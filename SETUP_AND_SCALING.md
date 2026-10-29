# Setup, ML-1M run, and scaling notes

Host used: 8x NVIDIA H100 80GB, driver 595 (CUDA 13.2), Python 3.10.12, `uv`.

## 1. Environment

```bash
cd /disk1/madnan/generative-recommenders
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
python -c "import torch, fbgemm_gpu, torchrec; print(torch.__version__, torch.cuda.is_available())"
```

Installed without pinning: torch 2.14.1+cu130, torchrec 1.9.2, fbgemm_gpu matching. The sanity import printed `2.14.1+cu130 True`.
README tests CUDA 12.4 and Python 3.10. If an fbgemm/torch ABI mismatch appears on another host, pin
`torch==2.6.0 fbgemm-gpu==1.1.0 torchrec==1.1.0` (torch from the cu124 index).

## 2. Data

```bash
mkdir -p tmp/ && python3 preprocess_public_data.py
```

Downloads and preprocesses ml-1m, ml-20m and amzn-books into `tmp/`. ML-1M output is in `tmp/ml-1m/`.

## 3. Run the public experiment

Smoke test (2 epochs, 2m25s, no errors):

```bash
cp configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin /tmp/smoke.gin
echo 'train_fn.num_epochs = 2' >> /tmp/smoke.gin
CUDA_VISIBLE_DEVICES=0 python3 main.py --gin_config_file=/tmp/smoke.gin --master_port=12346
```

Full run (101 epochs, single GPU, about 2 hours estimated from the smoke run):

```bash
mkdir -p logs/ml-1m-l200/
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin \
  --master_port=12345 2>&1 | tee logs/ml-1m-l200/hstu-sampled-softmax-n128-large-final.log
```

Logs: `logs/ml-1m-l200/`. TensorBoard: `tensorboard --logdir exps/ml-1m-l200/ --port 24001 --bind_all`.

The run completed (about 2 hours, ended 22:01 on 2026-10-28). The `.log` file contains only model-init lines, because logging is not configured in the spawned worker processes. Read metrics from TensorBoard (`eval_epoch/*` tags, final epoch 100) instead.

Final results, single H100, compared with the README HSTU-large row:

| Metric | This run | README |
|---|---|---|
| HR@10 | 0.3359 | 0.3294 |
| NDCG@10 | 0.1918 | 0.1893 |
| HR@50 | 0.5975 | 0.5935 |
| NDCG@50 | 0.2498 | 0.2481 |
| HR@200 | 0.7907 | 0.7839 |
| NDCG@200 | 0.2793 | 0.2771 |

All six metrics are 0.7% to 2.0% above the README values. The best epoch values (max over epochs) are within 0.005 of the final ones, so the run is stable. The event directory also holds the 2-epoch smoke run, which adds two early points to each series.

## 4. Single vs multi-GPU

- `main.py:66` sets `world_size = torch.cuda.device_count()` and spawns one process per visible GPU.
- Training uses NCCL and `DistributedDataParallel` (`research/trainer/train.py:73`, `:264`). This is data parallel only.
- Single GPU: `CUDA_VISIBLE_DEVICES=0`. Multi-GPU: `CUDA_VISIBLE_DEVICES=0,1,2,3` with a different `--master_port`.
- `local_batch_size=128` is per GPU, so the global batch is 128 x N with the same learning rate, and `drop_last` turns on for N > 1.
  Use one GPU to match the paper table, or set `local_batch_size = 128 / N`. Cross-rank handling of sampled-softmax negatives was not checked.
- The item embedding table is replicated on every GPU. There is no tensor, pipeline or embedding parallelism in this trainer.

## 5. When the model does not fit one GPU

| What is too big | Approach |
|---|---|
| Embedding tables | torchrec `DistributedModelParallel`. Already used by the DLRM-v3 trainer (`dlrm_v3/train/utils.py`). Not in `main.py`. |
| Dense weights or optimizer state | FSDP. Not in the repo. |
| Activations at long context | Fused attention kernels first, then recompute, then context (sequence) parallelism. Not in the repo. |
| A single layer's weights | Tensor parallelism. Unlikely for HSTU layers. Not in the repo. |

Tensor parallelism shards weights, so it helps least for long-context training, where activations dominate.
HSTU attention is pointwise (SiLU of QK^T, no softmax), which should make a ring-style context-parallel implementation simpler than for standard attention. This comes from the paper and was not checked against the kernels.
Inference has no KV cache in this repo (the README says cached incremental decoding is not included yet for HSTU Ultra), so long-context inference is a full forward pass per request.

## 6. Fused attention kernels

- DLRM-v3 path: already fused. `HammerModule.hammer_kernel()` returns `HammerKernel.TRITON` unless overridden (`common.py:144`). Pin it with `set_hammer_kernel(HammerKernel.TRITON)`.
- Research trainer (`main.py`): not fused. Attention is a dense `torch.einsum` (`research/modeling/sequential/hstu.py:203`), quadratic in memory. Using fused kernels there would need a code change.

## 7. Memory probes

Scripts: `scripts/memory_probes/mem_probe.py` (attention op) and `scripts/memory_probes/stack_probe.py` (full STU stack).
Run on a free GPU, for example `CUDA_VISIBLE_DEVICES=1 python scripts/memory_probes/stack_probe.py recompute 65536 262144 1048576`.
Modes for `stack_probe.py`: `recompute` (default config), `norecompute`, `ckpt`.

Attention op only, one layer, 4 heads x 128, bf16, batch 1, forward and backward:

| Seq len | PyTorch | Triton |
|---|---|---|
| 16K | 6.9 GiB | 0.1 GiB |
| 32K | 27.4 GiB | not run |
| 64K | OOM | 0.5 GiB |
| 256K | not run | 2.0 GiB |
| 1M | not run | 8.0 GiB |

Full stack, 5 layers, embedding 512, 4 heads x 128, 9.2M params, bf16, batch 1, forward and backward, Triton kernels:

| Tokens | Default (recompute on) | Recompute off |
|---|---|---|
| 64K | 2.1 GiB | 4.2 GiB |
| 128K | 3.6 GiB | 7.9 GiB |
| 256K | 7.1 GiB | 15.6 GiB |
| 512K | 14.1 GiB | 31.1 GiB |
| 1M | 28.1 GiB | 62.2 GiB |

Findings:

- Memory is linear in length, about 28 KiB per token with defaults. Extrapolating, the 80GB limit is near 2M tokens. This is an extrapolation, not a measurement.
- `STULayerConfig` already defaults `recompute_normed_x`, `recompute_uvqk` and `recompute_y` to True, which halves memory.
- Wrapping layers in `torch.utils.checkpoint` on top of that fails with `CheckpointError` because the Triton autograd function reads `saved_tensors` more than once (`ops/triton/triton_hstu_preprocess_and_attention.py:198`). Not worked around.
- Scope limits: random dense inputs, dense stack only. Embedding lookups, the preprocessor, the loss head and step time were not measured.

## 8. Datasets and models that exceed one GPU

- DLRM-v3 streaming configs use `HASH_SIZE_1B` embedding tables at dim 512 (`dlrm_v3/configs.py:482`), about 2 TB per table in fp32. That is an embedding problem, handled by torchrec sharding.
- The DLRM-v3 dense stack (5 layers, hidden 512, 4 heads x 128) is 9.2M params and fits one GPU at 1M tokens.
- Sizes for `movielens-13b` and `movielens-18b` were not verified. Streaming data paths point at `/home/linjianma` and were not available here.

## 9. Decision

For long-context dense training, stay single GPU with the fused Triton kernels and the default recompute flags. Use DDP only for throughput.
Consider context parallelism only above about 2M tokens, and measure the full model (embeddings, preprocessor, loss, step time) before building it.
