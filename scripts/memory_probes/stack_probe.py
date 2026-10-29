import fbgemm_gpu  # noqa
import sys

import torch
from torch.utils.checkpoint import checkpoint
from generative_recommenders.common import HammerKernel
from generative_recommenders.modules.stu import STULayer, STULayerConfig, STUStack

mode = sys.argv[1]  # recompute | norecompute | ckpt
LAYERS, E, H, D = 5, 512, 4, 128  # DLRM-v3 config
rc = mode != "norecompute"
cfg = STULayerConfig(
    embedding_dim=E, num_heads=H, hidden_dim=D, attention_dim=D,
    recompute_normed_x=rc, recompute_uvqk=rc, recompute_y=rc,
    target_aware=False, sort_by_length=False,
)
stack = STUStack([STULayer(cfg) for _ in range(LAYERS)]).cuda().to(torch.bfloat16)
stack.set_hammer_kernel(HammerKernel.TRITON)
stack.train()
print(f"params: {sum(p.numel() for p in stack.parameters())/1e6:.1f}M", flush=True)

for L in [int(x) for x in sys.argv[2:]]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        lens = torch.tensor([L], device="cuda", dtype=torch.int32)
        off = torch.tensor([0, L], device="cuda", dtype=torch.int32)
        nt = torch.zeros(1, device="cuda", dtype=torch.int32)
        x = torch.randn(L, E, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        kw = dict(x_lengths=lens, x_offsets=off, max_seq_len=L, num_targets=nt)
        if mode == "ckpt":
            h = x
            for layer in stack._stu_layers:
                h = checkpoint(lambda t, m=layer: m(x=t, **kw), h, use_reentrant=False)
            out = h
        else:
            out = stack(x=x, **kw)
        out.float().sum().backward()
        torch.cuda.synchronize()
        print(f"{mode} L={L}: peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
        del x, out
    except torch.OutOfMemoryError:
        print(f"{mode} L={L}: OOM", flush=True)
        break
