import fbgemm_gpu  # noqa
import sys

import torch
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.hstu_attention import hstu_mha

H, D = 4, 128  # DLRM-v3 config: 4 heads, 128 qk/linear dim
kernel = {"triton": HammerKernel.TRITON, "pytorch": HammerKernel.PYTORCH}[sys.argv[1]]
B = int(sys.argv[2])
for L in [int(x) for x in sys.argv[3:]]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        lens = torch.full((B,), L, device="cuda", dtype=torch.int32)
        off = torch.zeros(B + 1, device="cuda", dtype=torch.int32)
        off[1:] = torch.cumsum(lens, 0)
        N = B * L
        q, k, v = (
            torch.randn(N, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(3)
        )
        out = hstu_mha(
            max_seq_len=L, alpha=1.0 / D, q=q, k=k, v=v, seq_offsets=off,
            causal=True, kernel=kernel,
        )
        out.sum().backward()
        torch.cuda.synchronize()
        print(f"{sys.argv[1]} B={B} L={L}: peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
    except torch.OutOfMemoryError:
        print(f"{sys.argv[1]} B={B} L={L}: OOM", flush=True)
        break
