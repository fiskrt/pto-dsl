import ctypes
import os

import torch
import torch_npu  # noqa: F401

from ptodsl import do_bench
from ptodsl.test_util import get_test_device


M = 4224
N = 16384
K = 16384
BLOCK_DIM = 24
SWIZZLE_DIRECTION = 1
SWIZZLE_COUNT = 3
WARMUP_ITERS = 5
BENCH_ITERS = 20


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path):
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.call_kernel.restype = None

    def matmul_abt(
        a,
        b,
        *,
        block_dim=BLOCK_DIM,
        swizzle_direction=SWIZZLE_DIRECTION,
        swizzle_count=SWIZZLE_COUNT,
        stream_ptr=None,
    ):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_

        m = int(a.shape[0])
        k = int(a.shape[1])
        n = int(b.shape[0])
        c = torch.empty((m, n), device=a.device, dtype=a.dtype)

        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(a),
            torch_to_ctypes(b),
            torch_to_ctypes(c),
            m,
            n,
            k,
            swizzle_direction,
            swizzle_count,
        )
        return c

    return matmul_abt


def main():
    device = get_test_device()
    torch.npu.set_device(device)
    matmul_abt = load_lib("./matmul_kernel.so")

    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16, device=device)
    b = torch.randn(N, K, dtype=torch.float16, device=device)

    fn = lambda: matmul_abt(a, b)
    time_us = do_bench(
        fn,
        warmup_iters=WARMUP_ITERS,
        benchmark_iters=BENCH_ITERS,
        unit="us",
        flush_cache=False,
    )

    flops = 2.0 * M * N * K
    tflops = flops / time_us / 1e6

    print("---")
    print(f"(m, n, k)=({M}, {N}, {K})")
    print(f"TFLOPS: {tflops:.1f}")
    print(f"execution_time: {time_us:.5f} us")


if __name__ == "__main__":
    main()
