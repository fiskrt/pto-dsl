import ctypes
import os
import argparse

import torch
import torch_npu

from ptodsl.test_util import get_test_device


M = 1024
N = 1024
K = 1024
BLOCK_DIM = 24
VARIANTS = {
    "step1-baseline": "./build_artifacts/step1_baseline_kernel.so",
    "step2-doublebuffer": "./build_artifacts/step2_doublebuffer_kernel.so",
    "step3-swizzle": "./build_artifacts/step3_swizzle_kernel.so",
    "step4-manual-pipelining": "./build_artifacts/step4_manual_pipelining_kernel.so",
}


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
        ctypes.c_int
    ]
    lib.call_kernel.restype = None

    def matmul_abt(
        a,
        b,
        *,
        block_dim=24,
        stream_ptr=None,
    ):
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("matmul_abt expects 2D tensors: a[M,K], b[N,K]")
        if a.shape[1] != b.shape[1]:
            raise ValueError(
                f"K mismatch: a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}"
            )
        if a.dtype != torch.float16 or b.dtype != torch.float16:
            raise ValueError("matmul_abt currently supports float16 inputs only")

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
            k
        )
        return c

    return matmul_abt


def run_case(matmul_abt, a, b):
    c = matmul_abt(a, b, block_dim=BLOCK_DIM)
    torch.npu.synchronize()
    return c


def test_matmul():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()),
        default="step4-manual-pipelining",
        help="Which kernel variant to run.",
    )
    args = parser.parse_args()

    device = get_test_device()
    torch.npu.set_device(device)
    lib_path = VARIANTS[args.variant]
    matmul_abt = load_lib(lib_path)

    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16, device=device)
    b = torch.randn(N, K, dtype=torch.float16, device=device)
    run_case(matmul_abt, a, b)

    print(
        f"Ran variant={args.variant} ({lib_path}) "
        f"with m={M}, n={N}, k={K}, block_dim={BLOCK_DIM}"
    )


if __name__ == "__main__":
    test_matmul()
