import ctypes
import os
import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_npu

from ptodsl.test_util import get_test_device


BLOCK_DIM_LIST = [24]
M_LIST = [1024]
SHAPES_NK = [(1024, 1024)]


@dataclass
class CaseResult:
    m: int
    n: int
    k: int
    block_dim: int
    max_absdiff: float
    mean_absdiff: float


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


def run_case(matmul_abt, a, b, c_ref, *, block_dim):
    c = matmul_abt(
        a,
        b,
        block_dim=block_dim
    )
    torch.npu.synchronize()
    return CaseResult(
        m=int(a.shape[0]),
        n=int(b.shape[0]),
        k=int(a.shape[1]),
        block_dim=block_dim,
        max_absdiff=float((c - c_ref).abs().max().item()),
        mean_absdiff=float((c - c_ref).abs().mean().item()),
    )


def test_matmul():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=[
            "step1-baseline",
            "step2-doublebuffer",
            "step3-swizzle",
            "step4-manual-pipelining",
            "all",
        ],
        default="all",
        help="Which kernel variant to run.",
    )
    args = parser.parse_args()

    device = get_test_device()
    torch.npu.set_device(device)

    variants = {
        "step1-baseline": "./build_artifacts/step1_baseline_kernel.so",
        "step2-doublebuffer": "./build_artifacts/step2_doublebuffer_kernel.so",
        "step3-swizzle": "./build_artifacts/step3_swizzle_kernel.so",
        "step4-manual-pipelining": "./build_artifacts/step4_manual_pipelining_kernel.so",
    }
    if args.variant == "all":
        selected = [
            ("step1-baseline", variants["step1-baseline"]),
            ("step2-doublebuffer", variants["step2-doublebuffer"]),
            ("step3-swizzle", variants["step3-swizzle"]),
            ("step4-manual-pipelining", variants["step4-manual-pipelining"]),
        ]
    else:
        selected = [(args.variant, variants[args.variant])]

    torch.manual_seed(0)
    for variant_name, lib_path in selected:
        print(f"\n=== Running variant: {variant_name} ({lib_path}) ===")
        matmul_abt = load_lib(lib_path)

        for m in M_LIST:
            for n, k in SHAPES_NK:
                a = torch.randn(m, k, dtype=torch.float16, device=device)
                b = torch.randn(n, k, dtype=torch.float16, device=device)
                c_ref = F.linear(a, b)
                torch.npu.synchronize()

                for block_dim in BLOCK_DIM_LIST:
                    result = run_case(matmul_abt, a, b, c_ref, block_dim=block_dim)
                    
                print(f"(m, n, k)=({m}, {n}, {k})")


if __name__ == "__main__":
    test_matmul()
