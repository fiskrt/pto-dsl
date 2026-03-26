import ctypes
import os
import subprocess

import torch
import torch_npu  # noqa: F401

from ptodsl.test_util import get_test_device

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(THIS_DIR, "lib.so")
M = 16
K = 32
N = 32
FIFO_ELEMS = 1024
ATOL = 5e-2
RTOL = 5e-2


def ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def main() -> None:
    subprocess.run(["bash", "compile.sh"], check=True, cwd=THIS_DIR)

    device = get_test_device()
    torch.npu.set_device(device)
    torch.manual_seed(0)

    lib = ctypes.CDLL(LIB_PATH)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.call_kernel.restype = None

    a = torch.randn((M, K), dtype=torch.float16, device=device)
    b = torch.randn((K, N), dtype=torch.float16, device=device)
    bias = torch.randn((M, N), dtype=torch.float32, device=device)
    out = torch.zeros((M, N), dtype=torch.float32, device=device)
    fifo = torch.zeros((FIFO_ELEMS,), dtype=torch.float32, device=device)

    lib.call_kernel(
        1,
        torch.npu.current_stream()._as_parameter_,
        ptr(out),
        ptr(a),
        ptr(b),
        ptr(bias),
        ptr(fifo),
    )
    torch.npu.synchronize()

    ref = a.float().cpu() @ b.float().cpu() + bias.cpu()
    out_cpu = out.cpu()
    max_abs = float((out_cpu - ref).abs().max().item())
    print(f"max_abs={max_abs:.6f}")

    if not torch.allclose(out_cpu, ref, atol=ATOL, rtol=RTOL):
        raise SystemExit("validation failed")

    print("validation passed")


if __name__ == "__main__":
    main()
