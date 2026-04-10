import os


DEVICE_ENV_VAR = "PTODSL_TEST_DEVICE_ID"
DEFAULT_DEVICE_ID = "0"
DEFAULT_NUM_CUBE_CORES = 20
DEFAULT_NUM_VEC_CORES = DEFAULT_NUM_CUBE_CORES * 2
DEVICE_PREFIX = "npu:"


def get_num_cube_cores() -> int:
    """Return the number of cube (matrix) cores on the NPU."""
    try:
        import torch

        return int(getattr(torch.npu.get_device_properties(0), "cube_core_num"))
    except Exception as e:
        print(
            f"Warning: could not query cube_core_num ({e}); defaulting to {DEFAULT_NUM_CUBE_CORES}."
        )
        return DEFAULT_NUM_CUBE_CORES


def get_num_vec_cores() -> int:
    """Return the number of vector cores on the NPU."""
    try:
        import torch

        return int(getattr(torch.npu.get_device_properties(0), "vector_core_num"))
    except Exception as e:
        print(
            f"Warning: could not query vector_core_num ({e}); defaulting to {DEFAULT_NUM_VEC_CORES}."
        )
        return DEFAULT_NUM_VEC_CORES


def get_test_device() -> str:
    device_id = os.getenv(DEVICE_ENV_VAR)
    if not device_id:
        print(
            f"Warning: {DEVICE_ENV_VAR} is not set; defaulting to {DEFAULT_DEVICE_ID}."
        )
        device_id = DEFAULT_DEVICE_ID

    if device_id.startswith(DEVICE_PREFIX):
        return device_id
    return f"{DEVICE_PREFIX}{device_id}"
