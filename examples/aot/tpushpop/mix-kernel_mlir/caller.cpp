#ifndef KERNEL_CPP
#error "KERNEL_CPP must be defined at compile time."
#endif

#include <cstdint>

#include KERNEL_CPP

extern "C" void call_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *gmSlotBuffer)
{
    call_both<<<blockDim, nullptr, stream>>>((__gm__ float *)gmSlotBuffer);
}
