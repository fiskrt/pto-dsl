/*
Flow:
1. Cube loads A and B from GM through GlobalTensor views.
2. Cube copies those GM-backed matrix tiles into local matrix tiles:
   `aMat` at `0x0`, `bMat` at `0x20000`, then converts them to matmul inputs
   `aTile` and `bTile` and runs one `TMATMUL` into `acc`.
3. Cube `TPUSH`es the full `16x32` accumulator tile to the C2V pipe.
4. Vector `TPOP`s its `8x32` half-tile from that pushed accumulator, loads the
   matching `8x32` bias tile from GM, does `TADD`, and stores the result to GM.

Allocation summary:
- `GlobalTensor` objects are just GM views over `srcA`, `srcB`, `bias`, and `out`.
  They do not allocate local on-core memory themselves.
- The C2V FIFO is also explicit GM memory in this example: `fifoMem` is the GM slot
  buffer passed into `TPipe`, so cube writes the pushed accumulator tile into GM and
  vector reads it back from that same GM-backed FIFO.
- Cube local tiles:
  `aMat @ 0x0`, `bMat @ 0x20000`, `aTile @ 0x0`, `bTile @ 0x0`, `acc @ 0x0`.
- Vector local tiles:
  `biasTile @ 0x10000`, `outTile @ 0x20000`.
- The cross-core transfer is the matmul result: one full `AccTile<float, 16, 32>`
  produced on cube and split `up/down` so each vector subcore receives one `8x32`
  row half via `TPOP`.
*/
#include <pto/pto-inst.hpp>
#include <pto/common/fifo.hpp>

using namespace pto;

using In = half;
using Out = float;

constexpr uint32_t M = 16;
constexpr uint32_t K = 32;
constexpr uint32_t N = 32;
constexpr uint32_t VEC_CORES = 2;
constexpr uint32_t VEC_M = M / VEC_CORES;

#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

__global__ AICORE void runTPushPopMatmulAdd(__gm__ uint64_t *ffts, __gm__ Out *out, __gm__ In *srcA, __gm__ In *srcB,
                                            __gm__ Out *bias, __gm__ Out *fifoMem)
{
    set_ffts_base_addr((uint64_t)ffts);

    using GlobalA = GlobalTensor<In, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<In, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalBias = GlobalTensor<Out, Shape<1, 1, 1, VEC_M, N>, Stride<M * N, M * N, VEC_M * N, N, 1>>;
    using GlobalOut = GlobalTensor<Out, Shape<1, 1, 1, VEC_M, N>, Stride<M * N, M * N, VEC_M * N, N, 1>>;

    using TileMatA = Tile<TileType::Mat, In, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, In, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<In, M, K, M, K>;
    using RightTile = TileRight<In, K, N, K, N>;
    using AccTile = TileAcc<Out, M, N, M, N>;
    using VecTile = Tile<TileType::Vec, Out, VEC_M, N, BLayout::RowMajor, VEC_M, N>;

    using Pipe = TPipe<0, Direction::DIR_C2V, M * N * sizeof(Out), 2>;
    Pipe pipe((__gm__ void *)(uint64_t)fifoMem, 0x0, 0x0);

    if constexpr (DAV_CUBE) {
        TileMatA aMat;
        TileMatB bMat;
        LeftTile aTile;
        RightTile bTile;
        AccTile acc;
        TASSIGN(aMat, 0x0);
        TASSIGN(bMat, 0x20000);
        TASSIGN(aTile, 0x0);
        TASSIGN(bTile, 0x0);
        TASSIGN(acc, 0x0);

        GlobalA globalA(srcA);
        GlobalB globalB(srcB);

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        TLOAD(aMat, globalA);
        TLOAD(bMat, globalB);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TMOV(aTile, aMat);
        TMOV(bTile, bMat);

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        TMATMUL(acc, aTile, bTile);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        TPUSH<Pipe, AccTile, TileSplitAxis::TILE_UP_DOWN>(pipe, acc);

        pipe_barrier(PIPE_ALL);
    }

    if constexpr (DAV_VEC) {
        VecTile popped;
        VecTile biasTile;
        VecTile outTile;
        TASSIGN(biasTile, 0x10000);
        TASSIGN(outTile, 0x20000);

        uint32_t subBlock = get_subblockid();
        uint32_t offset = subBlock * VEC_M * N;
        GlobalBias globalBias(bias + offset);
        GlobalOut globalOut(out + offset);

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(pipe, popped);
        TLOAD(biasTile, globalBias);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        TADD(outTile, popped, biasTile);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(globalOut, outTile);

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        pipe_barrier(PIPE_ALL);
    }
}

void LaunchTPushPopMatmulAdd(uint8_t *ffts, uint8_t *out, uint8_t *srcA, uint8_t *srcB, uint8_t *bias, uint8_t *fifoMem,
                             void *stream)
{
    runTPushPopMatmulAdd<<<1, nullptr, stream>>>(
        reinterpret_cast<uint64_t *>(ffts), reinterpret_cast<Out *>(out), reinterpret_cast<In *>(srcA),
        reinterpret_cast<In *>(srcB), reinterpret_cast<Out *>(bias), reinterpret_cast<Out *>(fifoMem));
}
