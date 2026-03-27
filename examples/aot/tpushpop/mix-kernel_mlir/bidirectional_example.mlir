// Bidirectional pipe example.
//
// There are two logical FIFO pipes:
// - `c2v_fifo`: cube/kernel `@cube_kernel` pushes to vector/kernel `@vector_kernel`
// - `v2c_fifo`: vector/kernel `@vector_kernel` pushes to cube/kernel `@cube_kernel`
//
// `gm_slot_buffer` is the GM-backed slot storage for these pipes. The reserve/import
// ops connect each side of the same named FIFO, and `aic/aiv_initialize_pipe`
// binds those FIFO endpoints to the shared GM slot buffer plus each side's local
// consumer buffer.
//
// What is transferred:
// - Cube -> Vector: one full `16 x 16` `f32` accumulator tile via `pto.tpush_to_aiv`
//   with `split = 0` (no split). Vector receives that same logical `16 x 16` tile
//   with `pto.tpop_from_aic`, but in a vector tile type/layout.
// - Vector -> Cube: the doubled version of that received tile. Vector computes
//   `recv_tile + recv_tile` with `pto.tadd`, then sends that full `16 x 16` `f32`
//   result back with `pto.tpush_to_aic`. Cube receives it with `pto.tpop_from_aiv`
//   in a matrix tile type/layout.
//
// Shape summary:
// - All transferred tiles are `rows=16, cols=16, dtype=f32`
// - Cube-produced tile: `loc=acc`, `blayout=col_major`, `slayout=row_major`
// - Vector-produced return tile: `loc=vec`, `blayout=row_major`, `slayout=none_box`
// - Cube-consumed tile after V2C pop: `loc=mat`, `blayout=col_major`, `slayout=row_major`
// - Vector-consumed tile after C2V pop: `loc=vec`, `blayout=row_major`, `slayout=none_box`
module {

  func.func @call_both(%gm_slot_buffer: !pto.ptr<f32>) attributes {pto.entry} {
    func.call @cube_kernel(%gm_slot_buffer) : (!pto.ptr<f32>) -> ()
    func.call @vector_kernel(%gm_slot_buffer) : (!pto.ptr<f32>) -> ()
    return
  }

  func.func @cube_kernel(%gm_slot_buffer: !pto.ptr<f32>) attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    %v2c_local = pto.reserve_buffer {
      name = "v2c_fifo",
      size = 4096,
      location = #pto.address_space<mat>,
      auto = true
    } -> i32
    %c2v_import = pto.import_reserved_buffer {
      name = "c2v_fifo",
      peer_func = @vector_kernel
    } -> i32
    pto.aic_initialize_pipe {dir_mask = 3, slot_size = 1024}
      (gm_slot_buffer = %gm_slot_buffer : !pto.ptr<f32>,
       c2v_consumer_buf = %c2v_import : i32,
       v2c_consumer_buf = %v2c_local : i32)

    %acc_tile = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
    pto.tpush_to_aiv(%acc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>) {split = 0}

    %mat_tile = pto.tpop_from_aiv {split = 0}
      -> !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %left_tile = pto.alloc_tile : !pto.tile_buf<loc=left, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=row_major, fractal=512, pad=0>
    pto.tmov ins(%mat_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>) outs(%left_tile : !pto.tile_buf<loc=left, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=row_major, fractal=512, pad=0>)
    pto.tfree_from_aiv {split = 0}
    return
  }

  func.func @vector_kernel(%gm_slot_buffer: !pto.ptr<f32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c2v_local = pto.reserve_buffer {
      name = "c2v_fifo",
      size = 4096,
      location = #pto.address_space<vec>,
      auto = true
    } -> i32
    %v2c_import = pto.import_reserved_buffer {
      name = "v2c_fifo",
      peer_func = @cube_kernel
    } -> i32
    pto.aiv_initialize_pipe {dir_mask = 3, slot_size = 1024}
      (gm_slot_buffer = %gm_slot_buffer : !pto.ptr<f32>,
       c2v_consumer_buf = %c2v_local : i32,
       v2c_consumer_buf = %v2c_import : i32)

    %recv_tile = pto.tpop_from_aic {split = 0}
      -> !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %sum_tile = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tadd ins(%recv_tile, %recv_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%sum_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tfree_from_aic {split = 0}
    pto.tpush_to_aic(%sum_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) {split = 0}
    return
  }

}
