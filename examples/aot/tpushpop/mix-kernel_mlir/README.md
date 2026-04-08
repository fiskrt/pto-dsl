# Cross core communication with `pto.push_to_aiv` example

## Run

```bash
python run.py c2v
python run.py bidi
```

`c2v` is the default, so `python run.py` is the same as `python run.py c2v`.

## How C2V Communication Works

This example sends one `16x16 f32` tile from the Cube kernel to the Vector kernel.

- The host allocates one shared `gm_slot_buffer` and passes it to both kernels.
- The Vector kernel owns the C2V consumer buffer with `pto.reserve_buffer(name = "c2v_fifo")`.
- The Cube kernel refers to that same buffer with `pto.import_reserved_buffer(name = "c2v_fifo")`.
- Both sides call `*_initialize_pipe` with `dir_mask = 1`, which means `C2V`.
- Cube sends with `pto.tpush_to_aiv(...)`.
- Vector receives with `pto.tpop_from_aic(...)` and releases the consumed slot with `pto.tfree_from_aic`.

In the generated C++, this becomes the same `TPipe<..., Direction::DIR_C2V, ...>` on both sides:

- Cube: `TPUSH(pipe, acc_tile)`
- Vector: `TPOP(pipe, vec_tile)` then `TFREE(pipe)`

The important mental model is: `TPUSH`/`TPOP` are the real cross-core handoff, while `gm_slot_buffer` is the shared backing storage that makes the FIFO work.

## How Bidirectional Works

`bidi` starts the same way as `c2v`, but adds a return path:

- Cube computes `x @ x` and sends it to vector over C2V.
- Vector pops that tile, computes `tile + tile`, and pushes the doubled result back over V2C.
- Cube pops the returned tile and writes it to GM.

The important difference is that both sides initialize with `dir_mask = 3`, so the same mixed-kernel launch can use both directions of the pipe.
