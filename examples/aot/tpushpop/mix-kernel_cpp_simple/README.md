# Simple Cube To Vector `TPUSH`/`TPOP` Example

This is a stripped-down sibling of `mix-kernel_cpp`.

The kernel is fixed to a single `16x32 @ 32x32` matmul, followed by a bias add on the vector side:

- no tile loop
- no sanity mode
- no extra runner configuration

Run it with:

```bash
python run.py
```
