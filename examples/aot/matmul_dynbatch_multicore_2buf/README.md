```bash
python ./matmul_dsl.py | ptoas > mul.cpp
bash ./compile.sh
python ./run_matmul.py --no-benchmark
```
