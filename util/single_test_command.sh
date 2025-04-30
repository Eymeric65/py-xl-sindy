python single_test.py \
    --random-seed 10 \
    --experiment-folder "mujoco_align_data/cart_pole" \
    --max-time 20 \
    --forces-scale-vector 7 4.5 \
    --forces-period 3 \
    --forces-period-shift 0.5 \
    --sample-number 1000 \
    --optimization-function "hard_threshold_sparse_regression" \
    --algorithm "xlsindy" \
    --noise-level 0.0 \
    --no-implicit-regression

python single_test.py \
    --random-seed 10 \
    --experiment-folder "mujoco_align_data/cart_pole" \
    --max-time 20 \
    --forces-scale-vector 15 15 \
    --forces-period 3 \
    --forces-period-shift 0.5 \
    --sample-number 1000 \
    --optimization-function "lasso_regression" \
    --algorithm "xlsindy" \
    --noise-level 0.0 \
    --no-implicit-regression

python single_test.py \
    --random-seed 12 14 \
    --experiment-folder "mujoco_align_data/cart_pole" \
    --max-time 20 \
    --forces-scale-vector 3.049855884605056 1.639574110558981 \
    --forces-period 3 \
    --forces-period-shift 0.5 \
    --sample-number 1000 \
    --optimization-function "hard_threshold_sparse_regression" \
    --algorithm "xlsindy" \
    --noise-level 0.0 \
    --no-implicit-regression
# Compatible with old test catp_pole_1214_20250221_135709.json


python single_test.py \
    --random-seed 12 14 \
    --experiment-folder "mujoco_align_data/cart_pole" \
    --max-time 30 \
    --forces-scale-vector 8 8 \
    --forces-period 3 \
    --forces-period-shift 0.5 \
    --sample-number 1000 \
    --optimization-function "hard_threshold_sparse_regression" \
    --algorithm "xlsindy" \
    --noise-level 0.0 \
    --no-implicit-regression