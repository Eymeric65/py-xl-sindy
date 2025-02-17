python mujoco_generate_data.py \
--experiment-folder "mujoco_align_data/cart_pole_double" \
--max-time 20 \
--forces-scale-vector 1 1 1 \
--forces-period 3 \
--forces-period-shift 0.5 \
--random-seed 12 \
--sample-number 1000 

python align_data.py \
--experiment-file "cart_pole_double__20250214_164920" \
--optimization-function "lasso_regression" \
--algorithm "xlsindy"

python align_data.py \
--experiment-file "cart_pole_double__20250214_172043" \
--optimization-function ""hard_threshold_sparse_regression"" \
--algorithm "xlsindy"