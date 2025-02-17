python batch_launch.py \
--experiment-folder "mujoco_align_data/cart_pole_double" \
--max-time 20 \
--number-coordinate 3 \
--forces-span 1 4 \
--number-experiment 2

python batch_launch.py \
--experiment-folder "mujoco_align_data/cart_pole_double" \
--max-time 20 \
--number-coordinate 3 \
--forces-span 1 12 \
--number-experiment 25

python batch_launch.py \
--experiment-folder "mujoco_align_data/cart_pole_double" \
--max-time 20 \
--number-coordinate 3 \
--forces-span 1 12 \
--number-experiment 100

python batch_launch.py \
--experiment-folder "mujoco_align_data/double_pendulum_pm" \
--max-time 20 \
--number-coordinate 2 \
--forces-span 1 12 \
--number-experiment 2

python batch_launch.py \
--experiment-folder "mujoco_align_data/double_pendulum_pm" \
--max-time 20 \
--number-coordinate 2 \
--forces-span 1 12 \
--number-experiment 150

python batch_launch.py \
--experiment-folder "mujoco_align_data/cart_pole" \
--max-time 20 \
--number-coordinate 2 \
--forces-span 1 12 \
--number-experiment 2

python batch_launch.py \
--experiment-folder "mujoco_align_data/cart_pole" \
--max-time 20 \
--number-coordinate 2 \
--forces-span 1 12 \
--number-experiment 150

python batch_launch.py \
--experiment-folder "mujoco_align_data/cart_pole" \
--max-time 20 \
--number-coordinate 2 \
--forces-span 1 12 \
--number-experiment 2 \
--mode "generate" \
--random-seed 12 \
--sample-number 500

## --- Second set of experiment 20250217 ---

# Generate
python batch_launch.py \
--experiment-folder "mujoco_align_data/cart_pole" \
--max-time 20 \
--number-coordinate 2 \
--forces-span 1 12 \
--number-experiment 200 \
--mode "generate" \
--random-seed 12 \
--sample-number 500

python batch_launch.py \
--experiment-folder "mujoco_align_data/cart_pole_double" \
--max-time 20 \
--number-coordinate 3 \
--forces-span 1 12 \
--number-experiment 2 \
--mode "generate" \
--random-seed 13 \
--sample-number 500

python batch_launch.py \
--experiment-folder "mujoco_align_data/double_pendulum_pm" \
--max-time 20 \
--number-coordinate 2 \
--forces-span 1 12 \
--number-experiment 200 \
--mode "generate" \
--random-seed 14 \
--sample-number 1000

# Align

python batch_file_execute.py \
--script "align_data" \
--script_args "hard_threshold_sparse_regression" "xlsindy"

python batch_file_execute.py \
--script "align_data" \
--script_args "lasso_regression" "xlsindy"

# Populate metric

python batch_file_execute.py \
--script "exploration_metric" 

## -----------------------------------