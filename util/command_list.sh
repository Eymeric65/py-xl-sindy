python mujoco_align.py \
--experiment-folder "mujoco_align_data/double_pendulum_pm" \
--max-time 30 \
--no-real_mujoco_time \
--forces-scale-vector 10 2 \
--forces-period 3 \
--forces-period-shift 0.5 \
--regression \
--force-ideal-solution \
--generate-ideal-path

python mujoco_align.py \
--experiment-folder "mujoco_align_data/cart_pole" \
--max-time 30 \
--real_mujoco_time \
--forces-scale-vector 10 1 \
--forces-period 3 \
--forces-period-shift 0.5 \
--regression

python mujoco_align.py \
--experiment-folder "mujoco_align_data/cart_pole_double" \
--max-time 30 \
--real_mujoco_time \
--forces-scale-vector 10 1 2 \
--forces-period 3 \
--forces-period-shift 0.5 \
--regression

python mujoco_align.py \
--experiment-folder "mujoco_align_data/cart_pole" \
--max-time 30 \
--real_mujoco_time \
--forces-scale-vector 1 1 \
--forces-period 3 \
--forces-period-shift 0.5 \
--regression \
--force-ideal-solution \
--generate-ideal-path