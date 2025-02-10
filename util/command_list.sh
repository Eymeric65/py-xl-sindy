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
--forces-scale-vector 4 4 \
--forces-period 3 \
--forces-period-shift 0.5 \
--regression 

# True experiment, it works on double_pendulum_pm

python mujoco_align.py \
--experiment-folder "mujoco_align_data/double_pendulum_pm" \
--max-time 30 \
--no-real_mujoco_time \
--forces-scale-vector 8 8 \
--forces-period 3 \
--forces-period-shift 0.5 \
--regression \
--generate-ideal-path

# Test retrieve cartpole
python mujoco_align.py \
--experiment-folder "mujoco_align_data/cart_pole" \
--max-time 30 \
--no-real_mujoco_time \
--forces-scale-vector 2 3 \
--forces-period 3 \
--forces-period-shift 0.5 \
--regression \
--generate-ideal-path

#0.126 % variance 0.066% model

python mujoco_align.py \
--experiment-folder "mujoco_align_data/cart_pole_double" \
--max-time 30 \
--no-real_mujoco_time \
--forces-scale-vector 3 6 8 \
--forces-period 3 \
--forces-period-shift 0.5 \
--regression \
--generate-ideal-path \
--plot

python mujoco_align.py \
--experiment-folder "mujoco_align_data/cart_pole_double" \
--max-time 30 \
--no-real_mujoco_time \
--forces-scale-vector 3 6 8 \
--forces-period 3 \
--forces-period-shift 0.5 \
--regression 

python mujoco_align.py \
--experiment-folder "mujoco_align_data/cart_pole_double" \
--max-time 30 \
--no-real_mujoco_time \
--forces-scale-vector 1 1 1 \
--forces-period 3 \
--forces-period-shift 0.5 \
--regression 
