"""
quick script to create a validation database 
"""

import os
import glob
import numpy as np

# Directories
result_dir = "result"
output_dir = "mujoco_align_data"
os.makedirs(output_dir, exist_ok=True)

# Use glob to find files matching the pattern "<name>__*.npz"
pattern = os.path.join(result_dir, "*__*.npz")
file_list = glob.glob(pattern)

# Group files by base name (e.g. "double_pendulum_pm")
groups = {}
for file_path in file_list:
    fname = os.path.basename(file_path)
    base_name = fname.split("__")[0]
    groups.setdefault(base_name, []).append(file_path)

# Create a reproducible random generator
rng = np.random.default_rng(seed=42)

# Process each group of files
for base_name, files in groups.items():
    time_samples = []
    qpos_samples = []
    qvel_samples = []
    qacc_samples = []
    force_samples = []

    for file_path in files:
        sim_data = np.load(file_path)
        time_data   = sim_data['array1']
        qpos_data   = sim_data['array2']
        qvel_data   = sim_data['array3']
        qacc_data   = sim_data['array4']
        force_data  = sim_data['array5']

        # Determine sample size (ensuring at least one sample)
        n_samples = time_data.shape[0]

        if base_name=="cart_pole":
            print(n_samples)
        sample_size = max(1, int(np.ceil(0.1 * n_samples)))

        # Randomly select indices (without replacement)
        indices = rng.choice(n_samples, size=sample_size, replace=False)

        # Append the sampled data
        time_samples.append(time_data[indices])
        qpos_samples.append(qpos_data[indices])
        qvel_samples.append(qvel_data[indices])
        qacc_samples.append(qacc_data[indices])
        force_samples.append(force_data[indices])

    # Concatenate samples from all files in the group
    time_concat  = np.concatenate(time_samples, axis=0)
    qpos_concat  = np.concatenate(qpos_samples, axis=0)
    qvel_concat  = np.concatenate(qvel_samples, axis=0)
    qacc_concat  = np.concatenate(qacc_samples, axis=0)
    force_concat = np.concatenate(force_samples, axis=0)

    # Save the aggregated data to a new file
    output_path = os.path.join(output_dir, f"{base_name}.npz")
    np.savez(output_path,
             array1=time_concat,
             array2=qpos_concat,
             array3=qvel_concat,
             array4=qacc_concat,
             array5=force_concat)

    print(f"Saved {output_path} with {time_concat.shape[0]} samples.")
