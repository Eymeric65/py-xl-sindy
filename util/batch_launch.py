import subprocess
from concurrent.futures import ThreadPoolExecutor

# Function to execute a command
def run_command(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr

def mujoco_align_cmd_creator(exp,max_time,forces_scale):

    return     [
        "python", "mujoco_align.py",
        "--experiment-folder", exp,
        "--max-time", max_time,
        "--no-real_mujoco_time",
        "--forces-scale-vector", *forces_scale,
        "--forces-period", "3",
        "--forces-period-shift", "0.5",
        "--regression",
    ]


# List of commands to execute
commands = [
    [
        "python", "mujoco_align.py",
        "--experiment-folder", "mujoco_align_data/cart_pole_double",
        "--max-time", "30",
        "--no-real_mujoco_time",
        "--forces-scale-vector", "3", "6", "8",
        "--forces-period", "3",
        "--forces-period-shift", "0.5",
        "--regression",
        "--generate-ideal-path",
        "--plot"
    ],
    # Add more command lists as needed
]

# Execute commands in parallel
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(run_command, cmd) for cmd in commands]
    for future in futures:
        stdout, stderr = future.result()
        if stdout:
            print("Output:", stdout)
        if stderr:
            print("Error:", stderr)
