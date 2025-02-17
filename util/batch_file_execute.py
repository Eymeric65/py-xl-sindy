""" 
This script execute a python file for every file in result.
It is mainly used to execute :
- align_data.py
- exploration_metric.py
"""
import subprocess
from concurrent.futures import ThreadPoolExecutor

import os
import glob

from dataclasses import dataclass
from dataclasses import field
from typing import List
import tyro

@dataclass
class Args:
    script: str = "align_data"
    """The script to launch : can be from multiple type"""
    script_args :List[str] = field(default_factory=lambda:[])
    """The script argument to be passed (check the script to know order)"""

def run_command(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr


if __name__ == "__main__":

    args = tyro.cli(Args)

    if args.script =="align_data":

        def command_generator(filepath):

            return [
            "python", "align_data.py",
            "--experiment-file", filepath,
            "--optimization-function", args.script_args[0],
            "--algorithm",args.script_args[1]
            ]
        
    if args.script =="exploration_metric":

        def command_generator(filepath):

            return [
            "python", "exploration_metric.py",
            "--experiment-file", filepath,
            ]
        
    commands =[]

        # Loop through all .json files in the "result" folder
    for json_file in glob.glob("result/*.json"):
        # Remove the .json extension from the file path
        base_filepath = os.path.splitext(json_file)[0]
        #print(base_filepath)
        commands.append(command_generator(base_filepath))



    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_command, cmd) for cmd in commands]
        for future in futures:
            stdout, stderr = future.result()
            if stdout:
                print("Output:", stdout)
            if stderr:
                print("Error:", stderr)