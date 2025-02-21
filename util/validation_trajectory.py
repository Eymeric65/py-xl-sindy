""" 
This scripts generate a validation trajectory from mujoco and compare it with trajectory from all the retrieved model
"""
#tyro cly dependencies
from dataclasses import dataclass
from dataclasses import field
from typing import List
import tyro

import sys
import os
import importlib

import mujoco
import numpy as np 
import xlsindy

import matplotlib.pyplot as plt

import xlsindy.result_formatting

# loggin purpose
from datetime import datetime
import json


@dataclass
class Args:
    experiment_file: str = "None"
    """the experiment file (without extension)"""
    max_time: float = 10.0
    """the maximum time for the validation simulation"""
    random_seed:List[int] = field(default_factory=lambda:[2])
    """the random seed to add for the force function (only used for force function)"""




if __name__ == "__main__":

    args = tyro.cli(Args)

    # CLI validation
    if args.experiment_file == "None":
        raise ValueError("experiment_file should be provided, don't hesitate to invoke --help")

    with open(args.experiment_file+".json", 'r') as json_file:
        simulation_dict = json.load(json_file)

    folder_path = os.path.join(os.path.dirname(__file__), "mujoco_align_data/"+simulation_dict["input"]["experiment_folder"])
    sys.path.append(folder_path)

    # import the xlsindy_gen.py script
    xlsindy_gen = importlib.import_module("xlsindy_gen")

    try:
        xlsindy_component = eval(f"xlsindy_gen.xlsindy_component")
    except AttributeError:
        raise AttributeError(f"xlsindy_gen.py should contain a function named {args.algorithm}_component in order to work with algorithm {args.algorithm}")
    
    
    num_coordinates, time_sym, symbols_matrix, catalog_repartition, extra_info = xlsindy_component(mode=args.algorithm,random_seed=args.random_seed)        
