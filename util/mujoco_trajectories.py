
from dataclasses import dataclass
from dataclasses import field
from typing import List
import tyro

import sys
import os
import importlib


@dataclass
class Args:
    experiment_folder: str = None
    """the folder where the experiment data is stored : the mujoco environment.xml file and the xlsindy_gen.py script"""
    max_time: float = 10.0
    """the maximum time for the simulation"""
    real_mujoco_time: bool = True
    """if True, the simulation will be done in real time, otherwise, the simulation will be done as fast as possible"""
    forces_scale_vector: List[float] = field(default_factory=lambda: None)
    """the different scale for the forces vector to be applied, this can mimic an action mask over the system if some entry are 0"""
    forces_period: float = 3.0
    """the period for the forces function"""
    forces_period_shift: float = 0.5
    """the shift for the period of the forces function"""
    generate_ideal_path: bool = False
    """if True, the ideal simulation from the lagrangian provided will be generated"""


if __name__ == "__main__":

    args = tyro.cli(Args)

    # CLI validation
    if args.forces_scale_vector is None:
        raise ValueError("forces_scale_vector should be provided, don't hesitate to invoke --help")
    if args.experiment_folder is None:
        raise ValueError("experiment_folder should be provided, don't hesitate to invoke --help")
    else: # import the xlsindy_back_script
        folder_path = os.path.join(os.path.dirname(__file__), args.experiment_folder)
        sys.path.append(folder_path)

        # import the xlsindy_gen.py script
        xlsindy_gen = importlib.import_module("xlsindy_gen")

        try:
            xlsindy_component = xlsindy_gen.xlsindy_component
        except AttributeError:
            raise AttributeError("xlsindy_gen.py should contain a function named xlsindy_component")
        
        try:
            mujoco_transform = xlsindy_gen.mujoco_transform
        except AttributeError:
            mujoco_transform = None
        
        num_coordinates, time_sym, symbols_matrix, full_catalog, extra_info = xlsindy_component()

        # Mujoco environment path
        mujoco_xml = os.path.join(folder_path, "environment.xml")