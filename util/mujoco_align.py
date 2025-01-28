""" 
The goal of this script is to align xl_sindy algorithm with the Mujoco environment.
The script takes in input :
- the simulation folder containing the mujoco environment.xml file and the xlsindy_gen.py script
- the way forces function should be created
- some optional hyperparameters
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
import mujoco.viewer
import time
import numpy as np 
import xlsindy



@dataclass
class Args:
    experiment_folder: str = None
    """the folder where the experiment data is stored : the mujoco environment.xml file and the xlsindy_gen.py script"""
    mujoco_angle_offset:float = -3.14
    """the angle offset to be applied to the mujoco environment"""
    max_time: float = 10.0
    """the maximum time for the simulation"""
    real_mujoco_time: bool = True
    """if True, the simulation will be done in real time, otherwise, the simulation will be done as fast as possible"""
    num_coordinates: int = None
    """the number of coordinates in the mujoco environment"""
    forces_scale_vector: List[float] = field(default_factory=lambda: None)
    """the different scale for the forces vector to be applied, this can mimic an action mask over the system if some entry are 0"""
    forces_period: float = 3.0
    """the period for the forces function"""
    forces_period_shift: float = 0.5
    """the shift for the period of the forces function"""


if __name__ == "__main__":

    args = tyro.cli(Args)
    #print(args)

    # CLI validation
    if args.num_coordinates is None:
        raise ValueError("num_coordinates should be provided, don't hesitate to invoke --help")
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

        # Mujoco environment path
        mujoco_xml = os.path.join(folder_path, "environment.xml")


    # Random controller initialisation

    forces_function = xlsindy.dynamics_modeling.optimized_force_generator(
        component_count=args.num_coordinates,
        scale_vector=args.forces_scale_vector,
        time_end=args.max_time,
        period=args.forces_period,
        period_shift=args.forces_period_shift,
        augmentations=30,
    )


    # initialize Mujoco environment and controller

    mujoco_model = mujoco.MjModel.from_xml_path(mujoco_xml)
    mujoco_data = mujoco.MjData(mujoco_model)

    mujoco_time = []
    mujoco_qpos = []
    mujoco_qvel = []
    mujoco_qacc = []

    def random_controller(forces_function):

        def ret(model, data):

            forces = forces_function(data.time)
            data.qfrc_applied = forces

            mujoco_time.append(data.time)
            mujoco_qpos.append(data.qpos)
            mujoco_qvel.append(data.qvel)
            mujoco_qacc.append(data.qacc)

        return ret
    
    mujoco.set_mjcb_control(random_controller(forces_function)) # use this for the controller, could be made easier with using directly the data from mujoco.

    # Viewer of the experiment
    with mujoco.viewer.launch_passive(mujoco_model, mujoco_data) as viewer:

        time_start_simulation = time.time()
        while viewer.is_running() and mujoco_data.time < args.max_time:

            mujoco.mj_step(mujoco_model, mujoco_data)
            viewer.sync()
            
            if args.real_mujoco_time:
                time_until_next_step = mujoco_model.opt.timestep - (time.time() - time_start_simulation)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        viewer.close()

    # turn the result into a numpy array
    mujoco_time = np.array(mujoco_time)
    mujoco_qpos = np.array(mujoco_qpos)
    mujoco_qvel = np.array(mujoco_qvel)
    mujoco_qacc = np.array(mujoco_qacc)


    