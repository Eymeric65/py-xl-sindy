"""
This script is the second part of mujoco_gernerate_data and append the info file with regression data.
User can choose :
- the type of algorithm : Sindy, XLSindy
- the regression algorithm : coordinate descent (scipy lasso), hard treshold
- level of noise added to imported data
"""

#tyro cly dependencies
from dataclasses import dataclass
from dataclasses import field
import tyro

import numpy as np
import json

import sys
import os
import importlib

import xlsindy

@dataclass
class Args:
    experiment_file: str = "None"
    """the experiment file (without extension)"""
    optimization_function:str = "lasso_regression"
    """the regression function used in the regression"""
    experiment_folder: str = "None"
    """the folder where the experiment environent is stored : the mujoco environment.xml file and the xlsindy_gen.py script"""
    algorithm:str = "xlsindy"
    """the name of the algorithm used (for the moment "xlsindy" and "sindy" are the only possible)"""

if __name__ == "__main__":

    args = tyro.cli(Args)

    ## CLI validation
    if args.experiment_file == "None":
        raise ValueError("experiment_file should be provided, don't hesitate to invoke --help")
    if args.experiment_folder == "None":
        raise ValueError("experiment_folder should be provided, don't hesitate to invoke --help")
    else:
        folder_path = os.path.join(os.path.dirname(__file__), args.experiment_folder)
        sys.path.append(folder_path)

        # import the xlsindy_gen.py script
        xlsindy_gen = importlib.import_module("xlsindy_gen")

        try:
            xlsindy_component = eval(f"xlsindy_gen.{args.algorithm}_component")
        except AttributeError:
            raise AttributeError(f"xlsindy_gen.py should contain a function named {args.algorithm}_component in order to work with algorithm {args.algorithm}")
        
        try:
            forces_wrapper = xlsindy_gen.forces_wrapper
        except AttributeError:
            forces_wrapper = None
        
        num_coordinates, time_sym, symbols_matrix, full_catalog, extra_info = xlsindy_component()        

    regression_function=eval(f"xlsindy.optimization.{args.optimization_function}")

    with open(args.experiment_file+".json", 'r') as json_file:
        simulation_dict = json.load(json_file)

    sim_data = np.load('data.npz')
    imported_time = sim_data['array1']
    imported_qpos = sim_data['array2']
    imported_qvel = sim_data['array3']
    imported_qacc = sim_data['array4']
    imported_force = sim_data['array5']

    ## XLSINDY dependent
    solution, exp_matrix, _ = xlsindy.simulation.execute_regression(
    theta_values=imported_qpos,
    velocity_values = imported_qvel,
    acceleration_values = imported_qacc,
    time_symbol=time_sym,
    symbol_matrix=symbols_matrix,
    catalog=full_catalog,
    external_force= imported_force,
    hard_threshold = 1e-3,
    apply_normalization = True,
    regression_function=regression_function
    )

    modele_fit,friction_matrix = xlsindy.catalog_gen.create_solution_expression(solution[:, 0], full_catalog,num_coordinates=num_coordinates,first_order_friction=True)
    model_acceleration_func, _ = xlsindy.euler_lagrange.generate_acceleration_function(modele_fit, symbols_matrix, time_sym,first_order_friction=friction_matrix)
    model_dynamics_system = xlsindy.dynamics_modeling.dynamics_function(model_acceleration_func, forces_wrapper(forces_function))
    
    model_acc = []

    for i in range(len(imported_time)): # skip start

        base_vector = np.ravel(np.column_stack((imported_qpos[i],imported_qvel[i])))

        model_acc+= [model_dynamics_system(imported_time[i],base_vector)]

    model_acc = np.array(model_acc)

    model_acc = model_acc[:,1::2]
    
    ## ---------------------------