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
from typing import List
import tyro

import xlsindy

import numpy as np
import json

import sys
import os
import importlib

from jax import jit
from jax import vmap


@dataclass
class Args:
    experiment_file: str = "None"
    """the experiment file (without extension)"""
    optimization_function:str = "lasso_regression"
    """the regression function used in the regression"""
    algorithm:str = "xlsindy"
    """the name of the algorithm used (for the moment "xlsindy" and "sindy" are the only possible)"""
    noise_level:float = 0.0
    """the level of noise introduce in the experiment"""
    random_seed:List[int] = field(default_factory=lambda:[0])
    """the random seed for the noise"""
    validation_on_database:bool = True
    """if true validate the model on the database file"""

if __name__ == "__main__":

    args = tyro.cli(Args)

    ## CLI validation
    if args.experiment_file == "None":
        raise ValueError("experiment_file should be provided, don't hesitate to invoke --help")

    with open(args.experiment_file+".json", 'r') as json_file:
        simulation_dict = json.load(json_file)

    folder_path = os.path.join(os.path.dirname(__file__), "mujoco_align_data/"+simulation_dict["input"]["experiment_folder"])
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
    
    num_coordinates, time_sym, symbols_matrix, catalog_repartition, extra_info = xlsindy_component()        

    regression_function=eval(f"xlsindy.optimization.{args.optimization_function}")



    sim_data = np.load(args.experiment_file+".npz")

    rng=np.random.default_rng(args.random_seed)

    #load
    imported_time = sim_data['array1'] 
    imported_qpos = sim_data['array2']
    imported_qvel = sim_data['array3']
    imported_qacc = sim_data['array4']
    imported_force = sim_data['array5']

    #add noise
    imported_qpos+= rng.normal(loc=0,scale=args.noise_level,size=imported_qpos.shape)
    imported_qvel+= rng.normal(loc=0,scale=args.noise_level,size=imported_qvel.shape)
    imported_qacc+= rng.normal(loc=0,scale=args.noise_level,size=imported_qacc.shape)
    imported_force+= rng.normal(loc=0,scale=args.noise_level,size=imported_force.shape)

    ## XLSINDY dependent
    solution, exp_matrix, _ = xlsindy.simulation.execute_regression(
    theta_values=imported_qpos,
    velocity_values = imported_qvel,
    acceleration_values = imported_qacc,
    time_symbol=time_sym,
    symbol_matrix=symbols_matrix,
    catalog_repartition=catalog_repartition,
    external_force= imported_force,
    hard_threshold = 1e-3,
    apply_normalization = True,
    regression_function=regression_function
    )

    #modele_fit,friction_matrix = xlsindy.catalog_gen.create_solution_expression(solution[:, 0], full_catalog,num_coordinates=num_coordinates,first_order_friction=True)
    model_acceleration_func, valid_model = xlsindy.dynamics_modeling.generate_acceleration_function(solution,catalog_repartition, symbols_matrix, time_sym,lambdify_module="jax")
    model_dynamics_system = xlsindy.dynamics_modeling.dynamics_function_RK4_env(model_acceleration_func) 
    

    ## Analysis of result
    
    result_name = f"result__{args.algorithm}__{args.noise_level:.1e}__{args.optimization_function}"
    simulation_dict[result_name] = {}
    simulation_dict[result_name]["algoritm"]=args.algorithm
    simulation_dict[result_name]["noise_level"]=args.noise_level
    simulation_dict[result_name]["optimization_function"]=args.optimization_function

    simulation_dict[result_name]["catalog_len"]=extra_info["catalog_len"]

    simulation_dict[result_name]["ideal_solution"]=extra_info["ideal_solution_vector"]

    

    if valid_model:

        model_dynamics_system = vmap(model_dynamics_system, in_axes=(1,1),out_axes=1)

        model_acc = xlsindy.dynamics_modeling.vectorised_acceleration_generation(model_dynamics_system,imported_qpos,imported_qvel,imported_force)

        # Finally, select the columns of interest (e.g., every second column starting at index 1)
        model_acc = model_acc[:, 1::2]

        if args.validation_on_database:


            validation_data = np.load("mujoco_align_data/"+simulation_dict["input"]["experiment_folder"]+".npz")

            #load
            validation_time = validation_data['array1'] 
            validation_qpos = validation_data['array2']
            validation_qvel = validation_data['array3']
            validation_qacc = validation_data['array4']
            validation_force = validation_data['array5']


            validation_acc= xlsindy.dynamics_modeling.vectorised_acceleration_generation(model_dynamics_system,validation_qpos,validation_qvel,validation_force)
            
            RMSE_validation = xlsindy.result_formatting.relative_mse(validation_acc[3:-3],validation_qacc[3:-3])

            simulation_dict[result_name]["RMSE_validation"] = RMSE_validation
            print("estimate variance on validation is : ",RMSE_validation)

        simulation_dict[result_name]["solution"]=solution
        # Estimate of the variance between model and mujoco
        RMSE_acceleration = xlsindy.result_formatting.relative_mse(model_acc[3:-3],imported_qacc[3:-3])

        simulation_dict[result_name]["RMSE_acceleration"] = RMSE_acceleration
        print("estimate variance between mujoco and model is : ",RMSE_acceleration)

        # Sparsity difference
        non_null_term = np.argwhere(solution != 0) 

        ideal_solution = extra_info["ideal_solution_vector"]

        non_null_term=np.unique(np.concat((non_null_term,np.argwhere(ideal_solution != 0 )),axis=0),axis=0)

        sparsity_reference = np.count_nonzero( extra_info["ideal_solution_vector"] )
        sparsity_model = np.count_nonzero(solution)

        sparsity_percentage = 100*(sparsity_model-sparsity_reference)/sparsity_reference
        sparsity_difference = abs(sparsity_model-sparsity_reference)
        print("sparsity difference percentage : ",sparsity_percentage)
        print("sparsity difference number : ",sparsity_difference)

        simulation_dict[result_name]["sparsity_difference"] = sparsity_difference
        simulation_dict[result_name]["sparsity_difference_percentage"] = sparsity_percentage

        # Model RMSE comparison
        ideal_solution_norm_nn = xlsindy.result_formatting.normalise_solution(extra_info["ideal_solution_vector"])[*non_null_term.T]
        solution_norm_nn = xlsindy.result_formatting.normalise_solution(solution)[*non_null_term.T]

        RMSE_model = xlsindy.result_formatting.relative_mse(ideal_solution_norm_nn,solution_norm_nn)
        simulation_dict[result_name]["RMSE_model"] = RMSE_model
        print("RMSE model comparison : ",RMSE_model)

    if not valid_model:
        print("Skipped model verification, retrieval failed")

    simulation_dict = xlsindy.result_formatting.convert_to_strings(simulation_dict)
    print("print model ...")
    with open(args.experiment_file+".json", 'w') as file:
        json.dump(simulation_dict, file, indent=4)
