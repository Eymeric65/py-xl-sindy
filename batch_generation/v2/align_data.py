"""
This script is the second part of mujoco_gernerate_data and append the info file with regression data.
User can choose :
- the type of algorithm : Sindy, XLSindy, Mixed
- the regression algorithm : coordinate descent (scipy lasso), hard treshold
- level of noise added to imported data

Actually align_data.py is in developpement, Implicit explicit regression is under test

"""

# tyro cly dependencies
from dataclasses import dataclass
from dataclasses import field
from typing import List
import tyro

import xlsindy

import numpy as np
import json

import hashlib

import sys
import os
import importlib

from jax import jit
from jax import vmap

import pickle

import pandas as pd

from tqdm import tqdm

from logging import getLogger
import logging

logger = getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('align_data.log')
    ]
)

@dataclass
class Args:
    experiment_file: str = "None"
    """the experiment file (without extension)"""
    optimization_function: str = "lasso_regression"
    """the regression function used in the regression"""
    algorithm: str = "xlsindy"
    """the name of the algorithm used (for the moment "xlsindy" and "sindy" are the only possible)"""
    regression_type: str = "explicit"
    """the type of regression to use (explicit, implicit, mixed)"""
    noise_level: float = 0.0
    """the level of noise introduce in the experiment"""
    random_seed: List[int] = field(default_factory=lambda: [0])
    """the random seed for the noise"""
    skip_already_done: bool = True
    """if true, skip the experiment if already present in the result file"""
    print_graph: bool = False
    """if true, show the graph of the result"""

    def get_uid(self):
        hash_input = (
            self.optimization_function
            + self.algorithm
            + str(self.noise_level)
            + self.regression_type
            + str(self.random_seed)
        )
        return hashlib.md5(hash_input.encode()).hexdigest()

if __name__ == "__main__":

    args = tyro.cli(Args)

    ## CLI validation
    if args.experiment_file == "None":
        raise ValueError(
            "experiment_file should be provided, don't hesitate to invoke --help"
        )

    with open(args.experiment_file + ".json", "r") as json_file:
        simulation_dict = json.load(json_file)

    if args.skip_already_done:
        if args.get_uid() in simulation_dict["results"]:
            print("already aligned")
            exit()

    folder_path = simulation_dict["generation_settings"]["experiment_folder"]

    sys.path.append(folder_path)

    # import the xlsindy_gen.py script
    xlsindy_gen = importlib.import_module("xlsindy_gen")

    try:
        xlsindy_component = eval(f"xlsindy_gen.xlsindy_component")
    except AttributeError:
        raise AttributeError(
            f"xlsindy_gen.py should contain a function named {args.algorithm}_component in order to work with algorithm {args.algorithm}"
        )

    try:
        forces_wrapper = xlsindy_gen.forces_wrapper
    except AttributeError:
        forces_wrapper = None

    random_seed = simulation_dict["generation_settings"]["random_seed"] + args.random_seed
    print("random seed is :", random_seed)
    num_coordinates, time_sym, symbols_matrix, full_catalog, xml_content, extra_info = (
        xlsindy_component(mode=args.algorithm, random_seed=random_seed)
    )

    regression_function = eval(f"xlsindy.optimization.{args.optimization_function}")

    with open(simulation_dict["data_path"], 'rb') as f:
        sim_data = pickle.load(f)

    rng = np.random.default_rng(random_seed)

    # load
    imported_time = sim_data["simulation_time"]
    imported_qpos = sim_data["simulation_qpos"]
    imported_qvel = sim_data["simulation_qvel"]
    imported_qacc = sim_data["simulation_qacc"]
    imported_force = sim_data["force_vector"]


    # add noise
    imported_qpos += rng.normal(loc=0, scale=args.noise_level, size=imported_qpos.shape)*np.linalg.norm(imported_qpos)/imported_qpos.shape[0]
    imported_qvel += rng.normal(loc=0, scale=args.noise_level, size=imported_qvel.shape)*np.linalg.norm(imported_qvel)/imported_qvel.shape[0]
    imported_qacc += rng.normal(loc=0, scale=args.noise_level, size=imported_qacc.shape)*np.linalg.norm(imported_qacc)/imported_qacc.shape[0]
    imported_force += rng.normal(loc=0, scale=args.noise_level, size=imported_force.shape)*np.linalg.norm(imported_force)/imported_force.shape[0]

    ## XLSINDY dependent

    if args.regression_type == "implicit":

        solution, exp_matrix = xlsindy.simulation.regression_implicite(
            theta_values=imported_qpos,
            velocity_values=imported_qvel,
            acceleration_values=imported_qacc,
            time_symbol=time_sym,
            symbol_matrix=symbols_matrix,
            catalog_repartition=full_catalog,
            regression_function=regression_function,
        )

    elif args.regression_type == "explicit":

        solution, exp_matrix = xlsindy.simulation.regression_explicite(
            theta_values=imported_qpos,
            velocity_values=imported_qvel,
            acceleration_values=imported_qacc,
            time_symbol=time_sym,
            symbol_matrix=symbols_matrix,
            catalog_repartition=full_catalog,
            external_force=imported_force,
            regression_function=regression_function,
        )

    elif args.regression_type == "mixed":

        solution, exp_matrix = xlsindy.simulation.regression_mixed(
            theta_values=imported_qpos,
            velocity_values=imported_qvel,
            acceleration_values=imported_qacc,
            time_symbol=time_sym,
            symbol_matrix=symbols_matrix,
            catalog_repartition=full_catalog,
            external_force=imported_force,
            regression_function=regression_function,
        )

    # DEBUG
    # solution = extra_info["ideal_solution_vector"]
    # Apply hard thresholding to the solution
    threshold = 1e-2  # Adjust threshold value as needed
    solution = np.where(np.abs(solution)/np.linalg.norm(solution) < threshold, 0, solution)

    ##--------------------------------

    model_acceleration_func, valid_model = (
        xlsindy.dynamics_modeling.generate_acceleration_function(
            solution, 
            full_catalog,
            symbols_matrix,
            time_sym,
            lambdify_module="jax",
        )
    )
    model_dynamics_system = xlsindy.dynamics_modeling.dynamics_function_RK4_env(
        model_acceleration_func
    )

    ## Analysis of result


    simulation_dict["results"][args.get_uid()] = {}
    simulation_dict["results"][args.get_uid()]["algoritm"] = args.algorithm
    simulation_dict["results"][args.get_uid()]["noise_level"] = args.noise_level
    simulation_dict["results"][args.get_uid()]["optimization_function"] = args.optimization_function
    simulation_dict["results"][args.get_uid()]["random_seed"] = random_seed
    simulation_dict["results"][args.get_uid()]["catalog_len"] = extra_info["catalog_len"]
    simulation_dict["results"][args.get_uid()]["solution"] = solution
    simulation_dict["results"][args.get_uid()]["ideal_solution"] = extra_info["ideal_solution_vector"]

    if valid_model:


        # Acceleration comparison result

        model_dynamics_system = vmap(model_dynamics_system, in_axes=(1, 1), out_axes=1)
        
        model_coordinate = xlsindy.dynamics_modeling.vectorised_acceleration_generation(
            model_dynamics_system, imported_qpos, imported_qvel, imported_force
        )
        # Finally, select the columns of interest (e.g., every second column starting at index 1)
        model_acc = model_coordinate[:, 1::2]

        # Estimate of the variance between model and mujoco
        RMSE_acceleration = xlsindy.result_formatting.relative_mse(
            model_acc[3:-3], imported_qacc[3:-3]
        )

        simulation_dict["results"][args.get_uid()]["RMSE_acceleration"] = RMSE_acceleration
        print("estimate variance between mujoco and model is : ", RMSE_acceleration)

        # Trajectory comparison result

        model_acceleration_func_np, _ = (
            xlsindy.dynamics_modeling.generate_acceleration_function(
                solution, 
                full_catalog,
                symbols_matrix,
                time_sym,
                lambdify_module="numpy",
            )
        )

        trajectory_rng = np.random.default_rng(args.random_seed)

        # Initialise 
        simulation_time_g = np.empty((0,1))
        simulation_qpos_g = np.empty((0,num_coordinates))
        simulation_qvel_g = np.empty((0,num_coordinates))
        simulation_qacc_g = np.empty((0,num_coordinates))
        force_vector_g = np.empty((0,num_coordinates))

        if len(simulation_dict["generation_settings"]["initial_position"]) == 0:
            simulation_dict["generation_settings"]["initial_position"] = np.zeros((num_coordinates,2))

        for i in tqdm(range(simulation_dict["generation_settings"]["batch_number"]),desc="Generating batches", unit="batch"):

            # Initial condition
            initial_condition = np.array(simulation_dict["generation_settings"]["initial_position"]).reshape(num_coordinates,2) + extra_info["initial_condition"]

            if len(simulation_dict["generation_settings"]["initial_condition_randomness"]) == 1:
                initial_condition += trajectory_rng.normal(
                    loc=0, scale=simulation_dict["generation_settings"]["initial_condition_randomness"], size=initial_condition.shape
                )
            else:
                initial_condition += trajectory_rng.normal(
                    loc=0, scale=np.reshape(simulation_dict["generation_settings"]["initial_condition_randomness"], initial_condition.shape)
                )

            # Random controller initialisation. This is the only random place of the code Everything else is deterministic (except if non deterministic solver is used)
            forces_function = xlsindy.dynamics_modeling.optimized_force_generator(
                component_count=num_coordinates,
                scale_vector=simulation_dict["generation_settings"]["forces_scale_vector"],
                time_end=simulation_dict["generation_settings"]["max_time"],
                period=simulation_dict["generation_settings"]["forces_period"],
                period_shift=simulation_dict["generation_settings"]["forces_period_shift"],
                augmentations=10, # base is 40
                random_seed=[simulation_dict["generation_settings"]["random_seed"],i],
            )

            model_dynamics_system = xlsindy.dynamics_modeling.dynamics_function(model_acceleration_func_np,forces_function) 
            logger.info("Theorical initialized")
            try:
                simulation_time_m, phase_values = xlsindy.dynamics_modeling.run_rk45_integration(model_dynamics_system, initial_condition, simulation_dict["generation_settings"]["max_time"], max_step=0.005)
            except Exception as e:
                logger.error(f"An error occurred on the RK45 integration: {e}")
            logger.info("Theorical simulation done")

            simulation_qpos_m = phase_values[:, ::2]
            simulation_qvel_m = phase_values[:, 1::2]

            simulation_qacc_m = np.gradient(simulation_qvel_m, simulation_time_m, axis=0, edge_order=1)

            force_vector_m = forces_function(simulation_time_m.T).T

            if len(simulation_qvel_g) >0:
                simulation_time_m += np.max(simulation_time_g)
            # Concatenate the data
            simulation_time_g = np.concatenate((simulation_time_g, simulation_time_m.reshape(-1, 1)), axis=0)
            simulation_qpos_g = np.concatenate((simulation_qpos_g, simulation_qpos_m), axis=0)
            simulation_qvel_g = np.concatenate((simulation_qvel_g, simulation_qvel_m), axis=0)
            simulation_qacc_g = np.concatenate((simulation_qacc_g, simulation_qacc_m), axis=0)
            force_vector_g = np.concatenate((force_vector_g, force_vector_m), axis=0)

        # Generate the batch as a theory one

    if not valid_model:
        print("Skipped model verification, retrieval failed")

    simulation_dict = xlsindy.result_formatting.convert_to_lists(simulation_dict)
    
    print("print model ...")
    with open(args.experiment_file + ".json", "w") as file:
        json.dump(simulation_dict, file, indent=4)

    if args.print_graph and valid_model:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        for i in range(num_coordinates):
            plt.subplot(num_coordinates, 1, i + 1)
            plt.plot(imported_time, imported_qacc[:, i], label="mujoco")
            plt.plot(imported_time, model_acc[:, i], label="model")
            plt.title(f"acceleration of coordinate {i}")
            plt.legend()

        plt.savefig(f"{args.experiment_file}_{args.get_uid()}_acceleration_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(15, 12))

        # Plot position comparison
        for i in range(num_coordinates):
            plt.subplot(4, num_coordinates, i + 1)
            plt.plot(imported_time, imported_qpos[:, i], label="mujoco", alpha=0.7)
            plt.plot(simulation_time_g.flatten(), simulation_qpos_g[:, i], label="model", alpha=0.7)
            plt.title(f"Position coord {i}")
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Plot velocity comparison
        for i in range(num_coordinates):
            plt.subplot(4, num_coordinates, num_coordinates + i + 1)
            plt.plot(imported_time, imported_qvel[:, i], label="mujoco", alpha=0.7)
            plt.plot(simulation_time_g.flatten(), simulation_qvel_g[:, i], label="model", alpha=0.7)
            plt.title(f"Velocity coord {i}")
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Plot acceleration comparison
        for i in range(num_coordinates):
            plt.subplot(4, num_coordinates, 2 * num_coordinates + i + 1)
            plt.plot(imported_time, imported_qacc[:, i], label="mujoco", alpha=0.7)
            plt.plot(simulation_time_g.flatten(), simulation_qacc_g[:, i], label="model", alpha=0.7)
            plt.title(f"Acceleration coord {i}")
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Plot force comparison
        for i in range(num_coordinates):
            plt.subplot(4, num_coordinates, 3 * num_coordinates + i + 1)
            plt.plot(imported_time, imported_force[:, i], label="mujoco", alpha=0.7)
            plt.plot(simulation_time_g.flatten(), force_vector_g[:, i], label="model", alpha=0.7)
            plt.title(f"Force coord {i}")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{args.experiment_file}_{args.get_uid()}_full_dynamics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
