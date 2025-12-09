"""
VERIFIED FOR VERSION 2.1.3

Minimal example script to demonstrate functionality of py-xl-sindy package.

This script generates synthetic data from a mujoco simulation of a cartpole.
It then uses the py-xl-sindy package to identify a model from the data.

The identified model is then simulated and compared to the original data.

This minimal example is a bit more modular allowing :
- two types of catalog (Lagrange and Classical)
- different regression methods
- different system and friction

"""

import numpy as np
import sympy as sp 
import xlsindy
from generate_trajectory import generate_mujoco_trajectory,generate_theoretical_trajectory
from xlsindy.logger import setup_logger
from xlsindy.optimization import lasso_regression
import time
import matplotlib.pyplot as plt
import os
import sys 
import importlib

def mujoco_transform(pos, vel, acc):

    return -pos, -vel, -acc

def inverse_mujoco_transform(pos, vel, acc):
    if acc is not None:
        return -pos, -vel, -acc
    else:
        return -pos, -vel, None
    
logger = setup_logger(__name__,level="DEBUG")

if __name__=="__main__":
    
    num_coordinates = 2
    random_seed = [0]
    batch_number = 5
    max_time = 10.0
    initial_position = np.array([0.0, 0.0,0.0,0.,0.,0.])
    initial_condition_randomness = np.array([0.1])
    forces_scale_vector = np.array([.5, 0.0, 0.0])
    forces_period = 3.0
    forces_period_shift = 0.5
    # ratio of catalog lenght to sample size
    data_ratio = 10
    validation_time = 30.0
    noise_level = 0.01
    experiment_system = "cart_pole_double"
    damping_coefficients = np.array([-0, -0,-0])
    catalog_lenght = 20
    # Three mode available : "xlsindy", "mixed", "sindy"
    simulation_mode="sindy"

    ## 0. Import the metacode for the chosen experiment system

    # Add the experiment system folder to the path
    folder_path = os.path.join(os.path.dirname(__file__),"mujoco_align_data", experiment_system)
    sys.path.append(folder_path)
    # Import the metacode xlsindy_gen from the experiment system folder
    xlsindy_gen = importlib.import_module("xlsindy_gen")

    try:
        xlsindy_component = xlsindy_gen.xlsindy_component
    except AttributeError:
        raise AttributeError(
            "xlsindy_gen.py should contain a function named xlsindy_component"
        )

    try:
        mujoco_transform = xlsindy_gen.mujoco_transform
    except AttributeError:
        mujoco_transform = None

    try:
        inverse_mujoco_transform = xlsindy_gen.inverse_mujoco_transform
    except AttributeError:
        inverse_mujoco_transform = None

    num_coordinates, time_sym, symbols_matrix, catalog, xml_content, extra_info = (
        xlsindy_component( random_seed=random_seed, damping_coefficients=damping_coefficients,mode=simulation_mode,sindy_catalog_len=catalog_lenght)  # type: ignore
    )

    ideal_solution_vector = extra_info.get("ideal_solution_vector", None)
    if ideal_solution_vector is None:
        raise ValueError(
            "xlsindy_gen.py should return an ideal_solution_vector in the extra_info dictionary"
        )

    ## End import metacode 

    rng = np.random.default_rng(random_seed)


    # Create the mujoco trajectory
    (simulation_time_t, 
    simulation_qpos_t, 
    simulation_qvel_t, 
    simulation_qacc_t, 
    force_vector_t,
    _) = generate_mujoco_trajectory(
        num_coordinates,
        initial_position,
        initial_condition_randomness,
        random_seed,
        batch_number,
        max_time,
        xml_content,
        forces_scale_vector,
        forces_period,
        forces_period_shift,
        mujoco_transform,
        inverse_mujoco_transform
    )

    # Add noise
    simulation_qpos_t += rng.normal(loc=0, scale=noise_level, size=simulation_qpos_t.shape)*np.linalg.norm(simulation_qpos_t)/simulation_qpos_t.shape[0]
    simulation_qvel_t += rng.normal(loc=0, scale=noise_level, size=simulation_qvel_t.shape)*np.linalg.norm(simulation_qvel_t)/simulation_qvel_t.shape[0]
    simulation_qacc_t += rng.normal(loc=0, scale=noise_level, size=simulation_qacc_t.shape)*np.linalg.norm(simulation_qacc_t)/simulation_qacc_t.shape[0]
    force_vector_t += rng.normal(loc=0, scale=noise_level, size=force_vector_t.shape)*np.linalg.norm(force_vector_t)/force_vector_t.shape[0]

    # Use a fixed ratio of the data in respect with catalog size
    catalog_size = catalog.catalog_length
    data_ratio = data_ratio
    
    # Sample uniformly n samples from the imported arrays
    n_samples = int(catalog_size * data_ratio)
    total_samples = simulation_qpos_t.shape[0]

    if n_samples < total_samples:

        # Evenly spaced sampling (deterministic, uniform distribution)
        sample_indices = np.linspace(0, total_samples - 1, n_samples, dtype=int)
        
        # Apply sampling to all arrays
        simulation_qpos_t = simulation_qpos_t[sample_indices]
        simulation_qvel_t = simulation_qvel_t[sample_indices]
        simulation_qacc_t = simulation_qacc_t[sample_indices]
        force_vector_t = force_vector_t[sample_indices]
        
        logger.info(f"Sampled {n_samples} points uniformly from {total_samples} total samples")
    else:
        logger.info(f"Using all {total_samples} samples (requested {n_samples})")

    logger.info("Starting mixed regression")

    start_time = time.perf_counter()

    solution, exp_matrix = xlsindy.simulation.regression_mixed(
        theta_values=simulation_qpos_t,
        velocity_values=simulation_qvel_t,
        acceleration_values=simulation_qacc_t,
        time_symbol=time_sym,
        symbol_matrix=symbols_matrix,
        catalog_repartition=catalog,
        external_force=force_vector_t,
        regression_function=lasso_regression,
        noise_level=noise_level,
        pre_knowledge_indices=np.nonzero(forces_scale_vector)[0],
        pre_knowledge_type="external_forces_manual"
    )

    end_time = time.perf_counter()

    regression_time = end_time - start_time

    logger.info(f"Regression completed in {end_time - start_time:.2f} seconds")

    # Use the result to generate validation trajectory

    threshold = 1e-2  # Adjust threshold value as needed
    solution = np.where(np.abs(solution)/np.linalg.norm(solution) < threshold, 0, solution)

    model_acceleration_func_np, valid_model = (
        xlsindy.dynamics_modeling.generate_acceleration_function(
            solution, 
            catalog,
            symbols_matrix,
            time_sym,
            lambdify_module="numpy",
        )
    )



    (simulation_time_v, 
    simulation_qpos_v, 
    simulation_qvel_v, 
    simulation_qacc_v, 
    force_vector_v,
    _) = generate_mujoco_trajectory(
        num_coordinates,
        initial_position,
        initial_condition_randomness,
        random_seed+[0], # Ensure same seed as for data generation
        1,
        validation_time,
        xml_content,
        forces_scale_vector,
        forces_period,
        forces_period_shift,
        mujoco_transform,
        inverse_mujoco_transform
    )

    if valid_model:

        (simulation_time_vs, 
         simulation_qpos_vs, 
         simulation_qvel_vs, 
         simulation_qacc_vs, 
         force_vector_vs,
         _) = generate_theoretical_trajectory(
             num_coordinates,
             initial_position,
             initial_condition_randomness,
             random_seed+[0], # Ensure same seed as for data generation
             1,
             validation_time,
             solution,
             catalog,
             time_sym,
             symbols_matrix,
             forces_scale_vector,
             forces_period,
             forces_period_shift
         )
    else:
        logger.warning("Model is not valid, skipping validation trajectory generation")
    
    # Create a figure with 4 subplots stacked vertically
    fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('Trajectory Comparison: Mujoco vs. Theoretical', fontsize=16)

    # --- 1. Plot Position Data ---
    axes[0].plot(simulation_time_v, simulation_qpos_v, label='Mujoco Simulation')
    if valid_model:
        axes[0].plot(simulation_time_vs, simulation_qpos_vs, label='Theoretical Simulation', linestyle='--')
    axes[0].set_title('Position vs. Time')
    axes[0].set_ylabel('Position')
    axes[0].legend()
    axes[0].grid(True)

    # --- 2. Plot Velocity Data ---
    axes[1].plot(simulation_time_v, simulation_qvel_v, label='Mujoco Simulation')
    if valid_model:
        axes[1].plot(simulation_time_vs, simulation_qvel_vs, label='Theoretical Simulation', linestyle='--')
    axes[1].set_title('Velocity vs. Time')
    axes[1].set_ylabel('Velocity')
    axes[1].legend()
    axes[1].grid(True)

    # --- 3. Plot Acceleration Data ---
    axes[2].plot(simulation_time_v, simulation_qacc_v, label='Mujoco Simulation')
    if valid_model:
        axes[2].plot(simulation_time_vs, simulation_qacc_vs, label='Theoretical Simulation', linestyle='--')
    axes[2].set_title('Acceleration vs. Time')
    axes[2].set_ylabel('Acceleration')
    axes[2].legend()
    axes[2].grid(True)

    # --- 4. Plot Force Data ---
    axes[3].plot(simulation_time_v, force_vector_v, label='Mujoco Force')
    if valid_model:
        axes[3].plot(simulation_time_vs, force_vector_vs, label='Theoretical Force', linestyle='--')
    axes[3].set_title('Force vs. Time')
    axes[3].set_ylabel('Force')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend()
    axes[3].grid(True)

    # Improve layout to prevent labels from overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to make room for the suptitle

    # Display the plots
    plt.savefig("trajectory_comparison.png")


        
        
