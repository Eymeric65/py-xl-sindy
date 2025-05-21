"""
This script is used to test the whole pipeline of the project.
An ideal trajectory is generated and then we run a regression tets

I have decided to make this script in order to confirm that the implementation of regression algorithm is correct.
The script is less bothersome than running the whole benchmarking pipeline (generating, regressing, storing, plotting,...)
"""

# tyro cly dependencies
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

@dataclass
class Args:
    ## Randomness
    random_seed: List[int] = field(default_factory=lambda: [2])
    """the random seed of the experiment (only used for force function)"""
    ## Data generation
    mujoco_generation:bool = True
    """if true generate the data using mujoco otherwise use the theoritical generator (default true)"""
    experiment_folder: str = "None"
    """the folder where the experiment data is stored : the mujoco environment.xml file and the xlsindy_gen.py script"""
    max_time: float = 10.0
    """the maximum time for the simulation"""
    forces_scale_vector: List[float] = field(default_factory=lambda: [])
    """the different scale for the forces vector to be applied, this can mimic an action mask over the system if some entry are 0"""
    forces_period: float = 3.0
    """the period for the forces function"""
    forces_period_shift: float = 0.5
    """the shift for the period of the forces function"""
    sample_number: int = 1000
    """the number of sample for the experiment (ten times the lenght of the catalog works well)"""
    ## Data regression
    optimization_function: str = "lasso_regression"
    """the regression function used in the regression"""
    algorithm: str = "xlsindy"
    """the name of the algorithm used (for the moment "xlsindy" and "sindy" are the only possible)"""
    noise_level: float = 0.0
    """the level of noise introduce in the experiment"""
    implicit_regression:bool = False
    """if true, use the implicit regression function"""

if __name__ == "__main__":

### ----------------------------------- Part 0 , load the variable -----------------------------------

    args = tyro.cli(Args)

    # CLI validation
    if args.forces_scale_vector == []:
        raise ValueError(
            "forces_scale_vector should be provided, don't hesitate to invoke --help"
        )
    if args.experiment_folder == "None":
        raise ValueError(
            "experiment_folder should be provided, don't hesitate to invoke --help"
        )
    else:  # import the xlsindy_back_script
        folder_path = os.path.join(os.path.dirname(__file__), args.experiment_folder)
        sys.path.append(folder_path)

        # import the xlsindy_gen.py script
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


        num_coordinates, time_sym, symbols_matrix, full_catalog, extra_info = (
            xlsindy_component(mode=args.algorithm, random_seed=args.random_seed)
        )

        # Mujoco environment path
        mujoco_xml = os.path.join(folder_path, "environment.xml")

    print("INFO : Cli validated")
    ## TODO add a check for the number of forces scale vector in the input

### ----------------------- Part 1, generate the data using Mujoco or theorical ----------------------
    
    # Random controller initialisation. This is the only random place of the code Everything else is deterministic (except if non deterministic solver is used)
    forces_function = xlsindy.dynamics_modeling.optimized_force_generator(
        component_count=num_coordinates,
        scale_vector=args.forces_scale_vector,
        time_end=args.max_time,
        period=args.forces_period,
        period_shift=args.forces_period_shift,
        augmentations=10, # base is 40
        random_seed=args.random_seed,
    )

    if args.mujoco_generation : # Mujoco Generation
        
        # initialize Mujoco environment and controller
        mujoco_model = mujoco.MjModel.from_xml_path(mujoco_xml)
        mujoco_data = mujoco.MjData(mujoco_model)

        simulation_time = []
        simulation_qpos = []
        simulation_qvel = []
        simulation_qacc = []
        force_vector = []

        def random_controller(forces_function):

            def ret(model, data):

                forces = forces_function(data.time)
                data.qfrc_applied = forces

                force_vector.append(forces.copy())

                simulation_time.append(data.time)
                simulation_qpos.append(data.qpos.copy())
                simulation_qvel.append(data.qvel.copy())
                simulation_qacc.append(data.qacc.copy())

            return ret

        mujoco.set_mjcb_control(
            random_controller(forces_function)
        )  # use this for the controller, could be made easier with using directly the data from mujoco.

        print("INFO : Mujoco initialized")
        while mujoco_data.time < args.max_time:
            mujoco.mj_step(mujoco_model, mujoco_data)

        print("INFO : Mujoco simulation done")

        # turn the result into a numpy array
        simulation_time = np.array(simulation_time)
        simulation_qpos = np.array(simulation_qpos)
        simulation_qvel = np.array(simulation_qvel)
        simulation_qacc = np.array(simulation_qacc)
        force_vector = np.array(force_vector)

        simulation_qpos, simulation_qvel, simulation_qacc, force_vector = mujoco_transform(
            simulation_qpos, simulation_qvel, simulation_qacc, force_vector
        )

    else: # Theorical generation
        
        model_acceleration_func, valid_model = xlsindy.dynamics_modeling.generate_acceleration_function(
        extra_info["ideal_solution_vector"],
        full_catalog,
        symbols_matrix,
        time_sym,
        lambdify_module="numpy"
        )

        model_dynamics_system = xlsindy.dynamics_modeling.dynamics_function(model_acceleration_func,forces_function) 
        print("INFO : Theorical initialized")
        try:
            simulation_time, phase_values = xlsindy.dynamics_modeling.run_rk45_integration(model_dynamics_system, extra_info["initial_condition"], args.max_time, max_step=0.05)
        except Exception as e:
            print(f"An error occurred on the RK45 integration: {e}")
        print("INFO : Theorical simulation done")

        simulation_qpos = phase_values[:, ::2]
        simulation_qvel = phase_values[:, 1::2]

        simulation_qacc = np.gradient(simulation_qvel, simulation_time, axis=0, edge_order=1)

        force_vector = forces_function(simulation_time.T).T

    raw_sample_number = len(simulation_time)
    print( f"Raw simulation len {raw_sample_number}")

    subsample = raw_sample_number // args.sample_number

    if subsample == 0 :
        subsample =1

    start_truncation = 2

    simulation_time = simulation_time[start_truncation::subsample]
    simulation_qpos = simulation_qpos[start_truncation::subsample]
    simulation_qvel = simulation_qvel[start_truncation::subsample]
    simulation_qacc = simulation_qacc[start_truncation::subsample]
    force_vector = force_vector[start_truncation::subsample]

### --------------------------- Part 2, Regresssion on the Data using xlsindy ------------------------

    regression_function = eval(f"xlsindy.optimization.{args.optimization_function}")

    rng = np.random.default_rng(args.random_seed)

    simulation_qpos += rng.normal(loc=0, scale=args.noise_level, size=simulation_qpos.shape)
    simulation_qvel += rng.normal(loc=0, scale=args.noise_level, size=simulation_qvel.shape)
    simulation_qacc += rng.normal(loc=0, scale=args.noise_level, size=simulation_qacc.shape)
    force_vector += rng.normal(
        loc=0, scale=args.noise_level, size=force_vector.shape
    )

    print("INFO : Regression function initialized")

    if args.implicit_regression:

        solution, exp_matrix, _ = xlsindy.simulation.regression_implicite(
            theta_values=simulation_qpos,
            velocity_values=simulation_qvel,
            acceleration_values=simulation_qacc,
            time_symbol=time_sym,
            symbol_matrix=symbols_matrix,
            catalog_repartition=full_catalog,
            hard_threshold=1e-3,
            regression_function=regression_function,
        )

    else:
        
        solution, exp_matrix, residuals,covariange_matrix = xlsindy.simulation.regression_explicite(
            theta_values=simulation_qpos,
            velocity_values=simulation_qvel,
            acceleration_values=simulation_qacc,
            time_symbol=time_sym,
            symbol_matrix=symbols_matrix,
            catalog_repartition=full_catalog,
            external_force=force_vector,
            hard_threshold=1e-3,
            regression_function=regression_function,
        )

    print("INFO : Regression done")

    # Not used right now
    model_acceleration_func, valid_model = (
    xlsindy.dynamics_modeling.generate_acceleration_function(
        solution,
        full_catalog,
        symbols_matrix,
        time_sym,
        lambdify_module="jax",
    )
    )

### -------------------------------------- Part 3, Result analisys -----------------------------------

    # Ideal Residulas (only for debugging purposes)
    exp_matrix_amp,forces_vector = xlsindy.optimization.amputate_experiment_matrix(exp_matrix,0)
    ideal_solution_amp = np.delete(extra_info["ideal_solution_vector"],0,axis=0)


    print("forces difference",np.linalg.norm(forces_vector.flatten()-force_vector.T.flatten()))

    ideal_residuals = forces_vector- exp_matrix_amp @ ideal_solution_amp
    print("Ideal Residuals : ", np.linalg.norm(ideal_residuals)/np.linalg.norm(forces_vector))

    # Residuals
    print("Residuals : ", np.linalg.norm(residuals)/np.linalg.norm(forces_vector))



    # Sparsity of the model 
    sparsity_reference = np.count_nonzero(extra_info["ideal_solution_vector"])
    sparsity_model = np.count_nonzero(solution)

    sparsity_percentage = (
        100 * (sparsity_model - sparsity_reference) / sparsity_reference
    )
    sparsity_difference = abs(sparsity_model - sparsity_reference)
    print("sparsity difference percentage : ", sparsity_percentage)
    print("sparsity difference number : ", sparsity_difference)

    # Model RMSE comparison

    non_null_term = np.argwhere(solution.flatten() != 0)

    non_null_term = np.unique(
        np.concat(
            (non_null_term, np.argwhere(extra_info["ideal_solution_vector"].flatten() != 0)), axis=0
        ),
        axis=0,
    )

    ideal_solution_norm = xlsindy.result_formatting.normalise_solution(
        extra_info["ideal_solution_vector"]
    )

    solution_norm = xlsindy.result_formatting.normalise_solution(solution)

    RMSE_model = xlsindy.result_formatting.relative_mse(
        solution_norm[non_null_term], ideal_solution_norm[non_null_term]
    )

    print("RMSE model comparison : ", RMSE_model)

### ----------------------------------------- Part 4, Render -----------------------------------------

    # Function catalog rendering
    fig, axs = plt.subplots(4, 1,figsize=(10, 10))
    fig.suptitle("Experiment Results")

    graph = {
        "model_comp":axs[0],
        "residuals":axs[1],
        "ideal_residuals":axs[2],
        "debug":axs[3]
    }

    graph["model_comp"].set_title("Model Comparison")

    graph["model_comp"].bar(
        np.arange(len(solution_norm)),
        solution_norm[:, 0],
        width=1,
        label="Found Model",
    )

    bar_height_found = np.abs(solution) / np.max(np.abs(solution))
    graph["model_comp"].bar(
        np.arange(len(ideal_solution_norm)),
        ideal_solution_norm[:, 0],
        width=0.5,
        label="True Model",
    )

    graph["model_comp"].legend()

    graph["residuals"].set_title("Residuals")

    res = residuals.reshape(num_coordinates,-1).T
    ideal_res = ideal_residuals.reshape(num_coordinates,-1).T

    for i in range(num_coordinates):

        graph["residuals"].plot(
            simulation_time,
            res[:,i],
            label=f"Residuals q{i}",
        )
        graph["ideal_residuals"].plot(
            simulation_time,
            ideal_res[:,i],
            label=f"Ideal Residuals q{i}",
        )



    graph["residuals"].legend()
    graph["ideal_residuals"].legend()

    graph["debug"].plot(forces_vector)
    graph["debug"].plot(force_vector.T.flatten())

    fig.savefig("experiment_result.svg")
    plt.show()


