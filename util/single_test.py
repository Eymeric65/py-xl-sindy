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

@dataclass
class Args:
    ## Randomness
    random_seed: List[int] = field(default_factory=lambda: [2])
    """the random seed of the experiment (only used for force function)"""
    ## Data generation
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

### ------------------------------ Part 1, generate the data using Mujoco ----------------------------

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

        try:
            forces_wrapper = xlsindy_gen.forces_wrapper
        except AttributeError:
            forces_wrapper = None

        num_coordinates, time_sym, symbols_matrix, full_catalog, extra_info = (
            xlsindy_component(mode=args.algorithm, random_seed=args.random_seed)
        )

        # Mujoco environment path
        mujoco_xml = os.path.join(folder_path, "environment.xml")

    ## TODO add a check for the number of forces scale vector in the input
    
    # Random controller initialisation. This is the only random place of the code Everything else is deterministic (except if non deterministic solver is used)
    forces_function = xlsindy.dynamics_modeling.optimized_force_generator(
        component_count=num_coordinates,
        scale_vector=args.forces_scale_vector,
        time_end=args.max_time,
        period=args.forces_period,
        period_shift=args.forces_period_shift,
        augmentations=40,
        random_seed=args.random_seed,
    )

    # initialize Mujoco environment and controller

    mujoco_model = mujoco.MjModel.from_xml_path(mujoco_xml)
    mujoco_data = mujoco.MjData(mujoco_model)

    mujoco_time = []
    mujoco_qpos = []
    mujoco_qvel = []
    mujoco_qacc = []
    force_vector = []

    def random_controller(forces_function):

        def ret(model, data):

            forces = forces_function(data.time)
            data.qfrc_applied = forces

            force_vector.append(forces.copy())

            mujoco_time.append(data.time)
            mujoco_qpos.append(data.qpos.copy())
            mujoco_qvel.append(data.qvel.copy())
            mujoco_qacc.append(data.qacc.copy())

        return ret

    mujoco.set_mjcb_control(
        random_controller(forces_function)
    )  # use this for the controller, could be made easier with using directly the data from mujoco.

    while mujoco_data.time < args.max_time:
        mujoco.mj_step(mujoco_model, mujoco_data)

    # turn the result into a numpy array
    mujoco_time = np.array(mujoco_time)
    mujoco_qpos = np.array(mujoco_qpos)
    mujoco_qvel = np.array(mujoco_qvel)
    mujoco_qacc = np.array(mujoco_qacc)
    force_vector = np.array(force_vector)

    mujoco_qpos, mujoco_qvel, mujoco_qacc, force_vector = mujoco_transform(
        mujoco_qpos, mujoco_qvel, mujoco_qacc, force_vector
    )

    raw_sample_number = len(mujoco_time)

    subsample = raw_sample_number // args.sample_number
    start_truncation = 2

    mujoco_time = mujoco_time[start_truncation::subsample]
    mujoco_qpos = mujoco_qpos[start_truncation::subsample]
    mujoco_qvel = mujoco_qvel[start_truncation::subsample]
    mujoco_qacc = mujoco_qacc[start_truncation::subsample]
    force_vector = force_vector[start_truncation::subsample]

### --------------------------- Part 2, Regresssion on the Data using xlsindy ------------------------

    regression_function = eval(f"xlsindy.optimization.{args.optimization_function}")

    rng = np.random.default_rng(args.random_seed)

    mujoco_qpos += rng.normal(loc=0, scale=args.noise_level, size=mujoco_qpos.shape)
    mujoco_qvel += rng.normal(loc=0, scale=args.noise_level, size=mujoco_qvel.shape)
    mujoco_qacc += rng.normal(loc=0, scale=args.noise_level, size=mujoco_qacc.shape)
    force_vector += rng.normal(
        loc=0, scale=args.noise_level, size=force_vector.shape
    )

    if args.implicit_regression:

        solution, exp_matrix, _ = xlsindy.simulation.regression_implicite(
            theta_values=mujoco_qpos,
            velocity_values=mujoco_qvel,
            acceleration_values=mujoco_qacc,
            time_symbol=time_sym,
            symbol_matrix=symbols_matrix,
            catalog_repartition=full_catalog,
            hard_threshold=1e-3,
            regression_function=regression_function,
        )

    else:
        
        solution, exp_matrix, residuals,covariange_matrix = xlsindy.simulation.regression_explicite(
            theta_values=mujoco_qpos,
            velocity_values=mujoco_qvel,
            acceleration_values=mujoco_qacc,
            time_symbol=time_sym,
            symbol_matrix=symbols_matrix,
            catalog_repartition=full_catalog,
            external_force=force_vector,
            hard_threshold=1e-3,
            regression_function=regression_function,
        )

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

### -------------------------------------- Part 3, Result analysys -----------------------------------

    # Ideal Residulas (only for debugging purposes)
    exp_matrix_amp,forces_vector = xlsindy.optimization.amputate_experiment_matrix(exp_matrix,0)
    ideal_solution_amp = np.delete(extra_info["ideal_solution_vector"],0,axis=0)

    ideal_residuals = forces_vector- exp_matrix_amp @ ideal_solution_amp
    print("Ideal Residuals : ", np.linalg.norm(ideal_residuals))    

    # Residuals
    print("Residuals : ", np.linalg.norm(residuals))



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

    ideal_solution_norm_nn = xlsindy.result_formatting.normalise_solution(
        extra_info["ideal_solution_vector"]
    )[non_null_term]

    solution_norm_nn = xlsindy.result_formatting.normalise_solution(solution)[
        non_null_term
    ]

    RMSE_model = xlsindy.result_formatting.relative_mse(
        solution_norm_nn, ideal_solution_norm_nn
    )

    print("RMSE model comparison : ", RMSE_model)