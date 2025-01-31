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

import matplotlib.pyplot as plt

from sympy import latex



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
    #print(args)

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


    # Random controller initialisation

    forces_function = xlsindy.dynamics_modeling.optimized_force_generator(
        component_count=num_coordinates,
        scale_vector=args.forces_scale_vector,
        time_end=args.max_time,
        period=args.forces_period,
        period_shift=args.forces_period_shift,
        augmentations=40,
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
            mujoco_qpos.append(data.qpos.copy())
            mujoco_qvel.append(data.qvel.copy())
            mujoco_qacc.append(data.qacc.copy())

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


    # from mujoco paradigm to xlsindy paradigm, TODO : maybe this should be coded in xlsindy_gen.py script inside another function
    mujoco_qpos = np.cumsum(mujoco_qpos, axis=1) + args.mujoco_angle_offset
    mujoco_qvel = np.cumsum(mujoco_qvel, axis=1)
    mujoco_qacc = np.cumsum(mujoco_qacc, axis=1)

    # Goes into the xlsindy regression

    nb_t = len(mujoco_time)

    surfacteur = len(full_catalog) * 10
    subsample = nb_t // surfacteur
    
    solution, exp_matrix, t_values_s,_ = xlsindy.simulation.execute_regression(
    time_values=mujoco_time,
    theta_values=mujoco_qpos,
    time_symbol=time_sym,
    symbol_matrix=symbols_matrix,
    catalog=full_catalog,
    external_force_function= forces_function, # super strange need to be reworked at least should propose an alternative with the vector of forces over time
    noise_level = 0,
    truncation_level = 5,
    subsample_rate = subsample,
    hard_threshold = 1e-3,
    velocity_values = mujoco_qvel,
    acceleration_values = mujoco_qacc,
    use_regression = True,
    apply_normalization = True,
    )

    # Compare the result with the base environment 
    modele_fit = xlsindy.catalog_gen.create_solution_expression(solution[:, 0], full_catalog, friction_count=num_coordinates)

    model_acceleration_func, _ = xlsindy.euler_lagrange.generate_acceleration_function(modele_fit, symbols_matrix, time_sym,fluid_forces=solution[-2:, 0])
    model_dynamics_system = xlsindy.dynamics_modeling.dynamics_function(model_acceleration_func, forces_function)

    model_acc = []

    for i in range(len(mujoco_time)): # skip start

        base_vector = np.ravel(np.column_stack((mujoco_qpos[i],mujoco_qvel[i])))

        model_acc+= [model_dynamics_system(mujoco_time[i],base_vector)]

    model_acc = np.array(model_acc)

    model_acc = model_acc[:,1::2]

    ## Numerical value as a result
    def relative_mse(X, Y):
        """Relative Mean Squared Error (scale-invariant)"""
        return np.sqrt(np.mean(((X - Y) / (np.max(X) - np.min(X))) ** 2)) * 100
    
    # Estimate of the variance between model and mujoco
    RMSE_acceleration = relative_mse(model_acc[3:-3],mujoco_qacc[3:-3])
    print("estimate variance between mujoco and model is : ",RMSE_acceleration)

    # Sparsity difference
    sparsity_reference = np.count_nonzero( extra_info["ideal_solution_vector"] )
    sparsity_model = np.count_nonzero(solution)

    sparsity_percentage = 100*(sparsity_model-sparsity_reference)/sparsity_reference
    sparsity_difference = abs(sparsity_model-sparsity_reference)
    print("sparsity difference percentage : ",sparsity_percentage)
    print("sparsity difference number : ",sparsity_difference)

    # model comparison RMSE
    def normalise_solution(X):
        """Normalise the solution vector because Lagrangian can be translated and multiply by singleton"""

        max_ind = np.argmax(np.abs(X))
        return X /X[max_ind]

    RMSE_model = relative_mse(normalise_solution(extra_info["ideal_solution_vector"]),normalise_solution(solution))
    print("RMSE model comparison : ",RMSE_model)

    print("Regression finished plotting in progress ... ")
    # Matplot plotting for the results

    fig, ax = plt.subplots()
    for i in range(num_coordinates):
        ax.plot(mujoco_time, mujoco_qacc[:,i],label=f"mujoco $\\ddot{{q}}_{i}$")
        ax.plot(mujoco_time, model_acc[:,i],label=f"model $\\ddot{{q}}_{i}$")
    ax.legend()


    fig, ax = plt.subplots()
    for i in range(num_coordinates):
        ax.plot(mujoco_time, mujoco_qpos[:,i],label=f"q_{i}")
    ax.legend()


    fig, ax = plt.subplots()    

    

    non_null_term = np.argwhere(solution != 0) 

    if extra_info is not None:
        ideal_solution_vector = extra_info["ideal_solution_vector"]

        non_null_term=np.unique(np.concat((non_null_term,np.argwhere(ideal_solution_vector != 0 )),axis=0),axis=0)

        ax.bar(np.arange(len(non_null_term)), normalise_solution(ideal_solution_vector)[*non_null_term.T], width=1, label="True Model")

    ax.bar(np.arange(len(non_null_term)), normalise_solution(solution)[*non_null_term.T], width=0.5, label="Model Found")
    ax.legend()

    def label_catalog(catalog,non_null_term):
        """Convert the catalog into label"""
        res=[]
        for index in non_null_term[:,0]:

            if index > len(catalog)-1:
                res+=[f"fluid forces $v_{{{index-len(catalog)}}}$"]
            else:
                res+= [f"${latex(catalog[index])}$"]
        return res

    catalog_string = label_catalog(full_catalog,non_null_term)

    # Set x-axis tick label
    ax.set_xticklabels(catalog_string)
    ax.set_xticks(np.arange(len(non_null_term)))
    ax.tick_params(labelrotation=90)
    ax.set_xlabel("Function from catalog")

    plt.tight_layout()
    plt.show()  