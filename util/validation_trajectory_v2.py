"""
This script generates a validation trajectory from mujoco and compares it with trajectories from all the retrieved models.
"""

from dataclasses import dataclass, field
from typing import List
import tyro

import sys
import os
import importlib

import mujoco
import numpy as np 
import xlsindy

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import xlsindy.result_formatting

from datetime import datetime
import json

def auto_zoom_viewport(ax, x, y, margin=0.1):
    """
    Automatically zooms the viewport for a given plot.

    Parameters:
        ax (matplotlib.axes.Axes): The Axes object to modify.
        x (array-like): X data values.
        y (array-like): Y data values.
        margin (float): Additional margin (percentage of y-range) to add around the min/max y-values.
    """
    y_min, y_max = np.min(y), np.max(y)
    y_range = y_max - y_min
    y_margin = y_range * margin

    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

@dataclass
class Args:
    experiment_file: str = "None"
    """the experiment file (without extension)"""
    max_time: float = 10.0
    """the maximum time for the validation simulation"""
    random_seed: List[int] = field(default_factory=lambda: [])
    """the random seed to add for the force function (only used for force function)"""
    plot: bool = True
    """if True, plot the different validation trajectories"""

def convert_numbers(data):
    if isinstance(data, dict):
        return {key: convert_numbers(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numbers(item) for item in data]
    else:
        try:
            if "." in str(data):
                return float(data)
            return int(data)
        except (ValueError, TypeError):
            return data

if __name__ == "__main__":

    args = tyro.cli(Args)

    if args.experiment_file == "None":
        raise ValueError("experiment_file should be provided, don't hesitate to invoke --help")

    with open(args.experiment_file + ".json", 'r') as json_file:
        simulation_dict = json.load(json_file)
        simulation_dict = convert_numbers(simulation_dict)

    folder_path = os.path.join(os.path.dirname(__file__), "mujoco_align_data/" + simulation_dict["input"]["experiment_folder"])
    sys.path.append(folder_path)

    # Import the xlsindy_gen.py script
    xlsindy_gen = importlib.import_module("xlsindy_gen")
    try:
        xlsindy_component = eval("xlsindy_gen.xlsindy_component")
    except AttributeError:
        raise AttributeError("xlsindy_gen.py should contain a function named xlsindy_component")
    try:
        mujoco_transform = xlsindy_gen.mujoco_transform
    except AttributeError:
        mujoco_transform = None
    try:
        forces_wrapper = xlsindy_gen.forces_wrapper
    except AttributeError:
        forces_wrapper = None

    num_coordinates, time_sym, symbols_matrix, catalog_repartition, extra_info = xlsindy_component()

    forces_function = xlsindy.dynamics_modeling.optimized_force_generator(
        component_count=num_coordinates,
        scale_vector=simulation_dict["input"]["forces_scale_vector"],
        time_end=args.max_time,
        period=simulation_dict["input"]["forces_period"],
        period_shift=simulation_dict["input"]["forces_period_shift"],
        augmentations=40,
        random_seed=simulation_dict["input"]["random_seed"] + args.random_seed
    )

    mujoco_xml = os.path.join(folder_path, "environment.xml")
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

    mujoco.set_mjcb_control(random_controller(forces_function))

    # Run Mujoco simulation
    while mujoco_data.time < args.max_time:
        mujoco.mj_step(mujoco_model, mujoco_data)

    # Convert lists to numpy arrays and transform them
    mujoco_time = np.array(mujoco_time)
    mujoco_qpos = np.array(mujoco_qpos)
    mujoco_qvel = np.array(mujoco_qvel)
    mujoco_qacc = np.array(mujoco_qacc)
    force_vector = np.array(force_vector)

    if mujoco_transform is not None:
        mujoco_qpos, mujoco_qvel, mujoco_qacc, force_vector = mujoco_transform(mujoco_qpos, mujoco_qvel, mujoco_qacc, force_vector)

    # Collect Mujoco results as the first experiment
    exp_name = ["mujoco"]
    exp_time = [mujoco_time]
    exp_qpos = [mujoco_qpos]
    exp_qvel = [mujoco_qvel]
    exp_qacc = [mujoco_qacc]
    exp_info = [{"name": "mujoco", "algoritm": "mujoco", "optimization_function": None, "noise": 0}]

    # Process additional experiments from simulation_dict
    sim_key = [key for key in simulation_dict.keys() if key.startswith("result__")]
    for sim in sim_key:
        exp_dict = simulation_dict[sim]
        if "solution" in exp_dict:
            # Re-run component with given parameters
            _, _, _, catalog_repartition, extra_info = xlsindy_component(
                mode=exp_dict["algoritm"],
                random_seed=exp_dict["random_seed"],
                sindy_catalog_len=exp_dict["catalog_len"]
            )
            solution = np.array(exp_dict["solution"])
            model_acceleration_func, valid_model = xlsindy.dynamics_modeling.generate_acceleration_function(
                solution, catalog_repartition, symbols_matrix, time_sym, lambdify_module="jax"
            )
            model_dynamics_system = xlsindy.dynamics_modeling.dynamics_function(
                model_acceleration_func, forces_wrapper(forces_function)
            )
            time_values, phase_values = xlsindy.dynamics_modeling.run_rk45_integration(
                model_dynamics_system, extra_info["initial_condition"], args.max_time, max_step=0.1
            )
            theta_values = phase_values[:, ::2]
            velocity_values = phase_values[:, 1::2]
            acceleration_values = np.gradient(velocity_values, time_values, axis=0, edge_order=1)

            exp_name.append(sim)
            exp_time.append(time_values)
            exp_qpos.append(theta_values)
            exp_qvel.append(velocity_values)
            exp_qacc.append(acceleration_values)
            exp_info.append({
                "name": sim,
                "algoritm": exp_dict.get("algoritm", "default"),
                "optimization_function": exp_dict.get("optimization_function", "default"),
                "noise": exp_dict.get("noise_level", 0)
            })

    # --- Color assignment based on (algoritm, optimization_function) ---
    candidate_cmaps = ["Blues", "Oranges", "Greens", "Reds", "Purples", "Greys", "YlOrBr", "YlGn"]
    group_color_map = {}
    group_indices = {}
    for i, info in enumerate(exp_info):
        if info["algoritm"] == "mujoco":
            continue
        key = (info["algoritm"], info["optimization_function"])
        group_indices.setdefault(key, []).append(i)
    for key, indices in group_indices.items():
        indices.sort(key=lambda idx: exp_info[idx]["noise"])
        group_indices[key] = indices
        if key not in group_color_map:
            cmap_name = candidate_cmaps.pop(0) if candidate_cmaps else "viridis"
            group_color_map[key] = cmap_name

    def get_color(i):
        info = exp_info[i]
        key = (info["algoritm"], info["optimization_function"])
        indices = group_indices.get(key, [])
        if i in indices:
            pos = indices.index(i)
            n = len(indices)
            shade = 0.4 + (0.5 * pos / (n - 1)) if n > 1 else 0.7
            cmap = plt.get_cmap(group_color_map[key])
            return cmap(shade)
        return None

    # --- Plotting: one figure with three subplots (qpos, qvel, qacc) ---
    if args.plot:
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        for i, name in enumerate(exp_name):
            t    = exp_time[i]
            qpos = exp_qpos[i]
            qvel = exp_qvel[i]
            qacc = exp_qacc[i]

            if exp_info[i]["algoritm"] == "mujoco":
                line_style = '--'
                color = 'black'
            else:
                line_style = '-'
                color = get_color(i)

            # Only add a label if noise == 0 (and only for the first coordinate plotted)
            label = f'{name} ({exp_info[i]["algoritm"]})' if exp_info[i]["noise"] == 0 else None

            # Plot qpos in the first subplot
            if qpos.ndim == 1 or qpos.shape[1] == 1:
                axs[0].plot(t, qpos if qpos.ndim == 1 else qpos[:, 0],
                            linestyle=line_style, color=color, label=label)
            else:
                for j in range(qpos.shape[1]):
                    current_label = label if j == 0 else None
                    axs[0].plot(t, qpos[:, j],
                                linestyle=line_style, color=color, label=current_label)

            # Plot qvel in the second subplot
            if qvel.ndim == 1 or qvel.shape[1] == 1:
                axs[1].plot(t, qvel if qvel.ndim == 1 else qvel[:, 0],
                            linestyle=line_style, color=color, label=label)
            else:
                for j in range(qvel.shape[1]):
                    current_label = label if j == 0 else None
                    axs[1].plot(t, qvel[:, j],
                                linestyle=line_style, color=color, label=current_label)

            # Plot qacc in the third subplot
            if qacc.ndim == 1 or qacc.shape[1] == 1:
                axs[2].plot(t, qacc if qacc.ndim == 1 else qacc[:, 0],
                            linestyle=line_style, color=color, label=label)
            else:
                for j in range(qacc.shape[1]):
                    current_label = label if j == 0 else None
                    axs[2].plot(t, qacc[:, j],
                                linestyle=line_style, color=color, label=current_label)

        axs[0].set_ylabel('qpos')
        axs[1].set_ylabel('qvel')
        axs[2].set_ylabel('qacc')
        axs[2].set_xlabel('Time')

        # Render legends only if there is at least one label
        for ax in axs:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()

        plt.tight_layout()
        plt.show()
