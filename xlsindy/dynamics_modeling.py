"""
This module contain every function in order to integrate and generate the dynamic function.

"""


import numpy as np
from .render import print_progress
from scipy import interpolate
from scipy.integrate import RK45
from typing import List, Callable, Dict

import jax.numpy as jnp


def dynamics_function(
    acceleration_function: Callable[[np.ndarray], np.ndarray],
    external_forces: Callable[[np.ndarray], np.ndarray],
) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Transforms the acceleration function into something understandable by usual integration method.

    The acceleration function ( output of euler_lagrange.generate_acceleration_function() ) takes as input a numerical symbol matrix.
    It is not suitable for the majority of the integration function that need to take as input (t,[q0,q0_d,...,qn,qn_d]) and output (q0_d,q0_dd,...,qn_d,qn_dd).

    Args:
        acceleration_function (function): Array of functions representing accelerations.
        external_forces (function): Function returning external forces at time `t`.

    Returns:
        function: Dynamics function compatible with classical integration solver.
    """

    def func(t, state):
        state = np.reshape(state, (-1, 2))
        state_transposed = np.transpose(state)

        # Prepare input matrix for dynamics calculations as a numerical symbol matrix
        input_matrix = np.zeros(
            (state_transposed.shape[0] + 2, state_transposed.shape[1])
        )
        input_matrix[1:3, :] = state_transposed
        input_matrix[0, :] = external_forces(t)

        # Create the result use the same size as before
        result = np.zeros(state.shape)
        result[:, 0] = state[:, 1]
        result[:, 1] = acceleration_function(input_matrix)[:, 0]
        return np.reshape(result, (-1,))

    return func

def dynamics_function_fixed_external(
    acceleration_function: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], Callable[[float, np.ndarray], np.ndarray]]:
    """
    Transforms the acceleration function into something understandable by usual integration method. (will be deprecated in v2.0)

    The acceleration function ( output of euler_lagrange.generate_acceleration_function() ) takes as input a numerical symbol matrix.
    It is not suitable for the majority of the integration function that need to take as input (t,[q0,q0_d,...,qn,qn_d]) and output (q0_d,q0_dd,...,qn_d,qn_dd).
    Due to the fact that it has been modified to perform one time step RK4, it introduces slight overhead

    Args:
        acceleration_function (function): Array of functions representing accelerations.

    Returns:
        function: return a function that take in input fixed force vector and return a Dynamics function compatible with classical integration solver.
    """

    def ret_func(forces):

        def func(t, state):
            state = np.reshape(state, (-1, 2))
            state_transposed = np.transpose(state)

            # Prepare input matrix for dynamics calculations as a numerical symbol matrix
            input_matrix = np.zeros(
                (state_transposed.shape[0] + 2, state_transposed.shape[1])
            )
            input_matrix[1:3, :] = state_transposed
            input_matrix[0, :] = forces

            # Create the result use the same size as before
            result = np.zeros(state.shape)
            result[:, 0] = state[:, 1]
            result[:, 1] = acceleration_function(input_matrix)[:, 0]
            return np.reshape(result, (-1,))
        
        return func

    return ret_func

def dynamics_function_RK4_env(
    acceleration_function: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], Callable[[float, np.ndarray], np.ndarray]]:
    """
    Transforms the acceleration function into something compatible for the RK4 integration method. (into reinforcement-learning-sindy )

    The acceleration function ( output of euler_lagrange.generate_acceleration_function() ) takes as input a numerical symbol matrix.
    It is not suitable for the integration function of RK4 environment that need to take as input ([q0,q0_d,...,qn,qn_d],[f0,...,fn]) and output (q0_d,q0_dd,...,qn_d,qn_dd).
    Use jax jnp instead of numpy for better performance. need to be use in accordance with euler_lagrange.generate_acceleration_function(lambdify_module="jax")

    Args:
        acceleration_function (function): Array of functions representing accelerations.

    Returns:
        function: return a function that take in input fixed force vector and forces and return a Dynamics function compatible with classical integration solver.
    """

    def ret_func(state,forces):

        #print("state shape ",state.shape)
        #print("forces shape ",forces.shape)

        state = jnp.reshape(state,(-1, 2))

        state_transposed = jnp.transpose(state)

        # Prepare input matrix for dynamics calculations as a numerical symbol matrix
        #input_matrix = jnp.zeros(
        #   (state_transposed.shape[0] + 2, state_transposed.shape[1])
        #)
        #input_matrix[1:3, :] = state_transposed
        #input_matrix[0, :] = forces

        input_matrix = jnp.concatenate([jnp.reshape(forces,(1,-1)),state_transposed,jnp.zeros((1,forces.shape[0]))],axis=0)

        #print("input_matrix shape ",input_matrix.shape)

        # Create the result use the same size as before
        #result = jnp.zeros(state.shape)
        #result[:, 0] = state[:, 1]
        #result[:, 1] = acceleration_function(input_matrix)[:, 0]

        result = jnp.concatenate([jnp.reshape(state[:, 1],(-1,1)),jnp.reshape(acceleration_function(input_matrix)[:, 0],(-1,1))],axis=1)
        #print("result shape ",result.shape)
        return jnp.reshape(result, (-1,))       
    
    return ret_func


def run_rk45_integration(
    dynamics: Callable[[float, np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    time_end: float,
    max_step: float = 0.05,
) -> List[np.ndarray]:
    """
    Runs an RK45 integration on a dynamics model.

    Args:
        dynamics (function): Dynamics function for integration.
        initial_state (np.ndarray): Initial state of the system.
        time_end (float): End time for the integration.
        max_step (float, optional): Maximum step size for the integration. Defaults to 0.05.

    Returns:
        tuple: Arrays of time values and states.
    """
    initial_state_flat = np.reshape(initial_state, (-1,))
    model = RK45(dynamics, 0, initial_state_flat, time_end, max_step, 0.001, np.e**-6)

    time_values = []
    state_values = []

    try:
        while model.status != "finished":
            for _ in range(200):
                if model.status != "finished":
                    model.step()
                    time_values.append(model.t)
                    state_values.append(model.y)
            print_progress(model.t, time_end)

    except RuntimeError:
        print("RuntimeError in RK45 integration")

    return np.array(time_values), np.array(state_values)


def generate_random_force(
    time_end: float,
    current_augmentation: int,
    target_augmentation: int,
    period_initial: float,
    period_shift_initial: float,
    component_count: int,
) -> Callable[[float], np.ndarray]:
    """
    Recursively generates a random external force function with specified augmentations.

    Parameters:
        time_end (float): End time for the generated force.
        current_augmentation (int): Current augmentation step in recursion.
        target_augmentation (int): Target augmentation level.
        period_initial (float): Initial period for the force oscillations.
        period_shift_initial (float): Initial shift for random variations in the period.
        component_count (int): Number of components in the force vector.

    Returns:
        Callable[[float], np.ndarray]: A function that generates a random force vector over time.
    """
    if current_augmentation == target_augmentation:
        return lambda t: t * np.zeros((component_count, 1))

    # Recursive call to generate the baseline force function
    baseline_force_function = generate_random_force(
        time_end,
        current_augmentation + 1,
        target_augmentation,
        period_initial,
        period_shift_initial,
        component_count,
    )

    # Calculate period, shift, and variance for the current augmentation level
    multiplier = target_augmentation - current_augmentation
    period = period_initial / multiplier
    period_shift = period_shift_initial / multiplier
    variance = 1 / multiplier

    # Generate time points with random shifts
    time_points = np.arange(0, time_end + period, period)
    time_points += (np.random.random(len(time_points)) - 0.5) * 2 * period_shift

    # Generate random force values with variance
    force_values = (
        np.random.random_sample((component_count, len(time_points))) * 2 - 1
    ) * variance
    force_values += baseline_force_function(time_points)
    force_values /= np.std(force_values)  # Normalize to standard deviation of 1

    # Create an interpolating function to return force values over time
    return interpolate.CubicSpline(time_points, force_values, axis=1)


def optimized_force_generator(
    component_count: int,
    scale_vector: np.ndarray,
    time_end: float,
    period: float,
    period_shift: float,
    augmentations: int = 50,
) -> Callable[[float], np.ndarray]:
    """
    Generates an optimized force function, applying a scale vector to the generated force.

    Parameters:
        component_count (int): Number of components in the force vector.
        scale_vector (np.ndarray): Scaling factors for each component.
        time_end (float): End time for the generated force.
        period (float): Base period for force oscillations.
        period_shift (float): Base shift applied to the period for randomness.
        augmentations (int): Number of augmentations in the recursive force generation.

    Returns:
        Callable[[float], np.ndarray]: A function that returns the optimized force at time `t`.
    """
    scale_vector = np.reshape(scale_vector, (component_count, 1))

    # Generate the recursive force function
    base_force_function = generate_random_force(
        time_end, 0, augmentations, period, period_shift, component_count
    )

    def force_function(t: float) -> np.ndarray:
        force_value = base_force_function(t)
        # Apply scaling vector to each component
        if len(force_value.shape) == 1:
            return force_value * scale_vector.flatten()
        return force_value * scale_vector

    return force_function
