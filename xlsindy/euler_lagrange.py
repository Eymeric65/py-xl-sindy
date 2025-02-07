"""

This module primarly focus on symbolic variable and enable to do the different manipulation in order to get the experiment matrix

"""
import numpy as np
import sympy
import time
from typing import List, Callable, Dict, Tuple
import jax.numpy as jnp



def compute_euler_lagrange_equation(
    lagrangian_expr: sympy.Expr,
    symbol_matrix: np.ndarray,
    time_symbol: sympy.Symbol,
    coordinate_index: int,
) -> sympy.Expr:
    """
    Compute the Euler-Lagrange equation for a given generalized coordinate.

    Args:
        lagrangian_expr (sp.Expr): The symbolic expression of the Lagrangian.
        symbol_matrix (np.ndarray): The matrix of symbolic variables (external forces, positions, velocities, and accelerations).
        time_symbol (sp.Symbol): The symbolic variable representing time.
        coordinate_index (int): The index of the generalized coordinate for differentiation.

    Returns:
        sympy.Expr: The Euler-Lagrange equation for the specified generalized coordinate.
    """
    dL_dq = sympy.diff(lagrangian_expr, symbol_matrix[1, coordinate_index])
    dL_dq_dot = sympy.diff(lagrangian_expr, symbol_matrix[2, coordinate_index])
    time_derivative = sympy.diff(dL_dq_dot, time_symbol)

    for j in range(
        symbol_matrix.shape[1]
    ):  # One can says there is the smart move when using symbolic variable
        time_derivative = time_derivative.replace(
            sympy.Derivative(symbol_matrix[1, j], time_symbol), symbol_matrix[2, j]
        )
        time_derivative = time_derivative.replace(
            sympy.Derivative(symbol_matrix[2, j], time_symbol), symbol_matrix[3, j]
        )

    return dL_dq - time_derivative


def generate_acceleration_function(
    lagrangian: sympy.Expr,
    symbol_matrix: np.ndarray,
    time_symbol: sympy.Symbol,
    substitution_dict: Dict[str, float] = {},
    first_order_friction: np.ndarray = None,
    verbose: bool = False,
    use_clever_solve: bool = True,
    lambdify_module: str = "numpy",
) -> Tuple[Callable[[np.ndarray], np.ndarray], bool]:
    """
    Generate a function for computing accelerations based on the Lagrangian.

    This is actually a multi step process that will convert a Lagrangian into an acceleration function through euler lagrange theory.

    Some information about clever solve : there is two mains way to retrieve the acceleration from the other variable.
    The first one is to ask sympy to symbolically solve our equation and after to lambify it for use afterward.
    The main drawback of this is that when system is not perfectly retrieved it is theorically extremely hard to get a simple equation giving acceleration from the other variable.
    This usually lead to code running forever trying to solve this symbolic issue.

    The other way is to create a linear system of b=Ax where x are the acceleration coordinate and b is the force vector.
    At runtime one's need to replace every term in b and A and solve the linear equation (of dimension n so really fast)

    Args:
        lagrangian (sympy.Expr): The symbolic Lagrangian of the system.
        symbol_matrix (np.ndarray): Matrix containing symbolic variables (external forces, positions, velocities, accelerations).
        time_symbol (sp.Symbol): The time symbol in the Lagrangian.
        substitution_dict (dict): Dictionary of substitutions for simplifying expressions. Used in case where lagrangian is handmade with variable (like lenght of pendulum etc...)
        first_order_friction (np.ndarray): friction coefficient array (default is None).
        verbose (bool): If True, print timing information (default is False).
        use_clever_solve (bool): If True, use matrix-based solution for acceleration (default is True).

    Returns:
        function: A function that computes the accelerations given system state. takes as input a numerical symbol matrix
        bool: Whether the acceleration function generation was successful.
    """
    num_coords = symbol_matrix.shape[1]
    if first_order_friction is None:
        first_order_friction = np.zeros((num_coords,num_coords))

    accelerations = np.zeros(
        (num_coords, 1), dtype="object"
    )  # Initialisation of return acceleration array
    dynamic_equations = np.zeros((num_coords,), dtype="object")
    valid = True

    for i in range(num_coords):
        if verbose:
            start_time = time.time()

        dynamic_eq = compute_euler_lagrange_equation(
            lagrangian, symbol_matrix, time_symbol, i
        )  # Get every Euler_lagrange equation

        if verbose:
            print("Time to derive {}: {}".format(i, time.time() - start_time))

        dynamic_eq -= symbol_matrix[0, i]  # Add external forces
        
        # add visquous forces
        dynamic_eq += first_order_friction[i,:] @ symbol_matrix[2, :]

        # if i < (num_coords - 1): # WTF is this # 20250206 Awesome work # Okay this is applied to correct the rk4 integration
        #     dynamic_eq += symbol_matrix[0, i + 1]

        # dynamic_eq += fluid_forces[i] * symbol_matrix[2, i]  # Old friction paradigm

        # # Hardcoded dissipation matrix for chained system
        # if i < (num_coords - 1):
        #     dynamic_eq += fluid_forces[i + 1] * symbol_matrix[2, i]
        #     dynamic_eq += -fluid_forces[i + 1] * symbol_matrix[2, i + 1]
        #     dynamic_eq += symbol_matrix[0, i + 1]

        # if i > 0:
        #     dynamic_eq += -fluid_forces[i] * symbol_matrix[2, i - 1]

        if str(symbol_matrix[3, i]) in str(
            dynamic_eq
        ):  # If we have acceleration term (we should if we somewhat analyse a real system)
            dynamic_equations[i] = dynamic_eq.subs(substitution_dict)
        else:
            valid = False
            break

    if valid:
        if verbose:
            print("Dynamics {}: {}".format(len(dynamic_equations), dynamic_equations))
            start_time = time.time()

        if use_clever_solve:
            system_matrix, force_vector = np.empty(
                (num_coords, num_coords), dtype=object
            ), np.empty((num_coords, 1), dtype=object)

            for i in range(num_coords):
                equation = dynamic_equations[i]
                for j in range(num_coords):
                    equation = equation.collect(symbol_matrix[3, j])
                    term = equation.coeff(symbol_matrix[3, j])
                    system_matrix[i, j] = -term
                    equation -= term * symbol_matrix[3, j]

                force_vector[i, 0] = equation

            system_func = sympy.lambdify([symbol_matrix], system_matrix,lambdify_module)
            force_func = sympy.lambdify([symbol_matrix], force_vector,lambdify_module)

            if lambdify_module == "jax":

                def acceleration_solver(input_values):
                    system_eval = system_func(input_values)
                    force_eval = force_func(input_values)
                    return jnp.linalg.solve(system_eval, force_eval)

            else:

                def acceleration_solver(input_values):
                    system_eval = system_func(input_values)
                    force_eval = force_func(input_values)
                    return np.linalg.solve(system_eval, force_eval)
            
            acc_func = acceleration_solver
        else:
            solution = sympy.solve(dynamic_equations, symbol_matrix[3, :])
            accelerations[:, 0] = (
                list(solution.values())
                if isinstance(solution, dict)
                else list(solution[0])
            )
            acc_func = sympy.lambdify([symbol_matrix], accelerations)

        if verbose:
            print("Time to Lambdify: {}".format(time.time() - start_time))
    else: # Fail
        acc_func = None
    return acc_func, valid


def create_experiment_matrix(
    num_coords: int,
    catalog: List[sympy.Expr],
    symbol_matrix: np.ndarray,
    time_symbol: sympy.Symbol,
    position_values: np.ndarray,
    time_values: np.ndarray,
    subsample: int = 1,
    friction_order_one:bool = False,
    truncation: int = 0,
    velocity_values: np.ndarray = [],
    acceleration_values: np.ndarray = [],
) -> List[np.ndarray]:
    """
    Create the SINDy experiment matrix.

    For each function in the catalog (plus the friction term) create the times series of the euler-lagranged function for each coordinate.
    This matrix will afterward undergo the regression in order to retrieve the parse expression.

    Args:
        num_coords (int): Number of generalized coordinates.
        catalog (list): List of Lagrangian expressions to evaluate.
        symbol_matrix (sp.Matrix): Symbolic variable matrix for the system.
        time_values (np.array): Array of time values.
        position_values (np.array): Array of positions at each time step.
        time_step (float): Time interval between steps.
        subsample (int): Rate of subsampling the data (default is 1, no subsampling).
        friction_order_one (bool): Whether to add frictional forces (default is False). Use the model of friction matrix in the first order
        truncation (int): Number of initial time steps to truncate (default is 0).
        velocity_values (np.array): Array of velocities (default is empty).
        acceleration_values (np.array): Array of accelerations (default is empty).

    Returns:
        np.array: Experiment matrix.
        np.array: Subsampled time values.
    """
    sampled_steps = len(position_values[truncation::subsample])
    experiment_matrix = np.zeros(
        ((sampled_steps) * num_coords, len(catalog) + int(friction_order_one) * num_coords**2)
    )

    if not velocity_values.any():
        print("Using approximation for velocity")
        velocity_values = np.gradient(
            position_values, time_values, axis=0, edge_order=2
        )

    if not acceleration_values.any():
        print("Using approximation for acceleration")
        acceleration_values = np.gradient(
            velocity_values, time_values, axis=0, edge_order=2
        )

    q_matrix = np.zeros((symbol_matrix.shape[0], symbol_matrix.shape[1], sampled_steps))
    q_matrix[1, :, :] = np.transpose(position_values[truncation::subsample])
    q_matrix[2, :, :] = np.transpose(velocity_values[truncation::subsample])
    q_matrix[3, :, :] = np.transpose(acceleration_values[truncation::subsample])

    for i in range(num_coords):
        catalog_lagrange = list(
            map(
                lambda x: compute_euler_lagrange_equation(
                    x, symbol_matrix, time_symbol, i
                ),
                catalog,
            )
        )
        catalog_lambda = list(
            map(
                lambda x: sympy.lambdify([symbol_matrix], x, modules="numpy"),
                catalog_lagrange,
            )
        )

        for j, func in enumerate(catalog_lambda):
            experiment_matrix[i * sampled_steps : (i + 1) * sampled_steps, j] = func(
                q_matrix
            )

        if friction_order_one: # New friction paradigm (friction interaction matrix)

            #print( experiment_matrix[i * sampled_steps : (i + 1) * sampled_steps, len(catalog_lambda) + i*num_coords:len(catalog_lambda) +(i+1)*num_coords].shape)

            experiment_matrix[
                i * sampled_steps : (i + 1) * sampled_steps, len(catalog_lambda) + i*num_coords:len(catalog_lambda) +(i+1)*num_coords
            ] += velocity_values[truncation::subsample,:]

        # if friction: # Old friction paradigm (hard coded paired friction)
        #     experiment_matrix[
        #         i * sampled_steps : (i + 1) * sampled_steps, len(catalog_lambda) + i
        #     ] += velocity_values[truncation::subsample, i]

        #     if i < (num_coords - 1):
        #         experiment_matrix[
        #             i * sampled_steps : (i + 1) * sampled_steps,
        #             len(catalog_lambda) + i + 1,
        #         ] += (
        #             velocity_values[truncation::subsample, i]
        #             - velocity_values[truncation::subsample, i + 1]
        #         )

        #     if i > 0:
        #         experiment_matrix[
        #             i * sampled_steps : (i + 1) * sampled_steps, len(catalog_lambda) + i
        #         ] -= velocity_values[truncation::subsample, i - 1]

    return experiment_matrix, time_values[truncation::subsample]
