import numpy as np
import sympy
import time

def compute_euler_lagrange_equation(lagrangian_expr, symbol_matrix, time_symbol, coordinate_index):
    """
    Compute the Euler-Lagrange equation for a given generalized coordinate.

    Parameters:
    - lagrangian_expr (sp.Expr): The symbolic expression of the Lagrangian.
    - symbol_matrix (sp.Matrix): The matrix of symbolic variables (external forces, positions, velocities, and accelerations).
    - time_symbol (sp.Symbol): The symbolic variable representing time.
    - coordinate_index (int): The index of the generalized coordinate for differentiation.

    Returns:
    - sp.Expr: The Euler-Lagrange equation for the specified generalized coordinate.
    """
    dL_dq = sympy.diff(lagrangian_expr, symbol_matrix[1, coordinate_index])
    dL_dq_dot = sympy.diff(lagrangian_expr, symbol_matrix[2, coordinate_index])
    time_derivative = sympy.diff(dL_dq_dot, time_symbol)

    for j in range(symbol_matrix.shape[1]): # One can says there is the smart move when using symbolic variable
        time_derivative = time_derivative.replace(sympy.Derivative(symbol_matrix[1, j], time_symbol), symbol_matrix[2, j])
        time_derivative = time_derivative.replace(sympy.Derivative(symbol_matrix[2, j], time_symbol), symbol_matrix[3, j])

    return dL_dq - time_derivative

def generate_acceleration_function(lagrangian, symbol_matrix, time_symbol, substitution_dict, fluid_forces=[], verbose=False, use_clever_solve=True):
    """
    Generate a function for computing accelerations based on the Lagrangian.

    This is actually a multi step process that will convert a Lagrangian into an acceleration function through euler lagrange theory.

    Some information about clever solve : there is two mains way to retrieve the acceleration from the other variable.
    The first one is to ask sympy to symbolically solve our equation and after to lambify it for use afterward. 
    The main drawback of this is that when system is not perfectly retrieved it is theorically extremely hard to get a simple equation giving acceleration from the other variable.
    This usually lead to code running forever trying to solve this symbolic issue.

    Parameters:
    - lagrangian (sp.Expr): The symbolic Lagrangian of the system.
    - symbol_matrix (sp.Matrix): Matrix containing symbolic variables (external forces, positions, velocities, accelerations).
    - time_symbol (sp.Symbol): The time symbol in the Lagrangian.
    - substitution_dict (dict): Dictionary of substitutions for simplifying expressions. Used in case where lagrangian is handmade with variable (like lenght of pendulum etc...)
    - fluid_forces (list): List of external fluid dissipation coefficient affecting the system (default is None).
    - verbose (bool): If True, print timing information (default is False).
    - use_clever_solve (bool): If True, use matrix-based solution for acceleration (default is True).

    Returns:
    - function: A function that computes the accelerations given system state.
    - bool: Whether the acceleration function generation was successful.
    """
    num_coords = symbol_matrix.shape[1]
    fluid_forces = fluid_forces or [0] * num_coords

    accelerations = np.zeros((num_coords, 1), dtype="object") # Initialisation of return acceleration array
    dynamic_equations = np.zeros((num_coords,), dtype="object")
    valid = True

    for i in range(num_coords):
        if verbose:
            start_time = time.time()

        dynamic_eq = compute_euler_lagrange_equation(lagrangian, symbol_matrix, time_symbol, i)

        if verbose:
            print("Time to derive {}: {}".format(i, time.time() - start_time))

        dynamic_eq -= symbol_matrix[0, i]
        dynamic_eq += fluid_forces[i] * symbol_matrix[2, i]

        if i < (num_coords - 1):
            dynamic_eq += fluid_forces[i + 1] * symbol_matrix[2, i]
            dynamic_eq += - fluid_forces[i + 1] * symbol_matrix[2, i + 1]
            dynamic_eq += symbol_matrix[0, i + 1]

        if i > 0:
            dynamic_eq += -fluid_forces[i] * symbol_matrix[2, i - 1]

        if str(symbol_matrix[3, i]) in str(dynamic_eq):
            dynamic_equations[i] = dynamic_eq.subs(substitution_dict)
        else:
            valid = False
            break

    if valid:
        if verbose:
            print("Dynamics {}: {}".format(len(dynamic_equations), dynamic_equations))
            start_time = time.time()

        if use_clever_solve:
            system_matrix, force_vector = np.empty((num_coords, num_coords), dtype=object), np.empty((num_coords, 1), dtype=object)

            for i in range(num_coords):
                equation = dynamic_equations[i]
                for j in range(num_coords):
                    equation = equation.collect(symbol_matrix[3, j])
                    term = equation.coeff(symbol_matrix[3, j])
                    system_matrix[i, j] = -term
                    equation -= term * symbol_matrix[3, j]

                force_vector[i, 0] = equation

            system_func = sp.lambdify([symbol_matrix], system_matrix)
            force_func = sp.lambdify([symbol_matrix], force_vector)

            def acceleration_solver(input_values):
                system_eval = system_func(input_values)
                force_eval = force_func(input_values)
                return np.linalg.solve(system_eval, force_eval)

            acc_func = acceleration_solver
        else:
            solution = sp.solve(dynamic_equations, symbol_matrix[3, :])
            accelerations[:, 0] = list(solution.values()) if isinstance(solution, dict) else list(solution[0])
            acc_func = sp.lambdify([symbol_matrix], accelerations)

        if verbose:
            print("Time to Lambdify: {}".format(time.time() - start_time))

    return acc_func, valid

def solve_linear_system(dynamics, acceleration_symbols):
    """
    Solve a linear system for accelerations given dynamics equations.

    Parameters:
    - dynamics (list): List of dynamic equations.
    - acceleration_symbols (list): List of symbols for acceleration variables.

    Returns:
    - sp.FiniteSet: Solution set for the linear system of equations.
    """
    matrix_a = np.zeros((len(acceleration_symbols), len(acceleration_symbols)), dtype="object")
    vector_b = np.zeros((1, len(acceleration_symbols)), dtype="object")

    for i, equation in enumerate(dynamics):
        for j, acc_symbol in enumerate(acceleration_symbols):
            equation = sp.collect(equation, acc_symbol)
            matrix_a[i, j] = equation.coeff(acc_symbol)
            equation -= matrix_a[i, j] * acc_symbol

        vector_b[0, i] = equation

    return sp.linsolve((sp.Matrix(matrix_a), sp.Matrix(vector_b)), *acceleration_symbols)

def create_experiment_matrix(num_time_steps, num_coords, catalog, symbol_matrix, time_values, position_values, time_step, subsample=1, noise=0, friction=False, truncation=0, velocity_values=[], acceleration_values=[]):
    """
    Create an experiment matrix for system analysis based on Lagrangian catalog.

    Parameters:
    - num_time_steps (int): Total number of time steps in the data.
    - num_coords (int): Number of generalized coordinates.
    - catalog (list): List of Lagrangian expressions to evaluate.
    - symbol_matrix (sp.Matrix): Symbolic variable matrix for the system.
    - time_values (np.array): Array of time values.
    - position_values (np.array): Array of positions at each time step.
    - time_step (float): Time interval between steps.
    - subsample (int): Rate of subsampling the data (default is 1, no subsampling).
    - noise (float): Standard deviation of noise to add (default is 0).
    - friction (bool): Whether to add frictional forces (default is False).
    - truncation (int): Number of initial time steps to truncate (default is 0).
    - velocity_values (np.array): Array of velocities (default is empty).
    - acceleration_values (np.array): Array of accelerations (default is empty).

    Returns:
    - np.array: Experiment matrix.
    - np.array: Subsampled time values.
    """
    sampled_steps = len(position_values[truncation::subsample])
    experiment_matrix = np.zeros(((sampled_steps) * num_coords, len(catalog) + int(friction) * num_coords))

    if not velocity_values.any():
        print("Using approximation for velocity")
        velocity_values = np.gradient(position_values, time_values, axis=0, edge_order=2)

    if not acceleration_values.any():
        print("Using approximation for acceleration")
        acceleration_values = np.gradient(velocity_values, time_values, axis=0, edge_order=2)

    q_matrix = np.zeros((symbol_matrix.shape[0], symbol_matrix.shape[1], sampled_steps))
    q_matrix[1, :, :] = np.transpose(position_values[truncation::subsample])
    q_matrix[2, :, :] = np.transpose(velocity_values[truncation::subsample])
    q_matrix[3, :, :] = np.transpose(acceleration_values[truncation::subsample])

    for i in range(num_coords):
        catalog_lagrange = list(map(lambda x: compute_euler_lagrange_equation(x, symbol_matrix, time_step, i), catalog))
        catalog_lambda = list(map(lambda x: sp.lambdify([symbol_matrix], x, modules="numpy"), catalog_lagrange))

        for j, func in enumerate(catalog_lambda):
            experiment_matrix[i * sampled_steps:(i + 1) * sampled_steps, j] = func(q_matrix)

        if friction:
            experiment_matrix[i * sampled_steps:(i + 1) * sampled_steps, len(catalog_lambda) + i] += velocity_values[truncation::subsample, i]

            if i < (num_coords - 1):
                experiment_matrix[i * sampled_steps:(i + 1) * sampled_steps, len(catalog_lambda) + i + 1] += velocity_values[truncation::subsample, i] - velocity_values[truncation::subsample, i + 1]

            if i > 0:
                experiment_matrix[i * sampled_steps:(i + 1) * sampled_steps, len(catalog_lambda) + i] -= velocity_values[truncation::subsample, i - 1]

    return experiment_matrix, time_values[truncation::subsample]

def compute_covariance_matrix(exp_matrix, solution, num_coords):
    """
    Calculate the covariance matrix based on experimental data.

    Parameters:
    - exp_matrix (np.array): Experiment matrix.
    - solution (np.array): Solution array.
    - num_coords (int): Number of generalized coordinates.

    Returns:
    - np.array: Variance of each element in the experiment matrix.
    """
    cov_matrix = np.linalg.inv(exp_matrix.T @ exp_matrix)
    sampled_steps = int(exp_matrix.shape[0] / num_coords)
    variance = np.zeros((sampled_steps,))

    for i in range(num_coords):
        app = exp_matrix[i * sampled_steps:(i + 1) * sampled_steps, :]
        variance += (app @ cov_matrix @ app.T).diagonal()

    return variance
