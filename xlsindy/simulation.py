"""

This module enable user to launch nearly complete workflow in order to run Xl-Sindy simulation

"""

import numpy as np
from .dynamics_modeling import *
from .catalog_gen import *
from .euler_lagrange import *
from .optimization import *

import jax
import jax.numpy as jnp
from jax import lax


def execute_regression(
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
    acceleration_values: np.ndarray,
    time_symbol: sympy.Symbol,
    symbol_matrix: np.ndarray,
    catalog_repartition: np.ndarray,
    external_force: np.ndarray,
    hard_threshold: float = 1e-3,
    apply_normalization: bool = True,
    regression_function: Callable = lasso_regression,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    (DEPRECATED) will be removed in the future, should use TODO
    Executes regression for a dynamic system to estimate the system’s parameters.

    Parameters:
        theta_values (np.ndarray): Array of angular positions over time.
        symbol_list (np.ndarray): Symbolic variables for model construction.
        catalog_repartition (List[tuple]): a listing of the different part of the catalog used need to follow the following structure : [("lagrangian",lagrangian_catalog),...,("classical",classical_catalog,expand_matrix)]
        external_force (np.ndarray): array of external forces.
        time_step (float): Time step value.
        hard_threshold (float): Threshold below which coefficients are zeroed.
        velocity_values (np.ndarray): Array of velocities (optional).
        acceleration_values (np.ndarray): Array of accelerations (optional).
        apply_normalization (bool): Whether to normalize data.
        regression_function (Callable): the regression function used to make the retrieval

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Solution vector, experimental matrix, sampled time values, covariance matrix.
    """

    num_coordinates = theta_values.shape[1]

    catalog = expand_catalog(catalog_repartition, symbol_matrix, time_symbol)

    # Generate the experimental matrix from the catalog
    experimental_matrix = create_experiment_matrix(
        num_coordinates,
        catalog,
        symbol_matrix,
        theta_values,
        velocity_values,
        acceleration_values,
    )

    external_force_vec = np.reshape(external_force.T, (-1, 1))

    covariance_matrix = None
    solution = None

    # Normalize experimental matrix if required
    normalized_matrix, reduction_indices, variance_vector = normalize_experiment_matrix(
        experimental_matrix, null_effect=apply_normalization
    )

    # Perform Lasso regression to obtain coefficients
    coefficients = regression_function(external_force_vec, normalized_matrix)

    # Revert normalization to obtain solution in original scale
    solution = unnormalize_experiment(
        coefficients, variance_vector, reduction_indices, experimental_matrix
    )
    solution[np.abs(solution) < np.max(np.abs(solution)) * hard_threshold] = 0

    # Estimate covariance matrix based on Ordinary Least Squares (OLS)
    solution_flat = solution.flatten()
    nonzero_indices = np.nonzero(np.abs(solution_flat) > 0)[0]
    reduced_experimental_matrix = experimental_matrix[:, nonzero_indices]
    covariance_reduced = np.cov(reduced_experimental_matrix.T)

    covariance_matrix = np.zeros((solution.shape[0], solution.shape[0]))
    covariance_matrix[nonzero_indices[:, np.newaxis], nonzero_indices] = (
        covariance_reduced
    )

    residuals = external_force_vec - experimental_matrix @ solution
    sigma_squared = (
        1
        / (experimental_matrix.shape[0] - experimental_matrix.shape[1])
        * residuals.T
        @ residuals
    )
    covariance_matrix *= sigma_squared

    return solution, experimental_matrix, covariance_matrix

def regression_explicite(
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
    acceleration_values: np.ndarray,
    time_symbol: sympy.Symbol,
    symbol_matrix: np.ndarray,
    catalog_repartition: np.ndarray,
    external_force: np.ndarray,
    hard_threshold: float = 1e-3,
    regression_function: Callable = lasso_regression,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes regression for a dynamic system to estimate the system’s parameters. 
    This function can only be used with explicit system, meaning that external forces array need to be populated at maximum

    Parameters:
        theta_values (np.ndarray): Array of angular positions over time.
        symbol_list (np.ndarray): Symbolic variables for model construction.
        catalog_repartition (List[tuple]): a listing of the different part of the catalog used need to follow the following structure : [("lagrangian",lagrangian_catalog),...,("classical",classical_catalog,expand_matrix)]
        external_force (np.ndarray): array of external forces.
        time_step (float): Time step value.
        hard_threshold (float): Threshold below which coefficients are zeroed.
        velocity_values (np.ndarray): Array of velocities (optional).
        acceleration_values (np.ndarray): Array of accelerations (optional).
        regression_function (Callable): the regression function used to make the retrieval

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Solution vector, experimental matrix, sampled time values, covariance matrix.
    """

    num_coordinates = theta_values.shape[1]

    # Extend experiment matrix with external forces
    catalog_repartition = [("external_forces",None)]+catalog_repartition

    catalog = expand_catalog(catalog_repartition, symbol_matrix, time_symbol)

    # Generate the experimental matrix from the catalog
    ## TODO Jax this
    experimental_matrix = jax_create_experiment_matrix(
        num_coordinates,
        catalog,
        symbol_matrix,
        theta_values,
        velocity_values,
        acceleration_values,
        external_force,
    )

    covariance_matrix = None
    solution = None

    # Perform Lasso regression to obtain coefficients
    ## TODO Jax every regression_function
    
    solution = regression_function(experimental_matrix,0) # Mask the first one that is the external forces

    solution = np.reshape(solution,shape=(-1,1))

    solution[np.abs(solution) < np.max(np.abs(solution)) * hard_threshold] = 0

    # Estimate covariance matrix based on Ordinary Least Squares (OLS)
    solution_flat = solution.flatten()
    nonzero_indices = np.nonzero(np.abs(solution_flat) > 0)[0]
    reduced_experimental_matrix = experimental_matrix[:, nonzero_indices]
    covariance_reduced = np.cov(reduced_experimental_matrix.T)

    covariance_matrix = np.zeros((solution.shape[0], solution.shape[0]))
    covariance_matrix[nonzero_indices[:, np.newaxis], nonzero_indices] = (
        covariance_reduced
    )

    residuals = external_force_vec - experimental_matrix @ solution
    sigma_squared = (
        1
        / (experimental_matrix.shape[0] - experimental_matrix.shape[1])
        * residuals.T
        @ residuals
    )
    covariance_matrix *= sigma_squared

    return np.reshape(solution,shape=(-1,1)), experimental_matrix, covariance_matrix

# def _build_A_star(A, k):
#     # Build A without the k-th column using dynamic slicing
#     left = lax.dynamic_slice(A, (0, 0), (A.shape[0], k))
#     right = lax.dynamic_slice(A, (0, k+1), (A.shape[0], A.shape[1] - k - 1))
#     return jnp.concatenate([left, right], axis=1)

def _build_A_star(A, k):
    n = A.shape[1]
    
    if k == 0:
        # Remove first column
        return A[:, 1:]
    elif k == n - 1:
        # Remove last column
        return A[:, :-1]
    else:
        # Remove middle column
        return np.concatenate([A[:, :k], A[:, k+1:]], axis=1)

def _insert_zero(x_k, k, n):
    x_full = np.zeros(n)
    x_full[:k] = x_k[:k]
    x_full[k+1:] = x_k[k:]
    return x_full

def regression_implicite(
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
    acceleration_values: np.ndarray,
    time_symbol: sympy.Symbol,
    symbol_matrix: np.ndarray,
    catalog_repartition: np.ndarray,
    hard_threshold: float = 1e-3,
    regression_function: Callable = lasso_regression,
    sparsity_coefficient: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes regression for a dynamic system to estimate the system’s parameters. 
    This function can only be used with implicit system, meaning that no external forces are provided.
    Actually, it is an implementation of SYNDy-PI with the general catalog framework 

    Parameters:
        theta_values (np.ndarray): Array of angular positions over time.
        symbol_list (np.ndarray): Symbolic variables for model construction.
        catalog_repartition (List[tuple]): a listing of the different part of the catalog used need to follow the following structure : [("lagrangian",lagrangian_catalog),...,("classical",classical_catalog,expand_matrix)]
        external_force (np.ndarray): array of external forces.
        time_step (float): Time step value.
        hard_threshold (float): Threshold below which coefficients are zeroed.
        velocity_values (np.ndarray): Array of velocities (optional).
        acceleration_values (np.ndarray): Array of accelerations (optional).
        regression_function (Callable): the regression function used to make the retrieval

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Solution vector, experimental matrix, sampled time values, covariance matrix.
    """

    num_coordinates = theta_values.shape[1]

    catalog = expand_catalog(catalog_repartition, symbol_matrix, time_symbol)

    # Generate the experimental matrix from the catalog
    ## TODO Jax this
    experimental_matrix = create_experiment_matrix(
        num_coordinates,
        catalog,
        symbol_matrix,
        theta_values,
        velocity_values,
        acceleration_values,
    )

    m, n = experimental_matrix.shape

    def solve_k(k):
        
        A_k = np.reshape(experimental_matrix[:, k].T, (-1, 1))
        A_star_k = _build_A_star(experimental_matrix, k)
        print("OmegaProut",A_k.shape,A_star_k.shape)
        x_k = regression_function(A_k,A_star_k )
        print("lourdProut",x_k)
        x_full = _insert_zero(x_k, k, n)  # Put zero at the kth position
        print("TartProut")
        return x_full

    #x_all = jax.vmap(solve_k)(jnp.arange(n)) # not jit regression function

    x_all = []

    for k in range(n):
        print("solved k", k)
        x_all.append(solve_k(k))

    x_all = np.stack(x_all)


    # Step 1 : Hardtresholding
    max_val = np.max(np.abs(x_all))
    threshold = max_val * hard_threshold
    x_all = np.where(np.abs(x_all) < threshold, 0.0, x_all)

    # Step 2: compute sparsity = number of non-zeros per x
    sparsity = np.sum(x_all != 0, axis=1)
    min_sparsity = np.min(sparsity[sparsity > 0])

    print("min_sparsity", min_sparsity)
    print("sparsity", sparsity)

    max_allowed_sparsity = min_sparsity * sparsity_coefficient

    # Step 3: mask valid sparse solutions
    valid_mask = ( sparsity <= max_allowed_sparsity )& (sparsity > 0)

    print("valid_mask", valid_mask)
    print("x_all shape", x_all.shape)
    valid_solutions = x_all[valid_mask]

    # Step 4: average valid solutions
    x_final = np.mean(valid_solutions, axis=0)

    return np.reshape(x_final,shape=(-1,1)), experimental_matrix, None # covariance matrix not computed in this case