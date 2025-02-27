"""

This module enable user to launch nearly complete workflow in order to run Xl-Sindy simulation

"""



import numpy as np
from .dynamics_modeling import *
from .catalog_gen import *
from .euler_lagrange import *
from .optimization import *


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
    regression_function:Callable=lasso_regression
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
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

    catalog=expand_catalog(catalog_repartition,symbol_matrix,time_symbol)

    # Generate the experimental matrix from the catalog
    experimental_matrix = create_experiment_matrix(
        num_coordinates,
        catalog,
        symbol_matrix,
        time_symbol,
        theta_values,
        velocity_values,
        acceleration_values,
        friction_order_one=True,
    )

    external_force_vec = np.reshape(external_force.T,(-1,1))

    covariance_matrix = None
    solution = None

    # Normalize experimental matrix if required
    normalized_matrix, reduction_indices, variance_vector = (
        normalize_experiment_matrix(
            experimental_matrix, null_effect=apply_normalization
        )
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
