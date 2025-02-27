"""

This module include every function in order to run the optimisation step for getting the sparse solution

"""


import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from typing import Callable, Tuple, List


def condition_value(exp_matrix: np.ndarray, solution: np.ndarray) -> np.ndarray:
    """
    Calculate condition values based on the variance of the experimental matrix.

    Parameters:
        exp_matrix (np.ndarray): Experimental matrix.
        solution (np.ndarray): Solution vector.

    Returns:
        np.ndarray: Array of condition values.
    """
    return np.abs(np.var(exp_matrix, axis=0) * solution[:, 0])


def optimal_sampling(theta_values: np.ndarray, distance_threshold: float) -> np.ndarray:
    """
    (experimetnal) Selects optimal samples from a set of points based on a distance threshold.

    Parameters:
        theta_values (np.ndarray): Array of points.
        distance_threshold (float): Minimum distance to include a point.

    Returns:
        np.ndarray: Indices of selected points.
    """
    num_values = theta_values.shape[0]
    result_points = np.empty(theta_values.shape)
    selected_indices = np.zeros((num_values,), dtype=int)

    result_points[0, :] = theta_values[0, :]
    count = 1

    for i in range(1, num_values):
        point = theta_values[i, :]
        distance = np.sqrt(
            np.min(np.sum(np.power(result_points[:count, :] - point, 2), axis=1))
        )
        if distance > distance_threshold:
            result_points[count, :] = point
            selected_indices[count] = i
            count += 1

    return selected_indices[:count]

def normalize_experiment_matrix(
    exp_matrix: np.ndarray, null_effect: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalizes an experimental matrix by its variance and mean.

    Parameters:
        exp_matrix (np.ndarray): Experimental matrix to normalize.
        null_effect (bool): Whether to consider null effect in normalization.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Normalized matrix, reduction indices, and variance.
    """
    variance = np.var(exp_matrix, axis=0) * int(not null_effect) + int(null_effect)
    mean = np.mean(exp_matrix, axis=0) * int(not null_effect)

    reduction_indices = np.argwhere(variance != 0).flatten()
    reduced_matrix = exp_matrix[:, reduction_indices]
    normalized_matrix = (reduced_matrix - mean[reduction_indices]) / variance[
        reduction_indices
    ]

    return normalized_matrix, reduction_indices, variance


def unnormalize_experiment(
    coefficients: np.ndarray,
    variance: np.ndarray,
    reduction_indices: np.ndarray,
    exp_matrix: np.ndarray,
) -> np.ndarray:
    """
    Reverts normalization of a solution vector.

    Parameters:
        coefficients (np.ndarray): Normalized coefficients.
        variance (np.ndarray): Variance used for normalization.
        reduction_indices (np.ndarray): Indices used for dimensionality reduction.
        exp_matrix (np.ndarray): Original experimental matrix.

    Returns:
        np.ndarray: Unnormalized solution vector.
    """
    solution_unscaled = coefficients / variance[reduction_indices]
    friction_coefficient = -solution_unscaled[-1]

    solution = np.zeros((exp_matrix.shape[1], 1))
    solution[reduction_indices, 0] = solution_unscaled

    return solution


def covariance_vector(
    exp_matrix: np.ndarray, covariance_matrix: np.ndarray, num_time_steps: int
) -> np.ndarray:
    """
    Calculates the covariance vector across time steps for an experimental matrix.

    Parameters:
        exp_matrix (np.ndarray): Experimental matrix.
        covariance_matrix (np.ndarray): Covariance matrix.
        num_time_steps (int): Number of time steps.

    Returns:
        np.ndarray: Covariance vector summed across time steps.
    """
    result_matrix = exp_matrix @ covariance_matrix @ exp_matrix.T
    diagonal_covariance = np.diagonal(result_matrix).reshape(-1, num_time_steps)
    summed_covariance = np.sum(diagonal_covariance, axis=1)

    return summed_covariance


## Optimisation function

def hard_threshold_sparse_regression(
    forces_vector: np.ndarray,
    exp_matrix: np.ndarray,
    #catalog: np.ndarray,
    condition_func: Callable = condition_value,
    threshold: float = 0.03,
) -> np.ndarray:
    """
    Performs sparse regression with a hard threshold to select significant features.

    Parameters:
        exp_matrix (np.ndarray): Experimental matrix.
        forces_vector (np.ndarray): Forces vector.
        condition_func (Callable): Function to calculate condition values.
        threshold (float): Threshold for feature selection.

    Returns:
        np.ndarray: solution vector. shape (-1,)
    """
    solution, residuals, rank, _ = np.linalg.lstsq(
        exp_matrix, forces_vector, rcond=None
    )

    #print("solution shape",solution.shape)

    retained_solution = solution.copy()
    result_solution = np.zeros(solution.shape)
    active_indices = np.arange(len(solution))
    steps = []

    prev_num_indices = len(solution) + 1
    current_num_indices = len(solution)

    while current_num_indices < prev_num_indices:
        prev_num_indices = current_num_indices
        condition_values = condition_func(exp_matrix, retained_solution)
        steps.append((retained_solution, condition_values, active_indices))

        significant_indices = np.argwhere(
            condition_values / np.max(condition_values) > threshold
        ).flatten()
        active_indices = active_indices[significant_indices]
        exp_matrix = exp_matrix[:, significant_indices]

        retained_solution, residuals, rank, _ = np.linalg.lstsq(
            exp_matrix, forces_vector, rcond=None
        )
        current_num_indices = len(active_indices)

    result_solution[active_indices] = retained_solution

    result_solution = np.reshape(result_solution,(-1,))# flatten

    return result_solution # model_fit, result_solution, reduction_count, steps # deprecated


def lasso_regression(
    forces_vector: np.ndarray,
    normalized_exp_matrix: np.ndarray,
    max_iterations: int = 10**4,
    tolerance: float = 1e-5,
    eps: float = 5e-4,
) -> np.ndarray:
    """
    Performs Lasso regression to select sparse features.

    Parameters:
        forces_vector (np.ndarray): Dependent variable vector.
        normalized_exp_matrix (np.ndarray): Normalized experimental matrix.
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.
        eps (float): Regularization parameter.

    Returns:
        np.ndarray: Coefficients of the fitted model. shape (-1,)
    """
    y = forces_vector[:, 0]
    model_cv = LassoCV(
        cv=5, random_state=0, max_iter=max_iterations, eps=eps, tol=tolerance
    )
    model_cv.fit(normalized_exp_matrix, y)
    best_alpha = model_cv.alpha_

    lasso_model = Lasso(alpha=best_alpha, max_iter=max_iterations, tol=tolerance)
    lasso_model.fit(normalized_exp_matrix, y)

    return lasso_model.coef_