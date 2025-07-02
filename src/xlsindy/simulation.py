"""

This module enable user to launch nearly complete workflow in order to run Xl-Sindy simulation

"""

from typing import Callable, Tuple
import numpy as np
import sympy

from .euler_lagrange import create_experiment_matrix, jax_create_experiment_matrix
from .optimization import lasso_regression


import cvxpy as cp

from .catalog import CatalogRepartition


def regression_explicite(
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
    acceleration_values: np.ndarray,
    time_symbol: sympy.Symbol,
    symbol_matrix: np.ndarray,
    catalog_repartition: CatalogRepartition,
    external_force: np.ndarray,
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
    external_forces_mask = 0 # super bad ## need to automate the finding process

    catalog = catalog_repartition.expand_catalog()

    whole_experimental_matrix = jax_create_experiment_matrix(
        num_coordinates,
        catalog,
        symbol_matrix,
        theta_values,
        velocity_values,
        acceleration_values,
        external_force,
    )

    whole_solution = regression_function(whole_experimental_matrix,external_forces_mask) # Mask the first one that is the external forces
    whole_solution = np.reshape(whole_solution,shape=(-1,1))
    #whole_solution[np.abs(whole_solution) < np.max(np.abs(whole_solution)) * hard_threshold] = 0

    ## Regression data, residual, covariance matrix,.... everything need to be done on the truncated again matrix 

    # exp_matrix,forces_vector = amputate_experiment_matrix(whole_experimental_matrix,external_forces_mask)
    # solution = np.delete(whole_solution,external_forces_mask,axis=0)


    # # Estimate covariance matrix based on Ordinary Least Squares (OLS)

    # solution_flat = solution.flatten()
    # nonzero_indices = np.nonzero(np.abs(solution_flat) > 0)[0]
    # reduced_experimental_matrix = exp_matrix[:, nonzero_indices]
    # covariance_reduced = np.cov(reduced_experimental_matrix.T)

    # covariance_matrix = np.zeros((solution.shape[0], solution.shape[0]))
    # covariance_matrix[nonzero_indices[:, np.newaxis], nonzero_indices] = (
    #     covariance_reduced
    # )

    # Deprecated just here as a reminder of the math 

    # residuals = forces_vector - exp_matrix @ solution

    # sigma_squared = (
    #     1
    #     / (exp_matrix.shape[0] - exp_matrix.shape[1])
    #     * residuals.T
    #     @ residuals
    # )
    # covariance_matrix *= sigma_squared

    return whole_solution, whole_experimental_matrix

def regression_implicite(
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
    acceleration_values: np.ndarray,
    time_symbol: sympy.Symbol,
    symbol_matrix: np.ndarray,
    catalog_repartition: CatalogRepartition,
    l1_lambda = 1e-7,
    debug: bool = False,
    #regression_function: Callable = lasso_regression,
    #sparsity_coefficient: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes regression for a dynamic system to estimate the system’s parameters. 
    This function can only be used with implicit system, meaning that no external forces are provided.
    Actually, it is an implementation of SYNDy-PI with the general catalog framework 

    Parameters:
        theta_values (np.ndarray): Array of angular positions over time.
        symbol_list (np.ndarray): Symbolic variables for model construction.
        catalog_repartition (CatalogRepartition): a listing of the different part of the catalog used need to follow the following structure : [("lagrangian",lagrangian_catalog),...,("classical",classical_catalog,expand_matrix)]
        external_force (np.ndarray): array of external forces.
        time_step (float): Time step value.
        hard_threshold (float): Threshold below which coefficients are zeroed.
        velocity_values (np.ndarray): Array of velocities (optional).
        acceleration_values (np.ndarray): Array of accelerations (optional).
        regression_function (Callable): the regression function used to make the retrieval
        debug (bool): Whether to go into debug mode.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Solution vector, experimental matrix, sampled time values, covariance matrix.
    """

    # Need to erase the external forces from the catalog
    num_coordinates = theta_values.shape[1]

    catalog = catalog_repartition.expand_catalog()

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

    print("debug : some information")
    print(np.linalg.norm(experimental_matrix),np.var(experimental_matrix))


    # Normalization enable more stable regression over the different experiment

    sigma_max = np.linalg.svd(experimental_matrix, compute_uv=False)[0]
    A = experimental_matrix / sigma_max
    # Aparently, operator norm is better than frobenius norm for this kind of regression
    # I don't know why, but it was the case on the first test, lol

    #A = experimental_matrix /np.linalg.norm(experimental_matrix)

    m, n = A.shape

    X = cp.Variable((n,n))

    # Since norm of A is 1, norm1(X) is around ~1 l1_lambda constrain the overal precision of the solution
    # Before attain frobenius residual to l1_lambda order the sparsity is not enforced...
    # One should take advantage of this in order to envorce the good behavior
    obj = cp.Minimize(
        cp.norm(A @ X - A, "fro") + l1_lambda * cp.norm1(X)
    )


    prob = cp.Problem(obj, [cp.diag(X) == 0])
    prob.solve(verbose=True)

    if debug:
        return X.value, experimental_matrix
    
    else:
        solutions = _implicit_post_treatment( X.value)
        
        return solutions, experimental_matrix

def _implicit_post_treatment(
        solution: np.ndarray,
        deg_tol: float = 10,
        weight_distribution_threshold: float = 0.8,
) -> np.ndarray:
    """
        Second try as post treatment to recuperate the solution in the form of a unique vector.
    Explained in implicit_solution_analysis.ipynb
    
    Parameters:
        solution (np.ndarray): The solution matrix obtained from the regression.
        deg_tol (float): The cone of similarity the vector need to be to form a cluster
        weight_distribution_threshold (float): the minimal weight deviation that is acceptable for a cluster
    Returns:
        np.ndarray: The processed solution vector.
    """

    #Firstly create sparse cluster
    groups = _groups_homothetic_noisy(solution, deg_tol=deg_tol, atol=1e-12)

    filtered_groups=[]
    for i, group in enumerate(groups):
        if len(group) > 2:

            total_group = solution[:,group].sum(axis=1)
            weight= _weak_sparsity_rank_weighted(total_group)

            if weight>weight_distribution_threshold: 
                filtered_groups+= [group]

            print(f"Group {i}, weight {weight:.2f}: {group}")

    solutions = np.zeros((solution.shape[0],len(filtered_groups)))

    for i,group in enumerate(filtered_groups) :

        candidate_solution = solution[:, group]
        U, S, VT = np.linalg.svd(candidate_solution.T)
        solutions[:,i] += -VT[0].flatten()

    return solutions

def _groups_homothetic_noisy(mat, deg_tol=3.0, atol=1e-12):
    """
    Used in _implicit_post_treatment in order to create sparse cluster of same direction vector.

    Parameters:
        mat  : shape (m, n) – columns are the vectors
        deg_tol : angular tolerance in degrees

    Returns:
        list of lists with column indices that are homothetic

    """
    rad_tol = np.deg2rad(deg_tol)
    # 1) normalise columns (keep track of zeros separately)
    lengths = np.linalg.norm(mat, axis=0)
    keep   = np.where(lengths > atol)[0]
    if keep.size == 0:
        return []                     # nothing but zeros
    unit   = mat[:, keep] / lengths[keep]

    groups, pool = [], list(range(unit.shape[1]))
    while pool:
        j        = pool.pop(0)
        ref      = unit[:, j]
        hits     = [keep[j]]          # column index in the original matrix
        to_drop  = []
        # 2) compare with remaining columns
        for pos in pool:
            ang = np.arccos(
                np.clip(np.abs(ref @ unit[:, pos]), -1.0, 1.0)
            )
            if ang <= rad_tol:
                hits.append(keep[pos])
                to_drop.append(pos)
        # 3) remove the ones we already grouped
        for pos in reversed(to_drop):
            pool.remove(pos)
        groups.append(hits)
    return groups

def _weak_sparsity_rank_weighted(x):
    """
    Custom weak sparsity measure: weighted average of sorted abs(x) using inverse rank weights.

    Parameters:
        x: np.ndarray (1D vector)

    Returns:
        scalar value ∈ [0, 1], higher = more concentrated mass at top

    """
    x = np.abs(x)
    s = np.sort(x)[::-1]  # descending
    ranks = np.arange(1, len(s) + 1)

    weights = 1.0 / ranks
    weighted_sum = np.sum(s * weights)
    weight_total = np.sum(weights)

    return weighted_sum / weight_total if weight_total != 0 else 0.0

def combine_best_fit(solutions, v_ideal):
    """
    Return the best linear combination of a and b to fit v_ideal.

    Parameters:
    - solutions: disjoint solution array
    - v_ideal: target vector

    Returns:
    - v_hat: best fit vector (alpha * a + beta * b)
    - residual: L2 norm of the residual (||v_hat - v_ideal||)
    """

    x, _, _, _ = np.linalg.lstsq(solutions, v_ideal, rcond=None)
    v_hat = solutions @ x
    residual = np.linalg.norm(v_hat - v_ideal)
    return v_hat.reshape(-1,1), residual

## Mixed framework regression 

def regression_mixed(
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
    acceleration_values: np.ndarray,
    time_symbol: sympy.Symbol,
    symbol_matrix: np.ndarray,
    catalog_repartition: CatalogRepartition,
    external_force: np.ndarray,
    regression_function: Callable = lasso_regression,
):
    """
    Executes regression for a dynamic system to estimate the system's parameters.
    This function can be used with both explicit and implicit systems, and will performs a chain of implicit/explicit regression.

    The algorithm is the following:
    1. Find recursively the explicit part of the catalog from the external forces.
    2. Perform a first explicit regression on this part of the catalog to retrieve the coefficients
    3. Perform an implicit regression on the remaining part of the catalog (remaining part calculated from the result of explicit regression).

    [Need to explore this algorithm maybe]
    The algorithm is the following:
    1. Create the experimental matrix from the catalog.
    2. Search the "activated" part of the catalog where force are present.
    3. Perform a first explicit regression on this experiment matrix to retrieve the coefficients of the explicit part.
    4. Search if the solution "activate" another part of the catalog.
    5. Repeat 3 and 4 until no new part of the catalog is activated.
    6. Perform a final implicit regression on the remaining part of the catalog.
    
    Args:
        theta_values (np.ndarray): Array of angular positions over time.
        velocity_values (np.ndarray): Array of velocities (optional).
        acceleration_values (np.ndarray): Array of accelerations (optional).
        time_symbol (sympy.Symbol): Symbol representing time.
        symbol_matrix (np.ndarray): Matrix of symbolic variables for model construction.
        catalog_repartition (CatalogRepartition): Catalog containing the different parts used in the regression.
        external_force (np.ndarray): Array of external forces. Defaults to None.
        regression_function (Callable, optional): The regression function used to make the retrieval. Defaults to lasso_regression. ( Maybe change to CVxPY in the future)
        
    Returns:
        Tuple TODO
    """

    num_coordinates = theta_values.shape[1]

    catalog = catalog_repartition.expand_catalog()

    # Generate the experimental matrix from the catalog
    experimental_matrix = create_experiment_matrix(
        num_coordinates,
        catalog,
        symbol_matrix,
        theta_values,
        velocity_values,
        acceleration_values,
    )


