"""
Contains the function responsible for the classical part of the catalog.
"""

import sympy 
from typing import List
import numpy as np

def _create_solution_vector(
    coeff_matrix: np.ndarray, expand_matrix: np.ndarray
) -> np.ndarray:
    """
    Translate the coefficient matrix into a column vector corresponding to the ordering
    of the expanded catalog matrix (as produced by classical_sindy_expand_catalog).

    Args:
        coeff_matrix (np.ndarray): A matrix of shape (len(catalog), n) containing the coefficients.
        expand_matrix (np.ndarray): A binary matrix of shape (len(catalog), n) that indicates
                                    where each catalog function is applied (1 means applied).

    Returns:
        np.ndarray: A column vector of shape (expand_matrix.sum(), 1) containing the coefficients,
                    in the order that matches the expanded catalog.
    """
    # Flatten the expand matrix in row-major order and find indices where its value is 1.
    mask = expand_matrix.ravel() == 1

    # Use boolean indexing to select corresponding coefficients (works for any dtype).
    coeff_flat = coeff_matrix.ravel()[mask]

    # Reshape into a column vector.
    coeff_vector = coeff_flat.reshape(-1, 1)
    return coeff_vector

def _expand_catalog(
    catalog: List[sympy.Expr], expand_matrix: np.ndarray
) -> np.ndarray:
    """
    expand the catalog in the case of a classical SINDy experiment (for other forces in lagrangian case or full classical SINDy retrieval)

    Args:
        catalog (List[sympy.Expr]): the list of function to expand
        expand_matrix (np.ndarray): the expand information matrix has shape (len(catalog),n) and if set to one at line i and row p means that the function i should be aplied in the equation of the p coordinate

    Returns:
        np.ndarray: an array of shape (np.sum(expand_matrix),n) containing all the function
    """
    # Create the output array
    res = np.zeros((expand_matrix.sum(), expand_matrix.shape[1]), dtype=object)

    # Compute the cumulative row indices (flattened order, then reshaped)
    line_count = np.cumsum(expand_matrix.ravel()) - 1
    line_count = line_count.reshape(expand_matrix.shape)

    # Compute the product in a vectorized way
    prod = (expand_matrix * catalog[:, None]).ravel()
    indices = np.argwhere(prod != 0)

    # Create an array of column indices that match the row-major flattening order
    cols = np.tile(np.arange(expand_matrix.shape[1]), expand_matrix.shape[0])

    # Use fancy indexing to assign the values
    res[line_count.ravel()[indices], cols[indices]] = prod[indices]

    return res