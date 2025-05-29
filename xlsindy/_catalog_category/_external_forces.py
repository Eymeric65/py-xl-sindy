"""
Contains the function responsible for the external forces part of the catalog.
"""

from typing import List
import numpy as np

def _expand_catalog(
    forces_correlation: List[List[int]],
    symbol_matrix: np.ndarray,
) -> np.ndarray:
    """
    Add the external forces in the experiment matrix

    Returns:
        np.ndarray: an array of shape (1,n) containing all the function
    """

    num_coordinate = symbol_matrix.shape[1]

    res = np.empty((1, num_coordinate), dtype=object)

    for i,additive in enumerate(forces_correlation):

        for index in additive:

            if res[0,i] is None :

                res[0,i] = np.sign(index)*symbol_matrix[0,np.abs(index)-1]

            else:
                
                res[0,i] += np.sign(index)*symbol_matrix[0,np.abs(index)-1]

    return res

def _create_solution_vector() -> np.ndarray:
    """
    Create a solution vector for the external forces in the experiment matrix. (return -1)
    
    Returns:
        np.ndarray: A column vector of shape (1, 1) containing the coefficients for the external forces. (-1)
    """
    return np.array(-1).reshape(-1, 1)