"""
Contains the function responsible for the lagrangian part of the catalog.
"""

import sympy 
from typing import List, Union
import numpy as np
from .. import euler_lagrange

from ..catalog import CatalogCategory

class Lagrange(CatalogCategory):
    """
    Lagrange based catalog. 

    Args:
        interlink_list (List[List[int]]) : Presence of the forces on each of the coordinate, 1-indexed can be negative for retroactive forces.
        symbol_matrix (np.ndarray) : Symbolic variable matrix for the system.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def create_solution_vector(self, *args, **kwargs):
        raise NotImplementedError

    def expand_catalog(self):
        raise NotImplementedError

    def label(self):
        raise NotImplementedError


def _create_solution_vector(
    expression: sympy.Expr,
    catalog: List[Union[int, float]],
) -> np.ndarray:
    """
    Creates a solution vector by matching expression terms to a catalog.

    Args:
        expression (sympy.Expr): The equation to match.
        catalog (List[Union[int, float]]): List of functions or constants to match against.

    Returns:
        np.ndarray: Solution vector containg the coefficient in order that return*catalog=expression.
    """

    expanded_expression_terms = sympy.expand(
        sympy.expand_trig(expression)
    ).args  # Expand the expression in order to get base function (ex: x, x^2, sin(s), ...)
    solution_vector = np.zeros((len(catalog), 1))

    for term in expanded_expression_terms:
        for idx, catalog_term in enumerate(catalog):
            test = term / catalog_term
            if (
                len(test.args) == 0
            ):  # if test is a constant it means that catalog_term is inside equation
                solution_vector[idx, 0] = test

    return solution_vector

def _expand_catalog(
    catalog: List[sympy.Expr],
    symbol_matrix: np.ndarray,
    time_symbol: sympy.Symbol,
) -> np.ndarray:
    """
    expand the catalog in the case of a XlSINDy experiment

    Args:
        catalog (List[sympy.Expr]): the list of function to expand
        symbol_matrix (np.ndarray): The matrix of symbolic variables (external forces, positions, velocities, and accelerations).
        time_symbol (sp.Symbol): The symbolic variable representing time.

    Returns:
        np.ndarray: an array of shape (len(catalog),n) containing all the function
    """

    num_coordinate = symbol_matrix.shape[1]

    res = np.empty((len(catalog), num_coordinate), dtype=object)

    for i in range(num_coordinate):

        catalog_lagrange = list(
            map(
                lambda x: euler_lagrange.compute_euler_lagrange_equation(
                    x, symbol_matrix, time_symbol, i
                ),
                catalog,
            )
        )
        res[:, i] = catalog_lagrange

    return res