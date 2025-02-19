"""


This file is mainly to create and manage catalog of function that will be use in the xl-sindy algorithm afterward.
"""

import numpy as np
import sympy
from typing import List, Callable, Union

from sympy import latex

from . import euler_lagrange

def generate_symbolic_matrix(coord_count: int, t: sympy.Symbol) -> np.ndarray:
    """
    Creates a symbolic matrix representing external forces and state variables for a number of coordinates.

    This function create the matrix containing all the state variable with following derivatives and external forces.

    +-------------+-------------+-------------+-------------+
    | Fext0(t)    | Fext2(t)    | ...         | Fextn(t)    |
    +-------------+-------------+-------------+-------------+
    | q0(t)       | q2(t)       | ...         | qn(t)       |
    +-------------+-------------+-------------+-------------+
    | q0_d(t)     | q2_d(t)     | ...         | qn_d(t)     |
    +-------------+-------------+-------------+-------------+
    | q0_dd(t)    | q2_dd(t)    | ...         | qn_dd(t)    |
    +-------------+-------------+-------------+-------------+

    Args:
        coord_count (int): Number of coordinates.
        t (sympy.Symbol): Symbol representing time.

    Returns:
        np.ndarray: matrix of shape (4, n) containing symbolic expression.
    """
    symbolic_matrix = np.zeros((4, coord_count), dtype=object)
    symbolic_matrix[0, :] = [sympy.Function(f"Fext_{i}")(t) for i in range(coord_count)]
    symbolic_matrix[1, :] = [sympy.Function(f"q_{i}")(t) for i in range(coord_count)]
    #symbolic_matrix[2, :] = [sympy.Function(f"\\dot{{q_{i}}}")(t) for i in range(coord_count)]
    #symbolic_matrix[3, :] = [sympy.Function(f"\\ddot{{q_{i}}}")(t) for i in range(coord_count)]
    symbolic_matrix[2, :] = [sympy.Function(f"qd_{i}")(t) for i in range(coord_count)]
    symbolic_matrix[3, :] = [sympy.Function(f"qdd_{i}")(t) for i in range(coord_count)]
    return symbolic_matrix


def calculate_forces_vector(
    force_function: Callable[[float], np.ndarray], time_values: np.ndarray
) -> np.ndarray:
    """
    Calculates a vector of forces at given time values.

    Args:
        force_function (Callable[[float], np.ndarray]): Function that returns forces for given time values.
        time_values (np.ndarray): Time values to evaluate.

    Returns:
        np.ndarray: Reshaped force vector.
    """
    force_vector = force_function(time_values)
    force_vector[:-1, :] -= force_vector[
        1:, :
    ]  # Offset forces in order to get local joint forces
    print(force_vector.shape)
    return np.transpose(
        np.reshape(force_vector, (1, -1))
    )  # Makes everything flat for rendering afterwards


def _concatenate_function_symvar(
    function_catalog: List[Callable[[int], sympy.Expr]], q_terms: int
) -> List[sympy.Expr]:
    """
    Concatenates function with symbolic value.

    This function is made to convert the first function catalog into a total list of function.

    it aims to convert this :
        'function_catalog_1 = [lambda x: Symb[2,x]]' (which means : "the catalog of the derived of the general coordinate" )
    into :
        '[Symb[2,0],Symb[2,1],...,Symb[2,n]]' equivalent of [q0_d(t),q1_d(t),...,qn_d(t)]

    Args:
        function_catalog (List[Callable[[int], sympy.Expr]]): List of functions.
        q_terms (int): Number of terms.

    Returns:
        List[sympy.Expr]: List of function values.
    """
    result = []
    for func in function_catalog:
        for j in range(q_terms):
            result.append(func(j))
    return result


def _generate_combination_catalog(
    catalog: List[sympy.Expr], depth: int, func_idx: int, power: int, initial_power: int
) -> List[sympy.Expr]:
    """
    Recursively generates combinations from a catalog of constants or functions.

    The goal is to generate every combinaison of function at a certain power.
    Depth should always been greater or equal to the power (otherwise we won't be able to obtain component at the right power)


    Args:
        catalog (List[sympy.Expr]): List of constants or functions.
        depth (int): Recursion depth. >=power
        func_idx (int): Current function index.
        power (int): Current power level.
        initial_power (int): Initial power level.

    Returns:
        List[sympy.Expr]: List of combinations.
    """
    if depth == 0:  # Return Identity if depth is nul
        return [1]
    else:
        result = []  # Initialize result
        for i in range(
            func_idx + 1, len(catalog)
        ):  # for every next function in the catalog (this triangular approch account for multiplication permutation ability)
            res = _generate_combination_catalog(
                catalog, depth - 1, i, initial_power - 1, initial_power
            )  # get the combinaison at a depth after ( power is equal initial_power -1 because the function is used the line after)
            result += [
                res_elem * catalog[i] for res_elem in res
            ]  # append result with each of the succeding combination multiplied by the function
        if (
            power > 0
        ):  # if the actual function has still power left we can decrease power by 1 and concatenate with the combinaison at one depth and one power less
            res = _generate_combination_catalog(
                catalog, depth - 1, func_idx, power - 1, initial_power
            )
            result += [res_elem * catalog[func_idx] for res_elem in res]
        return result


def generate_full_catalog(
    function_catalog: List[sympy.Expr], q_terms: int, degree: int, power: int = None
) -> List[sympy.Expr]:
    """
    Generates a catalog of linear combinations from a function array until a certain degree/power.

    Args:
        function_catalog (List[sympy.Expr]): List of functions to use.
        q_terms (int): Number of general coordinate.
        degree (int): Maximum degree of combinations.
        power (int, optional): Maximum power level. Defaults to None, in which case it uses `degree`. Need to be inferior or equal to depth in order to generate at least function_i^power in the catalog

    Returns:
        List[sympy.Expr]: List of combined functions.
    """
    catalog = []
    if (
        power is None
    ):  # If no power is specified we assume that user want to generate singleton function^degree
        power = degree

    base_catalog = _concatenate_function_symvar(function_catalog, q_terms)

    for i in range(degree):  # generate for each depth
        catalog += _generate_combination_catalog(base_catalog, i + 1, 0, power, power)
    return catalog

def classical_sindy_expand_catalog(
        catalog:List[sympy.Expr],
        expand_matrix:np.ndarray
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

    # Create an array of column indices that match the row-major flattening order
    cols = np.tile(np.arange(expand_matrix.shape[1]), expand_matrix.shape[0])
    
    # Use fancy indexing to assign the values
    res[line_count.ravel(), cols] = prod

    return res

def lagrangian_sindy_expand_catalog(
        catalog:List[sympy.Expr],
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

    res=np.empty((len(catalog),num_coordinate),dtype=object)
    


    for i in range(num_coordinate):

        catalog_lagrange = list(
            map(
                lambda x: euler_lagrange.compute_euler_lagrange_equation(
                    x, symbol_matrix, time_symbol, i
                ),
                catalog,
            )
        )
        res[:,i] = catalog_lagrange

    return res

def expand_catalog(
        catalog_repartition:List[tuple],
        symbol_matrix: np.ndarray,
        time_symbol: sympy.Symbol,
):
    """
    create a global catalog for the regression system

    Args:
        catalog_repartition (List[tuple]): a listing of the different part of the catalog used need to follow the following structure : [("lagrangian",lagrangian_catalog),...,("classical",classical_catalog,expand_matrix)]
        symbol_matrix (np.ndarray): The matrix of symbolic variables (external forces, positions, velocities, and accelerations).
        time_symbol (sp.Symbol): The symbolic variable representing time.
    """

    res=[]

    #print("debug :",catalog_repartition)

    for catalog in catalog_repartition:

        name,*args=catalog

        if name == "lagrangian":

            res+=[lagrangian_sindy_expand_catalog(*args,symbol_matrix,time_symbol)]

        elif name == "classical":

            res+=[classical_sindy_expand_catalog(*args)]

        else:
            raise ValueError("catalog not recognised")
        
        print(res[-1].shape)
        
    return np.concatenate(res,axis=0)


def separate_catalog(
    catalog:np.ndarray,
    catalog_repartition:List[tuple]
):
    """
    separate a catalog that has been concatenated in the catalog_repartition order

    Args:
        catalog (np.ndarray)
    """

def create_solution_vector(
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

    # if friction_terms is None :
    #     friction_len = 0
    # else:
    #     friction_len = np.prod(friction_terms.shape)

    expanded_expression_terms = sympy.expand(
        sympy.expand_trig(expression)
    ).args  # Expand the expression in order to get base function (ex: x, x^2, sin(s), ...)
    solution_vector = np.zeros((len(catalog), 1))
    #solution_vector = np.zeros((len(catalog) + friction_len, 1))
    for term in expanded_expression_terms:
        for idx, catalog_term in enumerate(catalog):
            test = term / catalog_term
            if (
                len(test.args) == 0
            ):  # if test is a constant it means that catalog_term is inside equation
                solution_vector[idx, 0] = test

    # if friction_terms is not None:
    #     solution_vector[len(catalog):len(catalog)+friction_len] = np.reshape(friction_terms,(-1,1))

    # for i, friction_value in enumerate(friction_terms): # old friction paradigm
    #     solution_vector[len(catalog) + i] = friction_value
    return solution_vector


def create_solution_expression(
    solution_vector: np.ndarray, 
    catalog: List[sympy.Expr],
    num_coordinates:int = None,
    first_order_friction:bool=False
) -> sympy.Expr:
    """
    Constructs an expression from a solution vector and a catalog.

    Args:
        solution_vector (np.ndarray): Solution values.
        catalog (List[Union[int, float]]): List of functions or constants.
        num_coordinates (int): Number of coordinates, used to return the friction term
        first_order_friction (bool): Wether or not the generation used the first order friction paradigm

    Returns:
        sympy.Expr: Constructed solution expression, litteraly construct solution_vector*catalog.T. Friction term are excluded
        np.ndarray: First Order friction matrix
    """
    model_expression = 0
    for i in range(len(catalog)):
        model_expression += solution_vector[i] * catalog[i]

    if first_order_friction :
        if num_coordinates is None:
            raise TypeError("You should specify the number of coordinate if you want to retrieve the first order friction matrix")
        
        first_order_friction_matrix = np.reshape(solution_vector[len(catalog):len(catalog)+num_coordinates**2],(num_coordinates,num_coordinates))

    else:
        first_order_friction_matrix = None
    
    return model_expression,first_order_friction_matrix

def label_catalog(catalog,non_null_term):
    """Convert the catalog into label"""
    res=[]
    for index in non_null_term[:,0]:

        if index > len(catalog)-1:
            res+=[f"fluid forces $v_{{{index-len(catalog)}}}$"]
        else:
            res+= [f"${latex(catalog[index]).replace("qd","\\dot{{q}}")}$"]
    return res