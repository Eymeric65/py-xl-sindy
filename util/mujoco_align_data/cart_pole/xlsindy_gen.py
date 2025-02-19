"""
this script is used by mujoco_align.py in order to generate catalog of function and reference lagrangian for the xl_sindy algorithn

it can be used as a template for the xlsindy_back_script argument of the mujoco_align.py script and should strictly follow the input output format
"""


import xlsindy
import numpy as np
import sympy as sp

def xlsindy_component(): # Name of this function should not be changed
    """
    This function is used to generate backbone of the xl_sindy algorithm
    
    this version can be used as a template for the xlsindy_back_script argument of the mujoco_align.py script and should strictly follow the input output format.
    name of the function should not be changed
    
    Returns:
        np.ndarray: matrix of shape (4, n) containing symbolic expression.
        List[sympy.Expr]: List of combined functions.
        Dict: extra_info dictionnary containing extra information about the system
    """

    time_sym = sp.symbols("t")

    num_coordinates = 2

    symbols_matrix = xlsindy.catalog_gen.generate_symbolic_matrix(num_coordinates, time_sym)

    # Create the catalog (Mandatory part)
    function_catalog_1 = [lambda x: symbols_matrix[2, x]] # \dot{x}
    function_catalog_2 = [lambda x: sp.sin(symbols_matrix[1, x]), lambda x: sp.cos(symbols_matrix[1, x])]

    catalog_part1 = np.array(xlsindy.catalog_gen.generate_full_catalog(function_catalog_1, num_coordinates, 2))
    catalog_part2 = np.array(xlsindy.catalog_gen.generate_full_catalog(function_catalog_2, num_coordinates, 2))
    cross_catalog = np.outer(catalog_part2, catalog_part1)

    lagrange_catalog = np.concatenate((cross_catalog.flatten(), catalog_part1, catalog_part2)) # Maybe not ?

    friction_catalog = np.array([symbols_matrix[2, x] for x in range(num_coordinates)]) # Contain only \dot{q}_1 \dot{q}_2
    expand_matrix = np.ones((len(friction_catalog),num_coordinates),dtype=int)
    catalog_repartition=[("lagrangian",lagrange_catalog),("classical",friction_catalog,expand_matrix)]


    # give a reference lagrangian for the system analysed (optional) through the extra_info dictionary

    link_length = 1.0
    mass_base = 0.8
    mass_link = 0.5

    friction_coeff = [-0.8, -1.3]

    friction_forces = np.array([[friction_coeff[0],0],[0,friction_coeff[1]]])


    # Assign ideal model variables
    theta1 = symbols_matrix[1, 0]
    theta1_d = symbols_matrix[2, 0]
    theta1_dd = symbols_matrix[3, 0]

    theta2 = symbols_matrix[1, 1]
    theta2_d = symbols_matrix[2, 1]
    theta2_dd = symbols_matrix[3, 1]

    mb, ml, l1, g = sp.symbols("mb ml l1 g")
    substitutions = {"g": 9.81, "mb": mass_base, "ml": mass_link, "l1": link_length}

    # Lagrangian (L)
    Lagrangian = 1/2 * (ml+mb)*theta1_d**2 + 1/2*ml*l1**2*theta2_d**2 - ml*l1*sp.cos(theta2)*theta2_d*theta1_d-ml*g*l1*sp.cos(theta2)
    
    # Generate solution vector
    ideal_lagrangian_vector = xlsindy.catalog_gen.create_solution_vector(sp.expand_trig(Lagrangian.subs(substitutions)), lagrange_catalog)
    ideal_friction_vector = np.reshape(friction_forces,(-1,1))

    ideal_solution_vector=np.concatenate((ideal_lagrangian_vector,ideal_friction_vector),axis=0)
    # Create the extra_info dictionnary 
    extra_info = {
        "lagrangian": Lagrangian,
        "substitutions": substitutions,
        "friction_forces": friction_forces,
        "ideal_solution_vector": ideal_solution_vector,
        "initial_condition": np.array([[0, 0], [0, 0]]),
        "lagrange_catalog":lagrange_catalog,
        "friction_catalog":friction_catalog,
        "catalog_len": len(lagrange_catalog)+np.sum(expand_matrix)
    }

    

    return num_coordinates, time_sym, symbols_matrix,catalog_repartition, extra_info # extra_info is optionnal and should be set to None if not in use


def mujoco_transform(pos,vel,acc,forces):

    return -pos,-vel,-acc,forces

def forces_wrapper(fun):

    def wrapper(*args, **kwargs):

        forces = fun(*args, **kwargs)

        return forces
    
    return wrapper

