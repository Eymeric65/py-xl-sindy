"""
this script is used by mujoco_align.py in order to generate catalog of function and reference lagrangian for the xl_sindy algorithn

it can be used as a template for the xlsindy_back_script argument of the mujoco_align.py script and should strictly follow the input output format
"""


import xlsindy
import numpy as np
import sympy as sp

mujoco_angle_offset = np.pi


def xlsindy_component(mode="xlsindy"): # Name of this function should not be changed
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

    link1_length = 1.0
    link2_length = 1.0
    mass1 = 0.8
    mass2 = 0.4

    friction_coeff = [-1.0, -1.4]

    friction_forces = np.array([[friction_coeff[0]+friction_coeff[1],-friction_coeff[1]],[-friction_coeff[1],friction_coeff[1]]])
    friction_function = np.array([[symbols_matrix[2, x] for x in range(num_coordinates)]])
    # Assign ideal model variables
    theta1 = symbols_matrix[1, 0]
    theta1_d = symbols_matrix[2, 0]
    theta1_dd = symbols_matrix[3, 0]

    theta2 = symbols_matrix[1, 1]
    theta2_d = symbols_matrix[2, 1]
    theta2_dd = symbols_matrix[3, 1]

    m1, l1, m2, l2, g = sp.symbols("m1 l1 m2 l2 g")
    total_length = link1_length + link2_length
    substitutions = {"g": 9.81, "l1": link1_length, "m1": mass1, "l2": link2_length, "m2": mass2}

    # Lagrangian (L)
    Lagrangian = (0.5 * (m1 + m2) * l1 ** 2 * theta1_d ** 2 + 0.5 * m2 * l2 ** 2 * theta2_d ** 2 + m2 * l1 * l2 * theta1_d
        * theta2_d * sp.cos(theta1 - theta2) + (m1 + m2) * g * l1 * sp.cos(theta1) + m2 * g * l2 * sp.cos(theta2))  

    if mode=="xlsindy":
        # Create the catalog (Mandatory part)
        function_catalog_1 = [lambda x: symbols_matrix[2, x]]
        function_catalog_2 = [lambda x: sp.sin(symbols_matrix[1, x]), lambda x: sp.cos(symbols_matrix[1, x])]

        catalog_part1 = np.array(xlsindy.catalog_gen.generate_full_catalog(function_catalog_1, num_coordinates, 2))
        catalog_part2 = np.array(xlsindy.catalog_gen.generate_full_catalog(function_catalog_2, num_coordinates, 2))
        cross_catalog = np.outer(catalog_part2, catalog_part1)

        lagrange_catalog = np.concatenate(([1],cross_catalog.flatten(), catalog_part1, catalog_part2)) # Maybe not ?

        friction_catalog = friction_function.flatten() # Contain only \dot{q}_1 \dot{q}_2
        expand_matrix = np.ones((len(friction_catalog),num_coordinates),dtype=int)
        catalog_repartition=[("lagrangian",lagrange_catalog),("classical",friction_catalog,expand_matrix)]
        # give a reference lagrangian for the system analysed (optional) through the extra_info dictionary
        
        # Generate solution vector
        ideal_lagrangian_vector = xlsindy.catalog_gen.create_solution_vector(sp.expand_trig(Lagrangian.subs(substitutions)), lagrange_catalog)
        ideal_friction_vector = np.reshape(friction_forces,(-1,1))

        ideal_solution_vector=np.concatenate((ideal_lagrangian_vector,ideal_friction_vector),axis=0)
        catalog_len = len(lagrange_catalog)+np.sum(expand_matrix)

    elif mode =="sindy":

        newton_equations = xlsindy.euler_lagrange.newton_from_lagrangian(Lagrangian.subs(substitutions),symbols_matrix,time_sym)
        newton_system = []

        newton_equations += (friction_function@friction_forces).flatten()

        #print(newton_equations)

        for i in range(num_coordinates):

            newton_system+=[xlsindy.catalog_gen.get_additive_equation_term(sp.expand_trig(newton_equations[i]))]

        catalog_need, coeff_matrix, binary_matrix = xlsindy.catalog_gen.sindy_create_coefficient_matrices(newton_system)


        solution = xlsindy.catalog_gen.translate_coeff_matrix(coeff_matrix,binary_matrix)

        complete_catalog = catalog_need
        complete_expand_matrix =binary_matrix


        catalog_repartition=[("classical",complete_catalog,complete_expand_matrix)]
        ideal_solution_vector = solution
        catalog_len = len(complete_catalog)


    # Create the extra_info dictionnary 
    extra_info = {
        "lagrangian": Lagrangian,
        "substitutions": substitutions,
        "friction_forces": friction_forces,
        "ideal_solution_vector": ideal_solution_vector,
        "initial_condition": np.array([[np.pi, 0], [np.pi, 0]]),
        "catalog_len": catalog_len
    }

    return num_coordinates, time_sym, symbols_matrix, catalog_repartition, extra_info # extra_info is optionnal and should be set to None if not in use


def mujoco_transform(pos,vel,acc,forces):

    pos = -np.cumsum(pos, axis=1) + mujoco_angle_offset
    vel = -np.cumsum(vel, axis=1)
    acc = -np.cumsum(acc, axis=1)

    forces[:,:-1]-=forces[:,1:] 

    return pos,vel,acc,forces

def forces_wrapper(fun):

    def wrapper(*args, **kwargs):

        forces = fun(*args, **kwargs)

        forces[:-1]-=forces[1:]

        return forces
    
    return wrapper
