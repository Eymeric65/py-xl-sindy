"""
this script is used by mujoco_align.py in order to generate catalog of function and reference lagrangian for the xl_sindy algorithn

it can be used as a template for the xlsindy_back_script argument of the mujoco_align.py script and should strictly follow the input output format
"""

import xlsindy
import numpy as np
import sympy as sp
from typing import List

mujoco_angle_offset = np.pi


def xlsindy_component(
    mode: str = "xlsindy", random_seed: List[int] = [12], sindy_catalog_len: int = 289
):  # Name of this function should not be changed
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

    num_coordinates = 3

    symbols_matrix = xlsindy.catalog_gen.generate_symbolic_matrix(
        num_coordinates, time_sym
    )

    # give a reference lagrangian for the system analysed (optional) through the extra_info dictionary

    link1_length = 1.0
    link2_length = 1.0
    massb = 0.8
    mass1 = 0.2
    mass2 = 0.5

    friction_coeff = [-1.5, -1.8, -1.2]
    # friction_coeff = [-0.0,-0.0, -0.0]

    friction_forces = np.array(
        [
            [friction_coeff[0], 0, 0],
            [0, friction_coeff[1] + friction_coeff[2], -friction_coeff[2]],
            [0, -friction_coeff[2], friction_coeff[2]],
        ]
    )
    friction_function = np.array(
        [[symbols_matrix[2, x] for x in range(num_coordinates)]]
    )
    # Assign ideal model variables
    theta1 = symbols_matrix[1, 0]
    theta1_d = symbols_matrix[2, 0]
    theta1_dd = symbols_matrix[3, 0]

    theta2 = symbols_matrix[1, 1]
    theta2_d = symbols_matrix[2, 1]
    theta2_dd = symbols_matrix[3, 1]

    theta3 = symbols_matrix[1, 2]
    theta3_d = symbols_matrix[2, 2]
    theta3_dd = symbols_matrix[3, 2]

    mb, m1, l1, m2, l2, g = sp.symbols("mb m1 l1 m2 l2 g")
    substitutions = {
        "g": 9.81,
        "mb": massb,
        "l1": link1_length,
        "m1": mass1,
        "l2": link2_length,
        "m2": mass2,
    }

    Lagrangian = (
        0.5 * (m1 + m2) * l1**2 * theta2_d**2
        + 0.5 * m2 * l2**2 * theta3_d**2
        + m2
        * l1
        * l2
        * theta2_d
        * theta3_d
        * sp.cos(theta2 - theta3)  # these term are double pendulum related
        + (m1 + m2) * g * l1 * sp.cos(theta2)
        + m2 * g * l2 * sp.cos(theta3)  # these term are potential
        + 0.5 * (mb + m1 + m2) * theta1_d**2
        + ((m1 + m2) * l1 * sp.cos(theta2) * theta1_d * theta2_d)
        + m2 * l2 * sp.cos(theta3) * theta3_d * theta1_d
    )  # These terms are specific to the cartpole

    if mode == "xlsindy":
        # Create the catalog (Mandatory part)
        function_catalog_1 = [lambda x: symbols_matrix[2, x]]
        function_catalog_2 = [
            lambda x: sp.sin(symbols_matrix[1, x]),
            lambda x: sp.cos(symbols_matrix[1, x]),
        ]

        catalog_part1 = np.array(
            xlsindy.catalog_gen.generate_full_catalog(
                function_catalog_1, num_coordinates, 2
            )
        )
        catalog_part2 = np.array(
            xlsindy.catalog_gen.generate_full_catalog(
                function_catalog_2, num_coordinates, 2
            )
        )
        cross_catalog = np.outer(catalog_part2, catalog_part1)
        lagrange_catalog = np.concatenate(
            ([1], cross_catalog.flatten(), catalog_part1, catalog_part2)
        )  # Maybe not ?

        friction_catalog = (
            friction_function.flatten()
        )  # Contain only \dot{q}_1 \dot{q}_2
        expand_matrix = np.ones((len(friction_catalog), num_coordinates), dtype=int)
        catalog_repartition = [
            ("lagrangian", lagrange_catalog),
            ("classical", friction_catalog, expand_matrix),
        ]

        # Generate solution vector
        ideal_lagrangian_vector = xlsindy.catalog_gen.create_solution_vector(
            sp.expand_trig(Lagrangian.subs(substitutions)), lagrange_catalog
        )
        ideal_friction_vector = np.reshape(friction_forces, (-1, 1))

        ideal_solution_vector = np.concatenate(
            (ideal_lagrangian_vector, ideal_friction_vector), axis=0
        )
        catalog_len = len(lagrange_catalog) + np.sum(expand_matrix)

    elif mode == "sindy":

        newton_equations = xlsindy.euler_lagrange.newton_from_lagrangian(
            Lagrangian.subs(substitutions), symbols_matrix, time_sym
        )
        newton_system = []

        newton_equations += (friction_function @ friction_forces).flatten()

        # print(newton_equations)

        for i in range(num_coordinates):

            newton_system += [
                xlsindy.catalog_gen.get_additive_equation_term(newton_equations[i])
            ]

        catalog_need, coeff_matrix, binary_matrix = (
            xlsindy.catalog_gen.sindy_create_coefficient_matrices(newton_system)
        )

        # complete the catalog

        function_catalog_0 = [lambda x: symbols_matrix[3, x]]  # \ddot{x}
        function_catalog_1 = [lambda x: symbols_matrix[2, x]]  # \ddot{x}
        function_catalog_2 = [
            lambda x: sp.sin(symbols_matrix[1, x]),
            lambda x: sp.cos(symbols_matrix[1, x]),
        ]

        catalog_part0 = np.array(
            xlsindy.catalog_gen.generate_full_catalog(
                function_catalog_0, num_coordinates, 1
            )
        )
        catalog_part1 = np.array(
            xlsindy.catalog_gen.generate_full_catalog(
                function_catalog_1, num_coordinates, 2
            )
        )
        catalog_part2 = np.array(
            xlsindy.catalog_gen.generate_full_catalog(
                function_catalog_2, num_coordinates, 2
            )
        )

        lagrange_catalog = xlsindy.catalog_gen.cross_catalog(
            catalog_part1, catalog_part2
        )
        lagrange_catalog = xlsindy.catalog_gen.cross_catalog(
            lagrange_catalog, catalog_part0
        )
        # --------------------

        coeff_matrix, binary_matrix, catalog_need = xlsindy.catalog_gen.augment_catalog(
            num_coordinates,
            lagrange_catalog,
            coeff_matrix,
            binary_matrix,
            catalog_need,
            sindy_catalog_len,
            random_seed,
        )

        solution = xlsindy.catalog_gen.translate_coeff_matrix(
            coeff_matrix, binary_matrix
        )

        catalog_repartition = [("classical", catalog_need, binary_matrix)]
        ideal_solution_vector = solution
        catalog_len = np.sum(binary_matrix)

    # Create the extra_info dictionnary
    extra_info = {
        "lagrangian": Lagrangian,
        "substitutions": substitutions,
        "friction_forces": friction_forces,
        "ideal_solution_vector": ideal_solution_vector,
        "initial_condition": np.array([[0, 0], [np.pi, 0], [np.pi, 0]]),
        "catalog_len": catalog_len,
    }

    return (
        num_coordinates,
        time_sym,
        symbols_matrix,
        catalog_repartition,
        extra_info,
    )  # extra_info is optionnal and should be set to None if not in use


def mujoco_transform(pos, vel, acc, forces):

    pos[:, 1:] = np.cumsum(pos[:, 1:], axis=1) - mujoco_angle_offset
    vel[:, 1:] = np.cumsum(vel[:, 1:], axis=1)
    acc[:, 1:] = np.cumsum(acc[:, 1:], axis=1)

    forces[:, 1] -= forces[:, 2]

    return -pos, -vel, -acc, forces


def forces_wrapper(fun):

    def wrapper(*args, **kwargs):

        forces = fun(*args, **kwargs)

        forces[1] -= forces[2]

        return forces

    return wrapper
