from .render import printProgress
from scipy import interpolate
from scipy.integrate import RK45


def dynamics_function(acceleration_function, external_forces):
    """
    Transforms an array of acceleration functions into a dynamics function for integration.

    Args:
        acceleration_function (function): Array of functions representing accelerations.
        external_forces (function): Function returning external forces at time `t`.

    Returns:
        function: Dynamics function compatible with RK45 integration.
    """
    def func(t, state):
        state = np.reshape(state, (-1, 2))
        state_transposed = np.transpose(state)

        # Prepare input matrix for dynamics calculations
        input_matrix = np.zeros((state_transposed.shape[0] + 2, state_transposed.shape[1]))
        input_matrix[1:3, :] = state_transposed
        input_matrix[0, :] = external_forces(t)

        result = np.zeros(state.shape)
        result[:, 0] = state[:, 1]
        result[:, 1] = acceleration_function(input_matrix)[:, 0]
        return np.reshape(result, (-1,))

    return func

def dynamics_function_with_fixed_forces(acceleration_function):
    """
    Generates a dynamics function with fixed external forces.

    Args:
        acceleration_function (function): Function array representing accelerations.

    Returns:
        function: Dynamics function for use with RK45.
    """
    def func(t, state, fixed_external_forces):
        state = np.reshape(state, (-1, 2))
        state_transposed = np.transpose(state)

        # Prepare input matrix for dynamics calculations
        input_matrix = np.zeros((state_transposed.shape[0] + 2, state_transposed.shape[1]))
        input_matrix[1:3, :] = state_transposed
        input_matrix[0, :] = fixed_external_forces

        result = np.zeros(state.shape)
        result[:, 0] = state[:, 1]
        result[:, 1] = acceleration_function(input_matrix)[:, 0]
        return np.reshape(result, (-1,))

    return func

def run_rk45_integration(dynamics, initial_state, time_end, max_step=0.05):
    """
    Runs an RK45 integration on a dynamics model.

    Args:
        dynamics (function): Dynamics function for integration.
        initial_state (np.ndarray): Initial state of the system.
        time_end (float): End time for the integration.
        max_step (float, optional): Maximum step size for the integration. Defaults to 0.05.

    Returns:
        tuple: Arrays of time values and states.
    """
    initial_state_flat = np.reshape(initial_state, (-1,))
    model = RK45(dynamics, 0, initial_state_flat, time_end, max_step, 0.001, np.e ** -6)

    time_values = []
    state_values = []

    try:
        while model.status != "finished":
            for _ in range(200):
                if model.status != "finished":
                    model.step()
                    time_values.append(model.t)
                    state_values.append(model.y)
            printProgress(model.t, time_end)

    except RuntimeError:
        print("RuntimeError in RK45 integration")

    return np.array(time_values), np.array(state_values)