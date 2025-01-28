"""

This module contain some render function for the basics experiment

"""


import sys
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def animate_single_pendulum(
    length: float, angle_array: np.ndarray, time_array: np.ndarray
):
    """
    Animates a single pendulum based on its length, angle data, and time steps.

    Parameters:
        length (float): Length of the pendulum.
        angle_array (ndarray): Array of angular positions over time.
        time_array (ndarray): Array of time steps corresponding to angles.
    """
    x_coords = length * np.sin(angle_array[:, 0])
    y_coords = -length * np.cos(angle_array[:, 0])

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(
        autoscale_on=False, xlim=(-length, length), ylim=(-length, length)
    )
    ax.set_aspect("equal")
    ax.grid()

    (trace_line,) = ax.plot([], [], ".-", lw=1, ms=2)
    (pendulum_line,) = ax.plot([], [], "o-", lw=2)
    time_template = "time = %.1fs"
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    def update_frame(i):
        current_x = [0, x_coords[i]]
        current_y = [0, y_coords[i]]

        trace_x = x_coords[:i]
        trace_y = y_coords[:i]

        pendulum_line.set_data(current_x, current_y)
        trace_line.set_data(trace_x, trace_y)
        time_text.set_text(time_template % time_array[i])
        return pendulum_line, trace_line, time_text

    ani = animation.FuncAnimation(
        fig, update_frame, len(angle_array), interval=40, blit=True
    )
    plt.show()


def animate_double_pendulum(
    length1: float, length2: float, angle_array: np.ndarray, time_array: np.ndarray, fig= None
):
    """
    Animates a double pendulum based on its segment lengths, angles, and time steps.

    Parameters:
        length1 (float): Length of the first segment.
        length2 (float): Length of the second segment.
        angle_array (ndarray): Array of angular positions of both segments over time.
        time_array (ndarray): Array of time steps corresponding to angles.
    """
    total_length = length1 + length2

    x1 = length1 * np.sin(angle_array[:, 0])
    y1 = -length1 * np.cos(angle_array[:, 0])

    x2 = length2 * np.sin(angle_array[:, 2]) + x1
    y2 = -length2 * np.cos(angle_array[:, 2]) + y1
    if fig == None:
        fig = plt.figure(figsize=(5, 4))

    
    ax = fig.add_subplot(
        autoscale_on=False,
        xlim=(-total_length, total_length),
        ylim=(-total_length, total_length),
    )
    ax.set_aspect("equal")
    ax.grid()

    (trace_line,) = ax.plot([], [], ".-", lw=1, ms=2)
    (pendulum_line,) = ax.plot([], [], "o-", lw=2)
    time_template = "time = %.1fs"
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    def update_frame(i):
        current_x = [0, x1[i], x2[i]]
        current_y = [0, y1[i], y2[i]]

        trace_x = x2[:i]
        trace_y = y2[:i]

        pendulum_line.set_data(current_x, current_y)
        trace_line.set_data(trace_x, trace_y)
        time_text.set_text(time_template % time_array[i])
        return pendulum_line, trace_line, time_text

    ani = animation.FuncAnimation(
        fig, update_frame, len(angle_array), interval=40, blit=True
    )
    plt.show()


def print_progress(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    bar_length: int = 100,
):
    """
    Displays a progress bar in the terminal.

    Parameters:
        iteration (int): Current iteration.
        total (int): Total number of iterations.
        prefix (str, optional): Prefix for the progress bar.
        suffix (str, optional): Suffix for the progress bar.
        decimals (int, optional): Number of decimal places to show in the percentage.
        bar_length (int, optional): Length of the progress bar in characters.
    """
    format_str = "{0:." + str(decimals) + "f}"
    percentage_complete = format_str.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = "*" * filled_length + "-" * (bar_length - filled_length)
    sys.stdout.write(
        "\r%s |%s| %s%s %s" % (prefix, bar, percentage_complete, "%", suffix)
    ),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write("\n")
        sys.stdout.flush()
