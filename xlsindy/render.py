"""

This module contain some render function for the basics experiment

"""

import sys
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.patches import FancyBboxPatch

def plot_bipartite_graph_svg(x_names, b_names, edges, x_sol_indices, output_file="fancy_bipartite_graph.svg"):
    """
    Plots a modern, fancy bipartite graph with custom node shapes and updated colors,
    then saves the plot as an SVG file.
    
    Parameters:
      x_names (list of str): Names for the x variables.
      b_names (list of str): Names for the b groups.
      edges (list of tuple): Each tuple (x_name, b_name) represents an edge from an x-node to a b-node.
      x_sol_indices (list of int): List of indices (with respect to x_names) corresponding to important (solved) x nodes.
      output_file (str): Filename for the output SVG.
    
    Customizations:
      - x-nodes are drawn as plain rectangles.
      - b-nodes are drawn as rounded rectangles.
      - Edges originating from an important x-node are drawn in a modern red (thicker),
        others in a dashed soft gray-blue.
      - A modern pastel color scheme is used with generous margins.
    """
    
    # Build the bipartite graph.
    G = nx.Graph()
    for x in x_names:
        G.add_node(x, bipartite=0)
    for b in b_names:
        G.add_node(b, bipartite=1)
    for edge in edges:
        if edge[0] in x_names and edge[1] in b_names:
            G.add_edge(edge[0], edge[1])
        else:
            raise ValueError(f"Edge {edge} contains a node not provided in the node lists.")
    
    # Compute a bipartite layout.
    pos = nx.bipartite_layout(G, x_names)
    
    # Determine the set of important (solved) x nodes.
    important_x = {x_names[i] for i in x_sol_indices}
    
    # Separate edges: those that "originate" from an important x-node and the others.
    important_edges = []
    other_edges = []
    for u, v in G.edges():
        if u in x_names and v in b_names:
            xnode = u
        elif v in x_names and u in b_names:
            xnode = v
        else:
            continue
        if xnode in important_x:
            important_edges.append((u, v))
        else:
            other_edges.append((u, v))
    
    # Create the figure and axis.
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("#f0f2f5")  # Light modern background.
    
    # Draw edges first.
    nx.draw_networkx_edges(G, pos,
                           edgelist=important_edges,
                           edge_color="#FF8A80",  # Modern red for important edges.
                           width=2,
                           ax=ax)
    nx.draw_networkx_edges(G, pos,
                           edgelist=other_edges,
                           edge_color="#B0BEC5",  # Soft gray-blue.
                           style="dashed",
                           width=1,
                           ax=ax)
    
    # --- Custom drawing of nodes via patches ---
    # Define sizes for node patches (adjust if labels are very long).
    width_x, height_x = 0.5, 0.12  # For x nodes (plain rectangles).
    width_b, height_b = 0.45, 0.14  # For b nodes (rounded rectangles).
    
    # Colors using a modern pastel palette.
    color_important_x = "#A8E6CF"  # Pastel green.
    color_unsolved_x = "#FFD3B6"   # Pastel peach.
    color_b = "#BBDEFB"           # Pastel blue.
    
    node_offset = 0.2  # Offset for node labels.

    # Draw x nodes as plain rectangles.
    for node in x_names:
        cx, cy = pos[node]
        cx-= node_offset  # Offset for x nodes.
        lower_left = (cx - width_x/2, cy - height_x/2)
        facecolor = color_important_x if node in important_x else color_unsolved_x
        patch = FancyBboxPatch(lower_left, width_x, height_x,
                               boxstyle="round,pad=0.02",
                               linewidth=2,
                               edgecolor="black", facecolor=facecolor,
                               mutation_scale=1.5)
        ax.add_patch(patch)
        ax.text(cx, cy, node, ha="center", va="center", fontsize=10, color="black")
    
    # Draw b nodes as rounded rectangles.
    for node in b_names:
        cx, cy = pos[node]
        cx+= node_offset  # Offset for x nodes.
        lower_left = (cx - width_b/2, cy - height_b/2)
        patch = FancyBboxPatch(lower_left, width_b, height_b,
                               boxstyle="square,pad=0.02",
                               linewidth=2,
                               edgecolor="black", facecolor=color_b,
                               mutation_scale=1.5)
        ax.add_patch(patch)
        ax.text(cx, cy, node, ha="center", va="center", fontsize=10, color="black")
    
    # Final touches.
    plt.title("Fancy Bipartite Graph: x Variables and b Groups", fontsize=16, weight="bold", pad=20)
    ax.set_axis_off()
    # Increase margins so that nodes are not cut off.
    ax.margins(0.1)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Save the output as an SVG file.
    plt.savefig(output_file, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Graph saved as {output_file}")



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
    length1: float,
    length2: float,
    angle_array: np.ndarray,
    time_array: np.ndarray,
    fig=None,
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
