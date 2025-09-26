"""
This file contains functions used to visualize the CTRNN model.
"""
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import graphviz
from sklearn.decomposition import PCA
from typing import Optional, Union

def draw_ctrnn_net(node_list: list, node_inputs: dict, iznn: Optional[bool] = False, dir_name: Optional[str] = None, file_name: Optional[str] = None) -> None:
    """
    This function draws the CTRNN network structure.
    Args:
        node_list: A list of node IDs in the network.
        node_inputs: A dictionary where keys are node IDs and values are lists of input connections for each node.
            (I.e. each list contains tuples of (input_node_id, weight) for the node with the corresponding ID)
        iznn: Whether the network is an Izhikevich spiking neural network (IZNN). If True, the function gives correct labels.
        dir_name: Optional directory name to save the output file. If None, saves in the current directory.
        file_name: Optional file name to save the output file. If None, defaults to 'ctrnn_network'.
    Returns:
        None
    """

    dot = graphviz.Digraph()

    for node_id in node_list:
        dot.node(str(node_id))

    for node_id, inputs in node_inputs.items():
        for input_node_id, weight in inputs:
            dot.edge(str(input_node_id), str(node_id), label=str(weight))

    dot.render(file_name or 'iznn_network' if iznn else 'ctrnn_network', format='png', cleanup=True, directory=dir_name or '.')

def draw_ctrnn_dynamics(states: np.ndarray, uniform_time: bool = True, times: Optional[Union[np.ndarray, list]] = None, 
                        iznn: Optional[bool] = False, save: bool = False, show: bool = True, 
                        dir_name: Optional[str] = None, file_name: Optional[str] = None) -> None:
    """
    This function draws the dynamics of the CTRNN over time.
    Args:
        states: A 2D numpy array where each row corresponds to the state of the network at a given time step,
            and each column corresponds to a specific node's state.
        uniform_time: Whether the time steps are uniform. If True, the function generates a uniform time array.
        times: If uniform_time is False, a list or numpy array of time points corresponding to each row in states.
            If uniform_time is True, this parameter is ignored.
        iznn: Whether the network is an Izhikevich spiking neural network (IZNN). If True, the function gives correct labels.
        save: Whether to save the plot as a file. If False, the plot is shown interactively.
        show: Whether to display the plot interactively. If False, the plot is only saved to a file if 'save' is True.
        dir_name: Optional directory name to save the output file. If None, saves in the current directory.
        file_name: Optional file name to save the output file. If None, defaults to 'ctrnn_dynamics'.
    Returns:
        None
    """
    if uniform_time:
        times = np.arange(states.shape[0])
    else:
        if times is None or len(times) != states.shape[0]:
            raise ValueError("Invalid times array. Must be provided and match the number of time steps in states.")

    plt.figure()
    plt.title(f"{'IZNN' if iznn else 'CTRNN'} Dynamics")
    plt.xlabel("Time")
    plt.ylabel("Output")
    plt.grid()

    for i in range(states.shape[1]):
        plt.plot(times, states[:, i], label=f"Node {i+1}")

    plt.legend(loc="best")
    if save:
        plt.savefig(f"{dir_name + '/' if dir_name else ''}{file_name or 'iznn_dynamics' if iznn else 'ctrnn_dynamics'}.png")
    if show:
        plt.show()

def draw_ctrnn_trajectory(states: np.ndarray, n_components: int = 2, iznn: Optional[bool] = False, save: bool = False, show: bool = True, dir_name: Optional[str] = None, file_name: Optional[str] = None) -> None:
    """
    This function draws a trajectory of the CTRNN's state space.
    If there are more than 'n_components' nodes, the PCA is used to reduce the dimensionality to 'n_components'.
    Args:
        states: A 2D numpy array where each row corresponds to the state of the network at a given time step,
        and each column corresponds to a specific node's state.
        n_components: The number of components to reduce to (default is 2, max is 3).
        iznn: Whether the network is an Izhikevich spiking neural network (IZNN). If True, the function gives correct labels.
        save: Whether to save the plot as a file. If False, the plot is shown interactively.
        show: Whether to display the plot interactively. If False, the plot is only saved to a file if 'save' is True.
        dir_name: Optional directory name to save the output file. If None, saves in the current directory.
        file_name: Optional file name to save the output file. If None, defaults to 'ctrnn_face_portrait'.
    Returns:
        None
    Raises:
        ValueError: If n_components is not 1, 2, or 3.
    """
    if n_components < 1 or n_components > 3:
        raise ValueError("Invalid number of components. Must be 1, 2, or 3.")

    if states.shape[1] > n_components:
        pca = PCA(n_components=n_components)
        reduced_states = pca.fit_transform(states)
    elif states.shape[1] < n_components:
        reduced_states = states
        raise RuntimeWarning(f"Cannot reduce to {n_components} components from {states.shape[1]} nodes. Falling back to {states.shape[1]}D.")
    else:
        reduced_states = states

    if n_components == 1:
        plt.figure(figsize=(10,5))
        plt.title(f"{'IZNN' if iznn else 'CTRNN'} Trajectory ({n_components}D)")
        plt.xlabel("Time")
        plt.ylabel("Principal Component 1")
        plt.grid()

        plt.plot(range(reduced_states.shape[1]), reduced_states[:, 0], color='b', marker='o', markersize=3)
        # denoting the start and end points
        plt.text(range(reduced_states.shape[1])[0], reduced_states[0][0], 'Start', fontsize=12, color='green')
        plt.text(range(reduced_states.shape[1])[-1], reduced_states[-1][0], 'End', fontsize=12, color='red')

    elif n_components == 2:
        plt.figure(figsize=(10,10))
        plt.title(f"{'IZNN' if iznn else 'CTRNN'} Trajectory ({n_components}D)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid()

        plt.plot(reduced_states[:, 0], reduced_states[:, 1], color='b', marker='o', markersize=3)
        # denoting the start and end points
        plt.text(reduced_states[0][0], reduced_states[0][1], 'Start', fontsize=12, color='green')
        plt.text(reduced_states[-1][0], reduced_states[-1][1], 'End', fontsize=12, color='red')

    elif n_components == 3:
        plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        ax.grid()

        ax.set_title(f"{'IZNN' if iznn else 'CTRNN'} Trajectory ({n_components}D)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")

        ax.plot3D(reduced_states[:, 0], reduced_states[:, 1], reduced_states[:, 2], color='b', marker='o', markersize=3)
        # denoting the start and end points
        ax.text(reduced_states[0][0], reduced_states[0][1], reduced_states[0][2], 'Start', fontsize=12, color='green')
        ax.text(reduced_states[-1][0], reduced_states[-1][1], reduced_states[-1][2], 'End', fontsize=12, color='red')

    if save:
        plt.savefig(f"{dir_name + '/' if dir_name else ''}{file_name or 'iznn_trajectory' if iznn else 'ctrnn_trajectory'}.png")
    if show:
        plt.show()