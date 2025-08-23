"""
This file contains functions used to visualize the CTRNN model.
"""
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import graphviz
from sklearn.decomposition import PCA

def draw_ctrnn_net(node_list: list, node_inputs: dict):
    """
    This function draws the CTRNN network structure.
    Args:
        node_list: A list of node IDs in the network.
        node_inputs: A dictionary where keys are node IDs and values are lists of input connections for each node.
        (I.e. each list contains tuples of (input_node_id, weight) for the node with the corresponding ID)
    """

    dot = graphviz.Digraph()

    for node_id in node_list:
        dot.node(str(node_id))

    for node_id, inputs in node_inputs.items():
        for input_node_id, weight in inputs:
            dot.edge(str(input_node_id), str(node_id), label=str(weight))

    dot.render('ctrnn_network', format='png', cleanup=True)

def draw_ctrnn_dynamics(states: np.ndarray):
    """
    This function draws the dynamics of the CTRNN over time.
    Args:
        states: A 2D numpy array where each row corresponds to the state of the network at a given time step,
        and each column corresponds to a specific node's state.
    """

    plt.figure()
    plt.title("CTRNN Dynamics")
    plt.xlabel("Time")
    plt.ylabel("Output")
    plt.grid()

    for i in range(states.shape[0]):
        plt.plot(states[i], label=f"Node {i+1}")

    plt.legend(loc="best")
    plt.show()

def draw_ctrnn_face_portrait(states: np.ndarray, n_components: int = 2) -> None:
    """
    This function draws a face portrait of the CTRNN's state space.
    If there are more than 'n_components' nodes, the PCA is used to reduce the dimensionality to 'n_components'.
    Args:
        states: A 2D numpy array where each row corresponds to the state of the network at a given time step,
        and each column corresponds to a specific node's state.
        n_components: The number of components to reduce to (default is 2, max is 3).
    Returns:
        None
    Raises:
        ValueError: If n_components is not 1, 2, or 3.
    """
    if n_components < 1 or n_components > 3:
        raise ValueError("Invalid number of components. Must be 1, 2, or 3.")

    if states.shape[0] > n_components:
        pca = PCA(n_components=n_components)
        reduced_states = pca.fit_transform(states)
    else:
        reduced_states = states

    if n_components == 1:
        plt.figure(figsize=(10,5))
        plt.xlabel("Time")
        plt.ylabel("Principal Component 1")
        plt.grid()

        plt.scatter(range(reduced_states.shape[0]), reduced_states[:, 0], color='b', s=5)

        plt.show()
    elif n_components == 2:
        plt.figure(figsize=(10,10))
        plt.title("CTRNN Face Portrait")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid()

        for i in range(reduced_states.shape[0]):
            plt.scatter(reduced_states[i, 0], reduced_states[i, 1], color='b', s=5)

        plt.show()
    elif n_components == 3:
        plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        ax.grid()

        ax.set_title("CTRNN Face Portrait")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")

        ax.plot3D(reduced_states[:, 0], reduced_states[:, 1], reduced_states[:, 2], color='b')

        plt.show()