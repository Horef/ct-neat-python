Discretization of Continuous Network Dynamics
===============================================
.. index:: ! discretization
.. index:: ! disretizer
.. index:: continuous-time
.. index:: recurrent
.. index:: ! ctrnn

Historically, :term:`continuous-time` :term:`recurrent` neural networks (CTRNN) have been used in 
robotics and other fields to model systems that interact with their environment over continuous time.
In that case, the translation of the dynamics into the output is trivial as the network's state evolves continuously in response to inputs, 
and the output is typically read at specific time intervals.
Therefore, one of biggest questions when working with :term:`continuous-time` :term:`recurrent` neural networks (CTRNN) is how
to convert their continuous dynamics as a response to a given input into a **discrete** output.

One of the possible approaches, implemented in CT-NEAT, is one consistent with the scientific consensus on the way
biological brains work.
In this approach, an *output* of a network is loosely associated with a stable attractor (or lack of thereof) which the system falls into
as a response to a given input. In some cases, it is interesting to observe the dynamics not of a network as a whole,
but rather of a specific neuron or a group of neurons within the network, however, this requires many additional considerations
and is not yet implemented in CT-NEAT.
In other words, in this approach, the output of a network is not its state at a specific time, but rather a meta-analysis
of the network's behavior over a period of time.

The :py:mod:`disretizer` module provides such functionality to convert a :term:`continuous-time` :term:`recurrent` neural 
network's (CTRNN) dynamics into a discrete output.

The discretizer class works as follows:

1. When initialized, it takes any CTRNN network, a set of input values on which the network will be evaluated,
   a set of expected output values corresponding to the inputs, and parameters controlling the discretization process
   and any other controllable part of the network's dynamics (e.g., simulation time, time step, etc.).
2. For each input value, the network is simulated over a specified period of time, and its dynamics are recorded.
3. The recorded dynamics are then analyzed to identify stable attractors or patterns in the network's behavior. This
   step uses the functions defined in the :py:mod:`iznn.dynamic_attractors` module, which returns a vector encoding 
   the attractor (if one is found).
4. Over the space of the identified attractors, a clustering algorithm is applied to group similar attractors together.
   Either fixed number of clusters can be specified, or the algorithm can determine the optimal number of clusters
   based on the data.
5. Once each input has been associated with a cluster, the Hungarian (or Munkres) algorithm is used to optimally match 
   the identified clusters with the expected output values, minimizing the overall discrepancy between the network's 
   outputs and the expected outputs.
6. The final output of the discretizer is a mapping from each input to its corresponding discrete output, based on the
   identified attractors and the optimal matching process.

Here are the docstrings for the module:

.. automodule:: ctneat.discretizer
   :members:
   :undoc-members:
   :special-members: __init__