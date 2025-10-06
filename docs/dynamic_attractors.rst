Dynamical Attractor Determination and Description
====================================================

One of the most important aspects of understanding the behavior of :term:`continuous-time` :term:`recurrent` 
neural networks (CTRNN) is the identification and characterization of their dynamical attractors (or lack of thereof).

An attractor of a dynamical system is a set of numerical values toward which the system tends to converge over time for a 
given set of initial conditions. Attractors can take various forms, including fixed points, limit cycles, and strange attractors.
Where fixed points are single states that the system settles into, limit cycles are periodic orbits that the system repeatedly follows,
and strange attractors exhibit complex, non-repeating behavior that is sensitive to initial conditions (yet the overall structure remains invariant).

In general, finding an attractor of a dynamical system is a NP-Hard problem, and there is no single method that can be used to find
all attractors in all systems. Even for a specific system with a specific set of initial conditions this is impossible in general
as it can never be known if all possible states have been explored (and we cannot run the system indefinitely).
However, there are a number of heuristic methods which can be used to find attractors (or at least an attractor like behavior)
in certain scenarios and under certain limiting assumptions (like the maximal simulation time and the length of the attractor period).

:py:mod:`ctneat.iznn.dynamic_attractors` module provides tools for identifying and characterizing dynamical attractors in 
Continuous-Time Recurrent Neural Networks (CTRNNs). It includes functions for simulating network dynamics, 
resampling data, and clustering attractor states.

The central function in the module is :py:func:`dynamic_attractors_pipeline`, which is the full pipeline for finding
and characterizing attractors in a CTRNN. It proceeds as follows:

1. It receives the recordings of the network's voltages and firing states over time, along with parameters controlling
   the analysis process (like burn-in time, sampling rate, etc.).

2. It optionally discards an initial "burn-in" period of the simulation to allow the network to stabilize.

3. It uses an RQA (Recurrence Quantification Analysis) based method to identify potential attractor periods in the network's dynamics.
   This is done by analyzing the recurrence plot of the network's firing states to find diagonal lines that indicate periodic behavior.
   This method produces a determinizm score that indicates how deterministic (as opposed to chaotic or random) the dynamics are.

4. If significant determinism is found, the function proceeds to try to find and fingerprint the attractor.

   1. Depending on the parameters passed to the function, it can use either the full dynamics of all neurons in the network,
      or perform PCA (Principal Component Analysis) to reduce the dimensionality to a single dimension (the first principal component).
      Or simply superimpose the dynamics of all neurons into a single vector.
   2. Fast Fourier Transform (FFT) is applied to the selected dynamics to identify dominant frequencies in the network's behavior.
      And that frequency is used to determine the period of a supposed limit cycle attractor.
   3. If a plausible period is found, the function extracts the segment of the dynamics corresponding to one full cycle of the attractor
      and runs a fingerprinting analysis on it.
   4. The fingerprinting analysis can be done either based on the voltages of the neurons, or based on their firing states.
      The fingerprinting process produces a vector that encodes the attractor's characteristics.

5. If no significant determinism is found, or if no plausible attractor period is identified,
   unless the variable burn-in is used, the function returns None for the attractor fingerprint. 

6. If the variable burn-in is used, the function will iteratively increase the burn-in period and repeat the analysis
   until either an attractor is found or the maximum burn-in time is reached.

Here are the docstrings for the module:

.. automodule:: ctneat.iznn.dynamic_attractors
   :members:
   :undoc-members:
   :special-members: __init__