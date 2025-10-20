What's New
==========

What's new in ct-neat-python 1.1.1 (October 2025)
-------------------------------------------------
- Added an option to the Discretizer class to return the determinism value of the network's dynamics
  when no attractor is found for a given input. This allows users to gain insights into the network's behavior
  even when it does not settle into a stable state.

What's new in ct-neat-python 1.1.0 (October 2025)
-------------------------------------------------
- Added a completely new module for discretization of continuous network dynamics. This module provides a Discretizer class which can be used to convert
the continuous dynamics of a CTRNN into discrete states based on the attractors found in the network's behavior.
- Small improvements to many of the modules, docstrings and documentation.

What's new in ct-neat-python 1.0.1 (October 2025)
-------------------------------------------------
- Added an option for the dynamic attractors function to return the fingerprint in a form of a vector of floats,
instead of a string. This is needed for the Discretizer class which is currently in development.
- Started work on a Discretizer class which will be used to convert the continuous dynamics of CTRNNs into discrete states.
