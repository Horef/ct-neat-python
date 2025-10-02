Welcome to CT-NEAT-Python's documentation!
==========================================

:abbr:`NEAT (NeuroEvolution of Augmenting Topologies)` is a method developed by Kenneth O. Stanley for evolving arbitrary neural
networks. 
:abbr:`CT-NEAT (Continuous-Time NEAT)` is an extension of NEAT for evolving continuous-time recurrent neural networks (CTRNNs),
also developed by Stanley.
CT-NEAT-Python is a Python implementation of CT-NEAT, with minimal dependencies beyond the standard library.

Currently this library supports Python versions 3.8 through 3.13, as well as PyPy 3.

Many thanks to the authors of the original implementation: Cesar Gomes Miguel, Carolina Feher da Silva, and Marcio Lobo Netto.
And the later work and development by CodeReclaimers (on GitHub) - Alan and Kallada McIntyre, Matt and Miguel.

.. note::
  Some of the example code has additional dependencies. For your convenience there is a conda environment YAML file in the
  examples directory you can use to set up an environment that will support all of the current examples.
  TODO: Improve README.md file information for the examples.

For further information regarding general concepts and theory, please see `Selected Publications
<http://www.cs.ucf.edu/~kstanley/#publications>`_ on Stanley's website, or his recent `AMA on Reddit
<https://www.reddit.com/r/IAmA/comments/3xqcrk/im_ken_stanley_artificial_intelligence_professor>`_.

If you encounter any confusing or incorrect information in this documentation, please open an issue in the `GitHub project
<https://github.com/Horef/ct-neat-python>`_.

.. _toc-label:

Contents:

.. toctree::
   :maxdepth: 2

   neat_overview
   installation
   whats_new
   config_file
   xor_example
   customization
   activation
   ctrnn
   genome-interface
   reproduction-interface
   docstrings
   glossary

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

