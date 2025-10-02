
Installation
============

About The Examples
------------------

Because `ct-neat-python` is still changing fairly rapidly, attempting to run examples with a significantly newer or older
version of the library will result in errors.  It is best to obtain matching example/library code by using one of the
two methods outlined below:

Install ct-neat-python from PyPI using pip
------------------------------------------
To install the most recent release from PyPI, you should run the command (as root or using `sudo`
as necessary)::

    pip install ct-neat-python

Note that the examples are not included with the package installed from PyPI, so you should download the source archive and use the example code contained in it.

You may also just get the latest release source, and install it directly (as shown below)
instead of `pip`.

Install ct-neat-python from source using `pyproject.toml`
---------------------------------------------------------
Obtain the source code by either cloning the source repository::

    git clone https://github.com/Horef/ct-neat-python

Note that the most current code in the repository may not always be in the most polished state, but I do make sure the
tests pass and that most of the examples run.  If you encounter any problems, please open an `issue on GitHub
<https://github.com/Horef/ct-neat-python/issues>`_.

To install from source, simply run::

    pip install -e .

from the directory containing `pyproject.toml`.