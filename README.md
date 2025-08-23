<!-- [![Build Status](https://app.travis-ci.com/CodeReclaimers/neat-python.svg?branch=master)](https://app.travis-ci.com/github/CodeReclaimers/neat-python)
[![Coverage Status](https://coveralls.io/repos/CodeReclaimers/neat-python/badge.svg?branch=master&service=github)](https://coveralls.io/github/CodeReclaimers/neat-python?branch=master)
[![Downloads](https://static.pepy.tech/personalized-badge/neat-python?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/neat-python) -->

## About ##

NEAT (NeuroEvolution of Augmenting Topologies) is a method developed by Kenneth O. Stanley for evolving arbitrary neural
networks. Later NEAT was extended to evolve CT (Continuous Time) networks in different frameworks.
This project is a Python implementation of CT-NEAT (and has both pure NEAT and CT-NEAT implementations) with few dependencies beyond the standard library (like scikit-learn). It was
forked from the excellent project started by @MattKallada and continued by @CodeReclaimers (![neat-python](https://github.com/CodeReclaimers/neat-python?tab=readme-ov-file)) after their project was archived.

For further information regarding general concepts and theory, please see the 
[Selected Publications](http://www.cs.ucf.edu/~kstanley/#publications) on Stanley's page at the University of Central 
Florida (now somewhat dated), or the [publications page](https://www.kenstanley.net/papers) of his current website. (rtNEAT would be relevant for the CT element.)

`ct-neat-python` is licensed under the [3-clause BSD license](https://opensource.org/licenses/BSD-3-Clause).  It is
currently supported on Python 3.6 through 3.11, and pypy3.

## Getting Started ##

If you want to try ct-neat-python, please check out the repository, start playing with the examples (`examples/xor` is
a good place to start) and then try creating your own experiment.

The documentation is available on [Read The Docs](http://neat-python.readthedocs.io).

## Citing ##

Here are APA and Bibtex entries you can use to cite this project in a publication. The listed authors are the originators
and/or maintainers of all iterations of the project up to this point.  If you have contributed and would like your name added 
to the citation, please submit an issue or email s@unzim.com.

APA
```
Horef, S. McIntyre, A., Kallada, M., Miguel, C. G., Feher de Silva, C., & Netto, M. L. ct-neat-python [Computer software]
```

Bibtex
```
@software{Horef_ct-neat-python,
author = {Horef Sergiy, McIntyre, Alan and Kallada, Matt and Miguel, Cesar G. and Feher de Silva, Carolina and Netto, Marcio Lobo},
title = {{ct-neat-python}}
}
```

<!-- ## Thank you! ##
Many thanks to the folks who have [cited this repository](https://scholar.google.com/scholar?start=0&hl=en&as_sdt=5,34&sciodt=0,34&cites=15315010889003730796&scipsc=) in their own work.  -->
