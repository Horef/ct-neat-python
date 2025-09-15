from setuptools import setup, find_packages

setup(
    name='ct-neat-python',
    version='1.0.0',
    author='Sergiy Horef',
    author_email='s@unzim.com',
    maintainer='Sergiy Horef',
    maintainer_email='s@unzim.com',
    url='https://github.com/Horef/ct-neat-python',
    license="BSD",
    description='A CT (Continuous Time) NEAT (NeuroEvolution of Augmenting Topologies) implementation',
    long_description='Python implementation of NEAT (NeuroEvolution of Augmenting Topologies), a method ' +
                     'developed by Kenneth O. Stanley for evolving arbitrary neural networks.' +
                     'with Continuous Time dynamics.',
    long_description_content_type= 'text/x-rst',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'joblib'],
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering'
    ]
)
