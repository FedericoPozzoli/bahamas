from setuptools import setup, find_packages

setup(
    name='bahamas',
    version='0.1.0',
    description='BAyesian HAmiltonian Montecarlo Analysis for Stochastic gravitational wave signal',
    author='Federico Pozzoli',
    author_email='fpozzoli@uninsubria.it',
    packages=find_packages(include=['bahamas', 'bahamas.psd_strain', 'bahamas.psd_response', 'bahamas.method']),
    #package_data={
    #    'bahamas': ['data/*.h5'],
    #},
    install_requires=[
        'numpy',
        'jax',
        'jaxlib',
        'numpyro',
        'scipy',
        'matplotlib',
        'h5py',
        'pyyaml',
        'pandas',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
