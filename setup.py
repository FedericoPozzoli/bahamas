from setuptools import setup, find_packages

setup(
    name='bahamas',
    version='0.1.2',
    description='BAyesian HAmiltonian Montecarlo Analysis for Stochastic gravitational wave signal',
    author='Federico Pozzoli',
    author_email='fpozzoli@uninsubria.it',
    packages=find_packages(include=['bahamas', 'bahamas.psd_strain', 'bahamas.psd_response', 'bahamas.method', 'bahamas.utilities']),
    install_requires=[
        'numpy',
        'jax',
        'jaxlib',
        'numpyro',
        'scipy',
        'matplotlib',
        'h5py',
        'pyyaml',
        'nessai',
        'healpy'
    ],
    url='https://github.com/FedericoPozzoli/bahamas.git',
    license='Apache License 2.0',
    license_files=['LICENSE'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    entry_points={
    'console_scripts': [
        'bahamas_inference=bahamas.utilities.run_pe:main',
        'bahamas_data=bahamas.utilities.run_data:main',
        'bahamas_input=bahamas.utilities.run_input:main',
    ],
},
)
