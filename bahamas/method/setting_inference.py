"""
This module provides utilities for setting up Bayesian inference, including reading data, 
selecting likelihood methods, and initializing matrices for EGP modeling.

Functions:
    read_data(config): Reads HDF5 data and applies frequency masking if specified.
    set_inference(config, sources): Sets up the inference process, including likelihood and EGP matrix initialization.

Classes:
    InferenceMethod: Handles the selection of likelihood methods and related configurations.
    EGPMatrix: Initializes the EGP matrix for Gaussian Process modeling.

Dependencies:
    - egp (custom module)
    - method (custom module)
    - NumPy
    - h5py
"""
from bahamas.method import setting_hmc
from bahamas.method import setting_nessai 
from bahamas.logger_config import logger

import numpy as np
import h5py
import os

def get_first_part(path: str) -> str:
    """
    Extract the first part of a file path.

    Parameters:
    - path (str): The file path.

    Returns:
    - str: The first part of the file path.
    """
    return os.path.dirname(path)

def read_data(config):
    """
    Reads HDF5 data from the specified file and applies frequency masking if specified.

    Args:
        config (dict): Configuration containing the file path and optional frequency masking parameters ('f1', 'f2').

    Returns:
        tuple: A tuple containing:
            - data_list (list): List of data arrays for each segment.
            - response_list (list): List of response arrays for each segment.
            - freq_list (list): List of frequency arrays for each segment.
            - count (list): List of count arrays for each segment.
    """
    file_path = config['inference']['file'] + '.h5'
    gen2 = config.get('gen2')
    f1 = config.get('f1', None)
    f2 = config.get('f2', None)

    data_list = []
    response_list = []
    freq_list = []
    count = []

    with h5py.File(file_path, 'r') as f:
        nchunk = len(f)
        logger.info(f'Number of chunks: {nchunk}')
        for i in range(nchunk):
            chunk_name = f'chunk_{i+1}'
            group = f[chunk_name]
            data_chunk = np.array(group['data'])
            response_chunk = np.array(group['response'])
            freq_chunk = np.array(group['freq'])
            try:
                count_chunk = np.array(group['count'])
            except KeyError:
                count_chunk = np.ones_like(freq_chunk)

            # Apply frequency mask if f1 and f2 are provided
            if f1 is not None and f2 is not None:
                mask = (freq_chunk > f1) & (freq_chunk < f2)
                data_chunk = data_chunk[:, mask]
                response_chunk = response_chunk[:, mask]
                freq_chunk = freq_chunk[mask]
                count_chunk = count_chunk[mask]

            data_list.append(data_chunk)
            response_list.append(response_chunk)
            freq_list.append(freq_chunk)
            count.append(count_chunk)

    if any(key in config for key in ['gaps', 'chunk']):
            folder = get_first_part(config['inference']['file'])
            try:
                time_data = np.loadtxt(f'{folder}/time_interval.txt')
            except:
                time_data = np.loadtxt(f'../data/time_interval.txt')
 
            t1, t2 = time_data[0], time_data[1]
    else:
            t1, t2 = np.array([0]), np.array([0])

    return data_list, response_list, freq_list, count, gen2, t1, t2


class InferenceMethod:
    """
    Handles the selection of likelihood methods and related configurations.
    """

    def __init__(self, config, t1, t2, count):
        """
        Initialize the InferenceMethod class.

        Args:
            config (dict): Configuration dictionary.
            t1 (array): Start times for each segment.
            t2 (array): End times for each segment.
            count (array): Count of data points for each segment.
        """
        self.config = config
        self.dt = config['dt']
        self.T = config['T']
        self.t1 = t1
        self.t2 = t2
        self.count = count

        if 'chunk' in config:
            self.TOBS = config['chunk']['duration']

        elif 'gaps' in config:
            self.TOBS = config['gaps']['sched_period']
            
        else:
            self.TOBS = config['T']
        
        self.N = self.T // self.dt


    def select_method(self):
        """
        Select the likelihood method and configure it based on the sampler and mode.

        Returns:
            tuple: A tuple containing:
                - log_like (callable): The selected likelihood function.
                - t1 (array): Start times for each segment.
                - t2 (array): End times for each segment.
                - dof (array): Degrees of freedom for each segment.
        """
        mode = self.config['inference']['likelihood']
        sampler = self.config['inference']['sampler']

        if mode == 'Whittle':
            if sampler in ['NUTS']:
                log_like = setting_hmc.whittle_lik
                if 'beta' in self.config['inference']:
                    log_like = setting_hmc.beta_scaled_log_likelihood(log_like, beta=self.config['inference']['beta'])
            elif sampler in ['nested']:
                log_like = setting_nessai.whittle_lik
            dof = self._calculate_dof()

        elif mode == 'Gamma':
            if sampler in ['NUTS']:
                log_like = setting_hmc.gamma_lik
                if 'beta' in self.config['inference']:
                    log_like = setting_hmc.beta_scaled_log_likelihood(log_like, beta=self.config['inference']['beta'])
            elif sampler in ['nested']:
                log_like = setting_nessai.gamma_lik
            dof = self._calculate_dof(gamma=True)

        else:
            raise ValueError(f"Unknown Likelihood: {mode}")

        return log_like, dof


    def _calculate_dof(self, gamma=False):
        """
        Calculate the degrees of freedom for each segment.

        Args:
            gamma (bool): Whether to calculate DOF for the Gamma likelihood.

        Returns:
            tuple: A tuple containing:
                - t1 (array): Start times for each segment.
                - t2 (array): End times for each segment.
                - dof (array): Degrees of freedom for each segment.
        """
        if gamma:
            dof = np.array(self.count) 
        else:
            if any(key in self.config for key in ['gaps', 'chunk']):
                dof = (self.t2 - self.t1) / self.dt // 2
            else:
                dof = np.array([self.T / self.dt // 2])
            
        return  dof


def set(config, sources):
    """
    Set up the inference process, including likelihood and EGP matrix initialization.

    Args:
        config (dict): Configuration dictionary.
        sources (dict): Source parameters.

    Returns:
        tuple: A tuple containing:
            - log_like (callable): The selected likelihood function.
            - data (list): Observed data segments.
            - freqs (list): Frequency grids for each segment.
            - response (list): Response functions for each segment.
            - dt (float): Time step.
            - t1 (array): Start times for each segment.
            - t2 (array): End times for each segment.
            - dof (array): Degrees of freedom for each segment.
            - matrix_egp (np.ndarray or None): The initialized EGP matrix.
    """
    # Load Data
    data, response, freqs, count, gen2, t1, t2 = read_data(config=config)

    # Setup Inference Method
    infer = InferenceMethod(config, t1, t2, count)
    log_like, dof = infer.select_method()

    return log_like, data, freqs, response, config['dt'], t1, t2, dof, gen2