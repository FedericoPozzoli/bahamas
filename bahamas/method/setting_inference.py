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
from bahamas.psd_strain import egp
from bahamas.method import setting_hmc
from bahamas.method import setting_nessai 

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
    f1 = config.get('f1', None)
    f2 = config.get('f2', None)

    data_list = []
    response_list = []
    freq_list = []
    count = []

    with h5py.File(file_path, 'r') as f:
        nchunk = len(f)
        print(f'Number of chunks: {nchunk}')
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

    return data_list, response_list, freq_list, count


class InferenceMethod:
    """
    Handles the selection of likelihood methods and related configurations.
    """

    def __init__(self, config):
        """
        Initialize the InferenceMethod class.

        Args:
            config (dict): Configuration dictionary.
        """
        self.config = config
        self.dt = config['dt']
        self.T = config['T']
        self.TOBS = config['chunk']['duration'] if 'chunk' in config else config['T']
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
            t1, t2, dof = self._calculate_dof()

        elif mode == 'Gamma':
            if sampler in ['NUTS']:
                log_like = setting_hmc.gamma_lik
                if 'beta' in self.config['inference']:
                    log_like = setting_hmc.beta_scaled_log_likelihood(log_like, beta=self.config['inference']['beta'])
            elif sampler in ['nested']:
                log_like = setting_nessai.gamma_lik
            t1, t2, dof = self._calculate_dof(gamma=True)

        else:
            raise ValueError(f"Unknown Likelihood: {mode}")

        return log_like, t1, t2, dof

    def get_number_logbin(self, nbin, f1, f2):
        """
        Calculate the number of log bins for frequency binning.

        Args:
            nbin (int): Number of bins.
            f1 (float): Lower frequency limit.
            f2 (float): Upper frequency limit.

        Returns:
            np.ndarray: Array of counts for each log bin.
        """
        freqs = np.arange(0, 0.5 / self.dt, 1 / self.TOBS)
        log_min = np.log10(np.min(freqs[freqs > 0]))
        log_max = np.log10(np.max(freqs))
        log_bins = np.logspace(log_min, log_max, nbin + 1)

        counts = []
        for i in range(nbin):
            mask = (freqs > log_bins[i]) & (freqs < log_bins[i + 1])
            if np.any(mask):
                ref = 0.5 * (freqs[mask][0] + freqs[mask][-1])
                if f1 < ref < f2:
                    count = np.sum(mask)
                    counts.append(count)

        return np.array(counts)

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
        if any(key in self.config for key in ['chunk']):
            folder = get_first_part(self.config['inference']['file'])
            time_data = np.loadtxt(f'{folder}/time_interval.txt')
            t1, t2 = time_data[0], time_data[1]
            N = (t2 - t1) // self.dt
            if self.config['mod'] == 'lin':
                dof = N // self.config['nseg'] if gamma else (t2 - t1) / self.dt // 2
            elif self.config['mod'] == 'log':
                dof = self.get_number_logbin(self.config['nseg'], self.config['f1'], self.config['f2']) if gamma else (t2 - t1) / self.dt // 2
        else:
            t1, t2 = np.array([0]), np.array([0])
            if self.config['mod'] == 'lin':
                dof = np.array([self.N // self.config['nseg']]) if gamma else np.array([self.T / self.dt // 2])
            elif self.config['mod'] == 'log':
                dof = self.get_number_logbin(self.config['nseg'], self.config['f1'], self.config['f2']) if gamma else np.array([self.T / self.dt // 2])

        return t1, t2, dof


class EGPMatrix:
    """
    Initializes the EGP matrix for Gaussian Process modeling.
    """

    @staticmethod
    def initialize_matrix(config, sources, freqs):
        """
        Initialize the EGP matrix based on the configuration and sources.

        Args:
            config (dict): Configuration dictionary.
            sources (dict): Source parameters.
            freqs (list): Frequency arrays for each segment.

        Returns:
            np.ndarray or None: The initialized EGP matrix, or None if not applicable.
        """
        if 'egp' not in sources:
            return None

        num = next((param['num'] for param in sources['egp'] if 'num' in param), None)
        nodes = np.logspace(np.log10(config['f1']), np.log10(config['f2']), num)
        cov = egp.GP_SGWB_kernel_SE(np.log10(nodes), np.log10(nodes), parameter=[6.3], eps=1.0e-6)
        inv_cov = np.linalg.inv(cov)
        k_star = egp.GP_SGWB_kernel_SE(np.log10(freqs[0]), np.log10(nodes), parameter=[6.3])

        return np.dot(k_star, inv_cov)


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
            - count (list): Counts for each segment.
            - dt (float): Time step.
            - t1 (array): Start times for each segment.
            - t2 (array): End times for each segment.
            - dof (array): Degrees of freedom for each segment.
            - matrix_egp (np.ndarray or None): The initialized EGP matrix.
    """
    # Load Data
    data, response, freqs, count = read_data(config=config)

    # Setup Inference Method
    infer = InferenceMethod(config)
    log_like, t1, t2, dof = infer.select_method()

    # Initialize EGP Matrix
    matrix_egp = EGPMatrix.initialize_matrix(config, sources, freqs)
    
    return log_like, data, freqs, response, count, config['dt'], t1, t2, dof, matrix_egp