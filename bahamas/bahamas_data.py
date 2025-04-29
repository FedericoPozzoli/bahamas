"""
BAHAMAS Data Generation Script
This script is part of the BAHAMAS project, which focuses on simulating gravitational wave signals.
It is designed to generate synthetic data for gravitational wave detection and analysis.
The script includes functionalities for signal processing, Power Spectral Density (PSD) analysis, and data simulation.
It uses the BAHAMAS library for signal processing and data generation.

Class: SignalProcessor
    This class handles the signal processing, PSD analysis, and data simulation.
    It includes methods for computing frequency grids, simulating data, saving data to HDF5 files,
    and plotting the Power Spectral Density (PSD) of the generated signal.

Usage:
    The script can be run from the command line with the following arguments:
    --config: Path to the YAML file containing mission properties.
    --sources: Path to the YAML file containing source parameters.
"""

from bahamas.psd_strain import psd_function as psd
from bahamas.psd_response import response as resp

import matplotlib.pylab as plt
import numpy as np
import yaml
import argparse
import logging
import h5py
import os
import sys

# Configure logger
logger = logging.getLogger('BAHAMAS')
logger.setLevel(logging.DEBUG)

# Add a console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def get_last_part(path: str) -> str:
    """
    Extract the last part of a file path.

    Parameters:
    - path (str): The file path.

    Returns:
    - str: The last part of the file path.
    """
    return os.path.basename(path)

def get_colormap_colors(n, cmap_name="viridis"):
    """
    Return an array of n colors sampled from the specified colormap.
    
    Parameters:
    - n (int): Number of colors to return.
    - cmap_name (str): Name of the colormap (default: 'viridis').
    
    Returns:
    - List of RGB tuples.
    """
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / (n - 1)) for i in range(n)]  # Normalize indices

def parse_arguments():
    """
    Parse command-line arguments for the script.
    
    Returns:
    - argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Signal Processing and PSD Analysis')
    parser.add_argument('--config', type=str, required=True, help='YAML file containing mission properties')
    parser.add_argument('--sources', type=str, required=True, help='YAML file containing source parameters')
    return parser.parse_args()

def load_yaml(file_path):
    """
    Load a YAML file and return its contents.
    
    Parameters:
    - file_path (str): Path to the YAML file.
    
    Returns:
    - dict: Parsed YAML content.
    """
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        raise

class SignalProcessor:
    """
    A class to handle signal processing, PSD analysis, and data simulation.
    """
    def __init__(self, config, sources):
        """
        Initialize the SignalProcessor with configuration and source data.
        
        Parameters:
        - config (dict): Configuration parameters.
        - sources (dict): Source parameters.
        """
        self.T = config['T']
        self.dt = config['dt']
        self.nseg = config['nseg']
        self.file = config['file']
        self.fileAV = config['fileAV']
        self.sources = sources
        self.freq_tot = []
        self.N_tot = []
        self.T1, self.T2 = [], []

        if 'response_num' in config:
            self.custom_response = config['response_num']
            
            if 'gen' in config:
                self.gen = config['gen']
            else:
                self.gen = 2

            if 'cross_term' in config:
                self.cross_term = config['cross_term']
            else:
                self.cross_term = False

            if 'equal_arm' in config:
                self.equal_arm = config['equal_arm']
            else:
                self.equal_arm = 8.3

            if 'tdi' in config:
                self.tdi = config['tdi']
                self.ntdi = len(self.tdi)
            else:
                self.tdi = 'AE'
                self.ntdi = 2
            
        else:
            self.custom_response = False

    def compute_frequency_grid(self):
        """
        Compute the frequency grid for the signal.
        """
        df = 1 / self.T
        freq = np.arange(0, 1 / (2 * self.dt), df)
        self.freq_tot = [freq]
        self.N_tot = [self.T // self.dt]

    def handle_series(self):
        """
        Handle the series based on configuration (chunks, or full resolution).
        """
        
        if 'galactic_DWD_time' in sources or 'galactic_prototype' in sources:
            if 'chunk' not in config:
                logger.error('Use chunked or gapped data with cyclo galactic sources')
                sys.exit()

        elif 'chunk' in config:
            logger.info('Production of chunked data')
            tch = config['chunk']['duration']    
            self.TOBS = tch

            nch = self.T // tch
            N = tch // self.dt
            df = 1 / tch
            freqs = np.arange(0, 1 / (2*self.dt), df)
            self.freq_tot = [freqs.copy() for _ in range(nch)]
            self.N_tot = [N for _ in range(nch)]    
            tt = np.linspace(0, self.T, nch + 1)
            self.T1 = tt[:-1]
            self.T2 = tt[1:]

        else:
            logger.info('Production of full resolution series')
            self.T1, self.T2 = np.array([None]), np.array([None])
            self.TOBS = self.T
            self.compute_frequency_grid()

    def compute_response(self, freqs):
        """
        Compute the response for the given frequencies.
        
        Parameters:
        - freqs (array): Frequency grid.
        
        Returns:
        - tuple: Response arrays.
        """
        
        return resp.get_response(freqs, gen = self.gen, tdi = self.tdi, equal_arm = self.equal_arm, cross_term = self.cross_term)


    def simulate_data(self):
        """
        Simulate data based on the configuration and sources.
        """
        f_av, response, data, responseAV, dataAV, count = [], [], [], [], [], []
        for i, freqs in enumerate(self.freq_tot):
            
            response_tdi = self.compute_response(freqs)
    
            data_chunk_tdi, data_chunk_tdi_av, response_tdi_av = [], [], []
            for ind_tdi in range(self.ntdi):
                psd_tdi, _ = psd.model_psd(freqs, sources=self.sources, response=response_tdi[ind_tdi], injected=True, t1=self.T1[i], t2=self.T2[i], tdi=i)
                _, _tdi = psd.GP_freq(freqs, self.dt, psd=psd_tdi, seed=np.random.randint(0, 100))
                data_chunk_tdi.append(_tdi)

                if config['mod'] == 'lin':
                    dof = self.N_tot[i] // self.nseg
                    f, d_av_tdi, R_av_tdi = psd.average_chunks(freqs, 2*(self.dt**2 / self.TOBS)*np.abs(_tdi)**2, response_tdi[ind_tdi], dof)      
                    data_chunk_tdi_av.append(d_av_tdi)
                    response_tdi_av.append(R_av_tdi)
                    if ind_tdi == 0:                  
                        count.append(dof*np.ones_like(f))

                elif config['mod'] == 'log':
                    f, d_av_tdi, R_av_tdi, c = psd.average_log_chunks(freqs, 2*(self.dt**2 / self.TOBS)*np.abs(_tdi)**2, response_tdi[ind_tdi], config['nseg'])
                    data_chunk_tdi_av.append(d_av_tdi)
                    response_tdi_av.append(R_av_tdi)
                    if ind_tdi == 0:
                        count.append(c)

            data.append(data_chunk_tdi)
            response.append(response_tdi)

            dataAV.append(data_chunk_tdi_av)
            responseAV.append(response_tdi_av)
            
            f_av.append(f)

            if f[0] > 1e-4:
                logger.warning('Attention! Minimum frequency > 1e-4 Hz')

        self.data, self.response = data, response
        self.dataAV, self.responseAV = dataAV, responseAV
        self.f_av = f_av
        self.count = count
        
    def save_data(self):
        """
        Save the simulated data and responses to HDF5 files.
        """
        # Save raw data
        with h5py.File(self.file+'.h5', 'w') as f:
            for i, (data_chunk, response_chunk, freq_chunk) in enumerate(zip(self.data, self.response, self.freq_tot)):
                group = f.create_group(f'chunk_{i+1}')
                group.create_dataset('data', data=np.array(data_chunk, dtype=np.complex128))
                group.create_dataset('response', data=np.array(response_chunk, dtype=np.float64))
                group.create_dataset('freq', data=np.array(freq_chunk, dtype=np.float64))

        # Save averaged data
        with h5py.File(self.fileAV+'.h5', 'w') as f:
            for i, (data_chunk, response_chunk, freq_chunk, count_chunk) in enumerate(zip(self.dataAV, self.responseAV, self.f_av, self.count)):
                group = f.create_group(f'chunk_{i+1}')
                group.create_dataset('data', data=np.array(data_chunk, dtype=np.float64))
                group.create_dataset('response', data=np.array(response_chunk, dtype=np.float64))
                group.create_dataset('freq', data=np.array(freq_chunk, dtype=np.float64))
                group.create_dataset('count', data=np.array(count_chunk, dtype=np.int32))

        logger.info(f"Data saved in {self.file}.h5 and {self.fileAV}.h5")

        # Save time intervals if chunks are present
        if any(key in config for key in ['chunk']):
            logger.info("Saving start and end of chunk.")
            np.savetxt('../data/time_interval.txt', [self.T1, self.T2])

    def plot_psd(self):
        """
        Plot the Power Spectral Density (PSD) of the generated signal.
        """
        color = get_colormap_colors(n=len(self.T1)+1)
        logger.info("Plotting signals.")

        plt.figure(figsize=(8, 6))
        self.freq_output = np.arange(1e-5, 0.029, 1 / self.T)
        self.df = 1 / self.T

        if 'galactic_DWD_time' not in sources and 'galactic_prototype' not in sources:   
            self.RAA_output, self.REE_output = self.compute_response(self.freq_output)
            psd_totA, self.psdA_output = psd.model_psd(self.freq_output, sources=self.sources, response=self.RAA_output, injected=True)
            psd_totE, self.psdE_output = psd.model_psd(self.freq_output, sources=self.sources, response=self.REE_output, injected=True)
            for i, name in enumerate(sources.keys()):
                plt.loglog(self.freq_output, self.psdA_output[i], label=name)

        for i in range(len(self.T1)):
            plt.loglog(self.f_av[i], self.dataAV[i][0], alpha=0.3, color=color[i], linestyle='--', label=f'chunk {i+1}')
        plt.xlabel("Frequency")
        plt.xlim(1e-4, 0.029)
        plt.ylim(10**-45, 10**-38)
        plt.ylabel("Power Spectral Density")
        plt.title("PSD of the Generated Signal")
        plt.legend()
        name_ = get_last_part(config['fileAV'])
        name_folder = config['folder_plot']
        plt.savefig(f'{name_folder}data_{name_}.png', bbox_inches='tight')

        if 'chunk' in config:
            plt.figure(figsize=(10, 4))
            for j in range(len(self.T1)):
                ft = np.insert(self.data[j][0], 0, 0)
                ift = np.fft.irfft(ft)
                plt.plot(np.linspace(self.T1[j], self.T2[j], len(ift))[:-1], ift[:-1], rasterized=True, color='teal', alpha=0.3)
            plt.savefig('../data/mod_gal.png', bbox_inches='tight')


    def SNR2(self, Sh, Sn):
        """
        Compute the squared Signal-to-Noise Ratio (SNR).
        
        Parameters:
        - Sh (array): Signal PSD.
        - Sn (array): Noise PSD.
        
        Returns:
        - float: Squared SNR.
        """
        return 4 * self.T * self.df * np.sum((Sh / Sn)**2)

    def compute_SNR(self):
        """
        Compute and log the Signal-to-Noise Ratio (SNR) for each source.
        """
        if 'galactic_DWD_time' not in sources and 'galactic_prototype' not in sources:
            for i, name in enumerate(sources.keys()):
                if name == 'instr_noise':
                    SnA, SnE = self.psdA_output[i], self.psdE_output[i]

            for i, name in enumerate(sources.keys()):
                if name != 'instr_noise':
                    SNRA = self.SNR2(Sh=self.psdA_output[i], Sn=SnA)
                    SNRE = self.SNR2(Sh=self.psdE_output[i], Sn=SnE)
                    snr = np.sqrt(SNRA + SNRE)
                    logger.info(f'SNR of {name}: {round(snr)}')

if __name__ == '__main__':
    """
    Main function to execute the script.
    It parses command-line arguments, loads configuration files, and processes signals.
    """
    # Parse arguments and load configuration
    args = parse_arguments()
    config = load_yaml(args.config)
    sources = load_yaml(args.sources)['sources']

    # Initialize and process signals
    processor = SignalProcessor(config, sources)
    processor.handle_series()
    processor.simulate_data()
    processor.save_data()
    processor.plot_psd()
    processor.compute_SNR()
    logger.info("Processing completed.")
