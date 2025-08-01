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
from bahamas.method import gaps
from bahamas.logger_config import logger

import matplotlib.pylab as plt
import numpy as np
import yaml
import argparse

import h5py
import os
import sys

# Configure logger

def get_first_part(path: str) -> str:
    """
    Extract the first part of a file path.

    Parameters:
    - path (str): The file path.

    Returns:
    - str: The first part of the file path.
    """
    return os.path.dirname(path)

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
        self.config = config
        self.T = self.config['T']
        self.dt = self.config['dt']
        self.nseg = self.config['nseg']
        self.file = self.config['file']
        self.fileAV = self.config['fileAV']
        self.sources = sources
        
        self.freq_tot = []
        self.N_tot = []
        self.T1, self.T2 = [], []

        if 'response_num' in self.config:
            self.custom_response = config['response_num']
            
            if 'gen2' in config:
                self.gen2 = config['gen2']
            else:
                self.gen2 = False

            if 'cross_term' in config:
                self.cross_term = config['cross_term']
            else:
                self.cross_term = False

            if 'equal_arm' in config:
                self.equal_arm = config['equal_arm']
            else:
                self.equal_arm = 8.33

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
        if 'gaps' in self.config and 'chunk' in self.config:
            logger.error('Use chunked or gapped data')
            sys.exit()
        
        if 'galactic_DWD_time' in self.sources:
            if 'chunk' not in self.config:
                logger.error('Use chunked or gapped data with cyclo galactic sources')
                sys.exit()

        #if 'bulge_time' in self.sources:
        #    if 'chunk' not in self.config:
        #        logger.error('Use chunked or gapped data with cyclo bulge disk sources')
        #        sys.exit()

        #if 'disk_time' in self.sources:
        #    if 'chunk' not in self.config:
        #        logger.error('Use chunked or gapped data with cyclo bulge disk sources')
        #        sys.exit()

        if 'gaps' in self.config:
            logger.info('Production of gapped data')
            tg = self.config['gaps']
            self.start, self.end = gaps.generate_gaps(
                self.T, scheduled_gap=tg['sched_gap'], scheduled_period=tg['sched_period'],
                unscheduled_gap=tg['unsched_gap'], exp_scale=tg['exp_scale'],
                merge_threshold=tg['thresh'], duty_cycle=tg['duty_cycle']
            )
            for i in range(len(self.start)):
                df = 1 / (self.end[i] - self.start[i])
                N = (self.end[i] - self.start) // self.dt
          
                freqs = np.arange(0, 1 / (2 * self.dt), df)
                self.freq_tot.append(freqs)
                self.N_tot.append(N)
                self.T1.append(self.start[i])
                self.T2.append(self.end[i])
                logger.info(f"Chunk {i+1}: start={self.start[i]}, end={self.end[i]}, T ={self.T2[i] - self.T1[i]}")
          
            self.TOBS = self.config['gaps']['sched_period']

        elif 'chunk' in self.config:
            logger.info('Production of chunked data')
            tch = self.config['chunk']['duration']    
            self.TOBS = tch

            t0 = 0
            if 't0' in self.config:
                t0 = self.config['t0']

            nch = self.T // tch
            N = tch // self.dt

            # Create frequency grid
            df = 1 / tch
            freqs = np.arange(0, 1 / (2*self.dt), df)
            self.freq_tot = [freqs.copy() for _ in range(nch)]
            self.N_tot = [N for _ in range(nch)]    

            # Create time intervals
            tt = np.linspace(t0, t0 + self.T, nch + 1)
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
        if self.custom_response == False:
            # If custom response is not used, return None
            return [np.zeros_like(freqs) for _ in range(self.ntdi)]
        
        else:
            return resp.get_response(freqs, gen2 = self.gen2, tdi = self.tdi, equal_arm = self.equal_arm, cross_term = self.cross_term)


    def simulate_data(self):
        """
        Simulate data based on the configuration and sources.
        """
        f_av, response, data, responseAV, dataAV, count = [], [], [], [], [], []
        for i, freqs in enumerate(self.freq_tot):
            
            response_tdi = self.compute_response(freqs)
    
            data_chunk_tdi, data_chunk_tdi_av, response_tdi_av = [], [], []
            for ind_tdi in range(self.ntdi):
                psd_tdi, _ = psd.model_psd(freqs, sources=self.sources, response=response_tdi[ind_tdi], injected=True, t1=self.T1[i], t2=self.T2[i], tdi=ind_tdi, gen2 = self.gen2)
                _, _tdi = GP_freq(freqs, self.dt, psd=psd_tdi)
                data_chunk_tdi.append(_tdi)

                if self.config['mod'] == 'lin':
                    dof = self.N_tot[i] // self.nseg
                    f, d_av_tdi, R_av_tdi = average_chunks(freqs, 2*(self.dt**2 / self.TOBS)*np.abs(_tdi)**2, response_tdi[ind_tdi], dof)      
                    data_chunk_tdi_av.append(d_av_tdi)
                    response_tdi_av.append(R_av_tdi)
                    if ind_tdi == 0:                  
                        count.append(dof*np.ones_like(f))

                elif self.config['mod'] == 'log':
                    f, d_av_tdi, R_av_tdi, c = average_log_chunks(freqs, 2*(self.dt**2 / self.TOBS)*np.abs(_tdi)**2, response_tdi[ind_tdi], self.config['nseg'])
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

        # Save time intervals if chunks or gaps are present
        if any(key in self.config for key in ['chunk', 'gaps']):
            logger.info("Saving start and end of chunk.")
            name_folder = get_first_part(self.config['fileAV'])
            np.savetxt(f'{name_folder}/time_interval.txt', [self.T1, self.T2])

    def plot_psd(self):
        """
        Plot the Power Spectral Density (PSD) of the generated signal.
        """
        color = get_colormap_colors(n=len(self.T1)+1)
        logger.info("Plotting signals.")

        plt.figure(figsize=(8, 6))
        self.freq_output = np.arange(1e-5, 0.029, 1 / self.T)
        self.df = 1 / self.T


        if all(k not in self.sources for k in ['galactic_DWD_time']):
            self.RAA_output, self.REE_output = self.compute_response(self.freq_output)
            psd_totA, self.psdA_output = psd.model_psd(self.freq_output, sources=self.sources, response=self.RAA_output, injected=True, gen2 = self.gen2)
            psd_totE, self.psdE_output = psd.model_psd(self.freq_output, sources=self.sources, response=self.REE_output, injected=True, gen2 = self.gen2)
            for i, name in enumerate(self.sources.keys()):
                plt.loglog(self.freq_output, self.psdA_output[i], label=name)


        for i in range(len(self.T1)):
            plt.loglog(self.f_av[i], self.dataAV[i][0], alpha=0.3, color=color[i], linestyle='--', label=f'chunk {i+1}')
        plt.xlabel("Frequency")
        plt.xlim(1e-4, 0.029)
        plt.ylim(10**-48, 10**-38.5)
        plt.ylabel("Power Spectral Density")
        plt.title("PSD of the Generated Signal")
        plt.legend()
        name_ = get_last_part(self.config['fileAV'])
        name_folder = self.config['folder_plot']
        plt.savefig(f'{name_folder}data_{name_}.png', bbox_inches='tight')

        if 'chunk' in self.config:
            plt.figure(figsize=(10, 4))
            for j in range(len(self.T1)):
                ft = np.insert(self.data[j][0], 0, 0)
                ift = np.fft.irfft(ft)
                plt.plot(np.linspace(self.T1[j], self.T2[j], len(ift))[:-1], ift[:-1], rasterized=True, color='teal', alpha=0.3)
            plt.savefig(f'{name_folder}mod_gal.png', bbox_inches='tight')

        if 'gaps' in self.config:
            plt.figure(figsize=(10, 4))
            for j in range(len(self.T1)):
                ft = np.insert(self.data[j][0], 0, 0)
                ift = np.fft.irfft(ft)
                logger.info(f'Chunk {j+1} start: {self.T1[j]}, end: {self.T2[j]}')
                plt.plot(np.linspace(self.T1[j], self.T2[j], len(ift))[:-1], ift[:-1], rasterized=True, color='teal', alpha=0.3)
            plt.savefig(f'{name_folder}mod_gal.png', bbox_inches='tight')

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
        if all(k not in self.sources for k in ['galactic_DWD_time', 'disk_time', 'bulge_time']):
            for i, name in enumerate(self.sources.keys()):
                if name == 'instr_noise':
                    SnA, SnE = self.psdA_output[i], self.psdE_output[i]

            for i, name in enumerate(self.sources.keys()):
                if name != 'instr_noise':
                    SNRA = self.SNR2(Sh=self.psdA_output[i], Sn=SnA)
                    SNRE = self.SNR2(Sh=self.psdE_output[i], Sn=SnE)
                    snr = np.sqrt(SNRA + SNRE)
                    logger.info(f'SNR of {name}: {round(snr)}')


##################################################################################
# DATA GENERATION AND PROCESSING
##################################################################################

def GP_freq(freqs, dt, psd, time=False):
    """
    Generates a stationary Gaussian process using the inverse FFT method.

    Args:
        freqs (array): Frequency array.
        dt (float): Time step.
        psd (array): Power spectral density.
        seed (int): Random seed.
        time (bool or str): If True, returns time-domain signal. If 'both', returns both time and frequency domain.

    Returns:
        tuple: Frequency and Fourier coefficients, or time and signal, or both.
    """
    amp_r = np.random.normal(loc=np.zeros_like(freqs), scale=np.sqrt(psd * (len(freqs) / dt)))
    amp_i = np.random.normal(loc=np.zeros_like(freqs), scale=np.sqrt(psd * (len(freqs) / dt)))
    fourier_coeffs = (amp_r + 1j * amp_i) / np.sqrt(2)
    fourier_coeffs[0] = 0

    if time == True:
        x = np.fft.irfft(fourier_coeffs, n=len(freqs) * 2)
        t = np.arange(len(freqs) * 2) * dt
        return t, x
    elif time == 'both':
        x = np.fft.irfft(fourier_coeffs, n=len(freqs) * 2)
        t = np.arange(len(freqs) * 2) * dt
        return freqs, fourier_coeffs, t, x
    else:
        return freqs, fourier_coeffs
    

def average_chunks(freqs, data, response, chunk_size):
    """
    Averages data in chunks.

    Args:
        freqs (array): Frequency array.
        data (array): Data array.
        response (array): Response array.
        chunk_size (int): Size of each chunk.

    Returns:
        tuple: Averaged frequency, data, and response arrays.
    """
    data_chunks = np.array_split(data, len(data) // chunk_size)
    response_chunks = np.array_split(response, len(data) // chunk_size)
    freq_chunks = np.array_split(freqs, len(data) // chunk_size)

    d = [np.mean(chunk) for chunk in data_chunks]
    r = [np.mean(chunk) for chunk in response_chunks]
    f = [np.mean(chunk) for chunk in freq_chunks]

    return np.array(f), np.array(d), np.array(r)


def average_log_chunks(freqs, data, response, num_bins=50):
    """
    Averages data in logarithmic bins.

    Args:
        freqs (array): Frequency array.
        data (array): Data array.
        response (array): Response array.
        num_bins (int): Number of bins.

    Returns:
        tuple: Averaged frequency, data, response, and count arrays.
    """
    freqs = np.asarray(freqs)
    data = np.asarray(data)
    response = np.asarray(response)

    log_min = np.log10(np.min(freqs[freqs > 0]))
    log_max = np.log10(np.max(freqs))
    log_bins = np.logspace(log_min, log_max, num_bins + 1)

    f_avg, d_avg, r_avg, count = [], [], [], []

    for i in range(num_bins):
        mask = (freqs >= log_bins[i]) & (freqs < log_bins[i + 1])
        if np.any(mask):
            f_avg.append(0.5 * (freqs[mask][0] + freqs[mask][-1]))
            d_avg.append(np.mean(data[mask]))
            r_avg.append(np.mean(response[mask]))
            count.append(np.sum(mask))

    return np.array(f_avg), np.array(d_avg), np.array(r_avg), np.array(count)
