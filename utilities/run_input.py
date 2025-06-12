"""
This script processes input data from LISA data challenge files,
converts XYZ data to AET format, chunks the data into segments,
and saves the processed data to HDF5 files.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import math
import matplotlib

import matplotlib.colors as mcolors
import matplotlib
import numpy as np
import jax
import numpyro

from bahamas.bahamas_data import average_log_chunks, get_colormap_colors

import logging

# Enable 64-bit precision in NumPyro
numpyro.enable_x64()

# Set up logging
logger = logging.getLogger('BAHAMAS_input')
logger.setLevel(logging.DEBUG)

# Add a console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Change to DEBUG to see debug messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def AET(X,Y,Z):
    A = (Z - X)/math.sqrt(2.0)
    E = (X - 2.0*Y + Z)/math.sqrt(6.0)
    T = (X + Y + Z)/math.sqrt(3.0)
    return A, E, T

def XYZ2AET(file, Tobs):
    """
    Convert XYZ data to AET format from an HDF5 file.
    Args:
        file (str): Path to the HDF5 file containing XYZ data.
        Tobs (float): Observation time.
    Returns:
        tuple: A, E, T arrays and time array.
    """
    if not file.endswith('.h5'):
        logger.error(f'The file {file} is not an HDF5 file.')

    with h5py.File(file, 'r') as f:
    # List all groups in the file
        print("Keys in the file:", list(f.keys()))
        
        #get confnoise

        noise = f['instnoise']
        t = noise['tdi']['t'][:]
        #print("Keys in the instru group:", list(noise.keys()))
        X_inst = noise['tdi']['X'][:]
        Y_inst = noise['tdi']['Y'][:]
        Z_inst = noise['tdi']['Z'][:]

        confnoise = f['confnoise']
        X_conf = confnoise['tdi']['X'][:]
        Y_conf = confnoise['tdi']['Y'][:]
        Z_conf = confnoise['tdi']['Z'][:]

    X, Y, Z = X_inst + X_conf, Y_inst + Y_conf, Z_inst + Z_conf

    dt = t[1] - t[0]
    Tdata = t[-1]

    if Tobs > Tdata:
        logger.warning(f'Tobs is larger than the data length {Tdata}, setting Tobs to {Tdata}')
        Tobs = Tdata
    
    ind = Tobs / dt
    A, E, T = AET(X[:int(ind)], Y[:int(ind)], Z[:int(ind)])

    t = t[:int(ind)]

    
    return A, E, T, t
    
def chunk_data(chunk_T, n_chunks, nseg, A, E, t, path):
    """
    Chunk the data into segments of specified length and save to HDF5 files.
    Args:
        chunk_T (float): Length of each chunk in seconds.
        n_chunks (int): Number of chunks to create.
        nseg (int): Number of segments for averaging.
        A (np.ndarray): A data array.
        E (np.ndarray): E data array.
        t (np.ndarray): Time array.
        path (str): Path to save the output files.
    Returns:
        None: Saves the chunked data to HDF5 files.
    """
    colors = get_colormap_colors(n=n_chunks)

    chunk_ind = np.linspace(1, n_chunks, n_chunks, dtype=int)

    dt = t[1] - t[0]
    point = chunk_T // dt
    tot_point = int(point * n_chunks)

    Achunk = np.split(A[:tot_point], n_chunks)
    Echunk = np.split(E[:tot_point], n_chunks)
    chunk_t_arr = np.split(t[:tot_point], n_chunks)
    #get chunk start and end times
    T1 = [chunk_t_arr[i][0][0] for i in range(n_chunks)]
    T2 = [chunk_t_arr[i][-1][0] for i in range(n_chunks)]
    

    freqs = np.arange(0, 1 / (2 * dt) + 1 / chunk_T, 1 / chunk_T)
    window = np.kaiser(chunk_t_arr[0].size, beta=30)  
    norm = np.sum(window**2) / len(window)

    logger.info(f'Chunking data into {n_chunks} segments of {chunk_T} seconds each.')
    f_av, freq_tot, count, response, data, responseAV, dataAV = [], [], [], [], [], [], []

    plt.figure(figsize=(10, 6))
    for i in range(n_chunks):

        Af_chunk = np.fft.rfft(Achunk[i].T[0] * window) / norm**0.5
        Ef_chunk = np.fft.rfft(Echunk[i].T[0] * window) / norm**0.5

        f, dA, RA, c = average_log_chunks(freqs, 2 * (dt**2 / chunk_T) * np.abs(Af_chunk)**2, np.zeros_like(freqs), nseg)
        f, dE, RE, c = average_log_chunks(freqs, 2 * (dt**2 / chunk_T) * np.abs(Ef_chunk)**2, np.zeros_like(freqs), nseg)

        f_av.append(f)
        freq_tot.append(f)
        count.append(c)

        dataAV.append([dA, dE])
        responseAV.append([RA, RE])
      
        #data.append([data_noise[j][0] + Af_month , data_noise[j][1] + Ef_month])
        data.append([Af_chunk , Ef_chunk])
        response.append([np.zeros_like(freqs), np.zeros_like(freqs)])

        
        plt.loglog(f, (dA), color=colors[i], alpha = 0.3, lw=0.5, label = f'Chunk {i+1}', rasterized=True)
        plt.xlim(1e-4, 0.029)
        plt.ylim(1e-47, 1e-38)

    plt.savefig(f'{path}/chunked_data_{n_chunks}_chunks.png')
    plt.close()

    with h5py.File(f'{path}/data.h5', 'w') as f:
        for i, (data_chunk, response_chunk, freq_chunk) in enumerate(zip(data, response, freq_tot)):
            group = f.create_group(f'chunk_{i+1}')
            group.create_dataset('data', data=np.array(data_chunk, dtype=np.complex128))
            group.create_dataset('response', data=np.array(response_chunk, dtype=np.float64))
            group.create_dataset('freq', data=np.array(freq_chunk, dtype=np.float64))#

    # Save averaged data
    with h5py.File(f'{path}/data_av.h5', 'w') as f:
        for i, (data_chunk, response_chunk, freq_chunk, count_chunk) in enumerate(zip(dataAV, responseAV, f_av, count)):
            group = f.create_group(f'chunk_{i+1}')
            group.create_dataset('data', data=np.array(data_chunk, dtype=np.float64))
            group.create_dataset('response', data=np.array(response_chunk, dtype=np.float64))
            group.create_dataset('freq', data=np.array(freq_chunk, dtype=np.float64))
            group.create_dataset('count', data=np.array(count_chunk, dtype=np.int32))

    # Save start and end times of each chunk in txt
    np.savetxt(f'{path}/time_interval.txt',  [T1, T2])
    logger.info(f'Chunked data saved to {path}/data.h5 and {path}/data_av.h5')
    
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process input data for BAHAMAS.')
    parser.add_argument('--file', type=str, required=True, help='Path to the input HDF5 file containing XYZ data.')
    parser.add_argument('--chunk_T', type=float, default=2, help='Length of each chunk in weeks(default:  2).')
    parser.add_argument('--n_chunks', type=int, default=10, help='Number of chunks to create (default: 10).')
    parser.add_argument('--nseg', type=int, default=10, help='Number of segments for averaging (default: 10).')
    parser.add_argument('--path', type=str, required=True, help='Path to save the output files.')

    args = parser.parse_args()

    week = 7 * 24 * 3600  # Convert weeks to seconds
    nchunks = args.n_chunks
    chunk_T = args.chunk_T * week  # Convert chunk_T from weeks to seconds
    nseg = args.nseg

    # Validate input arguments
    logger.info(f"Processing file: {args.file}")
    A, E, T, t = XYZ2AET(args.file, chunk_T * nchunks)
    chunk_data(chunk_T, nchunks, nseg, A, E, t, args.path)
    logger.info(f"Data processing completed. Output saved to {args.path}/data.h5 and {args.path}/data_av.h5")

if __name__ == "__main__":
    main()
