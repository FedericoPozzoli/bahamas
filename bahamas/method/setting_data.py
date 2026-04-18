"""
Utility functions for generating Gaussian processes, averaging data, and estimating power spectral densities.
"""
import numpy as np
import h5py
import os


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
        mask = (freqs <= log_bins[i]) & (freqs < log_bins[i + 1])
        if np.any(mask):
            f_avg.append(0.5 * (freqs[mask][0] + freqs[mask][-1]))
            d_avg.append(np.mean(data[mask]))
            r_avg.append(np.mean(response[mask]))
            count.append(np.sum(mask))
            
    return np.array(f_avg), np.array(d_avg), np.array(r_avg), np.array(count)


def find_idx_coarse_graining(freq, freq_full, freq_coarse, mode='lin'):
    """
    Robust mapping from fine frequencies to coarse bins using linear or log binning.
    """
    freq = np.asarray(freq)
    freq_full = np.asarray(freq_full)
    freq_coarse = np.asarray(freq_coarse)

    # ---------- Linear binning ----------
    if mode == 'lin':  
        # Compute linear bin boundaries
        lin_bins = np.linspace(np.min(freq_full), np.max(freq_full), len(freq_coarse) + 1)

        # Digitize gives bin numbers in [1, len(freq_coarse)]
        idx = np.digitize(freq, lin_bins, right=False) - 1

        # Clip out-of-range values safely
        idx = np.clip(idx, 0, len(freq_coarse) - 1)

        return idx

    # ---------- Logarithmic binning ----------
    elif mode == 'log':
        # Only positive frequencies matter for log binning
        mask_pos = freq > 0
        freq_pos = freq[mask_pos]

        # Compute log-spaced bin boundaries
        log_min = np.log10(np.min(freq_full[freq_full > 0]))
        log_max = np.log10(np.max(freq_full))
        log_bins = np.logspace(log_min, log_max, len(freq_coarse) + 1)

        # Digitize gives bin numbers in [1, len(freq_coarse)]
        idx_pos = np.digitize(freq_pos, log_bins, right=False) - 1

        # Clip out-of-range values safely
        idx_pos = np.clip(idx_pos, 0, len(freq_coarse) - 1)

        # Prepare final index array
        indices = np.zeros(len(freq), dtype=int)
        indices[mask_pos] = idx_pos
        return indices

    else:
        raise ValueError("mode must be 'lin' or 'log'")


def psd_estimator(residuals, freq, fs=None, dt=None, window=None, 
                               mode='lin', n_segments=50):
    """
    Compute the averaged periodogram from residuals with proper normalization.
    
    This function takes time-domain residuals (or their FFT) and computes
    an averaged periodogram, which is an estimate of the power spectral density.
    Includes proper normalization for DFT-to-PSD conversion and windowing.
    
    Parameters
    ----------
    residuals : array_like
        Complex Fourier coefficients of the residuals, or time-domain data.
        If time-domain, FFT will be applied automatically.
    freq : array_like
        Frequency array corresponding to the residuals.
    fs : float, optional
        Sampling frequency in Hz. Required if residuals are in time domain.
        If not provided, will be inferred from freq array.
    dt : float, optional
        Time step (1/fs). Alternative to fs parameter.
    window : str or array_like, optional
        Window function to apply. Can be:
        - None: rectangular (no window)
        - 'hann', 'hamming', 'blackman', etc.: scipy window name
        - array: custom window array (must match length of time-domain data)
    mode : str, optional
        Averaging mode: 'linear' for uniform chunks, 'log' for logarithmic bins.
        Default is 'linear'.
    n_segments : int, optional
        Number of segments/bins for averaging. Default is 50.
    
    Returns
    -------
    freq_avg : ndarray
        Averaged frequency array.
    periodogram_avg : ndarray
        Averaged periodogram (PSD estimate) with proper normalization.
    counts : ndarray
        Number of frequency points in each averaged bin.
    
    Notes
    -----
    The normalization converts the periodogram to a proper PSD estimate:
    
    For DFT coefficients X[k]:
    PSD[k] = (2 * dt / (N * S2)) * |X[k]|^2
    
    where:
    - dt is the time step
    - N is the number of samples
    - S2 is the window power normalization: S2 = sum(window^2) / N
    
    For one-sided PSD (positive frequencies only), the factor of 2 accounts
    for the negative frequencies. DC and Nyquist components don't get this factor.
    
    """
    from scipy import signal as scipy_signal
    
    # Ensure inputs are numpy arrays
    residuals = np.asarray(residuals)
    freq = np.asarray(freq)
    
    # Determine sampling parameters
    if dt is None and fs is None:
        if len(freq) > 1:
            df = freq[1] - freq[0]
            dt = 1.0 / (2 * freq[-1]) if freq[-1] > 0 else 1.0 / df
            fs = 1.0 / dt
        else:
            raise ValueError("Cannot determine dt/fs from single frequency point")
    elif dt is not None:
        fs = 1.0 / dt
    else:
        dt = 1.0 / fs
    
    # Track if we're working with time-domain data
    time_domain = not np.iscomplexobj(residuals)
    
    # Apply window if specified (only for time-domain data)
    window_array = None
    if time_domain and window is not None:
        if isinstance(window, str):
            # Use scipy to generate window
            window_array = scipy_signal.get_window(window, len(residuals))
        else:
            window_array = np.asarray(window)
            if len(window_array) != len(residuals):
                raise ValueError("Window length must match residuals length")
        
        # Apply window
        residuals = residuals * window_array
    
    # Compute window power normalization factor
    if window_array is not None:
        S2 = np.sum(window_array**2) / len(window_array)
    else:
        S2 = 1.0  # Rectangular window
    
    # If residuals are real, assume time-domain and apply FFT
    if time_domain:
        N = len(residuals)
        residuals = np.fft.rfft(residuals)
        # Generate frequency array if not provided properly
        if len(freq) != len(residuals):
            freq = np.fft.rfftfreq(N, dt)
    else:
        # For frequency-domain input, infer N from frequency array
        N = 2 * (len(residuals) - 1)
    
    # Compute periodogram with proper normalization
    # Factor: 2 * dt / (N * S2) for one-sided PSD
    normalization_factor = 2.0 * dt / (N * S2)
    
    periodogram = normalization_factor * np.abs(residuals) ** 2
    
    # DC component (index 0) and Nyquist (if N is even) don't get factor of 2
    periodogram[0] /= 2.0
    if N % 2 == 0 and len(periodogram) > 1:
        periodogram[-1] /= 2.0
    
    
    # Now perform averaging
    if mode == 'lin':
        # Linear averaging in uniform chunks
        chunk_size = len(freq) // n_segments

        freq_chunks = np.array_split(freq, len(freq) // chunk_size)
        periodogram_chunks = np.array_split(periodogram, len(freq) // chunk_size)
        
        freq_avg = np.array([np.mean(chunk) for chunk in freq_chunks])
        periodogram_avg = np.array([np.mean(chunk) for chunk in periodogram_chunks])
        counts = np.array([len(chunk) for chunk in freq_chunks])
    

    elif mode == 'log':
        # Positive frequencies only, excluding first frequency if equal to log_min
        mask_pos = freq > 0
        freq_pos = freq[mask_pos]
        period_pos = periodogram[mask_pos]

        log_min = np.log10(freq_pos.min())
        log_max = np.log10(freq_pos.max())
        log_bins = np.logspace(log_min, log_max, n_segments + 1)

        # Exclude frequency equal to the first bin edge to match (freq > bins[i]) & (freq <= bins[i+1])
        mask_pos_strict = freq_pos > log_bins[0]
        freq_pos = freq_pos[mask_pos_strict]
        period_pos = period_pos[mask_pos_strict]

        # Assign each frequency to a bin (matches original loop)
        bin_idx = np.digitize(freq_pos, log_bins, right=True) - 1
        bin_idx = np.clip(bin_idx, 0, n_segments - 1)

        # Count points in each bin
        counts_all = np.bincount(bin_idx, minlength=n_segments)
        valid_bins = np.where(counts_all > 0)[0]

        # Sort frequencies by bin to find first/last per bin
        order = np.argsort(bin_idx)
        bin_sorted = bin_idx[order]
        freq_sorted = freq_pos[order]
        psd_sorted = period_pos[order]

        # Start and end indices for each bin
        edges = np.flatnonzero(np.diff(bin_sorted)) + 1
        starts = np.r_[0, edges]
        ends   = np.r_[edges, len(bin_sorted)]

        # Representative frequency: 0.5*(first + last)
        freq_avg_all = 0.5 * (freq_sorted[starts] + freq_sorted[ends-1])

        # PSD mean per bin
        psd_sum_all = np.add.reduceat(psd_sorted, starts)
        psd_avg_all = psd_sum_all / counts_all[valid_bins]

        # Output exactly matches original loop
        freq_avg = freq_avg_all
        periodogram_avg = psd_avg_all
        counts = counts_all[valid_bins]


    else:
        raise ValueError("mode must be 'lin' or 'log'")
    
    return freq_avg, periodogram_avg, counts