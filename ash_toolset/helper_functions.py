# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 14:02:51 2023

@author: Shanon
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
from scipy.io import wavfile
import soundfile as sf
from scipy.signal import butter
import scipy as sp
import datetime
from ash_toolset import constants as CN
from os.path import join as pjoin
import dearpygui.dearpygui as dpg
import librosa
from difflib import SequenceMatcher, _nlargest  # necessary imports of functions used by modified get_close_matches
from thefuzz import fuzz
from thefuzz import process
import functools
import os
import re
import requests
import logging
import gdown
from SOFASonix import SOFAFile
import sofar as sof
import csv
import random
import math
from scipy.signal import savgol_filter, medfilt, correlate
from scipy.ndimage import gaussian_filter1d
import noisereduce as nr
import queue
import threading
import time
import scipy.signal as signal
from scipy.stats import linregress
from os.path import exists
import concurrent.futures

def sort_names_by_values(names, values, descending=False):
    """
    Sorts a list of names based on corresponding integer values.

    Parameters:
        names (list of str): A list of names.
        values (list of int): A list of integer values corresponding to each name.
        descending (bool): If True, sorts in descending order. Defaults to False (ascending).

    Returns:
        list of str: A new list of names sorted according to the order of values.

    Raises:
        ValueError: If the input lists are not the same length.
        TypeError: If 'names' is not a list of strings or 'values' is not a list of integers.
    """
    # Validate input types
    if not isinstance(names, list) or not isinstance(values, list):
        raise TypeError("Both 'names' and 'values' must be lists.")
    
    if len(names) != len(values):
        raise ValueError("Both lists must be of the same length.")
    
    if not all(isinstance(name, str) for name in names):
        raise TypeError("All elements in 'names' must be strings.")
    
    if not all(isinstance(value, int) for value in values):
        raise TypeError("All elements in 'values' must be integers.")

    if not names:
        return []

    # Pair and sort based on the values
    paired_list = list(zip(names, values))
    sorted_pairs = sorted(paired_list, key=lambda pair: pair[1], reverse=descending)
    sorted_names = [name for name, _ in sorted_pairs]
    
    return sorted_names

def biased_spherical_coordinate_sampler(azim_src_set, elev_src_set, num_samples,
                                                                     biased_azimuth_centers=np.array([45, 135, 225, 315]),
                                                                     azimuth_bias_strength=20, plot_distribution=False):
    """
    Efficiently samples spherical coordinates (azimuth and elevation) with a bias
    towards specified azimuth angles, accounting for circular wrap-around.
    Optionally plots the distribution of selected azimuths.

    Args:
        azim_src_set (np.ndarray): Array of possible azimuth angles (degrees), sorted.
        elev_src_set (np.ndarray): Array of possible elevation angles (degrees).
        num_samples (int): The number of spherical coordinates to sample.
        biased_azimuth_centers (np.ndarray, optional): Array of azimuth angles
            (degrees) around which the bias will be applied. Defaults to [45, 135, 225, 315].
        azimuth_bias_strength (int, optional): Controls the strength of the azimuth 
            bias (higher value = stronger bias, narrower distribution). Defaults to 20. note: higher is weaker bias
        plot_distribution (bool, optional): If True, plots a histogram of the
            selected azimuth angles after sampling. Defaults to False.

    Returns:
        tuple: A tuple containing two lists:
            - selected_azimuths (list): List of randomly selected azimuth angles.
            - selected_elevations (list): List of randomly selected elevation angles.
    """
    selected_azimuths = []
    selected_elevations = []
    num_azim = len(azim_src_set)
    probabilities = np.zeros(num_azim, dtype=float)

    for i, azim in enumerate(azim_src_set):
        prob = 0.0
        for center in biased_azimuth_centers:
            diff = np.abs(azim - center)
            circular_diff = np.min([diff, 360 - diff])
            prob += np.exp(-(circular_diff**2) / (2 * azimuth_bias_strength**2))
        probabilities[i] = prob

    probabilities /= np.sum(probabilities)

    # Create a CDF for efficient random sampling
    cdf = np.cumsum(probabilities)

    for _ in range(num_samples):
        # Efficiently sample azimuth using CDF
        rand_val = random.random()
        selected_azimuth_index = np.searchsorted(cdf, rand_val)
        selected_azimuth = azim_src_set[selected_azimuth_index]
        selected_azimuths.append(selected_azimuth)

        # Randomly select an elevation angle (no bias)
        random_elevation_index = random.randint(0, len(elev_src_set) - 1)
        selected_elevation = elev_src_set[random_elevation_index]
        selected_elevations.append(selected_elevation)

    if plot_distribution:
        plt.figure(figsize=(10, 6))
        plt.hist(selected_azimuths, bins=np.arange(azim_src_set.min(), azim_src_set.max() + 10, 10), edgecolor='black')
        plt.xlabel("Azimuth Angle (degrees)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Selected Azimuth Angles (Circular, Efficient)")
        plt.xticks(np.arange(0, 361, 45))
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return selected_azimuths, selected_elevations






def octave_smooth(freqs, values, fraction=1/3):
    """
    Efficiently applies fractional octave smoothing to a frequency-domain spectrum.

    Args:
        freqs (np.ndarray): Array of frequencies (Hz), must be sorted.
        values (np.ndarray): Array of corresponding values (e.g., group delay).
        fraction (float, optional): The fractional octave band (e.g., 1/3 for third-octave).
                                     Defaults to 1/3.

    Returns:
        np.ndarray: Array of smoothed values.
    """
    smoothed = np.copy(values)
    log_freqs = np.log2(freqs, where=freqs > 0, out=np.full_like(freqs, np.nan))
    fraction_half = fraction / 2

    for i, f_log in enumerate(log_freqs):
        if np.isnan(f_log):
            continue

        lower_log_f = f_log - fraction_half
        upper_log_f = f_log + fraction_half

        lower_idx = np.searchsorted(log_freqs, lower_log_f, side='left')
        upper_idx = np.searchsorted(log_freqs, upper_log_f, side='right')

        if lower_idx < upper_idx:
            smoothed[i] = np.nanmean(values[lower_idx:upper_idx])
        # If no frequencies fall within the band, the original value is kept (already initialized)

    return smoothed

def octave_smooth_slow(freqs, values, fraction=1/3):
    smoothed = np.zeros_like(values)
    for i, f in enumerate(freqs):
        if f <= 0:
            smoothed[i] = values[i]
            continue
        f1 = f * 2**(-fraction / 2)
        f2 = f * 2**(fraction / 2)
        idx = np.where((freqs >= f1) & (freqs <= f2))[0]
        if len(idx) > 0:
            smoothed[i] = np.nanmean(values[idx])
        else:
            smoothed[i] = values[i]
    return smoothed


def calc_group_delay_from_ir(y, sr=None, n_fft=2048, hop_length=512, smoothing_type='hann', smoothing_window=31, system_delay_ms=None, verbose=False):
    """
    Calculates the group delay from a 1D NumPy array representing an impulse response.

    Args:
        y (np.ndarray): 1D NumPy array representing the impulse response.
        sr (int): Sample rate of the impulse response in Hz.
        n_fft (int, optional): Length of the FFT window. Defaults to 2048.
        hop_length (int, optional): Number of samples between successive FFT windows. Defaults to 512.
        smoothing_type (str, optional): Type of smoothing ('none', 'hann', 'gaussian', 'savitzky-golay', 'octave'). Defaults to 'hann'.
        smoothing_window (int, optional): Window size for smoothing (depends on the type). Defaults to 31.
        system_delay_ms (float, optional): An estimated constant system delay in milliseconds to subtract.
                                            If None, the function will attempt to estimate it.
                                            Defaults to None.

    Returns:
        tuple: A tuple containing two NumPy arrays:
               - frequencies (np.ndarray): Array of frequencies in Hz.
               - averaged_group_delay_ms (np.ndarray): Array of averaged group delay values in milliseconds.
    """
    import numpy as np
    from numpy.fft import fft
    from numpy import unwrap, diff
    from scipy.signal import savgol_filter, windows
    from scipy.ndimage import gaussian_filter1d

    

    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError("Input 'y' must be a 1D NumPy array.")
    if not isinstance(sr, (int, float)) or sr <= 0:
        raise ValueError("Input 'sr' must be a positive number representing the sample rate.")

    n_frames = (len(y) - n_fft) // hop_length + 1
    freq_bins = np.fft.fftfreq(n_fft, 1 / sr)
    group_delays = []

    if verbose:
        print("Starting group delay calculation...")

    for frame in range(n_frames):
        start_idx = frame * hop_length
        end_idx = start_idx + n_fft
        frame_data = y[start_idx:end_idx]

        yf = fft(frame_data, n=n_fft)
        phase = np.angle(yf)
        unwrapped_phase = unwrap(phase)
        delta_phase = diff(unwrapped_phase)
        delta_f = np.diff(freq_bins)
        group_delay = -delta_phase / (2 * np.pi * delta_f)
        group_delay_ms = group_delay * 1000
        group_delays.append(group_delay_ms)

    averaged_group_delay_ms = np.mean(group_delays, axis=0)

    if verbose:
        print(f"Group delay (before smoothing): Min = {np.nanmin(averaged_group_delay_ms)} ms, "
              f"Max = {np.nanmax(averaged_group_delay_ms)} ms, "
              f"Mean = {np.nanmean(averaged_group_delay_ms)} ms, "
              f"Std = {np.nanstd(averaged_group_delay_ms)} ms")

    if np.any(np.isnan(averaged_group_delay_ms)) and verbose:
        print("Warning: NaN values detected in the averaged group delay.")

    if smoothing_type == 'savitzky-golay' and smoothing_window > 1:
        smoothed_group_delay_ms = savgol_filter(averaged_group_delay_ms, smoothing_window, 3)
    elif smoothing_type == 'hann' and smoothing_window > 1 and smoothing_window % 2 == 1:
        window = windows.hann(smoothing_window)
        window /= np.sum(window)
        smoothed_group_delay_ms = np.convolve(averaged_group_delay_ms, window, mode='same')
    elif smoothing_type == 'gaussian' and smoothing_window > 0:
        smoothed_group_delay_ms = gaussian_filter1d(averaged_group_delay_ms, smoothing_window)
    elif smoothing_type == 'octave':
        freqs = freq_bins[1:len(averaged_group_delay_ms)+1]
        smoothed_group_delay_ms = octave_smooth(freqs, averaged_group_delay_ms, fraction=1/smoothing_window)
    else:
        smoothed_group_delay_ms = averaged_group_delay_ms

    if verbose:
        print(f"Group delay (after smoothing): Min = {np.nanmin(smoothed_group_delay_ms)} ms, "
              f"Max = {np.nanmax(smoothed_group_delay_ms)} ms, "
              f"Mean = {np.nanmean(smoothed_group_delay_ms)} ms, "
              f"Std = {np.nanstd(smoothed_group_delay_ms)} ms")

    if np.any(np.isnan(smoothed_group_delay_ms)) and verbose:
        print("Warning: NaN values detected in the smoothed group delay.")

    if system_delay_ms is None:
        if smoothed_group_delay_ms.size > 0:
            mid_freq_start = int(len(smoothed_group_delay_ms) * 0.1)
            mid_freq_end = int(len(smoothed_group_delay_ms) * 0.4)
            mid_freq_slice = smoothed_group_delay_ms[mid_freq_start:mid_freq_end]
            if mid_freq_slice.size > 0 and not np.all(np.isnan(mid_freq_slice)):
                system_delay_ms_estimated = np.nanmedian(mid_freq_slice)
                if not np.isnan(system_delay_ms_estimated):
                    smoothed_group_delay_ms -= system_delay_ms_estimated
                    if verbose:
                        print(f"Estimated system delay (ms): {system_delay_ms_estimated}")
    else:
        smoothed_group_delay_ms -= system_delay_ms
        if verbose:
            print(f"Applied system delay (ms): {system_delay_ms}")

    if np.all(smoothed_group_delay_ms == smoothed_group_delay_ms[0]) and verbose:
        print("Warning: Smoothed group delay is constant across all frequencies.")

    if np.any(np.isnan(smoothed_group_delay_ms)):
        print("Warning: Final result contains NaN values.")

    if verbose:
        print(f"Final group delay: Min = {np.nanmin(smoothed_group_delay_ms)} ms, "
              f"Max = {np.nanmax(smoothed_group_delay_ms)} ms, "
              f"Mean = {np.nanmean(smoothed_group_delay_ms)} ms, "
              f"Std = {np.nanstd(smoothed_group_delay_ms)} ms")
        print("Group delay calculation completed.")

    return freq_bins[1:len(smoothed_group_delay_ms)+1], smoothed_group_delay_ms


def save_group_delay_to_csv(frequencies, group_delay_ms, filename="group_delay_analysis.csv"):
    """
    Saves frequency and group delay data to a CSV file.

    Args:
        frequencies (np.ndarray): Array of frequency values (Hz).
        group_delay_ms (np.ndarray): Array of group delay values (ms).
        filename (str): Name of the CSV file to save. Defaults to 'group_delay_analysis.csv'.
    """
    if not len(frequencies) == len(group_delay_ms):
        raise ValueError("Frequencies and group delay arrays must be the same length.")
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frequency (Hz)", "Group Delay (ms)"])
        for f, gd in zip(frequencies, group_delay_ms):
            writer.writerow([f, gd])
    
    print(f"Group delay data saved to '{os.path.abspath(filename)}'")




def create_weighted_list(length: int, randomize: bool = False, seed: int = None, hot_index: int = None, verbose: bool = True) -> list:
    """
    Creates a list of specified length where the sum of all elements is 1.

    Modes:
    - One-hot: Set a specific index to 1, all others to 0 (if hot_index is specified).
    - Random: Generate random weights summing to 1 (if randomize=True).
    - Uniform: All values equal, summing to 1 (default).

    Parameters:
    length (int): The number of elements in the list (must be > 0).
    randomize (bool): If True, generate random weights.
    seed (int, optional): Seed for reproducible randomness (only used if randomize=True).
    hot_index (int, optional): If set, creates a one-hot list with this index set to 1.
    verbose (bool): If True, prints the generated weightings.

    Returns:
    list: A list of floats where the sum is 1.

    Raises:
    ValueError: If parameters are invalid.
    """
    if length <= 0:
        raise ValueError("Length must be a positive integer.")

    if hot_index is not None:
        if not 0 <= hot_index < length:
            raise ValueError(f"hot_index must be between 0 and {length - 1}")
        one_hot = [0.0] * length
        one_hot[hot_index] = 1.0
        if verbose:
            print(f"One-hot weighting (index {hot_index}): {one_hot}")
        return one_hot

    if randomize:
        rng = np.random.default_rng(seed)
        weights = rng.random(length)
        weights /= weights.sum()
        weights_list = weights.tolist()
        if verbose:
            print(f"Random weightings (seed={seed}): {weights_list}")
        return weights_list
    else:
        weights = np.ones(length) / length
        weights_list = weights.tolist()
        if verbose:
            print(f"Uniform weightings: {weights_list}")
        return weights_list

def combine_measurements_4d(arr: np.ndarray) -> np.ndarray:
    """
    Combines impulse response samples (dimension 4) of multiple measurements
    (dimension 2) by summing, reducing the number of measurements to 7.

    Args:
        arr: The input 4D NumPy array with shape (1, measurements, 2, samples).
             'measurements' is assumed to be at least 7.

    Returns:
        A 4D NumPy array with shape (1, 7, 2, samples), where the samples
        are summed from the original measurements.
    """

    if arr.ndim != 4:
        raise ValueError("Input array must be 4-dimensional.")

    if arr.shape[0] != 1:
        raise ValueError("Dimension 1 must have length 1.")

    if arr.shape[2] != 2:
        raise ValueError("Dimension 3 must have length 2.")

    num_measurements = arr.shape[1]

    if num_measurements < 7:
        raise ValueError("Dimension 2 must have length of at least 7.")

    # Calculate the number of measurements to combine per output measurement
    combine_per = num_measurements // 7

    # Calculate the remaining measurements after even division
    remainder = num_measurements % 7

    # Initialize the output array
    output_arr = np.zeros((1, 7, 2, arr.shape[3]))

    # Combine measurements
    for i in range(7):
        start_idx = i * combine_per
        end_idx = (i + 1) * combine_per

        # Handle the remainder: distribute it to the first 'remainder' output measurements
        if i < remainder:
            end_idx += 1

        output_arr[0, i, :, :] = np.sum(arr[0, start_idx:end_idx, :, :], axis=0)

    return output_arr

def reshape_array_to_two_dims(arr: np.ndarray) -> np.ndarray:
    """
    Reshapes a NumPy array to meet specific dimension requirements.
    Useful for reshaping SRIRs, where we only want 2 dimensions

    Args:
        arr: The input NumPy array of arbitrary dimensions.

    Returns:
        A NumPy array with the following shape:
        - The last dimension is the longest dimension from the input array.
        - The first dimension is the product of all other dimensions 
          merged together.
    """

    original_shape = arr.shape
    num_dims = len(original_shape)

    if num_dims == 0:
        return arr  # Or raise an exception, depending on desired behavior for scalar input

    # Find the longest dimension and its index
    longest_dim = 0
    longest_dim_index = -1  # Initialize to -1 to handle empty shapes correctly
    for i, dim in enumerate(original_shape):
        if dim > longest_dim:
            longest_dim = dim
            longest_dim_index = i

    if num_dims == 1:
        return arr.reshape(1, original_shape[0])  # Handle 1D array case

    # Calculate the shape of the first dimension of the result
    other_dims_product = 1
    for i in range(num_dims):
        if i != longest_dim_index:
            other_dims_product *= original_shape[i]

    # Create the new shape tuple
    new_shape = (other_dims_product, longest_dim)

    # Move the longest axis to the last position using transpose
    if longest_dim_index != num_dims - 1:
        axes = list(range(num_dims))
        axes.pop(longest_dim_index)
        axes.append(longest_dim_index)
        arr = arr.transpose(axes)

    # Reshape the array
    reshaped_arr = arr.reshape(new_shape)

    return reshaped_arr


def reshape_array_to_three_dims(arr: np.ndarray) -> np.ndarray:
    """
    Reshapes a NumPy array to meet specific dimension requirements.
    useful for reshaping MultiSpeakerBRIR, where we want to keep an axis with 2 channels, 3 dimensions in total

    Args:
        arr: The input NumPy array of arbitrary dimensions.

    Returns:
        A NumPy array with the following shape:
        - The last dimension is the longest dimension from the input array.
        - The 2nd last dimension is the dimension from the input array with length 2.
        - The first dimension is the product of all other dimensions merged together.
    """

    original_shape = arr.shape
    num_dims = len(original_shape)

    if num_dims < 2:
        raise ValueError("Input array must have at least 2 dimensions.")

    # Find the longest dimension and its index
    longest_dim = 0
    longest_dim_index = -1
    for i, dim in enumerate(original_shape):
        if dim > longest_dim:
            longest_dim = dim
            longest_dim_index = i

    # Find the dimension with length 2 and its index
    dim_2_index = -1
    for i, dim in enumerate(original_shape):
        if dim == 2:
            dim_2_index = i
            break  # Only need to find the first dimension with length 2

    if dim_2_index == -1:
        raise ValueError("Input array must have a dimension with length 2.")

    # Calculate the shape of the first dimension of the result
    other_dims_product = 1
    for i in range(num_dims):
        if i != longest_dim_index and i != dim_2_index:
            other_dims_product *= original_shape[i]

    # Create the new shape tuple
    new_shape = (other_dims_product, 2, longest_dim)

    # Move the longest axis and axis with length 2 to the last positions using transpose
    if longest_dim_index != num_dims - 1 or dim_2_index != num_dims - 2:
        axes = list(range(num_dims))
        axes.pop(longest_dim_index)
        axes.pop(axes.index(dim_2_index)) #remove dim_2 after longest_dim is removed.
        axes.extend([dim_2_index, longest_dim_index])
        arr = arr.transpose(axes)

    # Reshape the array
    reshaped_arr = arr.reshape(new_shape)

    return reshaped_arr














def average_measurement_groups(
    data: np.ndarray,
    group_size: int,
    shuffle: bool = False,
    seed: int = None,
    smoothing_method: str = "savgol",
    window_size: int = 115,#75,205
    polyorder: int = 3,#3
    gaussian_sigma: float = 1.0,
    ema_alpha: float = 0.3
) -> np.ndarray:
    """
    Averages every group of `group_size` rows in a 2D array, with optional shuffling and smoothing.

    Parameters:
        data (np.ndarray): 2D array of shape (measurements, samples)
        group_size (int): Number of measurements to average per group
        shuffle (bool): Whether to shuffle rows before grouping
        seed (int): Seed for random shuffle reproducibility
        smoothing_method (str): Smoothing type: 'savgol', 'moving_average', 'gaussian', 'median', 'ema'
        window_size (int): Window size for moving average, median, or Savitzky-Golay
        polyorder (int): Polynomial order for Savitzky-Golay filter
        gaussian_sigma (float): Sigma for Gaussian filter
        ema_alpha (float): Alpha for exponential moving average

    Returns:
        np.ndarray: Smoothed and averaged 2D array of shape (measurements // group_size, samples)
    """
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array (measurements x samples).")
    if group_size < 1:
        raise ValueError("group_size must be >= 1")

    num_measurements, num_samples = data.shape
    if num_measurements < group_size:
        raise ValueError(f"Array must have at least {group_size} measurements.")

    if shuffle:
        rng = np.random.default_rng(seed)
        data = rng.permutation(data)

    usable_rows = (num_measurements // group_size) * group_size
    data_trimmed = data[:usable_rows]
    reshaped = data_trimmed.reshape(-1, group_size, num_samples)
    averaged = reshaped.mean(axis=1)

    # Apply smoothing to each row if specified
    if smoothing_method:
        for i in range(averaged.shape[0]):
            row = averaged[i]
            if smoothing_method == "savgol" and window_size > 1 and window_size % 2 == 1 and polyorder < window_size:
                row = savgol_filter(row, window_length=window_size, polyorder=polyorder)
            elif smoothing_method == "moving_average":
                kernel = np.ones(window_size, dtype=np.float32) / window_size
                row = np.convolve(row, kernel, mode='same')
            elif smoothing_method == "gaussian":
                row = gaussian_filter1d(row, sigma=gaussian_sigma)
            elif smoothing_method == "median":
                row = medfilt(row, kernel_size=window_size)
            elif smoothing_method == "ema":
                row = exponential_moving_average(row, alpha=ema_alpha)
            averaged[i] = row

    return averaged


def load_average_and_save_npy_directory(
    directory: str,
    output_file: str,
    group_size: int = 7,#13 for combined_b, 20 for combined_c, 7 for d
    shuffle: bool = True,#True for combined_b
    seed: int = 120,#8,11,12 for combined_b, 19,29,80 for combined_c
    smoothing_method: str = "savgol",
    window_size: int = 2499,#115,135 for combined_b, 199 for combined_c, 39 for d
    polyorder: int = 3,
    gaussian_sigma: float = 1.0,
    ema_alpha: float = 0.3
) -> np.ndarray:
    """
    Loads all .npy files from a directory, crops arrays to same length,
    averages them in groups (with optional shuffle and smoothing),
    then saves the final combined array to disk.

    Parameters:
        directory (str): Folder containing .npy files (each 2D)
        output_file (str): Where to save the final result
        group_size (int): How many rows to average per group
        shuffle (bool): Shuffle rows before averaging
        seed (int): Random seed for reproducibility
        smoothing_method (str): 'savgol', 'moving_average', 'gaussian', 'median', 'ema'
        window_size (int): Smoothing window size
        polyorder (int): Savitzky-Golay polyorder
        gaussian_sigma (float): Sigma for Gaussian smoothing
        ema_alpha (float): Alpha for EMA

    Returns:
        np.ndarray: Final stacked and smoothed array
    """
    arrays = []
    min_samples = None
    npy_files = sorted(f for f in os.listdir(directory) if f.endswith('.npy'))

    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in directory: {directory}")

    print(f"Found {len(npy_files)} files. Loading...")

    for idx, file in enumerate(npy_files, start=1):
        path = os.path.join(directory, file)
        data = np.load(path)

        if data.ndim != 2:
            print(f"Skipping '{file}': not a 2D array.")
            continue

        if min_samples is None or data.shape[1] < min_samples:
            min_samples = data.shape[1]

        arrays.append(data)
        print(f"[{idx}/{len(npy_files)}] Loaded '{file}' with shape {data.shape}")

    processed = []
    for i, array in enumerate(arrays, start=1):
        cropped = array[:, :min_samples]
        grouped = average_measurement_groups(
            cropped,
            group_size=group_size,
            shuffle=shuffle,
            seed=seed,
            smoothing_method=smoothing_method,
            window_size=window_size,
            polyorder=polyorder,
            gaussian_sigma=gaussian_sigma,
            ema_alpha=ema_alpha
        )
        processed.append(grouped)
        print(f"Processed {i}/{len(arrays)} arrays -> shape {grouped.shape}")

    final_array = np.vstack(processed)
    print(f"\nFinal array shape: {final_array.shape}")  # New line added
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    np.save(output_file, final_array)
    print(f"Saved combined array to '{output_file}'")
    
    return final_array


def compute_best_shift(reference: np.ndarray, target: np.ndarray, max_shift: int = 1024) -> int:
    """
    Compute the lag that maximizes cross-correlation between reference and target.
    The max_shift parameter limits how far the target is allowed to shift.
    """
    corr = correlate(target, reference, mode="full")
    lags = np.arange(-len(target) + 1, len(reference))
    center = len(corr) // 2
    window = slice(center - max_shift, center + max_shift + 1)
    best_lag = lags[window][np.argmax(corr[window])]
    return best_lag


def plot_summed_measurements(file_path: str):
    """
    Loads a 2D NumPy array (measurements x samples) from the given file path,
    sums all measurements, and plots the resulting 1D array.

    Parameters:
        file_path (str): Path to the .npy file containing the 2D matrix.
    """
    # Load the matrix
    matrix = np.load(file_path)

    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D array, but got shape {matrix.shape}")

    # Sum across measurements (rows)
    summed = np.sum(matrix, axis=0)

    # Plot the result
    plt.figure(figsize=(10, 4))
    plt.plot(summed, label="Summed Measurements")
    plt.title("Sum of All Measurements")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_average_measurements(file_path: str):
    """
    Loads a 2D NumPy array (measurements x samples) from the given file path,
    computes the average of all measurements, and plots the resulting 1D array.
    Also counts how many measurements consist entirely of zeros.

    Parameters:
        file_path (str): Path to the .npy file containing the 2D matrix.
    """
    # Load the matrix
    matrix = np.load(file_path)

    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D array, but got shape {matrix.shape}")

    # Count rows that are all zeros
    zero_rows = np.all(matrix == 0, axis=1)
    num_zero_rows = np.sum(zero_rows)
    print(f"Number of all-zero measurements: {num_zero_rows}")

    # Compute average across measurements (rows)
    average = np.mean(matrix, axis=0)

    # Plot the result
    plt.figure(figsize=(10, 4))
    plt.plot(average, label="Average Measurement")
    plt.title("Average of All Measurements")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()















def expand_measurements_with_pitch_shift(
    measurement_array: np.ndarray,
    desired_measurements: int,
    fs: int = CN.SAMP_FREQ,
    pitch_range: tuple = (0, 24),
    shuffle: bool = False,
    antialias: bool = True,
    seed=65529189939976765123732762606216328531,
    plot_sample: int = 0,#0
    num_threads: int = 4,
    gui_logger=None,
    cancel_event=None,
    pitch_shift_comp=True,
    ignore_ms: float = CN.IGNORE_MS,  # 0
    report_progress=0
) -> tuple[np.ndarray, int]:
    """
    Expand a smaller measurement array to a desired number of measurements
    using randomized pitch shifting, with optional anti-aliasing, using multithreading.
    
    Returns:
        Tuple[np.ndarray, int]: The expanded 2D array and status code:
            0 = Success, 1 = Failure, 2 = Cancelled
    """

    def apply_pitch_shift(ir, fs, n_steps, antialias):
        if antialias:
            rate = 2 ** (-n_steps / 12.0)
            new_sr = int(fs * rate)
            resampled = librosa.resample(ir, orig_sr=fs, target_sr=new_sr, res_type="kaiser_best")
            shifted = librosa.effects.pitch_shift(resampled, sr=new_sr, n_steps=n_steps)
            shifted = librosa.resample(shifted, orig_sr=new_sr, target_sr=fs, res_type="kaiser_best")
        else:
            shifted = librosa.effects.pitch_shift(ir, sr=fs, n_steps=n_steps)
        return shifted

    measurement_array = measurement_array.astype(np.float64, copy=False)
    base_n, sample_len = measurement_array.shape
    output = np.empty((desired_measurements, sample_len), dtype=np.float64)
    status = 1
    
    # Convert ms to samples
    ignore_samples = int((ignore_ms / 1000.0) * fs)
    if ignore_samples >= sample_len:
        raise ValueError(f"ignore_samples ({ignore_samples}) must be less than the measurement length ({sample_len})")

    count_lock = threading.Lock()
    count = 0
    last_print_time = time.time()
    
    #set appropriate range
    # If both elements are numerically zero (e.g., 0 or 0.0)
    if abs(pitch_range[0]) < 1e-6 and abs(pitch_range[1]) < 1e-6:
        pitch_range = (0, 24)  # replace with default
        log_with_timestamp("Invalid pitch range provided, resetting to default (0,24)", gui_logger)

    try:
        if seed == "random":
            seed = np.random.SeedSequence().entropy
        rng = np.random.default_rng(seed)

        # Pre-generate pitch shift values
        pitch_shifts = rng.uniform(pitch_range[0], pitch_range[1], desired_measurements)

        if pitch_shift_comp:
            # --- Compensation Filter Setup ---
            n_fft = 2048
            truncate_len = 512
            f_min = 20
            f_max = 20000
            # Precompute FFT magnitudes of base measurements once
            #base_mags = np.abs(np.fft.rfft(measurement_array, n=n_fft, axis=1))
            base_mags = np.abs(np.fft.rfft(measurement_array[:, ignore_samples:], n=n_fft, axis=1))
            avg_mag = np.mean(base_mags, axis=0)
            # Sample compensation points across pitch range
            pitch_probe_points = np.linspace(pitch_range[0], pitch_range[1], 20)#increase to improve quality
            compensation_filters = {}
            log_with_timestamp("Building compensation curves...", gui_logger)
            max_base = 20  # Limit number of base measurements used
            for p in pitch_probe_points:
                # Pitch shift all base measurements for this probe once
                shifted_irs = [
                    apply_pitch_shift(measurement_array[i], fs, p, antialias)[ignore_samples:]
                    for i in range(min(base_n, max_base))
                ]
                
                # Compute FFT magnitudes once per shifted IR
                shifted_mags = np.abs(np.fft.rfft(np.stack(shifted_irs), n=n_fft, axis=1))
                shifted_mag_mean = np.mean(shifted_mags, axis=0)
    
                comp_mag = avg_mag / np.maximum(shifted_mag_mean, 1e-6)
                log_mag_db = savgol_filter(20 * np.log10(comp_mag), window_length=31, polyorder=3)
                smoothed_mag = 10 ** (log_mag_db / 20)
                impulse = build_min_phase_filter(
                    smoothed_mag, fs, n_fft=n_fft, truncate_len=truncate_len, f_min=f_min, f_max=f_max, band_limit=True
                )
                compensation_filters[p] = impulse

        if plot_sample > 0 and pitch_shift_comp:
            log_with_timestamp("Plotting compensation curves...", gui_logger)
            import matplotlib.pyplot as plt
            freqs = np.fft.rfftfreq(n_fft, 1 / fs)
            plt.figure(figsize=(10, 6))
            for p in sorted(pitch_probe_points):
                mag = np.abs(np.fft.rfft(compensation_filters[p], n=n_fft))
                mag_db = 20 * np.log10(np.maximum(mag, 1e-8))
                plt.plot(freqs, mag_db, label=f"{p:+.1f} semitones")
            plt.title("Band-limited, Min-Phase Compensation Curves")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Gain (dB)")
            plt.xscale("log")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        def process_one(i):
            nonlocal count, last_print_time
            if cancel_event and cancel_event.is_set():
                return None

            ir = measurement_array[i % base_n]
            pitch_shift = pitch_shifts[i]
            try:
                result = apply_pitch_shift(ir, fs, pitch_shift, antialias)

                if len(result) < sample_len:
                    padded = np.zeros(sample_len)
                    padded[:len(result)] = result
                    result = padded
                else:
                    result = result[:sample_len]

                if pitch_shift_comp:
                    # Find closest compensation filter for pitch shift
                    closest_pitch = min(compensation_filters.keys(), key=lambda x: abs(x - pitch_shift))
                    comp_filter = compensation_filters[closest_pitch]
    
                    # Use fftconvolve for faster convolution
                    result = sp.signal.fftconvolve(result, comp_filter, mode='same')

                with count_lock:
                    count += 1
                    current_time = time.time()
                    if count % 100 == 0 or count == desired_measurements or current_time - last_print_time >= 5.0:
                        log_with_timestamp(
                            f"Expansion Progress: {count}/{desired_measurements} measurements processed.", gui_logger
                        )
                        last_print_time = current_time
                        if report_progress > 0:
                            a_point = 0.1
                            b_point = 0.35
                            progress = a_point + (float(count) / desired_measurements) * (b_point - a_point)
                            update_gui_progress(report_progress, progress=progress)

                return (i, result)

            except Exception as e:
                log_with_timestamp(f"Error processing measurement {i}: {e}", gui_logger)
                return (i, None)

        # Keep your threading code unchanged:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_one, i) for i in range(desired_measurements)]
            for future in concurrent.futures.as_completed(futures):
                if cancel_event and cancel_event.is_set():
                    status = 2
                    log_with_timestamp("Expansion was cancelled by user.", gui_logger)
                    return output, status

                result = future.result()
                if result is None:
                    status = 2
                    log_with_timestamp("Expansion was cancelled by user.", gui_logger)
                    return output, status

                i, shifted_measurement = result
                if shifted_measurement is not None:
                    output[i] = shifted_measurement
                else:
                    output[i] = np.zeros(sample_len)

        status = 0
        log_with_timestamp("Expansion complete.", gui_logger)

        if shuffle:
            rng.shuffle(output)

    except Exception as e:
        log_with_timestamp(f"Exception in expansion: {e}", gui_logger)
        status = 1

    return output, status






def generate_sequence(start: int, multiplier: float, limit: int) -> list:
    """
    Generate a list of unique, rounded integers starting from a given number.
    Each subsequent number is the previous one multiplied by a float.
    Stops when the next value exceeds the specified limit.

    Parameters:
        start (int): The starting integer of the sequence.
        multiplier (float): The multiplier applied to each step (can be a float).
        limit (int): The maximum value the sequence can reach (inclusive).

    Returns:
        list: A list of unique, rounded integers in the generated sequence.
    """
    sequence = []
    seen = set()
    value = start

    while value <= limit:
        rounded = round(value)
        if rounded not in seen:
            sequence.append(rounded)
            seen.add(rounded)
        value *= multiplier

    return sequence



def reshape_to_4d(measurement_array: np.ndarray, first_dim: int) -> np.ndarray:
    """
    Reshape a 2D array (measurements x samples) into a 4D array with shape (first_dim x N x 1 x samples),
    redistributing measurements across the first three dimensions. If needed, fills remaining
    positions with zeros.

    Parameters:
        measurement_array (np.ndarray): A 2D NumPy array of shape (measurements x samples)
        first_dim (int): The desired size of the first dimension (e.g., 7)

    Returns:
        np.ndarray: A 4D array of shape (first_dim x N x 1 x samples)
    """
    measurements, samples = measurement_array.shape

    # Compute the number of slots needed per group
    N = math.ceil(measurements / first_dim)

    # Initialize the output array with zeros
    output = np.zeros((first_dim, N, 1, samples), dtype=measurement_array.dtype)

    # Fill the array by distributing measurements across first_dim and N
    for idx in range(measurements):
        group = idx % first_dim  # Group index along the first dimension
        slot = idx // first_dim  # Slot index along the second dimension
        output[group, slot, 0, :] = measurement_array[idx]

    return output

def reshape_to_4d_and_center_slots(measurement_array: np.ndarray, first_dim: int) -> np.ndarray:
    """
    Reshape a 2D array (measurements x samples) into a 4D array with shape (first_dim x N x 1 x samples),
    redistributing measurements across the first three dimensions. Zero-pads as needed.
    Then, for each group (axis 0), subtracts the per-sample mean across slots (axis 1),
    and prints the residual mean magnitude for verification.

    Parameters:
        measurement_array (np.ndarray): 2D array of shape (measurements x samples).
        first_dim (int): Desired size of the first dimension (e.g., 7).

    Returns:
        np.ndarray: A 4D array of shape (first_dim x N x 1 x samples), centered per group.
    """
    measurements, samples = measurement_array.shape
    N = math.ceil(measurements / first_dim)

    # Initialize output array
    output = np.zeros((first_dim, N, 1, samples), dtype=measurement_array.dtype)

    # Fill output array
    for idx in range(measurements):
        group = idx % first_dim
        slot = idx // first_dim
        output[group, slot, 0, :] = measurement_array[idx]

    # Subtract mean per sample across slots and compute residuals
    for g in range(first_dim):
        group_data = output[g, :, 0, :]                     # Shape: (N, samples)
        mean_per_sample = np.mean(group_data, axis=0)       # Shape: (samples,)
        output[g, :, 0, :] -= mean_per_sample               # Subtract across slots

        # Residual check
        residual_mean = np.mean(output[g, :, 0, :], axis=0)
        max_residual = np.max(np.abs(residual_mean))
        print(f"Group {g} residual mean after subtraction (max abs): {max_residual:.2e}")

    return output

def delete_all_files_in_directory(directory, verbose=True, gui_logger=None):
    """
    Deletes all files in the given directory and its subdirectories.
    
    Args:
        directory (str): The path to the directory.
        verbose (bool): If True, prints status messages. If False, suppresses output.
    """
    if not os.path.isdir(directory):
        if verbose:
            log_string = f"'{directory}' is not a valid directory. Unable to delete"
            log_with_timestamp(log_string, gui_logger=None)
        return

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                if verbose:
                    log_string = f"Deleted file: {file_path}"
                    log_with_timestamp(log_string, gui_logger=None)
            except Exception as e:
                if verbose:
                    log_string = f"Could not delete file: {file_path} (Reason: {e})"
                    log_with_timestamp(log_string, gui_logger=None)

def sofa_load_object(sofa_local_fname, gui_logger=None):
    """
    Loads a SOFA file and returns a dictionary of variables starting with 'sofa_'.
    
    Args:
        sofa_local_fname (str): Path to the local SOFA file.
        gui_logger (optional): Logger for GUI messages.
    
    Returns:
        dict: Dictionary of variables starting with 'sofa_'.
    """
    # Initialize an empty dictionary to store variables starting with 'sofa_'
    sofa_vars = {}
    
    try:
        #first try loading with SOFAsonix
        try:
            loadsofa = SOFAFile.load(sofa_local_fname, verbose=False)
            sofa_vars['sofa_data_ir'] = loadsofa.data_ir
            sofa_vars['sofa_samplerate'] = int(loadsofa.Data_SamplingRate[0])
            sofa_vars['sofa_source_positions'] = loadsofa.SourcePosition
            sofa_vars['sofa_convention_name'] = loadsofa.GLOBAL_SOFAConventions
            sofa_vars['sofa_version'] = loadsofa.GLOBAL_Version
            sofa_vars['sofa_convention_version'] = loadsofa.GLOBAL_SOFAConventionsVersion
        except:
            log_string = 'Unable to load SOFA file with SOFAsonix. Attempting to load with sofar'
            log_with_timestamp(log_string, gui_logger=None)
            
            try:
                #if fails, try loading with SOFAR
                loadsofa = sof.read_sofa(sofa_local_fname)#
                sofa_vars['sofa_data_ir'] = loadsofa.Data_IR
                sofa_vars['sofa_samplerate'] = int(loadsofa.Data_SamplingRate)
                sofa_vars['sofa_source_positions'] = loadsofa.SourcePosition
                sofa_vars['sofa_convention_name'] = loadsofa.GLOBAL_SOFAConventions
                sofa_vars['sofa_version'] = loadsofa.GLOBAL_Version
                sofa_vars['sofa_convention_version'] = loadsofa.GLOBAL_SOFAConventionsVersion
                
                log_string = 'Loaded Successfully with sofar'
                log_with_timestamp(log_string, gui_logger=None)
            except:
                try:
                    #if fails, try loading with SOFAR, with verify set to false
                    loadsofa = sof.read_sofa(sofa_local_fname, verify=False)#(sofa_local_fname, verify=False) verify=False ignores convention violations
                    sofa_vars['sofa_data_ir'] = loadsofa.Data_IR
                    sofa_vars['sofa_samplerate'] = int(loadsofa.Data_SamplingRate)
                    sofa_vars['sofa_source_positions'] = loadsofa.SourcePosition
                    sofa_vars['sofa_convention_name'] = loadsofa.GLOBAL_SOFAConventions
                    sofa_vars['sofa_version'] = loadsofa.GLOBAL_Version
                    sofa_vars['sofa_convention_version'] = loadsofa.GLOBAL_SOFAConventionsVersion
                    
                    log_string = 'Loaded Successfully with sofar'
                    log_with_timestamp(log_string, gui_logger=None)
                
                except:
                    log_string = 'Unable to load SOFA file. Likely due to unsupported convention version.'
                    log_with_timestamp(log_string, gui_logger=None)
            
                    raise ValueError('Unable to load SOFA file')
        
    
    except Exception as ex:

        log_string = 'SOFA load workflow failed'
        log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
        
    return sofa_vars
    

def load_convert_npy_to_float64(file_path: str, verbose: bool = False) -> np.ndarray:
    """
    Loads a .npy file, converts its data type from float32 to float64, and returns the converted array.
    
    Parameters:
    file_path (str): The path to the .npy file.
    verbose (bool): If True, prints information about the processing steps; otherwise, remains silent.
    
    Returns:
    np.ndarray: The converted array with dtype float64.
    """
    if verbose:
        log_string_a = f"Loading file: {file_path}"
        log_with_timestamp(log_string_a)

    # Load the .npy file
    data = np.load(file_path)
    
    if verbose:
        log_string_a = f"Original data type: {data.dtype}"
        log_with_timestamp(log_string_a)

    # Convert data to float64 if it is float32
    if data.dtype == np.float32:
        data = data.astype(np.float64)
        if verbose:
            log_string_a = "Data converted from float32 to float64."
            log_with_timestamp(log_string_a)
    else:
        if verbose:
            log_string_a = "Data type unchanged."
            log_with_timestamp(log_string_a)

    if verbose:
        log_string_a = "File processing complete."
        log_with_timestamp(log_string_a)

    return data



def get_array_memory_usage_mb(array: np.ndarray) -> None:
    """
    Calculates and prints the memory consumption of a NumPy array in megabytes,
    along with the data type.

    Args:
        array: The NumPy array to analyze.
    """
    memory_usage_bytes = array.nbytes
    memory_usage_mb = memory_usage_bytes / (1024 * 1024)  # Convert bytes to megabytes
    data_type = array.dtype

    print(f"Data type: {data_type}")
    print(f"Memory usage: {memory_usage_mb:.2f} MB")  # Print to 2 decimal places

def crop_array_last_dimension(array: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Crops the last dimension of a multi-dimensional NumPy array to a specified number of samples.

    Args:
        array: A multi-dimensional NumPy array.
        num_samples: The desired number of samples in the last dimension.

    Returns:
        A new NumPy array with the last dimension cropped to num_samples.

    Raises:
        ValueError: If num_samples is greater than the size of the last dimension.
    """

    if num_samples > array.shape[-1]:
        raise ValueError("num_samples cannot be greater than the size of the last dimension.")

    return array[..., :num_samples]


def get_crop_index(input_array, threshold=CN.THRESHOLD_CROP, tail_ignore=0):
    """
    Returns the index in a 1D NumPy array where the amplitude last exceeds a given threshold,
    ignoring a specified number of samples at the end (up to half the array length).

    Args:
        input_array (numpy.ndarray): The 1D NumPy array representing the impulse response.
        threshold (float): The amplitude threshold to determine the crop point.
        tail_ignore (int): Number of samples from the end of the array to ignore in the search.

    Returns:
        int: The crop index (last index before `tail_ignore` where the signal exceeds the threshold).
             Returns 0 if no value exceeds the threshold in the considered range.
    """
    array_len = len(input_array)
    max_tail_ignore = array_len // 2
    tail_ignore = min(tail_ignore, max_tail_ignore)

    if tail_ignore >= array_len:
        return 0  # Redundant check, but safe

    # Work with the valid portion of the array
    valid_range = input_array[:array_len - tail_ignore]
    abs_array = np.abs(valid_range)
    
    # Find the last index above the threshold
    indices_above_threshold = np.where(abs_array > threshold)[0]

    if indices_above_threshold.size == 0:
        return 0

    return indices_above_threshold[-1]

def get_crop_index_old(input_array, threshold=CN.THRESHOLD_CROP):
    """
    Calculates the index at which a 1D NumPy array should be cropped 
    based on a given threshold.  The input array is not modified.

    Args:
        input_array (numpy.ndarray): The 1D NumPy array.
        threshold (float): The threshold value used for cropping.

    Returns:
        int: The index at which the array should be cropped.
             Returns 0 if no suitable crop index is found.
    """
    reversed_array = input_array[::-1]
    try:
        crop_index = len(reversed_array) - np.argmax(np.abs(reversed_array) > threshold) - 1
    except ValueError:
        # np.argmax returns ValueError if the array is all False
        # In this case, return 0 as no crop index found
        crop_index = 0
    return crop_index



def crop_array(input_array, threshold):
    """
    Crops a 1D NumPy array based on a given threshold.

    Args:
        input_array (numpy.ndarray): The 1D NumPy array to crop.
        threshold (float): The threshold value used for cropping.

    Returns:
        numpy.ndarray: The cropped array.
    """
    reversed_array = input_array[::-1]
    try:
        crop_index = len(reversed_array) - np.argmax(np.abs(reversed_array) > threshold) - 1
        cropped_array = input_array[:crop_index]
    except ValueError:
        # np.argmax returns ValueError if the array is all False
        # In this case, return the original array
        cropped_array = input_array
    return cropped_array



def log_with_timestamp(log_string, gui_logger=None, log_type=0, exception=None):
    """
    Logs a message with an optional GUI logger and prefixes GUI logs with a timestamp.

    Args:
        log_string (str): The message to log.
        gui_logger (object, optional): A GUI logger object. Defaults to None.
        log_type (int): 0 = info, 1 = warning, 2 = error.
        exception (Exception, optional): An exception object to log. Defaults to None.
    """

    if CN.LOG_INFO == True:
        if exception:
            if log_type == 2:
                logging.error(log_string, exc_info=exception)
            else:
                logging.info(log_string, exc_info=exception) # If it's a warning or info
        else:
            logging.info(log_string)

    if CN.LOG_GUI == True and gui_logger is not None:
        timestamp = datetime.datetime.now().strftime("%I:%M %p")
        gui_log_string = f"[{timestamp}] {log_string}"
        if log_type == 0:
            gui_logger.log_info(gui_log_string)
        elif log_type == 1:
            gui_logger.log_warning(gui_log_string)
        else:
            gui_logger.log_error(gui_log_string)


def zero_pad_last_dimension(data, n_fft):
    """
    Zero-pads the last dimension of a NumPy array to a specified length.

    Args:
        data: The input NumPy array.
        n_fft: The desired length of the last dimension.

    Returns:
        A new NumPy array with the last dimension zero-padded to n_fft, or the original array if padding wasn't necessary.
        Returns None if there's an error (e.g., n_fft is smaller than the current last dimension).
    """

    try:
        original_shape = data.shape
        last_dim = data.shape[-1]

        if last_dim > n_fft:
            print("Warning: n_fft is smaller than the current last dimension. Data will be truncated.")
            padded_data = data[..., :n_fft]
            return padded_data


        if last_dim == n_fft:  # No padding needed
            #print("No padding needed")
            return data

        padding_length = n_fft - last_dim
        padding_shape = list(data.shape[:-1]) + [padding_length]  # Shape of the padding
        padding = np.zeros(padding_shape, dtype=data.dtype) # Preserve the original dtype
        padded_data = np.concatenate((data, padding), axis=-1)

        return padded_data

    except Exception as e:
        log_string = f"An error occurred: {e}"
        if CN.LOG_INFO == True:
            logging.error(log_string, exc_info=e)
        return None

def zero_pad_1d(arr, target_length):
    """
    Zero-pads a 1D NumPy array to a specified length.

    Args:
        arr (np.ndarray): A 1D NumPy array.
        target_length (int): The desired length of the output array.

    Returns:
        np.ndarray: A zero-padded 1D NumPy array of shape (target_length,).
    """

    current_length = arr.shape[0]

    # If the array is already the desired length or longer, return the original or truncated version
    if current_length >= target_length:
        return arr[:target_length]

    # Create a zero-padded array
    padded_arr = np.zeros(target_length, dtype=arr.dtype)
    
    # Copy the original array into the padded array
    padded_arr[:current_length] = arr

    return padded_arr


def combine_dims_old(a, i=0, n=1):
    """
    Combines dimensions of numpy array `a`, 
    starting at index `i`,
    and combining `n` dimensions
    """
    s = list(a.shape)
    combined = functools.reduce(lambda x,y: x*y, s[i:i+n+1])
    return np.reshape(a, s[:i] + [combined] + s[i+n+1:])

def combine_dims(a, start=0, count=2):
    """ Reshapes numpy array a by combining count dimensions, 
        starting at dimension index start 
        s[:start]: Keeps the dimensions before start.
        (-1,): This tells NumPy to infer the new dimension size.
        s[start+count:]: Keeps the dimensions after the combined ones
        example:
            a = np.random.rand(2, 3, 4, 5)  # Shape: (2, 3, 4, 5)
            b = combine_dims(a, start=1, count=2)  # Merge dimensions 1 and 2
            print(a.shape)  # (2, 3, 4, 5)
            print(b.shape)  # (2, 12, 5)  -> 3 * 4 = 12
        """
    s = a.shape
    return np.reshape(a, s[:start] + (-1,) + s[start+count:])


def get_elevation_list(spatial_res=0):
    """ 
    Function returns list of elevations based on spatial resolution
    """
    try:
        if spatial_res == 0:
            elevation_list_sel= CN.ELEV_ANGLES_WAV_LOW
        elif spatial_res == 1:
            elevation_list_sel= CN.ELEV_ANGLES_WAV_MED
        elif spatial_res == 2:
            elevation_list_sel= CN.ELEV_ANGLES_WAV_HI
        elif spatial_res == 3:
            elevation_list_sel= CN.ELEV_ANGLES_WAV_MAX
        else:
            elevation_list_sel= CN.ELEV_ANGLES_WAV_LOW
    except:
        elevation_list_sel= CN.ELEV_ANGLES_WAV_LOW
        
    return elevation_list_sel

def print_message(message):
    """
    Function to print a message
    """
    current_time = datetime.datetime.now()
    print(current_time, ': ', message)


def round_to_multiple(number, multiple):
    """
    function to round to a multiple of a base
    """
    return multiple * round(number / multiple)

def round_down_even(n):
    return 2 * int(n // 2) 

def plot_data(mag_response, title_name = 'Output', n_fft = 65536, samp_freq = 44100, y_lim_adjust = 0, y_lim_a=-25, y_lim_b=15, x_lim_adjust = 0,x_lim_a=20, x_lim_b=20000, save_plot=0, plot_path=CN.DATA_DIR_OUTPUT, normalise=1, level_ends=0, plot_type=0):
    """
    Function takes a magnitude reponse array as an input and plots the reponse
    :param mag_response: numpy array 1d, magnitude reponse array
    :param title_name: string, name of plot
    :param n_fft: int, fft size
    :param samp_freq: int, sample frequency in Hz
    :param y_lim_adjust: int, 1 = adjust y axis to specified range, 0 = no adjustment 
    :param x_lim_adjust: int, 1 = adjust x axis to specified range, 0 = no adjustment 
    :param save_plot: int, 1 = save plot to file, 0 = dont save plot
    :param plot_path: string, path to save plot 
    :param normalise: int, 0 = dont normalise, 1 = normalise low frequencies to 0db
    :param plot_type: int, 0 = matplotlib, 1 = dearpygui (series 1 - filter export), 2 = dearpygui (series 2 - quick config), 3 = dearpygui (series 3 - lfa)
    :return: None
    """

    #level ends of spectrum
    if level_ends == 1:
        mag_response = level_spectrum_ends(mag_response, 200, 19000, n_fft=n_fft)#320, 19000 340, 19000
        #octave smoothing
        mag_response = smooth_fft_octaves(data=mag_response, n_fft=n_fft)
        
    nUniquePts = int(np.ceil((n_fft+1)/2.0))
    sampling_ratio = samp_freq/n_fft
    freqArray = np.arange(0, nUniquePts, 1.0) * sampling_ratio    
    
    #mag_response = mag_response / float(n_fft)
    mag_response = mag_response[0:nUniquePts]
    mag_response_log = 20*np.log10(mag_response)
    
    
    #normalise to 0db
    if normalise == 1:
        mag_response_log = mag_response_log-np.mean(mag_response_log[0:200])
    elif normalise == 2:
        mag_response_log = mag_response_log-np.mean(mag_response_log[CN.SPECT_SNAP_M_F0:CN.SPECT_SNAP_M_F1])#1200:1800

    if plot_type == 0:
        plt.figure()
        plt.plot(freqArray, mag_response_log, color='k', label="FR")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.xscale("log")
        plt.grid()
        if x_lim_adjust == 1:
            plt.xlim([x_lim_a, x_lim_b])
        else:
            plt.xlim([20, 20000])
        if y_lim_adjust == 1:
            plt.ylim([np.ceil(mag_response_log.max())-y_lim_a, np.ceil(mag_response_log.max())+y_lim_b])
        plt.title(title_name)
        plt.show()
        
        if save_plot == 1:
            out_file_name = title_name + '.png'
            out_file_path = pjoin(plot_path, out_file_name)
            plt.savefig(out_file_path)
    elif plot_type == 1:#filter and dataset tab
        dpg.set_value('series_tag', [freqArray, mag_response_log])
        dpg.set_item_label('series_tag', title_name)
        if y_lim_adjust == 1:
            dpg.set_axis_limits("y_axis", y_lim_a, y_lim_b)
        else:
            dpg.set_axis_limits("y_axis", -20, 15)
        if x_lim_adjust == 1:
            dpg.set_axis_limits("x_axis", x_lim_a, x_lim_b)
        else:
            dpg.set_axis_limits("x_axis", 10, 20000)
    elif plot_type == 2:#QC tab
        dpg.set_value('qc_series_tag', [freqArray, mag_response_log])
        dpg.set_item_label('qc_series_tag', title_name)
        if y_lim_adjust == 1:
            dpg.set_axis_limits("qc_y_axis", y_lim_a, y_lim_b)
        else:
            dpg.set_axis_limits("qc_y_axis", -20, 15)
        if x_lim_adjust == 1:
            dpg.set_axis_limits("qc_x_axis", x_lim_a, x_lim_b)
        else:
            dpg.set_axis_limits("qc_x_axis", 10, 20000)
    elif plot_type == 3:#LF analysis
        dpg.set_value('qc_lfa_series_tag', [freqArray, mag_response_log])
        dpg.set_item_label('qc_lfa_series_tag', title_name)
        if y_lim_adjust == 1:
            dpg.set_axis_limits("qc_lfa_y_axis", y_lim_a, y_lim_b)
        else:
            dpg.set_axis_limits("qc_lfa_y_axis", -20, 15)
        if x_lim_adjust == 1:
            dpg.set_axis_limits("qc_lfa_x_axis", x_lim_a, x_lim_b)
        else:
            dpg.set_axis_limits("qc_lfa_x_axis", 10, 20000)
            
            
def plot_data_generic(
    plot_data_array, 
    freqs=None,
    title_name='Output',
    data_type='magnitude',  # NEW: 'magnitude' or 'group_delay'
    n_fft=CN.N_FFT,
    samp_freq=44100,
    y_lim_adjust=0, y_lim_a=-25, y_lim_b=15,
    x_lim_adjust=0, x_lim_a=20, x_lim_b=20000,
    save_plot=0,
    plot_path=CN.DATA_DIR_OUTPUT,
    normalise=1,
    level_ends=0,
    plot_type=0
):
    """
    Generalised plotting function to support magnitude or group delay plots.

    :param plot_data_array: numpy array (1D), magnitude or group delay data
    :param data_type: str, 'magnitude' or 'group_delay'
    """
    

    if level_ends == 1 and data_type == 'magnitude':
        plot_data_array = level_spectrum_ends(plot_data_array, 200, 19000, n_fft=n_fft)
        plot_data_array = smooth_fft_octaves(data=plot_data_array, n_fft=n_fft)

    nUniquePts = int(np.ceil((n_fft + 1) / 2.0))
    sampling_ratio = samp_freq / n_fft
    if freqs is None:
        # Default frequency array from FFT
        freqArray = np.arange(0, nUniquePts, 1.0) * sampling_ratio
        plot_data_array = plot_data_array[0:nUniquePts]
    else:
        freqArray = freqs[0:nUniquePts-2]
        plot_data_array = plot_data_array[0:nUniquePts-2]

    if data_type == 'magnitude':
        plot_vals = 20 * np.log10(np.maximum(plot_data_array, 1e-12))  # avoid log(0)
        if normalise == 1:
            plot_vals -= np.mean(plot_vals[0:200])
        elif normalise == 2:
            plot_vals -= np.mean(plot_vals[CN.SPECT_SNAP_M_F0:CN.SPECT_SNAP_M_F1])
        y_label = "Magnitude (dB)"
    elif data_type == 'group_delay':
        plot_vals = plot_data_array  # assumed to be in ms already
        y_label = "Group Delay (ms)"
    else:
        raise ValueError("Unsupported data_type: use 'magnitude' or 'group_delay'")

    if plot_type == 0:  # Matplotlib
        plt.figure()
        plt.plot(freqArray, plot_vals, color='k', label=title_name)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(y_label)
        plt.xscale("log")
        plt.grid()
        if x_lim_adjust:
            plt.xlim([x_lim_a, x_lim_b])
        else:
            plt.xlim([20, 20000])
        if y_lim_adjust:
            plt.ylim([y_lim_a, y_lim_b])
        plt.title(title_name)
        plt.show()

        if save_plot:
            out_file_name = title_name + '.png'
            out_file_path = pjoin(plot_path, out_file_name)
            plt.savefig(out_file_path)

    else:
        tag_map = {
            1: 'series_tag',
            2: 'qc_series_tag',
            3: 'qc_lfa_series_tag'
        }
        x_axis_map = {
            1: 'x_axis',
            2: 'qc_x_axis',
            3: 'qc_lfa_x_axis'
        }
        y_axis_map = {
            1: 'y_axis',
            2: 'qc_y_axis',
            3: 'qc_lfa_y_axis'
        }

        series_tag = tag_map.get(plot_type)
        x_axis_tag = x_axis_map.get(plot_type)
        y_axis_tag = y_axis_map.get(plot_type)

        if series_tag:
            
            dpg.set_value(series_tag, [[], []])
            dpg.set_value(series_tag, [freqArray, plot_vals])
            dpg.set_item_label(series_tag, title_name)

            # Set axis limits
            if x_lim_adjust:
                dpg.set_axis_limits(x_axis_tag, x_lim_a, x_lim_b)
            else:
                dpg.set_axis_limits(x_axis_tag, 10, 20000)

            if y_lim_adjust:
                dpg.set_axis_limits(y_axis_tag, y_lim_a, y_lim_b)
            else:
                dpg.set_axis_limits(y_axis_tag, -20, 15)

            # Set y-axis label
            dpg.set_item_label(y_axis_tag, y_label)
            
    # print("Plot summary:")
    # print(f"Freq range: {freqArray.min()} Hz – {freqArray.max()} Hz")
    # print(f"Group delay range: {plot_vals.min()} ms – {plot_vals.max()} ms")
    # print(f"Data points: {len(freqArray)}")
            
    # if np.any(~np.isfinite(plot_vals)):
    #     print(f"[WARNING] plot_vals contains invalid entries (NaN or inf)")

    # if len(freqArray) != len(plot_vals):
    #     print(f"[WARNING] Frequency and value arrays are mismatched: {len(freqArray)} vs {len(plot_vals)}")


def plot_geq(geq_dict, title_name = 'Output', y_lim_adjust = 0, save_plot=0, plot_path=CN.DATA_DIR_OUTPUT):
    """
    Function takes a magnitude reponse dictionary as an input and plots the reponse
    :param geq_dict: dictionary, graphic eq dictionary with magnitude reponse
    :param title_name: string, name of plot
    :param y_lim_adjust: int, 1 = adjust y axis to 30db range, 0 = no adjustment 
    :param save_plot: int, 1 = save plot to file, 0 = dont save plot
    :param plot_path: string, path to save plot 
    :return: None
    """   
    
   
    # split dictionary into keys and values
    keys = list(geq_dict.keys())
    keys_float = [float(i) for i in keys]
    values = np.array(list(geq_dict.values()))
    
    plt.figure()
    plt.plot(keys_float, values, color='k', label="FR")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xscale("log")
    plt.grid()
    plt.xlim([20, 20000])
    if y_lim_adjust == 1:
        plt.ylim([np.ceil(values.max())-25, np.ceil(values.max())+5])
    plt.title(title_name)
    plt.show()
    
    if save_plot == 1:
        out_file_name = title_name + '.png'
        out_file_path = pjoin(plot_path, out_file_name)
        plt.savefig(out_file_path)
    

def plot_td(signal, title_name = 'Output', samp_freq = 44100):
    """
    Function takes a time domain signal as an input and plots the reponse
    :param signal: numpy array, time domain signal
    :param title_name: string, name of plot
    :param samp_freq: int, sample frequency in Hz
    :return: None
    """       

    Time = np.linspace(0, len(signal) / samp_freq, num=len(signal))

    plt.figure()
    plt.plot(Time, signal, color='k', label="amplitude")
    plt.xlabel("Time (S)")
    plt.ylabel("Amplitude")
    plt.grid()
    #plt.xlim([20, 20000])
    #plt.ylim([np.ceil(mag_response_log.max())-20, np.ceil(mag_response_log.max())+5])
    plt.title(title_name)
    plt.show()


def mag2db(mag):
    """Convert a magnitude to decibels (dB).

    If A is magnitude,

        db = 20 * log10(A)

    Parameters
    ----------
    mag : float or ndarray
        input magnitude or array of magnitudes

    Returns
    -------
    db : float or ndarray
        corresponding values in decibels

    Examples
    --------
    >>> ct.mag2db(10.0)                                         # doctest: +SKIP
    20.0

    >>> ct.mag2db(np.array([1, 0.01]))                          # doctest: +SKIP
    array([  0., -40.])

    """
    return 20. * np.log10(mag)

def db2mag(db):
    """Convert a gain in decibels (dB) to a magnitude.

    If A is magnitude,

        db = 20 * log10(A)

    Parameters
    ----------
    db : float or ndarray
        input value or array of values, given in decibels

    Returns
    -------
    mag : float or ndarray
        corresponding magnitudes

    Examples
    --------
    >>> ct.db2mag(-40.0)                                        # doctest: +SKIP
    0.01

    >>> ct.db2mag(np.array([0, -20]))                           # doctest: +SKIP
    array([1. , 0.1])

    """
    return 10. ** (db / 20.)



def list_diff(list1,list2):
    """
    function to get difference of 2 lists
    """
    
    result = []

    for i in range(len(list1)):
        result.append(list1[i] - list2[i])
        
    return result
  
# function to write wav file 
def write2wav(file_name, data, samplerate = 44100, prevent_clipping = 0, bit_depth='PCM_24'):
    """
    Function takes a time domain signal as an input and writes wav file 
    :param data: numpy array, time domain signal
    :param file_name: string, name of wav
    :param samplerate: int, sample frequency in Hz
    :param prevent_clipping: int, 1 = reduce amplitude to prevent clipping
    :return: None
    """    
    
    #old method using scipy
    #write("example.wav", samplerate, data.astype(np.int32))
    
    #adjust gain
    if prevent_clipping == 1:
        max_amp = np.max(np.abs(data))
        if max_amp > 1:
            data = data/max_amp
    
    #new method using PySoundFile 
    #soundfile expects data in frames x channels, or one-dimensional data for mono files. librosa does it the other way around.
    sf.write(file_name, data, samplerate, bit_depth)
    
    
# def read_wav_file(audiofilename):
#     """
#     function to open a wav file
#     """
    
#     samplerate, x = wavfile.read(audiofilename)  # x is a numpy array of integers, representing the samples 
#     # scale to -1.0 -- 1.0
#     if x.dtype == 'int16':
#         nb_bits = 16  # -> 16-bit wav files
#     elif x.dtype == 'int32':
#         nb_bits = 32  # -> 32-bit wav files
#     max_nb_bit = float(2 ** (nb_bits - 1))
#     samples = x / (max_nb_bit + 1)  # samples is a numpy array of floats representing the samples 
    
#     #print(x.dtype)
    
#     return samplerate, samples
    
def read_wav_file(audiofilename):
    """
    Opens a WAV file and returns the sample rate and normalized audio samples.

    Uses `soundfile` for better support of various bit depths and formats.
    Always returns float32 or float64 in the range [-1.0, 1.0].

    Parameters:
        audiofilename (str): Path to the WAV file.

    Returns:
        samplerate (int): Sampling rate of the audio file.
        samples (np.ndarray): Normalized audio data (float32/float64).
    """
    samples, samplerate = sf.read(audiofilename, always_2d=False)
    return samplerate, samples

    
def resample_signal(signal, original_rate = CN.SAMP_FREQ, new_rate = 48000, axis=0, scale=False):
    """
    function to resample a signal. By default will upsample from 44100Hz to 48000Hz
    """  
    
    #Resample data
    
    #V1.0 implementation uses scipy resample method which is low quality
    # number_of_samples = round(len(signal) * float(new_rate) / original_rate)
    # resampled_signal = sps.resample(signal, number_of_samples) 
    
    #new versions use librosa
    resampled_signal = librosa.resample(signal, orig_sr=original_rate, target_sr=new_rate, res_type='kaiser_best', axis=axis, scale=scale )
    
    
    return resampled_signal

    
def normalize_array(ir):
    """
    Normalizes an impulse response array to the range [-1, 1] based on its maximum absolute value.
    
    Parameters:
        ir (np.ndarray): Input impulse response array to normalize.
        
    Returns:
        np.ndarray: Normalized impulse response array.
    """
    max_val = np.max(np.abs(ir))
    if max_val > 0:
        ir = ir / max_val
    return ir       


def resample_by_interpolation(signal, input_fs = 44100, output_fs = 48000):
    """
    function to resample a signal. By default will upsample from 44100Hz to 48000Hz
    does not contain a low-pass filter to prevent aliasing when downsampling (i.e. scale < 1).
    This function is derived from https://github.com/nwhitehead/swmixer/blob/master/swmixer.py, which was released under LGPL. 
    """  
    
    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal    




 # ==============================================================================
# Measure RT60
# sourced from: https://dsp.stackexchange.com/questions/86316/a-python-code-for-blind-estimation-of-rt60-from-recorded-audio
# ==============================================================================
def measure_rt60(h, fs=1, decay_db=60, plot=False, rt60_tgt=None):
    """
    Analyze the RT60 of an impulse response. Optionaly plots some useful information.

    Parameters
    ----------
    `h`: array_like
        The impulse response.
    `fs`: float or int, optional
        The sampling frequency of h (default to 1, i.e., samples).
    `decay_db`: float or int, optional
        The decay in decibels for which we actually estimate the time. Although
        we would like to estimate the RT60, it might not be practical. Instead,
        we measure the RT20 or RT30 and extrapolate to RT60.
    `plot`: bool, optional
        If set to ``True``, the power decay and different estimated values will
        be plotted (default False).
    `rt60_tgt`: float
        This parameter can be used to indicate a target RT60 to which we want
        to compare the estimated value.
    """

    h = np.array(h)
    fs = float(fs)

    # The power of the impulse response in dB
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    # remove the possibly all zero tail
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]

    # -5 dB headroom
    i_5db = np.min(np.where(-5 - energy_db > 0)[0])
    e_5db = energy_db[i_5db]
    t_5db = i_5db / fs

    # after decay
    i_decay = np.min(np.where(-5 - decay_db - energy_db > 0)[0])
    t_decay = i_decay / fs

    # compute the decay time
    decay_time = t_decay - t_5db
    est_rt60 = (60 / decay_db) * decay_time

    if plot:
        import matplotlib.pyplot as plt

        # Remove clip power below to minimum energy (for plotting purpose mostly)
        energy_min = energy[-1]
        energy_db_min = energy_db[-1]
        power[power < energy[-1]] = energy_min
        power_db = 10 * np.log10(power)
        power_db -= np.max(power_db)

        # time vector
        def get_time(x, fs):
            return np.arange(x.shape[0]) / fs - i_5db / fs

        T = get_time(power_db, fs)

        # plot power and energy
        plt.plot(get_time(energy_db, fs), energy_db, label="Energy")

        # now the linear fit
        plt.plot([0, est_rt60], [e_5db, -65], "--", label="Linear Fit")
        plt.plot(T, np.ones_like(T) * -60, "--", label="-60 dB")
        plt.vlines(
            est_rt60, energy_db_min, 0, linestyles="dashed", label="Estimated RT60"
        )

        if rt60_tgt is not None:
            plt.vlines(rt60_tgt, energy_db_min, 0, label="Target RT60")

        plt.legend()

        plt.show()

    return est_rt60


def calculate_rt60(ir: np.ndarray, fs=CN.FS, bands=CN.OCTAVE_BANDS):
    """
    Calculate the average RT60 for given impulse response in 1/3 octave bands.
    
    Parameters:
        ir (np.ndarray): The impulse response (1D array)
        fs (int): The sampling frequency in Hz
        bands (np.ndarray): Array of center frequencies for the octave bands
    
    Returns:
        float: The average RT60 across the octave bands
    """
    rt60_values = []

    # For each octave band, calculate the RT60
    for band in bands:
        # Design the bandpass filter for the current band
        low_cut = band / (2**(1/6))  # Lower bound for 1/3 octave
        high_cut = band * (2**(1/6))  # Upper bound for 1/3 octave

        # Bandpass filter the impulse response
        nyquist = fs / 2
        low = low_cut / nyquist
        high = high_cut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_ir = signal.filtfilt(b, a, ir)

        # Calculate the energy decay curve (EDC) for the filtered signal
        squared_ir = filtered_ir ** 2
        edc = np.cumsum(squared_ir[::-1])[::-1]  # Integrate from the end

        # Find the decay time (RT60)
        initial_level = np.max(edc)
        final_level = initial_level / 1000  # 60 dB down
        crossing_index = np.where(edc <= final_level)[0][0]
        
        # Estimate RT60 by linear regression on the decay portion of the EDC
        time = np.arange(len(edc)) / fs
        rt60 = time[crossing_index]  # This is a simplified estimate
        
        rt60_values.append(rt60)

    # Return the average RT60 over all bands
    return np.mean(rt60_values)

# Example usage:
# ir = np.array([...])  # Your impulse response
# fs = 44100  # Sample rate of your IR
# average_rt60 = calculate_rt60(ir, fs)
# print("Average RT60 across 1/3-octave bands:", average_rt60)


def bandpass_filter(ir, fs, center_freq, fraction=3, order=4):
    """
    Band-pass filters the input IR to isolate a specific 1/3-octave band.
    
    Parameters:
        ir (np.array): Input impulse response.
        fs (int): Sample rate in Hz.
        center_freq (float): Center frequency of the band.
        fraction (int): Defines the fraction of the octave (default is 3 for 1/3-octave).
        order (int): Order of the Butterworth filter.
    
    Returns:
        np.array: Filtered impulse response.
    """
    factor = 2 ** (1 / (2 * fraction))  # Band edges scale factor
    low = center_freq / factor
    high = center_freq * factor
    sos = signal.butter(order, [low, high], btype='bandpass', fs=fs, output='sos')
    return signal.sosfilt(sos, ir)

def calculate_schroeder_curve(ir):
    """
    Calculates the Schroeder integration curve in dB.
    
    Parameters:
        ir (np.array): Filtered impulse response.
    
    Returns:
        np.array: Decay curve in dB (Schroeder curve).
    """
    energy = np.cumsum(ir[::-1] ** 2)[::-1]  # Reverse cumulative sum of energy
    energy /= np.max(energy)  # Normalize to peak = 1
    return 10 * np.log10(np.maximum(energy, 1e-12))  # Convert to dB (avoid log(0))

def linear_fit_decay(sch_db, fs, start_db, end_db):
    """
    Performs a linear regression on the Schroeder curve between two dB values
    to estimate RT60 via extrapolation.
    
    Parameters:
        sch_db (np.array): Schroeder curve in dB.
        fs (int): Sample rate in Hz.
        start_db (float): Start level in dB (e.g., -5).
        end_db (float): End level in dB (e.g., -25 or -35).
    
    Returns:
        float or None: Estimated RT60 (T20, T30) in seconds, or None if invalid.
    """
    start_idx = np.argmax(sch_db <= start_db)
    end_idx = np.argmax(sch_db <= end_db)
    if end_idx <= start_idx:
        return None  # Invalid range
    t = np.arange(end_idx - start_idx) / fs
    y = sch_db[start_idx:end_idx]
    slope, intercept, *_ = linregress(t, y)
    return -60 / slope if slope != 0 else None

def calculate_topt_from_schroeder(sch_db, fs):
    """
    Calculates Topt: the best linear fit over a variable decay range.
    Starts at -5 dB and searches for the best end point up to -45 dB.
    
    Parameters:
        sch_db (np.array): Schroeder curve in dB.
        fs (int): Sample rate in Hz.
    
    Returns:
        float or None: Estimated Topt in seconds.
    """
    best_fit_error = float("inf")
    best_rt = None
    start_idx = np.argmax(sch_db <= -5)
    
    for end_db in range(-10, -45, -1):
        end_idx = np.argmax(sch_db <= end_db)
        if end_idx <= start_idx:
            continue
        t = np.arange(end_idx - start_idx) / fs
        y = sch_db[start_idx:end_idx]
        slope, intercept, r_value, _, _ = linregress(t, y)
        residuals = np.sum((y - (slope * t + intercept)) ** 2)
        if residuals < best_fit_error and slope < 0:
            best_fit_error = residuals
            best_rt = -60 / slope  # RT60 from slope
    
    return best_rt

def compute_band_rt60s(ir, fs=CN.FS, bands=CN.OCTAVE_BANDS):
    """
    Calculates Topt-based RT60 estimates across all specified 1/3-octave bands.
    
    Parameters:
        ir (np.array): Raw impulse response.
        fs (int): Sample rate in Hz.
        bands (list): Center frequencies of bands.
    
    Returns:
        dict: Mapping of band center frequency to estimated RT60 (Topt).
    """
    results = {}
    for center_freq in bands:
        # Band-pass filter the IR at this frequency band
        filtered_ir = bandpass_filter(ir, fs, center_freq)
        
        # Compute Schroeder decay curve
        sch_db = calculate_schroeder_curve(filtered_ir)
        
        # Compute Topt
        rt = calculate_topt_from_schroeder(sch_db, fs)
        results[center_freq] = rt
    
    return results


#Example Usage
# # Load your impulse response as `ir` (e.g., from WAV) and define the sample rate `fs`
# band_rt60s = compute_band_rt60s(ir, fs)
# # Print estimated RT60s for each frequency band
# for band, rt in band_rt60s.items():
#     print(f"{band} Hz: Topt ≈ {rt:.2f} s" if rt is not None else f"{band} Hz: N/A")
# Calculate average Topt across all bands
# topt_values = band_rt60s["Topt"]
# topt_mean = np.nanmean(topt_values)
# print(f"Average Topt across bands: {topt_mean:.3f} seconds")


def signal_lowpass_filter(data, cutoff, fs, order=5, method=1, filtfilt=False):
    """
    Function takes a time domain signal as an input and applies low pass filter
    :param data: numpy array, time domain signal
    :param cutoff: int, cutoff frequency in Hz
    :param fs: int, sample frequency in Hz
    :param order: int, filter order
    :param method: int, 1=butter, 2=chevy1, 3=cheby2, 4=ellip, 5=bessel
    :return: numpy array, low pass filtered signal
    """   
    #Say I have a digital butterworth lowpass filter with an order of 3 and a cutoff of 4Hz at -3dB. this filter should have a roll-off of 6*3 = 18dB/Octave.
    #48db/octave = order of 8?
    if filtfilt == True:#doubles the effective order of the filtering when compared to a simple forward filter. 
        order=int(order/2)
    
    if method == 1:
        #method 1
        sos = butter(order, cutoff, fs=fs, btype='low', output='sos', analog=False)

    elif method == 2:
        #method 2
        sos = sps.cheby1(order, 3, cutoff, btype='low', analog=False, output='sos', fs=fs)
    elif method == 3:
        #method 3
        sos = sps.cheby2(order, 40, cutoff, btype='low', analog=False, output='sos', fs=fs)
    elif method == 4:
        #method 4
        sos = sps.ellip(order, 5, 40, cutoff, btype='low', analog=False, output='sos', fs=fs)
    else:
        #method 5
        sos = sps.bessel(order, cutoff, btype='low', analog=False, output='sos', fs=fs)
        
    #A forward-backward digital filter using cascaded second-order sections.
    if filtfilt == True:
        y = sps.sosfiltfilt(sos, data, padtype='even', padlen=30000) #Uses sosfiltfilt instead of sosfilt → Ensures zero-phase distortion and more stable filtering.
    else:
        y = sps.sosfilt(sos, data)
        
        
    return y


def signal_highpass_filter(data, cutoff, fs, order=5, method=1, filtfilt=False):
    """
    Function takes a time domain signal as an input and applies high pass filter
    :param data: numpy array, time domain signal
    :param cutoff: int, cutoff frequency in Hz
    :param fs: int, sample frequency in Hz
    :param order: int, filter order
    :param method: int, 1=butter, 2=chevy1, 3=cheby2, 4=ellip, 5=bessel
    :return: numpy array, high pass filtered signal
    """  
    if filtfilt == True:#doubles the effective order of the filtering when compared to a simple forward filter. 
        order=int(order/2)
    
    if method == 1:
        #method 1
        sos = butter(order, cutoff, fs=fs, btype='high', output='sos', analog=False)
        
    elif method == 2:
        #method 2
        sos = sps.cheby1(order, 3, cutoff, btype='high', analog=False, output='sos', fs=fs)
    elif method == 3:
        #method 3
        sos = sps.cheby2(order, 40, cutoff, btype='high', analog=False, output='sos', fs=fs)
    elif method == 4:
        #method 4
        sos = sps.ellip(order, 5, 40, cutoff, btype='high', analog=False, output='sos', fs=fs)
    else:
        #method 5
        sos = sps.bessel(order, cutoff, btype='high', analog=False, output='sos', fs=fs)
        
    #A forward-backward digital filter using cascaded second-order sections.
    if filtfilt == True:
        y = sps.sosfiltfilt(sos, data, padtype='even', padlen=30000) #Uses sosfiltfilt instead of sosfilt → Ensures zero-phase distortion and more stable filtering.
    else:
        y = sps.sosfilt(sos, data)
    
    return y

def get_filter_sos( cutoff, fs, order=5, method=1, filtfilt=False, b_type='high'):
    """
    Function takes a time domain signal as an input and calculates SOS filter
    :param cutoff: int, cutoff frequency in Hz
    :param fs: int, sample frequency in Hz
    :param order: int, filter order
    :param method: int, 1=butter, 2=chevy1, 3=cheby2, 4=ellip, 5=bessel
    :param b_type: str, 'low' or 'high'
    :return: sos filter object
    """  
    if filtfilt == True:#doubles the effective order of the filtering when compared to a simple forward filter. 
        order=int(order/2)

    if method == 1:
        #method 1
        sos = butter(order, cutoff, fs=fs, btype=b_type, output='sos', analog=False)
        
    elif method == 2:
        #method 2
        sos = sps.cheby1(order, 3, cutoff, btype=b_type, analog=False, output='sos', fs=fs)
    elif method == 3:
        #method 3
        sos = sps.cheby2(order, 40, cutoff, btype=b_type, analog=False, output='sos', fs=fs)
    elif method == 4:
        #method 4
        sos = sps.ellip(order, 5, 40, cutoff, btype=b_type, analog=False, output='sos', fs=fs)
    else:
        #method 5
        sos = sps.bessel(order, cutoff, btype=b_type, analog=False, output='sos', fs=fs)
        

    return sos

def get_multiple_filter_sos(
    cutoffs: list,
    fs: float,
    order: int,
    filtfilt: bool = False,
    b_type: str = 'low'
) -> list:
    """
    Create a list of second-order section (SOS) filter objects based on multiple cutoff frequencies.

    Parameters:
        cutoffs (list): List of cutoff frequencies.
        fs (float): Sampling frequency.
        order (int): Filter order.
        filtfilt (bool): Whether zero-phase filtering is used (optional).
        b_type (str): Filter type ('low', 'high', 'bandpass', etc.).

    Returns:
        list: List of SOS filter objects (one per cutoff).
    """
    return [
        get_filter_sos(
            cutoff=cutoff,
            fs=fs,
            order=order,
            filtfilt=filtfilt,
            b_type=b_type
        )
        for cutoff in cutoffs
    ]


def apply_sos_filter(data, sos, filtfilt=False, axis=-1):
    """
    Applies a high or low pass filter to a time-domain signal using second-order sections (SOS).
    
    Parameters:
        data (np.ndarray): Input signal.
        sos (np.ndarray): Second-order sections filter coefficients.
        filtfilt (bool): If True, applies zero-phase filtering with filtfilt.
        axis (int): Axis along which to apply the filter.

    Returns:
        np.ndarray: Filtered signal.
    """
    data = np.asarray(data)
    
    if filtfilt:
        # Ensure padlen does not exceed the length of the signal along the axis
        signal_length = data.shape[axis]
        padlen = min(30000, signal_length - 1) if signal_length > 1 else 0

        y = sps.sosfiltfilt(sos, data, padtype='even', padlen=padlen, axis=axis)
    else:
        y = sps.sosfilt(sos, data, axis=axis)
    
    return y


def group_delay(sig):
    """
    function to calculate group delay
    """  
    
    b = np.fft.fft(sig)
    n_sig = np.multiply(sig, np.arange(len(sig)))
    br = np.fft.fft(n_sig)
    return np.divide(br, b + 0.01).real


# def smooth_fft(data, crossover_f=1000, win_size_a = 150, win_size_b = 750, n_fft=CN.N_FFT, fs=CN.FS):
#     """
#     Function to perform smoothing of fft mag response
#     :param data: numpy array, magnitude response of a signal
#     :param crossover_f: int, crossover frequency in Hz. Below this freq a smoothing window of win_size_a will be applied and win_size_b above this freq
#     :param win_size_a: int, smoothing window size in Hz for lower frequencies
#     :param win_size_b: int, smoothing window size in Hz for higher frequencies
#     :param n_fft: int, fft size
#     :return: numpy array, smoothed signal
#     """  
    
#     crossover_fb= int(round(crossover_f*(n_fft/fs)))
#     win_size_a=int(round(win_size_a*(n_fft/fs)))
#     win_size_b=int(round(win_size_b*(n_fft/fs)))
#     win_size_c=min(win_size_a,win_size_b)
    
#     n_unique_pts = int(np.ceil((n_fft+1)/2.0))
#     nyq_freq = n_unique_pts-1
#     #apply win size a to low frequencies
#     data_smooth_a = sp.ndimage.uniform_filter1d(data,size=win_size_a)
#     data_smooth_b =np.zeros(n_fft)
#     data_smooth_b[0:crossover_fb] = data_smooth_a[0:crossover_fb]
#     #apply win size b to high frequencies
#     data_smooth_b[crossover_fb:n_unique_pts] = sp.ndimage.uniform_filter1d(data_smooth_a,size=win_size_b)[crossover_fb:n_unique_pts]
#     data_smooth_c = sp.ndimage.uniform_filter1d(data_smooth_b,size=win_size_c)#final pass
    
#     # Make symmetric (mirror below Nyquist into upper half)
#     data_smooth_c[nyq_freq+1:] = data_smooth_c[1:nyq_freq][::-1]
#     return data_smooth_c

# def smooth_fft_octaves(data, fund_freq=120, win_size_base = 15, n_fft=CN.N_FFT, fs=CN.FS):
#     """
#     Function to perform smoothing of fft mag response
#     :param data: numpy array, magnitude response of a signal
#     :param crossover_f: int, crossover frequency in Hz. Below this freq a smoothing window of win_size_a will be applied and win_size_b above this freq
#     :param win_size_a: int, smoothing window size in Hz for lower frequencies
#     :param win_size_b: int, smoothing window size in Hz for higher frequencies
#     :param n_fft: int, fft size
#     :return: numpy array, smoothed signal
#     """ 
    
#     n_unique_pts = int(np.ceil((n_fft+1)/2.0))
#     nyq_freq = n_unique_pts-1
    
#     max_freq = int(fs/2)
#     num_octaves = int(np.log2(max_freq/fund_freq))
    
#     for idx in range(num_octaves):
#         power = np.power(2,idx)
#         curr_cutoff_f = fund_freq*power
#         curr_win_s_a = win_size_base#win_size_base*power
#         curr_win_s_b = win_size_base*power#curr_win_s_a*2
        
#         data = smooth_fft(data, crossover_f=curr_cutoff_f, win_size_a = curr_win_s_a, win_size_b = curr_win_s_b, n_fft=n_fft, fs=fs)
    
#     data_smooth_c = data
    
#     # Make symmetric (mirror below Nyquist into upper half)
#     data_smooth_c[nyq_freq+1:] = data_smooth_c[1:nyq_freq][::-1]
#     return data_smooth_c



def smooth_fft(
    data,
    crossover_f=1000,
    win_size_a=150,
    win_size_b=750,
    n_fft=CN.N_FFT,
    fs=CN.FS,
    to_full=False
):
    """
    Applies two-stage smoothing to a magnitude FFT spectrum with different smoothing window
    sizes below and above a crossover frequency.

    Parameters:
        data (np.ndarray): Input magnitude spectrum (half or full).
        crossover_f (float): Frequency in Hz at which smoothing window changes.
        win_size_a (float): Smoothing window size in Hz for frequencies < crossover_f.
        win_size_b (float): Smoothing window size in Hz for frequencies >= crossover_f.
        n_fft (int): FFT size used to compute the original spectrum.
        fs (int): Sampling rate in Hz.
        to_full (bool): If True and input is half-spectrum, output a mirrored full-spectrum.

    Returns:
        np.ndarray: Smoothed spectrum (same type as input unless to_full=True).
    """
    is_half = len(data) == n_fft // 2 + 1
    nyq_bin = n_fft // 2

    # Ensure we are smoothing the positive frequency part only
    spectrum = data[:nyq_bin + 1] if not is_half else data.copy()

    # Convert smoothing windows and crossover to bins
    crossover_bin = int(round(crossover_f * n_fft / fs))
    win_a_bins = max(1, int(round(win_size_a * n_fft / fs)))
    win_b_bins = max(1, int(round(win_size_b * n_fft / fs)))
    win_c_bins = min(win_a_bins, win_b_bins)

    # Apply smoothing with different window sizes
    smooth_a = sp.ndimage.uniform_filter1d(spectrum, size=win_a_bins)
    smooth_b = sp.ndimage.uniform_filter1d(spectrum, size=win_b_bins)

    # Combine both smoothed results at the crossover point
    combined = np.empty_like(spectrum)
    combined[:crossover_bin] = smooth_a[:crossover_bin]
    combined[crossover_bin:] = smooth_b[crossover_bin:]

    # Final pass of light smoothing
    smoothed = sp.ndimage.uniform_filter1d(combined, size=win_c_bins)

    if not is_half:
        # Full input: re-insert the mirrored upper half
        result = np.empty_like(data)
        result[:nyq_bin + 1] = smoothed
        result[nyq_bin + 1:] = smoothed[1:nyq_bin][::-1]
        return result

    if to_full:
        # Generate full spectrum from half
        result = np.empty(n_fft, dtype=smoothed.dtype)
        result[:nyq_bin + 1] = smoothed
        result[nyq_bin + 1:] = smoothed[1:nyq_bin][::-1]
        return result

    return smoothed

def smooth_fft_octaves(
    data,
    fund_freq=120,
    win_size_base=15,
    n_fft=CN.N_FFT,
    fs=CN.FS,
    to_full=False
):
    """
    Applies multi-band smoothing based on octave scaling of smoothing windows.
    This function performs a sequence of smoothing operations, increasing the smoothing
    window size with frequency (per-octave) to simulate logarithmic perceptual resolution.

    Parameters:
        data (np.ndarray): Magnitude spectrum (either full or half spectrum).
        fund_freq (float): Base frequency in Hz that defines the start of octave scaling.
        win_size_base (float): Base smoothing window size in Hz at the lowest octave.
        n_fft (int): FFT size used to calculate the spectrum.
        fs (int): Sampling rate in Hz.
        to_full (bool): If True and input is half-spectrum, mirror to full spectrum on return.

    Returns:
        np.ndarray: Octave-smoothed spectrum (same type as input unless to_full=True).
    """
    # Determine input format
    is_half = len(data) == n_fft // 2 + 1
    nyq_bin = n_fft // 2
    max_freq = fs / 2
    num_octaves = int(np.log2(max_freq / fund_freq))

    # Work on a copy of the input
    smoothed = data.copy()

    # Apply octave-band smoothing iteratively
    for i in range(num_octaves):
        factor = 2 ** i
        cutoff = fund_freq * factor
        win_a = win_size_base
        win_b = win_size_base * factor

        smoothed = smooth_fft(
            smoothed,
            crossover_f=cutoff,
            win_size_a=win_a,
            win_size_b=win_b,
            n_fft=n_fft,
            fs=fs,
            to_full=False  # Always keep it half during iteration
        )

    # Return in desired format
    if not is_half:
        return smoothed  # Already full-spectrum, nothing more to do

    if to_full:
        # Mirror to create full-spectrum
        full_spec = np.empty(n_fft, dtype=smoothed.dtype)
        full_spec[:nyq_bin + 1] = smoothed
        full_spec[nyq_bin + 1:] = smoothed[1:nyq_bin][::-1]
        return full_spec
    else:
        return smoothed



def mag_to_min_fir(data, n_fft=CN.N_FFT, out_win_size=4096, crop=1):
    """
    Function to create min phase FIR from a fft magnitude response
    :param data: numpy array, magnitude response of a signal
    :param out_win_size: int, number of samples desired in output signal. Will crop signal
    :param n_fft: int, fft size
    :param crop: int, 0 = leave fir samples as per fft size, 1 = crop to out_win_size
    :return: numpy array, time domain signal
    """  
    
    n_unique_pts = int(np.ceil((n_fft+1)/2.0))

    data_pad_zeros=np.zeros(n_fft)
    data_pad_ones=np.ones(n_fft)

    #create min phase FIR 
    data_ifft = np.fft.irfft(data[0:n_unique_pts])#zero phase and symmetric
    data_lin = np.fft.ifftshift(data_ifft)#linear phase
    #filter will have a magnitude response that approximates the square root of the original filter’s magnitude response.
    data_min = sp.signal.minimum_phase(data_lin[1:n_fft], 'homomorphic', n_fft)
    data_min_pad = data_pad_zeros.copy()
    #return to original mag response
    data_min_conv = sp.signal.convolve(data_min,data_min, 'full', 'direct')
    data_min_pad[0:n_fft-1] = data_min_conv[:]
    
    #apply window to result

    #fade out window
    fade_hanning_size=out_win_size*2
    fade_hanning_start=100#50
    hann_fade_full=np.hanning(fade_hanning_size)
    hann_fade = np.split(hann_fade_full,2)[1]
    fade_out_win = data_pad_ones.copy()
    fade_out_win[fade_hanning_start:fade_hanning_start+int(fade_hanning_size/2)] = hann_fade
    fade_out_win[fade_hanning_start+int(fade_hanning_size/2):]=data_pad_zeros[fade_hanning_start+int(fade_hanning_size/2):]

    #return result
    #result with original length and windowed
    data_out = np.multiply(data_min_pad[0:n_fft-1],fade_out_win[0:n_fft-1])
    
    if crop == 1:
        return data_out[0:out_win_size]
    else:
        return data_out
    
# def build_min_phase_filter(smoothed_mag, fs=CN.FS, n_fft=CN.N_FFT, truncate_len=4096, f_min=20, f_max=20000, band_limit=False):
#     """
#     Build a minimum-phase FIR filter from a smoothed magnitude response,
#     band-limited between f_min and f_max, truncated to truncate_len samples,
#     and applying a fade-out window on the tail.

#     Parameters:
#     - smoothed_mag: array, magnitude response to convert
#     - fs: int, sampling frequency (Hz)
#     - n_fft: int, FFT size used for frequency domain
#     - truncate_len: int, length of time domain impulse response after truncation
#     - f_min: float, minimum frequency (Hz) for band-limiting compensation
#     - f_max: float, maximum frequency (Hz) for band-limiting compensation
#     - band_limit: bool, if True apply band-limiting to the magnitude response.

#     Returns:
#     - impulse: truncated minimum-phase impulse response with fade-out applied
#     """
#     freqs = np.fft.rfftfreq(n_fft, 1/fs)

#     # Create mask to band-limit the magnitude response within [f_min, f_max]
#     if band_limit:
#         # Apply band-limiting to magnitude response
#         band_mask = (freqs >= f_min) & (freqs <= f_max)
#         band_mag = np.zeros_like(smoothed_mag)
#         band_mag[band_mask] = smoothed_mag[band_mask]
#     else:
#         # Use full magnitude response
#         band_mag = smoothed_mag.copy()

#     # Log magnitude (avoid log(0))
#     log_mag = np.log(np.maximum(band_mag, 1e-8))

#     # Compute real cepstrum by inverse FFT of log magnitude
#     cepstrum = np.fft.irfft(log_mag, n=n_fft)

#     # Enforce minimum phase symmetry by doubling cepstral coefficients (except 0th)
#     cepstrum[1:n_fft//2] *= 2
#     cepstrum[n_fft//2+1:] = 0  # zero out upper half for causality

#     # Reconstruct minimum phase spectrum by FFT of cepstrum, then exponentiate
#     min_phase_spec = np.exp(np.fft.rfft(cepstrum, n=n_fft))

#     # Transform back to time domain: impulse response of minimum phase filter
#     impulse = np.fft.irfft(min_phase_spec, n=n_fft)

#     # Truncate impulse response to desired length
#     impulse = impulse[:truncate_len]

#     # Apply fade-out window to tail: smoothly taper from 1 to 0
#     fade_out = np.hanning(2 * truncate_len)[truncate_len:]
#     impulse *= fade_out

#     return impulse    
    
def build_min_phase_filter(
    smoothed_mag,
    fs=CN.FS,
    n_fft=CN.N_FFT,
    truncate_len=4096,
    f_min=20,
    f_max=20000,
    band_limit=False
):
    """
    Build a minimum-phase FIR filter from a magnitude spectrum (half or full).
    
    Parameters:
        smoothed_mag (np.ndarray): Magnitude spectrum (half-spectrum or full-spectrum).
        fs (int): Sampling frequency in Hz.
        n_fft (int): FFT size.
        truncate_len (int): Length to truncate time-domain IR.
        f_min (float): Minimum band-limit frequency.
        f_max (float): Maximum band-limit frequency.
        band_limit (bool): Whether to band-limit the magnitude before conversion.

    Returns:
        np.ndarray: Minimum-phase FIR filter (real impulse response).
    """
    # Ensure half-spectrum
    is_half = len(smoothed_mag) == n_fft // 2 + 1
    if not is_half:
        smoothed_mag = np.abs(smoothed_mag[:n_fft // 2 + 1])

    freqs = np.fft.rfftfreq(n_fft, 1 / fs)

    # Band-limit magnitude if requested
    if band_limit:
        band_mask = (freqs >= f_min) & (freqs <= f_max)
        mag = np.zeros_like(smoothed_mag)
        mag[band_mask] = smoothed_mag[band_mask]
    else:
        mag = smoothed_mag.copy()

    # Avoid log(0)
    log_mag = np.log(np.maximum(mag, 1e-8))

    # Real cepstrum
    cepstrum = np.fft.irfft(log_mag, n=n_fft)

    # Enforce minimum-phase symmetry (real cepstrum trick)
    cepstrum[1:n_fft // 2] *= 2
    cepstrum[n_fft // 2 + 1:] = 0

    # Rebuild min-phase spectrum
    min_phase_spec = np.exp(np.fft.rfft(cepstrum, n=n_fft))

    # Time-domain impulse response
    impulse = np.fft.irfft(min_phase_spec, n=n_fft)

    # Truncate and apply fade-out window
    impulse = impulse[:truncate_len]
    fade_out = np.hanning(2 * truncate_len)[truncate_len:]
    impulse *= fade_out

    return impulse


    
#modify spectrum to have flat mag response at low and high ends
# def level_spectrum_ends(data, low_freq=20, high_freq=20000, n_fft=CN.N_FFT, fs=CN.FS, smooth_win=67):
#     """
#     Function to modify spectrum to have flat mag response at low and high ends (efficient version)
#     :param data: numpy array, magnitude response of a signal (length n_fft)
#     :param low_freq: int, frequency in Hz below which will become flat
#     :param high_freq: int, frequency in Hz above which will become flat
#     :param n_fft: int, fft size
#     :param fs: int, sample frequency in Hz
#     :param smooth_win: int, smoothing window size in Hz to be applied after leveling ends
#     :return: numpy array, spectrum with smooth ends (length n_fft)
#     """
#     smooth_win_samples = int(round(smooth_win * (n_fft / fs)))

#     data_mod = data.copy()
#     low_freq_bin = int(low_freq * n_fft / fs)
#     high_freq_bin = int(high_freq * n_fft / fs)

#     # Level the low and high ends using array slicing
#     if low_freq_bin > 0:
#         data_mod[:low_freq_bin] = data[low_freq_bin]
#     if high_freq_bin < n_fft:
#         data_mod[high_freq_bin:] = data[high_freq_bin -1] # Use the value at the boundary

#     # Apply slight smoothing
#     if smooth_win_samples > 0:
#         data_smooth = sp.ndimage.uniform_filter1d(data_mod, size=smooth_win_samples)
#     else:
#         data_smooth = data_mod

#     # Make conjugate symmetric (assuming the input 'data' represents the positive frequency spectrum)
#     n_unique_pts = int(np.ceil((n_fft + 1) / 2.0))
#     if len(data_smooth) == n_fft:
#         positive_spectrum = data_smooth[:n_unique_pts].copy()
#         negative_spectrum = positive_spectrum[1:-1][::-1]  # Reverse and exclude DC and Nyquist
#         data_smooth = np.concatenate((positive_spectrum, negative_spectrum))
#     elif len(data_smooth) == n_unique_pts -1: # handle case where input was only positive spectrum
#         positive_spectrum = data_smooth.copy()
#         negative_spectrum = positive_spectrum[1:][::-1]
#         data_smooth = np.concatenate((positive_spectrum, negative_spectrum))
#     elif len(data_smooth) == n_unique_pts:
#         positive_spectrum = data_smooth[:-1].copy()
#         negative_spectrum = positive_spectrum[1:][::-1]
#         data_smooth = np.concatenate((positive_spectrum, negative_spectrum))


#     return data_smooth

def level_spectrum_ends(
    data,
    low_freq=20,
    high_freq=20000,
    n_fft=CN.N_FFT,
    fs=CN.FS,
    smooth_win=67,
    to_full=False
):
    """
    Modify a magnitude spectrum to flatten low and high frequency ends with smoothing.
    
    Supports both rfft (half) and fft (full) spectra.
    
    Parameters:
        data (np.ndarray): Magnitude spectrum (length n_fft or n_fft//2 + 1)
        low_freq (float): Frequency below which to flatten (Hz)
        high_freq (float): Frequency above which to flatten (Hz)
        n_fft (int): FFT size that produced the spectrum
        fs (int): Sampling frequency (Hz)
        smooth_win (float): Smoothing window size (Hz)
        to_full (bool): If True and input is half spectrum, output will be full spectrum

    Returns:
        np.ndarray: Modified spectrum, same format as input unless to_full=True
    """
    is_half_spectrum = len(data) == n_fft // 2 + 1
    freq_res = fs / n_fft
    smooth_win_samples = max(1, int(round(smooth_win / freq_res)))

    # Define frequency bin bounds
    low_bin = int(low_freq / freq_res)
    high_bin = int(high_freq / freq_res)
    high_bin = min(high_bin, len(data) - 1)
    low_bin = min(low_bin, high_bin)

    # Leveling
    data_mod = data.copy()
    if low_bin > 0:
        data_mod[:low_bin] = data[low_bin]
    if high_bin < len(data_mod):
        data_mod[high_bin:] = data[high_bin - 1]

    # Smoothing
    if smooth_win_samples > 1:
        data_mod = sp.ndimage.uniform_filter1d(data_mod, size=smooth_win_samples)

    if is_half_spectrum and to_full:
        # Convert to full (conjugate symmetric) real-valued spectrum
        full_spectrum = np.empty(n_fft, dtype=data_mod.dtype)
        full_spectrum[:n_fft // 2 + 1] = data_mod
        # Mirror the spectrum (exclude DC and Nyquist)
        full_spectrum[n_fft // 2 + 1:] = data_mod[1:n_fft // 2][::-1]
        return full_spectrum

    return data_mod



def padarray(A, size):
    """
    function to pad a numpy array with zeros to specified size
    """  
    #numpy.pad with constant mode, pass a tuple as second argument to tell how many zeros to pad on each size, a (2, 3) for instance will pad 2 zeros on the left side and 3 zeros on the right side:
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')
    

def get_close_matches_lower(word, possibilities, n=3, cutoff=0.45):
    """
    function to find closest matching string in a list of strings
    """  
    if not n >  0:
        raise ValueError("n must be > 0: %r" % (n,))
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("cutoff must be in [0.0, 1.0]: %r" % (cutoff,))
    result = []
    s = SequenceMatcher()
    s.set_seq2(word)
    for x in possibilities:
        s.set_seq1(x.lower())  # lower-case for comparison
        if s.real_quick_ratio() >= cutoff and \
           s.quick_ratio() >= cutoff and \
           s.ratio() >= cutoff:
            result.append((s.ratio(), x))

    # Move the best scorers to head of list
    result = _nlargest(n, result)
    # Strip scores for the best n matches
    return [x for score, x in result]

def get_close_matches_fuzz_simple(word, possibilities, n=2):
    """
    function to find closest matching string (word) in a list of strings (possibilities)
    """ 
    
    result = process.extract(word, possibilities, limit=n, scorer=fuzz.token_set_ratio)
    
    return result


def get_close_matches_fuzz(word, possibilities, n=2, score_cutoff=60):
    """
    Function to find closest matching string (word) in a list of strings (possibilities)
    with improved accuracy using different fuzzy matching techniques and score cutoff.

    Args:
        word (str): The string to find matches for.
        possibilities (list): A list of strings to search within.
        n (int): The number of closest matches to return.
        score_cutoff (int): Minimum score for a match to be considered.

    Returns:
        list: A list of tuples containing the best matches and their scores.
    """

    results = []

    # 1. Token Sort Ratio with partial ratio boost
    token_sort_results = process.extract(word, possibilities, limit=n, scorer=fuzz.token_sort_ratio)

    for match, score in token_sort_results:
        if score >= score_cutoff:
            results.append((match, score))

    # 2. Token Set Ratio (handles out-of-order words better)
    token_set_results = process.extract(word, possibilities, limit=n, scorer=fuzz.token_set_ratio)

    for match, score in token_set_results:
        if score >= score_cutoff and (match, score) not in results:
            results.append((match, score))

    # 3. Partial Ratio (finds substrings) if initial results are poor.
    if not results:
        partial_results = process.extract(word, possibilities, limit=n, scorer=fuzz.partial_ratio)
        for match, score in partial_results:
            if score >= score_cutoff:
                results.append((match, score))

    # 4. Weighted Ratio (combines different ratios)
    weighted_results = process.extract(word, possibilities, limit=n, scorer=fuzz.WRatio)
    for match, score in weighted_results:
        if score >= score_cutoff and (match, score) not in results:
            results.append((match, score))

    # Sort results by score (descending)
    results.sort(key=lambda x: x[1], reverse=True)

    # Return top n results (or fewer if there are less)
    return results[:n]


def find_delay_and_roll_lf(arr_a, arr_b):
    """
    function to find delay between arrays a and b and return number of elements to be shifted/rolled in array b for alignment
    returns an int containing number of samples to roll array b by to align with a
    """ 
    
    samp_freq=44100
    
    #contants for TD alignment of BRIRs
    t_shift_interval = CN.T_SHIFT_INTERVAL
    min_t_shift = CN.MIN_T_SHIFT_D
    max_t_shift = CN.MAX_T_SHIFT_D
    num_intervals = int(np.abs((max_t_shift-min_t_shift)/t_shift_interval))
    order=CN.ORDER#7
    delay_win_min_t = CN.DELAY_WIN_MIN_A
    delay_win_max_t = CN.DELAY_WIN_MAX_A
    delay_win_hop_size = CN.DELAY_WIN_HOP_SIZE
    delay_win_hops = CN.DELAY_WIN_HOPS_A
    
    
    cutoff_alignment = 110 #3.1.2 = 110Hz
    
    #peak to peak within a sufficiently small sample window
    peak_to_peak_window = int(np.divide(samp_freq,cutoff_alignment)*0.95) #int(np.divide(samp_freq,cutoff_alignment)) 
    
    delay_eval_set = np.zeros((num_intervals))
    
    prior_airs = arr_a 
    
    calc_delay = 1
        
    if calc_delay == 1:
        
        this_air = arr_b
     
        if np.sum(np.abs(this_air)) > 0:
            
            #low pass of prior airs
            prior_air_lp = signal_lowpass_filter(prior_airs, cutoff_alignment, samp_freq, order)
            #low pass of this ir
            this_air_lp = signal_lowpass_filter(this_air, cutoff_alignment, samp_freq, order)
            
            for delay in range(num_intervals):
                
                #shift current air
                current_shift = min_t_shift+(delay*t_shift_interval)
                n_air_shift = np.roll(this_air_lp,current_shift)
                #add prior air to shifted current air
                sum_ir_lp = np.add(prior_air_lp,n_air_shift)
                peak_to_peak_iter=0
                for hop_id in range(delay_win_hops):
                    samples = hop_id*delay_win_hop_size
                    peak_to_peak = np.abs(np.max(sum_ir_lp[delay_win_min_t+samples:delay_win_min_t+samples+peak_to_peak_window])-np.min(sum_ir_lp[delay_win_min_t+samples:delay_win_min_t+samples+peak_to_peak_window]))
                    #if this window has larger pk to pk, store in iter var
                    if peak_to_peak > peak_to_peak_iter:
                        peak_to_peak_iter = peak_to_peak
                #store largest pk to pk distance of all windows into delay set
                delay_eval_set[delay] = peak_to_peak_iter
            
            #shift next room by delay that has largest peak to peak distance (method 4 and 5)
            index_shift = np.argmax(delay_eval_set[:])
            samples_shift=min_t_shift+(index_shift*t_shift_interval)
        else:
            samples_shift=0
    else:
        samples_shift=0
     
    return samples_shift


        
    #
    #rolling code
    #
    # air_data[set_num,this_air_idx,0,:] = np.roll(air_data[set_num,this_air_idx,0,:],samples_shift)#left
    
    # #set end of array to zero to remove any data shifted to end of array
    # if samples_shift < 0:
    #     air_data[set_num,this_air_idx,0,min_t_shift:] = air_data[set_num,this_air_idx,0,min_t_shift:]*0#left
    
    
    
    
def roll_distribute_concatenate_npy_datasets_v1(input_dir):
    """
    Concatenates 4D NumPy arrays from .npy files in a folder, handling variable
    lengths in the first two dimensions, redistributes items if dimension 1
    length is less than 7, and rolls dimension 4 based on delay calculations
    using the sum of arrays in dimension 4.

    Args:
        input_dir: The directory containing the input .npy files.

    Returns:
        The concatenated NumPy array, or None if no valid files are found.
    """

    data_list = []
    previous_sum = None  # Store the sum of the previous arrays in dimension 4

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".npy"):
            filepath = os.path.join(input_dir, filename)
            try:
                data = np.load(filepath)
                if data.ndim != 4:
                    print(f"Warning: {filename} does not have 4 dimensions. Skipping.")
                    continue

                dim1, dim2, dim3, dim4 = data.shape

                # Calculate and apply delay
                current_sum = np.sum(data, axis=(0, 1, 2))  # Sum along dimensions 0, 1, and 2
                
                if previous_sum is not None:
                    roll_samples = find_delay_and_roll_lf(previous_sum, current_sum)
                    if CN.LOG_INFO == True and CN.SHOW_DEV_TOOLS == True:
                        logging.info('(AIR) samples_shift = ' + str(roll_samples))
                    data = np.roll(data, roll_samples, axis=3)  # Roll dimension 4

                previous_sum = current_sum.copy()  # Store the current sum

                if dim1 < 7:
                    new_dim1 = 7
                    new_dim2 = int(dim2 * dim1 / new_dim1)  # Redistribute items

                    if new_dim2 == 0:
                        print(f"Warning: {filename} redistribution would result in dimension 2 having length 0. Skipping.")
                        continue

                    reshaped_data = np.zeros((new_dim1, new_dim2, dim3, dim4), dtype=data.dtype)

                    # Iterate and fill the new array
                    orig_i = 0
                    orig_j = 0
                    for new_i in range(new_dim1):
                        for new_j in range(new_dim2):
                            if orig_i < dim1 and orig_j < dim2:
                                reshaped_data[new_i, new_j, :, :] = data[orig_i, orig_j, :, :]
                                orig_j += 1
                                if orig_j >= dim2:
                                    orig_j = 0
                                    orig_i += 1

                    data = reshaped_data

                data_list.append(data)

            except Exception as e:
                print(f"Error loading {filename}: {e}")

    if not data_list:
        print("No valid .npy files found in the input directory.")
        return None

    concatenated_data = data_list[0]
    for data in data_list[1:]:
        concatenated_data = np.concatenate((concatenated_data, data), axis=1)

    return concatenated_data
    
def roll_distribute_concatenate_npy_datasets(input_dir):
    """
    Concatenates 4D NumPy arrays from .npy files in a folder, handling variable
    lengths in the first two dimensions, redistributes items if dimension 1
    length is less than 7, and rolls dimension 4 based on delay calculations
    using the cumulative sum of arrays in dimension 4.

    Args:
        input_dir: The directory containing the input .npy files.

    Returns:
        The concatenated NumPy array, or None if no valid files are found.
    """

    data_list = []
    cumulative_sum = None  # Store the cumulative sum of arrays in dimension 4
    n_fft=CN.N_FFT

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".npy"):
            filepath = os.path.join(input_dir, filename)
            try:
                data = np.load(filepath)
                if data.ndim != 4:
                    print(f"Warning: {filename} does not have 4 dimensions. Skipping.")
                    continue

                dim1, dim2, dim3, dim4 = data.shape
                
                # Crop dimension 4 if it exceeds n_fft
                if dim4 > n_fft:
                    data = data[:, :, :, :n_fft]
                    print('Cropped to n_fft')
                    dim4 = n_fft

                # Calculate current sum
                current_sum = np.sum(data, axis=(0, 1, 2))  # Sum along dimensions 0, 1, and 2

                # Calculate and apply delay using cumulative sum
                if cumulative_sum is not None:
                    roll_samples = find_delay_and_roll_lf(cumulative_sum, current_sum)
                    if CN.LOG_INFO == True and CN.SHOW_DEV_TOOLS == True:
                        logging.info('(AIR) samples_shift = ' + str(roll_samples))
                    data = np.roll(data, roll_samples, axis=3)  # Roll dimension 4

                # Update cumulative sum
                if cumulative_sum is None:
                    cumulative_sum = current_sum.copy()
                else:
                    cumulative_sum = cumulative_sum + current_sum  # Accumulate the sum

                # Redistribution logic
                if dim1 < 7:
                    new_dim1 = 7
                    new_dim2 = int(dim2 * dim1 / new_dim1)  # Redistribute items

                    if new_dim2 == 0:
                        print(f"Warning: {filename} redistribution would result in dimension 2 having length 0. Skipping.")
                        continue

                    reshaped_data = np.zeros((new_dim1, new_dim2, dim3, dim4), dtype=data.dtype)

                    # Iterate and fill the new array
                    orig_i = 0
                    orig_j = 0
                    for new_i in range(new_dim1):
                        for new_j in range(new_dim2):
                            if orig_i < dim1 and orig_j < dim2:
                                reshaped_data[new_i, new_j, :, :] = data[orig_i, orig_j, :, :]
                                orig_j += 1
                            if orig_j >= dim2:
                                orig_j = 0
                                orig_i += 1

                    data = reshaped_data

                data_list.append(data)

            except Exception as e:
                print(f"Error loading {filename}: {e}")

    if not data_list:
        print("No valid .npy files found in the input directory.")
        return None

    concatenated_data = data_list[0]
    for data in data_list[1:]:
        concatenated_data = np.concatenate((concatenated_data, data), axis=1)

    return concatenated_data    
    

def summarize_array(arr):
    """
    Summarizes key information about the structure of a NumPy array.

    Args:
        arr: The NumPy array to summarize.

    Returns:
        A string summarizing the array's shape, dimensions, and data type.
    """

    if not isinstance(arr, np.ndarray):
        return "Input is not a NumPy array."

    summary = f"Array Summary:\n"
    summary += f"  Shape: {arr.shape}\n"
    summary += f"  Dimensions: {arr.ndim}\n"
    summary += f"  Data Type: {arr.dtype}\n"

    return summary    
    
    
def sanitize_filename(filename):
    """
    Sanitizes a filename by replacing invalid Windows characters with underscores.

    Args:
        filename (str): The filename to sanitize.

    Returns:
        str: The sanitized filename.
    """
    # Characters invalid in Windows filenames: \ / : * ? " < > |
    invalid_chars = r'[\\/:*?"<>|]'
    sanitized_filename = re.sub(invalid_chars, '_', filename)
    return sanitized_filename    
    
    

def update_gui_progress(report_progress, progress=None, message=''):
    """
    function to update progress bar in gui
    """ 
    overlay_text=''
    if report_progress > 0:
        if report_progress == 3:#AS tab
            prev_progress = dpg.get_value('progress_bar_as')
            if progress == None:
                progress=prev_progress#use previous progress if not specified
            dpg.set_value("progress_bar_as", progress)
        elif report_progress == 2:
            prev_progress = dpg.get_value('progress_bar_brir')
            if progress == None:
                progress=prev_progress#use previous progress if not specified
            if progress == 0:
                overlay_text = str(message)
            else:
                if message == '':
                    overlay_text = str(int(progress*100))+'%'
                else:
                    overlay_text = str(int(progress*100))+'% - '+message
            dpg.set_value("progress_bar_brir", progress)
            dpg.configure_item("progress_bar_brir", overlay = overlay_text)
        else:
            prev_progress = dpg.get_value('qc_progress_bar_brir')
            if progress == None:
                progress=prev_progress#use previous progress if not specified
            if progress == 0:
                overlay_text = str(message)
            else:
                if message == '':
                    overlay_text = str(int(progress*100))+'%'
                else:
                    overlay_text = str(int(progress*100))+'% - '+message
            dpg.set_value("qc_progress_bar_brir", progress)
            dpg.configure_item("qc_progress_bar_brir", overlay = overlay_text)
              
def check_stop_thread(gui_logger=None):
    """
    function to check if stop thread has been flagged and updates gui
    """ 
    
    #exit if stop thread flag is true
    stop_thread_1 = dpg.get_item_user_data("qc_progress_bar_brir")
    stop_thread_2 = dpg.get_item_user_data("progress_bar_brir")
    if stop_thread_1 == True or stop_thread_2 == True:
        log_string = 'BRIR Processing cancelled by user'
        log_with_timestamp(log_string, gui_logger)
        return True
    
    return False
              
   
                 
              
def check_and_download_file(file_path, gdrive_link, download=False, gui_logger=None):
    """
    Checks if the specified file exists, and if not, downloads it from the provided Google Drive link.
    
    Args:
        file_path (str): Full path to the file to check.
        gdrive_link (str): Direct Google Drive download link.
        download (bool): Whether to attempt downloading the file if missing.
        gui_logger (callable, optional): Logger for GUI output.

    Returns:
        int: 0 = Success, 1 = Failure
    """
    status = 1

    try:
        log_with_timestamp(f"Checking for file: {file_path}", gui_logger)

        if exists(file_path):
            log_with_timestamp("File already exists.", gui_logger)
        elif download:
            log_with_timestamp("File not found. Starting download...", gui_logger)
            download_file(gdrive_link, file_path, gui_logger=gui_logger)
            log_with_timestamp(f"File downloaded to: {file_path}", gui_logger)

        status = 0  # success

    except Exception as ex:
        log_with_timestamp("Failed to check or download file.", gui_logger, log_type=2, exception=ex)

    return status 

def download_file(url, save_location, gui_logger=None):
    """
    Downloads a file from a URL and saves it to the specified location.

    Args:
        url (str): The URL of the file to download.
        save_location (str): The full path (including filename) where the file should be saved.
        gui_logger:  A logger object for GUI logging (optional).
    """
    # Example Usage
    # url = "https://www.example.com/somefile.txt"  # Replace with the actual URL
    # save_path = "/path/to/save/myfile.txt"      # Replace with the desired save location
    # download_file(url, save_path)
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_location), exist_ok=True)

    # Check if the file already exists
    if os.path.exists(save_location):
        log_string = f"File already exists at: {save_location}"
        log_with_timestamp(log_string, gui_logger)
        return True  # Treat as success since the file is already present 

    try:
        
        if 'drive.google' in url:
            gdown.download(url, save_location, fuzzy=True)#google drive link
        else:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes
    
            with open(save_location, 'wb') as file:
                file.write(response.content)

        log_string = f"File downloaded successfully and saved to: {save_location}"
        log_with_timestamp(log_string, gui_logger)
        return True

    except requests.exceptions.RequestException as e:
        log_string = f"Error downloading file: {e}"
        log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=e)#log error
        return False
    
    except Exception as ex:
        log_string = "An unexpected error occurred"
        log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error  
        return False
    
    

def get_files_with_extension(directory, extension):
    """
    Returns a list of all files in a directory that have a specified extension,
    with the extension stripped out of the filenames.

    Args:
        directory (str): The path to the directory to search.
        extension (str): The file extension to filter by (e.g., "txt", "pdf").
                          It should NOT include a leading dot.

    Returns:
        list: A list of strings, where each string is the name of a file
              in the directory (without the extension) that ends with the 
              specified extension.
              Returns an empty list if no matching files are found 
              or if the directory is not valid.
    """

    file_list = [] # Corrected line

    if not os.path.isdir(directory):
        return file_list  # Or you might want to raise an exception here

    try:
        for filename in os.listdir(directory):
            if filename.lower().endswith("." + extension.lower()):
                #remove extension
                name_without_ext = os.path.splitext(filename)[0]
                file_list.append(name_without_ext)
    except Exception as e:
        print(f"An error occurred: {e}")
        return file_list # Or you might want to raise an exception here

    return file_list   


def load_csv_as_dicts(csv_dir, csv_name): 
    """
    Generic CSV loader that detects and converts column data types automatically.
    
    Parameters:
        csv_dir (str): Directory containing the CSV file.
        csv_name (str): Name of the CSV file.
    
    Returns:
        List[dict]: A list of dictionaries representing each row with inferred data types.
    """
    data_list = []
    filepath = pjoin(csv_dir, csv_name)

    try:
        with open(filepath, encoding='utf-8-sig', newline='') as inputfile:
            reader = csv.DictReader(inputfile)

            for row in reader:
                parsed_row = {}
                for key, value in row.items():
                    value = value.strip()

                    # Try to convert to int
                    try:
                        parsed_row[key] = int(value) if '.' not in value else float(value)
                    except ValueError:
                        # Try to convert to float
                        try:
                            float_value = float(value)
                            # Keep as float unless it's effectively an integer
                            if float_value.is_integer():
                                parsed_row[key] = int(float_value)  # Store as int if no fractional part
                            else:
                                parsed_row[key] = float_value  # Store as float
                        except ValueError:
                            # If neither int nor float, store as string
                            parsed_row[key] = value

                data_list.append(parsed_row)

    except Exception as e:
        print(f"Failed to load CSV '{csv_name}': {e}")

    return data_list

def extract_column(data, column, condition_key=None, condition_value=None):
    """
    Extracts a column from a list of dictionaries without auto-conversion.
    Optionally filters rows based on a condition.

    Parameters:
        data (list of dict): The metadata loaded from CSV.
        column (str): The key (column name) to extract values from.
        condition_key (str, optional): Key to filter rows.
        condition_value (any, optional): Value that condition_key must match.

    Returns:
        list: Values from the specified column.
    """
    if condition_key and condition_value is not None:
        return [row.get(column) for row in data if row.get(condition_key) == condition_value]
    
    return [row.get(column) for row in data]

def find_dict_by_value(data_list, key, value):
    """
    Search a list of dictionaries and return the first dictionary
    where the specified key matches the given value.

    Parameters:
        data_list (list): A list of dictionaries to search through.
        key (str): The key to look for in each dictionary.
        value (any): The value to match against.

    Returns:
        dict or None: The first matching dictionary, or None if not found.
    """
    for item in data_list:
        # Get the value associated with the key and compare it to the target value
        if item.get(key) == value:
            return item

    # Return None if no match was found
    return None

def get_value_from_matching_dict(data_list, match_key, match_value, return_key):
    """
    Search a list of dictionaries for the first dictionary where `match_key` equals `match_value`,
    and return the value associated with `return_key` in that dictionary.

    Parameters:
        data_list (list): A list of dictionaries to search through.
        match_key (str): The key to match on.
        match_value (any): The value to search for in `match_key`.
        return_key (str): The key whose value should be returned from the matched dictionary.

    Returns:
        any or None: The value corresponding to `return_key`, or None if not found or key missing.
    """
    for item in data_list:
        # Check if the current dictionary matches the given key-value pair
        if item.get(match_key) == match_value:
            # Return the value for the specified return_key (or None if it's missing)
            return item.get(return_key)

    # If no match is found, return None
    return None

def load_append_and_save_npy_matrices_recursive(
    directory: str,
    output_folder: str,
    output_filename: str = "combined_transform_matrix.npy",
    ignore_keyword: str = "combined"
) -> np.ndarray:
    """
    Recursively loads all .npy files from a directory and its subdirectories,
    assuming each file contains a 2D array of shape (measurements, samples),
    then appends them along the measurement axis and saves the result to a specified folder.

    Files whose names contain the 'ignore_keyword' are skipped during the search.

    Parameters:
        directory (str): Root directory to search for .npy files.
        output_folder (str): Directory where the combined output file should be saved.
        output_filename (str): Name of the output .npy file.
        ignore_keyword (str): Substring that, if found in a filename, will cause the file to be ignored.

    Returns:
        np.ndarray: A 2D array of stacked matrices.

    Raises:
        ValueError: If arrays are not 2D or their sample dimensions don't match.
    """
    appended = []
    sample_shape = None
    file_count = 0

    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            if fname.endswith(".npy") and ignore_keyword not in fname:
                path = os.path.join(root, fname)
                arr = np.load(path)

                if arr.ndim != 2:
                    raise ValueError(f"{path} is not a 2D array.")

                if sample_shape is None:
                    sample_shape = arr.shape[1]
                elif arr.shape[1] != sample_shape:
                    raise ValueError(
                        f"{path} has a different number of columns ({arr.shape[1]}) than expected ({sample_shape})."
                    )

                print(f"Loaded: {path} | shape: {arr.shape}")
                appended.append(arr)
                file_count += 1

    if not appended:
        raise ValueError("No valid .npy files were found or loaded.")

    combined = np.vstack(appended)
    
    #apply fading too
    #combined_faded = apply_fade_to_matrix(combined,fade_in_samples=150,fade_out_samples=150)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the combined matrix
    output_path = os.path.join(output_folder, output_filename)
    np.save(output_path, combined)

    print(f"\nTotal files loaded: {file_count}")
    print(f"Combined matrix shape: {combined.shape}")
    print(f"Saved combined matrix to: {output_path}")

    return combined

def apply_fade_to_matrix(
    matrix: np.ndarray,
    fade_in_samples: int = 100,
    fade_out_samples: int = 100
) -> np.ndarray:
    """
    Applies a linear fade-in and fade-out window to the second dimension (samples)
    of a 2D matrix with shape (measurements, samples).

    Parameters:
        matrix (np.ndarray): Input 2D matrix of shape (measurements, samples).
        fade_in_samples (int): Number of samples over which to apply the fade-in.
        fade_out_samples (int): Number of samples over which to apply the fade-out.

    Returns:
        np.ndarray: Matrix with fade-in and fade-out applied.
    
    Raises:
        ValueError: If the matrix is not 2D or fades exceed matrix length.
    """
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be 2D.")
    
    num_samples = matrix.shape[1]
    if fade_in_samples + fade_out_samples > num_samples:
        raise ValueError("Fade durations exceed the number of samples.")

    fade_in = np.linspace(0, 1, fade_in_samples)
    fade_out = np.linspace(1, 0, fade_out_samples)
    sustain = np.ones(num_samples - fade_in_samples - fade_out_samples)

    window = np.concatenate([fade_in, sustain, fade_out])
    faded_matrix = matrix * window  # Broadcasting applies fade to each measurement row

    return faded_matrix

def compute_rms(x: np.ndarray) -> float:
    """Compute the RMS of a 1D NumPy array using float64 precision."""
    x = np.asarray(x, dtype=np.float64)
    return np.sqrt(np.mean(np.square(x, dtype=np.float64), dtype=np.float64))

def balance_rms_fade(array: np.ndarray, region_size: int = 3000) -> np.ndarray:
    """
    Scales each measurement (row) so that the RMS of the last region matches the RMS of the first,
    using a linear gain ramp across the samples. All calculations are done in float64 to avoid
    precision loss.
    
    Parameters:
        array (np.ndarray): 2D array of shape (M, N) to apply RMS balancing to.
        region_size (int): Number of samples to use for start/end RMS regions.
    
    Returns:
        np.ndarray: RMS-balanced array of the same shape and dtype as input.
    """
    # Ensure working in float64 precision
    array = np.asarray(array, dtype=np.float64)
    M, N = array.shape
    x = np.linspace(0, 1, N, dtype=np.float64)

    balanced = np.empty_like(array)

    for i in range(M):
        row = array[i]
        start_rms = compute_rms(row[:region_size])
        end_rms = compute_rms(row[-region_size:])
        
        gain_ratio = start_rms / end_rms if end_rms != 0 else 1.0

        # Gain ramp: linear interpolation from 1 to gain_ratio
        gain = 1.0 + (gain_ratio - 1.0) * x
        balanced[i] = row * gain

    # Cast back to original dtype if needed
    if array.dtype != balanced.dtype:
        balanced = balanced.astype(array.dtype)

    return balanced

def crop_samples(arr: np.ndarray, crop_length: int) -> np.ndarray:
    """
    Returns a cropped version of the input 2D array with specified sample length.

    Parameters:
        arr (np.ndarray): 2D array of shape (measurements x samples).
        crop_length (int): Number of samples to retain per row.

    Returns:
        np.ndarray: Cropped 2D array.
    """
    if arr.ndim != 2:
        raise ValueError("Input array must be 2D.")
    if crop_length > arr.shape[1]:
        raise ValueError("crop_length exceeds available samples.")
    return arr[:, :crop_length]

def crop_measurements(arr: np.ndarray, max_measurements: int) -> np.ndarray:
    """
    Returns a cropped version of the input 2D array with a limited number of measurements (rows).

    Parameters:
        arr (np.ndarray): 2D array of shape (measurements x samples).
        max_measurements (int): Maximum number of rows to keep.

    Returns:
        np.ndarray: Cropped 2D array with at most `max_measurements` rows.
    """
    if arr.ndim != 2:
        raise ValueError("Input array must be 2D.")
    if max_measurements > arr.shape[0]:
        raise ValueError("max_measurements exceeds number of available measurements.")
    return arr[:max_measurements, :]

def exponential_moving_average(x, alpha):
    """Applies exponential moving average smoothing to a 1D array."""
    ema = np.zeros_like(x)
    ema[0] = x[0]
    for i in range(1, len(x)):
        ema[i] = alpha * x[i] + (1 - alpha) * ema[i - 1]
    return ema

def plot_sample_value_distribution(data: np.ndarray, bins: int = 100):
    """
    Plot the distribution of sample values across all measurements in a 2D array.
    Highlights the mean and ±1 standard deviation.

    Parameters:
        data (np.ndarray): 2D array with shape (measurements, samples)
        bins (int): Number of histogram bins

    Returns:
        None (displays the plot)
    """
    if data.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array (measurements x samples)")

    # Flatten the data to a 1D array
    flattened = data.flatten()
    mean = np.mean(flattened)
    std = np.std(flattened)

    # Plot histogram
    plt.figure(figsize=(10, 4))
    counts, bin_edges, _ = plt.hist(flattened, bins=bins, density=True, color='skyblue',
                                    edgecolor='black', alpha=0.9, label='Histogram')

    # Overlay mean and standard deviation lines
    plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean:.3f}')
    plt.axvline(mean - std, color='orange', linestyle='--', linewidth=1.5, label=f'±1 Std = {std:.3f}')
    plt.axvline(mean + std, color='orange', linestyle='--', linewidth=1.5)

    # Formatting
    plt.title("Distribution of Sample Values")
    plt.xlabel("Sample Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()