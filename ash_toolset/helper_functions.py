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
                loadsofa = sof.read_sofa(sofa_local_fname, verify=False)#verify=False ignores convention violations
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
    n_fft=65536,
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
    
    
def read_wav_file(audiofilename):
    """
    function to open a wav file
    """
    
    samplerate, x = wavfile.read(audiofilename)  # x is a numpy array of integers, representing the samples 
    # scale to -1.0 -- 1.0
    if x.dtype == 'int16':
        nb_bits = 16  # -> 16-bit wav files
    elif x.dtype == 'int32':
        nb_bits = 32  # -> 32-bit wav files
    max_nb_bit = float(2 ** (nb_bits - 1))
    samples = x / (max_nb_bit + 1)  # samples is a numpy array of floats representing the samples 
    
    #print(x.dtype)
    
    return samplerate, samples
    
    
def resample_signal(signal, original_rate = CN.SAMP_FREQ, new_rate = 48000, axis=0, scale=True):
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


def apply_sos_filter(data, sos, filtfilt=False):
    """
    Function takes a time domain signal as an input and applies high or low pass filter
    :param data: numpy array, time domain signal
    :param sos:  sos filter object
    :param filtfilt: bool, flag to use filtfilt
    :return: numpy array, x pass filtered signal
    """  

    #A forward-backward digital filter using cascaded second-order sections.
    if filtfilt == True:
        y = sps.sosfiltfilt(sos, data, padtype='even', padlen=30000) #Uses sosfiltfilt instead of sosfilt → Ensures zero-phase distortion and more stable filtering.
    else:
        y = sps.sosfilt(sos, data)
    
    return y


def group_delay(sig):
    """
    function to calculate group delay
    """  
    
    b = np.fft.fft(sig)
    n_sig = np.multiply(sig, np.arange(len(sig)))
    br = np.fft.fft(n_sig)
    return np.divide(br, b + 0.01).real


def smooth_fft(data, crossover_f=1000, win_size_a = 150, win_size_b = 750, n_fft=65536, fs=44100):
    """
    Function to perform smoothing of fft mag response
    :param data: numpy array, magnitude response of a signal
    :param crossover_f: int, crossover frequency in Hz. Below this freq a smoothing window of win_size_a will be applied and win_size_b above this freq
    :param win_size_a: int, smoothing window size in Hz for lower frequencies
    :param win_size_b: int, smoothing window size in Hz for higher frequencies
    :param n_fft: int, fft size
    :return: numpy array, smoothed signal
    """  
    
    crossover_fb= int(round(crossover_f*(n_fft/fs)))
    win_size_a=int(round(win_size_a*(n_fft/fs)))
    win_size_b=int(round(win_size_b*(n_fft/fs)))
    win_size_c=min(win_size_a,win_size_b)
    
    n_unique_pts = int(np.ceil((n_fft+1)/2.0))
    nyq_freq = n_unique_pts-1
    #apply win size a to low frequencies
    data_smooth_a = sp.ndimage.uniform_filter1d(data,size=win_size_a)
    data_smooth_b =np.zeros(n_fft)
    data_smooth_b[0:crossover_fb] = data_smooth_a[0:crossover_fb]
    #apply win size b to high frequencies
    data_smooth_b[crossover_fb:n_unique_pts] = sp.ndimage.uniform_filter1d(data_smooth_a,size=win_size_b)[crossover_fb:n_unique_pts]
    data_smooth_c = sp.ndimage.uniform_filter1d(data_smooth_b,size=win_size_c)#final pass
    
    #make conjugate symmetric
    for freq in range(n_fft):
        if freq>nyq_freq:
            dist_from_nyq = np.abs(freq-nyq_freq)
            data_smooth_c[freq]=data_smooth_c[nyq_freq-dist_from_nyq]
    
    return data_smooth_c

def smooth_fft_octaves(data, fund_freq=120, win_size_base = 15, n_fft=65536, fs=44100):
    """
    Function to perform smoothing of fft mag response
    :param data: numpy array, magnitude response of a signal
    :param crossover_f: int, crossover frequency in Hz. Below this freq a smoothing window of win_size_a will be applied and win_size_b above this freq
    :param win_size_a: int, smoothing window size in Hz for lower frequencies
    :param win_size_b: int, smoothing window size in Hz for higher frequencies
    :param n_fft: int, fft size
    :return: numpy array, smoothed signal
    """ 
    
    n_unique_pts = int(np.ceil((n_fft+1)/2.0))
    nyq_freq = n_unique_pts-1
    
    max_freq = int(fs/2)
    num_octaves = int(np.log2(max_freq/fund_freq))
    
    for idx in range(num_octaves):
        power = np.power(2,idx)
        curr_cutoff_f = fund_freq*power
        curr_win_s_a = win_size_base#win_size_base*power
        curr_win_s_b = win_size_base*power#curr_win_s_a*2
        
        data = smooth_fft(data, crossover_f=curr_cutoff_f, win_size_a = curr_win_s_a, win_size_b = curr_win_s_b, n_fft=n_fft, fs=fs)
    
    data_smooth_c = data
    
    #make conjugate symmetric
    for freq in range(n_fft):
        if freq>nyq_freq:
            dist_from_nyq = np.abs(freq-nyq_freq)
            data_smooth_c[freq]=data_smooth_c[nyq_freq-dist_from_nyq]
    
    return data_smooth_c


def mag_to_min_fir(data, n_fft=65536, out_win_size=4096, crop=0):
    """
    Function to create min phase FIR from a fft mag response
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
    
#modify spectrum to have flat mag response at low and high ends
def level_spectrum_ends_old(data, low_freq=20, high_freq=20000, n_fft=65536, fs=44100, smooth_win = 67):
    """
    Function to modify spectrum to have flat mag response at low and high ends
    :param data: numpy array, magnitude response of a signal
    :param low_freq: int, frequency in Hz below which will become flat
    :param high_freq: int, frequency in Hz above which will become flat
    :param n_fft: int, fft size
    :param fs: int, sample frequency in Hz
    :param smooth_win: int, smoothing window size in Hz to be applied after leveling ends
    :return: numpy array, spectrum with smooth ends
    """     
    smooth_win=int(round(smooth_win*(n_fft/fs)))
    
    n_unique_pts = int(np.ceil((n_fft+1)/2.0))
    nyq_freq = n_unique_pts-1
    
    data_mod = data.copy()
    low_freq_bin = int(low_freq*n_fft/fs)
    high_freq_bin = int(high_freq*n_fft/fs)
    data_mod[0:low_freq_bin] = data[low_freq_bin]
    data_mod[high_freq_bin:n_fft] = data[high_freq_bin]
    
    #apply slight smoothing
    if smooth_win > 0:
        data_smooth = sp.ndimage.uniform_filter1d(data_mod,size=smooth_win)
    else:
        data_smooth=data_mod
    
    #make conjugate symmetric
    for freq in range(n_fft):
        if freq>nyq_freq:
            dist_from_nyq = np.abs(freq-nyq_freq)
            data_smooth[freq]=data_smooth[nyq_freq-dist_from_nyq]
            
    return data_smooth

def level_spectrum_ends(data, low_freq=20, high_freq=20000, n_fft=65536, fs=44100, smooth_win=67):
    """
    Function to modify spectrum to have flat mag response at low and high ends (efficient version)
    :param data: numpy array, magnitude response of a signal (length n_fft)
    :param low_freq: int, frequency in Hz below which will become flat
    :param high_freq: int, frequency in Hz above which will become flat
    :param n_fft: int, fft size
    :param fs: int, sample frequency in Hz
    :param smooth_win: int, smoothing window size in Hz to be applied after leveling ends
    :return: numpy array, spectrum with smooth ends (length n_fft)
    """
    smooth_win_samples = int(round(smooth_win * (n_fft / fs)))

    data_mod = data.copy()
    low_freq_bin = int(low_freq * n_fft / fs)
    high_freq_bin = int(high_freq * n_fft / fs)

    # Level the low and high ends using array slicing
    if low_freq_bin > 0:
        data_mod[:low_freq_bin] = data[low_freq_bin]
    if high_freq_bin < n_fft:
        data_mod[high_freq_bin:] = data[high_freq_bin -1] # Use the value at the boundary

    # Apply slight smoothing
    if smooth_win_samples > 0:
        data_smooth = sp.ndimage.uniform_filter1d(data_mod, size=smooth_win_samples)
    else:
        data_smooth = data_mod

    # Make conjugate symmetric (assuming the input 'data' represents the positive frequency spectrum)
    n_unique_pts = int(np.ceil((n_fft + 1) / 2.0))
    if len(data_smooth) == n_fft:
        positive_spectrum = data_smooth[:n_unique_pts].copy()
        negative_spectrum = positive_spectrum[1:-1][::-1]  # Reverse and exclude DC and Nyquist
        data_smooth = np.concatenate((positive_spectrum, negative_spectrum))
    elif len(data_smooth) == n_unique_pts -1: # handle case where input was only positive spectrum
        positive_spectrum = data_smooth.copy()
        negative_spectrum = positive_spectrum[1:][::-1]
        data_smooth = np.concatenate((positive_spectrum, negative_spectrum))
    elif len(data_smooth) == n_unique_pts:
        positive_spectrum = data_smooth[:-1].copy()
        negative_spectrum = positive_spectrum[1:][::-1]
        data_smooth = np.concatenate((positive_spectrum, negative_spectrum))


    return data_smooth


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
        if report_progress == 2:
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