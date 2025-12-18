# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 14:05:45 2024

@author: Shanon
"""

import numpy as np
from os.path import join as pjoin
import os
from scipy.io import wavfile
from ash_toolset import constants as CN
from ash_toolset import helper_functions as hf
from ash_toolset import hrir_processing
from pathlib import Path
import mat73
import scipy as sp
import scipy.io as sio
import random
import logging
from SOFASonix import SOFAFile
from ash_toolset import pyquadfilter
from csv import DictReader
import gdown
import concurrent.futures
import noisereduce as nr
import h5py
import soundfile as sf
import sofar as sfr
import time
from scipy.io import loadmat
import csv
import glob
from typing import Any, Dict, Optional, Tuple, List, Callable
from scipy.io.matlab import mat_struct
import json
from datetime import datetime

logger = logging.getLogger(__name__)
log_info=1

    
  


def find_valid_ir_and_sr(obj, gui_logger=None):
    """
    Recursively search a MATLAB-loaded object for a valid IR array and sample rate.
    Prioritizes specific key names and ignores keys containing 'raw', 'meta', or 'info'.

    Args:
        obj (Any): The MATLAB-loaded object (typically a dict from loadmat).
        gui_logger (Optional[Callable[[str], None]], optional):
            Logger object for GUI output. Defaults to None.

    Returns:
        Tuple[np.ndarray, int, bool]:
            - fir_array (np.ndarray): The extracted impulse response array.
            - sample_rate (int): The extracted or fallback sample rate.
            - inferred_flag (bool): True if sample rate was inferred, False otherwise.

    Raises:
        ValueError: If no valid impulse response array is found after exhaustive search.
    """
    fallback_sr = 48000
    # Prioritize common and unambiguous IR keys
    preferred_ir_keys = ['ImpulseResponse', 'RIR', 'ir', 'Bformat', 'h', 'W', 'X', 'Y', 'Z'] # Added 'h' as a common variable name
    # Prioritize common sample rate keys
    preferred_sr_keys = ['fs', 'samplingrate', 'samplerate', 'sr'] # Added 'samplerate' and 'sr'
    # Keywords to ignore (case-insensitive) - often denote raw, unprocessed, or metadata
    ignore_keywords = ['raw', 'meta', 'info', 'calibration', 'config', 'header', 'comments']


    def is_valid_ir(arr) -> bool:
        """
        Checks if an array is a valid candidate for an impulse response.
        Must be a NumPy array, floating point, and at least 1-dimensional.
        We allow 1D here, as etl_airs handles reshaping later.
        """
        return isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating) and arr.ndim >= 1

    def is_valid_sr(val) -> bool:
        """
        Checks if a value is a valid sample rate.
        Must be convertible to float, and within a reasonable audio sample rate range.
        """
        try:
            sr_val = float(val)
            # Typical audio sample rates (e.g., 8kHz to 192kHz)
            return 1000 <= sr_val <= 200000
        except (ValueError, TypeError):
            return False

    def flatten_field(field) -> Any:
        """
        Recursively flattens single-element NumPy arrays, lists, or tuples.
        This helps extract the actual content from MATLAB's sometimes wrapped structures.
        """
        try:
            # Handle list/tuple containing single element
            if isinstance(field, (list, tuple)) and len(field) == 1:
                return flatten_field(field[0]) # Recurse on the single element

            # Handle single-element numpy arrays
            while isinstance(field, np.ndarray) and field.size == 1:
                field = field.item() # Extract the scalar value
            return field
        except Exception as e:
            hf.log_with_timestamp(f"[DEBUG] Failed to flatten field: {field} (type: {type(field)}). Error: {e}", gui_logger)
            return field # Return original if flattening fails

    # Use lists to store all found candidates, along with their preference status and key
    # This allows for a final prioritization step after searching the entire structure.
    found_irs: List[Tuple[np.ndarray, bool, str]] = [] # (ir_array, is_preferred_key, original_key)
    found_srs: List[Tuple[int, bool, str]] = []     # (sample_rate, is_preferred_key, original_key)

    # Stack for Depth-First Search (DFS) traversal: elements are (key, value) tuples
    stack: List[Tuple[Optional[str], Any]] = []

    # Initialize stack with the top-level object, handling its type
    if isinstance(obj, dict):
        for k, v in obj.items():
            stack.append((k, v))
        hf.log_with_timestamp(f"[DEBUG] Initial object is a dictionary with {len(obj)} items.", gui_logger)
    elif isinstance(obj, mat_struct):
        for k in obj._fieldnames:
            try:
                stack.append((k, getattr(obj, k)))
            except AttributeError:
                hf.log_with_timestamp(f"[WARNING] Top-level mat_struct field '{k}' not accessible, skipping.", gui_logger)
        hf.log_with_timestamp(f"[DEBUG] Initial object is a mat_struct with {len(obj._fieldnames)} fields.", gui_logger)
    else:
        hf.log_with_timestamp(f"[INFO] Top-level object is not a dict or mat_struct (type: {type(obj)}). Attempting to process as a direct value.", gui_logger)
        # If the top-level object itself is an IR or SR, add it as a candidate
        obj_flat = flatten_field(obj)
        if is_valid_ir(obj_flat):
            found_irs.append((obj_flat, False, "root_value"))
            hf.log_with_timestamp(f"[INFO] Found root object as potential IR with shape {obj_flat.shape}.", gui_logger)
        elif is_valid_sr(obj_flat):
            found_srs.append((int(float(obj_flat)), False, "root_value"))
            hf.log_with_timestamp(f"[INFO] Found root object as potential SR: {int(float(obj_flat))}.", gui_logger)

    hf.log_with_timestamp(f"[INFO] Starting recursive search for IR and Sample Rate.", gui_logger)

    while stack:
        key, val = stack.pop() # Pop from the stack for DFS
        #hf.log_with_timestamp(f"[DEBUG] Inspecting key: '{key}' (type: {type(val)}).", gui_logger)

        # Skip common MATLAB internal/metadata keys
        if key and (key.startswith('__') or key.startswith('_sa_') or key.startswith('__header__')):
            #hf.log_with_timestamp(f"[DEBUG] Skipping MATLAB internal key: '{key}'.", gui_logger)
            continue

        # Flatten the current value to expose underlying data
        val_flat = flatten_field(val)

        # Check for keywords to ignore in the key name (case-insensitive)
        if key is not None:
            key_lower = key.lower()
            if any(ignore_kw in key_lower for ignore_kw in ignore_keywords):
                hf.log_with_timestamp(f"[INFO] Ignoring value for key '{key}' due to ignore keyword match.", gui_logger)
                continue # Skip processing this branch further

            # Check if current value is a valid Sample Rate candidate
            if is_valid_sr(val_flat):
                sr_is_preferred = any(pk.lower() == key_lower for pk in preferred_sr_keys)
                found_srs.append((int(float(val_flat)), sr_is_preferred, key))
                hf.log_with_timestamp(f"[INFO] Found potential sample rate {int(float(val_flat))} for key '{key}' (Preferred: {sr_is_preferred}).", gui_logger)
            # else:
            #     hf.log_with_timestamp(f"[DEBUG] Value for key '{key}' is not a valid sample rate (value: {val_flat}).", gui_logger)

            # Check if current value is a valid Impulse Response candidate
            if is_valid_ir(val_flat):
                ir_is_preferred = any(pk.lower() == key_lower for pk in preferred_ir_keys)
                found_irs.append((val_flat, ir_is_preferred, key))
                hf.log_with_timestamp(f"[INFO] Found potential IR in key '{key}' with shape {val_flat.shape} (Preferred: {ir_is_preferred}).", gui_logger)
            # else:
            #     hf.log_with_timestamp(f"[DEBUG] Value for key '{key}' is not a valid IR type (type: {type(val_flat)}, ndim: {getattr(val_flat, 'ndim', 'N/A')}, dtype: {getattr(val_flat, 'dtype', 'N/A')}).", gui_logger)

        # Recurse into nested structures (dicts, mat_structs, structured arrays, lists/tuples of objects)
        if isinstance(val_flat, dict):
            #hf.log_with_timestamp(f"[DEBUG] Recursing into dictionary for key '{key}' with {len(val_flat)} items.", gui_logger)
            for k, v in val_flat.items():
                stack.append((k, v))
        elif isinstance(val_flat, mat_struct):
            #hf.log_with_timestamp(f"[DEBUG] Recursing into mat_struct for key '{key}' with {len(val_flat._fieldnames)} fields.", gui_logger)
            for k in val_flat._fieldnames:
                try:
                    stack.append((k, getattr(val_flat, k)))
                except AttributeError:
                    hf.log_with_timestamp(f"[WARNING] Mat_struct field '{k}' not accessible in '{key}', skipping.", gui_logger)
        elif isinstance(val_flat, np.ndarray) and val_flat.dtype.names: # Structured numpy array
            #hf.log_with_timestamp(f"[DEBUG] Recursing into structured numpy array for key '{key}' with fields: {val_flat.dtype.names}.", gui_logger)
            for name in val_flat.dtype.names:
                try:
                    # Flatten each structured array field before pushing to stack
                    stack.append((name, flatten_field(val_flat[name])))
                except Exception as e:
                    hf.log_with_timestamp(f"[WARNING] Failed to access structured array field '{name}' in '{key}': {e}", gui_logger)
        elif isinstance(val_flat, (list, tuple)) and len(val_flat) > 0:
            # Iterate through lists/tuples, pushing non-primitive elements to stack
            #hf.log_with_timestamp(f"[DEBUG] Recursing into list/tuple for key '{key}' with {len(val_flat)} elements.", gui_logger)
            for idx, item in enumerate(val_flat):
                # Only push if it's a compound type (dict, mat_struct, np.ndarray)
                # or if it's a list/tuple that could contain more structures
                if isinstance(item, (dict, mat_struct, np.ndarray)) or \
                   (isinstance(item, (list, tuple)) and len(item) > 0):
                    # Use a descriptive key for list/tuple elements if available, otherwise None
                    item_key = f"{key}[{idx}]" if key else f"list_item_{idx}"
                    stack.append((item_key, item))
        # For other types (scalars, non-structured arrays already checked by is_valid_ir, etc.), stop recursing.

    # --- Final selection logic based on all found candidates ---
    hf.log_with_timestamp(f"[INFO] Search complete. Found {len(found_irs)} potential IRs and {len(found_srs)} potential SRs.", gui_logger)

    # Select the best Impulse Response: Prioritize preferred keys, then the first one found
    final_fir_array: Optional[np.ndarray] = None

    # Attempt to construct B-format IR if all components are found
    b_format_keys = ['W', 'X', 'Y', 'Z']
    b_format_map = {k.upper(): None for k in b_format_keys}
    
    for arr, _, key in found_irs:
        if key and key.upper() in b_format_map and is_valid_ir(arr):
            b_format_map[key.upper()] = arr
    
    if all(b_format_map[k] is not None for k in b_format_keys):
        try:
            # Check all arrays have the same length
            lengths = [arr.shape[-1] for arr in b_format_map.values()]
            if len(set(lengths)) == 1:
                combined_b_format = np.stack([b_format_map[k] for k in b_format_keys], axis=0)
                final_fir_array = combined_b_format
                hf.log_with_timestamp(
                    f"[INFO] Constructed B-format IR from keys {b_format_keys} with shape {final_fir_array.shape}.",
                    gui_logger
                )
            else:
                hf.log_with_timestamp(
                    f"[WARNING] B-format IR components found, but they have mismatched lengths: {lengths}. Skipping B-format construction.",
                    gui_logger
                )
        except Exception as e:
            hf.log_with_timestamp(
                f"[ERROR] Failed to construct B-format IR: {e}",
                gui_logger
            )
    
    # If B-format not successfully constructed, fall back to best single IR
    if final_fir_array is None:
        if found_irs:
            found_irs.sort(key=lambda x: x[1], reverse=True)
            final_fir_array = found_irs[0][0]
            hf.log_with_timestamp(f"[INFO] Selected IR from key '{found_irs[0][2]}' with shape {final_fir_array.shape}.", gui_logger)
        else:
            error_msg = "No valid impulse response array found in the .mat file after exhaustive search."
            hf.log_with_timestamp(f"[ERROR] {error_msg}", gui_logger)
            raise ValueError(error_msg)

    # Select the best Sample Rate: Prioritize preferred keys, then the first one found
    final_sample_rate: int
    final_inferred_flag: bool

    if found_srs:
        found_srs.sort(key=lambda x: x[1], reverse=True)
        final_sample_rate = found_srs[0][0]
        final_inferred_flag = False
        hf.log_with_timestamp(f"[INFO] Selected sample rate {final_sample_rate} from key '{found_srs[0][2]}'.", gui_logger)
    else:
        final_sample_rate = fallback_sr
        final_inferred_flag = True
        hf.log_with_timestamp(f"[WARNING] Sample rate not found after search, falling back to default {fallback_sr}. (Inferred)", gui_logger)

    return final_fir_array, final_sample_rate, final_inferred_flag




def etl_airs(
    folder_path,
    file_type=None,
    gui_logger=None,
    project_samplerate=CN.FS,
    normalize=False,
    noise_reduction=False,
    noise_tail_ratio=0.2,
    max_measurements=CN.MAX_IRS,
    max_samples=CN.N_FFT,
    cancel_event=None
):
    """
    Loads and standardizes impulse responses from a folder (including subfolders) into a 2D array.
    """

    supported_exts = ['.wav', '.mat', '.npy', '.sofa', '.hdf5']

    all_files = [
        f for ext in supported_exts
        for f in glob.glob(os.path.join(folder_path, '**', f'*{ext}'), recursive=True)
    ]

    ir_list = []
    max_samples_in_data = 0
    total_measurements = 0
    status = 1
    air_data = np.array([])

    for path in all_files:
        ext = os.path.splitext(path)[1].lower()
        ftype = file_type if file_type else ext.strip('.')

        if cancel_event and cancel_event.is_set():
            gui_logger.log_warning("Operation cancelled by user.")
            return air_data, 2

        try:
            samplerate = project_samplerate  # default

            # --- MAT files ---
            if ftype == 'mat':
                data = loadmat(path, squeeze_me=True, struct_as_record=False)
                ir, samplerate, inferred_sr = find_valid_ir_and_sr(data, gui_logger=None)
                ir = hf.reshape_array_to_two_dims(ir)
                hf.log_with_timestamp(f"Loaded MAT file '{os.path.basename(path)}' with shape {ir.shape}, SR={samplerate}", gui_logger)
                if inferred_sr:
                    hf.log_with_timestamp(f"[Warning] Sample rate inferred as 48000 Hz for '{os.path.basename(path)}' — not found in file.", gui_logger, log_type=1)

            # --- WAV files ---
            elif ftype == 'wav':
                ir, samplerate = sf.read(path)
                ir = np.array(ir)
                if ir.ndim == 1:
                    ir = ir[np.newaxis, :]
                elif ir.ndim == 2:
                    if ir.shape[1] <= 64:
                        ir = ir.T
                hf.log_with_timestamp(f"Loaded WAV file '{os.path.basename(path)}' with shape {ir.shape}", gui_logger)

            # --- SOFA files ---
            elif ftype == 'sofa':
                loadsofa = hf.sofa_load_object(path, gui_logger)
                ir = loadsofa['sofa_data_ir']
                samplerate = loadsofa['sofa_samplerate']
                ir = hf.reshape_array_to_two_dims(ir)

            # --- HDF5 files ---
            elif ftype == 'hdf5':
                with h5py.File(path, mode='r') as rir_dset:
                    if 'rir' not in rir_dset:
                        raise ValueError(f"No valid IR dataset found in {path}")
                    ir = np.array(rir_dset['rir'])
                    samplerate = int(rir_dset.attrs.get('SamplingRate', project_samplerate))
                if ir.ndim == 1:
                    ir = ir[np.newaxis, :]
                elif ir.ndim == 2 and ir.shape[0] < ir.shape[1]:
                    ir = ir.T

            # --- NPY files ---
            elif ftype == 'npy':
                ir = np.load(path)
                ir = hf.reshape_array_to_two_dims(ir)
                samplerate = 48000  # ← Assume NPY files are at 48 kHz
                hf.log_with_timestamp(f"Loaded NPY file '{os.path.basename(path)}' with assumed SR=48000 Hz", gui_logger)

            else:
                raise ValueError(f"Unsupported file type: {ftype}")

            # --- Reshape multidimensional IRs ---
            if ir.ndim > 2:
                sample_axis = np.argmax(ir.shape)
                n_samples = ir.shape[sample_axis]
                other_dims = np.prod([s for i, s in enumerate(ir.shape) if i != sample_axis])
                ir = np.reshape(np.moveaxis(ir, sample_axis, -1), (other_dims, n_samples))
            elif ir.ndim == 1:
                ir = ir[np.newaxis, :]

            # --- Resample if needed ---
            if samplerate != project_samplerate:
                ir = hf.resample_signal(ir, original_rate=samplerate, new_rate=project_samplerate, axis=1)
                hf.log_with_timestamp(f"Resampled from {samplerate} Hz to {project_samplerate} Hz for {os.path.basename(path)}", gui_logger)

            # --- Optional noise reduction ---
            if noise_reduction:
                try:
                    tail_len = max(512, int(ir.shape[1] * noise_tail_ratio))
                    y_noise = np.copy(ir[:, -tail_len:])
                    y_clean = nr.reduce_noise(y=ir, sr=project_samplerate, y_noise=y_noise, stationary=True)
                    ir = np.reshape(y_clean, ir.shape)
                    hf.log_with_timestamp(f"Noise reduction applied to '{os.path.basename(path)}' (tail={tail_len})", gui_logger)
                except Exception as e:
                    hf.log_with_timestamp(f"[Error] Noise reduction failed on {path}: {e}", gui_logger)

            # --- Optional normalization ---
            if normalize:
                ir = hf.normalize_array(ir)

            hf.log_with_timestamp(f"Loaded '{os.path.basename(path)}' with shape {ir.shape}", gui_logger)

            # --- Limit checks ---
            max_samples_in_data = max(max_samples_in_data, ir.shape[1])
            total_measurements += ir.shape[0]

            if max_measurements and total_measurements > max_measurements:
                remaining = max_measurements - (total_measurements - ir.shape[0])
                ir = ir[:remaining]

            ir_list.append(ir)

            if max_measurements and total_measurements >= max_measurements:
                break

        except Exception as e:
            hf.log_with_timestamp(f"[Error] Failed to load {path}: {e}", gui_logger)
            continue

    if len(ir_list) == 0:
        hf.log_with_timestamp("No valid impulse responses found in the specified folder.", gui_logger, log_type=2)
        raise ValueError("No valid impulse responses found in the specified folder.")

    # --- Pad or truncate to consistent length ---
    padded_list = []
    for ir in ir_list:
        if ir.shape[1] > max_samples:
            ir = ir[:, :max_samples]
            hf.log_with_timestamp(f"Truncated IR to {max_samples} samples.", gui_logger)
        elif ir.shape[1] < max_samples:
            pad_width = max_samples - ir.shape[1]
            ir = np.pad(ir, ((0, 0), (0, pad_width)), mode='constant')
            hf.log_with_timestamp(f"Padded IR to {max_samples} samples.", gui_logger)
        padded_list.append(ir)

    hf.log_with_timestamp(f"Total IRs loaded: {len(ir_list)}, Max samples: {max_samples_in_data}", gui_logger)
    air_data = np.concatenate(padded_list, axis=0)
    hf.log_with_timestamp(f"Final air_data shape: {air_data.shape}", gui_logger)

    status = 0
    return air_data, status


def prepare_air_dataset(
    ir_set='default_set_name',
    input_folder=None,
    gui_logger=None,
    wav_export=CN.EXPORT_WAVS_DEFAULT,
    use_user_folder=False,
    save_npy=False,
    desired_measurements=3000, noise_reduction_mode=False,
    pitch_range=(0, 12),
    long_mode=False, report_progress=0, cancel_event=None, f_alignment = 0, pitch_shift_comp=True, ignore_ms=CN.IGNORE_MS, subwoofer_mode=False
):
    """
    Loads and processes individual IR files from a dataset folder, extracting the impulse responses
    into a structured NumPy array for later use. Optionally applies pitch shifting to expand the dataset,
    and saves the result in both .npy and .wav formats.

    The final output array shape is: (num_measurements, num_channels, fft_size)

    Args:
        ir_set (str): Name of the impulse response set. Must match a folder under
                      ASH-Toolset/data/raw/ir_data/raw_airs.
        gui_logger (callable or None): Optional logging function for GUI output.
        wav_export (bool): If True, exports the processed IRs as .wav files.
        use_user_folder (bool): If True, uses a user-defined folder for output instead of default location.
        save_npy (bool): If True, saves the processed IRs as a .npy file.
        desired_measurements (int): Target number of IR measurements in the final dataset.
        pitch_range (tuple): Tuple of (min_semitones, max_semitones) to apply pitch shifting.
        long_mode (bool): If True, assumes long IRs and adjusts trimming and padding behavior.

    Returns:
       Status code: 0 = Success, 1 = Failure, 2 = Cancelled
    """
    status=1
    
    #set fft length based on AC space
    if long_mode == True or ir_set in CN.AC_SPACE_LIST_HIRT60:
        n_fft=CN.N_FFT_L
    else:
        n_fft=CN.N_FFT

    if input_folder == None:
        input_folder=ir_set

    #set noise reduction based on AC space
    noise_reduction=noise_reduction_mode
        
    samp_freq_ash=CN.SAMP_FREQ
    air_data=np.array([])
    ir_min_threshold=CN.IR_MIN_THRESHOLD
    
    #windows
    data_pad_zeros=np.zeros(n_fft)
    data_pad_ones=np.ones(n_fft)
    #direct sound fade in window
    direct_hanning_size=200#
    direct_hanning_start=61#
    hann_direct_full=np.hanning(direct_hanning_size)
    hann_direct = np.split(hann_direct_full,2)[0]
    direct_removal_win = data_pad_zeros.copy()
    direct_removal_win[direct_hanning_start:direct_hanning_start+int(direct_hanning_size/2)] = hann_direct
    direct_removal_win[direct_hanning_start+int(direct_hanning_size/2):]=data_pad_ones[direct_hanning_start+int(direct_hanning_size/2):]

    
    #input folder
    if use_user_folder == True:
        ir_data_folder = pjoin(CN.DATA_DIR_IRS_USER,input_folder)
    else:
        ir_data_folder = pjoin(CN.DATA_DIR_RAW, 'ir_data', 'raw_airs',input_folder)
    #output folder
    if CN.RISE_WINDOW == True:  
        air_out_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'prepped_airs',ir_set)
    else:
        air_out_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'full_airs',ir_set)
        
    #loop through folders
    try:
        
        log_string_a = 'Loading IRs from: '+ir_set
        hf.log_with_timestamp(log_string_a, gui_logger)
        
        #run etl function to load IRs
        air_data,status_code= etl_airs(
            folder_path=ir_data_folder,
            file_type=None,
            gui_logger=gui_logger,
            project_samplerate=samp_freq_ash,
            normalize=True,
            noise_reduction=noise_reduction,
            noise_tail_ratio=0.2,
            max_measurements=CN.MAX_IRS,  # Limit on the number of measurements (rows)
            max_samples=n_fft,cancel_event=cancel_event  # Limit on the number of samples (columns)
        )
        if status_code > 0:#failure or cancelled
            return air_data, status_code
        
        if cancel_event and cancel_event.is_set():
            gui_logger.log_warning("Operation cancelled by user.")
            return air_data, 2
        
        total_measurements = air_data.shape[0]
        log_string_a = 'Total measurements before expansion: '+str(total_measurements)
        hf.log_with_timestamp(log_string_a, gui_logger)

        # Shift raw IRs so that direct peak is at sample x
        index_peak_ref = 30  # Target alignment point
        for idx in range(total_measurements):
            index_peak_cur = np.argmax(np.abs(air_data[idx, :]))
            ir_shift = index_peak_ref - index_peak_cur
            air_data[idx, :] = np.roll(air_data[idx, :], ir_shift)
        
            if ir_shift < 0:
                air_data[idx, ir_shift:] = 0  # Zero out tail after shift  
         

        #section to expand dataset in cases where few input IRs are available
        #if total IRs below threshold
        apply_transform=0
        if total_measurements < ir_min_threshold and subwoofer_mode==False:
            log_string_a = 'Expanding dataset'
            align_start_time = time.time()
            hf.log_with_timestamp(log_string_a, gui_logger)
            apply_transform=1
            #expand dataset by creating new IRs using pitch shifting
            air_data, status_code = hf.expand_measurements_with_pitch_shift(
                measurement_array=air_data,
                desired_measurements=desired_measurements,pitch_range=pitch_range,gui_logger=gui_logger,
                cancel_event=cancel_event,report_progress=report_progress, pitch_shift_comp=pitch_shift_comp,ignore_ms=ignore_ms
            ) 
            if status_code > 0:#failure or cancelled
                return air_data, status_code
            align_end_time = time.time()
            log_string_a = f"Dataset expansion completed in {align_end_time - align_start_time:.2f} seconds."
            hf.log_with_timestamp(log_string_a)
        
        total_measurements = air_data.shape[0]
        log_string_a = 'Total measurements after expansion: '+str(total_measurements)
        hf.log_with_timestamp(log_string_a, gui_logger)

        # Normalize each AIR in the spectral domain
        fb_start = int(CN.SPECT_SNAP_F0 * CN.N_FFT / samp_freq_ash)
        fb_end = int(CN.SPECT_SNAP_F1 * CN.N_FFT / samp_freq_ash)
        for idx in range(total_measurements):
            data_fft = np.fft.rfft(air_data[idx, :])
            mag_fft = np.abs(data_fft)
            average_mag = np.mean(mag_fft[fb_start:fb_end])
            if average_mag > 0:
                air_data[idx, :] /= average_mag
                
        if cancel_event and cancel_event.is_set():
            gui_logger.log_warning("Operation cancelled by user.")
            return air_data, 2
        

        
 
    
        # Synchronize in time domain
        if total_measurements >= 2:
            t_shift_interval = CN.T_SHIFT_INTERVAL
            min_t_shift = CN.MIN_T_SHIFT_A
            max_t_shift = CN.MAX_T_SHIFT_A
            num_intervals = int(np.abs((max_t_shift - min_t_shift) / t_shift_interval))
            order = CN.ORDER
            delay_win_min_t = CN.DELAY_WIN_MIN_A
            delay_win_hop_size = CN.DELAY_WIN_HOP_SIZE
            delay_win_hops = CN.DELAY_WIN_HOPS_A

            #which alignment frequency to use?
            if f_alignment <= 0:
                if ir_set in CN.AC_SPACE_LIST_SUB:
                    cutoff_alignment = CN.CUTOFF_ALIGNMENT_SUBRIR
                else:
                    cutoff_alignment = CN.CUTOFF_ALIGNMENT_AIR
            else:
                cutoff_alignment=f_alignment

            peak_to_peak_window = int(np.divide(samp_freq_ash, cutoff_alignment) * 1.0)
            delay_eval_set = np.zeros((total_measurements, num_intervals))
            lp_sos = hf.get_filter_sos(cutoff=cutoff_alignment, fs=CN.FS, order=order, filtfilt=CN.FILTFILT_TDALIGN_AIR, b_type='low')

            align_start_time = time.time()
            log_string_a = f"Starting time-domain alignment for {total_measurements} impulse responses..."
            hf.log_with_timestamp(log_string_a, gui_logger)

            print_interval = 100  # Print every 100 IRs (adjust as needed)

            for idx in range(total_measurements):
                align_ir_2d(
                    idx, air_data, data_pad_zeros,
                    t_shift_interval, min_t_shift, max_t_shift, num_intervals, order,
                    delay_win_min_t, delay_win_hop_size, delay_win_hops,
                    cutoff_alignment, samp_freq_ash, delay_eval_set, peak_to_peak_window, lp_sos
                )

                if (idx + 1) % print_interval == 0 or idx == total_measurements - 1:
                    if report_progress > 0:
                        a_point = 0.35
                        b_point = 0.6
                        progress = a_point + (float(idx)/total_measurements)*(b_point-a_point)
                        hf.update_gui_progress(report_progress, progress=progress)
                    log_string_a = f"Aligned {idx + 1} of {total_measurements} IRs."   
                    hf.log_with_timestamp(log_string_a, gui_logger)
                    if cancel_event and cancel_event.is_set():
                        gui_logger.log_warning("Operation cancelled by user.")
                        return air_data, 2

            align_end_time = time.time()
            log_string_a = f"Time-domain alignment completed in {align_end_time - align_start_time:.2f} seconds."
            log_string_b = "Time-domain alignment completed"
            hf.log_with_timestamp(log_string_a)
            hf.log_with_timestamp(log_string_b, gui_logger)
        
    
        #remove direction portion of signal
        if CN.RISE_WINDOW is True:
            air_data *= direct_removal_win
                       
        #
        #set each AIR to 0 level again after removing direct portion
        log_string_a = 'Adjusting Levels'
        hf.log_with_timestamp(log_string_a, gui_logger)
        # fb_start = int(CN.SPECT_SNAP_F0 * CN.N_FFT / samp_freq_ash)
        # fb_end = int(CN.SPECT_SNAP_F1 * CN.N_FFT / samp_freq_ash)
        if subwoofer_mode == True:
            f_norm_start = 30#normalise at low frequencies
            f_norm_end = 140
        else:
            f_norm_start = CN.SPECT_SNAP_F0#use mid frequencies
            f_norm_end = CN.SPECT_SNAP_F1
        # Convert target frequency range to rfft bins
        fb_start = int(f_norm_start * n_fft / samp_freq_ash)
        fb_end = int(f_norm_end * n_fft / samp_freq_ash)
        for idx in range(total_measurements):
            data_fft = np.fft.rfft(air_data[idx, :])
            mag_fft = np.abs(data_fft)
            average_mag = np.mean(mag_fft[fb_start:fb_end])
            if average_mag > 0:
                air_data[idx, :] /= average_mag
        
  
        if save_npy == True:
            #limit to 32 bit if saving
            if air_data.dtype == np.float64:
                air_data = air_data.astype(np.float32)
                
            #save full size array (not combined)
            npy_file_name = ir_set+'_full.npy'
            out_file_path = pjoin(air_out_folder,npy_file_name)      
            #create dir if doesnt exist 
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            np.save(out_file_path,air_data)
            
            log_string_a = 'Exported numpy file to: ' + out_file_path 
            hf.log_with_timestamp(log_string_a, gui_logger)
        
        #also save to wav for testing
        if wav_export == True:
            out_file_name = ir_set+'_sample.wav'
            out_file_path = pjoin(air_out_folder,out_file_name)
            
            #create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            out_wav_array=np.zeros((n_fft,1))
            #grab IR
            out_wav_array[:,0] = np.copy(air_data[1,:])#take sample

            hf.write2wav(file_name=out_file_path, data=out_wav_array, prevent_clipping=1, samplerate=samp_freq_ash)
            
        status=0#success if reached this far
    
   
    except Exception as ex:
        log_string = 'Failed to complete IR set processing for: ' + ir_set 
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
  
    return air_data, status



def align_ir_2d(
    idx, air_data, data_pad_zeros,
    t_shift_interval, min_t_shift, max_t_shift, num_intervals, order,
    delay_win_min_t, delay_win_hop_size, delay_win_hops,
    cutoff_alignment, samp_freq_ash, delay_eval_set, peak_to_peak_window, lp_sos,
    debug=False
):
    """
    Optimized time-domain alignment for a single IR using peak-to-peak analysis.

    Parameters:
        idx (int): Index of the IR to align.
        air_data (np.ndarray): IR dataset of shape (N, T).
        data_pad_zeros (np.ndarray): Zero-initialized array of shape (T,) for temporary use.
        debug (bool): If True, prints detailed debug information for the alignment process.
    """
    crop_length = 16000
    samples_shift = 0

    if idx == 0:
        if debug:
            print(f"[Debug] IR {idx} is reference. No shift applied.")
        return  # First IR is reference

    # Efficient average of prior IRs
    prior_airs = np.mean(air_data[:idx, :crop_length], axis=0)

    this_air = air_data[idx][:crop_length]

    if not np.any(this_air):  # Skip if silent
        if debug:
            print(f"[Debug] IR {idx} is silent. No shift applied.")
        return

    # Filter both IRs
    prior_lp = hf.apply_sos_filter(prior_airs, lp_sos, filtfilt=CN.FILTFILT_TDALIGN_AIR)
    this_lp = hf.apply_sos_filter(this_air, lp_sos, filtfilt=CN.FILTFILT_TDALIGN_AIR)

    # Generate shifted versions
    shifts = min_t_shift + np.arange(num_intervals) * t_shift_interval
    shifted_ir_stack = np.empty((num_intervals, crop_length), dtype=this_lp.dtype)
    for i, shift in enumerate(shifts):
        shifted_ir_stack[i] = np.roll(this_lp, shift)

    summed = shifted_ir_stack + prior_lp

    # Evaluate peak-to-peak energy over sliding windows
    peak_to_peak_values = np.zeros(num_intervals)
    for hop_id in range(delay_win_hops):
        start = delay_win_min_t + hop_id * delay_win_hop_size
        end = start + peak_to_peak_window
        segment = summed[:, start:end]
        ptp_values = segment.max(axis=1) - segment.min(axis=1)
        np.maximum(peak_to_peak_values, ptp_values, out=peak_to_peak_values)

    # Find best shift
    best_index = np.argmax(peak_to_peak_values)
    samples_shift = shifts[best_index]
    delay_eval_set[idx, :] = peak_to_peak_values

    # Apply shift
    air_data[idx] = np.roll(air_data[idx], int(samples_shift))

    if samples_shift < 0:
        air_data[idx, min_t_shift:] = 0.0

    if debug:
        print(f"[Debug] IR {idx}: shift={samples_shift}, max_peak2peak={peak_to_peak_values[best_index]:.6f}")
        

def convert_airs_to_brirs(
    ir_set='default_set',
    ir_group='prepped_airs',
    air_dataset=np.array([]),
    gui_logger=None,
    use_user_folder=False,
    wav_export=CN.EXPORT_WAVS_DEFAULT, spatial_res=3, hrir_dataset=None, correction_factor=1.0,
    long_mode=False, report_progress=0, cancel_event=None, distr_mode=0, rise_time=5.1, subwoofer_mode=False, binaural_input=False, auto_shape_output=False
):
    """
    Converts a dataset of Ambisonic Impulse Responses (AIRs) into Binaural Room Impulse Responses (BRIRs)
    using directional convolution with HRIRs and optional equalization.

    Parameters:
    ----------
    ir_set : str
        Name of the impulse response set (e.g., room or recording environment identifier).
    ir_group : str
        Group of AIRs to use. Must be either 'avg_airs' or 'prepped_airs'.
    air_dataset : np.ndarray, optional
        2D numpy array of shape (measurements, samples), containing AIR measurements to convert.
    gui_logger : callable, optional
        Logger function or object to log processing events with timestamps.
    use_user_folder : bool, default=False
        If True, use a user-specified directory for saving outputs.
    wav_export : bool, default=False
        If True, exports resulting BRIRs to WAV files.
    long_mode : bool, default=False
        If True, use longer AIR processing with extended convolution (intended for high-reverb datasets).
    long_mode : int, default=0
        If 0 distributes measurements into contiguous blocks, if 1 distributes round-robin style

    Returns:
       Status code: 0 = Success, 1 = Failure, 2 = Cancelled
    """
    status=1
    #set fft length based on AC space
    if long_mode == True:
        n_fft=CN.N_FFT_L
    else:
        n_fft=CN.N_FFT
    
    mag_comp=CN.MAG_COMP
  
    samp_freq_ash=CN.SAMP_FREQ

    #impulse
    impulse=np.zeros(CN.N_FFT)
    impulse[0]=1
    brir_reverberation=np.array([])
  
    # Direct sound window for initial rise
    rise_samples = int((rise_time / 1000.0) * samp_freq_ash)
    direct_hanning_start = 61
    # Initialize all-zero window
    direct_removal_win = np.zeros(n_fft)
    # If rise_samples is too small, skip Hann window and apply instant transition
    if rise_samples < 2:
        # Instant fade: everything after the start point becomes 1
        direct_removal_win[direct_hanning_start:] = 1.0
    else:
        # Build Hann window (full window = rise + fall)
        direct_hanning_size = rise_samples * 2
        hann_direct_full = np.hanning(direct_hanning_size)
        # Take the rising half only
        hann_rise = hann_direct_full[:rise_samples]
        # Apply the rising window
        end_rise = direct_hanning_start + rise_samples
        direct_removal_win[direct_hanning_start:end_rise] = hann_rise
        # After the rise portion, set to 1
        direct_removal_win[end_rise:] = 1.0
    
    
 
    try:
        log_string_a = 'Starting IR to BRIR conversion for: '+ir_set
        hf.log_with_timestamp(log_string_a, gui_logger)
           
        #open input dataset if not provided, one saved earlier
        if  air_dataset.size == 0:
            air_in_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', ir_group,ir_set)
            npy_file_name = ir_set+'_full.npy'
            air_file_path = pjoin(air_in_folder,npy_file_name)  
            try:
                #try loading a single npy dataset
                air_dataset = np.load(air_file_path)
                summary_string = hf.summarize_array(air_dataset)
                log_string_a = 'loaded: '+air_file_path + ', array structure: '+summary_string
            except:
                #if that fails, try loading all npy files within the folder, align them, redistribute, and concatenate them
                try:
                    air_dataset = hf.roll_distribute_concatenate_npy_datasets(air_in_folder)
                    summary_string = hf.summarize_array(air_dataset)
                    log_string_a = 'Failed to load: '+air_file_path + ', loaded all npy files in: '+air_in_folder + ', new array structure: '+summary_string     
                except:
                    return brir_reverberation, 1
            hf.log_with_timestamp(log_string_a, gui_logger)
            # Ensure air_dataset is 2D: (measurements, samples)
            if air_dataset.ndim >= 2:
                num_samples = air_dataset.shape[-1]
                air_dataset = air_dataset.reshape(-1, num_samples)
                hf.log_with_timestamp(f"Reshaped AIR dataset to 2D: {air_dataset.shape[0]} measurements x {num_samples} samples.", gui_logger)
            else:
                hf.log_with_timestamp(f"Unsupported AIR shape: {air_dataset.shape}", gui_logger)
                return brir_reverberation, 1
        
        #get number of sets
        total_measurements = air_dataset.shape[0]

        #
        # Load HRIR dataset (user-provided or fallback to defaults)
        if hrir_dataset is not None and isinstance(hrir_dataset, np.ndarray) and hrir_dataset.size > 0:
            hrir_list = np.asarray(hrir_dataset, dtype=np.float64)
            hf.log_with_timestamp(
                f"Using user-provided HRIR dataset, shape: {hrir_list.shape}",
                gui_logger
            )
        else:
            #loads HRIRs from one of two preprocessed HRIR datasets from dummy head measurements. 
            #Max spatial resolution (nearest 2 degrees).
            #DF compensated but no high pass filter
            try:
                npy_fname = pjoin(CN.DATA_DIR_INT, 'hrir_dataset_comp_max_THK-KU-100.npy')
                hrir_list = hf.load_convert_npy_to_float64(npy_fname)
                hf.log_with_timestamp(f"Loaded HRIR dataset: {npy_fname}", gui_logger)
            except:
                npy_fname = pjoin(CN.DATA_DIR_INT, 'hrir_dataset_comp_max_TU-FABIAN.npy')
                hrir_list = hf.load_convert_npy_to_float64(npy_fname)
                hf.log_with_timestamp(f"Loaded HRIR dataset: {npy_fname}", gui_logger)
                    
        
        #
        # Normalize HRIR array shape for processing
        #
        hrir_arr = hrir_list
        # --- Case 1: 5D -> extract listener 0 ---
        if hrir_arr.ndim == 5:
            # shape: [listener, elev, azim, ch, samples]
            hrir_selected = hrir_arr[0]
        # --- Case 2: 4D -> already in good shape ---
        elif hrir_arr.ndim == 4:
            # shape: [elev, azim, ch, samples]
            hrir_selected = hrir_arr
        # --- Invalid case ---
        else:
            raise ValueError(
                f"HRIR array has invalid number of dimensions ({hrir_arr.ndim}). "
                f"Expected 4 or 5 dimensions."
            )
            
        # Infer / validate spatial resolution
        spatial_res = hrir_processing.infer_hrir_spatial_res(hrir_selected,spatial_res=spatial_res,logger=gui_logger)

        total_elev_hrir = len(hrir_selected)
        total_azim_hrir = len(hrir_selected[0])
        total_chan_hrir = len(hrir_selected[0][0])
        total_samples_hrir = len(hrir_selected[0][0][0])
        base_elev_idx = total_elev_hrir//2
        base_elev_idx_offset=total_elev_hrir//8
        
        #spatial metadata
        elev_min=CN.SPATIAL_RES_ELEV_MIN_IN[spatial_res] 
        elev_max=CN.SPATIAL_RES_ELEV_MAX_IN[spatial_res] 
        azim_min=CN.SPATIAL_RES_AZIM_MIN_IN[spatial_res] 
        azim_max=CN.SPATIAL_RES_AZIM_MAX_IN[spatial_res] 
        elev_nearest=CN.SPATIAL_RES_ELEV_NEAREST_IN[spatial_res] #as per hrir dataset
        azim_nearest=CN.SPATIAL_RES_AZIM_NEAREST_IN[spatial_res] 
        #define desired angles
        elev_min_dist=elev_min#-40
        elev_max_dist=elev_max+elev_nearest#62
        azim_min_dist=azim_min
        azim_max_dist=azim_max+azim_nearest
        elev_src_set = np.arange(elev_min_dist,elev_max_dist,elev_nearest)
        azim_src_set=np.arange(azim_min_dist,azim_max_dist,azim_nearest)

        
        #set number of output directions (brir sources) 
        if total_measurements < CN.IR_MIN_THRESHOLD_TRANSFORM:
            raise ValueError( f"AIR array has insufficient number of measurements ({total_measurements}). ")
        if auto_shape_output:#derive based on number of measurements
            num_brir_sources = max(2, total_measurements // 360)
        else:
            if subwoofer_mode == True:#only 1 set required if subwoofer response
                num_brir_sources=1
            elif total_measurements < CN.IR_MIN_THRESHOLD_FULLSET:#not enough measurements for 7.1 set, fallback to 5.1 set
                num_brir_sources=5
            else:
                num_brir_sources=7#default to 7 sources for a 7.1 configuration
   
        log_string_a = 'num_brir_sources: ' + str(num_brir_sources)
        hf.log_with_timestamp(log_string_a)
        
        # Create numpy array for new BRIR dataset
        brir_reverberation = np.zeros((CN.INTERIM_ELEVS, num_brir_sources, 2, n_fft))
        
        # Log start
        hf.log_with_timestamp("Estimating BRIRs from IRs", gui_logger)
        
        # Sampling spatial coordinates with bias
        num_samples = total_measurements
        biased_centers = np.array([40, 140, 220, 320])#45, 135, 225, 315
        strength = 50#lower is more biased
        azimuths_distribution, elevations_distribution = hf.biased_spherical_coordinate_sampler(
            azim_src_set, elev_src_set, num_samples,
            biased_azimuth_centers=biased_centers,
            azimuth_bias_strength=strength,
            plot_distribution=False
        )
        
        if distr_mode == 0:
            # Split measurements evenly across sources
            measurements_per_source = np.array_split(np.arange(total_measurements), num_brir_sources)
        else:
            # Distribute measurements round-robin across sources (as lists)
            measurements_per_source = [[] for _ in range(num_brir_sources)]
            for i in range(total_measurements):
                measurements_per_source[i % num_brir_sources].append(i)
        
        # Distribution index counter
        dist_counter = 0
        
        # Loop over each BRIR source direction 
        for source_num, group in enumerate(measurements_per_source):
            # Initialize accumulators for averaging
            brir_sum_l = np.zeros(n_fft)
            brir_sum_r = np.zeros(n_fft)
        
            if cancel_event and cancel_event.is_set():
                gui_logger.log_warning("Operation cancelled by user.")
                return brir_reverberation, 2
        
            for i, measurement in enumerate(group):
                # Copy AIR, wrap around if fewer measurements than sources
                curr_air = np.copy(air_dataset[measurement % total_measurements, :])
        
                if binaural_input:
                    # Alternate contribution between left and right channels
                    if i % 2 == 0:
                        brir_sum_l += curr_air[:n_fft]
                    else:
                        brir_sum_r += curr_air[:n_fft]
                else:
                    # Get HRIR from distributed coordinates
                    curr_azim_deg = azimuths_distribution[dist_counter]
                    curr_elev_deg = elevations_distribution[dist_counter]
                    curr_azim_id = int(curr_azim_deg / azim_nearest)
                    curr_elev_id = int((curr_elev_deg - elev_min) / elev_nearest)
        
                    curr_hrir_l = np.copy(hrir_selected[curr_elev_id][curr_azim_id][0][:])
                    curr_hrir_r = np.copy(hrir_selected[curr_elev_id][curr_azim_id][1][:])
        
                    # Convolve AIR with HRIRs
                    curr_brir_l = sp.signal.convolve(curr_air, curr_hrir_l, mode='full', method='auto')
                    curr_brir_r = sp.signal.convolve(curr_air, curr_hrir_r, mode='full', method='auto')
        
                    # Accumulate for averaging
                    brir_sum_l += curr_brir_l[:n_fft]
                    brir_sum_r += curr_brir_r[:n_fft]
        
                # Move to next direction
                dist_counter += 1
        
            # Assign averaged results
            if binaural_input:
                brir_reverberation[0, source_num, 0, :] = brir_sum_l / ((len(group) + 1) // 2)
                brir_reverberation[0, source_num, 1, :] = brir_sum_r / (len(group) // 2)
            else:
                brir_reverberation[0, source_num, 0, :] = brir_sum_l / len(group)
                brir_reverberation[0, source_num, 1, :] = brir_sum_r / len(group)


    
        
        # Window initial rise
        brir_reverberation[0, :, :, :] *= direct_removal_win  # broadcasted multiply over channel axis
        
        # Window tail end of BRIR
        fade_start = int( int(((CN.RT60_MAX_S)/1000)*CN.FS) * (n_fft / CN.N_FFT)) 
        l_fade_win_size = abs(fade_start - n_fft) * 2
        hann_l_fade_full = np.hanning(l_fade_win_size)
        win_l_fade_out = np.split(hann_l_fade_full, 2)[1]  # second half
        l_fade_out_win = np.ones(n_fft)
        l_fade_out_win[fade_start:] = win_l_fade_out
        brir_reverberation[0, :, :, :] *= l_fade_out_win  # broadcasted multiply
        
        # Normalize each direction to average magnitude between fb_start and fb_end       
        f_norm_start = CN.SPECT_SNAP_F0
        f_norm_end = CN.SPECT_SNAP_F1
        # Convert target frequency range to rfft bins
        fb_start = int(f_norm_start * n_fft / samp_freq_ash)
        fb_end = int(f_norm_end * n_fft / samp_freq_ash)
        # # Vectorized FFT over all directions and both channels
        # Convert to rfft domain (half spectrum)
        fft_data = np.fft.rfft(brir_reverberation[0, :, :, :CN.N_FFT], axis=-1)
        # Compute magnitudes within that frequency band
        fft_magnitudes = np.abs(fft_data[:, :, fb_start:fb_end])
        avg_magnitudes = np.mean(fft_magnitudes, axis=-1)  # shape: (num_brir_sources, 2)
        avg_mag = np.mean(avg_magnitudes, axis=1)  # shape: (num_brir_sources,)
        # Normalize each direction's time-domain signal by the magnitude
        brir_reverberation[0, :, :, :] /= avg_mag[:, np.newaxis, np.newaxis]
                
        #
        #optional: create correction filter from average response and target, then equalise each BRIR
        #
        if mag_comp and correction_factor >= 0.1:
            hf.log_with_timestamp("Compensating BRIRs", gui_logger)
            
            comp_win_size = 7 if subwoofer_mode == True else 7
        
            # Level ends of spectrum
            high_freq = 12000 if subwoofer_mode == True else 15000
            low_freq_in = 20 if subwoofer_mode == True else 30
            low_freq_in_target = 150 if subwoofer_mode == True else low_freq_in#make target level in sub frequencies
        
            # Load target
            target_path = pjoin(CN.DATA_DIR_INT, 'reverberation', 'reverb_target_mag_response.npy')
            brir_fft_target_mag = np.load(target_path)
            brir_fft_target_mag = hf.level_spectrum_ends(brir_fft_target_mag, low_freq_in_target, high_freq, smooth_win=10)#20
        
            for direc in range(num_brir_sources):
                if cancel_event and cancel_event.is_set():
                    gui_logger.log_warning("Operation cancelled by user.")
                    return brir_reverberation, 2
                
                # Extract BRIRs for this direction (shape: 2 x N)
                brirs = brir_reverberation[0, direc, :, :CN.N_FFT]  # shape: (2, N_FFT)
        
                # Compute average magnitude spectrum in dB
                brir_fft = np.fft.fft(brirs, axis=-1)  # shape: (2, N_FFT)
                brir_mag = np.abs(brir_fft)
                brir_db = hf.mag2db(brir_mag)
                brir_fft_avg_db = np.mean(brir_db, axis=0)
        
                # Convert back to magnitude, level ends, and smooth
                brir_fft_avg_mag = hf.db2mag(brir_fft_avg_db)
                brir_fft_avg_mag = hf.level_spectrum_ends(brir_fft_avg_mag, low_freq_in, high_freq, smooth_win=comp_win_size)#6
                brir_fft_avg_mag_sm = hf.smooth_freq_octaves(data=brir_fft_avg_mag, win_size_base = comp_win_size)
        
                # Compensation filter (smoothed and scaled)
                comp_mag = hf.db2mag((hf.mag2db(brir_fft_target_mag) - hf.mag2db(brir_fft_avg_mag_sm))*correction_factor)
                #comp_mag = hf.smooth_freq_octaves(comp_mag, win_size_base = 7)
                comp_eq_fir = hf.build_min_phase_filter(comp_mag)
        
                # Equalize both channels (loop still needed for convolution)
                for chan in range(total_chan_hrir):
                    eq_brir = sp.signal.convolve(brir_reverberation[0, direc, chan, :], comp_eq_fir, 'full', 'auto')
                    brir_reverberation[0, direc, chan, :] = eq_brir[:n_fft]
        
                # Optional plots
                if CN.PLOT_ENABLE:
                    hf.plot_data(brir_fft_target_mag, f"{direc}_brir_fft_target_mag", normalise=0)
                    hf.plot_data(brir_fft_avg_mag, f"{direc}_brir_fft_avg_mag", normalise=0)
                    hf.plot_data(brir_fft_avg_mag_sm, f"{direc}_brir_fft_avg_mag_sm", normalise=0)
                    hf.plot_data(comp_mag, f"{direc}_comp_mag", normalise=0)

        
        #
        #set each direction to 0 level again
        #
        # Normalize each direction to average magnitude between fb_start and fb_end    
        if subwoofer_mode == True:
            f_norm_start = 30#normalise at low frequencies
            f_norm_end = 140
        else:
            f_norm_start = CN.SPECT_SNAP_F0#use mid frequencies
            f_norm_end = CN.SPECT_SNAP_F1
        # Convert target frequency range to rfft bins
        fb_start = int(f_norm_start * n_fft / samp_freq_ash)
        fb_end = int(f_norm_end * n_fft / samp_freq_ash)
        # # Vectorized FFT over all directions and both channels
        # Convert to rfft domain (half spectrum)
        fft_data = np.fft.rfft(brir_reverberation[0, :, :, :CN.N_FFT], axis=-1)
        # Compute magnitudes within that frequency band
        fft_magnitudes = np.abs(fft_data[:, :, fb_start:fb_end])
        avg_magnitudes = np.mean(fft_magnitudes, axis=-1)  # shape: (num_brir_sources, 2)
        avg_mag = np.mean(avg_magnitudes, axis=1)  # shape: (num_brir_sources,)
        # Normalize each direction's time-domain signal by the magnitude
        brir_reverberation[0, :, :, :] /= avg_mag[:, np.newaxis, np.newaxis]


        
        if use_user_folder == True:
            brir_out_folder = pjoin(CN.DATA_DIR_AS_USER,ir_set)
        else:
            brir_out_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'est_brirs',ir_set)
        
        # Trim the BRIR reverberation array if needed (trim last dimension based on threshold)
        brir_reverberation = trim_brir_samples(brir_reverberation, relative_threshold=1e-5, ignore_last=2000)
  
        #
        #export wavs for testing
        if wav_export == True:
            for direc in range(num_brir_sources):
                out_file_name = ir_set+'_'+str(direc)+'_est_brir.wav'
                out_file_path = pjoin(brir_out_folder,out_file_name)
                
                #create dir if doesnt exist
                output_file = Path(out_file_path)
                output_file.parent.mkdir(exist_ok=True, parents=True)
                num_samples = brir_reverberation.shape[-1]
                out_wav_array=np.zeros((num_samples,2))
                #grab BRIR
                out_wav_array[:,0] = np.copy(brir_reverberation[0,direc,0,:])#L
                out_wav_array[:,1] = np.copy(brir_reverberation[0,direc,1,:])#R

                hf.write2wav(file_name=out_file_path, data=out_wav_array, prevent_clipping=1, samplerate=samp_freq_ash)
        
        if brir_reverberation.dtype == np.float64:
            brir_reverberation = brir_reverberation.astype(np.float32)
            
        if subwoofer_mode == True:#only 2 dimensions required if subwoofer response
            brir_reverberation=hf.remove_leading_singletons(brir_reverberation)
        
        #
        #save numpy array for later use in BRIR generation functions
        #
        if mag_comp == True:#
            npy_file_name =  'reverberation_dataset_' +ir_set+'.npy'
        else:
            npy_file_name =  'reverberation_dataset_nocomp_' +ir_set+'.npy'
    
        out_file_path = pjoin(brir_out_folder,npy_file_name)      
          
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        np.save(out_file_path,brir_reverberation)    
        
        log_string_a = 'Exported numpy file to: ' + out_file_path 
        hf.log_with_timestamp(log_string_a, gui_logger)
        
        status=0#success if reached this far
   
    
    except Exception as ex:

        log_string = 'Failed to complete AIR to BRIR processing for: ' + ir_set 
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
        
    return brir_reverberation, status

def trim_brir_samples(brir_array, relative_threshold=0.000005, ignore_last=2000):
    """
    Trims the last dimension (samples) of a NumPy array by removing trailing elements
    below a threshold relative to the max value, ignoring the last 'ignore_last' samples.

    Args:
        brir_array: The BRIR array to be trimmed.
        relative_threshold: The threshold for trimming (relative to the max value in the array).
        ignore_last: The number of last samples to ignore when checking for threshold.

    Returns:
        brir_array_trimmed: The trimmed BRIR array.
    """
    original_shape = brir_array.shape
    last_dim = brir_array.shape[-1]
    
    max_value = np.abs(brir_array).max()
    threshold = max_value * relative_threshold
    
    # Ignore the last 'ignore_last' samples
    last_valid_index = last_dim - 1 - ignore_last
    
    for i in range(last_valid_index, -1, -1):
        if np.abs(brir_array[..., i]).max() >= threshold:
            last_valid_index = i
            break

    brir_array_trimmed = brir_array[..., :last_valid_index + 1]
    
    return brir_array_trimmed

def write_as_metadata_csv(ir_set, data_rows, sub_mode=False, gui_logger=None):
    """
    Writes a metadata CSV file with predefined columns. Missing values are left blank.
    
    :param file_path: str, full path to the CSV file to write.
    :param data_rows: list of dicts, where each dict represents a row with keys matching the column headers.
    :param gui_logger: optional logger for GUI display.
    """
    
    if sub_mode == False:
        metadata_key=CN.USER_CSV_KEY#_metadata
    else:
        metadata_key=CN.ASI_USER_CSV_KEY#_asi-metadata
    
    fieldnames = CN.AC_SPACE_FIELD_NAMES
    file_folder = pjoin(CN.DATA_DIR_AS_USER,ir_set)
    filename = ir_set + metadata_key +'.csv'
    file_path = pjoin(file_folder,filename)

    try:
        # Ensure the output directory exists
        os.makedirs(file_folder, exist_ok=True)
        
        with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data_rows:
                complete_row = {key: row.get(key, "") for key in fieldnames}
                writer.writerow(complete_row)

        log_string = 'Exported metadata CSV to: ' + file_path
        if gui_logger:
            hf.log_with_timestamp(log_string, gui_logger)
        else:
            print(log_string)

    except Exception as e:
        error_message = f"Failed to write metadata CSV to {file_path}: {str(e)}"
        if gui_logger:
            hf.log_with_timestamp(error_message, gui_logger)
        else:
            print(error_message)


def write_sub_metadata_csv(ir_set, data_rows, gui_logger=None):
    """
    Writes a metadata CSV file with predefined columns. Missing values are left blank.
    
    :param file_path: str, full path to the CSV file to write.
    :param data_rows: list of dicts, where each dict represents a row with keys matching the column headers.
    :param gui_logger: optional logger for GUI display.
    """
    fieldnames = CN.SUB_FIELD_NAMES
    file_folder = pjoin(CN.DATA_DIR_AS_USER,ir_set)
    filename = ir_set + CN.SUB_USER_CSV_KEY +'.csv'
    file_path = pjoin(file_folder,filename)

    try:
        # Ensure the output directory exists
        os.makedirs(file_folder, exist_ok=True)
        
        with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data_rows:
                complete_row = {key: row.get(key, "") for key in fieldnames}
                writer.writerow(complete_row)

        log_string = 'Exported metadata CSV to: ' + file_path
        if gui_logger:
            hf.log_with_timestamp(log_string, gui_logger)
        else:
            print(log_string)

    except Exception as e:
        error_message = f"Failed to write metadata CSV to {file_path}: {str(e)}"
        if gui_logger:
            hf.log_with_timestamp(error_message, gui_logger)
        else:
            print(error_message)

    
 
    
    
    
    
    
    
def loadIR(sessionPath):
    """Load impulse response (IR) data

    Parameters
    ------
    sessionPath: Path to IR folder

    Returns
    ------
    pos_mic: Microphone positions of shape (numMic, 3)
    pos_src: Source positions of shape (numSrc, 3)
    fullIR: IR data of shape (numSrc, numMic, irLen)
    """
    pos_mic = np.load(sessionPath.joinpath("pos_mic.npy"))
    pos_src = np.load(sessionPath.joinpath("pos_src.npy"))

    numMic = pos_mic.shape[0]
    
    allIR = []
    irIndices = []
    for f in sessionPath.iterdir():
        if not f.is_dir():
            if f.stem.startswith("ir_"):
                allIR.append(np.load(f))
                irIndices.append(int(f.stem.split("_")[-1]))

    assert(len(allIR) == numMic)
    numSrc = allIR[0].shape[0]
    irLen = allIR[0].shape[-1]
    fullIR = np.zeros((numSrc, numMic, irLen))
    for i, ir in enumerate(allIR):
        assert(ir.shape[0] == numSrc)
        assert(ir.shape[-1] == irLen)
        fullIR[:, irIndices[i], :] = ir

    return pos_mic, pos_src, fullIR    
    



def calc_avg_room_target_mag(gui_logger=None):
    """
    function to calculate average magnitude response of various rooms
    :return: None
    """ 
    
    air_ref_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'full_airs')
    #ir_data_folder = pjoin(CN.DATA_DIR_RAW, 'ir_data', 'prepped_airs')
    
    n_fft=CN.N_FFT
    #impulse
    impulse=np.zeros(n_fft)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat = hf.mag2db(fr_flat_mag)
    
    num_airs_avg = 0
    air_fft_avg_db = fr_flat.copy()
    
    log_string_a = 'Mag response estimation loop running'
    hf.log_with_timestamp(log_string_a, gui_logger)
            
    #loop through folders
    try:
        for root, dirs, files in os.walk(air_ref_folder):
            for filename in files:
                if '.npy' in filename and 'broadcast_studio_a' in filename:
                    #read npy files
                    air_file_path = pjoin(root, filename)
                    air_set = np.load(air_file_path)
                    
                    #get number of sets
                    total_sets_air = len(air_set)
                    total_irs_air =  len(air_set[0])
                    total_chan_air = len(air_set[0][0])
                    total_irs_sel=120
                    
                    for set_num in range(total_sets_air):
                        #each set represents a new BRIR direction
                        for chan_air in range(total_chan_air):
                            for ir_air in range(total_irs_sel):
                                curr_air = np.copy(air_set[set_num,ir_air,chan_air,0:n_fft])
                                if np.sum(np.abs(curr_air)) > 0:
                                    air_current_fft = np.fft.fft(curr_air)#
                                    air_current_mag_fft=np.abs(air_current_fft)
                                    air_current_db_fft = hf.mag2db(air_current_mag_fft)     
                                    air_fft_avg_db = np.add(air_fft_avg_db,air_current_db_fft)
                                    num_airs_avg = num_airs_avg+1
                                else:
                                    print(str(filename))
                                    print(str(set_num))
                                    print(str(chan_air))
                                    print(str(ir_air))
    
    
        #divide by total number of brirs
        air_fft_avg_db = air_fft_avg_db/num_airs_avg
        #convert to mag
        air_fft_avg_mag = hf.db2mag(air_fft_avg_db)
        #level ends of spectrum
        air_fft_avg_mag = hf.level_spectrum_ends(air_fft_avg_mag, 40, 18000, smooth_win = 10)#150
        
        #smoothing
        #octave smoothing
        air_fft_avg_mag_sm = hf.smooth_freq_octaves(data=air_fft_avg_mag, n_fft=n_fft)
        
        #create min phase FIR
        #avg_room_min_fir = hf.build_min_phase_filter(air_fft_avg_mag_sm)
 
        if CN.PLOT_ENABLE == True:
            print(str(num_airs_avg))
            hf.plot_data(air_fft_avg_mag,'air_fft_avg_mag', normalise=0)
            hf.plot_data(air_fft_avg_mag_sm,'air_fft_avg_mag_sm', normalise=0)    
  
  
    except Exception as ex:
        log_string = 'Failed to complete reverb response processing'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
            
            
            
    
def calc_reverb_target_mag(gui_logger=None):
    """
    function to calculate average magnitude response of BRIR sets
    :return: None
    """ 
    
    brir_ref_folder = pjoin(CN.DATA_DIR_INT, 'reverberation', 'ref_bin')
    
    n_fft=CN.N_FFT
    
    samp_freq_ash=CN.SAMP_FREQ
    output_wavs=1
    
    #windows
    data_pad_zeros=np.zeros(n_fft)
    data_pad_ones=np.ones(n_fft)
    total_chan_brir=2
    
    #direct sound window for 2nd phase
    direct_hanning_size=300#350
    direct_hanning_start=51#101
    hann_direct_full=np.hanning(direct_hanning_size)
    hann_direct = np.split(hann_direct_full,2)[0]
    direct_removal_win = data_pad_zeros.copy()
    direct_removal_win[direct_hanning_start:direct_hanning_start+int(direct_hanning_size/2)] = hann_direct
    direct_removal_win[direct_hanning_start+int(direct_hanning_size/2):]=data_pad_ones[direct_hanning_start+int(direct_hanning_size/2):]
    
    #impulse
    impulse=np.zeros(n_fft)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat = hf.mag2db(fr_flat_mag)
    
    #loop through folders
    try:

        num_bairs_avg = 0
        brir_fft_avg_db = fr_flat.copy()
        
        for root, dirs, files in os.walk(brir_ref_folder):
            for filename in files:
                if '.npy' in filename:

                    #read npy files
                    air_file_path = pjoin(root, filename)
                    brir_reverberation = np.load(air_file_path)
                    
                    #get number of sets
                    total_elev_reverb = len(brir_reverberation)
                    total_azim_reverb = len(brir_reverberation[0])
                    total_chan_reverb = len(brir_reverberation[0][0])
                    total_samples_reverb = len(brir_reverberation[0][0][0])
      
                    #get diffuse field spectrum
                    for direc in range(total_azim_reverb):
                        for chan in range(total_chan_reverb):
                            brir_current = np.copy(brir_reverberation[0,direc,chan,0:n_fft])#brir_out[elev][azim][chan][:]
                            brir_current_fft = np.fft.fft(brir_current)#ensure left channel is taken for both channels
                            brir_current_mag_fft=np.abs(brir_current_fft)
                            brir_current_db_fft = hf.mag2db(brir_current_mag_fft)
                            
                            brir_fft_avg_db = np.add(brir_fft_avg_db,brir_current_db_fft)
                            
                            num_bairs_avg = num_bairs_avg+1
 
     
        #divide by total number of brirs
        brir_fft_avg_db = brir_fft_avg_db/num_bairs_avg
        #convert to mag
        brir_fft_avg_mag = hf.db2mag(brir_fft_avg_db)
        #level ends of spectrum
        brir_fft_avg_mag = hf.level_spectrum_ends(brir_fft_avg_mag, 40, 18000, smooth_win = 10)#150
        
        #smoothing
        #octave smoothing
        brir_fft_avg_mag_sm = hf.smooth_freq_octaves(data=brir_fft_avg_mag, n_fft=n_fft)
  
 
        if CN.PLOT_ENABLE == True:
            print(str(num_bairs_avg))
            hf.plot_data(brir_fft_avg_mag,'brir_fft_avg_mag', normalise=2)
            hf.plot_data(brir_fft_avg_mag_sm,'brir_fft_avg_mag_sm', normalise=2)    
  
        #
        #save numpy array for later use in BRIR generation functions
        #
        npy_file_name =  'reverb_target_mag_response.npy'
        brir_out_folder = pjoin(CN.DATA_DIR_INT, 'reverberation')
        out_file_path = pjoin(brir_out_folder,npy_file_name)      
          
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        np.save(out_file_path,brir_fft_avg_mag_sm)    
        
        log_string_a = 'Exported numpy file to: ' + out_file_path 
        hf.log_with_timestamp(log_string_a, gui_logger)
       

    
    except Exception as ex:
        log_string = 'Failed to complete BRIR reverb response processing'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
    
  

def calc_room_target_dataset(gui_logger=None):
    """
    function to calculate average magnitude response of various rooms
    :return: None
    """ 
    
    try:
        n_fft=CN.N_FFT
        #impulse
        impulse=np.zeros(n_fft)
        impulse[0]=1
        fr_flat_mag = np.abs(np.fft.fft(impulse))
        fr_flat = hf.mag2db(fr_flat_mag)
    
        log_string_a = 'Room target response calculation running'
        hf.log_with_timestamp(log_string_a, gui_logger)
        
        ROOM_TARGET_LIST_FIRS = ['Flat','ash_room_target_fir','harman_b_room_target_fir','hats_room_target_fir','toole_trained_room_target_fir','rtings_target_fir']#deprecated
    
        #create new array for old and new targets
        num_targets=len(ROOM_TARGET_LIST_FIRS)-1
        num_targets_new=num_targets*3
        fir_length=4096
        room_target_fir_arr=np.zeros((num_targets_new,fir_length))
        
        mag_range_a=1200
        mag_range_b=1800
        #load existing room targets from .mat
        #
        # load room target filter (FIR)
        
        mat_fname = pjoin(CN.DATA_DIR_INT, 'room_targets_firs.mat')
        room_target_mat = mat73.loadmat(mat_fname)
        
        target_id=0
        for room_target in ROOM_TARGET_LIST_FIRS:
            if room_target != 'Flat':
                #load fir from mat struct
                room_target_fir = room_target_mat[room_target]
                
                #normalise to 0db at 1khz
                room_target_fir_t=np.zeros(CN.N_FFT)
                room_target_fir_t[0:fir_length] = np.copy(room_target_fir[0:fir_length])
                data_fft = np.fft.fft(room_target_fir_t)
                mag_fft=np.abs(data_fft)
                average_mag = np.mean(mag_fft[mag_range_a:mag_range_b])
                room_target_fir[0:fir_length] = np.divide(room_target_fir[0:fir_length],average_mag)

                #smooth spectrum
                room_target_fir_t=np.zeros(CN.N_FFT)
                room_target_fir_t[0:fir_length] = np.copy(room_target_fir[0:fir_length])
                data_fft = np.fft.fft(room_target_fir_t)
                room_target_mag=np.abs(data_fft)
                #level ends of spectrum
                room_target_mag_new = hf.level_spectrum_ends(room_target_mag, 20, 20000, smooth_win = 3)#150
                room_target_mag_new_sm = hf.smooth_freq_octaves(data=room_target_mag_new, n_fft=n_fft,fund_freq=150,win_size_base = 3)
                #create min phase FIR
                room_target_min_fir = hf.build_min_phase_filter(room_target_mag_new_sm)


                #save into numpy array
                room_target_fir_arr[target_id,:] = np.copy(room_target_min_fir[0:fir_length])
                
                target_id=target_id+1
    
        #calculate new set with modified targets that are flat > 1000Hz
        for idx in range(num_targets):
            room_target_fir=np.zeros(CN.N_FFT)
            room_target_fir[0:fir_length] = np.copy(room_target_fir_arr[idx,:])
            data_fft = np.fft.fft(room_target_fir)
            room_target_mag=np.abs(data_fft)
            #level ends of spectrum
            room_target_mag_new = hf.level_spectrum_ends(room_target_mag, 20, 1000, smooth_win = 3)#150
            room_target_mag_new_sm = hf.smooth_freq(data=room_target_mag_new, n_fft=n_fft, crossover_f=2000, win_size_a = 3, win_size_b = 500)
            
            #create min phase FIR
            room_target_min_fir = hf.build_min_phase_filter(room_target_mag_new_sm)
            
            #store into numpy array
            new_id=num_targets+idx
            room_target_fir_arr[new_id,:] = np.copy(room_target_min_fir[0:fir_length])
            
            if CN.PLOT_ENABLE == True:
                print(str(idx))
                hf.plot_data(room_target_mag_new,'room_target_mag_new_'+str(idx), normalise=0)
                hf.plot_data(room_target_mag_new_sm,'room_target_mag_new_sm_'+str(idx), normalise=0)
                plot_name = 'room_target_min_fir_'+str(idx)
                hf.plot_td(room_target_min_fir[0:fir_length],plot_name)
            
        #calculate new set with modified targets that are flat > 5000Hz
        for idx in range(num_targets):
            room_target_fir=np.zeros(CN.N_FFT)
            room_target_fir[0:fir_length] = room_target_fir_arr[idx,:]
            data_fft = np.fft.fft(room_target_fir)
            room_target_mag=np.abs(data_fft)
            #level ends of spectrum
            room_target_mag_new = hf.level_spectrum_ends(room_target_mag, 20, 5000, smooth_win = 3)#150
            room_target_mag_new_sm = hf.smooth_freq(data=room_target_mag_new, n_fft=n_fft, crossover_f=2000, win_size_a = 3, win_size_b = 500)
            
            #create min phase FIR
            room_target_min_fir = hf.build_min_phase_filter(room_target_mag_new_sm)
            
            #store into numpy array
            new_id=num_targets*2+idx
            room_target_fir_arr[new_id,:] = np.copy(room_target_min_fir[0:fir_length])
            
            if CN.PLOT_ENABLE == True:
                print(str(idx))
                hf.plot_data(room_target_mag_new,'room_target_mag_new_'+str(idx), normalise=0)
                hf.plot_data(room_target_mag_new_sm,'room_target_mag_new_sm_'+str(idx), normalise=0)
                plot_name = 'room_target_min_fir_'+str(idx)
                hf.plot_td(room_target_min_fir[0:fir_length],plot_name)
            

      
        #create flat target and insert into start of numpy array
        #impulse
        impulse=np.zeros(fir_length)
        impulse[0]=1
        room_target_fir_arr_i=np.zeros((1,fir_length))
        room_target_fir_arr_i[0,:] = np.copy(impulse[0:fir_length])
        room_target_fir_arr_out=np.append(room_target_fir_arr_i,room_target_fir_arr,0)
        
    
        #
        #save numpy array for later use in BRIR generation functions
        #
        npy_file_name =  'room_targets_firs.npy'
        data_out_folder = CN.DATA_DIR_INT
        out_file_path = pjoin(data_out_folder,npy_file_name)      
          
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        np.save(out_file_path,room_target_fir_arr_out)    
        
        log_string_a = 'Exported numpy file to: ' + out_file_path 
        hf.log_with_timestamp(log_string_a, gui_logger)
        
        
    
    
    except Exception as ex:
        log_string = 'Failed to complete room target response processing'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
            
                


def acoustic_space_updates(download_updates=False, gui_logger=None):
    """
    Function finds latest versions of acoustic spaces, compares with current versions.
    Downloads updates when enabled.
    """

    try:
        hf.log_with_timestamp("Checking for acoustic space updates", gui_logger)

        # ---------------------------
        # Load LOCAL metadata
        # ---------------------------
        csv_directory = pjoin(CN.DATA_DIR_INT, "reverberation")
        metadata_file = pjoin(csv_directory, CN.REV_METADATA_FILE_NAME)

        local_meta_dict_list = []
        with open(metadata_file, encoding="utf-8-sig", newline="") as inputfile:
            reader = DictReader(inputfile)
            for row in reader:
                local_meta_dict_list.append(row)

        if not local_meta_dict_list:
            raise ValueError("local metadata is empty")

        # ---------------------------
        # Load ONLINE metadata from GitHub
        # ---------------------------
        ghub_meta_url = CN.AS_META_URL

        remote_meta_tmp = pjoin(csv_directory, "reverberation_metadata_latest.csv")

        # Prefer GitHub directly — always first
        hf.log_with_timestamp("Downloading latest metadata...", gui_logger)
        response = hf.download_file(url=ghub_meta_url, save_location=remote_meta_tmp, gui_logger=gui_logger)

        if response is not True:
            raise ValueError("Failed to download latest metadata")

        web_meta_dict_list = []
        with open(remote_meta_tmp, encoding="utf-8-sig", newline="") as inputfile:
            reader = DictReader(inputfile)
            for row in reader:
                web_meta_dict_list.append(row)

        if not web_meta_dict_list:
            raise ValueError("latest metadata is empty")

        mismatches = 0
        updates_perf = 0

        # -------------------------------------------------
        # Compare versions + download updates (Github preferred)
        # -------------------------------------------------
        for space_w in web_meta_dict_list:
            name_w = space_w.get("name_src")
            name_gui_w = space_w.get("name_gui")
            vers_w = space_w.get("version")

            ghub_url = space_w.get("ghub_link")  #
            primary_url = space_w.get("gdrive_link")
            alternate_url = space_w.get("alternative_link")

            rev_folder = space_w.get("folder")
            file_name = space_w.get("file_name")

            local_file_path = pjoin(csv_directory, rev_folder, file_name + ".npy")

            match_found = False
            update_required = False

            # ----- Compare local/remote versions -----
            for space_l in local_meta_dict_list:
                if space_l.get("name_src") == name_w:
                    match_found = True
                    if space_l.get("version") != vers_w:
                        mismatches += 1
                        update_required = True
                        hf.log_with_timestamp(
                            f"New version ({vers_w}) available for: {name_gui_w}",
                            gui_logger,
                        )
                    break

            if not match_found:
                mismatches += 1
                update_required = True
                hf.log_with_timestamp(
                    f"New acoustic space available: {name_gui_w}",
                    gui_logger,
                )

            # --------------------------------------------------
            # Download update (Github preferred)
            # --------------------------------------------------
            if download_updates and update_required:

                hf.log_with_timestamp(f"Downloading update for {name_gui_w}", gui_logger)

                # Github always first
                download_sources = [
                    ("github", ghub_url),
                    ("primary", primary_url),
                    ("alternate", alternate_url),
                ]

                success = False

                for source_name, url in download_sources:
                    if not url:
                        continue

                    hf.log_with_timestamp(f"Attempting {source_name} download...", gui_logger)

                    response = hf.download_file(url=url, save_location=local_file_path, gui_logger=gui_logger)
                    if response is True:
                        # Validate by loading
                        try:
                            arr = hf.load_convert_npy_to_float64(local_file_path)
                            if arr is not None and len(arr) > 0:
                                hf.log_with_timestamp(
                                    f"Successfully downloaded {name_gui_w} from {source_name}",
                                    gui_logger,
                                )
                                success = True
                                updates_perf += 1
                                break
                            else:
                                hf.log_with_timestamp(
                                    f"Invalid/empty dataset after {source_name} download.",
                                    gui_logger,
                                )
                        except Exception:
                            hf.log_with_timestamp(
                                f"Dataset failed to load after {source_name} download.",
                                gui_logger,
                            )
                    else:
                        hf.log_with_timestamp(f"{source_name.capitalize()} download failed.", gui_logger)

                if not success:
                    raise ValueError(f"Failed to download updated dataset for {name_gui_w}")

        # --------------------------------------------------
        # Replace local metadata file after updates
        # --------------------------------------------------
        if updates_perf >= 1:
            hf.download_file(
                url=ghub_meta_url,
                save_location=metadata_file,
                gui_logger=gui_logger
            )
            hf.log_with_timestamp("Local metadata updated to latest version.", gui_logger)

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------
        if mismatches == 0:
            hf.log_with_timestamp("No updates available", gui_logger)

        # --------------------------------------------------
        # NEW: Save timestamp of update check
        # --------------------------------------------------
        last_checked_path = pjoin(csv_directory, "last_checked_updates.json")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(last_checked_path, "w", encoding="utf-8") as f:
            json.dump({"last_checked": ts}, f, indent=4)

    except Exception as ex:
        hf.log_with_timestamp(
            log_string="Failed to validate versions or update data",
            gui_logger=gui_logger,
            log_type=2,
            exception=ex,
        )

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    