# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 14:05:45 2024

@author: Shanon
"""

import numpy as np
from os.path import join as pjoin
import os
from ash_toolset import constants as CN
from ash_toolset import helper_functions as hf
from ash_toolset import hrir_processing
from pathlib import Path
import scipy as sp
import logging
from csv import DictReader, DictWriter
import noisereduce as nr
import h5py
import soundfile as sf
import time
from scipy.io import loadmat
import csv
import glob
from typing import Any, Dict, Optional, Tuple, List, Callable
from scipy.io.matlab import mat_struct
import json
from datetime import datetime
import matplotlib.pyplot as plt
import tempfile

logger = logging.getLogger(__name__)
log_info=1

    
def extract_ir_from_struct(obj, gui_logger=None, fallback_sr=48000):
    ir_keys = {'impulseresponse', 'rir', 'ir', 'bformat', 'h', 'w', 'x', 'y', 'z'}
    sr_keys = {'fs', 'samplingrate', 'samplerate', 'sr'}
    ignore = {'raw', 'meta', 'info', 'calibration', 'config', 'header', 'sweep', 'measurement', 'comments'}

    def log(msg, level="INFO"):
        if gui_logger:
            hf.log_with_timestamp(f"[{level}] {msg}", gui_logger)

    def walk(item, key_path=""):
        # ... (Existing HDF5/Dict/mat_struct logic) ...
        if isinstance(item, (h5py.Group, h5py.File)):
            for k in item.keys(): yield from walk(item[k], k)
        elif isinstance(item, h5py.Dataset):
            yield key_path, item[()]
        elif isinstance(item, dict):
            for k, v in item.items(): yield from walk(v, k)
        elif isinstance(item, mat_struct):
            for k in item._fieldnames: yield from walk(getattr(item, k), k)
        
        # --- Handle NumPy Arrays (Crucial for the 6x2 Object Array) ---
        elif isinstance(item, np.ndarray):
            # A. Structured arrays
            if item.dtype.names:
                for k in item.dtype.names:
                    yield from walk(item[k], f"{key_path}.{k}")
            
            # B. Object/Cell Arrays (This handles your 6x2 grid)
            elif item.dtype == object:
                for idx, val in np.ndenumerate(item):
                    # idx will be (row, col) - e.g., (0, 0), (0, 1)...
                    yield from walk(val, f"{key_path}{list(idx)}")
            
            # C. Standard Numeric Data
            else:
                yield key_path, item

        elif isinstance(item, (list, tuple)) and len(item) > 0:
            for idx, i in enumerate(item): yield from walk(i, f"{key_path}[{idx}]")
        else:
            yield key_path, item

    log("Scanning structure...")
    candidates = list(walk(obj))
    found_irs = []
    found_srs = []

    # --- Step 1: Extract Sample Rate ---
    for k, v in candidates:
        try:
            # np.array(v).item() handles h5py 0-dim datasets and numpy scalars
            val_float = float(np.array(v).item())
            if 1000 <= val_float <= 200000:
                found_srs.append({'val': int(val_float), 'pref': k.lower() in sr_keys})
        except:
            continue
    
    found_srs.sort(key=lambda x: x['pref'], reverse=True)
    temp_sr = found_srs[0]['val'] if found_srs else fallback_sr
    inferred = not any(s['pref'] for s in found_srs) if found_srs else True

    # --- Step 2: Extract IRs ---
    for k, v in candidates:
        k_low = k.lower()
        if any(ig in k_low for ig in ignore): continue

        arr = np.array(v)
        
        # This will catch the 2nd column arrays but ignore the 1st column strings
        if np.issubdtype(arr.dtype, np.floating) and arr.size > 1:
            # We treat the array as a whole (even if it's 2 x Samples)
            duration_samples = max(arr.shape)
            is_reasonable = duration_samples < (temp_sr * 12) 

            found_irs.append({
                'val': arr, 
                'key': k, 
                'pref': any(ik in k_low for ik in ir_keys), 
                'reasonable': is_reasonable,
                'size': arr.size
            })

    # --- Step 3: Final Selection ---
    if not found_irs:
        raise ValueError("No valid Room Impulse Response found.")

    # Sort by heuristic: Reasonable > Preferred Name > Size
    found_irs.sort(key=lambda x: (x['reasonable'], x['pref'], x['size']), reverse=True)
    
    # 2. B-format check (Keeping this as it's a specific reconstruction case)
    b_map = {c['key'].upper(): c['val'] for c in found_irs 
             if c['key'].upper() in {'W', 'X', 'Y', 'Z'} and c['reasonable']}
    
    if len(b_map) == 4:
        final_ir = np.stack([np.squeeze(b_map[ch]) for ch in ['W', 'X', 'Y', 'Z']])
        log("Constructed B-Format IR from components.")
    else:
        # 2. Check if the best match is part of an object array/list
        best_match = found_irs[0]
        best_key = best_match['key']
        
        if "[" in best_key:
            base_name = best_key.split('[')[0]
            siblings = [c for c in found_irs if c['key'].startswith(base_name) and c['reasonable']]
            siblings.sort(key=lambda x: x['key'])
            
            if len(siblings) > 1:
                processed_siblings = []
                for s in siblings:
                    arr = s['val']
                    # Force to 2D: (Samples,) becomes (1, Samples)
                    if arr.ndim == 1:
                        arr = arr[np.newaxis, :]
                    # If it's (Samples, Channels), transpose it to (Channels, Samples)
                    elif arr.shape[0] > arr.shape[1] and arr.shape[1] < 10:
                        arr = arr.T
                        
                    processed_siblings.append(arr)
                
                # Join along the channel axis (0)
                final_ir = np.concatenate(processed_siblings, axis=0)
                log(f"Combined {len(siblings)} IRs. Final shape: {final_ir.shape}")
            else:
                final_ir = np.squeeze(best_match['val'])
        else:
            final_ir = np.squeeze(best_match['val'])
  
    
    log(f"Selected IR from index '{best_match['key']}'. Shape: {final_ir.shape}")
    

    return final_ir, temp_sr, inferred
    
  



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

    all_files = sorted([
        f for ext in supported_exts
        for f in glob.glob(os.path.join(folder_path, '**', f'*{ext}'), recursive=True)
    ])

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
                try:
                    # 1. Try standard SciPy load (v7 and older)
                    data = loadmat(path, squeeze_me=True, struct_as_record=False)
                    hf.log_with_timestamp(f"Loaded legacy MAT: {os.path.basename(path)}", gui_logger)
                except (NotImplementedError, ValueError):
                    # 2. Fallback for v7.3 (HDF5 format)
                    hf.log_with_timestamp(f"Detected v7.3 MAT (HDF5), attempting h5py load...", gui_logger)
                    # We open in 'r' mode. Note: extract_ir_from_struct must handle dict-like HDF5 objects
                    data = h5py.File(path, 'r')
                ir, samplerate, inferred_sr = extract_ir_from_struct(data, gui_logger=None)
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
                hf.log_with_timestamp(f"Loaded ir '{os.path.basename(path)}' with shape {ir.shape}", gui_logger)
                #ir = hf.reshape_array_to_two_dims(ir)
                ir = hf.reshape_array_to_two_dims_preserve_pairs(ir)
                hf.log_with_timestamp(f"Reshaped ir '{os.path.basename(path)}' to shape {ir.shape}", gui_logger)

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
                ir = hf.reshape_array_to_two_dims_preserve_pairs(ir)
                hf.log_with_timestamp("Reshaped multidimensional IR", gui_logger)
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
    desired_measurements=3000, noise_reduction_mode=False, remove_direct=True,
    pitch_range=(0, 12),
    tail_mode="short",  # options: "short", "long", "short windowed" 
    window_type="hanning",  # new parameter: "hanning" or "triangle"
    report_progress=0, cancel_event=None, f_alignment = 0, pitch_shift_comp=CN.AS_PS_COMP_LIST[1], spatial_exp_method=CN.AS_SPAT_EXP_LIST[1],
    ignore_ms=CN.IGNORE_MS, subwoofer_mode=False, binaural_mode=False, air_data=None, correction_factor=1.0
):
    """
    Loads and processes individual acoustic IR files from a dataset folder, extracting the impulse responses
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
        tail_mode (str): Tail handling mode.
        "short"          → n_fft = CN.N_FFT
        "long"           → n_fft = CN.N_FFT_L
        "short windowed" → n_fft = CN.N_FFT with Hann fade-out applied to tail

    Returns:
       Status code: 0 = Success, 1 = Failure, 2 = Cancelled
    """
    status=1
    

    # set fft length and tail handling mode
    tail_mode = tail_mode.lower()
    if tail_mode == "long":
        n_fft = CN.N_FFT_L
        apply_tail_fade = False
    elif tail_mode == "long windowed":
        n_fft = CN.N_FFT_L
        apply_tail_fade = True
    elif tail_mode == "short windowed":
        n_fft = CN.N_FFT
        apply_tail_fade = True
    else:  # "short" (default)
        n_fft = CN.N_FFT
        apply_tail_fade = False

    if input_folder is None:
        input_folder=ir_set

    #set noise reduction based on AC space
    noise_reduction=noise_reduction_mode
        
    samp_freq_ash=CN.SAMP_FREQ
    # --- Dataset state (arrays only, never None) ---
    air_data = np.asarray(air_data) if air_data is not None else np.array([])
    ir_min_threshold=CN.IR_MIN_THRESHOLD

    
    if subwoofer_mode == True:
        f_norm_start = 30#normalise at low frequencies
        f_norm_end = 140
    else:
        f_norm_start = CN.SPECT_SNAP_F0#use mid frequencies
        f_norm_end = CN.SPECT_SNAP_F1
    # Convert target frequency range to rfft bins
    fb_start = int(f_norm_start * n_fft / samp_freq_ash)
    fb_end = int(f_norm_end * n_fft / samp_freq_ash)
    
    # windows
    direct_hanning_size = 200
    direct_hanning_start = 61
    hann_direct = np.hanning(direct_hanning_size)[:direct_hanning_size // 2]
    direct_removal_win = np.ones(n_fft)
    direct_removal_win[:direct_hanning_start] = 0
    direct_removal_win[direct_hanning_start:direct_hanning_start + direct_hanning_size // 2] = hann_direct
    
    # tail fade-out window (used for short_windowed mode)
    if apply_tail_fade:
        if subwoofer_mode:
            fade_start = int(0.06 * n_fft)
        else:
            fade_start = int(0.25 * n_fft)
        fade_len = n_fft - fade_start
    
        if window_type.lower() == "hanning":
            tail_win_full = np.hanning(fade_len * 2)
            tail_fade = tail_win_full[fade_len:]  # second half
        elif window_type.lower() == "triangle":
            tail_fade = np.linspace(1, 0, fade_len)
        elif window_type.lower() == "cosine":
            # Cosine fade: cos(0) = 1 → cos(pi/2) = 0
            tail_fade = np.cos(np.linspace(0, np.pi / 2, fade_len))
        else:
            raise ValueError(f"Unsupported window_type for tail fade: {window_type}")
            
        # --- Strengthen window for subwoofer mode ---
        window_power = 3 if subwoofer_mode else 1
        tail_fade = tail_fade ** window_power

    
    #input folder
    ir_data_folder = pjoin(CN.DATA_DIR_IRS_USER,input_folder)
    
    #loop through folders
    try:
        
        # --------------------------------------------------
        # Load or use provided AIR data
        # --------------------------------------------------
        if air_data.size == 0:
            # Normal ETL path
            log_string_a = 'Loading IRs from: ' + ir_set
            hf.log_with_timestamp(log_string_a, gui_logger)
        
            air_data, status_code = etl_airs(
                folder_path=ir_data_folder,
                file_type=None,
                gui_logger=gui_logger,
                project_samplerate=samp_freq_ash,
                normalize=True,
                noise_reduction=noise_reduction,
                noise_tail_ratio=0.2,
                max_measurements=CN.MAX_IRS,
                max_samples=n_fft,
                cancel_event=cancel_event
            )
        
            if status_code > 0:
                return air_data, status_code
        
            if cancel_event and cancel_event.is_set():
                gui_logger.log_warning("Operation cancelled by user.")
                return air_data, 2
        
        else:
            # Bypass ETL — use provided data
            hf.log_with_timestamp("Using provided AIR dataset (ETL bypassed).",gui_logger)
            air_data = np.asarray(air_data)
            status_code = 0
            
        if air_data.size == 0:
            hf.log_with_timestamp(
                "Invalid or empty IR dataset.", 
                log_type=2, 
                gui_logger=gui_logger
            )
            return air_data, 1
        
        total_measurements = air_data.shape[0]
        log_string_a = 'Total measurements before expansion: '+str(total_measurements)
        hf.log_with_timestamp(log_string_a, gui_logger)

        # Shift raw IRs so that direct peak is at sample x
        index_peak_ref = 40#30  # Target alignment point
        if binaural_mode:
            step = 2  # L/R pair
        else:
            step = 1  # Mono
        for idx in range(0, total_measurements, step):
            # Determine the reference peak for the first channel in the pair
            index_peak_cur = np.argmax(np.abs(air_data[idx, :]))
            ir_shift = index_peak_ref - index_peak_cur
            # Apply the same shift to current channel
            air_data[idx, :] = np.roll(air_data[idx, :], ir_shift)
            if ir_shift < 0:
                air_data[idx, ir_shift:] = 0  # Zero out tail after shift
            # If binaural, also apply the same shift to the next channel in the pair
            if binaural_mode and (idx + 1 < total_measurements):
                air_data[idx + 1, :] = np.roll(air_data[idx + 1, :], ir_shift)
                if ir_shift < 0:
                    air_data[idx + 1, ir_shift:] = 0
         
        

        #section to expand dataset in cases where few input IRs are available
        #if total IRs below threshold #total_measurements < ir_min_threshold   total_measurements < desired_measurements
        if total_measurements < desired_measurements and subwoofer_mode==False and binaural_mode == False and spatial_exp_method != CN.AS_SPAT_EXP_LIST[0]:
      
            log_string_a = 'Expanding IR dataset'
            align_start_time = time.time()
            hf.log_with_timestamp(log_string_a, gui_logger)
            #expand dataset by creating new IRs using pitch shifting
            air_data, status_code = hf.expand_measurements(
                measurement_array=air_data,
                desired_measurements=desired_measurements,pitch_range=pitch_range,gui_logger=gui_logger,
                cancel_event=cancel_event,report_progress=report_progress, pitch_shift_comp=pitch_shift_comp,ignore_ms=ignore_ms,pitch_shift_mode=spatial_exp_method
            ) 
            if status_code > 0:#failure or cancelled
                return air_data, status_code
            align_end_time = time.time()
            log_string_a = f"Dataset expansion completed in {align_end_time - align_start_time:.2f} seconds."
            hf.log_with_timestamp(log_string_a)

        # --- Expand dataset for subwoofer mode if below threshold ---
        elif subwoofer_mode:
 
            if total_measurements < CN.IR_MIN_THRESHOLD_DUPLICATE:
     
                log_string_a = (
                    f"Low-frequency mode: duplicating {total_measurements} measurements "
                    f"to reach minimum threshold of {CN.IR_MIN_THRESHOLD_DUPLICATE}"
                )
                hf.log_with_timestamp(log_string_a, gui_logger)
            
                # Compute the smallest integer multiple that meets/exceeds the threshold
                n_repeat = int(np.ceil(CN.IR_MIN_THRESHOLD_DUPLICATE / total_measurements))
            
                # Duplicate measurements exactly n_repeat times
                air_data = np.tile(air_data, (n_repeat, 1))
            
                # Trim to exact multiple of original measurements
                final_count = total_measurements * n_repeat
                air_data = air_data[:final_count, :]
            
                total_measurements = air_data.shape[0]
                log_string_a = (
                    f"Low-frequency dataset extended by integer repetition. "
                    f"Total measurements: {total_measurements} "
                    f"(original count: {total_measurements // n_repeat}, repeated {n_repeat} times)"
                )
                hf.log_with_timestamp(log_string_a)

        total_measurements = air_data.shape[0]
        log_string_a = 'Total measurements after expansion: '+str(total_measurements)
        hf.log_with_timestamp(log_string_a, gui_logger)

        # Normalize each AIR in the spectral domain to 0db reference
        for idx in range(0, total_measurements, step):
            data_fft_L = np.fft.rfft(air_data[idx, :])
            mag_L = np.abs(data_fft_L)
            average_mag = np.mean(mag_L[fb_start:fb_end])
            if average_mag > 0:
                air_data[idx, :] /= average_mag
                if binaural_mode:
                    air_data[idx + 1, :] /= average_mag        
                
                
        if cancel_event and cancel_event.is_set():
            gui_logger.log_warning("Operation cancelled by user.")
            return air_data, 2
      
        # Synchronize in time domain
        if total_measurements >= 2:
            order = CN.ORDER
            
            #which alignment frequency to use?
            if f_alignment <= 0:
                cutoff_alignment = CN.CUTOFF_ALIGNMENT_AIR
            else:
                cutoff_alignment=f_alignment
           
            # ---- Alignment parameters ----
            shifts, win_starts, win_ends,peak_to_peak_window = compute_lf_alignment_params(
                samp_freq=samp_freq_ash,
                cutoff_hz=cutoff_alignment
            )
       
            lp_sos = hf.get_filter_sos(cutoff=cutoff_alignment, fs=CN.FS, order=order, filtfilt=CN.FILTFILT_TDALIGN_AIR, b_type='low')
         
            align_start_time = time.time()
            log_string_a = f"Starting time-domain alignment for {total_measurements} impulse responses..."
            hf.log_with_timestamp(log_string_a, gui_logger)

            print_interval = 100  # Print every 100 IRs (adjust as needed)

            for idx in range(0, total_measurements, step):
                
                start_sample=220#170,190,200
                lookahead_samples = start_sample + int(0.8*CN.FS/cutoff_alignment)#1.15
                align_ir_static_window(idx=idx, air_data=air_data, shifts=shifts, lp_sos=lp_sos, binaural_mode=binaural_mode, start_sample=start_sample, lookahead_samples=lookahead_samples)
       
                # --- Progress logging ---
                if (idx + step) % print_interval == 0 or idx + step >= total_measurements:
                    if report_progress > 0:
                        a_point, b_point = 0.35, 0.6
                        progress = a_point + (float(idx) / total_measurements) * (b_point - a_point)
                        hf.update_gui_progress(report_progress, progress=progress)
                    hf.log_with_timestamp(f"Aligned {min(idx + step, total_measurements)} of {total_measurements} IRs.", gui_logger)
                    if cancel_event and cancel_event.is_set():
                        gui_logger.log_warning("Operation cancelled by user.")
                        return air_data, 2
            
            align_end_time = time.time()
            log_string_a = f"Time-domain alignment completed in {align_end_time - align_start_time:.2f} seconds."
            log_string_b = "Time-domain alignment completed"
            hf.log_with_timestamp(log_string_a)
            hf.log_with_timestamp(log_string_b, gui_logger)
        
  
    
        #remove direction portion of signal
        if remove_direct:
            air_data *= direct_removal_win
            
        # apply fade-out to tail if enabled
        if apply_tail_fade:
            for idx in range(total_measurements):
                air_data[idx, fade_start:] *= tail_fade
                       
        #
        #set each AIR to 0 level after removing direct portion
        log_string_a = 'Adjusting Levels'
        hf.log_with_timestamp(log_string_a, gui_logger)
 
        for idx in range(0, total_measurements, step):
            data_fft_L = np.fft.rfft(air_data[idx, :])
            mag_L = np.abs(data_fft_L)
            average_mag = np.mean(mag_L[fb_start:fb_end])
            if average_mag > 0:
                air_data[idx, :] /= average_mag
                if binaural_mode:
                    air_data[idx + 1, :] /= average_mag 
                    
        # --------------------------------------------------
        # DEBUG: Mean AIR frequency response (overview)
        # --------------------------------------------------
        if CN.PLOT_ENABLE:
            try:
                # Use mono stepping if binaural
                step_dbg = 2 if binaural_mode else 1
                mags_db = []
                for idx in range(0, total_measurements, step_dbg):
                    x = air_data[idx, :]
                    # rFFT
                    X = np.fft.rfft(x)
                    mag = np.abs(X)
                    # Avoid log(0)
                    mag = np.maximum(mag, 1e-12)
                    # Convert to dB
                    mag_db = 20.0 * np.log10(mag)
                    mags_db.append(mag_db)
                mags_db = np.asarray(mags_db)  # shape: (N_meas, N_bins)
                # Mean in dB domain
                mean_mag_db = np.mean(mags_db, axis=0)
                # Frequency axis
                freqs = np.fft.rfftfreq(len(x), d=1.0 / samp_freq_ash)
                # Plot
                plt.figure(figsize=(9, 4))
                plt.semilogx(freqs, mean_mag_db, label="Mean AIR magnitude (dB)")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Magnitude (dB)")
                plt.title("Mean AIR Frequency Response (rFFT)")
                plt.grid(True, which="both", ls="--", alpha=0.4)
                plt.xlim(20, samp_freq_ash / 2)
                plt.legend()
                plt.tight_layout()
                plt.show()
                hf.log_with_timestamp("Plotted mean AIR frequency response (debug)", gui_logger)

            except Exception as e:
                print(f"[DEBUG] Mean AIR spectrum plot failed: {e}")
   
        status=0#success if reached this far
        
        log_string_a = 'Prepared IR Dataset'
        hf.log_with_timestamp(log_string_a, gui_logger)
    
   
    except Exception as ex:
        log_string = 'Failed to complete IR set processing for: ' + ir_set 
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
  
    return air_data, status








def convert_airs_to_brirs(
    ir_set='default_set',
    air_dataset=np.array([]),
    gui_logger=None, spatial_res=3, hrir_dataset=None, correction_factor=1.0, remove_direct=True, as_listener_type=CN.AS_LISTENER_TYPE_LIST[0],
    tail_mode="short", biased_centers=None, azimuth_spread = 48, distr_mode=CN.AS_DISTR_MODE_LIST[0], f_alignment = 100, lf_drr_comp=True, 
    report_progress=0, cancel_event=None,  rise_time=5.1, subwoofer_mode=False, binaural_mode=False, auto_shape_output=False, virtual_speakers=7, octave_smoothing_n=3
):
    """
    Convert Acoustic Impulse Responses (AIRs) into Binaural Room Impulse Responses (BRIRs).

    Processes AIRs by convolving them with HRIRs or in binaural mode, applies optional 
    direct sound removal, tail handling, and magnitude compensation, and outputs a 
    BRIR dataset suitable for playback or further processing.

    Parameters
    ----------
    ir_set : str
        Name of the IR set (e.g., room identifier).
    air_dataset : np.ndarray, optional
        AIR measurements array (num_measurements × num_samples). Loaded from disk if empty.
    gui_logger : callable, optional
        Logger for progress messages.
    spatial_res : int, default=3
        HRIR spatial resolution.
    hrir_dataset : np.ndarray, optional
        Preloaded HRIRs. Defaults are used if None.
    correction_factor : float
        Scale factor for magnitude compensation.
    remove_direct : bool
        Apply fade-in window to reduce direct sound.
    tail_mode : str
        How to handle BRIR tail: 'short', 'long', 'short windowed'.
    distr_mode : str, default="Sequential"
        Measurement distribution mode:
        - "Sequential": Split measurements into contiguous blocks.
        - "Round-Robin": Distribute measurements one-by-one across sources.
        - "Random": Shuffle measurements before distributing.
        - "Spread": Select evenly spaced measurements across the dataset.
    rise_time : float
        Rise time in ms for initial fade-in.
    subwoofer_mode : bool
        Simplified output for subwoofer-only response.
    binaural_mode : bool
        Alternate left/right contributions instead of using HRIRs.
    auto_shape_output : bool
        Automatically determine number of BRIR sources.

    Returns
    -------
    brir_reverberation : np.ndarray
        Generated BRIR dataset (sources × channels × samples).
    status : int
        Result code: 0=success, 1=failure, 2=cancelled.
    """
    status=1
    
    # set fft length and tail handling mode
    tail_mode = tail_mode.lower()
    if tail_mode == "long" or tail_mode == "long windowed":
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
    # If rise_samples is too small or user wants to retain direct sound, skip Hann window and apply instant transition
    if rise_samples < 2 or remove_direct == False:
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
            hf.log_with_timestamp(f"Unsupported input IR measurement array shape: {air_dataset.size}", gui_logger)
            return brir_reverberation, 1
        
        # Ensure AIRs are long enough for convolution / FFT window
        curr_air_len = air_dataset.shape[1]
        required_air_len = n_fft
        if curr_air_len < required_air_len:
            pad_len = required_air_len - curr_air_len
            hf.log_with_timestamp(f"Zero-padding AIR dataset from {curr_air_len} to {required_air_len} samples",gui_logger)
            air_dataset = np.pad(air_dataset,pad_width=((0, 0), (0, pad_len)),mode="constant")
        
        #get number of sets
        total_measurements = air_dataset.shape[0]

        #
        # Load HRIR dataset (user-provided or fallback to defaults)
        if hrir_dataset is not None and isinstance(hrir_dataset, np.ndarray) and hrir_dataset.size > 0:
            hrir_list = np.asarray(hrir_dataset, dtype=np.float64)
            hf.log_with_timestamp(f"Validating HRIR dataset, shape: {hrir_list.shape}", gui_logger)
        else:
            raise ValueError("Invalid HRIR array provided")
                    
        
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
            
        # Infer / validate spatial resolution with fallback provided
        spatial_res = hrir_processing.infer_hrir_spatial_res(hrir_selected,spatial_res=spatial_res,logger=gui_logger)
        hf.log_with_timestamp(f"HRIR grid: elev={hrir_selected.shape[0]}, azim={hrir_selected.shape[1]}")
        
        if hrir_selected.shape[2] != 2:
            raise ValueError(
                f"HRIR dataset must have exactly 2 channels (L/R), "
                f"got {hrir_selected.shape[2]}"
            )
        total_chan_hrir = len(hrir_selected[0][0])
        if hrir_selected.ndim != 4:
            raise ValueError("HRIR dataset must be 4D after normalization.")
        if hrir_selected.shape[-1] < 16:
            raise ValueError("HRIRs are too short to be valid.")
        
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
        if auto_shape_output:#derive based on number of measurements
            if subwoofer_mode == True:#only 1 set required if subwoofer response
                num_brir_sources=1
            elif total_measurements < CN.IR_MIN_THRESHOLD_FULLSET:#limited number of measurements
                num_brir_sources = max(2, total_measurements // 250)#calculate one source per 250 measurements
            else:
                num_brir_sources = max(2, total_measurements // 360)#calculate one source per 360 measurements
            
        else:#manual shape mode -> use when called by AS import tool
            num_brir_sources = virtual_speakers
   
        log_string_a = 'num_brir_sources: ' + str(num_brir_sources)
        hf.log_with_timestamp(log_string_a)
        
        # Create numpy array for new BRIR dataset
        brir_reverberation = np.zeros((CN.INTERIM_ELEVS, num_brir_sources, 2, n_fft))
        
        # Log start
        hf.log_with_timestamp("Estimating BRIRs from IRs", gui_logger)
        
        # Sampling spatial coordinates with bias
        num_spatial_samples = total_measurements
        if biased_centers is None or len(biased_centers) == 0:
            biased_centers = np.array([40, 140, 220, 320])
        
        azimuths_distribution, elevations_distribution = hf.biased_spherical_coordinate_sampler(
            azim_src_set, elev_src_set, num_spatial_samples,
            biased_azimuth_centers=biased_centers,
            azimuth_spread=azimuth_spread,
            plot_distribution=False
        )
        # --- GUARANTEED BINAURAL-SAFE RANDOM DISTRIBUTION ---
        if binaural_mode and total_measurements % 2 != 0:
            raise ValueError(
                "Binaural mode requires an even number of AIR measurements "
                "(interleaved L/R pairs)."
            )
        # Standardize string input to handle case sensitivity
        dist_mode_clean = distr_mode.capitalize() 
        # --- MEASUREMENT DISTRIBUTION LOGIC ---
        if dist_mode_clean == "Sequential":
            if binaural_mode:
                num_pairs = total_measurements // 2
                pair_indices = np.column_stack((np.arange(0, total_measurements, 2), np.arange(1, total_measurements, 2)))
                num_sources_eff = min(num_brir_sources, num_pairs)
                pair_groups = np.array_split(pair_indices, num_sources_eff)
                measurements_per_source = [group.reshape(-1) for group in pair_groups]
            else:
                measurements_per_source = np.array_split(np.arange(total_measurements), num_brir_sources)
    
        elif dist_mode_clean == "Round-robin":
            if binaural_mode:
                num_pairs = total_measurements // 2
                pair_indices = np.column_stack((np.arange(0, total_measurements, 2), np.arange(1, total_measurements, 2)))
                num_sources_eff = min(num_brir_sources, num_pairs)
                pair_groups = [[] for _ in range(num_sources_eff)]
                for i, pair in enumerate(pair_indices):
                    pair_groups[i % num_sources_eff].append(pair)
                measurements_per_source = [np.asarray(group).reshape(-1) for group in pair_groups]
            else:
                measurements_per_source = [[] for _ in range(num_brir_sources)]
                for i in range(total_measurements):
                    measurements_per_source[i % num_brir_sources].append(i)
    
        elif dist_mode_clean == "Random":
            if binaural_mode:
                num_pairs = total_measurements // 2
                pair_indices = np.column_stack((np.arange(0, total_measurements, 2), np.arange(1, total_measurements, 2)))
                np.random.shuffle(pair_indices)
                num_sources_eff = min(num_brir_sources, num_pairs)
                pair_groups = np.array_split(pair_indices, num_sources_eff)
                measurements_per_source = [group.reshape(-1) for group in pair_groups]
            else:
                shuffled_indices = np.random.permutation(total_measurements)
                measurements_per_source = np.array_split(shuffled_indices, num_brir_sources)
    
        elif dist_mode_clean == "Single":
            # --- ENFORCE 1 MEASUREMENT (OR PAIR) PER SOURCE ---
            if binaural_mode:
                num_pairs = total_measurements // 2
                # Use as many sources as requested, up to the total pairs available
                num_sources_eff = min(num_brir_sources, num_pairs)
                pair_indices = np.column_stack((np.arange(0, total_measurements, 2), np.arange(1, total_measurements, 2)))
                
                # Sample evenly across the dataset timeline
                pair_sel_idx = np.linspace(0, num_pairs - 1, num_sources_eff, dtype=int)
                selected_pairs = pair_indices[pair_sel_idx]
                measurements_per_source = [pair.reshape(-1) for pair in selected_pairs]
            else:
                num_sources_eff = min(num_brir_sources, total_measurements)
                sel_idx = np.linspace(0, total_measurements - 1, num_sources_eff, dtype=int)
                measurements_per_source = [np.asarray([i]) for i in sel_idx]
            
        else:
            valid_options = CN.AS_DISTR_MODE_LIST
            raise ValueError(f"Invalid distr_mode: '{distr_mode}'. Expected one of {valid_options}")
        
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
                # Copy AIR, wrap around if fewer measurements than binaural sources (rare case)
                curr_air = np.copy(air_dataset[measurement % total_measurements, :])
        
                if binaural_mode:
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
                    
                    if curr_elev_id >= hrir_selected.shape[0] or curr_azim_id >= hrir_selected.shape[1]:
                        hf.log_with_timestamp(f"HRIR index out of range (elev={curr_elev_id}, azim={curr_azim_id}); skipping",gui_logger)
                        continue
        
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
            if binaural_mode:
                left_count = (len(group) + 1) // 2
                right_count = len(group) // 2
                
                brir_reverberation[0, source_num, 0, :] = brir_sum_l / max(1, left_count)
                brir_reverberation[0, source_num, 1, :] = brir_sum_r / max(1, right_count)
            else:
                brir_reverberation[0, source_num, 0, :] = brir_sum_l / len(group)
                brir_reverberation[0, source_num, 1, :] = brir_sum_r / len(group)

        # DEBUG TAP: raw BRIR (pre-window, pre-normalisation)
        debug_write_brir("00_raw_brir", brir_reverberation, samp_freq_ash)

        # Window initial rise
        if remove_direct:
            brir_reverberation[0, :, :, :] *= direct_removal_win  # broadcasted multiply over channel axis
        
        # Window tail end of BRIR
        fade_start = int( int(((CN.RT60_MAX_S)/1000)*CN.FS) * (n_fft / CN.N_FFT)) 
        l_fade_win_size = abs(fade_start - n_fft) * 2
        hann_l_fade_full = np.hanning(l_fade_win_size)
        win_l_fade_out = np.split(hann_l_fade_full, 2)[1]  # second half
        l_fade_out_win = np.ones(n_fft)
        l_fade_out_win[fade_start:] = win_l_fade_out
        brir_reverberation[0, :, :, :] *= l_fade_out_win  # broadcasted multiply
        
        #set to 0 level at 1khz
        brir_reverberation = hf.normalize_brir_band(ir_data=brir_reverberation,n_fft=n_fft,fs=CN.FS,f_norm_start=CN.SPECT_SNAP_F0,f_norm_end=CN.SPECT_SNAP_F1)
        
        comp_win_size = 7
        # Level ends of spectrum
        high_freq = 12000 if subwoofer_mode == True else 15000
        low_freq_in = 20
        low_freq_in_target = 150 if subwoofer_mode == True else 90#make target level in sub frequencies
        # Load target
        target_path = pjoin(CN.DATA_DIR_INT, 'reverberation', 'reverb_target_mag_response.npy')
        brir_fft_target_mag = np.load(target_path)
        brir_fft_target_mag = hf.level_spectrum_ends(brir_fft_target_mag, low_freq_in_target, high_freq, smooth_win=comp_win_size)#20
        
        if cancel_event and cancel_event.is_set():
            gui_logger.log_warning("Operation cancelled by user.")
            return brir_reverberation, 2
        
        # DEBUG TAP: BRIR (window, normalisation)
        debug_write_brir("01_wind_brir", brir_reverberation, samp_freq_ash)
        
        #integrate boosted low frequency response to compensate for increase DRR in low frequencies
        comp_lf_response_drr=False
        if subwoofer_mode == False and binaural_mode == False and lf_drr_comp:

            # Capture mean response before integrating boosted response
            # Collect ALL BRIRs across directions and channels
            # Shape: (num_directions * 2, N_FFT)
            if comp_lf_response_drr:
                brirs_all = brir_reverberation[0, :, :, :CN.N_FFT].reshape(-1, CN.N_FFT)
                # FFT
                brir_fft = np.fft.fft(brirs_all, axis=-1)
                brir_mag = np.abs(brir_fft)
                brir_db = hf.mag2db(brir_mag)
                # Global average (directions + channels)
                brir_fft_avg_db = np.mean(brir_db, axis=0)   
                # Smooth + shape average response
                brir_fft_avg_mag = hf.db2mag(brir_fft_avg_db)
                brir_fft_avg_mag = hf.level_spectrum_ends(brir_fft_avg_mag,low_freq_in,high_freq,smooth_win=comp_win_size)
                brir_fft_avg_mag_sm_pre = hf.smooth_gaussian_octave(data=brir_fft_avg_mag,n_fft=CN.N_FFT,fraction=octave_smoothing_n)
    
     
            
            
            # --- Iterative Correction Loop ---
            max_iterations = 2
            tolerance = 0.15 # Aim for less than X% error
            delay_samples=int(0.015 * CN.FS)
            fade_len_samples=int(0.04 * CN.FS)#0.05
            
            
            for i in range(max_iterations):
                # 1. Measure current DRR
                results = calculate_relative_drr_change(
                    brir_reverberation=brir_reverberation,
                    air_dataset=air_dataset,
                    crossover_hz=f_alignment,
                    fade_len_samples=fade_len_samples,delay_samples=delay_samples
                )
                
                current_error = results['relative_change']
                hf.log_with_timestamp(f"Iteration {i+1}: Error is {current_error * 100:.2f}%")
                
                # If error is small enough, break
                if abs(current_error) < tolerance:
                    hf.log_with_timestamp("DRR target reached.")
                    break
                    
                # 2. Calculate Correction Mix
                # We multiply the current ratio by the inverse of the error to "nudge" it
                # Note: Because this version is complementary, we use the ratio of ratios
                step_mix = results['brir_ratio'] / (results['air_ratio'] + 1e-10)
                
                # 3. Apply correction
                brir_reverberation = adjust_lf_drr(
                    brir_reverberation, 
                    lf_mix=step_mix, 
                    fade_len_samples=fade_len_samples, 
                    crossover_hz=f_alignment,delay_samples=delay_samples
                )
 
    
    
    
            if comp_lf_response_drr:
                # Capture mean response after integrating boosted response
                # Collect ALL BRIRs across directions and channels
                # Shape: (num_directions * 2, N_FFT)
                brirs_all = brir_reverberation[0, :, :, :CN.N_FFT].reshape(-1, CN.N_FFT)
                # FFT
                brir_fft = np.fft.fft(brirs_all, axis=-1)
                brir_mag = np.abs(brir_fft)
                brir_db = hf.mag2db(brir_mag)
                # Global average (directions + channels)
                brir_fft_avg_db = np.mean(brir_db, axis=0)   
                # Smooth + shape average response
                brir_fft_avg_mag = hf.db2mag(brir_fft_avg_db)
                brir_fft_avg_mag = hf.level_spectrum_ends(brir_fft_avg_mag,low_freq_in,high_freq,smooth_win=comp_win_size)
                brir_fft_avg_mag_sm_post = hf.smooth_gaussian_octave(data=brir_fft_avg_mag,n_fft=CN.N_FFT,fraction=octave_smoothing_n)
                
                #calc comp filter
                # Compensation in dB
                comp_db = (hf.mag2db(brir_fft_avg_mag_sm_pre)- hf.mag2db(brir_fft_avg_mag_sm_post))
                # Back to magnitude
                comp_mag = hf.db2mag(comp_db)
                comp_mag = hf.smooth_gaussian_octave(data=comp_mag, n_fft=CN.N_FFT, fraction=octave_smoothing_n)#12
                
                comp_eq_fir = hf.build_min_phase_filter(comp_mag,truncate_len=4096)
                for direc in range(num_brir_sources): 
                    # Equalize both channels (loop still needed for convolution)
                    for chan in range(total_chan_hrir):
                        eq_brir = sp.signal.convolve(brir_reverberation[0, direc, chan, :], comp_eq_fir, 'full', 'auto')
                        brir_reverberation[0, direc, chan, :] = eq_brir[:n_fft]
                        
                        
                # comp_db = hf.mag2db(comp_mag)
                # #parametric EQ step:
                # brir_reverberation = hf.equalize_brirs_parametric(brir_reverberation, diff_db_override = comp_db, override_n_fft = CN.N_FFT)
            
                # ------------------------------------------------------------
                # Optional plots (GLOBAL)
                # ------------------------------------------------------------
                if CN.PLOT_ENABLE:
                    hf.plot_data(brir_fft_avg_mag_sm_pre, "brir_fft_avg_mag_sm_pre_int", normalise=0)
                    hf.plot_data(brir_fft_avg_mag_sm_post, "brir_fft_avg_mag_sm_post_int", normalise=0)
                    hf.plot_data(comp_mag, "comp_mag_int", normalise=0)
            
            
        
        #parametric EQ step:
        #brir_reverberation = hf.equalize_brirs_parametric(brir_reverberation)
        
        # DEBUG TAP: BRIR (window, normalisation)
        debug_write_brir("02_param_eq_brir", brir_reverberation, samp_freq_ash)
        
        

                
        #
        # optional: 1st phase: create correction filter from average response and target,
        # then equalise each BRIR (GLOBAL average across all directions)
        #
        if mag_comp and correction_factor >= 0.1:#and subwoofer_mode == True
            hf.log_with_timestamp("Calibrating BRIRs (global average, stage 1)", gui_logger)

            # ------------------------------------------------------------
            # Collect ALL BRIRs across directions and channels
            # ------------------------------------------------------------
            # Shape: (num_directions * 2, N_FFT)
            brirs_all = brir_reverberation[0, :, :, :CN.N_FFT].reshape(-1, CN.N_FFT)
        
            # FFT
            brir_fft = np.fft.fft(brirs_all, axis=-1)
            brir_mag = np.abs(brir_fft)
            brir_db = hf.mag2db(brir_mag)
        
            # Global average (directions + channels)
            brir_fft_avg_db = np.mean(brir_db, axis=0)
        
            # ------------------------------------------------------------
            # Smooth + shape average response
            # ------------------------------------------------------------
            brir_fft_avg_mag = hf.db2mag(brir_fft_avg_db)
            brir_fft_avg_mag = hf.level_spectrum_ends(brir_fft_avg_mag,low_freq_in,high_freq,smooth_win=comp_win_size)
        
            brir_fft_avg_mag_sm = hf.smooth_gaussian_octave(data=brir_fft_avg_mag,n_fft=CN.N_FFT,fraction=octave_smoothing_n)
        
            # ------------------------------------------------------------
            # Compensation curve
            # ------------------------------------------------------------
            comp_db = (hf.mag2db(brir_fft_target_mag)- hf.mag2db(brir_fft_avg_mag_sm)) * correction_factor
        
            comp_mag = hf.db2mag(comp_db)
        
            # Final smoothing
            comp_mag = hf.smooth_gaussian_octave(data=comp_mag,n_fft=CN.N_FFT,fraction=octave_smoothing_n)
        
            comp_db = hf.mag2db(comp_mag)
        
            # ------------------------------------------------------------
            # Optional plots (GLOBAL)
            # ------------------------------------------------------------
            if CN.PLOT_ENABLE:
                hf.plot_data(brir_fft_target_mag, "brir_fft_target_mag_P1", normalise=0)
                hf.plot_data(brir_fft_avg_mag, "brir_fft_avg_mag_GLOBAL_P1", normalise=0)
                hf.plot_data(brir_fft_avg_mag_sm, "brir_fft_avg_mag_sm_GLOBAL_P1", normalise=0)
                hf.plot_data(comp_mag, "comp_mag_GLOBAL_P1", normalise=0)
                
            #parametric EQ step:
            brir_reverberation = hf.equalize_brirs_parametric(brir_reverberation, diff_db_override = comp_db, override_n_fft = CN.N_FFT)
    
    
    
               
        #
        # optional: 2nd phase: create correction filter from average response and target,
        # then equalise each BRIR (GLOBAL average across all directions)
        #
        if mag_comp and correction_factor >= 0.1:#and subwoofer_mode == True
            hf.log_with_timestamp("Calibrating BRIRs (global average, stage 2)", gui_logger)

            # ------------------------------------------------------------
            # Collect ALL BRIRs across directions and channels
            # ------------------------------------------------------------
            # Shape: (num_directions * 2, N_FFT)
            brirs_all = brir_reverberation[0, :, :, :CN.N_FFT].reshape(-1, CN.N_FFT)
        
            # FFT
            brir_fft = np.fft.fft(brirs_all, axis=-1)
            brir_mag = np.abs(brir_fft)
            brir_db = hf.mag2db(brir_mag)
        
            # Global average (directions + channels)
            brir_fft_avg_db = np.mean(brir_db, axis=0)
        
            # ------------------------------------------------------------
            # Smooth + shape average response
            # ------------------------------------------------------------
            brir_fft_avg_mag = hf.db2mag(brir_fft_avg_db)
            brir_fft_avg_mag = hf.level_spectrum_ends(brir_fft_avg_mag,low_freq_in,high_freq,smooth_win=comp_win_size)
        
            brir_fft_avg_mag_sm = hf.smooth_gaussian_octave(data=brir_fft_avg_mag,n_fft=CN.N_FFT,fraction=octave_smoothing_n)
        
            # ------------------------------------------------------------
            # Compensation curve
            # ------------------------------------------------------------
            comp_db = (hf.mag2db(brir_fft_target_mag)- hf.mag2db(brir_fft_avg_mag_sm)) * correction_factor
        
            comp_mag = hf.db2mag(comp_db)
        
            # Final smoothing
            comp_mag = hf.smooth_gaussian_octave(data=comp_mag,n_fft=CN.N_FFT,fraction=octave_smoothing_n)
        
            comp_eq_fir = hf.build_min_phase_filter(comp_mag,truncate_len=4096)
        
            # ------------------------------------------------------------
            # Optional plots (GLOBAL)
            # ------------------------------------------------------------
            if CN.PLOT_ENABLE:
                hf.plot_data(brir_fft_target_mag, "brir_fft_target_mag_P2", normalise=0)
                hf.plot_data(brir_fft_avg_mag, "brir_fft_avg_mag_GLOBAL_P2", normalise=0)
                hf.plot_data(brir_fft_avg_mag_sm, "brir_fft_avg_mag_sm_GLOBAL_P2", normalise=0)
                hf.plot_data(comp_mag, "comp_mag_GLOBAL_P2", normalise=0)
                
            for direc in range(num_brir_sources):
                if cancel_event and cancel_event.is_set():
                    gui_logger.log_warning("Operation cancelled by user.")
                    return brir_reverberation, 2  
                
                # Equalize both channels (loop still needed for convolution)
                for chan in range(total_chan_hrir):
                    eq_brir = sp.signal.convolve(brir_reverberation[0, direc, chan, :], comp_eq_fir, 'full', 'auto')
                    brir_reverberation[0, direc, chan, :] = eq_brir[:n_fft]
        

    
            
    
        #
        #optional: 3rd phase: create correction filter from average response and target, then equalise each BRIR
        #
        if mag_comp and correction_factor >= 0.1:
            hf.log_with_timestamp("Calibrating BRIRs (per speaker)", gui_logger)
       
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
                brir_fft_avg_mag_sm = hf.smooth_gaussian_octave(data=brir_fft_avg_mag, n_fft=CN.N_FFT, fraction=octave_smoothing_n)#6 or 12?
        
                # Compensation in dB
                comp_db = (hf.mag2db(brir_fft_target_mag)- hf.mag2db(brir_fft_avg_mag_sm)) * correction_factor
                # Limit maximum boost
                #comp_db = np.clip(comp_db, -40.0, 20.0)
                # Back to magnitude
                comp_mag = hf.db2mag(comp_db)
                comp_mag = hf.smooth_gaussian_octave(data=comp_mag, n_fft=CN.N_FFT, fraction=octave_smoothing_n)#12
                
                comp_eq_fir = hf.build_min_phase_filter(comp_mag,truncate_len=4096)
        
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
    
        brir_reverberation = hf.normalize_brir_band(ir_data=brir_reverberation,n_fft=n_fft,fs=CN.FS,f_norm_start=f_norm_start,f_norm_end=f_norm_end)
    
        # # Trim the BRIR reverberation array if needed (trim last dimension based on threshold)
        brir_reverberation = trim_brir_samples(brir_reverberation, relative_threshold=1e-5)
        
        # DEBUG TAP: after direct sound removal
        debug_write_brir("03_processed_brir", brir_reverberation, samp_freq_ash)
        
        if subwoofer_mode == True:#only 2 dimensions required if subwoofer response
             brir_reverberation=hf.remove_leading_singletons(brir_reverberation)
             
    
        
        status=0#success if reached this far
   
    
    except Exception as ex:

        log_string = 'Failed to complete IR to BRIR processing for: ' + ir_set 
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
        
    return brir_reverberation, status









def adjust_lf_drr(
    brir_reverberation,
    lf_mix=1.0,
    fs=CN.FS,
    crossover_hz=100,
    fade_len_samples=512,
    delay_samples=0,  # New parameter: N samples before windows start
    order=5,
    filtfilt=True
):
    """
    Vectorized reconstruction with a pre-delay.
    - 0 to delay_samples: Pure Direct (Unity)
    - delay to delay+fade: Hann Crossover
    - delay+fade to n_samp//2: Scaled Plateau -> Ramp
    - n_samp//2 to end: Unity Noise Floor
    """
    if brir_reverberation.ndim != 4:
        raise ValueError("Expected brir_reverberation shape (1, dirs, ch, samples)")

    _, _, _, n_samp = brir_reverberation.shape
    
    # Ensure our window indices don't exceed the signal length
    total_window_range = delay_samples + fade_len_samples
    if total_window_range > n_samp:
        raise ValueError("delay_samples + fade_len_samples exceeds signal length")

    # 1. Primary Windows (Fade In/Out)
    hann_full = np.hanning(fade_len_samples * 2)
    fade_in_win = hann_full[:fade_len_samples]
    fade_out_win = hann_full[fade_len_samples:]

    # 2. Segmented Tail Window Construction
    target_tail_gain = 1.0 / (lf_mix + 1e-10)
    tail_win = np.ones(n_samp)
    
    # Ramp defined from the end of the fade window to halfway through the IR
    ramp_start = total_window_range
    ramp_end = n_samp // 2
    
    if ramp_end > ramp_start:
        ramp_len = ramp_end - ramp_start
        tail_win[ramp_start:ramp_end] = np.linspace(1.0, target_tail_gain, ramp_len)
        tail_win[ramp_end:] = target_tail_gain
    else:
        tail_win[ramp_start:] = target_tail_gain

    # 3. Apply Filters
    lp_sos = hf.get_filter_sos(cutoff=crossover_hz, fs=fs, order=order, b_type="low", filtfilt=filtfilt)
    hp_sos = hf.get_filter_sos(cutoff=crossover_hz, fs=fs, order=order, b_type="high", filtfilt=filtfilt)

    hf_part = hf.apply_sos_filter(brir_reverberation, hp_sos, filtfilt=filtfilt, axis=-1)
    lf_full = hf.apply_sos_filter(brir_reverberation, lp_sos, filtfilt=filtfilt, axis=-1)

    # 4. LF Reconstruction with Delay
    lf_direct = np.zeros_like(lf_full)
    lf_reverb = lf_full.copy()

    # --- Direct Path Logic ---
    # Part A: Samples before the delay (kept at unity)
    lf_direct[..., :delay_samples] = lf_full[..., :delay_samples]
    # Part B: Samples during the fade-out
    lf_direct[..., delay_samples:total_window_range] = (
        lf_full[..., delay_samples:total_window_range] * fade_out_win[np.newaxis, np.newaxis, np.newaxis, :]
    )

    # --- Reverb Path Logic ---
    # Part A: Zero out reverb energy before and during pre-delay
    lf_reverb[..., :delay_samples] = 0
    # Part B: Apply fade-in starting at delay_samples
    lf_reverb[..., delay_samples:total_window_range] *= fade_in_win[np.newaxis, np.newaxis, np.newaxis, :]
    # Part C: Apply the tail noise protection window
    lf_reverb *= tail_win[np.newaxis, np.newaxis, np.newaxis, :]

    # 5. Summation
    lf_total = lf_direct + (lf_mix * lf_reverb)
    brir_integrated = hf_part + lf_total

    hf.log_with_timestamp(f"LF Adjusted (Delay: {delay_samples}, Mix: {lf_mix:.4f})")
    
    return brir_integrated





def calculate_relative_drr_change(
    brir_reverberation,
    air_dataset,
    n_measurements=40,
    fs=CN.FS,
    crossover_hz=100,
    delay_samples=0,
    fade_len_samples=512,
    order=4,
    filtfilt=True,
    use_windows=False  # Optional toggle for windowed vs hard split
):
    """
    Calculates DRR change. 
    If use_windows=False: Uses a hard rectangular split at split_idx.
    If use_windows=True:  Uses Hann fade-in/out windows starting at delay_samples.
    """
    # Slice AIR dataset for efficiency
    air_subset = air_dataset[:n_measurements, :] if air_dataset.shape[0] > n_measurements else air_dataset

    # Get Filter coefficients - filtfilt included as requested
    lp_sos = hf.get_filter_sos(cutoff=crossover_hz, fs=fs, order=order, b_type="low", filtfilt=filtfilt)

    # Point where the crossover/split begins
    #split_idx = delay_samples + fade_len_samples
    
    # Dynamic Split Index: 
    # If windowing, we split at the end of the fade. 
    # If hard split, we split exactly at the delay mark.
    split_idx = delay_samples + fade_len_samples if use_windows else delay_samples

    # Prepare windows if needed
    if use_windows:
        hann_full = np.hanning(fade_len_samples * 2)
        fade_in_win = hann_full[:fade_len_samples]
        fade_out_win = hann_full[fade_len_samples:]

    def compute_weighted_rms_ratio(signal_batch):
        """
        Calculates Ratio = RMS(Direct Path) / RMS(Reverb Path)
        """
        # 1. Apply Low-Pass Filter
        sig_lp = hf.apply_sos_filter(signal_batch, lp_sos, filtfilt=filtfilt, axis=-1)
        
        if use_windows:
            # --- Windowed Logic ---
            sig_direct = np.zeros_like(sig_lp)
            # Full signal before delay
            sig_direct[..., :delay_samples] = sig_lp[..., :delay_samples]
            # Fade out during the transition window
            sig_direct[..., delay_samples:split_idx] = (
                sig_lp[..., delay_samples:split_idx] * fade_out_win
            )

            sig_reverb = np.copy(sig_lp)
            # No reverb before delay
            sig_reverb[..., :delay_samples] = 0
            # Fade in during the transition window
            sig_reverb[..., delay_samples:split_idx] *= fade_in_win
            # Rest of the tail is full reverb energy
        else:
            # --- Hard Split Logic ---
            sig_direct = sig_lp[..., :split_idx]
            sig_reverb = sig_lp[..., split_idx:]
        
        # 3. RMS Calculation
        rms_direct = np.sqrt(np.mean(np.square(sig_direct)))
        rms_reverb = np.sqrt(np.mean(np.square(sig_reverb)))
        
        return rms_direct / (rms_reverb + 1e-10)

    # Calculate ratios
    ratio_brir = compute_weighted_rms_ratio(brir_reverberation)
    ratio_air = compute_weighted_rms_ratio(air_subset)

    # Relative change calculation
    relative_change = (ratio_brir - ratio_air) / (ratio_air + 1e-10)

    mode_str = "Windowed" if use_windows else "Hard Split"
    hf.log_with_timestamp(
        f"{mode_str} DRR Analysis | Delay: {delay_samples} | Fade: {fade_len_samples}"
    )

    return {
        "relative_change": relative_change,
        "brir_ratio": ratio_brir,
        "air_ratio": ratio_air
    }









def debug_write_brir(tag, brir, fs):
    """
    Export first elevation, first source as a stereo WAV.
    Expected brir shape: [elev, source, ch, samples]
    """
    if not CN.DEBUG_EXPORT:
        return

    try:
        os.makedirs(CN.DEBUG_DIR, exist_ok=True)

        # Defensive checks
        if brir.ndim < 4:
            print(f"[DEBUG] {tag}: invalid brir shape {brir.shape}")
            return

        # Take first elevation + first source
        wav = brir[0, 0, :, :]

        # Ensure shape = (samples, channels)
        wav = wav.T  # (2, N) -> (N, 2)

        fname = f"{CN.DEBUG_DIR}/{tag}.wav"
        hf.write2wav(fname, wav, samplerate=fs, prevent_clipping=1)

        print(f"[DEBUG] Wrote {fname}  shape={wav.shape}")

    except Exception as e:
        print(f"[DEBUG] Failed to write {tag}: {e}")

def trim_brir_samples(brir_array, relative_threshold=0.000005, ignore_last=0):
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
    




            
            
    

  

                



    
def acoustic_space_updates(download_updates=False, gui_logger=None):
    """
    Finds latest versions of acoustic spaces and compares them with current versions.
    Downloads updates and updates the local metadata row-by-row upon success.
    Uses atomic replacement to prevent CSV corruption.
    """

    try:
        hf.log_with_timestamp("Checking for acoustic space updates", gui_logger)

        # ---------------------------
        # 1. Setup Paths & Load LOCAL
        # ---------------------------
        csv_directory = pjoin(CN.DATA_DIR_INT, "reverberation")
        metadata_file = pjoin(csv_directory, CN.REV_METADATA_FILE_NAME)

        local_meta_dict_list = []
        fieldnames = None

        if os.path.exists(metadata_file):
            with open(metadata_file, encoding="utf-8-sig", newline="") as inputfile:
                reader = DictReader(inputfile)
                # CLEAN HERE: Ensure local keys are clean strings
                reader.fieldnames = clean_csv_headers(reader.fieldnames)
                fieldnames = reader.fieldnames
                local_meta_dict_list = list(reader)

        if not local_meta_dict_list:
            hf.log_with_timestamp("Local metadata empty or missing. Initializing new list.", gui_logger)

        # ---------------------------
        # 2. Load ONLINE metadata
        # ---------------------------
        ghub_meta_url = CN.AS_META_URL
        remote_meta_tmp = pjoin(csv_directory, "reverberation_metadata_latest.csv")

        hf.log_with_timestamp("Downloading latest metadata...", gui_logger)
        response = hf.download_file(url=ghub_meta_url, save_location=remote_meta_tmp, gui_logger=gui_logger)

        if response is not True:
            raise ValueError("Failed to download latest metadata from GitHub")

        web_meta_dict_list = []
        with open(remote_meta_tmp, encoding="utf-8-sig", newline="") as inputfile:
            # CLEAN HERE: Ensure local keys are clean strings
            reader.fieldnames = clean_csv_headers(reader.fieldnames)
            reader = DictReader(inputfile)
            web_meta_dict_list = list(reader)

        if not web_meta_dict_list:
            raise ValueError("Latest metadata is empty")

        mismatches = 0
        updates_perf = 0

        # -------------------------------------------------
        # 3. Compare & Update Row-by-Row
        # -------------------------------------------------
        for space_w in web_meta_dict_list:
            id_w = space_w.get("id")
            name_gui_w = space_w.get("name_gui")
            vers_w = space_w.get("version")
            rev_folder = space_w.get("folder")
            file_name = space_w.get("file_name")
            
            local_file_path = pjoin(csv_directory, rev_folder, file_name + ".npy")

            update_required = False
            local_row_index = None

            # Find matching row in local data
            for i, space_l in enumerate(local_meta_dict_list):
                if space_l.get("id") == id_w:
                    local_row_index = i
                    if space_l.get("version") != vers_w:
                        mismatches += 1
                        update_required = True
                        hf.log_with_timestamp(f"Update available ({vers_w}): {name_gui_w}", gui_logger, log_type=1)
                    break

            if local_row_index is None:
                mismatches += 1
                update_required = True
                hf.log_with_timestamp(f"New space found: {name_gui_w}", gui_logger)

            # --------------------------------------------------
            # 4. Download and Atomic Save
            # --------------------------------------------------
            if download_updates and update_required:
                hf.log_with_timestamp(f"Attempting download: {name_gui_w}", gui_logger)

                download_sources = [
                    ("github", space_w.get("ghub_link")),
                    ("primary", space_w.get("gdrive_link")),
                    ("alternative", space_w.get("alternative_link")),
                ]

                success = False
                for source_name, url in download_sources:
                    if not url: continue
                    
                    dl_res = hf.download_file(url=url, save_location=local_file_path, gui_logger=gui_logger)
                    if dl_res is True:
                        try:
                            # Verify integrity of downloaded file
                            arr = hf.load_convert_npy_to_float64(local_file_path)
                            if arr is not None and len(arr) > 0:
                                success = True
                                break
                        except Exception:
                            hf.log_with_timestamp(f"Corrupt download from {source_name}", gui_logger, log_type=2)

                if success:
                    # Update memory
                    if local_row_index is not None:
                        local_meta_dict_list[local_row_index] = space_w
                    else:
                        local_meta_dict_list.append(space_w)

                    # --- ATOMIC WRITE TO CSV ---
                    # Ensures the original file is never corrupted if the write fails
                    fd, temp_path = tempfile.mkstemp(dir=csv_directory, suffix=".tmp")
                    try:
                        with os.fdopen(fd, 'w', encoding="utf-8-sig", newline="") as tmp_f:
                            # Ensure the first fieldname doesn't contain a literal BOM string 
                            # if it was inherited from an older read process.
                            # CLEAN HERE: Use the helper for the final write headers
                            final_fields = clean_csv_headers(fieldnames if fieldnames else list(space_w.keys()))
                            
                            writer = DictWriter(tmp_f, fieldnames=final_fields)
                            writer.writeheader()
                            writer.writerows(local_meta_dict_list)
                        
                        # Atomic swap
                        os.replace(temp_path, metadata_file)
                        updates_perf += 1
                        hf.log_with_timestamp(f"Successfully updated {name_gui_w} to v{vers_w}", gui_logger)
                    except Exception as e:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        raise e
                else:
                    raise ValueError(f"All download sources failed for {name_gui_w}")

        # ---------------------------
        # 5. Finalize & Cleanup
        # ---------------------------
        if mismatches == 0:
            hf.log_with_timestamp("No updates available", gui_logger)

        if os.path.exists(remote_meta_tmp):
            os.remove(remote_meta_tmp)

        # Save last check timestamp
        last_checked_path = pjoin(csv_directory, "last_checked_updates.json")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(last_checked_path, "w", encoding="utf-8") as f:
            json.dump({"last_checked": ts}, f, indent=4)

    except Exception as ex:
        hf.log_with_timestamp(
            log_string="Acoustic space update failed",
            gui_logger=gui_logger,
            log_type=2,
            exception=ex,
        )    
 
def clean_csv_headers(header_list):
    """
    Strips the UTF-8 BOM, removes surrounding whitespace, 
    and filters out empty strings from a list of header names.
    Works regardless of column order.
    """
    if not header_list:
        return []
    
    # 1. lstrip('\ufeff') removes the BOM if present
    # 2. strip() removes spaces/newlines
    # 3. 'if h' ensures we don't return empty strings as keys
    return [h.lstrip('\ufeff').strip() for h in header_list if h]



def save_reverberation_dataset(
    brir_reverberation,
    ir_set,
    data_dir=CN.DATA_DIR_AS_USER,
    prefix=CN.AS_USER_PREFIX,
    gui_logger=None,
    trim_threshold=1e-5
):
    """
    Save a reverberation BRIR dataset to disk as a .npy file.

    Parameters
    ----------
    brir_reverberation : np.ndarray
        Reverberation BRIR array to save.
    ir_set : str
        Name of the IR dataset (used for folder/file naming).
    data_dir : str
        Base directory where dataset will be saved.
    prefix : str
        Prefix for the saved filename.
    gui_logger : optional
        GUI logger for logging messages.
    trim_threshold : float, default=1e-5
        Threshold for trimming low-amplitude samples.
    
    Returns
    -------
    status : int
        0 if success.
    out_file_path : str
        Full path to the saved file.
    """

    # --- folder for output ---
    brir_out_folder = pjoin(data_dir, ir_set)

    # --- trim last dimension based on threshold ---
    brir_reverberation = trim_brir_samples(brir_reverberation, relative_threshold=trim_threshold)

    # --- convert to float32 if needed ---
    if brir_reverberation.dtype == np.float64:
        brir_reverberation = brir_reverberation.astype(np.float32)

    # --- file path ---
    npy_file_name = f"{prefix}{ir_set}.npy"
    out_file_path = pjoin(brir_out_folder, npy_file_name)
    output_file = Path(out_file_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # --- save ---
    np.save(out_file_path, brir_reverberation)

    # --- log ---
    log_string = f"Exported reverberation dataset to: {out_file_path}"
    hf.log_with_timestamp(log_string, gui_logger)

    return 0, out_file_path




def compute_lf_alignment_params(
    samp_freq,
    cutoff_hz,
    delay_win_min_t=150,#0,200
    delay_win_max_t=1000,
    delay_win_hop_size=25,
):
    """
    Compute wavelength-relative alignment parameters for LF BRIR alignment.
    """

    wavelength = samp_freq / cutoff_hz

    # ---- Peak-to-peak window ----
    peak_to_peak_window = int(np.round(1.25 * wavelength))#1.25 * wavelength
    peak_to_peak_window = int(np.clip(peak_to_peak_window,0.75 * wavelength,2.0 * wavelength))

    # ---- Shift resolution ----
    step = max(1, int(np.round(wavelength / 12)))
    step = min(step, 10)

    # ---- Shift span ----
    half_span = 0.5 * wavelength
    min_s = int(np.floor(-(half_span*3.0) / step) * step)#1.25,1.4,2.0 2.2 2.25
    max_s = int(np.ceil( (half_span*3.0) / step) * step)#1.25,1.4,2.0 2.2 1.75
    shifts = np.arange(min_s, max_s + step, step, dtype=int)

    # ---- Delay windows ----
    delay_win_hops = int((delay_win_max_t - delay_win_min_t) / delay_win_hop_size)
    win_starts = delay_win_min_t + np.arange(delay_win_hops) * delay_win_hop_size
    win_ends = win_starts + peak_to_peak_window

    return shifts, win_starts, win_ends, peak_to_peak_window





        
      



       
       
        
 
        
def align_ir_static_window(
    idx, air_data,
    shifts,
    lp_sos,
    binaural_mode=False,
    debug=False,
    plot=CN.PLOT_ENABLE,
    plot_indices=[10, 50, 80, 400, 500, 600, 750, 751, 752, 753, 754, 756, 758, 850, 950, 1000, 1500, 2000],
    score_mode="ptp",#ptp or balanced. ptp more accurate
    lookahead_samples=650,
    start_sample=170  # <--- New parameter: ignores everything before this index
):
    if idx == 0 or not np.any(air_data[idx]):
        return
    
    # Convert plot_indices to a set for O(1) lookup performance
    plot_set = set(plot_indices) if plot_indices is not None else set()

    fade_samples = 30
    crop_length = 4000
    win_len = min(lookahead_samples, crop_length)

    # 1. Capture the "Original" state before any processing for the plot
    original_raw = air_data[idx, :crop_length].copy()

    # 2. Reference (mean of previous) & Target (Current)
    prior = np.mean(air_data[:idx, :crop_length], axis=0)
    this = air_data[idx, :crop_length]

    # 3. Filter to focus on low-frequency timing/phase
    prior_lp = hf.apply_sos_filter(prior, lp_sos, filtfilt=CN.FILTFILT_TDALIGN_AIR)
    this_lp = hf.apply_sos_filter(this, lp_sos, filtfilt=CN.FILTFILT_TDALIGN_AIR)

    # 4. Build shifted stack
    shifts = np.asarray(shifts)
    n_shifts = len(shifts)
    N = len(this_lp)
    shifted_stack = np.zeros((n_shifts, N), dtype=this_lp.dtype)

    for i, s in enumerate(shifts):
        if s > 0:
            shifted_stack[i, s:] = this_lp[:-s]
        elif s < 0:
            shifted_stack[i, :s] = this_lp[-s:]
        else:
            shifted_stack[i] = this_lp

    # 5. Constructive Interference calculation
    summed = shifted_stack + prior_lp[None, :]

    # 6. Score based on the window (start_sample to win_len)
    # This ignores the first 'start_sample' samples for scoring purposes
    seg = summed[:, start_sample:win_len]

    if score_mode == "ptp":
        scores = seg.max(axis=1) - seg.min(axis=1)
    elif score_mode == "balanced":
        scores = (np.abs(seg.max(axis=1)) + np.abs(seg.min(axis=1))) * 0.5
    else:
        raise ValueError("Use 'ptp' or 'balanced'")

    best_idx = np.argmax(scores)
    best_shift = shifts[best_idx]
    best_score = scores[best_idx]

    # 7. Apply winning shift to the actual 2D dataset
    target_indices = [idx, idx + 1] if binaural_mode else [idx]
    
    for t_idx in target_indices:
        if t_idx < air_data.shape[0]:
            air_data[t_idx] = np.roll(air_data[t_idx], best_shift)
            
            # Clean up wrap-around
            if best_shift > 0:
                air_data[t_idx, :best_shift] = 0.0
            elif best_shift < 0:
                air_data[t_idx, best_shift:] = 0.0
                
            n_f = min(fade_samples, air_data.shape[1])
            fade_win = np.linspace(0.0, 1.0, n_f)
            air_data[t_idx, :n_f] *= fade_win

    if debug:
        print(f"[Align] IR {idx}: shift={best_shift}, score={best_score:.4f}")

    # 8. Plotting
    if plot and idx in plot_set:
        original_lp = hf.apply_sos_filter(original_raw, lp_sos, filtfilt=CN.FILTFILT_TDALIGN_AIR)
        aligned_lp = hf.apply_sos_filter(air_data[idx, :crop_length], lp_sos, filtfilt=CN.FILTFILT_TDALIGN_AIR)
        
        plt.figure(figsize=(12, 5))
        
        # Adjust normalization to ignore start_sample as well
        def norm(v): 
            view = v[start_sample:win_len]
            return v / (np.max(np.abs(view)) + 1e-12)

        plt.plot(norm(prior_lp), label="Reference (Mean)", color='black', alpha=0.4, linestyle='--')
        plt.plot(norm(original_lp), label="Original (Unaligned)", color='red', alpha=0.5)
        plt.plot(norm(aligned_lp), label=f"Aligned (Shift: {best_shift})", color='green', alpha=0.9, linewidth=1.5)
        
        plt.axvline(x=start_sample, color='orange', linestyle='--', label='Search Window Start', alpha=0.6)
        plt.axvline(x=win_len, color='blue', linestyle=':', label='Search Window End', alpha=0.5)
        
        plt.title(f"Time-Alignment Visualizer: IR {idx} (Static Window: {start_sample} to {win_len})")
        plt.xlabel("Samples")
        plt.ylabel("Normalized Amplitude")
        plt.legend(loc='upper right')
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()     