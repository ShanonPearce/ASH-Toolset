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

logger = logging.getLogger(__name__)
log_info=1

def extract_airs_from_recording(ir_set='fw', gui_logger=None):
    """
    function to extract individual IRs from a recording containing multiple IRs
    :param ir_set: str, name of impulse response set. Must correspond to a folder in ASH-Toolset\data\raw\ir_data\stream
    :return: None
    """
    
    min_len_s=1.0
    peak_mode=2#0=check amplitude peaks, 1= check gradient peaks, 2=both methods
    amp_thresh_d=0.80#0.8
    amp_thresh_l=amp_thresh_d*0.8
    grad_thresh_d=0.1#
    output_wavs=1
    peak_tshift=250
    
    ir_data_folder = pjoin(CN.DATA_DIR_RAW, 'ir_data', 'stream',ir_set)
    ir_out_folder = pjoin(CN.DATA_DIR_RAW, 'ir_data', 'split_airs',ir_set)
    
    try:
        log_string_a = 'Starting extract_airs_from_recording processing for: '+ir_set
        hf.log_with_timestamp(log_string_a, gui_logger)
    
        file_id=0
        for root, dirs, files in os.walk(ir_data_folder):
            for filename in files:
                if '.wav' in filename:
                    #read wav file
                    wav_fname = pjoin(root, filename)
                    samplerate, data = wavfile.read(wav_fname)
                    fir_array = data / (2.**31)
                    
                    #resample if sample rate is not 44100
                    if samplerate != CN.SAMP_FREQ:
                        fir_array = hf.resample_signal(fir_array, original_rate = samplerate, new_rate = CN.SAMP_FREQ)
                        log_string_a = 'source samplerate: ' + str(samplerate) + ', resampled to: '+ str(CN.SAMP_FREQ)
                        hf.log_with_timestamp(log_string_a, gui_logger)
                    
                    
                    #normalise IR
                    fir_array=np.divide(fir_array,np.max(fir_array))
                    fir_array_abs = np.abs(fir_array)
                    #take gradient with x spacing
                    fir_array_abs_combined = np.divide(np.add(np.abs(fir_array[:,0]),np.abs(fir_array[:,1])),2)
                    fir_array_abs_grad = np.gradient(fir_array_abs_combined, 1)
                    if CN.PLOT_ENABLE == True:
                        plot_name = 'fir_array_abs_grad_'+filename
                        hf.plot_td(fir_array_abs_grad[0:20000],plot_name)
                    
                    #find prominent peaks
                    if peak_mode == 0:
                        peaks_indices = np.where(fir_array_abs_combined > amp_thresh_d)[0]
                    elif peak_mode == 1:
                        peaks_indices = np.where(fir_array_abs_grad > grad_thresh_d)[0]
                    else:
                        peaks_indices = np.where( np.logical_or(fir_array_abs_grad > grad_thresh_d,fir_array_abs_combined > amp_thresh_d))[0]
                    #get time values of above indices
                    peaks_time= np.divide(peaks_indices,samplerate)
                    #get time since previous peak
                    peaks_time_diff=np.zeros(len(peaks_time))
                    for i in range(len(peaks_time)):
                        if i < len(peaks_time)-1:
                            peaks_time_diff[i]=peaks_time[i+1]-peaks_time[i]    

                    #find prominent peaks where delay between previous peak is sufficiently high
                    ir_start_indices = peaks_indices[peaks_time_diff >= min_len_s]

                    #shift to the right by x samples
                    ir_start_ind_shift=np.subtract(ir_start_indices,peak_tshift)
                    fir_array_split = np.split(fir_array,ir_start_ind_shift)
                        
                    #name of dataset
                    dataset_name=filename.split('.wav')[0]
                    #loop through array to export wav files for testing
                    for idx, x in enumerate(fir_array_split):
                        out_file_name = 'split_ir_'+str(file_id)+'_'+str(idx)+ '.wav'
                        out_file_path = pjoin(ir_out_folder,dataset_name,out_file_name)
                        
                        #create dir if doesnt exist
                        output_file = Path(out_file_path)
                        output_file.parent.mkdir(exist_ok=True, parents=True)
                        
                        start_idx=1500
                        end_idx=min(40000,len(x))
                        if output_wavs == 1 and idx>0 and np.max(np.abs(x[start_idx:end_idx]))<amp_thresh_l:#ensure split ir does not have secondary peak
                            hf.write2wav(file_name=out_file_path, data=x, samplerate=44100, prevent_clipping = 1)
                            
                    file_id=file_id+1
                    
        log_string_a = 'Completed extract_airs_from_recording processing for: '+ir_set
        hf.log_with_timestamp(log_string_a, gui_logger)
                    
    except Exception as ex:
        log_string = 'Failed to extract AIRs from recording for stream: ' + ir_set 
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error



def split_airs_to_set(ir_set='fw', gui_logger=None):
    """
    function to compile dataset of IRs
    saves air_reverberation array. num_irs * 2 channels * n_fft samples
    :param ir_set: str, name of impulse response set. Must correspond to a folder in ASH-Toolset\data\interim\ir_data\split_irs
    :return: None
    """
    
    #set fft length based on AC space
    if ir_set in CN.AC_SPACE_LIST_LOWRT60:
        n_fft=CN.N_FFT
    else:
        n_fft=CN.N_FFT_L
    
    samp_freq_ash=CN.SAMP_FREQ
    output_wavs=1
    
    #windows
    data_pad_zeros=np.zeros(n_fft)
    data_pad_ones=np.ones(n_fft)
    
    total_chan_irs=2
    total_chan_air=1

    #direct sound window for 2nd phase
    direct_hanning_size=300#350
    direct_hanning_start=5#101
    hann_direct_full=np.hanning(direct_hanning_size)
    hann_direct = np.split(hann_direct_full,2)[0]
    direct_removal_win_b = data_pad_zeros.copy()
    direct_removal_win_b[direct_hanning_start:direct_hanning_start+int(direct_hanning_size/2)] = hann_direct
    direct_removal_win_b[direct_hanning_start+int(direct_hanning_size/2):]=data_pad_ones[direct_hanning_start+int(direct_hanning_size/2):]
    
    fade_hanning_size=int(60000)
    fade_hanning_start=1000
    hann_fade_full=np.hanning(fade_hanning_size)
    hann_fade = np.split(hann_fade_full,2)[1]
    fade_out_win_s = data_pad_ones.copy()
    fade_out_win_s[fade_hanning_start:fade_hanning_start+int(fade_hanning_size/2)] = hann_fade
    fade_out_win_s[fade_hanning_start+int(fade_hanning_size/2):]=data_pad_zeros[fade_hanning_start+int(fade_hanning_size/2):]
    
    fade_hanning_size=int(120000)
    fade_hanning_start=5000
    hann_fade_full=np.hanning(fade_hanning_size)
    hann_fade = np.split(hann_fade_full,2)[1]
    fade_out_win_l = data_pad_ones.copy()
    fade_out_win_l[fade_hanning_start:fade_hanning_start+int(fade_hanning_size/2)] = hann_fade
    fade_out_win_l[fade_hanning_start+int(fade_hanning_size/2):]=data_pad_zeros[fade_hanning_start+int(fade_hanning_size/2):]
    
    
    
    #loop through folders
    ir_in_folder = pjoin(CN.DATA_DIR_RAW, 'ir_data', 'split_airs',ir_set)
    
    try:
        log_string_a = 'Starting split_airs_to_set processing for: '+ir_set
        hf.log_with_timestamp(log_string_a, gui_logger)

    
        #get number of IRs
        ir_counter=0
        for root, dirs, files in os.walk(ir_in_folder):
            for filename in files:
                if '.wav' in filename:
                    #read wav files
                    wav_fname = pjoin(root, filename)
                    samplerate, data = wavfile.read(wav_fname)
                    samp_freq=samplerate
                    fir_array = data / (2.**31)

                    try:
                        input_channels = len(fir_array[0])
                    except:
                        #reshape if mono
                        input_channels=1
                        fir_array=fir_array.reshape(-1, 1)
                    ir_counter=ir_counter+input_channels
                    
        #numpy array, num sets x num irs in each set x 2 channels x NFFT max samples
        total_irs=int(ir_counter)+1
        print(str(total_irs))
        air_data=np.zeros((total_irs,n_fft))

        ir_counter=0
        for root, dirs, files in os.walk(ir_in_folder):
            for filename in files:
                if '.wav' in filename:
                    #print('test')
                    
                    #read wav files
                    wav_fname = pjoin(root, filename)
                    samplerate, data = wavfile.read(wav_fname)
                    samp_freq=samplerate
                    fir_array = data / (2.**31)
                    fir_length = len(fir_array)
                    try:
                        input_channels = len(fir_array[0])
                    except:
                        #reshape if mono
                        input_channels=1
                        fir_array=fir_array.reshape(-1, 1)
                    
                    #append into set list
                    
                    extract_legth = min(n_fft,fir_length)
                    #load into numpy array
                    for chan in range(input_channels):
                        air_data[ir_counter,0:extract_legth]=fir_array[0:extract_legth,chan]#L and R
                        air_data[ir_counter,:] = np.multiply(air_data[ir_counter,:],direct_removal_win_b)#direct removal                
                        if 'split_ir_' in filename:
                            air_data[ir_counter,:] = np.multiply(air_data[ir_counter,:],fade_out_win_s)#late reflections fade out 
                        else:
                            air_data[ir_counter,:] = np.multiply(air_data[ir_counter,:],fade_out_win_l)#late reflections fade out 
                            
                        ir_counter=ir_counter+1

        if ir_counter >0:
            
            #create dir if doesnt exist
            air_out_folder = pjoin(CN.DATA_DIR_RAW, 'ir_data', 'raw_airs',ir_set)
        
            npy_file_name = ir_set+'_td_avg.npy'
        
            out_file_path = pjoin(air_out_folder,npy_file_name)      
              
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            np.save(out_file_path,air_data)
            
            log_string_a = 'Exported numpy file to: ' + out_file_path 
            hf.log_with_timestamp(log_string_a, gui_logger)
                
            log_string_a = 'Completed split_airs_to_set processing for: '+ir_set
            hf.log_with_timestamp(log_string_a, gui_logger)
    
    except Exception as ex:
        log_string = 'Failed to complete TD averaging for: ' + ir_set 
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
    
    
  


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
    max_samples=CN.N_FFT, cancel_event=None
):
    """
    Loads and standardizes impulse responses from a folder (including subfolders) into a 2D array.
    
    Parameters:
        folder_path (str): Root folder to search for IR files.
        file_type (str, optional): Override for file type: 'mat', 'wav', 'npy', 'sofa', or 'hdf5'.
        gui_logger (optional): Logger for GUI output.
        project_samplerate (int): Sample rate to resample all IRs to.
        normalize (bool): Whether to normalize IRs.
        noise_reduction (bool): Whether to apply noise reduction.
        noise_tail_ratio (float): Fraction of IR length to treat as noise.
        max_measurements (int): Maximum total number of measurements (rows).
        max_samples (int): Maximum number of samples (columns).

    Returns:
        air_data (np.ndarray): 2D array of impulse responses (n_measurements, n_samples).
    """

    # Define supported file extensions
    supported_exts = ['.wav', '.mat', '.npy', '.sofa', '.hdf5']

    # Recursively gather all supported files from the input folder
    all_files = [
        f for ext in supported_exts
        for f in glob.glob(os.path.join(folder_path, '**', f'*{ext}'), recursive=True)
    ]

    ir_list = []
    max_samples_in_data = 0
    total_measurements = 0
    status=1
    air_data=np.array([])

    # Iterate over each file found
    for path in all_files:
        ext = os.path.splitext(path)[1].lower()
        ftype = file_type if file_type else ext.strip('.')
        
        if cancel_event and cancel_event.is_set():
            gui_logger.log_warning("Operation cancelled by user.")
            return air_data, 2

        try:
            samplerate = project_samplerate  # default sample rate unless overridden

            # Load .mat file
            if ftype == 'mat':
                data = loadmat(path, squeeze_me=True, struct_as_record=False)
                ir, samplerate, inferred_sr = find_valid_ir_and_sr(data, gui_logger=None)
                ir = hf.reshape_array_to_two_dims(ir)
                hf.log_with_timestamp(f"Loaded MAT file '{os.path.basename(path)}' with shape {ir.shape}, SR={samplerate}", gui_logger)
                if inferred_sr:
                    hf.log_with_timestamp(f"[Warning] Sample rate inferred as 48000 Hz for '{os.path.basename(path)}' — not found in file.", gui_logger, log_type=1)

            # Load .wav file
            elif ftype == 'wav':
                ir, samplerate = sf.read(path)
                ir = np.array(ir)
                if ir.ndim == 1:
                    ir = ir[np.newaxis, :]  # mono
                elif ir.ndim == 2:
                    if ir.shape[1] <= 64:  # assume shape (samples, channels)
                        ir = ir.T  # now (channels, samples)
                hf.log_with_timestamp(f"Loaded WAV file '{os.path.basename(path)}' with shape {ir.shape}", gui_logger)

            # Load .sofa file using helper function
            elif ftype == 'sofa':
                loadsofa = hf.sofa_load_object(path, gui_logger)
                ir = loadsofa['sofa_data_ir']
                samplerate = loadsofa['sofa_samplerate']
                ir = hf.reshape_array_to_two_dims(ir)

            # Load .hdf5 file and extract dataset
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

            # Load .npy file
            elif ftype == 'npy':
                ir = np.load(path)
                ir = hf.reshape_array_to_two_dims(ir)
   

            else:
                raise ValueError(f"Unsupported file type: {ftype}")

            # Reshape multi-dimensional IRs to 2D array (n_measurements, n_samples)
            if ir.ndim > 2:
                sample_axis = np.argmax(ir.shape)
                n_samples = ir.shape[sample_axis]
                other_dims = np.prod([s for i, s in enumerate(ir.shape) if i != sample_axis])
                ir = np.reshape(np.moveaxis(ir, sample_axis, -1), (other_dims, n_samples))
            elif ir.ndim == 1:
                ir = ir[np.newaxis, :]

            # Resample if needed
            if ftype != 'npy' and samplerate != project_samplerate:
                ir = hf.resample_signal(ir, original_rate=samplerate, new_rate=project_samplerate, axis=1)
                hf.log_with_timestamp(f"Resampled from {samplerate} Hz to {project_samplerate} Hz for {os.path.basename(path)}", gui_logger)

            # Optional noise reduction using trailing portion of IR
            if noise_reduction:
                try:
                    tail_len = max(512, int(ir.shape[1] * noise_tail_ratio))
                    y_noise = np.copy(ir[:, -tail_len:])
                    y_clean = nr.reduce_noise(y=ir, sr=project_samplerate, y_noise=y_noise, stationary=True)
                    ir = np.reshape(y_clean, ir.shape)
                    hf.log_with_timestamp(f"Noise reduction applied to '{os.path.basename(path)}' (tail={tail_len})", gui_logger)
                except Exception as e:
                    hf.log_with_timestamp(f"[Error] Noise reduction failed on {path}: {e}", gui_logger)

            # Optional normalization
            if normalize:
                ir = hf.normalize_array(ir)

            hf.log_with_timestamp(f"Loaded '{os.path.basename(path)}' with shape {ir.shape}", gui_logger)

            # Update sample and measurement limits
            max_samples_in_data = max(max_samples_in_data, ir.shape[1])
            total_measurements += ir.shape[0]

            # Enforce global measurement limit
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
    # Pad or truncate all IRs to have consistent number of samples
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

    # Combine all individual IRs into a single 2D array
    hf.log_with_timestamp(f"Total IRs loaded: {len(ir_list)}, Max samples: {max_samples_in_data}", gui_logger)
    air_data = np.concatenate(padded_list, axis=0)
    hf.log_with_timestamp(f"Final air_data shape: {air_data.shape}", gui_logger)

    status=0#success if reached this far
    return air_data,status



def prepare_air_dataset(
    ir_set='default_set_name',
    input_folder=None,
    gui_logger=None,
    wav_export=False,
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
    if ir_set in CN.AC_SPACE_LIST_NR or noise_reduction_mode == True:
        noise_reduction=True
    else:
        noise_reduction=False
        
    samp_freq_ash=CN.SAMP_FREQ
    air_data=np.array([])
    ir_min_threshold=CN.IR_MIN_THRESHOLD
    
    #windows
    data_pad_zeros=np.zeros(n_fft)
    data_pad_ones=np.ones(n_fft)
    #direct sound fade in window
    direct_hanning_size=300#300
    direct_hanning_start=51#101
    hann_direct_full=np.hanning(direct_hanning_size)
    hann_direct = np.split(hann_direct_full,2)[0]
    direct_removal_win_b = data_pad_zeros.copy()
    direct_removal_win_b[direct_hanning_start:direct_hanning_start+int(direct_hanning_size/2)] = hann_direct
    direct_removal_win_b[direct_hanning_start+int(direct_hanning_size/2):]=data_pad_ones[direct_hanning_start+int(direct_hanning_size/2):]
    #optional fade out window
    if ir_set in CN.AC_SPACE_LIST_WINDOW:
        if n_fft == CN.N_FFT_L:
            fade_hanning_size=int(65536*2)
        else:
            fade_hanning_size=int(65536)
        if ir_set == 'outdoors_a':
            fade_hanning_start=32000
        elif ir_set == 'hall_a':
            fade_hanning_start=27000
        elif ir_set == 'seminar_room_a':
            fade_hanning_start=12000
        elif n_fft == CN.N_FFT_L:
            fade_hanning_start=35000
        else:
            fade_hanning_start=30000#
        hann_fade_full=np.hanning(fade_hanning_size)
        hann_fade = np.split(hann_fade_full,2)[1]
        fade_out_win = data_pad_ones.copy()
        fade_out_win[fade_hanning_start:fade_hanning_start+int(fade_hanning_size/2)] = hann_fade
        fade_out_win[fade_hanning_start+int(fade_hanning_size/2):]=data_pad_zeros[fade_hanning_start+int(fade_hanning_size/2):]
    
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
        
        # Optional fade out window
        if ir_set in CN.AC_SPACE_LIST_WINDOW:
            # Get average level in late reflections across all IRs
            average_mag_total = 0
            total_irs = 0
            for idx in range(total_measurements):
                segment = air_data[idx, 44100:CN.N_FFT]
                data_fft = np.fft.fft(segment)
                if np.sum(np.abs(data_fft)) > 0.0001:
                    mag_fft = np.abs(data_fft)
                    average_mag = np.mean(mag_fft[fb_start:fb_end])
                    total_irs += 1
                    average_mag_total += average_mag
            if total_irs > 0:
                average_mag_total = 1.10 * average_mag_total / total_irs
            else:
                average_mag_total = 0
            # Apply fade-out window based on condition
            for idx in range(total_measurements):
                segment = air_data[idx, 44100:CN.N_FFT]
                data_fft = np.fft.fft(segment)
                mag_fft = np.abs(data_fft)
                average_mag = np.mean(mag_fft[fb_start:fb_end])
                if average_mag > average_mag_total or ir_set in CN.AC_SPACE_LIST_WINDOW_ALL:
                    air_data[idx, :] *= fade_out_win
                    log_string_a = f'Window applied to IR index {idx}'
                    hf.log_with_timestamp(log_string_a)
        
 
    
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
            air_data *= direct_removal_win_b
                       
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
            out_wav_array[:,0] = np.copy(air_data[-2,:])#take sample

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
    wav_export=CN.EXPORT_WAVS_DEFAULT,
    long_mode=False, report_progress=0, cancel_event=None, distr_mode=0, rise_time=5.1, subwoofer_mode=False
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
    if long_mode == True or ir_set in CN.AC_SPACE_LIST_HIRT60:
        n_fft=CN.N_FFT_L
    else:
        n_fft=CN.N_FFT
    
    if ir_set in CN.AC_SPACE_LIST_NOCOMP:#ir_set in CN.AC_SPACE_LIST_SUB
        mag_comp=False
    else:
        mag_comp=CN.MAG_COMP
  
    samp_freq_ash=CN.SAMP_FREQ

    #impulse
    impulse=np.zeros(CN.N_FFT)
    impulse[0]=1
    brir_reverberation=np.array([])
  
    #direct sound window for 2nd phase
    #windows
    data_pad_zeros=np.zeros(n_fft)
    data_pad_ones=np.ones(n_fft)
    if ir_set in CN.AC_SPACE_LIST_SLOWRISE:
        direct_hanning_size=1000#    
    else:
        rise_samples = int((rise_time / 1000.0) * samp_freq_ash)
        direct_hanning_size=rise_samples*2#
    direct_hanning_start=51#
    hann_direct_full=np.hanning(direct_hanning_size)
    hann_direct = np.split(hann_direct_full,2)[0]
    direct_removal_win_b = data_pad_zeros.copy()
    direct_removal_win_b[direct_hanning_start:direct_hanning_start+int(direct_hanning_size/2)] = hann_direct
    direct_removal_win_b[direct_hanning_start+int(direct_hanning_size/2):]=data_pad_ones[direct_hanning_start+int(direct_hanning_size/2):]
    
    
 
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
        #load HRIR dataset
        #
        
        try:
            npy_fname = pjoin(CN.DATA_DIR_INT, 'hrir_dataset_comp_max_TU-FABIAN.npy')
            hrir_list = hf.load_convert_npy_to_float64(npy_fname)
        except:
            npy_fname = pjoin(CN.DATA_DIR_INT, 'hrir_dataset_comp_max_THK-KU-100.npy')
            hrir_list = hf.load_convert_npy_to_float64(npy_fname)
        
        
        hrir_id = 0#only one id available 
        hrir_selected = hrir_list[hrir_id]
        total_elev_hrir = len(hrir_selected)
        total_azim_hrir = len(hrir_selected[0])
        total_chan_hrir = len(hrir_selected[0][0])
        total_samples_hrir = len(hrir_selected[0][0][0])
        base_elev_idx = total_elev_hrir//2
        base_elev_idx_offset=total_elev_hrir//8
        
        #spatial metadata
        spatial_res=3#max resolution
        elev_min=CN.SPATIAL_RES_ELEV_MIN[spatial_res] 
        elev_max=CN.SPATIAL_RES_ELEV_MAX[spatial_res] 
        elev_nearest=CN.SPATIAL_RES_ELEV_NEAREST[spatial_res] #as per hrir dataset
        elev_nearest_process=CN.SPATIAL_RES_ELEV_NEAREST_PR[spatial_res] 
        azim_nearest=CN.SPATIAL_RES_AZIM_NEAREST[spatial_res] 
        azim_nearest_process=CN.SPATIAL_RES_AZIM_NEAREST_PR[spatial_res] 
        #define desired angles
        nearest_deg=2
        elev_min_dist=-40
        elev_max_dist=62#60
        azim_min_dist=0
        azim_max_dist=360
        elev_src_set = np.arange(elev_min_dist,elev_max_dist,elev_nearest)#np.arange(-40,58,elev_nearest)
        azim_src_set=np.arange(azim_min_dist,azim_max_dist,azim_nearest)

        
        #set number of output directions (brir sources)
        if n_fft == CN.N_FFT:
            num_brir_sources=7
        else:
            num_brir_sources=5#fewer sources if long reverb tail
        if total_measurements < CN.IR_MIN_THRESHOLD_FULLSET:
            num_brir_sources=5
        if subwoofer_mode == True:#only 1 set required if subwoofer response
            num_brir_sources=1
   
        log_string_a = 'num_brir_sources: ' + str(num_brir_sources)
        hf.log_with_timestamp(log_string_a)
        
        # Create numpy array for new BRIR dataset
        brir_reverberation = np.zeros((CN.INTERIM_ELEVS, num_brir_sources, 2, n_fft))
        
        # Log start
        hf.log_with_timestamp("Estimating BRIRs from IRs", gui_logger)
        
        # Sampling spatial coordinates with bias
        num_samples = total_measurements
        biased_centers = np.array([45, 135, 225, 315])
        strength = 51
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
        
            for measurement in group:
                # Copy AIR
                curr_air = np.copy(air_dataset[measurement, :])
        
                # Get HRIR from distributed coordinates
                curr_azim_deg = azimuths_distribution[dist_counter]
                curr_elev_deg = elevations_distribution[dist_counter]
                curr_azim_id = int(curr_azim_deg / azim_nearest)
                curr_elev_id = int((curr_elev_deg - elev_min) / elev_nearest)
        
                curr_hrir_l = np.copy(hrir_list[hrir_id][curr_elev_id][curr_azim_id][0][:])
                curr_hrir_r = np.copy(hrir_list[hrir_id][curr_elev_id][curr_azim_id][1][:])
        
                # Convolve AIR with HRIRs
                #print(f"curr_air shape: {curr_air.shape}, curr_hrir_l shape: {curr_hrir_l.shape}")#debugging
                curr_brir_l = sp.signal.convolve(curr_air, curr_hrir_l, mode='full', method='auto')
                curr_brir_r = sp.signal.convolve(curr_air, curr_hrir_r, mode='full', method='auto')
        
                # Accumulate for averaging
                brir_sum_l += curr_brir_l[:n_fft]
                brir_sum_r += curr_brir_r[:n_fft]
        
                # Move to next direction
                dist_counter += 1
        
            # Average results and assign to final BRIR array
            brir_reverberation[0, source_num, 0, :] = brir_sum_l / len(group)
            brir_reverberation[0, source_num, 1, :] = brir_sum_r / len(group)


    
        
        # Window initial rise
        if ir_set not in CN.AC_SPACE_LIST_SUB:
            brir_reverberation[0, :, :, :] *= direct_removal_win_b  # broadcasted multiply over channel axis
        
        # Window tail end of BRIR
        fade_start = int(44100 * (n_fft / CN.N_FFT))
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
        #optional: create compensation filter from average response and target, then equalise each BRIR
        #
        if mag_comp:
            hf.log_with_timestamp("Compensating BRIRs", gui_logger)
            
            comp_win_size = 7 if subwoofer_mode == True else 7
        
            # Level ends of spectrum
            high_freq = 12000 if ir_set in CN.AC_SPACE_LIST_SUB or subwoofer_mode == True else 15000
            low_freq_in = 20 if ir_set in CN.AC_SPACE_LIST_SUB or subwoofer_mode == True else 30
            low_freq_in_target = 150 if subwoofer_mode == True else low_freq_in#make level in sub frequencies
        
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
                brir_fft_avg_mag_sm = hf.smooth_fft_octaves(data=brir_fft_avg_mag, win_size_base = comp_win_size)
        
                # Compensation filter (smoothed)
                comp_mag = hf.db2mag(hf.mag2db(brir_fft_target_mag) - hf.mag2db(brir_fft_avg_mag_sm))
                #comp_mag = hf.smooth_fft_octaves(comp_mag, win_size_base = 7)
                #comp_eq_fir = hf.mag_to_min_fir(comp_mag, crop=1)
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
        if mag_comp == True or ir_set in CN.AC_SPACE_LIST_NOCOMP:#consider nocomp flagged cases as compensated
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

    
 
    
def raw_brirs_to_brir_set(ir_set='fw', df_comp=True, mag_comp=CN.MAG_COMP, lf_align=True,gui_logger=None):
    """
    function to convert RIRs to BRIRs
    :param ir_set: str, name of impulse response set. Must correspond to a folder in ASH-Toolset\data\interim\ir_data\<ir_group>
    :param df_comp: bool, true = apply diffuse field compensation, false = no compensation
    :param mag_comp: bool, true = apply target equalisation, false = no eq
    :param lf_align: bool, true = align brirs in time domain by peaks, false = no alignment
    :return: None
    """ 
    
    if ir_set in CN.AC_SPACE_LIST_LOWRT60:
        n_fft=CN.N_FFT
    else:
        n_fft=CN.N_FFT_L
    
    if ir_set in CN.AC_SPACE_LIST_NOROLL:
        lf_align=False
        
    #set noise reduction based on AC space
    if ir_set in CN.AC_SPACE_LIST_NR:
        noise_reduction=1
    else:
        noise_reduction=0
 
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
    direct_removal_win_b = data_pad_zeros.copy()
    direct_removal_win_b[direct_hanning_start:direct_hanning_start+int(direct_hanning_size/2)] = hann_direct
    direct_removal_win_b[direct_hanning_start+int(direct_hanning_size/2):]=data_pad_ones[direct_hanning_start+int(direct_hanning_size/2):]
    
    #impulse
    impulse=np.zeros(n_fft)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat = hf.mag2db(fr_flat_mag)
    #impulse
    impulse=np.zeros(CN.N_FFT)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat_s = hf.mag2db(fr_flat_mag)
    
    #loop through folders
    ir_in_folder = pjoin(CN.DATA_DIR_RAW, 'ir_data', 'raw_brirs',ir_set)
    
    try:
        
        log_string_a = 'Analysing inputs'
        hf.log_with_timestamp(log_string_a, gui_logger)
    
        #get number of IRs
        total_irs=0
        total_sofa_obj=0
        total_irs_per_sofa=0
        min_irs_per_sofa=10000
        hemi_split_mode=0
        for root, dirs, files in os.walk(ir_in_folder):
            for filename in files:
                #WAV mode: assumes 1 2 channel wav BRIR per direction
                if '.wav' in filename:
                    total_irs=total_irs+1
                    
                #SOFA mode assumes MultiSpeakerBRIR convention, dim mREn
                if '.sofa' in filename:
                    total_sofa_obj=total_sofa_obj+1
                    
                    #read SOFA file
                    sofa_fname = pjoin(root, filename)
          
                    #try sofasonix or sofar if fails
                    loadsofa = hf.sofa_load_object(sofa_fname)#use custom function to load object, returns dict
                    data = loadsofa['sofa_data_ir']
                    samplerate = loadsofa['sofa_samplerate']
         
                    #calculate total number of IRs available in a sofa obj
                    shape = data.shape
                    n_dims= data.ndim
                    fir_array = hf.reshape_array_to_three_dims(data)#brir case
 
                    input_measurements = len(fir_array)#The first dimension is the product of all other dimensions merged together (measurements). 2nd dimension is channels, 3rd dimension is samples
                    total_irs_per_sofa=input_measurements
                    if total_irs_per_sofa<min_irs_per_sofa:#find min number of irs in a sofa object
                        min_irs_per_sofa=total_irs_per_sofa
                        
                    #calculate total IRs across all sofa obj
                    total_irs=total_irs+input_measurements
          
        if total_irs == 5 or total_irs == 7 or total_irs > 7:
            num_out_dirs=total_irs#even if many IRs exist, store each as a distinct set
        else:
            raise ValueError('Invalid number of input BRIRs')
            
        log_string_a = 'num_out_dirs: ' + str(num_out_dirs)
        hf.log_with_timestamp(log_string_a, gui_logger)
        
        irs_per_dir=int((total_irs/num_out_dirs))
        dirs_per_hemi = int(np.ceil(num_out_dirs/2))
        #sofa calculations
        if total_sofa_obj > 0:
            #set limits
            irs_per_sofa_obj = int(total_irs/total_sofa_obj)
            #limit num irs per sofa to not exceed smallest sofa size
            irs_per_sofa_obj = min(irs_per_sofa_obj,min_irs_per_sofa)
     
            dirs_per_sofa_obj=int(np.ceil(num_out_dirs/total_sofa_obj)) 
            sofa_obj_per_hemi=int(np.ceil(total_sofa_obj/2))
 
            log_string_a = 'dirs_per_hemi: ' + str(dirs_per_hemi) + ', sofa_obj_per_hemi: ' + str(sofa_obj_per_hemi) + ', dirs_per_sofa_obj: ' + str(dirs_per_sofa_obj) + ', irs_per_dir: ' + str(irs_per_dir) + ', num_out_dirs: ' + str(num_out_dirs) + ', irs_per_sofa_obj: ' + str(irs_per_sofa_obj) + ', total_irs: ' + str(total_irs) + ', total_sofa_obj: ' + str(total_sofa_obj) 
            hf.log_with_timestamp(log_string_a, gui_logger)

        else: 
            log_string_a = 'dirs_per_hemi: ' + str(dirs_per_hemi) + ', irs_per_dir: ' + str(irs_per_dir) + ', num_out_dirs: ' + str(num_out_dirs)  + ', total_irs: ' + str(total_irs)
            hf.log_with_timestamp(log_string_a, gui_logger)
          
        #numpy array, num sets x num irs in each set x 2 channels x NFFT max samples
        brir_reverberation=np.zeros((CN.INTERIM_ELEVS,num_out_dirs,2,n_fft))
        
        #set counters
        dir_counter=0
        
        log_string_a = 'Loading IR data into array'
        hf.log_with_timestamp(log_string_a, gui_logger)

        
        #section to load data into array
        for root, dirs, files in os.walk(ir_in_folder):
            for filename in files:
                #WAV mode: assumes 1 2 channel wav BRIR per direction, simply places each wav into an output set
                if '.wav' in filename:

                    #read wav files
                    wav_fname = pjoin(root, filename)
                    samplerate, data = wavfile.read(wav_fname)
                    #samp_freq=samplerate
                    fir_array = data / (2.**31)
                    fir_length = len(fir_array)
                    extract_legth = min(n_fft,fir_length)
                    
                    #resample if sample rate is not 44100
                    if samplerate != samp_freq_ash:
                        fir_array = hf.resample_signal(fir_array, original_rate = samplerate, new_rate = samp_freq_ash)
                        log_string_a = 'source samplerate: ' + str(samplerate) + ', resampled to: '+ str(samp_freq_ash)
                        hf.log_with_timestamp(log_string_a, gui_logger)
                        populate_samples_re = round(fir_length * float(samp_freq_ash) / samplerate)#recalculate no. samples from resampled IR
                        extract_legth = min(n_fft,populate_samples_re)#smallest of nfft or resampled IR length
 
                    #load into numpy array
                    brir_reverberation[0,dir_counter,0,0:extract_legth]=fir_array[0:extract_legth,0]#L
                    brir_reverberation[0,dir_counter,1,0:extract_legth]=fir_array[0:extract_legth,1]#R
                    
                    dir_counter=dir_counter+1
        
                if '.sofa' in filename:
                    #read SOFA file
                    sofa_fname = pjoin(root, filename)
                    #try sofasonix or sofar if fails
                    loadsofa = hf.sofa_load_object(sofa_fname)#use custom function to load object, returns dict
                    data = loadsofa['sofa_data_ir']
                    samplerate = loadsofa['sofa_samplerate']
    
                    #calculate total number of IRs available in a sofa obj
                    shape = data.shape
                    n_dims= data.ndim
                    fir_array = hf.reshape_array_to_three_dims(data)#Reshapes to meet specific dimension requirements.
                    fir_array=np.transpose(fir_array)#transpose into required form: samples x channels x measurements
                    input_meas = len(fir_array[0])
                    fir_length = len(fir_array)
                    extract_legth = min(n_fft,fir_length)
                    
                    log_string_a = 'SOFA data_ir shape: ' + str(shape) + ', Input Dimensions: ' + str(n_dims)  + ', source samplerate: ' + str(samplerate) + ', input_meas: ' + str(input_meas) + ', fir_length: ' + str(fir_length)
                    hf.log_with_timestamp(log_string_a, gui_logger)
                    
                    #resample if sample rate is not 44100
                    if samplerate != samp_freq_ash:
                        fir_array = hf.resample_signal(fir_array, original_rate = samplerate, new_rate = samp_freq_ash)
                        log_string_a = 'source samplerate: ' + str(samplerate) + ', resampled to: '+ str(samp_freq_ash)
                        hf.log_with_timestamp(log_string_a, gui_logger)
                        populate_samples_re = round(fir_length * float(samp_freq_ash) / samplerate)#recalculate no. samples from resampled IR
                        extract_legth = min(n_fft,populate_samples_re)#smallest of nfft or resampled IR length
                
  
                    #sofa_mode == 0 sofa objects are divided evenly among all sets.
                    for ir_idx in range(irs_per_sofa_obj):
 
                        #load into numpy array
                        brir_reverberation[0,dir_counter,0,0:extract_legth]=fir_array[0:extract_legth,0,ir_idx]#L
                        brir_reverberation[0,dir_counter,1,0:extract_legth]=fir_array[0:extract_legth,1,ir_idx]#R
                        
                        #increment set counter
                        dir_counter=dir_counter+1
                        #dont exceed array size
                        if dir_counter >= num_out_dirs:
                            break
          
                    
        log_string_a = 'Shifting IRs'
        hf.log_with_timestamp(log_string_a, gui_logger)            
    
 
        #crop and shift raw IRs so that direct peak is at sample 50
        index_peak_ref = 40#40
        
        for dir_id in range(num_out_dirs):
            if ir_set in CN.AC_SPACE_LIST_NOROLL:
                if dir_id == 0:
                    index_peak_cur = np.argmax(np.abs(brir_reverberation[0,dir_id,0,:]))
                    ir_shift = index_peak_ref-index_peak_cur
            else:
                index_peak_cur = np.argmax(np.abs(brir_reverberation[0,dir_id,0,:]))
                ir_shift = index_peak_ref-index_peak_cur
            brir_reverberation[0,dir_id,0,:] = np.roll(brir_reverberation[0,dir_id,0,:],ir_shift)
            brir_reverberation[0,dir_id,1,:] = np.roll(brir_reverberation[0,dir_id,1,:],ir_shift)

            #set end of array to zero to remove any data shifted to end of array
            if ir_shift < 0:
                brir_reverberation[0,dir_id,0,ir_shift:] = brir_reverberation[0,dir_id,0,ir_shift:]*0#left
                brir_reverberation[0,dir_id,1,ir_shift:] = brir_reverberation[0,dir_id,1,ir_shift:]*0#right
        

        #remove direction portion of signal
        for dir_id in range(num_out_dirs):
            for chan in range(total_chan_brir):
                # RIR has been shifted so apply fade in window again to remove any overlap with HRIR
                brir_reverberation[0,dir_id,chan,:] = np.multiply(brir_reverberation[0,dir_id,chan,:],direct_removal_win_b)

        
        #perform time domain synchronous averaging
        #align in low frequencies
        if lf_align == True:
            log_string_a = 'Aligning in low frequencies'
            hf.log_with_timestamp(log_string_a, gui_logger)   
    
            #contants for TD alignment of BRIRs
            t_shift_interval = CN.T_SHIFT_INTERVAL
            min_t_shift = CN.MIN_T_SHIFT_B
            max_t_shift = CN.MAX_T_SHIFT_B
            num_intervals = int(np.abs((max_t_shift-min_t_shift)/t_shift_interval))
            order=CN.ORDER#was 6
            delay_win_min_t = CN.DELAY_WIN_MIN_A
            delay_win_max_t = CN.DELAY_WIN_MAX_A
            delay_win_hop_size = CN.DELAY_WIN_HOP_SIZE
            delay_win_hops = CN.DELAY_WIN_HOPS_A
            cutoff_alignment = CN.CUTOFF_ALIGNMENT_BRIR
            #peak to peak within a sufficiently small sample window
            peak_to_peak_window = int(np.divide(samp_freq_ash,cutoff_alignment)*0.95) #int(np.divide(samp_freq_ash,cutoff_alignment)) 
            
            delay_eval_set = np.zeros((num_out_dirs,num_intervals))
            

            #go through each room in the ordered list
            for ir in range(num_out_dirs-1):#room in range(total_airs-1)
                this_air_orig_idx=ir
                next_air_orig_idx=ir+1

                #method 2: take sum of all prior rooms and this room
                rooms_to_add = 0
                this_air = data_pad_zeros.copy()
                for cum_air in range(ir+1):
                    cum_air_orig_idx = cum_air
                    rooms_to_add = rooms_to_add+1
                    this_air = np.add(this_air,brir_reverberation[0,cum_air_orig_idx,0,:])
                this_air = np.divide(this_air,rooms_to_add) 

                calc_delay = 1
                    
                if calc_delay == 1:
                    next_air = np.copy(brir_reverberation[0,next_air_orig_idx,0,:])
                    this_ir_lp = hf.signal_lowpass_filter(this_air, cutoff_alignment, samp_freq_ash, order)
                    next_ir_lp = hf.signal_lowpass_filter(next_air, cutoff_alignment, samp_freq_ash, order)
                    
                    for delay in range(num_intervals):
                        
                        #shift next room BRIR
                        current_shift = min_t_shift+(delay*t_shift_interval)
                        next_ir_lp_shift = np.roll(next_ir_lp,current_shift)
                        #add current room BRIR to shifted next room BRIR
                        sum_ir_lp = np.add(this_ir_lp,next_ir_lp_shift)
                        #calculate group delay
     
                        peak_to_peak_iter=0
                        for hop_id in range(delay_win_hops):
                            samples = hop_id*delay_win_hop_size
                            peak_to_peak = np.abs(np.max(sum_ir_lp[delay_win_min_t+samples:delay_win_min_t+samples+peak_to_peak_window])-np.min(sum_ir_lp[delay_win_min_t+samples:delay_win_min_t+samples+peak_to_peak_window]))
                            #if this window has larger pk to pk, store in iter var
                            if peak_to_peak > peak_to_peak_iter:
                                peak_to_peak_iter = peak_to_peak
                        #store largest pk to pk distance of all windows into delay set
                        delay_eval_set[next_air_orig_idx,delay] = peak_to_peak_iter
                    
                    #shift next room by delay that has largest peak to peak distance (method 4 and 5)
                    index_shift = np.argmax(delay_eval_set[next_air_orig_idx,:])
    
                
                samples_shift=min_t_shift+(index_shift*t_shift_interval)
                brir_reverberation[0,next_air_orig_idx,0,:] = np.roll(brir_reverberation[0,next_air_orig_idx,0,:],samples_shift)#left
                brir_reverberation[0,next_air_orig_idx,1,:] = np.roll(brir_reverberation[0,next_air_orig_idx,1,:],samples_shift)#right
                
                #20240511: set end of array to zero to remove any data shifted to end of array
                if samples_shift < 0:
                    brir_reverberation[0,next_air_orig_idx,0,min_t_shift:] = brir_reverberation[0,next_air_orig_idx,0,min_t_shift:]*0#left
                    brir_reverberation[0,next_air_orig_idx,1,min_t_shift:] = brir_reverberation[0,next_air_orig_idx,1,min_t_shift:]*0#right
                        
     
        log_string_a = 'Summarising directions'
        hf.log_with_timestamp(log_string_a, gui_logger)   
          
        #
        #combine data so that no more than 7 output directions. 
        #Combines samples (dimension 4) of multiple measurements (dimension 2) by summing, reducing the number of measurements to 7.
        #
        if total_sofa_obj > 0:
            log_string_a = 'brir_reverberation shape before: ' + str(brir_reverberation.shape)
            hf.log_with_timestamp(log_string_a, gui_logger)
            
            brir_reverberation=hf.combine_measurements_4d(brir_reverberation)
            num_out_dirs=7
            
            log_string_a = 'brir_reverberation shape after: ' + str(brir_reverberation.shape)
            hf.log_with_timestamp(log_string_a, gui_logger)
        
        log_string_a = 'Applying DF EQ'
        hf.log_with_timestamp(log_string_a, gui_logger)  
        
        #
        #apply diffuse field compensation
        #
        if ir_set in CN.AC_SPACE_LIST_KU100:
            dummy_head='ku_100'
        else:
            dummy_head='other'
        
        if dummy_head == 'ku_100':
        
            #load compensation for ku100 BRIRs
            filename = 'ku_100_dummy_head_compensation.wav'
            wav_fname = pjoin(CN.DATA_DIR_INT, filename)
            samplerate, df_eq = wavfile.read(wav_fname)
            df_eq = df_eq / (2.**31)
        
        else:
  
            #determine magnitude response, assume 0 phase
            #set magnitude to 0dB
            #invert response
            #convert to IR
            #if using IFFT, result will be 0 phase and symmetric
            #if using ifftshift, result will be linear phase
            #get linear phase FIR, window, and convert to minimum phase
            #window min phase FIR again to remove artefacts
            
            num_bairs_avg = 0
            brir_fft_avg_db = fr_flat.copy()
            
            #get diffuse field spectrum
            for direc in range(num_out_dirs):
                for chan in range(total_chan_brir):
                    brir_current = np.copy(brir_reverberation[0,direc,chan,0:CN.N_FFT])#brir_out[elev][azim][chan][:]
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
            brir_fft_avg_mag = hf.level_spectrum_ends(brir_fft_avg_mag, 40, 19000, smooth_win = 10)#150
        
            #smoothing
            #octave smoothing
            brir_fft_avg_mag_sm = hf.smooth_fft_octaves(data=brir_fft_avg_mag, n_fft=n_fft, win_size_base = 20)
            
            #invert response
            brir_fft_avg_mag_inv = hf.db2mag(hf.mag2db(brir_fft_avg_mag_sm)*-1)
            #create min phase FIR
            #brir_df_inv_fir = hf.mag_to_min_fir(brir_fft_avg_mag_inv, crop=1)
            brir_df_inv_fir = hf.build_min_phase_filter(brir_fft_avg_mag_inv)
            df_eq = brir_df_inv_fir
            
        if df_comp == True:
            #convolve with inverse filter
            for direc in range(num_out_dirs):
                for chan in range(total_chan_brir):
                    brir_eq_b = np.copy(brir_reverberation[0,direc,chan,:])
                    #apply DF eq
                    brir_eq_b = sp.signal.convolve(brir_eq_b,df_eq, 'full', 'auto')
                    brir_reverberation[0,direc,chan,:] = np.copy(brir_eq_b[0:n_fft])
   

        #set each AIR to 0 level
        fb_start=int(CN.SPECT_SNAP_F0*CN.N_FFT/samp_freq_ash)
        fb_end=int(CN.SPECT_SNAP_F1*CN.N_FFT/samp_freq_ash)
        for dir_id in range(num_out_dirs):
            data_fft = np.fft.fft(brir_reverberation[0,dir_id,0,0:CN.N_FFT])
            mag_fft=np.abs(data_fft)
            average_mag_l = np.mean(mag_fft[fb_start:fb_end])
            data_fft = np.fft.fft(brir_reverberation[0,dir_id,1,0:CN.N_FFT])
            mag_fft=np.abs(data_fft)
            average_mag_r = np.mean(mag_fft[fb_start:fb_end])
            average_mag=(average_mag_l+average_mag_r)/2
            if average_mag > 0:
                for chan in range(total_chan_brir):
                    brir_reverberation[0,dir_id,chan,:] = np.divide(brir_reverberation[0,dir_id,chan,:],average_mag)
       
        
    
        #
        #optional: create compensation filter from average response and target, then equalise each BRIR
        #
        if mag_comp == True:
            
            log_string_a = 'Compensating FR'
            hf.log_with_timestamp(log_string_a, gui_logger)  
            
            num_bairs_avg = 0
            brir_fft_avg_db = fr_flat_s.copy()
            
            #calculate average response
            for direc in range(num_out_dirs):
                for chan in range(total_chan_brir):
                    brir_current = np.copy(brir_reverberation[0,direc,chan,0:CN.N_FFT])#brir_out[elev][azim][chan][:]
                    brir_current_fft = np.fft.fft(brir_current)#
                    brir_current_mag_fft=np.abs(brir_current_fft)
                    brir_current_db_fft = hf.mag2db(brir_current_mag_fft)
                    
                    brir_fft_avg_db = np.add(brir_fft_avg_db,brir_current_db_fft)
                    
                    num_bairs_avg = num_bairs_avg+1
            
            #divide by total number of brirs
            brir_fft_avg_db = brir_fft_avg_db/num_bairs_avg
            #convert to mag
            brir_fft_avg_mag = hf.db2mag(brir_fft_avg_db)
            #level ends of spectrum
            brir_fft_avg_mag = hf.level_spectrum_ends(brir_fft_avg_mag, 40, 17000, smooth_win = 10)#150
            brir_fft_avg_mag_sm = hf.smooth_fft_octaves(data=brir_fft_avg_mag, win_size_base = 20)
            
            #load target
            npy_file_name =  'reverb_target_mag_response.npy'
            reverb_folder = pjoin(CN.DATA_DIR_INT, 'reverberation')
            in_file_path = pjoin(reverb_folder,npy_file_name) 
            brir_fft_target_mag = np.load(in_file_path)
            brir_fft_target_mag = hf.level_spectrum_ends(brir_fft_target_mag, 40, 17000, smooth_win = 10)#150
            
            #create compensation filter
            comp_mag = hf.db2mag(np.subtract(hf.mag2db(brir_fft_target_mag),hf.mag2db(brir_fft_avg_mag_sm)))
            comp_mag = hf.smooth_fft_octaves(data=comp_mag, win_size_base = 30)
            #create min phase FIR
            #comp_eq_fir = hf.mag_to_min_fir(comp_mag, crop=1)
            comp_eq_fir = hf.build_min_phase_filter(comp_mag)
 
            #equalise each brir with comp filter
            for direc in range(num_out_dirs):
                for chan in range(total_chan_brir):
                    #convolve BRIR with filters
                    brir_eq_b = np.copy(brir_reverberation[0,direc,chan,:])#
                    #apply DF eq
                    brir_eq_b = sp.signal.convolve(brir_eq_b,comp_eq_fir, 'full', 'auto')
                    brir_reverberation[0,direc,chan,:] = np.copy(brir_eq_b[0:n_fft])#

            if CN.PLOT_ENABLE == True:
                print(str(num_bairs_avg))
                hf.plot_data(brir_fft_target_mag,'brir_fft_target_mag', normalise=0)
                hf.plot_data(brir_fft_avg_mag,'brir_fft_avg_mag', normalise=0)
                hf.plot_data(brir_fft_avg_mag_sm,'brir_fft_avg_mag_sm', normalise=0)  
                hf.plot_data(comp_mag,'comp_mag', normalise=0) 
        
   
        
    
    
    
        #
        #export wavs for testing
        #
        brir_out_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'comp_brirs',ir_set)
        for direc in range(num_out_dirs):
            out_file_name = ir_set+'_'+str(direc)+'_comp_brir.wav'
            out_file_path = pjoin(brir_out_folder,out_file_name)
            
            #create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            out_wav_array=np.zeros((n_fft,2))
            #grab BRIR
            out_wav_array[:,0] = np.copy(brir_reverberation[0,direc,0,:])#L
            out_wav_array[:,1] = np.copy(brir_reverberation[0,direc,1,:])#R
            
            if output_wavs == 1:
                hf.write2wav(file_name=out_file_path, data=out_wav_array, prevent_clipping=1, samplerate=samp_freq_ash)
        
        #
        #save numpy array for later use in BRIR generation functions
        #
        npy_file_name =  'reverberation_dataset_' +ir_set+'.npy'
    
        out_file_path = pjoin(brir_out_folder,npy_file_name)      
          
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        np.save(out_file_path,brir_reverberation)    
        
        log_string_a = 'Exported numpy file to: ' + out_file_path 
        hf.log_with_timestamp(log_string_a, gui_logger)
       

    
    except Exception as ex:
        log_string = 'Failed to complete BRIR processing for: ' + ir_set 
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
    
    
    
    
    
    
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
        air_fft_avg_mag_sm = hf.smooth_fft_octaves(data=air_fft_avg_mag, n_fft=n_fft)
        
        #create min phase FIR
        #avg_room_min_fir = hf.mag_to_min_fir(air_fft_avg_mag_sm, crop=1)
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
    direct_removal_win_b = data_pad_zeros.copy()
    direct_removal_win_b[direct_hanning_start:direct_hanning_start+int(direct_hanning_size/2)] = hann_direct
    direct_removal_win_b[direct_hanning_start+int(direct_hanning_size/2):]=data_pad_ones[direct_hanning_start+int(direct_hanning_size/2):]
    
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
        brir_fft_avg_mag_sm = hf.smooth_fft_octaves(data=brir_fft_avg_mag, n_fft=n_fft)
  
 
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
    
    
def calc_subrir(gui_logger=None):
    """
    function to calculate time domain average low frequency BRIR
    :return: None
    """ 
    n_fft=CN.N_FFT
    total_chan_brir=2
    output_wavs=1
    sub_eq_mode=CN.SUB_EQ_MODE
    samp_freq_ash=CN.SAMP_FREQ
  
    #windows
    data_pad_zeros=np.zeros(n_fft)
    data_pad_ones=np.ones(n_fft)
    
    #initial rise window for sub
    initial_hanning_size=50
    initial_hanning_start=0#190
    hann_initial_full=np.hanning(initial_hanning_size)
    hann_initial = np.split(hann_initial_full,2)[0]
    initial_removal_win_sub = data_pad_zeros.copy()
    initial_removal_win_sub[initial_hanning_start:initial_hanning_start+int(initial_hanning_size/2)] = hann_initial
    initial_removal_win_sub[initial_hanning_start+int(initial_hanning_size/2):]=data_pad_ones[initial_hanning_start+int(initial_hanning_size/2):]
    
    #window for late reflections
    target_rt=400#ms
    n_fade_win_start = int(((target_rt)/1000)*CN.FS)
    n_fade_win_size=np.abs(8000)*2
    wind_n_fade_full=np.bartlett(n_fade_win_size)
    win_n_fade_out = np.split(wind_n_fade_full,2)[1]
    #additional window to fade out noise
    n_fade_out_win = data_pad_zeros.copy()
    n_fade_out_win[0:n_fade_win_start] = data_pad_ones[0:n_fade_win_start]
    n_fade_out_win[n_fade_win_start:n_fade_win_start+int(n_fade_win_size/2)] = win_n_fade_out
    
    #final window for late reflections
    target_rt=400#ms
    f_fade_win_start = int(((target_rt)/1000)*CN.FS)
    f_fade_win_size=np.abs(20000)*2
    wind_f_fade_full=np.bartlett(f_fade_win_size)
    win_f_fade_out = np.split(wind_f_fade_full,2)[1]
    #additional window to fade out noise
    f_fade_out_win = data_pad_zeros.copy()
    f_fade_out_win[0:f_fade_win_start] = data_pad_ones[0:f_fade_win_start]
    f_fade_out_win[f_fade_win_start:f_fade_win_start+int(f_fade_win_size/2)] = win_f_fade_out
    
    #impulse
    impulse=np.zeros(n_fft)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat = hf.mag2db(fr_flat_mag)
    
    try:
        
 
        # load reference sub bass BRIR (FIR)
    
        selected_sub_brir= 'ash_sub_brir'       
        mat_fname = pjoin(CN.DATA_DIR_INT, 'sub_brir_dataset.mat')
        sub_brir_mat = mat73.loadmat(mat_fname)
        sub_brir_ir = np.zeros((2,n_fft))
        sub_brir_ir[0,0:CN.N_FFT] = sub_brir_mat[selected_sub_brir][0][0:CN.N_FFT]
        sub_brir_ir[1,0:CN.N_FFT] = sub_brir_mat[selected_sub_brir][1][0:CN.N_FFT]
        
        #variable crossover depending on RT60
        f_crossover_var=CN.CUTOFF_ALIGNMENT_SUBRIR#
        peak_to_peak_window_sub = int(np.divide(samp_freq_ash,f_crossover_var)*0.95)
        
        mag_range_a=int(30*(n_fft/CN.N_FFT))
        mag_range_b=int(290*(n_fft/CN.N_FFT))
        
        #set level of reference sub BRIR to 0 at low freqs
        data_fft = np.fft.fft(sub_brir_ir[0][:])
        mag_fft=np.abs(data_fft)
        average_mag = np.mean(mag_fft[mag_range_a:mag_range_b])
        if average_mag == 0:
            if CN.LOG_INFO == True:
                logging.info('0 magnitude detected')
        for chan in range(CN.TOTAL_CHAN_BRIR):
            sub_brir_ir[chan][:] = np.divide(sub_brir_ir[chan][:],average_mag)
        
        #
        #section for loading estimated sub BRIR datasets and integrating into reference
        #
        dir_id=0
        num_sub_sets_new=6#5 #input sets, not including reference sub brir
        num_sub_sets_total=num_sub_sets_new+1#total input sets, including reference sub brir
        eval_polarity = CN.EVAL_POLARITY
        #eval_polarity = False
        
        #create numpy array for new BRIR dataset   
        subrir_sets=np.zeros((num_sub_sets_total,1,total_chan_brir,n_fft))
        subrir_sets_interim=np.zeros((num_sub_sets_total,1,total_chan_brir,n_fft))
        
        #copy reference BRIR into subrir_sets array
        for chan in range(CN.TOTAL_CHAN_BRIR):
            subrir_sets[0,dir_id,chan,:] = np.copy(sub_brir_ir[chan][:])
            subrir_sets_interim[0,dir_id,chan,:] = np.copy(sub_brir_ir[chan][:])
        
        for sub_set_id in range(num_sub_sets_new):

            # load sub bass BRIR (estimated)
            if sub_set_id == 0:
                ir_set='sub_set_b'#woofer
            elif sub_set_id == 1:
                ir_set='sub_set_c'#subrir
            elif sub_set_id == 2:
                ir_set='sub_set_d'#various AIRs    
            elif sub_set_id == 3:
                ir_set='sub_set_e'#ASH listening set
            elif sub_set_id == 4:
                ir_set='sub_set_f'#studio A
            elif sub_set_id == 5:
                ir_set='sub_set_g'
            elif sub_set_id == 6:
                ir_set='sub_set_h'
            if ir_set == 'sub_set_e':#wav
                brir_reverberation=np.zeros((CN.INTERIM_ELEVS,1,2,n_fft))
                #read wav files
                brir_out_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'est_brirs',ir_set)
                if ir_set == 'sub_set_e':
                    filename='BRIR_R32_C1_E0_A30_eq.wav'
                # if sub_set_id == 6:
                #     filename='BRIR_R05_C2_E0_A-30 EQ.wav'
                wav_fname = pjoin(brir_out_folder, filename)
                samplerate, data = wavfile.read(wav_fname)
                #samp_freq=samplerate
                fir_array = data / (2.**31)
                fir_length = len(fir_array)
                extract_legth = min(n_fft,fir_length)
                
                #resample if sample rate is not 44100
                if samplerate != samp_freq_ash:
                    fir_array = hf.resample_signal(fir_array, original_rate = samplerate, new_rate = samp_freq_ash)
                    log_string_a = 'source samplerate: ' + str(samplerate) + ', resampled to: '+ str(samp_freq_ash)
                    hf.log_with_timestamp(log_string_a, gui_logger)
                    populate_samples_re = round(fir_length * float(samp_freq_ash) / samplerate)#recalculate no. samples from resampled IR
                    extract_legth = min(n_fft,populate_samples_re)#smallest of nfft or resampled IR length
        
                #load into numpy array
                brir_reverberation[0,0,0,0:extract_legth]=fir_array[0:extract_legth,0]#L
                brir_reverberation[0,0,1,0:extract_legth]=fir_array[0:extract_legth,1]#R
  
            
            else:
                brir_out_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'est_brirs',ir_set)
                npy_file_name =  'reverberation_dataset_' +ir_set+'.npy' #'reverberation_dataset_nocomp_'
                out_file_path = pjoin(brir_out_folder,npy_file_name)  
                brir_reverberation = np.load(out_file_path)
                
            

            #set level of subrir set est brir to 0 at low freqs
            #set each AIR to 0 level
            
            data_fft = np.fft.fft(brir_reverberation[0,dir_id,0,0:CN.N_FFT])
            mag_fft=np.abs(data_fft)
            average_mag_l = np.mean(mag_fft[mag_range_a:mag_range_b])
            data_fft = np.fft.fft(brir_reverberation[0,dir_id,1,0:CN.N_FFT])
            mag_fft=np.abs(data_fft)
            average_mag_r = np.mean(mag_fft[mag_range_a:mag_range_b])
            average_mag=(average_mag_l+average_mag_r)/2
            if average_mag > 0:
                for chan in range(total_chan_brir):
                    brir_reverberation[0,dir_id,chan,:] = np.divide(brir_reverberation[0,dir_id,chan,:],average_mag)
 
            #copy result into subrir_sets array
            for chan in range(CN.TOTAL_CHAN_BRIR):
                subrir_sets[sub_set_id+1,dir_id,chan,:] = np.copy(brir_reverberation[0,dir_id,chan,:])
                
            log_string_a = 'loaded sub_set_id: ' + str(sub_set_id )
            hf.log_with_timestamp(log_string_a, gui_logger)
        
        #
        #align BRIRs in time domain
        #
    
        
        delay_eval_set_sub_p = np.zeros((CN.NUM_INTERVALS_S))
        delay_eval_set_sub_n = np.zeros((CN.NUM_INTERVALS_S))
        #section to calculate best delay for next ir to align with this ir
        brir_sample = np.copy(sub_brir_ir[0][:])
        
        #go through each room in the ordered list
        for sub_set_id in range(num_sub_sets_total):#add 1 to include reference ir
            chan_ref=0
            this_ir_idx=sub_set_id
            
            if sub_set_id > 0:#ignore reference ir
                #take sum of all prior irs to get reference
                irs_to_add = 0
                prev_ir = data_pad_zeros.copy()
                for cum_air in range(sub_set_id):
                    irs_to_add = irs_to_add+1
                    prev_ir = np.add(prev_ir, subrir_sets[cum_air,dir_id,chan_ref,:])
                prev_ir = np.divide(prev_ir,irs_to_add) 
                brir_sample =prev_ir
            
                #current sample that will be shifted
                subrir_sample_p = np.copy(subrir_sets[sub_set_id,dir_id,chan_ref,:])#check first ir, first channel
                subrir_sample_n = np.multiply(np.copy(subrir_sets[sub_set_id,dir_id,chan_ref,:]),-1)
            
                #run once for positive polarity
                for delay in range(CN.NUM_INTERVALS_S):
                
                    #shift next ir (BRIR)
                    current_shift = CN.MIN_T_SHIFT_S+(delay*CN.T_SHIFT_INTERVAL_S)
                    subrir_shift_c = np.roll(subrir_sample_p,current_shift)
                    
                    #add current ir (SUBBRIR) to shifted next ir (BRIR)
                    sum_ir_c = np.add(brir_sample,subrir_shift_c)
            
                    #method 5: calculate distance from peak to peak within a 400 sample window
                    sum_ir_lp = hf.signal_lowpass_filter(sum_ir_c, f_crossover_var, CN.FS, CN.ORDER)
                    peak_to_peak_iter=0
                    for hop_id in range(CN.DELAY_WIN_HOPS):
                        samples = hop_id*CN.DELAY_WIN_HOP_SIZE
                        local_max=np.max(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+peak_to_peak_window_sub])
                        local_min=np.min(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+peak_to_peak_window_sub])
                        peak_to_peak = np.abs(local_max)#now only looking at positive peak, was np.abs(local_max-local_min)
                        #if this window has larger pk to pk, store in iter var
                        if peak_to_peak > peak_to_peak_iter:
                            peak_to_peak_iter = peak_to_peak
                    #store largest pk to pk distance of all windows into delay set
                    delay_eval_set_sub_p[delay] = peak_to_peak_iter
            
                peak_to_peak_max_p = np.max(delay_eval_set_sub_p[:])
                index_shift_p = np.argmax(delay_eval_set_sub_p[:])
                
                #run once for negative polarity
                for delay in range(CN.NUM_INTERVALS_S):
                
                    #shift next ir (BRIR)
                    current_shift = CN.MIN_T_SHIFT_S+(delay*CN.T_SHIFT_INTERVAL_S)
                    subrir_shift_c = np.roll(subrir_sample_n,current_shift)
                    
                    #add current ir (SUBBRIR) to shifted next ir (BRIR)
                    sum_ir_c = np.add(brir_sample,subrir_shift_c)
            
                    #method 5: calculate distance from peak to peak within a 400 sample window
                    sum_ir_lp = hf.signal_lowpass_filter(sum_ir_c, f_crossover_var, CN.FS, CN.ORDER)
                    peak_to_peak_iter=0
                    for hop_id in range(CN.DELAY_WIN_HOPS):
                        samples = hop_id*CN.DELAY_WIN_HOP_SIZE
                        local_max=np.max(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+peak_to_peak_window_sub])
                        local_min=np.min(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+peak_to_peak_window_sub])
                        peak_to_peak = np.abs(local_max)
                        #if this window has larger pk to pk, store in iter var
                        if peak_to_peak > peak_to_peak_iter:
                            peak_to_peak_iter = peak_to_peak
                    #store largest pk to pk distance of all windows into delay set
                    delay_eval_set_sub_n[delay] = peak_to_peak_iter
            
                peak_to_peak_max_n = np.max(delay_eval_set_sub_n[:])
                index_shift_n = np.argmax(delay_eval_set_sub_n[:])
                
                if peak_to_peak_max_p > peak_to_peak_max_n or eval_polarity == False:#or sub_set_id == 3 or sub_set_id == 6
                    index_shift=index_shift_p
                    sub_polarity=1
                else:
                    index_shift=index_shift_n
                    sub_polarity=-1
                
                #shift next ir by delay that has largest peak to peak distance
                samples_shift=CN.MIN_T_SHIFT_S+(index_shift*CN.T_SHIFT_INTERVAL_S)
                
            
                for chan in range(CN.TOTAL_CHAN_BRIR):
                    #roll subrir
                    subrir_sets[sub_set_id,dir_id,chan,:] = np.roll(subrir_sets[sub_set_id,dir_id,chan,:],samples_shift)
                    #change polarity if applicable
                    subrir_sets[sub_set_id,dir_id,chan,:] = np.multiply(subrir_sets[sub_set_id,dir_id,chan,:],sub_polarity)
                    #set end of array to zero to remove any data shifted to end of array
                    if samples_shift < 0:
                        subrir_sets[sub_set_id,dir_id,chan,samples_shift:] = subrir_sets[sub_set_id,dir_id,chan,samples_shift:]*0#left
                        subrir_sets[sub_set_id,dir_id,chan,:] = np.multiply(subrir_sets[sub_set_id,dir_id,chan,:],initial_removal_win_sub)
                    #also apply fade out window
                    subrir_sets[sub_set_id,dir_id,chan,:] = np.multiply(subrir_sets[sub_set_id,dir_id,chan,:],n_fade_out_win)
                
                
                if CN.LOG_INFO == True:
                    logging.info('delay index = ' + str(index_shift))
                    logging.info('sub polarity = ' + str(sub_polarity))
                    logging.info('samples_shift = ' + str(samples_shift))
                    logging.info('peak_to_peak_max_n = ' + str(peak_to_peak_max_n))
                    logging.info('peak_to_peak_max_p = ' + str(peak_to_peak_max_p))
        
                log_string_a = 'aligned sub_set_id: ' + str(sub_set_id )
                hf.log_with_timestamp(log_string_a, gui_logger)
                
        
        #alignment completed
        log_string_a = 'alignment complete'
        hf.log_with_timestamp(log_string_a, gui_logger)
        
        ir_set = 'sub_set_average'
        #folder for saving outputs
        brir_out_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'est_brirs',ir_set)
        
        #
        #save wavs
        #
        
        out_file_name = 'pre_eq_sub_brir_orginal.wav'
        out_file_path = pjoin(brir_out_folder,out_file_name)
        #create dir if doesnt exist
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        out_wav_array=np.zeros((n_fft,2))
        #grab BRIR
        out_wav_array[:,0] = np.copy(sub_brir_ir[0][:])#L
        out_wav_array[:,1] = np.copy(sub_brir_ir[1][:])#R
        if output_wavs == 1:
            hf.write2wav(file_name=out_file_path, data=out_wav_array, prevent_clipping=1, samplerate=samp_freq_ash)
    
        for sub_set_id in range(num_sub_sets_total):
            out_file_name = str(sub_set_id) + '_pre_eq_rolled_brir.wav'
            out_file_path = pjoin(brir_out_folder,out_file_name)
            #create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            out_wav_array=np.zeros((n_fft,2))
            #grab BRIR
            out_wav_array[:,0] = np.copy(subrir_sets[sub_set_id,dir_id,0,:])#L
            out_wav_array[:,1] = np.copy(subrir_sets[sub_set_id,dir_id,1,:])#R
            if output_wavs == 1:
                hf.write2wav(file_name=out_file_path, data=out_wav_array, prevent_clipping=1, samplerate=samp_freq_ash)
        
        #
        # first stage of EQ
        #
        
        for sub_set_id in range(num_sub_sets_total):
            sub_brir_ir_new = np.zeros((2,n_fft))
            sub_brir_ir_new[0,0:CN.N_FFT] = np.copy(subrir_sets[sub_set_id,dir_id,0,:])
            sub_brir_ir_new[1,0:CN.N_FFT] = np.copy(subrir_sets[sub_set_id,dir_id,1,:])
             
            if sub_set_id > 0:#ignore reference
                #auto filtering
                if sub_eq_mode == 1:
     
                    num_bairs_avg = 0
                    brir_fft_avg_db = fr_flat.copy()
                    #get diffuse field spectrum
                    for chan in range(total_chan_brir):
                        brir_current = np.copy(sub_brir_ir_new[chan,0:CN.N_FFT])
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
                    brir_fft_avg_mag_sm = hf.level_spectrum_ends(brir_fft_avg_mag, 5, 200, smooth_win = 15)#150
                    #smoothing
                    #octave smoothing
                    #brir_fft_avg_mag_sm = hf.smooth_fft_octaves(data=brir_fft_avg_mag_sm, n_fft=n_fft, win_size_base = 20)
                    #invert response
                    brir_fft_avg_mag_inv = hf.db2mag(hf.mag2db(brir_fft_avg_mag_sm)*-1)
                    #create min phase FIR
                    #brir_df_inv_fir = hf.mag_to_min_fir(brir_fft_avg_mag_inv, crop=1)
                    brir_df_inv_fir = hf.build_min_phase_filter(brir_fft_avg_mag_inv)
                    df_eq = brir_df_inv_fir
                    #convolve with inverse filter
                    for chan in range(total_chan_brir):
                        brir_eq_b = np.copy(sub_brir_ir_new[chan,:])
                        #apply DF eq
                        brir_eq_b = sp.signal.convolve(brir_eq_b,df_eq, 'full', 'auto')
                        sub_brir_ir_new[chan,0:CN.N_FFT] = np.copy(brir_eq_b[0:CN.N_FFT])
                    if CN.PLOT_ENABLE == True:
                        hf.plot_data(brir_fft_avg_mag,'brir_fft_avg_mag ' + str(sub_set_id))
                        hf.plot_data(brir_fft_avg_mag_sm,'brir_fft_avg_mag_sm ' + str(sub_set_id))
                        hf.plot_data(brir_fft_avg_mag_inv,'brir_fft_avg_mag_inv ' + str(sub_set_id))
                
                
                #manual filtering
                elif sub_eq_mode == 0:
                    
                    if sub_set_id == 1:
                        filter_type="peaking"
                        fc=18
                        sr=samp_freq_ash
                        q=2
                        gain_db=-1
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                
                        filter_type="peaking"
                        fc=38
                        sr=samp_freq_ash
                        q=6
                        gain_db=-3
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=43
                        sr=samp_freq_ash
                        q=4
                        gain_db=-1.0
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=56
                        sr=samp_freq_ash
                        q=6
                        gain_db=-5
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                 
                        filter_type="peaking"
                        fc=72
                        sr=samp_freq_ash
                        q=4.0
                        gain_db=-1
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=127
                        sr=samp_freq_ash
                        q=7.0
                        gain_db=-2.5
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=149
                        sr=samp_freq_ash
                        q=4.0
                        gain_db=-0.5
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=95
                        sr=samp_freq_ash
                        q=2.0
                        gain_db=0.5
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    elif sub_set_id == 2:
                        filter_type="peaking"
                        fc=25
                        sr=samp_freq_ash
                        q=5
                        gain_db=-7
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=38
                        sr=samp_freq_ash
                        q=5
                        gain_db=-2.5
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=48
                        sr=samp_freq_ash
                        q=6
                        gain_db=-5
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=76
                        sr=samp_freq_ash
                        q=6
                        gain_db=-2
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=109
                        sr=samp_freq_ash
                        q=6
                        gain_db=-3
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    elif sub_set_id == 3:
                        filter_type="peaking"
                        fc=48
                        sr=samp_freq_ash
                        q=2.5
                        gain_db=-4
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=59
                        sr=samp_freq_ash
                        q=4
                        gain_db=-7
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=82
                        sr=samp_freq_ash
                        q=5
                        gain_db=-5
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=111
                        sr=samp_freq_ash
                        q=6
                        gain_db=-9
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=130
                        sr=samp_freq_ash
                        q=7
                        gain_db=-4
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=150
                        sr=samp_freq_ash
                        q=2.5
                        gain_db=-2
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    elif sub_set_id == 4:
                        
                        filter_type="peaking"
                        fc=9
                        sr=samp_freq_ash
                        q=2.5
                        gain_db=-5
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=14
                        sr=samp_freq_ash
                        q=10
                        gain_db=-3
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=29
                        sr=samp_freq_ash
                        q=2.5
                        gain_db=-1
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=61
                        sr=samp_freq_ash
                        q=2.5
                        gain_db=-1
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=143
                        sr=samp_freq_ash
                        q=9
                        gain_db=-1
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=198
                        sr=samp_freq_ash
                        q=9
                        gain_db=-1
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                    elif sub_set_id == 5:
                        
                        filter_type="peaking"
                        fc=45
                        sr=samp_freq_ash
                        q=4
                        gain_db=-3
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=77
                        sr=samp_freq_ash
                        q=7
                        gain_db=-2.5
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=89
                        sr=samp_freq_ash
                        q=7
                        gain_db=-2.5
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=105
                        sr=samp_freq_ash
                        q=7
                        gain_db=-2.5
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
           
                        filter_type="peaking"
                        fc=119
                        sr=samp_freq_ash
                        q=10
                        gain_db=-0.5
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)                    
           
                        filter_type="peaking"
                        fc=133
                        sr=samp_freq_ash
                        q=6
                        gain_db=2
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=154
                        sr=samp_freq_ash
                        q=10
                        gain_db=-2
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=170
                        sr=samp_freq_ash
                        q=7
                        gain_db=3.5
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=186
                        sr=samp_freq_ash
                        q=10
                        gain_db=-4
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="lowshelf"
                        fc=20
                        sr=samp_freq_ash
                        q=1.0
                        gain_db=3.5
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                    elif sub_set_id == 6:
                        
                        filter_type="lowshelf"
                        fc=40
                        sr=samp_freq_ash
                        q=1.0
                        gain_db=8
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=26
                        sr=samp_freq_ash
                        q=4
                        gain_db=-3
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=27
                        sr=samp_freq_ash
                        q=7
                        gain_db=-1.5
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                        filter_type="peaking"
                        fc=65
                        sr=samp_freq_ash
                        q=2
                        gain_db=-3
                        pyquad = pyquadfilter.PyQuadFilter(sr)
                        pyquad.set_params(filter_type, fc, q, gain_db)
                        sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                        sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                        
                    
            #save interim result
            out_file_name = str(sub_set_id) + '_after_eq_rolled_brir.wav'
            out_file_path = pjoin(brir_out_folder,out_file_name)
            #create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            out_wav_array=np.zeros((n_fft,2))
            #grab BRIR
            out_wav_array[:,0] = np.copy(sub_brir_ir_new[0][:])#L
            out_wav_array[:,1] = np.copy(sub_brir_ir_new[1][:])#R
            if output_wavs == 1:
                hf.write2wav(file_name=out_file_path, data=out_wav_array, prevent_clipping=1, samplerate=samp_freq_ash) 
        
            #set level of subrir set est brir to 0 at low freqs
            #set each AIR to 0 level
            data_fft = np.fft.fft(sub_brir_ir_new[0][:])
            mag_fft=np.abs(data_fft)
            average_mag_l = np.mean(mag_fft[mag_range_a:mag_range_b])
            data_fft = np.fft.fft(sub_brir_ir_new[1][:])
            mag_fft=np.abs(data_fft)
            average_mag_r = np.mean(mag_fft[mag_range_a:mag_range_b])
            average_mag=(average_mag_l+average_mag_r)/2
            if average_mag > 0:
                for chan in range(total_chan_brir):
                    out_wav_array[:,chan] = np.divide(out_wav_array[:,chan],average_mag)
        
            #copy result into subrir_sets array
            for chan in range(CN.TOTAL_CHAN_BRIR):
                subrir_sets_interim[sub_set_id,dir_id,chan,:] = np.copy(sub_brir_ir_new[chan][:])
                
            log_string_a = 'applied EQ to sub_set_id: ' + str(sub_set_id )
            hf.log_with_timestamp(log_string_a, gui_logger)
  
        #ratios for merging
        #manual weighting
        #ratio_list = [0.05,0.21,0.16,0.14,0.28,0.16] #=sub_brir_dataset_a    
        #0(1), 3(4) & 4(5) are highest quality
        #20250405: new ratio = [0.05,0.30,0.05,0.05,0.25,0.25, 0.05]  #sub_brir_dataset_b
        #ratio_list = [0.02,0.02,0.01,0.02,0.90,0.02, 0.01] #sub_brir_dataset_c     
        #ratio_list = [0.01,0.30,0.01,0.01,0.01,0.65, 0.01] #sub_brir_dataset_d
        
        ratio_list = [0.01,0.30,0.01,0.01,0.01,0.65, 0.01]    #
        
        #calculated
        #ratio_list = hf.create_weighted_list(num_sub_sets_total, hot_index=1)
        #ratio_list = hf.create_weighted_list(num_sub_sets_total, randomize=True, seed=42)
        #ratio_list = hf.create_weighted_list(num_sub_sets_total)

        sub_brir_ir_new = np.zeros((2,n_fft))
        #prepopulate with reference subrir
        sub_brir_ir_new[0,0:CN.N_FFT] = np.multiply(sub_brir_ir[0][:],ratio_list[0])
        sub_brir_ir_new[1,0:CN.N_FFT] = np.multiply(sub_brir_ir[1][:],ratio_list[0])
        for sub_set_id in range(num_sub_sets_total):
            #merge
            for chan in range(CN.TOTAL_CHAN_BRIR):
                sub_brir_ir_new[chan,0:CN.N_FFT] = np.add(np.multiply(subrir_sets_interim[sub_set_id,dir_id,chan,:],ratio_list[sub_set_id]),np.copy(sub_brir_ir_new[chan,0:CN.N_FFT]))

        
        
        #save interim result
        out_file_name = 'pre_eq_after_merge_brir.wav'
        out_file_path = pjoin(brir_out_folder,out_file_name)
        #create dir if doesnt exist
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        out_wav_array=np.zeros((n_fft,2))
        #grab BRIR
        out_wav_array[:,0] = np.copy(sub_brir_ir_new[0][:])#L
        out_wav_array[:,1] = np.copy(sub_brir_ir_new[1][:])#R
        if output_wavs == 1:
            hf.write2wav(file_name=out_file_path, data=out_wav_array, prevent_clipping=1, samplerate=samp_freq_ash) 
        
        
        #auto filtering
        if sub_eq_mode == 1:

            num_bairs_avg = 0
            brir_fft_avg_db = fr_flat.copy()
            #get diffuse field spectrum
            for chan in range(total_chan_brir):
                brir_current = np.copy(sub_brir_ir_new[chan,0:CN.N_FFT])
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
            brir_fft_avg_mag_sm = hf.level_spectrum_ends(brir_fft_avg_mag, 3, 200, smooth_win = 3)#150
            #smoothing
            #octave smoothing
            #brir_fft_avg_mag_sm = hf.smooth_fft_octaves(data=brir_fft_avg_mag_sm, n_fft=n_fft, win_size_base = 20)
            #invert response
            brir_fft_avg_mag_inv = hf.db2mag(hf.mag2db(brir_fft_avg_mag_sm)*-1)
            #create min phase FIR
            #brir_df_inv_fir = hf.mag_to_min_fir(brir_fft_avg_mag_inv, crop=1)
            brir_df_inv_fir = hf.build_min_phase_filter(brir_fft_avg_mag_inv)
            df_eq = brir_df_inv_fir
            #convolve with inverse filter
            for chan in range(total_chan_brir):
                brir_eq_b = np.copy(sub_brir_ir_new[chan,:])
                #apply DF eq
                brir_eq_b = sp.signal.convolve(brir_eq_b,df_eq, 'full', 'auto')
                sub_brir_ir_new[chan,0:CN.N_FFT] = np.copy(brir_eq_b[0:CN.N_FFT])
            if CN.PLOT_ENABLE == True:
                hf.plot_data(brir_fft_avg_mag,'brir_fft_avg_mag after merge')
                hf.plot_data(brir_fft_avg_mag_sm,'brir_fft_avg_mag_sm after merge')
                hf.plot_data(brir_fft_avg_mag_inv,'brir_fft_avg_mag_inv after merge')
        
            #sub set B
        
            # filter_type="peaking"
            # fc=4
            # sr=samp_freq_ash
            # q=3.0
            # gain_db=-2.8
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            # filter_type="peaking"
            # fc=8
            # sr=samp_freq_ash
            # q=3.0
            # gain_db=3
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            # filter_type="peaking"
            # fc=13
            # sr=samp_freq_ash
            # q=3.5
            # gain_db=-3
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            # filter_type="peaking"
            # fc=19
            # sr=samp_freq_ash
            # q=4.0
            # gain_db=1.7
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            # filter_type="peaking"
            # fc=27
            # sr=samp_freq_ash
            # q=3.5
            # gain_db=-0.1
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            # filter_type="peaking"
            # fc=150
            # sr=samp_freq_ash
            # q=2.5
            # gain_db=-1
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            # filter_type="peaking"
            # fc=90
            # sr=samp_freq_ash
            # q=2.5
            # gain_db=-0.2
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            # filter_type="peaking"
            # fc=102
            # sr=samp_freq_ash
            # q=4.5
            # gain_db=0.9
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            # filter_type="peaking"
            # fc=112
            # sr=samp_freq_ash
            # q=6.5
            # gain_db=1.1
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            # filter_type="peaking"
            # fc=120
            # sr=samp_freq_ash
            # q=5.2
            # gain_db=-1.8
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            # filter_type="peaking"
            # fc=126
            # sr=samp_freq_ash
            # q=6.5
            # gain_db=1.4
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            #sub set C
        
            
            # filter_type="peaking"
            # fc=10
            # sr=samp_freq_ash
            # q=3.0
            # gain_db=-4
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            # filter_type="peaking"
            # fc=8
            # sr=samp_freq_ash
            # q=3.0
            # gain_db=-1.3
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
         
            # filter_type="peaking"
            # fc=15
            # sr=samp_freq_ash
            # q=7.0
            # gain_db=4
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            # filter_type="peaking"
            # fc=22
            # sr=samp_freq_ash
            # q=3.0
            # gain_db=-0.5
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            # filter_type="peaking"
            # fc=46
            # sr=samp_freq_ash
            # q=7.0
            # gain_db=-0.3
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            # filter_type="peaking"
            # fc=51
            # sr=samp_freq_ash
            # q=12.0
            # gain_db=2.1
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            # filter_type="peaking"
            # fc=55
            # sr=samp_freq_ash
            # q=8.0
            # gain_db=-0.7
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            # filter_type="peaking"
            # fc=117
            # sr=samp_freq_ash
            # q=7.0
            # gain_db=-0.5
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            
            # filter_type="peaking"
            # fc=131
            # sr=samp_freq_ash
            # q=12.0
            # gain_db=-1.0
            # pyquad = pyquadfilter.PyQuadFilter(sr)
            # pyquad.set_params(filter_type, fc, q, gain_db)
            # sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            # sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            
            #sub set D = N/A
            
            filter_type="peaking"
            fc=32
            sr=samp_freq_ash
            q=3.0
            gain_db=0.2
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            filter_type="peaking"
            fc=39
            sr=samp_freq_ash
            q=3.0
            gain_db=-0.2
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            filter_type="peaking"
            fc=91
            sr=samp_freq_ash
            q=10
            gain_db=0.1
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            filter_type="peaking"
            fc=126
            sr=samp_freq_ash
            q=7
            gain_db=0.2
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            filter_type="peaking"
            fc=130
            sr=samp_freq_ash
            q=10
            gain_db=-0.1
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            filter_type="peaking"
            fc=151
            sr=samp_freq_ash
            q=20
            gain_db=-1
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
        
        #manual filtering
        elif sub_eq_mode == 0:
  
            filter_type="lowshelf"
            fc=13
            sr=samp_freq_ash
            q=1.0
            gain_db=10.5#9.3
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            filter_type="peaking"
            fc=2
            sr=samp_freq_ash
            q=2.0
            gain_db=-6.0
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
           
            #add more filters as required
  

            if CN.LOG_INFO == True:
                logging.info('filter_type = ' + str(filter_type))
                logging.info('fc = ' + str(fc))
                logging.info('sr = ' + str(sr))
                logging.info('q = ' + str(q))
                logging.info('gain_db = ' + str(gain_db))
        
    
        #final windowing
        sub_brir_ir_new[0][:] = np.multiply(sub_brir_ir_new[0][:],f_fade_out_win)#L
        sub_brir_ir_new[1][:] = np.multiply(sub_brir_ir_new[1][:],f_fade_out_win)#R
    
        log_string_a = 'Merged and applied final EQ '
        hf.log_with_timestamp(log_string_a, gui_logger)
    
        #
        #export wavs for testing
        #

        
        out_file_name = 'after_eq_sub_brir_new.wav'
        out_file_path = pjoin(brir_out_folder,out_file_name)
        
        #create dir if doesnt exist
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        out_wav_array=np.zeros((n_fft,2))
        #grab BRIR
        out_wav_array[:,0] = np.copy(sub_brir_ir_new[0][:])#L
        out_wav_array[:,1] = np.copy(sub_brir_ir_new[1][:])#R
        
        if output_wavs == 1:
            hf.write2wav(file_name=out_file_path, data=out_wav_array, prevent_clipping=1, samplerate=samp_freq_ash)

        
        #
        #save numpy array for later use in BRIR generation functions
        #
        npy_file_name =  'sub_brir_new.npy'
        
        out_file_path = pjoin(brir_out_folder,npy_file_name)      
          
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        np.save(out_file_path,sub_brir_ir_new)    
        
        log_string_a = 'Exported numpy file to: ' + out_file_path 
        hf.log_with_timestamp(log_string_a, gui_logger)
    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to complete sub BRIR processing'
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
                room_target_mag_new_sm = hf.smooth_fft_octaves(data=room_target_mag_new, n_fft=n_fft,fund_freq=150,win_size_base = 3)
                #create min phase FIR
                #room_target_min_fir = hf.mag_to_min_fir(room_target_mag_new_sm, crop=1)
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
            room_target_mag_new_sm = hf.smooth_fft(data=room_target_mag_new, n_fft=n_fft, crossover_f=2000, win_size_a = 3, win_size_b = 500)
            
            #create min phase FIR
            #room_target_min_fir = hf.mag_to_min_fir(room_target_mag_new_sm, crop=1)
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
            room_target_mag_new_sm = hf.smooth_fft(data=room_target_mag_new, n_fft=n_fft, crossover_f=2000, win_size_a = 3, win_size_b = 500)
            
            #create min phase FIR
            #room_target_min_fir = hf.mag_to_min_fir(room_target_mag_new_sm, crop=1)
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
    Function finds latest versions of acoustic spaces, compares with current versions
    """
    
    
    try:
        
        #log results
        log_string = 'Checking for acoustic space updates'
        hf.log_with_timestamp(log_string, gui_logger)
        
        #read local metadata from reverberation_metadata.csv
        #place rows into dictionary list
        local_meta_dict_list = []

        #directories
        csv_directory = pjoin(CN.DATA_DIR_INT, 'reverberation')
        #read metadata from csv. Expects reverberation_metadata.csv 
        metadata_file_name = 'reverberation_metadata.csv'
        metadata_file = pjoin(csv_directory, metadata_file_name)
        with open(metadata_file, encoding='utf-8-sig', newline='') as inputfile:
            reader = DictReader(inputfile)
            for row in reader:#rows 2 and onward
                #store each row as a dictionary
                #append to list of dictionaries
                local_meta_dict_list.append(row)
                    
                
        #download latest metadata file from gdrive
        #read metadata file
        #place into dictionary list
        web_meta_dict_list = []
        
        #get version of online database
        url = "https://drive.google.com/file/d/14eX5wLiyMCuS4-2aYBfbWRMXFgYc6Bm-/view?usp=drive_link"
        dl_file = pjoin(csv_directory, 'reverberation_metadata_latest.csv')
        gdown.download(url, dl_file, fuzzy=True)

        with open(dl_file, encoding='utf-8-sig', newline='') as inputfile:
            reader = DictReader(inputfile)
            for row in reader:#rows 2 and onward
                #store each row as a dictionary
                #append to list of dictionaries
                web_meta_dict_list.append(row)
 
        mismatches=0
        updates_perf=0
        if not web_meta_dict_list:
            raise ValueError('latest metadata is empty')
        if not local_meta_dict_list:
            raise ValueError('local metadata is empty') 
            
        #for each space in latest dict list
        for space_w in web_meta_dict_list:
            name_w = space_w.get('name_src')
            name_gui_w = space_w.get('name_gui')
            vers_w = space_w.get('version')
            gdrive_link = space_w.get('gdrive_link')
            rev_folder = space_w.get('folder')
            file_name = space_w.get('file_name')
            dl_file = pjoin(csv_directory,rev_folder, file_name+'.npy')
            match_found=0
            update_required=0
            for space_l in local_meta_dict_list:
                name_l = space_l.get('name_src')
                vers_l = space_l.get('version')
                #find matching space in local version
                if name_w == name_l:
                    match_found=1
                    #compare version with local version
                    #case for mismatching versions for matching name
                    if vers_w != vers_l:
                        mismatches=mismatches+1 
                        update_required=1
                        #if not matching, print details
                        log_string = 'New version ('+vers_w+') available for: ' + name_gui_w
                        hf.log_with_timestamp(log_string, gui_logger)

                            
            #this space not found in local metadata, must be new space
            if match_found==0:
                mismatches=mismatches+1 
                update_required=1
                log_string = 'New acoustic space available: ' + name_gui_w
                hf.log_with_timestamp(log_string, gui_logger)
                    
            #if download updates enabled
            #for each version mismatch, download latest dataset from gdrive and place into relevant folder
            if download_updates == True and update_required > 0:
                log_string = 'Downloading update'
                hf.log_with_timestamp(log_string, gui_logger)
   
                gdown.download(gdrive_link, dl_file, fuzzy=True)
                
                log_string = 'Latest version of dataset: ' + name_gui_w + ' downloaded and saved to: ' + dl_file
                hf.log_with_timestamp(log_string, gui_logger)
                updates_perf=updates_perf+1
                    
        #finally, download latest metadata file and replace local file
        if updates_perf >=1: 
            url = "https://drive.google.com/file/d/14eX5wLiyMCuS4-2aYBfbWRMXFgYc6Bm-/view?usp=drive_link"
            dl_file = pjoin(csv_directory, 'reverberation_metadata.csv')
            gdown.download(url, dl_file, fuzzy=True)
        
        
        #if no mismatches flagged, print message
        if mismatches == 0:
            log_string = 'No updates available'
            hf.log_with_timestamp(log_string, gui_logger)

    except Exception as ex:
        log_string = 'Failed to validate versions or update data'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #print message
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    