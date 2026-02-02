# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 16:20:38 2025

@author: Shanon
"""

import numpy as np
from os.path import join as pjoin
from ash_toolset import helper_functions as hf
from ash_toolset import constants as CN
from csv import DictReader
import os
from math import sqrt
from pathlib import Path
import threading
import scipy as sp
import dearpygui.dearpygui as dpg
import logging
import json
import matplotlib.pyplot as plt

              
logger = logging.getLogger(__name__)
log_info=1



def get_listener_list(listener_type="", dataset_name="", gui_logger=None):
    """
    Retrieves a list of listeners based on the listener type and dataset name.

    Args:
        listener_type (str, optional): The type of listener ('Dummy Head / Head & Torso Simulator' or 'Human Listener').
        dataset_name (str, optional): The name of the dataset to filter by (only used for 'Human Listener').
        gui_logger (object, optional): A logger object for GUI-related logging.

    Returns:
        list: A list of listener names.
    """
    listener_list = []

    try:
        # --- Human or Dummy Head listeners ---
        if listener_type in ['Dummy Head / Head & Torso Simulator', 'Human Listener']:
            # Filter HRTF_DATA based on listener_type and dataset_name
            listener_list_filtered = [
                row['name_gui'] for row in CN.HRTF_DATA
                if row.get('hrtf_type') == listener_type
                and (dataset_name == "" or row.get('dataset') == dataset_name)
            ]
            # Sort alphabetically
            listener_list = sorted(listener_list_filtered)

        # --- Favourites ---
        elif listener_type == 'Favourites':
            try:
                if dpg.does_item_exist("hrtf_add_favourite"):
                    listener_list = dpg.get_item_user_data("hrtf_add_favourite") or CN.HRTF_BASE_LIST_FAV
                else:
                    listener_list = CN.HRTF_BASE_LIST_FAV
            except Exception:
                listener_list = CN.HRTF_BASE_LIST_FAV
            listener_list.sort()

        # --- User HRIR / Sofa entries ---
        else:
            listener_list = get_listener_list_user()
            if not listener_list:
                listener_list = [CN.HRTF_USER_SOFA_DEFAULT]

    except Exception as ex:
        log_string = f"Error occurred in get_listener_list: {ex}"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type=2, exception=ex)

    return listener_list

def get_listener_list_user(directory=CN.DATA_DIR_SOFA_USER, extension='sofa', gui_logger=None):
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
    except Exception as ex:
        log_string=f"Error occurred: {ex}"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
        return file_list # Or you might want to raise an exception here

    return file_list 



def get_name_short(listener_type="", dataset_name="", name_gui="", gui_logger=None):
    """
    Retrieves a short name based on the listener type, dataset name, and GUI name.

    Args:
        listener_type (str, optional): The type of listener ('Dummy Head / Head & Torso Simulator' or 'Human Listener').
        dataset_name (str, optional): The name of the dataset to filter by.
        name_gui (str, optional): The GUI name of the listener.
        gui_logger (object, optional): A logger object for GUI-related logging.

    Returns:
        str: The short name corresponding to the listener, or name_gui if not found.
    """
    

    
    name_short = name_gui  # default to GUI name

    try:
        if listener_type in ['Dummy Head / Head & Torso Simulator', 'Human Listener']:
            # Filter HRTF_DATA for the matching row
            for row in CN.HRTF_DATA:
                if (
                    row.get('hrtf_type') == listener_type
                    and row.get('dataset') == dataset_name
                    and row.get('name_gui') == name_gui
                ):
                    name_short = row.get('name_short', name_gui)
                    break  # stop after first match

        # Else case: user SOFA or favourites already defaults to name_gui

    except Exception as ex:
        log_string = f"Error occurred in get_name_short: {ex}"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type=2, exception=ex)

    return name_short



def get_hrtf_info_from_name_short(name_short: str = "", gui_logger=None) -> tuple[str, str, str]:
    """
    Retrieves HRTF type, dataset, and GUI name for a given name_short from HRTF_DATA.

    Args:
        name_short (str, optional): The unique short name to look up. Defaults to "".
        gui_logger (object, optional): A logger object for GUI-related logging. Defaults to None.

    Returns:
        tuple[str, str, str]: (hrtf_type, dataset, name_gui) if found, otherwise ("", "", "").
    """
    hrtf_type = ""
    dataset = ""
    name_gui = ""

    try:
        if name_short:
            # Filter HRTF_DATA for the matching name_short
            for row in CN.HRTF_DATA:
                if row.get("name_short") == name_short:
                    hrtf_type = row.get("hrtf_type", "")
                    dataset = row.get("dataset", "")
                    name_gui = row.get("name_gui", "")
                    break  # stop after first match

    except Exception as ex:
        log_string = f"Error occurred in get_hrtf_info_from_name_short for name_short='{name_short}': {ex}"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type=2, exception=ex)

    return hrtf_type, dataset, name_gui



def get_sofa_url(listener_type="", dataset_name="", name_gui="", gui_logger=None):
    """
    Retrieves a SOFA file URL based on the listener type, dataset name, and GUI name from HRTF_DATA.

    Args:
        listener_type (str, optional): The type of listener ('Dummy Head / Head & Torso Simulator' or 'Human Listener').
        dataset_name (str, optional): The name of the dataset to filter by.
        name_gui (str, optional): The GUI name of the listener.
        gui_logger (object, optional): A logger object for GUI-related logging.

    Returns:
        str: The SOFA file URL if found, otherwise "".
    """
    url = ""

    try:
        if listener_type in ['Dummy Head / Head & Torso Simulator', 'Human Listener']:
            # Filter HRTF_DATA for the matching row
            for row in CN.HRTF_DATA:
                if (
                    row.get('hrtf_type') == listener_type
                    and row.get('dataset') == dataset_name
                    and row.get('name_gui') == name_gui
                ):
                    url = row.get('url', "")
                    break  # stop after first match

        # else: url remains ""

    except Exception as ex:
        log_string = f"Error occurred in get_sofa_url for dataset='{dataset_name}', name_gui='{name_gui}': {ex}"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type=2, exception=ex)

    return url




def get_alternative_url(listener_type="", dataset_name="", name_gui="", gui_logger=None):
    """
    Retrieves the alternative SOFA file URL based on listener type, dataset name, and GUI name from HRTF_DATA.

    Args:
        listener_type (str, optional): The type of listener ('Dummy Head / Head & Torso Simulator' or 'Human Listener').
        dataset_name (str, optional): The name of the dataset to filter by.
        name_gui (str, optional): The GUI name of the listener.
        gui_logger (object, optional): A logger object for GUI-related logging.

    Returns:
        str: The alternative SOFA file URL if found, otherwise "".
    """
    url = ""

    try:
        if listener_type in ['Dummy Head / Head & Torso Simulator', 'Human Listener']:
            # Filter HRTF_DATA for the matching row
            for row in CN.HRTF_DATA:
                if (
                    row.get('hrtf_type') == listener_type
                    and row.get('dataset') == dataset_name
                    and row.get('name_gui') == name_gui
                ):
                    url = row.get('url_alternative', "")
                    break  # stop after first match

        # else: url remains ""

    except Exception as ex:
        log_string = f"Error occurred in get_alternative_url for dataset='{dataset_name}', name_gui='{name_gui}': {ex}"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type=2, exception=ex)

    return url



def get_polarity(listener_type="", dataset_name="", name_gui="", gui_logger=None):
    """
    Retrieves the polarity string for a listener based on listener type, dataset name, and GUI name from HRTF_DATA.

    Args:
        listener_type (str, optional): The type of listener ('Dummy Head / Head & Torso Simulator' or 'Human Listener').
        dataset_name (str, optional): The name of the dataset to filter by.
        name_gui (str, optional): The GUI name of the listener.
        gui_logger (object, optional): A logger object for GUI-related logging.

    Returns:
        str: The polarity value ('yes', 'no', etc.) if found; otherwise "no".
    """
    flip_polarity = "no"  # default

    try:
        if listener_type in ['Dummy Head / Head & Torso Simulator', 'Human Listener']:
            # Filter HRTF_DATA for the matching row
            for row in CN.HRTF_DATA:
                if (
                    row.get('hrtf_type') == listener_type
                    and row.get('dataset') == dataset_name
                    and row.get('name_gui') == name_gui
                ):
                    flip_polarity = row.get('flip_polarity', "no")
                    break  # stop after first match

        # else: flip_polarity stays as default "no"

    except Exception as ex:
        log_string = (
            f"Error occurred in get_polarity for dataset='{dataset_name}', "
            f"name_gui='{name_gui}': {ex}"
        )
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type=2, exception=ex)

    return flip_polarity




def get_gdrive_url_high(listener_type="", dataset_name="", name_gui="", gui_logger=None):
    """
    Retrieves the high-resolution GDrive URL for a listener based on listener type, dataset name, and GUI name from HRTF_DATA.

    Args:
        listener_type (str, optional): The type of listener ('Dummy Head / Head & Torso Simulator' or 'Human Listener').
        dataset_name (str, optional): The name of the dataset to filter by.
        name_gui (str, optional): The GUI name of the listener.
        gui_logger (object, optional): A logger object for GUI-related logging.

    Returns:
        str: The high-resolution GDrive URL if found; otherwise "".
    """
    url = ""

    try:
        if listener_type in ['Dummy Head / Head & Torso Simulator', 'Human Listener']:
            # Filter HRTF_DATA for the matching row
            for row in CN.HRTF_DATA:
                if (
                    row.get('hrtf_type') == listener_type
                    and row.get('dataset') == dataset_name
                    and row.get('name_gui') == name_gui
                ):
                    url = row.get('gdrive_url_high', "")
                    break  # stop after first match

        # else: url remains ""

    except Exception as ex:
        log_string = (
            f"Error occurred in get_gdrive_url_high for dataset='{dataset_name}', "
            f"name_gui='{name_gui}': {ex}"
        )
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type=2, exception=ex)

    return url
     
def hrir_metadata_updates(download_updates=False, gui_logger=None):
    """ 
    Function finds latest version of HRTF metadata file, compares with current versions
    """
    
    
    try:
        
        update_required=0
        
        #log results
        log_string = 'Checking for HRTF updates'
        hf.log_with_timestamp(log_string, gui_logger)
        
        #read local metadata from reverberation_metadata.csv
        #place rows into dictionary list
        local_meta_dict_list = []

        #directories
        csv_directory = CN.DATA_DIR_HRIR_NPY
        #read metadata from csv. Expects reverberation_metadata.csv 
        metadata_file_name = CN.HRIR_METADATA_NAME
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
        url = CN.HRTF_META_URL
        dl_file = pjoin(csv_directory, 'hrir_metadata_latest.csv')
        response = hf.download_file(url=url, save_location=dl_file, gui_logger=gui_logger)
        if response is not True:
            raise RuntimeError('Failed to download latest HRTF metadata')

        with open(dl_file, encoding='utf-8-sig', newline='') as inputfile:
            reader = DictReader(inputfile)
            for row in reader:#rows 2 and onward
                #store each row as a dictionary
                #append to list of dictionaries
                web_meta_dict_list.append(row)
 
        mismatches=0

        if not web_meta_dict_list:
            raise ValueError('latest metadata is empty')
        if not local_meta_dict_list:
            raise ValueError('local metadata is empty') 
            
        #for each space in latest dict list
        for space_w in web_meta_dict_list:
            name_w = space_w.get('name_short')
            vers_w = space_w.get('local_version')

            match_found=0
            
            for space_l in local_meta_dict_list:
                name_l = space_l.get('name_short')
                vers_l = space_l.get('local_version')
                #find matching hrtf in local version
                if name_w == name_l:
                    match_found=1
                    #compare version with local version
                    #case for mismatching versions for matching name
                    if vers_w != vers_l:
                        mismatches=mismatches+1 
                        update_required=1
                        #if not matching, print details
                        log_string = 'New version ('+vers_w+') available for: ' + name_w
                        hf.log_with_timestamp(log_string, gui_logger, log_type=1)

                            
            #this hrtf not found in local metadata, must be new space
            if match_found==0:
                mismatches=mismatches+1 
                update_required=1
                log_string = 'New hrtf available: ' + name_w
                hf.log_with_timestamp(log_string, gui_logger, log_type=1)
                    
         
        #finally, download latest metadata file and replace local file
        if download_updates == True and update_required > 0: 
            log_string = 'Updating list of HRTF datasets '
            hf.log_with_timestamp(log_string, gui_logger)
  
            dl_file = metadata_file
            response = hf.download_file(url=url, save_location=dl_file, gui_logger=gui_logger)
            if response is not True:
                raise RuntimeError('Failed to download latest HRTF metadata')
            
            log_string = 'list of HRTF datasets updated. Restart required.'
            hf.log_with_timestamp(log_string, gui_logger)
      
        
        #if no mismatches flagged, print message
        if mismatches == 0:
            log_string = 'No updates available'
            hf.log_with_timestamp(log_string, gui_logger)
                
                

    except Exception as ex:
        log_string = 'Failed to validate versions or update data'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
    
  
    
def infer_hrir_spatial_res(hrir_arr, spatial_res=None, logger=None):
    """
    Infer the highest spatial_res compatible with an HRIR dataset.

    Parameters
    ----------
    hrir_arr : np.ndarray
        HRIR array, 4D [elev, azim, ch, samples] or
        5D [listener, elev, azim, ch, samples]
    spatial_res : int or None, optional
        User-supplied spatial_res to validate
    logger : callable, optional
        Logger function

    Returns
    -------
    inferred_spatial_res : int
        Highest spatial_res index compatible with HRIR grid

    Raises
    ------
    ValueError
        If no spatial_res definitions match the HRIR dataset
        If provided spatial_res is incompatible
    """

    # Normalize HRIR shape
    if hrir_arr.ndim == 5:
        hrir = hrir_arr[0]
    elif hrir_arr.ndim == 4:
        hrir = hrir_arr
    else:
        raise ValueError(
            f"HRIR array must be 4D or 5D, got shape {hrir_arr.shape}"
        )

    total_elev, total_azim, total_chan, _ = hrir.shape

    if total_chan < 2:
        raise ValueError(
            f"HRIR dataset must have at least 2 channels, got {total_chan}"
        )

    matching_res = []

    # Test all spatial_res definitions
    for res in range(len(CN.SPATIAL_RES_ELEV_MIN_IN)):
        elev_min = CN.SPATIAL_RES_ELEV_MIN_IN[res]
        elev_max = CN.SPATIAL_RES_ELEV_MAX_IN[res]
        azim_min = CN.SPATIAL_RES_AZIM_MIN_IN[res]
        azim_max = CN.SPATIAL_RES_AZIM_MAX_IN[res]
        elev_step = CN.SPATIAL_RES_ELEV_NEAREST_IN[res]
        azim_step = CN.SPATIAL_RES_AZIM_NEAREST_IN[res]
    
        elev_expected = np.arange(elev_min, elev_max + elev_step, elev_step)
        azim_expected = np.arange(azim_min, azim_max + azim_step, azim_step)
    
        if total_elev == len(elev_expected) and total_azim == len(azim_expected):
            matching_res.append(res)

    if not matching_res:
        raise ValueError(
            f"HRIR grid ({total_elev} elev, {total_azim} azim) "
            f"does not match any known spatial_res definition"
        )

    inferred_res = max(matching_res)

    # Validate user-supplied spatial_res if provided
    if spatial_res is not None and spatial_res != inferred_res:
        hf.log_with_timestamp(
                f"Provided spatial_res={spatial_res} does not match HRIR dataset. "
                f"Inferred highest compatible spatial_res={inferred_res}"
            )

    hf.log_with_timestamp(
            f"Inferred HRIR spatial_res={inferred_res} "
            f"(elev={total_elev}, azim={total_azim})"
        )

    return inferred_res

def log_sofa_spatial_stats(sofa_source_positions, gui_logger=None):
    """
    Log spatial coverage and resolution statistics for a SOFA dataset.

    Logs:
        - Min / max azimuth
        - Min / max elevation
        - Min / median / max spacing (resolution) for azimuth and elevation
        - Min / max distance (if present)

    Parameters
    ----------
    sofa_source_positions : np.ndarray
        SOFA SourcePosition array.
        Expected shape (N, 2) or (N, 3):
            [:, 0] -> azimuth (deg)
            [:, 1] -> elevation (deg)
            [:, 2] -> distance (optional)
    """
    try:
        if sofa_source_positions is None or sofa_source_positions.size == 0:
            hf.log_with_timestamp(
                "SOFA spatial stats: Source positions array is empty.",
                gui_logger
            )
            return

        az = sofa_source_positions[:, 0].astype(float)
        el = sofa_source_positions[:, 1].astype(float)

        # Min / max coverage
        az_min, az_max = az.min(), az.max()
        el_min, el_max = el.min(), el.max()

        # Unique sorted values
        az_unique = np.unique(np.sort(az))
        el_unique = np.unique(np.sort(el))

        # Resolution (spacing)
        az_diffs = np.diff(az_unique) if az_unique.size > 1 else np.array([0.0])
        el_diffs = np.diff(el_unique) if el_unique.size > 1 else np.array([0.0])

        az_step_min = az_diffs.min()
        az_step_med = np.median(az_diffs)
        az_step_max = az_diffs.max()

        el_step_min = el_diffs.min()
        el_step_med = np.median(el_diffs)
        el_step_max = el_diffs.max()

        # Distance (optional)
        dist_str = ""
        if sofa_source_positions.shape[1] > 2:
            dist = sofa_source_positions[:, 2].astype(float)
            dist_min, dist_max = dist.min(), dist.max()
            dist_str = f"\n  Distance: {dist_min:.3f} to {dist_max:.3f}"

        log_string = (
            "SOFA spatial coverage:\n"
            f"  Azimuth: {az_min:.1f} deg to {az_max:.1f} deg "
            f"(step min={az_step_min:.2f}, med={az_step_med:.2f}, max={az_step_max:.2f} deg)\n"
            f"  Elevation: {el_min:.1f} deg to {el_max:.1f} deg "
            f"(step min={el_step_min:.2f}, med={el_step_med:.2f}, max={el_step_max:.2f} deg)"
            f"{dist_str}"
        )

        hf.log_with_timestamp(log_string, gui_logger)

    except Exception as ex:
        hf.log_with_timestamp(
            "Failed to compute SOFA spatial statistics.",
            gui_logger,
            log_type=2,
            exception=ex
        )
        


def generate_rounded_directions_from_indices(
        nearest_dir_indices,
        sofa_source_positions,
        spatial_res=2
):
    """
    Generate a list of rounded (azimuth, elevation) directions corresponding
    to a list of SOFA nearest-direction indices.

    Each index is mapped back to its source azimuth/elevation, then rounded
    according to the spatial resolution grid.

    Args:
        nearest_dir_indices (list[int]):
            List of SOFA measurement indices used during nearest-direction mapping.
            Duplicates are allowed and preserved.

        sofa_source_positions (np.ndarray):
            Array of shape (N,3) with columns [azimuth, elevation, distance].

        spatial_res (int):
            Spatial resolution index used to determine rounding increment.

    Returns:
        List of tuples:
            [(azimuth_deg, elevation_deg), ...] rounded directions
            (same length and ordering as nearest_dir_indices)
    """
    if (
        sofa_source_positions is None
        or len(sofa_source_positions) == 0
        or nearest_dir_indices is None
        or len(nearest_dir_indices) == 0
    ):
        return []

    round_deg = CN.SPATIAL_RES_AZIM_NEAREST_IN[spatial_res]

    rounded_directions = []

    for idx in nearest_dir_indices:
        idx = int(idx)

        if idx < 0 or idx >= sofa_source_positions.shape[0]:
            continue  # safety guard

        azim = float(sofa_source_positions[idx, 0])
        elev = float(sofa_source_positions[idx, 1])

        # Round to nearest spatial grid
        azim_r = round_deg * round(azim / round_deg)
        elev_r = round_deg * round(elev / round_deg)

        # Wrap azimuth to [0, 360)
        azim_r = azim_r % 360

        rounded_directions.append((int(azim_r), int(elev_r)))

    return rounded_directions





def adjust_ctf_based_on_directions(
    rounded_directions,
    n_fft=CN.N_FFT,
    spatial_res=2,
    plot=CN.PLOT_ENABLE
):
    """
    Generate a linear-scale CTF adjustment filter from a reference HRIR dataset
    for sparse directions, optionally plotting the CTFs.

    Uses RFFT to match pipeline.

    Args:
        rounded_directions (list of tuples): [(azim_deg, elev_deg), ...]
        n_fft (int): FFT length to match HRIR workflow
        spatial_res (int): Spatial resolution index
        plot (bool): If True, plot reference, uneven, and adjusted CTFs

    Returns:
        adj_mag_sm (np.ndarray): Adjustment filter (linear magnitude), shape (n_fft//2 + 1,)
    """

    # --- Spatial resolution setup ---
    if 0 <= spatial_res < CN.NUM_SPATIAL_RES:
        elev_min = CN.SPATIAL_RES_ELEV_MIN_IN[spatial_res]
        elev_max = CN.SPATIAL_RES_ELEV_MAX_IN[spatial_res]
        elev_step = CN.SPATIAL_RES_ELEV_NEAREST_IN[spatial_res]
        azim_step = CN.SPATIAL_RES_AZIM_NEAREST_IN[spatial_res]
        total_azims = max(1, int(360 / azim_step))
        total_elevs = max(1, int((elev_max - elev_min) / elev_step + 1))
    else:
        raise ValueError("Invalid spatial resolution")

    # --- Load reference HRIR dataset ---
    npy_fname = pjoin(CN.DATA_DIR_HRIR_NPY_INTRP, "hrir_averaged_reference.npy")
    hrir_list = hf.load_convert_npy_to_float64(npy_fname)

    # --- Remove singleton first dimension if present ---
    if hrir_list.ndim == 5 and hrir_list.shape[0] == 1:
        hrir_list = np.squeeze(hrir_list, axis=0)

    if hrir_list.ndim != 4:
        raise ValueError(f"Unexpected HRIR array shape: {hrir_list.shape}")

    # --- Flatten HRIR to (directions, channels, samples) ---
    elevs, azims, chans, samples = hrir_list.shape
    hrir_flat = hrir_list.reshape(-1, chans, samples)

    # --- Compute indices of selected directions ---
    selected_indices = []
    for azim, elev in rounded_directions:
        e_idx = int(round((elev - elev_min) / elev_step))
        e_idx = np.clip(e_idx, 0, elevs - 1)

        azim = azim % 360
        a_idx = int(round(azim / azim_step)) % azims
        a_idx = np.clip(a_idx, 0, azims - 1)

        flat_idx = e_idx * azims + a_idx
        selected_indices.append(flat_idx)
    selected_indices = list(set(selected_indices))  # avoid duplicates

    # --- FFT magnitude using RFFT ---
    n_rfft = n_fft // 2 + 1
    fft_all = np.abs(np.fft.rfft(hrir_flat, n=n_fft, axis=-1))  # shape (directions, channels, n_rfft)
    fft_sel = np.abs(np.fft.rfft(hrir_flat[selected_indices], n=n_fft, axis=-1))

    # --- Convert to dB ---
    fft_all_db = hf.mag2db(fft_all)
    fft_sel_db = hf.mag2db(fft_sel)

    # --- Log-domain averaging over directions and channels ---
    all_ctf_db = np.mean(fft_all_db, axis=(0, 1))
    sel_ctf_db = np.mean(fft_sel_db, axis=(0, 1))

    # --- Adjustment filter (DB DOMAIN) ---
    adj_ctf_db = all_ctf_db - sel_ctf_db

    # --- Convert back to linear ---
    adj_mag = hf.db2mag(adj_ctf_db)

    # --- Smooth (linear domain) ---
    #adj_mag_sm = hf.smooth_freq_octaves(adj_mag, n_fft=n_fft)
    adj_mag_sm = hf.smooth_gaussian_octave(data=adj_mag, n_fft=n_fft, fraction=12)

    # --- Optional plotting ---
    if plot:
        freq_axis = np.linspace(0, CN.FS / 2, n_rfft)
        f_min, f_max = 20, 20000
        freq_mask = (freq_axis >= f_min) & (freq_axis <= f_max)

        plt.figure(figsize=(8, 4))
        plt.semilogx(freq_axis[freq_mask], all_ctf_db[freq_mask], label="Reference CTF (all)")
        plt.semilogx(freq_axis[freq_mask], sel_ctf_db[freq_mask], label="Uneven CTF (selected)")
        plt.semilogx(freq_axis[freq_mask], adj_ctf_db[freq_mask], label="Adjustment (dB)")
        plt.semilogx(freq_axis[freq_mask], hf.mag2db(adj_mag_sm[freq_mask]), label="Adjustment (smoothed)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("CTF Adjustment Filter (Log-Domain)")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return adj_mag_sm


def sofa_normalise_source_positions(sofa_dict, gui_logger=None):
    """
    Normalise SOFA source positions so they are always:
        - spherical
        - degrees
        - azimuth: 0–360° (counter-clockwise)
        - elevation: -90° … +90°
        - columns: [azimuth, elevation, radius]

    Modifies sofa_dict in-place.

    Returns:
        dict with keys:
            converted (bool)
            original_type
            original_units
    """
    pos = sofa_dict.get("sofa_source_positions")
    pos_type = str(sofa_dict.get("sofa_source_position_type", "")).lower()
    pos_units = str(sofa_dict.get("sofa_source_position_units", "")).lower()

    if pos is None:
        raise ValueError("SOFA source positions missing")

    converted = False
    original_type = pos_type
    original_units = pos_units

    # --------------------------------------------------
    # Cartesian → spherical
    # --------------------------------------------------
    if pos_type == "cartesian":
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]

        r = np.sqrt(x**2 + y**2 + z**2)

        with np.errstate(divide="ignore", invalid="ignore"):
            az = np.degrees(np.arctan2(y, x))      # CCW, -180..180
            el = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))

        pos = np.column_stack((az, el, r))

        converted = True
        pos_type = "spherical"
        pos_units = "degree"

        hf.log_with_timestamp(
            "Converted SOFA SourcePosition from cartesian → spherical (degrees)",
            gui_logger
        )

    # --------------------------------------------------
    # Spherical radians → degrees
    # --------------------------------------------------
    if pos_type == "spherical" and "rad" in pos_units:
        pos = pos.copy()
        pos[:, 0:2] = np.degrees(pos[:, 0:2])

        converted = True
        pos_units = "degree"

        hf.log_with_timestamp(
            "Converted SOFA SourcePosition from radians → degrees",
            gui_logger
        )

    # --------------------------------------------------
    # Enforce canonical angular ranges
    # --------------------------------------------------
    if pos_type == "spherical":
        pos = pos.copy()

        # Azimuth → 0..360 CCW
        az = np.mod(pos[:, 0], 360.0)

        # Fold 360 → 0 explicitly (numerical safety)
        az[np.isclose(az, 360.0)] = 0.0
        
        pos[:, 0] = az

        # Elevation already -90..+90 by construction,
        # but clip for numerical safety
        el = pos[:, 1]
        pos[:, 1] = np.clip(el, -90.0, 90.0)

        sofa_dict["sofa_source_positions"] = pos
        sofa_dict["sofa_source_position_type"] = "spherical"
        sofa_dict["sofa_source_position_units"] = "degree"

    else:
        hf.log_with_timestamp(
            f"Unrecognised SOFA SourcePosition format "
            f"(type={pos_type}, units={pos_units})",
            gui_logger,
            log_type=1
        )

    return {
        "converted": converted,
        "original_type": original_type,
        "original_units": original_units,
    }

def sofa_workflow_new_dataset(hrtf_type, hrtf_dataset, hrtf_gui, hrtf_short, report_progress=0, gui_logger=None, spatial_res=2, direction_fix_gui=CN.HRTF_DIRECTION_FIX_LIST_GUI[0], apply_lf_suppression=False):
    """ 
    Function peforms the following workflow:
        try loading specified sofa object
        if not found, download from url and place into sofa directory
        load sofa object
        extract data and place into npy array (resample if required)
        df eq
        adjust levels
        save as npy dataset
    Returns:
       Status code: 0 = Success, 1 = Failure, 2 = Cancelled
    """

    total_samples_hrir=CN.TOTAL_SAMPLES_HRIR
    status=1
    metadata = None  # <--- initialize metadata
    try:
        hp_sos = None
        f_crossover_var = 100
        if apply_lf_suppression:
            hp_sos = hf.get_filter_sos(cutoff=f_crossover_var,fs=CN.FS,order=5,b_type='high')
            
        #try loading specified sofa object    
        if hrtf_type == 'Human Listener':
            hrir_dir_base = CN.DATA_DIR_SOFA
        elif hrtf_type == 'Dummy Head / Head & Torso Simulator':
            hrir_dir_base = CN.DATA_DIR_SOFA
        elif hrtf_type == 'User SOFA Input':
            hrir_dir_base = CN.DATA_DIR_SOFA_USER
        else:
            log_string = 'Invalid HRTF type: {hrtf_type}'
            hf.log_with_timestamp(log_string, gui_logger)
            return status, metadata
        #form sofa file name
        sofa_local_fname = pjoin(hrir_dir_base, f"{hrtf_short}.sofa")
            
        # Check if the file already exists
        if os.path.exists(sofa_local_fname):
            #attempt to load the sofa file
            
            loadsofa = hf.sofa_load_object(sofa_local_fname)#use custom function to load object, returns dict
            if not loadsofa:#empty dict returned
                log_string = 'Unable to load SOFA file. Likely due to unsupported convention version'
                hf.log_with_timestamp(log_string, gui_logger)
                return status, metadata
        else:
            log_string = 'local SOFA dataset not found. Attempting to download'
            hf.log_with_timestamp(log_string, gui_logger)

            if report_progress > 0:
                hf.update_gui_progress(report_progress=report_progress, message='Downloading required dataset')
     
            #exit if stop thread flag is true
            stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
            if stop_thread == True:
                status=2#2=cancelled
                return status, metadata
            
            #get URL
            url = get_sofa_url(listener_type=hrtf_type, dataset_name=hrtf_dataset, name_gui=hrtf_gui, gui_logger=gui_logger)
      
            #external sofa directory already formed above, download and save to that location
            response = hf.download_file(url=url, save_location=sofa_local_fname, gui_logger=gui_logger)
            
            if response == True:
                #attempt to load the sofa file
                loadsofa = hf.sofa_load_object(sofa_local_fname)#use custom function to load object, returns dict
                if not loadsofa:#empty dict returned
                    log_string = 'Unable to load SOFA file. Likely due to unsupported convention version'
                    hf.log_with_timestamp(log_string, gui_logger)
                    return status, metadata
            else:
                log_string = 'Request Failed. Attempting alternative link.'
                hf.log_with_timestamp(log_string, gui_logger)
                
                #try again with alternative link
                url = get_alternative_url(listener_type=hrtf_type, dataset_name=hrtf_dataset, name_gui=hrtf_gui, gui_logger=None)
                response = hf.download_file(url=url, save_location=sofa_local_fname, gui_logger=gui_logger)
                
                if response == True:
                    #attempt to load the sofa file
                    loadsofa = hf.sofa_load_object(sofa_local_fname)#use custom function to load object, returns dict
                    if not loadsofa:#empty dict returned
                        log_string = 'Unable to load SOFA file. Likely due to unsupported convention version'
                        hf.log_with_timestamp(log_string, gui_logger)
                        return status, metadata
                else:
                    log_string = 'Unable to download SOFA file. Request Failed.'
                    hf.log_with_timestamp(log_string, gui_logger)
                
                    return status, metadata
    
        #exit if stop thread flag is true
        stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
        if stop_thread == True:
            status=2#2=cancelled
            return status, metadata
        
        log_string = 'Processing SOFA dataset'
        hf.log_with_timestamp(log_string, gui_logger)
        if report_progress > 0:
            progress = 10/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)
         
        # View the convention parameters
        # Copy impulse response data
        convention_name = loadsofa['sofa_convention_name']
        sofa_data_ir = loadsofa['sofa_data_ir']
        sofa_samplerate = loadsofa['sofa_samplerate']
        # --- Enforce canonical SOFA angular convention ---
        norm_info = sofa_normalise_source_positions(loadsofa, gui_logger=gui_logger)
        if norm_info["converted"]:
            hf.log_with_timestamp(
                f"SOFA positions normalised "
                f"(type={norm_info['original_type']}, units={norm_info['original_units']})",
                gui_logger
            )
        # fetch (now guaranteed spherical degrees)
        sofa_source_positions = loadsofa['sofa_source_positions']
        
        #log some information about source positoins
        log_sofa_spatial_stats(sofa_source_positions, gui_logger)
        
        #get direction fix value
        direction_fix_value = hf.map_array_value_lookup(direction_fix_gui, CN.HRTF_DIRECTION_FIX_LIST_GUI, CN.HRTF_DIRECTION_FIX_LIST)
        #extract data and place into npy array conforming to ASH spatial structure (resample if required)
        hrir_out, nearest_dir_indices = sofa_dataset_transform(convention_name=convention_name, sofa_data_ir=sofa_data_ir,spatial_res=spatial_res, sofa_samplerate=sofa_samplerate, sofa_source_positions=sofa_source_positions,direction_fix_value=direction_fix_value, gui_logger=gui_logger)
        
        # --- Adjustment filter for sparse directions ---
        rounded_directions_input = generate_rounded_directions_from_indices(nearest_dir_indices=nearest_dir_indices, sofa_source_positions=sofa_source_positions, spatial_res=spatial_res)
        # Unique directions (for sparsity detection only)
        rounded_directions_unique = sorted(set(rounded_directions_input))
        #direction_counts = Counter(rounded_directions_input)
        #hf.log_with_timestamp(f"Direction usage histogram: {dict(direction_counts)}")
        
        # --- NEW: detect single-elevation datasets ---
        # Extract elevation column (index 1 in SOFA SourcePosition)
        elevations = sofa_source_positions[:, 1]
        # Use unique rounded elevations to avoid floating-point noise
        unique_elevations = np.unique(np.round(elevations, decimals=3))
        single_elevation_only = len(unique_elevations) == 1
        
        #exit if stop thread flag is true
        stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
        if stop_thread == True:
            status=2#2=cancelled
            return status, metadata
        
        ######## DF eq - calculate CTF, create inverse filter and apply to HRIRs
        #
        
        #impulse
        fr_flat = CN.FR_FLAT_DB_RFFT
        #coordinate system and dimensions
        if spatial_res >= 0 and spatial_res < CN.NUM_SPATIAL_RES:
            elev_min=CN.SPATIAL_RES_ELEV_MIN_IN[spatial_res] 
            elev_max=CN.SPATIAL_RES_ELEV_MAX_IN[spatial_res] 
            elev_nearest=CN.SPATIAL_RES_ELEV_NEAREST_IN[spatial_res] #as per hrir dataset
            azim_nearest=CN.SPATIAL_RES_AZIM_NEAREST_IN[spatial_res] 
        else:
            raise ValueError('Invalid spatial resolution')
        total_azim_hrir = int(360/azim_nearest)
        total_elev_hrir = int((elev_max-elev_min)/elev_nearest +1)
     
        log_string = 'Applying DF correction to HRIRs'
        hf.log_with_timestamp(log_string, gui_logger)
        
        #use multiple threads to calculate EQ
        results_list = []
        ts = []
        for elev in range(total_elev_hrir):
            list_indx=elev
            results_list.append(0)
            t = threading.Thread(target=calc_eq_for_hrirs, args = (hrir_out, elev, total_azim_hrir, fr_flat, results_list, list_indx))
            ts.append(t)
            t.start()
        for t in ts:
           t.join()
        #results_list will be populated with numpy arrays representing db response for each elevation
        num_results_avg = 0
        hrir_fft_avg_db = fr_flat.copy()
        for result in results_list:
            if np.sum(np.abs(result)) > 0:#some results might be flat 0 db
                num_results_avg = num_results_avg+1
                hrir_fft_avg_db = np.add(hrir_fft_avg_db,result)
        #divide by total number of elevations
        hrir_fft_avg_db = hrir_fft_avg_db/num_results_avg
        #convert to mag
        hrir_fft_avg_mag = hf.db2mag(hrir_fft_avg_db)

        # Only apply adjustment if spatially sparse input
        adjustment_filter = None
        if len(rounded_directions_unique) < CN.CTF_ADJUST_DIRS_THRESHOLD or single_elevation_only:#less than 16 directions is considered insufficient or if only one elevation
            hf.log_with_timestamp(f"Rounded source directions (unique): {rounded_directions_unique}")
            adjustment_filter = adjust_ctf_based_on_directions(rounded_directions_input, n_fft=CN.N_FFT, spatial_res=spatial_res)
            # Combine with DF response multiplicatively
            hrir_fft_avg_mag *= adjustment_filter
            
        #level ends of spectrum
        hrir_fft_avg_mag_lvl = hf.level_spectrum_ends(hrir_fft_avg_mag, 120, 18500, smooth_win = 7, n_fft=CN.N_FFT)#150, 18000
        #octave smoothing
        #hrir_fft_avg_mag = hf.smooth_freq_octaves(data=hrir_fft_avg_mag, n_fft=CN.N_FFT)
        #hrir_fft_avg_mag_lvl = hf.smooth_freq_octaves(data=hrir_fft_avg_mag_lvl, n_fft=CN.N_FFT)
        hrir_fft_avg_mag = hf.smooth_gaussian_octave(data=hrir_fft_avg_mag, n_fft=CN.N_FFT, fraction=6)
        hrir_fft_avg_mag_lvl = hf.smooth_gaussian_octave(data=hrir_fft_avg_mag_lvl, n_fft=CN.N_FFT, fraction=6)
        
        #invert response
        hrir_fft_avg_mag_inv = hf.db2mag(hf.mag2db(hrir_fft_avg_mag_lvl)*-1)
        #create min phase FIR
        hrir_df_inv_fir = hf.build_min_phase_filter(hrir_fft_avg_mag_inv, truncate_len=1024, n_fft=CN.N_FFT)
   
        if CN.PLOT_ENABLE == True:
            hf.plot_data(hrir_fft_avg_mag_lvl,'hrir_fft_avg_mag_lvl', normalise=0)  
        
        if report_progress > 0:
            progress = 15/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)
        
        #exit if stop thread flag is true
        stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
        if stop_thread == True:
            status=2#2=cancelled
            return status, metadata
        
        #perform EQ
        ts = []
        for elev in range(total_elev_hrir):
            t = threading.Thread(target=apply_eq_to_hrirs, args = (hrir_out, elev, total_samples_hrir, total_azim_hrir, hrir_df_inv_fir))
            ts.append(t)
            t.start()
        for t in ts:
           t.join()

        #exit if stop thread flag is true
        stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
        if stop_thread == True:
            status=2#2=cancelled
            return status, metadata

        #apply high pass filter for compatibility with acoustic spaces
        if apply_lf_suppression and hp_sos is not None:
            log_string = "Applying low-frequency conditioning"
            hf.log_with_timestamp(log_string, gui_logger)
            ts = []
            for elev in range(total_elev_hrir):
                t = threading.Thread(target=apply_hp_to_hrirs, args = (hrir_out, elev, total_samples_hrir, total_azim_hrir, hp_sos))
                ts.append(t)
                t.start()
            for t in ts:
               t.join()
  
        log_string = 'SOFA dataset processed'
        hf.log_with_timestamp(log_string, gui_logger)
        
        # Convert the data type to float32 and replace the original array, no longer need higher precision
        if hrir_out.dtype == np.float64:
            hrir_out = hrir_out.astype(np.float32)
            
        # #
        # #form output directory
        #use helper function to form file name
        npy_fname = get_hrir_file_path(hrtf_type=hrtf_type, hrtf_dataset=hrtf_dataset, hrtf_gui=hrtf_gui, hrtf_short=hrtf_short, spatial_res=spatial_res, gui_logger=gui_logger)
        
        # Derive directory from helper output
        output_file = Path(npy_fname)
        hrir_dir = str(output_file.parent)

        # Ensure output folder exists
        output_file = Path(npy_fname)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        # Save npy and log
        np.save(npy_fname, hrir_out)
        log_string = f"HRIR NPY dataset saved: {npy_fname}"
        hf.log_with_timestamp(log_string, gui_logger)  
        
        # --- NEW: Save CTF (Common Transfer Function) ---
        try:
            # 1) smoothed CTF (Common Transfer Filter)
            ctf_fname = pjoin(hrir_dir, f"{hrtf_short}_CTF.wav")
            hf.build_min_phase_and_save(mag_response=hrir_fft_avg_mag,file_path=ctf_fname,truncate_len=4096)
            log_string = f"CTF saved: {ctf_fname}"
            hf.log_with_timestamp(log_string, gui_logger)
        
            # 2) Smoothed + Level-ends CTF (Common Transfer Filter)
            ctf_le_fname = pjoin(hrir_dir, f"{hrtf_short}_CTF-LE.wav")
            hf.build_min_phase_and_save(mag_response=hrir_fft_avg_mag_lvl,file_path=ctf_le_fname,truncate_len=4096)
            log_string = f"CTF-LE saved: {ctf_le_fname}"
            hf.log_with_timestamp(log_string, gui_logger)
            
            # 3) Adjustment Filter (ONLY if created)
            ctf_adj_fname = None  # <-- define upfront
            if adjustment_filter is not None:
                ctf_adj_fname = pjoin(hrir_dir, f"{hrtf_short}_CTF-ADJ.wav")
                hf.build_min_phase_and_save(mag_response=adjustment_filter,file_path=ctf_adj_fname,truncate_len=1024)
                log_string = f"CTF-Adjustment saved: {ctf_adj_fname}"
                hf.log_with_timestamp(log_string, gui_logger)
        
        except Exception as ex_ctf:
            log_string = "Failed to generate/save CTF filters."
            hf.log_with_timestamp(log_string=log_string,gui_logger=gui_logger,log_type=2,exception=ex_ctf)
        
        # --- NEW: Save metadata JSON ---
        try:
            metadata = {
                "hrtf_type": hrtf_type,
                "hrtf_dataset": hrtf_dataset,
                "hrtf_gui": hrtf_gui,
                "hrtf_short": hrtf_short,
                "direction_fix_gui": direction_fix_gui,
                "spatial_res": spatial_res,
                "npy_file": npy_fname,
                "ctf_file": ctf_fname,
                "ctf_le_file": ctf_le_fname,
                "apply_lf_suppression": apply_lf_suppression,
                "f_crossover_var":f_crossover_var,
                # Adjustment filter (optional, metadata is truth)
                "has_adjustment_filter": adjustment_filter is not None,
                "ctf_adj_file": ctf_adj_fname
            }
            metadata_fname = pjoin(hrir_dir, f"{hrtf_short}_metadata.json")
            with open(metadata_fname, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            hf.log_with_timestamp(f"Metadata JSON saved: {metadata_fname}", gui_logger)
        except Exception as ex_meta:
            hf.log_with_timestamp("Failed to save metadata JSON.", gui_logger, log_type=2, exception=ex_meta)
            
        #finally return true since workflow was succesful
        status=0#success

    except Exception as ex:
        status=1#failed
        log_string = 'SOFA workflow failed'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
            
    return status, metadata


    


def calc_eq_for_hrirs(hrir_out, elev, total_azim_hrir, fr_flat, results_list, list_indx):
    """
    Calculate equalisation filter in parts for a single elevation using RFFT (vectorized over channels and azimuths).

    Args:
        hrir_out (np.ndarray): HRIR array, shape (elevs, azims, channels, samples)
        elev (int): Elevation index to process
        total_azim_hrir (int): Total number of azimuths
        fr_flat (np.ndarray): Reference flat spectrum in dB (length N_RFFT)
        results_list (list): List to store results
        list_indx (int): Index in results_list to store output
    """
    # Select all azimuths and channels for this elevation
    hrir_subset = hrir_out[elev, :total_azim_hrir, :, :]  # shape: (azims, channels, samples)

    # Compute RFFT along the samples axis
    hrir_fft = np.fft.rfft(hrir_subset, n=CN.N_FFT, axis=-1)  # shape: (azims, channels, N_RFFT)

    # Convert magnitude to dB and average over azimuths and channels
    hrir_fft_db = hf.mag2db(np.abs(hrir_fft))
    hrir_fft_avg_db = fr_flat + np.mean(hrir_fft_db, axis=(0, 1))  # avg over azim and chan

    results_list[list_indx] = hrir_fft_avg_db



def apply_eq_to_hrirs(hrir_out, elev, total_samples_hrir, total_azim_hrir, hrir_df_inv_fir):
    """
    Apply equalisation to HRIRs in numpy array for a single elevation.

    Args:
        hrir_out (np.ndarray): HRIR array, shape (elevs, azims, channels, samples)
        elev (int): Elevation index to process
        total_samples_hrir (int): Number of samples per HRIR
        total_azim_hrir (int): Total number of azimuths
        hrir_df_inv_fir (np.ndarray): EQ filter to apply
    """
    for azim in range(total_azim_hrir):
        for chan in range(CN.TOTAL_CHAN_BRIR):
            # Convolve HRIR with EQ filter
            hrir_eq = sp.signal.convolve(hrir_out[elev, azim, chan, :], hrir_df_inv_fir, mode='full')
            # Truncate or pad to original length
            hrir_out[elev, azim, chan, :] = hrir_eq[:total_samples_hrir]



  
def apply_hp_to_hrirs(hrir_out, elev, total_samples_hrir, total_azim_hrir, hp_sos):
    """
    Apply high-pass filtering to HRIRs in numpy array for a single elevation.

    Args:
        hrir_out (np.ndarray): HRIR array, shape (elevs, azims, channels, samples)
        elev (int): Elevation index to process
        total_samples_hrir (int): Number of samples per HRIR
        total_azim_hrir (int): Total number of azimuths
        hp_sos (np.ndarray): SOS filter coefficients for high-pass filter
    """
    for azim in range(total_azim_hrir):
        for chan in range(CN.TOTAL_CHAN_BRIR):
            # Apply high-pass filter
            hrir_filtered = hf.apply_sos_filter(hrir_out[elev, azim, chan, :], hp_sos, filtfilt=False)
            # Truncate or pad to original length
            hrir_out[elev, azim, chan, :] = hrir_filtered[:total_samples_hrir]



def sofa_dataset_transform(  
        convention_name,
        sofa_data_ir,
        sofa_samplerate,
        sofa_source_positions,
        gui_logger=None,
        spatial_res=2,
        direction_fix_value=None
):
    """
    Function performs the following transformations:
        extracts hrirs from sofa_data_ir 
        resamples if required depending on sofa_samplerate
        crops to 256 sample window
        places into an npy dataset using sofa_source_positions for nearest directions. 

    Returns:
        hrir_out (np.ndarray):
            Shape: (elevs, azims, channels, samples)

        nearest_dir_indices (list[int]):
            One entry per output HRIR cell, containing the SOFA measurement index
            actually used. Multiplicity preserved.
    """

    hrir_out = None
    nearest_dir_indices = []   # NEW

    try:
        cutoff = 10000
        fs = CN.FS
        order = 8
        lp_sos = hf.get_filter_sos(cutoff=cutoff, fs=fs, order=order, b_type='low')

        if convention_name in CN.SOFA_COMPAT_CONV:
            n_measurements, n_receivers, n_samples = sofa_data_ir.shape[:3]
        else:
            raise ValueError('Invalid SOFA convention.')

        # Spatial resolution setup
        if 0 <= spatial_res < CN.NUM_SPATIAL_RES:
            elev_min = CN.SPATIAL_RES_ELEV_MIN_IN[spatial_res]
            elev_max = CN.SPATIAL_RES_ELEV_MAX_IN[spatial_res]
            elev_step = CN.SPATIAL_RES_ELEV_NEAREST_IN[spatial_res]
            azim_step = CN.SPATIAL_RES_AZIM_NEAREST_IN[spatial_res]
        else:
            raise ValueError('Invalid spatial resolution')

        output_azims = max(1, int(360 / azim_step))
        output_elevs = max(1, int((elev_max - elev_min) / elev_step + 1))

        total_chan_hrir = min(CN.TOTAL_CHAN_HRIR, n_receivers)
        total_samples_hrir = CN.TOTAL_SAMPLES_HRIR

        hrir_out = np.zeros((output_elevs, output_azims,
                             total_chan_hrir, total_samples_hrir))

        populate_samples = min(total_samples_hrir, n_samples)
        samp_freq_ash = CN.SAMP_FREQ

        if sofa_samplerate != samp_freq_ash:
            populate_samples = round(populate_samples * float(samp_freq_ash) / sofa_samplerate)
            hf.log_with_timestamp('Resampling dataset', gui_logger)

        # ---------------------------------------
        # Main loop
        # ---------------------------------------
        for e in range(output_elevs):
            elev_deg = int(elev_min + e * elev_step)

            for a in range(output_azims):
                azim_deg = int(a * azim_step)

                nearest_dir_idx = sofa_find_nearest_direction(
                    sofa_source_positions=sofa_source_positions,
                    target_elevation=elev_deg,
                    target_azimuth=azim_deg,
                    direction_fix_value=direction_fix_value
                )

                nearest_dir_idx = max(0, min(nearest_dir_idx, n_measurements - 1))

                # NEW: record exact usage
                nearest_dir_indices.append(nearest_dir_idx)

                hrir_selected = np.zeros((total_chan_hrir, n_samples))

                for chan in range(total_chan_hrir):
                    if chan < n_receivers:
                        query_sofa_data(convention_name,
                                        hrir_selected,
                                        sofa_data_ir,
                                        nearest_dir_idx,
                                        chan)

                if sofa_samplerate != samp_freq_ash:
                    hrir_selected = hf.resample_signal(
                        hrir_selected,
                        original_rate=sofa_samplerate,
                        new_rate=samp_freq_ash,
                        axis=1,
                        scale=True
                    )

                hrir_selected = shift_2d_impulse_response(
                    hrir_selected, lp_sos, target_index=58
                )

                copy_n = min(populate_samples,
                             hrir_selected.shape[1],
                             total_samples_hrir)

                for chan in range(total_chan_hrir):
                    hrir_out[e, a, chan, 0:copy_n] = hrir_selected[chan, 0:copy_n]

        if sofa_samplerate != samp_freq_ash:
            hf.log_with_timestamp(
                f'Resampled {sofa_samplerate} to {samp_freq_ash}',
                gui_logger
            )

    except Exception as ex:
        hf.log_with_timestamp('SOFA transform failed',
                              gui_logger=gui_logger,
                              log_type=2,
                              exception=ex)

    return hrir_out, nearest_dir_indices   # NEW


def query_sofa_data(convention_name,hrir_selected,sofa_data_ir,nearest_dir_idx,chan):
    """
    Function to copy data from sofa dataset
    """
    
    if convention_name == 'GeneralFIRE':
        try: 
            hrir_selected[chan,:] = np.copy(sofa_data_ir[nearest_dir_idx,chan,0,:])#GeneralFIRE case, not tested. mREn
        except:
            hrir_selected[chan,:] = np.copy(sofa_data_ir[nearest_dir_idx,chan,:])
        
    elif convention_name == 'GeneralFIR-E':
        try: 
            hrir_selected[chan,:] = np.copy(sofa_data_ir[nearest_dir_idx,chan,:,0])#GeneralFIR-E case, not tested.  mrne
        except:
            hrir_selected[chan,:] = np.copy(sofa_data_ir[nearest_dir_idx,chan,:])
    else:
        hrir_selected[chan,:] = np.copy(sofa_data_ir[nearest_dir_idx,chan,:])





def sofa_find_nearest_direction(
        sofa_source_positions,
        target_elevation,
        target_azimuth,
        direction_fix_value=None,
        gui_logger=None,
        elevation_limits=(-90, 90)):
    """
    Returns index of closest SOFA direction.
    Optional misalignment compensation via direction_fix_value.

    Supported direction_fix_value options:

    String-based:
        - "flip_azimuth"         : reverse azimuth (CW ↔ CCW)
        - "front_back"           : rotate 180° front ↔ back
        - "left_start"           : 0° reference starts at left (-90° grid rotation, 90° head rotation)
        - "right_start"          : 0° reference starts at right (+90° grid rotation, -90° head rotation)
        - "invert_elevation"     : flip elevation sign
        - "back_cw_start"        : 0° at back, rotating clockwise

    Tuple-based:
        - ("offset", degrees)                : azimuth offset, 
        - ("elevation_offset", degrees)      : elevation offset
        - ("offsets", az_deg, el_deg)        : azimuth + elevation offset
    """

    # --------------------------
    # Apply correction BEFORE searching
    # --------------------------
    try:
        if direction_fix_value is not None:

            # ------------------
            # String-based fixes
            # ------------------
            if isinstance(direction_fix_value, str):

                if direction_fix_value == "flip_azimuth":
                    target_azimuth = (-target_azimuth) % 360# Mirror azimuth direction: clockwise ↔ counter-clockwise

                elif direction_fix_value == "front_back":
                    target_azimuth = (target_azimuth + 180) % 360# Rotate azimuth by 180° to swap front ↔ back # add 180 deg to rotate azimuth grid from back to front

                elif direction_fix_value == "left_start":# Dataset uses 0° at left; shift azimuth so 0° aligns with front
                    target_azimuth = (target_azimuth + 90) % 360  # Add 90° azimuth offset. # Intuition: imagine rotating the listener's head 90° CCW so that a dataset with 0° at left is reinterpreted as having 0° at the front.

                elif direction_fix_value == "right_start":# Dataset uses 0° at right; shift azimuth so 0° aligns with front
                    target_azimuth = (target_azimuth - 90) % 360  #add -90 deg offset # Intuition: imagine rotating the listener's head 90° CW so that a dataset with 0° at right is reinterpreted as having 0° at the front.

                elif direction_fix_value == "invert_elevation":
                    target_elevation = -target_elevation#down becomes up and up becomes down

                elif direction_fix_value == "back_cw_start":# Dataset uses 0° at back and clockwise azimuth; convert to front-facing CCW convention
                    target_azimuth = (target_azimuth + 180) % 360# Rotate azimuth by 180° to swap front ↔ back
                    target_azimuth = (-target_azimuth) % 360# Mirror azimuth direction: clockwise ↔ counter-clockwise

            # ------------------
            # Tuple-based fixes
            # ------------------
            elif isinstance(direction_fix_value, tuple):

                key = direction_fix_value[0]

                if key == "offset":
                    # Azimuth offset only (existing behaviour)
                    az_offset = float(direction_fix_value[1])
                    target_azimuth = (target_azimuth + az_offset) % 360 
                    #If az_offset is positive: shifting source CW  If az_offset is negative: shifting source CCW 

                elif key == "elevation_offset":
                    el_offset = float(direction_fix_value[1])
                    target_elevation += el_offset
                    #If el_offset is positive: shifting source up  If el_offset is negative: shifting source down 

                elif key == "offsets":
                    az_offset = float(direction_fix_value[1])
                    el_offset = float(direction_fix_value[2])
                    target_azimuth = (target_azimuth + az_offset) % 360
                    target_elevation += el_offset

            # ------------------
            # Clamp elevation (safety)
            # ------------------
            if elevation_limits is not None:
                el_min, el_max = elevation_limits
                target_elevation = max(el_min, min(el_max, target_elevation))

            hf.log_with_timestamp(
                f"Direction correction applied: {direction_fix_value} "
                f"(az={target_azimuth:.1f}, el={target_elevation:.1f})",
                gui_logger=gui_logger
            )

    except Exception as ex:
        hf.log_with_timestamp(
            "Direction correction failed",
            gui_logger=gui_logger,
            log_type=2,
            exception=ex
        )

    # --------------------------
    # Nearest lookup
    # --------------------------
    nearest_idx = 0
    nearest_distance = 1e9

    try:
        for i in range(sofa_source_positions.shape[0]):
            az = float(sofa_source_positions[i, 0])
            el = float(sofa_source_positions[i, 1])

            dist = sqrt(
                (el - target_elevation) ** 2 +
                (az - target_azimuth) ** 2
            )

            if dist < nearest_distance:
                nearest_distance = dist
                nearest_idx = i

    except Exception as ex:
        hf.log_with_timestamp(
            "Direction search failed",
            gui_logger=gui_logger,
            log_type=2,
            exception=ex
        )

    return nearest_idx



def shift_2d_impulse_response(arr, lp_sos, target_index=50):
    """
    Shifts a 2D NumPy array so that the earliest peak across all rows in the second 
    dimension (largest absolute value) aligns at the specified target index.

    Args:
        arr (np.ndarray): A 2D NumPy array of shape (N, M), where M is the length of each impulse response.
        target_index (int): The index where the earliest peak should be aligned.

    Returns:
        np.ndarray: A 2D NumPy array with all rows shifted by the same amount.
    """
    
    #apply low pass before shifting
    arr_lp = hf.apply_sos_filter(arr, lp_sos)

    N, M = arr.shape  # Get shape of array

    # Ensure target index does not exceed the valid range
    target_index = min(target_index, M - 1)

    # Find peak indices across all rows (global peak detection)
    #peak_indices = np.argmax(np.abs(arr_lp), axis=1)  # Find peak per row
    peak_indices = np.argmax(arr_lp, axis=1)  # check only positive values
    
    earliest_peak_index = np.min(peak_indices)  # Find the earliest peak

    # Compute global shift amount
    shift_amount = target_index - earliest_peak_index

    # Apply the same shift to all rows
    shifted_arr = np.roll(arr, shift_amount, axis=1)

    # Zero out wrapped-around values
    if shift_amount > 0:
        shifted_arr[:, :shift_amount] = 0  # Zero out leading elements
    elif shift_amount < 0:
        shifted_arr[:, shift_amount:] = 0  # Zero out trailing elements

    return shifted_arr


def shift_2d_impulse_response_cc(arr, hrir_sample, lp_sos, max_shift_samples=100):
    """
    Shifts a 2D impulse response array to align with a reference using cross-correlation between 
    the most energetic channels in both the target and reference.

    Args:
        arr (np.ndarray): 2D array of shape (channels, samples).
        hrir_sample (np.ndarray): Reference 2D array of shape (channels, samples).
        max_shift_samples (int): Max samples to shift (positive or negative).

    Returns:
        np.ndarray: Shifted 2D array aligned with the reference.
    """


    # Apply low-pass filter
    #apply low pass before shifting
    arr_lp = hf.apply_sos_filter(arr, lp_sos)
    ref_lp = hf.apply_sos_filter(hrir_sample, lp_sos)

    # Find channel with highest peak (abs) in both arrays
    best_channel_arr = np.argmax(np.max(np.abs(arr_lp), axis=1))
    best_channel_ref = np.argmax(np.max(np.abs(ref_lp), axis=1))

    # Cross-correlation between these two peak channels
    xcorr = np.correlate(arr_lp[best_channel_arr], ref_lp[best_channel_ref], mode='full')
    lag = np.argmax(xcorr) - (len(arr_lp[best_channel_arr]) - 1)

    # Clip lag
    lag = int(np.clip(lag, -max_shift_samples, max_shift_samples))

    # Apply shift
    shifted_arr = np.roll(arr, -lag, axis=1)

    # Zero out wrapped values
    if lag > 0:
        shifted_arr[:, -lag:] = 0
    elif lag < 0:
        shifted_arr[:, :-lag] = 0

    return shifted_arr











def circular_mean(phases):
    """
    Compute circular mean of phase (in radians) across listeners.
    phases: array [..., n_listeners]
    """
    return np.arctan2(
        np.mean(np.sin(phases), axis=0),
        np.mean(np.cos(phases), axis=0)
    )



# def build_averaged_listener_from_sets(hrir_sets, gui_logger=None,
#                                       interp_mode='modular',
#                                       align_directions=True,
#                                       align_listeners=False,
#                                       sample_rate=CN.SAMP_FREQ,
#                                       n_jobs=-1):
#     """
#     Build an averaged HRIR dataset by interpolating across listeners in the
#     frequency domain. Supports:
#         - complex interpolation (real/imag)  [recommended]
#         - modular phase interpolation (circular mean)

#     Parameters
#     ----------
#     hrir_sets : list[np.ndarray]
#         List of HRIR datasets shaped [elev, azim, ch, samples].
#     interp_mode : {'complex', 'modular'}
#         Phase interpolation mode.
#     align_directions : bool
#         Per-direction ITD alignment within listener.
#     align_listeners : bool
#         Global cross-listener alignment.
#     """

#     try:
#         hf.log_with_timestamp("Validating input HRIR datasets", gui_logger)

#         # ------------ Input Validation ------------
#         if not isinstance(hrir_sets, list) or len(hrir_sets) == 0:
#             raise ValueError("hrir_sets must be a non-empty list.")

#         shapes = [h.shape for h in hrir_sets]
#         if len(set(shapes)) != 1:
#             raise ValueError(f"Inconsistent HRIR shapes: {shapes}")

#         n_list = len(hrir_sets)
#         elev_n, azim_n, ch_n, N = hrir_sets[0].shape

#         # ------------ Filters ------------
#         lp_sos = hf.get_filter_sos(
#             cutoff=10000, fs=sample_rate, order=8, b_type='low'
#         )
        
#         # ------------ Stage 0: Low-Frequency Polarity Correction ------------
#         hf.log_with_timestamp("Checking for polarity inversions (Low-Freq focus)...", gui_logger)
#         # 1. Initialize a 1kHz low-pass filter using your helper
#         # We use a lower cutoff (1000Hz) to ignore chaotic high-frequency phase
#         lp_sos_polarity = hf.get_filter_sos(
#             cutoff=1000, fs=sample_rate, order=4, b_type='low'
#         )
#         # 2. Get and filter the reference listener
#         ref_idx = 0
#         ref_sig_raw = hrir_sets[ref_idx][elev_n // 2, 0, 0, :]
#         ref_sig_lp = hf.apply_sos_filter(ref_sig_raw, lp_sos_polarity)
#         # 3. Compare all other listeners to the reference
#         for i in range(1, n_list):
#             target_sig_raw = hrir_sets[i][elev_n // 2, 0, 0, :]
#             target_sig_lp = hf.apply_sos_filter(target_sig_raw, lp_sos_polarity)
#             # Dot product check: if negative, the low-frequency pulses are 180° out of phase
#             if np.dot(ref_sig_lp, target_sig_lp) < 0:
#                 hf.log_with_timestamp(f"Inverting polarity for listener {i} (detected at <1kHz)", gui_logger)
#                 hrir_sets[i] *= -1.0


#         # ------------ Stage 1: Per-listener direction alignment ------------
#         if align_directions:
#             hf.log_with_timestamp("Performing intra-listener alignment...", gui_logger)
#             for i in range(n_list):
#                 for el in range(elev_n):
#                     for az in range(azim_n):
#                         hrir_sets[i][el, az, :2, :] = shift_2d_impulse_response(
#                             hrir_sets[i][el, az, :2, :], lp_sos, target_index=55
#                         )

     
#         # ------------ Stage 2: Sub-sample Global alignment ------------
#         if align_listeners:
#             hf.log_with_timestamp("Performing sub-sample inter-listener alignment...", gui_logger)
            
#             # Use a reference (usually the first listener)
#             ref_hrir = hrir_sets[0][elev_n // 2, azim_n // 2, 0, :]
#             ref_fft = np.fft.rfft(ref_hrir)
        
#             for i in range(1, n_list):
#                 target_hrir = hrir_sets[i][elev_n // 2, azim_n // 2, 0, :]
#                 target_fft = np.fft.rfft(target_hrir)
                
#                 # Cross-correlation in frequency domain to find sub-sample shift
#                 # This calculates the phase difference between signals
#                 cross_pow = ref_fft * np.conj(target_fft)
#                 # The phase of the cross power spectrum is the delay
#                 # We find the slope of that phase via the peak of the IR
#                 correlation = np.fft.irfft(cross_pow)
                
#                 # Parabolic interpolation to find the fractional peak
#                 # (Standard trick for sub-sample accuracy)
#                 peak_idx = np.argmax(correlation)
#                 # ... [Insert parabolic fit logic here or use a simpler method] ...
                
#                 # Simple high-accuracy alternative: 
#                 # Shift the entire dataset in the Frequency Domain
#                 # shift = delay in samples (can be 1.45, 2.12, etc.)
#                 delay = peak_idx if peak_idx < len(correlation)/2 else peak_idx - len(correlation)
                
#                 # Apply the shift to ALL directions/channels for this listener
#                 freq_bins = np.fft.rfftfreq(N)
#                 shift_vector = np.exp(1j * 2 * np.pi * freq_bins * delay)
                
#                 # Shift in frequency domain
#                 for el in range(elev_n):
#                     for az in range(azim_n):
#                         for ch in range(ch_n):
#                             spec = np.fft.rfft(hrir_sets[i][el, az, ch, :])
#                             hrir_sets[i][el, az, ch, :] = np.fft.irfft(spec * shift_vector, n=N)
                            
        
                            
    

#         # ------------ Stage 3: Spectral leveling per listener ------------
#         hf.log_with_timestamp("Applying per-listener spectral leveling...", gui_logger)
#         mag_a, mag_b, n_fft = CN.SPECT_SNAP_M_F0, CN.SPECT_SNAP_M_F1, CN.N_FFT

#         subset_idx = [(el, az)
#                       for el in np.linspace(0, elev_n - 1, 3, dtype=int)
#                       for az in np.linspace(0, azim_n - 1, 5, dtype=int)]

#         for i in range(n_list):
#             mag_sum = 0.0
#             for el, az in subset_idx:
#                 padded = hf.padarray(hrir_sets[i][el, az, 0, :], n_fft)
#                 mag_sum += np.mean(
#                     np.abs(np.fft.fft(padded)[mag_a:mag_b])
#                 )
#             hrir_sets[i] /= (mag_sum / len(subset_idx))

#         # ------------ Stage 4: Convert to frequency domain ------------
#         hf.log_with_timestamp("Converting HRIRs to frequency domain...", gui_logger)
#         hrtf_sets = np.array(
#             [np.fft.rfft(hrir, axis=-1) for hrir in hrir_sets],
#             dtype=np.complex128
#         )   # shape: [n_list, elev, azim, ch, freq]

#         # ------------ Stage 5: Interpolation ------------
#         hf.log_with_timestamp(f"Interpolating across listeners (mode={interp_mode})...",
#                               gui_logger)

#         hrtf_avg = np.empty((elev_n, azim_n, ch_n, hrtf_sets.shape[-1]),
#                             dtype=np.complex128)

#         for el in range(elev_n):
#             for az in range(azim_n):
#                 for ch in range(ch_n):

#                     H = hrtf_sets[:, el, az, ch]   # shape: [n_list, n_freqs]

#                     if interp_mode == "modular":#most reliable
#                         # --- Magnitude (in dB) + Circular phase averaging ---

#                         # Magnitude & phase
#                         mag = np.abs(H)
#                         phase = np.angle(H)
#                         # Convert magnitude to dB (amplitude dB)
#                         mag_db = 20 * np.log10(np.clip(mag, 1e-12, None))
#                         # Average magnitudes IN dB
#                         mag_db_interp = np.mean(mag_db, axis=0)
#                         # Convert back to linear magnitude
#                         mag_interp = 10 ** (mag_db_interp / 20)
#                         # Circular phase averaging (preserves wrapped phase)
#                         phase_interp = circular_mean(phase)
#                         # Reconstruct complex HRTF
#                         hrtf_avg[el, az, ch] = mag_interp * np.exp(1j * phase_interp)
                        
        
                      
                 
#                     else:
#                         raise ValueError(f"Invalid interp_mode: {interp_mode}")

#         # ------------ Stage 6: Return to time domain ------------
#         hf.log_with_timestamp("Converting averaged HRTF back to time domain...", gui_logger)
#         hrir_avg = np.fft.irfft(hrtf_avg, n=N, axis=-1)

#         # ------------ Stage 7: Final dataset-level normalization ------------
#         hf.log_with_timestamp("Applying final dataset-level normalization...", gui_logger)
#         mag_sum = 0.0
#         for el, az in subset_idx:
#             padded = hf.padarray(hrir_avg[el, az, 0, :], n_fft)
#             mag_sum += np.mean(
#                 np.abs(np.fft.fft(padded)[mag_a:mag_b])
#             )
#         hrir_avg /= (mag_sum / len(subset_idx))

#         # ------------ Stage 8: Ensure final shape is [1, elev, azim, ch, N] ------------
#         if hrir_avg.ndim == 4:
#             hrir_avg = np.expand_dims(hrir_avg, axis=0)
#         elif hrir_avg.ndim != 5:
#             raise ValueError(f"Unexpected HRIR shape: {hrir_avg.shape}")

#         # ------------ Stage 9: Save output ------------
#         npy_fname = pjoin(CN.DATA_DIR_HRIR_NPY_INTRP,
#                           CN.HRTF_AVERAGED_NAME_FILE + ".npy")
#         Path(npy_fname).parent.mkdir(exist_ok=True, parents=True)

#         if hrir_avg.dtype == np.float64:
#             hrir_avg = hrir_avg.astype(np.float32)

#         np.save(npy_fname, hrir_avg)
#         hf.log_with_timestamp(f"Saved npy dataset: {npy_fname}", gui_logger)
#         hf.log_with_timestamp("HRIR averaging complete.", gui_logger)

#         return hrir_avg

#     except Exception as e:
#         hf.log_with_timestamp(f"Error building averaged listener: {e}", gui_logger)
#         return None


def build_averaged_listener_from_sets(hrir_sets, gui_logger=None,
                                      interp_mode='modular',
                                      align_directions=True,
                                      align_listeners=False,
                                      sample_rate=CN.SAMP_FREQ,
                                      n_jobs=-1):
    """
    Build an averaged HRIR dataset by interpolating across listeners in the
    frequency domain. Supports:
        - complex interpolation (real/imag)  [recommended]
        - modular phase interpolation (circular mean)

    Parameters
    ----------
    hrir_sets : list[np.ndarray]
        List of HRIR datasets shaped [elev, azim, ch, samples].
    interp_mode : {'complex', 'modular'}
        Phase interpolation mode.
    align_directions : bool
        Per-direction ITD alignment within listener.
    align_listeners : bool
        Global cross-listener alignment.
    """

    try:
        hf.log_with_timestamp("Validating input HRIR datasets", gui_logger)

        # ------------ Input Validation ------------
        if not isinstance(hrir_sets, list) or len(hrir_sets) == 0:
            raise ValueError("hrir_sets must be a non-empty list.")

        shapes = [h.shape for h in hrir_sets]
        if len(set(shapes)) != 1:
            raise ValueError(f"Inconsistent HRIR shapes: {shapes}")

        n_list = len(hrir_sets)
        elev_n, azim_n, ch_n, N = hrir_sets[0].shape

        # ------------ Filters ------------
        lp_sos = hf.get_filter_sos(
            cutoff=10000, fs=sample_rate, order=8, b_type='low'
        )
        
        # ------------ Stage 0: Low-Frequency Polarity Correction ------------
        hf.log_with_timestamp("Checking for polarity inversions (Low-Freq focus)...", gui_logger)
        # 1. Initialize a 1kHz low-pass filter using your helper
        # We use a lower cutoff (1000Hz) to ignore chaotic high-frequency phase
        lp_sos_polarity = hf.get_filter_sos(
            cutoff=1000, fs=sample_rate, order=4, b_type='low'
        )
        # 2. Get and filter the reference listener
        ref_idx = 0
        ref_sig_raw = hrir_sets[ref_idx][elev_n // 2, 0, 0, :]
        ref_sig_lp = hf.apply_sos_filter(ref_sig_raw, lp_sos_polarity)
        # 3. Compare all other listeners to the reference
        for i in range(1, n_list):
            target_sig_raw = hrir_sets[i][elev_n // 2, 0, 0, :]
            target_sig_lp = hf.apply_sos_filter(target_sig_raw, lp_sos_polarity)
            # Dot product check: if negative, the low-frequency pulses are 180° out of phase
            if np.dot(ref_sig_lp, target_sig_lp) < 0:
                hf.log_with_timestamp(f"Inverting polarity for listener {i} (detected at <1kHz)", gui_logger)
                hrir_sets[i] *= -1.0


        # ------------ Stage 1: Per-listener direction alignment ------------
        if align_directions:
            hf.log_with_timestamp("Performing intra-listener alignment...", gui_logger)
            for i in range(n_list):
                for el in range(elev_n):
                    for az in range(azim_n):
                        hrir_sets[i][el, az, :2, :] = shift_2d_impulse_response(
                            hrir_sets[i][el, az, :2, :], lp_sos, target_index=55
                        )

     
        # ------------ Stage 2: Sub-sample Global alignment ------------
        if align_listeners:
            hf.log_with_timestamp("Performing sub-sample inter-listener alignment...", gui_logger)
            
            # Use a reference (usually the first listener)
            ref_hrir = hrir_sets[0][elev_n // 2, azim_n // 2, 0, :]
            ref_fft = np.fft.rfft(ref_hrir)
        
            for i in range(1, n_list):
                target_hrir = hrir_sets[i][elev_n // 2, azim_n // 2, 0, :]
                target_fft = np.fft.rfft(target_hrir)
                
                # Cross-correlation in frequency domain to find sub-sample shift
                # This calculates the phase difference between signals
                cross_pow = ref_fft * np.conj(target_fft)
                # The phase of the cross power spectrum is the delay
                # We find the slope of that phase via the peak of the IR
                correlation = np.fft.irfft(cross_pow)
                
                # Parabolic interpolation to find the fractional peak
                # (Standard trick for sub-sample accuracy)
                peak_idx = np.argmax(correlation)
                # ... [Insert parabolic fit logic here or use a simpler method] ...
                
                # Simple high-accuracy alternative: 
                # Shift the entire dataset in the Frequency Domain
                # shift = delay in samples (can be 1.45, 2.12, etc.)
                delay = peak_idx if peak_idx < len(correlation)/2 else peak_idx - len(correlation)
                
                # Apply the shift to ALL directions/channels for this listener
                freq_bins = np.fft.rfftfreq(N)
                shift_vector = np.exp(1j * 2 * np.pi * freq_bins * delay)
                
                # Shift in frequency domain
                for el in range(elev_n):
                    for az in range(azim_n):
                        for ch in range(ch_n):
                            spec = np.fft.rfft(hrir_sets[i][el, az, ch, :])
                            hrir_sets[i][el, az, ch, :] = np.fft.irfft(spec * shift_vector, n=N)
                            
        
                            
    

        # ------------ Stage 3: Spectral leveling per listener ------------
        hf.log_with_timestamp("Applying per-listener spectral leveling...", gui_logger)
        mag_a, mag_b, n_fft = CN.SPECT_SNAP_M_F0, CN.SPECT_SNAP_M_F1, CN.N_FFT

        subset_idx = [(el, az)
                      for el in np.linspace(0, elev_n - 1, 3, dtype=int)
                      for az in np.linspace(0, azim_n - 1, 5, dtype=int)]

        for i in range(n_list):
            mag_sum = 0.0
            for el, az in subset_idx:
                padded = hf.padarray(hrir_sets[i][el, az, 0, :], n_fft)
                mag_sum += np.mean(
                    np.abs(np.fft.fft(padded)[mag_a:mag_b])
                )
            hrir_sets[i] /= (mag_sum / len(subset_idx))


        # ------------ Stage 5: Interpolation (time-domain output) ------------
        hf.log_with_timestamp(
            f"Interpolating across listeners (mode={interp_mode})...", gui_logger
        )
        
        pad_factor = 4
        n_fft_pad = N * pad_factor
        n_freqs_pad = n_fft_pad // 2 + 1
        
        hrir_avg = np.empty((elev_n, azim_n, ch_n, N), dtype=np.float64)
        
        for el in range(elev_n):
            for az in range(azim_n):
                for ch in range(ch_n):
        
                    # --- FFT per listener (zero-padded) ---
                    H = np.empty((n_list, n_freqs_pad), dtype=np.complex128)
        
                    for i in range(n_list):
                        hrir = hrir_sets[i][el, az, ch]  # length = N
                        hrir_padded = np.pad(
                            hrir,
                            (0, n_fft_pad - N),
                            mode="constant"
                        )
                        H[i] = np.fft.rfft(hrir_padded, n=n_fft_pad)
        
                    if interp_mode == "modular":
                        mag = np.abs(H)
                        phase = np.angle(H)
        
                        mag_db = 20 * np.log10(np.clip(mag, 1e-12, None))
                        mag_db_interp = np.mean(mag_db, axis=0)
                        mag_interp = 10 ** (mag_db_interp / 20)
        
                        phase_interp = circular_mean(phase)
        
                        # --- Interpolated padded spectrum ---
                        H_interp = mag_interp * np.exp(1j * phase_interp)
        
                        # --- Back to time domain (padded) ---
                        hrir_interp_pad = np.fft.irfft(H_interp, n=n_fft_pad)
        
                        # --- Time-domain crop (FINAL HRIR) ---
                        hrir_avg[el, az, ch, :] = hrir_interp_pad[:N]
        
                    else:
                        raise ValueError(f"Invalid interp_mode: {interp_mode}")
                        
     
        # ------------ Stage 7: Final dataset-level normalization ------------
        hf.log_with_timestamp("Applying final dataset-level normalization...", gui_logger)
        mag_sum = 0.0
        for el, az in subset_idx:
            padded = hf.padarray(hrir_avg[el, az, 0, :], n_fft)
            mag_sum += np.mean(
                np.abs(np.fft.fft(padded)[mag_a:mag_b])
            )
        hrir_avg /= (mag_sum / len(subset_idx))

        # ------------ Stage 8: Ensure final shape is [1, elev, azim, ch, N] ------------
        if hrir_avg.ndim == 4:
            hrir_avg = np.expand_dims(hrir_avg, axis=0)
        elif hrir_avg.ndim != 5:
            raise ValueError(f"Unexpected HRIR shape: {hrir_avg.shape}")

        # ------------ Stage 9: Save output ------------
        npy_fname = pjoin(CN.DATA_DIR_HRIR_NPY_INTRP,
                          CN.HRTF_AVERAGED_NAME_FILE + ".npy")
        Path(npy_fname).parent.mkdir(exist_ok=True, parents=True)

        if hrir_avg.dtype == np.float64:
            hrir_avg = hrir_avg.astype(np.float32)

        np.save(npy_fname, hrir_avg)
        hf.log_with_timestamp(f"Saved npy dataset: {npy_fname}", gui_logger)
        hf.log_with_timestamp("HRIR averaging complete.", gui_logger)

        return hrir_avg

    except Exception as e:
        hf.log_with_timestamp(f"Error building averaged listener: {e}", gui_logger)
        return None





def get_hrir_file_path(hrtf_type, hrtf_dataset, hrtf_gui, hrtf_short,
                       spatial_res=2, gui_logger=None, report_progress=None):
    """
    Resolve and return the full file path for a given HRIR dataset
    based on metadata such as type, GUI selection, and spatial resolution.

    Parameters
    ----------
    hrtf_type : str
        Type of HRIR (e.g., 'Human Listener', 'Dummy Head / Head & Torso Simulator', 'Favourites', 'User SOFA Input').
    hrtf_gui : str
        GUI-selected HRTF name.
    hrtf_dataset : str
        Dataset folder name.
    hrtf_short : str
        Short name of the HRTF file (without extension).
    spatial_res : int
        Spatial resolution index (0–3).
    gui_logger : optional
        Logger for GUI.
    report_progress : optional
        Progress callback for downloads (used in high-res sets).

    Returns
    -------
    str
        Full path to the .npy HRIR dataset file.

    Raises
    ------
    ValueError
        If invalid type combinations or dataset paths occur.
    """

    try:
        # --- 1. Determine Base Directory ---
        if hrtf_type == 'Human Listener':
            hrir_dir_base = CN.DATA_DIR_HRIR_NPY_HL

        elif hrtf_type == 'Dummy Head / Head & Torso Simulator':
            hrir_dir_base = CN.DATA_DIR_HRIR_NPY_DH

        elif hrtf_type == 'Favourites':
            if hrtf_gui == CN.HRTF_AVERAGED_NAME_GUI:
                hrir_dir_base = CN.DATA_DIR_HRIR_NPY_INTRP
            else:#hrtf metadata should have been translated before this point, should refer to a different hrtf_type
                raise ValueError(f"HRIR processing failed: invalid listener: {hrtf_gui} for 'Favourites' type")

        elif hrtf_type == 'User SOFA Input':
            # User-loaded or custom HRTF set
            hrir_dir_base = CN.DATA_DIR_HRIR_NPY_USER
        else:
            raise ValueError("Invalid HRTF type")

        # --- 2. Validate spatial resolution ---
        if not (0 <= spatial_res < CN.NUM_SPATIAL_RES):
            raise ValueError(f"Invalid spatial resolution index: {spatial_res}")

        # --- 3. Subdirectory by spatial resolution ---
        if spatial_res <= 2:
            sub_directory = 'h'
        else:
            sub_directory = 'm'


        # --- 4. Construct directory path ---
        hrir_dir = hrir_dir_base

        if hrtf_type != 'Favourites':#different folder structure for averaged listener
            hrir_dir = pjoin(hrir_dir, sub_directory)
            hrir_dir = pjoin(hrir_dir, hrtf_dataset)

        # --- 5. Final filename ---
        npy_fname = pjoin(hrir_dir, f"{hrtf_short}.npy")

        # --- 6. Logging and return ---
       # hf.log_with_timestamp(f"Resolved HRIR path: {npy_fname}", gui_logger)
        return npy_fname

    except Exception as ex:
        hf.log_with_timestamp("Failed to resolve HRIR file path.", gui_logger, log_type=2, exception=ex)
        raise
        

def hrtf_param_cleaning(hrtf_type, hrtf_dataset, hrtf_gui, hrtf_short):
    """
    Cleans and resolves BRIR HRTF parameter values, including handling favourites and user SOFA inputs.
    Returns the possibly modified (hrtf_type, hrtf_dataset, hrtf_gui, hrtf_short).
    """
    # --- Case 1: Averaged favourite ---
    if hrtf_type == 'Favourites' and hrtf_gui == CN.HRTF_AVERAGED_NAME_GUI:
        hrtf_short = CN.HRTF_AVERAGED_NAME_FILE

    # --- Case 2: User SOFA under favourite ---
    elif hrtf_type == 'Favourites' and hrtf_gui.startswith(CN.HRTF_USER_SOFA_PREFIX):
        hrtf_type = 'User SOFA Input'
        hrtf_dataset = CN.HRTF_DATASET_USER_DEFAULT
        hrtf_gui = hrtf_gui.removeprefix(CN.HRTF_USER_SOFA_PREFIX)
        hrtf_short = hrtf_gui
        
    # --- Case 3: no Favourites found ---    
    elif hrtf_type == 'Favourites' and hrtf_gui == CN.HRTF_BASE_LIST_FAV[0]:
        hrtf_type = hrtf_type
        hrtf_dataset = hrtf_dataset
        hrtf_gui = hrtf_gui
        hrtf_short = hrtf_gui

    # --- Case 4: Standard favourite (lookup) ---
    elif hrtf_type == 'Favourites':
        hrtf_type, dataset, name_gui = get_hrtf_info_from_name_short(name_short=hrtf_gui)#gui name is same as name short for favourites, do a lookup to get remaining metadata
        hrtf_type = hrtf_type
        hrtf_dataset = dataset
        hrtf_short = hrtf_gui
        hrtf_gui = name_gui
        
    #else, no modification to metadata required

    return hrtf_type, hrtf_dataset, hrtf_gui, hrtf_short




    

    
   
    
    
def load_hrirs_list(hrtf_dict_list,spatial_res=2, direction_fix_gui=CN.HRTF_DIRECTION_FIX_LIST_GUI[0], gui_logger=None, metadata_only=False, force_skip_sofa=False, apply_lf_suppression=False):
    """
    Load a list of HRIR datasets using fully-specified HRTF metadata dictionaries.

    Each dict must contain:
        hrtf_type, hrtf_dataset, hrtf_gui, hrtf_short

    Parameters:
        hrtf_dict_list (list[dict]): List of HRTF metadata dictionaries
        spatial_res (int): Desired spatial resolution for HRIRs
        direction_fix_gui (str): Selected direction-fix option from GUI
        gui_logger (callable|None): Optional logger for GUI messages
        metadata_only (bool): If True, only read metadata JSON without loading HRIRs

    Returns:
        metadata_only=False:
            (list of HRIR arrays, status_code, metadata_list)
        metadata_only=True:
            ([], status_code, metadata_list)
    """
    try:
        # Early exit if no metadata is provided
        if not hrtf_dict_list:
            hf.log_with_timestamp("No HRIR metadata provided.", gui_logger)
            return [], 1, []

        hrir_list_loaded = []  # Will store loaded HRIR arrays (empty if metadata_only)
        metadata_list = []     # Will store metadata for each HRTF

        # Iterate through each HRTF dictionary
        for meta in hrtf_dict_list:
            # Extract required fields
            hrtf_type = meta.get("hrtf_type")
            hrtf_dataset = meta.get("hrtf_dataset")
            hrtf_gui = meta.get("hrtf_gui")
            hrtf_short = meta.get("hrtf_short")
  
            # Ensure all keys are present
            if None in (hrtf_type, hrtf_dataset, hrtf_gui, hrtf_short):
                raise ValueError(f"Invalid dictionary entry (missing keys): {meta}")

            # Normalize / clean HRTF parameters
            hrtf_type, hrtf_dataset, hrtf_gui, hrtf_short = hrtf_param_cleaning(
                hrtf_type, hrtf_dataset, hrtf_gui, hrtf_short
            )
  
            hf.log_with_timestamp(
                f"Reading HRTF metadata: {hrtf_gui}",
                gui_logger,
            )

            # Resolve expected NPY file path for the HRIR
            npy_fname = get_hrir_file_path(hrtf_type=hrtf_type,hrtf_gui=hrtf_gui,hrtf_dataset=hrtf_dataset,hrtf_short=hrtf_short,spatial_res=spatial_res,gui_logger=gui_logger)

            # Resolve metadata JSON path
            metadata_fname = Path(npy_fname).with_name(f"{hrtf_short}_metadata.json")
            current_metadata = {}

            # Attempt to read existing metadata JSON
            if metadata_fname.exists():
                try:
                    with open(metadata_fname, "r", encoding="utf-8") as f:
                        current_metadata = json.load(f)
                except Exception as e:
                    hf.log_with_timestamp(
                        f"Failed to read metadata JSON for {hrtf_gui}: {e}",
                        gui_logger,
                    )
            else:
                hf.log_with_timestamp(
                    f"Metadata JSON not found for {hrtf_gui}.",
                    gui_logger,
                )


            # --------- METADATA-ONLY SHORT-CIRCUIT ---------
            if metadata_only:
                # Skip all loading, NPY creation, or SOFA workflows
                continue

            # --------- NORMAL LOADING PATH ---------
            build_npy_hrir_dataset = True

            # Check for Google Drive source or averaged HRTF
            gdrive_url = get_gdrive_url_high(listener_type=hrtf_type,dataset_name=hrtf_dataset,name_gui=hrtf_gui,gui_logger=gui_logger)

            if gdrive_url or hrtf_gui == CN.HRTF_AVERAGED_NAME_GUI:
                # If precompiled NPY already exists or HRTF is averaged, skip SOFA creation
                build_npy_hrir_dataset = False
            elif Path(npy_fname).exists() and current_metadata:
                # Check if direction fix has changed or CTF files are missing
                direction_fix_applied = current_metadata.get("direction_fix_gui")
                lf_suppression_applied = current_metadata.get("apply_lf_suppression")
                ctf_file = current_metadata.get("ctf_file")
                ctf_le_file = current_metadata.get("ctf_le_file")
                

                if (
                    direction_fix_applied == direction_fix_gui
                    and lf_suppression_applied == apply_lf_suppression
                    and ctf_file
                    and ctf_le_file
                    and Path(ctf_file).exists()
                    and Path(ctf_le_file).exists()
                ):
                    build_npy_hrir_dataset = False

     
            
            # If needed, recreate HRIR from SOFA
            if build_npy_hrir_dataset:
                if force_skip_sofa:
                    continue
                hf.log_with_timestamp(
                    f"Creating HRIR dataset from SOFA for {hrtf_gui}.",
                    gui_logger,
                )
                status, new_metadata = sofa_workflow_new_dataset(hrtf_type=hrtf_type, hrtf_dataset=hrtf_dataset, hrtf_gui=hrtf_gui, hrtf_short=hrtf_short, 
                                                   direction_fix_gui=direction_fix_gui, gui_logger=gui_logger, apply_lf_suppression=apply_lf_suppression)
                if status == 2:
                    # User cancelled workflow
                    return [], 2, []
                elif status != 0:
                    raise ValueError(f"HRIR processing failed for {hrtf_gui}")
                # Update metadata after SOFA workflow
                current_metadata = new_metadata or {}
                    
            # Append metadata to list
            metadata_list.append(current_metadata)

            # Load NPY HRIR array
            hrir_list = hf.load_convert_npy_to_float64(npy_fname)

            # Handle optional singleton first dimension
            if hrir_list.ndim == 5 and hrir_list.shape[0] == 1:
                hrir_list = np.squeeze(hrir_list, axis=0)

            hrir_list_loaded.append(hrir_list)

            # Log HRIR shape
            elev, azim, ch, samples = hrir_list.shape
            hf.log_with_timestamp(
                f"Loaded {hrtf_gui}: Elev={elev}, Azim={azim}, Ch={ch}, Samples={samples}",
                gui_logger,
            )

        # Return results: HRIR arrays (if loaded), status, metadata list
        return hrir_list_loaded, 0, metadata_list

    except Exception as e:
        hf.log_with_timestamp(f"Error loading HRIRs: {e}", gui_logger)
        return [], 1, []    