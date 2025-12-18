# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 16:20:38 2025

@author: Shanon
"""

import numpy as np
from os.path import join as pjoin
from ash_toolset import helper_functions as hf
from ash_toolset import constants as CN
from ash_toolset import brir_export
from csv import DictReader
import os
import gdown
from math import sqrt
from pathlib import Path
import threading
import scipy as sp
import dearpygui.dearpygui as dpg
import logging
from scipy.signal import correlate
from scipy.interpolate import Rbf
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

logger = logging.getLogger(__name__)
log_info=1

def get_listener_list(listener_type="", dataset_name="", max_res_only=False, gui_logger=None):
    """
    Retrieves a list of listeners based on the listener type and dataset name.
    
    Args:
        listener_type (str, optional): The type of listener ('Dummy Head / Head & Torso Simulator' or 'Human Listener'). Defaults to "".
        dataset_name (str, optional): The name of the dataset to filter by (only used for 'Human Listener'). Defaults to "".
        gui_logger (object, optional): A logger object for GUI-related logging. Defaults to None.
    
    Returns:
        list: A list of listener names.
    """
    
    listener_list = []
    
    try:
    
        if listener_type == 'Dummy Head / Head & Torso Simulator' or listener_type == 'Human Listener':
            
            #HRTF related - individual npy datasets
            listener_list_filtered = []
            #load lists from csv file
            try:
            
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
                        if max_res_only == True:
                            if row.get('dataset') == dataset_name and row.get('hrtf_type') ==listener_type and row.get('hrtf_index_max') != '': #added check
                                listener_list_filtered.append(row.get('name_gui'))
                        else:
                            if row.get('dataset') == dataset_name and row.get('hrtf_type') ==listener_type: #added check
                                listener_list_filtered.append(row.get('name_gui'))
                # Sort the list alphabetically
                listener_list_filtered.sort()
            except Exception as e:
                log_string=f"Error loading listener list: {e}"
                hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=e)#log error
                pass
            
            listener_list=listener_list_filtered
            
        elif listener_type == 'Favourites':
            if dpg.does_item_exist("fde_hrtf_add_favourite"):
                try:
                    listener_list = dpg.get_item_user_data("fde_hrtf_add_favourite")
                    if not listener_list:
                        listener_list = CN.HRTF_BASE_LIST_FAV
                except Exception:
                    listener_list = CN.HRTF_BASE_LIST_FAV
            else:
                listener_list = CN.HRTF_BASE_LIST_FAV
            # Sort the list alphabetically
            listener_list.sort()
            
        else:
            listener_list=get_listener_list_user()
            if not listener_list:
                listener_list = [CN.HRTF_USER_SOFA_DEFAULT]
            
    except Exception as ex:
        log_string=f"Error occurred: {ex}"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
        
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
    Retrieves a short name based on the listener type and dataset name and gui name.
    
    Args:
        listener_type (str, optional): The type of listener ('Dummy Head / Head & Torso Simulator' or 'Human Listener'). Defaults to "".
        dataset_name (str, optional): The name of the dataset to filter by (only used for 'Human Listener'). Defaults to "".
        gui_logger (object, optional): A logger object for GUI-related logging. Defaults to None.
    
    Returns:
        list: A list of listener names.
    """
    
    name_short = ""
    
    try:
    
 
        if listener_type == 'Dummy Head / Head & Torso Simulator' or listener_type == 'Human Listener':
            
            #HRTF related - individual npy datasets
            #load lists from csv file
            try:
            
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
                        if row.get('dataset') == dataset_name and row.get('hrtf_type') ==listener_type and row.get('name_gui') ==name_gui: #added check
                            name_short = row.get('name_short')
    
            except Exception as e:
                log_string=f"Error loading listener list: {e}"
                hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=e)#log error
                pass
   
        else:
            name_short = name_gui #case for user SOFA or favourite, use same as GUI name (file name)
            
    except Exception as ex:
        log_string=f"Error occurred: {ex}"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
   
        
    return name_short

def get_hrtf_info_from_name_short(name_short: str = "", gui_logger=None) -> tuple[str, str, str]:
    """
    Retrieves HRTF type, dataset, and GUI name for a given name_short.

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
            try:
                # directories
                csv_directory = CN.DATA_DIR_HRIR_NPY
                metadata_file_name = CN.HRIR_METADATA_NAME
                metadata_file = pjoin(csv_directory, metadata_file_name)

                with open(metadata_file, encoding="utf-8-sig", newline="") as inputfile:
                    reader = DictReader(inputfile)
                    for row in reader:
                        if row.get("name_short") == name_short:
                            hrtf_type = row.get("hrtf_type", "")
                            dataset = row.get("dataset", "")
                            name_gui = row.get("name_gui", "")
                            break  # name_short is unique, so stop here

            except Exception as e:
                log_string = f"Error loading HRTF info for name_short='{name_short}': {e}"
                hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type=2, exception=e)

    except Exception as ex:
        log_string = f"Error occurred: {ex}"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type=2, exception=ex)

    return hrtf_type, dataset, name_gui


def get_sofa_url(listener_type="", dataset_name="", name_gui="", gui_logger=None):
    """
    Retrieves a sofa conventions database url to a sofa file based on the listener type and dataset name and gui name.
    
    Args:
        listener_type (str, optional): The type of listener ('Dummy Head / Head & Torso Simulator' or 'Human Listener'). Defaults to "".
        dataset_name (str, optional): The name of the dataset to filter by (only used for 'Human Listener'). Defaults to "".
        gui_logger (object, optional): A logger object for GUI-related logging. Defaults to None.
    
    Returns:
        list: A list of listener names.
    """
    
    url = ""
    
    try:
    
 
        if listener_type == 'Dummy Head / Head & Torso Simulator' or listener_type == 'Human Listener':
            
            #HRTF related - individual npy datasets
            #load lists from csv file
            try:
            
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
                        if row.get('dataset') == dataset_name and row.get('hrtf_type') ==listener_type and row.get('name_gui') ==name_gui: #added check
                            url = row.get('url')
    
            except Exception as e:
                log_string=f"Error loading listener list: {e}"
                hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=e)#log error
                pass
   
        else:
            url = ""
            
    except Exception as ex:
        log_string=f"Error occurred: {ex}"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
        
    return url

def get_alternative_url(listener_type="", dataset_name="", name_gui="", gui_logger=None):
    """
    Retrieves a sofa conventions database url to a sofa file based on the listener type and dataset name and gui name.
    
    Args:
        listener_type (str, optional): The type of listener ('Dummy Head / Head & Torso Simulator' or 'Human Listener'). Defaults to "".
        dataset_name (str, optional): The name of the dataset to filter by (only used for 'Human Listener'). Defaults to "".
        gui_logger (object, optional): A logger object for GUI-related logging. Defaults to None.
    
    Returns:
        list: A list of listener names.
    """
    
    url = ""
    
    try:
    
        
        if listener_type == 'Dummy Head / Head & Torso Simulator' or listener_type == 'Human Listener':
            
            #HRTF related - individual npy datasets
            #load lists from csv file
            try:
            
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
                        if row.get('dataset') == dataset_name and row.get('hrtf_type') ==listener_type and row.get('name_gui') ==name_gui: #added check
                            url = row.get('url_alternative')
    
            except Exception as e:
                log_string=f"Error loading listener list: {e}"
                hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=e)#log error
                pass
   
        else:
            url = ""
            
    except Exception as ex:
        log_string=f"Error occurred: {ex}"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
        
    return url



def get_polarity(listener_type="", dataset_name="", name_gui="", gui_logger=None):
    """
    Retrieves a polarity string based on the listener type and dataset name and gui name.
    
    Args:
        listener_type (str, optional): The type of listener ('Dummy Head / Head & Torso Simulator' or 'Human Listener'). Defaults to "".
        dataset_name (str, optional): The name of the dataset to filter by (only used for 'Human Listener'). Defaults to "".
        gui_logger (object, optional): A logger object for GUI-related logging. Defaults to None.
    
    Returns:
        list: A list of listener names.
    """
    
    flip_polarity = "no"
    
    try:
  
        if listener_type == 'Dummy Head / Head & Torso Simulator' or listener_type == 'Human Listener':
            
            #HRTF related - individual npy datasets
            #load lists from csv file
            try:
            
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
                        if row.get('dataset') == dataset_name and row.get('hrtf_type') ==listener_type and row.get('name_gui') ==name_gui: #added check
                            flip_polarity = row.get('flip_polarity')
    
            except Exception as e:
                log_string=f"Error loading listener list: {e}"
                hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=e)#log error
                pass
   
        else:
            flip_polarity = ""
            
    except Exception as ex:
        log_string=f"Error occurred: {ex}"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
        
    return flip_polarity


def get_gdrive_url_max(listener_type="", dataset_name="", name_gui="", gui_logger=None):
    """
    Retrieves a max res npy url based on the listener type and dataset name and gui name.
    
    Args:
        listener_type (str, optional): The type of listener ('Dummy Head / Head & Torso Simulator' or 'Human Listener'). Defaults to "".
        dataset_name (str, optional): The name of the dataset to filter by (only used for 'Human Listener'). Defaults to "".
        gui_logger (object, optional): A logger object for GUI-related logging. Defaults to None.
    
    Returns:
        list: A list of listener names.
    """
    
    url = ""
    
    try:
    
        if listener_type == 'Dummy Head / Head & Torso Simulator' or listener_type == 'Human Listener':
            
            #HRTF related - individual npy datasets
            #load lists from csv file
            try:
            
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
                        if row.get('dataset') == dataset_name and row.get('hrtf_type') ==listener_type and row.get('name_gui') ==name_gui: #added check
                            url = row.get('gdrive_url_max')
    
            except Exception as e:
                log_string=f"Error loading listener list: {e}"
                hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=e)#log error
                pass
   
        else:
            url = ""
            
    except Exception as ex:
        log_string=f"Error occurred: {ex}"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
        
    return url

def get_gdrive_url_high(listener_type="", dataset_name="", name_gui="", gui_logger=None):
    """
    Retrieves a high res npy url based on the listener type and dataset name and gui name.
    
    Args:
        listener_type (str, optional): The type of listener ('Dummy Head / Head & Torso Simulator' or 'Human Listener'). Defaults to "".
        dataset_name (str, optional): The name of the dataset to filter by (only used for 'Human Listener'). Defaults to "".
        gui_logger (object, optional): A logger object for GUI-related logging. Defaults to None.
    
    Returns:
        list: A list of listener names.
    """
    
    url = ""
    
    try:
    
   
        if listener_type == 'Dummy Head / Head & Torso Simulator' or listener_type == 'Human Listener':
            
            #HRTF related - individual npy datasets
            #load lists from csv file
            try:
            
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
                        if row.get('dataset') == dataset_name and row.get('hrtf_type') ==listener_type and row.get('name_gui') ==name_gui: #added check
                            url = row.get('gdrive_url_high')
    
            except Exception as e:
                log_string=f"Error loading listener list: {e}"
                hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=e)#log error
                pass
   
        else:
            url = ""
            
    except Exception as ex:
        log_string=f"Error occurred: {ex}"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
        
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
                        hf.log_with_timestamp(log_string, gui_logger)

                            
            #this hrtf not found in local metadata, must be new space
            if match_found==0:
                mismatches=mismatches+1 
                update_required=1
                log_string = 'New hrtf available: ' + name_w
                hf.log_with_timestamp(log_string, gui_logger)
                    
         
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

def sofa_workflow_new_dataset(hrtf_type, hrtf_dataset, hrtf_gui, hrtf_short, report_progress=0, gui_logger=None, spatial_res=2, direction_fix_gui=CN.HRTF_DIRECTION_FIX_LIST_GUI[0]):
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
    
    samp_freq=CN.SAMP_FREQ
    total_samples_hrir=CN.TOTAL_SAMPLES_HRIR
    status=1
    
    try:
        f_crossover_var=120
        hp_sos = hf.get_filter_sos(cutoff=f_crossover_var, fs=CN.FS, order=CN.ORDER, b_type='high')
            
        #try loading specified sofa object    
        if hrtf_type == 'Human Listener':
            hrir_dir_base = CN.DATA_DIR_SOFA
        elif hrtf_type == 'Dummy Head / Head & Torso Simulator':
            hrir_dir_base = CN.DATA_DIR_SOFA
        elif hrtf_type == 'Favourites':
            log_string = 'Invalid HRTF type: Favourites'
            hf.log_with_timestamp(log_string, gui_logger)
            return status
        elif hrtf_type == 'User SOFA Input':
            hrir_dir_base = CN.DATA_DIR_SOFA_USER
        else:
            log_string = 'Invalid HRTF type'
            hf.log_with_timestamp(log_string, gui_logger)
            return status
        #form sofa file name
        sofa_local_fname = pjoin(hrir_dir_base, f"{hrtf_short}.sofa")
            
        # Check if the file already exists
        if os.path.exists(sofa_local_fname):
            #attempt to load the sofa file
            
            loadsofa = hf.sofa_load_object(sofa_local_fname)#use custom function to load object, returns dict
            if not loadsofa:#empty dict returned
                log_string = 'Unable to load SOFA file. Likely due to unsupported convention version'
                hf.log_with_timestamp(log_string, gui_logger)
                return status
        else:
            log_string = 'local SOFA dataset not found. Attempting to download'
            hf.log_with_timestamp(log_string, gui_logger)

            if report_progress > 0:
                hf.update_gui_progress(report_progress=report_progress, message='Downloading required dataset')
     
            #exit if stop thread flag is true
            stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
            if stop_thread == True:
                status=2#2=cancelled
                return status
            
            #get URL
            url = get_sofa_url(listener_type=hrtf_type, dataset_name=hrtf_dataset, name_gui=hrtf_gui, gui_logger=gui_logger)
            
            #print(url)
            
            #external sofa directory already formed above, download and save to that location
            response = hf.download_file(url=url, save_location=sofa_local_fname, gui_logger=gui_logger)
            
            if response == True:
                #attempt to load the sofa file
                loadsofa = hf.sofa_load_object(sofa_local_fname)#use custom function to load object, returns dict
                if not loadsofa:#empty dict returned
                    log_string = 'Unable to load SOFA file. Likely due to unsupported convention version'
                    hf.log_with_timestamp(log_string, gui_logger)
                    return status
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
                        return status
                else:
                    log_string = 'Unable to download SOFA file. Request Failed.'
                    hf.log_with_timestamp(log_string, gui_logger)
                
                    return status
    
        #exit if stop thread flag is true
        stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
        if stop_thread == True:
            status=2#2=cancelled
            return status
        
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
        sofa_source_positions = loadsofa['sofa_source_positions']    
        
        #get direction fix value
        direction_fix_value = hf.map_array_value_lookup(direction_fix_gui, CN.HRTF_DIRECTION_FIX_LIST_GUI, CN.HRTF_DIRECTION_FIX_LIST)
        #extract data and place into npy array conforming to ASH spatial structure (resample if required)
        hrir_out = sofa_dataset_transform(convention_name=convention_name, sofa_data_ir=sofa_data_ir,spatial_res=spatial_res, sofa_samplerate=sofa_samplerate, sofa_source_positions=sofa_source_positions,direction_fix_value=direction_fix_value, gui_logger=gui_logger)
        
        #exit if stop thread flag is true
        stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
        if stop_thread == True:
            status=2#2=cancelled
            return status
        
        #
        #DF eq
        #
        
        #impulse
        impulse=np.zeros(CN.N_FFT)
        impulse[0]=1
        fr_flat_mag = np.abs(np.fft.fft(impulse))
        fr_flat = hf.mag2db(fr_flat_mag)
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
        #determine spatial resolution parameters
        direction_matrix_process = brir_export.generate_direction_matrix(spatial_res=spatial_res, variant=0)
        
        log_string = 'Applying DF correction to HRIRs'
        hf.log_with_timestamp(log_string)
        
        #use multiple threads to calculate EQ
        results_list = []
        ts = []
        for elev in range(total_elev_hrir):
            list_indx=elev
            results_list.append(0)
            t = threading.Thread(target=calc_eq_for_hrirs, args = (hrir_out, elev, total_azim_hrir, direction_matrix_process, fr_flat, results_list, list_indx))
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
        #level ends of spectrum
        hrir_fft_avg_mag_lvl = hf.level_spectrum_ends(hrir_fft_avg_mag, 120, 18500, smooth_win = 7, n_fft=CN.N_FFT)#150, 18000
        #octave smoothing
        hrir_fft_avg_mag_sm = hf.smooth_freq_octaves(data=hrir_fft_avg_mag_lvl, n_fft=CN.N_FFT)
        #invert response
        hrir_fft_avg_mag_inv = hf.db2mag(hf.mag2db(hrir_fft_avg_mag_sm)*-1)
        #create min phase FIR
        hrir_df_inv_fir = hf.build_min_phase_filter(hrir_fft_avg_mag_inv, truncate_len=1024, n_fft=CN.N_FFT)
   
        if CN.PLOT_ENABLE == True:
            hf.plot_data(hrir_fft_avg_mag_sm,'hrir_fft_avg_mag_sm', normalise=0)  
        
        if report_progress > 0:
            progress = 15/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)
        
        #exit if stop thread flag is true
        stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
        if stop_thread == True:
            status=2#2=cancelled
            return status
        
        #perform EQ
        ts = []
        for elev in range(total_elev_hrir):
            t = threading.Thread(target=apply_eq_to_hrirs, args = (hrir_out, elev, total_samples_hrir, total_azim_hrir, direction_matrix_process, hrir_df_inv_fir))
            ts.append(t)
            t.start()
        for t in ts:
           t.join()

        #exit if stop thread flag is true
        stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
        if stop_thread == True:
            status=2#2=cancelled
            return status

        #apply high pass filter for compatibility with acoustic spaces
        ts = []
        for elev in range(total_elev_hrir):
            t = threading.Thread(target=apply_hp_to_hrirs, args = (hrir_out, elev, total_samples_hrir, total_azim_hrir, direction_matrix_process, hp_sos))
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
        
        # --- NEW: Save CTF (Common Transfer Function) - TWO VERSIONS ---
        try:
            # 1) Un-smoothed CTF (Common Transfer Filter)
            ctf_fname = pjoin(hrir_dir, f"{hrtf_short}_CTF.wav")
            hf.build_min_phase_and_save(
                mag_response=hrir_fft_avg_mag,
                file_path=ctf_fname
            )
            log_string = f"CTF saved: {ctf_fname}"
            hf.log_with_timestamp(log_string, gui_logger)
        
            # 2) Smoothed / Level-EQ CTF (Common Transfer Filter)
            ctf_le_fname = pjoin(hrir_dir, f"{hrtf_short}_CTF-LE.wav")
            hf.build_min_phase_and_save(
                mag_response=hrir_fft_avg_mag_sm,
                file_path=ctf_le_fname
            )
            log_string = f"CTF-LE saved: {ctf_le_fname}"
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
                "ctf_le_file": ctf_le_fname
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
            
    return status


def calc_eq_for_hrirs(hrir_out, elev, total_azim_hrir, direction_matrix_process, fr_flat, results_list, list_indx):
    """
    Function calculates equalisation filter in parts
    :param hrir_out: numpy array containing hrirs
    :return: None
    """ 
    set_id=0
    num_brirs_avg = 0
    hrir_fft_avg_db = fr_flat.copy()
    
    #get diffuse field spectrum
    for azim in range(total_azim_hrir):
        for chan in range(CN.TOTAL_CHAN_BRIR):
            #only apply if direction is in direction matrix
            if direction_matrix_process[elev][azim][0][0] == 1: 
                hrir_current = np.copy(hrir_out[set_id,elev,azim,chan,:])
                # print(f"Dimensions of the dataset: {hrir_current.shape}")
                # print(f"Number of dimensions: {hrir_current.ndim}")  # Added for clarity
                # print(f"Size of the array: {hrir_current.size}") # Added for total number of elements
                #zero pad
                hrir_current = hf.zero_pad_1d(hrir_current, CN.N_FFT)
                hrir_current_fft = np.fft.fft(hrir_current)
                hrir_current_mag_fft=np.abs(hrir_current_fft)
                hrir_current_db_fft = hf.mag2db(hrir_current_mag_fft)
                
                hrir_fft_avg_db = np.add(hrir_fft_avg_db,hrir_current_db_fft)
                
                num_brirs_avg = num_brirs_avg+1
    
    #divide by total number of brirs
    if num_brirs_avg > 0:
        hrir_fft_avg_db = hrir_fft_avg_db/num_brirs_avg
    
    results_list[list_indx]=hrir_fft_avg_db            
  
   
  
def apply_eq_to_hrirs(hrir_out, elev, total_samples_hrir, total_azim_hrir, direction_matrix_process, hrir_df_inv_fir):
    """
    Function applies equalisation to hrirs in numpy array
    :param hrir_out: numpy array containing hrirs
    :return: None
    """ 
    set_id=0
    for azim in range(total_azim_hrir):
        for chan in range(CN.TOTAL_CHAN_BRIR):  
            
            #only apply equalisation if direction is in direction matrix
            if direction_matrix_process[elev][azim][0][0] == 1: 
            
                #convolve BRIR with filters
                hrir_eq_b = np.copy(hrir_out[set_id,elev,azim,chan,:])#
                #apply DF eq
                hrir_eq_b = sp.signal.convolve(hrir_eq_b,hrir_df_inv_fir, 'full', 'auto')
                hrir_out[set_id,elev,azim,chan,:] = np.copy(hrir_eq_b[0:total_samples_hrir])
            
            else:
                hrir_out[set_id,elev,azim,chan,:] = np.zeros(total_samples_hrir)#zero out directions that wont be exported    
  
  
def apply_hp_to_hrirs(hrir_out, elev, total_samples_hrir, total_azim_hrir, direction_matrix_process, hp_sos):
    """
    Function applies high pass filtering to hrirs in numpy array
    :param hrir_out: numpy array containing hrirs
    :return: None
    """ 
    set_id=0
    f_crossover_var=120
    for azim in range(total_azim_hrir):
        for chan in range(CN.TOTAL_CHAN_BRIR):  
            
            #only apply equalisation if direction is in direction matrix
            if direction_matrix_process[elev][azim][0][0] == 1: 
            
                #convolve BRIR with filters
                hrir_eq_b = np.copy(hrir_out[set_id,elev,azim,chan,:])#
                #apply HP eq
                #hrir_eq_b = hf.signal_highpass_filter(hrir_eq_b, f_crossover_var, CN.FS, CN.ORDER)
                hrir_eq_b = hf.apply_sos_filter(hrir_eq_b, hp_sos)
                
                hrir_out[set_id,elev,azim,chan,:] = np.copy(hrir_eq_b[0:total_samples_hrir])
            
            else:
                hrir_out[set_id,elev,azim,chan,:] = np.zeros(total_samples_hrir)#zero out directions that wont be exported    
  
  
    
  
  
  



def sofa_dataset_transform(
        convention_name,
        sofa_data_ir,
        sofa_samplerate,
        sofa_source_positions,
        gui_logger=None,
        spatial_res=2,
        direction_fix_value=None      # <── NEW PARAM
):
    """ 
    Function peforms the following transformations:
        extracts hrirs from sofa_data_ir 
        resamples if required depending on sofa_samplerate
        crops to 256 sample window
        places into an npy dataset using sofa_source_positions for nearest directions. 
        Resulting NPY array conforms to ASH spatial format: 
        shape [listener][elev][azim][chan][samples], elev increasing with height (min<=elev<=max, midpoint=level), azim CCW (0<=azim<360, 0=front), chan L&R, variable sample length, variable elev limits, variable elev and azim intervals depending on spatial resolution.    
        
        Args:
        convention_name (str): sofa convention
        sofa_data_ir (npy array): data IR array from sofa file, usually 3 dimensions, measurements x receivers x samples
        sofa_samplerate (int):  sample rate
        sofa_source_positions(npy array):source positions array from sofa file, 3 spherical dimensions, azimuth (deg) x elevation (deg) x distance, 2d array: position x coordinate tuple
        reverse_azim_rot (bool): flag to reverse azimuth direction, positive = CCW by default
    """

    hrir_out = None

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

        hrir_out = np.zeros((1, output_elevs, output_azims,
                             total_chan_hrir, total_samples_hrir))

        populate_samples = min(total_samples_hrir, n_samples)
        samp_freq_ash = CN.SAMP_FREQ

        if sofa_samplerate != samp_freq_ash:
            populate_samples = round(populate_samples *
                                     float(samp_freq_ash) / sofa_samplerate)
            hf.log_with_timestamp('Resampling dataset', gui_logger)

        # ---------------------------------------
        # Main loop
        # ---------------------------------------
        for e in range(output_elevs):
            elev_deg = int(elev_min + e * elev_step)

            for a in range(output_azims):
                azim_deg = int(a * azim_step)

                # Pass through the new misalignment-correction logic
                nearest_dir_idx = sofa_find_nearest_direction(
                    sofa_source_positions=sofa_source_positions,
                    target_elevation=elev_deg,
                    target_azimuth=azim_deg,
                    direction_fix_value=direction_fix_value   # <── NEW
                )

                nearest_dir_idx = max(0, min(nearest_dir_idx, n_measurements - 1))

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
                    hrir_selected, lp_sos, target_index=58)

                copy_n = min(populate_samples,
                             hrir_selected.shape[1],
                             total_samples_hrir)

                for chan in range(total_chan_hrir):
                    hrir_out[0, e, a, chan, 0:copy_n] = \
                        np.copy(hrir_selected[chan, 0:copy_n])

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

    return hrir_out




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
        gui_logger=None):
    """
    Returns index of closest SOFA direction.
    Optional misalignment compensation via direction_fix.

    Supported direction_fix options:
        - "flip_azimuth"         : reverse azimuth (CW ↔ CCW)
        - "front_back"           : rotate 180° front ↔ back
        - "left_start"           : 0° reference starts at left (-90° shift)
        - "right_start"          : 0° reference starts at right (+90° shift)
        - "swap_left_right"      : mirror horizontally
        - "rotate_cw_90"         : rotate azimuth -90°
        - "rotate_ccw_90"        : rotate azimuth +90°
        - "invert_elevation"     : flip elevation sign
        - ("offset", degrees)    : arbitrary azimuth shift
        - "back_cw_start"       : dataset's 0° is behind listener and rotates clockwise
    """

    # --------------------------
    # Apply correction BEFORE searching
    # --------------------------
    try:
        if direction_fix_value is not None:

            # Apply simple string-based corrections
            if isinstance(direction_fix_value, str):

                if direction_fix_value == "flip_azimuth":
                    target_azimuth = (-target_azimuth) % 360

                elif direction_fix_value == "front_back":
                    target_azimuth = (target_azimuth + 180) % 360

                elif direction_fix_value == "left_start":
                    # Dataset's 0° is at left → shift target by +90
                    target_azimuth = (target_azimuth + 90) % 360

                elif direction_fix_value == "right_start":
                    # Dataset's 0° is at right → shift target by -90
                    target_azimuth = (target_azimuth - 90) % 360

                elif direction_fix_value == "swap_left_right":
                    target_azimuth = (360 - target_azimuth) % 360

                elif direction_fix_value == "rotate_ccw_90":
                    target_azimuth = (target_azimuth + 90) % 360

                elif direction_fix_value == "rotate_cw_90":
                    target_azimuth = (target_azimuth - 90) % 360

                elif direction_fix_value == "invert_elevation":
                    target_elevation = -target_elevation
                    
                elif direction_fix_value == "back_cw_start":
                    # Dataset's 0° is at the back AND rotates clockwise
                    # Equivalent to 180° shift + azimuth flip
                    target_azimuth = (target_azimuth + 180) % 360
                    target_azimuth = (-target_azimuth) % 360

            # Support ("offset", degrees) tuples
            elif isinstance(direction_fix_value, tuple) and direction_fix_value[0] == "offset":
                offset = float(direction_fix_value[1])
                target_azimuth = (target_azimuth + offset) % 360
                
            # Log which correction is used
            hf.log_with_timestamp(f"Direction correction applied: {direction_fix_value}",
                                  gui_logger=gui_logger)

    except Exception as ex:
        hf.log_with_timestamp("Direction correction failed",
                              gui_logger=gui_logger,
                              log_type=2,
                              exception=ex)

    # --------------------------
    # Nearest lookup
    # --------------------------
    n_positions = sofa_source_positions.shape[0]
    nearest_idx = 0
    nearest_distance = 1e9

    try:
        for i in range(n_positions):
            az = int(sofa_source_positions[i, 0])
            el = int(sofa_source_positions[i, 1])

            dist = sqrt((el - target_elevation) ** 2 +
                        (az - target_azimuth) ** 2)

            if dist < nearest_distance:
                nearest_distance = dist
                nearest_idx = i

    except Exception as ex:
        hf.log_with_timestamp("Direction search failed",
                              gui_logger=gui_logger,
                              log_type=2,
                              exception=ex)

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
        freqs = np.fft.rfftfreq(N, d=1 / sample_rate)

        # ------------ Filters ------------
        lp_sos = hf.get_filter_sos(
            cutoff=10000, fs=CN.FS, order=8, b_type='low'
        )

        # ------------ Stage 1: Per-listener direction alignment ------------
        if align_directions:
            hf.log_with_timestamp("Performing intra-listener alignment...", gui_logger)
            for i in range(n_list):
                for el in range(elev_n):
                    for az in range(azim_n):
                        hrir_sets[i][el, az, :2, :] = shift_2d_impulse_response(
                            hrir_sets[i][el, az, :2, :], lp_sos, target_index=55
                        )

        # ------------ Stage 2: Global alignment across listeners ------------
        if align_listeners:
            hf.log_with_timestamp("Performing inter-listener alignment...", gui_logger)
            global_ref = hrir_sets[0][elev_n // 2, 0, 0]
            for i in range(1, n_list):
                local_ref = hrir_sets[i][elev_n // 2, 0, 0]
                corr = correlate(local_ref, global_ref, mode='full')
                delay = np.argmax(corr) - len(local_ref) + 1
                if np.isfinite(delay):
                    hf.log_with_timestamp(
                        f"Aligning listener {i} globally (shift={delay} samples)",
                        gui_logger
                    )
                    hrir_sets[i] = np.roll(hrir_sets[i], -int(delay), axis=-1)
            hf.log_with_timestamp("Global alignment done.", gui_logger)

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

        # ------------ Stage 4: Convert to frequency domain ------------
        hf.log_with_timestamp("Converting HRIRs to frequency domain...", gui_logger)
        hrtf_sets = np.array(
            [np.fft.rfft(hrir, axis=-1) for hrir in hrir_sets],
            dtype=np.complex128
        )   # shape: [n_list, elev, azim, ch, freq]

        # ------------ Stage 5: Interpolation ------------
        hf.log_with_timestamp(f"Interpolating across listeners (mode={interp_mode})...",
                              gui_logger)

        hrtf_avg = np.empty((elev_n, azim_n, ch_n, hrtf_sets.shape[-1]),
                            dtype=np.complex128)

        for el in range(elev_n):
            for az in range(azim_n):
                for ch in range(ch_n):

                    H = hrtf_sets[:, el, az, ch]   # shape: [n_list, n_freqs]

                    if interp_mode == "complex":
                        # unreliable
                        real_interp = np.mean(H.real, axis=0)
                        imag_interp = np.mean(H.imag, axis=0)
                        hrtf_avg[el, az, ch] = real_interp + 1j * imag_interp

                    elif interp_mode == "modular_linear":
                        mag = np.abs(H)
                        phase = np.angle(H)
                        mag_interp = np.mean(mag, axis=0)
                        phase_interp = circular_mean(phase)
                        hrtf_avg[el, az, ch] = mag_interp * np.exp(1j * phase_interp)
                        
                    elif interp_mode == "modular":#most reliable
                        # --- Magnitude (in dB) + Circular phase averaging ---

                        # Magnitude & phase
                        mag = np.abs(H)
                        phase = np.angle(H)
                        # Convert magnitude to dB (amplitude dB)
                        mag_db = 20 * np.log10(np.clip(mag, 1e-12, None))
                        # Average magnitudes IN dB
                        mag_db_interp = np.mean(mag_db, axis=0)
                        # Convert back to linear magnitude
                        mag_interp = 10 ** (mag_db_interp / 20)
                        # Circular phase averaging (preserves wrapped phase)
                        phase_interp = circular_mean(phase)
                        # Reconstruct complex HRTF
                        hrtf_avg[el, az, ch] = mag_interp * np.exp(1j * phase_interp)

                    else:
                        raise ValueError(f"Invalid interp_mode: {interp_mode}")

        # ------------ Stage 6: Return to time domain ------------
        hf.log_with_timestamp("Converting averaged HRTF back to time domain...", gui_logger)
        hrir_avg = np.fft.irfft(hrtf_avg, n=N, axis=-1)

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
                raise ValueError("HRIR processing failed: invalid combination for 'Favourites' type")

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

    # --- Case 3: Standard favourite (lookup) ---
    elif hrtf_type == 'Favourites':
        hrtf_type, dataset, name_gui = get_hrtf_info_from_name_short(name_short=hrtf_gui)#gui name is same as name short for favourites, do a lookup to get remaining metadata
        hrtf_type = hrtf_type
        hrtf_dataset = dataset
        hrtf_short = hrtf_gui
        hrtf_gui = name_gui
        
    #else, no modification to metadata required

    return hrtf_type, hrtf_dataset, hrtf_gui, hrtf_short




    

    
   
    
def load_hrirs_list(hrtf_dict_list, spatial_res=2, direction_fix_gui=CN.HRTF_DIRECTION_FIX_LIST_GUI[0], direction_fix_trigger=False, gui_logger=None):
    """
    Load a list of HRIR datasets using fully-specified HRTF metadata dictionaries.
    Each dict must contain: hrtf_type, hrtf_dataset, hrtf_gui, hrtf_short.
    Returns: (list of HRIR arrays, status_code, metadata_list)
    """
    try:
        if not hrtf_dict_list:
            hf.log_with_timestamp("No HRIR metadata provided to load.", gui_logger)
            return [], 1, []

        hrir_list_loaded = []
        metadata_list = []

        for meta in hrtf_dict_list:
            hrtf_type = meta.get("hrtf_type")
            hrtf_dataset = meta.get("hrtf_dataset")
            hrtf_gui = meta.get("hrtf_gui")
            hrtf_short = meta.get("hrtf_short")
            if None in (hrtf_type, hrtf_dataset, hrtf_gui, hrtf_short):
                raise ValueError(f"Invalid dictionary entry (missing keys): {meta}")

            hrtf_type, hrtf_dataset, hrtf_gui, hrtf_short = hrtf_param_cleaning(
                hrtf_type, hrtf_dataset, hrtf_gui, hrtf_short)

            hf.log_with_timestamp(f"Loading HRTF data: {hrtf_gui}", gui_logger)

            npy_fname = get_hrir_file_path(
                hrtf_type=hrtf_type, hrtf_gui=hrtf_gui,
                hrtf_dataset=hrtf_dataset, hrtf_short=hrtf_short,
                spatial_res=spatial_res, gui_logger=gui_logger)

            metadata_fname = Path(npy_fname).with_name(f"{hrtf_short}_metadata.json")
            build_npy_hrir_dataset = True
            current_metadata = {}
            
            if direction_fix_trigger == False:
                direction_fix_gui=CN.HRTF_DIRECTION_FIX_LIST_GUI[0]#reset direction fix to default if not triggered, as it will be passed into SOFA workflow regardless

            # Determine whether to recreate HRIR via SOFA workflow. 
            gdrive_url = get_gdrive_url_high(listener_type=hrtf_type, dataset_name=hrtf_dataset,name_gui=hrtf_gui, gui_logger=gui_logger)
            # Non-empty or averaged hrtf, exempt from SOFA recreation
            if gdrive_url or hrtf_gui == CN.HRTF_AVERAGED_NAME_GUI:
                build_npy_hrir_dataset = False
                hf.log_with_timestamp(f"SOFA not available but NPY dataset is available, NPY HRIR dataset will be loaded directly for {hrtf_gui}.")
            #If direction fix is triggered or CTF is missing, recreate from SOFA to ensure npy dataset is up to date and has critical files
            elif Path(npy_fname).exists():
                if metadata_fname.exists():
                    try:
                        with open(metadata_fname, "r", encoding="utf-8") as f:
                            current_metadata = json.load(f)
                        ctf_file = current_metadata.get("ctf_file")
                        ctf_le_file = current_metadata.get("ctf_le_file")

                        if direction_fix_trigger == False and Path(ctf_file).exists() and Path(ctf_le_file).exists():
                            build_npy_hrir_dataset = False
                            hf.log_with_timestamp(f"NPY file and metadata OK, NPY HRIR dataset will be loaded directly for {hrtf_gui}.")
                        else:
                            hf.log_with_timestamp("Direction fix has been triggered or CTF files are missing, NPY HRIR dataset will be recreated from SOFA.",gui_logger)

                    except Exception as meta_e:
                        hf.log_with_timestamp(f"Failed to read metadata JSON: {meta_e}. NPY HRIR dataset will be recreated from SOFA.",gui_logger)
                else:
                    hf.log_with_timestamp(f"Metadata JSON not found for {hrtf_short}. NPY HRIR dataset will be recreated from SOFA.",gui_logger)
            else:
                hf.log_with_timestamp(f"NPY file not found for {hrtf_short}. NPY HRIR dataset will be recreated from SOFA.", gui_logger)

            # SOFA workflow if needed
            if build_npy_hrir_dataset:
                hf.log_with_timestamp(f"Creating HRIR dataset from SOFA for {hrtf_gui}.", gui_logger)
                status = sofa_workflow_new_dataset(
                    hrtf_type=hrtf_type, hrtf_dataset=hrtf_dataset,
                    hrtf_gui=hrtf_gui, hrtf_short=hrtf_short,
                    direction_fix_gui=direction_fix_gui, gui_logger=gui_logger)
                if status == 2:
                    hf.log_with_timestamp(f"SOFA workflow cancelled for {hrtf_gui}", gui_logger)
                    return [], 2, []
                elif status != 0:
                    raise ValueError(f"HRIR processing failed for {hrtf_gui} via SOFA workflow")

                if metadata_fname.exists():
                    try:
                        with open(metadata_fname, "r", encoding="utf-8") as f:
                            current_metadata = json.load(f)
                    except Exception:
                        current_metadata = {}

            # Load NPY
            hrir_list = hf.load_convert_npy_to_float64(npy_fname)
            hrir_selected = hrir_list[0]
            hrir_list_loaded.append(hrir_selected)
            metadata_list.append(current_metadata)

            elev, azim, ch, samples = hrir_selected.shape
            hf.log_with_timestamp(f"Loaded {hrtf_gui}: Elev={elev}, Azim={azim}, Ch={ch}, Samples={samples}")

        if len(hrir_list_loaded) > 1:
            hf.log_with_timestamp(f"{len(hrir_list_loaded)} HRIR datasets loaded.", gui_logger)

        return hrir_list_loaded, 0, metadata_list

    except Exception as e:
        hf.log_with_timestamp(f"Error loading HRIRs: {e}", gui_logger)
        return [], 1, []