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
import numpy as np
import os
import gdown
from math import sqrt
from pathlib import Path
import threading
import scipy as sp


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
                metadata_file_name = 'hrir_metadata.csv'
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
            
            
        else:
            listener_list=get_listener_list_user()
            if not listener_list:
                listener_list = ['No SOFA files found']
            
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
                metadata_file_name = 'hrir_metadata.csv'
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
            name_short = name_gui #case for user SOFA, use same as GUI name (file name)
            
    except Exception as ex:
        log_string=f"Error occurred: {ex}"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
   
        
    return name_short


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
                metadata_file_name = 'hrir_metadata.csv'
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
                metadata_file_name = 'hrir_metadata.csv'
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

def get_flip_azim_flag(listener_type="", dataset_name="", name_gui="", gui_logger=None):
    """
    Retrieves a flip azimuth flag based on the listener type and dataset name and gui name.
    
    Args:
        listener_type (str, optional): The type of listener ('Dummy Head / Head & Torso Simulator' or 'Human Listener'). Defaults to "".
        dataset_name (str, optional): The name of the dataset to filter by (only used for 'Human Listener'). Defaults to "".
        name_gui (str, optional): The GUI name to match in the dataset. Defaults to "".
        gui_logger (object, optional): A logger object for GUI-related logging. Defaults to None.

    Returns:
        bool: True if the flag is "yes", otherwise False.
    """
    
    flag = ""
    
    try:
    
 
        if listener_type == 'Dummy Head / Head & Torso Simulator' or listener_type == 'Human Listener':
            
            #HRTF related - individual npy datasets
            #load lists from csv file
            try:
            
                #directories
                csv_directory = CN.DATA_DIR_HRIR_NPY
                #read metadata from csv. Expects reverberation_metadata.csv
                metadata_file_name = 'hrir_metadata.csv'
                metadata_file = pjoin(csv_directory, metadata_file_name)
                with open(metadata_file, encoding='utf-8-sig', newline='') as inputfile:
                    reader = DictReader(inputfile)
                    for row in reader:#rows 2 and onward
                        #store each row as a dictionary
                        #append to list of dictionaries
                        if row.get('dataset') == dataset_name and row.get('hrtf_type') ==listener_type and row.get('name_gui') ==name_gui: #added check
                            flag = row.get('flip_azimuths_fb')
    
            except Exception as e:
                log_string=f"Error loading listener list: {e}"
                hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=e)#log error
                pass
   
        else:
            flag = ""
            
    except Exception as ex:
        log_string=f"Error occurred: {ex}"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
        
    return flag.lower() == "yes" if flag else False

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
                metadata_file_name = 'hrir_metadata.csv'
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
                metadata_file_name = 'hrir_metadata.csv'
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
                metadata_file_name = 'hrir_metadata.csv'
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
        metadata_file_name = 'hrir_metadata.csv'
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
        url = "https://drive.google.com/file/d/18ZEFwEKXQVhqpuyLp-FnyDZf35VJFyky/view?usp=drive_link"
        dl_file = pjoin(csv_directory, 'hrir_metadata_latest.csv')
        gdown.download(url, dl_file, fuzzy=True)

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
                
            url = "https://drive.google.com/file/d/18ZEFwEKXQVhqpuyLp-FnyDZf35VJFyky/view?usp=drive_link"
            dl_file = metadata_file
            gdown.download(url, dl_file, fuzzy=True)
            
            log_string = 'list of HRTF datasets updated. Restart required.'
            hf.log_with_timestamp(log_string, gui_logger)
      
        
        #if no mismatches flagged, print message
        if mismatches == 0:
            log_string = 'No updates available'
            hf.log_with_timestamp(log_string, gui_logger)
                
                

    except Exception as ex:
        log_string = 'Failed to validate versions or update data'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
    
  
    


def sofa_workflow_new_dataset(brir_hrtf_type, brir_hrtf_dataset, brir_hrtf, brir_hrtf_short, report_progress=0, gui_logger=None, spatial_res=2):
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
        
            
            
        #try loading specified sofa object    
        if brir_hrtf_type == 'Human Listener':
            hrir_dir_base = CN.DATA_DIR_SOFA
        elif brir_hrtf_type == 'Dummy Head / Head & Torso Simulator':
            hrir_dir_base = CN.DATA_DIR_SOFA
        else:
            hrir_dir_base = CN.DATA_DIR_SOFA_USER
        #form sofa file name
        sofa_local_fname = pjoin(hrir_dir_base, f"{brir_hrtf_short}.sofa")
            
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
            url = get_sofa_url(listener_type=brir_hrtf_type, dataset_name=brir_hrtf_dataset, name_gui=brir_hrtf, gui_logger=gui_logger)
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
                url = get_alternative_url(listener_type=brir_hrtf_type, dataset_name=brir_hrtf_dataset, name_gui=brir_hrtf, gui_logger=None)
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
        
        #extract data and place into npy array (resample if required)
        #flip_azimuths_fb = get_flip_azim_flag(listener_type=brir_hrtf_type, dataset_name=brir_hrtf_dataset, name_gui=brir_hrtf, gui_logger=None)
        hrir_out = sofa_dataset_transform(convention_name=convention_name, sofa_data_ir=sofa_data_ir, sofa_samplerate=sofa_samplerate, sofa_source_positions=sofa_source_positions, flip_azimuths_fb=False, reverse_azim_rot=False, gui_logger=gui_logger)
        
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
            elev_min=CN.SPATIAL_RES_ELEV_MIN[spatial_res] 
            elev_max=CN.SPATIAL_RES_ELEV_MAX[spatial_res] 
            elev_min_out=CN.SPATIAL_RES_ELEV_MIN_OUT[spatial_res] 
            elev_max_out=CN.SPATIAL_RES_ELEV_MAX_OUT[spatial_res] 
            elev_nearest=CN.SPATIAL_RES_ELEV_NEAREST[spatial_res] #as per hrir dataset
            elev_nearest_process=CN.SPATIAL_RES_ELEV_NEAREST_PR[spatial_res] 
            azim_nearest=CN.SPATIAL_RES_AZIM_NEAREST[spatial_res] 
            azim_nearest_process=CN.SPATIAL_RES_AZIM_NEAREST_PR[spatial_res] 
        else:
            raise ValueError('Invalid spatial resolution')
        total_azim_hrir = int(360/azim_nearest)
        total_elev_hrir = int((elev_max-elev_min)/elev_nearest +1)
        #determine spatial resolution parameters
        direction_matrix_process = brir_export.generate_direction_matrix(spatial_res=spatial_res, variant=0)
        
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
        hrir_fft_avg_mag = hf.level_spectrum_ends(hrir_fft_avg_mag, 150, 18000, smooth_win = 7, n_fft=CN.N_FFT)#100, 19000
        #octave smoothing
        hrir_fft_avg_mag_sm = hf.smooth_fft_octaves(data=hrir_fft_avg_mag, n_fft=CN.N_FFT)
        #invert response
        hrir_fft_avg_mag_inv = hf.db2mag(hf.mag2db(hrir_fft_avg_mag_sm)*-1)
        #create min phase FIR
        hrir_df_inv_fir = hf.mag_to_min_fir(hrir_fft_avg_mag_inv, crop=1, out_win_size=1024, n_fft=CN.N_FFT)
        

        
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
        
        #use multiple threads to perform EQ
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
        
        #apply high pass filter
        ts = []
        for elev in range(total_elev_hrir):
            t = threading.Thread(target=apply_hp_to_hrirs, args = (hrir_out, elev, total_samples_hrir, total_azim_hrir, direction_matrix_process))
            ts.append(t)
            t.start()
        for t in ts:
           t.join()
        
        
   
        log_string = 'SOFA dataset processed'
        hf.log_with_timestamp(log_string, gui_logger)
            
        #
        #form output directory
        if brir_hrtf_type == 'Human Listener':
            hrir_dir_base = CN.DATA_DIR_HRIR_NPY_HL
        elif brir_hrtf_type == 'Dummy Head / Head & Torso Simulator':
            hrir_dir_base = CN.DATA_DIR_HRIR_NPY_DH
        else:
            hrir_dir_base = CN.DATA_DIR_HRIR_NPY_USER
        sub_directory = 'h'
        #join spatial res subdirectory
        hrir_dir = pjoin(hrir_dir_base, sub_directory)
        # join dataset subdirectory
        if brir_hrtf_type != 'User SOFA Input':
            hrir_dir = pjoin(hrir_dir, brir_hrtf_dataset)
        # full filename
        npy_fname = pjoin(hrir_dir, f"{brir_hrtf_short}.npy")
                
        # Convert the data type to float32 and replace the original array, no longer need higher precision
        if hrir_out.dtype == np.float64:
            hrir_out = hrir_out.astype(np.float32)
            
        #save npy
        output_file = Path(npy_fname)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        np.save(npy_fname,hrir_out)    
        
        
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
    Function applies equalisation to brirs in numpy array
    :param hrir_out: numpy array containing brirs
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
  
  
def apply_hp_to_hrirs(hrir_out, elev, total_samples_hrir, total_azim_hrir, direction_matrix_process):
    """
    Function applies equalisation to brirs in numpy array
    :param hrir_out: numpy array containing brirs
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
                #apply DF eq
                hrir_eq_b = hf.signal_highpass_filter(hrir_eq_b, f_crossover_var, CN.FS, CN.ORDER)
                hrir_out[set_id,elev,azim,chan,:] = np.copy(hrir_eq_b[0:total_samples_hrir])
            
            else:
                hrir_out[set_id,elev,azim,chan,:] = np.zeros(total_samples_hrir)#zero out directions that wont be exported    
  
  
    
  
  
  
def sofa_dataset_transform(convention_name, sofa_data_ir, sofa_samplerate, sofa_source_positions, flip_azimuths_fb=False, reverse_azim_rot=False, gui_logger=None, spatial_res=2):

    """ 
    Function peforms the following transformations:
        extracts hrirs from sofa_data_ir 
        resamples if required depending on sofa_samplerate
        crops to 256 sample window
        places into an npy dataset using sofa_source_positions for nearest directions    
        
        Args:
        convention_name (str): sofa convention
        sofa_data_ir (npy array): data IR array from sofa file, usually 3 dimensions, measurements x receivers x samples
        sofa_samplerate (int):  sample rate
        sofa_source_positions(npy array):source positions array from sofa file, 3 spherical dimensions, azimuth (deg) x elevation (deg) x distance, 2d array: position x coordinate tuple
        reverse_azim_rot (bool): flag to reverse azimuth direction, positive = CCW by default
    """
    hrir_out=None
    try:
        #filters
        cutoff=10000#14000
        fs=CN.FS
        order=8
        method=1
        lp_sos = hf.get_filter_sos(cutoff=cutoff, fs=fs, order=order, b_type='low')
        
        if convention_name in CN.SOFA_COMPAT_CONV:
            #in GeneralFIRE convention, dim 1 = measurements, dim 2 = receivers (L + R), dim 3 = samples, dim 4 = emitters (speakers):
            #in GeneralFIR and SimpleFreeFieldHRIR conventions, dim 1 = measurements, dim 2 = receivers (L + R), dim 3 = samples e.g. size [648, 2, 2048]
            # Get the lengths of the first 3 dimensions
            n_measurements = sofa_data_ir.shape[0]
            n_receivers = sofa_data_ir.shape[1]
            n_samples = sofa_data_ir.shape[2]  
        else:
            raise ValueError('Invalid SOFA convention. Not yet supported.')
        
        #coordinate system and dimensions
        if spatial_res >= 0 and spatial_res < CN.NUM_SPATIAL_RES:
            elev_min=CN.SPATIAL_RES_ELEV_MIN[spatial_res] 
            elev_max=CN.SPATIAL_RES_ELEV_MAX[spatial_res] 
            elev_min_out=CN.SPATIAL_RES_ELEV_MIN_OUT[spatial_res] 
            elev_max_out=CN.SPATIAL_RES_ELEV_MAX_OUT[spatial_res] 
            elev_nearest=CN.SPATIAL_RES_ELEV_NEAREST[spatial_res] #as per hrir dataset
            elev_nearest_process=CN.SPATIAL_RES_ELEV_NEAREST_PR[spatial_res] 
            azim_nearest=CN.SPATIAL_RES_AZIM_NEAREST[spatial_res] 
            azim_nearest_process=CN.SPATIAL_RES_AZIM_NEAREST_PR[spatial_res] 
        else:
            raise ValueError('Invalid spatial resolution')
        output_azims = int(360/azim_nearest)
        output_elevs = int((elev_max-elev_min)/elev_nearest +1)
        total_chan_hrir=CN.TOTAL_CHAN_HRIR
        total_samples_hrir=CN.TOTAL_SAMPLES_HRIR
        total_sets=1
        set_id=0
        #create empty npy dataset
        #HRIR array will be populated with all HRIRs
        hrir_out=np.zeros((total_sets,output_elevs,output_azims,total_chan_hrir,total_samples_hrir))
        hrir_selected=np.zeros((total_chan_hrir,n_samples))
        #hrir_sample=np.zeros((total_chan_hrir,n_samples))
        
        populate_samples = min(total_samples_hrir,n_samples)
        samp_freq_ash =CN.SAMP_FREQ
        #resample if samp_freq is not 44100
        if sofa_samplerate != samp_freq_ash:
            populate_samples_re = round(populate_samples * float(samp_freq_ash) / sofa_samplerate)#recalculate no. samples 
            populate_samples = min(populate_samples,populate_samples_re)
            log_string = 'Resampling dataset'
            hf.log_with_timestamp(log_string, gui_logger)
            
        # #retrieve sample as reference for shifting
        # sample_elev_deg = 0
        # sample_azim_deg=0
        # #get nearest direction
        # nearest_dir_idx = sofa_find_nearest_direction(sofa_source_positions=sofa_source_positions, target_elevation=sample_elev_deg, target_azimuth=sample_azim_deg, flip_azimuths_fb=flip_azimuths_fb, reverse_azim_rot=reverse_azim_rot)
        # #grab sample HRIR
        # for chan in range(total_chan_hrir):
        #     query_sofa_data(convention_name,hrir_sample,sofa_data_ir,nearest_dir_idx,chan)
        # #resample if samp_freq is not 44100
        # if sofa_samplerate != samp_freq_ash:
        #     hrir_sample = hf.resample_signal(hrir_sample, original_rate = sofa_samplerate, new_rate = samp_freq_ash, axis=1, scale=True)
        # #shift
        # hrir_sample=shift_2d_impulse_response(hrir_sample, lp_sos, target_index=57)#46
                
        #for each direction in output array
        #for each elev and az
        for elev in range(output_elevs):
            elev_deg = int(elev_min + elev*elev_nearest)
            for azim in range(output_azims):
                azim_deg = int(azim*azim_nearest)
                #get nearest direction
                nearest_dir_idx = sofa_find_nearest_direction(sofa_source_positions=sofa_source_positions, target_elevation=elev_deg, target_azimuth=azim_deg, flip_azimuths_fb=flip_azimuths_fb, reverse_azim_rot=reverse_azim_rot)
                
                #grab HRIR of this direction
                hrir_selected=np.zeros((total_chan_hrir,n_samples))
                for chan in range(total_chan_hrir):
                    query_sofa_data(convention_name,hrir_selected,sofa_data_ir,nearest_dir_idx,chan)
        
                
                #resample if samp_freq is not 44100
                if sofa_samplerate != samp_freq_ash:
                    hrir_selected = hf.resample_signal(hrir_selected, original_rate = sofa_samplerate, new_rate = samp_freq_ash, axis=1, scale=True)
                    
                #shift
                hrir_selected=shift_2d_impulse_response(hrir_selected, lp_sos, target_index=58)#46
                #hrir_selected=shift_2d_impulse_response_cc(hrir_selected, hrir_sample, lp_sos)#use cross correlation to align in TD
                
                
                #place into npy array (also crops)
                for chan in range(total_chan_hrir):
                    hrir_out[set_id,elev,azim,chan,0:populate_samples] = np.copy(hrir_selected[chan,0:populate_samples])
                

        #resample if samp_freq is not 44100
        if sofa_samplerate != samp_freq_ash:
            log_string_a = 'source samplerate: ' + str(sofa_samplerate) + ', resampled to: '+ str(samp_freq_ash)
            hf.log_with_timestamp(log_string_a)
            
    except Exception as ex:
        log_string = 'SOFA transform failed'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error

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


def sofa_find_nearest_direction(sofa_source_positions, target_elevation, target_azimuth, flip_azimuths_fb=False, reverse_azim_rot=False, spatial_res=2, gui_logger=None):
    """
    Function returns an int corresponding to index of nearest available azimuth and elevation angle for a specified hrtf and azimuth and elevation
    Used to determine elevations and azimuths available to read from sofa dataset
    """
    
    nearest_idx=None
    
    try:
        
        target_elevation = int(target_elevation)
        target_azimuth = int(target_azimuth)
        if reverse_azim_rot == True and target_azimuth>0:
            target_azimuth = 360-target_azimuth
        elif flip_azimuths_fb == True:
            if target_azimuth <= 180:
                target_azimuth = 180-target_azimuth
            else:
                target_azimuth = 180+360-target_azimuth
            
        else:
            target_azimuth = int(target_azimuth)
        n_positions = sofa_source_positions.shape[0]
 
        if spatial_res >= CN.NUM_SPATIAL_RES:
            raise ValueError('Invalid spatial resolution')
 
        nearest_distance = 10000.0 #start with large number
        nearest_elevation = target_elevation
        nearest_azimuth = target_azimuth

        #for each elev and az, calculate distance
        for position in range(n_positions):
            coordinate = np.copy(sofa_source_positions[position,:])
            azim_deg = int(coordinate[0])
            elev_deg = int(coordinate[1])
            current_distance = sqrt(abs(elev_deg-target_elevation)**2 + abs(azim_deg-target_azimuth)**2)
            #store this direction if it is closer than previous
            if current_distance < nearest_distance:
                nearest_distance=current_distance
                nearest_elevation=elev_deg
                nearest_azimuth=azim_deg
                nearest_idx = position
                    

        
    except Exception as ex:
        log_string="Error occurred"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
        
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
