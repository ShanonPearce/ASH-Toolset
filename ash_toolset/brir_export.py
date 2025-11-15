# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 23:20:14 2023

@author: Shanon
"""


# import packages
import numpy as np
from os.path import join as pjoin
from pathlib import Path
import time
import logging
from ash_toolset import helper_functions as hf
from ash_toolset import constants as CN
import shutil
import os
from math import sqrt
from SOFASonix import SOFAFile
import json
from datetime import datetime
import glob
logger = logging.getLogger(__name__)
log_info=1

# get the start time
st = time.time()


def export_brir(brir_arr,  brir_name, primary_path, brir_dir_export=True, brir_ts_export=True, hesuvi_export=True, gui_logger=None, multichan_export=False, 
                multichan_mapping=CN.PRESET_16CH_LABELS[0],
                 spatial_res=1, sofa_export=False, reduce_dataset=False, brir_dict={}, sofa_conv=None, use_stored_brirs=False, brir_dict_list=[]):
    """
    Function to export a customised BRIR to WAV files
    :param brir_arr: numpy array, containing set of BRIRs. 4d array. d1 = elevations, d2 = azimuths, d3 = channels, d4 = samples
    :param acoustic_space: str, shorthand name of selected acoustic space
    :param brir_name: string, name of brir
    :param primary_path: string, base path to save files to
    :param brir_dir_export: int, 1 = export directional wav brirs
    :param brir_ts_export: int, 1 = export true stereo wav brirs
    :param hesuvi_export: int, 1 = export hesuvi compatible wav brirs
    :param direct_gain_db: float, adjust gain of direct sound in dB
    :param gui_logger: gui logger object for dearpygui
    :param spatial_res: int, spatial resolution, 0= low (-30 to 30 deg elev, nearest 15 deg elev, 5 deg azim) 1 = moderate (-45 to 45 deg elev, nearest 15 deg elev, 5 deg azim), 2 = high (-50 to 50 deg elev, nearest 5 deg elev, 5 deg azim), 3 = full (-50 to 50 deg elev, nearest 2 deg elev, 2 deg azim)
    :param samp_freq: int, sample frequency in Hz
    :param brir_dict: dict, contains inputs relating to channel directions
    :return: None
    """ 
    
    
    # get the start time
    st = time.time()
    brir_data =[]
    try:
        
        log_string = 'Preparing BRIRs for export'
        hf.log_with_timestamp(log_string, gui_logger) 
        
        
    
        if use_stored_brirs == False:#standard mode
            if not isinstance(brir_arr, np.ndarray) or brir_arr.ndim != 4:
                raise ValueError("brir_arr must be a 4D NumPy array [elev, azim, channels, samples]")
            total_samples_brir = len(brir_arr[0][0][0])    
            #gain adjustment
            max_amp = np.max(np.abs(brir_arr))
        else:
            if not brir_dict_list:
                raise ValueError("brir_dict_list is empty — expected at least one stored BRIR entry when use_stored_brirs=True.")
            sample_dict = brir_dict_list[0]
            sample_brir=sample_dict["out_wav_array"]
            total_samples_brir = len(sample_brir)    
            max_amp=1#already adjusted gain
  
        #hesuvi path
        if 'EqualizerAPO' in primary_path:
            hesuvi_path = pjoin(primary_path,'HeSuVi')#stored outside of project folder (within hesuvi installation)
        else:
            hesuvi_path = pjoin(primary_path, CN.PROJECT_FOLDER,'HeSuVi')#stored within project folder
    
        #get relevant information from dict
        if brir_dict:
            if brir_name == CN.FOLDER_BRIRS_LIVE: 
                direct_gain_db = brir_dict.get("qc_direct_gain_db")
                samp_freq = brir_dict.get("qc_samp_freq_int")
                bit_depth = brir_dict.get("qc_bit_depth")
            else:
                direct_gain_db = brir_dict.get("fde_direct_gain_db")
                samp_freq = brir_dict.get("fde_samp_freq_int")
                bit_depth = brir_dict.get("fde_bit_depth")
            
            
            hrtf_symmetry = brir_dict.get("hrtf_symmetry")
            early_refl_delay_ms = brir_dict.get("er_delay_time")
        else:
            raise ValueError('brir_dict not populated')
    

        
        
        #reduce gain if direct gain db is less than max value
        reduction_gain_db = (CN.DIRECT_GAIN_MAX-direct_gain_db)*-1/2
        reduction_gain = hf.db2mag(reduction_gain_db)
        reduction_gain_db_he = (reduction_gain_db)
        reduction_gain_he = hf.db2mag(reduction_gain_db_he)
        
        #output array length = input array length due to being already cropped
        out_wav_samples_44 = total_samples_brir
        #also calculate for 48kHz
        out_wav_samples_48 = round(out_wav_samples_44 * float(48000) / 44100) 
            
        if spatial_res >= 0 and spatial_res < CN.NUM_SPATIAL_RES:
            elev_min=CN.SPATIAL_RES_ELEV_MIN_IN[spatial_res] 
            elev_max=CN.SPATIAL_RES_ELEV_MAX_IN[spatial_res] 
            elev_nearest=CN.SPATIAL_RES_ELEV_NEAREST_IN[spatial_res] #as per hrir dataset
            elev_nearest_process=CN.SPATIAL_RES_ELEV_NEAREST_PR[spatial_res] 
            azim_nearest=CN.SPATIAL_RES_AZIM_NEAREST_IN[spatial_res] 
            azim_nearest_process=CN.SPATIAL_RES_AZIM_NEAREST_PR[spatial_res] 
            total_azim_brir = int(360/azim_nearest)#should be identical to above lengths
            total_elev_brir = int((elev_max-elev_min)/elev_nearest +1)#should be identical to above lengths
        else:
            raise ValueError('Invalid spatial resolution')
        
        #
        ## write set of directional WAVs
        #
   
        out_wav_array=np.zeros((out_wav_samples_44,2))
 
        #output directory - wav files
        brir_folder = brir_name
        out_file_dir_wav = pjoin(primary_path, CN.PROJECT_FOLDER_BRIRS,brir_folder)
        
 
        if brir_dir_export == True:
            Path(out_file_dir_wav).mkdir(exist_ok=True, parents=True)#never need to call mkdir inside the loop
            if not hf.check_write_permissions(out_file_dir_wav, gui_logger):
                # Skip exporting or raise exception
                return brir_data
            
            if reduce_dataset == True:
                #case for reduced dataset
                direction_matrix_out = generate_direction_matrix(spatial_res=spatial_res, variant=3, brir_dict=brir_dict)
            else:
                #regular case
                direction_matrix_out = generate_direction_matrix(spatial_res=spatial_res, variant=2)
            #also a matrix for storing select brirs
            direction_matrix_store = generate_direction_matrix(spatial_res=spatial_res, variant=1)
            
            #full brir dataset not provided, using dictionary list (previous output stored in memory)
            if use_stored_brirs == True:
                for elev in range(total_elev_brir):
                    elev_deg = int(elev_min + elev*elev_nearest)
                    elev_deg_wav=elev_deg
                    for azim in range(total_azim_brir):
                        azim_deg = int(azim*azim_nearest)
                        azim_deg_wav = int(0-azim_deg) if azim_deg < 180 else int(360-azim_deg)
                        if direction_matrix_out[elev][azim][0][0] == 1:  
                            
                            for data_dict in brir_dict_list:
                                elev_deg_dict = data_dict["elev_deg_wav"]
                                azim_deg_dict = data_dict["azim_deg_wav"]
                                
                                if elev_deg_dict == elev_deg_wav and azim_deg_dict == azim_deg_wav:#matching direction found
                                    out_wav_array = data_dict["out_wav_array"]
                                    #write wav
                                    out_file_name = 'BRIR' + '_E' + str(elev_deg_wav) + '_A' + str(azim_deg_wav) + '.wav'
                                    
                                    out_file_path = pjoin(out_file_dir_wav,out_file_name)
                                    
                                    # #create dir if doesnt exist
                                    # output_file = Path(out_file_path)
                                    # output_file.parent.mkdir(exist_ok=True, parents=True)
                                    
                                    #resample if samp_freq is not 44100
                                    if samp_freq != CN.SAMP_FREQ:
                                        out_wav_resampled = hf.resample_signal(out_wav_array, new_rate=samp_freq)
                                        hf.write2wav(file_name=out_file_path, data=out_wav_resampled, bit_depth=bit_depth, samplerate=samp_freq)
                                    else:
                                        hf.write2wav(file_name=out_file_path, data=out_wav_array, bit_depth=bit_depth, samplerate=samp_freq)
            #normal process, use full dataset
            else:
     
                        
                # removal of old files
                if reduce_dataset:
                    if os.path.exists(out_file_dir_wav):
                        files = glob.glob(pjoin(out_file_dir_wav, '*'))  # get all files in directory
                        deleted = 0
                        failed = 0
                        for f in files:
                            try:
                                os.remove(f)
                                deleted += 1
                            except Exception as e:
                                failed += 1
                                hf.log_with_timestamp(
                                    f"Warning: Could not delete file '{f}': {e}",
                                    gui_logger,
                                    log_type=1,
                                    exception=e
                                )
                        # summary log
                        hf.log_with_timestamp(
                            f"Cleared directory: {out_file_dir_wav} "
                            f"(deleted={deleted}, failed={failed})",
                            gui_logger
                        )
                    else:
                        os.makedirs(out_file_dir_wav)
  
                #print(str(elev_min))
                #print(str(elev_nearest))
                
                #write stereo wav for each elev and az
                
                for elev in range(total_elev_brir):
                    elev_deg = int(elev_min + elev*elev_nearest)
                    elev_deg_wav=elev_deg
                    for azim in range(total_azim_brir):
                        azim_deg = int(azim*azim_nearest)
                        azim_deg_wav = int(0-azim_deg) if azim_deg < 180 else int(360-azim_deg)
                        
                        if direction_matrix_out[elev][azim][0][0] == 1 or direction_matrix_store[elev][azim][0][0] == 1: 
           
                            brir_chunk = brir_arr[elev][azim]
                            if brir_chunk.shape[0] != 2:
                                raise ValueError(f"Expected 2 channels, got {brir_chunk.shape[0]} at elev {elev}, azim {azim}")
                            
                            samples_to_use = min(out_wav_samples_44, brir_chunk.shape[1])
                            #out_wav_array[:] = 0  # clear
                            out_wav_array = np.zeros((out_wav_samples_44, 2), dtype=brir_arr.dtype)
                            out_wav_array[:samples_to_use, :] = brir_chunk[:, :samples_to_use].T * (reduction_gain/max_amp)
                            
                            
                        if direction_matrix_store[elev][azim][0][0] == 1 and brir_name == CN.FOLDER_BRIRS_LIVE: 
                            # Create a dictionary with the required variables
                            data_dict = {
                                "elev_deg_wav": elev_deg_wav,
                                "azim_deg_wav": azim_deg_wav,
                                "out_wav_array": np.copy(out_wav_array) # Keep as NumPy array if suitable, before resampling
                            }
        
                            # Append the dictionary to the list for later use
                            brir_data.append(data_dict)
                            
                        if direction_matrix_out[elev][azim][0][0] == 1:  
                            #write wav
                            out_file_name = 'BRIR' + '_E' + str(elev_deg_wav) + '_A' + str(azim_deg_wav) + '.wav'
                            
                            out_file_path = pjoin(out_file_dir_wav,out_file_name)
                            
                            # #create dir if doesnt exist
                            # output_file = Path(out_file_path)
                            # output_file.parent.mkdir(exist_ok=True, parents=True)
                            
                            #resample if samp_freq is not 44100
                            if samp_freq != CN.SAMP_FREQ:
                                out_wav_resampled = hf.resample_signal(out_wav_array, new_rate=samp_freq)
                                hf.write2wav(file_name=out_file_path, data=out_wav_resampled, bit_depth=bit_depth, samplerate=samp_freq)
                            else:
                                hf.write2wav(file_name=out_file_path, data=out_wav_array, bit_depth=bit_depth, samplerate=samp_freq)
                            
  
            #finished
            log_string = 'BRIR WAV set saved to: ' + str(out_file_dir_wav)
            hf.log_with_timestamp(log_string, gui_logger) 
  
            

  
        #
        ## write set of HESUVI WAVs or True stereo WAVs or 16 Channel WAVs
        #
        
        if use_stored_brirs == False:
  
            if (brir_ts_export or hesuvi_export or multichan_export):
            
                # Validate spatial resolution
                if 0 <= spatial_res < CN.NUM_SPATIAL_RES:
                    elev_min = CN.SPATIAL_RES_ELEV_MIN_IN[spatial_res]
                    elev_nearest = CN.SPATIAL_RES_ELEV_NEAREST_IN[spatial_res]
                    azim_nearest = CN.SPATIAL_RES_AZIM_NEAREST_IN[spatial_res]
                else:
                    raise ValueError('Invalid spatial resolution')
            
                # Output directories
                out_file_dir_wav_44 = pjoin(hesuvi_path, 'hrir', '44')
                out_file_dir_wav_48 = pjoin(hesuvi_path, 'hrir')
            
                # Base sources
                SOURCES_DICT = [
                    {"name": "FC",  "dict_keys": ("h_elev_c", "h_azim_c"), "default_az": 0,  "he_channels": (6, 13)},
                    {"name": "FL",  "dict_keys": ("h_elev_fl", "h_azim_fl"), "default_az": 30, "he_channels": (0, 1), "ts_channels": (0, 1)},
                    {"name": "FR",  "dict_keys": ("h_elev_fr", "h_azim_fr"), "default_az": 330,"he_channels": (8, 7), "ts_channels": (2, 3)},
                    {"name": "SL",  "dict_keys": ("h_elev_sl", "h_azim_sl"), "default_az": 90, "he_channels": (2, 3)},
                    {"name": "SR",  "dict_keys": ("h_elev_sr", "h_azim_sr"), "default_az": 270,"he_channels": (10, 9)},
                    {"name": "BL",  "dict_keys": ("h_elev_rl", "h_azim_rl"), "default_az": 135,"he_channels": (4, 5)},
                    {"name": "BR",  "dict_keys": ("h_elev_rr", "h_azim_rr"), "default_az": 225,"he_channels": (12, 11)},
                ]
            
                # Initialize arrays
                brir_out_16ch   = np.zeros((CN.NUM_OUT_CHANNELS_MC, out_wav_samples_44))
                brir_out_44_he  = np.zeros((CN.NUM_OUT_CHANNELS_HE, out_wav_samples_44))
                brir_out_48_he  = np.zeros((CN.NUM_OUT_CHANNELS_HE, out_wav_samples_48))
                brir_out_44_ts  = np.zeros((CN.NUM_OUT_CHANNELS_TS, out_wav_samples_44))
            
                # --- Dynamic 16-channel mapping ---
                mapping_order = multichan_mapping.split("|")  # e.g., ['FL','FR','FC','LFE','SL','SR','BL','BR']
                
                # Identify LFE channels safely
                lfe_channels = None
                if "LFE" in mapping_order:
                    lfe_idx = mapping_order.index("LFE")
                    lfe_channels = (lfe_idx * 2, lfe_idx * 2 + 1)
            
                # Assign 16-channel indices dynamically for sources present
                for src in SOURCES_DICT:
                    if src["name"] in mapping_order:
                        idx = mapping_order.index(src["name"])
                        src["ch16"] = (idx * 2, idx * 2 + 1)
            
                # --- Loop over sources ---
                for src in SOURCES_DICT:
            
                    # Elevation and azimuth from GUI dict or defaults
                    if brir_dict:
                        selected_elev = int(brir_dict.get(src["dict_keys"][0], 0))
                        selected_az   = int(brir_dict.get(src["dict_keys"][1], 0))
                        dezired_az= 0-selected_az if selected_az<=0 else 360-selected_az
                    else:
                        selected_elev = 0
                        dezired_az = src["default_az"]
            
                    # Convert to array indices
                    elev_id = int((selected_elev - elev_min) / elev_nearest)
                    azim_id = int(dezired_az / azim_nearest)
            
                    # Load BRIR into zero-padded array
                    data_pad = np.zeros((out_wav_samples_44, 2))
                    data_pad[:, 0] = brir_arr[elev_id][azim_id][0, :out_wav_samples_44] * reduction_gain_he / max_amp
                    data_pad[:, 1] = brir_arr[elev_id][azim_id][1, :out_wav_samples_44] * reduction_gain_he / max_amp
            
                    # Resample to 48 kHz
                    data_pad_48k = hf.resample_signal(data_pad, new_rate=48000)
                    
                    # Enforce exact target length
                    target_len = out_wav_samples_48
                    if data_pad_48k.shape[0] > target_len:
                        data_pad_48k = data_pad_48k[:target_len, :]
                    elif data_pad_48k.shape[0] < target_len:
                        data_pad_48k = np.pad(data_pad_48k, ((0, target_len - data_pad_48k.shape[0]), (0, 0)))
            
                    # Assign to HESUVI channels
                    ch_left, ch_right = src["he_channels"]
                    brir_out_44_he[ch_left, :] = data_pad[:, 0]
                    brir_out_44_he[ch_right, :] = data_pad[:, 1]
                    brir_out_48_he[ch_left, :] = data_pad_48k[:, 0]
                    brir_out_48_he[ch_right, :] = data_pad_48k[:, 1]
            
                    # Assign to True Stereo if defined
                    if "ts_channels" in src:
                        ts_left, ts_right = src["ts_channels"]
                        brir_out_44_ts[ts_left, :] = data_pad[:, 0]
                        brir_out_44_ts[ts_right, :] = data_pad[:, 1]
            
                    # Assign to 16-channel array dynamically
                    if "ch16" in src:
                        ch16_left, ch16_right = src["ch16"]
                        brir_out_16ch[ch16_left, :] = data_pad[:, 0]
                        brir_out_16ch[ch16_right, :] = data_pad[:, 1]
            
                    # Copy FC data to LFE channels if both exist
                    if lfe_channels and src["name"] == "FC":
                        brir_out_16ch[lfe_channels[0], :] = data_pad[:, 0]
                        brir_out_16ch[lfe_channels[1], :] = data_pad[:, 1]
                            
    
                
                
            if brir_ts_export == True:
                #
                #true stereo
                #
                output_wav_ts = brir_out_44_ts
                output_wav_ts = output_wav_ts.transpose()
                out_file_name = 'BRIR_True_Stereo.wav'
                
                out_file_path = pjoin(out_file_dir_wav,out_file_name)
                if not hf.check_write_permissions(out_file_dir_wav, gui_logger):
                    # Skip exporting or raise exception
                    return brir_data
                
                #create dir if doesnt exist
                output_file = Path(out_file_path)
                output_file.parent.mkdir(exist_ok=True, parents=True)
                
                #resample if samp_freq is not 44100
                if samp_freq != CN.SAMP_FREQ:
                    output_wav_ts = hf.resample_signal(output_wav_ts, new_rate = samp_freq)
                
                hf.write2wav(file_name=out_file_path, data=output_wav_ts, bit_depth=bit_depth, samplerate=samp_freq)
            
                log_string = 'BRIR WAV True stereo saved to: ' + str(out_file_dir_wav)
                hf.log_with_timestamp(log_string, gui_logger) 
            
            if hesuvi_export == True:
                
                #
                #hesuvi 44khz
                #
                output_wav_he_44 = brir_out_44_he
                output_wav_he_44 = output_wav_he_44.transpose()
                out_file_name = '_'+brir_name + '.wav'
                
                out_file_path = pjoin(out_file_dir_wav_44, out_file_name)
                if not hf.check_write_permissions(out_file_dir_wav, gui_logger):
                    # Skip exporting or raise exception
                    return brir_data
                #create dir if doesnt exist
                output_file = Path(out_file_path)
                output_file.parent.mkdir(exist_ok=True, parents=True)
                
                hf.write2wav(file_name=out_file_path, data=output_wav_he_44, bit_depth=bit_depth, samplerate=44100)
                
                #
                #hesuvi 48khz
                #
                output_wav_he_48 = brir_out_48_he
    
                output_wav_he_48 = output_wav_he_48.transpose()
                out_file_name = '_'+brir_name + '.wav'
                
                out_file_path = pjoin(out_file_dir_wav_48, out_file_name)
                
                #create dir if doesnt exist
                output_file = Path(out_file_path)
                output_file.parent.mkdir(exist_ok=True, parents=True)
       
                hf.write2wav(file_name=out_file_path, data=output_wav_he_48, bit_depth=bit_depth, samplerate=48000)
            
                log_string = 'BRIR HESUVI WAV saved to: ' + str(out_file_dir_wav_48)
                hf.log_with_timestamp(log_string, gui_logger) 
            
            if multichan_export:
                #
                # 16-channel BRIR
                #
                output_wav_16ch = brir_out_16ch.transpose()  # shape (samples, channels)
                out_file_name = 'BRIR_16ch_' + brir_name + '.wav'
                out_file_path = pjoin(out_file_dir_wav, out_file_name)
                
                if not hf.check_write_permissions(out_file_dir_wav, gui_logger):
                    return brir_data
                
                # create dir if it doesn't exist
                output_file = Path(out_file_path)
                output_file.parent.mkdir(exist_ok=True, parents=True)
            
                # resample if samp_freq is not internal rate
                if samp_freq != CN.SAMP_FREQ:
                    output_wav_16ch = hf.resample_signal(output_wav_16ch, new_rate=samp_freq)
            
                # Write 16-channel WAV
                hf.write2wav(file_name=out_file_path, data=output_wav_16ch, bit_depth=bit_depth, samplerate=samp_freq)
                hf.log_with_timestamp('BRIR WAV 16-channel saved to: ' + str(out_file_dir_wav), gui_logger)
            
                # --- Write channel mapping to text file ---
                mapping_file_name = 'BRIR_16ch_channel_mapping.txt'
                mapping_file_path = pjoin(out_file_dir_wav, mapping_file_name)
            
                mapping_lines = []
                mapping_lines.append(f"Export order: {multichan_mapping}")
                mapping_lines.append("\nAssigned channel indices:")
            
                for src in SOURCES_DICT:
                    if "ch16" in src:
                        ch_left, ch_right = src["ch16"]
                        mapping_lines.append(f"{src['name']}: left={ch_left}, right={ch_right}")
                if lfe_channels:
                    mapping_lines.append(f"LFE: left={lfe_channels[0]}, right={lfe_channels[1]}")
            
                with open(mapping_file_path, "w") as f:
                    f.write("\n".join(mapping_lines))
            
                hf.log_with_timestamp('16-channel mapping saved to: ' + str(mapping_file_path), gui_logger)
    
  
            
            #
            ## write SOFA file
            #
             
            if sofa_export == True:
                export_sofa_brir(primary_path=primary_path,brir_arr=brir_arr, brir_set_name=brir_name, output_samples=out_wav_samples_44, spatial_res=spatial_res, samp_freq=samp_freq, sofa_conv=sofa_conv, gui_logger=gui_logger)
        
    
 
    
    except Exception as ex:
        log_string = 'Failed to export BRIRs'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
            
    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    if CN.LOG_INFO == True:
        logging.info('Execution time:' + str(elapsed_time) + ' seconds')
        
        
    return brir_data#return dict list even if empty
        

def ensure_dir(path):
    Path(path).parent.mkdir(exist_ok=True, parents=True)

def generate_direction_matrix(spatial_res=1, variant=1, brir_dict={}):
    """
    Function returns a numpy array containing a matrix of elevations and azimuths marked for export (variant) or processing
    :param spatial_res: int, spatial resolution, 0= low (-30 to 30 deg elev, nearest 15 deg elev, 5 deg azim) 1 = moderate (-45 to 45 deg elev, nearest 15 deg elev, 5 deg azim), 2 = high (-50 to 50 deg elev, nearest 5 deg elev, 5 deg azim), 3 = full (-50 to 50 deg elev, nearest 2 deg elev, 2 deg azim)
    :param variant: int,  0 = full range for processing, 1 = reduced set of directions intended for reducing post processing, 2 = reduced set of directions intended for wav export only, 3 = reduce dataset wav flagged (specified directions),
    """

    try:
 
        if spatial_res >= 0 and spatial_res < CN.NUM_SPATIAL_RES:
            elev_min=CN.SPATIAL_RES_ELEV_MIN_IN[spatial_res] 
            elev_max=CN.SPATIAL_RES_ELEV_MAX_IN[spatial_res] 
            elev_min_out=CN.SPATIAL_RES_ELEV_MIN_OUT[spatial_res] 
            elev_max_out=CN.SPATIAL_RES_ELEV_MAX_OUT[spatial_res] 
            elev_nearest=CN.SPATIAL_RES_ELEV_NEAREST_IN[spatial_res] #as per hrir dataset
            elev_nearest_process=CN.SPATIAL_RES_ELEV_NEAREST_PR[spatial_res] 
            elev_nearest_out=CN.SPATIAL_RES_ELEV_NEAREST_OUT[spatial_res] 
            azim_nearest=CN.SPATIAL_RES_AZIM_NEAREST_IN[spatial_res] 
            azim_nearest_process=CN.SPATIAL_RES_AZIM_NEAREST_PR[spatial_res] 
            azim_nearest_out=CN.SPATIAL_RES_AZIM_NEAREST_OUT[spatial_res] 
        else:
            raise ValueError('Invalid spatial resolution')
           
        output_azims = int(360/azim_nearest)
        output_elevs = int((elev_max-elev_min)/elev_nearest +1)
        
        #create numpy array to return
        direction_matrix=np.zeros((output_elevs,output_azims,1,1))  
        
        azim_horiz_range = CN.AZIM_HORIZ_RANGE
        elev_list = []
        azim_list = []
        #grab elev and azim data for reduced dataset case
        if variant == 3 and brir_dict: 
            elev_list=(brir_dict.get('elev_list'))
            azim_list=(brir_dict.get('azim_list'))

        #write stereo wav for each elev and az
        for elev in range(output_elevs):
            elev_deg = int(elev_min + elev*elev_nearest)
            for azim in range(output_azims):
                azim_deg = int(azim*azim_nearest)
                azim_deg_wav = int(0-azim_deg) if azim_deg < 180 else int(360-azim_deg)
                
 
                #reduced dataset flagged case - wav output
                if variant == 3:
                    if elev_deg in elev_list and (azim_deg_wav in azim_list or azim_deg in CN.AZIM_EXTRA_RANGE):
                        #populate matrix with 1 if direction applicable
                        direction_matrix[elev][azim][0][0] = 1
                #reduced set of directions for post processing or WAV output
                elif variant >= 1:
                    #limited elevation range for HRTF type 4 -> 250309: dont limit outputs
                    if spatial_res == 0 or spatial_res == 1: 
                        if (elev_deg >= elev_min_out and elev_deg <= elev_max_out) and elev_deg%elev_nearest_out == 0 and (azim_deg%CN.NEAREST_AZ_WAV == 0 or (elev_deg >= -30 and elev_deg <= 30 and azim_deg in azim_horiz_range)):  
                            #populate matrix with 1 if direction applicable
                            direction_matrix[elev][azim][0][0] = 1
                    elif spatial_res == 2 or spatial_res == 3: 
                        if variant == 2:#wav export variant is different only for high and max
                            if (elev_deg >= elev_min_out and elev_deg <= elev_max_out) and elev_deg%elev_nearest_out == 0 and azim_deg%azim_nearest_out == 0:  
                                #populate matrix with 1 if direction applicable
                                direction_matrix[elev][azim][0][0] = 1
                        else:
                            if elev_deg%elev_nearest_out == 0 and azim_deg%azim_nearest_out == 0:  
                                #populate matrix with 1 if direction applicable
                                direction_matrix[elev][azim][0][0] = 1

                else:#processing matrix
                    if (elev_deg%elev_nearest_process == 0 and azim_deg%azim_nearest_process == 0):
                        #populate matrix with 1 if direction applicable
                        direction_matrix[elev][azim][0][0] = 1
                        
        
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        
    return direction_matrix


def find_nearest_direction(target_elevation, target_azimuth, spatial_res=1):
    """
    Function returns a dict containing nearest available azimuth and elevation angle for a specified hrtf and azimuth and elevation
    Used to determine elevations and azimuths available to read from wav file dataset
    """
    
    try:
        
        target_elevation = int(target_elevation)
        target_azimuth = int(target_azimuth)
        
        
        if spatial_res >= 0 and spatial_res < CN.NUM_SPATIAL_RES:
            elev_min=CN.SPATIAL_RES_ELEV_MIN_IN[spatial_res] 
            elev_max=CN.SPATIAL_RES_ELEV_MAX_IN[spatial_res] 
            elev_min_out=CN.SPATIAL_RES_ELEV_MIN_OUT[spatial_res] 
            elev_max_out=CN.SPATIAL_RES_ELEV_MAX_OUT[spatial_res] 
            elev_nearest=CN.SPATIAL_RES_ELEV_NEAREST_IN[spatial_res] #as per hrir dataset
            elev_nearest_process=CN.SPATIAL_RES_ELEV_NEAREST_PR[spatial_res] 
            azim_nearest=CN.SPATIAL_RES_AZIM_NEAREST_IN[spatial_res] 
            azim_nearest_process=CN.SPATIAL_RES_AZIM_NEAREST_PR[spatial_res] 
        else:
            raise ValueError('Invalid spatial resolution')
        
        output_azims = int(360/azim_nearest)
        output_elevs = int((elev_max-elev_min)/elev_nearest +1)

        azim_horiz_range = CN.AZIM_HORIZ_RANGE
        
         
        nearest_distance = 1000.0 #start with large number
        nearest_elevation = target_elevation
        nearest_azimuth = target_azimuth

        #for each elev and az
        for elev in range(output_elevs):
            elev_deg = int(elev_min + elev*elev_nearest)
            for azim in range(output_azims):
                azim_deg = int(azim*azim_nearest)
                azim_deg_wav = int(0-azim_deg) if azim_deg < 180 else int(360-azim_deg)
                
                valid_dir = 0 
                
                if spatial_res == 0: 
                    if (elev_deg >= elev_min_out and elev_deg <= elev_max_out) and elev_deg%elev_nearest_process == 0 and (azim_deg%CN.NEAREST_AZ_WAV == 0 or (elev_deg >= -30 and elev_deg <= 30 and azim_deg in azim_horiz_range)):  
                        valid_dir = 1 
                elif spatial_res == 1: 
                    if (elev_deg >= elev_min_out and elev_deg <= elev_max_out) and elev_deg%elev_nearest_process == 0 and (azim_deg%CN.NEAREST_AZ_WAV == 0 or (elev_deg >= -30 and elev_deg <= 30 and azim_deg in azim_horiz_range)):  
                        valid_dir = 1 
                elif spatial_res == 2 or spatial_res == 3:
                    if (elev_deg >= elev_min_out and elev_deg <= elev_max_out) and elev_deg%elev_nearest_process == 0 and azim_deg%azim_nearest_process == 0:  
                        valid_dir = 1 
                else:
                    if (elev_deg%elev_nearest_process == 0 and azim_deg%azim_nearest_process == 0):
                        valid_dir = 1 
                
                if valid_dir == 1:  
                    current_distance = sqrt(abs(elev_deg-target_elevation)**2 + abs(azim_deg_wav-target_azimuth)**2)
                    #store this direction if it is closer than previous
                    if current_distance < nearest_distance:
                        nearest_distance=current_distance
                        nearest_elevation=elev_deg
                        nearest_azimuth=azim_deg_wav
                    
        out_dict = {'nearest_elevation': nearest_elevation, 'nearest_azimuth': nearest_azimuth, 'nearest_distance': nearest_distance}
        
        return out_dict
        
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        


def remove_brirs(primary_path, gui_logger=None):
    """
    Function deletes BRIRs and E-APO configs stored in a specified directory
    """
    out_file_dir_wav = pjoin(primary_path, CN.PROJECT_FOLDER_BRIRS)
    output_config_path = pjoin(primary_path, CN.PROJECT_FOLDER_CONFIGS_BRIR)
    output_hesuvi_path = pjoin(primary_path, CN.PROJECT_FOLDER,'HeSuVi','hrir')
    
    
    try:
        
        if os.path.exists(out_file_dir_wav):
            shutil.rmtree(out_file_dir_wav)
            log_string_a = 'Deleted folder and contents: ' + out_file_dir_wav 
            hf.log_with_timestamp(log_string_a, gui_logger) 
                
        if os.path.exists(output_config_path):
            shutil.rmtree(output_config_path)
            log_string_b = 'Deleted folder and contents: ' + output_config_path
            hf.log_with_timestamp(log_string_b, gui_logger) 
    
        if os.path.exists(output_hesuvi_path):
            shutil.rmtree(output_hesuvi_path)
            log_string_c = 'Deleted folder and contents: ' + output_hesuvi_path
            hf.log_with_timestamp(log_string_c, gui_logger)
    
    except Exception as ex:
        log_string = 'Failed to delete folders: ' + out_file_dir_wav + ' & ' + output_config_path
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
            
            
def remove_select_brirs(primary_path, brir_set, gui_logger=None):
    """
    Function deletes BRIRs in a specified directory
    """
    out_file_dir_brirs = pjoin(primary_path, CN.PROJECT_FOLDER_BRIRS)
    original_set_name =  brir_set.replace(" ", "_")

    try:
        
        for root, dirs, file in os.walk(out_file_dir_brirs):
            for dir in dirs:
                if original_set_name in dir:
                    folder_path=os.path.join(root, dir)
                    shutil.rmtree(folder_path)
                    #os.rmdir(folder_path)
                    log_string_a = 'Deleted folder and contents: ' + folder_path 
                    hf.log_with_timestamp(log_string_a, gui_logger)
            
 
    except Exception as ex:
        log_string = 'Failed to delete folder: ' + out_file_dir_brirs
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
            
            
def export_sofa_brir(primary_path, brir_arr, brir_set_name, spatial_res, output_samples, samp_freq, sofa_conv=None, gui_logger=None):
    """
    Function to export a customised BRIR to SOFA file
    :param brir_arr: numpy array, containing set of BRIRs. 4d array. d1 = elevations, d2 = azimuths, d3 = channels, d4 = samples
    :param brir_name: string, name of brir
    :param spatial_res: int, spatial resolution, 0= low (-30 to 30 deg elev, nearest 15 deg elev, 5 deg azim) 1 = moderate (-45 to 45 deg elev, nearest 15 deg elev, 5 deg azim), 2 = high (-50 to 50 deg elev, nearest 5 deg elev, 5 deg azim), 3 = full (-50 to 50 deg elev, nearest 2 deg elev, 2 deg azim)
    :param samp_freq: int, sample frequency in Hz
    :param output_samples: int, value in samples to crop output arrays assuming sample freq of 44.1kHz
    """
    
    now_datetime = datetime.now()
    
    try:
        if sofa_conv == None:
            sofa_conv='GeneralFIR'
        
        #dynamic matrix structure, limits depend on spatial resolution
        if spatial_res >= 0 and spatial_res < CN.NUM_SPATIAL_RES:
            elev_min=CN.SPATIAL_RES_ELEV_MIN_IN[spatial_res] 
            elev_max=CN.SPATIAL_RES_ELEV_MAX_IN[spatial_res] 
            elev_nearest=CN.SPATIAL_RES_ELEV_NEAREST_IN[spatial_res] #as per hrir dataset
            azim_nearest=CN.SPATIAL_RES_AZIM_NEAREST_IN[spatial_res] 
            
            
        else:
            raise ValueError('Invalid spatial resolution')
        
        total_elev_brir = len(brir_arr)
        total_azim_brir = len(brir_arr[0])
        total_chan_brir = len(brir_arr[0][0])
        total_samples_brir = len(brir_arr[0][0][0])
        
        #calculate size of dataset based on output elevations and azimuths
        total_directions = total_elev_brir*total_azim_brir
        
        #set number of output samples assuming base rate is 44.1kHz
        if output_samples > total_samples_brir:
            output_samples = max(total_samples_brir-1000,4410)  
            
        #recalculate based on selected sample rate
        output_samples_re = round(output_samples * float(samp_freq) / CN.SAMP_FREQ) 
        
        source_pos_array=np.zeros((total_directions,3)) 
        data_ir_array=np.zeros((total_directions,total_chan_brir,output_samples_re)) 
        

        radius=2#assume 2m distance
        source_pos_id=0
        for elev in range(total_elev_brir):
                elev_deg = int(elev_min + elev*elev_nearest)
                for azim in range(total_azim_brir):
                    azim_deg = int(azim*azim_nearest)
                    
                    #populate source position array
                    #tuple = (azimuth angles, elevation angles, radius)
                    source_pos_array[source_pos_id][0]=azim_deg
                    source_pos_array[source_pos_id][1]=elev_deg
                    source_pos_array[source_pos_id][2]=radius
                    
                    #grab input IR
                    pos_brir_array=np.zeros((output_samples,2))
                    #grab BRIR
                    pos_brir_array[:,0] = np.copy(brir_arr[elev,azim,0,0:output_samples])#L
                    pos_brir_array[:,1] = np.copy(brir_arr[elev,azim,1,0:output_samples])#R
                    
                    #resample if samp_freq is not 44100
                    if samp_freq != CN.SAMP_FREQ:
                        pos_brir_array = hf.resample_signal(pos_brir_array, new_rate = samp_freq)
                        
                    # --- FIX: ensure length consistency ---
                    if pos_brir_array.shape[0] != output_samples_re:
                        if pos_brir_array.shape[0] > output_samples_re:
                            pos_brir_array = pos_brir_array[:output_samples_re, :]
                        else:
                            pad_len = output_samples_re - pos_brir_array.shape[0]
                            pos_brir_array = np.pad(pos_brir_array, ((0, pad_len), (0, 0)))
                    
                    #populate data IR array
                    data_ir_array[source_pos_id,0,0:output_samples_re] = pos_brir_array[0:output_samples_re,0]#L
                    data_ir_array[source_pos_id,1,0:output_samples_re] = pos_brir_array[0:output_samples_re,1]#R
                    
 
                    #increment id
                    source_pos_id = source_pos_id+1
                    
        
        
        
        # with open(CN.METADATA_FILE) as fp:
        #     _info = json.load(fp)
        # __version__ = _info['version']
        app_version = CN.__version__
        
        # Create SOFAFile object with the latest SimpleFreeFieldHRIR convention
        sofa = SOFAFile(sofa_conv, sofaConventionsVersion=1.0, version=1.0)
        
        # Set dimensions
        sofa._M = total_directions#how many directions?
        sofa._N = total_samples_brir#how many samples?
        
        # View parameters of convention
        #sofa.view()
        
        
        """
        =============================== Attributes ====================================
        """
        
        # ----- Mandatory attributes -----
        sofa.GLOBAL_ApplicationName = "ASH Toolset"
        sofa.GLOBAL_ApplicationVersion = app_version
        sofa.GLOBAL_AuthorContact = "srpearce55@gmail.com"
        sofa.GLOBAL_Comment = "Generated by ASH Toolset"
        sofa.GLOBAL_History = "N/A"
        sofa.GLOBAL_License = "Creative Commons Attribution 4.0 International (CC-BY-4.0)"
        sofa.GLOBAL_Organization = "N/A"
        sofa.GLOBAL_References = "https://sourceforge.net/projects/ash-toolset/"
        sofa.GLOBAL_RoomType = "Customised Virtual Reverberant Room"
        sofa.GLOBAL_DateCreated = now_datetime
        sofa.GLOBAL_DateModified = now_datetime
        sofa.GLOBAL_Title = brir_set_name
        sofa.GLOBAL_DatabaseName = "ASH Toolset"
        sofa.GLOBAL_ListenerShortName = "Refer to title"
        sofa.ListenerPosition_Type = "cartesian"
        sofa.ListenerPosition_Units = "meter"
        sofa.ReceiverPosition_Type = "cartesian"
        sofa.ReceiverPosition_Units = "meter"
        sofa.SourcePosition_Type = "spherical"
        sofa.SourcePosition_Units = "degree, degree, meter"
        sofa.EmitterPosition_Type = "cartesian"
        sofa.EmitterPosition_Units = "meter"
        sofa.Data_SamplingRate_Units = "hertz"
        
        # ----- Non-Mandatory attributes -----
        
        
        """
        =============================== Double Variables ==============================
        """
        
        # ----- Mandatory double variables -----
        
        # Needs dimensions IC or MC
        sofa.ListenerPosition = np.atleast_2d(np.array([0, 0, 0]))
        
        #SimpleFreeFieldHRIR requires ListenerUp
        # Needs dimensions IC or MC
        if sofa_conv == 'SimpleFreeFieldHRIR':
            sofa.ListenerUp = np.atleast_2d(np.array([0, 0, 1]))
        
        #SimpleFreeFieldHRIR requires ListenerUp
        # Needs dimensions IC or MC
        if sofa_conv == 'SimpleFreeFieldHRIR':
            sofa.ListenerView = np.atleast_2d(np.array([1, 0, 0]))
        
        # Needs dimensions rCI or rCM
        sofa.ReceiverPosition = np.atleast_3d(np.array([[0, 0.09, 0],[0, -0.09, 0]]))
        
        # Needs dimensions IC or MC
        sofa.SourcePosition = source_pos_array
        
        # Needs dimensions eCI or eCM
        emitter_pos = np.zeros((1,3,1))
        sofa.EmitterPosition = emitter_pos
        
        # Needs dimensions mRn
        sofa.Data_IR = data_ir_array
        
        # Needs dimensions I
        sofa.Data_SamplingRate = [samp_freq]
        
        # Needs dimensions IR or MR
        data_delay = np.zeros((1,2))
        sofa.Data_Delay = data_delay
        
        # ----- Non-mandatory double variables -----
        
        
        """
        =============================== Export ========================================
        """
        
        # View parameters of convention
        #sofa.view()
        
        #todo: specify directory or move to desired directory
        project_dir_sofa = pjoin(primary_path, CN.PROJECT_FOLDER_BRIRS_SOFA)
        out_file_dir_sofa = pjoin(primary_path, CN.PROJECT_FOLDER_BRIRS_SOFA,brir_set_name)
        if not hf.check_write_permissions(project_dir_sofa, gui_logger):
            # Skip exporting or raise exception
            return 
        
        #create parent dir if doesnt exist
        output_file = Path(out_file_dir_sofa)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        #out_file_sofa = pjoin(out_file_dir_sofa,brir_set_name)
        
        # Save file upon completion
        #sofa.export(brir_set_name)
        sofa.export(output_file)
   
        log_string_a = 'Exported SOFA file: ' + brir_set_name 
        hf.log_with_timestamp(log_string_a, gui_logger)
  
    except Exception as ex:
        log_string = 'Failed to export SOFA file: ' + brir_set_name 
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
    


def export_sofa_ir(primary_path, ir_arr, ir_set_name, spatial_res=2, output_samples=CN.TOTAL_SAMPLES_HRIR, samp_freq=CN.SAMP_FREQ,
                   sofa_conv=None, gui_logger=None, is_brir=None):
    """
    Export either an HRIR or BRIR dataset to a SOFA file (auto-detect type by sample length).

    Parameters
    ----------
    primary_path : str
        Root project directory.
    ir_arr : np.ndarray
        4D array [elev, azim, ch, samples], HRIR or BRIR dataset.
    ir_set_name : str
        Name for output SOFA file.
    spatial_res : int
        Spatial resolution index (0=low ... 3=full).
    output_samples : int
        Target number of samples (at 44.1kHz base).
    samp_freq : int
        Sampling rate (Hz).
    sofa_conv : str, optional
        SOFA convention (default auto-selected).
    gui_logger : object, optional
        GUI logger instance.
    is_brir : bool, optional
        True for BRIR, False for HRIR. If None, determined automatically by length.
    """

    now_datetime = datetime.now()

    try:
        # --- Handle 5D input with singleton first dimension ---
        if ir_arr.ndim == 5 and ir_arr.shape[0] == 1:
            ir_arr = ir_arr[0]  # strip singleton first dimension
    
        # Ensure the array is now 4D
        if ir_arr.ndim != 4:
            raise ValueError(f"Expected 4D array [elev, azim, ch, samples], got shape {ir_arr.shape}")
    
        total_elev, total_azim, total_chan, total_samples = ir_arr.shape

        # --- 1. Auto-detect HRIR vs BRIR ---
        if is_brir is None:
            # Heuristic: HRIRs < ~2048 samples, BRIRs longer (contain room reverb)
            is_brir = total_samples >= 2048
            hf.log_with_timestamp(
                f"Auto-detected input type: {'BRIR' if is_brir else 'HRIR'} "
                f"(samples={total_samples})",
                gui_logger
            )

        # --- 2. Select SOFA convention ---
        if sofa_conv is None:
            sofa_conv = 'GeneralFIR' if is_brir else 'SimpleFreeFieldHRIR'

        # --- 3. Spatial Resolution ---
        #dynamic matrix structure, limits depend on spatial resolution
        if 0 <= spatial_res < CN.NUM_SPATIAL_RES:
            elev_min = CN.SPATIAL_RES_ELEV_MIN_IN[spatial_res]
            elev_max = CN.SPATIAL_RES_ELEV_MAX_IN[spatial_res]
            elev_step = CN.SPATIAL_RES_ELEV_NEAREST_IN[spatial_res]
            azim_step = CN.SPATIAL_RES_AZIM_NEAREST_IN[spatial_res]
        else:
            raise ValueError("Invalid spatial resolution index")

        # --- 4. Adjust output length ---
        if not is_brir:
            # HRIRs are short, use full length
            output_samples = total_samples
        elif output_samples > total_samples:
            # Clamp if BRIR shorter than target
            output_samples = max(total_samples - 1000, 4410)

        # --- 5. Resample according to sample rate ---
        output_samples_re = round(output_samples * float(samp_freq) / CN.SAMP_FREQ)

        hf.log_with_timestamp(
            f"Preparing SOFA export for '{ir_set_name}' ({'BRIR' if is_brir else 'HRIR'})",
            gui_logger
        )
        hf.log_with_timestamp(
            f"Resolution: {total_elev}×{total_azim}, Channels: {total_chan}, "
            f"Output samples: {output_samples_re}",
            gui_logger
        )

        total_directions = total_elev * total_azim
        source_pos_array = np.zeros((total_directions, 3))
        data_ir_array = np.zeros((total_directions, total_chan, output_samples_re))

        # --- 6. Fill IR data and position metadata ---
        radius = 2.0 if is_brir else 1.0
        src_idx = 0

        for elev in range(total_elev):
            elev_deg = int(elev_min + elev * elev_step)
            for azim in range(total_azim):
                azim_deg = int(azim * azim_step)
                source_pos_array[src_idx] = [azim_deg, elev_deg, radius]

                ir_pair = np.zeros((output_samples, total_chan))
                for ch in range(total_chan):
                    ir_pair[:, ch] = np.copy(ir_arr[elev, azim, ch, :output_samples])

                if samp_freq != CN.SAMP_FREQ:
                    ir_pair = hf.resample_signal(ir_pair, new_rate=samp_freq)
                    
                # --- FIX: ensure length consistency ---
                if ir_pair.shape[0] != output_samples_re:
                    if ir_pair.shape[0] > output_samples_re:
                        ir_pair = ir_pair[:output_samples_re, :]
                    else:
                        pad_len = output_samples_re - ir_pair.shape[0]
                        ir_pair = np.pad(ir_pair, ((0, pad_len), (0, 0)))

                # Transpose from (samples, channels) → (channels, samples) for SOFA format
                data_ir_array[src_idx, :, :output_samples_re] = ir_pair[:output_samples_re, :].T

                src_idx += 1

        # --- 7. Build SOFA File ---
        app_version = CN.__version__
        sofa = SOFAFile(sofa_conv, sofaConventionsVersion=1.0, version=1.0)
        sofa._M = total_directions
        sofa._N = total_samples
        
        # =============================== Attributes ====================================
        sofa.GLOBAL_ApplicationName = "ASH Toolset"
        sofa.GLOBAL_ApplicationVersion = app_version
        sofa.GLOBAL_AuthorContact = "srpearce55@gmail.com"
        #sofa.GLOBAL_Comment = "Generated by ASH Toolset (averaged HRTF export)"
        sofa.GLOBAL_History = "Averaged across multiple listener datasets"
        sofa.GLOBAL_License = "Creative Commons Attribution 4.0 International (CC-BY-4.0)"
        sofa.GLOBAL_Organization = "Independent Researcher"
        sofa.GLOBAL_References = "https://sourceforge.net/projects/ash-toolset/"
        #sofa.GLOBAL_RoomType = "Free field"
        sofa.GLOBAL_DateCreated = now_datetime
        sofa.GLOBAL_DateModified = now_datetime
        sofa.GLOBAL_Title = ir_set_name
        sofa.GLOBAL_DatabaseName = "ASH Toolset"
        sofa.GLOBAL_ListenerShortName = "Refer to title"
        sofa.ListenerPosition_Type = "cartesian"
        sofa.ListenerPosition_Units = "meter"
        sofa.ReceiverPosition_Type = "cartesian"
        sofa.ReceiverPosition_Units = "meter"
        sofa.SourcePosition_Type = "spherical"
        sofa.SourcePosition_Units = "degree, degree, meter"
        sofa.EmitterPosition_Type = "cartesian"
        sofa.EmitterPosition_Units = "meter"
        sofa.Data_SamplingRate_Units = "hertz"

        # --- 8. Metadata ---
        sofa.GLOBAL_Comment = f"Generated by ASH Toolset ({'BRIR' if is_brir else 'HRIR'})"
        sofa.GLOBAL_RoomType = "Reverberant" if is_brir else "Free field"
 

        sofa.ListenerPosition = np.atleast_2d([0, 0, 0])
        sofa.ReceiverPosition = np.atleast_3d([[0, 0.09, 0], [0, -0.09, 0]])
        sofa.SourcePosition = source_pos_array
        sofa.EmitterPosition = np.zeros((1, 3, 1))

        if sofa_conv == 'SimpleFreeFieldHRIR':
            sofa.ListenerUp = np.atleast_2d([0, 0, 1])
            sofa.ListenerView = np.atleast_2d([1, 0, 0])

        # --- 9. Audio Data ---
        sofa.Data_IR = data_ir_array
        sofa.Data_SamplingRate = [samp_freq]
        sofa.Data_Delay = np.zeros((1, 2))

        # --- 10. Export ---
        subfolder = CN.PROJECT_FOLDER_BRIRS_SOFA if is_brir else CN.PROJECT_FOLDER_HRIRS_SOFA
        export_dir = Path(pjoin(primary_path, subfolder))
        export_dir.mkdir(parents=True, exist_ok=True)

        output_file = pjoin(export_dir, ir_set_name)

        if not hf.check_write_permissions(export_dir, gui_logger):
            hf.log_with_timestamp("Write permission denied.", gui_logger)
            return

        sofa.export(output_file)
        hf.log_with_timestamp(f"Exported SOFA file: {output_file}", gui_logger)

    except Exception as ex:
        msg = f"Failed to export SOFA file: {ir_set_name}"
        hf.log_with_timestamp(msg, gui_logger, log_type=2, exception=ex)

