# -*- coding: utf-8 -*-


"""
Main routine of ASH-Tools.

Created on Sun Apr 28 11:25:06 2024
@author: Shanon Pearce
License: (see /LICENSE)
"""


import logging.config
from os.path import join as pjoin
from pathlib import Path
import configparser
from ash_toolset import constants as CN
from ash_toolset import hpcf_functions
from ash_toolset import helper_functions as hf
from ash_toolset import hrir_processing
from ash_toolset import callbacks as cb
from ash_toolset import e_apo_config_creation
import dearpygui.dearpygui as dpg
import dearpygui_extend as dpge
from dearpygui_ext import logger
import numpy as np
import json
import winreg as wrg
import ast
import ctypes
import math
import threading
import sys
import os
import shutil

def main():


    #logging
    logging.basicConfig(
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("log.log", mode="w"),
    ],
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logging.info('Started')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    
    #hide splash screen
    try:
        import pyi_splash
        pyi_splash.update_text('UI Loaded ...')
        pyi_splash.close()
        logging.info('Splash screen closed.')
    except:
        pass
    
    with open(CN.METADATA_FILE) as fp:
        _info = json.load(fp)
    __version__ = _info['version']
    
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    
    
    #
    #program code
    #
    
    # HpCF Database code
    database = pjoin(CN.DATA_DIR_OUTPUT,'hpcf_database.db')
    # create a database connection
    conn = hpcf_functions.create_connection(database)
    brands_list = hpcf_functions.get_brand_list(conn)
    brand_default=brands_list[0]
    hp_list_default = hpcf_functions.get_headphone_list(conn, brands_list[0])#index 0
    headphone_default = hp_list_default[0]
    sample_list_default = hpcf_functions.get_samples_list(conn, headphone_default)
    sample_default = 'Sample A' #sample_list_default[0]
    hpcf_db_dict =  {
    'database': database,
    'conn': conn,
    'brands_list': brands_list,
    'hp_list_default': hp_list_default,
    'sample_list_default': sample_list_default,
    'brand_default': brand_default,
    'headphone_default': headphone_default,
    'sample_default': sample_default
    }
    
 
    
    
    #start with flat response
    impulse=np.zeros(CN.N_FFT)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    
    #
    #Section for loading settings
    #
    
    def migrate_settings():
        """
        Migrate the settings file from the old application directory to the new user-specific config directory.
    
        This function checks if the new settings file already exists; if so, no action is taken.
        If the new settings file does not exist but the old settings file does, the old settings file
        is copied to the new location to preserve user configuration. After a successful copy,
        the old settings file is deleted to avoid confusion.
    
        This migration ensures that settings persist across application versions that changed
        the default storage location for the settings file.
    
        Note:
            - Assumes SETTINGS_FILE_NEW and SETTINGS_FILE_OLD are defined globally or imported.
            - Does not return any value.
        """
        try:
            if os.path.exists(CN.SETTINGS_FILE_NEW):
                return
    
            if os.path.exists(CN.SETTINGS_FILE_OLD):
                new_dir = os.path.dirname(CN.SETTINGS_FILE_NEW)
                os.makedirs(new_dir, exist_ok=True)
                logging.info(f"[INFO] Migrating settings from old to new path: {CN.SETTINGS_FILE_OLD} → {CN.SETTINGS_FILE_NEW}")
                shutil.copy2(CN.SETTINGS_FILE_OLD, CN.SETTINGS_FILE_NEW)
    
                # If copy successful, delete old settings file
                os.remove(CN.SETTINGS_FILE_OLD)
                logging.info(f"[INFO] Deleted old settings file: {CN.SETTINGS_FILE_OLD}")
    
        except Exception as e:
            logging.error(f"Failed to migrate settings file: {e}")

    
    def safe_get(config, key, expected_type, default): 
        """Safely get a value from config with fallback and type casting."""
        try:
            val = config['DEFAULT'].get(key, default)
    
            # Special handling for boolean (string to bool via literal_eval)
            if expected_type == bool:
                return ast.literal_eval(str(val))
            # Handle common literal types like list, dict, tuple, etc.
            elif expected_type == ast.literal_eval or expected_type in [list, dict, tuple]:
                return ast.literal_eval(val)
            # Safely cast if it's a callable (int, float, str, etc.)
            elif callable(expected_type):
                return expected_type(val)
            # Fallback if type is unknown or not callable
            else:
                logging.info(f"safe_get: Expected type for key '{key}' is not callable. Using raw value.")
                return val
    
        except Exception as e:
            logging.info(f"safe_get: Failed to load key '{key}' – {e}. Using default: {default}")
            return default
        
    def validate_choice(loaded_value, valid_list):
        """
        Ensure the loaded_value is present in valid_list.
        If not, return the first item in the list and log a fallback message.
    
        Args:
            loaded_value: The value to validate.
            valid_list (list): The list of valid options.
    
        Returns:
            A valid value from the list (either the loaded one or a fallback).
        """
        if loaded_value in valid_list:
            return loaded_value
        else:
            fallback = valid_list[0] if valid_list else None
            logging.info(f"validate_choice: '{loaded_value}' is not valid. Falling back to '{fallback}'.")
            return fallback
    
    #default values
    sample_freq_default=CN.SAMPLE_RATE_LIST[0]
    bit_depth_default=CN.BIT_DEPTH_LIST[0]
    brir_hp_type_default=CN.HP_COMP_LIST[2]
    show_hpcf_history_default=False
    spatial_res_default=CN.SPATIAL_RES_LIST[0]
    room_target_default=CN.ROOM_TARGET_LIST[1]
    direct_gain_default=2.0
    #acoustic space
    ac_space_default=CN.AC_SPACE_LIST_GUI[2]
    fir_hpcf_exp_default=True
    fir_st_hpcf_exp_default=False
    eapo_hpcf_exp_default=False
    geq_hpcf_exp_default=False
    geq_31_exp_default=False
    hesuvi_hpcf_exp_default=False
    dir_brir_exp_default=True
    ts_brir_exp_default=False
    hesuvi_brir_exp_default=False
    eapo_brir_exp_default=False
    sofa_brir_exp_default=False
    audio_channels_default=CN.AUDIO_CHANNELS[0]
    e_apo_upmix_method_default=CN.UPMIXING_METHODS[1]
    e_apo_side_delay_default=0
    e_apo_rear_delay_default=0
    auto_check_updates_default=False
    #E-APO config related settings
    e_apo_mute_default=False
    e_apo_gain_default=0.0
    e_apo_elev_angle_default=0
    e_apo_az_angle_fl_default=-30  
    e_apo_az_angle_fr_default=30  
    e_apo_az_angle_c_default=0  
    e_apo_az_angle_sl_default=-90  
    e_apo_az_angle_sr_default=90  
    e_apo_az_angle_rl_default=-135 
    e_apo_az_angle_rr_default=135 
    e_apo_enable_hpcf_default=False
    e_apo_enable_brir_default=False
    e_apo_autoapply_hpcf_default=False
    e_apo_prevent_clip_default=CN.AUTO_GAIN_METHODS[1]
    e_apo_hpcf_curr_default=''
    e_apo_hpcf_sel_default=''
    e_apo_brir_curr_default=''
    e_apo_brir_sel_default=''
    tab_selected_default=0
    hrtf_symmetry_default=CN.HRTF_SYM_LIST[0]
    er_rise_default=0
    qc_brir_hp_type_default=CN.HP_COMP_LIST[2]
    qc_room_target_default=CN.ROOM_TARGET_LIST[1]
    qc_direct_gain_default=2.0
    #acoustic space
    qc_ac_space_default=CN.AC_SPACE_LIST_GUI[1]
    qc_brand_default=brand_default
    qc_headphone_default=headphone_default
    qc_sample_default=sample_default
    #hrtf selection related
    brir_hrtf_type_list_default=CN.HRTF_TYPE_LIST
    brir_hrtf_type_default=CN.HRTF_TYPE_DEFAULT
    brir_hrtf_dataset_list_default=CN.HRTF_TYPE_DATASET_DICT.get(brir_hrtf_type_default)
    brir_hrtf_dataset_default=CN.HRTF_DATASET_DEFAULT
    hrtf_list_default=hrir_processing.get_listener_list(listener_type=brir_hrtf_type_default, dataset_name=brir_hrtf_dataset_default)
    hrtf_default=CN.HRTF_LISTENER_DEFAULT
    sofa_exp_conv_default=CN.SOFA_OUTPUT_CONV[0]
    crossover_f_mode_default=CN.SUB_FC_SETTING_LIST[0]
    crossover_f_default=CN.SUB_FC_DEFAULT
    sub_response_default = (CN.SUB_RESPONSE_LIST_GUI[3] if len(CN.SUB_RESPONSE_LIST_GUI) > 3 else CN.SUB_RESPONSE_LIST_GUI[0] if CN.SUB_RESPONSE_LIST_GUI else None)#use index 1 if available
    hp_rolloff_comp_default=False
    fb_filtering_default=False
    default_qc_brir_settings = {
    'qc_brir_hp_type_default': qc_brir_hp_type_default,
    'qc_room_target_default': qc_room_target_default,
    'qc_direct_gain_default': qc_direct_gain_default,
    'qc_ac_space_default': qc_ac_space_default,
    'qc_brand_default': qc_brand_default,
    'qc_headphone_default': qc_headphone_default,
    'qc_sample_default': qc_sample_default,
    'qc_hrtf_default': hrtf_default,
    'qc_brir_hrtf_type_default': brir_hrtf_type_default,
    'qc_brir_hrtf_dataset_default': brir_hrtf_dataset_default,
    'qc_crossover_f_mode_default': crossover_f_mode_default,
    'qc_crossover_f_default': crossover_f_default,
    'qc_sub_response_default': sub_response_default,
    'qc_hp_rolloff_comp_default': hp_rolloff_comp_default,
    'qc_fb_filtering_default': fb_filtering_default
    }#use in save settings function
    
    #loaded values - start with defaults
    sample_freq_loaded=sample_freq_default
    bit_depth_loaded=bit_depth_default
    brir_hp_type_loaded=brir_hp_type_default
    hrtf_loaded=hrtf_default
    spatial_res_loaded=spatial_res_default
    room_target_loaded=room_target_default
    direct_gain_loaded=direct_gain_default
    #acoustic space
    ac_space_loaded=ac_space_default
    fir_hpcf_exp_loaded=fir_hpcf_exp_default
    fir_st_hpcf_exp_loaded=fir_st_hpcf_exp_default
    eapo_hpcf_exp_loaded=eapo_hpcf_exp_default
    geq_hpcf_exp_loaded=geq_hpcf_exp_default
    geq_31_exp_loaded=geq_31_exp_default
    hesuvi_hpcf_exp_loaded=hesuvi_hpcf_exp_default
    dir_brir_exp_loaded=dir_brir_exp_default
    ts_brir_exp_loaded=ts_brir_exp_default
    hesuvi_brir_exp_loaded=hesuvi_brir_exp_default
    eapo_brir_exp_loaded=eapo_brir_exp_default
    sofa_brir_exp_loaded=sofa_brir_exp_default
    audio_channels_loaded=audio_channels_default
    auto_check_updates_loaded=auto_check_updates_default
    hrtf_symmetry_loaded=hrtf_symmetry_default
    er_rise_loaded=er_rise_default
    show_hpcf_history_loaded=show_hpcf_history_default
    #E-APO config related settings
    e_apo_upmix_method_loaded=e_apo_upmix_method_default
    e_apo_side_delay_loaded=e_apo_side_delay_default
    e_apo_rear_delay_loaded=e_apo_rear_delay_default
    e_apo_mute_fl_loaded=e_apo_mute_default
    e_apo_mute_fr_loaded=e_apo_mute_default
    e_apo_mute_c_loaded=e_apo_mute_default
    e_apo_mute_sl_loaded=e_apo_mute_default
    e_apo_mute_sr_loaded=e_apo_mute_default
    e_apo_mute_rl_loaded=e_apo_mute_default
    e_apo_mute_rr_loaded=e_apo_mute_default
    e_apo_gain_oa_loaded=e_apo_gain_default
    e_apo_gain_fl_loaded=e_apo_gain_default
    e_apo_gain_fr_loaded=e_apo_gain_default
    e_apo_gain_c_loaded=e_apo_gain_default
    e_apo_gain_sl_loaded=e_apo_gain_default
    e_apo_gain_sr_loaded=e_apo_gain_default
    e_apo_gain_rl_loaded=e_apo_gain_default
    e_apo_gain_rr_loaded=e_apo_gain_default
    e_apo_elev_angle_fl_loaded=e_apo_elev_angle_default
    e_apo_elev_angle_fr_loaded=e_apo_elev_angle_default
    e_apo_elev_angle_c_loaded=e_apo_elev_angle_default
    e_apo_elev_angle_sl_loaded=e_apo_elev_angle_default
    e_apo_elev_angle_sr_loaded=e_apo_elev_angle_default
    e_apo_elev_angle_rl_loaded=e_apo_elev_angle_default
    e_apo_elev_angle_rr_loaded=e_apo_elev_angle_default
    e_apo_az_angle_fl_loaded=e_apo_az_angle_fl_default  
    e_apo_az_angle_fr_loaded=e_apo_az_angle_fr_default  
    e_apo_az_angle_c_loaded=e_apo_az_angle_c_default 
    e_apo_az_angle_sl_loaded=e_apo_az_angle_sl_default
    e_apo_az_angle_sr_loaded=e_apo_az_angle_sr_default 
    e_apo_az_angle_rl_loaded=e_apo_az_angle_rl_default
    e_apo_az_angle_rr_loaded=e_apo_az_angle_rr_default
    e_apo_enable_hpcf_loaded=e_apo_enable_hpcf_default
    e_apo_enable_brir_loaded=e_apo_enable_brir_default
    e_apo_autoapply_hpcf_loaded=e_apo_autoapply_hpcf_default
    e_apo_prevent_clip_loaded=e_apo_prevent_clip_default
    e_apo_hpcf_curr_loaded = e_apo_hpcf_curr_default
    e_apo_hpcf_sel_loaded = e_apo_hpcf_sel_default
    e_apo_brir_curr_loaded = e_apo_brir_curr_default
    e_apo_brir_sel_loaded = e_apo_brir_sel_default
    e_apo_brir_sel_loaded = e_apo_brir_sel_default
    qc_brir_hp_type_loaded=qc_brir_hp_type_default
    qc_hrtf_loaded=hrtf_default
    qc_room_target_loaded=qc_room_target_default
    qc_direct_gain_loaded=qc_direct_gain_default
    qc_ac_space_loaded=qc_ac_space_default
    qc_brand_loaded=qc_brand_default
    qc_headphone_loaded=qc_headphone_default
    qc_sample_loaded=qc_sample_default
    #hrtf selection related
    brir_hrtf_type_loaded=brir_hrtf_type_default
    brir_hrtf_dataset_loaded=brir_hrtf_dataset_default
    qc_brir_hrtf_type_loaded=brir_hrtf_type_default
    qc_brir_hrtf_dataset_loaded=brir_hrtf_dataset_default
    sofa_exp_conv_loaded=sofa_exp_conv_default
    qc_crossover_f_mode_loaded=crossover_f_mode_default
    qc_crossover_f_loaded=crossover_f_default
    qc_sub_response_loaded=sub_response_default
    qc_hp_rolloff_comp_loaded=hp_rolloff_comp_default
    qc_fb_filtering_loaded=fb_filtering_default
    crossover_f_mode_loaded=crossover_f_mode_default
    crossover_f_loaded=crossover_f_default
    sub_response_loaded=sub_response_default
    hp_rolloff_comp_loaded=hp_rolloff_comp_default
    fb_filtering_loaded=fb_filtering_default
    
    #thread variables
    e_apo_conf_lock = threading.Lock()
    
    #get equalizer APO path
    try:
        key = wrg.OpenKey(wrg.HKEY_LOCAL_MACHINE, "Software\\EqualizerAPO")
        value =  wrg.QueryValueEx(key, "InstallPath")[0]
        e_apo_path= pjoin(value, 'config')
        #E APO was found in registry
    except:
        e_apo_path = None
    #set primary path to E APO path before attempting to load saved path
    if e_apo_path is not None:
        primary_path = e_apo_path
        primary_ash_path = pjoin(e_apo_path, CN.PROJECT_FOLDER)
        #print('found in registry')
    else:
        primary_path = 'C:\\Program Files\\EqualizerAPO\\config'
        primary_ash_path = 'C:\\Program Files\\EqualizerAPO\\config\\' + CN.PROJECT_FOLDER
    qc_primary_path=primary_path
    #try reading from settings.ini to get path
    try:
        migrate_settings()
        # then load from SETTINGS_FILE (which points to SETTINGS_FILE_NEW)
        #load settings
        config = configparser.ConfigParser()
        config.read(CN.SETTINGS_FILE)
        version_loaded = config['DEFAULT']['version']
        
        if version_loaded != __version__ and CN.LOG_INFO:
            logging.info(f"Settings version mismatch: file version={version_loaded}, expected={__version__}. Loading available settings anyway.")
            
        primary_path = safe_get(config, 'path', str, primary_path)
        primary_ash_path=pjoin(primary_path, CN.PROJECT_FOLDER)

        sample_freq_loaded = safe_get(config, 'sampling_frequency', str, sample_freq_default)
        bit_depth_loaded = safe_get(config, 'bit_depth', str, bit_depth_default)
        brir_hp_type_loaded = safe_get(config, 'brir_headphone_type', str, brir_hp_type_default)
        hrtf_loaded = safe_get(config, 'brir_hrtf', str, hrtf_default)
        spatial_res_loaded = safe_get(config, 'spatial_resolution', str, spatial_res_default)
        room_target_loaded = safe_get(config, 'brir_room_target', str, room_target_default)
        direct_gain_loaded = safe_get(config, 'brir_direct_gain', float, direct_gain_default)
        ac_space_loaded = safe_get(config, 'acoustic_space', str, ac_space_default)
        fir_hpcf_exp_loaded = safe_get(config, 'fir_hpcf_exp', ast.literal_eval, fir_hpcf_exp_default)
        fir_st_hpcf_exp_loaded = safe_get(config, 'fir_st_hpcf_exp', ast.literal_eval, fir_st_hpcf_exp_default)
        eapo_hpcf_exp_loaded = safe_get(config, 'eapo_hpcf_exp', ast.literal_eval, eapo_hpcf_exp_default)
        geq_hpcf_exp_loaded = safe_get(config, 'geq_hpcf_exp', ast.literal_eval, geq_hpcf_exp_default)
        geq_31_exp_loaded = safe_get(config, 'geq_31_exp', ast.literal_eval, geq_31_exp_default)
        hesuvi_hpcf_exp_loaded = safe_get(config, 'hesuvi_hpcf_exp', ast.literal_eval, hesuvi_hpcf_exp_default)
        dir_brir_exp_loaded = safe_get(config, 'dir_brir_exp', ast.literal_eval, dir_brir_exp_default)
        ts_brir_exp_loaded = safe_get(config, 'ts_brir_exp', ast.literal_eval, ts_brir_exp_default)
        hesuvi_brir_exp_loaded = safe_get(config, 'hesuvi_brir_exp', ast.literal_eval, hesuvi_brir_exp_default)
        eapo_brir_exp_loaded = safe_get(config, 'eapo_brir_exp', ast.literal_eval, eapo_brir_exp_default)
        sofa_brir_exp_loaded = safe_get(config, 'sofa_brir_exp', ast.literal_eval, sofa_brir_exp_default)
        auto_check_updates_loaded = safe_get(config, 'auto_check_updates', bool, auto_check_updates_default)
        hrtf_symmetry_loaded = safe_get(config, 'force_hrtf_symmetry', str, hrtf_symmetry_default)
        er_rise_loaded = safe_get(config, 'er_delay_time', float, er_rise_default)
        show_hpcf_history_loaded = safe_get(config, 'show_hpcf_history', bool, show_hpcf_history_default)
        e_apo_upmix_method_loaded = safe_get(config, 'upmix_method', str, e_apo_upmix_method_default)
        e_apo_side_delay_loaded = safe_get(config, 'side_delay', int, e_apo_side_delay_default)
        e_apo_rear_delay_loaded = safe_get(config, 'rear_delay', int, e_apo_rear_delay_default)
        e_apo_mute_fl_loaded = safe_get(config, 'mute_fl', bool, e_apo_mute_default)
        e_apo_mute_fr_loaded = safe_get(config, 'mute_fr', bool, e_apo_mute_default)
        e_apo_mute_c_loaded = safe_get(config, 'mute_c', bool, e_apo_mute_default)
        e_apo_mute_sl_loaded = safe_get(config, 'mute_sl', bool, e_apo_mute_default)
        e_apo_mute_sr_loaded = safe_get(config, 'mute_sr', bool, e_apo_mute_default)
        e_apo_mute_rl_loaded = safe_get(config, 'mute_rl', bool, e_apo_mute_default)
        e_apo_mute_rr_loaded = safe_get(config, 'mute_rr', bool, e_apo_mute_default)
        e_apo_gain_oa_loaded = safe_get(config, 'gain_oa', float, e_apo_gain_default)
        e_apo_gain_fl_loaded = safe_get(config, 'gain_fl', float, e_apo_gain_default)
        e_apo_gain_fr_loaded = safe_get(config, 'gain_fr', float, e_apo_gain_default)
        e_apo_gain_c_loaded = safe_get(config, 'gain_c', float, e_apo_gain_default)
        e_apo_gain_sl_loaded = safe_get(config, 'gain_sl', float, e_apo_gain_default)
        e_apo_gain_sr_loaded = safe_get(config, 'gain_sr', float, e_apo_gain_default)
        e_apo_gain_rl_loaded = safe_get(config, 'gain_rl', float, e_apo_gain_default)
        e_apo_gain_rr_loaded = safe_get(config, 'gain_rr', float, e_apo_gain_default)
        e_apo_elev_angle_fl_loaded = safe_get(config, 'elev_fl', int, e_apo_elev_angle_default)
        e_apo_elev_angle_fr_loaded = safe_get(config, 'elev_fr', int, e_apo_elev_angle_default)
        e_apo_elev_angle_c_loaded = safe_get(config, 'elev_c', int, e_apo_elev_angle_default)
        e_apo_elev_angle_sl_loaded = safe_get(config, 'elev_sl', int, e_apo_elev_angle_default)
        e_apo_elev_angle_sr_loaded = safe_get(config, 'elev_sr', int, e_apo_elev_angle_default)
        e_apo_elev_angle_rl_loaded = safe_get(config, 'elev_rl', int, e_apo_elev_angle_default)
        e_apo_elev_angle_rr_loaded = safe_get(config, 'elev_rr', int, e_apo_elev_angle_default)
        e_apo_az_angle_fl_loaded = safe_get(config, 'azim_fl', int, e_apo_az_angle_fl_default)
        e_apo_az_angle_fr_loaded = safe_get(config, 'azim_fr', int, e_apo_az_angle_fr_default)
        e_apo_az_angle_c_loaded = safe_get(config, 'azim_c', int, e_apo_az_angle_c_default)
        e_apo_az_angle_sl_loaded = safe_get(config, 'azim_sl', int, e_apo_az_angle_sl_default)
        e_apo_az_angle_sr_loaded = safe_get(config, 'azim_sr', int, e_apo_az_angle_sr_default)
        e_apo_az_angle_rl_loaded = safe_get(config, 'azim_rl', int, e_apo_az_angle_rl_default)
        e_apo_az_angle_rr_loaded = safe_get(config, 'azim_rr', int, e_apo_az_angle_rr_default)
        e_apo_enable_hpcf_loaded = safe_get(config, 'enable_hpcf', bool, e_apo_enable_hpcf_default)
        e_apo_enable_brir_loaded = safe_get(config, 'enable_brir', bool, e_apo_enable_brir_default)
        e_apo_autoapply_hpcf_loaded = safe_get(config, 'auto_apply_hpcf', bool, e_apo_autoapply_hpcf_default)
        e_apo_prevent_clip_loaded = safe_get(config, 'prevent_clip', str, e_apo_prevent_clip_default)
        e_apo_hpcf_curr_loaded = safe_get(config, 'hpcf_current', str, e_apo_hpcf_curr_default)
        e_apo_hpcf_sel_loaded = safe_get(config, 'hpcf_selected', str, e_apo_hpcf_sel_default)
        e_apo_brir_curr_loaded = safe_get(config, 'brir_set_current', str, e_apo_brir_curr_default)
        e_apo_brir_sel_loaded = safe_get(config, 'brir_set_selected', str, e_apo_brir_sel_default)
        audio_channels_loaded = safe_get(config, 'channel_config', str, audio_channels_default)
        tab_selected_loaded = safe_get(config, 'tab_selected', int, tab_selected_default)
        qc_brir_hp_type_loaded = safe_get(config, 'qc_brir_headphone_type', str, qc_brir_hp_type_default)
        qc_hrtf_loaded = safe_get(config, 'qc_brir_hrtf', str, hrtf_default)
        qc_room_target_loaded = safe_get(config, 'qc_brir_room_target', str, qc_room_target_default)
        qc_direct_gain_loaded = safe_get(config, 'qc_brir_direct_gain', float, qc_direct_gain_default)
        qc_ac_space_loaded = safe_get(config, 'qc_acoustic_space', str, qc_ac_space_default)
        qc_brand_loaded = safe_get(config, 'qc_brand', str, qc_brand_default)
        qc_headphone_loaded = safe_get(config, 'qc_headphone', str, qc_headphone_default)
        qc_sample_loaded = safe_get(config, 'qc_sample', str, qc_sample_default)
        brir_hrtf_type_loaded = safe_get(config, 'brir_hrtf_type', str, brir_hrtf_type_default)
        brir_hrtf_dataset_loaded = safe_get(config, 'brir_hrtf_dataset', str, brir_hrtf_dataset_default)
        qc_brir_hrtf_type_loaded = safe_get(config, 'qc_brir_hrtf_type', str, brir_hrtf_type_default)
        qc_brir_hrtf_dataset_loaded = safe_get(config, 'qc_brir_hrtf_dataset', str, brir_hrtf_dataset_default)
        sofa_exp_conv_loaded = safe_get(config, 'sofa_exp_conv', str, sofa_exp_conv_default)
        qc_crossover_f_mode_loaded = safe_get(config, 'qc_crossover_f_mode', str, crossover_f_mode_default)
        qc_crossover_f_loaded = safe_get(config, 'qc_crossover_f', int, crossover_f_default)
        qc_sub_response_loaded = safe_get(config, 'qc_sub_response', str, sub_response_default)
        qc_hp_rolloff_comp_loaded = safe_get(config, 'qc_hp_rolloff_comp', bool, hp_rolloff_comp_default)
        qc_fb_filtering_loaded = safe_get(config, 'qc_fb_filtering_mode', bool, fb_filtering_default)
        crossover_f_mode_loaded = safe_get(config, 'crossover_f_mode', str, crossover_f_mode_default)
        crossover_f_loaded = safe_get(config, 'crossover_f', int, crossover_f_default)
        sub_response_loaded = safe_get(config, 'sub_response', str, sub_response_default)
        hp_rolloff_comp_loaded = safe_get(config, 'hp_rolloff_comp', bool, hp_rolloff_comp_default)
        fb_filtering_loaded = safe_get(config, 'fb_filtering_mode', bool, fb_filtering_default)
        

        
    except Exception as e:
        logging.info(f"Error loading configuration: {e}")
        logging.info("Falling back to default values.")
        

    
    #set hesuvi path
    if 'EqualizerAPO' in primary_path:
        primary_hesuvi_path = pjoin(primary_path,'HeSuVi')#stored outside of project folder (within hesuvi installation)
    else:
        primary_hesuvi_path = pjoin(primary_path, CN.PROJECT_FOLDER,'HeSuVi')#stored within project folder

    #hp and sample lists based on loaded brand and headphone
    qc_hp_list_loaded = hpcf_functions.get_headphone_list(conn, qc_brand_loaded)#
    qc_sample_list_loaded = hpcf_functions.get_samples_list(conn, qc_headphone_loaded)
    

    #code to get gui width based on windows resolution
    gui_win_width_default=1722
    gui_win_height_default=717
    gui_win_width_loaded=gui_win_width_default
    gui_win_height_loaded=gui_win_height_default
    try:
        screen_width = ctypes.windll.user32.GetSystemMetrics(0)
        screen_height = ctypes.windll.user32.GetSystemMetrics(1)
        #print(str(screen_width))
        #print(str(screen_height))
        if screen_width < gui_win_width_default:
            gui_win_width_loaded=screen_width
        if screen_height < gui_win_height_default:
            gui_win_height_loaded=screen_height
    except:
        gui_win_width_loaded=gui_win_width_default
        gui_win_height_loaded=gui_win_height_default

            
    #adjust hrtf list based on loaded spatial resolution
    #also adjust file export selection
    dir_brir_exp_show=True
    ts_brir_exp_show=True
    hesuvi_brir_exp_show=True
    eapo_brir_exp_show=True
    sofa_brir_exp_show=True
    dir_brir_tooltip_show=True
    ts_brir_tooltip_show=True
    hesuvi_brir_tooltip_show=True
    eapo_brir_tooltip_show=True
    sofa_brir_tooltip_show=True
    brir_hrtf_dataset_list_loaded = CN.HRTF_TYPE_DATASET_DICT.get(brir_hrtf_type_loaded)
    if spatial_res_loaded == 'Max':#reduced list
        hrtf_list_loaded = hrir_processing.get_listener_list(listener_type=brir_hrtf_type_loaded, dataset_name=brir_hrtf_dataset_loaded, max_res_only=True)
    else:
        hrtf_list_loaded = hrir_processing.get_listener_list(listener_type=brir_hrtf_type_loaded, dataset_name=brir_hrtf_dataset_loaded)
    
    if spatial_res_loaded == 'Max':
        brir_hrtf_dataset_list_loaded = CN.HRTF_TYPE_DATASET_DICT.get('Dummy Head - Max Resolution')
        ts_brir_exp_loaded=False
        hesuvi_brir_exp_loaded=False
        eapo_brir_exp_loaded=False
        ts_brir_exp_show=False
        hesuvi_brir_exp_show=False
        eapo_brir_exp_show=False
        ts_brir_tooltip_show=False
        hesuvi_brir_tooltip_show=False
        eapo_brir_tooltip_show=False
    elif spatial_res_loaded == 'Medium' or spatial_res_loaded == 'Low':
        sofa_brir_exp_loaded=False
        sofa_brir_exp_show=False
        sofa_brir_tooltip_show=False
    
    spatial_res_list_loaded=CN.SPATIAL_RES_LIST_LIM
    if brir_hrtf_type_loaded == CN.HRTF_TYPE_LIST[0]:
        spatial_res_list_loaded=CN.SPATIAL_RES_LIST
    
    #qc dataset list based on loaded hrtf type
    qc_brir_hrtf_dataset_list_loaded = CN.HRTF_TYPE_DATASET_DICT.get(qc_brir_hrtf_type_loaded)
    #qc hrtf list based on dataset and hrtf type
    qc_hrtf_list_loaded = hrir_processing.get_listener_list(listener_type=qc_brir_hrtf_type_loaded, dataset_name=qc_brir_hrtf_dataset_loaded)


    #validate that loaded strings are within associated lists
    ac_space_loaded = validate_choice(ac_space_loaded, CN.AC_SPACE_LIST_GUI)
    qc_ac_space_loaded = validate_choice(qc_ac_space_loaded, CN.AC_SPACE_LIST_GUI)
    room_target_loaded = validate_choice(room_target_loaded, CN.ROOM_TARGET_LIST)
    qc_room_target_loaded = validate_choice(qc_room_target_loaded, CN.ROOM_TARGET_LIST)
    brir_hp_type_loaded = validate_choice(brir_hp_type_loaded, CN.HP_COMP_LIST)
    qc_brir_hp_type_loaded = validate_choice(qc_brir_hp_type_loaded, CN.HP_COMP_LIST)
    sub_response_loaded = validate_choice(sub_response_loaded, CN.SUB_RESPONSE_LIST_GUI)
    qc_sub_response_loaded = validate_choice(qc_sub_response_loaded, CN.SUB_RESPONSE_LIST_GUI)
    brir_hrtf_dataset_loaded = validate_choice(brir_hrtf_dataset_loaded, brir_hrtf_dataset_list_loaded)
    qc_brir_hrtf_dataset_loaded = validate_choice(qc_brir_hrtf_dataset_loaded, qc_brir_hrtf_dataset_list_loaded)
    hrtf_loaded = validate_choice(hrtf_loaded, hrtf_list_loaded)
    qc_hrtf_loaded = validate_choice(qc_hrtf_loaded, qc_hrtf_list_loaded)


    #
    # Equalizer APO related code
    #
    elevation_list_sel= CN.ELEV_ANGLES_WAV_LOW
  
    
    #
    ## GUI Functions 
    #

    #
    ## misc tools and settings
    #

            
    def reset_settings():
        """ 
        GUI function to reset settings
        """
        dpg.set_value("wav_sample_rate", sample_freq_default)
        dpg.set_value("wav_bit_depth", bit_depth_default)  
        dpg.set_value("brir_hp_type", brir_hp_type_default)
        dpg.set_value("qc_toggle_hpcf_history", show_hpcf_history_default)
        dpg.set_value("brir_spat_res", spatial_res_default)
        dpg.set_value("rm_target_list", room_target_default)
        dpg.set_value("direct_gain", direct_gain_default)
        dpg.set_value("direct_gain_slider", direct_gain_default)
        dpg.set_value("acoustic_space_combo", ac_space_default)
        dpg.set_value("crossover_f_mode", crossover_f_mode_default)
        dpg.set_value("crossover_f", crossover_f_default)
        dpg.set_value("sub_response", sub_response_default)
        dpg.set_value("hp_rolloff_comp", hp_rolloff_comp_default)
        dpg.set_value("fb_filtering", fb_filtering_default)
        
        dpg.configure_item('brand_list',items=brands_list)
        dpg.configure_item('headphone_list',items=hp_list_default)
        dpg.configure_item('sample_list',items=sample_list_default)
        dpg.set_value("brand_list", brand_default)
        dpg.set_value("headphone_list", headphone_default)
        dpg.set_value("sample_list", sample_default)
        dpg.configure_item('qc_brand_list',items=brands_list)
        dpg.configure_item('qc_headphone_list',items=hp_list_default)
        dpg.configure_item('qc_sample_list',items=sample_list_default)
        dpg.set_value("qc_brand_list", brand_default)
        dpg.set_value("qc_headphone_list", headphone_default)
        dpg.set_value("qc_sample_list", sample_default)
 
        dpg.set_value("qc_wav_sample_rate", sample_freq_default)
        dpg.set_value("qc_wav_bit_depth", bit_depth_default)
        dpg.set_value("qc_brir_hp_type", brir_hp_type_default)
        
        dpg.set_value("qc_rm_target_list", room_target_default)
        dpg.set_value("qc_direct_gain", direct_gain_default)
        dpg.set_value("qc_direct_gain_slider", direct_gain_default)
        dpg.set_value("qc_acoustic_space_combo", ac_space_default)
        dpg.set_value("qc_crossover_f_mode", crossover_f_mode_default)
        dpg.set_value("qc_crossover_f", crossover_f_default)
        dpg.set_value("qc_sub_response", sub_response_default)
        dpg.set_value("qc_hp_rolloff_comp", hp_rolloff_comp_default)
        dpg.set_value("qc_fb_filtering", fb_filtering_default)
        
        dpg.set_value("fir_hpcf_toggle", fir_hpcf_exp_default)
        dpg.set_value("fir_st_hpcf_toggle", fir_st_hpcf_exp_default)
        dpg.set_value("eapo_hpcf_toggle", eapo_hpcf_exp_default)
        dpg.set_value("geq_hpcf_toggle", geq_hpcf_exp_default)
        dpg.set_value("geq_31_hpcf_toggle", geq_31_exp_default)
        dpg.set_value("hesuvi_hpcf_toggle", hesuvi_hpcf_exp_default)
        dpg.set_value("dir_brir_toggle", dir_brir_exp_default)
        dpg.set_value("ts_brir_toggle", ts_brir_exp_default)
        dpg.set_value("hesuvi_brir_toggle", hesuvi_brir_exp_default)
        dpg.set_value("eapo_brir_toggle", eapo_brir_exp_default)
        dpg.set_value("sofa_brir_toggle", sofa_brir_exp_default)
        dpg.set_value("force_hrtf_symmetry", hrtf_symmetry_default)
        dpg.set_value("er_delay_time_tag", er_rise_default)
        dpg.set_value("e_apo_brir_conv", e_apo_enable_brir_default)
        dpg.set_value("e_apo_hpcf_conv", e_apo_enable_hpcf_default)
        dpg.set_value("e_apo_prevent_clip", e_apo_prevent_clip_default)
        #hrtf selection related
        dpg.configure_item('brir_hrtf',items=hrtf_list_default)
        dpg.set_value("brir_hrtf", hrtf_default)
        dpg.configure_item('qc_brir_hrtf',items=hrtf_list_default)
        dpg.set_value("qc_brir_hrtf", hrtf_default)
        dpg.set_value("brir_hrtf_type", brir_hrtf_type_default)
        dpg.configure_item('brir_hrtf_dataset',items=brir_hrtf_dataset_list_default)
        dpg.set_value("brir_hrtf_dataset", brir_hrtf_dataset_default)
        dpg.set_value("qc_brir_hrtf_type", brir_hrtf_type_default)
        dpg.configure_item('qc_brir_hrtf_dataset',items=brir_hrtf_dataset_list_default)
        dpg.set_value("qc_brir_hrtf_dataset", brir_hrtf_dataset_default)
        dpg.set_value("sofa_exp_conv", sofa_exp_conv_default )
        #reset progress bars
        cb.reset_hpcf_progress()
        cb.reset_brir_progress()
        cb.qc_reset_progress()
        cb.e_apo_toggle_hpcf_gui(app_data=False)
        cb.e_apo_toggle_brir_gui(app_data=False)
        
        #reset output directory
        if e_apo_path is not None:
            primary_path = e_apo_path
            primary_ash_path = pjoin(e_apo_path, CN.PROJECT_FOLDER)
        else:
            primary_path = 'C:\\Program Files\\EqualizerAPO\\config'
            primary_ash_path = 'C:\\Program Files\\EqualizerAPO\\config\\' + CN.PROJECT_FOLDER
        dpg.set_value('selected_folder_base', primary_path)
        dpg.set_value('selected_folder_ash', primary_ash_path)
        dpg.set_value('selected_folder_ash_tooltip', primary_ash_path)
        #hesuvi path
        if 'EqualizerAPO' in primary_path:
            hesuvi_path_selected = pjoin(primary_path,'HeSuVi')#stored outside of project folder (within hesuvi installation)
        else:
            hesuvi_path_selected = pjoin(primary_path, CN.PROJECT_FOLDER,'HeSuVi')#stored within project folder
        dpg.set_value('selected_folder_hesuvi', hesuvi_path_selected)
        dpg.set_value('selected_folder_hesuvi_tooltip', hesuvi_path_selected)

        reset_channel_config()

        cb.save_settings()
        
        #log results
        log_string = 'Settings have been reset to default '
        hf.log_with_timestamp(log_string, logz)
       




    def reset_channel_config(sender=None, app_data=None):
        """ 
        GUI function to reset channel config in E-APO config section
        """
        dpg.set_value("e_apo_upmix_method", e_apo_upmix_method_default)
        dpg.set_value("e_apo_side_delay", e_apo_side_delay_default)
        dpg.set_value("e_apo_rear_delay", e_apo_rear_delay_default)
        dpg.set_value("e_apo_mute_fl", e_apo_mute_default)
        dpg.set_value("e_apo_mute_fr", e_apo_mute_default)
        dpg.set_value("e_apo_mute_c", e_apo_mute_default)
        dpg.set_value("e_apo_mute_sl", e_apo_mute_default)
        dpg.set_value("e_apo_mute_sr", e_apo_mute_default)
        dpg.set_value("e_apo_mute_rl", e_apo_mute_default)
        dpg.set_value("e_apo_mute_rr", e_apo_mute_default)
        dpg.set_value("e_apo_gain_oa", e_apo_gain_default)
        dpg.set_value("e_apo_gain_fl", e_apo_gain_default)
        dpg.set_value("e_apo_gain_fr", e_apo_gain_default)
        dpg.set_value("e_apo_gain_c", e_apo_gain_default)
        dpg.set_value("e_apo_gain_sl", e_apo_gain_default)
        dpg.set_value("e_apo_gain_sr", e_apo_gain_default)
        dpg.set_value("e_apo_gain_rl", e_apo_gain_default)
        dpg.set_value("e_apo_gain_rr", e_apo_gain_default)
        dpg.set_value("e_apo_elev_angle_fl", e_apo_elev_angle_default)
        dpg.set_value("e_apo_elev_angle_fr", e_apo_elev_angle_default)
        dpg.set_value("e_apo_elev_angle_c", e_apo_elev_angle_default)
        dpg.set_value("e_apo_elev_angle_sl", e_apo_elev_angle_default)
        dpg.set_value("e_apo_elev_angle_sr", e_apo_elev_angle_default)
        dpg.set_value("e_apo_elev_angle_rl", e_apo_elev_angle_default)
        dpg.set_value("e_apo_elev_angle_rr", e_apo_elev_angle_default)
        dpg.set_value("e_apo_az_angle_fl", e_apo_az_angle_fl_default)
        dpg.set_value("e_apo_az_angle_fr", e_apo_az_angle_fr_default)
        dpg.set_value("e_apo_az_angle_c", e_apo_az_angle_c_default)
        dpg.set_value("e_apo_az_angle_sl", e_apo_az_angle_sl_default)
        dpg.set_value("e_apo_az_angle_sr", e_apo_az_angle_sr_default)
        dpg.set_value("e_apo_az_angle_rl", e_apo_az_angle_rl_default)
        dpg.set_value("e_apo_az_angle_rr", e_apo_az_angle_rr_default)
        dpg.set_value("hesuvi_elev_angle_fl", e_apo_elev_angle_default)
        dpg.set_value("hesuvi_elev_angle_fr", e_apo_elev_angle_default)
        dpg.set_value("hesuvi_elev_angle_c", e_apo_elev_angle_default)
        dpg.set_value("hesuvi_elev_angle_sl", e_apo_elev_angle_default)
        dpg.set_value("hesuvi_elev_angle_sr", e_apo_elev_angle_default)
        dpg.set_value("hesuvi_elev_angle_rl", e_apo_elev_angle_default)
        dpg.set_value("hesuvi_elev_angle_rr", e_apo_elev_angle_default)
        dpg.set_value("hesuvi_az_angle_fl", e_apo_az_angle_fl_default)
        dpg.set_value("hesuvi_az_angle_fr", e_apo_az_angle_fr_default)
        dpg.set_value("hesuvi_az_angle_c", e_apo_az_angle_c_default)
        dpg.set_value("hesuvi_az_angle_sl", e_apo_az_angle_sl_default)
        dpg.set_value("hesuvi_az_angle_sr", e_apo_az_angle_sr_default)
        dpg.set_value("hesuvi_az_angle_rl", e_apo_az_angle_rl_default)
        dpg.set_value("hesuvi_az_angle_rr", e_apo_az_angle_rr_default)
        
        dpg.set_value("e_apo_prevent_clip", e_apo_prevent_clip_default)
        #also reset channel config
        dpg.set_value("audio_channels_combo", audio_channels_default)
        cb.e_apo_select_channels(app_data=audio_channels_default,aquire_config=False)
        #update drawings for azimuths
        cb.e_apo_activate_direction()
        
        #finally rewrite config file
        cb.e_apo_config_acquire()
  

    def _hsv_to_rgb(h, s, v):
        """ 
        GUI function to convert hsv to rgb. HSV stands for hue, saturation, and value
        """
        if s == 0.0: return (v, v, v)
        i = int(h*6.) # XXX assume int() truncates!
        f = (h*6.)-i; p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
        if i == 0: return (255*v, 255*t, 255*p)
        if i == 1: return (255*q, 255*v, 255*p)
        if i == 2: return (255*p, 255*v, 255*t)
        if i == 3: return (255*p, 255*q, 255*v)
        if i == 4: return (255*t, 255*p, 255*v)
        if i == 5: return (255*v, 255*p, 255*q)
               
       

     
    def intialise_gui():
        """ 
        GUI function to perform further initial configuration of gui elements
        """
        
        # acoustic space manager: Add a logger object inside the child window
        if CN.SHOW_AS_TAB:
            import_console_log = logger.mvLogger(parent="import_console_window")
            dpg.configure_item("import_console_window", user_data=import_console_log)
            cb.update_ir_folder_list()
            cb.update_as_table_from_csvs()
        # #room target manager
        cb.update_room_target_list()
        
        #check for updates on start
        if auto_check_updates_loaded == True:
            #start thread
            thread = threading.Thread(target=cb.check_all_updates, args=(), daemon=True)
            thread.start()


        #inital configuration
        #update channel gui elements on load, will also write e-APO config again
        cb.e_apo_select_channels(app_data=dpg.get_value('audio_channels_combo'),aquire_config=False)
        #adjust active tab
        try:
            dpg.set_value("tab_bar", tab_selected_loaded)
        except Exception:
            pass
        #cb.save_settings()#not needed
        hpcf_is_active=dpg.get_value('e_apo_hpcf_conv')
        brir_is_active=dpg.get_value('e_apo_brir_conv')
        #show hpcf history
        cb.qc_show_hpcf_history(app_data=dpg.get_value('qc_toggle_hpcf_history'))

        cb.e_apo_toggle_hpcf_custom(app_data=hpcf_is_active, aquire_config=False)
        cb.e_apo_toggle_brir_custom(app_data=brir_is_active, aquire_config=False)
        #finally acquire config once
        cb.e_apo_config_acquire()
        

    
   
    
    
    #
    ## GUI CODE
    #
    
    #default plotting data
    default_x = []
    default_y = []
    for i in range(0, 500):
        default_x.append(i)
        default_y.append(i)
 
    dpg.create_context()
    
    image_location_fl = pjoin(CN.DATA_DIR_RAW, 'l_icon.png')
    image_location_fr = pjoin(CN.DATA_DIR_RAW, 'r_icon.png')
    image_location_c = pjoin(CN.DATA_DIR_RAW, 'c_icon.png')
    image_location_sl = pjoin(CN.DATA_DIR_RAW, 'sl_icon.png')
    image_location_sr = pjoin(CN.DATA_DIR_RAW, 'sr_icon.png')
    image_location_rl = pjoin(CN.DATA_DIR_RAW, 'rl_icon.png')
    image_location_rr = pjoin(CN.DATA_DIR_RAW, 'rr_icon.png')
    image_location_listener = pjoin(CN.DATA_DIR_RAW, 'listener_icon.png')
    width_fl, height_fl, channels_fl, data_fl = dpg.load_image(image_location_fl)
    width_fr, height_fr, channels_fr, data_fr = dpg.load_image(image_location_fr)
    width_c, height_c, channels_c, data_c = dpg.load_image(image_location_c)
    width_sl, height_sl, channels_sl, data_sl = dpg.load_image(image_location_sl)
    width_sr, height_sr, channels_sr, data_sr = dpg.load_image(image_location_sr)
    width_rl, height_rl, channels_rl, data_rl = dpg.load_image(image_location_rl)
    width_rr, height_rr, channels_rr, data_rr = dpg.load_image(image_location_rr)
    width_listener, height_listener, channels_listener, data_listener = dpg.load_image(image_location_listener)

    with dpg.texture_registry(show=False):
        dpg.add_static_texture(width=width_fl, height=height_fl, default_value=data_fl, tag='fl_image')
        dpg.add_static_texture(width=width_fr, height=height_fr, default_value=data_fr, tag='fr_image')
        dpg.add_static_texture(width=width_c, height=height_c, default_value=data_c, tag='c_image')
        dpg.add_static_texture(width=width_sl, height=height_sl, default_value=data_sl, tag='sl_image')
        dpg.add_static_texture(width=width_sr, height=height_sr, default_value=data_sr, tag='sr_image')
        dpg.add_static_texture(width=width_rl, height=height_rl, default_value=data_rl, tag='rl_image')
        dpg.add_static_texture(width=width_rr, height=height_rr, default_value=data_rr, tag='rr_image')
        dpg.add_static_texture(width=width_listener, height=height_listener, default_value=data_listener, tag='listener_image')

    # add a font registry
    with dpg.font_registry():
        # first argument ids the path to the .ttf or .otf file
        # in_file_path = pjoin(CN.DATA_DIR_EXT,'font', 'Lato-Medium.ttf')#SourceSansPro-Regular
        # default_font = dpg.add_font(in_file_path, 14)    
        # in_file_path = pjoin(CN.DATA_DIR_EXT,'font', 'Lato-Bold.ttf')#SourceSansPro-Regular
        # bold_font = dpg.add_font(in_file_path, 14)
        # in_file_path = pjoin(CN.DATA_DIR_EXT,'font', 'Lato-Bold.ttf')#SourceSansPro-Regular
        # bold_small_font = dpg.add_font(in_file_path, 13) 
        # in_file_path = pjoin(CN.DATA_DIR_EXT,'font', 'Lato-Bold.ttf')#SourceSansPro-Regular
        # large_font = dpg.add_font(in_file_path, 16)    
        # in_file_path = pjoin(CN.DATA_DIR_EXT,'font', 'Lato-Medium.ttf')#SourceSansPro-Regular
        # small_font = dpg.add_font(in_file_path, 13)
        
        in_file_path = pjoin(CN.DATA_DIR_EXT,'font', 'Roboto-Medium.ttf')#
        default_font = dpg.add_font(in_file_path, 14)    
        in_file_path = pjoin(CN.DATA_DIR_EXT,'font', 'Roboto-Bold.ttf')#
        bold_font = dpg.add_font(in_file_path, 14)
        in_file_path = pjoin(CN.DATA_DIR_EXT,'font', 'Roboto-Bold.ttf')#
        bold_small_font = dpg.add_font(in_file_path, 13) 
        in_file_path = pjoin(CN.DATA_DIR_EXT,'font', 'Roboto-Bold.ttf')#
        large_font = dpg.add_font(in_file_path, 16)    
        in_file_path = pjoin(CN.DATA_DIR_EXT,'font', 'Roboto-Medium.ttf')#
        small_font = dpg.add_font(in_file_path, 13)
        
        

    dpg.create_viewport(title='Audio Spatialisation for Headphones', width=gui_win_width_loaded, height=gui_win_height_loaded, small_icon=CN.ICON_LOCATION, large_icon=CN.ICON_LOCATION)
    
    with dpg.window(tag="Primary Window", horizontal_scrollbar=True):
        
        # set font of app
        dpg.bind_font(default_font)
        
        # Themes
        with dpg.theme(tag="__theme_a"):
            i=4.2#i=3
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(i/7.0, 0.3, 0.5))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(i/7.0, 0.8, 0.8))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(i/7.0, 0.6, 0.7))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, i*5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 7, 7)
        with dpg.theme(tag="__theme_b"):
            i=3.8#i=2
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(i/7.0, 0.3, 0.5))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(i/7.0, 0.8, 0.8))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(i/7.0, 0.6, 0.7))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, i*5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 7, 7)
        with dpg.theme(tag="__theme_c"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Header, (138, 138, 62), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (135, 163, 78), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (138, 138, 62), category=dpg.mvThemeCat_Core)
        with dpg.theme(tag="__theme_f"):
            i=3.9
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, _hsv_to_rgb(i/7.0, 0.05, 0.99))
        with dpg.theme(tag="__theme_g"):
            i=3.5
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, _hsv_to_rgb(i/7.0, 0.2, 0.55)) 
        with dpg.theme(tag="__theme_h"):
            i=3.9
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, _hsv_to_rgb(i/7.0, 0.2, 0.55)) 
  
        with dpg.theme(tag="__theme_i"):
            i=3.6
            j=3.6
            k=3.4
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, _hsv_to_rgb(i/7.0, 0.3, 0.5))   
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, _hsv_to_rgb(i/7.0, 0.3, 0.5)) 
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, _hsv_to_rgb(i/7.0, 0.5, 0.8)) 
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, _hsv_to_rgb(i/7.0, 0.5, 0.7)) 
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, _hsv_to_rgb(i/7.0, 0.5, 0.7))
                dpg.add_theme_color(dpg.mvPlotCol_TitleText, _hsv_to_rgb(i/7.0, 0.5, 0.7))
                dpg.add_theme_color(dpg.mvThemeCol_Header, _hsv_to_rgb(4.4/7.0, 0.1, 0.25))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, _hsv_to_rgb(j/7.0, 0.3, 0.5)) 
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, _hsv_to_rgb(j/7.0, 0.3, 0.5)) 
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, _hsv_to_rgb(j/7.0, 0.3, 0.5)) 
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(j/7.0, 0.3, 0.5)) 
                dpg.add_theme_color(dpg.mvPlotCol_Line, _hsv_to_rgb(k/7.0, 0.25, 0.6), category=dpg.mvThemeCat_Plots) 
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, _hsv_to_rgb(j/7.0, 0.7, 0.7))
        with dpg.theme(tag="__theme_j"):
            i=4.2
            j=4.2
            k=0.5
            with dpg.theme_component(dpg.mvAll):  
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, _hsv_to_rgb(i/7.0, 0.4*k, 0.5))   
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, _hsv_to_rgb(i/7.0, 0.3*k, 0.5)) 
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, _hsv_to_rgb(i/7.0, 0.5*k, 0.8)) 
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, _hsv_to_rgb(i/7.0, 0.5*k, 0.7)) 
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, _hsv_to_rgb(i/7.0, 0.5*k, 0.7))
                dpg.add_theme_color(dpg.mvPlotCol_TitleText, _hsv_to_rgb(i/7.0, 0.5*k, 0.7))
                dpg.add_theme_color(dpg.mvThemeCol_Header, _hsv_to_rgb(4.4/7.0, 0.1*k, 0.25))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, _hsv_to_rgb(j/7.0, 0.3*k, 0.5)) 
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, _hsv_to_rgb(j/7.0, 0.3*k, 0.5)) 
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, _hsv_to_rgb(j/7.0, 0.3*k, 0.5)) 
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(j/7.0, 0.3*k, 0.5)) 
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, _hsv_to_rgb(j/7.0, 0.7*k, 0.7))
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, _hsv_to_rgb(i/7.0, 0.2, 0.55)) 
        
        with dpg.theme() as modern_theme:
            i=3.9
            j=3.9
            k=3.8
            with dpg.theme_component(dpg.mvAll) as global_style:

                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (32, 32, 32), category=dpg.mvThemeCat_Core)  # Neutral dark gray
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (44, 44, 44), category=dpg.mvThemeCat_Core)   # Slightly lighter neutral dark gray
                dpg.add_theme_color(dpg.mvThemeCol_Border, (60, 60, 60), category=dpg.mvThemeCat_Core)
    
                dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 60, 60), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 40, 40), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 220, 220), category=dpg.mvThemeCat_Core)
                
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 4, 3, category=dpg.mvThemeCat_Core)#8, 6,
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 4, 4, category=dpg.mvThemeCat_Core)# 8, 8,
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4, category=dpg.mvThemeCat_Core)
                
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, _hsv_to_rgb(i/7.0, 0.4, 0.5))   
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, _hsv_to_rgb(i/7.0, 0.3, 0.5)) 
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, _hsv_to_rgb(i/7.0, 0.5, 0.8)) 
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, _hsv_to_rgb(i/7.0, 0.5, 0.7)) 
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, _hsv_to_rgb(i/7.0, 0.5, 0.7))
                dpg.add_theme_color(dpg.mvPlotCol_TitleText, _hsv_to_rgb(i/7.0, 0.5, 0.7))
                dpg.add_theme_color(dpg.mvThemeCol_Header, _hsv_to_rgb(4.4/7.0, 0.1, 0.25))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, _hsv_to_rgb(j/7.0, 0.3, 0.5)) 
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, _hsv_to_rgb(j/7.0, 0.3, 0.5)) 
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, _hsv_to_rgb(j/7.0, 0.3, 0.5)) 
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(j/7.0, 0.3, 0.5)) 
                dpg.add_theme_color(dpg.mvPlotCol_Line, _hsv_to_rgb(k/7.0, 0.25, 0.6), category=dpg.mvThemeCat_Plots) 
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, _hsv_to_rgb(j/7.0, 0.7, 0.7))

        
        dpg.bind_theme(modern_theme)#global_theme
        
        with dpg.tab_bar(tag='tab_bar'):

            with dpg.tab(label="Quick Configuration",tag='quick_config', parent="tab_bar"): 
                dpg.add_text("Apply Headphone Correction & Binaural Room Simulation in Equalizer APO")
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    #Section for HpCF Export
                    with dpg.child_window(width=550, height=610):
                        title_1_qc = dpg.add_text("Headphone Correction")
                        dpg.bind_item_font(title_1_qc, bold_font)
                        with dpg.child_window(autosize_x=True, height=390):
                            

                            
                            with dpg.group(horizontal=True):
                                with dpg.group():
                                    subtitle_1_qc = dpg.add_text("Select Headphone & Sample", tag='qc_hpcf_title')
                                    with dpg.tooltip("qc_hpcf_title"):
                                        dpg.add_text("Select a headphone from below list")
                                    dpg.bind_item_font(subtitle_1_qc, bold_font)
                                dpg.add_text("                                                      ")
                                dpg.add_checkbox(label="Show History", default_value = show_hpcf_history_loaded,  tag='qc_toggle_hpcf_history', callback=cb.qc_show_hpcf_history)
                                with dpg.tooltip("qc_toggle_hpcf_history"):
                                    dpg.add_text("Shows previously applied headphones")
                                dpg.add_text("     ")
                                
                                #dpg.add_button(label="Clear History",user_data="",tag="qc_clear_history", callback=cb.remove_hpcfs)
                                # Button to trigger the popup
                                dpg.add_button(label="Clear History", user_data="", tag="qc_clear_history_button",
                                               callback=lambda: dpg.configure_item("qc_clear_history_popup", show=True))
                                
                                # Optional tooltip
                                with dpg.tooltip("qc_clear_history_button"):
                                    dpg.add_text("Clear headphone correction history")
                                
                                # Confirmation popup
                                with dpg.popup("qc_clear_history_button", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="qc_clear_history_popup"):
                                    dpg.add_text("This will clear headphone correction history")
                                    dpg.add_separator()
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="OK", width=75, callback=cb.remove_hpcfs)
                                        dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("qc_clear_history_popup", show=False))
                                
                                
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_text("Search Brand:")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_input_text(label="", callback=cb.qc_filter_brand_list, width=105)
                                dpg.add_text("Search Headphone:")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_input_text(label="", callback=cb.qc_filter_headphone_list, width=209)
                            with dpg.group(horizontal=True, width=0):
                                with dpg.group():
                                    dpg.add_text("Brand")
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_listbox(brands_list, width=135, num_items=16, tag='qc_brand_list', default_value=qc_brand_loaded, callback=cb.qc_update_headphone_list)
                                with dpg.group():
                                    dpg.add_text("Headphone")
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_listbox(qc_hp_list_loaded, width=250, num_items=16, tag='qc_headphone_list', default_value=qc_headphone_loaded ,callback=cb.qc_update_sample_list)
                                with dpg.group():
                                    dpg.add_text("Sample")
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_listbox(qc_sample_list_loaded, width=115, num_items=16, default_value=qc_sample_loaded, tag='qc_sample_list', callback=cb.qc_plot_sample) 
                        
                        with dpg.child_window(autosize_x=True, height=176):
                            with dpg.group(horizontal=True):
                                subtitle_3_qc = dpg.add_text("Apply Headphone Correction in Equalizer APO")
                                dpg.bind_item_font(subtitle_3_qc, bold_font)
                                dpg.add_text("                                    ")
                                dpg.add_checkbox(label="Auto Apply Selection", default_value = e_apo_autoapply_hpcf_loaded,  tag='qc_auto_apply_hpcf_sel', callback=cb.e_apo_auto_apply_hpcf)
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_button(label=CN.PROCESS_BUTTON_HPCF,user_data="",tag="qc_hpcf_tag", callback=cb.qc_apply_hpcf_params, width=145)
                                dpg.bind_item_theme(dpg.last_item(), "__theme_a")
                                dpg.bind_item_font(dpg.last_item(), large_font)
                                with dpg.tooltip("qc_hpcf_tag"):
                                    dpg.add_text("This will apply the selected filter in Equalizer APO")
                                dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=30, width=356, overlay=CN.PROGRESS_START, tag="qc_progress_bar_hpcf")
                                dpg.bind_item_theme(dpg.last_item(), "__theme_h")
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="Enable Headphone Correction", default_value = e_apo_enable_hpcf_loaded,  tag='e_apo_hpcf_conv', callback=cb.e_apo_toggle_hpcf_gui, user_data=e_apo_conf_lock )
                                #dpg.add_text("                                    ")
                                
                            dpg.add_separator()
                            dpg.add_text("Current Filter: ")
                            dpg.bind_item_font(dpg.last_item(), bold_font)
                            dpg.add_text(default_value=e_apo_hpcf_curr_loaded,  tag='qc_e_apo_curr_hpcf')
                            dpg.bind_item_font(dpg.last_item(), bold_font)
                            dpg.bind_item_theme(dpg.last_item(), "__theme_f")
                            dpg.add_text(default_value=e_apo_hpcf_sel_loaded, tag='qc_e_apo_sel_hpcf',show=False, user_data=hpcf_db_dict)
                            
                    #Section for BRIR generation
                    with dpg.child_window(width=534, height=610):
                        title_2_qc = dpg.add_text("Binaural Room Simulation", tag='qc_brir_title',user_data=default_qc_brir_settings )
                        dpg.bind_item_font(title_2_qc, bold_font)
                        with dpg.tooltip("qc_brir_title"):
                            dpg.add_text("Customise a new binaural room simulation using below parameters")
                        with dpg.child_window(autosize_x=True, height=390):
                            with dpg.tab_bar(tag='qc_brir_tab_bar'):

                                with dpg.tab(label="Acoustics & EQ Parameters",tag='qc_acoustics_eq_tab', parent="qc_brir_tab_bar"): 
                                    #dpg.add_separator()
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Acoustic Space",tag='qc_acoustic_space_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            with dpg.group(horizontal=True):
                                                dpg.add_text("Sort by: ")
                                                dpg.add_combo(CN.AC_SPACE_LIST_SORT_BY, width=140, label="",  tag='qc_sort_by_as',default_value=CN.AC_SPACE_LIST_SORT_BY[0], callback=cb.qc_sort_ac_space)
                                            
                                            dpg.add_listbox(CN.AC_SPACE_LIST_GUI, label="",tag='qc_acoustic_space_combo',default_value=qc_ac_space_loaded, callback=cb.qc_update_ac_space, num_items=16, width=255)
                                            with dpg.tooltip("qc_acoustic_space_title"):
                                                dpg.add_text("This will determine the listening environment")
                                        with dpg.group():
                                            dpg.add_text("Direct Sound Gain (dB)", tag='qc_direct_gain_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_input_float(label="Direct Gain (dB)",width=140, format="%.1f", tag='qc_direct_gain', min_value=CN.DIRECT_GAIN_MIN, max_value=CN.DIRECT_GAIN_MAX, default_value=qc_direct_gain_loaded,min_clamped=True, max_clamped=True, callback=cb.qc_update_direct_gain)
                                            with dpg.tooltip("qc_direct_gain_title"):
                                                dpg.add_text("This will control the loudness of the direct signal")
                                                dpg.add_text("Higher values result in lower perceived distance, lower values result in higher perceived distance")
                                            dpg.add_slider_float(label="", min_value=CN.DIRECT_GAIN_MIN, max_value=CN.DIRECT_GAIN_MAX, default_value=qc_direct_gain_loaded, width=140,clamped=True, no_input=True, format="", callback=cb.qc_update_direct_gain_slider, tag='qc_direct_gain_slider')
                                            dpg.add_separator()
                                            dpg.add_text("Room Target", tag='qc_rm_target_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(CN.ROOM_TARGET_LIST, default_value=qc_room_target_loaded, num_items=7, width=235, tag='qc_rm_target_list', callback=cb.qc_select_room_target)
                                            with dpg.tooltip("qc_rm_target_title"):
                                                dpg.add_text("This will influence the overall balance of low and high frequencies")
                                            dpg.add_separator()
                                            dpg.add_text("Headphone Compensation", tag='qc_brir_hp_type_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(CN.HP_COMP_LIST, default_value=qc_brir_hp_type_loaded, num_items=4, width=235, callback=cb.qc_select_hp_comp, tag='qc_brir_hp_type')
                                            with dpg.tooltip("qc_brir_hp_type_title"):
                                                dpg.add_text("This should align with the listener's headphone type")
                                                dpg.add_text("Reduce to low strength if sound localisation or timbre is compromised")
                                                
                                with dpg.tab(label="Listener Selection",tag='qc_listener_tab', parent="qc_brir_tab_bar"): 
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Listener Type", tag='qc_brir_hrtf_type_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_radio_button(brir_hrtf_type_list_default, horizontal=False, tag= "qc_brir_hrtf_type", default_value=qc_brir_hrtf_type_loaded, callback=cb.qc_update_hrtf_dataset_list )
                                            with dpg.tooltip("qc_brir_hrtf_type_title"):
                                                dpg.add_text("User SOFA files must be placed in 'ASH Toolset\\_internal\\data\\user\\SOFA' folder")
                                            dpg.add_separator()
                                            dpg.add_text("Dataset", tag='qc_brir_hrtf_dataset_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(qc_brir_hrtf_dataset_list_loaded, default_value=qc_brir_hrtf_dataset_loaded, num_items=11, width=255, callback=cb.qc_update_hrtf_list, tag='qc_brir_hrtf_dataset')
                                        with dpg.group():
                                            dpg.add_text("Listener", tag='qc_brir_hrtf_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(qc_hrtf_list_loaded, default_value=qc_hrtf_loaded, num_items=17, width=240, callback=cb.qc_select_hrtf, tag='qc_brir_hrtf')
                                            with dpg.tooltip("qc_brir_hrtf_title"):
                                                dpg.add_text("This will influence the externalisation and localisation of sounds around the listener")
                                                
                                with dpg.tab(label="Low-frequency Extension",tag='qc_lfe_tab', parent="qc_brir_tab_bar"): 
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Integration Crossover Frequency", tag='qc_crossover_f_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_combo(CN.SUB_FC_SETTING_LIST, default_value=qc_crossover_f_mode_loaded, width=130, callback=cb.qc_update_crossover_f, tag='qc_crossover_f_mode')
                                            with dpg.tooltip("qc_crossover_f_mode"):
                                                dpg.add_text("Auto Select mode will select an optimal frequency for the selected acoustic space")
                                            dpg.add_input_int(label="Crossover Frequency (Hz)",width=140, tag='qc_crossover_f', min_value=CN.SUB_FC_MIN, max_value=CN.SUB_FC_MAX, default_value=qc_crossover_f_loaded,min_clamped=True, max_clamped=True, callback=cb.qc_update_crossover_f)
                                            with dpg.tooltip("qc_crossover_f"):
                                                dpg.add_text("Crossover Frequency can be adjusted to a value between 20Hz and 150Hz")
                                                dpg.add_text("This can be used to tune the integration of the cleaner LF response and original room response")
                                                dpg.add_text("Higher values may result in a smoother bass response")
                                            dpg.add_separator()
                                            dpg.add_text("Response for Low-frequency Extension", tag='qc_sub_response_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(CN.SUB_RESPONSE_LIST_GUI, default_value=qc_sub_response_loaded, num_items=7, width=240, callback=cb.qc_select_sub_brir, tag='qc_sub_response')
                                            with dpg.tooltip("qc_sub_response_title"):
                                                dpg.add_text("Refer to supporting information tab and filter preview for comparison")
                                            dpg.add_separator()
                                            dpg.add_text("Additonal EQ", tag='qc_hp_rolloff_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            with dpg.group(horizontal=True):
                                                dpg.add_checkbox(label="Compensate Headphone Roll-off", default_value = qc_hp_rolloff_comp_loaded,  tag='qc_hp_rolloff_comp', callback=cb.qc_update_brir_param)
                                                with dpg.tooltip("qc_hp_rolloff_comp"):
                                                    dpg.add_text("This will compensate the typical reduction in bass response at lower frequencies")
                                                dpg.add_text("      ")
                                                dpg.add_checkbox(label="Forward-Backward Filtering", default_value = qc_fb_filtering_loaded,  tag='qc_fb_filtering', callback=cb.qc_update_brir_param)
                                                with dpg.tooltip("qc_fb_filtering"):
                                                    dpg.add_text("This will eliminate delay introduced by the filters, however can introduce edge artefacts in some cases")
                                            
                                            dpg.add_separator()
                                            with dpg.group(horizontal=True):
                                                with dpg.group():
                                                    dpg.add_text("Analysis", tag='qc_analysis_title')
                                                    dpg.add_checkbox(label="Plot Integrated Response in Analysis Tab", default_value = qc_hp_rolloff_comp_loaded,  tag='qc_lf_analysis_toggle', callback=cb.qc_lf_analyse_toggle)
                                                dpg.add_text("          ")
                                                with dpg.group():
                                                    dpg.add_text("Plot Type", tag='qc_analysis_type_title')
                                                    dpg.add_radio_button(CN.SUB_PLOT_LIST, horizontal=True, default_value=CN.SUB_PLOT_LIST[0],   callback=cb.qc_lf_analyse_change_type, tag='qc_lf_analysis_type')
                                            
                                            
                                                
                        with dpg.child_window(autosize_x=True, height=175):
                            with dpg.group(horizontal=True):
                                subtitle_6_qc = dpg.add_text("Apply Simulation in Equalizer APO")
                                dpg.bind_item_font(subtitle_6_qc, bold_font)
                                dpg.add_text("                                                      ")
                  
                            dpg.add_separator()
                            
                            with dpg.group(horizontal=True):
                                dpg.add_button(label=CN.PROCESS_BUTTON_BRIR,user_data=CN.PROCESS_BRIRS_RUNNING,tag="qc_brir_tag", callback=cb.qc_apply_brir_params, width=145)#user data is thread running flag
                                dpg.bind_item_theme(dpg.last_item(), "__theme_a")
                                dpg.bind_item_font(dpg.last_item(), large_font)
                                with dpg.tooltip("qc_brir_tag"):
                                    dpg.add_text("This will generate the binaural simulation and apply it in Equalizer APO")   
                                dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=30, width=340, overlay=CN.PROGRESS_START, tag="qc_progress_bar_brir",user_data=CN.STOP_THREAD_FLAG)#user data is thread cancel flag
                                dpg.bind_item_theme(dpg.last_item(), "__theme_h")
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="Enable Binaural Room Simulation", default_value = e_apo_enable_brir_loaded,  tag='e_apo_brir_conv', callback=cb.e_apo_toggle_brir_gui, user_data=[])#user data will be a dict list
                                dpg.add_text("                                    ")
                     
                            dpg.add_separator()
                            dpg.add_text("Current Simulation: ")
                            dpg.bind_item_font(dpg.last_item(), bold_font)
                            dpg.add_text(default_value=e_apo_brir_curr_loaded,  tag='qc_e_apo_curr_brir_set' , wrap=506, user_data=False)#user data used for flagging use of below BRIR dict
                            dpg.bind_item_font(dpg.last_item(), bold_font)
                            dpg.bind_item_theme(dpg.last_item(), "__theme_f")
                            dpg.add_text(default_value=e_apo_brir_sel_loaded, tag='qc_e_apo_sel_brir_set',show=False, user_data={})#user data used for storing snapshot of BRIR dict 
                    #right most section
                    with dpg.group():    
                        #Section for channel config, plotting
                        with dpg.child_window(width=590, height=440):
                            with dpg.group(width=590):
                                with dpg.tab_bar(tag='qc_inner_tab_bar'):
                                    
                                    with dpg.tab(label="Channel Configuration", parent="qc_inner_tab_bar",tag='qc_cc_tab'):
                                        with dpg.group(horizontal=True):
                                            with dpg.group():
                                                with dpg.group(horizontal=True):
                                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                                    dpg.add_text("Configure Audio Channels                                                              ")
                                                    dpg.add_text("Reset Configuration:  ")
                                                    dpg.add_button(label="Reset to Default",  callback=reset_channel_config)
                                                dpg.add_separator()
                                                with dpg.group(horizontal=True):
                                                    dpg.add_text("Preamplification (dB) :  ")
                                                    dpg.add_input_float(label=" ",format="%.1f", width=100,min_value=-100, max_value=20,min_clamped=True,max_clamped=True, tag='e_apo_gain_oa',default_value=e_apo_gain_oa_loaded, callback=cb.e_apo_adjust_preamp)
                                                    dpg.add_text("  ")
                                                with dpg.group(horizontal=True):
                                                    dpg.add_text("Auto-Adjust Preamp: ")
                                                    dpg.add_combo(CN.AUTO_GAIN_METHODS, label="", width=160, default_value = e_apo_prevent_clip_loaded,  tag='e_apo_prevent_clip', callback=cb.e_apo_config_acquire_gui)
                                                    with dpg.tooltip("e_apo_prevent_clip"):
                                                        dpg.add_text("This will auto-adjust the preamp to prevent clipping or align low/mid frequency levels for 2.0 inputs")
                                                with dpg.group(horizontal=True):
                                                    dpg.add_text("Audio Channels:  ")
                                                    dpg.add_combo(CN.AUDIO_CHANNELS, width=180, label="",  tag='audio_channels_combo',default_value=audio_channels_loaded, callback=cb.e_apo_select_channels_gui)
                                                with dpg.group(horizontal=True):
                                                    with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                                        borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                                        borders_outerV=True, borders_innerV=True, delay_search=True):
                                                        #dpg.bind_item_font(dpg.last_item(), bold_font)
                                                        dpg.add_table_column(label="Input Channels")
                                                        dpg.add_table_column(label="Max. Peak Gain (dB)")
                                                        dpg.add_table_column(label="Average Peak Gain (dB)")
                                                        for i in range(3):
                                                            with dpg.table_row():
                                                                for j in range(3):
                                                                    if j == 0:#channels
                                                                        if i == 0:
                                                                            dpg.add_text('2.0 Stereo')
                                                                        elif i == 1:
                                                                            dpg.add_text('5.1 Surround')
                                                                        elif i == 2:
                                                                            dpg.add_text('7.1 Surround')
                                                                    elif j == 1:#peak gain
                                                                        if i == 0:
                                                                            dpg.add_text(tag='e_apo_gain_peak_2_0')
                                                                        elif i == 1:
                                                                            dpg.add_text(tag='e_apo_gain_peak_5_1')
                                                                        elif i == 2:
                                                                            dpg.add_text(tag='e_apo_gain_peak_7_1')
                                                                    elif j == 2:#average gain
                                                                        if i == 0:
                                                                            dpg.add_text(tag='e_apo_gain_avg_2_0')
                                                                        elif i == 1:
                                                                            dpg.add_text(tag='e_apo_gain_avg_5_1')
                                                                        elif i == 2:
                                                                            dpg.add_text(tag='e_apo_gain_avg_7_1')
                           
                                                    with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                                        borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                                        borders_outerV=True, borders_innerV=True, delay_search=True, tag='upmixing_table'):
                                                        #dpg.bind_item_font(dpg.last_item(), bold_font)
                                                        dpg.add_table_column(label="Upmixing Parameter")
                                                        dpg.add_table_column(label="Value")
                                                        for i in range(3):
                                                            with dpg.table_row():
                                                                for j in range(2):
                                                                    if j == 0:#channels
                                                                        if i == 0:
                                                                            dpg.add_text('Upmixing Method',tag='e_apo_upmix_method_text')
                                                                            with dpg.tooltip("e_apo_upmix_method_text"):
                                                                                dpg.add_text('Method A performs simple channel duplication')
                                                                                dpg.add_text('Method B includes Mid/Side channel separation')
                                                                        elif i == 1:
                                                                            dpg.add_text('Side Delay (samples)',tag='e_apo_side_delay_text')
                                                                            with dpg.tooltip("e_apo_side_delay_text"):
                                                                                dpg.add_text('Side channels are delayed by specified samples')
                                                                        elif i == 2:
                                                                            dpg.add_text('Rear Delay (samples)',tag='e_apo_rear_delay_text')
                                                                            with dpg.tooltip("e_apo_rear_delay_text"):
                                                                                dpg.add_text('Rear channels are delayed by specified samples')
                                                                    elif j == 1:#peak gain
                                                                        if i == 0:
                                                                            dpg.add_combo(CN.UPMIXING_METHODS, width=100, label="",  tag='e_apo_upmix_method',default_value=e_apo_upmix_method_loaded, callback=cb.e_apo_config_acquire_gui)
                                                                        elif i == 1:
                                                                            dpg.add_input_int(label=" ", width=100,min_value=-100, max_value=100, tag='e_apo_side_delay',min_clamped=True,max_clamped=True, default_value=e_apo_side_delay_loaded, callback=cb.e_apo_config_acquire_gui)
                                                                        elif i == 2:
                                                                            dpg.add_input_int(label=" ", width=100,min_value=-100, max_value=100, tag='e_apo_rear_delay',min_clamped=True,max_clamped=True, default_value=e_apo_rear_delay_loaded, callback=cb.e_apo_config_acquire_gui)  
                                                                        
                                                with dpg.group(horizontal=True):
                                                    with dpg.group():
                                                        with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                                            borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                                            borders_outerV=True, borders_innerV=True, delay_search=True):
                                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                                            dpg.add_table_column(label="Channel")
                                                            dpg.add_table_column(label="Mute")
                                                            dpg.add_table_column(label="Gain (dB)")
                                                            dpg.add_table_column(label="Elevation (°)")
                                                            dpg.add_table_column(label="Azimuth (°)")
    
                                                            for i in range(7):
                                                                with dpg.table_row():
                                                                    for j in range(5):
                                                                        if j == 0:#channel
                                                                            if i == 0:
                                                                                dpg.add_text("L")
                                                                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                                                            elif i == 1:
                                                                                dpg.add_text("R")
                                                                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                                                            elif i == 2:
                                                                                dpg.add_text("C + SUB")
                                                                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                                                            elif i == 3:
                                                                                dpg.add_text("SL")
                                                                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                                                            elif i == 4:
                                                                                dpg.add_text("SR")
                                                                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                                                            elif i == 5:
                                                                                dpg.add_text("RL")
                                                                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                                                            elif i == 6:
                                                                                dpg.add_text("RR")
                                                                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                                                        if j == 1:#Mute
                                                                            if i == 0:
                                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_fl',default_value=e_apo_mute_fl_loaded, callback=cb.e_apo_config_acquire_gui)
                                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                            elif i == 1:
                                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_fr',default_value=e_apo_mute_fr_loaded, callback=cb.e_apo_config_acquire_gui)
                                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                            elif i == 2:
                                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_c',default_value=e_apo_mute_c_loaded, callback=cb.e_apo_config_acquire_gui)
                                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                            elif i == 3:
                                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_sl',default_value=e_apo_mute_sl_loaded, callback=cb.e_apo_config_acquire_gui)
                                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                            elif i == 4:
                                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_sr',default_value=e_apo_mute_sr_loaded, callback=cb.e_apo_config_acquire_gui)
                                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                            elif i == 5:
                                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_rl',default_value=e_apo_mute_rl_loaded, callback=cb.e_apo_config_acquire_gui)
                                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                            elif i == 6:
                                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_rr',default_value=e_apo_mute_rr_loaded, callback=cb.e_apo_config_acquire_gui)
                                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                        if j == 2:#gain
                                                                            if i == 0:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_fl',min_clamped=True,max_clamped=True, default_value=e_apo_gain_fl_loaded, callback=cb.e_apo_config_acquire_gui)
                                                                            elif i == 1:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_fr',min_clamped=True,max_clamped=True,default_value=e_apo_gain_fr_loaded, callback=cb.e_apo_config_acquire_gui)
                                                                            elif i == 2:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_c',min_clamped=True,max_clamped=True,default_value=e_apo_gain_c_loaded, callback=cb.e_apo_config_acquire_gui)
                                                                            elif i == 3:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_sl',min_clamped=True,max_clamped=True,default_value=e_apo_gain_sl_loaded, callback=cb.e_apo_config_acquire_gui)
                                                                            elif i == 4:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_sr',min_clamped=True,max_clamped=True,default_value=e_apo_gain_sr_loaded, callback=cb.e_apo_config_acquire_gui)
                                                                            elif i == 5:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_rl',min_clamped=True,max_clamped=True,default_value=e_apo_gain_rl_loaded, callback=cb.e_apo_config_acquire_gui)
                                                                            elif i == 6:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_rr',min_clamped=True,max_clamped=True,default_value=e_apo_gain_rr_loaded, callback=cb.e_apo_config_acquire_gui)
                                                                        if j == 3:#elevation
                                                                            if i == 0:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_fl',default_value=e_apo_elev_angle_fl_loaded, callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_fl"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 1:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_fr',default_value=e_apo_elev_angle_fr_loaded, callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_fr"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 2:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_c',default_value=e_apo_elev_angle_c_loaded, callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_c"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 3:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_sl',default_value=e_apo_elev_angle_sl_loaded, callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_sl"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 4:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_sr',default_value=e_apo_elev_angle_sr_loaded, callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_sr"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 5:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_rl',default_value=e_apo_elev_angle_rl_loaded, callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_rl"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 6:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_rr',default_value=e_apo_elev_angle_rr_loaded, callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_rr"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                        if j == 4:#azimuth
                                                                            if i == 0:
                                                                                dpg.add_combo(CN.AZ_ANGLES_FL_WAV, width=70, label="",  tag='e_apo_az_angle_fl',default_value=e_apo_az_angle_fl_loaded, callback=cb.e_apo_activate_direction_gui)
                                                                            elif i == 1:
                                                                                dpg.add_combo(CN.AZ_ANGLES_FR_WAV, width=70, label="",  tag='e_apo_az_angle_fr',default_value=e_apo_az_angle_fr_loaded, callback=cb.e_apo_activate_direction_gui)
                                                                            elif i == 2:
                                                                                dpg.add_combo(CN.AZ_ANGLES_C_WAV, width=70, label="",  tag='e_apo_az_angle_c',default_value=e_apo_az_angle_c_loaded, callback=cb.e_apo_activate_direction_gui)
                                                                            elif i == 3:
                                                                                dpg.add_combo(CN.AZ_ANGLES_SL_WAV, width=70, label="",  tag='e_apo_az_angle_sl',default_value=e_apo_az_angle_sl_loaded, callback=cb.e_apo_activate_direction_gui)
                                                                            elif i == 4:
                                                                                dpg.add_combo(CN.AZ_ANGLES_SR_WAV, width=70, label="",  tag='e_apo_az_angle_sr',default_value=e_apo_az_angle_sr_loaded, callback=cb.e_apo_activate_direction_gui)
                                                                            elif i == 5:
                                                                                dpg.add_combo(CN.AZ_ANGLES_RL_WAV, width=70, label="",  tag='e_apo_az_angle_rl',default_value=e_apo_az_angle_rl_loaded, callback=cb.e_apo_activate_direction_gui)
                                                                            elif i == 6:
                                                                                dpg.add_combo(CN.AZ_ANGLES_RR_WAV, width=70, label="",  tag='e_apo_az_angle_rr',default_value=e_apo_az_angle_rr_loaded, callback=cb.e_apo_activate_direction_gui)
                                       
                                                    with dpg.drawlist(width=250, height=200, tag="channel_drawing"):
                                                        with dpg.draw_layer():
    
                                                            dpg.draw_circle([CN.X_START, CN.Y_START], CN.RADIUS, color=[163, 177, 184])           
                                                            with dpg.draw_node(tag="listener_drawing"):
                                                                dpg.apply_transform(dpg.last_item(), dpg.create_translation_matrix([CN.X_START, CN.Y_START]))
                                                                #dpg.draw_circle([0, 0], 20, color=[163, 177, 184], fill=[158,158,158])
                                                                #dpg.draw_text([-19, -8], 'Listener', color=[0, 0, 0],size=13)
                                                                dpg.draw_image('listener_image',[-30, -30],[30, 30])
                                                                
                                                                with dpg.draw_node(tag="fl_drawing"):
                                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(e_apo_az_angle_fl_loaded*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
                                                                    with dpg.draw_node(tag="fl_drawing_inner", user_data=45.0):
                                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(e_apo_az_angle_fl_loaded*-1))/180.0 , [0, 0, -1]))
                                                                        dpg.draw_image('fl_image',[-12, -12],[12, 12])
                                                                        
                                                                with dpg.draw_node(tag="fr_drawing"):
                                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(e_apo_az_angle_fr_loaded*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
                                                                    with dpg.draw_node(tag="fr_drawing_inner", user_data=45.0):
                                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(e_apo_az_angle_fr_loaded*-1))/180.0 , [0, 0, -1]))
                                                                        dpg.draw_image('fr_image',[-12, -12],[12, 12])
                                                                
                                                                with dpg.draw_node(tag="c_drawing"):
                                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(e_apo_az_angle_c_loaded*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
                                                                    with dpg.draw_node(tag="c_drawing_inner", user_data=45.0):
                                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(e_apo_az_angle_c_loaded*-1))/180.0 , [0, 0, -1]))
                                                                        dpg.draw_image('c_image',[-12, -12],[12, 12])
                                                                        
                                                                with dpg.draw_node(tag="sl_drawing"):
                                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(e_apo_az_angle_sl_loaded*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
                                                                    with dpg.draw_node(tag="sl_drawing_inner", user_data=45.0):
                                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(e_apo_az_angle_sl_loaded*-1))/180.0 , [0, 0, -1]))
                                                                        dpg.draw_image('sl_image',[-12, -12],[12, 12])
                                                                        
                                                                with dpg.draw_node(tag="sr_drawing"):
                                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(e_apo_az_angle_sr_loaded*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
                                                                    with dpg.draw_node(tag="sr_drawing_inner", user_data=45.0):
                                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(e_apo_az_angle_sr_loaded*-1))/180.0 , [0, 0, -1]))
                                                                        dpg.draw_image('sr_image',[-12, -12],[12, 12])
                                                                        
                                                                with dpg.draw_node(tag="rl_drawing"):
                                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(e_apo_az_angle_rl_loaded*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
                                                                    with dpg.draw_node(tag="rl_drawing_inner", user_data=45.0):
                                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(e_apo_az_angle_rl_loaded*-1))/180.0 , [0, 0, -1]))
                                                                        dpg.draw_image('rl_image',[-12, -12],[12, 12])
                                                                        
                                                                with dpg.draw_node(tag="rr_drawing"):
                                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(e_apo_az_angle_rr_loaded*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
                                                                    with dpg.draw_node(tag="rr_drawing_inner", user_data=45.0):
                                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(e_apo_az_angle_rr_loaded*-1))/180.0 , [0, 0, -1]))
                                                                        dpg.draw_image('rr_image',[-12, -12],[12, 12])
                                                
                                                  
                      
                                    with dpg.tab(label="Filter Preview", parent="qc_inner_tab_bar",tag='qc_fp_tab'):
                                        #plotting
                                        #dpg.add_separator()
                                        dpg.add_text("Select a filter from list to preview")
                                        # create plot
                                        with dpg.plot(label="Magnitude Response Plot", height=350, width=580):
                                            # optionally create legend
                                            dpg.add_plot_legend()
                                    
                                            # REQUIRED: create x and y axes
                                            dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)", tag="qc_x_axis", log_scale=True)
                                            dpg.set_axis_limits("qc_x_axis", 10, 20000)
                                            dpg.add_plot_axis(dpg.mvYAxis, label="Magnitude (dB)", tag="qc_y_axis")
                                            dpg.set_axis_limits("qc_y_axis", -20, 15)
                                    
                                            # series belong to a y axis
                                            dpg.add_line_series(default_x, default_y, label="Plot", parent="qc_y_axis", tag="qc_series_tag")
                                            #initial plot
                                            hf.plot_data(fr_flat_mag, title_name='', n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=2, plot_type=2)
                                    
                                    with dpg.tab(label="Low-frequency Analysis", parent="qc_inner_tab_bar",tag='qc_lfa_tab'):
                                        #plotting
                                        #dpg.add_separator()
                                        dpg.add_text("Enable in Low-frequency extension tab. Plot will be automatically updated after parameters are applied")
                                 
                                        # create plot
                                        with dpg.plot(label="Low-frequency Analysis", height=350, width=580):
                                            # optionally create legend
                                            dpg.add_plot_legend(tag="lfa_legend_tag")
                                            
                                            # REQUIRED: create x and y axes
                                            dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)", tag="qc_lfa_x_axis", log_scale=True)
                                            dpg.set_axis_limits("qc_x_axis", 10, 20000)
                                            dpg.add_plot_axis(dpg.mvYAxis, label="Magnitude (dB)", tag="qc_lfa_y_axis")
                                            dpg.set_axis_limits("qc_y_axis", -20, 15)
                                            
                                    
                                            # series belong to a y axis
                                            dpg.add_line_series(default_x, default_y, label="Plot", parent="qc_lfa_y_axis", tag="qc_lfa_series_tag")
                                            #initial plot
                                            hf.plot_data(fr_flat_mag, title_name='', n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=2, plot_type=3)
                                            
                                    
                     
                        #Section for misc settings
                        with dpg.group():
                            #section to store e apo config path
                            dpg.add_text(tag='qc_selected_folder_base', show=False)
                            dpg.set_value('qc_selected_folder_base', qc_primary_path)
                            #Section for wav settings
                            with dpg.child_window(width=590, height=90):
                                title_4 = dpg.add_text("IR Format", tag='qc_export_title')
                                with dpg.tooltip("qc_export_title"):
                                    dpg.add_text("Configure sample rate and bit depth of exported impulse responses")
                                    dpg.add_text("This should align with the format of the playback audio device")
                                dpg.bind_item_font(title_4, bold_font)
                                dpg.add_separator()
                                with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                    borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                    borders_outerV=True, borders_innerV=True, delay_search=True):
                                    #dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_table_column(label="  File Format  ")
                                    dpg.add_table_column(label="   Sample Rate")
                                    dpg.add_table_column(label="   Bit Depth")
                                    for i in range(1):
                                        with dpg.table_row():
                                            for j in range(3):
                                                if j == 0:#File Format
                                                    dpg.add_text("   WAV")
                                                elif j == 1:#Sample Rate
                                                    with dpg.group(horizontal=True):
                                                        dpg.add_text("  ")
                                                        dpg.add_radio_button(CN.SAMPLE_RATE_LIST, horizontal=True, tag= "qc_wav_sample_rate", default_value=sample_freq_loaded, callback=cb.sync_wav_sample_rate )
                                                        dpg.add_text("  ")
                                                elif j == 2:#Bit Depth
                                                    with dpg.group(horizontal=True):
                                                        dpg.add_text("  ")
                                                        dpg.add_radio_button(CN.BIT_DEPTH_LIST, horizontal=True, tag= "qc_wav_bit_depth", default_value=bit_depth_loaded, callback=cb.sync_wav_bit_depth)
                                                        dpg.add_text("  ")
                            with dpg.child_window(width=590, height=72):
                                dpg.add_text("Audio Device Configuration", tag='qc_ad_title')
                                with dpg.tooltip("qc_ad_title"):
                                    dpg.add_text("Ensure the playback audio device sample rate aligns with the IR sample rate above")
                                dpg.add_separator()
                                dpg.add_button(label="Open Sound Control Panel", callback=cb.open_sound_control_panel)

                            
     

            with dpg.tab(label="Filter & Dataset Export",tag='filter_management', parent="tab_bar"):
                dpg.bind_item_theme(dpg.last_item(), "__theme_i")
                dpg.add_text("Export Headphone Correction Filters & Binaural Room Simulation Datasets")
                #dpg.bind_item_font(dpg.last_item(), large_font)
                dpg.add_separator()
            
                with dpg.group(horizontal=True):
                    #Section for HpCF Export
                    with dpg.child_window(width=550, height=610):
                        title_1 = dpg.add_text("Headphone Correction Filters")
                        dpg.bind_item_font(title_1, bold_font)
                        with dpg.child_window(autosize_x=True, height=390):
                            subtitle_1 = dpg.add_text("Select a Headphone", tag='hpcf_title')
                            with dpg.tooltip("hpcf_title"):
                                dpg.add_text("Select a headphone from below list for filter export")
                            dpg.bind_item_font(subtitle_1, bold_font)
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_text("Search Brand:")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_input_text(label="", callback=cb.filter_brand_list, width=105)
                                dpg.add_text("Search Headphone:")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_input_text(label="", callback=cb.filter_headphone_list, width=209)
                            with dpg.group(horizontal=True, width=0):
                                with dpg.group():
                                    dpg.add_text("Brand")
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_listbox(brands_list, width=135, num_items=16, tag='brand_list', callback=cb.update_headphone_list)
                                with dpg.group():
                                    dpg.add_text("Headphone")
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_listbox(hp_list_default, width=250, num_items=16, tag='headphone_list', default_value=headphone_default ,callback=cb.update_sample_list)
                                with dpg.group():
                                    dpg.add_text("Sample")
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_listbox(sample_list_default, width=115, num_items=16, default_value=sample_default, tag='sample_list', user_data=headphone_default, callback=cb.plot_sample)
                                    with dpg.tooltip("sample_list"):
                                        dpg.add_text("Note: all samples will be exported. Select a sample to preview")
                        with dpg.child_window(autosize_x=True, height=88):
                            subtitle_2 = dpg.add_text("Select Files to Include in Export")
                            dpg.bind_item_font(subtitle_2, bold_font)
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="WAV FIR Filters", default_value = fir_hpcf_exp_loaded, callback=cb.export_hpcf_file_toggle, tag='fir_hpcf_toggle')
                                with dpg.tooltip("fir_hpcf_toggle"):
                                    dpg.add_text("Min phase FIRs in WAV format for convolution. 1 Channel")
                                    dpg.add_text("Required for Equalizer APO configuration updates")
                                dpg.add_checkbox(label="WAV Stereo FIR Filters", default_value = fir_st_hpcf_exp_loaded, callback=cb.export_hpcf_file_toggle, tag='fir_st_hpcf_toggle')
                                with dpg.tooltip("fir_st_hpcf_toggle"):
                                    dpg.add_text("Min phase FIRs in WAV format for convolution. 2 Channels")
                                dpg.add_checkbox(label="E-APO Configuration Files", default_value = eapo_hpcf_exp_loaded, callback=cb.export_hpcf_file_toggle, tag='eapo_hpcf_toggle')  
                                with dpg.tooltip("eapo_hpcf_toggle"):
                                    dpg.add_text("Equalizer APO configurations to perform convolution with FIR filters. Deprecated from V2.0.0 onwards")
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="Graphic EQ Filters (127 Bands)", default_value = geq_hpcf_exp_loaded, callback=cb.export_hpcf_file_toggle, tag='geq_hpcf_toggle')
                                with dpg.tooltip("geq_hpcf_toggle"):
                                    dpg.add_text("Graphic EQ configurations with 127 bands. Compatible with Equalizer APO and Wavelet")
                                dpg.add_checkbox(label="Graphic EQ Filters (31 Bands)", default_value = geq_31_exp_loaded, callback=cb.export_hpcf_file_toggle, tag='geq_31_hpcf_toggle')
                                with dpg.tooltip("geq_31_hpcf_toggle"):
                                    dpg.add_text("Graphic EQ configurations with 31 bands. Compatible with 31 band graphic equalizers including Equalizer APO")
                                dpg.add_checkbox(label="HeSuVi Filters", default_value = hesuvi_hpcf_exp_loaded, callback=cb.export_hpcf_file_toggle, tag='hesuvi_hpcf_toggle')
                                with dpg.tooltip("hesuvi_hpcf_toggle"):
                                    dpg.add_text("Graphic EQ configurations with 103 bands. Compatible with HeSuVi. Saved in HeSuVi\\eq folder")
                        with dpg.child_window(autosize_x=True, height=84):
                            subtitle_3 = dpg.add_text("Export Correction Filters")
                            dpg.bind_item_font(subtitle_3, bold_font)
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="Process",user_data="",tag="hpcf_tag", callback=cb.process_hpcfs, width=130)
                                dpg.bind_item_theme(dpg.last_item(), "__theme_b")
                                dpg.bind_item_font(dpg.last_item(), large_font)
                                with dpg.tooltip("hpcf_tag"):
                                    dpg.add_text("This will export the selected filters to the output directory")
                                    
                                dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=32, width=370, overlay=CN.PROGRESS_START_ALT, tag="progress_bar_hpcf")
                                dpg.bind_item_theme(dpg.last_item(), "__theme_g")
    
                    #Section for BRIR generation
                    with dpg.child_window(width=520, height=610):
                        title_2 = dpg.add_text("Binaural Room Simulation", tag='brir_title')
                        dpg.bind_item_font(title_2, bold_font)
                        with dpg.tooltip("brir_title"):
                            dpg.add_text("Customise a new binaural room simulation using below parameters")
                        with dpg.child_window(autosize_x=True, height=388):
                            with dpg.tab_bar(tag='brir_tab_bar'):

                                with dpg.tab(label="Acoustics & EQ Parameters",tag='acoustics_eq_tab', parent="brir_tab_bar"): 
  
                                    #dpg.add_separator()
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Acoustic Space",tag='acoustic_space_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            with dpg.group(horizontal=True):
                                                dpg.add_text("Sort by: ")
                                                dpg.add_combo(CN.AC_SPACE_LIST_SORT_BY, width=140, label="",  tag='sort_by_as',default_value=CN.AC_SPACE_LIST_SORT_BY[0], callback=cb.sort_ac_space)
                                            dpg.add_listbox(CN.AC_SPACE_LIST_GUI, label="",tag='acoustic_space_combo',default_value=ac_space_loaded, callback=cb.update_ac_space, num_items=16, width=250)
                                            with dpg.tooltip("acoustic_space_title"):
                                                dpg.add_text("This will determine the listening environment")
                                            
                                            
                                        with dpg.group():
                                            dpg.add_text("Direct Sound Gain (dB)", tag='direct_gain_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_input_float(label="Direct Gain (dB)",width=140, format="%.1f", tag='direct_gain', min_value=CN.DIRECT_GAIN_MIN, max_value=CN.DIRECT_GAIN_MAX, default_value=direct_gain_loaded,min_clamped=True, max_clamped=True, callback=cb.update_direct_gain)
                                            with dpg.tooltip("direct_gain_title"):
                                                dpg.add_text("This will control the loudness of the direct signal")
                                                dpg.add_text("Higher values result in lower perceived distance, lower values result in higher perceived distance")
                                            dpg.add_slider_float(label="", min_value=CN.DIRECT_GAIN_MIN, max_value=CN.DIRECT_GAIN_MAX, default_value=direct_gain_loaded, width=140,clamped=True, no_input=True, format="", callback=cb.update_direct_gain_slider, tag='direct_gain_slider')
                                            dpg.add_separator()
                                            dpg.add_text("Room Target", tag='rm_target_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(CN.ROOM_TARGET_LIST, default_value=room_target_loaded, num_items=6, width=230, tag='rm_target_list', callback=cb.select_room_target)
                                            with dpg.tooltip("rm_target_title"):
                                                dpg.add_text("This will influence the overall balance of low and high frequencies")
                                            dpg.add_separator()
                                            dpg.add_text("Headphone Compensation", tag='brir_hp_type_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(CN.HP_COMP_LIST, default_value=brir_hp_type_loaded, num_items=4, width=230, callback=cb.select_hp_comp, tag='brir_hp_type')
                                            with dpg.tooltip("brir_hp_type_title"):
                                                dpg.add_text("This should align with the listener's headphone type")
                                                dpg.add_text("Reduce to low strength if sound localisation or timbre is compromised")
                                                
                                with dpg.tab(label="Listener Selection",tag='listener_tab', parent="brir_tab_bar"): 
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Spatial Resolution", tag= "brir_spat_res_title")
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_radio_button(spatial_res_list_loaded, horizontal=True, tag= "brir_spat_res", default_value=spatial_res_loaded, callback=cb.select_spatial_resolution )
                                            with dpg.tooltip("brir_spat_res_title"):
                                                dpg.add_text("Increasing resolution will increase number of directions available but will also increase processing time and dataset size")
                                                dpg.add_text("'Low' is recommended unless additional directions or SOFA export is required")
                                            dpg.add_separator()    
                                            dpg.add_text("Listener Type", tag='brir_hrtf_type_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_radio_button(brir_hrtf_type_list_default, horizontal=False, tag= "brir_hrtf_type", default_value=brir_hrtf_type_loaded, callback=cb.update_hrtf_dataset_list )
                                            with dpg.tooltip("brir_hrtf_type_title"):
                                                dpg.add_text("User SOFA files must be placed in 'ASH Toolset\\_internal\\data\\user\\SOFA' folder")
                                            dpg.add_separator()
                                            dpg.add_text("Dataset", tag='brir_hrtf_dataset_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(brir_hrtf_dataset_list_loaded, default_value=brir_hrtf_dataset_loaded, num_items=9, width=255, callback=cb.update_hrtf_list, tag='brir_hrtf_dataset')
                                        with dpg.group():
                                            
                                            dpg.add_text("Listener", tag='brir_hrtf_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(hrtf_list_loaded, default_value=hrtf_loaded, num_items=17, width=230, callback=cb.select_hrtf, tag='brir_hrtf')
                                            with dpg.tooltip("brir_hrtf_title"):
                                                dpg.add_text("This will influence the externalisation and localisation of sounds around the listener")
                                        
                                with dpg.tab(label="Low-frequency Extension",tag='lfe_tab', parent="brir_tab_bar"): 
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Integration Crossover Frequency", tag='crossover_f_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_combo(CN.SUB_FC_SETTING_LIST, default_value=crossover_f_mode_loaded, width=130, callback=cb.update_crossover_f, tag='crossover_f_mode')
                                            with dpg.tooltip("crossover_f_mode"):
                                                dpg.add_text("Auto Select mode will select an optimal frequency for the selected acoustic space")
                                                
                                            dpg.add_input_int(label="Crossover Frequency (Hz)",width=140, tag='crossover_f', min_value=CN.SUB_FC_MIN, max_value=CN.SUB_FC_MAX, default_value=crossover_f_loaded,min_clamped=True, max_clamped=True, callback=cb.update_crossover_f)
                                            with dpg.tooltip("crossover_f"):
                                                dpg.add_text("Crossover Frequency can be adjusted to a value between 20Hz and 150Hz")
                                                dpg.add_text("This can be used to tune the integration of the cleaner LF response and original room response")
                                                dpg.add_text("Higher values may result in a smoother bass response")
                                            dpg.add_separator()
                                            dpg.add_text("Response for Low-frequency Extension", tag='sub_response_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(CN.SUB_RESPONSE_LIST_GUI, default_value=sub_response_loaded, num_items=9, width=240, callback=cb.select_sub_brir, tag='sub_response')
                                            with dpg.tooltip("sub_response_title"):
                                                dpg.add_text("Refer to supporting information tab and filter preview for comparison")
                                            dpg.add_separator()
                                            dpg.add_text("Additonal EQ", tag='hp_rolloff_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            with dpg.group(horizontal=True):
                                                dpg.add_checkbox(label="Compensate Headphone Roll-off", default_value = hp_rolloff_comp_loaded,  tag='hp_rolloff_comp', callback=cb.update_brir_param)
                                                with dpg.tooltip("hp_rolloff_comp"):
                                                    dpg.add_text("This will compensate the typical reduction in bass response at lower frequencies")
                                                dpg.add_text("      ")
                                                dpg.add_checkbox(label="Forward-Backward Filtering", default_value = fb_filtering_loaded,  tag='fb_filtering', callback=cb.qc_update_brir_param)
                                                with dpg.tooltip("fb_filtering"):
                                                    dpg.add_text("This will eliminate delay introduced by the filters, however can introduce edge artefacts in some cases")
                                            
                                            
                                        
                        with dpg.child_window(autosize_x=True, height=88):
                            subtitle_5 = dpg.add_text("Select Files to Include in Export")
                            dpg.bind_item_font(subtitle_5, bold_font)
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="Direction-specific WAV BRIRs", default_value = dir_brir_exp_loaded,  tag='dir_brir_toggle', callback=cb.export_brir_toggle, show=dir_brir_exp_show)
                                with dpg.tooltip("dir_brir_toggle", tag='dir_brir_tooltip', show=dir_brir_tooltip_show):
                                    dpg.add_text("Binaural Room Impulse Responses (BRIRs) in WAV format for convolution")
                                    dpg.add_text("2 channels per file. One file for each direction")
                                    dpg.add_text("Required for Equalizer APO configuration updates")
                                dpg.add_checkbox(label="True Stereo WAV BRIR", default_value = ts_brir_exp_loaded,  tag='ts_brir_toggle', callback=cb.export_brir_toggle, show=ts_brir_exp_show)
                                with dpg.tooltip("ts_brir_toggle", tag='ts_brir_tooltip', show=ts_brir_tooltip_show):
                                    dpg.add_text("True Stereo BRIR in WAV format for convolution")
                                    dpg.add_text("4 channels. One file representing L and R speakers")
                                dpg.add_checkbox(label="SOFA File", default_value = sofa_brir_exp_loaded,  tag='sofa_brir_toggle', callback=cb.export_brir_toggle, show=sofa_brir_exp_show)
                                with dpg.tooltip("sofa_brir_toggle", tag='sofa_brir_tooltip', show=sofa_brir_tooltip_show):
                                    dpg.add_text("BRIR dataset as a SOFA file")
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="HeSuVi WAV BRIRs", default_value = hesuvi_brir_exp_loaded,  tag='hesuvi_brir_toggle', callback=cb.export_brir_toggle, show=hesuvi_brir_exp_show)  
                                with dpg.tooltip("hesuvi_brir_toggle", tag='hesuvi_brir_tooltip', show=hesuvi_brir_tooltip_show):
                                    dpg.add_text("BRIRs in HeSuVi compatible WAV format. 14 channels, 44.1kHz and 48kHz")
                                dpg.add_checkbox(label="E-APO Configuration Files", default_value = eapo_brir_exp_loaded,  tag='eapo_brir_toggle', callback=cb.export_brir_toggle, show=eapo_brir_exp_show)
                                with dpg.tooltip("eapo_brir_toggle", tag='eapo_brir_tooltip', show=eapo_brir_tooltip_show):
                                    dpg.add_text("Equalizer APO configurations to perform convolution with BRIRs. Deprecated from V2.0.0 onwards")
                        with dpg.child_window(autosize_x=True, height=86):
                            subtitle_6 = dpg.add_text("Generate and Export Binaural Dataset")
                            dpg.bind_item_font(subtitle_6, bold_font)
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="Process",user_data=CN.PROCESS_BRIRS_RUNNING,tag="brir_tag", callback=cb.start_process_brirs, width=130)
                                dpg.bind_item_theme(dpg.last_item(), "__theme_b")
                                dpg.bind_item_font(dpg.last_item(), large_font)
                                with dpg.tooltip("brir_tag"):
                                    dpg.add_text("This will generate the binaural dataset and export to the output directory. This may take some time to process")
                                    
                                dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=32, width=340, overlay=CN.PROGRESS_START_ALT, tag="progress_bar_brir",user_data=CN.STOP_THREAD_FLAG)
                                dpg.bind_item_theme(dpg.last_item(), "__theme_g")
                
                    #right most section
                    with dpg.group():    
                        #Section for plotting
                        with dpg.child_window(width=600, height=467):
                            with dpg.group(width=600):
                                with dpg.tab_bar(tag='inner_tab_bar'):
                                    with dpg.tab(label="Filter Preview", parent="inner_tab_bar",tag='fp_tab'):
                                        # title_3 = dpg.add_text("Filter Preview")
                                        # dpg.bind_item_font(title_3, bold_font)
                                        #plotting
                                        #dpg.add_separator()
                                        dpg.add_text("Select a filter from list to preview")
                                        # create plot
                                        with dpg.plot(label="Magnitude Response Plot", height=390, width=585):
                                            # optionally create legend
                                            dpg.add_plot_legend()
                                    
                                            # REQUIRED: create x and y axes
                                            dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)", tag="x_axis", log_scale=True)
                                            dpg.set_axis_limits("x_axis", 10, 20000)
                                            dpg.add_plot_axis(dpg.mvYAxis, label="Magnitude (dB)", tag="y_axis")
                                            dpg.set_axis_limits("y_axis", -20, 15)
                                    
                                            # series belong to a y axis
                                            dpg.add_line_series(default_x, default_y, label="Plot", parent="y_axis", tag="series_tag")
                                            #initial plot
                                            hf.plot_data(fr_flat_mag, title_name='', n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=2, plot_type=1)
                                        
                                                                        
                                    with dpg.tab(label="HeSuVi Channel Configuration", parent="inner_tab_bar",tag='hcc_tab'):
                                        dpg.add_text("Adjust source directions for HeSuVi export")
                                        dpg.add_button(label="Reset to Default",  callback=reset_channel_config)
                                        with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                                            borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                                            borders_outerV=True, borders_innerV=True, delay_search=True):
                                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                                            dpg.add_table_column(label="Channel")
                                                            dpg.add_table_column(label="Elevation Angle (°)")
                                                            dpg.add_table_column(label="Azimuth Angle (°)")
    
                                                            for i in range(7):
                                                                with dpg.table_row():
                                                                    for j in range(3):
                                                                        if j == 0:#channel
                                                                            if i == 0:
                                                                                dpg.add_text("L")
                                                                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                                                            elif i == 1:
                                                                                dpg.add_text("R")
                                                                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                                                            elif i == 2:
                                                                                dpg.add_text("C + SUB")
                                                                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                                                            elif i == 3:
                                                                                dpg.add_text("SL")
                                                                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                                                            elif i == 4:
                                                                                dpg.add_text("SR")
                                                                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                                                            elif i == 5:
                                                                                dpg.add_text("RL")
                                                                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                                                            elif i == 6:
                                                                                dpg.add_text("RR")
                                                                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                                                        if j == 1:#elevation
                                                                            if i == 0:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='hesuvi_elev_angle_fl',default_value=e_apo_elev_angle_default)
                                                                                with dpg.tooltip("hesuvi_elev_angle_fl"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 1:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='hesuvi_elev_angle_fr',default_value=e_apo_elev_angle_default)
                                                                                with dpg.tooltip("hesuvi_elev_angle_fr"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 2:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='hesuvi_elev_angle_c',default_value=e_apo_elev_angle_default)
                                                                                with dpg.tooltip("hesuvi_elev_angle_c"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 3:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='hesuvi_elev_angle_sl',default_value=e_apo_elev_angle_default)
                                                                                with dpg.tooltip("hesuvi_elev_angle_sl"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 4:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='hesuvi_elev_angle_sr',default_value=e_apo_elev_angle_default)
                                                                                with dpg.tooltip("hesuvi_elev_angle_sr"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 5:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='hesuvi_elev_angle_rl',default_value=e_apo_elev_angle_default)
                                                                                with dpg.tooltip("hesuvi_elev_angle_rl"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 6:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='hesuvi_elev_angle_rr',default_value=e_apo_elev_angle_default)
                                                                                with dpg.tooltip("hesuvi_elev_angle_rr"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                        if j == 2:#azimuth
                                                                            if i == 0:
                                                                                dpg.add_combo(CN.AZ_ANGLES_FL_WAV, width=80, label="",  tag='hesuvi_az_angle_fl',default_value=e_apo_az_angle_fl_default)
                                                                                with dpg.tooltip("hesuvi_az_angle_fl"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                            elif i == 1:
                                                                                dpg.add_combo(CN.AZ_ANGLES_FR_WAV, width=80, label="",  tag='hesuvi_az_angle_fr',default_value=e_apo_az_angle_fr_default)
                                                                                with dpg.tooltip("hesuvi_az_angle_fr"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                            elif i == 2:
                                                                                dpg.add_combo(CN.AZ_ANGLES_C_WAV, width=80, label="",  tag='hesuvi_az_angle_c',default_value=e_apo_az_angle_c_default)
                                                                                with dpg.tooltip("hesuvi_az_angle_c"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                            elif i == 3:
                                                                                dpg.add_combo(CN.AZ_ANGLES_SL_WAV, width=80, label="",  tag='hesuvi_az_angle_sl',default_value=e_apo_az_angle_sl_default)
                                                                                with dpg.tooltip("hesuvi_az_angle_sl"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                            elif i == 4:
                                                                                dpg.add_combo(CN.AZ_ANGLES_SR_WAV, width=80, label="",  tag='hesuvi_az_angle_sr',default_value=e_apo_az_angle_sr_default)
                                                                                with dpg.tooltip("hesuvi_az_angle_sr"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                            elif i == 5:
                                                                                dpg.add_combo(CN.AZ_ANGLES_RL_WAV, width=80, label="",  tag='hesuvi_az_angle_rl',default_value=e_apo_az_angle_rl_default)
                                                                                with dpg.tooltip("hesuvi_az_angle_rl"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                            elif i == 6:
                                                                                dpg.add_combo(CN.AZ_ANGLES_RR_WAV, width=80, label="",  tag='hesuvi_az_angle_rr',default_value=e_apo_az_angle_rr_default)
                                                                                with dpg.tooltip("hesuvi_az_angle_rr"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                    
                               
                        #Section for Exporting files
                        with dpg.group(horizontal=True):
                            #Section for wav settings
                            with dpg.child_window(width=225, height=139):
                                title_4 = dpg.add_text("IR Format", tag='export_title')
                                with dpg.tooltip("export_title"):
                                    dpg.add_text("Configure sample rate and bit depth of exported impulse responses")
                                    dpg.add_text("This should align with the format of the playback audio device")
                                dpg.bind_item_font(title_4, bold_font)
                                dpg.add_separator()
                                with dpg.group():
                                    dpg.add_text("Select Sample Rate")
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_radio_button(CN.SAMPLE_RATE_LIST, horizontal=True, tag= "wav_sample_rate", default_value=sample_freq_loaded, callback=cb.update_brir_param )
                                #dpg.add_text("          ")
                                with dpg.group():
                                    dpg.add_text("Select Bit Depth")
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_radio_button(CN.BIT_DEPTH_LIST, horizontal=True, tag= "wav_bit_depth", default_value=bit_depth_loaded, callback=cb.update_brir_param)
                                    
                            #output locations
                            with dpg.child_window(width=371, height=139):
                                with dpg.group(horizontal=True):
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_text("Output Locations", tag='out_dir_title')
                                    with dpg.tooltip("out_dir_title"):
                                        dpg.add_text("'EqualizerAPO\\config' directory should be selected if using Equalizer APO")
                                        dpg.add_text("Main outputs will be saved under 'ASH-Outputs' sub directory") 
                                        dpg.add_text("HeSuVi outputs will be saved in 'EqualizerAPO\\config\\HeSuVi'")  
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_text("           ")
                                    dpg.add_button(label="Open Folder",user_data="",tag="open_folder_tag", callback=cb.open_output_folder)
                                    dpg.add_text("      ")
                                    dpge.add_file_browser(width=800,height=600,label='Change Folder',show_as_window=True, dirs_only=True,show_ok_cancel=True, allow_multi_selection=False, collapse_sequences=True,callback=cb.show_selected_folder)
                                dpg.add_separator()
                                dpg.add_text("Main Outputs:")
                                dpg.bind_item_font(dpg.last_item(), bold_small_font)
                                dpg.add_text(tag='selected_folder_ash')
                                dpg.bind_item_font(dpg.last_item(), small_font)
                                with dpg.tooltip("selected_folder_ash"):
                                    dpg.add_text("Location to save correction filters and binaural datasets",tag="selected_folder_ash_tooltip")
                                dpg.add_text(tag='selected_folder_base',show=False)
                                dpg.add_text("HeSuVi Outputs:")
                                dpg.bind_item_font(dpg.last_item(), bold_small_font)
                                dpg.add_text(tag='selected_folder_hesuvi')
                                dpg.bind_item_font(dpg.last_item(), small_font)
                                with dpg.tooltip("selected_folder_hesuvi"):
                                    dpg.add_text("Location to save HeSuVi files",tag="selected_folder_hesuvi_tooltip")
                                
                            dpg.set_value('selected_folder_ash', primary_ash_path)
                            dpg.set_value('selected_folder_ash_tooltip', primary_ash_path)
                            dpg.set_value('selected_folder_base', primary_path)
                            dpg.set_value('selected_folder_hesuvi', primary_hesuvi_path)
                            dpg.set_value('selected_folder_hesuvi_tooltip', primary_hesuvi_path)
    

                        
            if CN.SHOW_AS_TAB:
                with dpg.tab(label="Acoustic Space Import", tag='as_import_tab', parent="tab_bar"):     
                    dpg.bind_item_theme(dpg.last_item(), "__theme_j")
                    dpg.add_text("Import New Acoustic Spaces from Impulse Response (IR) Files")
                    dpg.add_separator()
                
                    with dpg.group(horizontal=True):
                        # Panel 1 - Left
                        with dpg.child_window(width=400, height=610):
                            with dpg.child_window(width=385, height=470):
                                dpg.add_text("Set Parameters")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_separator()
                                with dpg.group(horizontal=True):
                                    dpg.add_text("IR Folder")
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_text("             ")
                                    dpg.add_button(label="Refresh Folder List", callback=cb.update_ir_folder_list, tag="refresh_folders_btn")
                                    with dpg.tooltip("refresh_folders_btn"):
                                        dpg.add_text("Click to reload the list of available IR folders")
                                    dpg.add_text("   ")
                                    dpg.add_button(label="Open Input Folder", tag="open_user_ir_folder_button", callback=cb.open_user_input_as_folder)
                                    with dpg.tooltip("open_user_ir_folder_button"):
                                        dpg.add_text("Click to open in explorer the folder where the user IRs are stored")
                                        dpg.add_text("IRs should be stored toether in a single subfolder, for example: 'Room A'")
                                        dpg.add_text("Supported file types: wav, sofa, mat, npy, hdf5")
                                        dpg.add_text("Files can have any number of channels and samples and any bit depth and sample rate")
                                dpg.add_listbox(items=[], label="", tag="ir_folder_list", callback=cb.folder_selected_callback, width=330, num_items=4)
                                with dpg.tooltip("ir_folder_list"):
                                    dpg.add_text("Choose an IR folder from the list")
                    
                                dpg.add_text("Name (optional)")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_input_text(label="", tag="space_name", width=300)
                                with dpg.tooltip("space_name"):
                                    dpg.add_text("Enter a name for the new acoustic space. If left blank, folder name will be used")
                    
                                dpg.add_text("Description (optional)")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_input_text(label="", tag="space_description", width=300, multiline=False, height=50)
                                with dpg.tooltip("space_description"):
                                    dpg.add_text("Enter a brief description of the acoustic space (optional)")
                                
                                with dpg.group(horizontal=True):
                                    with dpg.group():
                                        dpg.add_text("Long Reverb Tail Mode")
                                        dpg.bind_item_font(dpg.last_item(), bold_font)
                                        dpg.add_checkbox(label="Enable", tag="long_tail_mode")
                                        with dpg.tooltip("long_tail_mode"):
                                            dpg.add_text("Only enable if the IRs have long decay tails (> 1.5 seconds).")
                                            dpg.add_text("This will increase processing time")
                                        dpg.add_text("Low-frequency Mode")
                                        dpg.bind_item_font(dpg.last_item(), bold_font)
                                        dpg.add_checkbox(label="Enable", tag="as_subwoofer_mode")
                                        with dpg.tooltip("as_subwoofer_mode"):
                                            dpg.add_text("Enable if the IRs are low frequency measurements")
                                            dpg.add_text("This will make the result available under Low-frequency responses")
                                    
                                    
                                    dpg.add_text("                    ")
                                    with dpg.group():
                                        dpg.add_text("Noise Reduction")
                                        dpg.bind_item_font(dpg.last_item(), bold_font)
                                        dpg.add_checkbox(label="Enable", tag="noise_reduction_mode")
                                        with dpg.tooltip("noise_reduction_mode"):
                                            dpg.add_text("Enable if the IRs have high noise floor")
                                        dpg.add_text("Rise Time (ms)")
                                        dpg.bind_item_font(dpg.last_item(), bold_font)
                                        dpg.add_input_float(label="", tag="as_rise_time", width=120,default_value=5.0, min_value=0.0, max_value=20.0, format="%.2f",min_clamped=True, max_clamped=True)
                                        with dpg.tooltip("as_rise_time"):
                                            dpg.add_text("This will apply a fade in window of specified duration")
                                            dpg.add_text("Min: 0, Max: 20")
                                with dpg.group(horizontal=True):
                                    with dpg.group():
                                        dpg.add_text("Desired Directions")
                                        dpg.bind_item_font(dpg.last_item(), bold_font)
                                        dpg.add_input_int(label="", tag="unique_directions", width=120, default_value=1750, min_value=1000, max_value=3500)
                                        with dpg.tooltip("unique_directions"):
                                            dpg.add_text("Specify the desired number of source directions for spatial sampling")
                                            dpg.add_text("Min: 1000, Max: 3000. Decrease to reduce processing time")
                                    dpg.add_text("                     ")
                                    with dpg.group():
                                        dpg.add_text("Alignment Frequency (Hz)")
                                        dpg.bind_item_font(dpg.last_item(), bold_font)
                                        dpg.add_input_int(label="", tag="alignment_freq", width=120, default_value=110, min_value=50, max_value=150)
                                        with dpg.tooltip("alignment_freq"):
                                            dpg.add_text("Specify the cutoff frequency used for time domain alignment")
                                            dpg.add_text("Min: 50, Max: 150.")
                                    
                                with dpg.group(horizontal=True):
                                    with dpg.group():
                                        dpg.add_text("Pitch Shift Range")
                                        dpg.bind_item_font(dpg.last_item(), bold_font)
                                        dpg.add_input_float(label="Low", tag="pitch_range_low", width=120,default_value=0.0, min_value=-48.0, max_value=0.0, format="%.2f",min_clamped=True, max_clamped=True)
                                        with dpg.tooltip("pitch_range_low"):
                                            dpg.add_text("Set the minimum pitch shift in semitones (can be fractional)")
                                            dpg.add_text("Used to expand dataset with new simulated source directions")
                                            dpg.add_text("Only used in cases where few IRs are supplied")
                                            dpg.add_text("Min: -48, Max: 0")
                                        dpg.add_input_float(label="High", tag="pitch_range_high", width=120,default_value=24.0, min_value=0.0, max_value=48.0, format="%.2f",min_clamped=True, max_clamped=True)
                                        with dpg.tooltip("pitch_range_high"):
                                            dpg.add_text("Set the maximum pitch shift in semitones (can be fractional).")
                                            dpg.add_text("Used to expand dataset with new simulated source directions")
                                            dpg.add_text("Only used in cases where few IRs are supplied")
                                            dpg.add_text("Min: 0, Max: 48")
                                    dpg.add_text("           ")
                                    with dpg.group():
                                        dpg.add_text("Pitch Shift Compensation")
                                        dpg.bind_item_font(dpg.last_item(), bold_font)
                                        dpg.add_checkbox(label="Enable", tag="pitch_shift_comp")
                                        with dpg.tooltip("pitch_shift_comp"):
                                            dpg.add_text("Enable to correct pitch of new directions after expanding dataset")
                                            dpg.add_text("This may introduce artefacts")
                                        
                    
                                
                    
                            with dpg.child_window(width=385, height=120):
                                dpg.add_text("Process IRs")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_separator()
                    
                                with dpg.group(horizontal=True):
                                    dpg.add_text("Selected Folder:", tag="selected_folder_label")
                                    dpg.add_text("", tag="selected_folder_display")
                                    with dpg.tooltip("selected_folder_display"):
                                        dpg.add_text("Displays the selected IR folder")
                    
                                with dpg.group(horizontal=True):
                                    dpg.add_button(
                                        label="Start Processing",
                                        callback=cb.launch_processing_thread,
                                        tag="start_processing_btn",
                                        user_data={"ir_processing_running": False}
                                    )
                                    with dpg.tooltip("start_processing_btn"):
                                        dpg.add_text("Begin processing the selected acoustic space")
                                        dpg.add_text("This may take a few minutes to complete")
                                    dpg.add_text("   ")
                                    dpg.add_button(label="Cancel Processing", tag="cancel_processing_button", callback=cb.cancel_processing_callback, user_data={"cancel_event": threading.Event()})
                    
                                with dpg.group(horizontal=True):
                                    dpg.add_text("Progress: ")
                                    dpg.add_progress_bar(tag="progress_bar_as", default_value=0.0, width=290)
                                with dpg.tooltip("progress_bar_as"):
                                    dpg.add_text("Displays the processing progress.")
                
                        with dpg.group(horizontal=False):
                            # Panel 2 - Middle (Imported Acoustic Spaces Table)
                            with dpg.child_window(width=1285, height=310):
                                dpg.add_text("Imported Acoustic Spaces", tag="imported_as_title")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                with dpg.tooltip("imported_as_title"):
                                    dpg.add_text("Processed IRs are displayed below")
                                    #dpg.add_text("Restart is required for newly processed IRs to be displayed in other tabs")
                                
                                
                                with dpg.group(horizontal=True):
                                    dpg.add_button(label="Refresh Table", tag="refresh_as_table_button", callback=cb.update_as_table_from_csvs)
                                    with dpg.tooltip("refresh_as_table_button"):
                                        dpg.add_text("Searches for imported acoustic spaces.")
                                    dpg.add_text("      ")
                                    dpg.add_button(label="Open Dataset Folder", tag="open_dataset_folder_button", callback=cb.open_user_int_as_folder)
                                    with dpg.tooltip("open_dataset_folder_button"):
                                        dpg.add_text("Click to open in explorer the folder where the processed IRs are stored")
                                    dpg.add_text("      ")
                                    dpg.add_button(label="Delete Selected", tag="delete_selected_button", callback=lambda: dpg.configure_item("del_processed_popup", show=True))
                                    with dpg.tooltip("delete_selected_button"):
                                        dpg.add_text("Delete selected row and associated data")
                    
                                with dpg.popup("delete_selected_button", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="del_processed_popup"):
                                    dpg.add_text("Selected Acoustic Space will be deleted")
                                    dpg.add_separator()
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="OK", width=75, callback=cb.delete_selected_callback)
                                        dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("del_processed_popup", show=False))
                    
                                dpg.add_text("", tag="selected_ir_rows", show=False, user_data=[])
                                with dpg.table(tag="processed_irs_table", header_row=True,
                                       resizable=True,
                                       policy=dpg.mvTable_SizingStretchProp,
                                       borders_innerH=True, borders_outerH=True,
                                       borders_innerV=True, borders_outerV=True,
                                       row_background=True):
                        
                                    #dpg.add_table_column(label="")  # Selectable column (no header)
                                    dpg.add_table_column(label="Name", init_width_or_weight=60)
                                    dpg.add_table_column(label="RT60 (ms)", init_width_or_weight=20)
                                    dpg.add_table_column(label="Description", init_width_or_weight=330)
                    
                            # Panel 3 - Right (Logger)
                            with dpg.child_window(width=1285, height=296, tag="import_console_window"):
                                pass  # Logger will be added here later

                        
                        
       
                        
            with dpg.tab(label="Room Target Generator",tag='room_target_tab', parent="tab_bar"):    
                dpg.bind_item_theme(dpg.last_item(), "__theme_j")
                dpg.add_text("Create new Room Targets")
                dpg.add_separator()
            
                with dpg.group(horizontal=True):
                    # Panel 1 - Left
                    with dpg.child_window(width=400, height=492):
                        with dpg.child_window(width=385, height=352):
                            dpg.add_text("Set Parameters")
                            dpg.bind_item_font(dpg.last_item(), bold_font)
                            dpg.add_separator()
   
                            dpg.add_text("Name (optional)")
                            dpg.bind_item_font(dpg.last_item(), bold_font)
                            dpg.add_input_text(label="", tag="target_name", width=270, callback=cb.update_room_target_name_display)
                            with dpg.tooltip("target_name"):
                                dpg.add_text("Enter a name for the new room target. If left blank, parameters will be used")
                            dpg.add_spacer(height=10)
                            dpg.add_text("Low-shelf Filter")
                            dpg.bind_item_font(dpg.last_item(), bold_font)
                        
                            dpg.add_input_int(
                                label="Frequency (Hz)", 
                                tag="low_shelf_freq", 
                                width=200, 
                                default_value=100, 
                                min_value=20, 
                                max_value=1000, 
                                min_clamped=True, 
                                max_clamped=True, 
                                callback=cb.generate_room_target_callback, 
                                user_data={"generation_running": False, "save_to_file": False}
                            )
                            with dpg.tooltip("low_shelf_freq"):
                                dpg.add_text("Set the cutoff frequency for the low-shelf filter.\nRecommended range: 20-1000 Hz")
                        
                            dpg.add_input_float(label="Gain (dB)", tag="low_shelf_gain", width=200, default_value=6.0, min_value=-6.0, max_value=18.0, format="%.2f", 
                                                min_clamped=True, max_clamped=True, callback=cb.generate_room_target_callback, user_data={"generation_running": False, "save_to_file": False})
                            with dpg.tooltip("low_shelf_gain"):
                                dpg.add_text("Set the gain for the low-shelf filter.\nNegative values attenuate low frequencies.")
                        
                            dpg.add_input_float(label="Q-Factor", tag="low_shelf_q", width=200, default_value=0.707, min_value=0.1, max_value=5.0, format="%.2f",
                                                min_clamped=True, max_clamped=True, callback=cb.generate_room_target_callback, user_data={"generation_running": False, "save_to_file": False})
                            with dpg.tooltip("low_shelf_q"):
                                dpg.add_text("Set the Q-factor (slope) for the low-shelf filter.\nLower values = broader effect.")
                        
                            dpg.add_spacer(height=10)
                            #dpg.add_separator()
                            dpg.add_text("High-shelf Filter")
                            dpg.bind_item_font(dpg.last_item(), bold_font)
                        
                            dpg.add_input_int(
                                label="Frequency (Hz)", 
                                tag="high_shelf_freq", 
                                width=200, 
                                default_value=7000, 
                                min_value=1000, 
                                max_value=20000, 
                                min_clamped=True, 
                                max_clamped=True, 
                                callback=cb.generate_room_target_callback, 
                                user_data={"generation_running": False, "save_to_file": False}
                            )
                            with dpg.tooltip("high_shelf_freq"):
                                dpg.add_text("Set the cutoff frequency for the high-shelf filter.\nRecommended range: 1-20 kHz")
                        
                            dpg.add_input_float(label="Gain (dB)", tag="high_shelf_gain", width=200, default_value=-4.0, min_value=-18.0, max_value=6.0, format="%.2f",
                                                min_clamped=True, max_clamped=True, callback=cb.generate_room_target_callback, user_data={"generation_running": False, "save_to_file": False})
                            with dpg.tooltip("high_shelf_gain"):
                                dpg.add_text("Set the gain for the high-shelf filter.\nPositive values boost high frequencies.")
                        
                            dpg.add_input_float(label="Q-Factor", tag="high_shelf_q", width=200, default_value=0.4, min_value=0.1, max_value=5.0, format="%.2f", 
                                                min_clamped=True, max_clamped=True, callback=cb.generate_room_target_callback, user_data={"generation_running": False, "save_to_file": False})
                            with dpg.tooltip("high_shelf_q"):
                                dpg.add_text("Set the Q-factor (slope) for the high-shelf filter.\nLower values = broader effect.")
               
                        with dpg.child_window(width=385, height=120):
                            dpg.add_text("Generate Room Target")
                            dpg.bind_item_font(dpg.last_item(), bold_font)
                            dpg.add_separator()
                        
                            with dpg.group(horizontal=True):
                                dpg.add_text("Room Target Name:", tag="room_target_name_label")
                                dpg.add_text("", tag="room_target_name_display")
                                with dpg.tooltip("room_target_name_display"):
                                    dpg.add_text("Displays the name entered above for the new Room Target.")
                        
                            with dpg.group(horizontal=True):
                                dpg.add_button(
                                    label="Generate Target",
                                    tag="generate_room_target_btn",
                                    callback=cb.generate_room_target_callback,
                                    user_data={"generation_running": False, "save_to_file": True}
                                )
                                with dpg.tooltip("generate_room_target_btn"):
                                    dpg.add_text("Apply the low/high shelf filter settings and generate a new room target curve.")
                                    dpg.add_text("Newly created targets will be available in the quick config and dataset export tabs")
                        
                            with dpg.group(horizontal=True):
                                dpg.add_text("Progress:")
                                dpg.add_progress_bar(tag="progress_bar_target_gen", default_value=0.0, width=290)
                            with dpg.tooltip("progress_bar_target_gen"):
                                dpg.add_text("Displays progress while generating the new room target.")
            
                    with dpg.group(horizontal=False):
                        with dpg.group(horizontal=True):  # Overall horizontal layout for left plot + right list
                            # Plot Panel - Left
                            with dpg.child_window(width=580, height=492):
                                dpg.add_text("Room Target Preview", tag="target_plot_title")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                with dpg.tooltip("target_plot_title"):
                                    dpg.add_text("Displays the magnitude response of the selected room target.")
                        
                                with dpg.plot(label="Magnitude Response Plot", height=450, width=560):
                                    dpg.add_plot_legend()
                                    dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)", tag="rt_x_axis", log_scale=True)
                                    dpg.set_axis_limits("rt_x_axis", 10, 20000)
                                    dpg.add_plot_axis(dpg.mvYAxis, label="Magnitude (dB)", tag="rt_y_axis")
                                    dpg.set_axis_limits("rt_y_axis", -18, 18)
                                    dpg.add_line_series([], [], label="Target", parent="rt_y_axis", tag="rt_plot_series")
                        
                            # Listbox Panel - Right
                            with dpg.child_window(width=690, height=492):  # Adjusted width to fit total ~1285
                                dpg.add_text("Generated Room Targets", tag="generated_targets_title")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                with dpg.tooltip("generated_targets_title"):
                                    dpg.add_text("Displays the list of user-defined room targets")
                                    dpg.add_text("Targets will also be available in the quick config and dataset export tabs")
                        
                                with dpg.group(horizontal=True):
                                    dpg.add_button(label="Refresh List", tag="refresh_targets_button", callback=cb.update_room_target_list)
                                    with dpg.tooltip("refresh_targets_button"):
                                        dpg.add_text("Scans for newly generated room target filters.")
                        
                                    dpg.add_text("      ")
                                    dpg.add_button(label="Open Targets Folder", tag="open_target_folder_button", callback=cb.open_user_rt_folder)
                                    with dpg.tooltip("open_target_folder_button"):
                                        dpg.add_text("Opens the directory where room targets are stored.")
                        
                                    dpg.add_text("      ")
                                    dpg.add_button(label="Delete Selected", tag="delete_selected_target_button", callback=lambda: dpg.configure_item("del_target_popup", show=True))
                                    with dpg.tooltip("delete_selected_target_button"):
                                        dpg.add_text("Deletes the selected room target and associated data.")
                                        dpg.add_text("Restart is required for room target lists to refresh in other tabs")
                        
                                with dpg.popup("delete_selected_target_button", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="del_target_popup"):
                                    dpg.add_text("Selected Room Target will be deleted")
                                    dpg.add_separator()
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="OK", width=75, callback=cb.delete_selected_target_callback)
                                        dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("del_target_popup", show=False))
                        
                                dpg.add_text("", tag="selected_target_name", show=False, user_data=None)
                        
                                dpg.add_listbox(
                                    items=[],
                                    tag="room_target_listbox",
                                    width=650,
                                    num_items=23,
                                    callback=cb.on_room_target_selected  # Should trigger update to plot
                                )
                
                        # # Panel 3 - below (Logger)
                        # with dpg.child_window(width=1285, height=244, tag="rt_console_window"):
                        #     pass  # Logger will be added here later
                        
            with dpg.tab(label="Supporting Information",tag='support_tab', parent="tab_bar"):    
                dpg.bind_item_theme(dpg.last_item(), "__theme_j")
                with dpg.collapsing_header(label="Acoustic Spaces", default_open=True):
                    with dpg.child_window(auto_resize_x=True, auto_resize_y=True):
                        #dpg.add_text("Acoustic Spaces")
                        #dpg.bind_item_font(dpg.last_item(), bold_font)
                        data_length = len(CN.AC_SPACE_LIST_GUI)
    
                        with dpg.table(
                            header_row=True,
                            policy=dpg.mvTable_SizingFixedFit,
                            resizable=False,
                            borders_outerH=True,
                            borders_innerH=True,
                            no_host_extendX=True,
                            borders_outerV=True,
                            borders_innerV=True,
                            delay_search=True
                        ):
                            dpg.add_table_column(label="Name")
                            dpg.add_table_column(label="RT60 (ms)")
                            dpg.add_table_column(label="Description", width_fixed=True, init_width_or_weight=600)
                            dpg.add_table_column(label="Source Dataset", width_fixed=True, init_width_or_weight=850)
                        
                            for i in range(data_length):
                                with dpg.table_row():
                                    dpg.add_text(CN.AC_SPACE_LIST_GUI[i])
                                    dpg.add_text(CN.AC_SPACE_MEAS_R60[i])
                                    dpg.add_text(CN.AC_SPACE_DESCR[i], wrap=0)
                                    dpg.add_text(CN.AC_SPACE_DATASET[i], wrap=0)
                                
                with dpg.collapsing_header(label="Spatial Resolutions", default_open=True):
                    with dpg.child_window(auto_resize_x=True, auto_resize_y=True):
                        #dpg.add_text("Spatial Resolutions")
                        dpg.bind_item_font(dpg.last_item(), bold_font)
                        with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                            borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                            borders_outerV=True, borders_innerV=True, delay_search=True):
                            #dpg.bind_item_font(dpg.last_item(), bold_font)
                            dpg.add_table_column(label="Resolution")
                            dpg.add_table_column(label="Elevation Range")
                            dpg.add_table_column(label="Elevation Steps")
                            dpg.add_table_column(label="Azimuth Range")
                            dpg.add_table_column(label="Azimuth Steps")
                            for i in range(len(CN.SPATIAL_RES_LIST)):
                                with dpg.table_row():
                                    for j in range(5):
                                        if j == 0:#Resolution
                                            dpg.add_text(CN.SPATIAL_RES_LIST[i])
                                        elif j == 1:#Elevation Range
                                            dpg.add_text(CN.SPATIAL_RES_ELEV_RNG[i])
                                        elif j == 2:#Elevation Steps
                                            dpg.add_text(CN.SPATIAL_RES_ELEV_STP[i])
                                        elif j == 3:#Azimuth Range
                                            dpg.add_text(CN.SPATIAL_RES_AZIM_RNG[i])
                                        elif j == 4:#Azimuth Steps
                                            dpg.add_text(CN.SPATIAL_RES_AZIM_STP[i])
                with dpg.collapsing_header(label="Low-frequency Responses", default_open=True):  
                    #Section to show sub brir information
                    with dpg.child_window(auto_resize_x=True, auto_resize_y=True):
                        #dpg.add_text("Subwoofer Responses")
                        dpg.bind_item_font(dpg.last_item(), bold_font)
                        with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                            borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                            borders_outerV=True, borders_innerV=True, delay_search=True):
                            #dpg.bind_item_font(dpg.last_item(), bold_font)
                            dpg.add_table_column(label="Name")
                            dpg.add_table_column(label="Acoustic Space")
                            dpg.add_table_column(label="RT60 (ms)")
                            dpg.add_table_column(label="Frequency Range")
                            dpg.add_table_column(label="Tolerance")
                            dpg.add_table_column(label="Comments")
                            dpg.add_table_column(label="Source Type")
                            dpg.add_table_column(label="Listener Type")
                            dpg.add_table_column(label="Source Dataset")
                            for i in range(len(CN.SUB_RESPONSE_LIST_GUI)):
                                with dpg.table_row():
                                    dpg.add_text(CN.SUB_RESPONSE_LIST_GUI[i])
                                    dpg.add_text(CN.SUB_RESPONSE_LIST_AS[i],wrap =100)
                                    dpg.add_text(CN.SUB_RESPONSE_LIST_R60[i])
                                    dpg.add_text(CN.SUB_RESPONSE_LIST_RANGE[i])
                                    dpg.add_text(CN.SUB_RESPONSE_LIST_TOL[i])
                                    dpg.add_text(CN.SUB_RESPONSE_LIST_COMM[i])
                                    dpg.add_text(CN.SUB_RESPONSE_LIST_SOURCE[i])
                                    dpg.add_text(CN.SUB_RESPONSE_LIST_LISTENER[i])
                                    dpg.add_text(CN.SUB_RESPONSE_LIST_DATASET[i])
                             
                with dpg.collapsing_header(label="Supported SOFA Conventions", default_open=True):
                    with dpg.child_window(auto_resize_x=True, auto_resize_y=True):               
                        #dpg.add_text("Supported SOFA Conventions")
                        #dpg.bind_item_font(dpg.last_item(), bold_font)
                        #dpg.add_separator()
                        with dpg.group(horizontal=True):
                            with dpg.group():
                                dpg.add_text("Read")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                    borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                    borders_outerV=True, borders_innerV=True, delay_search=True):
                                    #dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_table_column(label="Convention")
                                    dpg.add_table_column(label="Version")
                                    dpg.add_table_column(label="SOFAConventionsVersion")
                                    for i in range(len(CN.SOFA_COMPAT_CONV)):
                                        with dpg.table_row():
                                            for j in range(3):
                                                if j == 0:#Convention
                                                    dpg.add_text(CN.SOFA_COMPAT_CONV[i])
                                                elif j == 1:#Version
                                                    dpg.add_text(CN.SOFA_COMPAT_VERS[i],wrap =100)
                                                elif j == 2:#SOFAConventionsVersion
                                                    dpg.add_text(CN.SOFA_COMPAT_CONVERS[i])
                            
                            with dpg.group():
                                dpg.add_text("Write")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                    borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                    borders_outerV=True, borders_innerV=True, delay_search=True):
                                    #dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_table_column(label="Convention")
                                    dpg.add_table_column(label="Version")
                                    dpg.add_table_column(label="SOFAConventionsVersion")
                                    for i in range(len(CN.SOFA_OUTPUT_CONV)):
                                        with dpg.table_row():
                                            for j in range(3):
                                                if j == 0:#Convention
                                                    dpg.add_text(CN.SOFA_OUTPUT_CONV[i])
                                                elif j == 1:#Version
                                                    dpg.add_text(CN.SOFA_OUTPUT_VERS[i],wrap =100)
                                                elif j == 2:#SOFAConventionsVersion
                                                    dpg.add_text(CN.SOFA_OUTPUT_CONVERS[i])
                                        
            
            with dpg.tab(label="Additional Tools & Settings",tag='additional_tools', parent="tab_bar"):    
                dpg.bind_item_theme(dpg.last_item(), "__theme_j")
                with dpg.group(horizontal=True):
                    with dpg.group():  
                        with dpg.group(horizontal=True):

                            #Section for database
                            with dpg.child_window(width=200, height=150):
                                dpg.add_text("App")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_separator()
                                dpg.add_text("Check for Updates on Start")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_checkbox(label="Auto Check for Updates", default_value = auto_check_updates_loaded,  tag='check_updates_start_tag', callback=cb.save_settings)
                                dpg.add_text("Check for App Updates")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Check for Updates",user_data="",tag="app_version_tag", callback=cb.check_app_version)
                                with dpg.tooltip("app_version_tag"):
                                    dpg.add_text("This will check for updates to the app and show versions in the log")   
                            with dpg.child_window(width=224, height=150):
                                dpg.add_text("Headphone Correction Filters")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_separator()
                                dpg.add_text("Check for Headphone Filter Updates")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Check for Updates",user_data="",tag="hpcf_db_version_tag", callback=cb.check_db_version)
                                with dpg.tooltip("hpcf_db_version_tag"):
                                    dpg.add_text("This will check for updates to the headphone correction filter dataset and show versions in the log")
                                dpg.add_text("Download Latest Headphone Filters")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Download Latest Dataset",user_data="",tag="hpcf_db_download_tag", callback=cb.download_latest_db)
                                with dpg.tooltip("hpcf_db_download_tag"):
                                    dpg.add_text("This will download latest version of the dataset and replace local version")
                            with dpg.child_window(width=224, height=150):
                                dpg.add_text("Acoustic Spaces")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_separator()
                                dpg.add_text("Check for Acoustic Space Updates")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Check for Updates",user_data="",tag="as_version_tag", callback=cb.check_as_versions)
                                with dpg.tooltip("as_version_tag"):
                                    dpg.add_text("This will check for updates to acoustic space datasets and show new versions in the log")
                                dpg.add_text("Download Acoustic Space Updates")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Download Latest Datasets",user_data="",tag="as_download_tag", callback=cb.download_latest_as_sets)
                                with dpg.tooltip("as_download_tag"):
                                    dpg.add_text("This will download any updates to acoustic space datasets and replace local versions")
                            with dpg.child_window(width=224, height=150):
                                dpg.add_text("HRTF Datasets")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_separator()
                                dpg.add_text("Check for HRTF Dataset Updates")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Check for Updates",user_data="",tag="hrtf_version_tag", callback=cb.check_hrtf_versions)
                                with dpg.tooltip("hrtf_version_tag"):
                                    dpg.add_text("This will check for updates to HRTF datasets and show new updates in the log")
                                dpg.add_text("Update HRTF Dataset List")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Download Latest Dataset List",user_data="",tag="hrtf_download_tag", callback=cb.download_latest_hrtf_sets)
                                with dpg.tooltip("hrtf_download_tag"):
                                    dpg.add_text("This will download the latest list of HRTF datasets. Restart required if updates found")
                            #Section to reset settngs
                            with dpg.child_window(width=190, height=150):
                                dpg.add_text("Inputs")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_separator()
                                dpg.add_text("Reset All Settings to Default")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Reset Settings",user_data="",tag="reset_settings_tag", callback=reset_settings)  
                            with dpg.child_window(width=240, height=150):
                                dpg.add_text("Outputs")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_separator()
                                dpg.add_text("Delete All Exported Headphone Filters")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Delete Headphone Filters",user_data="",tag="remove_hpcfs_tag", callback=cb.remove_hpcfs)
                                with dpg.tooltip("remove_hpcfs_tag"):
                                    dpg.add_text("Warning: this will delete all headphone filters that have been exported to the output directory")
                                dpg.add_text("Delete All Exported Binaural Datasets")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Delete Binaural Datasets",user_data="",tag="remove_brirs_tag", callback=cb.remove_brirs)
                                with dpg.tooltip("remove_brirs_tag"):
                                    dpg.add_text("Warning: this will delete all BRIRs that have been generated and exported to the output directory")  
                            #Section for misc settings
                            with dpg.child_window(width=340, height=150):
                                dpg.add_text("Misc. Settings")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_separator()    
                                with dpg.group(horizontal=True):
                                    with dpg.group(): 
                                        dpg.add_text("Force Left/Right Symmetry")
                                        dpg.bind_item_font(dpg.last_item(), bold_font)
                                        dpg.add_combo(CN.HRTF_SYM_LIST, default_value=hrtf_symmetry_loaded, width=130, callback=cb.qc_update_brir_param, tag='force_hrtf_symmetry')
                                        with dpg.tooltip("force_hrtf_symmetry"):
                                            dpg.add_text("This will mirror the left or right sides of the HATS / dummy head") 
                                            dpg.add_text("Applies to the direct sound. Reverberation is not modified") 
                                        dpg.add_text("Early Reflection Delay (ms)")
                                        dpg.bind_item_font(dpg.last_item(), bold_font)
                                        dpg.add_input_float(label=" ",width=140, format="%.1f", tag='er_delay_time_tag', min_value=CN.ER_RISE_MIN, max_value=CN.ER_RISE_MAX, default_value=er_rise_loaded,min_clamped=True, max_clamped=True, callback=cb.qc_update_brir_param)
                                        with dpg.tooltip("er_delay_time_tag"):
                                            dpg.add_text("This will increase the time between the direct sound and early reflections") 
                                            dpg.add_text("This can be used to increase perceived distance") 
                                            dpg.add_text("Min. 0ms, Max. 10ms")
                                    dpg.add_text("  ")
                                    with dpg.group(): 
                                        dpg.add_text("SOFA Export Convention")
                                        dpg.bind_item_font(dpg.last_item(), bold_font)
                                        dpg.add_combo(CN.SOFA_OUTPUT_CONV, default_value=sofa_exp_conv_loaded, width=150, callback=cb.save_settings, tag='sofa_exp_conv')
                                    
                #section for logging
                with dpg.child_window(width=1690, height=482, tag="console_window",user_data=None):
                    dpg.add_text("Primary Log",tag='log_text',user_data=__version__)
                    dpg.bind_item_font(dpg.last_item(), bold_font)
            

    
    dpg.setup_dearpygui()
    
    logz=logger.mvLogger(parent="console_window")
    dpg.configure_item('console_window',user_data=logz)#store as user data

    #section to log tool version on startup
    #log results
    log_string = 'Started ASH Toolset - Version: ' + __version__
    hf.log_with_timestamp(log_string, logz)
  
    dpg.show_viewport()
    dpg.set_primary_window("Primary Window", True)
    dpg.configure_item("Primary Window", horizontal_scrollbar=True)
    
    #inital configuration
    intialise_gui()
        
    dpg.start_dearpygui()
 
    dpg.destroy_context()
        
    #finally close the connection
    conn.close()

    logging.info('Finished') 










if __name__ == '__main__':
    
    main()
    
