# -*- coding: utf-8 -*-


"""
Main routine of ASH-Tools.

Created on Sun Apr 28 11:25:06 2024
@author: Shanon Pearce
License: (see /LICENSE)
"""

def main():
    import logging.config
    from os.path import join as pjoin
    from pathlib import Path
    import configparser
    from ash_toolset import brir_generation
    from ash_toolset import brir_export
    from ash_toolset import e_apo_config_creation
    from ash_toolset import constants as CN
    from ash_toolset import hpcf_functions
    from ash_toolset import helper_functions as hf
    from ash_toolset import air_processing
    from ash_toolset import hrir_processing
    from scipy.io import wavfile
    import mat73
    import dearpygui.dearpygui as dpg
    import dearpygui_extend as dpge
    from dearpygui_ext import logger
    import numpy as np
    import csv
    import json
    import winreg as wrg
    import ast
    import ctypes
    import math
    from time import time
    from time import sleep
    import threading
    import scipy as sp
    import sys
    import os
    import importlib


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
    
    #start with flat response
    impulse=np.zeros(CN.N_FFT)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))

    #default values
    sample_freq_default=CN.SAMPLE_RATE_LIST[0]
    bit_depth_default=CN.BIT_DEPTH_LIST[0]
    brir_hp_type_default='Over/On-Ear Headphones - High Strength'
    
    spatial_res_default=CN.SPATIAL_RES_LIST[0]
    room_target_default=CN.ROOM_TARGET_LIST[1]
    direct_gain_default=2.0
    #acoustic space
    ac_space_default=CN.AC_SPACE_LIST_GUI[0]
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
    e_apo_prevent_clip_default=True
    e_apo_hpcf_curr_default=''
    e_apo_hpcf_sel_default=''
    e_apo_brir_curr_default=''
    e_apo_brir_sel_default=''
    tab_selected_default=0
    hrtf_symmetry_default=CN.HRTF_SYM_LIST[0]
    er_rise_default=0
    qc_brir_hp_type_default='Over/On-Ear Headphones - High Strength'
    qc_room_target_default=CN.ROOM_TARGET_LIST[1]
    qc_direct_gain_default=2.0
    #acoustic space
    qc_ac_space_default=CN.AC_SPACE_LIST_GUI[0]
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
        #load settings
        config = configparser.ConfigParser()
        config.read(CN.SETTINGS_FILE)
        version_loaded = config['DEFAULT']['version']
        if __version__ == version_loaded:
            
            sample_freq_loaded = config['DEFAULT']['sampling_frequency']
            bit_depth_loaded = config['DEFAULT']['bit_depth']
            brir_hp_type_loaded = config['DEFAULT']['brir_headphone_type']
            hrtf_loaded=config['DEFAULT']['brir_hrtf']
            spatial_res_loaded=config['DEFAULT']['spatial_resolution']
            room_target_loaded=config['DEFAULT']['brir_room_target']
            direct_gain_loaded=float(config['DEFAULT']['brir_direct_gain'])
            #acoustic space
            ac_space_loaded=config['DEFAULT']['acoustic_space']
            fir_hpcf_exp_loaded=ast.literal_eval(config['DEFAULT']['fir_hpcf_exp']) 
            fir_st_hpcf_exp_loaded=ast.literal_eval(config['DEFAULT']['fir_st_hpcf_exp'])
            eapo_hpcf_exp_loaded=ast.literal_eval(config['DEFAULT']['eapo_hpcf_exp'])
            geq_hpcf_exp_loaded=ast.literal_eval(config['DEFAULT']['geq_hpcf_exp'])
            geq_31_exp_loaded=ast.literal_eval(config['DEFAULT']['geq_31_exp'])
            hesuvi_hpcf_exp_loaded=ast.literal_eval(config['DEFAULT']['hesuvi_hpcf_exp'])
            dir_brir_exp_loaded=ast.literal_eval(config['DEFAULT']['dir_brir_exp'])
            ts_brir_exp_loaded=ast.literal_eval(config['DEFAULT']['ts_brir_exp'])
            hesuvi_brir_exp_loaded=ast.literal_eval(config['DEFAULT']['hesuvi_brir_exp'])
            eapo_brir_exp_loaded=ast.literal_eval(config['DEFAULT']['eapo_brir_exp'])
            sofa_brir_exp_loaded=ast.literal_eval(config['DEFAULT']['sofa_brir_exp'])
            auto_check_updates_loaded=ast.literal_eval(config['DEFAULT']['auto_check_updates'])
            hrtf_symmetry_loaded=config['DEFAULT']['force_hrtf_symmetry']
            er_rise_loaded=float(config['DEFAULT']['er_delay_time'])
            base_folder_loaded = config['DEFAULT']['path']
            primary_path=base_folder_loaded
            primary_ash_path=pjoin(base_folder_loaded, CN.PROJECT_FOLDER)
            #E-APO config related settings
            e_apo_upmix_method_loaded=config['DEFAULT']['upmix_method']
            e_apo_side_delay_loaded=int(config['DEFAULT']['side_delay'])
            e_apo_rear_delay_loaded=int(config['DEFAULT']['rear_delay'])
            e_apo_mute_fl_loaded=ast.literal_eval(config['DEFAULT']['mute_fl'])
            e_apo_mute_fr_loaded=ast.literal_eval(config['DEFAULT']['mute_fr'])
            e_apo_mute_c_loaded=ast.literal_eval(config['DEFAULT']['mute_c'])
            e_apo_mute_sl_loaded=ast.literal_eval(config['DEFAULT']['mute_sl'])
            e_apo_mute_sr_loaded=ast.literal_eval(config['DEFAULT']['mute_sr'])
            e_apo_mute_rl_loaded=ast.literal_eval(config['DEFAULT']['mute_rl'])
            e_apo_mute_rr_loaded=ast.literal_eval(config['DEFAULT']['mute_rr'])
            e_apo_gain_oa_loaded=float(config['DEFAULT']['gain_oa'])
            e_apo_gain_fl_loaded=float(config['DEFAULT']['gain_fl'])
            e_apo_gain_fr_loaded=float(config['DEFAULT']['gain_fr'])
            e_apo_gain_c_loaded=float(config['DEFAULT']['gain_c'])
            e_apo_gain_sl_loaded=float(config['DEFAULT']['gain_sl'])
            e_apo_gain_sr_loaded=float(config['DEFAULT']['gain_sr'])
            e_apo_gain_rl_loaded=float(config['DEFAULT']['gain_rl'])
            e_apo_gain_rr_loaded=float(config['DEFAULT']['gain_rr'])
            e_apo_elev_angle_fl_loaded=int(config['DEFAULT']['elev_fl'])
            e_apo_elev_angle_fr_loaded=int(config['DEFAULT']['elev_fr'])
            e_apo_elev_angle_c_loaded=int(config['DEFAULT']['elev_c'])
            e_apo_elev_angle_sl_loaded=int(config['DEFAULT']['elev_sl'])
            e_apo_elev_angle_sr_loaded=int(config['DEFAULT']['elev_sr'])
            e_apo_elev_angle_rl_loaded=int(config['DEFAULT']['elev_rl'])
            e_apo_elev_angle_rr_loaded=int(config['DEFAULT']['elev_rr'])
            e_apo_az_angle_fl_loaded=int(config['DEFAULT']['azim_fl'])
            e_apo_az_angle_fr_loaded=int(config['DEFAULT']['azim_fr'])  
            e_apo_az_angle_c_loaded=int(config['DEFAULT']['azim_c'] )
            e_apo_az_angle_sl_loaded=int(config['DEFAULT']['azim_sl'])
            e_apo_az_angle_sr_loaded=int(config['DEFAULT']['azim_sr'] )
            e_apo_az_angle_rl_loaded=int(config['DEFAULT']['azim_rl'])
            e_apo_az_angle_rr_loaded=int(config['DEFAULT']['azim_rr'])
            e_apo_enable_hpcf_loaded=ast.literal_eval(config['DEFAULT']['enable_hpcf'])
            e_apo_enable_brir_loaded=ast.literal_eval(config['DEFAULT']['enable_brir'])
            e_apo_autoapply_hpcf_loaded=ast.literal_eval(config['DEFAULT']['auto_apply_hpcf'])
            e_apo_prevent_clip_loaded=ast.literal_eval(config['DEFAULT']['prevent_clip'])
            e_apo_hpcf_curr_loaded = config['DEFAULT']['hpcf_current']
            e_apo_hpcf_sel_loaded = config['DEFAULT']['hpcf_selected']
            e_apo_brir_curr_loaded = config['DEFAULT']['brir_set_current']
            e_apo_brir_sel_loaded = config['DEFAULT']['brir_set_selected']
            audio_channels_loaded=config['DEFAULT']['channel_config']
            tab_selected_loaded=int(config['DEFAULT']['tab_selected']) 
            qc_brir_hp_type_loaded = config['DEFAULT']['qc_brir_headphone_type']
            qc_hrtf_loaded=config['DEFAULT']['qc_brir_hrtf']
            qc_room_target_loaded=config['DEFAULT']['qc_brir_room_target']
            qc_direct_gain_loaded=float(config['DEFAULT']['qc_brir_direct_gain'])
            qc_ac_space_loaded=config['DEFAULT']['qc_acoustic_space']
            qc_brand_loaded=config['DEFAULT']['qc_brand']
            qc_headphone_loaded=config['DEFAULT']['qc_headphone']
            qc_sample_loaded=config['DEFAULT']['qc_sample']
            brir_hrtf_type_loaded=config['DEFAULT']['brir_hrtf_type']
            brir_hrtf_dataset_loaded=config['DEFAULT']['brir_hrtf_dataset']
            qc_brir_hrtf_type_loaded=config['DEFAULT']['qc_brir_hrtf_type']
            qc_brir_hrtf_dataset_loaded=config['DEFAULT']['qc_brir_hrtf_dataset']
            sofa_exp_conv_loaded=config['DEFAULT']['sofa_exp_conv']
            
        else:
            raise ValueError('Settings not loaded due to version mismatch')
        
    except:
        pass
    
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


    #
    # Equalizer APO related code
    #
    elevation_list_sel= CN.ELEV_ANGLES_WAV_LOW
  
    
    #
    ## GUI Functions - HPCFs
    #

    def filter_brand_list(sender, app_data):
        """ 
        GUI function to update list of brands based on input text
        """
        
        search_str = app_data
        #update brand list with filtered set
        brand_list_specific = hpcf_functions.search_brand_list(conn, search_str)
  
        if brand_list_specific and brand_list_specific != None:
            brand_list_chosen = brand_list_specific.copy()
        else:
            brand_list_chosen = brands_list.copy()
            
        dpg.configure_item('brand_list',items=brand_list_chosen)
        #reset brand value to first brand
        dpg.configure_item('brand_list',show=False)
        dpg.configure_item('brand_list',default_value=brand_list_chosen[0])
        dpg.configure_item('brand_list',show=True)

        #update headphone list
        brand = brand_list_chosen[0]
        hp_list_specific = hpcf_functions.get_headphone_list(conn, brand)
        dpg.configure_item('headphone_list',items=hp_list_specific)
        
        #also update sample list
        headphone = hp_list_specific[0]
        sample_list_specific = hpcf_functions.get_samples_list(conn, headphone)
        sample_list_sorted = (sorted(sample_list_specific))
        dpg.configure_item('sample_list',items=sample_list_sorted)
        dpg.configure_item('sample_list',user_data=headphone)
        
        #also update plot to Sample A
        sample = 'Sample A'
        hpcf_functions.hpcf_to_plot(conn, headphone, sample, plot_type=1)
        
        #reset sample list to Sample A
        dpg.configure_item('sample_list',show=False)
        dpg.configure_item('sample_list',default_value='Sample A')
        dpg.configure_item('sample_list',show=True)
        

   
    def qc_filter_brand_list(sender, app_data):
        """ 
        GUI function to update list of brands based on input text
        """
        
        search_str = app_data
        #update brand list with filtered set
        brand_list_specific = hpcf_functions.search_brand_list(conn, search_str)
  
        if brand_list_specific and brand_list_specific != None:
            brand_list_chosen = brand_list_specific.copy()
        else:
            brand_list_chosen = brands_list.copy()
            
        dpg.configure_item('qc_brand_list',items=brand_list_chosen)
        #reset brand value to first brand
        dpg.configure_item('qc_brand_list',show=False)
        dpg.configure_item('qc_brand_list',default_value=brand_list_chosen[0])
        dpg.configure_item('qc_brand_list',show=True)

        #update headphone list
        brand = brand_list_chosen[0]
        hp_list_specific = hpcf_functions.get_headphone_list(conn, brand)
        dpg.configure_item('qc_headphone_list',items=hp_list_specific)
        
        #also update sample list
        headphone = hp_list_specific[0]
        sample_list_specific = hpcf_functions.get_samples_list(conn, headphone)
        sample_list_sorted = (sorted(sample_list_specific))
        dpg.configure_item('qc_sample_list',items=sample_list_sorted)

        #also update plot to Sample A
        sample = 'Sample A'
        hpcf_functions.hpcf_to_plot(conn, headphone, sample, plot_type=2)
        
        #reset sample list to Sample A
        dpg.configure_item('qc_sample_list',show=False)
        dpg.configure_item('qc_sample_list',default_value='Sample A')
        dpg.configure_item('qc_sample_list',show=True)
        dpg.set_value("qc_toggle_hpcf_history", False)

 
    def filter_headphone_list(sender, app_data):
        """ 
        GUI function to update list of headphone based on input text
        """
        
        search_str = app_data
        #update brand list with filtered set
        headphone_list_specific = hpcf_functions.search_headphone_list(conn, search_str)
 
        if headphone_list_specific and headphone_list_specific != None:
            
            
            #clear out brand list
            dpg.configure_item('brand_list',items=[])
            
            #update headphone list
            dpg.configure_item('headphone_list',items=headphone_list_specific)
            
            #reset headphone value to first headphone
            dpg.configure_item('headphone_list',show=False)
            dpg.configure_item('headphone_list',default_value=headphone_list_specific[0])
            dpg.configure_item('headphone_list',show=True)
            
            #also update sample list
            headphone = headphone_list_specific[0]
            sample_list_specific = hpcf_functions.get_samples_list(conn, headphone)
            sample_list_sorted = (sorted(sample_list_specific))
            dpg.configure_item('sample_list',items=sample_list_sorted)
            dpg.configure_item('sample_list',user_data=headphone)
            
            #also update plot to Sample A
            sample = 'Sample A'
            hpcf_functions.hpcf_to_plot(conn, headphone, sample, plot_type=1)
            
            #reset sample list to Sample A
            dpg.configure_item('sample_list',show=False)
            dpg.configure_item('sample_list',default_value='Sample A')
            dpg.configure_item('sample_list',show=True)
            

        else:
            #reset brand list
            dpg.configure_item('brand_list',items=brands_list)
            #reset brand value to first brand
            dpg.configure_item('brand_list',show=False)
            dpg.configure_item('brand_list',default_value=brands_list[0])
            dpg.configure_item('brand_list',show=True)
            #update headphone list
            brand = brands_list[0]
            hp_list_specific = hpcf_functions.get_headphone_list(conn, brand)
            dpg.configure_item('headphone_list',items=hp_list_specific)
            
            #also update sample list
            headphone = hp_list_specific[0]
            sample_list_specific = hpcf_functions.get_samples_list(conn, headphone)
            sample_list_sorted = (sorted(sample_list_specific))
            dpg.configure_item('sample_list',items=sample_list_sorted)
            dpg.configure_item('sample_list',user_data=headphone)
            
            #also update plot to Sample A
            sample = 'Sample A'
            hpcf_functions.hpcf_to_plot(conn, headphone, sample, plot_type=1)
            
            #reset sample list to Sample A
            dpg.configure_item('sample_list',show=False)
            dpg.configure_item('sample_list',default_value='Sample A')
            dpg.configure_item('sample_list',show=True)

    
    def qc_filter_headphone_list(sender, app_data):
        """ 
        GUI function to update list of headphone based on input text
        """
        
        search_str = app_data
        #update brand list with filtered set
        headphone_list_specific = hpcf_functions.search_headphone_list(conn, search_str)
 
        if headphone_list_specific and headphone_list_specific != None:
            
            
            #clear out brand list
            dpg.configure_item('qc_brand_list',items=[])
            
            #update headphone list
            dpg.configure_item('qc_headphone_list',items=headphone_list_specific)
            
            #reset headphone value to first headphone
            dpg.configure_item('qc_headphone_list',show=False)
            dpg.configure_item('qc_headphone_list',default_value=headphone_list_specific[0])
            dpg.configure_item('qc_headphone_list',show=True)
            
            #also update sample list
            headphone = headphone_list_specific[0]
            sample_list_specific = hpcf_functions.get_samples_list(conn, headphone)
            sample_list_sorted = (sorted(sample_list_specific))
            dpg.configure_item('qc_sample_list',items=sample_list_sorted)
  
            #also update plot to Sample A
            sample = 'Sample A'
            hpcf_functions.hpcf_to_plot(conn, headphone, sample, plot_type=2)
            
            #reset sample list to Sample A
            dpg.configure_item('qc_sample_list',show=False)
            dpg.configure_item('qc_sample_list',default_value='Sample A')
            dpg.configure_item('qc_sample_list',show=True)

        else:
            #reset brand list
            dpg.configure_item('qc_brand_list',items=brands_list)
            #reset brand value to first brand
            dpg.configure_item('qc_brand_list',show=False)
            dpg.configure_item('qc_brand_list',default_value=brands_list[0])
            dpg.configure_item('qc_brand_list',show=True)
            #update headphone list
            brand = brands_list[0]
            hp_list_specific = hpcf_functions.get_headphone_list(conn, brand)
            dpg.configure_item('qc_headphone_list',items=hp_list_specific)
            
            #also update sample list
            headphone = hp_list_specific[0]
            sample_list_specific = hpcf_functions.get_samples_list(conn, headphone)
            sample_list_sorted = (sorted(sample_list_specific))
            dpg.configure_item('qc_sample_list',items=sample_list_sorted)
            
            #also update plot to Sample A
            sample = 'Sample A'
            hpcf_functions.hpcf_to_plot(conn, headphone, sample, plot_type=2)
            
            #reset sample list to Sample A
            dpg.configure_item('qc_sample_list',show=False)
            dpg.configure_item('qc_sample_list',default_value='Sample A')
            dpg.configure_item('qc_sample_list',show=True)
        dpg.set_value("qc_toggle_hpcf_history", False)
  
    def qc_show_hpcf_history(sender, app_data):
        """ 
        GUI function to update list of headphone based on exported hpcf files
        """
        
        output_path = dpg.get_value('qc_selected_folder_base')
        hp_list_out_latest = e_apo_config_creation.get_exported_hp_list(output_path)
        search_str = hp_list_out_latest
        #update brand list with filtered set
        headphone_list_specific = hpcf_functions.search_headphones_in_list(conn, search_str)
        
        #set list values to previous values if toggle disabled
        #brand=dpg.get_value('qc_brand_list')
        headphone = dpg.get_value('qc_headphone_list')
        sample = dpg.get_value('qc_sample_list')
        brand=hpcf_functions.get_brand(conn, headphone)
 
        if headphone_list_specific and headphone_list_specific != None and app_data == True:
            
            
            #clear out brand list
            dpg.configure_item('qc_brand_list',items=[])
            
            #update headphone list
            dpg.configure_item('qc_headphone_list',items=headphone_list_specific)
            
            #reset headphone value to first headphone
            dpg.configure_item('qc_headphone_list',show=False)
            dpg.configure_item('qc_headphone_list',default_value=headphone_list_specific[0])
            dpg.configure_item('qc_headphone_list',show=True)
            
            #also update sample list
            headphone = headphone_list_specific[0]
            sample_list_specific = hpcf_functions.get_samples_list(conn, headphone)
            sample_list_sorted = (sorted(sample_list_specific))
            dpg.configure_item('qc_sample_list',items=sample_list_sorted)
  
            #also update plot to Sample A
            sample = 'Sample A'
            hpcf_functions.hpcf_to_plot(conn, headphone, sample, plot_type=2)
            
            #reset sample list to Sample A
            dpg.configure_item('qc_sample_list',show=False)
            dpg.configure_item('qc_sample_list',default_value='Sample A')
            dpg.configure_item('qc_sample_list',show=True)

        else:
            #reset brand list
            dpg.configure_item('qc_brand_list',items=brands_list)
            #reset brand value to first brand
            dpg.configure_item('qc_brand_list',show=False)
            dpg.configure_item('qc_brand_list',default_value=brand)
            dpg.configure_item('qc_brand_list',show=True)
            #update headphone list
            hp_list_specific = hpcf_functions.get_headphone_list(conn, brand)
            dpg.configure_item('qc_headphone_list',items=hp_list_specific)
            #also update sample list
            sample_list_specific = hpcf_functions.get_samples_list(conn, headphone)
            sample_list_sorted = (sorted(sample_list_specific))
            dpg.configure_item('qc_sample_list',items=sample_list_sorted)
            #also update plot
            hpcf_functions.hpcf_to_plot(conn, headphone, sample, plot_type=2)
            #reset sample list
            dpg.configure_item('qc_sample_list',show=False)
            dpg.configure_item('qc_sample_list',default_value=sample)
            dpg.configure_item('qc_sample_list',show=True)
            
        
        
        #reset progress
        qc_reset_progress()
        save_settings()
    
    def update_headphone_list(sender, app_data):
        """ 
        GUI function to update list of headphones based on selected brand
        """
        
        #update headphone list
        brand = app_data
        hp_list_specific = hpcf_functions.get_headphone_list(conn, brand)
        dpg.configure_item('headphone_list',items=hp_list_specific)
        
        #also update sample list
        headphone = hp_list_specific[0]
        sample_list_specific = hpcf_functions.get_samples_list(conn, headphone)
        sample_list_sorted = (sorted(sample_list_specific))
        dpg.configure_item('sample_list',items=sample_list_sorted)
        dpg.configure_item('sample_list',user_data=headphone)
        
        #also update plot to Sample A
        sample = 'Sample A'
        hpcf_functions.hpcf_to_plot(conn, headphone, sample, plot_type=1)
        
        #reset sample list to Sample A
        dpg.configure_item('sample_list',show=False)
        dpg.configure_item('sample_list',default_value='Sample A')
        dpg.configure_item('sample_list',show=True)
        
        #reset progress
        reset_hpcf_progress()
        save_settings()
      
    def qc_update_headphone_list(sender, app_data):
        """ 
        GUI function to update list of headphones based on selected brand
        """
        
        #update headphone list
        brand = app_data
        hp_list_specific = hpcf_functions.get_headphone_list(conn, brand)
        dpg.configure_item('qc_headphone_list',items=hp_list_specific)
        
        #also update sample list
        headphone = hp_list_specific[0]
        sample_list_specific = hpcf_functions.get_samples_list(conn, headphone)
        sample_list_sorted = (sorted(sample_list_specific))
        dpg.configure_item('qc_sample_list',items=sample_list_sorted)

        #also update plot to Sample A
        sample = 'Sample A'
        hpcf_functions.hpcf_to_plot(conn, headphone, sample, plot_type=2)
        
        #reset sample list to Sample A
        dpg.configure_item('qc_sample_list',show=False)
        dpg.configure_item('qc_sample_list',default_value='Sample A')
        dpg.configure_item('qc_sample_list',show=True)

        
        #reset progress
        qc_reset_progress()
        save_settings()
        
        #run hpcf processing if auto apply setting enabled
        auto_apply_on = dpg.get_value('qc_auto_apply_hpcf_sel') 
        if auto_apply_on == True:
            qc_process_hpcfs()

    def update_sample_list(sender, app_data):
        """ 
        GUI function to update list of samples based on selected headphone
        """
        
        #update sample list
        headphone = app_data
        sample_list_specific = hpcf_functions.get_samples_list(conn, headphone)
        sample_list_sorted = (sorted(sample_list_specific))
        dpg.configure_item('sample_list',items=sample_list_sorted)
        dpg.configure_item('sample_list',user_data=headphone)
        
        #also update plot to Sample A
        sample = 'Sample A'
        hpcf_functions.hpcf_to_plot(conn, headphone, sample, plot_type=1)
        
        #reset sample list to Sample A
        dpg.configure_item('sample_list',show=False)
        dpg.configure_item('sample_list',default_value='Sample A')
        dpg.configure_item('sample_list',show=True)

        #reset progress
        reset_hpcf_progress()
        save_settings()
        
    def qc_update_sample_list(sender, app_data):
        """ 
        GUI function to update list of samples based on selected headphone
        """
        
        #update sample list
        headphone = app_data
        sample_list_specific = hpcf_functions.get_samples_list(conn, headphone)
        sample_list_sorted = (sorted(sample_list_specific))
        dpg.configure_item('qc_sample_list',items=sample_list_sorted)
        dpg.configure_item('qc_sample_list',user_data=headphone)
        
        #also update plot to Sample A
        sample = 'Sample A'
        hpcf_functions.hpcf_to_plot(conn, headphone, sample, plot_type=2)
        
        #reset sample list to Sample A
        dpg.configure_item('qc_sample_list',show=False)
        dpg.configure_item('qc_sample_list',default_value='Sample A')
        dpg.configure_item('qc_sample_list',show=True)
  
        
        #reset progress
        qc_reset_progress()
        save_settings()
        
        #run hpcf processing if auto apply setting enabled
        auto_apply_on = dpg.get_value('qc_auto_apply_hpcf_sel') 
        if auto_apply_on == True:
            qc_process_hpcfs()
        
    
    def plot_sample(sender, app_data, user_data):
        """ 
        GUI function to plot a selected sample
        """
        headphone_selected=dpg.get_value('headphone_list')
        sample = app_data 
        hpcf_functions.hpcf_to_plot(conn, headphone_selected, sample, plot_type=1)
        #reset progress
        reset_hpcf_progress()
        save_settings()
 
    def qc_plot_sample(sender, app_data, user_data):
        """ 
        GUI function to plot a selected sample
        """
        headphone_selected=dpg.get_value('qc_headphone_list')
        sample = app_data 
        hpcf_functions.hpcf_to_plot(conn, headphone_selected, sample, plot_type=2)
        #reset progress
        qc_reset_progress()
        save_settings()
        
        #run hpcf processing if auto apply setting enabled
        auto_apply_on = dpg.get_value('qc_auto_apply_hpcf_sel') 
        if auto_apply_on == True:
            qc_process_hpcfs()
 
 
    def export_hpcf_file_toggle(sender, app_data):
        """ 
        GUI function to trigger save and refresh
        """

        save_settings()
        #reset progress
        reset_hpcf_progress()


    def process_hpcfs(sender, app_data, user_data):
        """ 
        GUI function to process HPCFs
        """
        
        output_path = dpg.get_value('selected_folder_base')

        headphone = dpg.get_value('headphone_list')
        fir_export = dpg.get_value('fir_hpcf_toggle')
        fir_stereo_export = dpg.get_value('fir_st_hpcf_toggle')
        geq_export = dpg.get_value('geq_hpcf_toggle')
        geq_31_export = dpg.get_value('geq_31_hpcf_toggle')
        geq_103_export = False
        hesuvi_export = dpg.get_value('hesuvi_hpcf_toggle')
        eapo_export = dpg.get_value('eapo_hpcf_toggle')

        samp_freq_str = dpg.get_value('wav_sample_rate')
        samp_freq_int = CN.SAMPLE_RATE_DICT.get(samp_freq_str)
        bit_depth_str = dpg.get_value('wav_bit_depth')
        bit_depth = CN.BIT_DEPTH_DICT.get(bit_depth_str)
  
        hpcf_functions.hpcf_to_file_bulk(conn, primary_path=output_path, headphone=headphone, fir_export = fir_export, fir_stereo_export = fir_stereo_export, geq_export = geq_export, samp_freq=samp_freq_int, bit_depth=bit_depth, 
                                         geq_31_export = geq_31_export, geq_103_export = geq_103_export, hesuvi_export = hesuvi_export, eapo_export=eapo_export, gui_logger=logz, report_progress=2)
   
        save_settings()
       
    def qc_apply_hpcf_params(sender=None, app_data=None):
        """ 
        GUI function to apply hpcf parameters
        """
        force_output=False
        #check if saved hpcf set name is matching with currently selected params
        hpcf_name_full = calc_hpcf_name(full_name=True)
        hpcf_name = calc_hpcf_name(full_name=False)
        sel_hpcf_set=dpg.get_value('qc_e_apo_sel_hpcf')
        # if hpcf_name in sel_hpcf_set:#if only sample rate or bit depth changed, force write output
        #     force_output=True
        #if matching, enable hpcf conv in config
        if hpcf_name_full == sel_hpcf_set:
            dpg.set_value("e_apo_hpcf_conv", True)
            dpg.set_value("qc_e_apo_curr_hpcf", hpcf_name)
            dpg.set_value("qc_progress_bar_hpcf", 1)
            dpg.configure_item("qc_progress_bar_hpcf", overlay = CN.PROGRESS_FIN)
            e_apo_config_acquire()
        else:#else run hpcf processing from scratch
            qc_process_hpcfs(app_data=force_output)
                
       
    def qc_process_hpcfs(sender=None, app_data=False, user_data=None):
        """ 
        GUI function to process HPCFs
        """
        
        output_path = dpg.get_value('qc_selected_folder_base')
        headphone = dpg.get_value('qc_headphone_list')
        sample = dpg.get_value('qc_sample_list')
        fir_export = True
        fir_stereo_export = False
        geq_export = False
        geq_31_export = False
        geq_103_export = False
        hesuvi_export = False
        eapo_export = False
        samp_freq_str = dpg.get_value('qc_wav_sample_rate')
        samp_freq_int = CN.SAMPLE_RATE_DICT.get(samp_freq_str)
        bit_depth_str = dpg.get_value('qc_wav_bit_depth')
        bit_depth = CN.BIT_DEPTH_DICT.get(bit_depth_str)
        force_output=app_data

        hpcf_functions.hpcf_to_file_bulk(conn, primary_path=output_path, headphone=headphone, fir_export = fir_export, fir_stereo_export = fir_stereo_export, geq_export = geq_export, samp_freq=samp_freq_int, bit_depth=bit_depth, 
                                         geq_31_export = geq_31_export, geq_103_export = geq_103_export, hesuvi_export = hesuvi_export, eapo_export=eapo_export, gui_logger=logz, report_progress=1, force_output=force_output)
   

        #finally rewrite config file
        dpg.set_value("e_apo_hpcf_conv", True)
        e_apo_config_acquire()
        #update current hpcf text
        filter_name = headphone + ' ' + sample
        filter_name_full = headphone + ' ' + sample + ' ' + samp_freq_str + ' ' + bit_depth_str
        dpg.set_value("qc_e_apo_curr_hpcf", filter_name)
        dpg.set_value("qc_e_apo_sel_hpcf", filter_name_full) 
        save_settings(update_hpcf_pars=True)
       
    #
    ## GUI Functions - BRIRs
    #
    
    def select_spatial_resolution(sender, app_data):
        """ 
        GUI function to update spatial resolution based on input
        """
        
        brir_hrtf_type = dpg.get_value('brir_hrtf_type')
        
        #update hrtf list based on spatial resolution
        #also update file format selection based on spatial resolution
        #set some to false and hide irrelevant options
        if app_data == 'Max':

            dpg.set_value("ts_brir_toggle", False)
            dpg.set_value("hesuvi_brir_toggle", False)
            dpg.set_value("eapo_brir_toggle", False)
            
            dpg.configure_item("ts_brir_toggle", show=False)
            dpg.configure_item("hesuvi_brir_toggle", show=False)
            dpg.configure_item("eapo_brir_toggle", show=False)
            dpg.configure_item("sofa_brir_toggle", show=True)
 
            dpg.configure_item("ts_brir_tooltip", show=False)
            dpg.configure_item("hesuvi_brir_tooltip", show=False)
            dpg.configure_item("eapo_brir_tooltip", show=False)
            dpg.configure_item("sofa_brir_tooltip", show=True)
            
        elif app_data == 'High':

            dpg.configure_item("ts_brir_toggle", show=True)
            dpg.configure_item("hesuvi_brir_toggle", show=True)
            dpg.configure_item("eapo_brir_toggle", show=True)
            dpg.configure_item("sofa_brir_toggle", show=True)
 
            dpg.configure_item("ts_brir_tooltip", show=True)
            dpg.configure_item("hesuvi_brir_tooltip", show=True)
            dpg.configure_item("eapo_brir_tooltip", show=True)
            dpg.configure_item("sofa_brir_tooltip", show=True)
            
        else:

            dpg.set_value("sofa_brir_toggle", False)

            dpg.configure_item("ts_brir_toggle", show=True)
            dpg.configure_item("hesuvi_brir_toggle", show=True)
            dpg.configure_item("eapo_brir_toggle", show=True)
            dpg.configure_item("sofa_brir_toggle", show=False)

            dpg.configure_item("ts_brir_tooltip", show=True)
            dpg.configure_item("hesuvi_brir_tooltip", show=True)
            dpg.configure_item("eapo_brir_tooltip", show=True)
            dpg.configure_item("sofa_brir_tooltip", show=False)
        
        if app_data == 'Max':
            if brir_hrtf_type == CN.HRTF_TYPE_LIST[0]:
                brir_hrtf_dataset_list_new = CN.HRTF_TYPE_DATASET_DICT.get('Dummy Head - Max Resolution')
                dpg.configure_item('brir_hrtf_dataset',items=brir_hrtf_dataset_list_new)
                brir_hrtf_dataset_new = dpg.get_value('brir_hrtf_dataset')
                hrtf_list_new = hrir_processing.get_listener_list(listener_type=brir_hrtf_type, dataset_name=brir_hrtf_dataset_new, max_res_only=True)
                dpg.configure_item('brir_hrtf',items=hrtf_list_new)
        else:
            if brir_hrtf_type == CN.HRTF_TYPE_LIST[0]:
                brir_hrtf_dataset_list_new = CN.HRTF_TYPE_DATASET_DICT.get(brir_hrtf_type)
                dpg.configure_item('brir_hrtf_dataset',items=brir_hrtf_dataset_list_new)
                brir_hrtf_dataset_new = dpg.get_value('brir_hrtf_dataset')
                hrtf_list_new = hrir_processing.get_listener_list(listener_type=brir_hrtf_type, dataset_name=brir_hrtf_dataset_new)
                dpg.configure_item('brir_hrtf',items=hrtf_list_new)
                
        #reset progress bar
        reset_brir_progress()
        
        save_settings()
        

    def select_room_target(sender, app_data):
        """ 
        GUI function to update brir based on input
        """

        target_sel = app_data
        
        #run plot
        try:

            # populate room target dictionary for plotting
            # load room target filters (FIR)
            npy_fname = pjoin(CN.DATA_DIR_INT, 'room_targets_firs.npy')
            room_target_mat = np.load(npy_fname)
            #create dictionary
            target_mag_dict = {} 
            for idx, target in enumerate(CN.ROOM_TARGET_LIST_SHORT):
                room_target_fir=np.zeros(CN.N_FFT)
                room_target_fir[0:4096] = room_target_mat[idx]
                data_fft = np.fft.fft(room_target_fir)
                room_target_mag=np.abs(data_fft)
                room_target_name=CN.ROOM_TARGET_LIST[idx]
                target_mag_dict.update({room_target_name: room_target_mag})
        
            mag_response = target_mag_dict.get(target_sel)
            plot_tile = target_sel + ' frequency response'
            hf.plot_data(mag_response, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=2, plot_type=1)
    
        except:
            pass

        #reset progress bar
        reset_brir_progress()
        
        save_settings()
 
    def select_hrtf(sender=None, app_data=None):
        """ 
        GUI function to update brir based on input
        """
        
        brir_dict=get_brir_dict()
        brir_hrtf_type=brir_dict.get('brir_hrtf_type')
        brir_hrtf_dataset=brir_dict.get('brir_hrtf_dataset')
        brir_hrtf = brir_dict.get('brir_hrtf')
        brir_hrtf_short=brir_dict.get('brir_hrtf_short')
        try:
            #run plot
            spat_res_int=0
            if brir_hrtf_type == 'Human Listener':
                hrir_dir_base = CN.DATA_DIR_HRIR_NPY_HL
            elif brir_hrtf_type == 'Dummy Head / Head & Torso Simulator':
                hrir_dir_base = CN.DATA_DIR_HRIR_NPY_DH
            else:
                hrir_dir_base = CN.DATA_DIR_HRIR_NPY_USER#user sofa npy set
            sub_directory = 'h'
            #join spatial res subdirectory
            hrir_dir = pjoin(hrir_dir_base, sub_directory)
            # join dataset subdirectory
            if brir_hrtf_type != 'User SOFA Input':
                hrir_dir = pjoin(hrir_dir, brir_hrtf_dataset)
            # full filename
            npy_fname = pjoin(hrir_dir, f"{brir_hrtf_short}.npy")
            #load npy files
            hrir_list = np.load(npy_fname)
            hrir_selected = hrir_list[0]
            #set metadata
            total_elev_hrir = len(hrir_selected)
            total_azim_hrir = len(hrir_selected[0])
            total_chan_hrir = len(hrir_selected[0][0])
            total_samples_hrir = len(hrir_selected[0][0][0])
            elev_min=CN.SPATIAL_RES_ELEV_MIN[spat_res_int] 
            elev_nearest=CN.SPATIAL_RES_ELEV_NEAREST[spat_res_int] #as per hrir dataset
            azim_nearest=CN.SPATIAL_RES_AZIM_NEAREST[spat_res_int] 
            #grab hrir for specific direction
            for elev in range(total_elev_hrir):
                elev_deg = int(elev_min + elev*elev_nearest)
                for azim in range(total_azim_hrir):
                    azim_deg = int(azim*azim_nearest)
                    if elev_deg == 0 and azim_deg == 330:  
                        chan=1
                        hrir=np.zeros(CN.N_FFT)
                        hrir[0:total_samples_hrir] = hrir_selected[elev][azim][chan][0:total_samples_hrir]
                        data_fft = np.fft.fft(hrir)
                        hrtf_mag=np.abs(data_fft)
            
            mag_response = hrtf_mag
            plot_tile = 'HRTF sample: ' + brir_hrtf_short + ' 0° elevation, 30° azimuth, right ear'
            hf.plot_data(mag_response, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=2, level_ends=1, plot_type=1)
            
        except:
            hf.plot_data(fr_flat_mag, title_name='No Preview Available', n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=2, plot_type=1)

     
        #reset progress bar
        reset_brir_progress()
        
        save_settings()
    
    def select_hp_comp(sender, app_data):
        """ 
        GUI function to update brir based on input
        """
        
        hp_type = dpg.get_value("brir_hp_type")
        pinna_comp_int = CN.HP_COMP_LIST.index(hp_type)
        pinna_comp = pinna_comp_int
        
        #run plot
        try:
            # load pinna comp filter (FIR)
            npy_fname = pjoin(CN.DATA_DIR_INT, 'headphone_pinna_comp_fir.npy')
            pinna_comp_fir = np.load(npy_fname)

 
            # load additional headphone eq
            apply_add_hp_eq = 0
            if pinna_comp == 2:
                filename = 'additional_comp_for_over_&_on_ear_headphones.wav'
                apply_add_hp_eq = 1
            elif pinna_comp == 0:
                filename = 'additional_comp_for_in_ear_headphones.wav'
                apply_add_hp_eq = 1
            if apply_add_hp_eq > 0:
                wav_fname = pjoin(CN.DATA_DIR_INT, filename)
                samplerate, data_addit_eq = wavfile.read(wav_fname)
                data_addit_eq = data_addit_eq / (2.**31)
            #apply pinna compensation
            brir_eq_b=np.copy(impulse)
            if pinna_comp >= 2:
                brir_eq_b = sp.signal.convolve(brir_eq_b,pinna_comp_fir, 'full', 'auto')
            #apply additional eq for headphones
            if apply_add_hp_eq > 0:
                brir_eq_b = sp.signal.convolve(brir_eq_b,data_addit_eq, 'full', 'auto')
            pinna_comp_fir=np.zeros(CN.N_FFT)
            pinna_comp_fir[0:1024] = brir_eq_b[0:1024]
            data_fft = np.fft.fft(pinna_comp_fir)
            mag_response=np.abs(data_fft)
            plot_tile = 'Headphone Compensation: ' + hp_type
            hf.plot_data(mag_response, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=2, level_ends=1, plot_type=1)

        except:
            pass

        #reset progress bar
        reset_brir_progress()
        
        save_settings()
    
    
    def update_direct_gain(sender, app_data):
        """ 
        GUI function to update brir based on input
        """
        d_gain=app_data
        dpg.set_value("direct_gain_slider", d_gain)

        #reset progress bar
        reset_brir_progress()
        
        save_settings()
    
    def update_direct_gain_slider(sender, app_data):
        """ 
        GUI function to update brir based on input
        """
        
        d_gain=app_data
        dpg.set_value("direct_gain", d_gain)

        #reset progress bar
        reset_brir_progress()
        
        save_settings()
 
    
    def qc_select_room_target(sender, app_data):
        """ 
        GUI function to update brir based on input
        """
        target_sel = app_data
        #run plot
        try:

            # populate room target dictionary for plotting
            # load room target filters (FIR)
            npy_fname = pjoin(CN.DATA_DIR_INT, 'room_targets_firs.npy')
            room_target_mat = np.load(npy_fname)
            #create dictionary
            target_mag_dict = {} 
            for idx, target in enumerate(CN.ROOM_TARGET_LIST_SHORT):
                room_target_fir=np.zeros(CN.N_FFT)
                room_target_fir[0:4096] = room_target_mat[idx]
                data_fft = np.fft.fft(room_target_fir)
                room_target_mag=np.abs(data_fft)
                room_target_name=CN.ROOM_TARGET_LIST[idx]
                target_mag_dict.update({room_target_name: room_target_mag})
        
            mag_response = target_mag_dict.get(target_sel)
            plot_tile = target_sel + ' frequency response'
            hf.plot_data(mag_response, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=2, plot_type=2)
    
        except:
            pass

        #reset progress bar
        qc_reset_progress()
        save_settings()

 
    def qc_select_hrtf(sender=None, app_data=None):
        """ 
        GUI function to update brir based on input
        """
        brir_dict=get_brir_dict()
        brir_hrtf_type=brir_dict.get('qc_brir_hrtf_type')
        brir_hrtf_dataset=brir_dict.get('qc_brir_hrtf_dataset')
        brir_hrtf = brir_dict.get('qc_brir_hrtf')
        brir_hrtf_short=brir_dict.get('qc_brir_hrtf_short')
        try:
            #run plot
            spat_res_int=0
            if brir_hrtf_type == 'Human Listener':
                hrir_dir_base = CN.DATA_DIR_HRIR_NPY_HL
            elif brir_hrtf_type == 'Dummy Head / Head & Torso Simulator':
                hrir_dir_base = CN.DATA_DIR_HRIR_NPY_DH
            else:
                hrir_dir_base = CN.DATA_DIR_HRIR_NPY_USER#user sofa npy set
            sub_directory = 'h'
            #join spatial res subdirectory
            hrir_dir = pjoin(hrir_dir_base, sub_directory)
            # join dataset subdirectory
            if brir_hrtf_type != 'User SOFA Input':
                hrir_dir = pjoin(hrir_dir, brir_hrtf_dataset)
            # full filename
            npy_fname = pjoin(hrir_dir, f"{brir_hrtf_short}.npy")
            #load npy files
            hrir_list = np.load(npy_fname)
            hrir_selected = hrir_list[0]
            #set metadata
            total_elev_hrir = len(hrir_selected)
            total_azim_hrir = len(hrir_selected[0])
            total_chan_hrir = len(hrir_selected[0][0])
            total_samples_hrir = len(hrir_selected[0][0][0])
            elev_min=CN.SPATIAL_RES_ELEV_MIN[spat_res_int] 
            elev_nearest=CN.SPATIAL_RES_ELEV_NEAREST[spat_res_int] #as per hrir dataset
            azim_nearest=CN.SPATIAL_RES_AZIM_NEAREST[spat_res_int] 
            #grab hrir for specific direction
            for elev in range(total_elev_hrir):
                elev_deg = int(elev_min + elev*elev_nearest)
                for azim in range(total_azim_hrir):
                    azim_deg = int(azim*azim_nearest)
                    if elev_deg == 0 and azim_deg == 330:  
                        chan=1
                        hrir=np.zeros(CN.N_FFT)
                        hrir[0:total_samples_hrir] = hrir_selected[elev][azim][chan][0:total_samples_hrir]
                        data_fft = np.fft.fft(hrir)
                        hrtf_mag=np.abs(data_fft)
            
            mag_response = hrtf_mag
            plot_tile = 'HRTF sample: ' + brir_hrtf_short + ' 0° elevation, 30° azimuth, right ear'
            hf.plot_data(mag_response, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=2, level_ends=1, plot_type=2)
            
        except:
            hf.plot_data(fr_flat_mag, title_name='No Preview Available', n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=2, plot_type=2)

     
        #reset progress bar
        qc_reset_progress()
        save_settings()

    def qc_select_hp_comp(sender, app_data):
        """ 
        GUI function to update brir based on input
        """
        
        hp_type = dpg.get_value("qc_brir_hp_type")
        pinna_comp_int = CN.HP_COMP_LIST.index(hp_type)
        pinna_comp = pinna_comp_int
        
        #run plot
        try:
            # load pinna comp filter (FIR)
            npy_fname = pjoin(CN.DATA_DIR_INT, 'headphone_pinna_comp_fir.npy')
            pinna_comp_fir = np.load(npy_fname)

 
            # load additional headphone eq
            apply_add_hp_eq = 0
            if pinna_comp == 2:
                filename = 'additional_comp_for_over_&_on_ear_headphones.wav'
                apply_add_hp_eq = 1
            elif pinna_comp == 0:
                filename = 'additional_comp_for_in_ear_headphones.wav'
                apply_add_hp_eq = 1
            if apply_add_hp_eq > 0:
                wav_fname = pjoin(CN.DATA_DIR_INT, filename)
                samplerate, data_addit_eq = wavfile.read(wav_fname)
                data_addit_eq = data_addit_eq / (2.**31)
            #apply pinna compensation
            brir_eq_b=np.copy(impulse)
            if pinna_comp >= 2:
                brir_eq_b = sp.signal.convolve(brir_eq_b,pinna_comp_fir, 'full', 'auto')
            #apply additional eq for headphones
            if apply_add_hp_eq > 0:
                brir_eq_b = sp.signal.convolve(brir_eq_b,data_addit_eq, 'full', 'auto')
            pinna_comp_fir=np.zeros(CN.N_FFT)
            pinna_comp_fir[0:1024] = brir_eq_b[0:1024]
            data_fft = np.fft.fft(pinna_comp_fir)
            mag_response=np.abs(data_fft)
            plot_tile = 'Headphone Compensation: ' + hp_type
            hf.plot_data(mag_response, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=2, level_ends=1, plot_type=2)

        except:
            pass

        #reset progress bar
        qc_reset_progress()
        save_settings()
        
    
    def qc_update_direct_gain(sender, app_data):
        """ 
        GUI function to update brir based on input
        """
        d_gain=app_data
        dpg.set_value("qc_direct_gain_slider", d_gain)

        #reset progress bar
        qc_reset_progress()
        save_settings()
        

    def qc_update_direct_gain_slider(sender, app_data):
        """ 
        GUI function to update brir based on input
        """
        
        d_gain=app_data
        dpg.set_value("qc_direct_gain", d_gain)

        #reset progress bar
        qc_reset_progress()   
        save_settings()
        

    
    def update_brir_param(sender, app_data):
        """ 
        GUI function to update brir based on input
        """

        #reset progress bar
        reset_brir_progress()
        save_settings()
        
    
    def qc_update_brir_param(sender, app_data):
        """ 
        GUI function to update brir based on input
        """

        #reset progress bar
        qc_reset_progress()
        save_settings()
        
    def export_brir_toggle(sender, app_data):
        """ 
        GUI function to update settings based on toggle
        """
        
        #reset progress bar
        reset_brir_progress()
        save_settings()
    
    def sync_wav_sample_rate(sender, app_data):
        """ 
        GUI function to update settings based on toggle
        """
        dpg.set_value("wav_sample_rate", app_data)
        
        #reset progress bar
        qc_reset_progress()
        
        save_settings()
    
    def sync_wav_bit_depth(sender, app_data):
        """ 
        GUI function to update settings based on toggle
        """
        dpg.set_value("wav_bit_depth", app_data)
        
        #reset progress bar
        qc_reset_progress()
        
        save_settings()
    
 
     
    def update_hrtf_dataset_list(sender, app_data):
        """ 
        GUI function to update list of hrtf datasets based on selected hrtf type
        """
        if app_data != None:
            brir_hrtf_type_new= app_data
            #update spatial res to valid list
            if brir_hrtf_type_new == CN.HRTF_TYPE_LIST[0]:
                dpg.configure_item('brir_spat_res',items=CN.SPATIAL_RES_LIST)
            else:
                #change to smaller list
                dpg.configure_item('brir_spat_res',items=CN.SPATIAL_RES_LIST_LIM)
                spat_res = dpg.get_value("brir_spat_res")
                if spat_res == 'Max':#reduce from max to high
                    dpg.configure_item('brir_spat_res',show=False)
                    dpg.configure_item('brir_spat_res',default_value='High')
                    dpg.configure_item('brir_spat_res',show=True)

            brir_hrtf_dataset_list_new = CN.HRTF_TYPE_DATASET_DICT.get(brir_hrtf_type_new)
            #update dataset list with filtered type
            dpg.configure_item('brir_hrtf_dataset',items=brir_hrtf_dataset_list_new)
            brir_hrtf_dataset_new=brir_hrtf_dataset_list_new[0]
            #reset dataset value to first dataset
            dpg.configure_item('brir_hrtf_dataset',show=False)
            dpg.configure_item('brir_hrtf_dataset',default_value=brir_hrtf_dataset_new)
            dpg.configure_item('brir_hrtf_dataset',show=True)
            
            #hrtf list based on dataset and hrtf type
            hrtf_list_new = hrir_processing.get_listener_list(listener_type=brir_hrtf_type_new, dataset_name=brir_hrtf_dataset_new)
            dpg.configure_item('brir_hrtf',items=hrtf_list_new)
            brir_hrtf_new=hrtf_list_new[0]
            #reset dataset value to first dataset
            dpg.configure_item('brir_hrtf',show=False)
            dpg.configure_item('brir_hrtf',default_value=brir_hrtf_new)
            dpg.configure_item('brir_hrtf',show=True)
            select_hrtf()
            #reset progress bar
            reset_brir_progress()
            save_settings()
        
    def update_hrtf_list(sender, app_data):
        """ 
        GUI function to update list of hrtfs based on selected dataset
        """
        if app_data != None:
            brir_hrtf_type_new = (dpg.get_value("brir_hrtf_type"))
            brir_hrtf_dataset_new= app_data

            #hrtf list based on dataset and hrtf type
            spat_res = dpg.get_value("brir_spat_res")
            if spat_res == 'Max':#reduced list
                hrtf_list_new = hrir_processing.get_listener_list(listener_type=brir_hrtf_type_new, dataset_name=brir_hrtf_dataset_new, max_res_only=True)
            else:
                hrtf_list_new = hrir_processing.get_listener_list(listener_type=brir_hrtf_type_new, dataset_name=brir_hrtf_dataset_new)
            dpg.configure_item('brir_hrtf',items=hrtf_list_new)
            brir_hrtf_new=hrtf_list_new[0]
            #reset dataset value to first dataset
            dpg.configure_item('brir_hrtf',show=False)
            dpg.configure_item('brir_hrtf',default_value=brir_hrtf_new)
            dpg.configure_item('brir_hrtf',show=True)
            select_hrtf()
            #reset progress bar
            reset_brir_progress()
            save_settings()
           
    def qc_update_hrtf_dataset_list(sender, app_data):
        """ 
        GUI function to update list of hrtf datasets based on selected hrtf type
        """
        if app_data != None:
            qc_brir_hrtf_type_new= app_data
            qc_brir_hrtf_dataset_list_new = CN.HRTF_TYPE_DATASET_DICT.get(qc_brir_hrtf_type_new)
            #update dataset list with filtered type
            dpg.configure_item('qc_brir_hrtf_dataset',items=qc_brir_hrtf_dataset_list_new)
            qc_brir_hrtf_dataset_new=qc_brir_hrtf_dataset_list_new[0]
            #reset dataset value to first dataset
            dpg.configure_item('qc_brir_hrtf_dataset',show=False)
            dpg.configure_item('qc_brir_hrtf_dataset',default_value=qc_brir_hrtf_dataset_new)
            dpg.configure_item('qc_brir_hrtf_dataset',show=True)
            
            #qc hrtf list based on dataset and hrtf type
            qc_hrtf_list_new = hrir_processing.get_listener_list(listener_type=qc_brir_hrtf_type_new, dataset_name=qc_brir_hrtf_dataset_new)
            dpg.configure_item('qc_brir_hrtf',items=qc_hrtf_list_new)
            qc_brir_hrtf_new=qc_hrtf_list_new[0]
            #reset dataset value to first dataset
            dpg.configure_item('qc_brir_hrtf',show=False)
            dpg.configure_item('qc_brir_hrtf',default_value=qc_brir_hrtf_new)
            dpg.configure_item('qc_brir_hrtf',show=True)
            qc_select_hrtf()
            #reset progress bar
            qc_reset_progress()
            save_settings()

    def qc_update_hrtf_list(sender, app_data):
        """ 
        GUI function to update list of hrtfs based on selected dataset
        """
        if app_data != None:
            qc_brir_hrtf_type_new = (dpg.get_value("qc_brir_hrtf_type"))
            qc_brir_hrtf_dataset_new= app_data

            #qc hrtf list based on dataset and hrtf type
            qc_hrtf_list_new = hrir_processing.get_listener_list(listener_type=qc_brir_hrtf_type_new, dataset_name=qc_brir_hrtf_dataset_new)
            dpg.configure_item('qc_brir_hrtf',items=qc_hrtf_list_new)
            qc_brir_hrtf_new=qc_hrtf_list_new[0]
            #reset dataset value to first dataset
            dpg.configure_item('qc_brir_hrtf',show=False)
            dpg.configure_item('qc_brir_hrtf',default_value=qc_brir_hrtf_new)
            dpg.configure_item('qc_brir_hrtf',show=True)
            
            qc_select_hrtf()
            #reset progress bar
            qc_reset_progress()
            save_settings()
            
            
    def start_process_brirs(sender, app_data, user_data):
        """ 
        GUI function to start or stop head tracking thread
        """
        #thread bool
        process_brirs_running=dpg.get_item_user_data("brir_tag")
        
        if process_brirs_running == False:

            #set thread running flag
            process_brirs_running = True
            #update user data
            dpg.configure_item('brir_tag',user_data=process_brirs_running)
            dpg.configure_item('brir_tag',label="Cancel")
            
            #set stop thread flag flag
            stop_thread_flag = False
            #update user data
            dpg.configure_item('progress_bar_brir',user_data=stop_thread_flag)
            
            #start thread
            thread = threading.Thread(target=process_brirs, args=(), daemon=True)
            thread.start()
   
        else:
            
            #set stop thread flag flag
            stop_thread_flag = True
            #update user data
            dpg.configure_item('progress_bar_brir',user_data=stop_thread_flag)

    def process_brirs(sender=None, app_data=None, user_data=None):
        """ 
        GUI function to process BRIRs
        """

        brir_directional_export = (dpg.get_value("dir_brir_toggle"))
        brir_ts_export = (dpg.get_value("ts_brir_toggle"))
        hesuvi_export = (dpg.get_value("hesuvi_brir_toggle"))
        eapo_export = (dpg.get_value("eapo_brir_toggle"))
        sofa_export = (dpg.get_value("sofa_brir_toggle"))
        target = dpg.get_value("rm_target_list")
        room_target_int = CN.ROOM_TARGET_LIST.index(target)
        room_target = room_target_int
        direct_gain_db = dpg.get_value("direct_gain")
        direct_gain_db = round(direct_gain_db,1)#round to nearest .1 dB
        ac_space = dpg.get_value("acoustic_space_combo")
        ac_space_int = CN.AC_SPACE_LIST_GUI.index(ac_space)
        ac_space_short = CN.AC_SPACE_LIST_SHORT[ac_space_int]
        ac_space_src = CN.AC_SPACE_LIST_SRC[ac_space_int]
        hp_type = dpg.get_value("brir_hp_type")
        pinna_comp_int = CN.HP_COMP_LIST.index(hp_type)
        pinna_comp = pinna_comp_int
        output_path = dpg.get_value('selected_folder_base')
        samp_freq_str = dpg.get_value('wav_sample_rate')
        samp_freq_int = CN.SAMPLE_RATE_DICT.get(samp_freq_str)
        bit_depth_str = dpg.get_value('wav_bit_depth')
        bit_depth = CN.BIT_DEPTH_DICT.get(bit_depth_str)
        spat_res = dpg.get_value("brir_spat_res")
        spat_res_int = CN.SPATIAL_RES_LIST.index(spat_res)
        hrtf_symmetry = dpg.get_value('force_hrtf_symmetry')
        er_delay_time = dpg.get_value('er_delay_time_tag')
        er_delay_time = round(er_delay_time,1)#round to nearest .1 dB
        brir_dict=get_brir_dict()
        brir_hrtf_short=brir_dict.get('brir_hrtf_short')
        sofa_conv=dpg.get_value("sofa_exp_conv")
        
        #calculate name
        brir_name = brir_hrtf_short + '_'+ac_space_short + '_' + str(direct_gain_db) + 'dB_' + CN.ROOM_TARGET_LIST_SHORT[room_target] + '_' + CN.HP_COMP_LIST_SHORT[pinna_comp]
        
        """
        #Run BRIR integration
        """
        brir_gen, status = brir_generation.generate_integrated_brir(brir_name=brir_name, direct_gain_db=direct_gain_db, room_target=room_target, spatial_res=spat_res_int, early_refl_delay_ms=er_delay_time,
                                                            pinna_comp=pinna_comp, report_progress=2, gui_logger=logz, acoustic_space=ac_space_src, hrtf_symmetry=hrtf_symmetry, brir_dict=brir_dict)
        
        """
        #Run BRIR export
        """
   
        if brir_gen.size != 0 and status == 0:
            brir_export.export_brir(brir_arr=brir_gen, acoustic_space=ac_space_src, brir_name=brir_name, primary_path=output_path, samp_freq=samp_freq_int, 
                                bit_depth=bit_depth, brir_dir_export=brir_directional_export, brir_ts_export=brir_ts_export, hesuvi_export=hesuvi_export, 
                                gui_logger=logz, direct_gain_db=direct_gain_db, spatial_res=spat_res_int, sofa_export=sofa_export, brir_dict=brir_dict, sofa_conv=sofa_conv)
        
            #set progress to 100 as export is complete (assume E-APO export time is negligible)
            progress = 100/100
            hf.update_gui_progress(report_progress=2, progress=progress, message='Processed')
        elif status == 1:
            progress = 0/100
            hf.update_gui_progress(report_progress=2, progress=progress, message='Failed to generate dataset. Refer to log')
        elif status == 2:
            progress = 0/100
            hf.update_gui_progress(report_progress=2, progress=progress, message='Cancelled')
            

        if eapo_export == True and brir_gen.size != 0 and status == 0:
            """
            #Run E-APO Config creator for BRIR convolution
            """
            e_apo_config_creation.write_e_apo_configs_brirs(brir_name=brir_name, primary_path=output_path)
 
        
        #also reset thread running flag
        process_brirs_running = False
        #update user data
        dpg.configure_item('brir_tag',user_data=process_brirs_running)
        dpg.configure_item('brir_tag',label="Process")
        #set stop thread flag flag
        stop_thread_flag = False
        #update user data
        dpg.configure_item('progress_bar_brir',user_data=stop_thread_flag)
    
    def e_apo_toggle_brir_gui(sender=None, app_data=None):
        """ 
        GUI function to toggle brir convolution
        app_data is the toggle
        
        """
        aquire_config=True
        use_dict_list=False
        force_run_process=False

        e_apo_toggle_brir_custom(app_data=app_data, aquire_config=aquire_config, use_dict_list=use_dict_list, force_run_process=force_run_process)

    
    def e_apo_toggle_brir_custom(app_data=None, aquire_config=True, use_dict_list=False, force_run_process=False):
        """ 
        GUI function to toggle brir convolution - with custom parameters passed in
        app_data is the toggle
        
        """
        
        process_brirs_running=dpg.get_item_user_data("qc_brir_tag")
        if app_data == False:#toggled off
            dpg.set_value("qc_e_apo_curr_brir_set", '')
            #call main config writer function
            if aquire_config==True:#custom parameter will be none if called by gui
                e_apo_config_acquire()
            
            if process_brirs_running == True:
                #stop processing if already processing brirs
                qc_stop_process_brirs()
            else:
                #reset progress
                dpg.set_value("qc_progress_bar_brir", 0)
                dpg.configure_item("qc_progress_bar_brir", overlay = CN.PROGRESS_START)
            
        else:#toggled on
            #check if saved brir set name is matching with currently selected params
            brir_name_full = calc_brir_set_name(full_name=True)
            brir_name = calc_brir_set_name(full_name=False)
            sel_brir_set=dpg.get_value('qc_e_apo_sel_brir_set')
            #if matching and not forced to run, enable brir conv in config
            if brir_name_full == sel_brir_set and force_run_process==False:#custom parameter will be none if called by gui
                dpg.set_value("e_apo_brir_conv", True)
                dpg.set_value("qc_e_apo_curr_brir_set", brir_name)
                dpg.set_value("qc_progress_bar_brir", 1)
                dpg.configure_item("qc_progress_bar_brir", overlay = CN.PROGRESS_FIN)
                e_apo_activate_direction(force_reset=True)#run in case direction not found in case of reduced dataset
                if aquire_config==True:#custom parameter will be none if called by gui
                    e_apo_config_acquire()
            else:#else run brir processing from scratch
                if process_brirs_running == False:#only start if not already running
                    qc_start_process_brirs(use_dict_list=use_dict_list)

    


    def qc_apply_brir_params(sender=None, app_data=None):
        """ 
        GUI function to apply brir parameters, used for button press
        """

        #check if saved brir set name is matching with currently selected params
        brir_name = calc_brir_set_name(full_name=False)
        brir_name_full = calc_brir_set_name(full_name=True)
        sel_brir_set=dpg.get_value('qc_e_apo_sel_brir_set')
        #if matching, enable brir conv in config
        if brir_name_full == sel_brir_set:
            dpg.set_value("e_apo_brir_conv", True)
            dpg.set_value("qc_e_apo_curr_brir_set", brir_name)
            dpg.set_value("qc_progress_bar_brir", 1)
            dpg.configure_item("qc_progress_bar_brir", overlay = CN.PROGRESS_FIN)
            e_apo_activate_direction(force_reset=True)#run in case direction not found in case of reduced dataset
            e_apo_config_acquire()
        else:#else run brir processing from scratch
            qc_start_process_brirs()#this may trigger a cancel if already running
            
       
    def qc_start_process_brirs(use_dict_list=False):
        """ 
        GUI function to start or stop processing of BRIRs thread
        """
        #thread bool
        process_brirs_running=dpg.get_item_user_data("qc_brir_tag")
        
        if process_brirs_running == False:#if not already running

            #set thread running flag
            process_brirs_running = True
            #update user data
            dpg.configure_item('qc_brir_tag',user_data=process_brirs_running)
            dpg.configure_item('qc_brir_tag',label="Cancel")
            
            #reset stop thread flag flag
            stop_thread_flag = False
            #update user data
            dpg.configure_item('qc_progress_bar_brir',user_data=stop_thread_flag)
            
            #start thread
            thread = threading.Thread(target=qc_process_brirs, args=(use_dict_list,), daemon=True)
            thread.start()
   
        else:#if already running, cancel it
            
            qc_stop_process_brirs()

    def qc_stop_process_brirs():
        """ 
        GUI function to stop processing of BRIRs thread
        """
        #set stop thread flag flag
        stop_thread_flag = True
        #update user data
        dpg.configure_item('qc_progress_bar_brir',user_data=stop_thread_flag)
        

    def qc_process_brirs(use_dict_list=False):
        """ 
        GUI function to process BRIRs
        use_dict_list: bool, true = this function should use a prepopulated dict list containing BRIR data, false = generate new brir dataset
        """
        #grab parameters
        brir_directional_export = True
        brir_ts_export = False
        hesuvi_export = False
        sofa_export = False
        target = dpg.get_value("qc_rm_target_list")
        room_target_int = CN.ROOM_TARGET_LIST.index(target)
        room_target = room_target_int
        direct_gain_db = dpg.get_value("qc_direct_gain")
        direct_gain_db = round(direct_gain_db,1)#round to nearest .1 dB
        ac_space = dpg.get_value("qc_acoustic_space_combo")
        ac_space_int = CN.AC_SPACE_LIST_GUI.index(ac_space)
        ac_space_src = CN.AC_SPACE_LIST_SRC[ac_space_int]
        hp_type = dpg.get_value("qc_brir_hp_type")
        pinna_comp_int = CN.HP_COMP_LIST.index(hp_type)
        pinna_comp = pinna_comp_int
        output_path = dpg.get_value('qc_selected_folder_base')
        samp_freq_str = dpg.get_value('qc_wav_sample_rate')
        samp_freq_int = CN.SAMPLE_RATE_DICT.get(samp_freq_str)
        bit_depth_str = dpg.get_value('qc_wav_bit_depth')
        bit_depth = CN.BIT_DEPTH_DICT.get(bit_depth_str)
        hrtf_symmetry = dpg.get_value('force_hrtf_symmetry')
        spat_res_int = 0
        er_delay_time = dpg.get_value('er_delay_time_tag')
        er_delay_time = round(er_delay_time,1)#round to nearest .1 dB
        reduce_dataset = True#(dpg.get_value("qc_brir_reduce_dataset"))
        brir_dict_list=dpg.get_item_user_data("e_apo_brir_conv")#contains previously processed brirs
        brir_name = calc_brir_set_name(full_name=False)
        brir_name_full = calc_brir_set_name(full_name=True)
        out_dataset_name = CN.FOLDER_BRIRS_LIVE 
        force_use_brir_dict=dpg.get_item_user_data("qc_e_apo_curr_brir_set")
        if use_dict_list == False and force_use_brir_dict == False:#grab relevant config data from gui elements
            brir_dict=get_brir_dict()
        else:
            brir_dict=dpg.get_item_user_data("qc_e_apo_sel_brir_set")#grab from previously stored values
        
        """
        #Run BRIR integration
        """
        if use_dict_list == False or not brir_dict_list:#run brir integration process
            brir_gen, status = brir_generation.generate_integrated_brir(brir_name=out_dataset_name, direct_gain_db=direct_gain_db, room_target=room_target, spatial_res=spat_res_int, early_refl_delay_ms=er_delay_time, 
                                                            pinna_comp=pinna_comp, report_progress=1, gui_logger=logz, acoustic_space=ac_space_src, hrtf_symmetry=hrtf_symmetry, reduce_dataset=reduce_dataset, brir_dict=brir_dict)
        else:
            brir_gen= np.array([])
            status=0
        
        """
        #Run BRIR export
        """
        if (brir_gen.size != 0 and status == 0) or (use_dict_list == True and brir_dict_list):#either brir dataset was generated or dict list was provided
            #mute and disable conv before exporting files -> avoids conflicts
            gain_oa_selected=dpg.get_value('e_apo_gain_oa')
            dpg.set_value("e_apo_gain_oa", CN.EAPO_MUTE_GAIN)
            dpg.set_value("e_apo_brir_conv", False)
            e_apo_config_acquire(estimate_gain=False)
            #run export function
            brir_dict_list_new = brir_export.export_brir(brir_arr=brir_gen, acoustic_space=ac_space_src, brir_name=out_dataset_name, primary_path=output_path, samp_freq=samp_freq_int, 
                                bit_depth=bit_depth, brir_dir_export=brir_directional_export, brir_ts_export=brir_ts_export, hesuvi_export=hesuvi_export, 
                                gui_logger=logz, direct_gain_db=direct_gain_db, spatial_res=spat_res_int, sofa_export=sofa_export, reduce_dataset=reduce_dataset, brir_dict=brir_dict,
                                use_dict_list=use_dict_list, brir_dict_list=brir_dict_list)
        
            #set progress to 100 as export is complete (assume E-APO export time is negligible)
            progress = 100/100
            hf.update_gui_progress(report_progress=1, progress=progress, message='Processed')
            #rewrite config file
            dpg.set_value("e_apo_brir_conv", True)
            #save dict list within gui element
            if brir_dict_list_new:#only update if not empty list
                dpg.configure_item('e_apo_brir_conv',user_data=brir_dict_list_new)
            if use_dict_list == False:#only update if normal process
                #update current brir set text
                dpg.set_value("qc_e_apo_curr_brir_set", brir_name)
                dpg.set_value("qc_e_apo_sel_brir_set", brir_name_full)
            #unmute before writing configs once more
            dpg.set_value("e_apo_gain_oa", gain_oa_selected)
            dpg.set_value("e_apo_brir_conv", True)
            #wait before updating config
            sleep(0.1)
            #update directions to previously stored values if flagged
            if use_dict_list == True or force_use_brir_dict == True:
                e_apo_update_direction(aquire_config=False, brir_dict_new=brir_dict)
            #Reset user data flag
            dpg.configure_item('qc_e_apo_curr_brir_set',user_data=False)
            #rewrite config file
            e_apo_config_acquire()
            
            if use_dict_list == False:#only update if normal process
                save_settings(update_brir_pars=True)
                #if live set, also write a file containing name of dataset
                out_file_path = pjoin(output_path, CN.PROJECT_FOLDER_BRIRS,out_dataset_name,'dataset_name.txt')
                with open(out_file_path, 'w') as file:
                    file.write(brir_name)
            else:
                save_settings(update_brir_pars=False)
                
        elif status == 1:
            progress = 0/100
            hf.update_gui_progress(report_progress=1, progress=progress, message='Failed to generate dataset. Refer to log')
        elif status == 2:
            progress = 0/100
            hf.update_gui_progress(report_progress=1, progress=progress, message='Cancelled')
  
        #also reset thread running flag
        process_brirs_running = False
        #update user data
        dpg.configure_item('qc_brir_tag',user_data=process_brirs_running)
        dpg.configure_item('qc_brir_tag',label=CN.PROCESS_BUTTON_BRIR)
        #set stop thread flag flag
        stop_thread_flag = False
        #update user data
        dpg.configure_item('qc_progress_bar_brir',user_data=stop_thread_flag)
        if status == 0:
            #reset progress in case of changed settings
            qc_reset_progress()
    
    def calc_brir_set_name(full_name=True):
        """ 
        GUI function to calculate brir set name from currently selected parameters
        """
        target = dpg.get_value("qc_rm_target_list")
        room_target_int = CN.ROOM_TARGET_LIST.index(target)
        room_target = room_target_int
        direct_gain_db = dpg.get_value("qc_direct_gain")
        direct_gain_db = round(direct_gain_db,1)#round to nearest .1 dB
        ac_space = dpg.get_value("qc_acoustic_space_combo")
        ac_space_int = CN.AC_SPACE_LIST_GUI.index(ac_space)
        ac_space_short = CN.AC_SPACE_LIST_SHORT[ac_space_int]
        hp_type = dpg.get_value("qc_brir_hp_type")
        pinna_comp_int = CN.HP_COMP_LIST.index(hp_type)
        pinna_comp = pinna_comp_int
        sample_rate = dpg.get_value('qc_wav_sample_rate')
        bit_depth = dpg.get_value('qc_wav_bit_depth')
        hrtf_symmetry = dpg.get_value('force_hrtf_symmetry')
        er_delay_time = dpg.get_value('er_delay_time_tag')
        er_delay_time = round(er_delay_time,1)#round to nearest .1 dB
        brir_dict=get_brir_dict()
        qc_brir_hrtf_short=brir_dict.get('qc_brir_hrtf_short')
        if full_name==True:
            brir_name = qc_brir_hrtf_short + ' '+ac_space_short + ' ' + str(direct_gain_db) + 'dB ' + CN.ROOM_TARGET_LIST_SHORT[room_target] + ' ' + CN.HP_COMP_LIST_SHORT[pinna_comp] + ' ' + sample_rate + ' ' + bit_depth + ' ' + hrtf_symmetry + ' ' + str(er_delay_time) 
        else:
            brir_name = qc_brir_hrtf_short + ' '+ac_space_short + ' ' + str(direct_gain_db) + 'dB ' + CN.ROOM_TARGET_LIST_SHORT[room_target] + ' ' + CN.HP_COMP_LIST_SHORT[pinna_comp]
    
        return brir_name
    
    def calc_hpcf_name(full_name=True):
        """ 
        GUI function to calculate hpcf from currently selected parameters
        """
        headphone = dpg.get_value('qc_headphone_list')
        sample = dpg.get_value('qc_sample_list')
        sample_rate = dpg.get_value('qc_wav_sample_rate')
        bit_depth = dpg.get_value('qc_wav_bit_depth')
        if full_name==True:
            filter_name = headphone + ' ' + sample + ' ' + sample_rate + ' ' + bit_depth
        else:
            filter_name = headphone + ' ' + sample
    
    
        return filter_name
    
    #
    ## misc tools and settings
    #

    def show_selected_folder(sender, files, cancel_pressed):
        """ 
        GUI function to process selected folder
        """
        if not cancel_pressed:
            base_folder_selected=files[0]
            ash_folder_selected=pjoin(base_folder_selected, CN.PROJECT_FOLDER)
            dpg.set_value('selected_folder_base', base_folder_selected)
            dpg.set_value('selected_folder_ash', ash_folder_selected)
            dpg.set_value('selected_folder_ash_tooltip', ash_folder_selected)
            #hesuvi path
            if 'EqualizerAPO' in base_folder_selected:
                hesuvi_path_selected = pjoin(base_folder_selected,'HeSuVi')#stored outside of project folder (within hesuvi installation)
            else:
                hesuvi_path_selected = pjoin(base_folder_selected, CN.PROJECT_FOLDER,'HeSuVi')#stored within project folder
            dpg.set_value('selected_folder_hesuvi', hesuvi_path_selected)
            dpg.set_value('selected_folder_hesuvi_tooltip', hesuvi_path_selected)
            save_settings()
            
    def reset_settings():
        """ 
        GUI function to reset settings
        """
        dpg.set_value("wav_sample_rate", sample_freq_default)
        dpg.set_value("wav_bit_depth", bit_depth_default)  
        dpg.set_value("brir_hp_type", brir_hp_type_default)
        
        dpg.set_value("brir_spat_res", spatial_res_default)
        dpg.set_value("rm_target_list", room_target_default)
        dpg.set_value("direct_gain", direct_gain_default)
        dpg.set_value("direct_gain_slider", direct_gain_default)
        dpg.set_value("acoustic_space_combo", ac_space_default)
        
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
        reset_hpcf_progress()
        reset_brir_progress()
        qc_reset_progress()
        e_apo_toggle_hpcf_gui(app_data=False)
        e_apo_toggle_brir_gui(app_data=False)
        
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

        save_settings()
        
        #log results
        log_string = 'Settings have been reset to default '
        hf.log_with_timestamp(log_string, logz)
        
    def save_settings(update_hpcf_pars=False,update_brir_pars=False):
        """ 
        GUI function to save settings
        """
        
        #load previous settings
        #try reading from settings.ini to get path
        try:
            #load settings
            config = configparser.ConfigParser()
            config.read(CN.SETTINGS_FILE)
            version_loaded = config['DEFAULT']['version']
            if __version__ == version_loaded:

                qc_brir_hp_type_loaded = config['DEFAULT']['qc_brir_headphone_type']
                qc_room_target_loaded=config['DEFAULT']['qc_brir_room_target']
                qc_direct_gain_loaded=(config['DEFAULT']['qc_brir_direct_gain'])
                qc_ac_space_loaded=config['DEFAULT']['qc_acoustic_space']
                qc_brand_loaded=config['DEFAULT']['qc_brand']
                qc_headphone_loaded=config['DEFAULT']['qc_headphone']
                qc_sample_loaded=config['DEFAULT']['qc_sample']
                qc_hrtf_loaded=config['DEFAULT']['qc_brir_hrtf']
                qc_brir_hrtf_type_loaded=config['DEFAULT']['qc_brir_hrtf_type']
                qc_brir_hrtf_dataset_loaded=config['DEFAULT']['qc_brir_hrtf_dataset']
                
            else:
                raise ValueError('Settings not loaded due to version mismatch')
            
        except:
            qc_brir_hp_type_loaded=qc_brir_hp_type_default
            qc_room_target_loaded=qc_room_target_default
            qc_direct_gain_loaded=qc_direct_gain_default
            qc_ac_space_loaded=qc_ac_space_default
            qc_brand_loaded=qc_brand_default
            qc_headphone_loaded=qc_headphone_default
            qc_sample_loaded=qc_sample_default
            qc_hrtf_loaded=hrtf_default
            qc_brir_hrtf_type_loaded=brir_hrtf_type_default
            qc_brir_hrtf_dataset_loaded=brir_hrtf_dataset_default
        
        #dont save hpcf settings unless flagged
        autoapply_hpcf=(dpg.get_value('qc_auto_apply_hpcf_sel'))
        if autoapply_hpcf == True:
            update_hpcf_pars=True
        


        #dont save current hpcf settings unless flagged
        if update_hpcf_pars == True:
            qc_headphone_sel_str=dpg.get_value('qc_headphone_list')
            qc_sample_sel_str=dpg.get_value('qc_sample_list')
            qc_brand_sel_str = hpcf_functions.get_brand(conn, qc_headphone_sel_str)
        else:
            qc_headphone_sel_str=qc_headphone_loaded
            qc_sample_sel_str=qc_sample_loaded
            qc_brand_sel_str = qc_brand_loaded

        #dont save brir settings unless flagged
        if update_brir_pars == True:
            #qc settings
            qc_hp_type_str = dpg.get_value('qc_brir_hp_type')
            qc_room_target_str = dpg.get_value('qc_rm_target_list')
            qc_direct_gain_str = str(dpg.get_value('qc_direct_gain'))
            qc_ac_space_str=dpg.get_value('qc_acoustic_space_combo')
            qc_hrtf_str = dpg.get_value('qc_brir_hrtf')
            qc_hrtf_type_str = str(dpg.get_value('qc_brir_hrtf_type'))
            qc_hrtf_dataset_str = str(dpg.get_value('qc_brir_hrtf_dataset'))
        else:
            #qc settings
            qc_hp_type_str = qc_brir_hp_type_loaded
            qc_room_target_str = qc_room_target_loaded
            qc_direct_gain_str = str(qc_direct_gain_loaded)
            qc_ac_space_str=qc_ac_space_loaded
            qc_hrtf_str = qc_hrtf_loaded
            qc_hrtf_type_str = qc_brir_hrtf_type_loaded
            qc_hrtf_dataset_str = qc_brir_hrtf_dataset_loaded


        try:
            #save folder name to config file
            config = configparser.ConfigParser()
            
            config['DEFAULT']['path'] = dpg.get_value('selected_folder_base')    # update
            config['DEFAULT']['sampling_frequency'] = dpg.get_value('wav_sample_rate') 
            config['DEFAULT']['bit_depth'] = dpg.get_value('wav_bit_depth')    # update
            config['DEFAULT']['brir_headphone_type'] = dpg.get_value('brir_hp_type')    # update
            config['DEFAULT']['brir_hrtf'] = dpg.get_value('brir_hrtf')    # update
            config['DEFAULT']['spatial_resolution'] = dpg.get_value('brir_spat_res')
            config['DEFAULT']['brir_room_target'] = dpg.get_value('rm_target_list')    # update
            config['DEFAULT']['brir_direct_gain'] = str(dpg.get_value('direct_gain'))    # update
            config['DEFAULT']['acoustic_space'] = dpg.get_value('acoustic_space_combo')
            config['DEFAULT']['version'] = __version__    # update
            config['DEFAULT']['fir_hpcf_exp'] = str(dpg.get_value('fir_hpcf_toggle'))
            config['DEFAULT']['fir_st_hpcf_exp'] = str(dpg.get_value('fir_st_hpcf_toggle'))
            config['DEFAULT']['eapo_hpcf_exp'] = str(dpg.get_value('eapo_hpcf_toggle'))
            config['DEFAULT']['geq_hpcf_exp'] = str(dpg.get_value('geq_hpcf_toggle'))
            config['DEFAULT']['geq_31_exp'] = str(dpg.get_value('geq_31_hpcf_toggle'))
            config['DEFAULT']['hesuvi_hpcf_exp'] = str(dpg.get_value('hesuvi_hpcf_toggle'))
            config['DEFAULT']['dir_brir_exp'] = str(dpg.get_value('dir_brir_toggle'))
            config['DEFAULT']['ts_brir_exp'] = str(dpg.get_value('ts_brir_toggle'))
            config['DEFAULT']['hesuvi_brir_exp'] = str(dpg.get_value('hesuvi_brir_toggle')) 
            config['DEFAULT']['eapo_brir_exp'] = str(dpg.get_value('eapo_brir_toggle'))
            config['DEFAULT']['sofa_brir_exp'] = str(dpg.get_value('sofa_brir_toggle'))
            config['DEFAULT']['auto_check_updates'] = str(dpg.get_value('check_updates_start_tag'))
            config['DEFAULT']['force_hrtf_symmetry'] = str(dpg.get_value('force_hrtf_symmetry'))
            config['DEFAULT']['er_delay_time'] = str(dpg.get_value('er_delay_time_tag'))
            config['DEFAULT']['mute_fl'] = str(dpg.get_value('e_apo_mute_fl'))
            config['DEFAULT']['mute_fr'] = str(dpg.get_value('e_apo_mute_fr'))
            config['DEFAULT']['mute_c'] = str(dpg.get_value('e_apo_mute_c'))
            config['DEFAULT']['mute_sl'] = str(dpg.get_value('e_apo_mute_sl'))
            config['DEFAULT']['mute_sr'] = str(dpg.get_value('e_apo_mute_sr'))
            config['DEFAULT']['mute_rl'] = str(dpg.get_value('e_apo_mute_rl'))
            config['DEFAULT']['mute_rr'] = str(dpg.get_value('e_apo_mute_rr'))
            config['DEFAULT']['gain_oa'] = str(dpg.get_value('e_apo_gain_oa'))
            config['DEFAULT']['gain_fl'] = str(dpg.get_value('e_apo_gain_fl'))
            config['DEFAULT']['gain_fr'] = str(dpg.get_value('e_apo_gain_fr'))
            config['DEFAULT']['gain_c'] = str(dpg.get_value('e_apo_gain_c'))
            config['DEFAULT']['gain_sl'] = str(dpg.get_value('e_apo_gain_sl'))
            config['DEFAULT']['gain_sr'] = str(dpg.get_value('e_apo_gain_sr'))
            config['DEFAULT']['gain_rl'] = str(dpg.get_value('e_apo_gain_rl'))
            config['DEFAULT']['gain_rr'] = str(dpg.get_value('e_apo_gain_rr'))
            config['DEFAULT']['elev_fl'] = str(dpg.get_value('e_apo_elev_angle_fl'))
            config['DEFAULT']['elev_fr'] = str(dpg.get_value('e_apo_elev_angle_fr'))
            config['DEFAULT']['elev_c'] = str(dpg.get_value('e_apo_elev_angle_c'))
            config['DEFAULT']['elev_sl'] = str(dpg.get_value('e_apo_elev_angle_sl'))
            config['DEFAULT']['elev_sr'] = str(dpg.get_value('e_apo_elev_angle_sr'))
            config['DEFAULT']['elev_rl'] = str(dpg.get_value('e_apo_elev_angle_rl'))
            config['DEFAULT']['elev_rr'] = str(dpg.get_value('e_apo_elev_angle_rr'))
            config['DEFAULT']['azim_fl'] = str(dpg.get_value('e_apo_az_angle_fl'))
            config['DEFAULT']['azim_fr'] = str(dpg.get_value('e_apo_az_angle_fr'))
            config['DEFAULT']['azim_c'] = str(dpg.get_value('e_apo_az_angle_c'))
            config['DEFAULT']['azim_sl'] = str(dpg.get_value('e_apo_az_angle_sl'))
            config['DEFAULT']['azim_sr'] = str(dpg.get_value('e_apo_az_angle_sr'))
            config['DEFAULT']['azim_rl'] = str(dpg.get_value('e_apo_az_angle_rl'))
            config['DEFAULT']['azim_rr'] = str(dpg.get_value('e_apo_az_angle_rr'))
            
            config['DEFAULT']['upmix_method'] = str(dpg.get_value('e_apo_upmix_method'))
            config['DEFAULT']['side_delay'] = str(dpg.get_value('e_apo_side_delay'))
            config['DEFAULT']['rear_delay'] = str(dpg.get_value('e_apo_rear_delay'))

            config['DEFAULT']['tab_selected'] = str(dpg.get_value('tab_bar'))
            
            config['DEFAULT']['enable_hpcf'] = str(dpg.get_value('e_apo_hpcf_conv'))
            config['DEFAULT']['auto_apply_hpcf'] = str(dpg.get_value('qc_auto_apply_hpcf_sel'))
            config['DEFAULT']['hpcf_current'] = str(dpg.get_value('qc_e_apo_curr_hpcf'))
            config['DEFAULT']['hpcf_selected'] = str(dpg.get_value('qc_e_apo_sel_hpcf'))
            config['DEFAULT']['enable_brir'] = str(dpg.get_value('e_apo_brir_conv'))

            config['DEFAULT']['brir_set_current'] = str(dpg.get_value('qc_e_apo_curr_brir_set'))
            config['DEFAULT']['brir_set_selected'] = str(dpg.get_value('qc_e_apo_sel_brir_set'))
            config['DEFAULT']['channel_config'] = str(dpg.get_value('audio_channels_combo'))
            config['DEFAULT']['prevent_clip'] = str(dpg.get_value('e_apo_prevent_clip'))
            config['DEFAULT']['qc_brir_headphone_type'] = qc_hp_type_str    # update
            
            config['DEFAULT']['qc_brir_room_target'] = qc_room_target_str    # update
            config['DEFAULT']['qc_brir_direct_gain'] = qc_direct_gain_str    # update
            config['DEFAULT']['qc_acoustic_space'] = qc_ac_space_str
            config['DEFAULT']['qc_brand'] = qc_brand_sel_str
            config['DEFAULT']['qc_headphone'] = qc_headphone_sel_str
            config['DEFAULT']['qc_sample'] = qc_sample_sel_str
            #hrtf selection related
            config['DEFAULT']['brir_hrtf_type'] = str(dpg.get_value('brir_hrtf_type'))
            config['DEFAULT']['brir_hrtf_dataset'] = str(dpg.get_value('brir_hrtf_dataset'))
            config['DEFAULT']['qc_brir_hrtf'] = qc_hrtf_str    # update
            config['DEFAULT']['qc_brir_hrtf_type'] = qc_hrtf_type_str
            config['DEFAULT']['qc_brir_hrtf_dataset'] = qc_hrtf_dataset_str
            config['DEFAULT']['sofa_exp_conv'] = str(dpg.get_value('sofa_exp_conv'))

            with open(CN.SETTINGS_FILE, 'w') as configfile:    # save
                config.write(configfile)

        except Exception as e: 
            logging.error(f"Failed to write to settings.ini Error: {e}")
            log_string = 'Failed to write to settings.ini'
            hf.log_with_timestamp(log_string, logz)
        
    

    def remove_brirs(sender, app_data, user_data):
        """ 
        GUI function to delete generated BRIRs
        """
        base_folder_selected=dpg.get_value('selected_folder_base')
        brir_export.remove_brirs(base_folder_selected, gui_logger=logz)    
        base_folder_selected=dpg.get_value('qc_selected_folder_base')
        brir_export.remove_brirs(base_folder_selected, gui_logger=logz)  
        #disable brir convolution
        dpg.set_value("qc_e_apo_sel_brir_set", 'Deleted')
        dpg.set_value("e_apo_brir_conv", False)
        e_apo_toggle_brir_gui(app_data=False)
        
        
        
    def remove_hpcfs(sender, app_data, user_data):
        """ 
        GUI function to remove generated HpCFs
        """
        base_folder_selected=dpg.get_value('selected_folder_base')
        hpcf_functions.remove_hpcfs(base_folder_selected, gui_logger=logz) 
        base_folder_selected=dpg.get_value('qc_selected_folder_base')
        hpcf_functions.remove_hpcfs(base_folder_selected, gui_logger=logz) 
        #disable hpcf convolution
        dpg.set_value("e_apo_hpcf_conv", False)
        dpg.set_value("qc_e_apo_sel_hpcf", 'Deleted')
        e_apo_toggle_hpcf_gui(app_data=False)
        

    #
    # Equalizer APO configuration functions
    #
    
    def e_apo_auto_apply_hpcf(sender, app_data, user_data):
        """ 
        GUI function to toggle auto apply hpcf convolution
        """
        if app_data == True:
            e_apo_toggle_hpcf_gui(app_data=True)
        

    def e_apo_toggle_hpcf_gui(sender=None, app_data=True):
        """ 
        GUI function to toggle hpcf convolution
        """
        aquire_config=True
        e_apo_toggle_hpcf_custom(app_data=app_data, aquire_config=aquire_config)
            
    def e_apo_toggle_hpcf_custom(app_data=True, aquire_config=True):
        """ 
        GUI function to toggle hpcf convolution
        """
        force_output=False
        if app_data == False:
            dpg.set_value("qc_e_apo_curr_hpcf", '')
            #call main config writer function
            if aquire_config==True or aquire_config==None:#custom parameter will be none if called by gui
                e_apo_config_acquire()
            #reset progress
            dpg.set_value("qc_progress_bar_hpcf", 0)
            dpg.configure_item("qc_progress_bar_hpcf", overlay = CN.PROGRESS_START)
        else:
            #check if saved hpcf set name is matching with currently selected params
            hpcf_name_full = calc_hpcf_name(full_name=True)
            hpcf_name = calc_hpcf_name(full_name=False)
            if hpcf_name in hpcf_name_full:#if only sample rate or bit depth changed, force write output
                force_output=True
            sel_hpcf_set=dpg.get_value('qc_e_apo_sel_hpcf')
            #if matching, enable hpcf conv in config
            if hpcf_name_full == sel_hpcf_set:
                dpg.set_value("e_apo_hpcf_conv", True)
                dpg.set_value("qc_e_apo_curr_hpcf", hpcf_name)
                dpg.set_value("qc_progress_bar_hpcf", 1)
                dpg.configure_item("qc_progress_bar_hpcf", overlay = CN.PROGRESS_FIN)
                if aquire_config==True or aquire_config==None:#custom parameter will be none if called by gui
                    e_apo_config_acquire()
            else:#else run hpcf processing from scratch
                qc_process_hpcfs(app_data=force_output)

    def get_brir_dict():
        """ 
        GUI function to get inputs relating to brirs and store in a dict, returns a dict
        """
                
        #brir related selections
        enable_brir_selected=dpg.get_value('e_apo_brir_conv')
        brir_set_folder=CN.FOLDER_BRIRS_LIVE
        brir_set_name=dpg.get_value('qc_e_apo_sel_brir_set')
        mute_fl_selected=dpg.get_value('e_apo_mute_fl')
        mute_fr_selected=dpg.get_value('e_apo_mute_fr')
        mute_c_selected=dpg.get_value('e_apo_mute_c')
        mute_sl_selected=dpg.get_value('e_apo_mute_sl')
        mute_sr_selected=dpg.get_value('e_apo_mute_sr')
        mute_rl_selected=dpg.get_value('e_apo_mute_rl')
        mute_rr_selected=dpg.get_value('e_apo_mute_rr')
        gain_oa_selected=dpg.get_value('e_apo_gain_oa')
        gain_fl_selected=dpg.get_value('e_apo_gain_fl')
        gain_fr_selected=dpg.get_value('e_apo_gain_fr')
        gain_c_selected=dpg.get_value('e_apo_gain_c')
        gain_sl_selected=dpg.get_value('e_apo_gain_sl')
        gain_sr_selected=dpg.get_value('e_apo_gain_sr')
        gain_rl_selected=dpg.get_value('e_apo_gain_rl')
        gain_rr_selected=dpg.get_value('e_apo_gain_rr')
        elev_fl_selected=dpg.get_value('e_apo_elev_angle_fl')
        elev_fr_selected=dpg.get_value('e_apo_elev_angle_fr')
        elev_c_selected=dpg.get_value('e_apo_elev_angle_c')
        elev_sl_selected=dpg.get_value('e_apo_elev_angle_sl')
        elev_sr_selected=dpg.get_value('e_apo_elev_angle_sr')
        elev_rl_selected=dpg.get_value('e_apo_elev_angle_rl')
        elev_rr_selected=dpg.get_value('e_apo_elev_angle_rr')
        azim_fl_selected=dpg.get_value('e_apo_az_angle_fl')
        azim_fr_selected=dpg.get_value('e_apo_az_angle_fr')
        azim_c_selected=dpg.get_value('e_apo_az_angle_c')
        azim_sl_selected=dpg.get_value('e_apo_az_angle_sl')
        azim_sr_selected=dpg.get_value('e_apo_az_angle_sr')
        azim_rl_selected=dpg.get_value('e_apo_az_angle_rl')
        azim_rr_selected=dpg.get_value('e_apo_az_angle_rr')
        
        #hesuvi brir related selections
        h_elev_fl_selected=dpg.get_value('hesuvi_elev_angle_fl')
        h_elev_fr_selected=dpg.get_value('hesuvi_elev_angle_fr')
        h_elev_c_selected=dpg.get_value('hesuvi_elev_angle_c')
        h_elev_sl_selected=dpg.get_value('hesuvi_elev_angle_sl')
        h_elev_sr_selected=dpg.get_value('hesuvi_elev_angle_sr')
        h_elev_rl_selected=dpg.get_value('hesuvi_elev_angle_rl')
        h_elev_rr_selected=dpg.get_value('hesuvi_elev_angle_rr')
        h_azim_fl_selected=dpg.get_value('hesuvi_az_angle_fl')
        h_azim_fr_selected=dpg.get_value('hesuvi_az_angle_fr')
        h_azim_c_selected=dpg.get_value('hesuvi_az_angle_c')
        h_azim_sl_selected=dpg.get_value('hesuvi_az_angle_sl')
        h_azim_sr_selected=dpg.get_value('hesuvi_az_angle_sr')
        h_azim_rl_selected=dpg.get_value('hesuvi_az_angle_rl')
        h_azim_rr_selected=dpg.get_value('hesuvi_az_angle_rr')
        
        #hrtf type, dataset, hrtf name
        brir_hrtf_type_selected=dpg.get_value('brir_hrtf_type')
        #print(brir_hrtf_type_selected)
        brir_hrtf_dataset_selected=dpg.get_value('brir_hrtf_dataset')
        brir_hrtf_selected=dpg.get_value('brir_hrtf')
        brir_hrtf_short = hrir_processing.get_name_short(listener_type=brir_hrtf_type_selected, dataset_name=brir_hrtf_dataset_selected, name_gui=brir_hrtf_selected)
        qc_brir_hrtf_type_selected=dpg.get_value('qc_brir_hrtf_type')
        qc_brir_hrtf_dataset_selected=dpg.get_value('qc_brir_hrtf_dataset')
        qc_brir_hrtf_selected=dpg.get_value('qc_brir_hrtf')
        qc_brir_hrtf_short = hrir_processing.get_name_short(listener_type=qc_brir_hrtf_type_selected, dataset_name=qc_brir_hrtf_dataset_selected, name_gui=qc_brir_hrtf_selected)
        
        elev_list = [
             int(elev_fl_selected),
             int(elev_fr_selected),
             int(elev_c_selected),
             int(elev_sl_selected),
             int(elev_sr_selected),
             int(elev_rl_selected),
             int(elev_rr_selected),
         ]
        azim_list = [
             int(azim_fl_selected),
             int(azim_fr_selected),
             int(azim_c_selected),
             int(azim_sl_selected),
             int(azim_sr_selected),
             int(azim_rl_selected),
             int(azim_rr_selected),
         ]
     
        brir_dict = {'enable_conv': enable_brir_selected, 'brir_set_folder': brir_set_folder, 'brir_set_name': brir_set_name, 'mute_fl': mute_fl_selected, 'mute_fr': mute_fr_selected, 'mute_c': mute_c_selected, 'mute_sl': mute_sl_selected,
                     'mute_sr': mute_sr_selected, 'mute_rl': mute_rl_selected, 'mute_rr': mute_rr_selected, 'gain_oa': gain_oa_selected, 'gain_fl': gain_fl_selected, 'gain_fr': gain_fr_selected, 'gain_c': gain_c_selected,
                     'gain_sl': gain_sl_selected, 'gain_sr': gain_sr_selected, 'gain_rl': gain_rl_selected, 'gain_rr': gain_rr_selected, 'elev_fl': elev_fl_selected, 'elev_fr': elev_fr_selected,
                     'elev_c': elev_c_selected, 'elev_sl': elev_sl_selected, 'elev_sr': elev_sr_selected, 'elev_rl': elev_rl_selected, 'elev_rr': elev_rr_selected, 'azim_fl': azim_fl_selected,
                     'azim_fr': azim_fr_selected, 'azim_c': azim_c_selected, 'azim_sl': azim_sl_selected, 'azim_sr': azim_sr_selected, 'azim_rl': azim_rl_selected, 'azim_rr': azim_rr_selected, 'elev_list': elev_list, 'azim_list': azim_list,
                     'brir_hrtf_type': brir_hrtf_type_selected, 'brir_hrtf_dataset': brir_hrtf_dataset_selected, 'brir_hrtf': brir_hrtf_selected, 'brir_hrtf_short': brir_hrtf_short,  
                     'qc_brir_hrtf_type': qc_brir_hrtf_type_selected, 'qc_brir_hrtf_dataset': qc_brir_hrtf_dataset_selected, 'qc_brir_hrtf': qc_brir_hrtf_selected, 'qc_brir_hrtf_short': qc_brir_hrtf_short,
                     'h_elev_fl': h_elev_fl_selected, 'h_elev_fr': h_elev_fr_selected,
                     'h_elev_c': h_elev_c_selected, 'h_elev_sl': h_elev_sl_selected, 'h_elev_sr': h_elev_sr_selected, 'h_elev_rl': h_elev_rl_selected, 'h_elev_rr': h_elev_rr_selected, 'h_azim_fl': h_azim_fl_selected,
                     'h_azim_fr': h_azim_fr_selected, 'h_azim_c': h_azim_c_selected, 'h_azim_sl': h_azim_sl_selected, 'h_azim_sr': h_azim_sr_selected, 'h_azim_rl': h_azim_rl_selected, 'h_azim_rr': h_azim_rr_selected,}
  
        return brir_dict
        


    def e_apo_config_acquire(estimate_gain=True):
        """ 
        GUI function to acquire lock on function to write updates to custom E-APO config
        """

        e_apo_conf_lock.acquire()
        e_apo_config_write(estimate_gain=estimate_gain)
        e_apo_conf_lock.release()
        
    def e_apo_config_acquire_gui(sender=None, app_data=None):
        """ 
        GUI function to acquire lock on function to write updates to custom E-APO config
        """
        estimate_gain=True
        e_apo_conf_lock.acquire()
        e_apo_config_write(estimate_gain=estimate_gain)
        e_apo_conf_lock.release()
          

    def e_apo_config_write(estimate_gain=True):
        """ 
        GUI function to write updates to custom E-APO config
        """
        base_folder_selected=dpg.get_value('qc_selected_folder_base')
        
        #hpcf related selections
        enable_hpcf_selected=dpg.get_value('e_apo_hpcf_conv')
        headphone_selected=dpg.get_value('qc_headphone_list')
        sample_selected=dpg.get_value('qc_sample_list')
        #brand_selected=dpg.get_value('qc_brand_list')
        brand_selected = hpcf_functions.get_brand(conn, headphone_selected)
        
        hpcf_dict = {'enable_conv': enable_hpcf_selected, 'brand': brand_selected, 'headphone': headphone_selected, 'sample': sample_selected}
        
        #brir related selections
        enable_brir_selected=dpg.get_value('e_apo_brir_conv')
        brir_set_folder=CN.FOLDER_BRIRS_LIVE
        gain_oa_selected=dpg.get_value('e_apo_gain_oa')

        brir_dict = get_brir_dict()
  
        audio_channels=dpg.get_value('audio_channels_combo')
        upmix_method=dpg.get_value('e_apo_upmix_method')
        side_delay=dpg.get_value('e_apo_side_delay')
        rear_delay=dpg.get_value('e_apo_rear_delay')
        
        #get spatial resolution for this brir set
        spatial_res_sel = 0
        
        #run function to write custom config
        gain_conf = e_apo_config_creation.write_ash_e_apo_config(primary_path=base_folder_selected, hpcf_dict=hpcf_dict, brir_dict=brir_dict, audio_channels=audio_channels, gui_logger=logz, spatial_res=spatial_res_sel, upmix_method=upmix_method, side_delay=side_delay, rear_delay=rear_delay)
 
        #run function to load the custom config file in config.txt
        gain_oa_selected=dpg.get_value('e_apo_gain_oa')
        if enable_hpcf_selected == True or enable_brir_selected == True or gain_oa_selected == CN.EAPO_MUTE_GAIN:
            load_config = True
        else:
            load_config = False
            
        prevent_clipping = dpg.get_value('e_apo_prevent_clip')
  
        #also update estimated peak gain
        if estimate_gain == True:
            #average gain
            est_pk_gain_2 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '2.0 Stereo', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder, calc_mode=1))
            dpg.set_value("e_apo_gain_avg_2_0", str(est_pk_gain_2))
            est_pk_gain_5 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '5.1 Surround', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder, calc_mode=1))
            dpg.set_value("e_apo_gain_avg_5_1", str(est_pk_gain_5))
            est_pk_gain_7 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '7.1 Surround', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder, calc_mode=1))
            dpg.set_value("e_apo_gain_avg_7_1", str(est_pk_gain_7))
            #peak gain
            est_pk_gain_2 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '2.0 Stereo', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder))
            dpg.set_value("e_apo_gain_peak_2_0", str(est_pk_gain_2))
            est_pk_gain_5 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '5.1 Surround', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder))
            dpg.set_value("e_apo_gain_peak_5_1", str(est_pk_gain_5))
            est_pk_gain_7 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '7.1 Surround', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder))
            dpg.set_value("e_apo_gain_peak_7_1", str(est_pk_gain_7))
            
            
            #if clipping prevention enabled, grab 2.0 peak gain, calc new gain and rewrite the custom config with gain override
            one_chan_mute=False
            mute_fl=brir_dict.get('mute_fl')
            mute_fr=brir_dict.get('mute_fr')
            if (mute_fl == True or mute_fr == True):#if at least one channel is muted, dont adjust gain
                one_chan_mute=True
            
            if prevent_clipping == True and load_config == True and one_chan_mute == False:
                gain_override = float(est_pk_gain_2*-1)-0.1#change polarity and reduce slightly
                gain_override = min(gain_override, 20.0)#limit to max of 20db
                dpg.set_value("e_apo_gain_oa", gain_oa_selected+gain_override)
                #run function to write custom config
                gain_conf = e_apo_config_creation.write_ash_e_apo_config(primary_path=base_folder_selected, hpcf_dict=hpcf_dict, brir_dict=brir_dict, audio_channels=audio_channels, gui_logger=logz, spatial_res=spatial_res_sel, gain_override=gain_override, upmix_method=upmix_method, side_delay=side_delay, rear_delay=rear_delay)
                #run function to load the custom config file in config.txt
                #if true, edit config.txt to include the custom config
                e_apo_config_creation.include_ash_e_apo_config(primary_path=base_folder_selected, enabled=load_config)
                #average gain
                est_pk_gain_2 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '2.0 Stereo', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder, calc_mode=1))
                dpg.set_value("e_apo_gain_avg_2_0", str(est_pk_gain_2))
                est_pk_gain_5 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '5.1 Surround', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder, calc_mode=1))
                dpg.set_value("e_apo_gain_avg_5_1", str(est_pk_gain_5))
                est_pk_gain_7 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '7.1 Surround', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder, calc_mode=1))
                dpg.set_value("e_apo_gain_avg_7_1", str(est_pk_gain_7))
                #peak gain
                est_pk_gain_2 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '2.0 Stereo', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder))
                dpg.set_value("e_apo_gain_peak_2_0", str(est_pk_gain_2))
                est_pk_gain_5 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '5.1 Surround', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder))
                dpg.set_value("e_apo_gain_peak_5_1", str(est_pk_gain_5))
                est_pk_gain_7 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '7.1 Surround', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder))
                dpg.set_value("e_apo_gain_peak_7_1", str(est_pk_gain_7))
            else:
                e_apo_config_creation.include_ash_e_apo_config(primary_path=base_folder_selected, enabled=load_config)
        else:
            #edit config.txt to include the custom config - only run once
            e_apo_config_creation.include_ash_e_apo_config(primary_path=base_folder_selected, enabled=load_config)
            
        #also save settings
        save_settings()   


    def e_apo_adjust_preamp(sender=None, app_data=None):
        """ 
        GUI function to process updates in E-APO config section
        """
        #turn off auto adjust preamp option
        dpg.set_value("e_apo_prevent_clip", False)
        e_apo_config_acquire()




    def e_apo_update_direction(sender=None, app_data=None, aquire_config=False, brir_dict_new={}):
        """ 
        GUI function to process updates to directions in E-APO config section
        """
        
        if brir_dict_new:
            #if brir dict is provided, grab values
            elev_fl=brir_dict_new.get('elev_fl')
            elev_fr=brir_dict_new.get('elev_fr')
            elev_c=brir_dict_new.get('elev_c')
            elev_sl=brir_dict_new.get('elev_sl')
            elev_sr=brir_dict_new.get('elev_sr')
            elev_rl=brir_dict_new.get('elev_rl')
            elev_rr=brir_dict_new.get('elev_rr')
            azim_fl=brir_dict_new.get('azim_fl')
            azim_fr=brir_dict_new.get('azim_fr')
            azim_c=brir_dict_new.get('azim_c')
            azim_sl=brir_dict_new.get('azim_sl')
            azim_sr=brir_dict_new.get('azim_sr')
            azim_rl=brir_dict_new.get('azim_rl')
            azim_rr=brir_dict_new.get('azim_rr')
            #then update gui values to align
            dpg.set_value("e_apo_elev_angle_fl", elev_fl)
            dpg.set_value("e_apo_elev_angle_fr", elev_fr)
            dpg.set_value("e_apo_elev_angle_c", elev_c)
            dpg.set_value("e_apo_elev_angle_sl", elev_sl)
            dpg.set_value("e_apo_elev_angle_sr", elev_sr)
            dpg.set_value("e_apo_elev_angle_rl", elev_rl)
            dpg.set_value("e_apo_elev_angle_rr", elev_rr)
            dpg.set_value("e_apo_az_angle_fl", azim_fl)
            dpg.set_value("e_apo_az_angle_fr", azim_fr)
            dpg.set_value("e_apo_az_angle_c", azim_c)
            dpg.set_value("e_apo_az_angle_sl", azim_sl)
            dpg.set_value("e_apo_az_angle_sr", azim_sr)
            dpg.set_value("e_apo_az_angle_rl", azim_rl)
            dpg.set_value("e_apo_az_angle_rr", azim_rr)
        else:
            #else grab azimuths from gui elements and proceed. No update required to gui values
            brir_dict_new=get_brir_dict()
            azim_fl=brir_dict_new.get('azim_fl')
            azim_fr=brir_dict_new.get('azim_fr')
            azim_c=brir_dict_new.get('azim_c')
            azim_sl=brir_dict_new.get('azim_sl')
            azim_sr=brir_dict_new.get('azim_sr')
            azim_rl=brir_dict_new.get('azim_rl')
            azim_rr=brir_dict_new.get('azim_rr')
        
        #update gui elements and write new config 
  
        #update each azimuth based drawing
        azimuth=int(azim_fl)
        dpg.apply_transform("fl_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("fl_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_fr)
        dpg.apply_transform("fr_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("fr_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_c)
        dpg.apply_transform("c_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("c_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_sl)
        dpg.apply_transform("sl_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("sl_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_sr)
        dpg.apply_transform("sr_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("sr_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_rl)
        dpg.apply_transform("rl_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("rl_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_rr)
        dpg.apply_transform("rr_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("rr_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
            
        #finally rewrite config file
        if aquire_config == True or aquire_config == None:#custom parameter will be none if called by gui
            e_apo_config_acquire()


    def e_apo_activate_direction_gui(sender=None, app_data=None):
        """ 
        GUI function to process updates to directions in E-APO config section
        """
        aquire_config=True 
        force_reset=False
        e_apo_activate_direction(aquire_config=aquire_config, force_reset=force_reset)

    def e_apo_activate_direction(aquire_config=False, force_reset=False):
        """ 
        GUI function to process updates to directions in E-APO config section
        """
        #get current selections
        base_folder_selected=dpg.get_value('qc_selected_folder_base')
        channels_selected=dpg.get_value('audio_channels_combo')
        brir_set_folder=CN.FOLDER_BRIRS_LIVE
        brir_dict=get_brir_dict()
        #print(brir_dict)
        
        #Get brir_dict for desired directions, store in a gui element
        dpg.configure_item('qc_e_apo_sel_brir_set',user_data=brir_dict)
        
        #run function to check if all brirs currently exist (returns true if brirs are disabled)
        all_brirs_found = e_apo_config_creation.dataset_all_brirs_found(primary_path=base_folder_selected, brir_set=brir_set_folder, brir_dict=brir_dict, channel_config = channels_selected)
        
        #if some files are missing (due to reduced dataset size)
        if all_brirs_found == False:
            
            try:
                #load previous settings
                config = configparser.ConfigParser()
                config.read(CN.SETTINGS_FILE)
                version_loaded = config['DEFAULT']['version']
                if __version__ == version_loaded:
                    e_apo_elev_angle_fl_loaded=int(config['DEFAULT']['elev_fl'])
                    e_apo_elev_angle_fr_loaded=int(config['DEFAULT']['elev_fr'])
                    e_apo_elev_angle_c_loaded=int(config['DEFAULT']['elev_c'])
                    e_apo_elev_angle_sl_loaded=int(config['DEFAULT']['elev_sl'])
                    e_apo_elev_angle_sr_loaded=int(config['DEFAULT']['elev_sr'])
                    e_apo_elev_angle_rl_loaded=int(config['DEFAULT']['elev_rl'])
                    e_apo_elev_angle_rr_loaded=int(config['DEFAULT']['elev_rr'])
                    e_apo_az_angle_fl_loaded=int(config['DEFAULT']['azim_fl'])
                    e_apo_az_angle_fr_loaded=int(config['DEFAULT']['azim_fr'])  
                    e_apo_az_angle_c_loaded=int(config['DEFAULT']['azim_c'] )
                    e_apo_az_angle_sl_loaded=int(config['DEFAULT']['azim_sl'])
                    e_apo_az_angle_sr_loaded=int(config['DEFAULT']['azim_sr'] )
                    e_apo_az_angle_rl_loaded=int(config['DEFAULT']['azim_rl'])
                    e_apo_az_angle_rr_loaded=int(config['DEFAULT']['azim_rr'])
                    
                else:
                    raise ValueError('Settings not loaded due to version mismatch')
                    
                #update directions to previously saved values
                dpg.set_value("e_apo_elev_angle_fl", e_apo_elev_angle_fl_loaded)
                dpg.set_value("e_apo_elev_angle_fr", e_apo_elev_angle_fr_loaded)
                dpg.set_value("e_apo_elev_angle_c", e_apo_elev_angle_c_loaded)
                dpg.set_value("e_apo_elev_angle_sl", e_apo_elev_angle_sl_loaded)
                dpg.set_value("e_apo_elev_angle_sr", e_apo_elev_angle_sr_loaded)
                dpg.set_value("e_apo_elev_angle_rl", e_apo_elev_angle_rl_loaded)
                dpg.set_value("e_apo_elev_angle_rr", e_apo_elev_angle_rr_loaded)
                dpg.set_value("e_apo_az_angle_fl", e_apo_az_angle_fl_loaded)
                dpg.set_value("e_apo_az_angle_fr", e_apo_az_angle_fr_loaded)
                dpg.set_value("e_apo_az_angle_c", e_apo_az_angle_c_loaded)
                dpg.set_value("e_apo_az_angle_sl", e_apo_az_angle_sl_loaded)
                dpg.set_value("e_apo_az_angle_sr", e_apo_az_angle_sr_loaded)
                dpg.set_value("e_apo_az_angle_rl", e_apo_az_angle_rl_loaded)
                dpg.set_value("e_apo_az_angle_rr", e_apo_az_angle_rr_loaded)
                
            except:
                pass
            
            log_string = 'Selected direction was not found'
            hf.log_with_timestamp(log_string, logz)
            
            brir_dict_list=dpg.get_item_user_data("e_apo_brir_conv")
            
            #reset progress and disable brir conv as not started yet
            if force_reset == True and not brir_dict_list:#only when triggered by apply button or toggle and no stored brir data
                dpg.set_value("qc_e_apo_curr_brir_set", '')
                dpg.set_value("qc_progress_bar_brir", 0)
                dpg.configure_item("qc_progress_bar_brir", overlay = CN.PROGRESS_START)
                dpg.set_value("e_apo_brir_conv", False)
            
            #If files not found, check if dict list is not empty, 
            if brir_dict_list:#list is not empty
                #if list populated, export new wavs from dict list for matching directions 
                log_string = 'Exporting missing direction(s)'
                hf.log_with_timestamp(log_string, logz)
                #attempt to export wavs
                e_apo_toggle_brir_custom(app_data=True, use_dict_list=True, force_run_process=True)#do not use button due to cancellation logic
            else:
                #Update user data to flag that stored BRIR dict should be used for azimuths
                dpg.configure_item('qc_e_apo_curr_brir_set',user_data=True)
                log_string = 'Processing and exporting missing direction(s)'
                hf.log_with_timestamp(log_string, logz)
                #If list not populated, trigger new brir processing (likely restarted app)
                e_apo_toggle_brir_custom(app_data=True, use_dict_list=False, force_run_process=True)#do not use button due to cancellation logic
                
  
        #use updated azimuths
        brir_dict_new=get_brir_dict()
        azim_fl_selected=brir_dict_new.get('azim_fl')
        azim_fr_selected=brir_dict_new.get('azim_fr')
        azim_c_selected=brir_dict_new.get('azim_c')
        azim_sl_selected=brir_dict_new.get('azim_sl')
        azim_sr_selected=brir_dict_new.get('azim_sr')
        azim_rl_selected=brir_dict_new.get('azim_rl')
        azim_rr_selected=brir_dict_new.get('azim_rr')
        
        #update gui elements and write new config 
  
        #update each azimuth based drawing
        azimuth=int(azim_fl_selected)
        dpg.apply_transform("fl_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("fl_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_fr_selected)
        dpg.apply_transform("fr_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("fr_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_c_selected)
        dpg.apply_transform("c_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("c_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_sl_selected)
        dpg.apply_transform("sl_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("sl_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_sr_selected)
        dpg.apply_transform("sr_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("sr_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_rl_selected)
        dpg.apply_transform("rl_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("rl_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_rr_selected)
        dpg.apply_transform("rr_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("rr_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
            
        #finally rewrite config file
        if aquire_config == True:#custom parameter will be none if called by gui
            e_apo_config_acquire()




    def mute_all_chans():
        """ 
        GUI function to mute channel config in E-APO config section
        """
        dpg.set_value("e_apo_mute_fl", True)
        dpg.set_value("e_apo_mute_fr", True)
        dpg.set_value("e_apo_mute_c", True)
        dpg.set_value("e_apo_mute_sl", True)
        dpg.set_value("e_apo_mute_sr", True)
        dpg.set_value("e_apo_mute_rl", True)
        dpg.set_value("e_apo_mute_rr", True)

    def unmute_all_chans():
        """ 
        GUI function to unmute channel config in E-APO config section
        """
        dpg.set_value("e_apo_mute_fl", e_apo_mute_default)
        dpg.set_value("e_apo_mute_fr", e_apo_mute_default)
        dpg.set_value("e_apo_mute_c", e_apo_mute_default)
        dpg.set_value("e_apo_mute_sl", e_apo_mute_default)
        dpg.set_value("e_apo_mute_sr", e_apo_mute_default)
        dpg.set_value("e_apo_mute_rl", e_apo_mute_default)
        dpg.set_value("e_apo_mute_rr", e_apo_mute_default)

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
        e_apo_select_channels(app_data=audio_channels_default,aquire_config=False)
        #update drawings for azimuths
        e_apo_activate_direction()
        
        #finally rewrite config file
        e_apo_config_acquire()
  
    def e_apo_select_channels_gui(sender=None, app_data=None):
        """ 
        GUI function to process updates in E-APO config section
        """
        aquire_config=True
        e_apo_select_channels(app_data=app_data, aquire_config=aquire_config)  
  
    def e_apo_select_channels(app_data=None, aquire_config=True):
        """ 
        GUI function to process updates in E-APO config section
        """

        try:
            if app_data == '2.0 Stereo Upmix to 7.1':
                dpg.configure_item("upmixing_table", show=True)
            else:
                dpg.configure_item("upmixing_table", show=False)
                
            if app_data == '2.0 Stereo':  
                dpg.configure_item("e_apo_mute_c", show=False)
                dpg.configure_item("e_apo_mute_sl", show=False)
                dpg.configure_item("e_apo_mute_sr", show=False)
                dpg.configure_item("e_apo_mute_rl", show=False)
                dpg.configure_item("e_apo_mute_rr", show=False) 
                dpg.configure_item("e_apo_gain_c", show=False)
                dpg.configure_item("e_apo_gain_sl", show=False)
                dpg.configure_item("e_apo_gain_sr", show=False)
                dpg.configure_item("e_apo_gain_rl", show=False)
                dpg.configure_item("e_apo_gain_rr", show=False)
                dpg.configure_item("e_apo_elev_angle_c", show=False)
                dpg.configure_item("e_apo_elev_angle_sl", show=False)
                dpg.configure_item("e_apo_elev_angle_sr", show=False)
                dpg.configure_item("e_apo_elev_angle_rl", show=False)
                dpg.configure_item("e_apo_elev_angle_rr", show=False)
                dpg.configure_item("e_apo_az_angle_c", show=False)
                dpg.configure_item("e_apo_az_angle_sl", show=False)
                dpg.configure_item("e_apo_az_angle_sr", show=False)
                dpg.configure_item("e_apo_az_angle_rl", show=False)
                dpg.configure_item("e_apo_az_angle_rr", show=False)
                dpg.configure_item("c_drawing", show=False)
                dpg.configure_item("c_drawing_inner", show=False)
                dpg.configure_item("sl_drawing", show=False)
                dpg.configure_item("sl_drawing_inner", show=False)
                dpg.configure_item("sr_drawing", show=False)
                dpg.configure_item("sr_drawing_inner", show=False)
                dpg.configure_item("rl_drawing", show=False)
                dpg.configure_item("rl_drawing_inner", show=False)
                dpg.configure_item("rr_drawing", show=False)
                dpg.configure_item("rr_drawing_inner", show=False)
                dpg.configure_item("e_apo_gain_peak_5_1", show=False)
                dpg.configure_item("e_apo_gain_peak_7_1", show=False)
                dpg.configure_item("e_apo_gain_avg_5_1", show=False)
                dpg.configure_item("e_apo_gain_avg_7_1", show=False)
    
            elif app_data == '5.1 Surround':  
                dpg.configure_item("e_apo_mute_c", show=True)
                dpg.configure_item("e_apo_mute_sl", show=False)
                dpg.configure_item("e_apo_mute_sr", show=False)
                dpg.configure_item("e_apo_mute_rl", show=True)
                dpg.configure_item("e_apo_mute_rr", show=True)
                dpg.configure_item("e_apo_gain_c", show=True)
                dpg.configure_item("e_apo_gain_sl", show=False)
                dpg.configure_item("e_apo_gain_sr", show=False)
                dpg.configure_item("e_apo_gain_rl", show=True)
                dpg.configure_item("e_apo_gain_rr", show=True)
                dpg.configure_item("e_apo_elev_angle_c", show=True)
                dpg.configure_item("e_apo_elev_angle_sl", show=False)
                dpg.configure_item("e_apo_elev_angle_sr", show=False)
                dpg.configure_item("e_apo_elev_angle_rl", show=True)
                dpg.configure_item("e_apo_elev_angle_rr", show=True)
                dpg.configure_item("e_apo_az_angle_c", show=True)
                dpg.configure_item("e_apo_az_angle_sl", show=False)
                dpg.configure_item("e_apo_az_angle_sr", show=False)
                dpg.configure_item("e_apo_az_angle_rl", show=True)
                dpg.configure_item("e_apo_az_angle_rr", show=True)
                dpg.configure_item("c_drawing", show=True)
                dpg.configure_item("c_drawing_inner", show=True)
                dpg.configure_item("sl_drawing", show=False)
                dpg.configure_item("sl_drawing_inner", show=False)
                dpg.configure_item("sr_drawing", show=False)
                dpg.configure_item("sr_drawing_inner", show=False)
                dpg.configure_item("rl_drawing", show=True)
                dpg.configure_item("rl_drawing_inner", show=True)
                dpg.configure_item("rr_drawing", show=True)
                dpg.configure_item("rr_drawing_inner", show=True)
                dpg.configure_item("e_apo_gain_peak_5_1", show=True)
                dpg.configure_item("e_apo_gain_peak_7_1", show=False)
                dpg.configure_item("e_apo_gain_avg_5_1", show=True)
                dpg.configure_item("e_apo_gain_avg_7_1", show=False)
            
            elif app_data == '7.1 Surround' or app_data == '2.0 Stereo Upmix to 7.1':
                dpg.configure_item("e_apo_mute_c", show=True)
                dpg.configure_item("e_apo_mute_sl", show=True)
                dpg.configure_item("e_apo_mute_sr", show=True)
                dpg.configure_item("e_apo_mute_rl", show=True)
                dpg.configure_item("e_apo_mute_rr", show=True)
                dpg.configure_item("e_apo_gain_c", show=True)
                dpg.configure_item("e_apo_gain_sl", show=True)
                dpg.configure_item("e_apo_gain_sr", show=True)
                dpg.configure_item("e_apo_gain_rl", show=True)
                dpg.configure_item("e_apo_gain_rr", show=True)
                dpg.configure_item("e_apo_elev_angle_c", show=True)
                dpg.configure_item("e_apo_elev_angle_sl", show=True)
                dpg.configure_item("e_apo_elev_angle_sr", show=True)
                dpg.configure_item("e_apo_elev_angle_rl", show=True)
                dpg.configure_item("e_apo_elev_angle_rr", show=True)
                dpg.configure_item("e_apo_az_angle_c", show=True)
                dpg.configure_item("e_apo_az_angle_sl", show=True)
                dpg.configure_item("e_apo_az_angle_sr", show=True)
                dpg.configure_item("e_apo_az_angle_rl", show=True)
                dpg.configure_item("e_apo_az_angle_rr", show=True) 
                dpg.configure_item("c_drawing", show=True)
                dpg.configure_item("c_drawing_inner", show=True)
                dpg.configure_item("sl_drawing", show=True)
                dpg.configure_item("sl_drawing_inner", show=True)
                dpg.configure_item("sr_drawing", show=True)
                dpg.configure_item("sr_drawing_inner", show=True)
                dpg.configure_item("rl_drawing", show=True)
                dpg.configure_item("rl_drawing_inner", show=True)
                dpg.configure_item("rr_drawing", show=True)
                dpg.configure_item("rr_drawing_inner", show=True)
                dpg.configure_item("e_apo_gain_peak_5_1", show=True)
                dpg.configure_item("e_apo_gain_peak_7_1", show=True)
                dpg.configure_item("e_apo_gain_avg_5_1", show=True)
                dpg.configure_item("e_apo_gain_avg_7_1", show=True)
                
            elif app_data == '7.1 Downmix to Stereo':
                dpg.configure_item("e_apo_mute_c", show=True)
                dpg.configure_item("e_apo_mute_sl", show=True)
                dpg.configure_item("e_apo_mute_sr", show=True)
                dpg.configure_item("e_apo_mute_rl", show=True)
                dpg.configure_item("e_apo_mute_rr", show=True)
                dpg.configure_item("e_apo_gain_c", show=True)
                dpg.configure_item("e_apo_gain_sl", show=True)
                dpg.configure_item("e_apo_gain_sr", show=True)
                dpg.configure_item("e_apo_gain_rl", show=True)
                dpg.configure_item("e_apo_gain_rr", show=True)
                dpg.configure_item("e_apo_elev_angle_c", show=False)
                dpg.configure_item("e_apo_elev_angle_sl", show=False)
                dpg.configure_item("e_apo_elev_angle_sr", show=False)
                dpg.configure_item("e_apo_elev_angle_rl", show=False)
                dpg.configure_item("e_apo_elev_angle_rr", show=False)
                dpg.configure_item("e_apo_az_angle_c", show=False)
                dpg.configure_item("e_apo_az_angle_sl", show=False)
                dpg.configure_item("e_apo_az_angle_sr", show=False)
                dpg.configure_item("e_apo_az_angle_rl", show=False)
                dpg.configure_item("e_apo_az_angle_rr", show=False)
                dpg.configure_item("c_drawing", show=False)
                dpg.configure_item("c_drawing_inner", show=False)
                dpg.configure_item("sl_drawing", show=False)
                dpg.configure_item("sl_drawing_inner", show=False)
                dpg.configure_item("sr_drawing", show=False)
                dpg.configure_item("sr_drawing_inner", show=False)
                dpg.configure_item("rl_drawing", show=False)
                dpg.configure_item("rl_drawing_inner", show=False)
                dpg.configure_item("rr_drawing", show=False)
                dpg.configure_item("rr_drawing_inner", show=False)
                dpg.configure_item("e_apo_gain_peak_5_1", show=True)
                dpg.configure_item("e_apo_gain_peak_7_1", show=True)
                dpg.configure_item("e_apo_gain_avg_5_1", show=True)
                dpg.configure_item("e_apo_gain_avg_7_1", show=True)

        except Exception as e:
            print(f"An error occurred: {e}")  # Log the error
            
        #finally rewrite config file
        if aquire_config == True:#custom parameter will be none if called by gui
            e_apo_config_acquire()
            

 
    def reset_brir_progress():
        """ 
        GUI function to reset progress bar
        """
        #if not already running
        #thread bool
        process_brirs_running=dpg.get_item_user_data("brir_tag")
        if process_brirs_running == False:
            #reset progress bar
            dpg.set_value("progress_bar_brir", 0)
            dpg.configure_item("progress_bar_brir", overlay = CN.PROGRESS_START_ALT)

    def reset_hpcf_progress():
        """ 
        GUI function to reset progress bar
        """
        dpg.set_value("progress_bar_hpcf", 0)
        dpg.configure_item("progress_bar_hpcf", overlay = CN.PROGRESS_START_ALT)

    def qc_reset_progress():
        """ 
        GUI function to reset progress bar
        """
        
        #reset brir progress if applicable
        process_brirs_running=dpg.get_item_user_data("qc_brir_tag")
        brir_conv_enabled=dpg.get_value('e_apo_brir_conv')
        #if not already running
        #thread bool
        if process_brirs_running == False:
            #check if saved brir set name is matching with currently selected params
            brir_name = calc_brir_set_name(full_name=True)
            sel_brir_set=dpg.get_value('qc_e_apo_sel_brir_set')
            if brir_name != sel_brir_set:
                #reset progress bar
                dpg.set_value("qc_progress_bar_brir", 0)
                dpg.configure_item("qc_progress_bar_brir", overlay = CN.PROGRESS_START)
            elif brir_name == sel_brir_set and sel_brir_set !='' and brir_conv_enabled==True:
                dpg.set_value("qc_progress_bar_brir", 1)
                dpg.configure_item("qc_progress_bar_brir", overlay = CN.PROGRESS_FIN)

        #reset hpcf progress if applicable
        filter_name = calc_hpcf_name()
        sel_hpcf = dpg.get_value('qc_e_apo_sel_hpcf')
        hpcf_conv_enabled=dpg.get_value('e_apo_hpcf_conv')
        if filter_name != sel_hpcf:
            #reset progress bar
            dpg.set_value("qc_progress_bar_hpcf", 0)
            dpg.configure_item("qc_progress_bar_hpcf", overlay = CN.PROGRESS_START)
        elif filter_name == sel_hpcf and sel_hpcf !='' and hpcf_conv_enabled==True:
            dpg.set_value("qc_progress_bar_hpcf", 1)
            dpg.configure_item("qc_progress_bar_hpcf", overlay = CN.PROGRESS_FIN)
        

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
               
        
    #
    ## GUI Functions - Additional DEV tools
    #    

    def print_summary(sender, app_data, user_data):
        """ 
        GUI function to print summary of recent HpCFs
        """
        
        hpcf_functions.get_recent_hpcfs(conn, gui_logger=logz)     
        
    def generate_brir_reverb(sender, app_data, user_data):
        """ 
        GUI function to process BRIR reverberation data
        """
        #Run BRIR reverb synthesis
        brir_generation.generate_reverberant_brir(gui_logger=logz)

    def check_app_version(sender, app_data, user_data):
        """ 
        GUI function to check app version
        """
        hpcf_functions.check_for_app_update(gui_logger=logz)
    
    def check_db_version(sender, app_data, user_data):
        """ 
        GUI function to check db version
        """
        hpcf_functions.check_for_database_update(conn=conn, gui_logger=logz)

    def download_latest_db(sender, app_data, user_data):
        """ 
        GUI function to download latest db
        """
        hpcf_functions.downlod_latest_database(conn=conn, gui_logger=logz)
        
    def check_as_versions(sender, app_data, user_data):
        """ 
        GUI function to check acoustic space versions
        """
        air_processing.acoustic_space_updates(download_updates=False, gui_logger=logz)

    def download_latest_as_sets(sender, app_data, user_data):
        """ 
        GUI function to download latest acoustic spaces
        """
        air_processing.acoustic_space_updates(download_updates=True, gui_logger=logz)
    
    def check_hrtf_versions(sender, app_data, user_data):
        """ 
        GUI function to check hrtf versions
        """
        hrir_processing.hrir_metadata_updates(download_updates=False, gui_logger=logz)

    def download_latest_hrtf_sets(sender, app_data, user_data):
        """ 
        GUI function to download latest hrtf metadata
        """
        hrir_processing.hrir_metadata_updates(download_updates=True, gui_logger=logz)
        
    def check_all_updates():
        """ 
        GUI function to check for all updates
        """
        hpcf_functions.check_for_app_update(gui_logger=logz)
        hpcf_functions.check_for_database_update(conn=conn, gui_logger=logz)
        air_processing.acoustic_space_updates(download_updates=False, gui_logger=logz)
        hrir_processing.hrir_metadata_updates(download_updates=False, gui_logger=logz)

    
    def calc_hpcf_averages(sender, app_data, user_data):
        """ 
        GUI function to calculate hpcf averages
        """
        hpcf_functions.hpcf_generate_averages(conn, gui_logger=logz)
     
    def calc_hpcf_variants(sender, app_data, user_data):
        """ 
        GUI function to calculate hpcf averages
        """
        hpcf_functions.hpcf_generate_variants(conn, gui_logger=logz)
        
    def crop_hpcfs(sender, app_data, user_data):
        """ 
        GUI function to calculate hpcf averages
        """
        hpcf_functions.crop_hpcf_firs(conn, gui_logger=logz)    
        
    def renumber_hpcf_samples(sender, app_data, user_data):
        """ 
        GUI function to renumber hpcf samples to remove gaps
        """
        hpcf_functions.renumber_headphone_samples(conn, gui_logger=logz)
 
    def generate_hpcf_summary(sender, app_data, user_data):
        """ 
        GUI function to generate hpcf summary sheet
        """
        
        measurement_folder_name = dpg.get_value('hp_measurements_tag')
        in_ear_set = dpg.get_value('in_ear_set_tag')
        hpcf_functions.generate_hp_summary_sheet(conn=conn, measurement_folder_name=measurement_folder_name, in_ear_set = in_ear_set, gui_logger=logz)

    def calc_new_hpcfs(sender, app_data, user_data):
        """ 
        GUI function to calculate new hpcfs
        """
        
        measurement_folder_name = dpg.get_value('hp_measurements_tag')
        in_ear_set = dpg.get_value('in_ear_set_tag')
        hpcf_functions.calculate_new_hpcfs(conn=conn, measurement_folder_name=measurement_folder_name, in_ear_set = in_ear_set, gui_logger=logz)

    def rename_hp_hp(sender, app_data, user_data):
        """ 
        GUI function to rename a headphone in db
        """
        
        current_hp_name = dpg.get_value('headphone_list')
        new_hp_name = dpg.get_value('hpcf_rename_tag')
        if new_hp_name != "Headphone Name" and new_hp_name != None:
            hpcf_functions.rename_hpcf_headphone(conn=conn, headphone_old=current_hp_name, headphone_new=new_hp_name, gui_logger=logz)
            
            #also update headphone list and sample list
            sender=None
            app_data=dpg.get_value('brand_list')
            update_headphone_list(sender, app_data)
  
    def rename_hp_sample(sender, app_data, user_data):
        """ 
        GUI function to rename a headphone sample in db
        """
        
        current_hp_name = dpg.get_value('headphone_list')
        sample = dpg.get_value('sample_list')
        new_hp_name = dpg.get_value('hpcf_rename_tag')
        if new_hp_name != "Headphone Name" and new_hp_name != None:
            hpcf_functions.rename_hpcf_headphone_and_sample(conn=conn, headphone_old=current_hp_name, sample_old=sample, headphone_new=new_hp_name, gui_logger=logz)
            
            #also update headphone list and sample list
            sender=None
            app_data=dpg.get_value('brand_list')
            update_headphone_list(sender, app_data)
            
    def bulk_rename_sample(sender, app_data, user_data):
        """ 
        GUI function to rename a headphone sample in db
        """
        
        old_sample_name = dpg.get_value('sample_find_tag')
        new_sample_name = dpg.get_value('sample_replace_tag')
        
        if new_sample_name != "New Name" and new_sample_name != None:
            hpcf_functions.rename_hpcf_sample_name_bulk(conn=conn, sample_name_old=old_sample_name, sample_name_new=new_sample_name, gui_logger=logz)
            
            #also update headphone list and sample list
            sender=None
            app_data=dpg.get_value('brand_list')
            update_headphone_list(sender, app_data)
        

    def delete_hp(sender, app_data, user_data):
        """ 
        GUI function to delete a hp from db
        """
        headphone = dpg.get_value('headphone_list')
        
        hpcf_functions.delete_headphone(conn=conn, headphone=headphone, gui_logger=logz)
        
        #also update headphone list and sample list
        sender=None
        app_data=dpg.get_value('brand_list')
        update_headphone_list(sender, app_data)
    

    def delete_sample(sender, app_data, user_data):
        """ 
        GUI function to delete a sample from db
        """
        headphone = dpg.get_value('headphone_list')
        sample = dpg.get_value('sample_list')
        
        hpcf_functions.delete_headphone_sample(conn=conn, headphone=headphone, sample=sample, gui_logger=logz)
        
        #also update headphone list and sample list
        sender=None
        app_data=dpg.get_value('brand_list')
        update_headphone_list(sender, app_data)

    def create_db_from_wav(sender, app_data, user_data):
        """ 
        GUI function to recreate db from wavs
        """
        hpcf_functions.hpcf_wavs_to_database(conn, gui_logger=logz)

    def run_extract_airs(sender, app_data, user_data):
        """ 
        GUI function to run extract airs from recording function
        """
        
        dataset_name=dpg.get_value('air_dataset_name_tag')
        air_processing.extract_airs_from_recording(ir_set=dataset_name, gui_logger=logz)

    def run_split_airs_to_set(sender, app_data, user_data):
        """ 
        GUI function to run split airs to air set function
        """
        
        dataset_name=dpg.get_value('air_dataset_name_tag')
        air_processing.split_airs_to_set(ir_set=dataset_name, gui_logger=logz)

    def run_raw_air_to_dataset(sender, app_data, user_data):
        """ 
        GUI function to run RAW AIR to dataset function
        """
        
        dataset_name=dpg.get_value('air_dataset_name_tag')
        air_processing.prepare_air_set(ir_set=dataset_name, gui_logger=logz)
        
    def run_air_to_brir(sender, app_data, user_data):
        """ 
        GUI function to run AIR to BRIR dataset function
        """
        
        dataset_name=dpg.get_value('air_dataset_name_tag')
        air_processing.airs_to_brirs(ir_set=dataset_name, gui_logger=logz)
    
    def run_raw_to_brir(sender, app_data, user_data):
        """ 
        GUI function to run AIR to BRIR dataset function
        """
        dataset_name=dpg.get_value('air_dataset_name_tag')
        air_processing.raw_brirs_to_brir_set(ir_set=dataset_name, gui_logger=logz)
    
    def run_mono_cue(sender, app_data, user_data):
        """ 
        GUI function to run mono cues processing function
        """
        brir_generation.process_mono_cues(gui_logger=logz)
        
    def run_mono_cue_hp(sender, app_data, user_data):
        """ 
        GUI function to calculate new hpcfs
        """
        
        measurement_folder_name = dpg.get_value('hp_measurements_tag')
        in_ear_set = dpg.get_value('in_ear_set_tag')
        hpcf_functions.process_mono_hp_cues(conn=conn, measurement_folder_name=measurement_folder_name, in_ear_set = in_ear_set, gui_logger=logz)
        
    def generate_hrir_dataset(sender, app_data, user_data):
        """ 
        GUI function to run AIR to BRIR dataset function
        """
        spat_res = dpg.get_value("brir_spat_res")
        spat_res_int = CN.SPATIAL_RES_LIST.index(spat_res)
        brir_generation.preprocess_hrirs(spatial_res=spat_res_int, gui_logger=logz)
    
    def calc_reverb_target(sender, app_data, user_data):
        """ 
        GUI function to run calc reverb target function
        """
        air_processing.calc_reverb_target_mag(gui_logger=logz)
        
    def run_sub_brir_calc(sender, app_data, user_data):
        """ 
        GUI function to run calc sub brir function
        """
        air_processing.calc_subrir(gui_logger=logz)
        
    def run_room_target_calc(sender, app_data, user_data):
        """ 
        GUI function to run calc avg room target function
        """
        air_processing.calc_room_target_dataset(gui_logger=logz)
     
    def open_output_folder(sender, app_data, user_data):
        """ 
        GUI function to open output folder in windows explorer
        """
        ash_folder_selected=dpg.get_value('selected_folder_ash')
        base_folder_selected=dpg.get_value('selected_folder_base')
        try:
            os.startfile(ash_folder_selected)
        except Exception: 
            try:
                os.startfile(base_folder_selected)
            except Exception as e: 
                #print(e)
                #print('Failed')
                pass
        
    def intialise_gui():
        """ 
        GUI function to perform further initial configuration of gui elements
        """
        #inital configuration
        #update channel gui elements on load, will also write e-APO config again
        e_apo_select_channels(app_data=dpg.get_value('audio_channels_combo'),aquire_config=False)
        #adjust active tab
        try:
            dpg.set_value("tab_bar", tab_selected_loaded)
        except Exception:
            pass
        save_settings()
        hpcf_is_active=dpg.get_value('e_apo_hpcf_conv')
        brir_is_active=dpg.get_value('e_apo_brir_conv')

        e_apo_toggle_hpcf_custom(app_data=hpcf_is_active, aquire_config=False)
        e_apo_toggle_brir_custom(app_data=brir_is_active, aquire_config=False)
        #finally acquire config once
        e_apo_config_acquire()
        
    
        
     
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
    width_fl, height_fl, channels_fl, data_fl = dpg.load_image(image_location_fl)
    width_fr, height_fr, channels_fr, data_fr = dpg.load_image(image_location_fr)
    width_c, height_c, channels_c, data_c = dpg.load_image(image_location_c)
    width_sl, height_sl, channels_sl, data_sl = dpg.load_image(image_location_sl)
    width_sr, height_sr, channels_sr, data_sr = dpg.load_image(image_location_sr)
    width_rl, height_rl, channels_rl, data_rl = dpg.load_image(image_location_rl)
    width_rr, height_rr, channels_rr, data_rr = dpg.load_image(image_location_rr)

    with dpg.texture_registry(show=False):
        dpg.add_static_texture(width=width_fl, height=height_fl, default_value=data_fl, tag='fl_image')
        dpg.add_static_texture(width=width_fr, height=height_fr, default_value=data_fr, tag='fr_image')
        dpg.add_static_texture(width=width_c, height=height_c, default_value=data_c, tag='c_image')
        dpg.add_static_texture(width=width_sl, height=height_sl, default_value=data_sl, tag='sl_image')
        dpg.add_static_texture(width=width_sr, height=height_sr, default_value=data_sr, tag='sr_image')
        dpg.add_static_texture(width=width_rl, height=height_rl, default_value=data_rl, tag='rl_image')
        dpg.add_static_texture(width=width_rr, height=height_rr, default_value=data_rr, tag='rr_image')

    # add a font registry
    with dpg.font_registry():
        # first argument ids the path to the .ttf or .otf file
        in_file_path = pjoin(CN.DATA_DIR_EXT,'font', 'Lato-Medium.ttf')#SourceSansPro-Regular
        default_font = dpg.add_font(in_file_path, 14)    
        in_file_path = pjoin(CN.DATA_DIR_EXT,'font', 'Lato-Bold.ttf')#SourceSansPro-Regular
        bold_font = dpg.add_font(in_file_path, 14)
        in_file_path = pjoin(CN.DATA_DIR_EXT,'font', 'Lato-Bold.ttf')#SourceSansPro-Regular
        bold_small_font = dpg.add_font(in_file_path, 13) 
        in_file_path = pjoin(CN.DATA_DIR_EXT,'font', 'Lato-Bold.ttf')#SourceSansPro-Regular
        large_font = dpg.add_font(in_file_path, 16)    
        in_file_path = pjoin(CN.DATA_DIR_EXT,'font', 'Lato-Medium.ttf')#SourceSansPro-Regular
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
        with dpg.theme(tag="__theme_d"):
            i=4
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(i/7.0, 0.6, 0.6))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(i/7.0, 0.8, 0.8))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(i/7.0, 0.7, 0.7))  
        with dpg.theme(tag="__theme_e"):
            i=3.1
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, _hsv_to_rgb(i/7.0, 0.6, 0.8))
        with dpg.theme(tag="__theme_f"):
            i=3.9
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, _hsv_to_rgb(i/7.0, 0.05, 0.99))
        with dpg.theme(tag="__theme_g"):
            i=3.5
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, _hsv_to_rgb(i/7.0, 0.2, 0.5)) 
        with dpg.theme(tag="__theme_h"):
            i=3.9
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, _hsv_to_rgb(i/7.0, 0.2, 0.5)) 
        with dpg.theme() as global_theme:
            i=3.9
            j=3.9
            k=3.8
            with dpg.theme_component(dpg.mvAll):
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
            with dpg.theme_component(dpg.mvAll):  
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
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, _hsv_to_rgb(j/7.0, 0.7, 0.7))
        dpg.bind_theme(global_theme)
        
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
                                dpg.add_text("                                                                             ")
                                dpg.add_checkbox(label="Show History", default_value = False,  tag='qc_toggle_hpcf_history', callback=qc_show_hpcf_history)
                                with dpg.tooltip("qc_toggle_hpcf_history"):
                                    dpg.add_text("Shows previously applied headphones")
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_text("Search Brand:")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_input_text(label="", callback=qc_filter_brand_list, width=105)
                                dpg.add_text("Search Headphone:")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_input_text(label="", callback=qc_filter_headphone_list, width=209)
                            with dpg.group(horizontal=True, width=0):
                                with dpg.group():
                                    dpg.add_text("Brand")
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_listbox(brands_list, width=135, num_items=16, tag='qc_brand_list', default_value=qc_brand_loaded, callback=qc_update_headphone_list)
                                with dpg.group():
                                    dpg.add_text("Headphone")
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_listbox(qc_hp_list_loaded, width=250, num_items=16, tag='qc_headphone_list', default_value=qc_headphone_loaded ,callback=qc_update_sample_list)
                                with dpg.group():
                                    dpg.add_text("Sample")
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_listbox(qc_sample_list_loaded, width=115, num_items=16, default_value=qc_sample_loaded, tag='qc_sample_list', callback=qc_plot_sample) 
                        with dpg.child_window(autosize_x=True, height=176):
                            with dpg.group(horizontal=True):
                                subtitle_3_qc = dpg.add_text("Apply Headphone Correction in Equalizer APO")
                                dpg.bind_item_font(subtitle_3_qc, bold_font)
                                dpg.add_text("                                    ")
                                dpg.add_checkbox(label="Auto Apply Selection", default_value = e_apo_autoapply_hpcf_loaded,  tag='qc_auto_apply_hpcf_sel', callback=e_apo_auto_apply_hpcf)
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_button(label=CN.PROCESS_BUTTON_HPCF,user_data="",tag="qc_hpcf_tag", callback=qc_apply_hpcf_params, width=145)
                                dpg.bind_item_theme(dpg.last_item(), "__theme_a")
                                dpg.bind_item_font(dpg.last_item(), large_font)
                                with dpg.tooltip("qc_hpcf_tag"):
                                    dpg.add_text("This will apply the selected filter in Equalizer APO")
                                dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=30, width=356, overlay=CN.PROGRESS_START, tag="qc_progress_bar_hpcf")
                                dpg.bind_item_theme(dpg.last_item(), "__theme_h")
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="Enable Headphone Correction", default_value = e_apo_enable_hpcf_loaded,  tag='e_apo_hpcf_conv', callback=e_apo_toggle_hpcf_gui)
                                #dpg.add_text("                                    ")
                                
                            dpg.add_separator()
                            dpg.add_text("Current Filter: ")
                            dpg.bind_item_font(dpg.last_item(), bold_font)
                            dpg.add_text(default_value=e_apo_hpcf_curr_loaded,  tag='qc_e_apo_curr_hpcf')
                            dpg.bind_item_font(dpg.last_item(), large_font)
                            dpg.bind_item_theme(dpg.last_item(), "__theme_f")
                            dpg.add_text(default_value=e_apo_hpcf_sel_loaded, tag='qc_e_apo_sel_hpcf',show=False)
                    #Section for BRIR generation
                    with dpg.child_window(width=534, height=610):
                        title_2_qc = dpg.add_text("Binaural Room Simulation", tag='qc_brir_title')
                        dpg.bind_item_font(title_2_qc, bold_font)
                        with dpg.tooltip("qc_brir_title"):
                            dpg.add_text("Customise a new binaural room simulation using below parameters")
                        with dpg.child_window(autosize_x=True, height=390):
                            with dpg.tab_bar(tag='qc_brir_tab_bar'):

                                with dpg.tab(label="Acoustics & EQ Parameters",tag='qc_acoustics_eq_tab', parent="qc_brir_tab_bar"): 
                                    dpg.add_separator()
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Acoustic Space",tag='qc_acoustic_space_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(CN.AC_SPACE_LIST_GUI, label="",tag='qc_acoustic_space_combo',default_value=qc_ac_space_loaded, callback=qc_update_brir_param, num_items=17, width=255)
                                            with dpg.tooltip("qc_acoustic_space_title"):
                                                dpg.add_text("This will determine the listening environment")
                                        with dpg.group():
                                            dpg.add_text("Direct Sound Gain (dB)", tag='qc_direct_gain_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_input_float(label="Direct Gain (dB)",width=140, format="%.1f", tag='qc_direct_gain', min_value=CN.DIRECT_GAIN_MIN, max_value=CN.DIRECT_GAIN_MAX, default_value=qc_direct_gain_loaded,min_clamped=True, max_clamped=True, callback=qc_update_direct_gain)
                                            with dpg.tooltip("qc_direct_gain_title"):
                                                dpg.add_text("This will control the loudness of the direct signal")
                                                dpg.add_text("Higher values result in lower perceived distance, lower values result in higher perceived distance")
                                            dpg.add_slider_float(label="", min_value=CN.DIRECT_GAIN_MIN, max_value=CN.DIRECT_GAIN_MAX, default_value=qc_direct_gain_loaded, width=140,clamped=True, no_input=True, format="", callback=qc_update_direct_gain_slider, tag='qc_direct_gain_slider')
                                            
                                            dpg.add_text("Room Target", tag='qc_rm_target_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(CN.ROOM_TARGET_LIST, default_value=qc_room_target_loaded, num_items=7, width=235, tag='qc_rm_target_list', callback=qc_select_room_target)
                                            with dpg.tooltip("qc_rm_target_title"):
                                                dpg.add_text("This will influence the overall balance of low and high frequencies")
                                            
                                            dpg.add_text("Headphone Compensation", tag='qc_brir_hp_type_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(CN.HP_COMP_LIST, default_value=qc_brir_hp_type_loaded, num_items=4, width=235, callback=qc_select_hp_comp, tag='qc_brir_hp_type')
                                            with dpg.tooltip("qc_brir_hp_type_title"):
                                                dpg.add_text("This should align with the listener's headphone type")
                                                dpg.add_text("Reduce to low strength if sound localisation or timbre is compromised")
                                                
                                with dpg.tab(label="Listener Selection",tag='qc_listener_tab', parent="qc_brir_tab_bar"): 
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Listener Type", tag='qc_brir_hrtf_type_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_radio_button(brir_hrtf_type_list_default, horizontal=False, tag= "qc_brir_hrtf_type", default_value=qc_brir_hrtf_type_loaded, callback=qc_update_hrtf_dataset_list )
                                            with dpg.tooltip("qc_brir_hrtf_type"):
                                                dpg.add_text("User SOFA files must be placed in 'ASH Toolset\\_internal\data\\user\\SOFA' folder")
                                            dpg.add_text("Dataset Name", tag='qc_brir_hrtf_dataset_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(qc_brir_hrtf_dataset_list_loaded, default_value=qc_brir_hrtf_dataset_loaded, num_items=11, width=255, callback=qc_update_hrtf_list, tag='qc_brir_hrtf_dataset')
                                        with dpg.group():
                                            dpg.add_text("Listener", tag='qc_brir_hrtf_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(qc_hrtf_list_loaded, default_value=qc_hrtf_loaded, num_items=17, width=240, callback=qc_select_hrtf, tag='qc_brir_hrtf')
                                            with dpg.tooltip("qc_brir_hrtf_title"):
                                                dpg.add_text("This will influence the externalisation and localisation of sounds around the listener")
                                                
                        with dpg.child_window(autosize_x=True, height=175):
                            with dpg.group(horizontal=True):
                                subtitle_6_qc = dpg.add_text("Apply Simulation in Equalizer APO")
                                dpg.bind_item_font(subtitle_6_qc, bold_font)
                                dpg.add_text("                                                      ")
                  
                            dpg.add_separator()
                            
                            with dpg.group(horizontal=True):
                                dpg.add_button(label=CN.PROCESS_BUTTON_BRIR,user_data=CN.PROCESS_BRIRS_RUNNING,tag="qc_brir_tag", callback=qc_apply_brir_params, width=145)#user data is thread running flag
                                dpg.bind_item_theme(dpg.last_item(), "__theme_a")
                                dpg.bind_item_font(dpg.last_item(), large_font)
                                with dpg.tooltip("qc_brir_tag"):
                                    dpg.add_text("This will generate the binaural simulation and apply it in Equalizer APO")   
                                dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=30, width=340, overlay=CN.PROGRESS_START, tag="qc_progress_bar_brir",user_data=CN.STOP_THREAD_FLAG)#user data is thread cancel flag
                                dpg.bind_item_theme(dpg.last_item(), "__theme_h")
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="Enable Binaural Room Simulation", default_value = e_apo_enable_brir_loaded,  tag='e_apo_brir_conv', callback=e_apo_toggle_brir_gui, user_data=[])#user data will be a dict list
                                dpg.add_text("                                    ")
                     
                            dpg.add_separator()
                            dpg.add_text("Current Simulation: ")
                            dpg.bind_item_font(dpg.last_item(), bold_font)
                            dpg.add_text(default_value=e_apo_brir_curr_loaded,  tag='qc_e_apo_curr_brir_set' , wrap=510, user_data=False)#user data used for flagging use of below BRIR dict
                            dpg.bind_item_font(dpg.last_item(), bold_font)
                            dpg.bind_item_theme(dpg.last_item(), "__theme_f")
                            dpg.add_text(default_value=e_apo_brir_sel_loaded, tag='qc_e_apo_sel_brir_set',show=False, user_data={})#user data used for storing snapshot of BRIR dict 
                    #right most section
                    with dpg.group():    
                        #Section for channel config, plotting
                        with dpg.child_window(autosize_x=True, height=422):
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
                                                dpg.add_input_float(label=" ",format="%.1f", width=100,min_value=-100, max_value=20,min_clamped=True,max_clamped=True, tag='e_apo_gain_oa',default_value=e_apo_gain_oa_loaded, callback=e_apo_adjust_preamp)
                                                dpg.add_text("  ")
                                                dpg.add_checkbox(label="Auto-adjust Preamp ", default_value = e_apo_prevent_clip_loaded,  tag='e_apo_prevent_clip', callback=e_apo_config_acquire_gui)
                                                with dpg.tooltip("e_apo_prevent_clip"):
                                                    dpg.add_text("This will auto-adjust the preamp to prevent clipping for 2.0 inputs")
                                            with dpg.group(horizontal=True):
                                                dpg.add_text("Audio Channels:  ")
                                                dpg.add_combo(CN.AUDIO_CHANNELS, width=180, label="",  tag='audio_channels_combo',default_value=audio_channels_loaded, callback=e_apo_select_channels_gui)
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
                                                                        dpg.add_combo(CN.UPMIXING_METHODS, width=100, label="",  tag='e_apo_upmix_method',default_value=e_apo_upmix_method_loaded, callback=e_apo_config_acquire_gui)
                                                                    elif i == 1:
                                                                        dpg.add_input_int(label=" ", width=100,min_value=-100, max_value=100, tag='e_apo_side_delay',min_clamped=True,max_clamped=True, default_value=e_apo_side_delay_loaded, callback=e_apo_config_acquire_gui)
                                                                    elif i == 2:
                                                                        dpg.add_input_int(label=" ", width=100,min_value=-100, max_value=100, tag='e_apo_rear_delay',min_clamped=True,max_clamped=True, default_value=e_apo_rear_delay_loaded, callback=e_apo_config_acquire_gui)  
                                                                    
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
                                                                            dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_fl',default_value=e_apo_mute_fl_loaded, callback=e_apo_config_acquire_gui)
                                                                            dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                        elif i == 1:
                                                                            dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_fr',default_value=e_apo_mute_fr_loaded, callback=e_apo_config_acquire_gui)
                                                                            dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                        elif i == 2:
                                                                            dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_c',default_value=e_apo_mute_c_loaded, callback=e_apo_config_acquire_gui)
                                                                            dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                        elif i == 3:
                                                                            dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_sl',default_value=e_apo_mute_sl_loaded, callback=e_apo_config_acquire_gui)
                                                                            dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                        elif i == 4:
                                                                            dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_sr',default_value=e_apo_mute_sr_loaded, callback=e_apo_config_acquire_gui)
                                                                            dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                        elif i == 5:
                                                                            dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_rl',default_value=e_apo_mute_rl_loaded, callback=e_apo_config_acquire_gui)
                                                                            dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                        elif i == 6:
                                                                            dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_rr',default_value=e_apo_mute_rr_loaded, callback=e_apo_config_acquire_gui)
                                                                            dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                    if j == 2:#gain
                                                                        if i == 0:
                                                                            dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_fl',min_clamped=True,max_clamped=True, default_value=e_apo_gain_fl_loaded, callback=e_apo_config_acquire_gui)
                                                                        elif i == 1:
                                                                            dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_fr',min_clamped=True,max_clamped=True,default_value=e_apo_gain_fr_loaded, callback=e_apo_config_acquire_gui)
                                                                        elif i == 2:
                                                                            dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_c',min_clamped=True,max_clamped=True,default_value=e_apo_gain_c_loaded, callback=e_apo_config_acquire_gui)
                                                                        elif i == 3:
                                                                            dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_sl',min_clamped=True,max_clamped=True,default_value=e_apo_gain_sl_loaded, callback=e_apo_config_acquire_gui)
                                                                        elif i == 4:
                                                                            dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_sr',min_clamped=True,max_clamped=True,default_value=e_apo_gain_sr_loaded, callback=e_apo_config_acquire_gui)
                                                                        elif i == 5:
                                                                            dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_rl',min_clamped=True,max_clamped=True,default_value=e_apo_gain_rl_loaded, callback=e_apo_config_acquire_gui)
                                                                        elif i == 6:
                                                                            dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_rr',min_clamped=True,max_clamped=True,default_value=e_apo_gain_rr_loaded, callback=e_apo_config_acquire_gui)
                                                                    if j == 3:#elevation
                                                                        if i == 0:
                                                                            dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_fl',default_value=e_apo_elev_angle_fl_loaded, callback=e_apo_activate_direction_gui)
                                                                            with dpg.tooltip("e_apo_elev_angle_fl"):
                                                                                dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                        elif i == 1:
                                                                            dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_fr',default_value=e_apo_elev_angle_fr_loaded, callback=e_apo_activate_direction_gui)
                                                                            with dpg.tooltip("e_apo_elev_angle_fr"):
                                                                                dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                        elif i == 2:
                                                                            dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_c',default_value=e_apo_elev_angle_c_loaded, callback=e_apo_activate_direction_gui)
                                                                            with dpg.tooltip("e_apo_elev_angle_c"):
                                                                                dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                        elif i == 3:
                                                                            dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_sl',default_value=e_apo_elev_angle_sl_loaded, callback=e_apo_activate_direction_gui)
                                                                            with dpg.tooltip("e_apo_elev_angle_sl"):
                                                                                dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                        elif i == 4:
                                                                            dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_sr',default_value=e_apo_elev_angle_sr_loaded, callback=e_apo_activate_direction_gui)
                                                                            with dpg.tooltip("e_apo_elev_angle_sr"):
                                                                                dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                        elif i == 5:
                                                                            dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_rl',default_value=e_apo_elev_angle_rl_loaded, callback=e_apo_activate_direction_gui)
                                                                            with dpg.tooltip("e_apo_elev_angle_rl"):
                                                                                dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                        elif i == 6:
                                                                            dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_rr',default_value=e_apo_elev_angle_rr_loaded, callback=e_apo_activate_direction_gui)
                                                                            with dpg.tooltip("e_apo_elev_angle_rr"):
                                                                                dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                    if j == 4:#azimuth
                                                                        if i == 0:
                                                                            dpg.add_combo(CN.AZ_ANGLES_FL_WAV, width=70, label="",  tag='e_apo_az_angle_fl',default_value=e_apo_az_angle_fl_loaded, callback=e_apo_activate_direction_gui)
                                                                        elif i == 1:
                                                                            dpg.add_combo(CN.AZ_ANGLES_FR_WAV, width=70, label="",  tag='e_apo_az_angle_fr',default_value=e_apo_az_angle_fr_loaded, callback=e_apo_activate_direction_gui)
                                                                        elif i == 2:
                                                                            dpg.add_combo(CN.AZ_ANGLES_C_WAV, width=70, label="",  tag='e_apo_az_angle_c',default_value=e_apo_az_angle_c_loaded, callback=e_apo_activate_direction_gui)
                                                                        elif i == 3:
                                                                            dpg.add_combo(CN.AZ_ANGLES_SL_WAV, width=70, label="",  tag='e_apo_az_angle_sl',default_value=e_apo_az_angle_sl_loaded, callback=e_apo_activate_direction_gui)
                                                                        elif i == 4:
                                                                            dpg.add_combo(CN.AZ_ANGLES_SR_WAV, width=70, label="",  tag='e_apo_az_angle_sr',default_value=e_apo_az_angle_sr_loaded, callback=e_apo_activate_direction_gui)
                                                                        elif i == 5:
                                                                            dpg.add_combo(CN.AZ_ANGLES_RL_WAV, width=70, label="",  tag='e_apo_az_angle_rl',default_value=e_apo_az_angle_rl_loaded, callback=e_apo_activate_direction_gui)
                                                                        elif i == 6:
                                                                            dpg.add_combo(CN.AZ_ANGLES_RR_WAV, width=70, label="",  tag='e_apo_az_angle_rr',default_value=e_apo_az_angle_rr_loaded, callback=e_apo_activate_direction_gui)
                                   
                                                with dpg.drawlist(width=250, height=200, tag="channel_drawing"):
                                                    with dpg.draw_layer():

                                                        dpg.draw_circle([CN.X_START, CN.Y_START], CN.RADIUS, color=[163, 177, 184])           
                                                        with dpg.draw_node(tag="listener_drawing"):
                                                            dpg.apply_transform(dpg.last_item(), dpg.create_translation_matrix([CN.X_START, CN.Y_START]))
                                                            dpg.draw_circle([0, 0], 20, color=[163, 177, 184], fill=[158,158,158])
                                                            dpg.draw_text([-19, -8], 'Listener', color=[0, 0, 0],size=13)
                                                            
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
                                with dpg.tab(label="Supporting Information", parent="qc_inner_tab_bar",tag='qc_si_tab'):
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Acoustic Properties")
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            data_length = len(CN.AC_SPACE_LIST_GUI)
                                            third_length = data_length // 3
                                            remainder = data_length % 3
                                            # Calculate the end indices for each table
                                            table1_end = third_length + (1 if remainder > 0 else 0)
                                            table2_end = table1_end + third_length + (1 if remainder > 1 else 0)
                                            table3_end = data_length
                                            with dpg.group(horizontal=True):
                                                # Table 1
                                                with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                                               borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                                               borders_outerV=True, borders_innerV=True, delay_search=True):
                                            
                                                    dpg.add_table_column(label="Name")
                                                    dpg.add_table_column(label="RT60 (ms)")
                                            
                                                    for i in range(table1_end):
                                                        with dpg.table_row():
                                                            dpg.add_text(CN.AC_SPACE_LIST_GUI[i])
                                                            dpg.add_text(CN.AC_SPACE_MEAS_R60[i])
                                            
                                                # Table 2
                                                with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                                               borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                                               borders_outerV=True, borders_innerV=True, delay_search=True):
                                            
                                                    dpg.add_table_column(label="Name")
                                                    dpg.add_table_column(label="RT60 (ms)")
                                            
                                                    for i in range(table1_end, table2_end):
                                                        with dpg.table_row():
                                                            dpg.add_text(CN.AC_SPACE_LIST_GUI[i])
                                                            dpg.add_text(CN.AC_SPACE_MEAS_R60[i])
                                            
                                                # Table 3
                                                with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                                               borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                                               borders_outerV=True, borders_innerV=True, delay_search=True):
                                            
                                                    dpg.add_table_column(label="Name")
                                                    dpg.add_table_column(label="RT60 (ms)")
                                            
                                                    for i in range(table2_end, table3_end):
                                                        with dpg.table_row():
                                                            dpg.add_text(CN.AC_SPACE_LIST_GUI[i])
                                                            dpg.add_text(CN.AC_SPACE_MEAS_R60[i])
                                                                        
                                    #Section to show equalizer apo directory
                                    with dpg.group():
                                        dpg.add_separator()
                                        dpg.add_text("Equalizer APO Config Path")
                                        dpg.bind_item_font(dpg.last_item(), bold_font)
                                        dpg.add_text(tag='qc_selected_folder_base')
                                        dpg.set_value('qc_selected_folder_base', qc_primary_path)
                                        
                                        dpg.add_separator()
                                        dpg.add_text("Supported SOFA Conventions")
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
    
                        #Section for misc settings
                        with dpg.group():
                            #Section for wav settings
                            with dpg.child_window(width=590, height=100):
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
                                                        dpg.add_radio_button(CN.SAMPLE_RATE_LIST, horizontal=True, tag= "qc_wav_sample_rate", default_value=sample_freq_loaded, callback=sync_wav_sample_rate )
                                                        dpg.add_text("  ")
                                                elif j == 2:#Bit Depth
                                                    with dpg.group(horizontal=True):
                                                        dpg.add_text("  ")
                                                        dpg.add_radio_button(CN.BIT_DEPTH_LIST, horizontal=True, tag= "qc_wav_bit_depth", default_value=bit_depth_loaded, callback=sync_wav_bit_depth)
                                                        dpg.add_text("  ")

                            
     

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
                                dpg.add_input_text(label="", callback=filter_brand_list, width=105)
                                dpg.add_text("Search Headphone:")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_input_text(label="", callback=filter_headphone_list, width=209)
                            with dpg.group(horizontal=True, width=0):
                                with dpg.group():
                                    dpg.add_text("Brand")
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_listbox(brands_list, width=135, num_items=16, tag='brand_list', callback=update_headphone_list)
                                with dpg.group():
                                    dpg.add_text("Headphone")
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_listbox(hp_list_default, width=250, num_items=16, tag='headphone_list', default_value=headphone_default ,callback=update_sample_list)
                                with dpg.group():
                                    dpg.add_text("Sample")
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_listbox(sample_list_default, width=115, num_items=16, default_value=sample_default, tag='sample_list', user_data=headphone_default, callback=plot_sample)
                                    with dpg.tooltip("sample_list"):
                                        dpg.add_text("Note: all samples will be exported. Select a sample to preview")
                        with dpg.child_window(autosize_x=True, height=88):
                            subtitle_2 = dpg.add_text("Select Files to Include in Export")
                            dpg.bind_item_font(subtitle_2, bold_font)
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="WAV FIR Filters", default_value = fir_hpcf_exp_loaded, callback=export_hpcf_file_toggle, tag='fir_hpcf_toggle')
                                with dpg.tooltip("fir_hpcf_toggle"):
                                    dpg.add_text("Min phase FIRs in WAV format for convolution. 1 Channel")
                                    dpg.add_text("Required for Equalizer APO configuration updates")
                                dpg.add_checkbox(label="WAV Stereo FIR Filters", default_value = fir_st_hpcf_exp_loaded, callback=export_hpcf_file_toggle, tag='fir_st_hpcf_toggle')
                                with dpg.tooltip("fir_st_hpcf_toggle"):
                                    dpg.add_text("Min phase FIRs in WAV format for convolution. 2 Channels")
                                dpg.add_checkbox(label="E-APO Configuration Files", default_value = eapo_hpcf_exp_loaded, callback=export_hpcf_file_toggle, tag='eapo_hpcf_toggle')  
                                with dpg.tooltip("eapo_hpcf_toggle"):
                                    dpg.add_text("Equalizer APO configurations to perform convolution with FIR filters. Deprecated from V2.0.0 onwards")
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="Graphic EQ Filters (127 Bands)", default_value = geq_hpcf_exp_loaded, callback=export_hpcf_file_toggle, tag='geq_hpcf_toggle')
                                with dpg.tooltip("geq_hpcf_toggle"):
                                    dpg.add_text("Graphic EQ configurations with 127 bands. Compatible with Equalizer APO and Wavelet")
                                dpg.add_checkbox(label="Graphic EQ Filters (31 Bands)", default_value = geq_31_exp_loaded, callback=export_hpcf_file_toggle, tag='geq_31_hpcf_toggle')
                                with dpg.tooltip("geq_31_hpcf_toggle"):
                                    dpg.add_text("Graphic EQ configurations with 31 bands. Compatible with 31 band graphic equalizers including Equalizer APO")
                                dpg.add_checkbox(label="HeSuVi Filters", default_value = hesuvi_hpcf_exp_loaded, callback=export_hpcf_file_toggle, tag='hesuvi_hpcf_toggle')
                                with dpg.tooltip("hesuvi_hpcf_toggle"):
                                    dpg.add_text("Graphic EQ configurations with 103 bands. Compatible with HeSuVi. Saved in HeSuVi\eq folder")
                        with dpg.child_window(autosize_x=True, height=84):
                            subtitle_3 = dpg.add_text("Export Correction Filters")
                            dpg.bind_item_font(subtitle_3, bold_font)
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="Process",user_data="",tag="hpcf_tag", callback=process_hpcfs, width=130)
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
  
                                    dpg.add_separator()
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Acoustic Space",tag='acoustic_space_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(CN.AC_SPACE_LIST_GUI, label="",tag='acoustic_space_combo',default_value=ac_space_loaded, callback=update_brir_param, num_items=17, width=250)
                                            with dpg.tooltip("acoustic_space_title"):
                                                dpg.add_text("This will determine the listening environment")
                                            
                                            
                                        with dpg.group():
                                            dpg.add_text("Direct Sound Gain (dB)", tag='direct_gain_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_input_float(label="Direct Gain (dB)",width=140, format="%.1f", tag='direct_gain', min_value=CN.DIRECT_GAIN_MIN, max_value=CN.DIRECT_GAIN_MAX, default_value=direct_gain_loaded,min_clamped=True, max_clamped=True, callback=update_direct_gain)
                                            with dpg.tooltip("direct_gain_title"):
                                                dpg.add_text("This will control the loudness of the direct signal")
                                                dpg.add_text("Higher values result in lower perceived distance, lower values result in higher perceived distance")
                                            dpg.add_slider_float(label="", min_value=CN.DIRECT_GAIN_MIN, max_value=CN.DIRECT_GAIN_MAX, default_value=direct_gain_loaded, width=140,clamped=True, no_input=True, format="", callback=update_direct_gain_slider, tag='direct_gain_slider')
                                     
                                            dpg.add_text("Room Target", tag='rm_target_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(CN.ROOM_TARGET_LIST, default_value=room_target_loaded, num_items=7, width=230, tag='rm_target_list', callback=select_room_target)
                                            with dpg.tooltip("rm_target_title"):
                                                dpg.add_text("This will influence the overall balance of low and high frequencies")
     
                                            dpg.add_text("Headphone Compensation", tag='brir_hp_type_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(CN.HP_COMP_LIST, default_value=brir_hp_type_loaded, num_items=4, width=230, callback=select_hp_comp, tag='brir_hp_type')
                                            with dpg.tooltip("brir_hp_type_title"):
                                                dpg.add_text("This should align with the listener's headphone type")
                                                dpg.add_text("Reduce to low strength if sound localisation or timbre is compromised")
                                                
                                with dpg.tab(label="Listener Selection",tag='listener_tab', parent="brir_tab_bar"): 
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Spatial Resolution", tag= "brir_spat_res_title")
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_radio_button(spatial_res_list_loaded, horizontal=True, tag= "brir_spat_res", default_value=spatial_res_loaded, callback=select_spatial_resolution )
                                            with dpg.tooltip("brir_spat_res_title"):
                                                dpg.add_text("Increasing resolution will increase number of directions available but will also increase processing time and dataset size")
                                                dpg.add_text("'Low' is recommended unless additional directions or SOFA export is required")
                                                
                                            dpg.add_text("Listener Type", tag='brir_hrtf_type_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_radio_button(brir_hrtf_type_list_default, horizontal=False, tag= "brir_hrtf_type", default_value=brir_hrtf_type_loaded, callback=update_hrtf_dataset_list )
                                            with dpg.tooltip("brir_hrtf_type"):
                                                dpg.add_text("User SOFA files must be placed in 'ASH Toolset\\_internal\data\\user\\SOFA' folder")
                                            dpg.add_text("Dataset Name", tag='brir_hrtf_dataset_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(brir_hrtf_dataset_list_loaded, default_value=brir_hrtf_dataset_loaded, num_items=9, width=255, callback=update_hrtf_list, tag='brir_hrtf_dataset')
                                        with dpg.group():
                                            
                                            dpg.add_text("Listener", tag='brir_hrtf_title')
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_listbox(hrtf_list_loaded, default_value=hrtf_loaded, num_items=17, width=230, callback=select_hrtf, tag='brir_hrtf')
                                            with dpg.tooltip("brir_hrtf_title"):
                                                dpg.add_text("This will influence the externalisation and localisation of sounds around the listener")
                                        
                        with dpg.child_window(autosize_x=True, height=88):
                            subtitle_5 = dpg.add_text("Select Files to Include in Export")
                            dpg.bind_item_font(subtitle_5, bold_font)
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="Direction-specific WAV BRIRs", default_value = dir_brir_exp_loaded,  tag='dir_brir_toggle', callback=export_brir_toggle, show=dir_brir_exp_show)
                                with dpg.tooltip("dir_brir_toggle", tag='dir_brir_tooltip', show=dir_brir_tooltip_show):
                                    dpg.add_text("Binaural Room Impulse Responses (BRIRs) in WAV format for convolution")
                                    dpg.add_text("2 channels per file. One file for each direction")
                                    dpg.add_text("Required for Equalizer APO configuration updates")
                                dpg.add_checkbox(label="True Stereo WAV BRIR", default_value = ts_brir_exp_loaded,  tag='ts_brir_toggle', callback=export_brir_toggle, show=ts_brir_exp_show)
                                with dpg.tooltip("ts_brir_toggle", tag='ts_brir_tooltip', show=ts_brir_tooltip_show):
                                    dpg.add_text("True Stereo BRIR in WAV format for convolution")
                                    dpg.add_text("4 channels. One file representing L and R speakers")
                                dpg.add_checkbox(label="SOFA File", default_value = sofa_brir_exp_loaded,  tag='sofa_brir_toggle', callback=export_brir_toggle, show=sofa_brir_exp_show)
                                with dpg.tooltip("sofa_brir_toggle", tag='sofa_brir_tooltip', show=sofa_brir_tooltip_show):
                                    dpg.add_text("BRIR dataset as a SOFA file")
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="HeSuVi WAV BRIRs", default_value = hesuvi_brir_exp_loaded,  tag='hesuvi_brir_toggle', callback=export_brir_toggle, show=hesuvi_brir_exp_show)  
                                with dpg.tooltip("hesuvi_brir_toggle", tag='hesuvi_brir_tooltip', show=hesuvi_brir_tooltip_show):
                                    dpg.add_text("BRIRs in HeSuVi compatible WAV format. 14 channels, 44.1kHz and 48kHz")
                                dpg.add_checkbox(label="E-APO Configuration Files", default_value = eapo_brir_exp_loaded,  tag='eapo_brir_toggle', callback=export_brir_toggle, show=eapo_brir_exp_show)
                                with dpg.tooltip("eapo_brir_toggle", tag='eapo_brir_tooltip', show=eapo_brir_tooltip_show):
                                    dpg.add_text("Equalizer APO configurations to perform convolution with BRIRs. Deprecated from V2.0.0 onwards")
                        with dpg.child_window(autosize_x=True, height=86):
                            subtitle_6 = dpg.add_text("Generate and Export Binaural Dataset")
                            dpg.bind_item_font(subtitle_6, bold_font)
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="Process",user_data=CN.PROCESS_BRIRS_RUNNING,tag="brir_tag", callback=start_process_brirs, width=130)
                                dpg.bind_item_theme(dpg.last_item(), "__theme_b")
                                dpg.bind_item_font(dpg.last_item(), large_font)
                                with dpg.tooltip("brir_tag"):
                                    dpg.add_text("This will generate the binaural dataset and export to the output directory. This may take some time to process")
                                    
                                dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=32, width=340, overlay=CN.PROGRESS_START_ALT, tag="progress_bar_brir",user_data=CN.STOP_THREAD_FLAG)
                                dpg.bind_item_theme(dpg.last_item(), "__theme_g")
                
                    #right most section
                    with dpg.group():    
                        #Section for plotting
                        with dpg.child_window(autosize_x=True, height=467):
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
                                with dpg.tab(label="Supporting Information", parent="inner_tab_bar",tag='si_tab'):
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Acoustic Properties")
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                                borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                                borders_outerV=True, borders_innerV=True, delay_search=True):
                                                #dpg.bind_item_font(dpg.last_item(), bold_font)
                                                dpg.add_table_column(label="Name")
                                                dpg.add_table_column(label="RT60 (ms)")
                                                for i in range(len(CN.AC_SPACE_LIST_GUI)):
                                                    with dpg.table_row():
                                                        for j in range(2):
                                                            if j == 0:#name
                                                                dpg.add_text(CN.AC_SPACE_LIST_GUI[i])
                                                            else:#rt60
                                                                dpg.add_text(CN.AC_SPACE_MEAS_R60[i])
                                                        
                                        with dpg.group():
                                            dpg.add_text("Spatial Resolutions")
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                                borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                                borders_outerV=True, borders_innerV=True, delay_search=True):
                                                #dpg.bind_item_font(dpg.last_item(), bold_font)
                                                dpg.add_table_column(label="Resolution")
                                                dpg.add_table_column(label="Elevation Range")
                                                dpg.add_table_column(label="Elev. Steps")
                                                dpg.add_table_column(label="Azimuth Range")
                                                dpg.add_table_column(label="Azim. Steps")
                                                for i in range(len(CN.SPATIAL_RES_LIST)):
                                                    with dpg.table_row():
                                                        for j in range(5):
                                                            if j == 0:#Resolution
                                                                dpg.add_text(CN.SPATIAL_RES_LIST[i])
                                                            elif j == 1:#Elevation Range
                                                                dpg.add_text(CN.SPATIAL_RES_ELEV_RNG[i],wrap =100)
                                                            elif j == 2:#Elevation Steps
                                                                dpg.add_text(CN.SPATIAL_RES_ELEV_STP[i])
                                                            elif j == 3:#Azimuth Range
                                                                dpg.add_text(CN.SPATIAL_RES_AZIM_RNG[i])
                                                            elif j == 4:#Azimuth Steps
                                                                dpg.add_text(CN.SPATIAL_RES_AZIM_STP[i])
                                            dpg.add_text("")                    
                                            dpg.add_text("Supported Input SOFA Conventions")
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
                                            
                                            dpg.add_text("Supported Output SOFA Conventions")
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
                                    dpg.add_radio_button(CN.SAMPLE_RATE_LIST, horizontal=True, tag= "wav_sample_rate", default_value=sample_freq_loaded, callback=update_brir_param )
                                #dpg.add_text("          ")
                                with dpg.group():
                                    dpg.add_text("Select Bit Depth")
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_radio_button(CN.BIT_DEPTH_LIST, horizontal=True, tag= "wav_bit_depth", default_value=bit_depth_loaded, callback=update_brir_param)
                                    
                            #output locations
                            with dpg.child_window(width=371, height=139):
                                with dpg.group(horizontal=True):
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_text("Output Locations", tag='out_dir_title')
                                    with dpg.tooltip("out_dir_title"):
                                        dpg.add_text("'EqualizerAPO\config' directory should be selected if using Equalizer APO")
                                        dpg.add_text("Main outputs will be saved under 'ASH-Outputs' sub directory") 
                                        dpg.add_text("HeSuVi outputs will be saved in 'EqualizerAPO\config\HeSuVi'")  
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_text("           ")
                                    dpg.add_button(label="Open Folder",user_data="",tag="open_folder_tag", callback=open_output_folder)
                                    dpg.add_text("      ")
                                    dpge.add_file_browser(width=800,height=600,label='Change Folder',show_as_window=True, dirs_only=True,show_ok_cancel=True, allow_multi_selection=False, collapse_sequences=True,callback=show_selected_folder)
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
    
                with dpg.collapsing_header(label="Developer Tools",show=CN.SHOW_DEV_TOOLS):
                    with dpg.group(horizontal=True):
                        with dpg.group():
                    
                            #Section for HRIRs and BRIRs
                            with dpg.child_window(width=300, height=110):
                                dpg.add_text("Regenerate Generic Room Reverberation Data")
                                dpg.add_button(label="Click Here to Regenerate",user_data="",tag="brir_reverb_tag", callback=generate_brir_reverb)
                                with dpg.tooltip("brir_reverb_tag"):
                                    dpg.add_text("This will regenerate the Generic Room reverberation data used to generate BRIRs. Requires brir_dataset_compensated in data\interim folder. It may take some time to process")
                                dpg.add_text("Regenerate HRIR Dataset")
                                dpg.add_button(label="Click Here to Regenerate",user_data="",tag="hrir_dataset_tag", callback=generate_hrir_dataset)
                                with dpg.tooltip("hrir_dataset_tag"):
                                    dpg.add_text("This will regenerate the HRIR dataset used to generate BRIRs. Requires hrir_dataset_compensated .mat files in data\interim folder.")
                            #Section for database
                            with dpg.child_window(width=300, height=60):
                                dpg.add_text("Rebuild HpCF Database from WAVs")
                                dpg.add_button(label="Click Here to Rebuild",user_data="",tag="hpcf_db_create", callback=create_db_from_wav)
                                with dpg.tooltip("hpcf_db_create"):
                                    dpg.add_text("This will rebuild the HpCF database from WAV FIRs. Requires WAV FIRs in data\interim\hpcf_wavs folder")
                            #Section for detailing HpCFs
                            with dpg.child_window(width=300, height=60):
                                dpg.add_text("Print Summary of Recent HpCFs")
                                dpg.add_button(label="Click Here to Print",user_data="",tag="hpcf_print_summary_tag", callback=print_summary)
                        with dpg.group():
                            #Section for HpCF bulk functions
                            with dpg.child_window(width=340, height=120):
                                with dpg.group(horizontal=True):
                                    with dpg.group():
                                        dpg.add_text("Calculate HpCF Averages")
                                        dpg.add_button(label="Click Here to Calculate",user_data="",tag="hpcf_average_tag", callback=calc_hpcf_averages)
                                        with dpg.tooltip("hpcf_average_tag"):
                                            dpg.add_text("This will create averaged HpCFs for headphones with multiple samples and no existing average")
                                        dpg.add_text("Renumber HpCF Samples")
                                        dpg.add_button(label="Click Here to Renumber",user_data="",tag="hpcf_renumber_tag", callback=renumber_hpcf_samples)
                                        with dpg.tooltip("hpcf_renumber_tag"):
                                            dpg.add_text("This will renumber HpCF samples to ensure consecutive numbers")
                                    with dpg.group():
                                        dpg.add_text("Calculate HpCF Variants")
                                        dpg.add_button(label="Click Here to Calculate",user_data="",tag="hpcf_variant_tag", callback=calc_hpcf_variants)
                                        dpg.add_text("Crop All HpCFs")
                                        dpg.add_button(label="Click Here to Crop",user_data="",tag="hpcf_crop_tag", callback=crop_hpcfs)
                            #Section for HpCFs
                            with dpg.child_window(width=340, height=180):
                                dpg.add_text("Enter Name of Headphone Measurements Folder")
                                dpg.add_input_text(label="input text", default_value="Folder Name",tag="hp_measurements_tag")
                                with dpg.tooltip("hp_measurements_tag"):
                                    dpg.add_text("Enter name of folder containing headphone measurements as it appears in data \ raw \ headphone_measurements")
                                dpg.add_checkbox(label="In-ear Headphone Set", default_value = False,tag="in_ear_set_tag")
                                with dpg.tooltip("in_ear_set_tag"):
                                    dpg.add_text("Select this option if measurements folder contains in ear headphones")
                                dpg.add_text("Generate HpCF Summary Sheet")
                                dpg.add_button(label="Click Here to Generate",user_data="",tag="hpcf_summary_tag", callback=generate_hpcf_summary)
                                with dpg.tooltip("hpcf_summary_tag"):
                                    dpg.add_text("Creates CSV containing summary of measurements in above folder")
                                dpg.add_text("Calculate new HpCFs")
                                dpg.add_button(label="Click Here to Calculate",user_data="",tag="hpcf_new_calc_tag", callback=calc_new_hpcfs) 
                                with dpg.tooltip("hpcf_new_calc_tag"):
                                    dpg.add_text("Calculates new HpCFs from measurements and adds to the database")
                        with dpg.group():
                            #Section for deleting HpCFs
                            with dpg.child_window(width=280, height=120):
                                dpg.add_text("Delete Headphone from DB (selected headphone)")
                                dpg.add_button(label="Click Here to Delete",user_data="",tag="hpcf_delete_hp_tag", callback=delete_hp)
                                dpg.add_text("Delete Sample from DB (selected sample)")
                                dpg.add_button(label="Click Here to Delete",user_data="",tag="hpcf_delete_sample_tag", callback=delete_sample)
        
                            #Section for modifying existing HpCFs
                            with dpg.child_window(width=280, height=180):
                                dpg.add_text("Enter New Name for Headphone")
                                dpg.add_input_text(label="input text", default_value="Headphone Name",tag="hpcf_rename_tag",width=190)
                                dpg.add_text("Rename Headphone (selected headphone)")
                                dpg.add_button(label="Click Here to Rename",user_data="",tag="hpcf_rename_hp_tag", callback=rename_hp_hp)
                                with dpg.tooltip("hpcf_rename_hp_tag"):
                                    dpg.add_text("Renames HpCFs for the selected headphone to the specified name")
                                dpg.add_text("Rename Headphone (selected sample)")
                                dpg.add_button(label="Click Here to Rename",user_data="",tag="hpcf_rename_sample_tag", callback=rename_hp_sample)
                                with dpg.tooltip("hpcf_rename_sample_tag"):
                                    dpg.add_text("Renames headphone field for the selected sample to the specified name")
                        with dpg.group():
                            with dpg.group(horizontal=True):
                                #Section for renaming existing HpCFs - find and replace
                                with dpg.child_window(width=420, height=120):
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Enter Sample Name to Replace")
                                            dpg.add_input_text(label="input text", default_value="Sample Name",tag="sample_find_tag",width=150)
                                            dpg.add_text("Enter New Name for Sample")
                                            dpg.add_input_text(label="input text", default_value="New Name",tag="sample_replace_tag",width=150)
                                        dpg.add_text("   ")    
                                        with dpg.group():
                                            dpg.add_text("Bulk Rename Sample")
                                            dpg.add_button(label="Click Here to Rename",user_data="",tag="bulk_rename_sample_tag", callback=bulk_rename_sample)
                                            with dpg.tooltip("bulk_rename_sample_tag"):
                                                dpg.add_text("Bulk renames sample name across all headphones")
                                #Section for running HRTF processing functions
                                with dpg.child_window(width=200, height=120):
                                    with dpg.group():
                                        dpg.add_text("Mono Cue Processing (DRIR)")
                                        dpg.add_button(label="Click Here to Run",user_data="",tag="mono_cue_tag", callback=run_mono_cue)
                                        dpg.add_text("Mono Cue Processing (HP)")
                                        dpg.add_button(label="Click Here to Run",user_data="",tag="mono_cue_hp_tag", callback=run_mono_cue_hp)
                            #Section for running AIR processing functions
                            with dpg.child_window(width=720, height=180):
                                with dpg.group(horizontal=True):
                                    with dpg.group():
                                        dpg.add_text("Enter name of AIR dataset")
                                        dpg.add_input_text(label="input text", default_value="dataset_name",tag="air_dataset_name_tag",width=150)
                                        #dpg.add_input_int(label="Desired AIR sets", width=75,min_value=0, max_value=20, tag='air_output_sets_tag',default_value=0)
                                with dpg.group(horizontal=True):
                                    with dpg.group():
                                        dpg.add_text("Raw AIRs to AIR Dataset")
                                        dpg.add_button(label="Click Here to Run",user_data="",tag="raw_air_to_dataset_tag", callback=run_raw_air_to_dataset)
                                    dpg.add_text("  ")
                                    with dpg.group():
                                        dpg.add_text("AIR to BRIR Estimation")
                                        dpg.add_button(label="Click Here to Run",user_data="",tag="air_to_brir_tag", callback=run_air_to_brir)
                                    dpg.add_text("  ")
                                    with dpg.group():
                                        dpg.add_text("Raw BRIR Compensation")
                                        dpg.add_button(label="Click Here to Run",user_data="",tag="raw_brir_comp_tag", callback=run_raw_to_brir)
                                    dpg.add_text("  ")
                                    with dpg.group():
                                        dpg.add_text("Raw AIR extraction")
                                        dpg.add_button(label="Click Here to Run",user_data="",tag="air_extraction_tag", callback=run_extract_airs)
                                    dpg.add_text("  ")
                                    with dpg.group():
                                        dpg.add_text("Split AIRs to Set")
                                        dpg.add_button(label="Click Here to Run",user_data="",tag="split_air_set_tag", callback=run_split_airs_to_set)
                                    dpg.add_text("  ")
                                with dpg.group(horizontal=True):
                                    with dpg.group():
                                        dpg.add_text("Calculate reverb target resp.")
                                        dpg.add_button(label="Click Here to Run",user_data="",tag="calc_reverb_target_tag", callback=calc_reverb_target)
                                    dpg.add_text("  ")
                                    with dpg.group():
                                        dpg.add_text("Sub BRIR Generation")
                                        dpg.add_button(label="Click Here to Run",user_data="",tag="calc_sub_brir_tag", callback=run_sub_brir_calc)
                                    dpg.add_text("  ")
                                    with dpg.group():
                                        dpg.add_text("Generate room target set")
                                        dpg.add_button(label="Click Here to Run",user_data="",tag="calc_room_target_tag", callback=run_room_target_calc)
        
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
                                dpg.add_checkbox(label="Auto Check for Updates", default_value = auto_check_updates_loaded,  tag='check_updates_start_tag', callback=save_settings)
                                dpg.add_text("Check for App Updates")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Check for Updates",user_data="",tag="app_version_tag", callback=check_app_version)
                                with dpg.tooltip("app_version_tag"):
                                    dpg.add_text("This will check for updates to the app and show versions in the log")   
                            with dpg.child_window(width=224, height=150):
                                dpg.add_text("Headphone Correction Filters")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_separator()
                                dpg.add_text("Check for Headphone Filter Updates")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Check for Updates",user_data="",tag="hpcf_db_version_tag", callback=check_db_version)
                                with dpg.tooltip("hpcf_db_version_tag"):
                                    dpg.add_text("This will check for updates to the headphone correction filter dataset and show versions in the log")
                                dpg.add_text("Download Latest Headphone Filters")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Download Latest Dataset",user_data="",tag="hpcf_db_download_tag", callback=download_latest_db)
                                with dpg.tooltip("hpcf_db_download_tag"):
                                    dpg.add_text("This will download latest version of the dataset and replace local version")
                            with dpg.child_window(width=224, height=150):
                                dpg.add_text("Acoustic Spaces")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_separator()
                                dpg.add_text("Check for Acoustic Space Updates")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Check for Updates",user_data="",tag="as_version_tag", callback=check_as_versions)
                                with dpg.tooltip("as_version_tag"):
                                    dpg.add_text("This will check for updates to acoustic space datasets and show new versions in the log")
                                dpg.add_text("Download Acoustic Space Updates")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Download Latest Datasets",user_data="",tag="as_download_tag", callback=download_latest_as_sets)
                                with dpg.tooltip("as_download_tag"):
                                    dpg.add_text("This will download any updates to acoustic space datasets and replace local versions")
                            with dpg.child_window(width=224, height=150):
                                dpg.add_text("HRTF Datasets")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_separator()
                                dpg.add_text("Check for HRTF Dataset Updates")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Check for Updates",user_data="",tag="hrtf_version_tag", callback=check_hrtf_versions)
                                with dpg.tooltip("hrtf_version_tag"):
                                    dpg.add_text("This will check for updates to HRTF datasets and show new updates in the log")
                                dpg.add_text("Update HRTF Dataset List")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Download Latest Dataset List",user_data="",tag="hrtf_download_tag", callback=download_latest_hrtf_sets)
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
                                dpg.add_button(label="Delete Headphone Filters",user_data="",tag="remove_hpcfs_tag", callback=remove_hpcfs)
                                with dpg.tooltip("remove_hpcfs_tag"):
                                    dpg.add_text("Warning: this will delete all headphone filters that have been exported to the output directory")
                                dpg.add_text("Delete All Exported Binaural Datasets")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Delete Binaural Datasets",user_data="",tag="remove_brirs_tag", callback=remove_brirs)
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
                                        dpg.add_combo(CN.HRTF_SYM_LIST, default_value=hrtf_symmetry_loaded, width=130, callback=qc_update_brir_param, tag='force_hrtf_symmetry')
                                        with dpg.tooltip("force_hrtf_symmetry"):
                                            dpg.add_text("This will mirror the left or right sides of the HATS / dummy head") 
                                            dpg.add_text("Applies to the direct sound. Reverberation is not modified") 
                                        dpg.add_text("Early Reflection Delay (ms)")
                                        dpg.bind_item_font(dpg.last_item(), bold_font)
                                        dpg.add_input_float(label=" ",width=140, format="%.1f", tag='er_delay_time_tag', min_value=CN.ER_RISE_MIN, max_value=CN.ER_RISE_MAX, default_value=er_rise_loaded,min_clamped=True, max_clamped=True, callback=qc_update_brir_param)
                                        with dpg.tooltip("er_delay_time_tag"):
                                            dpg.add_text("This will increase the time between the direct sound and early reflections") 
                                            dpg.add_text("This can be used to increase perceived distance") 
                                            dpg.add_text("Min. 0ms, Max. 10ms")
                                    with dpg.group(): 
                                        dpg.add_text("SOFA Export Convention")
                                        dpg.bind_item_font(dpg.last_item(), bold_font)
                                        dpg.add_combo(CN.SOFA_OUTPUT_CONV, default_value=sofa_exp_conv_loaded, width=150, callback=save_settings, tag='sofa_exp_conv')
                                    
                #section for logging
                with dpg.child_window(width=1690, height=482, tag="console_window"):
                    dpg.add_text("Log")
                    dpg.bind_item_font(dpg.last_item(), bold_font)


    dpg.setup_dearpygui()
    logz=logger.mvLogger(parent="console_window")

    #section to log tool version on startup
    #log results
    log_string = 'Started ASH Toolset - Version: ' + __version__
    hf.log_with_timestamp(log_string, logz)
    
    #check for updates on start
    if auto_check_updates_loaded == True:
        #start thread
        thread = threading.Thread(target=check_all_updates, args=(), daemon=True)
        thread.start()

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