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
    
    with open(CN.METADATA_FILE) as fp:
        _info = json.load(fp)
    __version__ = _info['version']
    
    #
    #program code
    #
    
    #start with flat response
    impulse=np.zeros(CN.N_FFT)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))

    #default values
    sample_freq_default=CN.SAMPLE_RATE_LIST[0]
    bit_depth_default=CN.BIT_DEPTH_LIST[0]
    brir_hp_type_default='Over/On-Ear Headphones - High Strength'
    hrtf_default=CN.HRTF_LIST_NUM[0]
    spatial_res_default=CN.SPATIAL_RES_LIST[0]
    room_target_default=CN.ROOM_TARGET_LIST[1]
    direct_gain_default=2.0
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
    autosize_win_default=True
    gui_win_width_default=1738#1660
    gui_win_height_default=1039#1034
    audio_channels_default=CN.AUDIO_CHANNELS[0]
    show_filter_sect_default=True
    show_eapo_sect_default=True
    auto_check_updates_default=False
    #E-APO config related settings
    e_apo_mute_default=False
    e_apo_gain_default=0
    e_apo_elev_angle_default=0
    e_apo_az_angle_fl_default=-30  
    e_apo_az_angle_fr_default=30  
    e_apo_az_angle_c_default=0  
    e_apo_az_angle_sl_default=-90  
    e_apo_az_angle_sr_default=90  
    e_apo_az_angle_rl_default=-135 
    e_apo_az_angle_rr_default=135 
    e_apo_enable_default=False
    e_apo_enable_hpcf_default=False
    e_apo_enable_brir_default=False
    e_apo_hp_selected_default=''
    e_apo_sample_selected_default=''
    e_apo_brir_selected_default=''

    #acoustic space
    ac_space_default=CN.AC_SPACE_LIST_GUI[0]
    
    #loaded values - start with defaults
    sample_freq_loaded=sample_freq_default
    bit_depth_loaded=bit_depth_default
    brir_hp_type_loaded=brir_hp_type_default
    hrtf_loaded=hrtf_default
    spatial_res_loaded=spatial_res_default
    room_target_loaded=room_target_default
    direct_gain_loaded=direct_gain_default
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
    autosize_win_loaded = autosize_win_default
    audio_channels_loaded=audio_channels_default
    show_filter_sect_loaded=show_filter_sect_default
    show_eapo_sect_loaded=show_eapo_sect_default
    auto_check_updates_loaded=auto_check_updates_default
    #E-APO config related settings
    e_apo_mute_fl_loaded=e_apo_mute_default
    e_apo_mute_fr_loaded=e_apo_mute_default
    e_apo_mute_c_loaded=e_apo_mute_default
    e_apo_mute_sl_loaded=e_apo_mute_default
    e_apo_mute_sr_loaded=e_apo_mute_default
    e_apo_mute_rl_loaded=e_apo_mute_default
    e_apo_mute_rr_loaded=e_apo_mute_default
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
    e_apo_enable_loaded=e_apo_enable_default
    e_apo_enable_hpcf_loaded=e_apo_enable_hpcf_default
    e_apo_enable_brir_loaded=e_apo_enable_brir_default
    e_apo_hp_selected_loaded = e_apo_hp_selected_default
    e_apo_sample_selected_loaded = e_apo_sample_selected_default
    e_apo_brir_selected_loaded = e_apo_brir_selected_default
    #acoustic space
    ac_space_loaded=ac_space_default
    #thread variables
    e_apo_conf_lock = threading.Lock()
    
    #code to get gui width based on windows resolution
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
    
    #get equalizer APO path
    try:
        key = wrg.OpenKey(wrg.HKEY_LOCAL_MACHINE, "Software\\EqualizerAPO")
        value =  wrg.QueryValueEx(key, "InstallPath")[0]
        e_apo_path= pjoin(value, 'config')
        
        #E APO was found in registry, so also enable config functions
        #e_apo_enable_loaded=True
        
    except:
        e_apo_path = None
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
            autosize_win_loaded=ast.literal_eval(config['DEFAULT']['autosize_win'])
            show_filter_sect_loaded=ast.literal_eval(config['DEFAULT']['show_filter_section'])
            show_eapo_sect_loaded=ast.literal_eval(config['DEFAULT']['show_eapo_section'])
            auto_check_updates_loaded=ast.literal_eval(config['DEFAULT']['auto_check_updates'])
            base_folder_loaded = config['DEFAULT']['path']
            primary_path=base_folder_loaded
            primary_ash_path=pjoin(base_folder_loaded, CN.PROJECT_FOLDER)
            #E-APO config related settings
            e_apo_mute_fl_loaded=ast.literal_eval(config['DEFAULT']['mute_fl'])
            e_apo_mute_fr_loaded=ast.literal_eval(config['DEFAULT']['mute_fr'])
            e_apo_mute_c_loaded=ast.literal_eval(config['DEFAULT']['mute_c'])
            e_apo_mute_sl_loaded=ast.literal_eval(config['DEFAULT']['mute_sl'])
            e_apo_mute_sr_loaded=ast.literal_eval(config['DEFAULT']['mute_sr'])
            e_apo_mute_rl_loaded=ast.literal_eval(config['DEFAULT']['mute_rl'])
            e_apo_mute_rr_loaded=ast.literal_eval(config['DEFAULT']['mute_rr'])
            e_apo_gain_fl_loaded=int(config['DEFAULT']['gain_fl'])
            e_apo_gain_fr_loaded=int(config['DEFAULT']['gain_fr'])
            e_apo_gain_c_loaded=int(config['DEFAULT']['gain_c'])
            e_apo_gain_sl_loaded=int(config['DEFAULT']['gain_sl'])
            e_apo_gain_sr_loaded=int(config['DEFAULT']['gain_sr'])
            e_apo_gain_rl_loaded=int(config['DEFAULT']['gain_rl'])
            e_apo_gain_rr_loaded=int(config['DEFAULT']['gain_rr'])
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
            e_apo_enable_loaded=ast.literal_eval(config['DEFAULT']['enable_e_apo'])
            e_apo_enable_hpcf_loaded=ast.literal_eval(config['DEFAULT']['enable_hpcf'])
            e_apo_enable_brir_loaded=ast.literal_eval(config['DEFAULT']['enable_brir'])
            e_apo_hp_selected_loaded = config['DEFAULT']['headphone_selected']
            e_apo_sample_selected_loaded = config['DEFAULT']['sample_selected']
            e_apo_brir_selected_loaded = config['DEFAULT']['brir_set_selected']
            audio_channels_loaded=config['DEFAULT']['channel_config']

            #acoustic space
            ac_space_loaded=config['DEFAULT']['acoustic_space']
            
        else:
            raise ValueError('Settings not loaded due to version mismatch')
        
    except:
        if e_apo_path is not None:
            primary_path = e_apo_path
            primary_ash_path = pjoin(e_apo_path, CN.PROJECT_FOLDER)
        else:
            primary_path = 'C:\\Program Files\\EqualizerAPO\\config'
            primary_ash_path = 'C:\\Program Files\\EqualizerAPO\\config\\' + CN.PROJECT_FOLDER
    #hesuvi path
    if 'EqualizerAPO' in primary_path:
        primary_hesuvi_path = pjoin(primary_path,'HeSuVi')#stored outside of project folder (within hesuvi installation)
    else:
        primary_hesuvi_path = pjoin(primary_path, CN.PROJECT_FOLDER,'HeSuVi')#stored within project folder

    
    #adjust window size if setting enabled
    if autosize_win_loaded == True:
        if show_filter_sect_loaded == True and show_eapo_sect_loaded == False:
            if gui_win_height_loaded > 733:
                gui_win_height_loaded=733
        elif show_filter_sect_loaded == False and show_eapo_sect_loaded == True:
            if gui_win_height_loaded > 430:
                gui_win_height_loaded=430
                gui_win_width_loaded=1650
            
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
    if spatial_res_loaded == 'Max':
        hrtf_list_loaded = CN.HRTF_LIST_FULL_RES_NUM
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
        hrtf_list_loaded = CN.HRTF_LIST_NUM
        sofa_brir_exp_loaded=False
        sofa_brir_exp_show=False
        sofa_brir_tooltip_show=False
    else:
        hrtf_list_loaded = CN.HRTF_LIST_NUM
    
 
    #
    # HpCF Database code
    #
    database = pjoin(CN.DATA_DIR_OUTPUT,'hpcf_database.db')
    # create a database connection
    conn = hpcf_functions.create_connection(database)
    brands_list = hpcf_functions.get_brand_list(conn)
    hp_list_default = hpcf_functions.get_headphone_list(conn, brands_list[0])#index 0
    headphone_default = hp_list_default[0]
    sample_list_default = hpcf_functions.get_samples_list(conn, headphone_default)
    sample_default = 'Sample A' #sample_list_default[0]
    
    default_hpcf_settings = {'headphone': headphone_default, 'hpcf_export': 1, 'fir_export': int(fir_hpcf_exp_loaded), 'fir_stereo_export': int(fir_st_hpcf_exp_loaded), 'geq_export': int(geq_hpcf_exp_loaded), 'geq_31_export': int(geq_31_exp_loaded), 'geq_103_export': 0, 'hesuvi_export': int(hesuvi_hpcf_exp_loaded), 'eapo_export': int(eapo_hpcf_exp_loaded)}
    
    #
    # Equalizer APO related code
    #
    #exported brir lists
    brir_list_out_default = e_apo_config_creation.get_exported_brir_list(primary_path)
    if e_apo_brir_selected_loaded and e_apo_brir_selected_loaded != '' and e_apo_brir_selected_loaded in brir_list_out_default:#apply loaded brir set
        brir_out_default=e_apo_brir_selected_loaded
    elif brir_list_out_default and brir_list_out_default != None:#or apply first brir set in list
        brir_out_default = brir_list_out_default[0]
    else:
        brir_out_default=''
    
    #Whenever a brir set is selected, determine spatial resolution, then get elevation list and update list in gui element
    if brir_out_default and brir_out_default != None:
        spatial_res_sel = e_apo_config_creation.get_spatial_res_from_dir(primary_path=primary_path, brir_set=brir_out_default)
        elevation_list_sel = hf.get_elevation_list(spatial_res_sel)
    else:
        elevation_list_sel= CN.ELEV_ANGLES_WAV_LOW
    
    #exported headphone and sample lists
    hp_list_out_default = e_apo_config_creation.get_exported_hp_list(primary_path)
    if e_apo_hp_selected_loaded and e_apo_hp_selected_loaded != '' and e_apo_hp_selected_loaded in hp_list_out_default:#apply loaded headphone
        headphone_out_default=e_apo_hp_selected_loaded
        sample_list_out_default = e_apo_config_creation.get_exported_sample_list(headphone=headphone_out_default,primary_path=primary_path)
        if e_apo_sample_selected_loaded and e_apo_sample_selected_loaded != '' and e_apo_sample_selected_loaded in sample_list_out_default:#apply loaded sample
            sample_out_default=e_apo_sample_selected_loaded
        elif sample_list_out_default and sample_list_out_default != None:#or apply first sample in list
            sample_out_default = sample_list_out_default[0]
        else:
            sample_out_default=''
    elif hp_list_out_default and hp_list_out_default != None:#or apply first headphone and sample in list
        headphone_out_default = hp_list_out_default[0]
        sample_list_out_default = e_apo_config_creation.get_exported_sample_list(headphone=headphone_out_default,primary_path=primary_path)
        if sample_list_out_default and sample_list_out_default != None:
            sample_out_default = sample_list_out_default[0]
        else:
            sample_out_default=''
    else:
        headphone_out_default=''
        sample_list_out_default=[]
        sample_out_default=''
  
    
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
        
        #also update hpcf button user data
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("hpcf_tag")
        modified_dict = current_dict.copy()
        modified_dict.update({'headphone': headphone})
        #update user data
        dpg.configure_item('hpcf_tag',user_data=modified_dict)
   
            
            
        
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
            
            #also update hpcf button user data
            #get user data from process hpcf button, contains a dict
            current_dict = dpg.get_item_user_data("hpcf_tag")
            modified_dict = current_dict.copy()
            modified_dict.update({'headphone': headphone})
            #update user data
            dpg.configure_item('hpcf_tag',user_data=modified_dict)
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
            
            #also update hpcf button user data
            #get user data from process hpcf button, contains a dict
            current_dict = dpg.get_item_user_data("hpcf_tag")
            modified_dict = current_dict.copy()
            modified_dict.update({'headphone': headphone})
            #update user data
            dpg.configure_item('hpcf_tag',user_data=modified_dict)
    
    
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
        
        #also update hpcf button user data
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("hpcf_tag")
        modified_dict = current_dict.copy()
        modified_dict.update({'headphone': headphone})
        #update user data
        dpg.configure_item('hpcf_tag',user_data=modified_dict)
        
        #reset progress
        dpg.set_value("progress_bar_hpcf", 0)
        dpg.configure_item("progress_bar_hpcf", overlay = 'Progress')
        

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
        
        #also update hpcf button user data
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("hpcf_tag")
        modified_dict = current_dict.copy()
        modified_dict.update({'headphone': headphone})
        #update user data
        dpg.configure_item('hpcf_tag',user_data=modified_dict)
        
        #reset progress
        dpg.set_value("progress_bar_hpcf", 0)
        dpg.configure_item("progress_bar_hpcf", overlay = 'Progress')
        
 
    def plot_sample(sender, app_data, user_data):
        """ 
        GUI function to plot a selected sample
        """
        
        headphone = user_data
        sample = app_data 
        hpcf_functions.hpcf_to_plot(conn, headphone, sample, plot_type=1)
 
    def export_fir_toggle(sender, app_data):
        """ 
        GUI function to update hpcf dictionary based on toggle
        """
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("hpcf_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        modified_dict.update({'fir_export': int(app_data)})
 
        #update user data
        dpg.configure_item('hpcf_tag',user_data=modified_dict)
        save_settings()
        #reset progress
        dpg.set_value("progress_bar_hpcf", 0)
        dpg.configure_item("progress_bar_hpcf", overlay = 'Progress')
        
    def export_fir_stereo_toggle(sender, app_data):
        """ 
        GUI function to update hpcf dictionary based on toggle
        """
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("hpcf_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        modified_dict.update({'fir_stereo_export': int(app_data)})
  
        #update user data
        dpg.configure_item('hpcf_tag',user_data=modified_dict)
        save_settings()
        #reset progress
        dpg.set_value("progress_bar_hpcf", 0)
        dpg.configure_item("progress_bar_hpcf", overlay = 'Progress')

    def export_geq_toggle(sender, app_data):
        """ 
        GUI function to update hpcf dictionary based on toggle
        """
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("hpcf_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        modified_dict.update({'geq_export': int(app_data)})
  
        #update user data
        dpg.configure_item('hpcf_tag',user_data=modified_dict)
        save_settings()
        #reset progress
        dpg.set_value("progress_bar_hpcf", 0)
        dpg.configure_item("progress_bar_hpcf", overlay = 'Progress')
    

    def export_geq_31_toggle(sender, app_data):
        """ 
        GUI function to update hpcf dictionary based on toggle
        """
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("hpcf_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        modified_dict.update({'geq_31_export': int(app_data)})
   
        #update user data
        dpg.configure_item('hpcf_tag',user_data=modified_dict)
        save_settings()
        #reset progress
        dpg.set_value("progress_bar_hpcf", 0)
        dpg.configure_item("progress_bar_hpcf", overlay = 'Progress')

    def export_hesuvi_hpcf_toggle(sender, app_data):
        """ 
        GUI function to update hpcf dictionary based on toggle
        """
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("hpcf_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        modified_dict.update({'hesuvi_export': int(app_data)})
  
        #update user data
        dpg.configure_item('hpcf_tag',user_data=modified_dict)
        save_settings()
        #reset progress
        dpg.set_value("progress_bar_hpcf", 0)
        dpg.configure_item("progress_bar_hpcf", overlay = 'Progress')

    def export_eapo_hpcf_toggle(sender, app_data):
        """ 
        GUI function to update hpcf dictionary based on toggle
        """
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("hpcf_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        modified_dict.update({'eapo_export': int(app_data)})
  
        #update user data
        dpg.configure_item('hpcf_tag',user_data=modified_dict)
        save_settings()
        #reset progress
        dpg.set_value("progress_bar_hpcf", 0)
        dpg.configure_item("progress_bar_hpcf", overlay = 'Progress')


    def process_hpcfs(sender, app_data, user_data):
        """ 
        GUI function to process HPCFs
        """
        
        output_path = dpg.get_value('selected_folder_base')
        
        current_dict = dpg.get_item_user_data("hpcf_tag")
        headphone = current_dict.get('headphone')
        fir_export = current_dict.get('fir_export')
        fir_stereo_export = current_dict.get('fir_stereo_export')
        geq_export = current_dict.get('geq_export')
        geq_31_export = current_dict.get('geq_31_export')
        geq_103_export = current_dict.get('geq_103_export')
        hesuvi_export = current_dict.get('hesuvi_export')
        eapo_export = current_dict.get('eapo_export')
        hpcf_export = current_dict.get('hpcf_export')
        
        samp_freq_str = dpg.get_value('wav_sample_rate')
        samp_freq_int = CN.SAMPLE_RATE_DICT.get(samp_freq_str)
        bit_depth_str = dpg.get_value('wav_bit_depth')
        bit_depth = CN.BIT_DEPTH_DICT.get(bit_depth_str)
        
        
        if hpcf_export == 1:
            hpcf_functions.hpcf_to_file_bulk(conn, primary_path=output_path, headphone=headphone, fir_export = fir_export, fir_stereo_export = fir_stereo_export, geq_export = geq_export, samp_freq=samp_freq_int, bit_depth=bit_depth, 
                                             geq_31_export = geq_31_export, geq_103_export = geq_103_export, hesuvi_export = hesuvi_export, eapo_export=eapo_export, gui_logger=logz, report_progress=True)
   
            #also populate headphone list in E-APO config section
            hp_list_out_latest = e_apo_config_creation.get_exported_hp_list(output_path)
            dpg.configure_item('e_apo_load_hp',items=hp_list_out_latest)
            #check if sample list is empty, if empty then populate with sample list for new hp
            sample_out_latest = dpg.get_value('e_apo_load_sample')
            if not sample_out_latest or sample_out_latest == None:
                headphone_out_selected = hp_list_out_latest[0]
                sample_list_out_new = e_apo_config_creation.get_exported_sample_list(headphone=headphone_out_selected,primary_path=output_path)
                dpg.configure_item('e_apo_load_sample',items=sample_list_out_new)
                    
            
        #finally rewrite config file
        e_apo_config_acquire()
        
    #
    ## GUI Functions - BRIRs
    #
    
    def select_spatial_resolution(sender, app_data):
        """ 
        GUI function to update spatial resolution based on input
        """
        
        #update hrtf list based on spatial resolution
        #also update file format selection based on spatial resolution
        #set some to false and hide irrelevant options
        if app_data == 'Max':
            dpg.configure_item('brir_hrtf',items=CN.HRTF_LIST_FULL_RES_NUM)
            
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
            dpg.configure_item('brir_hrtf',items=CN.HRTF_LIST_NUM)  

            dpg.configure_item("ts_brir_toggle", show=True)
            dpg.configure_item("hesuvi_brir_toggle", show=True)
            dpg.configure_item("eapo_brir_toggle", show=True)
            dpg.configure_item("sofa_brir_toggle", show=True)
 
            dpg.configure_item("ts_brir_tooltip", show=True)
            dpg.configure_item("hesuvi_brir_tooltip", show=True)
            dpg.configure_item("eapo_brir_tooltip", show=True)
            dpg.configure_item("sofa_brir_tooltip", show=True)
            
        else:
            dpg.configure_item('brir_hrtf',items=CN.HRTF_LIST_NUM)  

            dpg.set_value("sofa_brir_toggle", False)

            dpg.configure_item("ts_brir_toggle", show=True)
            dpg.configure_item("hesuvi_brir_toggle", show=True)
            dpg.configure_item("eapo_brir_toggle", show=True)
            dpg.configure_item("sofa_brir_toggle", show=False)

            dpg.configure_item("ts_brir_tooltip", show=True)
            dpg.configure_item("hesuvi_brir_tooltip", show=True)
            dpg.configure_item("eapo_brir_tooltip", show=True)
            dpg.configure_item("sofa_brir_tooltip", show=False)
        
        
        #reset progress bar
        reset_progress()
        
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
        reset_progress()
        
        save_settings()
 
    def select_hrtf(sender, app_data):
        """ 
        GUI function to update brir based on input
        """
        
        #run plot
        spat_res = dpg.get_value("brir_spat_res")
        spat_res_int = CN.SPATIAL_RES_LIST.index(spat_res)
        hrtf = dpg.get_value("brir_hrtf")
        if spat_res == 'Max':
            hrtf_type = CN.HRTF_LIST_FULL_RES_NUM.index(hrtf)+1
        else:
            hrtf_type = CN.HRTF_LIST_NUM.index(hrtf)+1
        hrtf_index = hrtf_type-1
        if spat_res_int <= 2:
            try:
                npy_fname = pjoin(CN.DATA_DIR_INT, 'hrir_dataset_compensated_high.npy')
                #load npy files
                hrir_list = np.load(npy_fname)
                hrir_selected = hrir_list[hrtf_index]
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
                plot_tile = 'HRTF sample: ' + hrtf + ' 0° elevation, 30° azimuth, right ear'
                hf.plot_data(mag_response, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=0, level_ends=1, plot_type=1)
                
            except:
                pass
     
        #reset progress bar
        reset_progress()
        
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
        reset_progress()
        
        save_settings()
    
    
    def update_direct_gain(sender, app_data):
        """ 
        GUI function to update brir based on input
        """
        d_gain=app_data
        dpg.set_value("direct_gain_slider", d_gain)

        #reset progress bar
        reset_progress()
        
        save_settings()
    
    def update_direct_gain_slider(sender, app_data):
        """ 
        GUI function to update brir based on input
        """
        
        d_gain=app_data
        dpg.set_value("direct_gain", d_gain)

        #reset progress bar
        reset_progress()
        
        save_settings()
 
    def update_brir_param(sender, app_data):
        """ 
        GUI function to update brir based on input
        """

        #reset progress bar
        reset_progress()
        
        save_settings()
    
        
    def export_brir_toggle(sender, app_data):
        """ 
        GUI function to update settings based on toggle
        """
        
        #reset progress bar
        reset_progress()
        
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

        brir_directional_export = int(dpg.get_value("dir_brir_toggle"))
  
        brir_ts_export = int(dpg.get_value("ts_brir_toggle"))
  
        hesuvi_export = int(dpg.get_value("hesuvi_brir_toggle"))
  
        eapo_export = int(dpg.get_value("eapo_brir_toggle"))
  
        sofa_export = int(dpg.get_value("sofa_brir_toggle"))
        
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
   
        brir_set_export = CN.BRIR_EXPORT_ENABLE
        
        samp_freq_str = dpg.get_value('wav_sample_rate')
        samp_freq_int = CN.SAMPLE_RATE_DICT.get(samp_freq_str)
        bit_depth_str = dpg.get_value('wav_bit_depth')
        bit_depth = CN.BIT_DEPTH_DICT.get(bit_depth_str)
        
        spat_res = dpg.get_value("brir_spat_res")
        spat_res_int = CN.SPATIAL_RES_LIST.index(spat_res)
        
        hrtf = dpg.get_value("brir_hrtf")
        if spat_res == 'Max':
            hrtf_type = CN.HRTF_LIST_FULL_RES_NUM.index(hrtf)+1
        else:
            hrtf_type = CN.HRTF_LIST_NUM.index(hrtf)+1
 
        if brir_set_export == 1:
        
            """
            #Run BRIR integration
            """
            brir_gen = brir_generation.generate_integrated_brir(hrtf_type=hrtf_type, direct_gain_db=direct_gain_db, room_target=room_target, spatial_res=spat_res_int, 
                                                                pinna_comp=pinna_comp, report_progress=True, gui_logger=logz, acoustic_space=ac_space_src)
            
            """
            #Run BRIR export
            """
            #calculate name
            #depends on reverb reduction
            #depends on spatial resolution
            if spat_res == 'Max':
                brir_name = CN.HRTF_LIST_FULL_RES_SHORT[hrtf_type-1] + '_'+ac_space_short + '_' + str(direct_gain_db) + 'dB_' + CN.ROOM_TARGET_LIST_SHORT[room_target] + '_' + CN.HP_COMP_LIST_SHORT[pinna_comp]
            else:    
                brir_name = CN.HRTF_LIST_SHORT[hrtf_type-1] + '_'+ac_space_short + '_' + str(direct_gain_db) + 'dB_' + CN.ROOM_TARGET_LIST_SHORT[room_target] + '_' + CN.HP_COMP_LIST_SHORT[pinna_comp]
                
            
            if brir_gen.size != 0:
                brir_export.export_brir(brir_arr=brir_gen, acoustic_space=ac_space_src, hrtf_type=hrtf_type, brir_name=brir_name, primary_path=output_path, samp_freq=samp_freq_int, 
                                    bit_depth=bit_depth, brir_dir_export=brir_directional_export, brir_ts_export=brir_ts_export, hesuvi_export=hesuvi_export, 
                                    gui_logger=logz, direct_gain_db=direct_gain_db, spatial_res=spat_res_int, sofa_export=sofa_export)
            
                #set progress to 100 as export is complete (assume E-APO export time is negligible)
                progress = 100/100
                dpg.set_value("progress_bar_brir", progress)
                dpg.configure_item("progress_bar_brir", overlay = str(int(progress*100))+'%')
            else:
                #set progress to 0 as process ended early
                progress = 0
                dpg.set_value("progress_bar_brir", progress)
                dpg.configure_item("progress_bar_brir", overlay = str(int(progress*100))+'%')
            
            #update BRIR list in E-APO section
            brir_list_out_latest = e_apo_config_creation.get_exported_brir_list(output_path)
            dpg.configure_item('e_apo_load_brir_set',items=brir_list_out_latest)
            
            #Whenever a brir set is selected, determine spatial resolution, then get elevation list and update list in gui element
            brir_out_sel=dpg.get_value('e_apo_load_brir_set')
            update_elevations_list(brir_out_sel)
        
        if eapo_export == 1 and brir_gen.size != 0:
            """
            #Run E-APO Config creator for BRIR convolution
            """
            e_apo_config_creation.write_e_apo_configs_brirs(brir_name=brir_name, primary_path=output_path, hrtf_type=hrtf_type)
            
        #finally rewrite config file
        e_apo_config_acquire()
        
        #also reset thread running flag
        process_brirs_running = False
        #update user data
        dpg.configure_item('brir_tag',user_data=process_brirs_running)
        dpg.configure_item('brir_tag',label="Process")
        #set stop thread flag flag
        stop_thread_flag = False
        #update user data
        dpg.configure_item('progress_bar_brir',user_data=stop_thread_flag)
        
    #
    ## GUI Functions - Additional DEV tools
    #    
       
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
    
    
    def check_all_updates():
        """ 
        GUI function to check for all updates
        """
        hpcf_functions.check_for_app_update(gui_logger=logz)
        hpcf_functions.check_for_database_update(conn=conn, gui_logger=logz)
        air_processing.acoustic_space_updates(download_updates=False, gui_logger=logz)

    
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
        dpg.set_value("brir_hrtf", hrtf_default)
        dpg.set_value("brir_spat_res", spatial_res_default)
        dpg.configure_item('brir_hrtf',items=CN.HRTF_LIST_NUM)
        dpg.set_value("rm_target_list", room_target_default)
        dpg.set_value("direct_gain", direct_gain_default)
        dpg.set_value("direct_gain_slider", direct_gain_default)
        
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
    
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("hpcf_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        modified_dict.update({'eapo_export': int(eapo_hpcf_exp_default)})
        modified_dict.update({'fir_export': int(fir_hpcf_exp_default)})
        modified_dict.update({'fir_stereo_export': int(fir_st_hpcf_exp_default)})
        modified_dict.update({'geq_export': int(geq_hpcf_exp_default)})
        modified_dict.update({'geq_31_export': int(geq_31_exp_default)})
        modified_dict.update({'hesuvi_export': int(hesuvi_hpcf_exp_default)})
        #update user data
        dpg.configure_item('hpcf_tag',user_data=modified_dict)
        
        #reset progress bar
        reset_progress()
        
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
        dpg.set_value('selected_folder_hesuvi', primary_hesuvi_path)
        dpg.set_value('selected_folder_hesuvi_tooltip', primary_hesuvi_path)
        dpg.set_value("acoustic_space_combo", ac_space_default)
        
        reset_channel_config()
        
        save_settings()
        
    def save_settings():
        """ 
        GUI function to save settings
        """
        base_folder_selected=dpg.get_value('selected_folder_base')
        samp_freq_str = dpg.get_value('wav_sample_rate')
        bit_depth_str = dpg.get_value('wav_bit_depth')
        hp_type_str = dpg.get_value('brir_hp_type')
        hrtf_str = dpg.get_value('brir_hrtf')
        brir_spat_res_str = dpg.get_value('brir_spat_res')
        room_target_str = dpg.get_value('rm_target_list')
        direct_gain_str = str(dpg.get_value('direct_gain'))
        fir_hpcf_exp_str= str(dpg.get_value('fir_hpcf_toggle'))
        fir_st_hpcf_exp_str= str(dpg.get_value('fir_st_hpcf_toggle'))
        eapo_hpcf_exp_str= str(dpg.get_value('eapo_hpcf_toggle'))
        geq_hpcf_exp_str= str(dpg.get_value('geq_hpcf_toggle'))
        geq_31_exp_str= str(dpg.get_value('geq_31_hpcf_toggle'))
        hesuvi_hpcf_exp_str= str(dpg.get_value('hesuvi_hpcf_toggle'))
        dir_brir_exp_str= str(dpg.get_value('dir_brir_toggle'))
        ts_brir_exp_str= str(dpg.get_value('ts_brir_toggle'))
        hesuvi_brir_exp_str= str(dpg.get_value('hesuvi_brir_toggle'))
        eapo_brir_exp_str= str(dpg.get_value('eapo_brir_toggle'))
        sofa_brir_exp_str= str(dpg.get_value('sofa_brir_toggle'))
        autosize_win_str = str(dpg.get_value('resize_window_tag'))
        show_filter_sect_str = str(dpg.get_value('show_filter_sect_tag'))
        show_eapo_sect_str = str(dpg.get_value('show_eapo_sect_tag'))
        auto_check_updates_str = str(dpg.get_value('check_updates_start_tag'))
        enable_e_apo_str=str(dpg.get_value('e_apo_live_config'))
        enable_hpcf_str=str(dpg.get_value('e_apo_hpcf_conv'))
        headphone_load_str=str(dpg.get_value('e_apo_load_hp'))
        sample_load_str=str(dpg.get_value('e_apo_load_sample'))
        enable_brir_str=str(dpg.get_value('e_apo_brir_conv'))
        brir_set_str=str(dpg.get_value('e_apo_load_brir_set'))
        channel_config_str=str(dpg.get_value('audio_channels_combo'))
        
        mute_fl_str=str(dpg.get_value('e_apo_mute_fl'))
        mute_fr_str=str(dpg.get_value('e_apo_mute_fr'))
        mute_c_str=str(dpg.get_value('e_apo_mute_c'))
        mute_sl_str=str(dpg.get_value('e_apo_mute_sl'))
        mute_sr_str=str(dpg.get_value('e_apo_mute_sr'))
        mute_rl_str=str(dpg.get_value('e_apo_mute_rl'))
        mute_rr_str=str(dpg.get_value('e_apo_mute_rr'))
        gain_fl_str=str(dpg.get_value('e_apo_gain_fl'))
        gain_fr_str=str(dpg.get_value('e_apo_gain_fr'))
        gain_c_str=str(dpg.get_value('e_apo_gain_c'))
        gain_sl_str=str(dpg.get_value('e_apo_gain_sl'))
        gain_sr_str=str(dpg.get_value('e_apo_gain_sr'))
        gain_rl_str=str(dpg.get_value('e_apo_gain_rl'))
        gain_rr_str=str(dpg.get_value('e_apo_gain_rr'))
        
        elev_fl_str=str(dpg.get_value('e_apo_elev_angle_fl'))
        elev_fr_str=str(dpg.get_value('e_apo_elev_angle_fr'))
        elev_c_str=str(dpg.get_value('e_apo_elev_angle_c'))
        elev_sl_str=str(dpg.get_value('e_apo_elev_angle_sl'))
        elev_sr_str=str(dpg.get_value('e_apo_elev_angle_sr'))
        elev_rl_str=str(dpg.get_value('e_apo_elev_angle_rl'))
        elev_rr_str=str(dpg.get_value('e_apo_elev_angle_rr'))
        azim_fl_str=str(dpg.get_value('e_apo_az_angle_fl'))
        azim_fr_str=str(dpg.get_value('e_apo_az_angle_fr'))
        azim_c_str=str(dpg.get_value('e_apo_az_angle_c'))
        azim_sl_str=str(dpg.get_value('e_apo_az_angle_sl'))
        azim_sr_str=str(dpg.get_value('e_apo_az_angle_sr'))
        azim_rl_str=str(dpg.get_value('e_apo_az_angle_rl'))
        azim_rr_str=str(dpg.get_value('e_apo_az_angle_rr'))
    
        ac_space_str=dpg.get_value('acoustic_space_combo')
        
        try:
            #save folder name to config file
            config = configparser.ConfigParser()
            
            config['DEFAULT']['path'] = base_folder_selected    # update
            config['DEFAULT']['sampling_frequency'] = samp_freq_str 
            config['DEFAULT']['bit_depth'] = bit_depth_str    # update
            config['DEFAULT']['brir_headphone_type'] = hp_type_str    # update
            config['DEFAULT']['brir_hrtf'] = hrtf_str    # update
            config['DEFAULT']['spatial_resolution'] = brir_spat_res_str
            config['DEFAULT']['brir_room_target'] = room_target_str    # update
            config['DEFAULT']['brir_direct_gain'] = direct_gain_str    # update
            config['DEFAULT']['version'] = __version__    # update
            config['DEFAULT']['fir_hpcf_exp'] = fir_hpcf_exp_str
            config['DEFAULT']['fir_st_hpcf_exp'] = fir_st_hpcf_exp_str
            config['DEFAULT']['eapo_hpcf_exp'] = eapo_hpcf_exp_str
            config['DEFAULT']['geq_hpcf_exp'] = geq_hpcf_exp_str
            config['DEFAULT']['geq_31_exp'] = geq_31_exp_str
            config['DEFAULT']['hesuvi_hpcf_exp'] = hesuvi_hpcf_exp_str
            config['DEFAULT']['dir_brir_exp'] = dir_brir_exp_str
            config['DEFAULT']['ts_brir_exp'] = ts_brir_exp_str
            config['DEFAULT']['hesuvi_brir_exp'] = hesuvi_brir_exp_str 
            config['DEFAULT']['eapo_brir_exp'] = eapo_brir_exp_str
            config['DEFAULT']['sofa_brir_exp'] = sofa_brir_exp_str
            config['DEFAULT']['autosize_win'] = autosize_win_str
            config['DEFAULT']['show_filter_section'] = show_filter_sect_str
            config['DEFAULT']['show_eapo_section'] = show_eapo_sect_str
            config['DEFAULT']['auto_check_updates'] = auto_check_updates_str
            config['DEFAULT']['enable_e_apo']=enable_e_apo_str
            config['DEFAULT']['enable_hpcf'] = enable_hpcf_str
            config['DEFAULT']['headphone_selected'] = headphone_load_str
            config['DEFAULT']['sample_selected'] = sample_load_str
            config['DEFAULT']['enable_brir'] = enable_brir_str
            config['DEFAULT']['brir_set_selected'] = brir_set_str
            config['DEFAULT']['channel_config'] = channel_config_str
            config['DEFAULT']['mute_fl'] = mute_fl_str
            config['DEFAULT']['mute_fr'] = mute_fr_str
            config['DEFAULT']['mute_c'] = mute_c_str
            config['DEFAULT']['mute_sl'] = mute_sl_str
            config['DEFAULT']['mute_sr'] = mute_sr_str
            config['DEFAULT']['mute_rl'] = mute_rl_str
            config['DEFAULT']['mute_rr'] = mute_rr_str
            config['DEFAULT']['gain_fl'] = gain_fl_str
            config['DEFAULT']['gain_fr'] = gain_fr_str
            config['DEFAULT']['gain_c'] = gain_c_str
            config['DEFAULT']['gain_sl'] = gain_sl_str
            config['DEFAULT']['gain_sr'] = gain_sr_str
            config['DEFAULT']['gain_rl'] = gain_rl_str
            config['DEFAULT']['gain_rr'] = gain_rr_str
            config['DEFAULT']['elev_fl'] = elev_fl_str
            config['DEFAULT']['elev_fr'] = elev_fr_str
            config['DEFAULT']['elev_c'] = elev_c_str
            config['DEFAULT']['elev_sl'] = elev_sl_str
            config['DEFAULT']['elev_sr'] = elev_sr_str
            config['DEFAULT']['elev_rl'] = elev_rl_str
            config['DEFAULT']['elev_rr'] = elev_rr_str
            config['DEFAULT']['azim_fl'] = azim_fl_str
            config['DEFAULT']['azim_fr'] = azim_fr_str
            config['DEFAULT']['azim_c'] = azim_c_str
            config['DEFAULT']['azim_sl'] = azim_sl_str
            config['DEFAULT']['azim_sr'] = azim_sr_str
            config['DEFAULT']['azim_rl'] = azim_rl_str
            config['DEFAULT']['azim_rr'] = azim_rr_str

            config['DEFAULT']['acoustic_space'] = ac_space_str

            with open(CN.SETTINGS_FILE, 'w') as configfile:    # save
                config.write(configfile)
        except:
            log_string = 'Failed to write to settings.ini'
            logz.log_info(log_string)
        
    
    def update_sample_rate(sender, app_data, user_data):
        """ 
        GUI function to save sample rate
        """
        save_settings()

    def update_bit_depth(sender, app_data, user_data):
        """ 
        GUI function to save sample rate
        """
        save_settings()
        
    def print_summary(sender, app_data, user_data):
        """ 
        GUI function to print summary of recent HpCFs
        """
        
        hpcf_functions.get_recent_hpcfs(conn, gui_logger=logz)     
        
    def remove_brirs(sender, app_data, user_data):
        """ 
        GUI function to delete generated BRIRs
        """
        base_folder_selected=dpg.get_value('selected_folder_base')
        brir_export.remove_brirs(base_folder_selected, gui_logger=logz)     
        
        #update BRIR list in E-APO section
        brir_list_out_latest = e_apo_config_creation.get_exported_brir_list(base_folder_selected)
        dpg.configure_item('e_apo_load_brir_set',items=brir_list_out_latest)
        dpg.set_value("e_apo_load_brir_set", '')
        #update current BRIR set text
        dpg.set_value("e_apo_curr_brir_set", '')
        
        #call main config writer function
        e_apo_config_acquire()
        
    def remove_hpcfs(sender, app_data, user_data):
        """ 
        GUI function to remove generated HpCFs
        """
        base_folder_selected=dpg.get_value('selected_folder_base')
        hpcf_functions.remove_hpcfs(base_folder_selected, gui_logger=logz)     
        
        #update hp and sample list in E-APO section
        hp_list_out_latest = e_apo_config_creation.get_exported_hp_list(base_folder_selected)
        dpg.configure_item('e_apo_load_hp',items=hp_list_out_latest)
        dpg.set_value("e_apo_load_hp", '')
        
        sample_list_out_latest=[]
        dpg.configure_item('e_apo_load_sample',items=sample_list_out_latest)
        dpg.set_value("e_apo_load_sample", '')
        
        #call main config writer function
        e_apo_config_acquire()
        
    def remove_select_brirs(sender, app_data, user_data):
        """ 
        GUI function to delete specified generated BRIRs
        """
        base_folder_selected=dpg.get_value('selected_folder_base')
        brir_selected=dpg.get_value('e_apo_load_brir_set')
        if brir_selected:
            brir_export.remove_select_brirs(base_folder_selected, brir_selected, gui_logger=logz)  
            #update brir list in E-APO section
            brir_list_out_latest = e_apo_config_creation.get_exported_brir_list(base_folder_selected)
            dpg.configure_item('e_apo_load_brir_set',items=brir_list_out_latest)
            
            #Whenever a brir set is selected, determine spatial resolution, then get elevation list and update list in gui element
            brir_out_sel=dpg.get_value('e_apo_load_brir_set')
            update_elevations_list(brir_out_sel)
            #update current BRIR set text
            dpg.set_value("e_apo_curr_brir_set", brir_out_sel)
            
            #call main config writer function
            e_apo_config_acquire()
            
        dpg.configure_item("del_brir_popup", show=False)
        
    def remove_select_hpcfs(sender, app_data, user_data):
        """ 
        GUI function to remove specified generated HpCFs
        """
        base_folder_selected=dpg.get_value('selected_folder_base')
        hp_selected=dpg.get_value('e_apo_load_hp')
        if hp_selected:
            hpcf_functions.remove_select_hpcfs(base_folder_selected, hp_selected, gui_logger=logz)  
        
            #update hp and sample list in E-APO section
            hp_list_out_latest = e_apo_config_creation.get_exported_hp_list(base_folder_selected)
            dpg.configure_item('e_apo_load_hp',items=hp_list_out_latest)
            #if no headphones left in list, empty out sample list
            if not hp_list_out_latest or hp_list_out_latest == None:
                sample_list_out_latest=[]
                dpg.configure_item('e_apo_load_sample',items=sample_list_out_latest)
                dpg.set_value("e_apo_load_sample", '')
            else:
                #update sample list
                headphone =dpg.get_value('e_apo_load_hp')
                sample_list_specific = e_apo_config_creation.get_exported_sample_list(headphone=headphone,primary_path=base_folder_selected)
                sample_list_sorted = (sorted(sample_list_specific))
                dpg.configure_item('e_apo_load_sample',items=sample_list_sorted)
                
            #call main config writer function
            e_apo_config_acquire()
            
        dpg.configure_item("del_hp_popup", show=False)
        
    #
    # Equalizer APO configuration functions
    #

    def e_apo_config_acquire(sender=None, app_data=None):
        """ 
        GUI function to acquire lock on function to write updates to custom E-APO config
        """
        e_apo_conf_lock.acquire()
        e_apo_config_write()
        e_apo_conf_lock.release()

    def e_apo_config_write(sender=None, app_data=None):
        """ 
        GUI function to write updates to custom E-APO config
        """
        base_folder_selected=dpg.get_value('selected_folder_base')
        
        #hpcf related selections
        enable_hpcf_selected=dpg.get_value('e_apo_hpcf_conv')
        headphone_selected=dpg.get_value('e_apo_load_hp')
        sample_selected=dpg.get_value('e_apo_load_sample')
        
        hpcf_dict = {'enable_conv': enable_hpcf_selected, 'headphone': headphone_selected, 'sample': sample_selected}
        
        #brir related selections
        enable_brir_selected=dpg.get_value('e_apo_brir_conv')
        brir_set_selected=dpg.get_value('e_apo_load_brir_set')
        
        mute_fl_selected=dpg.get_value('e_apo_mute_fl')
        mute_fr_selected=dpg.get_value('e_apo_mute_fr')
        mute_c_selected=dpg.get_value('e_apo_mute_c')
        mute_sl_selected=dpg.get_value('e_apo_mute_sl')
        mute_sr_selected=dpg.get_value('e_apo_mute_sr')
        mute_rl_selected=dpg.get_value('e_apo_mute_rl')
        mute_rr_selected=dpg.get_value('e_apo_mute_rr')
        
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
 
        brir_dict = {'enable_conv': enable_brir_selected, 'brir_set': brir_set_selected, 'mute_fl': mute_fl_selected, 'mute_fr': mute_fr_selected, 'mute_c': mute_c_selected, 'mute_sl': mute_sl_selected,
                     'mute_sr': mute_sr_selected, 'mute_rl': mute_rl_selected, 'mute_rr': mute_rr_selected, 'gain_fl': gain_fl_selected, 'gain_fr': gain_fr_selected, 'gain_c': gain_c_selected,
                     'gain_sl': gain_sl_selected, 'gain_sr': gain_sr_selected, 'gain_rl': gain_rl_selected, 'gain_rr': gain_rr_selected, 'elev_fl': elev_fl_selected, 'elev_fr': elev_fr_selected,
                     'elev_c': elev_c_selected, 'elev_sl': elev_sl_selected, 'elev_sr': elev_sr_selected, 'elev_rl': elev_rl_selected, 'elev_rr': elev_rr_selected, 'azim_fl': azim_fl_selected,
                     'azim_fr': azim_fr_selected, 'azim_c': azim_c_selected, 'azim_sl': azim_sl_selected, 'azim_sr': azim_sr_selected, 'azim_rl': azim_rl_selected, 'azim_rr': azim_rr_selected}
  
        audio_channels=dpg.get_value('audio_channels_combo')
        
        #get spatial resolution for this brir set
        spatial_res_sel = e_apo_config_creation.get_spatial_res_from_dir(primary_path=base_folder_selected, brir_set=brir_set_selected)
        
        #run function to write custom config
        e_apo_config_creation.write_ash_e_apo_config(primary_path=base_folder_selected, hpcf_dict=hpcf_dict, brir_dict=brir_dict, audio_channels=audio_channels, gui_logger=logz, spatial_res=spatial_res_sel)
 
        #run function to load the custom config file in config.txt
        load_config = dpg.get_value('e_apo_live_config')
        #if true, edit config.txt to include the custom config
        e_apo_config_creation.include_ash_e_apo_config(primary_path=base_folder_selected, enabled=load_config)
        
        #also save settings
        save_settings()
     
    def e_apo_enable_auto_conf(sender, app_data):
        """ 
        GUI function to process updates in E-APO config section
        """
        if app_data:
            #enable e-apo live updates if not already enabled
            dpg.set_value("e_apo_live_config", True)
            
        #call main config writer function
        e_apo_config_acquire()
     
    def e_apo_select_hp(sender, app_data):
        """ 
        GUI function to process updates in E-APO config section
        """
        #update sample list
        headphone = app_data
        base_folder_selected=dpg.get_value('selected_folder_base')
        sample_list_specific = e_apo_config_creation.get_exported_sample_list(headphone=headphone,primary_path=base_folder_selected)
        sample_list_sorted = (sorted(sample_list_specific))
        dpg.configure_item('e_apo_load_sample',items=sample_list_sorted)
        
        #enable e-apo live updates if not already enabled
        dpg.set_value("e_apo_live_config", True)
        dpg.set_value("e_apo_hpcf_conv", True)
        #call main config writer function
        e_apo_config_acquire()
  
    def e_apo_select_brir(sender, app_data):
        """ 
        GUI function to process updates in E-APO config section
        """
        #enable e-apo live updates if not already enabled
        dpg.set_value("e_apo_live_config", True)
        dpg.set_value("e_apo_brir_conv", True)
        #update current BRIR set text
        dpg.set_value("e_apo_curr_brir_set", app_data)
        
        #Whenever a brir set is selected, determine spatial resolution, then get elevation list and update list in gui element
        brir_out_sel=app_data
        update_elevations_list(brir_out_sel)
        
        #call main config writer function
        e_apo_config_acquire()
        
    def update_elevations_list(brir_out_sel):
        """ 
        GUI function to update elevation list in E-APO config section
        """
        #Whenever a brir set is selected, determine spatial resolution, then get elevation list and update list in gui element
        if brir_out_sel and brir_out_sel != None:
            spatial_res_sel = e_apo_config_creation.get_spatial_res_from_dir(primary_path=primary_path, brir_set=brir_out_sel)
            elevation_list_sel = hf.get_elevation_list(spatial_res_sel)
        else:
            elevation_list_sel= CN.ELEV_ANGLES_WAV_LOW
        dpg.configure_item('e_apo_elev_angle_fl',items=elevation_list_sel)
        dpg.configure_item('e_apo_elev_angle_fr',items=elevation_list_sel)
        dpg.configure_item('e_apo_elev_angle_c',items=elevation_list_sel)
        dpg.configure_item('e_apo_elev_angle_sl',items=elevation_list_sel)
        dpg.configure_item('e_apo_elev_angle_sr',items=elevation_list_sel)
        dpg.configure_item('e_apo_elev_angle_rl',items=elevation_list_sel)
        dpg.configure_item('e_apo_elev_angle_rr',items=elevation_list_sel)
        
    def e_apo_select_sample(sender, app_data):
        """ 
        GUI function to process updates in E-APO config section
        """
        #enable e-apo live updates if not already enabled
        dpg.set_value("e_apo_live_config", True)
        dpg.set_value("e_apo_hpcf_conv", True)
        #call main config writer function
        e_apo_config_acquire()
        

    def e_apo_config_azim_fl(sender=None, app_data=None):
        """ 
        GUI function to process updates in E-APO config section
        """
        if app_data:
            azimuth=int(app_data)
            dpg.apply_transform("fl_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([90, 0]))
            dpg.apply_transform("fl_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
            
            e_apo_config_acquire()
        
    def e_apo_config_azim_fr(sender=None, app_data=None):
        """ 
        GUI function to process updates in E-APO config section
        """
        if app_data:
            azimuth=int(app_data)
            dpg.apply_transform("fr_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([90, 0]))
            dpg.apply_transform("fr_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
            
            e_apo_config_acquire()
        
    def e_apo_config_azim_c(sender=None, app_data=None):
        """ 
        GUI function to process updates in E-APO config section
        """
        if app_data:
            azimuth=int(app_data)
            dpg.apply_transform("c_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([90, 0]))
            dpg.apply_transform("c_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
            
            e_apo_config_acquire()
        
    def e_apo_config_azim_sl(sender=None, app_data=None):
        """ 
        GUI function to process updates in E-APO config section
        """
        if app_data:
            azimuth=int(app_data)
            dpg.apply_transform("sl_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([90, 0]))
            dpg.apply_transform("sl_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
            
            e_apo_config_acquire()
        
    def e_apo_config_azim_sr(sender=None, app_data=None):
        """ 
        GUI function to process updates in E-APO config section
        """
        if app_data:
            azimuth=int(app_data)
            dpg.apply_transform("sr_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([90, 0]))
            dpg.apply_transform("sr_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
            
            e_apo_config_acquire()
        
    def e_apo_config_azim_rl(sender=None, app_data=None):
        """ 
        GUI function to process updates in E-APO config section
        """
        if app_data:
            azimuth=int(app_data)
            dpg.apply_transform("rl_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([90, 0]))
            dpg.apply_transform("rl_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
            
            e_apo_config_acquire()
        
    def e_apo_config_azim_rr(sender=None, app_data=None):
        """ 
        GUI function to process updates in E-APO config section
        """
        if app_data:
            azimuth=int(app_data)
            dpg.apply_transform("rr_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([90, 0]))
            dpg.apply_transform("rr_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
            
            e_apo_config_acquire()

    def reset_channel_config(sender=None, app_data=None):
        """ 
        GUI function to reset channel config in E-APO config section
        """
        dpg.set_value("e_apo_mute_fl", e_apo_mute_default)
        dpg.set_value("e_apo_mute_fr", e_apo_mute_default)
        dpg.set_value("e_apo_mute_c", e_apo_mute_default)
        dpg.set_value("e_apo_mute_sl", e_apo_mute_default)
        dpg.set_value("e_apo_mute_sr", e_apo_mute_default)
        dpg.set_value("e_apo_mute_rl", e_apo_mute_default)
        dpg.set_value("e_apo_mute_rr", e_apo_mute_default)
        
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
        
        e_apo_config_azim_fl(app_data=e_apo_az_angle_fl_default)
        e_apo_config_azim_fr(app_data=e_apo_az_angle_fr_default)
        e_apo_config_azim_c(app_data=e_apo_az_angle_c_default)
        e_apo_config_azim_sl(app_data=e_apo_az_angle_sl_default)
        e_apo_config_azim_sr(app_data=e_apo_az_angle_sr_default)
        e_apo_config_azim_rl(app_data=e_apo_az_angle_rl_default)
        e_apo_config_azim_rr(app_data=e_apo_az_angle_rr_default)
 
        #finally rewrite config file
        e_apo_config_acquire()
  
    def e_apo_select_channels(sender=None, app_data=None):
        """ 
        GUI function to process updates in E-APO config section
        """
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
        
        elif app_data == '7.1 Surround':
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


        #finally rewrite config file
        e_apo_config_acquire()

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
               
    def reset_progress():
        """ 
        GUI function to reset progress bar
        """
        #if not already running
        #thread bool
        process_brirs_running=dpg.get_item_user_data("brir_tag")
        if process_brirs_running == False:
            #reset progress bar
            dpg.set_value("progress_bar_brir", 0)
            dpg.configure_item("progress_bar_brir", overlay = 'Progress')

    def update_progress(sender, app_data):
        """ 
        GUI function to update progress bar
        """
        #print(str(app_data))
        value = dpg.get_value("Drag int")
        dpg.set_value("progress_bar_brir", value/100)
        dpg.configure_item("progress_bar_brir", overlay = str(value)+'%')

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
                dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(i/7.0, 0.40, 0.6))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(i/7.0, 0.8, 0.8))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(i/7.0, 0.6, 0.7))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, i*5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 8)
        with dpg.theme(tag="__theme_b"):
            i=3.8#i=2
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(i/7.0, 0.40, 0.6))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(i/7.0, 0.8, 0.8))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(i/7.0, 0.6, 0.7))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, i*5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 8)
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
        with dpg.theme() as global_theme:
            i=3.9
            j=3.9
            k=3.8
            l=3.5
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, _hsv_to_rgb(i/7.0, 0.5, 0.5))   
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, _hsv_to_rgb(i/7.0, 0.5, 0.5)) 
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, _hsv_to_rgb(i/7.0, 0.5, 0.8)) 
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, _hsv_to_rgb(i/7.0, 0.5, 0.7)) 
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, _hsv_to_rgb(i/7.0, 0.5, 0.7))
                dpg.add_theme_color(dpg.mvPlotCol_TitleText, _hsv_to_rgb(i/7.0, 0.5, 0.7))
                dpg.add_theme_color(dpg.mvThemeCol_Header, _hsv_to_rgb(4.4/7.0, 0.1, 0.25))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, _hsv_to_rgb(j/7.0, 0.3, 0.5)) 
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, _hsv_to_rgb(j/7.0, 0.3, 0.5)) 
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, _hsv_to_rgb(j/7.0, 0.3, 0.5)) 
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(j/7.0, 0.3, 0.5)) 
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, _hsv_to_rgb(l/7.0, 0.2, 0.5)) 
                dpg.add_theme_color(dpg.mvPlotCol_Line, _hsv_to_rgb(k/7.0, 0.25, 0.6), category=dpg.mvThemeCat_Plots) 
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, _hsv_to_rgb(j/7.0, 0.7, 0.7))
        
        dpg.bind_theme(global_theme)
        
        with dpg.collapsing_header(label="Filter Creation",default_open = show_filter_sect_loaded):
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
                                listbox_1 = dpg.add_listbox(brands_list, width=135, num_items=16, tag='brand_list', callback=update_headphone_list)
                            with dpg.group():
                                dpg.add_text("Headphone")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                listbox_2 = dpg.add_listbox(hp_list_default, width=250, num_items=16, tag='headphone_list', default_value=headphone_default ,callback=update_sample_list)
                            with dpg.group():
                                dpg.add_text("Sample")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                listbox_3 = dpg.add_listbox(sample_list_default, width=115, num_items=16, default_value=sample_default, tag='sample_list', user_data=headphone_default, callback=plot_sample)
                                with dpg.tooltip("sample_list"):
                                    dpg.add_text("Note: all samples will be exported. Select a sample to preview")
                    with dpg.child_window(autosize_x=True, height=88):
                        subtitle_2 = dpg.add_text("Select Files to Include in Export")
                        dpg.bind_item_font(subtitle_2, bold_font)
                        dpg.add_separator()
                        with dpg.group(horizontal=True):
                            dpg.add_checkbox(label="WAV FIR Filters", default_value = fir_hpcf_exp_loaded, callback=export_fir_toggle, tag='fir_hpcf_toggle')
                            with dpg.tooltip("fir_hpcf_toggle"):
                                dpg.add_text("Min phase FIRs in WAV format for convolution. 1 Channel")
                                dpg.add_text("Required for Equalizer APO configuration updates")
                            dpg.add_checkbox(label="WAV Stereo FIR Filters", default_value = fir_st_hpcf_exp_loaded, callback=export_fir_stereo_toggle, tag='fir_st_hpcf_toggle')
                            with dpg.tooltip("fir_st_hpcf_toggle"):
                                dpg.add_text("Min phase FIRs in WAV format for convolution. 2 Channels")
                            dpg.add_checkbox(label="E-APO Configuration Files", default_value = eapo_hpcf_exp_loaded, callback=export_eapo_hpcf_toggle, tag='eapo_hpcf_toggle')  
                            with dpg.tooltip("eapo_hpcf_toggle"):
                                dpg.add_text("Equalizer APO configurations to perform convolution with FIR filters. Deprecated from V2.0.0 onwards")
                        with dpg.group(horizontal=True):
                            dpg.add_checkbox(label="Graphic EQ Filters (127 Bands)", default_value = geq_hpcf_exp_loaded, callback=export_geq_toggle, tag='geq_hpcf_toggle')
                            with dpg.tooltip("geq_hpcf_toggle"):
                                dpg.add_text("Graphic EQ configurations with 127 bands. Compatible with Equalizer APO and Wavelet")
                            dpg.add_checkbox(label="Graphic EQ Filters (31 Bands)", default_value = geq_31_exp_loaded, callback=export_geq_31_toggle, tag='geq_31_hpcf_toggle')
                            with dpg.tooltip("geq_31_hpcf_toggle"):
                                dpg.add_text("Graphic EQ configurations with 31 bands. Compatible with 31 band graphic equalizers including Equalizer APO")
                            dpg.add_checkbox(label="HeSuVi Filters", default_value = hesuvi_hpcf_exp_loaded, callback=export_hesuvi_hpcf_toggle, tag='hesuvi_hpcf_toggle')
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
                            dpg.configure_item('hpcf_tag',user_data=default_hpcf_settings)
                            with dpg.tooltip("hpcf_tag"):
                                dpg.add_text("This will export the selected filters to the output directory")
                                
                            dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=32, width=370, overlay="Progress", tag="progress_bar_hpcf")

                #Section for BRIR generation
                with dpg.child_window(width=520, height=610):
                    title_2 = dpg.add_text("Binaural Room Simulation", tag='brir_title')
                    dpg.bind_item_font(title_2, bold_font)
                    
                    with dpg.child_window(autosize_x=True, height=388):
                        subtitle_4 = dpg.add_text("Set Parameters", tag='brir_param_title')
                        with dpg.tooltip("brir_param_title"):
                            dpg.add_text("Customise a new binaural room simulation using below parameters")
                        dpg.bind_item_font(subtitle_4, bold_font)
                        dpg.add_separator()
                        with dpg.group(horizontal=True):
                            with dpg.group():
                                dpg.add_text("Acoustic Space",tag='acoustic_space_title')
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_listbox(CN.AC_SPACE_LIST_GUI, label="",tag='acoustic_space_combo',default_value=ac_space_loaded, callback=update_brir_param, num_items=6, width=250)
                                with dpg.tooltip("acoustic_space_title"):
                                    dpg.add_text("This will determine the listening environment")
                                dpg.add_text("Spatial Resolution", tag= "brir_spat_res_title")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_radio_button(CN.SPATIAL_RES_LIST, horizontal=True, tag= "brir_spat_res", default_value=spatial_res_loaded, callback=select_spatial_resolution )
                                with dpg.tooltip("brir_spat_res_title"):
                                    dpg.add_text("Increasing resolution will increase number of directions available but will also increase processing time and dataset size")
                                    dpg.add_text("'Low' is recommended unless additional directions or SOFA export is required")
                                dpg.add_text("Dummy Head / Head & Torso Simulator", tag='brir_hrtf_title')
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                listbox_4 = dpg.add_listbox(hrtf_list_loaded, default_value=hrtf_loaded, num_items=6, width=250, callback=select_hrtf, tag='brir_hrtf')
                                with dpg.tooltip("brir_hrtf_title"):
                                    dpg.add_text("This will influence the externalisation and localisation of sounds around the listener")
                            with dpg.group():
                                dpg.add_text("Direct Sound Gain (dB)", tag='direct_gain_title')
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_input_float(label="Direct Gain (dB)",width=140, format="%.1f", tag='direct_gain', min_value=CN.DIRECT_GAIN_MIN, max_value=CN.DIRECT_GAIN_MAX, default_value=direct_gain_loaded,min_clamped=True, max_clamped=True, callback=update_direct_gain)
                                with dpg.tooltip("direct_gain_title"):
                                    dpg.add_text("This will control the loudness of the direct signal")
                                    dpg.add_text("Higher values result in lower perceived distance, lower values result in higher perceived distance")
                                dpg.add_slider_float(label="", min_value=CN.DIRECT_GAIN_MIN, max_value=CN.DIRECT_GAIN_MAX, default_value=direct_gain_loaded, width=140,clamped=True, no_input=True, format="", callback=update_direct_gain_slider, tag='direct_gain_slider')
                                
                                dpg.add_text("Headphone Compensation", tag='brir_hp_type_title')
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                listbox_5 = dpg.add_listbox(CN.HP_COMP_LIST, default_value=brir_hp_type_loaded, num_items=4, width=230, callback=select_hp_comp, tag='brir_hp_type')
                                with dpg.tooltip("brir_hp_type_title"):
                                    dpg.add_text("This should align with the listener's headphone type")
                                    dpg.add_text("Reduce to low strength if sound localisation or timbre is compromised")
                                
                                dpg.add_text("Room Target", tag='rm_target_title')
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                listbox_6 = dpg.add_listbox(CN.ROOM_TARGET_LIST, default_value=room_target_loaded, num_items=6, width=230, tag='rm_target_list', callback=select_room_target)
                                with dpg.tooltip("rm_target_title"):
                                    dpg.add_text("This will influence the overall balance of low and high frequencies")
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
                            dpg.bind_item_theme(dpg.last_item(), "__theme_a")
                            dpg.bind_item_font(dpg.last_item(), large_font)
                            with dpg.tooltip("brir_tag"):
                                dpg.add_text("This will generate the binaural dataset and export to the output directory. This may take some time to process")
                                
                            dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=32, width=340, overlay="Progress", tag="progress_bar_brir",user_data=CN.STOP_THREAD_FLAG)
            
                #right most section
                with dpg.group():    
                    #Section for plotting
                    with dpg.child_window(width=604, height=467):
                        with dpg.tab_bar():
                            with dpg.tab(label="Filter Preview"):
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
                                    #hpcf_functions.hpcf_to_plot(conn, headphone_default, sample_default, plot_type=1)
                                    hf.plot_data(fr_flat_mag, title_name='', n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=2, plot_type=1)
                            with dpg.tab(label="Supporting Information"):
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
                           
                    #Section for Exporting files
                    with dpg.group(horizontal=True):
                        #Section for wav settings
                        with dpg.child_window(width=225, height=139):
                            title_4 = dpg.add_text("WAV Settings", tag='export_title')
                            dpg.bind_item_font(title_4, bold_font)
                            dpg.add_separator()
                            with dpg.group():
                                dpg.add_text("Select Sample Rate")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_radio_button(CN.SAMPLE_RATE_LIST, horizontal=True, tag= "wav_sample_rate", default_value=sample_freq_loaded, callback=update_sample_rate )
                            #dpg.add_text("          ")
                            with dpg.group():
                                dpg.add_text("Select Bit Depth")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_radio_button(CN.BIT_DEPTH_LIST, horizontal=True, tag= "wav_bit_depth", default_value=bit_depth_loaded, callback=update_bit_depth)
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
                                dpg.add_text("                                         ")
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
                        
  
        with dpg.collapsing_header(label="Equalizer APO Configuration",default_open = show_eapo_sect_loaded):
            #Section for managing E-APO configurations
            with dpg.child_window(width=1690, height=325):
                with dpg.group(horizontal=True):
                    dpg.bind_item_font(dpg.last_item(), bold_font)
                    dpg.add_checkbox(label="Auto-Configure 'config.txt'", default_value = e_apo_enable_loaded,  tag='e_apo_live_config', callback=e_apo_config_acquire)
                    dpg.bind_item_theme(dpg.last_item(), "__theme_e")
                    with dpg.tooltip("e_apo_live_config"):
                        dpg.add_text("Auto-configure 'config.txt' to apply selected filters")
                    dpg.add_text("  ")
                    dpg.add_checkbox(label="Enable Headphone Correction", default_value = e_apo_enable_hpcf_loaded,  tag='e_apo_hpcf_conv', callback=e_apo_enable_auto_conf)
                    dpg.bind_item_theme(dpg.last_item(), "__theme_e")
                    with dpg.tooltip("e_apo_hpcf_conv"):
                        dpg.add_text("Enable convolution with selected headphone correction WAV filter")
                    dpg.add_text("  ")
                    dpg.add_checkbox(label="Enable Binaural Room Simulation", default_value = e_apo_enable_brir_loaded,  tag='e_apo_brir_conv', callback=e_apo_enable_auto_conf)
                    dpg.bind_item_theme(dpg.last_item(), "__theme_e")
                    with dpg.tooltip("e_apo_brir_conv"):
                        dpg.add_text("Enable convolution with selected BRIR WAV files")

                with dpg.group(horizontal=True):
                    with dpg.child_window(width=CN.HP_SEL_WIN_WIDTH_FULL, tag='e_apo_load_hp_win'):
                        with dpg.group(horizontal=True):
                            with dpg.group():
                                with dpg.group(horizontal=True):
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_text("Select Headphone & Sample", tag='e_apo_hp_title')
                                    with dpg.tooltip("e_apo_hp_title"):
                                        dpg.add_text("Exported FIR filters will be shown below")
                                    dpg.add_text("          ")
                                    dpg.add_button(label="  Delete Selected Headphone  ")
                                    with dpg.popup(dpg.last_item(), modal=True, mousebutton=dpg.mvMouseButton_Left, tag="del_hp_popup"):
                                        dpg.add_text("Saved filters for selected headphone will be deleted.")
                                        dpg.add_separator()
                                        with dpg.group(horizontal=True):
                                            dpg.add_button(label="OK", width=75, callback=remove_select_hpcfs)
                                            dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("del_hp_popup", show=False))
                                dpg.add_separator()
                                
                                with dpg.group(horizontal=True):
                                    listbox_7 = dpg.add_listbox(hp_list_out_default,default_value=headphone_out_default, num_items=12, width=CN.HP_SEL_LIST_WIDTH_FULL, tag='e_apo_load_hp', callback=e_apo_select_hp)
                                    listbox_8 = dpg.add_listbox(sample_list_out_default,default_value=sample_out_default, num_items=12, width=120, tag='e_apo_load_sample', callback=e_apo_select_sample)
                    with dpg.child_window(width=CN.BRIR_SEL_WIN_WIDTH_FULL, tag='e_apo_load_brir_win'):
                        with dpg.group(horizontal=True):
                            dpg.bind_item_font(dpg.last_item(), bold_font)
                            dpg.add_text("Select Binaural Simulation", tag='e_apo_brir_title')
                            with dpg.tooltip("e_apo_brir_title"):
                                dpg.add_text("Exported BRIR WAV datasets will be shown below")
                            dpg.add_text("                                            ")
                            dpg.add_button(label="  Delete Selected Dataset  ")
                            with dpg.popup(dpg.last_item(), modal=True, mousebutton=dpg.mvMouseButton_Left, tag="del_brir_popup"):
                                dpg.add_text("Selected dataset will be deleted.")
                                dpg.add_separator()
                                with dpg.group(horizontal=True):
                                    dpg.add_button(label="OK", width=75, callback=remove_select_brirs)
                                    dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("del_brir_popup", show=False))
                        dpg.add_separator()
                        
                        listbox_9 = dpg.add_listbox(brir_list_out_default, num_items=12, default_value=brir_out_default, width=CN.BRIR_SEL_LIST_WIDTH_FULL, tag='e_apo_load_brir_set', callback=e_apo_select_brir)
                    
                    with dpg.child_window(width=670):
                        with dpg.group(horizontal=True):
                            with dpg.group():
                                with dpg.group(horizontal=True):
                                    dpg.bind_item_font(dpg.last_item(), bold_font)
                                    dpg.add_text("Channel Configuration                  ")
                                    dpg.add_button(label="Reset All", width=54, callback=reset_channel_config)
                                    dpg.add_text("                    Audio Channels: ")
                                    dpg.add_combo(CN.AUDIO_CHANNELS, width=200, label="",  tag='audio_channels_combo',default_value=audio_channels_loaded, callback=e_apo_select_channels)
                                dpg.add_separator()
                                
                                with dpg.group(horizontal=True):
                                    dpg.add_text("Current Simulation: ")
                                    dpg.add_text(default_value=brir_out_default,  tag='e_apo_curr_brir_set')
                                    dpg.bind_item_theme(dpg.last_item(), "__theme_f")
                                with dpg.group(horizontal=True):
                                    with dpg.group():
                                        #dpg.add_text(" ")
                                        with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                            borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                            borders_outerV=True, borders_innerV=True, delay_search=True):
                                            dpg.bind_item_font(dpg.last_item(), bold_font)
                                            dpg.add_table_column(label="Channel")
                                            dpg.add_table_column(label="Mute")
                                            dpg.add_table_column(label="Gain (dB)")
                                            dpg.add_table_column(label="Elevation Angle (°)")
                                            dpg.add_table_column(label="Azimuth Angle (°)")
                                            tooltip_gain = 'Positive values may result in clipping'
                                            tooltip_elevation = 'Positive values are above the listener while negative values are below the listener'
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
                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_fl',default_value=e_apo_mute_fl_loaded, callback=e_apo_config_acquire)
                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                            elif i == 1:
                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_fr',default_value=e_apo_mute_fr_loaded, callback=e_apo_config_acquire)
                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                            elif i == 2:
                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_c',default_value=e_apo_mute_c_loaded, callback=e_apo_config_acquire)
                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                            elif i == 3:
                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_sl',default_value=e_apo_mute_sl_loaded, callback=e_apo_config_acquire)
                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                            elif i == 4:
                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_sr',default_value=e_apo_mute_sr_loaded, callback=e_apo_config_acquire)
                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                            elif i == 5:
                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_rl',default_value=e_apo_mute_rl_loaded, callback=e_apo_config_acquire)
                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                            elif i == 6:
                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_rr',default_value=e_apo_mute_rr_loaded, callback=e_apo_config_acquire)
                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                        if j == 2:#gain
                                                            if i == 0:
                                                                dpg.add_input_int(label=" ", width=75,min_value=-100, max_value=20, tag='e_apo_gain_fl',default_value=e_apo_gain_fl_loaded, callback=e_apo_config_acquire)
                                                                with dpg.tooltip("e_apo_gain_fl"):
                                                                    dpg.add_text(tooltip_gain)
                                                            elif i == 1:
                                                                dpg.add_input_int(label=" ", width=75,min_value=-100, max_value=20, tag='e_apo_gain_fr',default_value=e_apo_gain_fr_loaded, callback=e_apo_config_acquire)
                                                                with dpg.tooltip("e_apo_gain_fr"):
                                                                    dpg.add_text(tooltip_gain)
                                                            elif i == 2:
                                                                dpg.add_input_int(label=" ", width=75,min_value=-100, max_value=20, tag='e_apo_gain_c',default_value=e_apo_gain_c_loaded, callback=e_apo_config_acquire)
                                                                with dpg.tooltip("e_apo_gain_c"):
                                                                    dpg.add_text(tooltip_gain)
                                                            elif i == 3:
                                                                dpg.add_input_int(label=" ", width=75,min_value=-100, max_value=20, tag='e_apo_gain_sl',default_value=e_apo_gain_sl_loaded, callback=e_apo_config_acquire)
                                                                with dpg.tooltip("e_apo_gain_sl"):
                                                                    dpg.add_text(tooltip_gain)
                                                            elif i == 4:
                                                                dpg.add_input_int(label=" ", width=75,min_value=-100, max_value=20, tag='e_apo_gain_sr',default_value=e_apo_gain_sr_loaded, callback=e_apo_config_acquire)
                                                                with dpg.tooltip("e_apo_gain_sr"):
                                                                    dpg.add_text(tooltip_gain)
                                                            elif i == 5:
                                                                dpg.add_input_int(label=" ", width=75,min_value=-100, max_value=20, tag='e_apo_gain_rl',default_value=e_apo_gain_rl_loaded, callback=e_apo_config_acquire)
                                                                with dpg.tooltip("e_apo_gain_rl"):
                                                                    dpg.add_text(tooltip_gain)
                                                            elif i == 6:
                                                                dpg.add_input_int(label=" ", width=75,min_value=-100, max_value=20, tag='e_apo_gain_rr',default_value=e_apo_gain_rr_loaded, callback=e_apo_config_acquire)
                                                                with dpg.tooltip("e_apo_gain_rr"):
                                                                    dpg.add_text(tooltip_gain)
                                                        if j == 3:#elevation
                                                            if i == 0:
                                                                dpg.add_combo(elevation_list_sel, width=100, label="",  tag='e_apo_elev_angle_fl',default_value=e_apo_elev_angle_fl_loaded, callback=e_apo_config_acquire)
                                                                with dpg.tooltip("e_apo_elev_angle_fl"):
                                                                    dpg.add_text(tooltip_elevation)
                                                            elif i == 1:
                                                                dpg.add_combo(elevation_list_sel, width=100, label="",  tag='e_apo_elev_angle_fr',default_value=e_apo_elev_angle_fr_loaded, callback=e_apo_config_acquire)
                                                                with dpg.tooltip("e_apo_elev_angle_fr"):
                                                                    dpg.add_text(tooltip_elevation)
                                                            elif i == 2:
                                                                dpg.add_combo(elevation_list_sel, width=100, label="",  tag='e_apo_elev_angle_c',default_value=e_apo_elev_angle_c_loaded, callback=e_apo_config_acquire)
                                                                with dpg.tooltip("e_apo_elev_angle_c"):
                                                                    dpg.add_text(tooltip_elevation)
                                                            elif i == 3:
                                                                dpg.add_combo(elevation_list_sel, width=100, label="",  tag='e_apo_elev_angle_sl',default_value=e_apo_elev_angle_sl_loaded, callback=e_apo_config_acquire)
                                                                with dpg.tooltip("e_apo_elev_angle_sl"):
                                                                    dpg.add_text(tooltip_elevation)
                                                            elif i == 4:
                                                                dpg.add_combo(elevation_list_sel, width=100, label="",  tag='e_apo_elev_angle_sr',default_value=e_apo_elev_angle_sr_loaded, callback=e_apo_config_acquire)
                                                                with dpg.tooltip("e_apo_elev_angle_sr"):
                                                                    dpg.add_text(tooltip_elevation)
                                                            elif i == 5:
                                                                dpg.add_combo(elevation_list_sel, width=100, label="",  tag='e_apo_elev_angle_rl',default_value=e_apo_elev_angle_rl_loaded, callback=e_apo_config_acquire)
                                                                with dpg.tooltip("e_apo_elev_angle_rl"):
                                                                    dpg.add_text(tooltip_elevation)
                                                            elif i == 6:
                                                                dpg.add_combo(elevation_list_sel, width=100, label="",  tag='e_apo_elev_angle_rr',default_value=e_apo_elev_angle_rr_loaded, callback=e_apo_config_acquire)
                                                                with dpg.tooltip("e_apo_elev_angle_rr"):
                                                                    dpg.add_text(tooltip_elevation)
                                                        if j == 4:#azimuth
                                                            if i == 0:
                                                                dpg.add_combo(CN.AZ_ANGLES_FL_WAV, width=100, label="",  tag='e_apo_az_angle_fl',default_value=e_apo_az_angle_fl_loaded, callback=e_apo_config_azim_fl)
                                                            elif i == 1:
                                                                dpg.add_combo(CN.AZ_ANGLES_FR_WAV, width=100, label="",  tag='e_apo_az_angle_fr',default_value=e_apo_az_angle_fr_loaded, callback=e_apo_config_azim_fr)
                                                            elif i == 2:
                                                                dpg.add_combo(CN.AZ_ANGLES_C_WAV, width=100, label="",  tag='e_apo_az_angle_c',default_value=e_apo_az_angle_c_loaded, callback=e_apo_config_azim_c)
                                                            elif i == 3:
                                                                dpg.add_combo(CN.AZ_ANGLES_SL_WAV, width=100, label="",  tag='e_apo_az_angle_sl',default_value=e_apo_az_angle_sl_loaded, callback=e_apo_config_azim_sl)
                                                            elif i == 4:
                                                                dpg.add_combo(CN.AZ_ANGLES_SR_WAV, width=100, label="",  tag='e_apo_az_angle_sr',default_value=e_apo_az_angle_sr_loaded, callback=e_apo_config_azim_sr)
                                                            elif i == 5:
                                                                dpg.add_combo(CN.AZ_ANGLES_RL_WAV, width=100, label="",  tag='e_apo_az_angle_rl',default_value=e_apo_az_angle_rl_loaded, callback=e_apo_config_azim_rl)
                                                            elif i == 6:
                                                                dpg.add_combo(CN.AZ_ANGLES_RR_WAV, width=100, label="",  tag='e_apo_az_angle_rr',default_value=e_apo_az_angle_rr_loaded, callback=e_apo_config_azim_rr)
                       
                                    with dpg.drawlist(width=250, height=210, tag="channel_drawing"):
            
                                        with dpg.draw_layer():
                                            
                                            radius=90
                                            x_start=110
                                            y_start=105#115
                                            dpg.draw_circle([x_start, y_start], radius, color=[163, 177, 184])           
                                            with dpg.draw_node(tag="listener_drawing"):
                                                dpg.apply_transform(dpg.last_item(), dpg.create_translation_matrix([x_start, y_start]))
                                                dpg.draw_circle([0, 0], 20, color=[163, 177, 184], fill=[158,158,158])
                                                dpg.draw_text([-19, -8], 'Listener', color=[0, 0, 0],size=13)
                                                
                                                with dpg.draw_node(tag="fl_drawing"):
                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(e_apo_az_angle_fl_loaded*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([radius, 0]))
                                                    with dpg.draw_node(tag="fl_drawing_inner", user_data=45.0):
                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(e_apo_az_angle_fl_loaded*-1))/180.0 , [0, 0, -1]))
                                                        dpg.draw_image('fl_image',[-12, -12],[12, 12])
                                                        
                                                with dpg.draw_node(tag="fr_drawing"):
                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(e_apo_az_angle_fr_loaded*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([radius, 0]))
                                                    with dpg.draw_node(tag="fr_drawing_inner", user_data=45.0):
                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(e_apo_az_angle_fr_loaded*-1))/180.0 , [0, 0, -1]))
                                                        dpg.draw_image('fr_image',[-12, -12],[12, 12])
                                                
                                                with dpg.draw_node(tag="c_drawing"):
                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(e_apo_az_angle_c_loaded*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([radius, 0]))
                                                    with dpg.draw_node(tag="c_drawing_inner", user_data=45.0):
                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(e_apo_az_angle_c_loaded*-1))/180.0 , [0, 0, -1]))
                                                        dpg.draw_image('c_image',[-12, -12],[12, 12])
                                                        
                                                with dpg.draw_node(tag="sl_drawing"):
                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(e_apo_az_angle_sl_loaded*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([radius, 0]))
                                                    with dpg.draw_node(tag="sl_drawing_inner", user_data=45.0):
                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(e_apo_az_angle_sl_loaded*-1))/180.0 , [0, 0, -1]))
                                                        dpg.draw_image('sl_image',[-12, -12],[12, 12])
                                                        
                                                with dpg.draw_node(tag="sr_drawing"):
                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(e_apo_az_angle_sr_loaded*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([radius, 0]))
                                                    with dpg.draw_node(tag="sr_drawing_inner", user_data=45.0):
                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(e_apo_az_angle_sr_loaded*-1))/180.0 , [0, 0, -1]))
                                                        dpg.draw_image('sr_image',[-12, -12],[12, 12])
                                                        
                                                with dpg.draw_node(tag="rl_drawing"):
                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(e_apo_az_angle_rl_loaded*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([radius, 0]))
                                                    with dpg.draw_node(tag="rl_drawing_inner", user_data=45.0):
                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(e_apo_az_angle_rl_loaded*-1))/180.0 , [0, 0, -1]))
                                                        dpg.draw_image('rl_image',[-12, -12],[12, 12])
                                                        
                                                with dpg.draw_node(tag="rr_drawing"):
                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(e_apo_az_angle_rr_loaded*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([radius, 0]))
                                                    with dpg.draw_node(tag="rr_drawing_inner", user_data=45.0):
                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(e_apo_az_angle_rr_loaded*-1))/180.0 , [0, 0, -1]))
                                                        dpg.draw_image('rr_image',[-12, -12],[12, 12])
   
        with dpg.collapsing_header(label="Additional Tools & Settings",default_open = True):
            with dpg.group(horizontal=True):
                with dpg.group():  
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            
                            #Section for database
                            with dpg.child_window(width=224, height=318):
                                dpg.add_text("Check for Updates on Start")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_checkbox(label="Auto Check for Updates", default_value = auto_check_updates_loaded,  tag='check_updates_start_tag', callback=save_settings)
                                dpg.add_separator()
                                dpg.add_text("Check for App Updates")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Check for Updates",user_data="",tag="app_version_tag", callback=check_app_version)
                                with dpg.tooltip("app_version_tag"):
                                    dpg.add_text("This will check for updates to the app and show versions in the log")   
                            #with dpg.child_window(width=224, height=110):
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
                            #with dpg.child_window(width=224, height=135):
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
                        with dpg.group():
                            #Section to reset settngs
                            with dpg.child_window(width=250, height=65):
                                dpg.add_text("Reset Settings to Default")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Reset Settings",user_data="",tag="reset_settings_tag", callback=reset_settings)  
                            with dpg.child_window(width=250, height=110):
                                dpg.add_text("Delete All Exported Headphone Filters")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Delete Headphone Filters",user_data="",tag="remove_hpcfs_tag", callback=remove_hpcfs)
                                with dpg.tooltip("remove_hpcfs_tag"):
                                    dpg.add_text("Warning: this will delete all headphone filters that have been exported to output directory")
                                dpg.add_text("Delete All Exported Binaural Datasets")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_button(label="Delete Binaural Datasets",user_data="",tag="remove_brirs_tag", callback=remove_brirs)
                                with dpg.tooltip("remove_brirs_tag"):
                                    dpg.add_text("Warning: this will delete all BRIRs that have been generated and exported to output directory")  
                            #Section to reduce size of window
                            with dpg.child_window(width=250, height=135):
                                dpg.add_text("Auto-size App Window on Start")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_checkbox(label="Auto-size Window", default_value = autosize_win_loaded,  tag='resize_window_tag', callback=save_settings)
                                with dpg.tooltip("resize_window_tag"):
                                    dpg.add_text("Requires restart")
                                dpg.add_text("Show/Hide Sections on Start")
                                dpg.bind_item_font(dpg.last_item(), bold_font)
                                dpg.add_checkbox(label="Show Filter Creation", default_value = show_filter_sect_loaded,  tag='show_filter_sect_tag', callback=save_settings)
                                with dpg.tooltip("show_filter_sect_tag"):
                                    dpg.add_text("Requires restart")
                                dpg.add_checkbox(label="Show E-APO Configuration", default_value = show_eapo_sect_loaded,  tag='show_eapo_sect_tag', callback=save_settings)
                                with dpg.tooltip("show_eapo_sect_tag"):
                                    dpg.add_text("Requires restart")
                        
                #section for logging
                with dpg.child_window(width=1200, height=318, tag="console_window"):
                    dpg.add_text("Log")
                    
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

    


    dpg.setup_dearpygui()
    logz=logger.mvLogger(parent="console_window")

    #section to log tool version on startup
    #log results
    log_string = 'Audio Spatialisation for Headphones Toolset. Version: ' + __version__
    logz.log_info(log_string)
    
    #check for updates on start
    if auto_check_updates_loaded == True:
        #start thread
        thread = threading.Thread(target=check_all_updates, args=(), daemon=True)
        thread.start()

    dpg.show_viewport()
    dpg.set_primary_window("Primary Window", True)
    
    dpg.configure_item("Primary Window", horizontal_scrollbar=True)
    e_apo_select_channels(app_data=dpg.get_value('audio_channels_combo'))#update channel gui elements on load

    
    dpg.start_dearpygui()
    

    dpg.destroy_context()
        
    #finally close the connection
    conn.close()

    logging.info('Finished') 

if __name__ == '__main__':
    main()