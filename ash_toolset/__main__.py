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
    import mat73
    import dearpygui.dearpygui as dpg
    import dearpygui_extend as dpge
    from dearpygui_ext import logger
    import numpy as np
    import csv
    import json
    import winreg as wrg
    
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
    
    
    
    
    #
    #program code
    #
    
    #get equalizer APO path
    try:
        key = wrg.OpenKey(wrg.HKEY_LOCAL_MACHINE, "Software\\EqualizerAPO")
        value =  wrg.QueryValueEx(key, "InstallPath")[0]
        e_apo_path=value
    except:
        e_apo_path = None
    #try reading from settings.ini to get path
    try:
        config = configparser.ConfigParser()
        config.read(CN.SETTINGS_FILE)
        base_folder_loaded = config['DEFAULT']['path']
        primary_path=base_folder_loaded
        primary_ash_path=pjoin(base_folder_loaded, CN.PROJECT_FOLDER)
    except:
        if e_apo_path is not None:
            primary_path = e_apo_path
            primary_ash_path = pjoin(e_apo_path, CN.PROJECT_FOLDER)
        else:
            primary_path = 'C:\Program Files\EqualizerAPO'
            primary_ash_path = 'C:\Program Files\EqualizerAPO\ASH-Custom-Set'
    
    #
    # populate room target dictionary for plotting
    #
    # load room target filters (FIR)
    mat_fname = pjoin(CN.DATA_DIR_INT, 'room_targets_firs.mat')
    room_target_mat = mat73.loadmat(mat_fname)
    #start with flat response
    impulse=np.zeros(CN.N_FFT)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    #create dictionary
    target_mag_dict = {'Flat': fr_flat_mag} 
    
    for idx, target in enumerate(CN.ROOM_TARGET_LIST_FIRS):
        if idx > 0:
            room_target_fir=np.zeros(CN.N_FFT)
            room_target_fir[0:4096] = room_target_mat[target]
            data_fft = np.fft.fft(room_target_fir)
            room_target_mag=np.abs(data_fft)
            
            room_target_name=CN.ROOM_TARGET_LIST[idx]
    
            target_mag_dict.update({room_target_name: room_target_mag})
    
    
    
    
    
    
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
    
    
    default_hpcf_settings = {'headphone': headphone_default, 'hpcf_export': 1, 'fir_export': 1, 'fir_stereo_export': 1, 'geq_export': 1, 'geq_31_export': 1, 'geq_32_export': 0, 'hesuvi_export': 1, 'eapo_export': 1}
    
    hrtf_default = 1
    direct_gain_default = 4.0
    reverb_gain_default = direct_gain_default*-1
    room_target_default = 1
    pinna_comp_default = 1
    rt60_default = 400
    default_brir_settings = {'hrtf': hrtf_default, 'direct_gain': direct_gain_default, 'room_target': room_target_default, 'pinna_comp': pinna_comp_default, 'rt60': rt60_default,  
                             'brir_export': 1, 'brir_directional_export':1, 'brir_ts_export': 1, 'hesuvi_export': 1, 'eapo_export': 1}
    
    
    
    #
    ## GUI Functions - HPCFs
    #
    
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
        
 
    def plot_filter(sender, app_data):
        """ 
        GUI function to plot a selected filter
        """
        
        target = app_data
        #print(target)
        mag_response = target_mag_dict.get(target)
        
        #run plot
        plot_tile = target + ' frequency response'
        hf.plot_data(mag_response, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, plot_type=1)
        

    def export_fir_toggle(sender, app_data):
        """ 
        GUI function to update hpcf dictionary based on toggle
        """
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("hpcf_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        if app_data == True:
            modified_dict.update({'fir_export': 1})
        elif app_data == False:
            modified_dict.update({'fir_export': 0})    
        #update user data
        dpg.configure_item('hpcf_tag',user_data=modified_dict)
        
    def export_fir_stereo_toggle(sender, app_data):
        """ 
        GUI function to update hpcf dictionary based on toggle
        """
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("hpcf_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        if app_data == True:
            modified_dict.update({'fir_stereo_export': 1})
        elif app_data == False:
            modified_dict.update({'fir_stereo_export': 0})    
        #update user data
        dpg.configure_item('hpcf_tag',user_data=modified_dict)

    def export_geq_toggle(sender, app_data):
        """ 
        GUI function to update hpcf dictionary based on toggle
        """
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("hpcf_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        if app_data == True:
            modified_dict.update({'geq_export': 1})
        elif app_data == False:
            modified_dict.update({'geq_export': 0})    
        #update user data
        dpg.configure_item('hpcf_tag',user_data=modified_dict)
    

    def export_geq_31_toggle(sender, app_data):
        """ 
        GUI function to update hpcf dictionary based on toggle
        """
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("hpcf_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        if app_data == True:
            modified_dict.update({'geq_31_export': 1})
        elif app_data == False:
            modified_dict.update({'geq_31_export': 0})    
        #update user data
        dpg.configure_item('hpcf_tag',user_data=modified_dict)

    def export_hesuvi_hpcf_toggle(sender, app_data):
        """ 
        GUI function to update hpcf dictionary based on toggle
        """
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("hpcf_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        if app_data == True:
            modified_dict.update({'hesuvi_export': 1})
        elif app_data == False:
            modified_dict.update({'hesuvi_export': 0})    
        #update user data
        dpg.configure_item('hpcf_tag',user_data=modified_dict)

    def export_eapo_hpcf_toggle(sender, app_data):
        """ 
        GUI function to update hpcf dictionary based on toggle
        """
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("hpcf_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        if app_data == True:
            modified_dict.update({'eapo_export': 1})
        elif app_data == False:
            modified_dict.update({'eapo_export': 0})    
        #update user data
        dpg.configure_item('hpcf_tag',user_data=modified_dict)
    
 
    def export_hpcf_toggle(sender, app_data):
        """ 
        GUI function to update hpcf dictionary based on toggle
        """
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("hpcf_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        if app_data == True:
            modified_dict.update({'hpcf_export': 1})
        elif app_data == False:
            modified_dict.update({'hpcf_export': 0})    
        #update user data
        dpg.configure_item('hpcf_tag',user_data=modified_dict)

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
        geq_32_export = current_dict.get('geq_32_export')
        hesuvi_export = current_dict.get('hesuvi_export')
        eapo_export = current_dict.get('eapo_export')
        hpcf_export = current_dict.get('hpcf_export')
        
        if hpcf_export == 1:
            hpcf_functions.hpcf_to_file_bulk(conn, primary_path=output_path, headphone=headphone, fir_export = fir_export, fir_stereo_export = fir_stereo_export, geq_export = geq_export, 
                                             geq_31_export = geq_31_export, geq_32_export = geq_32_export, hesuvi_export = hesuvi_export, eapo_export=eapo_export, gui_logger=logz, report_progress=1)
   
    #
    ## GUI Functions - BRIRs
    #
    
 
    def export_brir_toggle(sender, app_data):
        """ 
        GUI function to update brir dictionary based on toggle
        """
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("brir_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        if app_data == True:
            modified_dict.update({'brir_export': 1})
        elif app_data == False:
            modified_dict.update({'brir_export': 0})    
        #update user data
        dpg.configure_item('brir_tag',user_data=modified_dict)
    
    def export_dir_brirs_toggle(sender, app_data):
        """ 
        GUI function to update brir dictionary based on toggle
        """
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("brir_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        if app_data == True:
            modified_dict.update({'brir_directional_export': 1})
        elif app_data == False:
            modified_dict.update({'brir_directional_export': 0})    
        #update user data
        dpg.configure_item('brir_tag',user_data=modified_dict)
 
    def export_ts_brirs_toggle(sender, app_data):
        """ 
        GUI function to update brir dictionary based on toggle
        """
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("brir_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        if app_data == True:
            modified_dict.update({'brir_ts_export': 1})
        elif app_data == False:
            modified_dict.update({'brir_ts_export': 0})    
        #update user data
        dpg.configure_item('brir_tag',user_data=modified_dict)
 
    def export_eapo_brir_toggle(sender, app_data):
        """ 
        GUI function to update brir dictionary based on toggle
        """
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("brir_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        if app_data == True:
            modified_dict.update({'eapo_export': 1})
        elif app_data == False:
            modified_dict.update({'eapo_export': 0})    
        #update user data
        dpg.configure_item('brir_tag',user_data=modified_dict)

    def export_hesuvi_brir_toggle(sender, app_data):
        """ 
        GUI function to update brir dictionary based on toggle
        """
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("brir_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        if app_data == True:
            modified_dict.update({'hesuvi_export': 1})
        elif app_data == False:
            modified_dict.update({'hesuvi_export': 0})    
        #update user data
        dpg.configure_item('brir_tag',user_data=modified_dict)

    def select_room_target(sender, app_data):
        """ 
        GUI function to update brir dictionary based on input
        """
        
        target = app_data
        
        #run plot
        mag_response = target_mag_dict.get(target)
        plot_tile = target + ' frequency response'
        hf.plot_data(mag_response, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=2, plot_type=1)
        
        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("brir_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        room_target_int = CN.ROOM_TARGET_LIST.index(target)
        #print(room_target_int)
        modified_dict.update({'room_target': room_target_int})
        #update user data
        dpg.configure_item('brir_tag',user_data=modified_dict)
        
        #reset progress bar
        dpg.set_value("progress_bar_brir", 0)
        dpg.configure_item("progress_bar_brir", overlay = 'progress')
 
    def update_direct_gain(sender, app_data):
        """ 
        GUI function to update brir dictionary based on input
        """
        
        #gain_db = app_data*-1
        gain_db = app_data

        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("brir_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        #print(gain_db)
        modified_dict.update({'direct_gain': gain_db})
        #update user data
        dpg.configure_item('brir_tag',user_data=modified_dict)
        
        #reset progress bar
        dpg.set_value("progress_bar_brir", 0)
        dpg.configure_item("progress_bar_brir", overlay = 'progress')
    
    def update_rt60(sender, app_data):
        """ 
        GUI function to update brir dictionary based on input
        """
        
        target_rt60 = app_data

        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("brir_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        #print(target_rt60)
        modified_dict.update({'rt60': target_rt60})
        #update user data
        dpg.configure_item('brir_tag',user_data=modified_dict)
        
        #reset progress bar
        dpg.set_value("progress_bar_brir", 0)
        dpg.configure_item("progress_bar_brir", overlay = 'progress')
    
    def update_hrtf(sender, app_data):
        """ 
        GUI function to update brir dictionary based on input
        """
        
        hrtf = app_data

        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("brir_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        room_target_int = CN.HRTF_LIST_NUM.index(hrtf)+1
        #print(room_target_int)
        modified_dict.update({'hrtf': room_target_int})
        #update user data
        dpg.configure_item('brir_tag',user_data=modified_dict)
        
        #reset progress bar
        dpg.set_value("progress_bar_brir", 0)
        dpg.configure_item("progress_bar_brir", overlay = 'progress')
    
    def update_hp_type(sender, app_data):
        """ 
        GUI function to update brir dictionary based on input
        """
        
        hp_type = app_data

        #get user data from process hpcf button, contains a dict
        current_dict = dpg.get_item_user_data("brir_tag")
        modified_dict = current_dict.copy()
        #change value in dict
        pinna_comp_int = CN.HP_COMP_LIST.index(hp_type)
        #print(pinna_comp_int)
        modified_dict.update({'pinna_comp': pinna_comp_int})
        #update user data
        dpg.configure_item('brir_tag',user_data=modified_dict)
        
        #reset progress bar
        dpg.set_value("progress_bar_brir", 0)
        dpg.configure_item("progress_bar_brir", overlay = 'progress')
    
 
    def process_brirs(sender, app_data, user_data):
        """ 
        GUI function to process BRIRs
        """
        
        
        output_path = dpg.get_value('selected_folder_base')
        
        current_dict = dpg.get_item_user_data("brir_tag")
        hrtf_type = current_dict.get('hrtf')
        direct_gain_db = current_dict.get('direct_gain')
        direct_gain_db = round(direct_gain_db,1)#round to nearest .1 dB
        room_target = current_dict.get('room_target')
        apply_pinna_comp = current_dict.get('pinna_comp')
        target_rt60 = current_dict.get('rt60')
        hesuvi_export = current_dict.get('hesuvi_export')
        eapo_export = current_dict.get('eapo_export')
        brir_set_export = current_dict.get('brir_export')
        brir_ts_export = current_dict.get('brir_ts_export')
        brir_directional_export = current_dict.get('brir_directional_export')
        
        if brir_set_export == 1:
        
            """
            #Run BRIR integration
            """
            brir_gen = brir_generation.generate_integrated_brir(hrtf_type=hrtf_type, direct_gain_db=direct_gain_db, room_target=room_target, 
                                                                apply_pinna_comp=apply_pinna_comp, target_rt60=target_rt60, report_progress=1, gui_logger=logz)
            
            """
            #Run BRIR export
            """
            #calculate name
            brir_name = CN.HRTF_LIST_SHORT[hrtf_type-1] +'_'+ str(target_rt60) + 'ms_' + str(direct_gain_db) + 'dB_' + CN.ROOM_TARGET_LIST_SHORT[room_target] + '_' + CN.HP_COMP_LIST_SHORT[apply_pinna_comp]
            brir_export.export_brir(brir_arr=brir_gen, hrtf_type=hrtf_type, target_rt60=target_rt60, brir_name=brir_name, primary_path=output_path, 
                                    brir_dir_export=brir_directional_export, brir_ts_export=brir_ts_export, hesuvi_export=hesuvi_export, report_progress=1, gui_logger=logz, direct_gain_db=direct_gain_db)
            
            #set progress to 100 as export is complete (assume E-APO export time is negligible)
            progress = 100/100
            dpg.set_value("progress_bar_brir", progress)
            dpg.configure_item("progress_bar_brir", overlay = str(int(progress*100))+'%')
        
        if eapo_export == 1:
            """
            #Run E-APO Config creator for BRIR convolution
            """
            e_apo_config_creation.write_e_apo_configs_brirs(brir_name=brir_name, primary_path=output_path, hrtf_type=hrtf_type)
            
       
    #
    ## GUI Functions - Additional DEV tools
    #    
       
    def generate_brir_reverb(sender, app_data, user_data):
        """ 
        GUI function to process BRIR reverberation data
        """

        #Run BRIR reverb synthesis
        brir_generation.generate_reverberant_brir(gui_logger=logz)

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
        
    def calc_hpcf_averages(sender, app_data, user_data):
        """ 
        GUI function to calculate hpcf averages
        """
        
        hpcf_functions.hpcf_generate_averages(conn, gui_logger=logz)
        
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
        hpcf_functions.generate_hp_summary_sheet(conn=conn, measurement_folder_name=measurement_folder_name, gui_logger=logz)

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
            
            try:
                #save folder name to config file
                config = configparser.ConfigParser()
                config['DEFAULT']['path'] = base_folder_selected    # update
                with open(CN.SETTINGS_FILE, 'w') as configfile:    # save
                    config.write(configfile)
            except:
                log_string = 'Failed to write to settings.ini'
                logz.log_info(log_string)
            

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
                

    def update_progress(sender, app_data):
        """ 
        GUI function to update progress bar
        """
        print(str(app_data))
        value = dpg.get_value("Drag int")
        dpg.set_value("progress_bar_brir", value/100)
        dpg.configure_item("progress_bar_brir", overlay = str(value)+'%')
             

    
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
    
    # add a font registry
    with dpg.font_registry():
        # first argument ids the path to the .ttf or .otf file
        in_file_path = pjoin(CN.DATA_DIR_EXT, 'Lato-Regular.ttf')#SourceSansPro-Regular
        default_font = dpg.add_font(in_file_path, 14)    
    
    
    dpg.create_viewport(title='Audio Spatialisation for Headphones', width=1650, height=900, small_icon=CN.ICON_LOCATION, large_icon=CN.ICON_LOCATION)
    
    with dpg.window(tag="Primary Window"):
        
        # set font of app
        dpg.bind_font(default_font)
        
        # Themes
        with dpg.theme(tag="__theme_a"):
            i=3
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(i/7.0, 0.6, 0.6))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(i/7.0, 0.8, 0.8))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(i/7.0, 0.7, 0.7))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, i*5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 8)
        with dpg.theme(tag="__theme_b"):
            i=2
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(i/7.0, 0.6, 0.6))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(i/7.0, 0.8, 0.8))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(i/7.0, 0.7, 0.7))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, i*5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 8)

        

        with dpg.collapsing_header(label="Headphone Correction and Binaural Room Simulation",default_open = True, leaf =True):
                              
            with dpg.group(horizontal=True):
                #Section for HpCF Export
                with dpg.child_window(width=550, height=510):
                    dpg.add_text("Headphone Correction Filters (HpCFs)")
                    #dpg.add_checkbox(label="Export HpCFs", default_value = True, callback=export_hpcf_toggle)
                    with dpg.child_window(autosize_x=True, height=350):
                        dpg.add_text("Select Headphone")
                        with dpg.group(horizontal=True, width=0):
                            with dpg.group():
                                dpg.add_text("Brand")
                                listbox_1 = dpg.add_listbox(brands_list, width=200, num_items=15, tag='brand_list', callback=update_headphone_list)
                            with dpg.group():
                                dpg.add_text("Headphone")
                                listbox_2 = dpg.add_listbox(hp_list_default, width=200, num_items=15, tag='headphone_list', default_value=headphone_default ,callback=update_sample_list)
                            with dpg.group():
                                dpg.add_text("Sample")
                                listbox_3 = dpg.add_listbox(sample_list_default, width=100, num_items=15, default_value=sample_default, tag='sample_list', user_data=headphone_default, callback=plot_sample)
                                with dpg.tooltip("sample_list"):
                                    dpg.add_text("Note: all samples will be exported. Select a sample to preview")
                    with dpg.child_window(autosize_x=True, height=115):
                        dpg.add_text("Select Files to Include in Export")
                        with dpg.group(horizontal=True):
                            dpg.add_checkbox(label="FIR Filters", default_value = True, callback=export_fir_toggle, tag='fir_hpcf_toggle')
                            with dpg.tooltip("fir_hpcf_toggle"):
                                dpg.add_text("Min phase WAV FIRs for convolution. 1 Channel, 24 bit depth, 44.1Khz")
                            dpg.add_checkbox(label="Stereo FIR Filters", default_value = True, callback=export_fir_stereo_toggle, tag='fir_st_hpcf_toggle')
                            with dpg.tooltip("fir_st_hpcf_toggle"):
                                dpg.add_text("Min phase WAV FIRs for convolution. 2 Channels, 24 bit depth, 44.1Khz")
                            dpg.add_checkbox(label="E-APO Configuration Files", default_value = True, callback=export_eapo_hpcf_toggle, tag='eapo_hpcf_toggle')  
                            with dpg.tooltip("eapo_hpcf_toggle"):
                                dpg.add_text("Equalizer APO configurations to perform convolution with FIR filters")
                        with dpg.group(horizontal=True):
                            dpg.add_checkbox(label="Graphic EQ Filters", default_value = True, callback=export_geq_toggle, tag='geq_hpcf_toggle')
                            with dpg.tooltip("geq_hpcf_toggle"):
                                dpg.add_text("Graphic EQ configurations with 127 bands. Compatible with Equalizer APO and Wavelet")
                            dpg.add_checkbox(label="Graphic EQ Filters (31 Bands)", default_value = True, callback=export_geq_31_toggle, tag='geq_31_hpcf_toggle')
                            with dpg.tooltip("geq_31_hpcf_toggle"):
                                dpg.add_text("Graphic EQ configurations with 31 bands. Compatible with 31 band graphic equalizers including Equalizer APO")
                            dpg.add_checkbox(label="HeSuVi Filters", default_value = True, callback=export_hesuvi_hpcf_toggle, tag='hesuvi_hpcf_toggle')
                            with dpg.tooltip("hesuvi_hpcf_toggle"):
                                dpg.add_text("Graphic EQ configurations with 127 bands. Compatible with HeSuVi. Saved in HeSuVi\eq folder")
                

                #Section for BRIR generation
                with dpg.child_window(width=460, height=510):
                    dpg.add_text("Binaural Room Impulse Responses (BRIRs)")
                    #dpg.add_checkbox(label="Generate BRIRs", default_value = True, callback=export_brir_toggle)
                    with dpg.child_window(autosize_x=True, height=350):
                        with dpg.group():
                            dpg.add_text("Select Gain for Direct Sound (dB)")
                            dpg.add_input_float(label="Direct Gain (dB)", format="%.02f", tag='direct_gain', min_value=CN.DIRECT_GAIN_MIN, max_value=CN.DIRECT_GAIN_MAX, default_value=direct_gain_default,min_clamped=True, max_clamped=True, callback=update_direct_gain)
                            with dpg.tooltip("direct_gain"):
                                dpg.add_text("Higher values will result in lower perceived distance. Lower values result in higher perceived distance")
                                #dpg.add_text("Higher values will sound closer. Lower values will sound further away")
                            dpg.add_text("Select Target RT60 Reverberation Time (ms)")
                            dpg.add_input_int(label="Target RT60 (ms)", tag='target_rt60', default_value=400, min_value=200, max_value=1250, min_clamped=True, max_clamped=True, callback=update_rt60)
                            with dpg.tooltip("target_rt60"):
                                dpg.add_text("Select a value between 200ms and 1250ms")
                            with dpg.group(horizontal=True):
                                with dpg.group():
                                    dpg.add_text("Select Dummy Head / Head & Torso Simulator")
                                    listbox_4 = dpg.add_listbox(CN.HRTF_LIST_NUM, default_value='01: Neumann KU 100', num_items=11, width=220, callback=update_hrtf)
                                with dpg.group():
                                    dpg.add_text("Select Headphone Type")
                                    listbox_5 = dpg.add_listbox(CN.HP_COMP_LIST, default_value='Over-Ear/On-Ear Headphones', width=200, callback=update_hp_type)
                                    dpg.add_text("Select Room Target")
                                    listbox_6 = dpg.add_listbox(CN.ROOM_TARGET_LIST, default_value='ASH Target', num_items=6, width=200, tag='rm_target_list', callback=select_room_target)
                    with dpg.child_window(autosize_x=True, height=115):
                        dpg.add_text("Select Files to Include in Export")
                        with dpg.group(horizontal=True):
                            dpg.add_checkbox(label="Direction Specific WAVs", default_value = True, callback=export_dir_brirs_toggle, tag='dir_brir_toggle')
                            with dpg.tooltip("dir_brir_toggle"):
                                dpg.add_text("Directional WAV BRIRs for convolution. 2 Channels, 24 bit depth, 44.1Khz")
                            dpg.add_checkbox(label="True Stereo WAVs", default_value = True, callback=export_ts_brirs_toggle, tag='ts_brir_toggle')
                            with dpg.tooltip("ts_brir_toggle"):
                                dpg.add_text("True Stereo WAV BRIRs for convolution. 4 Channels, 24 bit depth, 44.1Khz")
                        with dpg.group(horizontal=True):
                            dpg.add_checkbox(label="HeSuVi WAVs", default_value = True, callback=export_hesuvi_brir_toggle, tag='hesuvi_brir_toggle')  
                            with dpg.tooltip("hesuvi_brir_toggle"):
                                dpg.add_text("HeSuVi compatible WAV BRIRs. 14 Channels, 24 bit depth, 44.1Khz and 48Khz")
                            dpg.add_checkbox(label="E-APO Configuration Files", default_value = True, callback=export_eapo_brir_toggle, tag='eapo_brir_toggle')
                            with dpg.tooltip("eapo_brir_toggle"):
                                dpg.add_text("Equalizer APO configurations to perform convolution with BRIRs")
                    
                #Section for plotting
                with dpg.child_window(width=580, height=510):
                    dpg.add_text("Filter Preview")
                    #plotting
                    with dpg.child_window(width=560, height=470):
                        dpg.add_text("Select a filter from list to preview")
                        # create plot
                        with dpg.plot(label="Magnitude Response Plot", height=420, width=550):
                            # optionally create legend
                            dpg.add_plot_legend()
                    
                            # REQUIRED: create x and y axes
                            dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)", tag="x_axis", log_scale=True)
                            dpg.set_axis_limits("x_axis", 10, 20000)
                            dpg.add_plot_axis(dpg.mvYAxis, label="Magnitude (DB)", tag="y_axis")
                            dpg.set_axis_limits("y_axis", -20, 15)
                    
                            # series belong to a y axis
                            dpg.add_line_series(default_x, default_y, label="Plot", parent="y_axis", tag="series_tag")
                            #initial plot
                            hpcf_functions.hpcf_to_plot(conn, headphone_default, sample_default, plot_type=1)
            #Section for exporting files and logging
            with dpg.group(horizontal=True):
                #Section for Exporting files
                with dpg.child_window(width=550, height=305):
                    dpg.add_text("BRIR and HpCF Export")
                    with dpg.child_window(width=530, height=70):
                        dpg.add_text("Output Directory")
                        with dpg.group(horizontal=True):
                            dpge.add_file_browser(width=800,height=600,label='Choose Folder',show_as_window=True, dirs_only=True,show_ok_cancel=True, allow_multi_selection=False, collapse_sequences=True,callback=show_selected_folder)
                            dpg.add_text(tag='selected_folder_ash')
                            dpg.add_text(tag='selected_folder_base',show=False)
                            with dpg.tooltip("selected_folder_ash"):
                                dpg.add_text("Location to save HpCFs and BRIRs. Files will be saved under ASH-Custom-Set sub directory")
                    dpg.set_value('selected_folder_ash', primary_ash_path)
                    dpg.set_value('selected_folder_base', primary_path)
                    dpg.add_text("HpCFs")
                    dpg.add_button(label="Click Here to Process HpCFs",user_data="",tag="hpcf_tag", callback=process_hpcfs)
                    dpg.bind_item_theme(dpg.last_item(), "__theme_b")
                    dpg.configure_item('hpcf_tag',user_data=default_hpcf_settings)
                    with dpg.tooltip("hpcf_tag"):
                        dpg.add_text("This will export the selected HpCFs to above directory")
                    dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=30, width=530, overlay="progress", tag="progress_bar_hpcf")
                    
                    dpg.add_text("BRIRs")
                    dpg.add_button(label="Click Here to Process BRIRs",user_data="",tag="brir_tag", callback=process_brirs)
                    dpg.bind_item_theme(dpg.last_item(), "__theme_a")
                    dpg.configure_item('brir_tag',user_data=default_brir_settings)
                    with dpg.tooltip("brir_tag"):
                        dpg.add_text("This will generate the customised BRIRs and export to above directory. It may take a minute to process")
                    dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=30, width=530, overlay="progress", tag="progress_bar_brir")
                    #dpg.add_text("")
                    
                    #dpg.add_drag_int(label="Drag int", callback = update_progress, tag="Drag int")
                #section for logging
                with dpg.child_window(width=1048, height=305, tag="console_window"):
                    dpg.add_text("Log")
                
                
            with dpg.collapsing_header(label="Additional Tools"):
                with dpg.group(horizontal=True):
                    with dpg.group():
                        #Section for BRIRs
                        with dpg.child_window(width=400, height=120):
                            dpg.add_text("Regenerate Reverberation Data for BRIR Generation")
                            dpg.add_button(label="Click Here to Regenerate",user_data="",tag="brir_reverb_tag", callback=generate_brir_reverb)
                            with dpg.tooltip("brir_reverb_tag"):
                                dpg.add_text("This will regenerate the reverberation data used to generate BRIRs. Requires brir_dataset_compensated in data\interim folder. It may take some time to process")
                        #Section for database
                        with dpg.child_window(width=400, height=180):
                            dpg.add_text("Check for HpCF Database Updates")
                            dpg.add_button(label="Click Here to Check Versions",user_data="",tag="hpcf_db_version_tag", callback=check_db_version)
                            with dpg.tooltip("hpcf_db_version_tag"):
                                dpg.add_text("This will display version of local HpcF database and latest version in the log")
                            dpg.add_text("Download latest HpCF Database")
                            dpg.add_button(label="Click Here to Download",user_data="",tag="hpcf_db_download_tag", callback=download_latest_db)
                            with dpg.tooltip("hpcf_db_download_tag"):
                                dpg.add_text("This will download latest version of HpcF database and replace local version")
                            dpg.add_text("Rebuild HpCF Database from WAVs")
                            dpg.add_button(label="Click Here to Rebuild",user_data="",tag="hpcf_db_create", callback=create_db_from_wav)
                            with dpg.tooltip("hpcf_db_create"):
                                dpg.add_text("This will rebuild the HpCF database from WAV FIRs. Requires WAV FIRs in data\interim\hpcf_wavs folder")
                    with dpg.group():
                        #Section for HpCF bulk functions
                        with dpg.child_window(width=400, height=120):
                            dpg.add_text("Calculate HpCF Averages")
                            dpg.add_button(label="Click Here to Calculate",user_data="",tag="hpcf_average_tag", callback=calc_hpcf_averages)
                            with dpg.tooltip("hpcf_average_tag"):
                                dpg.add_text("This will create averaged HpCFs for headphones with multiple samples and no existing average")
                            dpg.add_text("Renumber HpCF Samples")
                            dpg.add_button(label="Click Here to Renumber",user_data="",tag="hpcf_renumber_tag", callback=renumber_hpcf_samples)
                            with dpg.tooltip("hpcf_renumber_tag"):
                                dpg.add_text("This will renumber HpCF samples to ensure consecutive numbers")
                        #Section for HpCFs
                        with dpg.child_window(width=400, height=180):
                            dpg.add_text("Enter Name of Headphone Measurements Folder")
                            dpg.add_input_text(label="input text", default_value="Folder Name",tag="hp_measurements_tag")
                            with dpg.tooltip("hp_measurements_tag"):
                                dpg.add_text("Enter name of folder containing headphone measurements as it appears in data \ raw \ headphone_measurements")
                            dpg.add_text("Generate HpCF Summary Sheet")
                            dpg.add_button(label="Click Here to Generate",user_data="",tag="hpcf_summary_tag", callback=generate_hpcf_summary)
                            with dpg.tooltip("hpcf_summary_tag"):
                                dpg.add_text("Creates CSV containing summary of measurements in above folder")
                            dpg.add_checkbox(label="In-ear Headphone Set", default_value = False,tag="in_ear_set_tag")
                            with dpg.tooltip("in_ear_set_tag"):
                                dpg.add_text("Select this option if measurements folder contains in ear headphones")
                            dpg.add_text("Calculate new HpCFs")
                            dpg.add_button(label="Click Here to Calculate",user_data="",tag="hpcf_new_calc_tag", callback=calc_new_hpcfs) 
                            with dpg.tooltip("hpcf_new_calc_tag"):
                                dpg.add_text("Calculates new HpCFs from measurements and adds to the database")
                    with dpg.group():
                        #Section for deleting HpCFs
                        with dpg.child_window(width=400, height=120):
                            dpg.add_text("Delete Headphone (selected headphone)")
                            dpg.add_button(label="Click Here to Delete",user_data="",tag="hpcf_delete_hp_tag", callback=delete_hp)
                            dpg.add_text("Delete Sample (selected sample)")
                            dpg.add_button(label="Click Here to Delete",user_data="",tag="hpcf_delete_sample_tag", callback=delete_sample)

                        #Section for modifying existing HpCFs
                        with dpg.child_window(width=400, height=180):
                            dpg.add_text("Enter New Name for Headphone")
                            dpg.add_input_text(label="input text", default_value="Headphone Name",tag="hpcf_rename_tag")
                            dpg.add_text("Rename Headphone (selected headphone)")
                            dpg.add_button(label="Click Here to Rename",user_data="",tag="hpcf_rename_hp_tag", callback=rename_hp_hp)
                            with dpg.tooltip("hpcf_rename_hp_tag"):
                                dpg.add_text("Renames HpCFs for the selected headphone to the specified name")
                            dpg.add_text("Rename Headphone (selected sample)")
                            dpg.add_button(label="Click Here to Rename",user_data="",tag="hpcf_rename_sample_tag", callback=rename_hp_sample)
                            with dpg.tooltip("hpcf_rename_sample_tag"):
                                dpg.add_text("Renames headphone field for the selected sample to the specified name")
                    with dpg.group():
                        #Section for renaming existing HpCFs - find and replace
                        with dpg.child_window(width=400, height=304):
                            dpg.add_text("Enter Sample Name to Replace")
                            dpg.add_input_text(label="input text", default_value="Sample Name",tag="sample_find_tag")
                            dpg.add_text("Enter New Name for Sample")
                            dpg.add_input_text(label="input text", default_value="New Name",tag="sample_replace_tag")
                            dpg.add_text("Bulk Rename Sample")
                            dpg.add_button(label="Click Here to Rename",user_data="",tag="bulk_rename_sample_tag", callback=bulk_rename_sample)
                            with dpg.tooltip("bulk_rename_sample_tag"):
                                dpg.add_text("Bulk renames sample name across all headphones")

                
    #dpg.show_font_manager()    
    
    dpg.setup_dearpygui()
    logz=logger.mvLogger(parent="console_window")
    #logz.log_info("test string")
    
    #section to log tool version on startup
    #log results
    with open(CN.METADATA_FILE) as fp:
        _info = json.load(fp)
    __version__ = _info['version']
    log_string = 'Audio Spatialisation for Headphones Toolset. Version: ' + __version__
    logz.log_info(log_string)
    
    
    dpg.show_viewport()
    dpg.set_primary_window("Primary Window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()
        
    #finally close the connection
    conn.close()

    logging.info('Finished') 

if __name__ == '__main__':
    main()