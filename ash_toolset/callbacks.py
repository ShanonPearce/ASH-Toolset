# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 19:16:49 2025

@author: Shanon
"""

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
import dearpygui.dearpygui as dpg
import numpy as np
import ast
import math
from time import sleep
import threading
import scipy as sp
import os


#
## GUI Functions - HPCFs
#

def filter_brand_list(sender, app_data):
    """ 
    GUI function to update list of brands based on input text
    """
    
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    brands_list = hpcf_db_dict['brands_list']
    
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
    
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    brands_list = hpcf_db_dict['brands_list']
    
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
    
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    brands_list = hpcf_db_dict['brands_list']
    
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
    
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    brands_list = hpcf_db_dict['brands_list']
    
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
  
def qc_show_hpcf_history(sender=None, app_data=None):
    """ 
    GUI function to update list of headphone based on exported hpcf files
    """
    
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    brands_list = hpcf_db_dict['brands_list']
    
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
        dpg.configure_item('qc_headphone_list',default_value=headphone)#default_value=headphone_list_specific[0])
        dpg.configure_item('qc_headphone_list',show=True)
        
        #also update sample list
        #headphone = headphone_list_specific[0]
        sample_list_specific = hpcf_functions.get_samples_list(conn, headphone)
        sample_list_sorted = (sorted(sample_list_specific))
        dpg.configure_item('qc_sample_list',items=sample_list_sorted)
  
        #also update plot to Sample A
        #sample = 'Sample A'
        hpcf_functions.hpcf_to_plot(conn, headphone, sample, plot_type=2)
        
        #reset sample list to Sample A
        dpg.configure_item('qc_sample_list',show=False)
        dpg.configure_item('qc_sample_list',default_value=sample)#default_value='Sample A'
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
    
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    
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
    
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    
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
    
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    
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
    
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    
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
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    
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
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    
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
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
    
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
    
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
    
    
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
    filter_name = calc_hpcf_name(full_name=False)
    filter_name_full = calc_hpcf_name(full_name=True)
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
    update_brir_param()
    

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
        hf.plot_data(mag_response, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1,y_lim_a=-12, y_lim_b=12, save_plot=0, normalise=2, plot_type=1)

    except:
        pass

    #reset progress bar
    update_brir_param()
 
def select_hrtf(sender=None, app_data=None):
    """ 
    GUI function to update brir based on input
    """
    fr_flat_mag=CN.FR_FLAT_MAG
    
    
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
    update_brir_param()

def select_hp_comp(sender, app_data):
    """ 
    GUI function to update brir based on input
    """
    impulse=CN.IMPULSE
    hp_type = dpg.get_value("brir_hp_type")
    pinna_comp_int = CN.HP_COMP_LIST.index(hp_type)
    pinna_comp = pinna_comp_int
    
    #run plot
    try:
        # load pinna comp filter (FIR)
        npy_fname = pjoin(CN.DATA_DIR_INT, 'headphone_ear_comp_dataset.npy')
        ear_comp_fir_dataset = hf.load_convert_npy_to_float64(npy_fname)
        pinna_comp_fir_loaded = ear_comp_fir_dataset[pinna_comp,:]
        


        pinna_comp_fir=np.zeros(CN.N_FFT)
        pinna_comp_fir[0:1024] = pinna_comp_fir_loaded[0:1024]
        data_fft = np.fft.fft(pinna_comp_fir)
        mag_response=np.abs(data_fft)
        plot_tile = 'Headphone Compensation: ' + hp_type
        hf.plot_data(mag_response, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1,y_lim_a=-10, y_lim_b=10, save_plot=0, normalise=2, level_ends=1, plot_type=1)

    except:
        pass

    #reset progress bar
    update_brir_param()




def update_direct_gain(sender, app_data):
    """ 
    GUI function to update brir based on input
    """
    d_gain=app_data
    dpg.set_value("direct_gain_slider", d_gain)

    #reset progress bar
    update_brir_param()

def update_direct_gain_slider(sender, app_data):
    """ 
    GUI function to update brir based on input
    """
    
    d_gain=app_data
    dpg.set_value("direct_gain", d_gain)

    #reset progress bar
    update_brir_param()
 
def update_crossover_f(sender=None, app_data=None, user_data=None):
    """ 
    GUI function to make updates to freq crossover parameters
    """
    if app_data == CN.SUB_FC_SETTING_LIST[0]:
        ac_space = dpg.get_value("acoustic_space_combo")
        ac_space_int = CN.AC_SPACE_LIST_GUI.index(ac_space)
        ac_space_src = CN.AC_SPACE_LIST_SRC[ac_space_int]
        f_crossover_var,order_var= brir_generation.get_ac_f_crossover(name_src=ac_space_src)
        dpg.set_value("crossover_f", f_crossover_var)
        
    if type(app_data) is int:
        dpg.set_value("crossover_f_mode", CN.SUB_FC_SETTING_LIST[1])#set mode to custom
        
    update_brir_param()

def update_ac_space(sender, app_data, user_data):
    """ 
    GUI function to make updates to acoustic spaces
    """
    crossover_mode = dpg.get_value("crossover_f_mode")
    if crossover_mode == CN.SUB_FC_SETTING_LIST[0]:
        update_crossover_f(app_data=crossover_mode)
        
    update_brir_param()

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
        hf.plot_data(mag_response, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, y_lim_a=-12, y_lim_b=12, save_plot=0, normalise=2, plot_type=2)

    except:
        pass

    #reset progress bar
    qc_update_brir_param()

 
def qc_select_hrtf(sender=None, app_data=None):
    """ 
    GUI function to update brir based on input
    """
    fr_flat_mag=CN.FR_FLAT_MAG
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
    qc_update_brir_param()

def qc_select_hp_comp(sender, app_data):
    """ 
    GUI function to update brir based on input
    """
    impulse=CN.IMPULSE
    hp_type = dpg.get_value("qc_brir_hp_type")
    pinna_comp_int = CN.HP_COMP_LIST.index(hp_type)
    pinna_comp = pinna_comp_int
    
    #run plot
    try:
   
        # load pinna comp filter (FIR)
        npy_fname = pjoin(CN.DATA_DIR_INT, 'headphone_ear_comp_dataset.npy')
        ear_comp_fir_dataset = hf.load_convert_npy_to_float64(npy_fname)
        pinna_comp_fir_loaded = ear_comp_fir_dataset[pinna_comp,:]
        
        pinna_comp_fir=np.zeros(CN.N_FFT)
        pinna_comp_fir[0:1024] = pinna_comp_fir_loaded[0:1024]
        data_fft = np.fft.fft(pinna_comp_fir)
        mag_response=np.abs(data_fft)
        plot_tile = 'Headphone Compensation: ' + hp_type
        hf.plot_data(mag_response, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, y_lim_a=-10, y_lim_b=10, save_plot=0, normalise=2, level_ends=1, plot_type=2)

    except:
        pass

    #reset progress bar
    qc_update_brir_param()
    
     
def qc_update_crossover_f(sender=None, app_data=None, user_data=None):
    """ 
    GUI function to make updates to freq crossover parameters
    """
    if app_data == CN.SUB_FC_SETTING_LIST[0]:
        ac_space = dpg.get_value("qc_acoustic_space_combo")
        ac_space_int = CN.AC_SPACE_LIST_GUI.index(ac_space)
        ac_space_src = CN.AC_SPACE_LIST_SRC[ac_space_int]
        f_crossover_var,order_var= brir_generation.get_ac_f_crossover(name_src=ac_space_src)
        dpg.set_value("qc_crossover_f", f_crossover_var)
        
    if type(app_data) is int:
        dpg.set_value("qc_crossover_f_mode", CN.SUB_FC_SETTING_LIST[1])#set mode to custom
        
    qc_update_brir_param()
    
def qc_lf_analyse_toggle(sender=None, app_data=None, user_data=None):
    """ 
    GUI function to make updates to LF analysis panel
    """
    #if toggled
    if app_data == True:
        lfa_type = dpg.get_value("qc_lf_analysis_type")
        if lfa_type == 'Magnitude':
            qc_lf_analyse_mag()
        if lfa_type == 'Group Delay':
            qc_lf_analyse_gd()
    else:
        #initial plot
        impulse=np.zeros(CN.N_FFT)
        impulse[0]=1
        fr_flat_mag = np.abs(np.fft.fft(impulse))
        hf.plot_data(fr_flat_mag, title_name='', n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=2, plot_type=3)
            
    qc_update_brir_param()
        
def qc_lf_analyse_change_type(sender=None, app_data=None, user_data=None):
    """ 
    GUI function to make updates to LF analysis panel
    """
    #if toggled
    activated = dpg.get_value("qc_lf_analysis_toggle")
    if activated == True:
        if app_data == 'Magnitude':
            qc_lf_analyse_mag()
        if app_data == 'Group Delay':
            qc_lf_analyse_gd()
            
    qc_update_brir_param()
    
def qc_lf_analyse_mag():
    """ 
    GUI function to plot magnitude response of exported BRIR in LFs
    """
    n_fft=CN.N_FFT
    peak_gain=0#default value
    impulse=np.zeros(n_fft)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat = hf.mag2db(fr_flat_mag)
    brir_in_arr=np.zeros(n_fft)
    
    primary_path=dpg.get_value('qc_selected_folder_base')
    brir_set=CN.FOLDER_BRIRS_LIVE
    brir_set_formatted = brir_set.replace(" ", "_")
    brirs_path = pjoin(primary_path, CN.PROJECT_FOLDER_BRIRS, brir_set_formatted)
    elev=0
    azim_fl=-30
    azim_fr=30
    total_irs=0
    brir_name_wav_l = 'BRIR' + '_E' + str(elev) + '_A' + str(azim_fl) + '.wav'
    brir_name_wav_r = 'BRIR' + '_E' + str(elev) + '_A' + str(azim_fr) + '.wav'
    #brir_name = calc_brir_set_name(full_name=False)
    brir_name = dpg.get_value('qc_e_apo_curr_brir_set')
    
    try:
        #sum 4x channels
        input_chan=0
        total_irs=total_irs+1
        wav_fname = pjoin(brirs_path, brir_name_wav_l)
        samplerate, fir_array = hf.read_wav_file(wav_fname)
        fir_length = min(len(fir_array),n_fft)
        brir_in_arr[0:fir_length]=np.add(brir_in_arr[0:fir_length],fir_array[0:fir_length,input_chan])
        input_chan=1
        total_irs=total_irs+1
        brir_in_arr[0:fir_length]=np.add(brir_in_arr[0:fir_length],fir_array[0:fir_length,input_chan])
        
        input_chan=0
        total_irs=total_irs+1
        wav_fname = pjoin(brirs_path, brir_name_wav_r)
        samplerate, fir_array = hf.read_wav_file(wav_fname)
        fir_length = min(len(fir_array),n_fft)
        brir_in_arr[0:fir_length]=np.add(brir_in_arr[0:fir_length],fir_array[0:fir_length,input_chan])
        input_chan=1
        total_irs=total_irs+1
        brir_in_arr[0:fir_length]=np.add(brir_in_arr[0:fir_length],fir_array[0:fir_length,input_chan])
        
        brir_in_arr=np.divide(brir_in_arr,total_irs)
        
        data_fft = np.fft.fft(brir_in_arr[0:CN.N_FFT])
        mag_fft=np.abs(data_fft)
    
        plot_title = brir_name 
        
        hf.plot_data_generic(mag_fft, title_name=plot_title, data_type='magnitude', n_fft=n_fft, samp_freq=samplerate, y_lim_adjust = 1, y_lim_a=-10, y_lim_b=10, x_lim_adjust = 1,x_lim_a=10, x_lim_b=200, save_plot=0, normalise=1,  plot_type=3)
        
    except Exception as ex:
        
        logging.error("Error occurred", exc_info = ex) 
        
        hf.plot_data(fr_flat_mag, title_name='File not found. Apply parameters to generate dataset', n_fft=n_fft, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=1, plot_type=3)
    
 
def qc_lf_analyse_gd():
    """ 
    GUI function to plot group delay response of exported BRIR in LFs
    """
    n_fft=CN.N_FFT
    peak_gain=0#default value
    impulse=np.zeros(n_fft)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat = hf.mag2db(fr_flat_mag)
    brir_in_arr_l=np.zeros(n_fft)
    
    primary_path=dpg.get_value('qc_selected_folder_base')
    brir_set=CN.FOLDER_BRIRS_LIVE
    brir_set_formatted = brir_set.replace(" ", "_")
    brirs_path = pjoin(primary_path, CN.PROJECT_FOLDER_BRIRS, brir_set_formatted)
    elev_fl=0
    azim_fl=-30
    brir_name_wav = 'BRIR' + '_E' + str(elev_fl) + '_A' + str(azim_fl) + '.wav'
    #brir_name = calc_brir_set_name(full_name=False)
    brir_name = dpg.get_value('qc_e_apo_curr_brir_set')
    
    try:

        wav_fname = pjoin(brirs_path, brir_name_wav)
        samplerate, fir_array = hf.read_wav_file(wav_fname)
        fir_length = min(len(fir_array),n_fft)
        brir_in_arr_l[0:fir_length]=fir_array[0:fir_length,0]
  
        freqs, gd = hf.calc_group_delay_from_ir(y=brir_in_arr_l, sr=samplerate, n_fft=n_fft, hop_length=512, smoothing_window=18, smoothing_type = 'octave', system_delay_ms=None)
   
        plot_title = brir_name 

        hf.plot_data_generic(gd, freqs=freqs, title_name=plot_title, data_type='group_delay',n_fft=n_fft, samp_freq=samplerate, y_lim_adjust = 1, y_lim_a=-150, y_lim_b=150, x_lim_adjust = 1,x_lim_a=10, x_lim_b=200,  plot_type=3)
        
    except Exception as ex:
        
        logging.error("Error occurred", exc_info = ex) 
        
        hf.plot_data(fr_flat_mag, title_name='File not found. Apply parameters to generate dataset', n_fft=n_fft, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=1, plot_type=3)
    
     
 
    

def qc_update_ac_space(sender, app_data, user_data):
    """ 
    GUI function to make updates to acoustic spaces
    """
    crossover_mode = dpg.get_value("qc_crossover_f_mode")
    if crossover_mode == CN.SUB_FC_SETTING_LIST[0]:
        qc_update_crossover_f(app_data=crossover_mode)
        
    qc_update_brir_param()


def qc_sort_ac_space(sender, app_data, user_data):
    """ 
    GUI function to make updates to acoustic spaces
    """
    if app_data == 'Name':
        dpg.configure_item('qc_acoustic_space_combo',items=CN.AC_SPACE_LIST_GUI)
    elif app_data == 'Reverberation Time':
        unsorted_list = CN.AC_SPACE_LIST_GUI
        values_list = CN.AC_SPACE_MEAS_R60
        sorted_list = hf.sort_names_by_values(names=unsorted_list, values=values_list, descending=False)
        dpg.configure_item('qc_acoustic_space_combo',items=sorted_list)
  
def sort_ac_space(sender, app_data, user_data):
    """ 
    GUI function to make updates to acoustic spaces
    """
    if app_data == 'Name':
        dpg.configure_item('acoustic_space_combo',items=CN.AC_SPACE_LIST_GUI)
    elif app_data == 'Reverberation Time':
        unsorted_list = CN.AC_SPACE_LIST_GUI
        values_list = CN.AC_SPACE_MEAS_R60
        sorted_list = hf.sort_names_by_values(names=unsorted_list, values=values_list, descending=False)
        dpg.configure_item('acoustic_space_combo',items=sorted_list)
  




def qc_update_direct_gain(sender, app_data):
    """ 
    GUI function to update brir based on input
    """
    d_gain=app_data
    dpg.set_value("qc_direct_gain_slider", d_gain)

    #reset progress bar
    qc_update_brir_param()
    

def qc_update_direct_gain_slider(sender, app_data):
    """ 
    GUI function to update brir based on input
    """
    
    d_gain=app_data
    dpg.set_value("qc_direct_gain", d_gain)

    #reset progress bar
    qc_update_brir_param()
    
def select_sub_brir(sender, app_data):
    """ 
    GUI function to update brir based on input
    """

    logz=dpg.get_item_user_data("console_window")#contains logger
    sub_response=app_data
    
    #run plot
    try:
        
        file_name = brir_generation.get_sub_f_name(sub_response=sub_response, gui_logger=logz)
        npy_fname = pjoin(CN.DATA_DIR_SUB, file_name+'.npy')
        sub_brir_npy = hf.load_convert_npy_to_float64(npy_fname)
        sub_brir_ir = np.zeros(CN.N_FFT)
        sub_brir_ir[0:CN.N_FFT] = sub_brir_npy[0][0:CN.N_FFT]
        data_fft = np.fft.fft(sub_brir_ir)
        sub_mag=np.abs(data_fft)
    
        plot_tile = sub_response
        hf.plot_data(sub_mag, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, y_lim_a=-8, y_lim_b=8, x_lim_adjust = 1,x_lim_a=10, x_lim_b=150, save_plot=0, normalise=1, plot_type=1)


    except:
        pass

    #reset progress bar
    update_brir_param()
    
def qc_select_sub_brir(sender, app_data):
    """ 
    GUI function to update brir based on input
    """

    logz=dpg.get_item_user_data("console_window")#contains logger
    sub_response=app_data
    
    #run plot
    try:
        
        file_name = brir_generation.get_sub_f_name(sub_response=sub_response, gui_logger=logz)
        npy_fname = pjoin(CN.DATA_DIR_SUB, file_name+'.npy')
        sub_brir_npy = hf.load_convert_npy_to_float64(npy_fname)
        sub_brir_ir = np.zeros(CN.N_FFT)
        sub_brir_ir[0:CN.N_FFT] = sub_brir_npy[0][0:CN.N_FFT]
        data_fft = np.fft.fft(sub_brir_ir)
        sub_mag=np.abs(data_fft)
    
        plot_tile = sub_response
        hf.plot_data(sub_mag, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, y_lim_a=-8, y_lim_b=8, x_lim_adjust = 1,x_lim_a=10, x_lim_b=150, save_plot=0, normalise=1, plot_type=2)


    except:
        pass

    #reset progress bar
    qc_update_brir_param()


def update_brir_param(sender=None, app_data=None):
    """ 
    GUI function to update brir based on input
    """

    #reset progress bar
    reset_brir_progress()
    save_settings()
    

def qc_update_brir_param(sender=None, app_data=None):
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
    logz=dpg.get_item_user_data("console_window")#contains logger
    #grab parameters
    brir_directional_export = (dpg.get_value("dir_brir_toggle"))
    brir_ts_export = (dpg.get_value("ts_brir_toggle"))
    hesuvi_export = (dpg.get_value("hesuvi_brir_toggle"))
    eapo_export = (dpg.get_value("eapo_brir_toggle"))
    sofa_export = (dpg.get_value("sofa_brir_toggle"))
    spat_res = dpg.get_value("brir_spat_res")
    spat_res_int = CN.SPATIAL_RES_LIST.index(spat_res)
    output_path = dpg.get_value('selected_folder_base')
    sofa_conv=dpg.get_value("sofa_exp_conv")
    
    #grab parameters
    brir_dict_params=get_brir_dict()
    
    #calculate name
    room_target = brir_dict_params.get("room_target")
    direct_gain_db = brir_dict_params.get("direct_gain_db")
    pinna_comp = brir_dict_params.get("pinna_comp")
    brir_hrtf_short=brir_dict_params.get('brir_hrtf_short')
    ac_space_short= brir_dict_params.get("ac_space_short")
    brir_name = brir_hrtf_short + '_'+ac_space_short + '_' + str(direct_gain_db) + 'dB_' + CN.ROOM_TARGET_LIST_SHORT[room_target] + '_' + CN.HP_COMP_LIST_SHORT[pinna_comp]
    
    

    
    
    
    """
    #Run BRIR integration
    """
    
    brir_gen, status = brir_generation.generate_integrated_brir(brir_name=brir_name, spatial_res=spat_res_int, report_progress=2, gui_logger=logz, brir_dict=brir_dict_params)
    
    """
    #Run BRIR export
    """
   
    if brir_gen.size != 0 and status == 0:
    
        brir_export.export_brir(brir_arr=brir_gen, brir_name=brir_name, primary_path=output_path, 
                             brir_dir_export=brir_directional_export, brir_ts_export=brir_ts_export, hesuvi_export=hesuvi_export, 
                            gui_logger=logz,  spatial_res=spat_res_int, sofa_export=sofa_export,  brir_dict=brir_dict_params, sofa_conv=sofa_conv)
        
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
    logz=dpg.get_item_user_data("console_window")#contains logger
    #preset
    brir_directional_export = True
    brir_ts_export = False
    hesuvi_export = False
    sofa_export = False
    spat_res_int = 0
    out_dataset_name = CN.FOLDER_BRIRS_LIVE 
    brir_name = calc_brir_set_name(full_name=False)
    brir_name_full = calc_brir_set_name(full_name=True)
    reduce_dataset = True
    output_path = dpg.get_value('qc_selected_folder_base')
    
    #grab parameters
    brir_dict_params=get_brir_dict()
 
   
    # target = dpg.get_value("qc_rm_target_list")
    # room_target_int = CN.ROOM_TARGET_LIST.index(target)
    # room_target = room_target_int
    # direct_gain_db = dpg.get_value("qc_direct_gain")
    # direct_gain_db = round(direct_gain_db,1)#round to nearest .1 dB
    # ac_space = dpg.get_value("qc_acoustic_space_combo")
    # ac_space_int = CN.AC_SPACE_LIST_GUI.index(ac_space)
    # ac_space_src = CN.AC_SPACE_LIST_SRC[ac_space_int]
    # hp_type = dpg.get_value("qc_brir_hp_type")
    # pinna_comp_int = CN.HP_COMP_LIST.index(hp_type)
    # pinna_comp = pinna_comp_int
    # samp_freq_str = dpg.get_value('qc_wav_sample_rate')
    # samp_freq_int = CN.SAMPLE_RATE_DICT.get(samp_freq_str)
    # bit_depth_str = dpg.get_value('qc_wav_bit_depth')
    # bit_depth = CN.BIT_DEPTH_DICT.get(bit_depth_str)
    # hrtf_symmetry = dpg.get_value('force_hrtf_symmetry')
    # er_delay_time = dpg.get_value('er_delay_time_tag')
    # er_delay_time = round(er_delay_time,1)#round to nearest .1 dB
    
    brir_dict_list=dpg.get_item_user_data("e_apo_brir_conv")#contains previously processed brirs
    
    #which brir dict to use for output, only used to specify directons
    force_use_brir_dict=dpg.get_item_user_data("qc_e_apo_curr_brir_set")
    if use_dict_list == False and force_use_brir_dict == False:#grab relevant config data from gui elements
        brir_dict_out=get_brir_dict()
    else:
        brir_dict_out=dpg.get_item_user_data("qc_e_apo_sel_brir_set")#grab from previously stored values, in case of direction change where brirs dont exist
    

    """
    #Run BRIR integration
    """
    if use_dict_list == False or not brir_dict_list:#run brir integration process
        brir_gen, status = brir_generation.generate_integrated_brir(brir_name=out_dataset_name, spatial_res=spat_res_int, report_progress=1, gui_logger=logz, brir_dict=brir_dict_params)
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
        brir_dict_list_new = brir_export.export_brir(brir_arr=brir_gen, brir_name=out_dataset_name, primary_path=output_path, 
                             brir_dir_export=brir_directional_export, brir_ts_export=brir_ts_export, hesuvi_export=hesuvi_export, 
                            gui_logger=logz,  spatial_res=spat_res_int, sofa_export=sofa_export, reduce_dataset=reduce_dataset, brir_dict=brir_dict_out,
                            use_dict_list=use_dict_list, brir_dict_list=brir_dict_list)
    
        #set progress to 100 as export is complete (assume E-APO export time is negligible)
        progress = 100/100
        hf.update_gui_progress(report_progress=1, progress=progress, message='Processed')
        #rewrite config file
        dpg.set_value("e_apo_brir_conv", True)
        #save dict list within gui element
        if brir_dict_list_new:#only update if not empty list
            old_data=dpg.get_item_user_data("e_apo_brir_conv")
            # Prevent memory leak by clearing old references
            if isinstance(old_data, list):
                old_data.clear()
            dpg.configure_item('e_apo_brir_conv',user_data=brir_dict_list_new)#use later if changing directions
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
            e_apo_update_direction(aquire_config=False, brir_dict_new=brir_dict_out)
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
    #plot LF analysis if toggled
    activated = dpg.get_value("qc_lf_analysis_toggle")
    qc_lf_analyse_toggle(app_data=activated)

def calc_brir_set_name(full_name=True):
    """ 
    GUI function to calculate brir set name from currently selected parameters
    """

    
    brir_dict=get_brir_dict()
    room_target = brir_dict.get("qc_room_target")
    direct_gain_db = brir_dict.get("qc_direct_gain_db")
    ac_space_short = brir_dict.get("qc_ac_space_short")
    pinna_comp = brir_dict.get("qc_pinna_comp")
    sample_rate = brir_dict.get("qc_samp_freq_str")
    bit_depth = brir_dict.get("qc_bit_depth_str")
    hrtf_symmetry = brir_dict.get("hrtf_symmetry")
    er_delay_time = brir_dict.get("er_delay_time")
    qc_brir_hrtf_short=brir_dict.get('qc_brir_hrtf_short')
    
    qc_crossover_f=brir_dict.get('qc_crossover_f')
    qc_sub_response=brir_dict.get('qc_sub_response')
    qc_sub_response_short=brir_dict.get('qc_sub_response_short')
    qc_hp_rolloff_comp=brir_dict.get('qc_hp_rolloff_comp')
    qc_fb_filtering=brir_dict.get('qc_fb_filtering')
    
    if full_name==True:
        brir_name = qc_brir_hrtf_short + ' '+ac_space_short + ' ' + str(direct_gain_db) + 'dB ' + CN.ROOM_TARGET_LIST_SHORT[room_target] + ' ' + CN.HP_COMP_LIST_SHORT[pinna_comp] + ' ' + sample_rate + ' ' + bit_depth + ' ' + hrtf_symmetry + ' ' + str(er_delay_time) + ' ' + str(qc_crossover_f) + ' ' + str(qc_sub_response) + ' ' + str(qc_hp_rolloff_comp) + ' ' + str(qc_fb_filtering) 
    else:
        brir_name = qc_brir_hrtf_short + ', '+ac_space_short + ', ' + str(direct_gain_db) + 'dB, ' + CN.ROOM_TARGET_LIST_SHORT[room_target] + ', ' + qc_sub_response_short+ '-' +str(qc_crossover_f) + ', ' + CN.HP_COMP_LIST_SHORT[pinna_comp] 

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
        filter_name = headphone + ', ' + sample


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
       
    
def save_settings(update_hpcf_pars=False,update_brir_pars=False):
    """ 
    GUI function to save settings
    """
    __version__=dpg.get_item_user_data("log_text")#contains version
    logz=dpg.get_item_user_data("console_window")#contains logger
    default_qc_brir_settings=dpg.get_item_user_data("qc_brir_title")#contains default settings
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    
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
            qc_crossover_f_mode_loaded=config['DEFAULT']['qc_crossover_f_mode']
            qc_crossover_f_loaded=int(config['DEFAULT']['qc_crossover_f'])
            qc_sub_response_loaded=config['DEFAULT']['qc_sub_response']
            qc_hp_rolloff_comp_loaded=ast.literal_eval(config['DEFAULT']['qc_hp_rolloff_comp'])
            qc_fb_filtering_loaded=ast.literal_eval(config['DEFAULT']['qc_fb_filtering'])
            
        else:
            raise ValueError('Settings not loaded due to version mismatch')
        
    except:

        qc_brir_hp_type_loaded = default_qc_brir_settings['qc_brir_hp_type_default']
        qc_room_target_loaded = default_qc_brir_settings['qc_room_target_default']
        qc_direct_gain_loaded = default_qc_brir_settings['qc_direct_gain_default']
        qc_ac_space_loaded = default_qc_brir_settings['qc_ac_space_default']
        qc_brand_loaded = default_qc_brir_settings['qc_brand_default']
        qc_headphone_loaded = default_qc_brir_settings['qc_headphone_default']
        qc_sample_loaded = default_qc_brir_settings['qc_sample_default']
        qc_hrtf_loaded = default_qc_brir_settings['qc_hrtf_default']
        qc_brir_hrtf_type_loaded = default_qc_brir_settings['qc_brir_hrtf_type_default']
        qc_brir_hrtf_dataset_loaded = default_qc_brir_settings['qc_brir_hrtf_dataset_default']
        qc_crossover_f_mode_loaded = default_qc_brir_settings['qc_crossover_f_mode_default']
        qc_crossover_f_loaded = default_qc_brir_settings['qc_crossover_f_default']
        qc_sub_response_loaded = default_qc_brir_settings['qc_sub_response_default']
        qc_hp_rolloff_comp_loaded = default_qc_brir_settings['qc_hp_rolloff_comp_default']
        qc_fb_filtering_loaded = default_qc_brir_settings['qc_fb_filtering_default']
    
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
        qc_crossover_f_mode_str = str(dpg.get_value('qc_crossover_f_mode'))
        qc_crossover_f_str = str(dpg.get_value('qc_crossover_f'))
        qc_sub_response_str = str(dpg.get_value('qc_sub_response'))
        qc_hp_rolloff_comp_str = str(dpg.get_value('qc_hp_rolloff_comp'))
        qc_fb_filtering_str = str(dpg.get_value('qc_fb_filtering'))
    else:
        #qc settings
        qc_hp_type_str = qc_brir_hp_type_loaded
        qc_room_target_str = qc_room_target_loaded
        qc_direct_gain_str = str(qc_direct_gain_loaded)
        qc_ac_space_str=qc_ac_space_loaded
        qc_hrtf_str = qc_hrtf_loaded
        qc_hrtf_type_str = qc_brir_hrtf_type_loaded
        qc_hrtf_dataset_str = qc_brir_hrtf_dataset_loaded
        
        qc_crossover_f_mode_str = qc_crossover_f_mode_loaded
        qc_crossover_f_str = str(qc_crossover_f_loaded)
        qc_sub_response_str = qc_sub_response_loaded
        qc_hp_rolloff_comp_str = str(qc_hp_rolloff_comp_loaded)
        qc_fb_filtering_str = str(qc_fb_filtering_loaded)


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
        config['DEFAULT']['show_hpcf_history'] = str(dpg.get_value('qc_toggle_hpcf_history'))
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
        
        config['DEFAULT']['qc_crossover_f_mode'] = qc_crossover_f_mode_str
        config['DEFAULT']['qc_crossover_f'] = qc_crossover_f_str
        config['DEFAULT']['qc_sub_response'] = qc_sub_response_str
        config['DEFAULT']['qc_hp_rolloff_comp'] = qc_hp_rolloff_comp_str
        config['DEFAULT']['qc_fb_filtering'] = qc_fb_filtering_str
        config['DEFAULT']['crossover_f_mode'] = str(dpg.get_value('crossover_f_mode'))
        config['DEFAULT']['crossover_f'] = str(dpg.get_value('crossover_f'))
        config['DEFAULT']['sub_response'] = str(dpg.get_value('sub_response'))
        config['DEFAULT']['hp_rolloff_comp'] = str(dpg.get_value('hp_rolloff_comp'))
        config['DEFAULT']['fb_filtering'] = str(dpg.get_value('fb_filtering'))

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
    logz=dpg.get_item_user_data("console_window")#contains logger
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
    logz=dpg.get_item_user_data("console_window")#contains logger
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
    
    #QC params
    qc_target = dpg.get_value("qc_rm_target_list")
    qc_room_target_int = CN.ROOM_TARGET_LIST.index(qc_target)
    qc_room_target = qc_room_target_int
    qc_direct_gain_db = dpg.get_value("qc_direct_gain")
    qc_direct_gain_db = round(qc_direct_gain_db,1)#round to nearest .1 dB
    qc_ac_space = dpg.get_value("qc_acoustic_space_combo")
    qc_ac_space_int = CN.AC_SPACE_LIST_GUI.index(qc_ac_space)
    qc_ac_space_short = CN.AC_SPACE_LIST_SHORT[qc_ac_space_int]
    qc_ac_space_src = CN.AC_SPACE_LIST_SRC[qc_ac_space_int]
    qc_hp_type = dpg.get_value("qc_brir_hp_type")
    qc_pinna_comp_int = CN.HP_COMP_LIST.index(qc_hp_type)
    qc_pinna_comp = qc_pinna_comp_int
    qc_samp_freq_str = dpg.get_value('qc_wav_sample_rate')
    qc_samp_freq_int = CN.SAMPLE_RATE_DICT.get(qc_samp_freq_str)
    qc_bit_depth_str = dpg.get_value('qc_wav_bit_depth')
    qc_bit_depth = CN.BIT_DEPTH_DICT.get(qc_bit_depth_str)
    
    #brir params
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
    samp_freq_str = dpg.get_value('wav_sample_rate')
    samp_freq_int = CN.SAMPLE_RATE_DICT.get(samp_freq_str)
    bit_depth_str = dpg.get_value('wav_bit_depth')
    bit_depth = CN.BIT_DEPTH_DICT.get(bit_depth_str)
    spat_res = dpg.get_value("brir_spat_res")
    spat_res_int = CN.SPATIAL_RES_LIST.index(spat_res)
    
    #misc
    hrtf_symmetry = dpg.get_value('force_hrtf_symmetry')
    er_delay_time = dpg.get_value('er_delay_time_tag')
    er_delay_time = round(er_delay_time,1)#round to nearest .1 dB
    
    #low freq
    qc_crossover_f_mode = dpg.get_value('qc_crossover_f_mode')
    qc_crossover_f = dpg.get_value('qc_crossover_f')
    qc_sub_response = dpg.get_value('qc_sub_response')
    qc_sub_response_int = CN.SUB_RESPONSE_LIST_GUI.index(qc_sub_response)
    qc_sub_response_short = CN.SUB_RESPONSE_LIST_SHORT[qc_sub_response_int]
    qc_hp_rolloff_comp = dpg.get_value('qc_hp_rolloff_comp')
    qc_fb_filtering = dpg.get_value('qc_fb_filtering')
    crossover_f_mode = dpg.get_value('crossover_f_mode')
    crossover_f = dpg.get_value('crossover_f')
    sub_response = dpg.get_value('sub_response')
    sub_response_int = CN.SUB_RESPONSE_LIST_GUI.index(sub_response)
    sub_response_short = CN.SUB_RESPONSE_LIST_SHORT[sub_response_int]
    hp_rolloff_comp = dpg.get_value('hp_rolloff_comp')
    fb_filtering = dpg.get_value('fb_filtering')

    brir_dict = {
        'enable_conv': enable_brir_selected, 'brir_set_folder': brir_set_folder, 
        'brir_set_name': brir_set_name, 'mute_fl': mute_fl_selected, 'mute_fr': mute_fr_selected, 
        'mute_c': mute_c_selected, 'mute_sl': mute_sl_selected, 'mute_sr': mute_sr_selected, 
        'mute_rl': mute_rl_selected, 'mute_rr': mute_rr_selected, 'gain_oa': gain_oa_selected, 
        'gain_fl': gain_fl_selected, 'gain_fr': gain_fr_selected, 'gain_c': gain_c_selected, 
        'gain_sl': gain_sl_selected, 'gain_sr': gain_sr_selected, 'gain_rl': gain_rl_selected, 
        'gain_rr': gain_rr_selected, 
    
        # Individual Elevation Variables
        'elev_fl': elev_list[0], 'elev_fr': elev_list[1], 'elev_c': elev_list[2], 
        'elev_sl': elev_list[3], 'elev_sr': elev_list[4], 'elev_rl': elev_list[5], 
        'elev_rr': elev_list[6], 
    
        # Individual Azimuth Variables
        'azim_fl': azim_list[0], 'azim_fr': azim_list[1], 'azim_c': azim_list[2], 
        'azim_sl': azim_list[3], 'azim_sr': azim_list[4], 'azim_rl': azim_list[5], 
        'azim_rr': azim_list[6], 
    
        # Elevation and Azimuth Lists
        'elev_list': elev_list, 'azim_list': azim_list, 
    
        # HRTF selections
        'brir_hrtf_type': brir_hrtf_type_selected, 'brir_hrtf_dataset': brir_hrtf_dataset_selected, 
        'brir_hrtf': brir_hrtf_selected, 'brir_hrtf_short': brir_hrtf_short, 
        'qc_brir_hrtf_type': qc_brir_hrtf_type_selected, 'qc_brir_hrtf_dataset': qc_brir_hrtf_dataset_selected, 
        'qc_brir_hrtf': qc_brir_hrtf_selected, 'qc_brir_hrtf_short': qc_brir_hrtf_short, 
    
        # Hesuvi (h_) Elevation Variables
        'h_elev_fl': h_elev_fl_selected, 'h_elev_fr': h_elev_fr_selected, 
        'h_elev_c': h_elev_c_selected, 'h_elev_sl': h_elev_sl_selected, 
        'h_elev_sr': h_elev_sr_selected, 'h_elev_rl': h_elev_rl_selected, 
        'h_elev_rr': h_elev_rr_selected, 
    
        # Hesuvi (h_) Azimuth Variables
        'h_azim_fl': h_azim_fl_selected, 'h_azim_fr': h_azim_fr_selected, 
        'h_azim_c': h_azim_c_selected, 'h_azim_sl': h_azim_sl_selected, 
        'h_azim_sr': h_azim_sr_selected, 'h_azim_rl': h_azim_rl_selected, 
        'h_azim_rr': h_azim_rr_selected, 
    
        # QC room and acoustic space selections
        'qc_room_target': qc_room_target, 'qc_room_target_int': qc_room_target_int, 'qc_direct_gain_db': qc_direct_gain_db, 
        'qc_ac_space_short': qc_ac_space_short, 'qc_ac_space_src': qc_ac_space_src, 
        'qc_pinna_comp': qc_pinna_comp, 'qc_samp_freq_int': qc_samp_freq_int, 'qc_samp_freq_str': qc_samp_freq_str,   
        'qc_bit_depth': qc_bit_depth, 'qc_bit_depth_str': qc_bit_depth_str, 
    
        # General BRIR selections
        'room_target': room_target, 'room_target_int': room_target_int, 'direct_gain_db': direct_gain_db, 
        'ac_space_short': ac_space_short, 'ac_space_src': ac_space_src, 
        'pinna_comp': pinna_comp, 'samp_freq_int': samp_freq_int, 'samp_freq_str': samp_freq_str, 
        'bit_depth': bit_depth, 'bit_depth_str': bit_depth_str, 'spat_res_int': spat_res_int, 
    
        # Additional variables
        'hrtf_symmetry': hrtf_symmetry, 'er_delay_time': er_delay_time,
        
        # Additional variables
        'qc_crossover_f_mode': qc_crossover_f_mode, 'qc_crossover_f': qc_crossover_f, 'qc_sub_response': qc_sub_response, 'qc_sub_response_short': qc_sub_response_short, 'qc_hp_rolloff_comp': qc_hp_rolloff_comp,
        'qc_fb_filtering': qc_fb_filtering, 'crossover_f_mode': crossover_f_mode, 'crossover_f': crossover_f, 'sub_response': sub_response, 'sub_response_short': sub_response_short, 
        'hp_rolloff_comp': hp_rolloff_comp, 'fb_filtering': fb_filtering
    }



    return brir_dict
    


def e_apo_config_acquire(estimate_gain=True):
    """ 
    GUI function to acquire lock on function to write updates to custom E-APO config
    """
    e_apo_conf_lock=dpg.get_item_user_data("e_apo_hpcf_conv")#contains lock
    e_apo_conf_lock.acquire()
    e_apo_config_write(estimate_gain=estimate_gain)
    e_apo_conf_lock.release()
    
def e_apo_config_acquire_gui(sender=None, app_data=None):
    """ 
    GUI function to acquire lock on function to write updates to custom E-APO config
    """
    e_apo_conf_lock=dpg.get_item_user_data("e_apo_hpcf_conv")#contains lock
    estimate_gain=True
    e_apo_conf_lock.acquire()
    e_apo_config_write(estimate_gain=estimate_gain)
    e_apo_conf_lock.release()
      

def e_apo_config_write(estimate_gain=True):
    """ 
    GUI function to write updates to custom E-APO config
    """
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
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
        
        if prevent_clipping != CN.AUTO_GAIN_METHODS[0] and load_config == True and one_chan_mute == False:
            
            if prevent_clipping == 'Align Low Frequencies':
                #peak gain - low frequencies
                est_pk_gain_reference = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '2.0 Stereo', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder,freq_mode = 'Align Low Frequencies'))
                constant_reduction=0.3
            elif prevent_clipping == 'Align Mid Frequencies':
                est_pk_gain_reference = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '2.0 Stereo', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder,freq_mode = 'Align Mid Frequencies'))
                constant_reduction=0.2
            else:
                est_pk_gain_reference = est_pk_gain_2
                constant_reduction=0.1
            gain_override = float(est_pk_gain_reference*-1)-constant_reduction#change polarity and reduce slightly
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
    dpg.set_value("e_apo_prevent_clip", CN.AUTO_GAIN_METHODS[0])
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
    __version__=dpg.get_item_user_data("log_text")#contains version
    logz=dpg.get_item_user_data("console_window")#contains logger
    
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
    


#
## GUI Functions - Additional DEV tools
#    

def print_summary(sender, app_data, user_data):
    """ 
    GUI function to print summary of recent HpCFs
    """
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
    hpcf_functions.get_recent_hpcfs(conn, gui_logger=logz)     
    
def generate_brir_reverb(sender, app_data, user_data):
    """ 
    GUI function to process BRIR reverberation data
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    #Run BRIR reverb synthesis
    brir_generation.generate_reverberant_brir(gui_logger=logz)

def check_app_version(sender, app_data, user_data):
    """ 
    GUI function to check app version
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    hpcf_functions.check_for_app_update(gui_logger=logz)

def check_db_version(sender, app_data, user_data):
    """ 
    GUI function to check db version
    """
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
    hpcf_functions.check_for_database_update(conn=conn, gui_logger=logz)

def download_latest_db(sender, app_data, user_data):
    """ 
    GUI function to download latest db
    """
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
    hpcf_functions.downlod_latest_database(conn=conn, gui_logger=logz)
    
def check_as_versions(sender, app_data, user_data):
    """ 
    GUI function to check acoustic space versions
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    air_processing.acoustic_space_updates(download_updates=False, gui_logger=logz)

def download_latest_as_sets(sender, app_data, user_data):
    """ 
    GUI function to download latest acoustic spaces
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    air_processing.acoustic_space_updates(download_updates=True, gui_logger=logz)

def check_hrtf_versions(sender, app_data, user_data):
    """ 
    GUI function to check hrtf versions
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    hrir_processing.hrir_metadata_updates(download_updates=False, gui_logger=logz)

def download_latest_hrtf_sets(sender, app_data, user_data):
    """ 
    GUI function to download latest hrtf metadata
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    hrir_processing.hrir_metadata_updates(download_updates=True, gui_logger=logz)
    
def check_all_updates():
    """ 
    GUI function to check for all updates
    """
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
    hpcf_functions.check_for_app_update(gui_logger=logz)
    hpcf_functions.check_for_database_update(conn=conn, gui_logger=logz)
    air_processing.acoustic_space_updates(download_updates=False, gui_logger=logz)
    hrir_processing.hrir_metadata_updates(download_updates=False, gui_logger=logz)


def calc_hpcf_averages(sender, app_data, user_data):
    """ 
    GUI function to calculate hpcf averages
    """
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
    hpcf_functions.hpcf_generate_averages(conn, gui_logger=logz)
 
def calc_hpcf_variants(sender, app_data, user_data):
    """ 
    GUI function to calculate hpcf averages
    """
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
    hpcf_functions.hpcf_generate_variants(conn, gui_logger=logz)
    
def crop_hpcfs(sender, app_data, user_data):
    """ 
    GUI function to calculate hpcf averages
    """
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
    hpcf_functions.crop_hpcf_firs(conn, gui_logger=logz)    
    
def renumber_hpcf_samples(sender, app_data, user_data):
    """ 
    GUI function to renumber hpcf samples to remove gaps
    """
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
    hpcf_functions.renumber_headphone_samples(conn, gui_logger=logz)
 
def generate_hpcf_summary(sender, app_data, user_data):
    """ 
    GUI function to generate hpcf summary sheet
    """
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
    measurement_folder_name = dpg.get_value('hp_measurements_tag')
    in_ear_set = dpg.get_value('in_ear_set_tag')
    hpcf_functions.generate_hp_summary_sheet(conn=conn, measurement_folder_name=measurement_folder_name, in_ear_set = in_ear_set, gui_logger=logz)

def calc_new_hpcfs(sender, app_data, user_data):
    """ 
    GUI function to calculate new hpcfs
    """
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
    measurement_folder_name = dpg.get_value('hp_measurements_tag')
    in_ear_set = dpg.get_value('in_ear_set_tag')
    hpcf_functions.calculate_new_hpcfs(conn=conn, measurement_folder_name=measurement_folder_name, in_ear_set = in_ear_set, gui_logger=logz)

def rename_hp_hp(sender, app_data, user_data):
    """ 
    GUI function to rename a headphone in db
    """
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
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
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
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
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
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
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
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
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
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
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
    hpcf_functions.hpcf_wavs_to_database(conn, gui_logger=logz)

def run_extract_airs(sender, app_data, user_data):
    """ 
    GUI function to run extract airs from recording function
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    dataset_name=dpg.get_value('air_dataset_name_tag')
    air_processing.extract_airs_from_recording(ir_set=dataset_name, gui_logger=logz)

def run_split_airs_to_set(sender, app_data, user_data):
    """ 
    GUI function to run split airs to air set function
    """
    logz=dpg.get_item_user_data("console_window")#contains logger

    dataset_name=dpg.get_value('air_dataset_name_tag')
    air_processing.split_airs_to_set(ir_set=dataset_name, gui_logger=logz)

def run_raw_air_to_dataset(sender, app_data, user_data):
    """ 
    GUI function to run RAW AIR to dataset function
    """
    logz=dpg.get_item_user_data("console_window")#contains logger

    dataset_name=dpg.get_value('air_dataset_name_tag')
    air_processing.prepare_air_set(ir_set=dataset_name, gui_logger=logz)
    
def run_air_to_brir(sender, app_data, user_data):
    """ 
    GUI function to run AIR to BRIR dataset function
    """
    logz=dpg.get_item_user_data("console_window")#contains logger

    dataset_name=dpg.get_value('air_dataset_name_tag')
    air_processing.airs_to_brirs(ir_set=dataset_name, gui_logger=logz)

def run_raw_to_brir(sender, app_data, user_data):
    """ 
    GUI function to run AIR to BRIR dataset function
    """
    logz=dpg.get_item_user_data("console_window")#contains logger

    dataset_name=dpg.get_value('air_dataset_name_tag')
    air_processing.raw_brirs_to_brir_set(ir_set=dataset_name, gui_logger=logz)

def run_mono_cue(sender, app_data, user_data):
    """ 
    GUI function to run mono cues processing function
    """
    logz=dpg.get_item_user_data("console_window")#contains logger

    #brir_generation.process_mono_cues(gui_logger=logz)
    #250414 new version
    brir_generation.process_mono_cues_v2(gui_logger=logz)
    
def run_mono_cue_hp(sender, app_data, user_data):
    """ 
    GUI function to calculate new hpcfs
    """
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
    measurement_folder_name = dpg.get_value('hp_measurements_tag')
    in_ear_set = dpg.get_value('in_ear_set_tag')
    hpcf_functions.process_mono_hp_cues(conn=conn, measurement_folder_name=measurement_folder_name, in_ear_set = in_ear_set, gui_logger=logz)
    
def generate_hrir_dataset(sender, app_data, user_data):
    """ 
    GUI function to run AIR to BRIR dataset function
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    spat_res = dpg.get_value("brir_spat_res")
    spat_res_int = CN.SPATIAL_RES_LIST.index(spat_res)
    brir_generation.preprocess_hrirs(spatial_res=spat_res_int, gui_logger=logz)

def calc_reverb_target(sender, app_data, user_data):
    """ 
    GUI function to run calc reverb target function
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    air_processing.calc_reverb_target_mag(gui_logger=logz)
    
def run_sub_brir_calc(sender, app_data, user_data):
    """ 
    GUI function to run calc sub brir function
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    air_processing.calc_subrir(gui_logger=logz)
    
def run_room_target_calc(sender, app_data, user_data):
    """ 
    GUI function to run calc avg room target function
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
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

 

