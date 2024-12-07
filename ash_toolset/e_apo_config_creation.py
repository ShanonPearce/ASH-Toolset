# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 23:20:14 2023

@author: Shanon
"""


# import packages
from os.path import join as pjoin
import os
from pathlib import Path
import time
import logging
from datetime import date
from ash_toolset import constants as CN
from ash_toolset import brir_export
from scipy.io import wavfile
import numpy as np
from ash_toolset import helper_functions as hf

today = str(date.today())

logger = logging.getLogger(__name__)
log_info=1





def write_e_apo_configs_brirs(brir_name, primary_path, hrtf_type):
    """
    Function creates equalizer APO configuration files for specified brir name
    :param hrtf_type: int, selected HRTF type
    :param brir_name: string, name of brir
    :param primary_path: string, base path to save files to
    :return: None
    """ 
    
    # get the start time
    st = time.time()


    try:
    
        brir_folder = brir_name
        output_config_path = pjoin(primary_path, CN.PROJECT_FOLDER_CONFIGS_BRIR,brir_folder)
        
        conv_brir_rel_dir = 'Convolution: ..\\..\\..\\BRIRs\\'+brir_name+'\\'
        brir_part_b = 'BRIR'
        brir_part_c = '_E0_'
        brir_part_d = '.wav'
        
        elev_range = ['','_Elevated_Front']
        
        #for every speaker config, write file
        for m in elev_range:
            
            if m == '_Elevated_Front':
                if hrtf_type in CN.HRTF_TYPE_LIST_LIM_RES:
                    elev='30'
                else:
                    elev='15'
            else:
                elev='0'
                    
            for n in range(CN.NUM_SPEAK_CONFIGS):
                
                out_file_name = CN.CHANNEL_CONFIGS[n][0] +m+ '_Config.txt'
                out_file_path = pjoin(output_config_path, out_file_name)
                
                #create dir if doesnt exist
                output_file = Path(out_file_path)
                output_file.parent.mkdir(exist_ok=True, parents=True)
        
                #write to txt file
                with open(out_file_path, 'w') as f:
                    f.write('BRIR convolution configuration for Equalizer APO')
                    f.write('\n')
                    f.write('Speaker setup: ' + CN.CHANNEL_CONFIGS[n][2])
                    f.write('\n')
                    f.write('Dated: ' + today)
                    f.write('\n')
                    f.write('\n')
                    f.write('Channel: ALL')
                    f.write('\n')
                    f.write('Preamp: '+ str(CN.HRTF_GAIN_LIST_NUM[hrtf_type-1]) +' dB')#also adjust gain
                    f.write('\n')
                    if CN.CHANNEL_CONFIGS[n][1] == '2.0' or CN.CHANNEL_CONFIGS[n][1] == '2.0W' or CN.CHANNEL_CONFIGS[n][1] == '2.0N':
                        copy_string = 'Copy: L_INPUT_L_EAR=L L_INPUT_R_EAR=L R_INPUT_L_EAR=R R_INPUT_R_EAR=R'
                    elif CN.CHANNEL_CONFIGS[n][1] == '7.1N':
                        copy_string = 'Copy: L_INPUT_L_EAR=L L_INPUT_R_EAR=L R_INPUT_L_EAR=R R_INPUT_R_EAR=R C_INPUT_L_EAR=C+SUB C_INPUT_R_EAR=C+SUB RL_INPUT_L_EAR=RL RL_INPUT_R_EAR=RL RR_INPUT_L_EAR=RR RR_INPUT_R_EAR=RR SL_INPUT_L_EAR=SL SL_INPUT_R_EAR=SL SR_INPUT_L_EAR=SR SR_INPUT_R_EAR=SR'
                    elif CN.CHANNEL_CONFIGS[n][1] == '7.1W':
                        copy_string = 'Copy: L_INPUT_L_EAR=L L_INPUT_R_EAR=L R_INPUT_L_EAR=R R_INPUT_R_EAR=R C_INPUT_L_EAR=C+SUB C_INPUT_R_EAR=C+SUB RL_INPUT_L_EAR=RL RL_INPUT_R_EAR=RL RR_INPUT_L_EAR=RR RR_INPUT_R_EAR=RR SL_INPUT_L_EAR=SL SL_INPUT_R_EAR=SL SR_INPUT_L_EAR=SR SR_INPUT_R_EAR=SR'
                    elif CN.CHANNEL_CONFIGS[n][1] == '5.1':
                        copy_string = 'Copy: L_INPUT_L_EAR=L L_INPUT_R_EAR=L R_INPUT_L_EAR=R R_INPUT_R_EAR=R C_INPUT_L_EAR=C+SUB C_INPUT_R_EAR=C+SUB RL_INPUT_L_EAR=RL RL_INPUT_R_EAR=RL RR_INPUT_L_EAR=RR RR_INPUT_R_EAR=RR'
                    f.write(copy_string)
                    f.write('\n')
                    f.write('Channel: L_INPUT_L_EAR L_INPUT_R_EAR')
                    f.write('\n')
                    if CN.CHANNEL_CONFIGS[n][1] == '2.0W' and hrtf_type != 4:
                        f.write(conv_brir_rel_dir + brir_part_b + '_E' + elev +'_A' + CN.AZIM_DICT['WIDE_FL'] + brir_part_d)
                    elif CN.CHANNEL_CONFIGS[n][1] == '2.0N' and hrtf_type != 4:
                        f.write(conv_brir_rel_dir + brir_part_b + '_E' + elev +'_A' + CN.AZIM_DICT['NARROW_FL'] + brir_part_d)
                    else:
                        f.write(conv_brir_rel_dir + brir_part_b + '_E' + elev +'_A' + CN.AZIM_DICT['FL'] + brir_part_d)
                    f.write('\n')
                    f.write('Channel: R_INPUT_L_EAR R_INPUT_R_EAR')
                    f.write('\n')
                    if CN.CHANNEL_CONFIGS[n][1] == '2.0W' and hrtf_type != 4:
                        f.write(conv_brir_rel_dir + brir_part_b + '_E' + elev +'_A' + CN.AZIM_DICT['WIDE_FR'] + brir_part_d)
                    elif CN.CHANNEL_CONFIGS[n][1] == '2.0N' and hrtf_type != 4:
                        f.write(conv_brir_rel_dir + brir_part_b + '_E' + elev +'_A' + CN.AZIM_DICT['NARROW_FR'] + brir_part_d)
                    else:
                        f.write(conv_brir_rel_dir + brir_part_b + '_E' + elev +'_A' + CN.AZIM_DICT['FR'] + brir_part_d)
                    f.write('\n')
                    if CN.CHANNEL_CONFIGS[n][1] == '7.1N' or CN.CHANNEL_CONFIGS[n][1] == '7.1W' or CN.CHANNEL_CONFIGS[n][1] == '5.1':
                        f.write('Channel: C_INPUT_L_EAR C_INPUT_R_EAR')
                        f.write('\n')
                        f.write(conv_brir_rel_dir + brir_part_b + brir_part_c +'A' + CN.AZIM_DICT['FC'] + brir_part_d)
                        f.write('\n')
                    
                        f.write('Channel: RL_INPUT_L_EAR RL_INPUT_R_EAR')
                        f.write('\n')
                        if CN.CHANNEL_CONFIGS[n][1] == '5.1':
                            f.write(conv_brir_rel_dir + brir_part_b + brir_part_c +'A' + CN.AZIM_DICT['WIDEST_BL'] + brir_part_d)
                        elif CN.CHANNEL_CONFIGS[n][1] == '7.1W':
                            f.write(conv_brir_rel_dir + brir_part_b + brir_part_c +'A' + CN.AZIM_DICT['WIDE_BL'] + brir_part_d)
                        elif CN.CHANNEL_CONFIGS[n][1] == '7.1N':
                            f.write(conv_brir_rel_dir + brir_part_b + brir_part_c +'A' + CN.AZIM_DICT['NARROW_BL'] + brir_part_d)
                        f.write('\n')
                        f.write('Channel: RR_INPUT_L_EAR RR_INPUT_R_EAR')
                        f.write('\n')
                        if CN.CHANNEL_CONFIGS[n][1] == '5.1':
                            f.write(conv_brir_rel_dir + brir_part_b + brir_part_c +'A' + CN.AZIM_DICT['WIDEST_BR'] + brir_part_d)
                        elif CN.CHANNEL_CONFIGS[n][1] == '7.1W':
                            f.write(conv_brir_rel_dir + brir_part_b + brir_part_c +'A' + CN.AZIM_DICT['WIDE_BR'] + brir_part_d)
                        elif CN.CHANNEL_CONFIGS[n][1] == '7.1N':
                            f.write(conv_brir_rel_dir + brir_part_b + brir_part_c +'A' + CN.AZIM_DICT['NARROW_BR'] + brir_part_d)
                        f.write('\n')
                    if CN.CHANNEL_CONFIGS[n][1] == '7.1N' or CN.CHANNEL_CONFIGS[n][1] == '7.1W':
                        f.write('Channel: SL_INPUT_L_EAR SL_INPUT_R_EAR')
                        f.write('\n')
                        f.write(conv_brir_rel_dir + brir_part_b + brir_part_c +'A' + CN.AZIM_DICT['SL'] + brir_part_d)
                        f.write('\n')
                        
                        f.write('Channel: SR_INPUT_L_EAR SR_INPUT_R_EAR')
                        f.write('\n')
                        f.write(conv_brir_rel_dir + brir_part_b + brir_part_c +'A' + CN.AZIM_DICT['SR'] + brir_part_d)
                        f.write('\n')
                        
                    
                    if CN.CHANNEL_CONFIGS[n][1] == '2.0' or CN.CHANNEL_CONFIGS[n][1] == '2.0W' or CN.CHANNEL_CONFIGS[n][1] == '2.0N':
                        copy_string = 'Copy: L=L_INPUT_L_EAR+R_INPUT_L_EAR R=L_INPUT_R_EAR+R_INPUT_R_EAR'
                    elif CN.CHANNEL_CONFIGS[n][1] == '7.1N':
                        copy_string = 'Copy: L=L_INPUT_L_EAR+R_INPUT_L_EAR+C_INPUT_L_EAR+RL_INPUT_L_EAR+RR_INPUT_L_EAR+SL_INPUT_L_EAR+SR_INPUT_L_EAR R=L_INPUT_R_EAR+R_INPUT_R_EAR+C_INPUT_R_EAR+RL_INPUT_R_EAR+RR_INPUT_R_EAR+SL_INPUT_R_EAR+SR_INPUT_R_EAR C=0*C RL=0*RL RR=0*RR SL=0*SL SR=0*SR SUB=0*SUB'
                    elif CN.CHANNEL_CONFIGS[n][1] == '7.1W':
                        copy_string = 'Copy: L=L_INPUT_L_EAR+R_INPUT_L_EAR+C_INPUT_L_EAR+RL_INPUT_L_EAR+RR_INPUT_L_EAR+SL_INPUT_L_EAR+SR_INPUT_L_EAR R=L_INPUT_R_EAR+R_INPUT_R_EAR+C_INPUT_R_EAR+RL_INPUT_R_EAR+RR_INPUT_R_EAR+SL_INPUT_R_EAR+SR_INPUT_R_EAR C=0*C RL=0*RL RR=0*RR SL=0*SL SR=0*SR SUB=0*SUB'
                    elif CN.CHANNEL_CONFIGS[n][1] == '5.1':
                        copy_string = 'Copy: L=L_INPUT_L_EAR+R_INPUT_L_EAR+C_INPUT_L_EAR+RL_INPUT_L_EAR+RR_INPUT_L_EAR R=L_INPUT_R_EAR+R_INPUT_R_EAR+C_INPUT_R_EAR+RL_INPUT_R_EAR+RR_INPUT_R_EAR C=0*C RL=0*RL RR=0*RR SL=0 SR=0 SUB=0*SUB'
                    f.write(copy_string)
        
        if CN.LOG_INFO == 1:
            logging.info('Equalizer APO Configurations saved to: ' + str(output_config_path))

        
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
            
    # get the end time
    et = time.time()
    
    # get the execution time
    elapsed_time = et - st
    if CN.LOG_INFO == 1:
        logging.info('Execution time:' + str(elapsed_time) + ' seconds')
        
        

def write_e_apo_configs_hpcfs(brand, headphone, sample, primary_path):
    """
    Function creates equalizer APO configuration files for specified hpcf name
    :param brand: string, headphone brand
    :param headphone: string, name of headphone
    :param sample: string, name of sample
    :param primary_path: string, base path to save files to
    :return: None
    """ 

    # get the start time
    st = time.time()


    try:
    
        #hpcf wav directory
        out_file_dir_wav = pjoin(primary_path, CN.PROJECT_FOLDER_HPCFS,'FIRs',brand)
        hpcf_name_wav = headphone + ' ' + sample + '.wav'
        hpcf_name_wav = hpcf_name_wav.replace(" ", "_")
    
        #hpcf EAPO config path
        output_config_path = pjoin(primary_path, CN.PROJECT_FOLDER_CONFIGS_HPCF,brand)
    
        out_file_name = headphone + ' ' + sample + '.txt'
        out_file_name = out_file_name.replace(" ", "_")
        out_file_path = pjoin(output_config_path, out_file_name)
        
        #create dir if doesnt exist
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        conv_hpcf_rel_dir = 'Convolution: ..\\..\\..\\HpCFs\\FIRs\\'+brand+'\\'
        
        #write to txt file
        with open(out_file_path, 'w') as f:
            f.write('HpCF convolution configuration for Equalizer APO')
            f.write('\n')
            f.write('Headphone: ' + headphone)
            f.write('\n')
            f.write('Sample: ' + sample)
            f.write('\n')
            f.write('Dated: ' + today)
            f.write('\n')
            f.write('Channel: ALL')
            f.write('\n')
            f.write('Preamp: 6 dB')#also adjust gain
            f.write('\n')
            f.write(conv_hpcf_rel_dir + hpcf_name_wav)
        
        
        if CN.LOG_INFO == 1:
            logging.info('Equalizer APO Configurations saved to: ' + str(output_config_path))

        
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
            
    # get the end time
    et = time.time()
    
    # get the execution time
    elapsed_time = et - st
    if CN.LOG_INFO == 1:
        logging.info('Execution time:' + str(elapsed_time) + ' seconds')
        
 
def write_ash_e_apo_config(primary_path, hpcf_dict, brir_dict, audio_channels, spatial_res=1, gui_logger=None):
    """
    Function creates equalizer APO configuration file to perform relevant convolution commands. This file needs to be loaded into config.txt
    :param primary_path: string, base path to save files to
    :param hpcf_dict: dict, dictionary containing hpcf related metadata
    :param brir_dict: dict, dictionary containing brir related metadata
    :param primary_path: string, selected audio channel configuration
    :param spatial_res: int, spatial resolution, 0= low, 1 = moderate, 2 = high, 3 = full
    :return: float containing overall gain
    """ 
    out_file=''
    additional_gain=CN.HRTF_GAIN_ADDIT#
    #impulse
    fft_size=CN.N_FFT
    impulse=np.zeros(fft_size)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat = hf.mag2db(fr_flat_mag)
    gain_overall=0.0
    
    try:
        #get variables from dicts
        enable_hpcf_conv = hpcf_dict.get('enable_conv')
        brand = hpcf_dict.get('brand')
        headphone = hpcf_dict.get('headphone')
        sample = hpcf_dict.get('sample')
        if not headphone or headphone == '' or not sample or sample == '':
            enable_hpcf_conv=False #dont proceed with hpcf convolution if no headphone or sample provided
        
        enable_brir_conv = brir_dict.get('enable_conv')
        brir_set_folder=brir_dict.get('brir_set_folder')
        brir_set_name=brir_dict.get('brir_set_name')
        if not brir_set_name or brir_set_name == '' or not brir_set_folder or brir_set_folder == '':
            enable_brir_conv=False #dont proceed with brir convolution if no set name provided
        
        mute_fl=brir_dict.get('mute_fl')
        mute_fr=brir_dict.get('mute_fr')
        mute_c=brir_dict.get('mute_c')
        mute_sl=brir_dict.get('mute_sl')
        mute_sr=brir_dict.get('mute_sr')
        mute_rl=brir_dict.get('mute_rl')
        mute_rr=brir_dict.get('mute_rr')
        gain_oa=brir_dict.get('gain_oa')
        gain_fl=brir_dict.get('gain_fl')
        gain_fr=brir_dict.get('gain_fr')
        gain_c=brir_dict.get('gain_c')
        gain_sl=brir_dict.get('gain_sl')
        gain_sr=brir_dict.get('gain_sr')
        gain_rl=brir_dict.get('gain_rl')
        gain_rr=brir_dict.get('gain_rr')
        if mute_fl == True:
            gain_fl=-100
        if mute_fr == True:
            gain_fr=-100
        if mute_c == True:
            gain_c=-100
        if mute_sl == True:
            gain_sl=-100
        if mute_sr == True:
            gain_sr=-100
        if mute_rl == True:
            gain_rl=-100
        if mute_rr == True:
            gain_rr=-100
            
        elev_fl=brir_dict.get('elev_fl')
        elev_fr=brir_dict.get('elev_fr')
        elev_c=brir_dict.get('elev_c')
        elev_sl=brir_dict.get('elev_sl')
        elev_sr=brir_dict.get('elev_sr')
        elev_rl=brir_dict.get('elev_rl')
        elev_rr=brir_dict.get('elev_rr')
        
        azim_fl=brir_dict.get('azim_fl')
        azim_fr=brir_dict.get('azim_fr')
        azim_c=brir_dict.get('azim_c')
        azim_sl=brir_dict.get('azim_sl')
        azim_sr=brir_dict.get('azim_sr')
        azim_rl=brir_dict.get('azim_rl')
        azim_rr=brir_dict.get('azim_rr')
        
        #get htrf type from file name
        hrtf_type=get_hrtf_type_from_dir(brir_set_name, spatial_res=spatial_res)

            
        nearest_dir_dict_fl = brir_export.find_nearest_direction(hrtf_type=hrtf_type ,target_elevation=elev_fl , target_azimuth=azim_fl, spatial_res=spatial_res)
        nearest_dir_dict_fr = brir_export.find_nearest_direction(hrtf_type=hrtf_type ,target_elevation=elev_fr , target_azimuth=azim_fr, spatial_res=spatial_res)
        nearest_dir_dict_c = brir_export.find_nearest_direction(hrtf_type=hrtf_type ,target_elevation=elev_c , target_azimuth=azim_c, spatial_res=spatial_res)
        nearest_dir_dict_sl = brir_export.find_nearest_direction(hrtf_type=hrtf_type ,target_elevation=elev_sl , target_azimuth=azim_sl, spatial_res=spatial_res)
        nearest_dir_dict_sr = brir_export.find_nearest_direction(hrtf_type=hrtf_type ,target_elevation=elev_sr , target_azimuth=azim_sr, spatial_res=spatial_res)
        nearest_dir_dict_rl = brir_export.find_nearest_direction(hrtf_type=hrtf_type ,target_elevation=elev_rl , target_azimuth=azim_rl, spatial_res=spatial_res)
        nearest_dir_dict_rr = brir_export.find_nearest_direction(hrtf_type=hrtf_type ,target_elevation=elev_rr , target_azimuth=azim_rr, spatial_res=spatial_res)
        elev_fl=nearest_dir_dict_fl.get('nearest_elevation')
        elev_fr=nearest_dir_dict_fr.get('nearest_elevation')
        elev_c=nearest_dir_dict_c.get('nearest_elevation')
        elev_sl=nearest_dir_dict_sl.get('nearest_elevation')
        elev_sr=nearest_dir_dict_sr.get('nearest_elevation')
        elev_rl=nearest_dir_dict_rl.get('nearest_elevation')
        elev_rr=nearest_dir_dict_rr.get('nearest_elevation')
        azim_fl=nearest_dir_dict_fl.get('nearest_azimuth')
        azim_fr=nearest_dir_dict_fr.get('nearest_azimuth')
        azim_c=nearest_dir_dict_c.get('nearest_azimuth')
        azim_sl=nearest_dir_dict_sl.get('nearest_azimuth')
        azim_sr=nearest_dir_dict_sr.get('nearest_azimuth')
        azim_rl=nearest_dir_dict_rl.get('nearest_azimuth')
        azim_rr=nearest_dir_dict_rr.get('nearest_azimuth')

        #find file names for desired brirs
        brir_set_formatted = brir_set_folder.replace(" ", "_")
        brirs_path = pjoin(primary_path, CN.PROJECT_FOLDER_BRIRS, brir_set_formatted)
        
        brir_name_wav_fl = 'BRIR' + '_E' + str(elev_fl) + '_A' + str(azim_fl) + '.wav'
        brir_name_wav_fr = 'BRIR' + '_E' + str(elev_fr) + '_A' + str(azim_fr) + '.wav'
        brir_name_wav_c = 'BRIR' + '_E' + str(elev_c) + '_A' + str(azim_c) + '.wav'
        brir_name_wav_sl = 'BRIR' + '_E' + str(elev_sl) + '_A' + str(azim_sl) + '.wav'
        brir_name_wav_sr = 'BRIR' + '_E' + str(elev_sr) + '_A' + str(azim_sr) + '.wav'
        brir_name_wav_rl = 'BRIR' + '_E' + str(elev_rl) + '_A' + str(azim_rl) + '.wav'
        brir_name_wav_rr = 'BRIR' + '_E' + str(elev_rr) + '_A' + str(azim_rr) + '.wav'
  
  
        #find brand name and file name for headphone - search through hpcf folder
        hpcf_name_formatted = headphone.replace(" ", "_")
        sample_name_formatted = sample.replace(" ", "_")
        brand_formatted = brand.replace(" ", "_")
        hpcf_name_wav = hpcf_name_formatted + '_' + sample_name_formatted + '.wav'

            
        #hpcf EAPO config path
        output_config_path = pjoin(primary_path, CN.PROJECT_FOLDER_CONFIGS)
    
        out_file_name = 'ASH_Toolset_Config.txt'
        out_file = pjoin(output_config_path, out_file_name)
        
        #create dir if doesnt exist
        output_file = Path(out_file)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        
        #relative directories
        conv_hpcf_rel_dir = 'Convolution: ..\\HpCFs\\FIRs\\'+brand_formatted+'\\'
        conv_hpcf_command = 'Convolution: ..\\HpCFs\\FIRs\\'+brand_formatted+'\\'+ hpcf_name_wav
        conv_brir_rel_dir = 'Convolution: ..\\BRIRs\\'+brir_set_formatted+'\\'
        
        
        #calculate overall gain to prevent clipping
        output_fr_db = fr_flat
        max_level_db=0
        if enable_hpcf_conv == True:
            #assume -6dB is peak for all hpcfs
            max_level_db=max_level_db-6
        if enable_brir_conv == True:
            #find gain to apply based on HATS
            hrtf_gain=0
            if spatial_res == 3:
                gain_list=CN.HRTF_GAIN_LIST_FULL_RES_NUM
            else:
                gain_list=CN.HRTF_GAIN_LIST_NUM
            hrtf_index = hrtf_type-1
            hrtf_gain = gain_list[hrtf_index]*-1
            #find gain to apply based on acoustic space
            as_gain = get_as_gain_from_dir(brir_set_name)*-1
                    
            max_level_db = max_level_db + hrtf_gain + as_gain
        if enable_hpcf_conv == False and enable_brir_conv == False:
            max_level_db=max_level_db+0
        max_level_db = max_level_db + additional_gain
        gain_overall=gain_overall-max_level_db+gain_oa
        gain_overall = round(gain_overall,1)
        if gain_overall > 50 or gain_overall < -100:
            log_string = 'invalid gains calculated: ' + str(gain_overall)
            if CN.LOG_GUI == 1 and gui_logger != None:
                gui_logger.log_warning(log_string)
        
        #write to txt file
        with open(out_file, 'w') as f:
            #metadata
            f.write('# HpCF and BRIR convolution configuration for Equalizer APO')
            f.write('\n')
            f.write('# Headphone: ' + headphone)
            f.write('\n')
            f.write('# Sample: ' + sample)
            f.write('\n')
            f.write('# BRIR Set: ' + brir_set_name)
            f.write('\n')
            f.write('# Dated: ' + today)
            f.write('\n')
            f.write('\n')
            
            #section to apply gains
            f.write('Channel: ALL')
            f.write('\n')
            f.write('Preamp: '+ str(gain_overall) +' dB')#also adjust gain
            f.write('\n')
            f.write('\n')
            
            #section to apply specific channel gains
            f.write('Channel: L')
            f.write('\n')
            f.write('Preamp: '+ str(gain_fl) +' dB')#also adjust gain
            f.write('\n')
            f.write('Channel: R')
            f.write('\n')
            f.write('Preamp: '+ str(gain_fr) +' dB')#also adjust gain
            f.write('\n')
            if audio_channels == '7.1 Surround' or audio_channels == '5.1 Surround':
                f.write('Channel: C')
                f.write('\n')
                f.write('Preamp: '+ str(gain_c) +' dB')#also adjust gain
                f.write('\n') 
                f.write('Channel: SUB')
                f.write('\n')
                f.write('Preamp: '+ str(gain_c) +' dB')#also adjust gain
                f.write('\n') 
                f.write('Channel: RL')
                f.write('\n')
                f.write('Preamp: '+ str(gain_rl) +' dB')#also adjust gain
                f.write('\n')
                f.write('Channel: RR')
                f.write('\n')
                f.write('Preamp: '+ str(gain_rr) +' dB')#also adjust gain
                f.write('\n')
            if audio_channels == '7.1 Surround':
                f.write('Channel: SL')
                f.write('\n')
                f.write('Preamp: '+ str(gain_sl) +' dB')#also adjust gain
                f.write('\n')
                f.write('Channel: SR')
                f.write('\n')
                f.write('Preamp: '+ str(gain_sr) +' dB')#also adjust gain
                f.write('\n')
            f.write('\n')
            
            #downmixing if applicable
            if audio_channels == '7.1 Downmix to Stereo':
                copy_string = 'Copy: L=1*L+0.707*C+0.707*SUB+0.707*SL+0.707*RL R=1*R+0.707*C+0.707*SUB+0.707*SR+0.707*RR'
                f.write(copy_string)
                f.write('\n')
                f.write('\n')
                
            #section to perform brir convolution
            if enable_brir_conv == True:
                if audio_channels == '2.0 Stereo' or audio_channels == '7.1 Downmix to Stereo':
                    copy_string = 'Copy: L_INPUT_L_EAR=L L_INPUT_R_EAR=L R_INPUT_L_EAR=R R_INPUT_R_EAR=R'
                elif audio_channels == '7.1 Surround':
                    copy_string = 'Copy: L_INPUT_L_EAR=L L_INPUT_R_EAR=L R_INPUT_L_EAR=R R_INPUT_R_EAR=R C_INPUT_L_EAR=C+SUB C_INPUT_R_EAR=C+SUB RL_INPUT_L_EAR=RL RL_INPUT_R_EAR=RL RR_INPUT_L_EAR=RR RR_INPUT_R_EAR=RR SL_INPUT_L_EAR=SL SL_INPUT_R_EAR=SL SR_INPUT_L_EAR=SR SR_INPUT_R_EAR=SR'
                elif audio_channels == '5.1 Surround':
                    copy_string = 'Copy: L_INPUT_L_EAR=L L_INPUT_R_EAR=L R_INPUT_L_EAR=R R_INPUT_R_EAR=R C_INPUT_L_EAR=C+SUB C_INPUT_R_EAR=C+SUB RL_INPUT_L_EAR=RL RL_INPUT_R_EAR=RL RR_INPUT_L_EAR=RR RR_INPUT_R_EAR=RR'
                else:
                    copy_string = 'Copy: L_INPUT_L_EAR=L L_INPUT_R_EAR=L R_INPUT_L_EAR=R R_INPUT_R_EAR=R'
                f.write(copy_string)
                f.write('\n')
                f.write('Channel: L_INPUT_L_EAR L_INPUT_R_EAR')
                f.write('\n')
                f.write(conv_brir_rel_dir + brir_name_wav_fl)
                f.write('\n')
                f.write('Channel: R_INPUT_L_EAR R_INPUT_R_EAR')
                f.write('\n')
                f.write(conv_brir_rel_dir + brir_name_wav_fr)
                f.write('\n')
                if audio_channels == '7.1 Surround' or audio_channels == '5.1 Surround':
                    f.write('Channel: C_INPUT_L_EAR C_INPUT_R_EAR')
                    f.write('\n')
                    f.write(conv_brir_rel_dir + brir_name_wav_c)
                    f.write('\n')              
                    f.write('Channel: RL_INPUT_L_EAR RL_INPUT_R_EAR')
                    f.write('\n')
                    f.write(conv_brir_rel_dir + brir_name_wav_rl)
                    f.write('\n')
                    f.write('Channel: RR_INPUT_L_EAR RR_INPUT_R_EAR')
                    f.write('\n')
                    f.write(conv_brir_rel_dir + brir_name_wav_rr)
                    f.write('\n')
                if audio_channels == '7.1 Surround':
                    f.write('Channel: SL_INPUT_L_EAR SL_INPUT_R_EAR')
                    f.write('\n')
                    f.write(conv_brir_rel_dir + brir_name_wav_sl)
                    f.write('\n')
                    f.write('Channel: SR_INPUT_L_EAR SR_INPUT_R_EAR')
                    f.write('\n')
                    f.write(conv_brir_rel_dir + brir_name_wav_sr)
                    f.write('\n')

                if audio_channels == '2.0 Stereo' or audio_channels == '7.1 Downmix to Stereo':
                    copy_string = 'Copy: L=0*L+L_INPUT_L_EAR+R_INPUT_L_EAR R=0*R+L_INPUT_R_EAR+R_INPUT_R_EAR C=0*C RL=0*RL RR=0*RR SL=0*SL SR=0*SR SUB=0*SUB'
                elif audio_channels == '7.1 Surround':
                    copy_string = 'Copy: L=0*L+L_INPUT_L_EAR+R_INPUT_L_EAR+C_INPUT_L_EAR+RL_INPUT_L_EAR+RR_INPUT_L_EAR+SL_INPUT_L_EAR+SR_INPUT_L_EAR R=0*R+L_INPUT_R_EAR+R_INPUT_R_EAR+C_INPUT_R_EAR+RL_INPUT_R_EAR+RR_INPUT_R_EAR+SL_INPUT_R_EAR+SR_INPUT_R_EAR C=0*C RL=0*RL RR=0*RR SL=0*SL SR=0*SR SUB=0*SUB'
                elif audio_channels == '5.1 Surround':
                    copy_string = 'Copy: L=0*L+L_INPUT_L_EAR+R_INPUT_L_EAR+C_INPUT_L_EAR+RL_INPUT_L_EAR+RR_INPUT_L_EAR R=0*R+L_INPUT_R_EAR+R_INPUT_R_EAR+C_INPUT_R_EAR+RL_INPUT_R_EAR+RR_INPUT_R_EAR C=0*C RL=0*RL RR=0*RR SL=0*SL SR=0*SR SUB=0*SUB'
                f.write(copy_string)
                f.write('\n')
                f.write('\n')
                
            #section to perform hpcf convolution
            if enable_hpcf_conv == True:
                f.write('Channel: L R')
                f.write('\n')
                f.write(conv_hpcf_command)
                f.write('\n')
                f.write('\n')
  
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to write config file: ' + out_file
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_warning(log_string)
            
        return CN.EAPO_ERROR_CODE
            
    return gain_overall
    
    
def include_ash_e_apo_config(primary_path, enabled=False):
    """
    Function loads ASH_Toolset_Config equalizer APO configuration file in config.txt to perform relevant convolution commands.
    :param primary_path: string, base path to save files to
    :return: None
    """    
    status_code=0
    try:
        #hpcf EAPO config path
        output_config_path = pjoin(primary_path, CN.PROJECT_FOLDER_CONFIGS)
        
        #previously generated custom config file
        hesuvi_file_name = 'hesuvi.txt'
        custom_file_name = 'ASH_Toolset_Config.txt'
        custom_file = pjoin(output_config_path, custom_file_name)
        custom_file_rel = CN.PROJECT_FOLDER + '\\E-APO-Configs\\'+custom_file_name
        
        if os.path.isfile(custom_file):
        
            #config.txt file located in e-apo config directory
            e_apo_config_path = pjoin(primary_path, 'config.txt')
            
            if os.path.isfile(e_apo_config_path):
            
                #read from config.txt
                altered_lines = []
                include_custom_exists = 0
                with open(e_apo_config_path, "r", encoding='utf-8') as f:
                    for line in f.readlines():
                        altered_line=line
                        #comment out line if hesuvi config is active and so is ash toolset config
                        if hesuvi_file_name in line and '#' not in line and enabled == True:
                            altered_line = '# '+line
                        #comment out line if a previous format convolution config is active
                        if ('HpCF-Convolution' in line or 'BRIR-Convolution' in line) and '#' not in line and enabled == True:
                            altered_line = '# '+line
                        #if include custom config command already exists in config.txt
                        if custom_file_name in line:
                            include_custom_exists=1
                            #shouldnt be included (comment out)
                            #only if not already commented out
                            if enabled == False and '#' not in line: 
                                altered_line = '# '+line
                            #should be included (remove comment)
                            #only if already commented out
                            elif enabled == True and '#' in line: 
                                altered_line = line.replace("# ", "")
                            #rename config from old version (pre v2.4.0)
                            if CN.PROJECT_FOLDER_SSD in line: 
                                altered_line = altered_line.replace(CN.PROJECT_FOLDER_SSD, CN.PROJECT_FOLDER)
                                
                        altered_lines.append(altered_line)
                        
                        
                #write back to config.txt
                with open(e_apo_config_path, "w") as f:
                    #f.write('\n'.join(altered_lines) + '\n')
                    f.write(''.join(altered_lines))
                    #also write include line if enabled and not already included
                    if include_custom_exists == 0 and enabled == True:
                        if 'EqualizerAPO\\config\\'+ CN.PROJECT_FOLDER +'\\E-APO-Configs' in output_config_path:
                            include_string = 'Include: '+custom_file_rel#use relative directory if project folder is in same folder as config.txt
                        else:
                            include_string = 'Include: '+custom_file#use full path
                        f.write('\n' + include_string )
                    
                    
            else:
                raise ValueError('config.txt not modified. file not found: '+ str(e_apo_config_path))
    
        else:
            raise ValueError('config.txt not modified. file not found: '+ str(custom_file))
    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)   
        return CN.EAPO_ERROR_CODE
    
    return status_code
    
def get_exported_brir_list(primary_path):
    """
    Function reads BRIR folders in output directory and returns list of folders.
    :param primary_path: string, base path to save files to
    :return: list
    """    
    
    try:
        
        try:
            #find previous location (pre v2.4.0) and rename folder if found
            old_proj_path = pjoin(primary_path, CN.PROJECT_FOLDER_SSD)
            new_proj_path = pjoin(primary_path, CN.PROJECT_FOLDER)
            if os.path.exists(old_proj_path):
                #rename folder
                os.rename(old_proj_path, new_proj_path)
        except Exception as ex:
            logging.error("Error occurred", exc_info = ex)  
 
        brirs_path = pjoin(primary_path, CN.PROJECT_FOLDER_BRIRS)

        #list all names in path 
        directory_list = list()
        for root, dirs, files in os.walk(brirs_path, topdown=False):
            for name in dirs:
                if 'dB' in name:
                    extracted_brir_name =  name.replace("_", " ")
                    directory_list.append(extracted_brir_name)
        
        return sorted(directory_list, key=str.casefold)
    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)   
        return []
    
def get_spatial_res_from_dir(primary_path, brir_set):
    """
    Function reads BRIR folders in output directory and returns int containing estimated spatial resolution.
    :param primary_path: string, base path
    :param brir_set: string, name of brir set
    :return: int
    """  
    
    spatial_res=1#default value
    
    try:

        #find file names for desired brirs
        brir_set_formatted = brir_set.replace(" ", "_")
        brirs_path = pjoin(primary_path, CN.PROJECT_FOLDER_BRIRS, brir_set_formatted)

        spatial_res_low=0
        spatial_res_med=0 
        spatial_res_high=0 
        spatial_res_full=0
        
        #find specific directions to flag spatial res
        for root, dirs, files in os.walk(brirs_path):
            for filename in files:
                if '_E-50_' in filename:
                    spatial_res_high=1 
                if '_E-45_' in filename:
                    spatial_res_med=1 
                if '_E-30_' in filename:
                    spatial_res_low=1 
                if '_E-2_' in filename:
                    spatial_res_full=1     
  
        if spatial_res_full == 1:
            spatial_res=3
        elif spatial_res_high == 1:
            spatial_res=2
        elif spatial_res_med == 1:
            spatial_res=1
        elif spatial_res_low == 1:
            spatial_res=0

    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)   
        
    return spatial_res
    
def get_hrtf_type_from_dir(brir_set, spatial_res=1):
    """
    Function reads BRIR folders in output directory and returns int containing hrtf type.
    :param brir_set: string, name of brir set
    :param spatial_res: int, spatial resolution
    :return: int
    """  
    
    hrtf_type=1#default value
    hrtf_index=0
    
    try:

        #find hrtf type from file name
        brir_set_formatted = brir_set.replace(" ", "_")
        if spatial_res == 3:
            hrtf_list=CN.HRTF_LIST_FULL_RES_SHORT
        else:
            hrtf_list=CN.HRTF_LIST_SHORT
        for idx, x in enumerate(hrtf_list):
                hrtf_name_delim = x+'_'
                if hrtf_name_delim in brir_set_formatted:
                    hrtf_type=idx+1
                    hrtf_index=idx
    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)   
    
    return hrtf_type
    
def get_as_gain_from_dir(brir_set):
    """
    Function reads BRIR folders in output directory and returns float containing gain for acoustic space.
    :param brir_set: string, name of brir set
    :param spatial_res: int, spatial resolution
    :return: float containing gain for acoustic space
    """  
    
    gain=0#default value
    
    try:

        #find hrtf type from file name
        brir_set_formatted = brir_set.replace(" ", "_")       
        ac_list=CN.AC_SPACE_LIST_SHORT

        for idx, x in enumerate(ac_list):
                ac_name_delim = '_'+x+'_'
                if ac_name_delim in brir_set_formatted:
                    gain=CN.AC_SPACE_GAINS[idx]
    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)   
    
    return gain
     
    
def get_exported_hp_list(primary_path):
    """
    Function reads HpCF folders in output directory and returns list of headphones.
    :param primary_path: string, base path to save files to
    :return: None
    """    
    #delimiters
    delimiter_a = '_Sample'
    delimiter_b = '_Average'

    hpcf_fir_path = pjoin(primary_path, CN.PROJECT_FOLDER_HPCFS,'FIRs')
    
    hpcf_file_list = []
    hpcf_hp_list = []
    
    try:
    
        for root, dirs, files in os.walk(hpcf_fir_path):
            for filename in files:
                if '.wav' in filename:
                    hpcf_file_list.append(filename)
                    
                    name_before_a = filename.split(delimiter_a)[0]
                    name_before_a_b = name_before_a.split(delimiter_b)[0]
                    extracted_hp_name =  name_before_a_b.replace("_", " ")
                    
                    hpcf_hp_list.append(extracted_hp_name)
        
        return sorted(list(set(hpcf_hp_list)), key=str.casefold)
    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)  
        return []
    
def get_exported_sample_list(headphone, primary_path):
    """
    Function reads HpCF folders in output directory for a specified headphone and returns list of samples.
    :param primary_path: string, base path to save files to
    :return: None
    """    
    
    #hpcf_name_formatted = headphone.replace(" ", "_")
    hpcf_name_formatted = headphone.replace(" ", "_")+'_'
                
    hpcf_fir_path = pjoin(primary_path, CN.PROJECT_FOLDER_HPCFS,'FIRs')
    
    hpcf_file_list = []
    hpcf_sample_list = []
    delimiter_a = '.wav'
    
    try:
    
        for root, dirs, files in os.walk(hpcf_fir_path):
            for filename in files:
                if hpcf_name_formatted in filename:
                    hpcf_file_list.append(filename)
                    
                    name_after_hp = filename.split(hpcf_name_formatted)[1]
                    name_before_a = name_after_hp.split(delimiter_a)[0]
                    extracted_sample_name =  name_before_a.replace("_", " ")
                    
                    if extracted_sample_name.startswith('Sample ') or extracted_sample_name.startswith('sample ') or extracted_sample_name.startswith('Average') or extracted_sample_name.startswith('average'):
                        hpcf_sample_list.append(extracted_sample_name)
        
        return sorted(hpcf_sample_list, key=str.casefold)
    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex) 
        return []
    
def est_peak_gain_from_dir(primary_path, brir_set, hpcf_dict, brir_dict, gain_config=0, channel_config = '2.0 Stereo'):
    """
    Function reads BRIR folders in output directory and returns float containing estimated peak gain.
    :param primary_path: string, base path
    :param brir_set: string, name of brir set
    :return: float containing estimated peak gain
    """  
    n_fft=CN.N_FFT
    peak_gain=0#default value
    impulse=np.zeros(n_fft)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat = hf.mag2db(fr_flat_mag)
    
    try:
        enable_hpcf_conv = hpcf_dict.get('enable_conv')
        enable_brir_conv = brir_dict.get('enable_conv')
        
        spatial_res=0
        mute_fl=brir_dict.get('mute_fl')
        mute_fr=brir_dict.get('mute_fr')
        mute_c=brir_dict.get('mute_c')
        mute_sl=brir_dict.get('mute_sl')
        mute_sr=brir_dict.get('mute_sr')
        mute_rl=brir_dict.get('mute_rl')
        mute_rr=brir_dict.get('mute_rr')
        gain_oa=brir_dict.get('gain_oa')
        gain_fl=brir_dict.get('gain_fl')
        gain_fr=brir_dict.get('gain_fr')
        gain_c=brir_dict.get('gain_c')
        gain_sl=brir_dict.get('gain_sl')
        gain_sr=brir_dict.get('gain_sr')
        gain_rl=brir_dict.get('gain_rl')
        gain_rr=brir_dict.get('gain_rr')
        if mute_fl == True:
            gain_fl=-100
        if mute_fr == True:
            gain_fr=-100
        if mute_c == True:
            gain_c=-100
        if mute_sl == True:
            gain_sl=-100
        if mute_sr == True:
            gain_sr=-100
        if mute_rl == True:
            gain_rl=-100
        if mute_rr == True:
            gain_rr=-100
        gain_fl_mag=hf.db2mag(gain_fl)
        gain_fr_mag=hf.db2mag(gain_fr)
        gain_c_mag=hf.db2mag(gain_c)
        gain_sl_mag=hf.db2mag(gain_sl)
        gain_sr_mag=hf.db2mag(gain_sr)
        gain_rl_mag=hf.db2mag(gain_rl)
        gain_rr_mag=hf.db2mag(gain_rr)
        elev_fl=brir_dict.get('elev_fl')
        elev_fr=brir_dict.get('elev_fr')
        elev_c=brir_dict.get('elev_c')
        elev_sl=brir_dict.get('elev_sl')
        elev_sr=brir_dict.get('elev_sr')
        elev_rl=brir_dict.get('elev_rl')
        elev_rr=brir_dict.get('elev_rr')
        azim_fl=brir_dict.get('azim_fl')
        azim_fr=brir_dict.get('azim_fr')
        azim_c=brir_dict.get('azim_c')
        azim_sl=brir_dict.get('azim_sl')
        azim_sr=brir_dict.get('azim_sr')
        azim_rl=brir_dict.get('azim_rl')
        azim_rr=brir_dict.get('azim_rr')
        #get htrf type from file name
        hrtf_type=get_hrtf_type_from_dir(brir_set, spatial_res=spatial_res)  
        nearest_dir_dict_fl = brir_export.find_nearest_direction(hrtf_type=hrtf_type ,target_elevation=elev_fl , target_azimuth=azim_fl, spatial_res=spatial_res)
        nearest_dir_dict_fr = brir_export.find_nearest_direction(hrtf_type=hrtf_type ,target_elevation=elev_fr , target_azimuth=azim_fr, spatial_res=spatial_res)
        nearest_dir_dict_c = brir_export.find_nearest_direction(hrtf_type=hrtf_type ,target_elevation=elev_c , target_azimuth=azim_c, spatial_res=spatial_res)
        nearest_dir_dict_sl = brir_export.find_nearest_direction(hrtf_type=hrtf_type ,target_elevation=elev_sl , target_azimuth=azim_sl, spatial_res=spatial_res)
        nearest_dir_dict_sr = brir_export.find_nearest_direction(hrtf_type=hrtf_type ,target_elevation=elev_sr , target_azimuth=azim_sr, spatial_res=spatial_res)
        nearest_dir_dict_rl = brir_export.find_nearest_direction(hrtf_type=hrtf_type ,target_elevation=elev_rl , target_azimuth=azim_rl, spatial_res=spatial_res)
        nearest_dir_dict_rr = brir_export.find_nearest_direction(hrtf_type=hrtf_type ,target_elevation=elev_rr , target_azimuth=azim_rr, spatial_res=spatial_res)
        elev_fl=nearest_dir_dict_fl.get('nearest_elevation')
        elev_fr=nearest_dir_dict_fr.get('nearest_elevation')
        elev_c=nearest_dir_dict_c.get('nearest_elevation')
        elev_sl=nearest_dir_dict_sl.get('nearest_elevation')
        elev_sr=nearest_dir_dict_sr.get('nearest_elevation')
        elev_rl=nearest_dir_dict_rl.get('nearest_elevation')
        elev_rr=nearest_dir_dict_rr.get('nearest_elevation')
        azim_fl=nearest_dir_dict_fl.get('nearest_azimuth')
        azim_fr=nearest_dir_dict_fr.get('nearest_azimuth')
        azim_c=nearest_dir_dict_c.get('nearest_azimuth')
        azim_sl=nearest_dir_dict_sl.get('nearest_azimuth')
        azim_sr=nearest_dir_dict_sr.get('nearest_azimuth')
        azim_rl=nearest_dir_dict_rl.get('nearest_azimuth')
        azim_rr=nearest_dir_dict_rr.get('nearest_azimuth')
        brir_name_wav_fl = 'BRIR' + '_E' + str(elev_fl) + '_A' + str(azim_fl) + '.wav'
        brir_name_wav_fr = 'BRIR' + '_E' + str(elev_fr) + '_A' + str(azim_fr) + '.wav'
        brir_name_wav_c = 'BRIR' + '_E' + str(elev_c) + '_A' + str(azim_c) + '.wav'
        brir_name_wav_sl = 'BRIR' + '_E' + str(elev_sl) + '_A' + str(azim_sl) + '.wav'
        brir_name_wav_sr = 'BRIR' + '_E' + str(elev_sr) + '_A' + str(azim_sr) + '.wav'
        brir_name_wav_rl = 'BRIR' + '_E' + str(elev_rl) + '_A' + str(azim_rl) + '.wav'
        brir_name_wav_rr = 'BRIR' + '_E' + str(elev_rr) + '_A' + str(azim_rr) + '.wav'
        
        result_spectrum_l = np.copy(fr_flat)
        result_spectrum_r = np.copy(fr_flat)

        if enable_hpcf_conv == True:
            #get max gain from HpCF
            brand_folder = hpcf_dict.get('brand')
            brand_formatted = brand_folder.replace(" ", "_")
            headphone = hpcf_dict.get('headphone')
            headphone_formatted = headphone.replace(" ", "_")
            sample = hpcf_dict.get('sample')
            sample_formatted = sample.replace(" ", "_")
            out_file_dir_wav = pjoin(primary_path, CN.PROJECT_FOLDER_HPCFS,'FIRs',brand_formatted)
            hpcf_name_wav = headphone_formatted + '_' + sample_formatted + '.wav'
            out_file_path = pjoin(out_file_dir_wav, hpcf_name_wav)
            samplerate, data = wavfile.read(out_file_path)
            fir_array = data / (2.**31)
            data_pad=np.zeros(n_fft)
            data_pad[0:(CN.HPCF_FIR_LENGTH-1)]=fir_array[0:(CN.HPCF_FIR_LENGTH-1)]
            data_fft = np.fft.fft(data_pad)
            output_fr = np.abs(data_fft)
            output_fr_db=hf.mag2db(output_fr)
            result_spectrum_l=np.add(result_spectrum_l,output_fr_db)
            result_spectrum_r=np.add(result_spectrum_r,output_fr_db)
            max_level_hpcf_db = round(np.max(output_fr_db),1)

        if enable_brir_conv == True:
            #get max gain from BRIRs
            #find file names for desired brirs
            brir_set_formatted = brir_set.replace(" ", "_")
            brirs_path = pjoin(primary_path, CN.PROJECT_FOLDER_BRIRS, brir_set_formatted)
            brir_in_arr_l=np.zeros(n_fft)
            brir_in_arr_r=np.zeros(n_fft)
            #find specific directions
            for root, dirs, files in os.walk(brirs_path):
                for filename in files:
                    if channel_config == '2.0 Stereo' or channel_config == '5.1 Surround' or channel_config == '7.1 Surround':
                        if filename == brir_name_wav_fl:
                            #read wav files
                            wav_fname = pjoin(root, filename)
                            samplerate, fir_array = hf.read_wav_file(wav_fname)
                            fir_length = min(len(fir_array),n_fft)
                            #load into separate array for each channel
                            brir_in_arr_l[0:fir_length]=np.add(brir_in_arr_l[0:fir_length],np.multiply(fir_array[0:fir_length,0],gain_fl_mag))
                            brir_in_arr_r[0:fir_length]=np.add(brir_in_arr_r[0:fir_length],np.multiply(fir_array[0:fir_length,1],gain_fl_mag))
                        if filename == brir_name_wav_fr:
                            #read wav files
                            wav_fname = pjoin(root, filename)
                            samplerate, fir_array = hf.read_wav_file(wav_fname)
                            fir_length = min(len(fir_array),n_fft)
                            #load into separate array for each channel
                            brir_in_arr_l[0:fir_length]=np.add(brir_in_arr_l[0:fir_length],np.multiply(fir_array[0:fir_length,0],gain_fr_mag))
                            brir_in_arr_r[0:fir_length]=np.add(brir_in_arr_r[0:fir_length],np.multiply(fir_array[0:fir_length,1],gain_fr_mag))
                    if channel_config == '5.1 Surround' or channel_config == '7.1 Surround':
                        if filename == brir_name_wav_c:#center
                            #read wav files
                            wav_fname = pjoin(root, filename)
                            samplerate, fir_array = hf.read_wav_file(wav_fname)
                            fir_length = min(len(fir_array),n_fft)
                            #load into separate array for each channel
                            brir_in_arr_l[0:fir_length]=np.add(brir_in_arr_l[0:fir_length],np.multiply(fir_array[0:fir_length,0],gain_c_mag))
                            brir_in_arr_r[0:fir_length]=np.add(brir_in_arr_r[0:fir_length],np.multiply(fir_array[0:fir_length,1],gain_c_mag))
                        if filename == brir_name_wav_c:#subwoofer
                            #read wav files
                            wav_fname = pjoin(root, filename)
                            samplerate, fir_array = hf.read_wav_file(wav_fname)
                            fir_length = min(len(fir_array),n_fft)
                            #load into separate array for each channel
                            brir_in_arr_l[0:fir_length]=np.add(brir_in_arr_l[0:fir_length],np.multiply(fir_array[0:fir_length,0],gain_c_mag))
                            brir_in_arr_r[0:fir_length]=np.add(brir_in_arr_r[0:fir_length],np.multiply(fir_array[0:fir_length,1],gain_c_mag))
                        if filename == brir_name_wav_rl:
                            #read wav files
                            wav_fname = pjoin(root, filename)
                            samplerate, fir_array = hf.read_wav_file(wav_fname)
                            fir_length = min(len(fir_array),n_fft)
                            #load into separate array for each channel
                            brir_in_arr_l[0:fir_length]=np.add(brir_in_arr_l[0:fir_length],np.multiply(fir_array[0:fir_length,0],gain_rl_mag))
                            brir_in_arr_r[0:fir_length]=np.add(brir_in_arr_r[0:fir_length],np.multiply(fir_array[0:fir_length,1],gain_rl_mag))
                        if filename == brir_name_wav_rr:
                            #read wav files
                            wav_fname = pjoin(root, filename)
                            samplerate, fir_array = hf.read_wav_file(wav_fname)
                            fir_length = min(len(fir_array),n_fft)
                            #load into separate array for each channel
                            brir_in_arr_l[0:fir_length]=np.add(brir_in_arr_l[0:fir_length],np.multiply(fir_array[0:fir_length,0],gain_rr_mag))
                            brir_in_arr_r[0:fir_length]=np.add(brir_in_arr_r[0:fir_length],np.multiply(fir_array[0:fir_length,1],gain_rr_mag))
                    if channel_config == '7.1 Surround':
                        if filename == brir_name_wav_sl:
                            #read wav files
                            wav_fname = pjoin(root, filename)
                            samplerate, fir_array = hf.read_wav_file(wav_fname)
                            fir_length = min(len(fir_array),n_fft)
                            #load into separate array for each channel
                            brir_in_arr_l[0:fir_length]=np.add(brir_in_arr_l[0:fir_length],np.multiply(fir_array[0:fir_length,0],gain_sl_mag))
                            brir_in_arr_r[0:fir_length]=np.add(brir_in_arr_r[0:fir_length],np.multiply(fir_array[0:fir_length,1],gain_sl_mag))
                        if filename == brir_name_wav_sr:
                            #read wav files
                            wav_fname = pjoin(root, filename)
                            samplerate, fir_array = hf.read_wav_file(wav_fname)
                            fir_length = min(len(fir_array),n_fft)
                            #load into separate array for each channel
                            brir_in_arr_l[0:fir_length]=np.add(brir_in_arr_l[0:fir_length],np.multiply(fir_array[0:fir_length,0],gain_sr_mag))
                            brir_in_arr_r[0:fir_length]=np.add(brir_in_arr_r[0:fir_length],np.multiply(fir_array[0:fir_length,1],gain_sr_mag))             
            #calculate peak gain from BRIRs
            #need to take into account: overall gain from config, individual channel gains, hpcf gain, & brir gains
            data_fft = np.fft.fft(brir_in_arr_l[0:CN.N_FFT])
            mag_fft=np.abs(data_fft)
            db_fft = hf.mag2db(mag_fft)
            result_spectrum_l=np.add(result_spectrum_l,db_fft)
            
            data_fft = np.fft.fft(brir_in_arr_r[0:CN.N_FFT])
            mag_fft=np.abs(data_fft)
            db_fft = hf.mag2db(mag_fft)
            result_spectrum_r=np.add(result_spectrum_r,db_fft)
            
        max_db_l = np.max(result_spectrum_l)
        max_db_r = np.max(result_spectrum_r)
        
        #finalise gain
        peak_gain = peak_gain+np.mean([max_db_l,max_db_r])
        if enable_brir_conv == True or enable_hpcf_conv == True:
            peak_gain = peak_gain+gain_config
        peak_gain = round(peak_gain,1)
        
        

    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)   
        
    return peak_gain