# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 23:20:14 2023

@author: Shanon
"""


# import packages
from os.path import join as pjoin
from pathlib import Path
import time
import logging
from datetime import date
from ash_toolset import constants as CN

today = str(date.today())

logger = logging.getLogger(__name__)
log_info=1





def write_e_apo_configs_brirs(brir_name, primary_path, hrtf_type):
    """
    Function creates equalizer APO configuration files for specified brir name
    :param hrtf_type: int, selected HRTF type: 1 = KU100, 2 = kemar L pinna, 3 = kemar N pinna, 4 = B&K 4128 HATS, 5 = DADEC, 6 = HEAD acoustics HMSII.2, 7 = G.R.A.S.KEMAR (new), 8 = Bruel & Kjaer Type 4128C (BKwHA)
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
                if hrtf_type == 4:
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
                    f.write('Preamp: -14.5 dB')#also adjust gain
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