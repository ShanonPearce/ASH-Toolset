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
from datetime import date
from datetime import datetime

logger = logging.getLogger(__name__)
log_info=1

# get the start time
st = time.time()


def export_brir(brir_arr, acoustic_space, hrtf_type, brir_name, primary_path, brir_dir_export=1, brir_ts_export=1, hesuvi_export=1, gui_logger=None, direct_gain_db=CN.DIRECT_GAIN_MAX, samp_freq=44100, bit_depth='PCM_24', spatial_res=1, sofa_export=0):
    """
    Function to export a customised BRIR to WAV files
    :param brir_arr: numpy array, containing set of BRIRs. 4d array. d1 = elevations, d2 = azimuths, d3 = channels, d4 = samples
    :param hrtf_type: int, selected HRTF type starts from 1
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
    :return: None
    """ 
    
    
    # get the start time
    st = time.time()

    try:
        
        log_string = 'Preparing BRIRs for export'
        if CN.LOG_INFO == 1:
            logging.info(log_string)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
    
        total_elev_brir = len(brir_arr)
        total_azim_brir = len(brir_arr[0])
        total_chan_brir = len(brir_arr[0][0])
        total_samples_brir = len(brir_arr[0][0][0])    
    
        #hesuvi path
        if 'EqualizerAPO' in primary_path:
            hesuvi_path = pjoin(primary_path,'HeSuVi')#stored outside of project folder (within hesuvi installation)
        else:
            hesuvi_path = pjoin(primary_path, CN.PROJECT_FOLDER,'HeSuVi')#stored within project folder
    
        #larger reverb times will need additional samples
        ac_space_int = CN.AC_SPACE_LIST_SRC.index(acoustic_space)
        est_rt60 = CN.AC_SPACE_EST_R60[ac_space_int]
        
        if est_rt60 <=400:
            out_wav_samples_44 = 33075
        elif est_rt60 <750:
            out_wav_samples_44 = 44100
        elif est_rt60 <1000:
            out_wav_samples_44 = 55125
        elif est_rt60 <=1250:
            out_wav_samples_44 = 63945    
        elif est_rt60 <=1500:
            out_wav_samples_44 = 99225 
        else:
            out_wav_samples_44 = 127890
        if out_wav_samples_44 > total_samples_brir:
            out_wav_samples_44 = max(total_samples_brir-1000,4410)  
            
        out_wav_samples_48 = round(out_wav_samples_44 * float(48000) / 44100) 
            
        #gain adjustment
        max_amp = np.max(np.abs(brir_arr))
        
        #reduce gain if direct gain db is less than max value
        reduction_gain_db = (CN.DIRECT_GAIN_MAX-direct_gain_db)*-1/2
        reduction_gain = hf.db2mag(reduction_gain_db)
        
        #
        ## write set of WAVs
        #
   
        out_wav_array=np.zeros((out_wav_samples_44,2))
 
        #output directory - wav files
        brir_folder = brir_name
        out_file_dir_wav = pjoin(primary_path, CN.PROJECT_FOLDER_BRIRS,brir_folder)
 
        if brir_dir_export == 1:
            
            direction_matrix_out = generate_direction_matrix(hrtf_type, spatial_res=spatial_res, output_variant=2)
            
                    
            if spatial_res >= 0 and spatial_res < CN.NUM_SPATIAL_RES:
                elev_min=CN.SPATIAL_RES_ELEV_MIN[spatial_res] 
                elev_max=CN.SPATIAL_RES_ELEV_MAX[spatial_res] 
                elev_nearest=CN.SPATIAL_RES_ELEV_NEAREST[spatial_res] #as per hrir dataset
                elev_nearest_process=CN.SPATIAL_RES_ELEV_NEAREST_PR[spatial_res] 
                azim_nearest=CN.SPATIAL_RES_AZIM_NEAREST[spatial_res] 
                azim_nearest_process=CN.SPATIAL_RES_AZIM_NEAREST_PR[spatial_res] 
            else:
                raise ValueError('Invalid spatial resolution')
        
            #print(str(elev_min))
            #print(str(elev_nearest))
            
            
            
            #write stereo wav for each elev and az
            for elev in range(total_elev_brir):
                elev_deg = int(elev_min + elev*elev_nearest)
                for azim in range(total_azim_brir):
                    azim_deg = int(azim*azim_nearest)
                    azim_deg_wav = int(0-azim_deg) if azim_deg < 180 else int(360-azim_deg)
                    if direction_matrix_out[elev][azim][0][0] == 1:  
                        
                        out_wav_array=np.zeros((out_wav_samples_44,2))
                        #grab BRIR
                        out_wav_array[:,0] = np.copy(brir_arr[elev][azim][0][0:out_wav_samples_44])*reduction_gain/max_amp#L
                        out_wav_array[:,1] = np.copy(brir_arr[elev][azim][1][0:out_wav_samples_44])*reduction_gain/max_amp#R
                        
                        #write wav
                        out_file_name = 'BRIR' + '_E' + str(elev_deg) + '_A' + str(azim_deg_wav) + '.wav'
                        
                        out_file_path = pjoin(out_file_dir_wav,out_file_name)
                        
                        #create dir if doesnt exist
                        output_file = Path(out_file_path)
                        output_file.parent.mkdir(exist_ok=True, parents=True)
                        
                        #resample if samp_freq is not 44100
                        if samp_freq != 44100:
                            out_wav_array = hf.resample_signal(out_wav_array, new_rate = samp_freq)
                        
                        hf.write2wav(file_name=out_file_path, data=out_wav_array, bit_depth=bit_depth, samplerate=samp_freq)
    
            log_string = 'BRIR WAV set saved to: ' + str(out_file_dir_wav)
            if CN.LOG_INFO == 1:
                logging.info(log_string)
            if CN.LOG_GUI == 1 and gui_logger != None:
                gui_logger.log_info(log_string)
 
        #
        ## write SOFA file
        #
         
        if sofa_export == 1:
            export_sofa_brir(primary_path=primary_path,brir_arr=brir_arr, brir_set_name=brir_name, est_rt60=est_rt60, spatial_res=spatial_res, samp_freq=samp_freq, gui_logger=gui_logger)
    
        #
        ## write set of HESUVI WAVs
        #
        
        if brir_ts_export == 1 or hesuvi_export == 1:
        
            if spatial_res >= 0 and spatial_res < CN.NUM_SPATIAL_RES:
                elev_min=CN.SPATIAL_RES_ELEV_MIN[spatial_res] 
                elev_nearest=CN.SPATIAL_RES_ELEV_NEAREST[spatial_res] #as per hrir dataset
                azim_nearest=CN.SPATIAL_RES_AZIM_NEAREST[spatial_res] 
            else:
                raise ValueError('Invalid spatial resolution')
            
            #output directory - HESUVI wav files
            out_file_dir_wav_44 = pjoin(hesuvi_path,'hrir','44')
            out_file_dir_wav_48 = pjoin(hesuvi_path,'hrir')
            
                
            brir_out_44_he=np.zeros((CN.NUM_OUT_CHANNELS_HE,out_wav_samples_44))
            brir_out_48_he=np.zeros((CN.NUM_OUT_CHANNELS_HE,out_wav_samples_48))
            brir_out_44_ts=np.zeros((CN.NUM_OUT_CHANNELS_TS,out_wav_samples_44))
            
            selected_elev=0
            elev_id=int((selected_elev-elev_min)/elev_nearest)
        
            #load each input azimuth
            for n in range(CN.NUM_SOURCE_AZIM):
                dezired_az=0
                #change azimuth depending on source azim ID
                if n == 0 :
                    dezired_az=0
                elif n == 1:
                    dezired_az=30
                elif n == 2:
                    dezired_az=330
                elif n == 3:
                    dezired_az=90
                elif n == 4:
                    dezired_az=270
                elif n == 5:
                    dezired_az=135
                elif n == 6:
                    dezired_az=225
                
                dezired_azim_id = int(dezired_az/azim_nearest)    

                #load into zero pad array
                data_pad=np.zeros((out_wav_samples_44,2))
                data_pad[0:(out_wav_samples_44),0]=np.copy(brir_arr[elev_id][dezired_azim_id][0][0:out_wav_samples_44])/max_amp#L
                data_pad[0:(out_wav_samples_44),1]=np.copy(brir_arr[elev_id][dezired_azim_id][1][0:out_wav_samples_44])/max_amp#R
      
                #create a copy and resample to 48kHz
                data_pad_48k=np.zeros((out_wav_samples_48,2))           
                data_pad_48k = hf.resample_signal(data_pad)
      
                #place each channel into output array as per HeSuVi channel mapping
                if n == 0 :#C
                    brir_out_44_he[6,:]=data_pad[0:out_wav_samples_44,0]
                    brir_out_44_he[13,:]=data_pad[0:out_wav_samples_44,1]
                    brir_out_48_he[6,:]=data_pad_48k[0:out_wav_samples_48,0]
                    brir_out_48_he[13,:]=data_pad_48k[0:out_wav_samples_48,1]
                elif n == 1:#FL
                    brir_out_44_he[0,:]=data_pad[0:out_wav_samples_44,0]
                    brir_out_44_he[1,:]=data_pad[0:out_wav_samples_44,1]
                    brir_out_48_he[0,:]=data_pad_48k[0:out_wav_samples_48,0]
                    brir_out_48_he[1,:]=data_pad_48k[0:out_wav_samples_48,1]
                elif n == 2:#FR
                    brir_out_44_he[8,:]=data_pad[0:out_wav_samples_44,0]
                    brir_out_44_he[7,:]=data_pad[0:out_wav_samples_44,1]
                    brir_out_48_he[8,:]=data_pad_48k[0:out_wav_samples_48,0]
                    brir_out_48_he[7,:]=data_pad_48k[0:out_wav_samples_48,1]
                elif n == 3:#SL
                    brir_out_44_he[2,:]=data_pad[0:out_wav_samples_44,0]
                    brir_out_44_he[3,:]=data_pad[0:out_wav_samples_44,1]
                    brir_out_48_he[2,:]=data_pad_48k[0:out_wav_samples_48,0]
                    brir_out_48_he[3,:]=data_pad_48k[0:out_wav_samples_48,1]
                elif n == 4:#SR
                    brir_out_44_he[10,:]=data_pad[0:out_wav_samples_44,0]
                    brir_out_44_he[9,:]=data_pad[0:out_wav_samples_44,1]
                    brir_out_48_he[10,:]=data_pad_48k[0:out_wav_samples_48,0]
                    brir_out_48_he[9,:]=data_pad_48k[0:out_wav_samples_48,1]
                elif n == 5:#BL
                    brir_out_44_he[4,:]=data_pad[0:out_wav_samples_44,0]
                    brir_out_44_he[5,:]=data_pad[0:out_wav_samples_44,1]
                    brir_out_48_he[4,:]=data_pad_48k[0:out_wav_samples_48,0]
                    brir_out_48_he[5,:]=data_pad_48k[0:out_wav_samples_48,1]
                elif n == 6:#BR
                    brir_out_44_he[12,:]=data_pad[0:out_wav_samples_44,0]
                    brir_out_44_he[11,:]=data_pad[0:out_wav_samples_44,1]
                    brir_out_48_he[12,:]=data_pad_48k[0:out_wav_samples_48,0]
                    brir_out_48_he[11,:]=data_pad_48k[0:out_wav_samples_48,1]
     
                
                #place each channel into output array as per true stereo channel mapping
                if n == 1:
                    brir_out_44_ts[0,:]=data_pad[0:out_wav_samples_44,0]
                    brir_out_44_ts[1,:]=data_pad[0:out_wav_samples_44,1]
                elif n == 2:
                    brir_out_44_ts[2,:]=data_pad[0:out_wav_samples_44,0]
                    brir_out_44_ts[3,:]=data_pad[0:out_wav_samples_44,1]
            
            
        if brir_ts_export == 1:
            #
            #true stereo
            #
            output_wav_ts = brir_out_44_ts
            output_wav_ts = output_wav_ts.transpose()
            out_file_name = 'BRIR_True_Stereo.wav'
            
            out_file_path = pjoin(out_file_dir_wav,out_file_name)
            
            #create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            #resample if samp_freq is not 44100
            if samp_freq != 44100:
                output_wav_ts = hf.resample_signal(output_wav_ts, new_rate = samp_freq)
            
            hf.write2wav(file_name=out_file_path, data=output_wav_ts, bit_depth=bit_depth, samplerate=samp_freq)
        
            log_string = 'BRIR WAV True stereo saved to: ' + str(out_file_dir_wav)
            if CN.LOG_INFO == 1:
                logging.info(log_string)
            if CN.LOG_GUI == 1 and gui_logger != None:
                gui_logger.log_info(log_string)
        
        if hesuvi_export == 1:
            
            #
            #hesuvi 44khz
            #
            output_wav_he_44 = brir_out_44_he
            output_wav_he_44 = output_wav_he_44.transpose()
            out_file_name = '_'+brir_name + '.wav'
            
            out_file_path = pjoin(out_file_dir_wav_44, out_file_name)
            
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
            if CN.LOG_INFO == 1:
                logging.info(log_string)
            if CN.LOG_GUI == 1 and gui_logger != None:
                gui_logger.log_info(log_string)
        
 
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to export BRIRs'
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_error(log_string)
            
    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    if CN.LOG_INFO == 1:
        logging.info('Execution time:' + str(elapsed_time) + ' seconds')
        
        
        
        

def generate_direction_matrix(hrtf_type, spatial_res=1, output_variant=1):
    """
    Function returns a numpy array containing a matrix of elevations and azimuths marked for export (output_variant) or processing
    :param hrtf_type: int, select HRTF type, starts from 1
    :param spatial_res: int, spatial resolution, 0= low (-30 to 30 deg elev, nearest 15 deg elev, 5 deg azim) 1 = moderate (-45 to 45 deg elev, nearest 15 deg elev, 5 deg azim), 2 = high (-50 to 50 deg elev, nearest 5 deg elev, 5 deg azim), 3 = full (-50 to 50 deg elev, nearest 2 deg elev, 2 deg azim)
    :param output_variant: int, 1 = reduced set of directions intended for reducing post processing, 2 = reduced set of directions intended for wav export only, 0 = no reduction
    """

    try:
 
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
        
        #create numpy array to return
        direction_matrix=np.zeros((output_elevs,output_azims,1,1))  
        
        azim_horiz_range = CN.AZIM_HORIZ_RANGE

        #write stereo wav for each elev and az
        for elev in range(output_elevs):
            elev_deg = int(elev_min + elev*elev_nearest)
            for azim in range(output_azims):
                azim_deg = int(azim*azim_nearest)
                
                #reduced set of directions for post processing or WAV output
                if output_variant >= 1:
                    #limited elevation range for HRTF type 4
                    if hrtf_type == 4 and spatial_res < 3:
                        if (elev_deg >= -30 and elev_deg <= 30 and elev_deg%15 == 0 and azim_deg%15 == 0):  
                            #populate matrix with 1 if direction applicable
                            direction_matrix[elev][azim][0][0] = 1
                    elif spatial_res == 0: 
                        if (elev_deg >= elev_min_out and elev_deg <= elev_max_out) and elev_deg%elev_nearest_process == 0 and (azim_deg%CN.NEAREST_AZ_WAV == 0 or (elev_deg >= -30 and elev_deg <= 30 and azim_deg in azim_horiz_range)):  
                            #populate matrix with 1 if direction applicable
                            direction_matrix[elev][azim][0][0] = 1
                    elif spatial_res == 1: 
                        if (elev_deg >= elev_min_out and elev_deg <= elev_max_out) and elev_deg%elev_nearest_process == 0 and (azim_deg%CN.NEAREST_AZ_WAV == 0 or (elev_deg >= -30 and elev_deg <= 30 and azim_deg in azim_horiz_range)):  
                            #populate matrix with 1 if direction applicable
                            direction_matrix[elev][azim][0][0] = 1
                    elif spatial_res == 2 or spatial_res == 3: 
                        if output_variant == 2:#wav export variant
                            if (elev_deg >= elev_min_out and elev_deg <= elev_max_out) and elev_deg%elev_nearest_process == 0 and azim_deg%azim_nearest_process == 0:  
                                #populate matrix with 1 if direction applicable
                                direction_matrix[elev][azim][0][0] = 1
                        else:
                            if elev_deg%elev_nearest_process == 0 and azim_deg%azim_nearest_process == 0:  
                                #populate matrix with 1 if direction applicable
                                direction_matrix[elev][azim][0][0] = 1
                    else:
                        if (elev_deg%elev_nearest_process == 0 and azim_deg%azim_nearest_process == 0):
                            #populate matrix with 1 if direction applicable
                            direction_matrix[elev][azim][0][0] = 1
 
                else:
                    if (elev_deg%elev_nearest_process == 0 and azim_deg%azim_nearest_process == 0):
                        #populate matrix with 1 if direction applicable
                        direction_matrix[elev][azim][0][0] = 1
                        
        
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        
    return direction_matrix


def find_nearest_direction(hrtf_type, target_elevation, target_azimuth, spatial_res=1):
    """
    Function returns a dict containing nearest available azimuth and elevation angle for a specified hrtf and azimuth and elevation
    Used to determine elevations and azimuths available to read from wav file dataset
    """
    
    try:
        
        target_elevation = int(target_elevation)
        target_azimuth = int(target_azimuth)
        
        
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
                
                if hrtf_type == 4 and spatial_res < 3:
                    if (elev_deg >= -30 and elev_deg <= 30 and elev_deg%15 == 0 and azim_deg%15 == 0):  
                        valid_dir = 1 
                elif spatial_res == 0: 
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
            if CN.LOG_INFO == 1:
                logging.info(log_string_a)
            if CN.LOG_GUI == 1 and gui_logger != None:
                gui_logger.log_info(log_string_a)
                
        if os.path.exists(output_config_path):
            shutil.rmtree(output_config_path)
            log_string_b = 'Deleted folder and contents: ' + output_config_path
            if CN.LOG_INFO == 1:
                logging.info(log_string_b)
            if CN.LOG_GUI == 1 and gui_logger != None:
                gui_logger.log_info(log_string_b)
    
        if os.path.exists(output_hesuvi_path):
            shutil.rmtree(output_hesuvi_path)
            log_string_c = 'Deleted folder and contents: ' + output_hesuvi_path
            if CN.LOG_INFO == 1:
                logging.info(log_string_c)
            if CN.LOG_GUI == 1 and gui_logger != None:
                gui_logger.log_info(log_string_c)
    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to delete folders: ' + out_file_dir_wav + ' & ' + output_config_path
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_error(log_string)
            
            
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
                    if CN.LOG_INFO == 1:
                        logging.info(log_string_a)
                    if CN.LOG_GUI == 1 and gui_logger != None:
                        gui_logger.log_info(log_string_a)
            
 
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to delete folder: ' + out_file_dir_brirs
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_error(log_string)
            
            
def export_sofa_brir(primary_path, brir_arr, brir_set_name, spatial_res, est_rt60, samp_freq, gui_logger=None):
    """
    Function to export a customised BRIR to SOFA file
    :param brir_arr: numpy array, containing set of BRIRs. 4d array. d1 = elevations, d2 = azimuths, d3 = channels, d4 = samples
    :param brir_name: string, name of brir
    :param spatial_res: int, spatial resolution, 0= low (-30 to 30 deg elev, nearest 15 deg elev, 5 deg azim) 1 = moderate (-45 to 45 deg elev, nearest 15 deg elev, 5 deg azim), 2 = high (-50 to 50 deg elev, nearest 5 deg elev, 5 deg azim), 3 = full (-50 to 50 deg elev, nearest 2 deg elev, 2 deg azim)
    :param samp_freq: int, sample frequency in Hz
    :param est_rt60: int, value in ms for target reverberation time
    """
    
    now_datetime = datetime.now()
    
    try:
        
        if spatial_res >= 0 and spatial_res < CN.NUM_SPATIAL_RES:
            elev_min=CN.SPATIAL_RES_ELEV_MIN[spatial_res] 
            elev_max=CN.SPATIAL_RES_ELEV_MAX[spatial_res] 
            elev_nearest=CN.SPATIAL_RES_ELEV_NEAREST[spatial_res] #as per hrir dataset
            azim_nearest=CN.SPATIAL_RES_AZIM_NEAREST[spatial_res] 
            
            
        else:
            raise ValueError('Invalid spatial resolution')
        
        total_elev_brir = len(brir_arr)
        total_azim_brir = len(brir_arr[0])
        total_chan_brir = len(brir_arr[0][0])
        total_samples_brir = len(brir_arr[0][0][0])
        
        #calculate size of dataset based on output elevations and azimuths
        total_directions = total_elev_brir*total_azim_brir
        
        #set number of output samples assuming base rate is 44.1kHz
        if est_rt60 <=300:
            output_samples = 22050
        elif est_rt60 <=400:
            output_samples = 33075
        elif est_rt60 <750:
            output_samples = 44100
        elif est_rt60 <1000:
            output_samples = 55125
        elif est_rt60 <=1250:
            output_samples = 63945    
        elif est_rt60 <=1500:
            output_samples = 99225
        else:
            output_samples = 127890
        if output_samples > total_samples_brir:
            output_samples = max(total_samples_brir-1000,4410)  
            
        #recalculate based on selected sample rate
        output_samples_re = round(output_samples * float(samp_freq) / 44100) 
        
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
                    if samp_freq != 44100:
                        pos_brir_array = hf.resample_signal(pos_brir_array, new_rate = samp_freq)
                    
                    #populate data IR array
                    data_ir_array[source_pos_id,0,0:output_samples_re] = pos_brir_array[0:output_samples_re,0]#L
                    data_ir_array[source_pos_id,1,0:output_samples_re] = pos_brir_array[0:output_samples_re,1]#R
                    
 
                    #increment id
                    source_pos_id = source_pos_id+1
                    
        
        
        
        with open(CN.METADATA_FILE) as fp:
            _info = json.load(fp)
        __version__ = _info['version']
        
        # Create SOFAFile object with the latest SimpleFreeFieldHRIR convention
        sofa = SOFAFile("GeneralFIR", sofaConventionsVersion=1.0, version=1.0)
        
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
        sofa.GLOBAL_ApplicationVersion = __version__
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
        
        # Needs dimensions IC or MC
        #sofa.ListenerUp = np.atleast_2d(np.array([0, 0, 1]))
        
        # Needs dimensions IC or MC
        #sofa.ListenerView = np.atleast_2d(np.array([1, 0, 0]))
        
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
        out_file_dir_sofa = pjoin(primary_path, CN.PROJECT_FOLDER_BRIRS_SOFA,brir_set_name)
        
        #create parent dir if doesnt exist
        output_file = Path(out_file_dir_sofa)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        #out_file_sofa = pjoin(out_file_dir_sofa,brir_set_name)
        
        # Save file upon completion
        #sofa.export(brir_set_name)
        sofa.export(output_file)
   
        log_string_a = 'Exported SOFA file: ' + brir_set_name 
        if CN.LOG_INFO == 1:
            logging.info(log_string_a)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string_a)
  
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to export SOFA file: ' + brir_set_name 
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_error(log_string)
    
