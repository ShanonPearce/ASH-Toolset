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

logger = logging.getLogger(__name__)
log_info=1

# get the start time
st = time.time()


def export_brir(brir_arr, hrtf_type, target_rt60, brir_name, primary_path, brir_dir_export=1, brir_ts_export=1, hesuvi_export=1, report_progress=0, gui_logger=None, direct_gain_db=CN.DIRECT_GAIN_MAX, samp_freq=44100, bit_depth='PCM_24'):
    """
    Function to export a customised BRIR to WAV files
    :param brir_arr: numpy array, containing set of BRIRs. 4d array. d1 = elevations, d2 = azimuths, d3 = channels, d4 = samples
    :param hrtf_type: int, selected HRTF type: 1 = KU100, 2 = kemar L pinna, 3 = kemar N pinna, 4 = B&K 4128 HATS, 5 = DADEC, 6 = HEAD acoustics HMSII.2, 7 = G.R.A.S.KEMAR (new), 8 = Bruel & Kjaer Type 4128C (BKwHA)
    :param target_rt60: int, value in ms for target reverberation time
    :param brir_name: string, name of brir
    :param primary_path: string, base path to save files to
    :param brir_dir_export: int, 1 = export directional wav brirs
    :param brir_ts_export: int, 1 = export true stereo wav brirs
    :param hesuvi_export: int, 1 = export hesuvi compatible wav brirs
    :param direct_gain_db: float, adjust gain of direct sound in dB
    :param report_progress: int, 1 = update progress to progress bar in gui, set to 0 if no gui
    :param gui_logger: gui logger object for dearpygui
    :return: None
    """ 
    
    
    # get the start time
    st = time.time()

    try:
    
        #hesuvi path
        if 'EqualizerAPO' in primary_path:
            hesuvi_path = pjoin(primary_path,'HeSuVi')#pjoin(primary_path, 'config','HeSuVi')      
        else:
            hesuvi_path = pjoin(primary_path, CN.PROJECT_FOLDER,'HeSuVi')  #pjoin(primary_path, CN.PROJECT_FOLDER,'config','HeSuVi')  
    
        #larger reverb times will need additional samples
        if target_rt60 <=800:
            out_wav_samples_44 = 44100
        else:
            out_wav_samples_44 = 55125
        out_wav_samples_48 = round(out_wav_samples_44 * float(48000) / 44100) 
            
        #gain adjustment
        max_amp = np.max(np.abs(brir_arr))
        
        #reduce gain if direct gain db is less than max value
        reduction_gain_db = (CN.DIRECT_GAIN_MAX-direct_gain_db)*-1/2
        reduction_gain = hf.db2mag(reduction_gain_db)
        
        #
        ## write set of WAVs
        #
   
        total_elev_brir = len(brir_arr)
        total_azim_brir = len(brir_arr[0])
        total_chan_brir = len(brir_arr[0][0])
        total_samples_brir = len(brir_arr[0][0][0])
        
        out_wav_array=np.zeros((out_wav_samples_44,2))
 
        #output directory - wav files
        brir_folder = brir_name
        out_file_dir_wav = pjoin(primary_path, CN.PROJECT_FOLDER_BRIRS,brir_folder)
 
        if brir_dir_export == 1:
            
            direction_matrix = generate_direction_matrix(hrtf_type)
            
            #write stereo wav for each elev and az
            for elev in range(total_elev_brir):
                elev_deg = int(CN.MIN_ELEV + elev*CN.NEAREST_ELEV)
                for azim in range(total_azim_brir):
                    azim_deg = int(azim*CN.NEAREST_AZ_HRIR)
                    azim_deg_wav = int(0-azim_deg) if azim_deg < 180 else int(360-azim_deg)
                    if direction_matrix[elev][azim][0][0] == 1:  
                        
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
        ## write set of HESUVI WAVs
        #
        
        #output directory - HESUVI wav files
        out_file_dir_wav_44 = pjoin(hesuvi_path,'hrir','44')
        out_file_dir_wav_48 = pjoin(hesuvi_path,'hrir')
        
            
        brir_out_44_he=np.zeros((CN.NUM_OUT_CHANNELS_HE,out_wav_samples_44))
        brir_out_48_he=np.zeros((CN.NUM_OUT_CHANNELS_HE,out_wav_samples_48))
        brir_out_44_ts=np.zeros((CN.NUM_OUT_CHANNELS_TS,out_wav_samples_44))
        
        selected_elev=0
        elev_id=int((selected_elev-CN.MIN_ELEV)/CN.NEAREST_ELEV)
    
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
            
            dezired_azim_id = int(dezired_az/CN.NEAREST_AZ_HRIR)    
                
            
            #load into zero pad array
            data_pad=np.zeros((65536,2))
            data_pad[0:(out_wav_samples_44),0]=np.copy(brir_arr[elev_id][dezired_azim_id][0][0:out_wav_samples_44])/max_amp#L
            data_pad[0:(out_wav_samples_44),1]=np.copy(brir_arr[elev_id][dezired_azim_id][1][0:out_wav_samples_44])/max_amp#R
  
            #create a copy and resample to 48kHz
            data_pad_48k=np.zeros((65536,2))
            
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
            gui_logger.log_info(log_string)
            
    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    if CN.LOG_INFO == 1:
        logging.info('Execution time:' + str(elapsed_time) + ' seconds')
        
        
        
        

def generate_direction_matrix(hrtf_type):
    """
    Function returns a numpy array containing a matrix of elevations and azimuths marked for export
    """
    
    #create numpy array to return
    direction_matrix=np.zeros((CN.OUTPUT_ELEVS,CN.OUTPUT_AZIMS,1,1))  
    
    try:
        
        #limited elevation range for HRTF type 4
        if hrtf_type == 4:
            elev_out_range = {-30,0,30}
        else:
            elev_out_range = {-45,-30,-15,0,15,30,45}  
        
        azim_horiz_range = {20,25,35,40,340,335,325,320}
        
         
        

        #write stereo wav for each elev and az
        for elev in range(CN.OUTPUT_ELEVS):
            elev_deg = int(CN.MIN_ELEV + elev*CN.NEAREST_ELEV)
            for azim in range(CN.OUTPUT_AZIMS):
                azim_deg = int(azim*CN.NEAREST_AZ_HRIR)
                if (elev_deg in elev_out_range and azim_deg%CN.NEAREST_AZ_WAV == 0) or (hrtf_type != 4 and (elev_deg == 0 or elev_deg == -15 or elev_deg == 15) and (azim_deg in azim_horiz_range)):  
                    
                    #populate matrix with 1 if direction applicable
                    direction_matrix[elev][azim][0][0] = 1
                    

        
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        
    return direction_matrix