# -*- coding: utf-8 -*-
"""
BRIR generation routine of ASH-Toolset.

Created on Sun Apr 28 11:25:06 2024
@author: Shanon Pearce
License: (see /LICENSE)
"""


# import packages
import numpy as np
from os.path import join as pjoin
import mat73
import time
import logging
from ash_toolset import helper_functions as hf
from ash_toolset import constants as CN
from ash_toolset import brir_export
from ash_toolset import hrir_processing
import scipy as sp
from scipy.io import wavfile
import os
from pathlib import Path
import concurrent.futures
from csv import DictReader
from ash_toolset import pyquadfilter

logger = logging.getLogger(__name__)




def generate_integrated_brir(brir_name,  spatial_res=1, report_progress=0, gui_logger=None, brir_dict={}):   
    """
    Function to generate customised BRIR from below parameters

    :param hrtf_type: int, select HRTF type: starts from 1
    :param direct_gain_db: float, adjust gain of direct sound in dB
    :param room_target: int, room target id
    :param pinna_comp: int, 0 = equalised for in ear headphones (with additional eq), 1 = in ear headphones (without additional eq), 2 = over/on ear headphones (with additional eq), 3 = over/on ear headphones (without additional eq)
    :param early_refl_rise_ms: float, ms length of early reflection rise window, default = 0 ms (instant rise)
    :param reduce_reverb: bool, True = enable reverb reduction, False = no reverb reduction
    :param spatial_res: int, spatial resolution, 0= low, 1 = moderate, 2 = high, 3 = full
    :param report_progress: int, 1 = update progress to progress bar (1st tab) in gui, 2 = 2nd tab, set to 0 if not using the gui
    :param gui_logger: gui logger object for dearpygui
    Returns:
        tuple: (np.ndarray, int, str)
            - NumPy array: numpy array containing set of BRIRs. 4d array. d1 = elevations, d2 = azimuths, d3 = channels, d4 = samples
            - Status code: 0 = Success, 1 = Failure, 2 = Cancelled
    """ 

    # get the start time
    st = time.time()
    brir_out= np.array([])
    status=1
    #num_workers = os.cpu_count()  #4 Or os.cpu_count() or any other value
  
    try:
        
        #get relevant information from dict
        if brir_dict:
            if brir_name == CN.FOLDER_BRIRS_LIVE: 
                brir_hrtf_type=brir_dict.get('qc_brir_hrtf_type')
                brir_hrtf_dataset=brir_dict.get('qc_brir_hrtf_dataset')
                brir_hrtf = brir_dict.get('qc_brir_hrtf')
                brir_hrtf_short=brir_dict.get('qc_brir_hrtf_short')
                room_target_name = brir_dict.get("qc_room_target")
                direct_gain_db = brir_dict.get("qc_direct_gain_db")
                acoustic_space= brir_dict.get("qc_ac_space_src")
                pinna_comp = brir_dict.get("qc_pinna_comp")
                crossover_f=brir_dict.get('qc_crossover_f')
                sub_response=brir_dict.get('qc_sub_response')
                hp_rolloff_comp=brir_dict.get('qc_hp_rolloff_comp')
                fb_filtering=brir_dict.get('qc_fb_filtering')
            else:
                brir_hrtf_type=brir_dict.get('brir_hrtf_type')
                brir_hrtf_dataset=brir_dict.get('brir_hrtf_dataset')
                brir_hrtf = brir_dict.get('brir_hrtf')
                brir_hrtf_short=brir_dict.get('brir_hrtf_short')
                room_target_name = brir_dict.get("room_target")
                direct_gain_db = brir_dict.get("direct_gain_db")
                acoustic_space= brir_dict.get("ac_space_src")
                pinna_comp = brir_dict.get("pinna_comp")
                crossover_f=brir_dict.get('crossover_f')
                sub_response=brir_dict.get('sub_response')
                hp_rolloff_comp=brir_dict.get('hp_rolloff_comp')
                fb_filtering=brir_dict.get('fb_filtering')
            
            hrtf_symmetry = brir_dict.get("hrtf_symmetry")
            early_refl_delay_ms = brir_dict.get("er_delay_time")
        else:
            raise ValueError('brir_dict not populated')
        
        #exit if stop thread flag is true
        stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
        if stop_thread == True:
            brir_out= np.array([])
            status=2#2=cancelled
            return brir_out,status
  
        #limit IR size
        target_rt60=CN.RT60_MAX_L
        if acoustic_space in CN.AC_SPACE_LIST_LOWRT60:
            n_fft=CN.N_FFT
            target_rt60=CN.RT60_MAX_S
        else:
            n_fft=CN.N_FFT_L      

        #variable crossover depending on acoustic space
        f_crossover_var=crossover_f
       # _,order_var= get_ac_f_crossover(name_src=acoustic_space, gui_logger=gui_logger)#f_crossover_var,order_var
        
        reverb_data=CN.reverb_data
        order_var = CN.extract_column(data=reverb_data, column='order_crossover', condition_key='name_src', condition_value=acoustic_space, return_all_matches=False)
        fade_start = CN.extract_column(data=reverb_data, column='fade_start', condition_key='name_src', condition_value=acoustic_space, return_all_matches=False)
        est_rt60 = CN.extract_column(data=reverb_data, column='est_rt60', condition_key='name_src', condition_value=acoustic_space, return_all_matches=False)
        as_folder = CN.extract_column(data=reverb_data, column='folder', condition_key='name_src', condition_value=acoustic_space, return_all_matches=False)
        as_file_name = CN.extract_column(data=reverb_data, column='file_name', condition_key='name_src', condition_value=acoustic_space, return_all_matches=False)
        
        #filters
        if f_crossover_var < CN.FILTFILT_THRESH_F:
            fb_filtering=True#force true if below threshold due to strong delays
        lp_sos = hf.get_filter_sos(cutoff=f_crossover_var, fs=CN.FS, order=order_var, filtfilt=fb_filtering, b_type='low')
        hp_sos = hf.get_filter_sos(cutoff=f_crossover_var, fs=CN.FS, order=order_var, filtfilt=fb_filtering, b_type='high')

        # ac_space_int = CN.AC_SPACE_LIST_SRC.index(acoustic_space)
        # fade_start = CN.AC_SPACE_FADE_START[ac_space_int]
        # est_rt60 = CN.AC_SPACE_EST_R60[ac_space_int]

        #impulse
        impulse=np.zeros(CN.N_FFT)
        impulse[0]=1
        fr_flat_mag = np.abs(np.fft.rfft(impulse))
        fr_flat = hf.mag2db(fr_flat_mag)

        #windows
        data_pad_zeros=np.zeros(n_fft)
        data_pad_ones=np.ones(n_fft)

        #initial rise window for sub
        initial_hanning_size=60
        initial_hanning_start=0#190
        hann_initial_full=np.hanning(initial_hanning_size)
        hann_initial = np.split(hann_initial_full,2)[0]
        initial_removal_win_sub = data_pad_zeros.copy()
        initial_removal_win_sub[initial_hanning_start:initial_hanning_start+int(initial_hanning_size/2)] = hann_initial
        initial_removal_win_sub[initial_hanning_start+int(initial_hanning_size/2):]=data_pad_ones[initial_hanning_start+int(initial_hanning_size/2):]
        #initial rise window for reverb
        initial_hanning_size=100
        initial_hanning_start=70
        hann_initial_full=np.hanning(initial_hanning_size)
        hann_initial = np.split(hann_initial_full,2)[0]
        initial_removal_win_rev = data_pad_zeros.copy()
        initial_removal_win_rev[initial_hanning_start:initial_hanning_start+int(initial_hanning_size/2)] = hann_initial
        initial_removal_win_rev[initial_hanning_start+int(initial_hanning_size/2):]=data_pad_ones[initial_hanning_start+int(initial_hanning_size/2):]

        #direction matrices
        
        #get matrix of desired directions to process
        direction_matrix_process = brir_export.generate_direction_matrix(spatial_res=spatial_res, variant=0)
        #get matrix of desired output directions
        #regular case
        direction_matrix_post = brir_export.generate_direction_matrix(spatial_res=spatial_res, variant=1)


        log_string = 'Loading input filters'
        hf.log_with_timestamp(log_string, gui_logger)
        if report_progress > 0:
            progress = 1/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)

        #
        # load room target filter (FIR)
  

        room_target_fir = CN.ROOM_TARGETS_DICT[room_target_name]["impulse_response"]
            
        #
        # load pinna comp filter (FIR)
        #


        npy_fname = pjoin(CN.DATA_DIR_INT, 'headphone_ear_comp_dataset.npy')
        ear_comp_fir_dataset = hf.load_convert_npy_to_float64(npy_fname)
        pinna_comp_fir = ear_comp_fir_dataset[pinna_comp,:]
        
        #
        # load additional headphone eq
        #

        
        data_lf_comp_eq = np.copy(impulse[0:32])
        if hp_rolloff_comp == True:
            filename = 'low_frequency_roll-off_compensation.wav'
            wav_fname = pjoin(CN.DATA_DIR_INT, filename)
            samplerate, data_lf_comp_eq = hf.read_wav_file(wav_fname)


        log_string = 'Loading reverberation data'
        hf.log_with_timestamp(log_string, gui_logger)
        if report_progress > 0:
            progress = 2/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)
        
        #
        # load sub bass BRIR (FIR)
        #

        sub_data=CN.sub_data
        sub_file_name = CN.extract_column(data=sub_data, column='file_name', condition_key='name_gui', condition_value=sub_response, return_all_matches=False)
        sub_folder = CN.extract_column(data=sub_data, column='folder', condition_key='name_gui', condition_value=sub_response, return_all_matches=False)
        #file_name = get_sub_f_name(sub_response=sub_response, gui_logger=gui_logger)
        if sub_folder == 'sub' or sub_folder == 'lf_brir':#default sub responses
            npy_fname = pjoin(CN.DATA_DIR_SUB, sub_file_name+'.npy')
        else:#user sub response
            file_folder = pjoin(CN.DATA_DIR_AS_USER,sub_response)
            npy_fname = pjoin(file_folder, sub_file_name+'.npy')
        sub_brir_npy = hf.load_convert_npy_to_float64(npy_fname)
        sub_brir_ir = np.zeros((2,n_fft))
        # Use only available samples in each channel (up to CN.N_FFT)
        available_samples = min(CN.N_FFT, sub_brir_npy.shape[-1])
        sub_brir_ir[0, :available_samples] = sub_brir_npy[0, :available_samples]
        sub_brir_ir[1, :available_samples] = sub_brir_npy[1, :available_samples]
        
        

        #
        # Load BRIR reverberation data
        #
        if as_folder == 'user':
            brir_rev_folder = pjoin(CN.DATA_DIR_AS_USER,acoustic_space)
        else:
            brir_rev_folder = pjoin(CN.DATA_DIR_INT, 'reverberation', as_folder)
        #npy_file_name =  'reverberation_dataset_' +acoustic_space+'.npy'
        npy_file_name =  as_file_name+'.npy'
        npy_file_path = pjoin(brir_rev_folder,npy_file_name) 
        try:
            brir_reverberation = hf.load_convert_npy_to_float64(npy_file_path)
        except Exception as e:
            log_string=f"Error loading reverberation data: {e}"
            hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=e)#log error
            raise ValueError('Reverberation data not loaded')
            
        #total_elev_reverb = len(brir_reverberation)
        total_azim_reverb = len(brir_reverberation[0])
        #total_chan_reverb = len(brir_reverberation[0][0])
        #total_samples_reverb = len(brir_reverberation[0][0][0])
        nearest_azim_reverb = int(360/total_azim_reverb)
        
        #zero pad reverberation array due to variable length
        brir_reverberation=hf.zero_pad_last_dimension(data=brir_reverberation, n_fft=n_fft)
        
        
        log_string = 'Loading HRTF data'
        hf.log_with_timestamp(log_string, gui_logger)
        if report_progress > 0:
            progress = 5/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)

        #
        # load HRIRs
        #
        

        
          
        # print(brir_hrtf_type)
        # print(brir_hrtf_dataset)
        # print(brir_hrtf)
        # print(brir_hrtf_short)
        
        if brir_hrtf_type == 'Human Listener':
            hrir_dir_base = CN.DATA_DIR_HRIR_NPY_HL
        elif brir_hrtf_type == 'Dummy Head / Head & Torso Simulator':
            hrir_dir_base = CN.DATA_DIR_HRIR_NPY_DH
        else:
            hrir_dir_base = CN.DATA_DIR_HRIR_NPY_USER#user sofa npy set
        
        if spatial_res >= 0 and spatial_res < CN.NUM_SPATIAL_RES:
            if spatial_res <= 2:
                sub_directory = 'h'
            elif spatial_res == 3:
                sub_directory = 'm'
                status_subroutine = check_download_max_hrtf_datasets(download=True, gui_logger=gui_logger, report_progress=report_progress)
                if status_subroutine == 1:#failed
                    raise ValueError('HRIR processing failed due to error downloading HRTF dataset')
            #join spatial res subdirectory
            hrir_dir = pjoin(hrir_dir_base, sub_directory)
            # join dataset subdirectory
            if brir_hrtf_type != 'User SOFA Input':
                hrir_dir = pjoin(hrir_dir, brir_hrtf_dataset)
            # full filename
            npy_fname = pjoin(hrir_dir, f"{brir_hrtf_short}.npy")
            # load specified hrir dataset
                
            elev_min=CN.SPATIAL_RES_ELEV_MIN[spatial_res] 
            elev_max=CN.SPATIAL_RES_ELEV_MAX[spatial_res] 
            elev_nearest=CN.SPATIAL_RES_ELEV_NEAREST[spatial_res] #as per hrir dataset
            elev_nearest_process=CN.SPATIAL_RES_ELEV_NEAREST_PR[spatial_res] 
            azim_nearest=CN.SPATIAL_RES_AZIM_NEAREST[spatial_res] 
            azim_nearest_process=CN.SPATIAL_RES_AZIM_NEAREST_PR[spatial_res] 
        else:
            raise ValueError('Invalid spatial resolution. No hrir dataset loaded.')

        #first try to load npy dataset
        #if failed, proceed with SOFA workflow
        try:
            #load npy files
            hrir_list = hf.load_convert_npy_to_float64(npy_fname)
        except:
            log_string = 'local HRIR dataset not found. Proceeding with SOFA workflow'
            hf.log_with_timestamp(log_string, gui_logger) 
            if report_progress > 0:
                hf.update_gui_progress(report_progress=report_progress, message=log_string)
        
            #run sofa workflow
            status_subroutine = hrir_processing.sofa_workflow_new_dataset(brir_hrtf_type=brir_hrtf_type, brir_hrtf_dataset=brir_hrtf_dataset, brir_hrtf=brir_hrtf, brir_hrtf_short=brir_hrtf_short, gui_logger=gui_logger, report_progress=report_progress)
            if status_subroutine == 0:#success
                #finally load the npy file
                hrir_list = hf.load_convert_npy_to_float64(npy_fname)
            elif status_subroutine == 2:#cancelled
                brir_out= np.array([])
                status=2#2=cancelled
                return brir_out,status
            else:
                raise ValueError('HRIR processing failed due to error in SOFA workflow')
        
        
        hrir_selected = hrir_list[0]#first dimension is unitary
        total_elev_hrir = len(hrir_selected)
        total_azim_hrir = len(hrir_selected[0])
        total_chan_hrir = len(hrir_selected[0][0])
        total_samples_hrir = len(hrir_selected[0][0][0])
        base_elev_idx = total_elev_hrir//2
    
  
        #output_azims = int(360/azim_nearest)#should be identical to above lengths
        #output_elevs = int((elev_max-elev_min)/elev_nearest +1)#should be identical to above lengths
        
        #BRIR_out array will be populated BRIRs for all directions
        brir_out=np.zeros((total_elev_hrir,total_azim_hrir,2,n_fft))   

  
   
        log_string = 'Adjusting levels'
        hf.log_with_timestamp(log_string, gui_logger)
        if report_progress > 0:
            progress = 20/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)
            

      
        #adjust levels of HRIRs for DRR scaling. Assume 0dB starting point
        #round to nearest 2 decimal places
        direct_gain_db = round(direct_gain_db,2)
        #limit gain
        if direct_gain_db > CN.DIRECT_GAIN_MAX:
            direct_gain_db = CN.DIRECT_GAIN_MAX
        elif direct_gain_db < CN.DIRECT_GAIN_MIN:
            direct_gain_db = CN.DIRECT_GAIN_MIN   
        direct_gain = hf.db2mag(direct_gain_db)
        hrir_selected *= direct_gain#Use Vectorized NumPy Operations
        
        #flip polarity if flagged
        flip_polarity = hrir_processing.get_polarity(listener_type=brir_hrtf_type, dataset_name=brir_hrtf_dataset, name_gui=brir_hrtf, gui_logger=gui_logger)
        if flip_polarity == 'yes':
            hrir_selected *= -1
     
        #apply left right symmetry if enabled
        if hrtf_symmetry != CN.HRTF_SYM_LIST[0]:
            for elev in range(total_elev_hrir):
                for azim in range(total_azim_hrir):
                    if ('Left' in hrtf_symmetry and azim <= int((total_azim_hrir)/2)) or ('Right' in hrtf_symmetry and (azim >= int((total_azim_hrir)/2) or azim==0)):
                        #get id of mirrored azimuth
                        azim_mirror=total_azim_hrir-azim
                        if azim_mirror==total_azim_hrir:
                            azim_mirror=0
                        for chan in range(total_chan_hrir):
                            #get id of mirrored channel
                            chan_mirror = total_chan_hrir-1-chan
                            #replace hrir with mirrored hrir
                            hrir_selected[elev][azim][chan][:] = np.copy(hrir_selected[elev][azim_mirror][chan_mirror][:])
        
        
        
        
        
        log_string = 'Preparing reverberation data'
        hf.log_with_timestamp(log_string, gui_logger) 
        if report_progress > 0:
            progress = 30/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)
            
        
        #exit if stop thread flag is true
        stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
        if stop_thread == True:
            brir_out= np.array([])
            status=2#2=cancelled
            return brir_out,status
        
        #
        ## Integration of BRIRs with HRIRs through TD interpolation and DRR scaling
        #
 
        #
        ## Reverberation shaping
        #
        
        #get start id for noise reduction fade out
        rt60_comp_factor=0.95
        
        n_fade_win_start = int(((fade_start*rt60_comp_factor)/1000)*CN.FS)
        if fade_start > 0:
            noise_fade = True
        else:
            noise_fade = False

        #get start id for reverberation snapshot for reverb reduction
        #convert to sample ID to get start of reverb response
        rt60_snap_start = int(((target_rt60)/1000)*CN.FS)
 
        #generate hann windows for reverb EQ
        l_fade_win_size=np.abs(rt60_snap_start-n_fft)*2
        n_fade_win_size=np.abs(4000)*2
        if CN.WINDOW_TYPE == 1:
            wind_l_fade_full=np.hanning(l_fade_win_size)
            wind_n_fade_full=np.hanning(n_fade_win_size)
        elif CN.WINDOW_TYPE == 2:
            wind_l_fade_full=np.bartlett(l_fade_win_size)
            wind_n_fade_full=np.bartlett(n_fade_win_size)
        elif CN.WINDOW_TYPE == 3: 
            wind_l_fade_full=np.blackman(l_fade_win_size)
            wind_n_fade_full=np.blackman(n_fade_win_size)
        elif CN.WINDOW_TYPE == 4:
            wind_l_fade_full=np.hamming(l_fade_win_size)
            wind_n_fade_full=np.hamming(n_fade_win_size)
        else:
            wind_l_fade_full=np.bartlett(l_fade_win_size)
            wind_n_fade_full=np.bartlett(n_fade_win_size)
        
        win_l_fade_out = np.split(wind_l_fade_full,2)[1]
        win_n_fade_out = np.split(wind_n_fade_full,2)[1]
        
        if noise_fade == True:
            #additional window to fade out noise
            n_fade_out_win = data_pad_zeros.copy()
            n_fade_out_win[0:n_fade_win_start] = data_pad_ones[0:n_fade_win_start]
            n_fade_out_win[n_fade_win_start:n_fade_win_start+int(n_fade_win_size/2)] = win_n_fade_out
        else:
            #additional window to fade out tail end of late reflections
            n_fade_out_win = data_pad_zeros.copy()
            n_fade_out_win[0:rt60_snap_start] = data_pad_ones[0:rt60_snap_start]
            n_fade_out_win[rt60_snap_start:] = win_l_fade_out
        
        
        #for each azim
        for azim in range(total_azim_reverb):
            #apply fade out for noise reduction
            brir_reverberation[0][azim][0][:] = np.multiply(brir_reverberation[0][azim][0][:],n_fade_out_win)
            brir_reverberation[0][azim][1][:] = np.multiply(brir_reverberation[0][azim][1][:],n_fade_out_win)

            #shift reflections by specified delay
            if early_refl_delay_ms > 0:
                reverb_delay_samples = int((early_refl_delay_ms/1000)*CN.FS)
                brir_reverberation[0][azim][0][:] = np.roll(brir_reverberation[0][azim][0][:],reverb_delay_samples)
                brir_reverberation[0][azim][1][:] = np.roll(brir_reverberation[0][azim][1][:],reverb_delay_samples)
                #also zero out response before delay
                brir_reverberation[0][azim][0][0:reverb_delay_samples] = np.multiply(brir_reverberation[0][azim][0][0:reverb_delay_samples],0)
                brir_reverberation[0][azim][1][0:reverb_delay_samples] = np.multiply(brir_reverberation[0][azim][1][0:reverb_delay_samples],0)
            
                
        #
        # TD alignment of BRIRs with HRIRs
        #
        
        #grab a sample hrir and brir used to td alignment
        # elev_ind = int((0-elev_min)/elev_nearest)
        # azim_ind = int(30/azim_nearest)
        # hrir_sample = np.copy(hrir_selected[elev_ind][azim_ind][0][:])
        # hrir_sample=hf.zero_pad_last_dimension(data=hrir_sample, n_fft=n_fft)#zero pad
        # reverb_brir_align(brir_reverberation, n_fft, f_crossover_var, hrir_sample, initial_removal_win_rev, total_azim_reverb, order_var)




        log_string = 'Integrating HRIRs and reverberation'
        hf.log_with_timestamp(log_string, gui_logger) 
        if report_progress > 0:
            progress = 40/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)

        #
        # grab BRIRs from interim matrix and place in output matrix
        #
        for elev in range(total_elev_hrir):
            for azim in range(total_azim_hrir):
                
                azim_deg = int(azim*azim_nearest)
                #case for multiple directions on horizontal plane, every x deg azimuth
                if total_azim_reverb > 7:
                    #map hrir azimuth to appropriate brir azimuth
                    #round azim to nearest X deg and get new ID
                    brir_azim = hf.round_to_multiple(azim_deg,nearest_azim_reverb)#brir_reverberation is nearest 5 deg
                    if brir_azim >= 360:
                        brir_azim = 0
                    brir_azim_ind = int(brir_azim/nearest_azim_reverb)#get index
                #case for minimal set (7 directions)
                elif total_azim_reverb == 7:
                    if azim_deg < 15 or azim_deg > 345:
                        brir_azim_ind=0
                    elif azim_deg < 60:
                        brir_azim_ind=1
                    elif azim_deg < 120:
                        brir_azim_ind=2
                    elif azim_deg < 180: 
                        brir_azim_ind=3
                    elif azim_deg <= 240:
                        brir_azim_ind=4
                    elif azim_deg <= 300:
                        brir_azim_ind=5
                    else:
                        brir_azim_ind=6
                #case for minimal set (5 directions)
                elif total_azim_reverb == 5:
                    if azim_deg < 15 or azim_deg > 345:
                        brir_azim_ind=0
                    elif azim_deg < 120:
                        brir_azim_ind=1
                    elif azim_deg < 180: 
                        brir_azim_ind=2
                    elif azim_deg <= 240:
                        brir_azim_ind=3
                    else:
                        brir_azim_ind=4
                else:
                    raise ValueError('Unable to process BRIR reverberation data. Invalid number of directions: ' + str(total_azim_reverb) )
                
                for chan in range(CN.TOTAL_CHAN_BRIR):
                    brir_out[elev][azim][chan][0:n_fft] = np.copy(brir_reverberation[0][brir_azim_ind][chan][0:n_fft])
   
    
   
    
        #
        #add HRIR into output BRIR array
        #use multiple threads to integrate HRIRs into output BRIR array
        # Use ThreadPoolExecutor to integrate HRIRs into output BRIR array
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(combine_hrirs_brirs, brir_out, elev, n_fft, total_azim_hrir, hrir_selected, total_samples_hrir) for elev in range(total_elev_hrir)]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Optionally, handle any exceptions here
        
        #clear out hrir array since no longer used
        hrir_selected = None
        brir_reverberation = None
        
   
 
 
        log_string = 'Processing low frequencies'
        hf.log_with_timestamp(log_string, gui_logger)
    
        if report_progress > 0:
            progress = 50/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)
        
        
        #exit if stop thread flag is true
        stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
        if stop_thread == True:
            brir_out= np.array([])
            status=2#2=cancelled
            return brir_out,status
        
        #low freq reference ranges
        mag_range_a=int(30*(n_fft/CN.N_FFT))
        mag_range_a2=int(80*(n_fft/CN.N_FFT))#50
        mag_range_b=int(290*(n_fft/CN.N_FFT))
        
        #grab a sample brir used to td alignment
        elev_ind = int((0-elev_min)/elev_nearest)
        azim_ind = int(30/azim_nearest)
        brir_sample = np.copy(brir_out[elev_ind][azim_ind][0][:])
        
        #set level of HF BRIR to 0 at low freqs
        data_fft_b = np.fft.rfft(brir_sample)
        mag_fft_b=np.abs(data_fft_b)
        average_mag_b = np.mean(mag_fft_b[mag_range_a2:mag_range_b])
        if average_mag_b == 0:
            if CN.LOG_INFO == True:
                logging.info('0 magnitude detected')
        for elev in range(total_elev_hrir):
            for azim in range(total_azim_hrir):
                for chan in range(total_chan_hrir):      
                    #only apply gain if direction is in direction matrix
                    if direction_matrix_process[elev][azim][0][0] == 1: 
                        brir_out[elev][azim][chan][:] = np.divide(brir_out[elev][azim][chan][:],average_mag_b)
        
        
        
        #
        # Sub BRIR integration
        #
        
  
        if CN.ENABLE_SUB_INTEGRATION == True:
        
            sub_brir_align(sub_brir_ir, n_fft, f_crossover_var, brir_sample, initial_removal_win_sub, order_var, fb_filtering)
            
            if report_progress > 0:
                progress = 60/100
                hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)
            
            #
            #apply low pass filter to sub BRIR
            #
            brir_eq_a_l = np.copy(sub_brir_ir[0][:])
            brir_eq_a_r = np.copy(sub_brir_ir[1][:])
            # #apply lp filter
            # brir_eq_b_l = hf.signal_lowpass_filter(brir_eq_a_l, f_crossover_var, CN.FS, order_var)
            # brir_eq_b_r = hf.signal_lowpass_filter(brir_eq_a_r, f_crossover_var, CN.FS, order_var)
            #apply lp filter
            brir_eq_b_l = hf.apply_sos_filter(brir_eq_a_l, lp_sos, filtfilt=fb_filtering)
            brir_eq_b_r = hf.apply_sos_filter(brir_eq_a_r, lp_sos, filtfilt=fb_filtering)
            
            
            sub_brir_ir[0][:] = brir_eq_b_l[0:n_fft] 
            sub_brir_ir[1][:] = brir_eq_b_r[0:n_fft]
            
            #apply fade out win if enabled
            sub_brir_ir[0][:] = np.multiply(sub_brir_ir[0][:],n_fade_out_win)
            sub_brir_ir[1][:] = np.multiply(sub_brir_ir[1][:],n_fade_out_win)
            
       

            
            #use multiple threads to integrate sub brir into output array
            # Use ThreadPoolExecutor to integrate sub brir
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(integrate_sub_brirs, brir_out, elev, n_fft, total_azim_hrir, direction_matrix_process,
                                         sub_brir_ir, hp_sos, fb_filtering)
                           for elev in range(total_elev_hrir)]
                for future in concurrent.futures.as_completed(futures):
                    future.result()  # Optionally, handle exceptions

        
    

        log_string = 'Applying EQ'
        hf.log_with_timestamp(log_string, gui_logger)
        if report_progress > 0:
            progress = 70/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)
         
            
        #exit if stop thread flag is true
        stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
        if stop_thread == True:
            brir_out= np.array([])
            status=2#2=cancelled
            return brir_out,status
            


        #
        ## EQ correction
        #
        

        #use multiple threads to calculate EQ
        # Use ThreadPoolExecutor to calculate EQ
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results_list = [0] * total_elev_hrir  # Initialize results list, contains a zero element for each elevation to start with
            futures = [executor.submit(calc_eq_for_brirs, brir_out, elev, total_azim_hrir, direction_matrix_process, fr_flat, results_list, elev) for elev in range(total_elev_hrir)]
    
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Optionally, handle exceptions

       
           
        #results_list will be populated with numpy arrays representing db response for each elevation
        num_results_avg = 0
        brir_fft_avg_db = fr_flat.copy()
        for result in results_list:
            if np.sum(np.abs(result)) > 0:#some results might be flat 0 db
                num_results_avg = num_results_avg+1
                brir_fft_avg_db = np.add(brir_fft_avg_db,result)
        #divide by total number of elevations
        brir_fft_avg_db = brir_fft_avg_db/num_results_avg
 
        #convert to mag
        brir_fft_avg_mag = hf.db2mag(brir_fft_avg_db)
        
        #level ends of spectrum
        brir_fft_avg_mag_sm = hf.level_spectrum_ends(brir_fft_avg_mag, 20, 18500, smooth_win = 5, n_fft=CN.N_FFT)#40, 19000, smooth_win = 7
        
 
        #octave smoothing
        brir_fft_avg_mag_sm = hf.smooth_fft_octaves(data=brir_fft_avg_mag_sm, n_fft=CN.N_FFT, win_size_base = 5, fund_freq=170)#win_size_base = 10, fund_freq=140 win_size_base = 7, fund_freq=160
        
        
        
        
        #invert response
        brir_fft_avg_mag_inv = hf.db2mag(hf.mag2db(brir_fft_avg_mag_sm)*-1)
        #create min phase FIR
        #brir_df_inv_fir = hf.mag_to_min_fir(brir_fft_avg_mag_inv, crop=1, out_win_size=2048, n_fft=CN.N_FFT)
        brir_df_inv_fir = hf.build_min_phase_filter(smoothed_mag=brir_fft_avg_mag_inv,  truncate_len=2048, n_fft=CN.N_FFT)#new method
        
        if report_progress > 0:
            progress = 80/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)
        
        #exit if stop thread flag is true
        stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
        if stop_thread == True:
            brir_out= np.array([])
            status=2#2=cancelled
            return brir_out,status


        
        #section to prepare to crop BRIR output
        #larger reverb times will need additional samples
       #estimate output array length based on RT60
        if est_rt60 <=400:#determines the most samples to keep based on estimated rt60
            out_samples_est = 33075
        elif est_rt60 <750:
            out_samples_est = 44100
        elif est_rt60 <1000:
            out_samples_est = 55125
        elif est_rt60 <=1250:
            out_samples_est = 63945    
        elif est_rt60 <=1500:
            out_samples_est = 99225 
        else:
            out_samples_est = 127890
        if out_samples_est > n_fft:
            out_samples_est = max(n_fft-1000,4410)      
        #attempt to trim array. Calculate point where amplitude falls below threshold, later discards remaining samples
        brir_sample = np.abs(np.copy(brir_out[elev_ind][azim_ind][0][:]))
        norm_factor = np.max(brir_sample)
        ref_array = np.divide(brir_sample,norm_factor*2)
        crop_samples = hf.get_crop_index(ref_array, tail_ignore=10000)
        
        #hf.plot_td(ref_array[10000:])
        #print(crop_samples)
        #print(out_samples_est)
        
        if crop_samples < 4000 or crop_samples > out_samples_est:
            crop_samples=out_samples_est
        #get end index for convolutions
        
        #print(crop_samples)
        # brir_sample = brir_out[0][0][0][:]
        # crop_samples = hf.get_crop_index(brir_sample)
        # #print(crop_samples)
        
        #use multiple threads to perform EQ
        #Use ThreadPoolExecutor to perform EQ
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(apply_eq_to_brirs, brir_out, elev, n_fft, total_azim_hrir, direction_matrix_post,
                                    brir_df_inv_fir, room_target_fir,  pinna_comp_fir, data_lf_comp_eq, crop_samples)
                       for elev in range(total_elev_hrir)]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Optionally, handle exceptions


        if CN.PLOT_ENABLE == True:
            hf.plot_data(brir_fft_avg_mag,'brir_fft_avg_mag')
            hf.plot_data(brir_fft_avg_mag_sm,'brir_fft_avg_mag_sm')
            hf.plot_data(brir_fft_avg_mag_inv,'brir_fft_avg_mag_inv')

        #exit if stop thread flag is true
        stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
        if stop_thread == True:
            brir_out= np.array([])
            status=2#2=cancelled
            return brir_out,status


        log_string = 'Finalising dataset'
        status=0#0=success
        hf.log_with_timestamp(log_string, gui_logger)
        if report_progress > 0:
            progress = 90/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)
            
        
        #crop before returning
        brir_out=hf.crop_array_last_dimension(brir_out, crop_samples)
        # Convert the data type to float32 and replace the original array, no longer need higher precision
        if brir_out.dtype == np.float64:
            brir_out = brir_out.astype(np.float32)
            
            
            

    except Exception as ex:
        status=1#1=failed
        log_string = 'Failed to generate BRIRs'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
  

        
    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    log_string = 'Execution time:' + str(elapsed_time) + ' seconds'
    hf.log_with_timestamp(log_string)
        
    #return the integrated BRIR
    return brir_out,status


def sub_brir_align(sub_brir_ir, n_fft, f_crossover_var, brir_sample, initial_removal_win_sub, order_var, fb_filtering):
    """
    Function performs time domain alignment of sub brir array
    :param sub_brir_ir: numpy array containing sub brirs
    :return: None
    """ 
    #create low pass filter
    lp_sos = hf.get_filter_sos(cutoff=f_crossover_var, fs=CN.FS, order=order_var, filtfilt=CN.FILTFILT_TDALIGN, b_type='low')
    
    #calculate limits for time domain shifting based on crossover frequency
    total_shift_quota = int(CN.FS / f_crossover_var)
    shift_quota_per_pol = int(total_shift_quota / 2)  # Only half size needed due to flipping polarities
    t_shift_interval = 10
    
    # Round min_t_shift down to the nearest multiple of t_shift_interval
    min_t_shift = int(np.floor((0 - shift_quota_per_pol / 2) / t_shift_interval) * t_shift_interval)
    # Round max_t_shift up to the nearest multiple of t_shift_interval
    max_t_shift = int(np.ceil((0 + shift_quota_per_pol / 2) / t_shift_interval) * t_shift_interval)
    
    # if fb_filtering == True:#symmetrical case
    #     # Round min_t_shift down to the nearest multiple of t_shift_interval
    #     min_t_shift = int(np.floor((0 - shift_quota_per_pol / 2) / t_shift_interval) * t_shift_interval)
    #     # Round max_t_shift up to the nearest multiple of t_shift_interval
    #     max_t_shift = int(np.ceil((0 + shift_quota_per_pol / 2) / t_shift_interval) * t_shift_interval)
    # else:#bias towards lower values
    #     # Round min_t_shift down to the nearest multiple of t_shift_interval
    #     min_t_shift = int(np.floor((0 - shift_quota_per_pol * (2/3)) / t_shift_interval) * t_shift_interval)
    #     # Round max_t_shift up to the nearest multiple of t_shift_interval
    #     max_t_shift = int(np.ceil((0 + shift_quota_per_pol * (1/3)) / t_shift_interval) * t_shift_interval)
        
    # Calculate the number of intervals
    num_intervals = int(np.abs((max_t_shift - min_t_shift) / t_shift_interval))
    
    # print(str(min_t_shift))
    # print(str(max_t_shift))
    # print(str(num_intervals))
    
    peak_to_peak_window = int(np.divide(CN.FS,f_crossover_var)*0.95)
    
    delay_eval_set_sub_p = np.zeros((num_intervals))
    delay_eval_set_sub_n = np.zeros((num_intervals))

    #take sum of reference BRIR and Sub BRIR, low pass, compare peak to peak distance in low freqs
    #then calculate distance from peak to peak within a 400 sample window 
    
    mag_range_a=int(30*(n_fft/CN.N_FFT))
    mag_range_b=int(290*(n_fft/CN.N_FFT))
    

    #set level of sub BRIR to 0 at low freqs
    data_fft = np.fft.rfft(sub_brir_ir[0][:])
    mag_fft=np.abs(data_fft)
    average_mag = np.mean(mag_fft[mag_range_a:mag_range_b])
    if average_mag == 0:
        if CN.LOG_INFO == True:
            logging.info('0 magnitude detected')
    for chan in range(CN.TOTAL_CHAN_BRIR):
        sub_brir_ir[chan][:] = np.divide(sub_brir_ir[chan][:],average_mag)



 
    subrir_sample_p = np.copy(sub_brir_ir[0][:])#check first ir, first channel
    subrir_sample_n = np.multiply(np.copy(sub_brir_ir[0][:]),-1)

    #apply low pass before shifting
    # brir_sample_lp = hf.signal_lowpass_filter(brir_sample, f_crossover_var, CN.FS, order_var)
    # subrir_sample_p_lp = hf.signal_lowpass_filter(subrir_sample_p, f_crossover_var, CN.FS, order_var)
    # subrir_sample_n_lp = hf.signal_lowpass_filter(subrir_sample_n, f_crossover_var, CN.FS, order_var)
    
    brir_sample_lp = hf.apply_sos_filter(brir_sample, lp_sos, filtfilt=CN.FILTFILT_TDALIGN)
    subrir_sample_p_lp = hf.apply_sos_filter(subrir_sample_p, lp_sos, filtfilt=CN.FILTFILT_TDALIGN)
    subrir_sample_n_lp = hf.apply_sos_filter(subrir_sample_n, lp_sos, filtfilt=CN.FILTFILT_TDALIGN)

    #run once for positive polarity
    for delay in range(num_intervals):
    
        #shift SUBBRIR
        current_shift = min_t_shift+(delay*t_shift_interval)
        subrir_shift_c = np.roll(subrir_sample_p_lp,current_shift)
        
        #add (BRIR) to shifted SUBBRIR
        sum_ir_lp = np.add(brir_sample_lp,subrir_shift_c)

        #calculate distance from peak to peak within a 400 sample window
        peak_to_peak_iter=0
        for hop_id in range(CN.DELAY_WIN_HOPS):
            samples = hop_id*CN.DELAY_WIN_HOP_SIZE
            local_max=np.max(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+peak_to_peak_window])
            local_min=np.min(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+peak_to_peak_window])
            if CN.PEAK_MEAS_MODE == 1:
                peak_to_peak = np.abs(local_max-local_min)
            else:
                peak_to_peak = np.abs(local_max)
            #if this window has larger pk to pk, store in iter var
            if peak_to_peak > peak_to_peak_iter:
                peak_to_peak_iter = peak_to_peak
        #store largest pk to pk distance of all windows into delay set
        delay_eval_set_sub_p[delay] = peak_to_peak_iter

    peak_to_peak_max_p = np.max(delay_eval_set_sub_p[:])
    index_shift_p = np.argmax(delay_eval_set_sub_p[:])
    
    #run once for negative polarity
    for delay in range(num_intervals):
    
        #shift SUBBRIR
        current_shift = min_t_shift+(delay*t_shift_interval)
        subrir_shift_c = np.roll(subrir_sample_n_lp,current_shift)
        
        #add (BRIR) to shifted SUBBRIR
        sum_ir_lp = np.add(brir_sample_lp,subrir_shift_c)

        #calculate distance from peak to peak within a 400 sample window
        peak_to_peak_iter=0
        for hop_id in range(CN.DELAY_WIN_HOPS):
            samples = hop_id*CN.DELAY_WIN_HOP_SIZE
            local_max=np.max(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+peak_to_peak_window])
            local_min=np.min(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+peak_to_peak_window])
            if CN.PEAK_MEAS_MODE == 1:
                peak_to_peak = np.abs(local_max-local_min)
            else:
                peak_to_peak = np.abs(local_max)
            #if this window has larger pk to pk, store in iter var
            if peak_to_peak > peak_to_peak_iter:
                peak_to_peak_iter = peak_to_peak
        #store largest pk to pk distance of all windows into delay set
        delay_eval_set_sub_n[delay] = peak_to_peak_iter

    peak_to_peak_max_n = np.max(delay_eval_set_sub_n[:])
    index_shift_n = np.argmax(delay_eval_set_sub_n[:])
    
    if peak_to_peak_max_p > peak_to_peak_max_n or CN.EVAL_POLARITY == False:
        index_shift=index_shift_p
        sub_polarity=1
        #shift subrir by delay that has largest peak to peak distance
        samples_shift=min_t_shift+(index_shift*t_shift_interval)
    else:
        index_shift=index_shift_n
        sub_polarity=-1
        #shift subrir by delay that has largest peak to peak distance
        samples_shift=min_t_shift+(index_shift*t_shift_interval)
    
    
    

    for chan in range(CN.TOTAL_CHAN_BRIR):
        #roll subrir
        sub_brir_ir[chan][:] = np.roll(sub_brir_ir[chan][:],samples_shift)
        #change polarity if applicable
        sub_brir_ir[chan][:] = np.multiply(sub_brir_ir[chan][:],sub_polarity)
        #set end of array to zero to remove any data shifted to end of array
        if samples_shift < 0:
            sub_brir_ir[chan][samples_shift:] = sub_brir_ir[chan][samples_shift:]*0#left
            sub_brir_ir[chan][:] = np.multiply(sub_brir_ir[chan][:],initial_removal_win_sub)
    

    if CN.LOG_INFO == True and CN.SHOW_DEV_TOOLS == True:
        logging.info('(SUB) delay index = ' + str(index_shift))
        logging.info('(SUB) samples_shift = ' + str(samples_shift))
        logging.info('(SUB) sub polarity = ' + str(sub_polarity))
        logging.info('(SUB) peak_to_peak_max_n = ' + str(peak_to_peak_max_n))
        logging.info('(SUB) peak_to_peak_max_p = ' + str(peak_to_peak_max_p))


def reverb_brir_align(brir_reverberation, n_fft, f_crossover_var, hrir_sample, initial_removal_win_rev, total_azim_reverb, order_var):
    """
    Function performs time domain alignment of reverb brir array with hrirs
    """ 
    #eval_polarity=CN.EVAL_POLARITY
    eval_polarity=False

    brir_sample = np.copy(brir_reverberation[0][1][0][:])#use id 1 (FL)
    
    peak_to_peak_window = int(np.divide(CN.FS,f_crossover_var)*0.95)
    
    delay_eval_set_reverb_p = np.zeros((CN.NUM_INTERVALS_R))
    delay_eval_set_reverb_n = np.zeros((CN.NUM_INTERVALS_R))

    #take sum of reference HRIR and Reverb BRIR, low pass, compare peak to peak distance in low freqs
    #then calculate distance from peak to peak within a 400 sample window 
    
    reverb_sample_p = np.copy(brir_sample)#check first ir, first channel
    reverb_sample_n = np.multiply(np.copy(brir_sample),-1)

    #run once for positive polarity
    for delay in range(CN.NUM_INTERVALS_R):
    
        #shift (BRIR)
        current_shift = CN.MIN_T_SHIFT_R+(delay*CN.T_SHIFT_INTERVAL_R)
        reverb_shift_c = np.roll(reverb_sample_p,current_shift)
        
        #add (HRIR) to shifted (BRIR)
        sum_ir_c = np.add(hrir_sample,reverb_shift_c)

        #calculate distance from peak to peak within a 400 sample window
        sum_ir_lp = hf.signal_lowpass_filter(sum_ir_c, f_crossover_var, CN.FS, order_var)
        peak_to_peak_iter=0
        for hop_id in range(CN.DELAY_WIN_HOPS):
            samples = hop_id*CN.DELAY_WIN_HOP_SIZE
            local_max=np.max(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+peak_to_peak_window])
            local_min=np.min(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+peak_to_peak_window])
            if CN.PEAK_MEAS_MODE == 1:
                peak_to_peak = np.abs(local_max-local_min)
            else:
                peak_to_peak = np.abs(local_max)
            #if this window has larger pk to pk, store in iter var
            if peak_to_peak > peak_to_peak_iter:
                peak_to_peak_iter = peak_to_peak
        #store largest pk to pk distance of all windows into delay set
        delay_eval_set_reverb_p[delay] = peak_to_peak_iter

    peak_to_peak_max_p = np.max(delay_eval_set_reverb_p[:])
    index_shift_p = np.argmax(delay_eval_set_reverb_p[:])
    
    #run once for negative polarity
    for delay in range(CN.NUM_INTERVALS_R):
    
        #shift (BRIR)
        current_shift = CN.MIN_T_SHIFT_R+(delay*CN.T_SHIFT_INTERVAL_R)
        reverb_shift_c = np.roll(reverb_sample_n,current_shift)
        
        #add (HRIR) to shifted (BRIR)
        sum_ir_c = np.add(hrir_sample,reverb_shift_c)

        #calculate distance from peak to peak within a 400 sample window
        sum_ir_lp = hf.signal_lowpass_filter(sum_ir_c, f_crossover_var, CN.FS, order_var)
        peak_to_peak_iter=0
        for hop_id in range(CN.DELAY_WIN_HOPS):
            samples = hop_id*CN.DELAY_WIN_HOP_SIZE
            local_max=np.max(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+peak_to_peak_window])
            local_min=np.min(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+peak_to_peak_window])
            if CN.PEAK_MEAS_MODE == 1:
                peak_to_peak = np.abs(local_max-local_min)
            else:
                peak_to_peak = np.abs(local_max)
            #if this window has larger pk to pk, store in iter var
            if peak_to_peak > peak_to_peak_iter:
                peak_to_peak_iter = peak_to_peak
        #store largest pk to pk distance of all windows into delay set
        delay_eval_set_reverb_n[delay] = peak_to_peak_iter

    peak_to_peak_max_n = np.max(delay_eval_set_reverb_n[:])
    index_shift_n = np.argmax(delay_eval_set_reverb_n[:])
    
    if peak_to_peak_max_p > peak_to_peak_max_n or eval_polarity == False:
        index_shift=index_shift_p
        sub_polarity=1
        #shift BRIR by delay that has largest peak to peak distance
        samples_shift=CN.MIN_T_SHIFT_R+(index_shift*CN.T_SHIFT_INTERVAL_R)
    else:
        index_shift=index_shift_n
        sub_polarity=-1
        #shift BRIR by delay that has largest peak to peak distance
        samples_shift=CN.MIN_T_SHIFT_R+(index_shift*CN.T_SHIFT_INTERVAL_R)
    
    
    

    for azim in range(total_azim_reverb):
        #roll BRIR
        brir_reverberation[0][azim][0][:] = np.roll(brir_reverberation[0][azim][0][:],samples_shift)
        brir_reverberation[0][azim][1][:] = np.roll(brir_reverberation[0][azim][1][:],samples_shift)
        #change polarity if applicable
        brir_reverberation[0][azim][0][:] = np.multiply(brir_reverberation[0][azim][0][:],sub_polarity)
        brir_reverberation[0][azim][1][:] = np.multiply(brir_reverberation[0][azim][1][:],sub_polarity)
        
        if samples_shift < 0:
            #set end of array to zero to remove any data shifted to end of array
            brir_reverberation[0][azim][0][samples_shift:] = np.multiply(brir_reverberation[0][azim][0][samples_shift:],0)
            brir_reverberation[0][azim][1][samples_shift:] = np.multiply(brir_reverberation[0][azim][1][samples_shift:],0)
            #apply window to start
            brir_reverberation[0][azim][0][:] = np.multiply(brir_reverberation[0][azim][0][:],initial_removal_win_rev)
            brir_reverberation[0][azim][1][:] = np.multiply(brir_reverberation[0][azim][1][:],initial_removal_win_rev)
     

    if CN.LOG_INFO == True and CN.SHOW_DEV_TOOLS == True:
        logging.info('(Reverb) delay index = ' + str(index_shift))
        logging.info('(Reverb) samples_shift = ' + str(samples_shift))
        logging.info('(Reverb) polarity = ' + str(sub_polarity))
        logging.info('(Reverb) peak_to_peak_max_n = ' + str(peak_to_peak_max_n))
        logging.info('(Reverb) peak_to_peak_max_p = ' + str(peak_to_peak_max_p))
        
        
        

def combine_hrirs_brirs(brir_out, elev, n_fft, total_azim_hrir, hrir_selected, total_samples_hrir, swap_azimuths=False):
    """
    Function integrates HRIRs into BRIR dataset
    :param brir_out: numpy array containing brirs
    :param swap_azimuths: bool flag, if true will swap left and right azimuths. Useful for cases where azimuth angles have opposite direction to standard CCW
    :return: None
    """ 
    
    for azim in range(total_azim_hrir):
        if swap_azimuths == True and azim>0:#swap L and R if flagged and not 0 id
            azim_hrir = total_azim_hrir-azim
        else:
            azim_hrir = azim
        for chan in range(CN.TOTAL_CHAN_BRIR):      
            brir_out[elev][azim][chan][0:total_samples_hrir] = brir_out[elev][azim][chan][0:total_samples_hrir] + hrir_selected[elev][azim][chan][0:total_samples_hrir]
    



def calc_eq_for_brirs(brir_out, elev, total_azim_hrir, direction_matrix_process, fr_flat, results_list, list_indx):
    """
    Function calculates equalisation filter in parts
    :param brir_out: numpy array containing brirs
    :return: None
    """ 
    
    num_brirs_avg = 0
    brir_fft_avg_db = fr_flat.copy()
    
    #get diffuse field spectrum
    for azim in range(total_azim_hrir):
        for chan in range(CN.TOTAL_CHAN_BRIR):
            #only apply if direction is in direction matrix
            if direction_matrix_process[elev][azim][0][0] == 1: 
                brir_current = brir_out[elev, azim, chan, 0:CN.N_FFT]
                brir_current_fft = np.fft.rfft(brir_current)
                brir_current_mag_fft=np.abs(brir_current_fft)
                brir_current_db_fft = hf.mag2db(brir_current_mag_fft)
                
                brir_fft_avg_db = np.add(brir_fft_avg_db,brir_current_db_fft)
                
                num_brirs_avg = num_brirs_avg+1
    
    #divide by total number of brirs
    if num_brirs_avg > 0:
        brir_fft_avg_db = brir_fft_avg_db/num_brirs_avg
    
    results_list[list_indx]=brir_fft_avg_db
    

def apply_eq_to_brirs(brir_out, elev, n_fft, total_azim_hrir, direction_matrix_post, brir_df_inv_fir, room_target_fir,  pinna_comp_fir, data_lf_comp_eq, crop_samples):
    """
    Function applies equalisation to brirs in numpy array
    :param brir_out: numpy array containing brirs
    :return: None
    """ 
    
 
    for azim in range(total_azim_hrir):
        for chan in range(CN.TOTAL_CHAN_BRIR):  
            
            #only apply equalisation if direction is in direction matrix
            if direction_matrix_post[elev][azim][0][0] == 1: 
            
                #convolve BRIR with filters
                brir_eq_b = np.copy(brir_out[elev,azim,chan,0:crop_samples])#brir_eq_a = np.copy(brir_out[elev][azim][chan][:])
                #apply DF eq
                brir_eq_b = sp.signal.convolve(brir_eq_b,brir_df_inv_fir, 'full', 'auto')
                #apply room target
                brir_eq_b = sp.signal.convolve(brir_eq_b,room_target_fir, 'full', 'auto')
                #apply pinna compensation
                brir_eq_b = sp.signal.convolve(brir_eq_b,pinna_comp_fir, 'full', 'auto')
                #apply additional eq for headphones
                brir_eq_b = sp.signal.convolve(brir_eq_b,data_lf_comp_eq, 'full', 'auto')
                
                replace_samples=min(len(brir_eq_b),n_fft)
                brir_out[elev,azim,chan,0:replace_samples] = np.copy(brir_eq_b[0:replace_samples])
                if replace_samples < n_fft:
                    brir_out[elev,azim,chan,replace_samples:n_fft]=0
            
            else:
                brir_out[elev,azim,chan,:] = np.zeros(n_fft)#zero out directions that wont be exported





def integrate_sub_brirs(brir_out, elev, n_fft, total_azim_hrir, direction_matrix_process, sub_brir_ir, hp_sos, fb_filtering):
    """
    Function applies final stages of sub brir integration
    :param brir_out: numpy array containing brirs
    :return: None
    """ 

    #
    #apply high pass filter to hrir+brir
    #
    for azim in range(total_azim_hrir):
        for chan in range(CN.TOTAL_CHAN_BRIR):  
            #only apply equalisation if direction is in direction matrix
            if direction_matrix_process[elev][azim][0][0] == 1: 
            
                brir_eq_a_h = np.copy(brir_out[elev][azim][chan][:])
                
                #apply hp filter
                #brir_eq_b_h = hf.signal_highpass_filter(brir_eq_a_h, f_crossover_var, CN.FS, order_var)
                brir_eq_b_h = hf.apply_sos_filter(brir_eq_a_h, hp_sos, filtfilt=fb_filtering)
                
                brir_out[elev][azim][chan][:] = brir_eq_b_h[0:n_fft] 
  
    #add SUB BRIR into output BRIR array
    for azim in range(total_azim_hrir):
        for chan in range(CN.TOTAL_CHAN_BRIR):
            #only apply if direction is in direction matrix
            if direction_matrix_process[elev][azim][0][0] == 1: 
                brir_out[elev][azim][chan][:] = brir_out[elev][azim][chan][:] + sub_brir_ir[chan][:]





def preprocess_hrirs(spatial_res=1, gui_logger=None):
    """
    Function applies level matching to hrirs in MAT dataset and saves as NPY dataset
    :param gui_logger: gui logger object for dearpygui
    :return: None
    """ 
    
    # get the start time
    st = time.time()
    n_fft=CN.N_FFT
    lf_align=False#False
    h_pass=True

    try:
    
        if spatial_res >= 0 and spatial_res < CN.NUM_SPATIAL_RES:
            if spatial_res <= 2:
                spatial_res_str = 'high'
                #this dataset includes all hrirs up to high spatial resolution. Elevations from -60 to 60deg in 5 deg steps, Azimuths from 0 to 360dg in 5deg steps
                mat_fname = pjoin(CN.DATA_DIR_INT, 'hrir_dataset_compensated_high.mat')
            elif spatial_res == 3:
                spatial_res_str = 'max'
                #this dataset includes all hrirs at full spatial resolution. Elevations from -40 to 60deg in 2 deg steps, Azimuths from 0 to 360dg in 2deg steps
                mat_fname = pjoin(CN.DATA_DIR_INT, 'hrir_dataset_compensated_max.mat')
            elev_min=CN.SPATIAL_RES_ELEV_MIN[spatial_res] 
            elev_max=CN.SPATIAL_RES_ELEV_MAX[spatial_res] 
            elev_nearest=CN.SPATIAL_RES_ELEV_NEAREST[spatial_res] #as per hrir dataset
            azim_nearest=CN.SPATIAL_RES_AZIM_NEAREST[spatial_res] 
        else:
            log_string = 'Invalid spatial resolution. No hrir dataset loaded.'
            hf.log_with_timestamp(log_string, gui_logger)

        #old format mat files
        #annots = loadmat(mat_fname)
        #new format MATLAB 7.3 files
        hrirs = mat73.loadmat(mat_fname)
        
        #load matrix, get list of numpy arrays
        hrir_list = [[element for element in upperElement] for upperElement in hrirs['ash_input_hrir']]
        #grab desired hrtf. by default returns 9x72x2x65536 array
        num_hrtfs = len(hrir_list)
        hrir_selected_ref = hrir_list[0]
        total_elev_hrir = len(hrir_selected_ref)
        total_azim_hrir = len(hrir_selected_ref[0])
        total_chan_hrir = len(hrir_selected_ref[0][0])
        total_samples_hrir = len(hrir_selected_ref[0][0][0])
        base_elev_idx = total_elev_hrir//2
        print(str(total_samples_hrir))
        
        # take 0 deg azim as reference
        index_peak_ref = np.argmax(hrir_selected_ref[base_elev_idx][0][0][:])
        hrir_ref_sample = np.copy(hrir_selected_ref[base_elev_idx][0][0][:])
        #HRIR array will be populated with all HRIRs
        hrir_out=np.zeros((num_hrtfs,total_elev_hrir,total_azim_hrir,total_chan_hrir,total_samples_hrir))

        for hrtf_idx in range(num_hrtfs):
            
            hrir_selected = hrir_list[hrtf_idx]
            hrtf_type=hrtf_idx+1
  
            if (spatial_res <= 2) and (hrtf_type >= 5 and hrtf_type <= 8):
                base_azim_idx = total_azim_hrir//2
            else:
                base_azim_idx=0
            #
            #apply level matching
            ## set HRIR level to 0
            
            mag_range_a=CN.SPECT_SNAP_M_F0 #mag_range_a=int(CN.SPECT_SNAP_F0*n_fft/CN.FS)
            mag_range_b=CN.SPECT_SNAP_M_F1 #mag_range_b=int(CN.SPECT_SNAP_F1*n_fft/CN.FS)
       
            polarity=1
            #invert polarity of HRTF 4 (04: B&K Type 4128) and 6 (06: DADEC (MMHR-HRIR)) to align with reference
            if (spatial_res <= 2) and (hrtf_type in CN.HRTF_TYPE_LIST_FLIP_POL):
                polarity=-1
            
            for elev in range(total_elev_hrir):
                for azim in range(total_azim_hrir):
                    #direction specific gain
                    avg_mag_sum = 0
                    for chan in range(total_chan_hrir):
                        data_fft = np.fft.fft(hf.padarray(hrir_selected[elev][azim][chan][:],n_fft))#also zero pads array in place
                        mag_fft=np.abs(data_fft)
                        #avg_mag_azim = np.mean(mag_fft[mag_range_a:mag_range_b])
                        avg_mag_azim = np.mean(mag_fft[mag_range_a:mag_range_b])
                        avg_mag_sum=avg_mag_sum+avg_mag_azim
                    avg_mag=avg_mag_sum/total_chan_hrir
                    
                    for chan in range(total_chan_hrir):
                        hrir_selected[elev][azim][chan][:] = np.divide(hrir_selected[elev][azim][chan][:],avg_mag)
                        hrir_selected[elev][azim][chan][:] = np.multiply(hrir_selected[elev][azim][chan][:],polarity)

            #
            #apply time domain alignment
            #

            for elev in range(total_elev_hrir):
                for azim in range(total_azim_hrir):
                    azim_deg = int(azim*azim_nearest)
                    #take left channel if azim < 180 deg, otherwise take right channel
                    if azim_deg <= 180:
                        index_peak_cur = np.argmax(np.abs(hrir_selected[elev][azim][0][:]))
                    else:
                        index_peak_cur = np.argmax(np.abs(hrir_selected[elev][azim][1][:]))    
                    hrir_shift = index_peak_ref-index_peak_cur
                    
                    for chan in range(total_chan_hrir):
                        hrir_selected[elev][azim][chan][:] = np.roll(hrir_selected[elev][azim][chan][:],hrir_shift)


            #
            #apply high pass filter to hrirs
            #
            if h_pass == True:
                #f_crossover_var=CN.F_CROSSOVER
                f_crossover_var=180
                for elev in range(total_elev_hrir):
                    for azim in range(total_azim_hrir):
                        for chan in range(total_chan_hrir):  
                            hrir_eq_a_h = np.copy(hrir_selected[elev][azim][chan][:])
                            #apply hp filter
                            hrir_eq_b_h = hf.signal_highpass_filter(hrir_eq_a_h, f_crossover_var, CN.FS, CN.ORDER)
                            hrir_selected[elev][azim][chan][0:total_samples_hrir] = hrir_eq_b_h[0:total_samples_hrir] 


            if lf_align == True:
                #after aligning in high freqs, align in low freqs
                fc_alignment=1000#500

                delay_eval_set_sub_p = np.zeros((CN.NUM_INTERVALS_H))

                #section to calculate best delay for next ir to align with this ir
                for elev in range(total_elev_hrir):
                    for azim in range(total_azim_hrir):
                        azim_deg = int(azim*azim_nearest)
                        if azim_deg <= 180:
                            hrir_sample_p = np.copy(hrir_selected[elev][azim][0][:])
                        else:
                            hrir_sample_p = np.copy(hrir_selected[elev][azim][1][:]) 
    
                        #run once for positive polarity
                        for delay in range(CN.NUM_INTERVALS_H):
                        
                            #shift next ir (hrir)
                            current_shift = CN.MIN_T_SHIFT_H+(delay*CN.T_SHIFT_INTERVAL_H)
                            hrir_shift_c = np.roll(hrir_sample_p,current_shift)
                            
                            #add current ir (ref hrir) to shifted next ir (hrir)
                            sum_ir_c = np.add(hrir_ref_sample,hrir_shift_c)
            
                            #method 5: calculate distance from peak to peak
                            sum_ir_lp = hf.signal_lowpass_filter(sum_ir_c, fc_alignment, CN.FS, CN.ORDER)
                            local_max=np.max(sum_ir_lp)
                            local_min=np.min(sum_ir_lp)
                            if CN.PEAK_MEAS_MODE == 1:
                                peak_to_peak = np.abs(local_max-local_min)
                            else:
                                peak_to_peak = np.abs(local_max)
                
                            #store largest pk to pk distance of all windows into delay set
                            delay_eval_set_sub_p[delay] = peak_to_peak

                        index_shift_p = np.argmax(delay_eval_set_sub_p[:])
        
                        index_shift=index_shift_p
                        #shift next ir by delay that has largest peak to peak distance
                        samples_shift=CN.MIN_T_SHIFT_H+(index_shift*CN.T_SHIFT_INTERVAL_H)
         
                        for chan in range(CN.TOTAL_CHAN_BRIR):
                            #roll hrir
                            hrir_selected[elev][azim][chan][:] = np.roll(hrir_selected[elev][azim][chan][:],samples_shift)
                            #set end of array to zero to remove any data shifted to end of array
                            if samples_shift < 0:
                                hrir_selected[elev][azim][chan][samples_shift:] = hrir_selected[elev][azim][chan][samples_shift:]*0#left
  
            #populate hrir dataset
            for elev in range(total_elev_hrir):
                for azim in range(total_azim_hrir):
                    azim_deg = int(azim*azim_nearest)
                    
                    #correct azimuth angle for MMHR datasets
                    if (spatial_res <= 2) and (hrtf_type >= 5 and hrtf_type <= 8):
                        if azim_deg <= 180:
                            azim_deg_in = 180-azim_deg
                        else:
                            azim_deg_in = 180+360-azim_deg
                        azim_in = int(azim_deg_in/azim_nearest)
                    else:
                        azim_in = azim
                        
                    for chan in range(total_chan_hrir):
                        hrir_out[hrtf_idx,elev,azim,chan,0:total_samples_hrir] = np.copy(hrir_selected[elev][azim_in][chan][0:total_samples_hrir])
     
            log_string = 'level adjustment and time alignment completed for hrtf idx: ' + str(hrtf_idx)
            hf.log_with_timestamp(log_string, gui_logger)

        #save resulting dataset to a numpy file
        if lf_align == True:
            npy_file_name =  'hrir_dataset_compensated_lfa_' +spatial_res_str+'.npy'
        elif h_pass == True:
            npy_file_name =  'hrir_dataset_compensated_hp_' +spatial_res_str+'.npy'
        else:
            npy_file_name =  'hrir_dataset_compensated_' +spatial_res_str+'.npy'
    
        out_file_path = pjoin(CN.DATA_DIR_INT,npy_file_name)      

        np.save(out_file_path,hrir_out)    

        log_string = 'Binary saved in data folder'
        hf.log_with_timestamp(log_string, gui_logger)
  
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        
    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    if CN.LOG_INFO == True:
        logging.info('Execution time:' + str(elapsed_time) + ' seconds')
   
        
   
def process_mono_cues(gui_logger=None):
    """
    Function performs statistical analysis on monoaural cues
    """    

    ir_set='mono_cues_set_a'
    brir_ref_folder = pjoin(CN.DATA_DIR_INT, 'mono_cues', ir_set)
    
    n_fft=CN.N_FFT
    
    ash_samp_freq=44100
    output_wavs=1

    total_chan_brir=2
  
    #impulse
    impulse=np.zeros(n_fft)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat = hf.mag2db(fr_flat_mag)
    
    
    try:
        #
        #section for DRIR cues
        #
        num_bairs_avg = 0
        brir_fft_avg_db = fr_flat.copy()
        #loop through folders
        for root, dirs, files in os.walk(brir_ref_folder):
            for filename in files:
                if '.wav' in filename:
                    #print('test')
                    
                    #read wav files
                    wav_fname = pjoin(root, filename)
                    samplerate, data = wavfile.read(wav_fname)
                    #samp_freq=samplerate
                    fir_array = data / (2.**31)
                    fir_length = len(fir_array)
                    
                    #resample if sample rate is not 44100
                    if samplerate != ash_samp_freq:
                        fir_array = hf.resample_signal(fir_array, original_rate = samplerate, new_rate = ash_samp_freq)
                        log_string_a = 'source samplerate: ' + str(samplerate) + ', resampled to: '+ str(ash_samp_freq)
                        hf.log_with_timestamp(log_string_a, gui_logger)
 
                    extract_legth = min(n_fft,fir_length)
                    
                    for chan in range(total_chan_brir):
                        if ('_A-90' in filename ) or ('_A90' in filename ): #if ('_A-90' in filename and chan == 0) or ('_A90' in filename and chan == 1):
                            brir_current=np.zeros(n_fft)
                            brir_current[0:extract_legth] = np.copy(fir_array[0:extract_legth,chan])#brir_out[elev][azim][chan][:]
                            brir_current_fft = np.fft.fft(brir_current)#ensure left channel is taken for both channels
                            brir_current_mag_fft=np.abs(brir_current_fft)
                            brir_current_db_fft = hf.mag2db(brir_current_mag_fft)
                            
                            brir_fft_avg_db = np.add(brir_fft_avg_db,brir_current_db_fft)
                            
                            num_bairs_avg = num_bairs_avg+1

        #divide by total number of brirs
        brir_fft_avg_db = brir_fft_avg_db/num_bairs_avg
        #convert to mag
        brir_fft_avg_mag = hf.db2mag(brir_fft_avg_db)
        #level ends of spectrum
        brir_fft_avg_mag = hf.level_spectrum_ends(brir_fft_avg_mag, 450, 18000, smooth_win = 10)#150
        #smoothing
        #octave smoothing
        brir_fft_avg_mag_sm = hf.smooth_fft_octaves(data=brir_fft_avg_mag, n_fft=n_fft, win_size_base = 25)
        drir_90_l_mean_mag = np.copy(brir_fft_avg_mag_sm)
        if CN.PLOT_ENABLE == True:
            print(str(num_bairs_avg))
            hf.plot_data(brir_fft_avg_mag_sm,'brir_fft_avg_mag_sm', normalise=1)    
 
        #compensation
        filename = 'diffuse_field_eq_for_in_ear_headphones.wav'
        wav_fname = pjoin(CN.DATA_DIR_INT, filename)
        samplerate, data_addit_eq = wavfile.read(wav_fname)
        data_addit_eq = data_addit_eq / (2.**31)
        #convert to mag response
        in_ear_eq_fir=np.zeros(CN.N_FFT)
        in_ear_eq_fir[0:1024] = data_addit_eq[0:1024]
        data_fft = np.fft.fft(in_ear_eq_fir)#
        in_ear_eq_mag=np.abs(data_fft)
        in_ear_eq_db=hf.mag2db(in_ear_eq_mag)
        
        drir_90_l_mean_db = hf.mag2db(drir_90_l_mean_mag)
        drir_90_l_mean_db_comp = np.subtract(drir_90_l_mean_db,in_ear_eq_db)
        #convert to mag
        drir_90_l_mean_comp = hf.db2mag(drir_90_l_mean_db_comp)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(drir_90_l_mean_comp,'drir_90_l_mean_comp', normalise=1)  
    
    
        #
        # section for HP cues
        #
        ir_set='hp_cues'
        hp_folder = 'on_ear'
        hp_mag_in_folder = pjoin(CN.DATA_DIR_INT, 'mono_cues', ir_set,hp_folder)
        
        num_sets_avg = 0
        hp_fft_avg_db = fr_flat.copy()
        #loop through folders
        for root, dirs, files in os.walk(hp_mag_in_folder):
            for filename in files:
                if '.npy' in filename:
                    #read npy files
                    npy_file_path = pjoin(root, filename)
                    hp_current_mag_fft = np.load(npy_file_path)
                    hp_current_db_fft = hf.mag2db(hp_current_mag_fft)
                    hp_fft_avg_db = np.add(hp_fft_avg_db,hp_current_db_fft)
                    num_sets_avg = num_sets_avg+1

        #divide by total number of brirs
        hp_fft_avg_db = hp_fft_avg_db/num_sets_avg
        #convert to mag
        brir_fft_avg_mag = hf.db2mag(hp_fft_avg_db)
        #level ends of spectrum
        brir_fft_avg_mag = hf.level_spectrum_ends(brir_fft_avg_mag, 450, 18000, smooth_win = 20)#150
        #smoothing
        #octave smoothing
        brir_fft_avg_mag_sm = hf.smooth_fft_octaves(data=brir_fft_avg_mag, n_fft=n_fft, win_size_base = 20)
        hp_cue_onear_mean_mag = np.copy(brir_fft_avg_mag_sm)
 
        if CN.PLOT_ENABLE == True:
            print(str(num_sets_avg))
            hf.plot_data(brir_fft_avg_mag_sm,'brir_fft_avg_mag_sm', normalise=1) 
        
        
        ir_set='hp_cues'
        hp_folder = 'over_ear'
        hp_mag_in_folder = pjoin(CN.DATA_DIR_INT, 'mono_cues', ir_set,hp_folder)
        
        num_sets_avg = 0
        hp_fft_avg_db = fr_flat.copy()
        #loop through folders
        for root, dirs, files in os.walk(hp_mag_in_folder):
            for filename in files:
                if '.npy' in filename:
                    #read npy files
                    npy_file_path = pjoin(root, filename)
                    hp_current_mag_fft = np.load(npy_file_path)
                    hp_current_db_fft = hf.mag2db(hp_current_mag_fft)
                    hp_fft_avg_db = np.add(hp_fft_avg_db,hp_current_db_fft)
                    num_sets_avg = num_sets_avg+1

        #divide by total number of brirs
        hp_fft_avg_db = hp_fft_avg_db/num_sets_avg
        #convert to mag
        brir_fft_avg_mag = hf.db2mag(hp_fft_avg_db)
        #level ends of spectrum
        brir_fft_avg_mag = hf.level_spectrum_ends(brir_fft_avg_mag, 450, 18000, smooth_win = 20)#150
        #smoothing
        #octave smoothing
        brir_fft_avg_mag_sm = hf.smooth_fft_octaves(data=brir_fft_avg_mag, n_fft=n_fft, win_size_base = 20)
        hp_cue_overear_mean_mag = np.copy(brir_fft_avg_mag_sm)
 
        if CN.PLOT_ENABLE == True:
            print(str(num_sets_avg))
            hf.plot_data(brir_fft_avg_mag_sm,'brir_fft_avg_mag_sm', normalise=1) 
        
        ir_set='hp_cues'
        hp_folder = 'in_ear'
        hp_mag_in_folder = pjoin(CN.DATA_DIR_INT, 'mono_cues', ir_set,hp_folder)
        
        num_sets_avg = 0
        hp_fft_avg_db = fr_flat.copy()
        #loop through folders
        for root, dirs, files in os.walk(hp_mag_in_folder):
            for filename in files:
                if '.npy' in filename:
                    #read npy files
                    npy_file_path = pjoin(root, filename)
                    hp_current_mag_fft = np.load(npy_file_path)
                    hp_current_db_fft = hf.mag2db(hp_current_mag_fft)
                    hp_fft_avg_db = np.add(hp_fft_avg_db,hp_current_db_fft)
                    num_sets_avg = num_sets_avg+1

        #divide by total number of brirs
        hp_fft_avg_db = hp_fft_avg_db/num_sets_avg
        #convert to mag
        brir_fft_avg_mag = hf.db2mag(hp_fft_avg_db)
        #level ends of spectrum
        brir_fft_avg_mag = hf.level_spectrum_ends(brir_fft_avg_mag, 450, 18000, smooth_win = 20)#150
        #smoothing
        #octave smoothing
        brir_fft_avg_mag_sm = hf.smooth_fft_octaves(data=brir_fft_avg_mag, n_fft=n_fft, win_size_base = 20)
        hp_cue_inear_mean_mag = np.copy(brir_fft_avg_mag_sm)
        if CN.PLOT_ENABLE == True:
            print(str(num_sets_avg))
            hf.plot_data(brir_fft_avg_mag_sm,'brir_fft_avg_mag_sm', normalise=1) 
        
        #compare difference between over ear and in ear
        hp_overear_db_fft = hf.mag2db(hp_cue_overear_mean_mag)
        hp_inear_db_fft = hf.mag2db(hp_cue_inear_mean_mag)
        hp_fft_diff_db = np.subtract(hp_overear_db_fft,hp_inear_db_fft)
        #convert to mag
        hp_cue_diff_mean_mag = hf.db2mag(hp_fft_diff_db)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(hp_cue_diff_mean_mag,'hp_cue_diff_mean_mag', normalise=1)
            
        #compare difference between on ear and in ear
        hp_onear_db_fft = hf.mag2db(hp_cue_onear_mean_mag)
        hp_inear_db_fft = hf.mag2db(hp_cue_inear_mean_mag)
        hp_fft_diff_db = np.subtract(hp_onear_db_fft,hp_inear_db_fft)
        #convert to mag
        hp_cue_onear_diff_mean_mag = hf.db2mag(hp_fft_diff_db)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(hp_cue_onear_diff_mean_mag,'hp_cue_onear_diff_mean_mag', normalise=1)
            
        #compare difference between over ear and on ear
        hp_onear_db_fft = hf.mag2db(hp_cue_onear_mean_mag)
        hp_overear_db_fft = hf.mag2db(hp_cue_overear_mean_mag)
        hp_fft_diff_db = np.subtract(hp_onear_db_fft,hp_overear_db_fft)
        #convert to mag
        hp_cue_onear_overear_diff_mean_mag = hf.db2mag(hp_fft_diff_db)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(hp_cue_onear_overear_diff_mean_mag,'hp_cue_onear_overear_diff_mean_mag', normalise=1)
  
        #
        # load pinna comp filter (FIR)
        #
        mat_fname = pjoin(CN.DATA_DIR_INT, 'headphone_pinna_comp_fir.mat')
        pinna_comp_mat = mat73.loadmat(mat_fname)
        pinna_comp_fir=np.zeros(CN.N_FFT)
        pinna_comp_fir[0:4096] = pinna_comp_mat['ash_hp_pinna_comp_fir'][0:4096]
        data_fft = np.fft.fft(pinna_comp_fir)
        mag_response=np.abs(data_fft)
        pinna_comp_db_fft = hf.mag2db(mag_response)
        #invert so that no longer inverse
        pinna_comp_db_pos = np.multiply(pinna_comp_db_fft,-1)
        #convert to mag
        pinna_comp_pos = hf.db2mag(pinna_comp_db_pos)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(pinna_comp_pos,'pinna_comp_pos', normalise=1)
  
        #average the estimates with weighting to get good approximate
        weightings = [0.16,0.24,0.60]#[0.12,0.30,0.58]
        hp_est_a_db_fft = hf.mag2db(drir_90_l_mean_comp)
        hp_est_b_db_fft = hf.mag2db(hp_cue_diff_mean_mag)
        hp_est_c_db_fft = hf.mag2db(pinna_comp_pos)
        hp_est_avg_db = fr_flat.copy()
        hp_est_avg_db = np.add(np.multiply(hp_est_a_db_fft,weightings[0]),hp_est_avg_db)
        hp_est_avg_db = np.add(np.multiply(hp_est_b_db_fft,weightings[1]),hp_est_avg_db)
        hp_est_avg_db = np.add(np.multiply(hp_est_c_db_fft,weightings[2]),hp_est_avg_db)
        #convert to mag
        hp_est_avg_mag = hf.db2mag(hp_est_avg_db)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(drir_90_l_mean_comp,'drir_90_l_mean_comp', normalise=1)
            hf.plot_data(hp_cue_diff_mean_mag,'hp_cue_diff_mean_mag', normalise=1)
            hf.plot_data(pinna_comp_pos,'pinna_comp_pos', normalise=1)
            hf.plot_data(hp_est_avg_mag,'hp_est_avg_mag', normalise=1)
    
        #create min phase inverse FIR
        pinna_comp_db_o = hf.mag2db(hp_est_avg_mag)
        #invert so that inverse
        pinna_comp_db_inv = np.multiply(pinna_comp_db_o,-1)
        #convert to mag
        pinna_comp_inv = hf.db2mag(pinna_comp_db_inv)
        #pinna_comp_inv_min_fir = hf.mag_to_min_fir(pinna_comp_inv, crop=1)
        pinna_comp_inv_min_fir = hf.build_min_phase_filter(smoothed_mag=pinna_comp_inv,  n_fft=CN.N_FFT)#new method
    
        #save numpy array for later use in BRIR generation functions
        
        npy_file_name =  'headphone_pinna_comp_fir.npy'
        brir_out_folder = CN.DATA_DIR_INT
        out_file_path = pjoin(brir_out_folder,npy_file_name)      
          
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        #np.save(out_file_path,pinna_comp_inv_min_fir)    
        
        log_string_a = 'Exported numpy file to: ' + out_file_path 
        hf.log_with_timestamp(log_string_a, gui_logger)
       

    
    except Exception as ex:
        log_string = 'Failed to complete BRIR processing'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error

   
def process_mono_cues_v2(gui_logger=None):
    """
    Function performs statistical analysis on monoaural cues
    """    

    ir_set='mono_cues_set_a'
    brir_ref_folder = pjoin(CN.DATA_DIR_INT, 'mono_cues', ir_set)
    
    n_fft=CN.N_FFT
    
    ash_samp_freq=44100
    output_wavs=1

    total_chan_brir=2
    
    #level ends
    level_f_a=450
    level_f_b=17000
  
    #impulse
    impulse=np.zeros(n_fft)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat = hf.mag2db(fr_flat_mag)
    
    
    try:
   
        #read in-ear headphone equalisation from WAV file
        filename = 'diffuse_field_eq_for_in_ear_headphones.wav'
        wav_fname = pjoin(CN.DATA_DIR_INT, filename)
        samplerate, data_addit_eq = wavfile.read(wav_fname)
        data_addit_eq = data_addit_eq / (2.**31)
        #convert to mag response
        in_ear_eq_fir=np.zeros(CN.N_FFT)
        in_ear_eq_fir[0:1024] = data_addit_eq[0:1024]
        data_fft = np.fft.fft(in_ear_eq_fir)#
        in_ear_eq_mag=np.abs(data_fft)
        in_ear_eq_db=hf.mag2db(in_ear_eq_mag)
   
    
        #
        # section for HP cues
        #
        ir_set='hp_cues'
        hp_folder = 'on_ear'
        hp_mag_in_folder = pjoin(CN.DATA_DIR_INT, 'mono_cues', ir_set,hp_folder)
        
        num_sets_avg = 0
        hp_fft_avg_db = fr_flat.copy()
        #loop through folders
        for root, dirs, files in os.walk(hp_mag_in_folder):
            for filename in files:
                if '.npy' in filename:
                    #read npy files
                    npy_file_path = pjoin(root, filename)
                    hp_current_mag_fft = np.load(npy_file_path)
                    hp_current_db_fft = hf.mag2db(hp_current_mag_fft)
                    hp_fft_avg_db = np.add(hp_fft_avg_db,hp_current_db_fft)
                    num_sets_avg = num_sets_avg+1

        #divide by total number of brirs
        hp_fft_avg_db = hp_fft_avg_db/num_sets_avg
        #convert to mag
        hptf_fft_avg_mag = hf.db2mag(hp_fft_avg_db)
        #level ends of spectrum
        hptf_fft_avg_mag_le = hf.level_spectrum_ends(hptf_fft_avg_mag, level_f_a, level_f_b, smooth_win = 20)#150
        #smoothing
        #octave smoothing
        hptf_fft_avg_mag_sm = hf.smooth_fft_octaves(data=hptf_fft_avg_mag_le, n_fft=n_fft, win_size_base = 20)
        hp_cue_onear_mean_mag = np.copy(hptf_fft_avg_mag_sm)
 
        if CN.PLOT_ENABLE == True:
            print(str(num_sets_avg))
            hf.plot_data(hptf_fft_avg_mag,'hptf_fft_avg_mag', normalise=1) 
            hf.plot_data(hptf_fft_avg_mag_sm,'hptf_fft_avg_mag_sm', normalise=1) 
        
        
        ir_set='hp_cues'
        hp_folder = 'over_ear'
        hp_mag_in_folder = pjoin(CN.DATA_DIR_INT, 'mono_cues', ir_set,hp_folder)
        
        num_sets_avg = 0
        hp_fft_avg_db = fr_flat.copy()
        #loop through folders
        for root, dirs, files in os.walk(hp_mag_in_folder):
            for filename in files:
                if '.npy' in filename:
                    #read npy files
                    npy_file_path = pjoin(root, filename)
                    hp_current_mag_fft = np.load(npy_file_path)
                    hp_current_db_fft = hf.mag2db(hp_current_mag_fft)
                    hp_fft_avg_db = np.add(hp_fft_avg_db,hp_current_db_fft)
                    num_sets_avg = num_sets_avg+1

        #divide by total number of brirs
        hp_fft_avg_db = hp_fft_avg_db/num_sets_avg
        #convert to mag
        hptf_fft_avg_mag = hf.db2mag(hp_fft_avg_db)
        #level ends of spectrum
        hptf_fft_avg_mag_le = hf.level_spectrum_ends(hptf_fft_avg_mag, level_f_a, level_f_b, smooth_win = 20)#150
        #smoothing
        #octave smoothing
        hptf_fft_avg_mag_sm = hf.smooth_fft_octaves(data=hptf_fft_avg_mag_le, n_fft=n_fft, win_size_base = 20)
        hp_cue_overear_mean_mag = np.copy(hptf_fft_avg_mag_sm)
 
        if CN.PLOT_ENABLE == True:
            print(str(num_sets_avg))
            hf.plot_data(hptf_fft_avg_mag,'hptf_fft_avg_mag', normalise=1) 
            hf.plot_data(hptf_fft_avg_mag_sm,'hptf_fft_avg_mag_sm', normalise=1) 
        
        ir_set='hp_cues'
        hp_folder = 'in_ear'
        hp_mag_in_folder = pjoin(CN.DATA_DIR_INT, 'mono_cues', ir_set,hp_folder)
        
        num_sets_avg = 0
        hp_fft_avg_db = fr_flat.copy()
        #loop through folders
        for root, dirs, files in os.walk(hp_mag_in_folder):
            for filename in files:
                if '.npy' in filename:
                    #read npy files
                    npy_file_path = pjoin(root, filename)
                    hp_current_mag_fft = np.load(npy_file_path)
                    hp_current_db_fft = hf.mag2db(hp_current_mag_fft)
                    hp_fft_avg_db = np.add(hp_fft_avg_db,hp_current_db_fft)
                    num_sets_avg = num_sets_avg+1

        #divide by total number of brirs
        hp_fft_avg_db = hp_fft_avg_db/num_sets_avg
        #convert to mag
        hptf_fft_avg_mag = hf.db2mag(hp_fft_avg_db)
        #level ends of spectrum
        hptf_fft_avg_mag_le = hf.level_spectrum_ends(hptf_fft_avg_mag, level_f_a, level_f_b, smooth_win = 20)#150
        #smoothing
        #octave smoothing
        hptf_fft_avg_mag_sm = hf.smooth_fft_octaves(data=hptf_fft_avg_mag_le, n_fft=n_fft, win_size_base = 20)
        hp_cue_inear_mean_mag = np.copy(hptf_fft_avg_mag_sm)
        if CN.PLOT_ENABLE == True:
            print(str(num_sets_avg))
            hf.plot_data(hptf_fft_avg_mag,'hptf_fft_avg_mag', normalise=1) 
            hf.plot_data(hptf_fft_avg_mag_sm,'hptf_fft_avg_mag_sm', normalise=1) 
        
        #compare difference between over ear and in ear
        hp_overear_db_fft = hf.mag2db(hp_cue_overear_mean_mag)
        hp_inear_db_fft = hf.mag2db(hp_cue_inear_mean_mag)
        hp_fft_diff_db = np.subtract(hp_overear_db_fft,hp_inear_db_fft)
        #convert to mag
        hp_cue_overear_diff_mean_mag = hf.db2mag(hp_fft_diff_db)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(hp_cue_overear_diff_mean_mag,'hp_cue_overear_diff_mean_mag', normalise=1)
            
        #compare difference between on ear and in ear
        hp_onear_db_fft = hf.mag2db(hp_cue_onear_mean_mag)
        hp_inear_db_fft = hf.mag2db(hp_cue_inear_mean_mag)
        hp_fft_diff_db = np.subtract(hp_onear_db_fft,hp_inear_db_fft)
        #convert to mag
        hp_cue_onear_diff_mean_mag = hf.db2mag(hp_fft_diff_db)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(hp_cue_onear_diff_mean_mag,'hp_cue_onear_diff_mean_mag', normalise=1)
            
        #compare difference between over ear and on ear
        hp_onear_db_fft = hf.mag2db(hp_cue_onear_mean_mag)
        hp_overear_db_fft = hf.mag2db(hp_cue_overear_mean_mag)
        hp_fft_diff_db = np.subtract(hp_onear_db_fft,hp_overear_db_fft)
        #apply additional EQ for DF headphones
        hp_fft_diff_db = np.subtract(hp_fft_diff_db,in_ear_eq_db*0.2)
        #convert to mag
        hp_cue_onear_overear_diff_mean_db = hp_fft_diff_db
        hp_cue_onear_overear_diff_mean_mag = hf.db2mag(hp_fft_diff_db)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(hp_cue_onear_overear_diff_mean_mag,'hp_cue_onear_overear_diff_mean_mag', normalise=1)
  
        #
        # load pinna comp filter (FIR)
        #
        npy_fname = pjoin(CN.DATA_DIR_INT, 'headphone_pinna_comp_fir.npy')
        pinna_comp_fir_short = np.load(npy_fname)
        pinna_comp_fir=np.zeros(CN.N_FFT)
        pinna_comp_fir[0:4096] = pinna_comp_fir_short[0:4096]
        data_fft = np.fft.fft(pinna_comp_fir)
        mag_response=np.abs(data_fft)
        pinna_comp_db_fft = hf.mag2db(mag_response)
        #invert so that no longer inverse
        pinna_comp_db_pos = np.multiply(pinna_comp_db_fft,-1)
        #convert to mag
        pinna_comp_pos = hf.db2mag(pinna_comp_db_pos)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(pinna_comp_pos,'pinna_comp_pos', normalise=1)
  
    
        #average the estimates with weighting to get good approximate
        weightings = [0.70,0.30]#[0.75,0.25] [0.70,0.30] [0.67,0.33]
        hp_est_a_db_fft = hf.mag2db(pinna_comp_pos)
        hp_est_b_db_fft = hf.mag2db(hp_cue_overear_diff_mean_mag)
        hp_est_avg_db = fr_flat.copy()
        hp_est_avg_db = np.add(np.multiply(hp_est_a_db_fft,weightings[0]),hp_est_avg_db)
        hp_est_avg_db = np.add(np.multiply(hp_est_b_db_fft,weightings[1]),hp_est_avg_db)
        #convert to mag
        hp_est_avg_mag = hf.db2mag(hp_est_avg_db)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(hp_est_avg_mag,'hp_est_avg_mag', normalise=1)
        
        
        
  
        # load additional headphone eq
        filename = 'additional_comp_for_over_&_on_ear_headphones.wav'
        wav_fname = pjoin(CN.DATA_DIR_INT, filename)
        samplerate, data_addit_eq = wavfile.read(wav_fname)
        data_addit_eq = data_addit_eq / (2.**31)
        hp_comp_fir=np.zeros(CN.N_FFT)
        hp_comp_fir[0:1024] = data_addit_eq[0:1024]
        data_fft = np.fft.fft(hp_comp_fir)
        mag_response=np.abs(data_fft)
        db_response = hf.mag2db(mag_response)
        #invert so that no longer inverse
        db_response = np.multiply(db_response,-1)
        over_ear_high_db = np.add(hp_est_avg_db,db_response)
        
        #on ear eq
        on_ear_low_db = np.add(hp_est_avg_db,hp_cue_onear_overear_diff_mean_db)
        on_ear_high_db = np.add(on_ear_low_db,db_response)
        
        # load additional headphone eq
        filename = 'additional_comp_for_in_ear_headphones.wav'
        wav_fname = pjoin(CN.DATA_DIR_INT, filename)
        samplerate, data_addit_eq = wavfile.read(wav_fname)
        data_addit_eq = data_addit_eq / (2.**31)
        hp_comp_fir=np.zeros(CN.N_FFT)
        hp_comp_fir[0:1024] = data_addit_eq[0:1024]
        data_fft = np.fft.fft(hp_comp_fir)
        mag_response=np.abs(data_fft)
        db_response = hf.mag2db(mag_response)
        #invert so that no longer inverse
        db_response = np.multiply(db_response,-1)
        in_ear_high_db = np.add(fr_flat,db_response)
        
     
        #final curves
        in_ear_low_mag = fr_flat_mag.copy()
        in_ear_high_mag = hf.db2mag(in_ear_high_db)
        over_ear_low_mag = hp_est_avg_mag.copy()
        over_ear_high_mag = over_ear_high_db.copy()
        on_ear_low_mag = hf.db2mag(on_ear_low_db)
        on_ear_high_mag = hf.db2mag(on_ear_high_db)
        
        if CN.PLOT_ENABLE == True:
            hf.plot_data(in_ear_low_mag,'in_ear_low_mag', normalise=1)
            hf.plot_data(in_ear_high_mag,'in_ear_high_mag', normalise=1)
            hf.plot_data(over_ear_low_mag,'over_ear_low_mag', normalise=1)
            hf.plot_data(over_ear_high_mag,'over_ear_high_mag', normalise=1)
            hf.plot_data(on_ear_low_mag,'on_ear_low_mag', normalise=1)
            hf.plot_data(on_ear_high_mag,'on_ear_high_mag', normalise=1)
    
        fir_length=1024
        num_filters=6
        npy_out=np.zeros((num_filters,fir_length))      
  
        #create min phase inverse FIR
        in_ear_high_db = hf.mag2db(in_ear_high_mag)
        in_ear_low_db = hf.mag2db(in_ear_low_mag)
        over_ear_high_db = hf.mag2db(over_ear_high_mag)
        over_ear_low_db = hf.mag2db(over_ear_low_mag)
        on_ear_high_db = hf.mag2db(on_ear_high_mag)
        on_ear_low_db = hf.mag2db(on_ear_low_mag)
        #invert so that inverse
        in_ear_high_db_inv = np.multiply(in_ear_high_db,-1)
        in_ear_low_db_inv = in_ear_low_db
        over_ear_high_db_inv = np.multiply(over_ear_high_db,-1)
        over_ear_low_db_inv = np.multiply(over_ear_low_db,-1)
        on_ear_high_db_inv = np.multiply(on_ear_high_db,-1)
        on_ear_low_db_inv = np.multiply(on_ear_low_db,-1)
        #convert to mag
        in_ear_high_inv = hf.db2mag(in_ear_high_db_inv)
        in_ear_low_inv = hf.db2mag(in_ear_low_db_inv)
        over_ear_high_inv = hf.db2mag(over_ear_high_db_inv)
        over_ear_low_inv = hf.db2mag(over_ear_low_db_inv)
        on_ear_high_inv = hf.db2mag(on_ear_high_db_inv)
        on_ear_low_inv = hf.db2mag(on_ear_low_db_inv)
        #min phase FIRs
        # in_ear_high_inv_min_fir = hf.mag_to_min_fir(in_ear_high_inv, crop=1, out_win_size=fir_length)
        # in_ear_low_inv_min_fir = hf.mag_to_min_fir(in_ear_low_inv, crop=1, out_win_size=fir_length)
        # over_ear_high_inv_min_fir = hf.mag_to_min_fir(over_ear_high_inv, crop=1, out_win_size=fir_length)
        # over_ear_low_inv_min_fir = hf.mag_to_min_fir(over_ear_low_inv, crop=1, out_win_size=fir_length)
        # on_ear_high_inv_min_fir = hf.mag_to_min_fir(on_ear_high_inv, crop=1, out_win_size=fir_length)
        # on_ear_low_inv_min_fir = hf.mag_to_min_fir(on_ear_low_inv, crop=1, out_win_size=fir_length)
        #new method
        in_ear_high_inv_min_fir = hf.build_min_phase_filter(in_ear_high_inv,  truncate_len=fir_length)
        in_ear_low_inv_min_fir = hf.build_min_phase_filter(in_ear_low_inv,  truncate_len=fir_length)
        over_ear_high_inv_min_fir = hf.build_min_phase_filter(over_ear_high_inv,  truncate_len=fir_length)
        over_ear_low_inv_min_fir = hf.build_min_phase_filter(over_ear_low_inv,  truncate_len=fir_length)
        on_ear_high_inv_min_fir = hf.build_min_phase_filter(on_ear_high_inv,  truncate_len=fir_length)
        on_ear_low_inv_min_fir = hf.build_min_phase_filter(on_ear_low_inv,  truncate_len=fir_length)
        
    
        #HP_COMP_LIST_SHORT = ['In-Ear-High','In-Ear-Low','Over-Ear-High','Over-Ear-Low','On-Ear-High','On-Ear-Low']
    
        #store in a npy array
        npy_out[0,:]=in_ear_high_inv_min_fir
        npy_out[1,:]=in_ear_low_inv_min_fir
        npy_out[2,:]=over_ear_high_inv_min_fir
        npy_out[3,:]=over_ear_low_inv_min_fir
        npy_out[4,:]=on_ear_high_inv_min_fir
        npy_out[5,:]=on_ear_low_inv_min_fir
    
    
    
    
        # #save numpy array for later use in BRIR generation functions
        
        npy_file_name =  'headphone_ear_comp_dataset.npy'
        brir_out_folder = CN.DATA_DIR_INT
        out_file_path = pjoin(brir_out_folder,npy_file_name)      
          
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        np.save(out_file_path,npy_out)    
        
        log_string_a = 'Exported numpy file to: ' + out_file_path 
        hf.log_with_timestamp(log_string_a, gui_logger)
       

    
    except Exception as ex:
        log_string = 'Failed to complete BRIR processing'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error

def process_mono_cues_v3(gui_logger=None):
    """
    Function performs statistical analysis on monoaural cues
    """    

    ir_set='mono_cues_set_a'
    brir_ref_folder = pjoin(CN.DATA_DIR_INT, 'mono_cues', ir_set)
    
    n_fft=CN.N_FFT
    
    ash_samp_freq=44100
    output_wavs=1

    total_chan_brir=2
    
    #level ends
    level_f_a=450
    level_f_b=17000
  
    #impulse
    impulse=np.zeros(n_fft)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat_db = hf.mag2db(fr_flat_mag)
    
    
    try:
   
        #read in-ear headphone equalisation from WAV file
        filename = 'diffuse_field_eq_for_in_ear_headphones.wav'
        wav_fname = pjoin(CN.DATA_DIR_INT, filename)
        samplerate, data_addit_eq = wavfile.read(wav_fname)
        data_addit_eq = data_addit_eq / (2.**31)
        #convert to mag response
        in_ear_eq_fir=np.zeros(CN.N_FFT)
        in_ear_eq_fir[0:1024] = data_addit_eq[0:1024]
        data_fft = np.fft.fft(in_ear_eq_fir)#
        in_ear_eq_mag=np.abs(data_fft)
        in_ear_eq_db=hf.mag2db(in_ear_eq_mag)
   
    
        #
        # section for HP cues
        #
        ir_set='hp_cues'
        hp_folder = 'on_ear'
        hp_mag_in_folder = pjoin(CN.DATA_DIR_INT, 'mono_cues', ir_set,hp_folder)
        
        num_sets_avg = 0
        hp_fft_avg_db = fr_flat_db.copy()
        #loop through folders
        for root, dirs, files in os.walk(hp_mag_in_folder):
            for filename in files:
                if '.npy' in filename:
                    #read npy files
                    npy_file_path = pjoin(root, filename)
                    hp_current_mag_fft = np.load(npy_file_path)
                    hp_current_db_fft = hf.mag2db(hp_current_mag_fft)
                    hp_fft_avg_db = np.add(hp_fft_avg_db,hp_current_db_fft)
                    num_sets_avg = num_sets_avg+1

        #divide by total number of brirs
        hp_fft_avg_db = hp_fft_avg_db/num_sets_avg
        #convert to mag
        hptf_fft_avg_mag = hf.db2mag(hp_fft_avg_db)
        #level ends of spectrum
        hptf_fft_avg_mag_le = hf.level_spectrum_ends(hptf_fft_avg_mag, level_f_a, level_f_b, smooth_win = 20)#150
        #smoothing
        #octave smoothing
        hptf_fft_avg_mag_sm = hf.smooth_fft_octaves(data=hptf_fft_avg_mag_le, n_fft=n_fft, win_size_base = 20)
        hp_cue_onear_mean_mag = np.copy(hptf_fft_avg_mag_sm)
 
        if CN.PLOT_ENABLE == True:
            print(str(num_sets_avg))
            hf.plot_data(hptf_fft_avg_mag,'hptf_fft_avg_mag', normalise=1) 
            hf.plot_data(hptf_fft_avg_mag_sm,'hptf_fft_avg_mag_sm', normalise=1) 
        
        
        ir_set='hp_cues'
        hp_folder = 'over_ear'
        hp_mag_in_folder = pjoin(CN.DATA_DIR_INT, 'mono_cues', ir_set,hp_folder)
        
        num_sets_avg = 0
        hp_fft_avg_db = fr_flat_db.copy()
        #loop through folders
        for root, dirs, files in os.walk(hp_mag_in_folder):
            for filename in files:
                if '.npy' in filename:
                    #read npy files
                    npy_file_path = pjoin(root, filename)
                    hp_current_mag_fft = np.load(npy_file_path)
                    hp_current_db_fft = hf.mag2db(hp_current_mag_fft)
                    hp_fft_avg_db = np.add(hp_fft_avg_db,hp_current_db_fft)
                    num_sets_avg = num_sets_avg+1

        #divide by total number of brirs
        hp_fft_avg_db = hp_fft_avg_db/num_sets_avg
        #convert to mag
        hptf_fft_avg_mag = hf.db2mag(hp_fft_avg_db)
        #level ends of spectrum
        hptf_fft_avg_mag_le = hf.level_spectrum_ends(hptf_fft_avg_mag, level_f_a, level_f_b, smooth_win = 20)#150
        #smoothing
        #octave smoothing
        hptf_fft_avg_mag_sm = hf.smooth_fft_octaves(data=hptf_fft_avg_mag_le, n_fft=n_fft, win_size_base = 20)
        hp_cue_overear_mean_mag = np.copy(hptf_fft_avg_mag_sm)
 
        if CN.PLOT_ENABLE == True:
            print(str(num_sets_avg))
            hf.plot_data(hptf_fft_avg_mag,'hptf_fft_avg_mag', normalise=1) 
            hf.plot_data(hptf_fft_avg_mag_sm,'hptf_fft_avg_mag_sm', normalise=1) 
        
        ir_set='hp_cues'
        hp_folder = 'in_ear'
        hp_mag_in_folder = pjoin(CN.DATA_DIR_INT, 'mono_cues', ir_set,hp_folder)
        
        num_sets_avg = 0
        hp_fft_avg_db = fr_flat_db.copy()
        #loop through folders
        for root, dirs, files in os.walk(hp_mag_in_folder):
            for filename in files:
                if '.npy' in filename:
                    #read npy files
                    npy_file_path = pjoin(root, filename)
                    hp_current_mag_fft = np.load(npy_file_path)
                    hp_current_db_fft = hf.mag2db(hp_current_mag_fft)
                    hp_fft_avg_db = np.add(hp_fft_avg_db,hp_current_db_fft)
                    num_sets_avg = num_sets_avg+1

        #divide by total number of brirs
        hp_fft_avg_db = hp_fft_avg_db/num_sets_avg
        #convert to mag
        hptf_fft_avg_mag = hf.db2mag(hp_fft_avg_db)
        #level ends of spectrum
        hptf_fft_avg_mag_le = hf.level_spectrum_ends(hptf_fft_avg_mag, level_f_a, level_f_b, smooth_win = 20)#150
        #smoothing
        #octave smoothing
        hptf_fft_avg_mag_sm = hf.smooth_fft_octaves(data=hptf_fft_avg_mag_le, n_fft=n_fft, win_size_base = 20)
        hp_cue_inear_mean_mag = np.copy(hptf_fft_avg_mag_sm)
        if CN.PLOT_ENABLE == True:
            print(str(num_sets_avg))
            hf.plot_data(hptf_fft_avg_mag,'hptf_fft_avg_mag', normalise=1) 
            hf.plot_data(hptf_fft_avg_mag_sm,'hptf_fft_avg_mag_sm', normalise=1) 
        
        #compare difference between over ear and in ear
        hp_overear_db_fft = hf.mag2db(hp_cue_overear_mean_mag)
        hp_inear_db_fft = hf.mag2db(hp_cue_inear_mean_mag)
        hp_fft_diff_db = np.subtract(hp_overear_db_fft,hp_inear_db_fft)
        #convert to mag
        hp_cue_overear_diff_mean_mag = hf.db2mag(hp_fft_diff_db)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(hp_cue_overear_diff_mean_mag,'hp_cue_overear_diff_mean_mag', normalise=1)
            
        #compare difference between on ear and in ear
        hp_onear_db_fft = hf.mag2db(hp_cue_onear_mean_mag)
        hp_inear_db_fft = hf.mag2db(hp_cue_inear_mean_mag)
        hp_fft_diff_db = np.subtract(hp_onear_db_fft,hp_inear_db_fft)
        #convert to mag
        hp_cue_onear_diff_mean_mag = hf.db2mag(hp_fft_diff_db)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(hp_cue_onear_diff_mean_mag,'hp_cue_onear_diff_mean_mag', normalise=1)
            
        #compare difference between over ear and on ear
        hp_onear_db_fft = hf.mag2db(hp_cue_onear_mean_mag)
        hp_overear_db_fft = hf.mag2db(hp_cue_overear_mean_mag)
        hp_fft_diff_db = np.subtract(hp_onear_db_fft,hp_overear_db_fft)
        #apply additional EQ for DF headphones
        hp_fft_diff_db = np.subtract(hp_fft_diff_db,in_ear_eq_db*0.2)
        #convert to mag
        hp_cue_onear_overear_diff_mean_db = hp_fft_diff_db
        hp_cue_onear_overear_diff_mean_mag = hf.db2mag(hp_fft_diff_db)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(hp_cue_onear_overear_diff_mean_mag,'hp_cue_onear_overear_diff_mean_mag', normalise=1)
  
        #
        # load pinna comp filter (FIR)
        #
        npy_fname = pjoin(CN.DATA_DIR_INT, 'headphone_pinna_comp_fir.npy')
        pinna_comp_fir_short = np.load(npy_fname)
        pinna_comp_fir=np.zeros(CN.N_FFT)
        pinna_comp_fir[0:4096] = pinna_comp_fir_short[0:4096]
        data_fft = np.fft.fft(pinna_comp_fir)
        mag_response=np.abs(data_fft)
        pinna_comp_db_fft = hf.mag2db(mag_response)
        #invert so that no longer inverse
        pinna_comp_db_pos = np.multiply(pinna_comp_db_fft,-1)
        #convert to mag
        pinna_comp_pos = hf.db2mag(pinna_comp_db_pos)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(pinna_comp_pos,'pinna_comp_pos', normalise=1)
  
    
        #average the estimates with weighting to get good approximate
        weightings = [0.70,0.30]#[0.75,0.25] [0.70,0.30] [0.67,0.33]
        hp_est_a_db_fft = hf.mag2db(pinna_comp_pos)
        hp_est_b_db_fft = hf.mag2db(hp_cue_overear_diff_mean_mag)
        hp_est_avg_db = fr_flat_db.copy()
        hp_est_avg_db = np.add(np.multiply(hp_est_a_db_fft,weightings[0]),hp_est_avg_db)
        hp_est_avg_db = np.add(np.multiply(hp_est_b_db_fft,weightings[1]),hp_est_avg_db)
        #convert to mag
        hp_est_avg_mag = hf.db2mag(hp_est_avg_db)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(hp_est_avg_mag,'hp_est_avg_mag', normalise=1)
        
        
        
  
        # load additional headphone eq
        filename = 'additional_comp_for_over_&_on_ear_headphones.wav'
        wav_fname = pjoin(CN.DATA_DIR_INT, filename)
        samplerate, data_addit_eq = wavfile.read(wav_fname)
        data_addit_eq = data_addit_eq / (2.**31)
        hp_comp_fir=np.zeros(CN.N_FFT)
        hp_comp_fir[0:1024] = data_addit_eq[0:1024]
        data_fft = np.fft.fft(hp_comp_fir)
        mag_response=np.abs(data_fft)
        db_response = hf.mag2db(mag_response)
        #invert so that no longer inverse
        db_response = np.multiply(db_response,-1)
        over_ear_add_eq_db = db_response
        over_ear_add_eq_mag = hf.db2mag(db_response)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(over_ear_add_eq_mag,'over_ear_add_eq_mag', normalise=1)

        
        #on ear eq
        on_ear_low_db = np.add(hp_est_avg_db,hp_cue_onear_overear_diff_mean_db)
        on_ear_high_db = np.add(on_ear_low_db,db_response)
        
        # load additional headphone eq
        filename = 'additional_comp_for_in_ear_headphones.wav'
        wav_fname = pjoin(CN.DATA_DIR_INT, filename)
        samplerate, data_addit_eq = wavfile.read(wav_fname)
        data_addit_eq = data_addit_eq / (2.**31)
        hp_comp_fir=np.zeros(CN.N_FFT)
        hp_comp_fir[0:1024] = data_addit_eq[0:1024]
        data_fft = np.fft.fft(hp_comp_fir)
        mag_response=np.abs(data_fft)
        db_response = hf.mag2db(mag_response)
        #invert so that no longer inverse
        db_response = np.multiply(db_response,-1)
        in_ear_high_db = np.add(fr_flat_db,db_response)
        
        
        #20250602: new high strength EQ for over ear
        samp_freq_ash=CN.FS
        #a
        #sequence of peaking filters
        filter_type="peaking"
        fc=2100
        sr=samp_freq_ash
        q=2.0
        gain_db=-4.0
        pyquad = pyquadfilter.PyQuadFilter(sr)
        pyquad.set_params(filter_type, fc, q, gain_db)
        oe_hs_comp_a_fir = np.copy(impulse)
        oe_hs_comp_a_fir = pyquad.filter(oe_hs_comp_a_fir)
        #
        filter_type="peaking"
        fc=4000
        sr=samp_freq_ash
        q=2.0
        gain_db=2.0
        pyquad = pyquadfilter.PyQuadFilter(sr)
        pyquad.set_params(filter_type, fc, q, gain_db)
        oe_hs_comp_a_fir = pyquad.filter(oe_hs_comp_a_fir)
        #
        #grab frequency response
        data_fft = np.fft.fft(oe_hs_comp_a_fir[0][0:n_fft])
        oe_hs_comp_a_mag=np.abs(data_fft)
        oe_hs_comp_a_db = hf.mag2db(oe_hs_comp_a_mag)
        
        #b
        #sequence of peaking filters
        filter_type="peaking"
        fc=2200
        sr=samp_freq_ash
        q=3.0
        gain_db=-4.0
        pyquad = pyquadfilter.PyQuadFilter(sr)
        pyquad.set_params(filter_type, fc, q, gain_db)
        oe_hs_comp_b_fir = np.copy(impulse)
        oe_hs_comp_b_fir = pyquad.filter(oe_hs_comp_b_fir)
        #
        filter_type="peaking"
        fc=5500
        sr=samp_freq_ash
        q=2.5
        gain_db=1.5
        pyquad = pyquadfilter.PyQuadFilter(sr)
        pyquad.set_params(filter_type, fc, q, gain_db)
        oe_hs_comp_b_fir = pyquad.filter(oe_hs_comp_b_fir)
        #
        filter_type="peaking"
        fc=6500
        sr=samp_freq_ash
        q=3.0
        gain_db=3.0
        pyquad = pyquadfilter.PyQuadFilter(sr)
        pyquad.set_params(filter_type, fc, q, gain_db)
        oe_hs_comp_b_fir = pyquad.filter(oe_hs_comp_b_fir)
        #
        filter_type="peaking"
        fc=10200
        sr=samp_freq_ash
        q=6.0
        gain_db=-4.0
        pyquad = pyquadfilter.PyQuadFilter(sr)
        pyquad.set_params(filter_type, fc, q, gain_db)
        oe_hs_comp_b_fir = pyquad.filter(oe_hs_comp_b_fir)
        #
        filter_type="peaking"
        fc=7500
        sr=samp_freq_ash
        q=5.0
        gain_db=2.0
        pyquad = pyquadfilter.PyQuadFilter(sr)
        pyquad.set_params(filter_type, fc, q, gain_db)
        oe_hs_comp_b_fir = pyquad.filter(oe_hs_comp_b_fir)
        #grab frequency response
        data_fft = np.fft.fft(oe_hs_comp_b_fir[0][0:n_fft])
        oe_hs_comp_b_mag=np.abs(data_fft)
        oe_hs_comp_b_db = hf.mag2db(oe_hs_comp_b_mag)
        
        #c
        #sequence of peaking filters
        filter_type="peaking"
        fc=2900
        sr=samp_freq_ash
        q=2.5
        gain_db=-5.0
        pyquad = pyquadfilter.PyQuadFilter(sr)
        pyquad.set_params(filter_type, fc, q, gain_db)
        oe_hs_comp_c_fir = np.copy(impulse)
        oe_hs_comp_c_fir = pyquad.filter(oe_hs_comp_c_fir)
        #
        filter_type="peaking"
        fc=5500
        sr=samp_freq_ash
        q=2.0
        gain_db=2.0
        pyquad = pyquadfilter.PyQuadFilter(sr)
        pyquad.set_params(filter_type, fc, q, gain_db)
        oe_hs_comp_c_fir = pyquad.filter(oe_hs_comp_c_fir)
        #
        filter_type="peaking"
        fc=5000
        sr=samp_freq_ash
        q=3.0
        gain_db=1.5
        pyquad = pyquadfilter.PyQuadFilter(sr)
        pyquad.set_params(filter_type, fc, q, gain_db)
        oe_hs_comp_c_fir = pyquad.filter(oe_hs_comp_c_fir)
        #
        filter_type="peaking"
        fc=3400
        sr=samp_freq_ash
        q=5.0
        gain_db=2.0
        pyquad = pyquadfilter.PyQuadFilter(sr)
        pyquad.set_params(filter_type, fc, q, gain_db)
        oe_hs_comp_c_fir = pyquad.filter(oe_hs_comp_c_fir)
        #
        filter_type="peaking"
        fc=10500
        sr=samp_freq_ash
        q=5.0
        gain_db=-3.0
        pyquad = pyquadfilter.PyQuadFilter(sr)
        pyquad.set_params(filter_type, fc, q, gain_db)
        oe_hs_comp_c_fir = pyquad.filter(oe_hs_comp_c_fir)
        #grab frequency response
        data_fft = np.fft.fft(oe_hs_comp_c_fir[0][0:n_fft])
        oe_hs_comp_c_mag=np.abs(data_fft)
        oe_hs_comp_c_db = hf.mag2db(oe_hs_comp_c_mag)
        
        if CN.PLOT_ENABLE == True:
            #print(impulse.shape)
            #print(oe_hs_comp_a_fir.shape)
            #print(oe_hs_comp_a_mag.shape)
            hf.plot_data(oe_hs_comp_a_mag,'oe_hs_comp_a_mag', normalise=1)
            hf.plot_data(oe_hs_comp_b_mag,'oe_hs_comp_b_mag', normalise=1)
            hf.plot_data(oe_hs_comp_c_mag,'oe_hs_comp_c_mag', normalise=1)
            
        #average the 3 new estimates with weighting to get good approximate
        weightings = [0.40,0.30,0.30]#
        oe_hs_comp_avg_db = fr_flat_db.copy()
        oe_hs_comp_avg_db = np.add(np.multiply(oe_hs_comp_a_db,weightings[0]),oe_hs_comp_avg_db)
        oe_hs_comp_avg_db = np.add(np.multiply(oe_hs_comp_b_db,weightings[1]),oe_hs_comp_avg_db)
        oe_hs_comp_avg_db = np.add(np.multiply(oe_hs_comp_c_db,weightings[2]),oe_hs_comp_avg_db)
        #invert so that no longer inverse
        oe_hs_comp_avg_db = np.multiply(oe_hs_comp_avg_db,-1)
        #convert to mag
        oe_hs_comp_avg_mag = hf.db2mag(oe_hs_comp_avg_db)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(oe_hs_comp_avg_mag,'oe_hs_comp_avg_mag', normalise=1)
        
        #average the new and old estimates with weighting to get good approximate
        weightings = [0.50,0.50]#
        oe_hs_new_avg_db = fr_flat_db.copy()
        oe_hs_new_avg_db = np.add(np.multiply(oe_hs_comp_avg_db,weightings[0]),oe_hs_new_avg_db)
        oe_hs_new_avg_db = np.add(np.multiply(over_ear_add_eq_db,weightings[1]),oe_hs_new_avg_db)
        #convert to mag
        oe_hs_new_avg_mag = hf.db2mag(oe_hs_new_avg_db)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(oe_hs_new_avg_mag,'oe_hs_new_avg_mag', normalise=1)
            
        over_ear_high_db = np.add(hp_est_avg_db,oe_hs_new_avg_db)
        over_ear_high_mag = hf.db2mag(over_ear_high_db)
        if CN.PLOT_ENABLE == True:
            hf.plot_data(over_ear_high_mag,'over_ear_high_mag', normalise=1)
     
        #final curves
        in_ear_low_mag = fr_flat_mag.copy()
        in_ear_high_mag = hf.db2mag(in_ear_high_db)
        over_ear_low_mag = hp_est_avg_mag.copy()
        over_ear_high_mag = hf.db2mag(over_ear_high_db)
        on_ear_low_mag = hf.db2mag(on_ear_low_db)
        on_ear_high_mag = hf.db2mag(on_ear_high_db)
        
        if CN.PLOT_ENABLE == True:
            hf.plot_data(in_ear_low_mag,'in_ear_low_mag', normalise=1)
            hf.plot_data(in_ear_high_mag,'in_ear_high_mag', normalise=1)
            hf.plot_data(over_ear_low_mag,'over_ear_low_mag', normalise=1)
            hf.plot_data(over_ear_high_mag,'over_ear_high_mag', normalise=1)
            hf.plot_data(on_ear_low_mag,'on_ear_low_mag', normalise=1)
            hf.plot_data(on_ear_high_mag,'on_ear_high_mag', normalise=1)
    
        fir_length=1024
        num_filters=6
        npy_out=np.zeros((num_filters,fir_length))      
  
        #create min phase inverse FIR
        in_ear_high_db = hf.mag2db(in_ear_high_mag)
        in_ear_low_db = hf.mag2db(in_ear_low_mag)
        over_ear_high_db = hf.mag2db(over_ear_high_mag)
        over_ear_low_db = hf.mag2db(over_ear_low_mag)
        on_ear_high_db = hf.mag2db(on_ear_high_mag)
        on_ear_low_db = hf.mag2db(on_ear_low_mag)
        #invert so that inverse
        in_ear_high_db_inv = np.multiply(in_ear_high_db,-1)
        in_ear_low_db_inv = in_ear_low_db
        over_ear_high_db_inv = np.multiply(over_ear_high_db,-1)
        over_ear_low_db_inv = np.multiply(over_ear_low_db,-1)
        on_ear_high_db_inv = np.multiply(on_ear_high_db,-1)
        on_ear_low_db_inv = np.multiply(on_ear_low_db,-1)
        #convert to mag
        in_ear_high_inv = hf.db2mag(in_ear_high_db_inv)
        in_ear_low_inv = hf.db2mag(in_ear_low_db_inv)
        over_ear_high_inv = hf.db2mag(over_ear_high_db_inv)
        over_ear_low_inv = hf.db2mag(over_ear_low_db_inv)
        on_ear_high_inv = hf.db2mag(on_ear_high_db_inv)
        on_ear_low_inv = hf.db2mag(on_ear_low_db_inv)
        #min phase FIRs
        #new method
        in_ear_high_inv_min_fir = hf.build_min_phase_filter(in_ear_high_inv,  truncate_len=fir_length)
        in_ear_low_inv_min_fir = hf.build_min_phase_filter(in_ear_low_inv,  truncate_len=fir_length)
        over_ear_high_inv_min_fir = hf.build_min_phase_filter(over_ear_high_inv,  truncate_len=fir_length)
        over_ear_low_inv_min_fir = hf.build_min_phase_filter(over_ear_low_inv,  truncate_len=fir_length)
        on_ear_high_inv_min_fir = hf.build_min_phase_filter(on_ear_high_inv,  truncate_len=fir_length)
        on_ear_low_inv_min_fir = hf.build_min_phase_filter(on_ear_low_inv,  truncate_len=fir_length)
        
    
        #HP_COMP_LIST_SHORT = ['In-Ear-High','In-Ear-Low','Over-Ear-High','Over-Ear-Low','On-Ear-High','On-Ear-Low']
    
        #store in a npy array
        npy_out[0,:]=in_ear_high_inv_min_fir
        npy_out[1,:]=in_ear_low_inv_min_fir
        npy_out[2,:]=over_ear_high_inv_min_fir
        npy_out[3,:]=over_ear_low_inv_min_fir
        npy_out[4,:]=on_ear_high_inv_min_fir
        npy_out[5,:]=on_ear_low_inv_min_fir
    
    
    
    
        # #save numpy array for later use in BRIR generation functions
        
        npy_file_name =  'headphone_ear_comp_dataset.npy'
        brir_out_folder = CN.DATA_DIR_INT
        out_file_path = pjoin(brir_out_folder,npy_file_name)      
          
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        np.save(out_file_path,npy_out)    
        
        log_string_a = 'Exported numpy file to: ' + out_file_path 
        hf.log_with_timestamp(log_string_a, gui_logger)
       

    
    except Exception as ex:
        log_string = 'Failed to complete BRIR processing'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error


def check_download_max_hrtf_datasets(download=False, gui_logger=None, report_progress=0):
    """ 
    Function checks if max resolution hrtf datasets are locally available, if not found it will download from google drive
    Returns:
       Status code: 0 = Success, 1 = Failure, 2 = Cancelled
    """
    status=1
    message=''
    
    try:
        
        #log results
        log_string = 'Checking for missing datasets'
        hf.log_with_timestamp(log_string, gui_logger)
        
        #for each max res dataset
        brir_hrtf_dataset_list_loaded = CN.HRTF_TYPE_DATASET_DICT.get('Dummy Head - Max Resolution')
        for dataset in brir_hrtf_dataset_list_loaded:
            
            #for each hrtf in that dataset
            hrtf_list_new = hrir_processing.get_listener_list(listener_type='Dummy Head / Head & Torso Simulator', dataset_name=dataset, max_res_only=True)
            for hrtf in hrtf_list_new:
                #url
                gdrive_url = hrir_processing.get_gdrive_url_max(listener_type='Dummy Head / Head & Torso Simulator', dataset_name=dataset, name_gui=hrtf )
                
                update_required=True
        
                #expected file location
                brir_hrtf_short = hrir_processing.get_name_short(listener_type='Dummy Head / Head & Torso Simulator', dataset_name=dataset, name_gui=hrtf )
                hrir_dir_base = CN.DATA_DIR_HRIR_NPY_DH
                sub_directory = 'm'
                #join spatial res subdirectory
                hrir_dir_sr = pjoin(hrir_dir_base, sub_directory)
                # Create dataset subdirectory
                hrir_dir_ds = pjoin(hrir_dir_sr, dataset)
                # full filename
                dl_file = pjoin(hrir_dir_ds, f"{brir_hrtf_short}.npy")
                
                # check if file exists
                if os.path.exists(dl_file):
                    update_required = False
        
                #file already available
                if update_required==False:
                    log_string = brir_hrtf_short + '.npy found'
                    hf.log_with_timestamp(log_string, gui_logger)
                        
                #if download updates enabled
                #download dataset from gdrive and place into relevant folder
                if download == True and update_required == True:
                    try:
                        if report_progress > 0:
                            message = 'Downloading required dataset(s)'
                            hf.update_gui_progress(report_progress=report_progress, message=message)
                    except:
                        pass
                    
                    log_string = 'Downloading ' + brir_hrtf_short + '.npy'
                    hf.log_with_timestamp(log_string, gui_logger)
           
                    gdrive_link = gdrive_url
                    hf.download_file(gdrive_link, dl_file, gui_logger=gui_logger)
                    #gdown.download(gdrive_link, dl_file, fuzzy=True)
                    
                    log_string = brir_hrtf_short + '.npy' + ' downloaded and saved to: ' + dl_file
                    hf.log_with_timestamp(log_string, gui_logger)
                    
                status=0#success if reached this far
       
        

    except Exception as ex:
        log_string = 'Failed to validate versions or update data'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
    
    
    return status







def get_ac_f_crossover(name_src="", gui_logger=None):
    """
    Retrieves a crossover freq (int) for lf integration for a specific acoustic space
    
    Args:
        name_src (str, optional): Name of acoustic space
        gui_logger (object, optional): A logger object for GUI-related logging. Defaults to None.
    
    Returns:
        fc: crossover freq (int), order (int)
    """
    
    fc = CN.F_CROSSOVER
    order = CN.ORDER
    
    try:

        #load lists from csv file
        try:
        
            #directories
            csv_directory = pjoin(CN.DATA_DIR_INT, 'reverberation')
            #read metadata from csv. Expects reverberation_metadata.csv 
            metadata_file_name = 'reverberation_metadata.csv'
            metadata_file = pjoin(csv_directory, metadata_file_name)
            with open(metadata_file, encoding='utf-8-sig', newline='') as inputfile:
                reader = DictReader(inputfile)
                for row in reader:#rows 2 and onward
                    #store each row as a dictionary
                    #append to list of dictionaries
                    if row.get('name_src') == name_src: #added check
                        fc = int(row.get('f_crossover'))
                        order = int(row.get('order_crossover'))
 
        except Exception as e:
            log_string=f"Error loading reverberation metadata: {e}"
            hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=e)#log error
            pass
            
    except Exception as ex:
        log_string=f"Error occurred: {ex}"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
   
        
    return fc,order

def get_sub_f_name(sub_response="", gui_logger=None):
    """
    Retrieves a file name to load sub brir dataset for a specified sub name
    
    Args:
        sub_response (str, optional): Name of sub
        gui_logger (object, optional): A logger object for GUI-related logging. Defaults to None.
    
    Returns:
        fc: crossover freq (int), order (int)
    """
    
    file_name = ''
    
    try:

        #load lists from csv file
        try:
        
            #directories
            csv_directory = pjoin(CN.DATA_DIR_INT, 'sub')
            #read metadata from csv. Expects reverberation_metadata.csv 
            metadata_file_name = 'sub_brir_metadata.csv'
            metadata_file = pjoin(csv_directory, metadata_file_name)
            with open(metadata_file, encoding='utf-8-sig', newline='') as inputfile:
                reader = DictReader(inputfile)
                for row in reader:#rows 2 and onward
                    #store each row as a dictionary
                    #append to list of dictionaries
                    if row.get('name_gui') == sub_response: #added check
                        file_name = (row.get('file_name'))
 
        except Exception as e:
            log_string=f"Error loading sub metadata: {e}"
            hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=e)#log error
            pass
            
    except Exception as ex:
        log_string=f"Error occurred: {ex}"
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
   
        
    return file_name
