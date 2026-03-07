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
import time
import logging
from ash_toolset import helper_functions as hf
from ash_toolset import constants as CN
from ash_toolset import brir_export
from ash_toolset import hrir_processing
import scipy as sp
import os
from pathlib import Path
import concurrent.futures

import json
from datetime import datetime
from ash_toolset import air_processing

logger = logging.getLogger(__name__)




def generate_integrated_brir(brir_name,  spatial_res=1, report_progress=0, gui_logger=None, brir_meta_dict={}):   
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
        if brir_meta_dict:
            brir_hrtf_type=brir_meta_dict.get('brir_hrtf_type')
            brir_hrtf_dataset=brir_meta_dict.get('brir_hrtf_dataset')
            brir_hrtf_gui = brir_meta_dict.get('brir_hrtf')
            brir_hrtf_short=brir_meta_dict.get('brir_hrtf_short')
            room_target_name = brir_meta_dict.get("room_target")
            direct_gain_db = brir_meta_dict.get("direct_gain_db")
            acoustic_space= brir_meta_dict.get("ac_space_gui")
            brir_hp_comp = brir_meta_dict.get("brir_hp_comp")
            brir_elf = brir_meta_dict.get("brir_elf")
            crossover_f=brir_meta_dict.get('crossover_f')
            sub_response=brir_meta_dict.get('sub_response')
            hp_rolloff_comp=brir_meta_dict.get('hp_rolloff_comp')
            fb_filtering=brir_meta_dict.get('fb_filtering')
            hrtf_symmetry = brir_meta_dict.get("hrtf_symmetry")
            hrtf_low_freq_suppression = brir_meta_dict.get("hrtf_low_freq_suppression")
            early_refl_delay_ms = brir_meta_dict.get("er_delay_time")
            hrtf_polarity = brir_meta_dict.get("hrtf_polarity")
            hrtf_direction_misalign_comp = brir_meta_dict.get("hrtf_direction_misalign_comp")
            hrtf_df_cal_mode = brir_meta_dict.get("hrtf_df_cal_mode")
            reverb_tail_crop_db = brir_meta_dict.get("reverb_tail_crop_db")
            brir_df_cal_mode = brir_meta_dict.get("brir_df_cal_mode")
            brir_max_length=brir_meta_dict.get("brir_max_length")
            octave_smoothing_n=brir_meta_dict.get("octave_smoothing_n")
            brir_df_cal_factor=brir_meta_dict.get("brir_df_cal_factor")
            gen_fir_length=brir_meta_dict.get("gen_fir_length")
        else:
            raise ValueError('brir_meta_dict not populated')
            
   
        #exit if stop thread flag is true
        stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
        if stop_thread == True:
            brir_out= np.array([])
            status=2#2=cancelled
            return brir_out,status
  
        #limit IR size
        if acoustic_space in CN.AC_SPACE_LIST_LOWRT60:
            n_fft=CN.N_FFT #short mode
        else:
            n_fft=CN.N_FFT_L   #long mode   

        #variable crossover depending on acoustic space
        f_crossover_var=crossover_f
     
        reverb_data=CN.reverb_data
        order_var = CN.extract_column(data=reverb_data, column='order_crossover', condition_key='name_gui', condition_value=acoustic_space, return_all_matches=False)
        fade_start = CN.extract_column(data=reverb_data, column='fade_start', condition_key='name_gui', condition_value=acoustic_space, return_all_matches=False)
        meas_rt60 = CN.extract_column(data=reverb_data, column='meas_rt60', condition_key='name_gui', condition_value=acoustic_space, return_all_matches=False)
        as_folder = CN.extract_column(data=reverb_data, column='folder', condition_key='name_gui', condition_value=acoustic_space, return_all_matches=False)
        as_file_name = CN.extract_column(data=reverb_data, column='file_name', condition_key='name_gui', condition_value=acoustic_space, return_all_matches=False)
        as_id = CN.extract_column(data=reverb_data, column='id', condition_key='name_gui', condition_value=acoustic_space, return_all_matches=False)
        as_url_primary = CN.extract_column(data=reverb_data, column='gdrive_link', condition_key='name_gui', condition_value=acoustic_space, return_all_matches=False)
        as_url_alt = CN.extract_column(data=reverb_data, column='alternative_link', condition_key='name_gui', condition_value=acoustic_space, return_all_matches=False)
        as_url_ghub = CN.extract_column(data=reverb_data, column='ghub_link', condition_key='name_gui', condition_value=acoustic_space, return_all_matches=False)
        
        #filters
        if f_crossover_var < CN.FILTFILT_THRESH_F:
            fb_filtering=True#force true if below threshold due to strong delays
        lp_sos = hf.get_filter_sos(cutoff=f_crossover_var, fs=CN.FS, order=order_var, filtfilt=fb_filtering, b_type='low')
        hp_sos = hf.get_filter_sos(cutoff=f_crossover_var, fs=CN.FS, order=order_var, filtfilt=fb_filtering, b_type='high')

 
        #impulse
        impulse=CN.IMPULSE
        fr_flat = CN.FR_FLAT_DB_RFFT

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

        #direction matrices
        
        #get matrix of desired directions to process
        direction_matrix_process = brir_export.generate_direction_matrix(spatial_res=spatial_res, variant=0)
        #get matrix of desired output directions
        #regular case
        direction_matrix_post = brir_export.generate_direction_matrix(spatial_res=spatial_res, variant=1)


        log_string = 'Loading filters'
        hf.log_with_timestamp(log_string, gui_logger)
        if report_progress > 0:
            progress = 1/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)

        #
        # load room target filter (FIR)
  
        room_target_fir = CN.ROOM_TARGETS_DICT[room_target_name]["impulse_response"]
            
        #
        # load pinna comp filter (FIR)
 
        pinna_comp_fir = CN.HP_COMP_DICT[brir_hp_comp]["impulse_response"]
        
        #
        # load Equal loudness filter (FIR)
 
        brir_elf_fir = CN.ELF_DICT[brir_elf]["impulse_response"]
        
        #
        # load additional headphone eq
  
        data_lf_comp_eq = np.copy(impulse[0:32])
        if hp_rolloff_comp == True:
            filename = 'low_frequency_roll-off_compensation.wav'
            wav_fname = pjoin(CN.DATA_DIR_INT, filename)
            samplerate, data_lf_comp_eq = hf.read_wav_file(wav_fname)

        # load low frequency BRIR (FIR)
        #

        sub_data=CN.sub_data
        sub_file_name = CN.extract_column(data=sub_data, column='file_name', condition_key='name_gui', condition_value=sub_response, return_all_matches=False)
        sub_folder = CN.extract_column(data=sub_data, column='folder', condition_key='name_gui', condition_value=sub_response, return_all_matches=False)
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
        
   
        
        ################## Load BRIR reverberation data
        #
        
        # -------------------------------
        # Periodic reverberation update check (every 7 days)
        # -------------------------------
        try:
            csv_directory = pjoin(CN.DATA_DIR_INT, "reverberation")
            last_checked_path = pjoin(csv_directory, "last_checked_updates.json")
            needs_update = True  # default to True if file missing or unreadable
        
            if os.path.exists(last_checked_path):
                try:
                    with open(last_checked_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        last_checked_str = data.get("last_checked")
                        if last_checked_str:
                            last_checked_dt = datetime.strptime(last_checked_str, "%Y-%m-%d %H:%M:%S")
                            hours_elapsed = (datetime.now() - last_checked_dt).total_seconds() / 3600
                            needs_update = hours_elapsed > 168
                except Exception:
                    # If file is corrupted, force update
                    needs_update = True
        
            if needs_update:
                log_string = 'Downloading updates'
                hf.log_with_timestamp(log_string, gui_logger)
                if report_progress > 0:
                    progress = 1/100
                    hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)
                    
                air_processing.acoustic_space_updates(download_updates=True, gui_logger=gui_logger)
        except Exception as e:
            hf.log_with_timestamp(f"Failed to check/update acoustic spaces: {e}", gui_logger)
        
        log_string = 'Loading reverberation data'
        hf.log_with_timestamp(log_string, gui_logger)
        if report_progress > 0:
            progress = 2/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)
            
        if as_folder == 'user':
            brir_rev_folder = pjoin(CN.DATA_DIR_INT, 'reverberation', as_folder, as_id)
        else:
            brir_rev_folder = pjoin(CN.DATA_DIR_INT, 'reverberation', as_folder)
        
        npy_file_name = as_file_name + '.npy'
        npy_file_path = pjoin(brir_rev_folder, npy_file_name)
    
        try:
            # --- Try loading from local file ---
            brir_reverberation = hf.load_convert_npy_to_float64(npy_file_path)
        
            if brir_reverberation is None or len(brir_reverberation) == 0:
                hf.log_with_timestamp('Reverberation dataset is empty.', gui_logger)
                raise ValueError('Reverberation dataset is empty.')
        
        except Exception:
            if as_folder == 'user':
                log_string = 'Unable to load user reverberation dataset'
                hf.log_with_timestamp(log_string, gui_logger)
                raise ValueError(log_string)
                
            hf.log_with_timestamp('Local dataset not found or invalid. Proceeding to download.', gui_logger)
        
            first_choice = ("github", as_url_ghub)
            second_choice = ("primary", as_url_primary)
        
            # --- Ordered download sources ---
            download_sources = [
                first_choice,
                second_choice,
                ("alternate", as_url_alt),
            ]
        
            brir_reverberation = None
            success = False
        
            for source_name, url in download_sources:
                if not url:
                    continue  # Skip if URL missing or empty
        
                hf.log_with_timestamp(f"Attempting {source_name} download...", gui_logger)
                response = hf.download_file(url=url, save_location=npy_file_path, gui_logger=gui_logger)
        
                if response is True:
                    try:
                        brir_reverberation = hf.load_convert_npy_to_float64(npy_file_path)
                    except Exception:
                        brir_reverberation = None
        
                    if brir_reverberation is not None and len(brir_reverberation) > 0:
                        hf.log_with_timestamp(f"Successfully loaded reverberation data from {source_name} link.", gui_logger)
                        success = True
                        break
                    else:
                        hf.log_with_timestamp(f"Dataset empty or invalid after {source_name} download.", gui_logger)
                else:
                    hf.log_with_timestamp(f"{source_name.capitalize()} request failed.", gui_logger)
        
            if not success:
                log_string = 'All download attempts failed. Unable to load reverberation data.'
                hf.log_with_timestamp(log_string, gui_logger)
                raise ValueError(log_string)
           
        receivers, sources, channels, samples = brir_reverberation.shape
        log_string = (
            f"Loaded Reverberation Dataset: "
            f"Receivers={receivers}, Sources={sources}, "
            f"Channels={channels}, Samples={samples}."
        )
        hf.log_with_timestamp(log_string)
           
        total_azim_reverb = len(brir_reverberation[0])
        nearest_azim_reverb = max(int(360/total_azim_reverb),1)
        
        #zero pad reverberation array due to variable length
        brir_reverberation=hf.zero_pad_last_dimension(data=brir_reverberation, n_fft=n_fft)
        
        brir_reverberation = hf.normalize_brir_band(ir_data=brir_reverberation,n_fft=n_fft,fs=CN.FS,f_norm_start=CN.SPECT_SNAP_F0,f_norm_end=CN.SPECT_SNAP_F1,analysis_samples=10000)
        
        #grab spatial format
        if spatial_res >= 0 and spatial_res < CN.NUM_SPATIAL_RES:      
            elev_min=CN.SPATIAL_RES_ELEV_MIN_IN[spatial_res] 
            elev_nearest=CN.SPATIAL_RES_ELEV_NEAREST_IN[spatial_res] #as per hrir dataset
            azim_nearest=CN.SPATIAL_RES_AZIM_NEAREST_IN[spatial_res] 
        else:
            raise ValueError('Invalid spatial resolution')
            
        log_string = 'Loading HRTF data'
        #hf.log_with_timestamp(log_string, gui_logger)
        if report_progress > 0:
            progress = 5/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)

        
        ############################## load HRIRs and prepare HRIR array
        #
        
        # load hrir dataset
        hrtf_dict_list = [
            {
                "hrtf_type": brir_hrtf_type,
                "hrtf_dataset": brir_hrtf_dataset,       # dataset name
                "hrtf_gui": brir_hrtf_gui,        # GUI-friendly name
                "hrtf_short": brir_hrtf_short      # short name used for file
            }
        ]

        
        hrir_list, status, hrir_metadata_list = hrir_processing.load_hrirs_list(hrtf_dict_list=hrtf_dict_list, spatial_res=spatial_res, direction_fix_gui=hrtf_direction_misalign_comp, 
                                                                                gui_logger=gui_logger, brir_meta_dict=brir_meta_dict)
        if status != 0 or not hrir_list:
            raise ValueError(f"Failed to load HRIR dataset: {brir_hrtf_short}")
            
        # Extract the only item, unitary first dimension is already removed if present
        hrir_selected = hrir_list[0]
        # hrir_data now guaranteed shape: [elev][azim][chan][samples]
     
        total_elev_hrir = len(hrir_selected)
        total_azim_hrir = len(hrir_selected[0])
        total_chan_hrir = len(hrir_selected[0][0])
        total_samples_hrir = len(hrir_selected[0][0][0])
 
        ############################## DF calibration CTF loading
        #
        ctf_mag_db = None          # main CTF magnitude in dB
        ctf_adj_mag_db = None      # adjustment filter magnitude in dB
        df_cal_reversal = False
        ctf_loaded = False         # True if main CTF loaded
        ctf_adj_loaded = False # True if both main CTF and adjustment filter loaded
        
        # Only proceed if metadata exists
        if hrir_metadata_list:
            hrir_metadata = hrir_metadata_list[0]
        
            # --- Attempt DF calibration main CTF ---
            if hrtf_df_cal_mode != CN.HRTF_DF_CAL_MODE_LIST[0]:
                df_cal_reversal = True
                desired_ctf = "ctf_le_file" if hrtf_df_cal_mode == CN.HRTF_DF_CAL_MODE_LIST[2] else "ctf_file"
                ctf_path = hrir_metadata.get(desired_ctf, None)
        
                if ctf_path and Path(ctf_path).exists():
                    # Read main CTF WAV
                    samplerate, ctf_data = hf.read_wav_file(ctf_path)
                    if ctf_data.ndim > 1:
                        ctf_data = ctf_data[:, 0]  # take first channel
                    ctf_fir = np.zeros(CN.N_FFT)
                    copy_len = min(len(ctf_data), CN.N_FFT)
                    ctf_fir[:copy_len] = ctf_data[:copy_len]
                    ctf_mag = np.abs(np.fft.rfft(ctf_fir))
                    ctf_mag_db = hf.mag2db(ctf_mag)
                    ctf_loaded = True
                    hf.log_with_timestamp(f"HRTF Diffuse-field calibration mode activated: {hrtf_df_cal_mode}", gui_logger)
                else:
                    missing_reason = "Metadata missing CTF key" if not ctf_path else "CTF file not found"
                    hf.log_with_timestamp(f"{missing_reason}: '{desired_ctf}' for {brir_hrtf_short}", log_type=1)
                    df_cal_reversal = False
        
            # --- Always attempt adjustment filter if metadata flag exists ---
            if hrir_metadata.get("has_adjustment_filter", False):
                ctf_adj_path = hrir_metadata.get("ctf_adj_file", None)
                if ctf_adj_path and Path(ctf_adj_path).exists():
                    samplerate, adj_data = hf.read_wav_file(ctf_adj_path)
                    if adj_data.ndim > 1:
                        adj_data = adj_data[:, 0]
                    ctf_adj_fir = np.zeros(CN.N_FFT)
                    copy_len = min(len(adj_data), CN.N_FFT)
                    ctf_adj_fir[:copy_len] = adj_data[:copy_len]
                    ctf_adj_mag_db = hf.mag2db(np.abs(np.fft.rfft(ctf_adj_fir)))
                    hf.log_with_timestamp(f"Adjustment filter loaded from: {ctf_adj_path}", gui_logger)
        
                    # Set flag if adjustment filter are loaded
                    ctf_adj_loaded = True
                else:
                    hf.log_with_timestamp("Adjustment filter missing or file not found, skipping.", log_type=1)
                                
                        
        
        # If metadata list is empty or DF calibration not required, df_cal_reversal remains False
  
        ##################### further HRIR preparation
      
        #output_azims = int(360/azim_nearest)#should be identical to above lengths
        #output_elevs = int((elev_max-elev_min)/elev_nearest +1)#should be identical to above lengths
        
        #BRIR_out array will be populated BRIRs for all directions
        brir_out=np.zeros((total_elev_hrir,total_azim_hrir,2,n_fft))   

        log_string = 'Adjusting HRTF levels'
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
        if hrtf_polarity == 'Recommended' or hrtf_polarity == 'Auto Select':
            flip_polarity = hrir_processing.get_polarity(listener_type=brir_hrtf_type, dataset_name=brir_hrtf_dataset, name_gui=brir_hrtf_gui, gui_logger=gui_logger)
        elif hrtf_polarity == 'Original':
            flip_polarity='no'
        elif hrtf_polarity == 'Reversed':
            flip_polarity='yes'
        else:
            flip_polarity='no'   
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
        
        
        
        ################ Reverberation shaping and preparation
        #
        
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
   
        
 
            
        ###################### Noise reduction at tail ends + reflection delays

        # Determine if noise fade is needed
        noise_fade = fade_start > 0
        
        # Compute sample indices
        rt60_comp_factor = 0.95
        n_fade_win_start = int((fade_start * rt60_comp_factor / 1000) * CN.FS)
        rt60_snap_start = int(n_fft * 0.9)
        
        # Generate fade windows
        l_fade_win_size = abs(rt60_snap_start - n_fft) * 2
        n_fade_win_size = 4000 * 2
        win_l_fade_out = np.hanning(l_fade_win_size)[l_fade_win_size // 2:]
        win_n_fade_out = np.hanning(n_fade_win_size)[n_fade_win_size // 2:]
        
        # Construct overall fade-out window
        n_fade_out_win = data_pad_zeros.copy()
        if noise_fade:
            n_fade_out_win[:n_fade_win_start] = data_pad_ones[:n_fade_win_start]
            n_fade_out_win[n_fade_win_start:n_fade_win_start + n_fade_win_size // 2] = win_n_fade_out
        else:
            n_fade_out_win[:rt60_snap_start] = data_pad_ones[:rt60_snap_start]
            n_fade_out_win[rt60_snap_start:] = win_l_fade_out
        
        # Apply fade and optional early reflection delay (vectorized)
        # Select all slices in dim0, all azimuths, all dim1, all channels
        brir_slice = brir_reverberation[:, :total_azim_reverb, :, :]  # shape: (dim0, azim, dim1, chan, samples)
        
        # Apply fade-out (broadcast over last axis)
        brir_slice *= n_fade_out_win
        
        # Apply early reflection delay if needed
        if early_refl_delay_ms > 0:
            delay_samples = int((early_refl_delay_ms / 1000) * CN.FS)
            brir_slice = np.roll(brir_slice, shift=delay_samples, axis=-1)
            brir_slice[..., :delay_samples] = 0  # zero out prepended samples
        
        # Write back to original array
        brir_reverberation[:, :total_azim_reverb, :, :] = brir_slice
            
        
        
        

                
        ################ HRIR and Reverberation Integration
        #
        
        log_string = 'Integrating HRIRs and reverberation'
        hf.log_with_timestamp(log_string, gui_logger) 
        if report_progress > 0:
            progress = 40/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)

      
    
        # grab reverberant BRIRs from interim matrix and place in output matrix
        if meas_rt60 > 0:
            for elev in range(total_elev_hrir):
                for azim in range(total_azim_hrir): 
                    azim_deg = int(azim*azim_nearest)
                    #case for minimal set (7 directions)
                    if total_azim_reverb == 7:
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
                    #case for multiple directions on horizontal plane, every x deg azimuth
                    elif total_azim_reverb > 0:
                        #map hrir azimuth to appropriate brir azimuth
                        #round azim to nearest X deg and get new ID
                        brir_azim = hf.round_to_multiple(azim_deg,nearest_azim_reverb)#brir_reverberation is nearest X deg, variable
                        if brir_azim >= 360:
                            brir_azim = 0
                        brir_azim_ind = int(brir_azim/nearest_azim_reverb)#get index
                    else:
                        raise ValueError('Unable to process BRIR reverberation data. Invalid number of reverberation sources: ' + str(total_azim_reverb) )
                    
                    for chan in range(CN.TOTAL_CHAN_BRIR):
                        brir_out[elev][azim][chan][0:n_fft] = np.copy(brir_reverberation[0][brir_azim_ind][chan][0:n_fft])
    
        
   
    
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
        
        if report_progress > 0:
            progress = 50/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)
   
        #################### low frequency BRIR integration
        #
 
        #exit if stop thread flag is true
        stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
        if stop_thread == True:
            brir_out= np.array([])
            status=2#2=cancelled
            return brir_out,status
        
        #low freq reference ranges
        f_norm_start=50
        f_norm_end=180
        fb_start = int(f_norm_start * n_fft / CN.FS)
        fb_end = int(f_norm_end * n_fft / CN.FS)
        
        #grab a sample brir used for td alignment and level alignment
        elev_ind = int((0-elev_min)/elev_nearest)
        azim_ind = int(30/azim_nearest)
        brir_sample = np.copy(brir_out[elev_ind][azim_ind][0][:])
        
        #set level of HF BRIR to 0 at low freqs
        data_fft_b = np.fft.rfft(brir_sample)
        mag_fft_b=np.abs(data_fft_b)
        average_mag_b = np.mean(mag_fft_b[fb_start:fb_end])
        if average_mag_b == 0:
            if CN.LOG_INFO == True:
                logging.info('0 magnitude detected')
        for elev in range(total_elev_hrir):
            for azim in range(total_azim_hrir):
                for chan in range(total_chan_hrir):      
                    #only apply gain if direction is in direction matrix
                    if direction_matrix_process[elev][azim][0][0] == 1: 
                        brir_out[elev][azim][chan][:] = np.divide(brir_out[elev][azim][chan][:],average_mag_b)
   
        log_string = 'Processing low frequencies'
        hf.log_with_timestamp(log_string, gui_logger)
       
        #integreate low frequency BRIRs into BRIRs. Only perform if enabled 
        if CN.ENABLE_SUB_INTEGRATION == True and f_crossover_var >= CN.MIN_FILT_FREQ:
     
            sub_brir_align(sub_brir_ir, n_fft, f_crossover_var, brir_sample, initial_removal_win_sub, order_var)
            
            if report_progress > 0:
                progress = 60/100
                hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)
            
            ############# apply low pass filter to sub BRIR
            brir_eq_a_l = np.copy(sub_brir_ir[0][:])
            brir_eq_a_r = np.copy(sub_brir_ir[1][:])
            #apply lp filter
            brir_eq_b_l = hf.apply_sos_filter(brir_eq_a_l, lp_sos, filtfilt=fb_filtering)
            brir_eq_b_r = hf.apply_sos_filter(brir_eq_a_r, lp_sos, filtfilt=fb_filtering)
            sub_brir_ir[0][:] = brir_eq_b_l[0:n_fft] 
            sub_brir_ir[1][:] = brir_eq_b_r[0:n_fft]
            #apply fade out win
            sub_brir_ir[0][:] = np.multiply(sub_brir_ir[0][:],n_fade_out_win)
            sub_brir_ir[1][:] = np.multiply(sub_brir_ir[1][:],n_fade_out_win)
      
            #use multiple threads to integrate sub brir into output array
            # Use ThreadPoolExecutor to integrate sub brir
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(integrate_sub_brirs, brir_out, elev, n_fft, total_azim_hrir, direction_matrix_process, sub_brir_ir, hp_sos, fb_filtering)
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
   
        
        ############### EQ correction
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
        # Limit maximum boost to +X dB
        brir_fft_avg_db = np.clip(brir_fft_avg_db, np.median(brir_fft_avg_db)-20.0, np.median(brir_fft_avg_db)+40.0)
        #convert to mag
        brir_fft_avg_mag = hf.db2mag(brir_fft_avg_db)      
        #level ends of spectrum
        brir_fft_avg_mag_sm = hf.level_spectrum_ends(brir_fft_avg_mag, 15, 18500, smooth_win = octave_smoothing_n, n_fft=CN.N_FFT)#40, 19000, smooth_win = 7 
        #octave smoothing
        brir_fft_avg_mag_sm = hf.smooth_gaussian_octave(data=brir_fft_avg_mag_sm, n_fft=CN.N_FFT, fraction=octave_smoothing_n)
        
        #include CTF if DF calibration reversal is enabled
        if ctf_adj_loaded and ctf_adj_mag_db is not None:
            # safely add adjustment to DF correction curve due to incomplete sphere
            brir_fft_avg_mag_sm = hf.db2mag(np.add(hf.mag2db(brir_fft_avg_mag_sm),ctf_adj_mag_db*0.8))#slight reduction in strength due to reverberation
            hf.log_with_timestamp("HRTF CTF Adjustment response available, applying to integrated response CTF.")

        
        #invert response and scale by user selected strength factor
        brir_df_cal_factor=max(0.05,brir_df_cal_factor)
        brir_fft_avg_mag_inv = hf.db2mag(hf.mag2db(brir_fft_avg_mag_sm)*-1*brir_df_cal_factor)
        
        #include CTF if DF calibration reversal is enabled
        if df_cal_reversal:
            if ctf_loaded and ctf_mag_db is not None:
                # safely add CTF to inverse correction curve to reintroduce it
                brir_fft_avg_mag_inv = hf.db2mag(np.add(hf.mag2db(brir_fft_avg_mag_inv),ctf_mag_db))
            else:
                hf.log_with_timestamp("HRTF CTF DF response not available, skipping HRTF DF reintegration.", gui_logger)
                
        #create min phase FIR
        if brir_df_cal_mode == CN.BRIR_DF_CAL_MODE_LIST[0]:
            brir_df_inv_fir = hf.build_min_phase_filter(smoothed_mag=brir_fft_avg_mag_inv,  truncate_len=gen_fir_length, n_fft=CN.N_FFT)#v4.0.0 truncate_len=4096
        elif brir_df_cal_mode == CN.BRIR_DF_CAL_MODE_LIST[1]:
            # 1. Define your impulse (e.g., 8192 samples long)
            n_fft_param = 8192 
            impulse = np.zeros((1, 1, n_fft_param))
            impulse[0, 0, 0] = 1.0  # Set the first sample to 1.0
            # 2. Run your function with the override
            # This will return the EQ-IR as a (1, 1, n_fft) array
            eq_ir_dataset = hf.equalize_brirs_parametric(brir_dataset=impulse, diff_db_override=hf.mag2db(brir_fft_avg_mag_inv), override_n_fft=CN.N_FFT, 
                                                         num_filters=50,low_freq_cut = 12.0, high_freq_cut = 18000.0)
            # 3. Extract the 1D array for easy convolution later
            brir_df_inv_fir = eq_ir_dataset.flatten()
        
        if report_progress > 0:
            progress = 80/100
            hf.update_gui_progress(report_progress=report_progress, progress=progress, message=log_string)
        
        #exit if stop thread flag is true
        stop_thread = hf.check_stop_thread(gui_logger=gui_logger)
        if stop_thread == True:
            brir_out= np.array([])
            status=2#2=cancelled
            return brir_out,status


        
        ################# section to crop BRIR output
        #larger reverb times will need additional samples
        #estimate output array length based on RT60
        #Below determines the most samples to keep based on estimated rt60
        if meas_rt60 <=0:
            out_samples_max = 512
        elif meas_rt60 <=400:
            out_samples_max = 33075
        elif meas_rt60 <=1250:
            out_samples_max = 64500    
        elif meas_rt60 <=1500:
            out_samples_max = 99225 
        else:
            out_samples_max = 127890
        if out_samples_max > n_fft:
            out_samples_max = max(n_fft,4000) #assumes at least 4000 samples available as fallback     
        #attempt to trim array. Calculate point where amplitude falls below threshold, later discards remaining samples
        brir_sample = np.abs(np.copy(brir_out[elev_ind][azim_ind][0][:]))
        norm_factor = np.max(brir_sample)
        ref_array = np.divide(brir_sample,norm_factor*2)
        crop_samples = hf.get_crop_index_relative(ref_array, tail_ignore=8000,head_ignore=100, threshold_db=reverb_tail_crop_db) #crop_samples = hf.get_crop_index(ref_array, tail_ignore=10000)
        if crop_samples < 2000 or crop_samples > out_samples_max:#ensure it stays within limits. Not suitable for short IRs (<50ms)
            crop_samples=out_samples_max
        if crop_samples > brir_max_length:#clamp to user specified max length
            crop_samples = brir_max_length
 
        #merge filters into single EQ filter
        eq_filter = brir_df_inv_fir
        for f in (room_target_fir, pinna_comp_fir, data_lf_comp_eq, brir_elf_fir):
            if f is not None:
                eq_filter = sp.signal.convolve(eq_filter, f, mode='full', method='auto')
        
        #use multiple threads to perform EQ
        #Use ThreadPoolExecutor to perform EQ
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(apply_eq_to_brirs, brir_out, elev, n_fft, total_azim_hrir, direction_matrix_post, eq_filter, crop_samples)
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
            
        # # ----------------------------------------
        # # Final peak normalization (anti-clipping)
        # # ----------------------------------------
        # peak = np.max(np.abs(brir_out))
        # if peak > 0:
        #     brir_out /= peak
            
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



def sub_brir_align(sub_brir_ir, n_fft, f_crossover_var,brir_sample, initial_removal_win_sub, order_var):
    """
    Time-domain alignment of sub BRIR array
    """

    # --- Low-pass + shift limits ---
    lp_sos = hf.get_filter_sos(
        cutoff=f_crossover_var,
        fs=CN.FS,
        order=order_var,
        filtfilt=CN.FILTFILT_TDALIGN,
        b_type="low",
    )

    t_shift_interval = 10
    shift_quota = int(1.05*CN.FS / f_crossover_var / 2)
    min_t_shift = int(np.floor(-shift_quota / 2 / t_shift_interval) * t_shift_interval)
    max_t_shift = int(np.ceil(+shift_quota / 2 / t_shift_interval) * t_shift_interval)
    shifts = np.arange(min_t_shift, max_t_shift, t_shift_interval)

    peak_to_peak_window = int(CN.FS / f_crossover_var * 0.95)

    # --- Normalise sub BRIR low-frequency level ---
    fb_start = int(20 * n_fft / CN.FS)
    fb_end = int(170 * n_fft / CN.FS)

    mag = np.abs(np.fft.rfft(sub_brir_ir[0]))
    avg_mag = np.mean(mag[fb_start:fb_end]) or 1.0

    sub_brir_ir[:] /= avg_mag

    # --- Reference + filtered signals ---
    brir_lp = hf.apply_sos_filter(brir_sample, lp_sos, filtfilt=CN.FILTFILT_TDALIGN)
    sub_lp = hf.apply_sos_filter(sub_brir_ir[0], lp_sos, filtfilt=CN.FILTFILT_TDALIGN)

    # --- Peak-to-peak evaluation helper ---
    def eval_alignment(sub_lp_sig):
        scores = np.zeros(len(shifts))
        for i, s in enumerate(shifts):
            summed = brir_lp + np.roll(sub_lp_sig, s)
            best = 0.0
            for hop in range(CN.DELAY_WIN_HOPS):
                start = CN.DELAY_WIN_MIN_T + hop * CN.DELAY_WIN_HOP_SIZE
                seg = summed[start : start + peak_to_peak_window]
                best = max(best, np.max(seg) - np.min(seg))
            scores[i] = best
        return scores

    # --- Evaluate both polarities ---
    scores_p = eval_alignment(sub_lp)
    scores_n = eval_alignment(-sub_lp)

    use_positive = (
        np.max(scores_p) > np.max(scores_n)
        or not CN.EVAL_POLARITY
    )

    scores = scores_p if use_positive else scores_n
    sub_polarity = 1 if use_positive else -1

    idx = int(np.argmax(scores))
    samples_shift = shifts[idx]

    # --- Apply shift + polarity ---
    for ch in range(CN.TOTAL_CHAN_BRIR):
        sub_brir_ir[ch] = np.roll(sub_brir_ir[ch], samples_shift) * sub_polarity
        if samples_shift < 0:
            sub_brir_ir[ch][samples_shift:] = 0
            sub_brir_ir[ch] *= initial_removal_win_sub

    # --- Logging ---
    if CN.LOG_INFO and CN.SHOW_DEV_TOOLS:
        logging.info(f"(SUB) delay index = {idx}")
        logging.info(f"(SUB) samples_shift = {samples_shift}")
        logging.info(f"(SUB) sub polarity = {sub_polarity}")
        logging.info(f"(SUB) peak_to_peak_max_p = {np.max(scores_p)}")
        logging.info(f"(SUB) peak_to_peak_max_n = {np.max(scores_n)}")
        
        
        

    
def combine_hrirs_brirs(brir_out, elev, n_fft, total_azim_hrir, hrir_selected, total_samples_hrir, swap_azimuths=False):
    """
    Integrate HRIRs into BRIR dataset more efficiently.
    Adds hrir_selected to brir_out for all channels in one go.
    Optionally swaps azimuths.
    """
    for azim in range(total_azim_hrir):
        azim_hrir = total_azim_hrir - azim if swap_azimuths and azim > 0 else azim
        # Add all channels in one slice
        brir_out[elev, azim, :, :total_samples_hrir] += hrir_selected[elev, azim_hrir, :, :total_samples_hrir]


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
    
    
    
    


def apply_eq_to_brirs(brir_out, elev, n_fft, total_azim_hrir, direction_matrix_post, eq_filter, crop_samples):
    """
    Apply a precomputed EQ filter to BRIRs for a single elevation.

    Args:
        brir_out (np.ndarray): BRIR array, shape (elevs, azims, channels, samples)
        elev (int): Elevation index to process
        n_fft (int): Number of samples to keep per BRIR
        total_azim_hrir (int): Total azimuths in BRIR array
        direction_matrix_post (np.ndarray): Mask of directions to process
        eq_filter (np.ndarray): Precomputed combined EQ filter
        crop_samples (int): Number of samples to take from input before convolution
    """

    for azim in range(total_azim_hrir):
        for chan in range(CN.TOTAL_CHAN_BRIR):

            if direction_matrix_post[elev, azim, 0, 0] == 1:
                # Take the portion to convolve
                brir_eq = brir_out[elev, azim, chan, :crop_samples]
                brir_eq = sp.signal.convolve(brir_eq, eq_filter, mode='full', method='auto')

                # Replace into output
                replace_samples = min(len(brir_eq), n_fft)
                brir_out[elev, azim, chan, :replace_samples] = brir_eq[:replace_samples]

                if replace_samples < n_fft:
                    brir_out[elev, azim, chan, replace_samples:] = 0
            else:
                # zero out directions that won't be exported
                brir_out[elev, azim, chan, :] = 0


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
                brir_eq_b_h = hf.apply_sos_filter(brir_eq_a_h, hp_sos, filtfilt=fb_filtering)
                
                brir_out[elev][azim][chan][:] = brir_eq_b_h[0:n_fft] 
  
    #add SUB BRIR (already low passed) into output BRIR array
    for azim in range(total_azim_hrir):
        for chan in range(CN.TOTAL_CHAN_BRIR):
            #only apply if direction is in direction matrix
            if direction_matrix_process[elev][azim][0][0] == 1: 
                brir_out[elev][azim][chan][:] = brir_out[elev][azim][chan][:] + sub_brir_ir[chan][:]




        
   



def process_mono_cues(gui_logger=None):
    """
    Refined function: Generates separate Headphone Compensation and 
    Equal Loudness (Perceptual) datasets.
    """
    from ash_toolset import pyquadfilter

    try:
        n_fft = CN.N_FFT
        fir_length = 1024
        fr_flat_db = CN.FR_FLAT_DB_FFT
        level_f_a, level_f_b = 450, 17000
        impulse = np.zeros(n_fft)
        impulse[0] = 1

        # --- Internal Helpers ---
        def load_wav_mag(fname):
            sr, data = hf.read_wav_file(pjoin(CN.DATA_DIR_INT, fname))
            fir = np.zeros(n_fft)
            fir[:min(len(data), 1024)] = data[:min(len(data), 1024)]
            return hf.mag2db(np.abs(np.fft.fft(fir)))

        def load_npy_avg_mag(folder):
            avg_db = fr_flat_db.copy()
            count = 0
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.endswith('.npy'):
                        avg_db += hf.mag2db(np.load(pjoin(root, f)))
                        count += 1
            avg_db /= max(count, 1)
            mag = hf.db2mag(avg_db)
            mag = hf.level_spectrum_ends(mag, level_f_a, level_f_b, smooth_win=20)
            return hf.smooth_gaussian_octave(data=mag, n_fft=n_fft, fraction=6)

        def apply_peaking_filters(filter_seq, impulse_in, samp_freq):
            fir_out = np.copy(impulse_in)
            for f in filter_seq:
                pyquad = pyquadfilter.PyQuadFilter(samp_freq)
                pyquad.set_params("peaking", f['fc'], f['q'], f['gain_db'])
                fir_out = pyquad.filter(fir_out)
            return fir_out

        # --- 1. Load & Process Base Mags ---
        # Note: in_ear_eq_db is kept if needed for future logic, but high-strength vars are gone.
        hp_cue_mean_mag = {m: load_npy_avg_mag(pjoin(CN.DATA_DIR_INT, 'mono_cues', 'hp_cues', f)) 
                           for m, f in zip(['onear', 'overear', 'inear'], ['on_ear', 'over_ear', 'in_ear'])}

        # Difference curve between Over-Ear and In-Ear
        oe_ie_diff_mag = hf.db2mag(hf.mag2db(hp_cue_mean_mag['overear']) - hf.mag2db(hp_cue_mean_mag['inear']))

        # Load Pinna Reference
        pinna_fir = np.zeros(n_fft)
        pinna_short = np.load(pjoin(CN.DATA_DIR_INT, 'headphone_pinna_comp_fir.npy'))
        pinna_fir[:4096] = pinna_short[:4096]
        pinna_comp_pos = hf.db2mag(-hf.mag2db(np.abs(np.fft.fft(pinna_fir))))

        # External Over-minus-In difference curves
        new_diff_mag = np.abs(np.fft.fft(np.pad(hf.read_wav_file(pjoin(CN.DATA_DIR_INT, 'over_minus_in_avg.wav'))[1][:1024], (0, n_fft-1024))))
        new_diff_mag_v2 = np.abs(np.fft.fft(np.pad(hf.read_wav_file(pjoin(CN.DATA_DIR_INT, 'over_minus_in_avg_selected_sources.wav'))[1][:1024], (0, n_fft-1024))))

        # --- 2. Build Derived Curves ---
        # HP Estimate (Weighted average for Over-Ear interaction)
        hp_est_avg_mag = hf.db2mag(
            0.25 * hf.mag2db(pinna_comp_pos) + 
            0.40 * hf.mag2db(oe_ie_diff_mag) + 
            0.10 * hf.mag2db(new_diff_mag) + 
            0.25 * hf.mag2db(new_diff_mag_v2)
        )
        
        # --- NEW: Plotting Base Components ---
        if CN.PLOT_ENABLE:
            base_components = [
                (pinna_comp_pos, "Pinna Comp Pos (Reference)"),
                (oe_ie_diff_mag, "OE-IE Mean Difference"),
                (new_diff_mag, "External Over-In Avg"),
                (new_diff_mag_v2, "External Over-In Selected Sources")
            ]
            for mag, title in base_components:
                hf.plot_data(mag.flatten(), title_name=f"BASE COMP: {title}", normalise=1)

        over_ear_add_eq_mag = hf.db2mag(-load_wav_mag('additional_comp_for_over_&_on_ear_headphones.wav'))
        
        # Headphone Comp Bases
        over_ear_low_db = hf.mag2db(hp_est_avg_mag)

        # Equal Loudness (ELF) Base
        hs_filters = {
                    # 'a': [{'fc': 2100, 'q': 2.0, 'gain_db': -4.0}, {'fc': 4000, 'q': 2.0, 'gain_db': 2.0}],
                    'b': [{'fc': 2200, 'q': 3.0, 'gain_db': -4.0}, {'fc': 5500, 'q': 2.5, 'gain_db': 1.5},
                          {'fc': 6500, 'q': 3.0, 'gain_db': 3.0}, {'fc': 10200, 'q': 6.0, 'gain_db': -4.0},
                          {'fc': 7500, 'q': 5.0, 'gain_db': 2.0}],
                    'c': [{'fc': 2900, 'q': 2.5, 'gain_db': -5.0}, {'fc': 5500, 'q': 2.0, 'gain_db': 2.0},
                          {'fc': 5000, 'q': 3.0, 'gain_db': 1.5}, {'fc': 3400, 'q': 5.0, 'gain_db': 2.0},
                          {'fc': 10500, 'q': 5.0, 'gain_db': -3.0}],
                    'd': [{'fc': 3000, 'q': 2.0, 'gain_db': -6.0}, {'fc': 1500, 'q': 3.0, 'gain_db': 1.5}, {'fc': 5900, 'q': 3.0, 'gain_db': 3.5},
                          {'fc': 4500, 'q': 3.0, 'gain_db': 2.5}, {'fc': 9500, 'q': 7.0, 'gain_db': -4.0}, {'fc': 3600, 'q': 5.0, 'gain_db': -2.0},
                          {'fc': 10800, 'q': 4.0, 'gain_db': -2.0}, {'fc': 7700, 'q': 7.0, 'gain_db': 2.0}, {'fc': 6300, 'q': 7.0, 'gain_db': 1.0}],
                    'e': [{'fc': 1400, 'q': 2.5, 'gain_db': 1.0}, {'fc': 2100, 'q': 4.0, 'gain_db': -2.5}, {'fc': 2800, 'q': 2.5, 'gain_db': -3.0},
                          {'fc': 4500, 'q': 4.0, 'gain_db': -3.0}, {'fc': 5900, 'q': 2.0, 'gain_db': 1.0}, {'fc': 6600, 'q': 8.0, 'gain_db': -1.5},
                          {'fc': 7200, 'q': 4.0, 'gain_db': 4.0}, {'fc': 8200, 'q': 4.0, 'gain_db': 2.0}, {'fc': 10100, 'q': 9.0, 'gain_db': -7.0}, {'fc': 11000, 'q': 7.0, 'gain_db': -5.0}]
                }
        
        oe_hs_comp_mags = []
        for seq in hs_filters.values():
            fir = apply_peaking_filters(seq, impulse, CN.SAMP_FREQ)
            oe_hs_comp_mags.append(hf.mag2db(np.abs(np.fft.fft(fir, n=n_fft))))

        oe_hs_comp_avg_db = np.mean(oe_hs_comp_mags, axis=0) * -1 #will be inverted once more, returning original curve
        oe_hs_new_avg_db = (oe_hs_comp_avg_db * 0.70) + (hf.mag2db(over_ear_add_eq_mag) * 0.30)
        
        in_ear_high_db = (-load_wav_mag('additional_comp_for_in_ear_headphones.wav'))

        # --- 3. Final Export Preparation ---
        # Collection 1: Headphone Comp (Flat, Flat/None, Over-Ear Low)
        hp_comp_mags = [np.zeros(n_fft), np.zeros(n_fft), over_ear_low_db]
        hp_comp_labels = ["None", "In-Ear (Flat Interaction)", "Over-Ear Low"]

        # Collection 2: Equal Loudness (Flat, In-Ear high/Perceptual, OE HS Avg)
        elf_mags = [np.zeros(n_fft), in_ear_high_db, oe_hs_new_avg_db]
        elf_labels = ["None", "In-Ear Perceptual", "Over-Ear Perceptual"]

        # --- 4. Plotting Export Curves ---
        if CN.PLOT_ENABLE:
            # Helper to handle the 1D conversion and plot both states
            def plot_pair(mag_db, label, category):
                # 1. Plot the "Natural" interaction curve (as measured)
                hf.plot_data(
                    hf.db2mag(mag_db).flatten(), 
                    title_name=f"{category} INTERACTION: {label}", 
                    normalise=1
                )
                # 2. Plot the "Inverted" compensation curve (as applied by FIR)
                # This is what build_min_phase_filter actually uses: hf.db2mag(-mag_db)
                hf.plot_data(
                    hf.db2mag(-mag_db).flatten(), 
                    title_name=f"{category} COMPENSATION (INVERTED): {label}", 
                    normalise=1
                )

            # Plot Headphone Compensation pairs
            for mag_db, label in zip(hp_comp_mags, hp_comp_labels):
                plot_pair(mag_db, label, "HP COMP")

            # Plot Equal Loudness pairs
            for mag_db, label in zip(elf_mags, elf_labels):
                plot_pair(mag_db, label, "ELF")

        # --- 5. Export to FIR ---
        # Using .flatten() here as well to ensure build_min_phase_filter 
        # receives a clean 1D magnitude vector.
        hp_comp_npy = np.array([
            hf.build_min_phase_filter(hf.db2mag(-m).flatten(), truncate_len=fir_length) 
            for m in hp_comp_mags
        ])
        
        elf_npy = np.array([
            hf.build_min_phase_filter(hf.db2mag(-m).flatten(), truncate_len=fir_length) 
            for m in elf_mags
        ])

        # Save Logic
        paths = {
            'hp': pjoin(CN.DATA_DIR_INT, 'headphone_comp', 'headphone_ear_comp_dataset_new.npy'),
            'elf': pjoin(CN.DATA_DIR_INT, 'loudness_comp', 'loudness_comp_dataset_new.npy')
        }

        for p in paths.values():
            Path(p).parent.mkdir(exist_ok=True, parents=True)
            
        np.save(paths['hp'], hp_comp_npy)
        np.save(paths['elf'], elf_npy)

        hf.log_with_timestamp(f"Exported {len(hp_comp_npy)} HP Comp and {len(elf_npy)} ELF filters.", gui_logger)

    except Exception as ex:
        hf.log_with_timestamp('Failed to complete mono cue processing', gui_logger=gui_logger, log_type=2, exception=ex)