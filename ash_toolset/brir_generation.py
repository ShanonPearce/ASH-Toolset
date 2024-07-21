# -*- coding: utf-8 -*-
"""
Main routine of ASH-Tools.

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
import scipy as sp
import matplotlib.pyplot as plt
from scipy.io import wavfile
import dearpygui.dearpygui as dpg

logger = logging.getLogger(__name__)



def generate_reverberant_brir(gui_logger=None):
    """
    Function generates binaural reverberation data ready for integration with HRIRs
    :param gui_logger: gui logger object for dearpygui
    :return: None
    """ 
    
    # get the start time
    st = time.time()

    try:
    
        #
        ### variables
        #
    
        brir_reverberation=np.zeros((CN.INTERIM_ELEVS,CN.OUTPUT_AZIMS,2,CN.N_FFT))
        
        #windows
        data_pad_zeros=np.zeros(CN.N_FFT)
        data_pad_ones=np.ones(CN.N_FFT)
        
        #direct sound window
        direct_hanning_size=450#400
        direct_hanning_start=104#101
        hann_direct_full=np.hanning(direct_hanning_size)
        hann_direct = np.split(hann_direct_full,2)[0]
        direct_removal_win = data_pad_zeros.copy()
        direct_removal_win[direct_hanning_start:direct_hanning_start+int(direct_hanning_size/2)] = hann_direct
        direct_removal_win[direct_hanning_start+int(direct_hanning_size/2):]=data_pad_ones[direct_hanning_start+int(direct_hanning_size/2):]
        
        #direct sound window for 2nd phase
        direct_hanning_size=600#350
        direct_hanning_start=104#101
        hann_direct_full=np.hanning(direct_hanning_size)
        hann_direct = np.split(hann_direct_full,2)[0]
        direct_removal_win_b = data_pad_zeros.copy()
        direct_removal_win_b[direct_hanning_start:direct_hanning_start+int(direct_hanning_size/2)] = hann_direct
        direct_removal_win_b[direct_hanning_start+int(direct_hanning_size/2):]=data_pad_ones[direct_hanning_start+int(direct_hanning_size/2):]

 
        #
        # load BRIRs
        #
        
        mat_fname = pjoin(CN.DATA_DIR_INT, 'brir_dataset_compensated.mat')
        #new format MATLAB 7.3 file
        brirs = mat73.loadmat(mat_fname)
        #load matrix, get list of numpy arrays
        brir_list = [[element for element in upperElement] for upperElement in brirs['ash_input_brir']]
        #grab desired hrtf. Returns 56x24x2x65536 array
        
        if CN.PLOT_ENABLE == 1:
            room_selected = brir_list[0]
            brir_selected = room_selected[0]
            hf.plot_td(brir_selected[0][0:44100],'Input BRIR room 0 az 0')
        
        
        total_rooms = len(brir_list)
        total_azim_brir = len(brir_list[0])
        total_chan_brir = len(brir_list[0][0])
        total_samples_brir = len(brir_list[0][0][0])
        

        log_string = 'Input BRIRs loaded'
        if CN.LOG_INFO == 1:
            logging.info(log_string)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
        #
        # section to rearrange BRIR array
        #
        
        
        if CN.ROLL_ROOM == 1:
            brir_room_roll = 17
            brir_list = np.roll(brir_list,brir_room_roll,axis=0)
        
        #
        # section to create ordered list of rooms smallest to largest
        #
        
        room_energy_set = np.zeros((total_rooms,2))
        #get ratio of energy in first 4000 samples to energy in next 12000 samples
        for room in range(total_rooms):
            room_energy_set[room][1] = np.divide(np.sum(np.abs(brir_list[room][0][0][0:2000])),np.sum(np.abs(brir_list[room][0][0][2000:16000])))
            room_energy_set[room][0] = room
        #sort by largest ratio first
        room_ind_desc_size = np.flip(np.argsort( room_energy_set[:,1] ))
        room_energy_set_sorted = room_energy_set[room_ind_desc_size]
        
        
        if CN.SORT_ROOMS_BY_SIZE == 1:
            room_order_list=np.copy(room_ind_desc_size)
        else:
            room_order_list=(np.copy(room_energy_set[:,0])).astype(int)

        #
        ## windowing of BRIRs for direct removal
        #
        
        for room in range(total_rooms):
            for azim in range(total_azim_brir):
                for chan in range(total_chan_brir):
                    brir_list[room][azim][chan][:] = np.multiply(brir_list[room][azim][chan][:],direct_removal_win)

        if CN.PLOT_ENABLE == 1:
            room_selected = brir_list[0]
            brir_selected = room_selected[0]
            hf.plot_td(brir_selected[0][0:44100],'Windowed BRIR room 0 az 0')
            hf.plot_td(direct_removal_win,'direct_removal_win')
            hf.plot_td(direct_removal_win_b,'direct_removal_win_b')
        

        log_string = 'Direct signal removed'
        if CN.LOG_INFO == 1:
            logging.info(log_string)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
        
        #
        ## set BRIR level to 0dB
        #
        
        for room in range(total_rooms):
            #energy=np.sum(brir_list[room][azim][chan][:]*brir_list[room][azim][chan][:])
            
            data_fft = np.fft.fft(brir_list[room][0][0][:])#ensure left channel is taken for both channels
            mag_fft=np.abs(data_fft)
            average_mag = np.mean(mag_fft[150:5500])#100:7500
            
            #sum_amp = np.sum(np.abs(data_fft))
            
            if average_mag == 0:
                hf.print_message('0 mag detected')
                
            for azim in range(total_azim_brir):
                for chan in range(total_chan_brir):
                    brir_list[room][azim][chan][:] = np.divide(brir_list[room][azim][chan][:],average_mag)
        
        
        if CN.PLOT_ENABLE == 1:
            room_selected = brir_list[0]
            brir_selected = room_selected[0]
            hf.plot_td(brir_selected[0][0:44100],'level matched BRIR room 0 az 0')
        

        log_string = 'BRIRs level matched'
        if CN.LOG_INFO == 1:
            logging.info(log_string)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
        
        #
        ## TD alignment of BRIRs
        #
        
        #for each room, 
        #add current BRIR and next room's BRIR together
        #calculate group delay of combined BRIR at 0-200Hz
        #add to array
        #shift next BRIR by x (50?) samples, repeat above steps
        #find TD shift (delay) where group delay is minimised
        #shift next room BRIR by that delay
        
        #method 1: take current room, compare group delay
        #method 2: take sum of all prior rooms and this room, compare group delay in low freqs
        #method 3: take sum of all prior rooms and this room, compare group delay in low freqs (first x samples)
        #method 4: take sum of all prior rooms and this room, low pass, compare peak to peak distance in low freqs
        #method 5: Same as 4 but calculate distance from peak to peak within a X sample window
        
        delay_eval_set = np.zeros((total_rooms,total_azim_brir,CN.NUM_INTERVALS))
        
        #go through each room in the ordered list
        for i, room in enumerate(room_order_list[0:total_rooms-1]):#room in range(total_rooms-1)
            this_room_orig_idx=room_order_list[i]
            next_room_orig_idx=room_order_list[i+1]
            for azim in range(total_azim_brir):
                #section to calculate best delay for next room to align with this room
                azim_deg = int(azim*CN.NEAREST_AZ_BRIR)
                
                if CN.ALIGNMENT_METHOD == 1:
                    #method 1: take current room 
                    this_room = np.copy(brir_list[this_room_orig_idx][azim][0][:])
                else:
                    #method 2: take sum of all prior rooms and this room
                    rooms_to_add = 0
                    this_room = data_pad_zeros.copy()
                    for cum_room in range(i+1):
                        cum_room_orig_idx = room_order_list[cum_room]
                        rooms_to_add = rooms_to_add+1
                        this_room = np.add(this_room,brir_list[cum_room_orig_idx][azim][0][:])
                    this_room = np.divide(this_room,rooms_to_add) 
        
                if CN.ALIGN_LIMIT_AZIM == 1:
                    if azim_deg <= 0: #30:
                        calc_delay = 1
                    else:
                        calc_delay = 0
                else:
                    calc_delay = 1
                    
                if calc_delay == 1:
                    next_room = np.copy(brir_list[next_room_orig_idx][azim][0][:])
                    for delay in range(CN.NUM_INTERVALS):
                        
                        #shift next room BRIR
                        current_shift = CN.MIN_T_SHIFT+(delay*CN.T_SHIFT_INTERVAL)
                        n_room_shift = np.roll(next_room,current_shift)
                        #add current room BRIR to shifted next room BRIR
                        sum_ir = np.add(this_room,n_room_shift)
                        #calculate group delay
                        
                        #Method 3: calculate group delay vs freq (crop IR first)
                        if CN.ALIGNMENT_METHOD == 3:
                            #crop IR
                            sum_ir_pad = data_pad_zeros.copy()
                            sum_ir_pad[CN.DELAY_WIN_MIN_T:CN.DELAY_WIN_MAX_T] = sum_ir[CN.DELAY_WIN_MIN_T:CN.DELAY_WIN_MAX_T]
                            grp_delay = hf.group_delay(sum_ir_pad)
                        #Methods 1-2: calculate group delay vs freq
                        elif CN.ALIGNMENT_METHOD <= 2:
                            grp_delay = hf.group_delay(sum_ir)
                        
                        #Methods 1-3: Calculate average group delay for this alignment and store in array
                        if CN.ALIGNMENT_METHOD <= 3:
                            #convert to ms
                            grp_delay_ms = grp_delay*1000/CN.FS
                            #plt.plot(grp_delay_ms)
                            rms_grp_delay = np.sqrt(np.mean(grp_delay_ms[CN.GRP_DELAY_MIN_F:CN.GRP_DELAY_MAX_F]**2))
                            #store in array
                            delay_eval_set[next_room_orig_idx][azim][delay] = rms_grp_delay
                        #Method 4: calculate distance from peak to peak for this alignment and store in array
                        elif CN.ALIGNMENT_METHOD == 4:
                            sum_ir_lp = hf.signal_lowpass_filter(sum_ir, CN.CUTOFF_ALIGNMENT, CN.FS, CN.ORDER)
                            peak_to_peak = np.abs(np.max(sum_ir_lp[CN.DELAY_WIN_MIN_T:CN.DELAY_WIN_MAX_T])-np.min(sum_ir_lp[CN.DELAY_WIN_MIN_T:CN.DELAY_WIN_MAX_T]))
                            delay_eval_set[next_room_orig_idx][azim][delay] = peak_to_peak
                        #method 5: calculate distance from peak to peak within a X sample window
                        elif CN.ALIGNMENT_METHOD == 5:
                            sum_ir_lp = hf.signal_lowpass_filter(sum_ir, CN.CUTOFF_ALIGNMENT, CN.FS, CN.ORDER)
                            peak_to_peak_iter=0
                            for samples in range(len(sum_ir_lp[CN.DELAY_WIN_MIN_T:CN.DELAY_WIN_MAX_T])):
                                peak_to_peak = np.abs(np.max(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+CN.PEAK_TO_PEAK_WINDOW])-np.min(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+CN.PEAK_TO_PEAK_WINDOW]))
                                #if this window has larger pk to pk, store in iter var
                                if peak_to_peak > peak_to_peak_iter:
                                    peak_to_peak_iter = peak_to_peak
                            #store largest pk to pk distance of all windows into delay set
                            delay_eval_set[next_room_orig_idx][azim][delay] = peak_to_peak_iter
                    
                    #shift next room by delay that has largest peak to peak distance (method 4 and 5)
                    if CN.ALIGNMENT_METHOD == 4 or CN.ALIGNMENT_METHOD == 5:
                        index_shift = np.argmax(delay_eval_set[next_room_orig_idx][azim][:])
                    else:
                        index_shift = np.argmin(delay_eval_set[next_room_orig_idx][azim][:])
                
                samples_shift=CN.MIN_T_SHIFT+(index_shift*CN.T_SHIFT_INTERVAL)
                brir_list[next_room_orig_idx][azim][0][:] = np.roll(brir_list[next_room_orig_idx][azim][0][:],samples_shift)#left
                brir_list[next_room_orig_idx][azim][1][:] = np.roll(brir_list[next_room_orig_idx][azim][1][:],samples_shift)#right
                
                #20240511: set end of array to zero to remove any data shifted to end of array
                if samples_shift < 0:
                    brir_list[next_room_orig_idx][azim][0][CN.MIN_T_SHIFT:] = brir_list[next_room_orig_idx][azim][0][CN.MIN_T_SHIFT:]*0#left
                    brir_list[next_room_orig_idx][azim][1][CN.MIN_T_SHIFT:] = brir_list[next_room_orig_idx][azim][0][CN.MIN_T_SHIFT:]*0#right
                
            
            if CN.LOG_INFO == 1:
                logging.info('delay index = ' + str(index_shift) + ' for room ' + str(next_room_orig_idx))
             

        log_string = 'BRIRs aligned'
        if CN.LOG_INFO == 1:
            logging.info(log_string)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)

        #adjust levels of BRIRs -> divide by total rooms to be integrated
        
        num_sel_rooms = total_rooms
        
        for room in range(total_rooms):
            for azim in range(total_azim_brir):
                for chan in range(total_chan_brir):
                    brir_list[room][azim][chan][:] = np.divide(brir_list[room][azim][chan][:],num_sel_rooms)
                    
                    #section to apply different weighting to each room based on reverb quantities
                    if CN.ROOM_WEIGHTING_DESC == 1:
                        room_order = np.where(room_order_list == room)[0][0]
                        room_weight=np.divide(total_rooms-room_order,total_rooms)*2
                        brir_list[room][azim][chan][:] = np.multiply(brir_list[room][azim][chan][:],room_weight)
        

        log_string = 'Gains applied'
        if CN.LOG_INFO == 1:
            logging.info(log_string)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
    
        #
        #add BRIRs into output BRIR array
        #
        for room in range(total_rooms):
            for elev in range(CN.INTERIM_ELEVS):
                for azim in range(CN.OUTPUT_AZIMS):
                    for chan in range(total_chan_brir):
                        #round azim to nearest X deg and get new ID
                        azim_deg = int(azim*CN.NEAREST_AZ_HRIR)
                        brir_azim = hf.round_to_multiple(azim_deg,15)
                        if brir_azim >= 360:
                            brir_azim = 0
                        brir_azim_ind = int(brir_azim/CN.NEAREST_AZ_BRIR)
                        
                        brir_reverberation[elev][azim][chan][:] = np.add(brir_reverberation[elev][azim][chan][:],brir_list[room][brir_azim_ind][chan][:])
        
                        #plot each BRIR
                        if CN.PLOT_ENABLE == 1 and elev == 0 and azim == 0 and chan == 0:
                            
                            sample_brir = np.copy(brir_reverberation[elev][azim][0][:])
                            sample_brir_lp = hf.signal_lowpass_filter(sample_brir, CN.F_CROSSOVER, CN.FS, CN.ORDER)
                            plot_name = 'Cum Sum BRIR LP Room: 0 to Room: ' + str(room)
                            hf.plot_td(sample_brir_lp[0:1024],plot_name)
                            plot_name = 'Cum Sum BRIR Reverberation, Room: 0 to Room: ' + str(room)
                            hf.plot_td(sample_brir[500:44100],plot_name)
 

        log_string = 'BRIRs combined'
        if CN.LOG_INFO == 1:
            logging.info(log_string)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
    

        #
        # BRIR has been shifted so apply fade in window again to remove any overlap with HRIR
        #
        for elev in range(CN.INTERIM_ELEVS):
            for azim in range(CN.OUTPUT_AZIMS):
                for chan in range(total_chan_brir):
                    brir_reverberation[elev][azim][chan][:] = np.multiply(brir_reverberation[elev][azim][chan][:],direct_removal_win_b)

        #
        # set BRIR level to 0 again due to shifting
        #            
        data_fft = np.fft.fft(brir_reverberation[0][0][0][:])
        mag_fft=np.abs(data_fft)
        average_mag = np.mean(mag_fft[100:7500])

        for elev in range(CN.INTERIM_ELEVS):
            for azim in range(CN.OUTPUT_AZIMS):
                for chan in range(total_chan_brir):
                    brir_reverberation[elev][azim][chan][:] = np.divide(brir_reverberation[elev][azim][chan][:],average_mag)

        #plot BRIR
        if CN.PLOT_ENABLE == 1:
            
            sample_brir = np.copy(brir_reverberation[0][0][0][:])
            sample_brir_lp = hf.signal_lowpass_filter(sample_brir, CN.F_CROSSOVER, CN.FS, CN.ORDER)
            plot_name = 'step 8 Sum BRIR LP All Rooms'
            hf.plot_td(sample_brir_lp[0:1024],plot_name)

        #
        ## save brir_reverberation array to a file to retrieve later
        #

        
        out_file_path = CN.BASE_DIR_PATH/'data'/'interim'/'brir_reverberation_data.npy'
        
        np.save(out_file_path,brir_reverberation)
 

        log_string = 'Binary saved in data folder'
        if CN.LOG_INFO == 1:
            logging.info(log_string)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
  
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        
    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    if CN.LOG_INFO == 1:
        logging.info('Execution time:' + str(elapsed_time) + ' seconds')
        
        
        


def generate_integrated_brir(hrtf_type, direct_gain_db, room_target, pinna_comp, target_rt60, report_progress=0, gui_logger=None):   
    """
    Function to generate customised BRIR from below parameters

    :param hrtf_type: int, select HRTF type: 1 = KU100, 2 = kemar L pinna, 3 = kemar N pinna, 4 = B&K 4128 HATS, 5 = DADEC, 6 = HEAD acoustics HMSII.2, 7 = G.R.A.S.KEMAR (new), 8 = Bruel & Kjaer Type 4128C (BKwHA)
    :param direct_gain_db: float, adjust gain of direct sound in dB
    :param room_target: int, 0 = flat (no room target), 1 = ASH Room target, 2 = Harman target
    :param pinna_comp: int, 0 = equalised for in ear headphones (with additional eq), 1 = in ear headphones (without additional eq), 2 = over/on ear headphones (with additional eq), 3 = over/on ear headphones (without additional eq)
    :param target_rt60: int, value in ms for target reverberation time
    :param report_progress: int, 1 = update progress to progress bar in gui, set to 0 if no gui
    :param gui_logger: gui logger object for dearpygui
    :return: numpy array containing set of BRIRs. 4d array. d1 = elevations, d2 = azimuths, d3 = channels, d4 = samples
    """ 
    
    
    # get the start time
    st = time.time()
    
    #BRIR_out array will be populated BRIRs for all directions
    brir_out=np.zeros((CN.OUTPUT_ELEVS,CN.OUTPUT_AZIMS,2,CN.N_FFT))   
    #brir_out_selection array will be populated BRIRs for only selected directions
    brir_out_selection=np.zeros((CN.OUTPUT_ELEVS,CN.OUTPUT_AZIMS,2,CN.N_FFT))  

    try:

        #impulse
        impulse=np.zeros(CN.N_FFT)
        impulse[0]=1
        fr_flat_mag = np.abs(np.fft.fft(impulse))
        fr_flat = hf.mag2db(fr_flat_mag)

        #windows
        data_pad_zeros=np.zeros(CN.N_FFT)
        data_pad_ones=np.ones(CN.N_FFT)

        #initial rise window
        initial_hanning_size=40
        initial_hanning_start=0#190
        hann_initial_full=np.hanning(initial_hanning_size)
        hann_initial = np.split(hann_initial_full,2)[0]
        initial_removal_win = data_pad_zeros.copy()
        initial_removal_win[initial_hanning_start:initial_hanning_start+int(initial_hanning_size/2)] = hann_initial
        initial_removal_win[initial_hanning_start+int(initial_hanning_size/2):]=data_pad_ones[initial_hanning_start+int(initial_hanning_size/2):]

        #fade out window
        fade_hanning_size=65536
        fade_hanning_start=int(fade_hanning_size/2)
        hann_fade_full=np.hanning(fade_hanning_size)
        hann_fade = np.split(hann_fade_full,2)[1]
        fade_out_win = data_pad_ones.copy()
        fade_out_win[fade_hanning_start:fade_hanning_start+int(fade_hanning_size/2)] = hann_fade
        fade_out_win[fade_hanning_start+int(fade_hanning_size/2):]=data_pad_zeros[fade_hanning_start+int(fade_hanning_size/2):]

        #
        # load room target filter (FIR)
        if room_target > 0:
            mat_fname = pjoin(CN.DATA_DIR_INT, 'room_targets_firs.mat')
            room_target_mat = mat73.loadmat(mat_fname)
            room_target_fir = room_target_mat[CN.ROOM_TARGET_LIST_FIRS[room_target]]
        
        #
        # load pinna comp filter (FIR)
        #
        mat_fname = pjoin(CN.DATA_DIR_INT, 'headphone_pinna_comp_fir.mat')
        pinna_comp_mat = mat73.loadmat(mat_fname)
        pinna_comp_fir = pinna_comp_mat['ash_hp_pinna_comp_fir'][0:4096]
        
        #
        # load additional gain for SUB BRIR (FIR)
        #
        mat_fname = pjoin(CN.DATA_DIR_INT, 'brir_eq_filters.mat')
        sub_eq_fir_mat = mat73.loadmat(mat_fname)
        sub_eq_fir = sub_eq_fir_mat['ash_sub_eq_fir'][0:16384]
        
        #
        # load RT60 Target Response (FIR)
        #
        mat_fname = pjoin(CN.DATA_DIR_INT, 'room_rt60_target_response_fir.mat')
        rt60_target_mat = mat73.loadmat(mat_fname)
        rt60_target_fir=np.zeros(CN.N_FFT)
        rt60_target_fir[0:4096] = rt60_target_mat['ref_room_late_fir'][0:4096]
        data_fft = np.fft.fft(rt60_target_fir)#ensure left channel is taken for both channels
        rt60_target_mag=np.abs(data_fft)
        
        
        #
        # load additional headphone eq
        #
        apply_add_hp_eq = 0
        if CN.APPLY_ADD_HP_EQ > 0:
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


        log_string = 'Input filters loaded'
        if CN.LOG_INFO == 1:
            logging.info(log_string)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
        if report_progress == 1:
            progress = 2/100
            dpg.set_value("progress_bar_brir", progress)
            dpg.configure_item("progress_bar_brir", overlay = str(int(progress*100))+'%')
        
        #
        # load sub bass BRIR (FIR)
        #
        
        #deprecated: load smaller variant if target RT60 is low
        selected_sub_brir = ''
        if target_rt60 < 350:
            selected_sub_brir= 'ash_sub_brir'#'ash_sub_brir_small'
        else:
            selected_sub_brir= 'ash_sub_brir'
                
        mat_fname = pjoin(CN.DATA_DIR_INT, 'sub_brir_dataset.mat')
        sub_brir_mat = mat73.loadmat(mat_fname)
        sub_brir_ir = np.zeros((2,CN.N_FFT))
        sub_brir_ir[0,:] = sub_brir_mat[selected_sub_brir][0][0:CN.N_FFT]
        sub_brir_ir[1,:] = sub_brir_mat[selected_sub_brir][1][0:CN.N_FFT]
        
        #variable crossover depending on RT60
        f_crossover_var=CN.F_CROSSOVER
        if target_rt60 < 350:
            f_crossover_var=145#145
        else:
            f_crossover_var=145#145


        log_string = 'Low frequency response loaded'
        if CN.LOG_INFO == 1:
            logging.info(log_string)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
        if report_progress == 1:
            progress = 5/100
            dpg.set_value("progress_bar_brir", progress)
            dpg.configure_item("progress_bar_brir", overlay = str(int(progress*100))+'%')
            
        #
        # Load BRIR reverberation data
        #
        brir_file_path = CN.BASE_DIR_PATH/'data'/'interim'/'brir_reverberation_data.npy'
        brir_reverberation = np.load(brir_file_path)
        
        
        #plot an input BRIR
        if CN.PLOT_ENABLE == 1:  
            sample_brir = np.copy(brir_reverberation[0][0][0][:])
            plot_name = 'Cum Sum BRIR Reverberation, Room: 0 to Room: 71'
            hf.plot_td(sample_brir[500:44100],plot_name)
        
        
        #
        # load HRIRs
        #
        #elev ranges from -60 to +60 deg in 15 deg intervals
        #azim ranges from 0 to 360 deg in 5 deg intervals
        
        mat_fname = pjoin(CN.DATA_DIR_INT, 'hrir_dataset_compensated.mat')
        
        #old format mat files
        #annots = loadmat(mat_fname)
        #new format MATLAB 7.3 files
        hrirs = mat73.loadmat(mat_fname)
        
        #load matrix, get list of numpy arrays
        hrir_list = [[element for element in upperElement] for upperElement in hrirs['ash_input_hrir']]
        #grab desired hrtf. Returns 9x72x2x65536 array
        hrir_selected = hrir_list[hrtf_type-1]
        total_elev_hrir = len(hrir_selected)
        total_azim_hrir = len(hrir_selected[0])
        total_chan_hrir = len(hrir_selected[0][0])
        total_samples_hrir = len(hrir_selected[0][0][0])
        
        if CN.PLOT_ENABLE == 1:
            elev_selected = hrir_selected[0]
            azim_selected = elev_selected[0]
            hf.plot_td(azim_selected[0][0:total_samples_hrir],'input HRIR elev 0 az 0')
        
   
        log_string = 'HRIRs loaded'
        if CN.LOG_INFO == 1:
            logging.info(log_string)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
        if report_progress == 1:
            progress = 7/100
            dpg.set_value("progress_bar_brir", progress)
            dpg.configure_item("progress_bar_brir", overlay = str(int(progress*100))+'%')
            
        #
        ## set HRIR level to 0
        #
        mag_range_a=200
        mag_range_b=4000
        avg_mag_sum = 0
        for azim in range(total_azim_hrir):
            data_fft = np.fft.fft(hf.padarray(hrir_selected[4][azim][0][:],CN.N_FFT))
            mag_fft=np.abs(data_fft)
            avg_mag_azim = np.mean(mag_fft[mag_range_a:mag_range_b])
            avg_mag_sum=avg_mag_sum+avg_mag_azim
        avg_mag=avg_mag_sum/total_azim_hrir
        
        
        polarity=1
        #invert polarity of HRTF 4 (04: B&K Type 4128) and 6 (06: DADEC (MMHR-HRIR)) to align with reference
        if hrtf_type == 4 or hrtf_type == 6:
            polarity=-1
        
        for elev in range(total_elev_hrir):
            for azim in range(total_azim_hrir):
                for chan in range(total_chan_hrir):
                    hrir_selected[elev][azim][chan][:] = np.divide(hrir_selected[elev][azim][chan][:],avg_mag)
                    hrir_selected[elev][azim][chan][:] = np.multiply(hrir_selected[elev][azim][chan][:],polarity)
        
        if CN.PLOT_ENABLE == 1:
            elev_selected = hrir_selected[0]
            azim_selected = elev_selected[0]
            hf.plot_td(azim_selected[0][0:total_samples_hrir],'0 level HRIR elev 0 az 0')
  
        #adjust levels of HRIRs for DRR scaling. Assume 0dB starting point
        #round to nearest 2 decimal places
        direct_gain_db = round(direct_gain_db,2)
        #limit gain
        if direct_gain_db > 10:
            direct_gain_db = 10
        elif direct_gain_db < -10:
            direct_gain_db = -10
        
        direct_gain = hf.db2mag(direct_gain_db)
        
        for elev in range(total_elev_hrir):
            for azim in range(total_azim_hrir):
                for chan in range(total_chan_hrir):
                    hrir_selected[elev][azim][chan][:] = np.multiply(hrir_selected[elev][azim][chan][:],CN.DIRECT_SCALING_FACTOR)
                    hrir_selected[elev][azim][chan][:] = np.multiply(hrir_selected[elev][azim][chan][:],direct_gain)
        

        log_string = 'HRIR levels adjusted'
        if CN.LOG_INFO == 1:
            logging.info(log_string)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
        if report_progress == 1:
            progress = 10/100
            dpg.set_value("progress_bar_brir", progress)
            dpg.configure_item("progress_bar_brir", overlay = str(int(progress*100))+'%')
            
        #
        # align HRIRs in time domain
        #
        
        # take 0 deg azim as reference
        index_peak_ref = np.argmax(np.abs(hrir_selected[4][0][0][:]))
        
        
        for elev in range(total_elev_hrir):
            for azim in range(total_azim_hrir):
                azim_deg = int(azim*CN.NEAREST_AZ_HRIR)
                #take left channel if azim < 180 deg, otherwise take right channel
                if azim_deg < 180:
                    index_peak_cur = np.argmax(np.abs(hrir_selected[elev][azim][0][:]))
                else:
                    index_peak_cur = np.argmax(np.abs(hrir_selected[elev][azim][1][:]))    
                hrir_shift = index_peak_ref-index_peak_cur
                
                for chan in range(total_chan_hrir):
                    hrir_selected[elev][azim][chan][:] = np.roll(hrir_selected[elev][azim][chan][:],hrir_shift)
 

        log_string = 'HRIRs aligned'
        if CN.LOG_INFO == 1:
            logging.info(log_string)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
        if report_progress == 1:
            progress = 15/100
            dpg.set_value("progress_bar_brir", progress)
            dpg.configure_item("progress_bar_brir", overlay = str(int(progress*100))+'%')
            
        #
        ## Integration of BRIRs with HRIRs through TD interpolation and DRR scaling
        #
        
        #
        #apply high pass filter to hrirs
        #
        for elev in range(total_elev_hrir):
            for azim in range(total_azim_hrir):
                for chan in range(total_chan_hrir):  
                    hrir_eq_a_h = np.copy(hrir_selected[elev][azim][chan][:])
                    #apply hp filter
                    hrir_eq_b_h = hf.signal_highpass_filter(hrir_eq_a_h, CN.F_CROSSOVER, CN.FS, CN.ORDER)
                    hrir_selected[elev][azim][chan][:] = hrir_eq_b_h[0:CN.N_FFT] 

        #
        # grab BRIRs from interim matrix and place in output matrix
        #
        for elev in range(total_elev_hrir):
            for azim in range(total_azim_hrir):
                for chan in range(CN.TOTAL_CHAN_BRIR):
                    brir_out[elev][azim][chan][:] = np.copy(brir_reverberation[0][azim][chan][:])
   
        #plot BRIR and HRIR
        if CN.PLOT_ENABLE == 1:
            
            sample_brir = np.copy(brir_out[4][0][0][:])
            sample_brir_lp = hf.signal_lowpass_filter(sample_brir, CN.F_CROSSOVER, CN.FS, CN.ORDER)
            plot_name = 'Sum BRIR LP All Rooms'
            hf.plot_td(sample_brir_lp[0:1024],plot_name)
            plot_name = 'Sum BRIR All Rooms'
            hf.plot_td(sample_brir[500:44100],plot_name)
            
            sample_hrir = np.copy(hrir_selected[4][0][0][0:total_samples_hrir])
            plot_name = 'HRIR before integration'
            hf.plot_td(sample_hrir[0:total_samples_hrir],plot_name)
    
        #
        #add HRIR into output BRIR array
        #
        for elev in range(total_elev_hrir):
            for azim in range(total_azim_hrir):
                for chan in range(total_chan_hrir):      
                    brir_out[elev][azim][chan][0:total_samples_hrir] = brir_out[elev][azim][chan][0:total_samples_hrir] + hrir_selected[elev][azim][chan][0:total_samples_hrir]
        
        
        #plot integrated BRIR and HRIR
        if CN.PLOT_ENABLE == 1:
            sample_brir = np.copy(brir_out[4][0][0][:])
            plot_name = 'Intergrated HRIR and BRIR All Rooms'
            hf.plot_td(sample_brir[0:44100],plot_name)
            
            sample_brir = np.copy(brir_out[8][71][0][:])
            plot_name = 'Intergrated HRIR and BRIR All Rooms'
            hf.plot_td(sample_brir[0:44100],plot_name)
 
 
        log_string = 'HRIRs integrated with reverberation'
        if CN.LOG_INFO == 1:
            logging.info(log_string)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
        if report_progress == 1:
            progress = 25/100
            dpg.set_value("progress_bar_brir", progress)
            dpg.configure_item("progress_bar_brir", overlay = str(int(progress*100))+'%')
        
        #
        ## Reverb shaping and EQ
        #
        
        #apply to each room, before TD interpolation
        
        #limit RT60 target
        if target_rt60 < 200:
            target_rt60=200
        elif target_rt60 > 1250:
            target_rt60=1250

        rt60_comp_factor=1
        #convert to sample ID to get start of reverb response
        rt60_snap_start = int(((target_rt60*rt60_comp_factor)/1000)*CN.FS)
        r_fade_hanning_start=200
        
        #generate hann windows for reverb EQ
        r_fade_hanning_size=np.abs(rt60_snap_start-r_fade_hanning_start)*2
        if CN.WINDOW_TYPE == 1:
            hann_r_fade_full=np.hanning(r_fade_hanning_size)
        elif CN.WINDOW_TYPE == 2:
            hann_r_fade_full=np.bartlett(r_fade_hanning_size)
        elif CN.WINDOW_TYPE == 3:
            hann_r_fade_full=np.blackman(r_fade_hanning_size)    
        elif CN.WINDOW_TYPE == 4:
            hann_r_fade_full=np.hamming(r_fade_hanning_size)   
        else:
            hann_r_fade_full=np.bartlett(r_fade_hanning_size)
        

        hann_r_fade_out = np.split(hann_r_fade_full,2)[1]
        hann_r_fade_in = np.split(hann_r_fade_full,2)[0]
        
        #fade in window
        r_fade_in_win = data_pad_zeros.copy()
        r_fade_in_win[r_fade_hanning_start:r_fade_hanning_start+int(r_fade_hanning_size/2)] = hann_r_fade_in
        r_fade_in_win[r_fade_hanning_start+int(r_fade_hanning_size/2):]=data_pad_ones[r_fade_hanning_start+int(r_fade_hanning_size/2):]
        
        #fade out window 
        r_fade_out_win = data_pad_zeros.copy()
        r_fade_out_win[0:r_fade_hanning_start] = data_pad_ones[0:r_fade_hanning_start]
        r_fade_out_win[r_fade_hanning_start:r_fade_hanning_start+int(r_fade_hanning_size/2)] = hann_r_fade_out
        r_fade_out_win[r_fade_hanning_start+int(r_fade_hanning_size/2):]=data_pad_zeros[r_fade_hanning_start+int(r_fade_hanning_size/2):]
        
        
        if CN.PLOT_ENABLE == 1:
            plt.plot(r_fade_in_win)
            plt.plot(r_fade_out_win)
        
        #for each room
        #get current reverb response
        #calc difference between target and current response
        #create min phase FIR for reverb reduction
        #for each azim
        #create copy and apply fade out window
        #create copy and apply fade in window
        #convolve fade in windowed BRIR with reverb reduction FIR
        #add both together

        num_brirs_avg = 0
        brir_fft_avg_db = fr_flat.copy()
        brir_current=data_pad_zeros.copy() 
        #get current reverb response from diffuse field spectrum (NEW METHOD)
        for elev in range(total_elev_hrir):
            for azim in range(total_azim_hrir):
                for chan in range(total_chan_hrir):   
                    brir_current[0:CN.N_FFT-rt60_snap_start] = np.copy(brir_out[elev][azim][chan][rt60_snap_start:CN.N_FFT])
                    brir_current_fft = np.fft.fft(brir_current)#
                    brir_current_mag_fft=np.abs(brir_current_fft)
                    brir_current_db_fft = hf.mag2db(brir_current_mag_fft)
                    
                    brir_fft_avg_db = np.add(brir_fft_avg_db,brir_current_db_fft)
                    
                    num_brirs_avg = num_brirs_avg+1

        #divide by total number of brirs
        brir_fft_avg_db = brir_fft_avg_db/num_brirs_avg
        #convert to mag
        brir_fft_avg_mag = hf.db2mag(brir_fft_avg_db)
        room_late_mag_fft=brir_fft_avg_mag.copy()
        
        #smoothing
        #room_late_mag_sm = hf.smooth_fft(room_late_mag_fft)
        room_late_mag_sm = hf.smooth_fft(room_late_mag_fft, 1000, 50, 750)
        room_late_mag_sm = hf.smooth_fft(room_late_mag_sm, 30000, 30, 1600)#secondary smoothing across all freqs
        room_late_db_fft = hf.mag2db(room_late_mag_sm)
        rt60_target_db = hf.mag2db(rt60_target_mag)
        #calc difference between target and current response
        room_late_comp_db = np.subtract(rt60_target_db,room_late_db_fft)
        room_late_comp_mag = hf.db2mag(room_late_comp_db)
        #level ends of spectrum
        room_late_comp_mag = hf.level_spectrum_ends(room_late_comp_mag, 20, 19000)
        #room_late_comp_mag = hf.smooth_fft(room_late_comp_mag)
        
        #if mag of current room < mag of target, do not apply any filter (already below target)   
        #divide by zero warning can be ignored due to this condition (room late mag wont be greater than target)
        if np.mean(room_late_mag_fft)>np.mean(rt60_target_mag):
            
            if CN.LOG_INFO == 1:
                logging.info('shaping reverberation')
            
            #create min phase FIR for reverb reduction
            room_late_comp_min = hf.mag_to_min_fir(room_late_comp_mag)
   
            #for each azim
            for elev in range(total_elev_hrir):
                for azim in range(total_azim_hrir):
                    #create copy and apply fade out window
                    brir_fade_out_l = np.multiply(brir_out[elev][azim][0][:],r_fade_out_win)
                    brir_fade_out_r = np.multiply(brir_out[elev][azim][1][:],r_fade_out_win)
                    #create copy and apply fade in window
                    brir_fade_in_l = np.multiply(brir_out[elev][azim][0][:],r_fade_in_win)
                    brir_fade_in_r = np.multiply(brir_out[elev][azim][1][:],r_fade_in_win)
                    #convolve fade in windowed BRIR with reverb reduction FIR
                    brir_fade_in_filt_l = sp.signal.convolve(brir_fade_in_l,room_late_comp_min, 'full', 'auto')
                    brir_fade_in_filt_r = sp.signal.convolve(brir_fade_in_r,room_late_comp_min, 'full', 'auto')
                    #add both together
                    brir_combined_l = np.add(brir_fade_out_l,brir_fade_in_filt_l[0:CN.N_FFT])
                    brir_combined_r = np.add(brir_fade_out_r,brir_fade_in_filt_r[0:CN.N_FFT])
                    #replace previous BRIR in set
                    brir_out[elev][azim][0][:] = np.copy(brir_combined_l)
                    brir_out[elev][azim][1][:] = np.copy(brir_combined_r)
                
        
        if CN.PLOT_ENABLE == 1:
            hf.plot_data(room_late_mag_fft,'room_late_mag_fft')
            hf.plot_data(room_late_mag_sm,'room_late_mag_sm')
            hf.plot_data(rt60_target_mag,'rt60_target_mag')
            hf.plot_data(room_late_comp_mag,'room_late_comp_mag')


        log_string = 'Reverberation time adjusted'
        if CN.LOG_INFO == 1:
            logging.info(log_string)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
        if report_progress == 1:
            progress = 50/100
            dpg.set_value("progress_bar_brir", progress)
            dpg.configure_item("progress_bar_brir", overlay = str(int(progress*100))+'%')
        #
        # Sub BRIR integration
        #
        
        
        delay_eval_set_sub = np.zeros((CN.NUM_INTERVALS_C))
        
        #method 1: take current room, compare group delay
        #method 2: take sum of all prior rooms and this room, compare group delay in low freqs
        #method 3: take sum of all prior rooms and this room, compare group delay in low freqs (first x samples)
        #method 4: take sum of all prior rooms and this room, low pass, compare peak to peak distance in low freqs
        #method 5: Same as 4 but calculate distance from peak to peak within a 400 sample window 
        
        if CN.ENABLE_SUB_INTEGRATION == 1:
        
            #set level of sub BRIR to 0 at low freqs
            data_fft = np.fft.fft(sub_brir_ir[0][:])
            mag_fft=np.abs(data_fft)
            average_mag = np.mean(mag_fft[30:290])
            if average_mag == 0:
                if CN.LOG_INFO == 1:
                    logging.info('0 magnitude detected')
            for chan in range(CN.TOTAL_CHAN_BRIR):
                sub_brir_ir[chan][:] = np.divide(sub_brir_ir[chan][:],average_mag)
        
            #set level of HF BRIR to 0 at low freqs
            data_fft_b = np.fft.fft(brir_out[0][0][0][:])
            mag_fft_b=np.abs(data_fft_b)
            average_mag_b = np.mean(mag_fft_b[50:290])
            if average_mag_b == 0:
                if CN.LOG_INFO == 1:
                    logging.info('0 magnitude detected')
            for elev in range(total_elev_hrir):
                for azim in range(total_azim_hrir):
                    for chan in range(total_chan_hrir):      
                        brir_out[elev][azim][chan][:] = np.divide(brir_out[elev][azim][chan][:],average_mag_b)

            #section to calculate best delay for next room to align with this room
            this_room_c = np.copy(brir_out[0][0][0][:])

            next_room_c = np.copy(sub_brir_ir[0][:])#check first room, fist azim, first channel

            for delay in range(CN.NUM_INTERVALS_C):
            
                #shift next room BRIR
                current_shift = CN.MIN_T_SHIFT_C+(delay*CN.T_SHIFT_INTERVAL_C)
                n_room_shift_c = np.roll(next_room_c,current_shift)
                
                #add current room BRIR to shifted next room BRIR
                sum_ir_c = np.add(this_room_c,n_room_shift_c)
                
                #Method 3: calculate group delay vs freq (crop IR first)
                if CN.ALIGNMENT_METHOD == 3:
                    #crop IR
                    sum_ir_pad = data_pad_zeros.copy()
                    sum_ir_pad[CN.DELAY_WIN_MIN_T:CN.DELAY_WIN_MAX_T] = sum_ir_c[CN.DELAY_WIN_MIN_T:CN.DELAY_WIN_MAX_T]
                    grp_delay = hf.group_delay(sum_ir_pad)
                #Methods 1-2: calculate group delay vs freq
                elif CN.ALIGNMENT_METHOD <= 2:
                    grp_delay = hf.group_delay(sum_ir_c)
                
                #Methods 1-3: Calculate average group delay for this alignment and store in array
                if CN.ALIGNMENT_METHOD <= 3:
                    #convert to ms
                    grp_delay_ms = grp_delay*1000/CN.FS
                    #average group delay within desired freq range
                    rms_grp_delay = np.sqrt(np.mean(grp_delay_ms[CN.GRP_DELAY_MIN_F:CN.GRP_DELAY_MAX_F]**2))
                    #store in array
                    delay_eval_set_sub[delay] = rms_grp_delay
                #Method 4: calculate distance from peak to peak for this alignment and store in array
                elif CN.ALIGNMENT_METHOD == 4:
                    sum_ir_lp = hf.signal_lowpass_filter(sum_ir_c, CN.CUTOFF_SUB, CN.FS, CN.ORDER)
                    peak_to_peak = np.abs(np.max(sum_ir_lp[CN.DELAY_WIN_MIN_T:CN.DELAY_WIN_MAX_T])-np.min(sum_ir_lp[CN.DELAY_WIN_MIN_T:CN.DELAY_WIN_MAX_T]))
                    delay_eval_set_sub[delay] = peak_to_peak
                #method 5: calculate distance from peak to peak within a 400 sample window
                elif CN.ALIGNMENT_METHOD == 5:
                    sum_ir_lp = hf.signal_lowpass_filter(sum_ir_c, CN.CUTOFF_SUB, CN.FS, CN.ORDER)
                    peak_to_peak_iter=0
                    for samples in range(len(sum_ir_lp[CN.DELAY_WIN_MIN_T:CN.DELAY_WIN_MAX_T])):
                        peak_to_peak = np.abs(np.max(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+CN.PEAK_TO_PEAK_WINDOW_SUB])-np.min(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+CN.PEAK_TO_PEAK_WINDOW_SUB]))
                        #if this window has larger pk to pk, store in iter var
                        if peak_to_peak > peak_to_peak_iter:
                            peak_to_peak_iter = peak_to_peak
                    #store largest pk to pk distance of all windows into delay set
                    delay_eval_set_sub[delay] = peak_to_peak_iter
            
            #shift next room by delay that has largest peak to peak distance (method 4 and 5)
            if CN.ALIGNMENT_METHOD == 4 or CN.ALIGNMENT_METHOD == 5:
                index_shift = np.argmax(delay_eval_set_sub[:])
            #shift next room by delay that has lowest grp delay (methods 1-3)
            else:
                index_shift = np.argmin(delay_eval_set_sub[:])
            samples_shift=CN.MIN_T_SHIFT_C+(index_shift*CN.T_SHIFT_INTERVAL_C)
            

            for chan in range(CN.TOTAL_CHAN_BRIR):
                sub_brir_ir[chan][:] = np.roll(sub_brir_ir[chan][:],samples_shift)
            

            if CN.LOG_INFO == 1:
                logging.info('delay index = ' + str(index_shift))
            
            #
            #apply low pass filter to sub BRIR
            #
            brir_eq_a_l = np.copy(sub_brir_ir[0][:])
            brir_eq_a_r = np.copy(sub_brir_ir[1][:])
            #apply lp filter
            brir_eq_b_l = hf.signal_lowpass_filter(brir_eq_a_l, f_crossover_var, CN.FS, CN.ORDER)
            brir_eq_b_r = hf.signal_lowpass_filter(brir_eq_a_r, f_crossover_var, CN.FS, CN.ORDER)
            sub_brir_ir[0][:] = brir_eq_b_l[0:CN.N_FFT] 
            sub_brir_ir[1][:] = brir_eq_b_r[0:CN.N_FFT]
            
            #plot BRIR before adding SUB 
            if CN.PLOT_ENABLE == 1:
                sample_brir = np.copy(brir_out[4][0][0][:])
                sample_brir_lp = hf.signal_lowpass_filter(sample_brir, f_crossover_var, CN.FS, CN.ORDER)
                plot_name = 'Sum BRIR + HRIR LP All Rooms before SUB Int'
                hf.plot_td(sample_brir_lp[0:1024],plot_name)
                
                sample_sub_lp = np.copy(sub_brir_ir[0][:])#already low passed
                plot_name = 'Sub IR after shifting before int'
                hf.plot_td(sample_sub_lp[0:1024],plot_name)
            
            #
            #apply high pass filter to hrir+brir
            #
            for elev in range(total_elev_hrir):
                for azim in range(total_azim_hrir):
                    for chan in range(total_chan_hrir):  
                        brir_eq_a_h = np.copy(brir_out[elev][azim][chan][:])
                        #apply hp filter
                        brir_eq_b_h = hf.signal_highpass_filter(brir_eq_a_h, f_crossover_var, CN.FS, CN.ORDER)
                        brir_out[elev][azim][chan][:] = brir_eq_b_h[0:CN.N_FFT] 
  
            #add SUB BRIR into output BRIR array
            for elev in range(total_elev_hrir):
                for azim in range(total_azim_hrir):
                    for chan in range(total_chan_hrir):      
                        brir_out[elev][azim][chan][:] = brir_out[elev][azim][chan][:] + sub_brir_ir[chan][:]
        
            #plot BRIR after adding SUB 
            if CN.PLOT_ENABLE == 1:
                
                sample_brir = np.copy(brir_out[4][0][0][:])
                sample_brir_lp = hf.signal_lowpass_filter(sample_brir, f_crossover_var, CN.FS, CN.ORDER)
                plot_name = 'Sum BRIR LP All Rooms after SUB Int'
                hf.plot_td(sample_brir_lp[0:1024],plot_name)


        log_string = 'Low frequency response integrated'
        if CN.LOG_INFO == 1:
            logging.info(log_string)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
        if report_progress == 1:
            progress = 75/100
            dpg.set_value("progress_bar_brir", progress)
            dpg.configure_item("progress_bar_brir", overlay = str(int(progress*100))+'%')
            
        #
        ## EQ correction
        #
        
        # HRIR has been filtered, so apply window to remove any aliasing
        for elev in range(total_elev_hrir):
            for azim in range(total_azim_hrir):
                for chan in range(CN.TOTAL_CHAN_BRIR):
                    brir_out[elev][azim][chan][:] = np.multiply(brir_out[elev][azim][chan][:],initial_removal_win)

        
        #determine magnitude response, assume 0 phase
        #set magnitude to 0dB
        #invert response
        #convert to IR
        #if using IFFT, result will be 0 phase and symmetric
        #if using ifftshift, result will be linear phase
        #get linear phase FIR, window, and convert to minimum phase
        #window min phase FIR again to remove artefacts
        
        num_brirs_avg = 0
        brir_fft_avg_db = fr_flat.copy()
        
        #get diffuse field spectrum
        for elev in range(total_elev_hrir):
            for azim in range(total_azim_hrir):
                for chan in range(total_chan_hrir):      
                    brir_current = brir_out[elev][azim][chan][:]
                    brir_current_fft = np.fft.fft(brir_current)#ensure left channel is taken for both channels
                    brir_current_mag_fft=np.abs(brir_current_fft)
                    brir_current_db_fft = hf.mag2db(brir_current_mag_fft)
                    
                    brir_fft_avg_db = np.add(brir_fft_avg_db,brir_current_db_fft)
                    
                    num_brirs_avg = num_brirs_avg+1
        
        #divide by total number of brirs
        brir_fft_avg_db = brir_fft_avg_db/num_brirs_avg
        #convert to mag
        brir_fft_avg_mag = hf.db2mag(brir_fft_avg_db)
        #level ends of spectrum
        brir_fft_avg_mag = hf.level_spectrum_ends(brir_fft_avg_mag, 40, 19000, smooth_win = 10)#150

        #smoothing - 5 stage process
        brir_fft_avg_mag_sm = hf.smooth_fft(brir_fft_avg_mag, 10000, 20, 1600)
        brir_fft_avg_mag_sm = hf.smooth_fft(brir_fft_avg_mag_sm, 1000, 20, 800)
        brir_fft_avg_mag_sm = hf.smooth_fft(brir_fft_avg_mag_sm, 700, 20, 400)
        brir_fft_avg_mag_sm = hf.smooth_fft(brir_fft_avg_mag_sm, 400, 20, 200)
        brir_fft_avg_mag_sm = hf.smooth_fft(brir_fft_avg_mag_sm, 150, 40, 10)
        brir_fft_avg_mag_sm = hf.smooth_fft(brir_fft_avg_mag_sm, 30000, 20, 1600)
        
        #invert response
        brir_fft_avg_mag_inv = hf.db2mag(hf.mag2db(brir_fft_avg_mag_sm)*-1)
        #create min phase FIR
        brir_df_inv_fir = hf.mag_to_min_fir(brir_fft_avg_mag_inv)
        
        #get matrix of desired output directions
        direction_matrix = brir_export.generate_direction_matrix(hrtf_type)

        for elev in range(total_elev_hrir):
            for azim in range(total_azim_hrir):
                for chan in range(total_chan_hrir):  
                    
                    #only apply equalisation if direction is in direction matrix
                    if direction_matrix[elev][azim][0][0] == 1: 
                    
                        #convolve BRIR with filters
                        brir_eq_a = np.copy(brir_out[elev][azim][chan][:])
                        #apply DF eq
                        brir_eq_b = sp.signal.convolve(brir_eq_a,brir_df_inv_fir, 'full', 'auto')
                        #apply room target
                        if room_target >= 1:
                            brir_eq_b = sp.signal.convolve(brir_eq_b,room_target_fir, 'full', 'auto')
                        #apply pinna compensation
                        if pinna_comp >= 2:
                            brir_eq_b = sp.signal.convolve(brir_eq_b,pinna_comp_fir, 'full', 'auto')
                        #apply additional eq for sub bass
                        if CN.ENABLE_SUB_INTEGRATION == 1 and CN.APPLY_SUB_EQ == 1:
                            brir_eq_b = sp.signal.convolve(brir_eq_b,sub_eq_fir, 'full', 'auto')
                        #apply additional eq for headphones
                        if CN.APPLY_ADD_HP_EQ > 0 and apply_add_hp_eq > 0:
                            brir_eq_b = sp.signal.convolve(brir_eq_b,data_addit_eq, 'full', 'auto')
                        brir_out_selection[elev][azim][chan][:] = brir_eq_b[0:CN.N_FFT]

        if CN.PLOT_ENABLE == 1:
            hf.plot_data(mag_fft,'mag_fft SUBRIR')
            hf.plot_data(brir_fft_avg_mag,'brir_fft_avg_mag')
            hf.plot_data(brir_current_mag_fft,'brir_current_mag_fft')
            hf.plot_data(brir_fft_avg_mag_sm,'brir_fft_avg_mag_sm')
            hf.plot_data(brir_fft_avg_mag_inv,'brir_fft_avg_mag_inv')


        log_string = 'Frequency response calibrated'
        if CN.LOG_INFO == 1:
            logging.info(log_string)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
        if report_progress == 1:
            progress = 90/100
            dpg.set_value("progress_bar_brir", progress)
            dpg.configure_item("progress_bar_brir", overlay = str(int(progress*100))+'%')

    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to generate BRIRs'
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
        
    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    if CN.LOG_INFO == 1:
        logging.info('Execution time:' + str(elapsed_time) + ' seconds')
        
    #return the integrated BRIR
    #return brir_out
    return brir_out_selection