# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 14:05:45 2024

@author: Shanon
"""

import numpy as np
from os.path import join as pjoin
import os
from scipy.io import wavfile
from ash_toolset import constants as CN
from ash_toolset import helper_functions as hf
from pathlib import Path
import mat73
import scipy as sp
import scipy.io as sio
import random
import logging
from SOFASonix import SOFAFile
from ash_toolset import pyquadfilter
if CN.SHOW_DEV_TOOLS == True:
    import noisereduce as nr
    import h5py
    import soundfile as sf

logger = logging.getLogger(__name__)
log_info=1

def extract_airs_from_recording(ir_set='fw', gui_logger=None):
    """
    function to extract individual IRs from a recording containing multiple IRs
    :param ir_set: str, name of impulse response set. Must correspond to a folder in ASH-Toolset\data\raw\ir_data\stream
    :return: None
    """
    
    min_len_s=1.0
    ampl_thresh=0.75#0.8
    output_wavs=1
    peak_tshift=50
    
    ir_data_folder = pjoin(CN.DATA_DIR_RAW, 'ir_data', 'stream',ir_set)
    ir_out_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'split_airs',ir_set)
    
    try:
    
        for root, dirs, files in os.walk(ir_data_folder):
            for filename in files:
                if '.wav' in filename:
                    #read wav file
                    wav_fname = pjoin(root, filename)
                    samplerate, data = wavfile.read(wav_fname)
                    fir_array = data / (2.**31)
                    
                    #resample if sample rate is not 44100
                    if samplerate != 44100:
                        fir_array = hf.resample_signal(fir_array, new_rate = 44100)
                        print('source samplerate: ' + str(samplerate))
                        print('resampled to 44100Hz')
                    
                    #find prominent peaks
                    peaks_indices = np.where(np.abs(fir_array) > ampl_thresh)[0]
                    #get time values of above indices
                    peaks_time= np.divide(peaks_indices,samplerate)
                    #get time since previous peak
                    peaks_time_diff=np.zeros(len(peaks_time))
                    for i in range(len(peaks_time)):
                        # #old method
                        # if i > 0:
                        #     peaks_time_diff[i]=peaks_time[i]-peaks_time[i-1]
                        #new method
                        if i < len(peaks_time)-1:
                            peaks_time_diff[i]=peaks_time[i+1]-peaks_time[i]    
                    #combine array
                    c1 = np.expand_dims(peaks_indices, axis=1)
                    c2 = np.expand_dims(peaks_time_diff, axis=1)
                    peaks_ind_tdiff = np.hstack((c1,c2))
                    #find prominent peaks where delay between previous peak is sufficiently high
                    ir_start_indices = peaks_indices[peaks_time_diff >= min_len_s]
                    #shift to the right by x samples
                    ir_start_ind_shift=np.subtract(ir_start_indices,peak_tshift)
                    fir_array_split = np.split(fir_array,ir_start_ind_shift)
                        
                    #name of dataset
                    dataset_name=filename.split('.wav')[0]
                    #loop through array to export wav files for testing
                    for idx, x in enumerate(fir_array_split):
                        out_file_name = dataset_name+'_'+str(idx)+ '.wav'
                        out_file_path = pjoin(ir_out_folder,dataset_name,out_file_name)
                        
                        #create dir if doesnt exist
                        output_file = Path(out_file_path)
                        output_file.parent.mkdir(exist_ok=True, parents=True)
                        
                        if output_wavs == 1 and idx>0:
                            hf.write2wav(file_name=out_file_path, data=x, samplerate=44100)
                    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to extract AIRs from recording for stream: ' + ir_set 
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_error(log_string)
    
    
    
    
    
def td_average_airs(ir_set='fw', gui_logger=None):
    """
    function to perform time domain averaging of multiple IRs
    saves air_reverberation array. num_out_sets * 2 channels * CN.N_FFT samples
    :param ir_set: str, name of impulse response set. Must correspond to a folder in ASH-Toolset\data\interim\ir_data\split_irs
    :return: None
    """
    
    samp_freq=44100
    output_wavs=1
    
    #windows
    data_pad_zeros=np.zeros(CN.N_FFT)
    data_pad_ones=np.ones(CN.N_FFT)
    
    total_chan_air=1
    lf_align=1
    num_out_sets=7
    
    #direct sound window for 2nd phase
    direct_hanning_size=300#350
    direct_hanning_start=51#101
    hann_direct_full=np.hanning(direct_hanning_size)
    hann_direct = np.split(hann_direct_full,2)[0]
    direct_removal_win_b = data_pad_zeros.copy()
    direct_removal_win_b[direct_hanning_start:direct_hanning_start+int(direct_hanning_size/2)] = hann_direct
    direct_removal_win_b[direct_hanning_start+int(direct_hanning_size/2):]=data_pad_ones[direct_hanning_start+int(direct_hanning_size/2):]
    
    #loop through folders
    ir_in_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'split_airs',ir_set)
    
    try:
    
        #get number of IRs
        ir_counter=0
        for root, dirs, files in os.walk(ir_in_folder):
            for filename in files:
                if '.wav' in filename:
                    ir_counter=ir_counter+1
                    
        #numpy array, num sets x num irs in each set x 2 channels x NFFT max samples
        irs_per_set=int(ir_counter/num_out_sets)+2
        air_data=np.zeros((num_out_sets,irs_per_set,2,CN.N_FFT))
        air_reverberation=np.zeros((num_out_sets,2,CN.N_FFT))
        
        set_counter=0
        ir_counter=0
        for root, dirs, files in os.walk(ir_in_folder):
            for filename in files:
                if '.wav' in filename:
                    #print('test')
                    
                    #read wav files
                    wav_fname = pjoin(root, filename)
                    samplerate, data = wavfile.read(wav_fname)
                    samp_freq=samplerate
                    fir_array = data / (2.**31)
                    fir_length = len(fir_array)
                    
                    #append into set list
                    current_set = set_counter%num_out_sets
                    
                    if current_set == 0 and set_counter>0:
                        ir_counter=ir_counter+1#increments every num_out_sets
                    
                    set_counter=set_counter+1
                    
                    extract_legth = min(CN.N_FFT,fir_length)
                    #load into numpy array
                    air_data[current_set,ir_counter,0,0:extract_legth]=fir_array[0:extract_legth,0]#L
                    air_data[current_set,ir_counter,1,0:extract_legth]=fir_array[0:extract_legth,1]#R
                    
      
        #perform time domain synchronous averaging
        #align in low frequencies
        if lf_align == 1:
 
            #contants for TD alignment of BRIRs
            t_shift_interval = CN.T_SHIFT_INTERVAL
            min_t_shift = CN.MIN_T_SHIFT_A
            max_t_shift = CN.MAX_T_SHIFT_A
            num_intervals = int(np.abs((max_t_shift-min_t_shift)/t_shift_interval))
            order=7#default 6
            delay_win_min_t = CN.DELAY_WIN_MIN_A
            delay_win_max_t = CN.DELAY_WIN_MAX_A
            delay_win_hop_size = CN.DELAY_WIN_HOP_SIZE
            delay_win_hops = CN.DELAY_WIN_HOPS_A
            
            cutoff_alignment = CN.CUTOFF_ALIGNMENT_AIR
            #peak to peak within a sufficiently small sample window
            peak_to_peak_window = int(np.divide(samp_freq,cutoff_alignment)*0.95) #int(np.divide(samp_freq,cutoff_alignment)) 
            
            
            delay_eval_set = np.zeros((num_out_sets,irs_per_set,num_intervals))
            
            for set_num in range(num_out_sets):
                #go through each room in the ordered list
                for ir in range(irs_per_set-1):#room in range(total_airs-1)
                    this_air_orig_idx=ir
                    next_air_orig_idx=ir+1
    
                    #method 2: take sum of all prior rooms and this room
                    rooms_to_add = 0
                    this_air = data_pad_zeros.copy()
                    for cum_air in range(ir+1):
                        cum_air_orig_idx = cum_air
                        rooms_to_add = rooms_to_add+1
                        this_air = np.add(this_air,air_data[set_num,cum_air_orig_idx,0,:])
                    this_air = np.divide(this_air,rooms_to_add) 
    
                    calc_delay = 1
                        
                    if calc_delay == 1:
                        next_air = np.copy(air_data[set_num,next_air_orig_idx,0,:])
                        for delay in range(num_intervals):
                            
                            #shift next room BRIR
                            current_shift = min_t_shift+(delay*t_shift_interval)
                            n_air_shift = np.roll(next_air,current_shift)
                            #add current room BRIR to shifted next room BRIR
                            sum_ir = np.add(this_air,n_air_shift)
                            #calculate group delay
         
                            sum_ir_lp = hf.signal_lowpass_filter(sum_ir, cutoff_alignment, samp_freq, order)
                            peak_to_peak_iter=0
                            for hop_id in range(delay_win_hops):
                                samples = hop_id*delay_win_hop_size
                                peak_to_peak = np.abs(np.max(sum_ir_lp[delay_win_min_t+samples:delay_win_min_t+samples+peak_to_peak_window])-np.min(sum_ir_lp[delay_win_min_t+samples:delay_win_min_t+samples+peak_to_peak_window]))
                                #if this window has larger pk to pk, store in iter var
                                if peak_to_peak > peak_to_peak_iter:
                                    peak_to_peak_iter = peak_to_peak
                            #store largest pk to pk distance of all windows into delay set
                            delay_eval_set[set_num,next_air_orig_idx,delay] = peak_to_peak_iter
                        
                        #shift next room by delay that has largest peak to peak distance (method 4 and 5)
                        index_shift = np.argmax(delay_eval_set[set_num,next_air_orig_idx,:])
        
                    
                    samples_shift=min_t_shift+(index_shift*t_shift_interval)
                    air_data[set_num,next_air_orig_idx,0,:] = np.roll(air_data[set_num,next_air_orig_idx,0,:],samples_shift)#left
                    air_data[set_num,next_air_orig_idx,1,:] = np.roll(air_data[set_num,next_air_orig_idx,1,:],samples_shift)#right
                    
                    #20240511: set end of array to zero to remove any data shifted to end of array
                    if samples_shift < 0:
                        air_data[set_num,next_air_orig_idx,0,min_t_shift:] = air_data[set_num,next_air_orig_idx,0,min_t_shift:]*0#left
                        air_data[set_num,next_air_orig_idx,1,min_t_shift:] = air_data[set_num,next_air_orig_idx,1,min_t_shift:]*0#right
                        
     
        #remove direction portion of signal?
        #add RIRs into output RIR array
        for set_num in range(num_out_sets):
            #add RIRs into output RIR array
            for ir in range(irs_per_set):
                for chan in range(total_chan_air):
                    air_reverberation[set_num,chan,:] = np.add(air_reverberation[set_num,chan,:],air_data[set_num,ir,chan,:])
            for chan in range(total_chan_air):
                # RIR has been shifted so apply fade in window again to remove any overlap with HRIR
                air_reverberation[set_num,chan,:] = np.multiply(air_reverberation[set_num,chan,:],direct_removal_win_b)
            
       
        #create dir if doesnt exist
        air_out_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'avg_airs',ir_set)
    
        npy_file_name = ir_set+'_td_avg.npy'
    
        out_file_path = pjoin(air_out_folder,npy_file_name)      
          
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        np.save(out_file_path,air_reverberation)
        
        #also save to wav for testing
        for set_num in range(num_out_sets):
            out_file_name = ir_set+'_'+str(set_num)+'_td_avg.wav'
            out_file_path = pjoin(air_out_folder,out_file_name)
            
            #create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            out_wav_array=np.zeros((CN.N_FFT,2))
            #grab BRIR
            out_wav_array[:,0] = np.copy(air_reverberation[set_num,0,:])#L
            out_wav_array[:,1] = np.copy(air_reverberation[set_num,1,:])#R
            
            if output_wavs == 1:
                hf.write2wav(file_name=out_file_path, data=out_wav_array, prevent_clipping=1, samplerate=samp_freq)
    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to complete TD averaging for: ' + ir_set 
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_error(log_string)
    
  
def irs_to_air_set(ir_set='default_set_name', num_out_sets=None, gui_logger=None):
    """
    function to load individual WAV IR files and insert into numpy array for later use
    saves air_reverberation array. num_out_sets * 2 channels * CN.N_FFT samples
    :param ir_set: str, name of impulse response set. Must correspond to a folder in ASH-Toolset\data\raw\ir_data\raw_airs
    :param num_out_sets: int, number of output sets
    :return: None
    """
    
    #set fft length based on AC space
    if ir_set in CN.AC_SPACE_LIST_LOWRT60:
        n_fft=CN.N_FFT
    else:
        n_fft=CN.N_FFT_L
    
    #param sofa_mode: int, 0 = divide sofa objects evenly among sets, 1 = each sofa object has a corresponding set
    if ir_set in CN.AC_SPACE_LIST_SOFA_1:
        sofa_mode=1
    elif ir_set in CN.AC_SPACE_LIST_SOFA_2:
        sofa_mode=2
    elif ir_set in CN.AC_SPACE_LIST_SOFA_3:
        sofa_mode=3
    elif ir_set in CN.AC_SPACE_LIST_SOFA_4:
        sofa_mode=4
    elif ir_set in CN.AC_SPACE_LIST_SOFA_5:
        sofa_mode=5
    else:
        sofa_mode=0
        
    #set noise reduction based on AC space
    if ir_set in CN.AC_SPACE_LIST_NR:
        noise_reduction=1
    else:
        noise_reduction=0
        
    samp_freq=44100
    output_wavs=1
    
    total_chan_air=1

    #windows
    data_pad_zeros=np.zeros(n_fft)
    data_pad_ones=np.ones(n_fft)

    #direct sound window for 2nd phase
    direct_hanning_size=300#350
    direct_hanning_start=51#101
    hann_direct_full=np.hanning(direct_hanning_size)
    hann_direct = np.split(hann_direct_full,2)[0]
    direct_removal_win_b = data_pad_zeros.copy()
    direct_removal_win_b[direct_hanning_start:direct_hanning_start+int(direct_hanning_size/2)] = hann_direct
    direct_removal_win_b[direct_hanning_start+int(direct_hanning_size/2):]=data_pad_ones[direct_hanning_start+int(direct_hanning_size/2):]
    
    #optional fade out window
    if ir_set in CN.AC_SPACE_LIST_WINDOW and n_fft == CN.N_FFT_L:
        fade_hanning_size=int(65536*2)
        fade_hanning_start=int(65536/2)
        hann_fade_full=np.hanning(fade_hanning_size)
        hann_fade = np.split(hann_fade_full,2)[1]
        fade_out_win = data_pad_ones.copy()
        fade_out_win[fade_hanning_start:fade_hanning_start+int(fade_hanning_size/2)] = hann_fade
        fade_out_win[fade_hanning_start+int(fade_hanning_size/2):]=data_pad_zeros[fade_hanning_start+int(fade_hanning_size/2):]
    
    #loop through folders
    ir_data_folder = pjoin(CN.DATA_DIR_RAW, 'ir_data', 'raw_airs',ir_set)

    try:
        log_string_a = 'Starting irs_to_air_set processing for: '+ir_set
        if CN.LOG_INFO == 1:
            logging.info(log_string_a)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string_a)

        #
        #get number of IRs to define size of dataset
        #
        total_irs=0#also count individual channels
        total_sofa_obj=0
        total_irs_per_sofa=0
        min_irs_per_sofa=10000
        hemi_split_mode=0
        for root, dirs, files in os.walk(ir_data_folder):
            for filename in files:
                if 'left_hem' in dirs or 'left_hem' in root:
                    hemi_split_mode=1
                    
                    
                if 'imp' in filename and ir_set in CN.AC_SPACE_LIST_RWCP:
                    wav_fname = pjoin(root, filename)
                    sig, fs = sf.read(wav_fname, channels=1, samplerate=44100, format='RAW', subtype='FLOAT')
                    samplerate=fs
                    fir_array = sig
                    try:
                        input_channels = len(fir_array[0])
                        if ir_set in CN.AC_SPACE_LIST_LIM_CHANS:#optional limiting of input channels
                            input_channels=min(input_channels,CN.AC_SPACE_CHAN_LIMITED)
                    except:
                        #reshape if mono
                        input_channels=1
                        fir_array=fir_array.reshape(-1, 1)
                    total_irs=total_irs+input_channels
                    
                    
                    
                if '.wav' in filename:
                    #read wav file
                    wav_fname = pjoin(root, filename)
                    samplerate, data = wavfile.read(wav_fname)
                    fir_array = data / (2.**31)
                    try:
                        input_channels = len(fir_array[0])
                        if ir_set in CN.AC_SPACE_LIST_LIM_CHANS:#optional limiting of input channels
                            input_channels=min(input_channels,CN.AC_SPACE_CHAN_LIMITED)
                    except:
                        #reshape if mono
                        input_channels=1
                        fir_array=fir_array.reshape(-1, 1)
                    total_irs=total_irs+input_channels
                    
                if '.npy' in filename:
                    #total_irs=total_irs+1#assume 1 channel per npy
                    
                    #read npy file
                    npy_fname = pjoin(root, filename)
                    fir_array = np.load(npy_fname)
                    shape = fir_array.shape
                    n_dims= fir_array.ndim
                    # Sampling rate
                    samplerate = 48000 #sample rate is assumed to be 48000Hz (MESHRIR)
                    
                    try:
                        input_channels = len(fir_array[0])
                    except:
                        #reshape if mono
                        input_channels=1
                        fir_array=fir_array.reshape(-1, 1)
                    #case where array needs to be transposed
                    if input_channels > 1000:
                        fir_array=np.transpose(fir_array)
                        #input_channels=1
                        input_channels = len(fir_array[0])
                        
                    total_irs=total_irs+input_channels
                    
                    log_string_a = 'npy data_ir shape: ' + str(shape) + ', Input Dimensions: ' + str(n_dims) + ', source samplerate: ' + str(samplerate) + ', input_channels: ' + str(input_channels)
                    if CN.LOG_INFO == 1:
                        logging.info(log_string_a)
                    if CN.LOG_GUI == 1 and gui_logger != None:
                        gui_logger.log_info(log_string_a)    
                        
    
                if '.mat' in filename:
                    mat_fname = pjoin(root, filename)
                    try:
                        data = mat73.loadmat(mat_fname)
                    except:
                        data = sio.loadmat(mat_fname)
                        
                    structs = list(data.keys())
                    #print('structs in this dataset:', structs)
                    
                    if ir_set in CN.AC_SPACE_LIST_ISO:
                        mat_struct=data['ImpulseResponse']  
                        samplerate = 48000
                        fir_array = mat_struct
                        shape = fir_array.shape
                        n_dims= fir_array.ndim
                    elif ir_set in CN.AC_SPACE_LIST_SUBRIRDB:
                        samplerate_m = 48000
                        mat_struct=data['db']  
                        samplerate = mat_struct['samplingRate'][0][0][0][0]
                        fir_array = mat_struct['RIR'][0][0]
                        shape = fir_array.shape
                        n_dims= fir_array.ndim
                        if n_dims == 3:
                            #case where dim 0 and 1 are > 1:
                            fir_array=hf.combine_dims(fir_array)
                    else:
                        mat_struct=data['recordings']  
                        samplerate = mat_struct['fs'][0][0][0][0]
                        fir_array = mat_struct['ir'][0][0]
                        shape = fir_array.shape
                        n_dims= fir_array.ndim
                    
                    try:
                        input_channels = len(fir_array[0])
                    except:
                        #reshape if mono
                        input_channels=1
                        fir_array=fir_array.reshape(-1, 1)
                    #case where array needs to be transposed
                    if input_channels > 1000:
                        fir_array=np.transpose(fir_array)
                        #input_channels=1
                        input_channels = len(fir_array[0])
                        
                    total_irs=total_irs+input_channels
                    
                    log_string_a = 'mat data_ir shape: ' + str(shape) + ', Input Dimensions: ' + str(n_dims) + ', source samplerate: ' + str(samplerate) + ', input_channels: ' + str(input_channels)
                    if CN.LOG_INFO == 1:
                        logging.info(log_string_a)
                    if CN.LOG_GUI == 1 and gui_logger != None:
                        gui_logger.log_info(log_string_a)
                    
    
                if '.sofa' in filename:
                    total_sofa_obj=total_sofa_obj+1
                    
                    #read SOFA file
                    sofa_fname = pjoin(root, filename)
                    loadsofa = SOFAFile.load(sofa_fname, verbose=False)
                    data = loadsofa.data_ir
                    samplerate = loadsofa.Data_SamplingRate
                    #calculate total number of IRs available in a sofa obj
                    shape = data.shape
                    n_dims= data.ndim
                    if n_dims == 4:
                        fir_array=data[0][0]
                    elif n_dims == 3:
                        #case where dim 2 is length 1 (reversed dimensions)
                        if len(data[0][0]) == 1:
                            data=np.transpose(data)
                            fir_array=data[0]
                        #case where dim 0 and 1 are > 1
                        elif len(data) > 1:
                            fir_array=hf.combine_dims(data)
                        else:
                            fir_array=data[0]
                    else:
                        raise ValueError('Invalid SOFA dimensions: '+ str(n_dims))
 
                    input_channels = len(fir_array)
                    total_irs_per_sofa=input_channels
                    if total_irs_per_sofa<min_irs_per_sofa:#find min number of irs in a sofa object
                        min_irs_per_sofa=total_irs_per_sofa
                        
                    #calculate total IRs across all sofa obj
                    total_irs=total_irs+input_channels
                    
                if '.hdf5' in filename:
                    hdf_fname = pjoin(root, filename)
                    rir_dset = h5py.File(hdf_fname, mode='r')
      
                    #print('RIR matrix shape', rir_dset.shape)
                    
                    #samplerate = rir_dset.attrs['sampling_rate']#attribute doesnt exist in reduced dataset
                    samplerate = 48000 #sample rate is assumed to be 48000Hz (dechorate)
                    num_rooms_total=4
                    
                    rooms = list(rir_dset.keys())
                    room=1
                    print('Rooms in this dataset:', rooms)
                    data=rir_dset[rooms[room]]
                    print('RIR matrix shape', data.shape)
                    data=np.transpose(data)
                    print('RIR matrix shape', data.shape)
                    fir_array=hf.combine_dims(data)
                    print('fir_array matrix shape', fir_array.shape)
                    fir_array=np.transpose(fir_array)
                    print('fir_array matrix shape', fir_array.shape)
                    
                    input_channels = len(fir_array[0])
                    #calculate total IRs
                    total_irs=total_irs+(input_channels*num_rooms_total)



        max_irs=CN.MAX_IRS
        total_irs = min(total_irs,max_irs)
        #set limits
        if ir_set in CN.AC_SPACE_LIST_SUB:
            num_out_sets=1
        elif ir_set in CN.AC_SPACE_LIST_LIM_SETS:
            num_out_sets=5
        elif num_out_sets == None or num_out_sets == 0:
            if total_irs >= 525:
                num_out_sets=7
            else:
                num_out_sets=5 
        #limit num out sets to not exceed number of IRs
        num_out_sets = min(total_irs,num_out_sets)
        #numpy array, num sets x num irs in each set x y channels x NFFT max samples
        irs_per_set=int((total_irs/num_out_sets)+1)
        sets_per_hemi = int(np.ceil(num_out_sets/2))
        
        #sofa calculations
        if total_sofa_obj > 0:
            #set limits
            irs_per_sofa_obj = int(total_irs/total_sofa_obj)
            #limit num irs per sofa to not exceed smallest sofa size
            irs_per_sofa_obj = min(irs_per_sofa_obj,min_irs_per_sofa)
     
            sets_per_sofa_obj=int(np.ceil(num_out_sets/total_sofa_obj)) 
            sofa_obj_per_hemi=int(np.ceil(total_sofa_obj/2))
 
            log_string_a = 'sets_per_hemi: ' + str(sets_per_hemi) + ', sofa_obj_per_hemi: ' + str(sofa_obj_per_hemi) + ', sets_per_sofa_obj: ' + str(sets_per_sofa_obj) + ', irs_per_set: ' + str(irs_per_set) + ', num_out_sets: ' + str(num_out_sets) + ', irs_per_sofa_obj: ' + str(irs_per_sofa_obj) + ', total_irs: ' + str(total_irs) + ', total_sofa_obj: ' + str(total_sofa_obj) 
            if CN.LOG_INFO == 1:
                logging.info(log_string_a)
            if CN.LOG_GUI == 1 and gui_logger != None:
                gui_logger.log_info(log_string_a)

        else: 
            log_string_a = 'sets_per_hemi: ' + str(sets_per_hemi) + ', irs_per_set: ' + str(irs_per_set) + ', num_out_sets: ' + str(num_out_sets)  + ', total_irs: ' + str(total_irs)
            if CN.LOG_INFO == 1:
                logging.info(log_string_a)
            if CN.LOG_GUI == 1 and gui_logger != None:
                gui_logger.log_info(log_string_a)
            
        
        log_string_a = 'num_out_sets: ' + str(num_out_sets) + ', total_irs: ' + str(total_irs)
        if CN.LOG_INFO == 1:
            logging.info(log_string_a)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string_a)
        
        air_data=np.zeros((num_out_sets,irs_per_set,total_chan_air,n_fft))
        air_sample=np.zeros((num_out_sets,total_chan_air,n_fft))
    
        #load wav IRs
        set_counter=0
        current_ir=0
        current_set=0
        current_sofa=0
        current_hemi=0
        
        set_counter_l=0
        current_ir_l=0
        current_set_l=0
        set_counter_r=0
        current_ir_r=0
        current_set_r=0
        for root, dirs, files in os.walk(ir_data_folder):
            for filename in files:
                
                #case for sofa files
                if '.sofa' in filename:
                    #read SOFA file
                    sofa_fname = pjoin(root, filename)
                    loadsofa = SOFAFile.load(sofa_fname, verbose=False)
                    data = loadsofa.data_ir
                    samplerate = int(loadsofa.Data_SamplingRate[0])
                    #calculate total number of IRs available in a sofa obj
                    shape = data.shape
                    n_dims= data.ndim
                    if n_dims == 4:
                        fir_array=data[0][0]
                    elif n_dims == 3:
                        #case where dim 2 is length 1 (reversed dimensions)
                        if len(data[0][0]) == 1:
                            data=np.transpose(data)
                            fir_array=data[0]
                        #case where dim 0 and 1 are > 1
                        elif len(data) > 1:
                            fir_array=hf.combine_dims(data)
                        else:
                            fir_array=data[0]
                    else:
                        raise ValueError('Invalid SOFA dimensions: '+ str(n_dims))
 
                    
                    fir_array=np.transpose(fir_array)
                    input_channels = len(fir_array[0])
                    fir_length = len(fir_array)
                    extract_legth = min(n_fft,fir_length)
                    
                    log_string_a = 'SOFA data_ir shape: ' + str(shape) + ', Input Dimensions: ' + str(n_dims) + ', sofa_mode: ' + str(sofa_mode) + ', source samplerate: ' + str(samplerate) + ', input_channels: ' + str(input_channels) + ', fir_length: ' + str(fir_length)
                    if CN.LOG_INFO == 1:
                        logging.info(log_string_a)
                    if CN.LOG_GUI == 1 and gui_logger != None:
                        gui_logger.log_info(log_string_a)
                    
                    #resample if sample rate is not 44100
                    if samplerate != samp_freq:
                        fir_array = hf.resample_signal(fir_array, new_rate = samp_freq)
                        log_string_a = 'source samplerate: ' + str(samplerate) + ', resampled to 44100Hz'
                        if CN.LOG_INFO == 1:
                            logging.info(log_string_a)
                        if CN.LOG_GUI == 1 and gui_logger != None:
                            gui_logger.log_info(log_string_a)
                
                    #apply noise reduction if enabled
                    if noise_reduction == 1:#if noise_reduction == 1 and 'idxX' in filename:
                        print("Noise reduction enabled")
                        if ir_set == 'audio_lab_g' or ir_set == 'audio_lab_h':
                            noise_samples = 60000
                        else:
                            noise_samples = 10000
                        #[shape=(# frames,) or (# channels, # frames)], real-valued
                        #One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
                        orig_shape = fir_array.shape
                        y = np.reshape(fir_array, (input_channels, -1))
                        y_noise = np.copy(y[:,int(noise_samples*-1):])#noise sample will be last x samples
                        y_clean = nr.reduce_noise(y=y, sr=samp_freq, y_noise=y_noise,stationary=True)
                        #You have to use the reshaping and then reshape back before saving:
                        fir_array=np.reshape(y_clean, orig_shape)
                    
                    if sofa_mode == 0:
                        #sofa objects are divided evenly among all sets.
                        for ir_idx in range(irs_per_sofa_obj):
     
                            #load into numpy array
                            air_data[current_set,current_ir,0,0:extract_legth]=np.copy(fir_array[0:extract_legth,ir_idx])#mono
                            
                            #increment set counter
                            set_counter=set_counter+1
                            #calculate new current set
                            current_set = set_counter%num_out_sets
                            #increment current ir every num_out_sets
                            if current_set == 0:
                                current_ir=current_ir+1
                    elif sofa_mode == 1:
                        #sofa objects are divided evenly among specific sets
                        for ir_idx in range(irs_per_sofa_obj):

                            if current_set < num_out_sets:
                                #load into numpy array
                                air_data[current_set,current_ir,0,0:extract_legth]=np.copy(fir_array[0:extract_legth,ir_idx])#mono
                            
                            #increment set counter
                            set_counter=set_counter+1
                            #calculate new current set
                            current_set = set_counter%sets_per_sofa_obj + current_sofa*sets_per_sofa_obj
                            #increment current ir every num_out_sets
                            if current_set == current_sofa*sets_per_sofa_obj:
                                current_ir=current_ir+1
    
                        #move onto next sofa object
                        current_sofa=current_sofa+1
                        #calculate new current set
                        set_counter=0
                        current_set = set_counter%sets_per_sofa_obj + current_sofa*sets_per_sofa_obj
                        #reset ir counter
                        current_ir=0
                    elif sofa_mode == 2:
                        #sofa objects are divided among left and right hemispheres
                        for ir_idx in range(irs_per_sofa_obj):

                            if current_set < num_out_sets:
                                #load into numpy array
                                air_data[current_set,current_ir,0,0:extract_legth]=np.copy(fir_array[0:extract_legth,ir_idx])#mono
                            
                            #increment set counter
                            set_counter=set_counter+1
                            #calculate new current set
                            current_set = set_counter%sets_per_hemi + current_hemi*sets_per_hemi
                            
                            #increment current ir every num sets_per_hemi
                            if current_set == current_hemi*sets_per_hemi:
                                current_ir=current_ir+1
    
                        #move onto next sofa object
                        current_sofa=current_sofa+1
                        if current_sofa >= sofa_obj_per_hemi:
                            current_hemi=current_hemi+1
                            #calculate new current set
                            set_counter=0
                            current_set = set_counter%sets_per_hemi + current_hemi*sets_per_hemi
                            #reset ir counter
                            current_ir=0
                    
                    elif sofa_mode == 3:
                        #sofa objects are divided among left and right hemispheres based on sub folders
                        #case for hemi split method
                        if 'left_hem' in root:
                            current_hemi=0
                            for chan_idx in range(input_channels):
    
                                #load into numpy array
                                if current_set_l < num_out_sets and current_ir_l < irs_per_set:
                                    air_data[current_set_l,current_ir_l,0,0:extract_legth]=np.copy(fir_array[0:extract_legth,chan_idx])#mono
                                
                                #increment set counter
                                set_counter_l=set_counter_l+1
                                #calculate new current set
                                current_set_l = set_counter_l%sets_per_hemi + current_hemi*sets_per_hemi
                                
                                #increment current ir every num sets_per_hemi
                                if current_set_l == current_hemi*sets_per_hemi:
                                    current_ir_l=current_ir_l+1
                        if 'right_hem' in root:
                            current_hemi=1
                            #calculate new current set
                            current_set_r = set_counter_r%sets_per_hemi + current_hemi*sets_per_hemi
                            for chan_idx in range(input_channels):
         
                                #load into numpy array
                                if current_set_r < num_out_sets and current_ir_r < irs_per_set:
                                    air_data[current_set_r,current_ir_r,0,0:extract_legth]=np.copy(fir_array[0:extract_legth,chan_idx])#mono
                                
                                #increment set counter
                                set_counter_r=set_counter_r+1
                                #calculate new current set
                                current_set_r = set_counter_r%sets_per_hemi + current_hemi*sets_per_hemi
                                
                                #increment current ir every num sets_per_hemi
                                if current_set_r == current_hemi*sets_per_hemi:
                                    current_ir_r=current_ir_r+1
                    
                    elif sofa_mode == 4:
                        #sofa IRs are divided among left and right hemispheres. 1st half of sofa obj goes to left hem, 2nd half goes to right. shared evenly across sofa objs
                        for chan_idx in range(irs_per_sofa_obj):#input_channels

                            if chan_idx < int(irs_per_sofa_obj/2):
                                current_hemi=0
                                
                                #load into numpy array
                                if current_set_l < num_out_sets and current_ir_l < irs_per_set:
                                    air_data[current_set_l,current_ir_l,0,0:extract_legth]=np.copy(fir_array[0:extract_legth,chan_idx])#mono
                                
                                #increment set counter
                                set_counter_l=set_counter_l+1
                                #calculate new current set
                                current_set_l = set_counter_l%sets_per_hemi + current_hemi*sets_per_hemi
                                
                                #increment current ir every num sets_per_hemi
                                if current_set_l == current_hemi*sets_per_hemi:
                                    current_ir_l=current_ir_l+1
                                
                            else:
                                current_hemi=1
                                
                                #calculate new current set
                                current_set_r = set_counter_r%sets_per_hemi + current_hemi*sets_per_hemi
         
                                #load into numpy array
                                if current_set_r < num_out_sets and current_ir_r < irs_per_set:
                                    air_data[current_set_r,current_ir_r,0,0:extract_legth]=np.copy(fir_array[0:extract_legth,chan_idx])#mono
                                
                                #increment set counter
                                set_counter_r=set_counter_r+1
                                #calculate new current set
                                current_set_r = set_counter_r%sets_per_hemi + current_hemi*sets_per_hemi
                                
                                #increment current ir every num sets_per_hemi
                                if current_set_r == current_hemi*sets_per_hemi:
                                    current_ir_r=current_ir_r+1
                            
                    elif sofa_mode == 5:
                        #sofa IRs are divided among left and right hemispheres. 1st half of sofa obj goes to left hem, 2nd half goes to right
                        for chan_idx in range(input_channels):#input_channels

                            if chan_idx < int(input_channels/2):
                                current_hemi=0
                                
                                #load into numpy array
                                if current_set_l < num_out_sets and current_ir_l < irs_per_set:
                                    air_data[current_set_l,current_ir_l,0,0:extract_legth]=np.copy(fir_array[0:extract_legth,chan_idx])#mono
                                
                                #increment set counter
                                set_counter_l=set_counter_l+1
                                #calculate new current set
                                current_set_l = set_counter_l%sets_per_hemi + current_hemi*sets_per_hemi
                                
                                #increment current ir every num sets_per_hemi
                                if current_set_l == current_hemi*sets_per_hemi:
                                    current_ir_l=current_ir_l+1
                                
                            else:
                                current_hemi=1
                                
                                #calculate new current set
                                current_set_r = set_counter_r%sets_per_hemi + current_hemi*sets_per_hemi
         
                                #load into numpy array
                                if current_set_r < num_out_sets and current_ir_r < irs_per_set:
                                    air_data[current_set_r,current_ir_r,0,0:extract_legth]=np.copy(fir_array[0:extract_legth,chan_idx])#mono
                                
                                #increment set counter
                                set_counter_r=set_counter_r+1
                                #calculate new current set
                                current_set_r = set_counter_r%sets_per_hemi + current_hemi*sets_per_hemi
                                
                                #increment current ir every num sets_per_hemi
                                if current_set_r == current_hemi*sets_per_hemi:
                                    current_ir_r=current_ir_r+1
                    
                    else:
                        raise ValueError('Invalid SOFA mode')
                        
                    
                #case for matlab files
                if '.mat' in filename:
                    mat_fname = pjoin(root, filename)
                    try:
                        data = mat73.loadmat(mat_fname)
                    except:
                        data = sio.loadmat(mat_fname)
                    
                    if ir_set in CN.AC_SPACE_LIST_ISO:
                        mat_struct=data['ImpulseResponse']  
                        samplerate = 48000
                        fir_array = mat_struct
                        shape = fir_array.shape
                        n_dims= fir_array.ndim
                    elif ir_set in CN.AC_SPACE_LIST_SUBRIRDB:
                        samplerate_m = 48000
                        mat_struct=data['db']  
                        samplerate = mat_struct['samplingRate'][0][0][0][0]
                        fir_array = mat_struct['RIR'][0][0]
                        shape = fir_array.shape
                        n_dims= fir_array.ndim
                        if n_dims == 3:
                            #case where dim 0 and 1 are > 1:
                            fir_array=hf.combine_dims(fir_array)
                    else:
                        mat_struct=data['recordings']  
                        samplerate = mat_struct['fs'][0][0][0][0]
                        fir_array = mat_struct['ir'][0][0]
                        shape = fir_array.shape
                        n_dims= fir_array.ndim
                    
                    
                    try:
                        input_channels = len(fir_array[0])
                    except:
                        #reshape if mono
                        input_channels=1
                        fir_array=fir_array.reshape(-1, 1)
                    #case where array needs to be transposed
                    if input_channels > 1000:
                        fir_array=np.transpose(fir_array)
                        input_channels = len(fir_array[0])

                    log_string_a = 'mat data_ir shape: ' + str(shape) + ', Input Dimensions: ' + str(n_dims) + ', source samplerate: ' + str(samplerate) + ', input_channels: ' + str(input_channels)
                    if CN.LOG_INFO == 1:
                        logging.info(log_string_a)
                    if CN.LOG_GUI == 1 and gui_logger != None:
                        gui_logger.log_info(log_string_a)

                    #resample if sample rate is not 44100
                    if samplerate != samp_freq:
                        fir_array = hf.resample_signal(fir_array, new_rate = samp_freq)
                        log_string_a = 'source samplerate: ' + str(samplerate) + ', resampled to 44100Hz'
                        if CN.LOG_INFO == 1:
                            logging.info(log_string_a)
                        if CN.LOG_GUI == 1 and gui_logger != None:
                            gui_logger.log_info(log_string_a)
                            
                    fir_length = len(fir_array)
                    extract_legth = min(n_fft,fir_length)
                    
                    #apply noise reduction if enabled
                    if noise_reduction == 1:#if noise_reduction == 1 and 'idxX' in filename:
                        print("Noise reduction enabled")
                        noise_samples = 20000
                        #[shape=(# frames,) or (# channels, # frames)], real-valued
                        #One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
                        orig_shape = fir_array.shape
                        y = np.reshape(fir_array, (input_channels, -1))
                        y_noise = np.copy(y[:,int(noise_samples*-1):])#noise sample will be last x samples
                        y_clean = nr.reduce_noise(y=y, sr=samp_freq, y_noise=y_noise,stationary=True)
                        #You have to use the reshaping and then reshape back before saving:
                        fir_array=np.reshape(y_clean, orig_shape)
                    
                    #each channel goes into a different set
                    if hemi_split_mode == 0:
                        for chan_idx in range(input_channels):
   
                            #load into numpy array
                            air_data[current_set,current_ir,0,0:extract_legth]=np.copy(fir_array[0:extract_legth,chan_idx])#mono
                            
                            #increment set counter
                            set_counter=set_counter+1
                            #calculate new current set
                            current_set = set_counter%num_out_sets
                            #increment current ir every num_out_sets
                            if current_set == 0:
                                current_ir=current_ir+1
                    else:
                        #case for hemi split method
                        if 'left_hem' in root:
                            current_hemi=0
                            for chan_idx in range(input_channels):
    
                                #load into numpy array
                                if current_set_l < num_out_sets and current_ir_l < irs_per_set:
                                    air_data[current_set_l,current_ir_l,0,0:extract_legth]=np.copy(fir_array[0:extract_legth,chan_idx])#mono
                                
                                #increment set counter
                                set_counter_l=set_counter_l+1
                                #calculate new current set
                                current_set_l = set_counter_l%sets_per_hemi + current_hemi*sets_per_hemi
                                
                                #increment current ir every num sets_per_hemi
                                if current_set_l == current_hemi*sets_per_hemi:
                                    current_ir_l=current_ir_l+1
                        if 'right_hem' in root:
                            current_hemi=1
                            #calculate new current set
                            current_set_r = set_counter_r%sets_per_hemi + current_hemi*sets_per_hemi
                            for chan_idx in range(input_channels):
         
                                #load into numpy array
                                if current_set_r < num_out_sets and current_ir_r < irs_per_set:
                                    air_data[current_set_r,current_ir_r,0,0:extract_legth]=np.copy(fir_array[0:extract_legth,chan_idx])#mono
                                
                                #increment set counter
                                set_counter_r=set_counter_r+1
                                #calculate new current set
                                current_set_r = set_counter_r%sets_per_hemi + current_hemi*sets_per_hemi
                                
                                #increment current ir every num sets_per_hemi
                                if current_set_r == current_hemi*sets_per_hemi:
                                    current_ir_r=current_ir_r+1
                
                
                #case for numpy arrays
                if '.npy' in filename:
                    #read npy file
                    npy_fname = pjoin(root, filename)
                    fir_array = np.load(npy_fname)
                    
                    # Sampling rate
                    samplerate = 48000 #sample rate is assumed to be 48000
  
                    try:
                        input_channels = len(fir_array[0])
                    except:
                        #reshape if mono
                        input_channels=1
                        fir_array=fir_array.reshape(-1, 1)
                    #case where array needs to be transposed
                    if input_channels > 1000:
                        fir_array=np.transpose(fir_array)
                        #input_channels=1
                        input_channels = len(fir_array[0])
                        
                    #resample if sample rate is not 44100
                    if samplerate != samp_freq:
                        fir_array = hf.resample_signal(fir_array, new_rate = samp_freq)
                        log_string_a = 'source samplerate: ' + str(samplerate) + ', resampled to 44100Hz'
                        if CN.LOG_INFO == 1:
                            logging.info(log_string_a)
                        if CN.LOG_GUI == 1 and gui_logger != None:
                            gui_logger.log_info(log_string_a)
                    
                    fir_length = len(fir_array)
                    extract_legth = min(n_fft,fir_length)
                    
                    #apply noise reduction if enabled
                    if noise_reduction == 1:
                        noise_samples = 15000
                        #[shape=(# frames,) or (# channels, # frames)], real-valued
                        #One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
                        orig_shape = fir_array.shape
                        y = np.reshape(fir_array, (input_channels, -1))
                        y_noise = np.copy(y[:,int(noise_samples*-1):])#noise sample will be last x samples
                        y_clean = nr.reduce_noise(y=y, sr=samp_freq, y_noise=y_noise,stationary=True)
                        #You have to use the reshaping and then reshape back before saving:
                        fir_array=np.reshape(y_clean, orig_shape)
                    
                    if hemi_split_mode == 0:
                        #each channel goes into a different set
                        for chan_idx in range(input_channels):

                            #load into numpy array
                            air_data[current_set,current_ir,0,0:extract_legth]=np.copy(fir_array[0:extract_legth,chan_idx])#mono
                            
                            #increment set counter
                            set_counter=set_counter+1
                            #calculate new current set
                            current_set = set_counter%num_out_sets
                            #increment current ir every num_out_sets
                            if current_set == 0:
                                current_ir=current_ir+1
                    else:
                        #case for hemi split method
                        if 'left_hem' in root:
                            current_hemi=0
                            for chan_idx in range(input_channels):
    
                                #load into numpy array
                                if current_set_l < num_out_sets and current_ir_l < irs_per_set:
                                    air_data[current_set_l,current_ir_l,0,0:extract_legth]=np.copy(fir_array[0:extract_legth,chan_idx])#mono
                                
                                #increment set counter
                                set_counter_l=set_counter_l+1
                                #calculate new current set
                                current_set_l = set_counter_l%sets_per_hemi + current_hemi*sets_per_hemi
                                
                                #increment current ir every num sets_per_hemi
                                if current_set_l == current_hemi*sets_per_hemi:
                                    current_ir_l=current_ir_l+1
                        if 'right_hem' in root:
                            current_hemi=1
                            #calculate new current set
                            current_set_r = set_counter_r%sets_per_hemi + current_hemi*sets_per_hemi
                            for chan_idx in range(input_channels):
         
                                #load into numpy array
                                if current_set_r < num_out_sets and current_ir_r < irs_per_set:
                                    air_data[current_set_r,current_ir_r,0,0:extract_legth]=np.copy(fir_array[0:extract_legth,chan_idx])#mono
                                
                                #increment set counter
                                set_counter_r=set_counter_r+1
                                #calculate new current set
                                current_set_r = set_counter_r%sets_per_hemi + current_hemi*sets_per_hemi
                                
                                #increment current ir every num sets_per_hemi
                                if current_set_r == current_hemi*sets_per_hemi:
                                    current_ir_r=current_ir_r+1
                        
                #case for hdf5
                if '.hdf5' in filename:
                    hdf_fname = pjoin(root, filename)
                    rir_dset = h5py.File(hdf_fname, mode='r')
      
                    #print('RIR matrix shape', rir_dset.shape)
                    
                    #samplerate = rir_dset.attrs['sampling_rate']#attribute doesnt exist in reduced dataset
                    samplerate = 48000 #sample rate is assumed to be 48000Hz (dechorate)
                    num_rooms_total=4
                    
                    rooms = list(rir_dset.keys())
                    print('Rooms in this dataset:', rooms)
                    
                    for room in range(num_rooms_total):
         
                        data=rir_dset[rooms[room+1]]
                        print('RIR matrix shape', data.shape)
                        data=np.transpose(data)
                        print('RIR matrix shape', data.shape)
                        fir_array=hf.combine_dims(data)
                        print('fir_array matrix shape', fir_array.shape)
                        fir_array=np.transpose(fir_array)
                        print('fir_array matrix shape', fir_array.shape)
        
                        #resample if sample rate is not 44100
                        if samplerate != samp_freq:
                            fir_array = hf.resample_signal(fir_array, new_rate = samp_freq)
                            log_string_a = 'source samplerate: ' + str(samplerate) + ', resampled to 44100Hz'
                            if CN.LOG_INFO == 1:
                                logging.info(log_string_a)
                            if CN.LOG_GUI == 1 and gui_logger != None:
                                gui_logger.log_info(log_string_a)
                          
                        fir_length = len(fir_array)
                        extract_legth = min(n_fft,fir_length)
                        #extract_legth=18000#crop
                        print('extract_legth ' + str(extract_legth))
                        
                        input_channels = len(fir_array[0])        
                                
                        if room < int(num_rooms_total/2):
                            current_hemi=0
                        else:
                            current_hemi=1
                            
                        if current_hemi == 0:
                            for chan_idx in range(input_channels):
             
                                #load into numpy array
                                if current_set_l < num_out_sets and current_ir_l < irs_per_set:
                                    air_data[current_set_l,current_ir_l,0,0:extract_legth]=np.copy(fir_array[0:extract_legth,chan_idx])#mono
                                
                                #increment set counter
                                set_counter_l=set_counter_l+1
                                #calculate new current set
                                current_set_l = set_counter_l%sets_per_hemi + current_hemi*sets_per_hemi
                                
                                #increment current ir every num sets_per_hemi
                                if current_set_l == current_hemi*sets_per_hemi:
                                    current_ir_l=current_ir_l+1
                        else:
                            #calculate new current set
                            current_set_r = set_counter_r%sets_per_hemi + current_hemi*sets_per_hemi
                            for chan_idx in range(input_channels):
            
                                #load into numpy array
                                if current_set_r < num_out_sets and current_ir_r < irs_per_set:
                                    air_data[current_set_r,current_ir_r,0,0:extract_legth]=np.copy(fir_array[0:extract_legth,chan_idx])#mono
                                
                                #increment set counter
                                set_counter_r=set_counter_r+1
                                #calculate new current set
                                current_set_r = set_counter_r%sets_per_hemi + current_hemi*sets_per_hemi
                                
                                #increment current ir every num sets_per_hemi
                                if current_set_r == current_hemi*sets_per_hemi:
                                    current_ir_r=current_ir_r+1
                    
                  
     
                #case for RWCP WAVs
                if 'imp' in filename and ir_set in CN.AC_SPACE_LIST_RWCP:
                    wav_fname = pjoin(root, filename)
                    sig, fs = sf.read(wav_fname, channels=1, samplerate=44100, format='RAW', subtype='FLOAT')
                    samplerate=fs
                    fir_array = sig
                    fir_length = len(fir_array)
                    
                    extract_legth = min(n_fft,fir_length)
                    ir_cutoff=0
                    
                    #resample if sample rate is not 44100
                    if samplerate != samp_freq:
                        fir_array = hf.resample_signal(fir_array, new_rate = samp_freq)
                        log_string_a = 'source samplerate: ' + str(samplerate) + ', resampled to 44100Hz'
                        if CN.LOG_INFO == 1:
                            logging.info(log_string_a)
                        if CN.LOG_GUI == 1 and gui_logger != None:
                            gui_logger.log_info(log_string_a)
                        
                    try:
                        input_channels = len(fir_array[0])
                    except:
                        #reshape if mono
                        input_channels=1
                        fir_array=fir_array.reshape(-1, 1)
                    
                    #apply noise reduction if enabled
                    if noise_reduction == 1:
                        noise_samples = 10000
                        #[shape=(# frames,) or (# channels, # frames)], real-valued
                        #One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
                        orig_shape = fir_array.shape
                        y = np.reshape(fir_array, (input_channels, -1))
                        y_noise = np.copy(y[:,int(noise_samples*-1):])#noise sample will be last x samples
                        y_clean = nr.reduce_noise(y=y, sr=samp_freq, y_noise=y_noise,stationary=True)
                        #You have to use the reshaping and then reshape back before saving:
                        fir_array=np.reshape(y_clean, orig_shape)

                    if hemi_split_mode == 0:
                        #each channel goes into a different set
                        for chan_idx in range(input_channels):
                            
                            #load into numpy array
                            if current_ir < irs_per_set:
                                air_data[current_set,current_ir,0,0:extract_legth]=np.copy(fir_array[0+ir_cutoff:extract_legth+ir_cutoff,chan_idx])#mono
                            
                            #increment set counter
                            set_counter=set_counter+1
                            #calculate new current set
                            current_set = set_counter%num_out_sets
                            #increment current ir every num_out_sets
                            if current_set == 0:
                                current_ir=current_ir+1
                    else:
                        #case for hemi split method
                        if 'left_hem' in root:
                            current_hemi=0
                            for chan_idx in range(input_channels):
             
                                #load into numpy array
                                if current_set_l < num_out_sets and current_ir_l < irs_per_set:
                                    air_data[current_set_l,current_ir_l,0,0:extract_legth]=np.copy(fir_array[0+ir_cutoff:extract_legth+ir_cutoff,chan_idx])#mono
                                
                                #increment set counter
                                set_counter_l=set_counter_l+1
                                #calculate new current set
                                current_set_l = set_counter_l%sets_per_hemi + current_hemi*sets_per_hemi
                                
                                #increment current ir every num sets_per_hemi
                                if current_set_l == current_hemi*sets_per_hemi:
                                    current_ir_l=current_ir_l+1
                        if 'right_hem' in root:
                            current_hemi=1
                            #calculate new current set
                            current_set_r = set_counter_r%sets_per_hemi + current_hemi*sets_per_hemi
                            for chan_idx in range(input_channels):
            
                                #load into numpy array
                                if current_set_r < num_out_sets and current_ir_r < irs_per_set:
                                    air_data[current_set_r,current_ir_r,0,0:extract_legth]=np.copy(fir_array[0+ir_cutoff:extract_legth+ir_cutoff,chan_idx])#mono
                                
                                #increment set counter
                                set_counter_r=set_counter_r+1
                                #calculate new current set
                                current_set_r = set_counter_r%sets_per_hemi + current_hemi*sets_per_hemi
                                
                                #increment current ir every num sets_per_hemi
                                if current_set_r == current_hemi*sets_per_hemi:
                                    current_ir_r=current_ir_r+1
     
        
                #case for WAV IRs
                if '.wav' in filename:
                    #read wav file
                    wav_fname = pjoin(root, filename)
                    samplerate, data = wavfile.read(wav_fname)
                    fir_array = data / (2.**31)
                    fir_length = len(fir_array)
                    extract_legth = min(n_fft,fir_length)
                    #set a cutoff for IR (ignore first x samples) if applicable
                    if ir_set in CN.AC_SPACE_LIST_CUTOFF:
                        ir_cutoff=80000
                    else:
                        ir_cutoff=0
                    
                    #resample if sample rate is not 44100
                    if samplerate != samp_freq:
                        fir_array = hf.resample_signal(fir_array, new_rate = samp_freq)
                        log_string_a = 'source samplerate: ' + str(samplerate) + ', resampled to 44100Hz'
                        if CN.LOG_INFO == 1:
                            logging.info(log_string_a)
                        if CN.LOG_GUI == 1 and gui_logger != None:
                            gui_logger.log_info(log_string_a)
                        
                    try:
                        input_channels = len(fir_array[0])
                    except:
                        #reshape if mono
                        input_channels=1
                        fir_array=fir_array.reshape(-1, 1)
                    
                    #apply noise reduction if enabled
                    if noise_reduction == 1 and extract_legth > 60000:
                        print('noise reduction enabled')
                        if ir_set == 'audio_lab_e':
                            noise_samples = 100000#100000
                        else:
                            noise_samples = 44100
                        
                        #[shape=(# frames,) or (# channels, # frames)], real-valued
                        #One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
                        orig_shape = fir_array.shape
                        y = np.reshape(fir_array, (input_channels, -1))
                        y_noise = np.copy(y[:,int(noise_samples*-1):])#noise sample will be last x samples
                        y_clean = nr.reduce_noise(y=y, sr=samp_freq, y_noise=y_noise,stationary=True)
                        #You have to use the reshaping and then reshape back before saving:
                        fir_array=np.reshape(y_clean, orig_shape)

                    if ir_set in CN.AC_SPACE_LIST_LIM_CHANS:#optional limiting of input channels
                        input_channels=min(input_channels,CN.AC_SPACE_CHAN_LIMITED)
                        
                    if hemi_split_mode == 0:
                        #each channel goes into a different set
                        for chan_idx in range(input_channels):
                            
                            #load into numpy array
                            if current_ir < irs_per_set:
                                air_data[current_set,current_ir,0,0:extract_legth]=np.copy(fir_array[0+ir_cutoff:extract_legth+ir_cutoff,chan_idx])#mono
                            
                            #increment set counter
                            set_counter=set_counter+1
                            #calculate new current set
                            current_set = set_counter%num_out_sets
                            #increment current ir every num_out_sets
                            if current_set == 0:
                                current_ir=current_ir+1
                    else:
                        #case for hemi split method
                        if 'left_hem' in root:
                            current_hemi=0
                            for chan_idx in range(input_channels):
             
                                #load into numpy array
                                if current_set_l < num_out_sets and current_ir_l < irs_per_set:
                                    air_data[current_set_l,current_ir_l,0,0:extract_legth]=np.copy(fir_array[0+ir_cutoff:extract_legth+ir_cutoff,chan_idx])#mono
                                
                                #increment set counter
                                set_counter_l=set_counter_l+1
                                #calculate new current set
                                current_set_l = set_counter_l%sets_per_hemi + current_hemi*sets_per_hemi
                                
                                #increment current ir every num sets_per_hemi
                                if current_set_l == current_hemi*sets_per_hemi:
                                    current_ir_l=current_ir_l+1
                        if 'right_hem' in root:
                            current_hemi=1
                            #calculate new current set
                            current_set_r = set_counter_r%sets_per_hemi + current_hemi*sets_per_hemi
                            for chan_idx in range(input_channels):
            
                                #load into numpy array
                                if current_set_r < num_out_sets and current_ir_r < irs_per_set:
                                    air_data[current_set_r,current_ir_r,0,0:extract_legth]=np.copy(fir_array[0+ir_cutoff:extract_legth+ir_cutoff,chan_idx])#mono
                                
                                #increment set counter
                                set_counter_r=set_counter_r+1
                                #calculate new current set
                                current_set_r = set_counter_r%sets_per_hemi + current_hemi*sets_per_hemi
                                
                                #increment current ir every num sets_per_hemi
                                if current_set_r == current_hemi*sets_per_hemi:
                                    current_ir_r=current_ir_r+1
                                
                
    
                    
        #crop raw IRs so that direct peak is at sample x
        index_peak_ref = 30#40
        for set_num in range(num_out_sets):
            for ir in range(irs_per_set):
                index_peak_cur = np.argmax(np.abs(air_data[set_num,ir,0,:]))
                ir_shift = index_peak_ref-index_peak_cur
                #print(str(ir_shift))
                
                air_data[set_num,ir,0,:] = np.roll(air_data[set_num,ir,0,:],ir_shift)

                #set end of array to zero to remove any data shifted to end of array
                if ir_shift < 0:
                    air_data[set_num,ir,0,ir_shift:] = air_data[set_num,ir,0,ir_shift:]*0#left
 
        #set each AIR to 0 level
        fb_start=int(CN.SPECT_SNAP_F0*CN.N_FFT/samp_freq)
        fb_end=int(CN.SPECT_SNAP_F1*CN.N_FFT/samp_freq)
        for set_num in range(num_out_sets):
            for ir in range(irs_per_set):
                data_fft = np.fft.fft(air_data[set_num,ir,0,0:CN.N_FFT])
                mag_fft=np.abs(data_fft)
                average_mag = np.mean(mag_fft[fb_start:fb_end])
                if average_mag > 0:
                    for chan in range(total_chan_air):
                        air_data[set_num,ir,chan,:] = np.divide(air_data[set_num,ir,chan,:],average_mag)
        
        #optional fade out window
        if ir_set in CN.AC_SPACE_LIST_WINDOW and n_fft == CN.N_FFT_L:
            for set_num in range(num_out_sets):
                for ir in range(irs_per_set):
                    for chan in range(total_chan_air):
                        air_data[set_num,ir,chan,:] = np.multiply(air_data[set_num,ir,chan,:],fade_out_win)
            
        
        #if total input IRs is more than twice num out sets, synchronise in time domain
        if irs_per_set >= 2:
            
            #contants for TD alignment of BRIRs
            t_shift_interval = CN.T_SHIFT_INTERVAL
            min_t_shift = CN.MIN_T_SHIFT_A
            max_t_shift = CN.MAX_T_SHIFT_A
            num_intervals = int(np.abs((max_t_shift-min_t_shift)/t_shift_interval))
            order=7#default 6
            delay_win_min_t = CN.DELAY_WIN_MIN_A
            delay_win_max_t = CN.DELAY_WIN_MAX_A
            delay_win_hop_size = CN.DELAY_WIN_HOP_SIZE
            delay_win_hops = CN.DELAY_WIN_HOPS_A
            
            if ir_set in CN.AC_SPACE_LIST_SUB:
                cutoff_alignment = 80#100,90
            else:
                cutoff_alignment = CN.CUTOFF_ALIGNMENT_AIR
            #peak to peak within a sufficiently small sample window
            peak_to_peak_window = int(np.divide(samp_freq,cutoff_alignment)*0.95) #int(np.divide(samp_freq,cutoff_alignment)) 
            
            delay_eval_set = np.zeros((num_out_sets,irs_per_set,num_intervals))
            
            for set_num in range(num_out_sets):
                #go through each room in the ordered list
                for ir in range(irs_per_set):#room in range(total_airs-1)
                    this_air_idx=ir
  
                    #method 2: take sum of all prior sets and this set
                    rooms_to_add = 0
                    prior_airs = data_pad_zeros.copy()
                    #all sets up to and including this set
                    for cum_set in range(set_num+1):
                        if cum_set < set_num:#previous sets
                            for cum_air in range(irs_per_set):#all irs in previous sets
                                cum_air_orig_idx = cum_air
                                rooms_to_add = rooms_to_add+1
                                prior_airs = np.add(prior_airs,air_data[cum_set,cum_air_orig_idx,0,:])
                        elif cum_set == set_num:#current set up to but not including current IR
                            for cum_air in range(this_air_idx):
                                cum_air_orig_idx = cum_air
                                rooms_to_add = rooms_to_add+1
                                prior_airs = np.add(prior_airs,air_data[cum_set,cum_air_orig_idx,0,:])
                    
                    if rooms_to_add > 0:
                        prior_airs = np.divide(prior_airs,rooms_to_add) 
    
                    if set_num == 0 and ir == 0:
                        calc_delay = 0
                    else:
                        calc_delay = 1
                        
                    if calc_delay == 1:
                        this_air = np.copy(air_data[set_num,this_air_idx,0,:])
                        if np.sum(np.abs(this_air)) > 0:
                            for delay in range(num_intervals):
                                
                                #shift next room BRIR
                                current_shift = min_t_shift+(delay*t_shift_interval)
                                n_air_shift = np.roll(this_air,current_shift)
                                #add current room BRIR to shifted next room BRIR
                                sum_ir = np.add(prior_airs,n_air_shift)
                                #calculate group delay
             
                                sum_ir_lp = hf.signal_lowpass_filter(sum_ir, cutoff_alignment, samp_freq, order)
                                peak_to_peak_iter=0
                                for hop_id in range(delay_win_hops):
                                    samples = hop_id*delay_win_hop_size
                                    peak_to_peak = np.abs(np.max(sum_ir_lp[delay_win_min_t+samples:delay_win_min_t+samples+peak_to_peak_window])-np.min(sum_ir_lp[delay_win_min_t+samples:delay_win_min_t+samples+peak_to_peak_window]))
                                    #if this window has larger pk to pk, store in iter var
                                    if peak_to_peak > peak_to_peak_iter:
                                        peak_to_peak_iter = peak_to_peak
                                #store largest pk to pk distance of all windows into delay set
                                delay_eval_set[set_num,this_air_idx,delay] = peak_to_peak_iter
                            
                            #shift next room by delay that has largest peak to peak distance (method 4 and 5)
                            index_shift = np.argmax(delay_eval_set[set_num,this_air_idx,:])
                            samples_shift=min_t_shift+(index_shift*t_shift_interval)
                        else:
                            samples_shift=0
                    else:
                        samples_shift=0
  
                    air_data[set_num,this_air_idx,0,:] = np.roll(air_data[set_num,this_air_idx,0,:],samples_shift)#left
                    
                    #set end of array to zero to remove any data shifted to end of array
                    if samples_shift < 0:
                        air_data[set_num,this_air_idx,0,min_t_shift:] = air_data[set_num,this_air_idx,0,min_t_shift:]*0#left
        
        if CN.RISE_WINDOW == True:   
            #remove direction portion of signal?
            for set_num in range(num_out_sets):
                for ir in range(irs_per_set):
                   for chan in range(total_chan_air):
                       air_data[set_num,ir,chan,:] = np.multiply(air_data[set_num,ir,chan,:],direct_removal_win_b)
        
        #populate sample array: take first air for each set only
        for set_num in range(num_out_sets):
            for chan in range(total_chan_air):
                air_sample[set_num,chan,:] = np.add(air_sample[set_num,chan,:],air_data[set_num,0,chan,:])
         
        if CN.RISE_WINDOW == True:  
            air_out_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'prepped_airs',ir_set)
        else:
            air_out_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'full_airs',ir_set)

        #save full size array (not combined)
        npy_file_name = ir_set+'_full.npy'
        out_file_path = pjoin(air_out_folder,npy_file_name)      
        #create dir if doesnt exist 
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        np.save(out_file_path,air_data)
        
        log_string_a = 'Exported numpy file to: ' + out_file_path 
        if CN.LOG_INFO == 1:
            logging.info(log_string_a)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string_a)
        
        #also save to wav for testing
        for set_num in range(num_out_sets):
            out_file_name = ir_set+'_'+str(set_num)+'_sample.wav'
            out_file_path = pjoin(air_out_folder,out_file_name)
            
            #create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            out_wav_array=np.zeros((n_fft,1))
            #grab BRIR
            out_wav_array[:,0] = np.copy(air_sample[set_num,0,:])#L
            
            if output_wavs == 1:
                hf.write2wav(file_name=out_file_path, data=out_wav_array, prevent_clipping=1, samplerate=samp_freq)
    
    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to complete AIR set processing for: ' + ir_set 
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_error(log_string)
  
    
    
def airs_to_brirs(ir_set='fw', ir_group='prepped_airs', out_directions=None, align_hrirs=True, source_directions=1,rand_sources=True, gui_logger=None):
    """
    function to convert RIRs to BRIRs
    :param ir_set: str, name of impulse response set. Must correspond to a folder in ASH-Toolset\data\interim\ir_data\<ir_group>
    :param ir_group: str, name of IR group. Either avg_airs or prepped_airs
    :param out_directions: int, desired number of output directions. Min = 1, max is limited by number of sets and channels in loaded ir set.
    :param align_hrirs: true = align hrirs in time domain by peaks, false = no alignment
    :param source_directions: int, number of source directions per brir
    :param rand_sources: bool, true = apply random source directions, false = circular motion
    :return: None
    """
    
    if ir_set in CN.AC_SPACE_LIST_LOWRT60:
        n_fft=CN.N_FFT
    else:
        n_fft=CN.N_FFT_L
    
    if ir_set in CN.AC_SPACE_LIST_NOCOMP:
        mag_comp=False
    else:
        mag_comp=CN.MAG_COMP
    
    output_wavs=1
    samp_freq=44100
    
    #impulse
    impulse=np.zeros(CN.N_FFT)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat = hf.mag2db(fr_flat_mag)
    
    #open input RIR
    air_in_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', ir_group,ir_set)
    npy_file_name = ir_set+'_full.npy'
    
    air_file_path = pjoin(air_in_folder,npy_file_name)  
    
    try:
        log_string_a = 'Starting airs_to_brirs processing for: '+ir_set
        if CN.LOG_INFO == 1:
            logging.info(log_string_a)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string_a)
            
        air_reverberation = np.load(air_file_path)
        
        #get number of sets
        total_sets_air = len(air_reverberation)
        total_irs_air =  len(air_reverberation[0])
        total_chan_air = len(air_reverberation[0][0])
        
        #direct sound window for 2nd phase
        #windows
        data_pad_zeros=np.zeros(n_fft)
        data_pad_ones=np.ones(n_fft)
        direct_hanning_size=200#300,250
        direct_hanning_start=51#101
        hann_direct_full=np.hanning(direct_hanning_size)
        hann_direct = np.split(hann_direct_full,2)[0]
        direct_removal_win_b = data_pad_zeros.copy()
        direct_removal_win_b[direct_hanning_start:direct_hanning_start+int(direct_hanning_size/2)] = hann_direct
        direct_removal_win_b[direct_hanning_start+int(direct_hanning_size/2):]=data_pad_ones[direct_hanning_start+int(direct_hanning_size/2):]
        
        #
        #load HRIR dataset
        #
        #mat_fname = pjoin(CN.DATA_DIR_INT, 'hrir_dataset_compensated_high.mat')
        mat_fname = pjoin(CN.DATA_DIR_INT, 'hrir_dataset_compensated_max.mat')
        spatial_res=3
        
        elev_min=CN.SPATIAL_RES_ELEV_MIN[spatial_res] 
        elev_max=CN.SPATIAL_RES_ELEV_MAX[spatial_res] 
        elev_nearest=CN.SPATIAL_RES_ELEV_NEAREST[spatial_res] #as per hrir dataset
        elev_nearest_process=CN.SPATIAL_RES_ELEV_NEAREST_PR[spatial_res] 
        azim_nearest=CN.SPATIAL_RES_AZIM_NEAREST[spatial_res] 
        azim_nearest_process=CN.SPATIAL_RES_AZIM_NEAREST_PR[spatial_res] 
        
        #new format MATLAB 7.3 files
        hrirs = mat73.loadmat(mat_fname)
        #load matrix, get list of numpy arrays
        hrir_list = [[element for element in upperElement] for upperElement in hrirs['ash_input_hrir']]
        #grab desired hrtf. by default returns 9x72x2x65536 array
        hrtf_type=1
        hrir_selected = hrir_list[hrtf_type-1]
        total_elev_hrir = len(hrir_selected)
        total_azim_hrir = len(hrir_selected[0])
        total_chan_hrir = len(hrir_selected[0][0])
        total_samples_hrir = len(hrir_selected[0][0][0])
        base_elev_idx = total_elev_hrir//2
        base_elev_idx_offset=total_elev_hrir//8
        
        #define desired angles
        if ir_set in CN.AC_SPACE_LIST_VARIED_R:#reduced directions if variable reverb time
            azim_src_range_a=np.arange(0,60,2)
            azim_src_range_b=np.arange(120,240,2)
            azim_src_range_c=np.arange(300,360,2)
        else:
            azim_src_range_a=np.arange(0,70,2)
            azim_src_range_b=np.arange(110,250,2)
            azim_src_range_c=np.arange(290,360,2)
        
        azim_src_range_ab = np.append(azim_src_range_a,azim_src_range_b)
        azim_src_range_abc = np.append(azim_src_range_ab,azim_src_range_c)
        azim_src_set=azim_src_range_abc

        elev_src_set = np.arange(-40,50,2)

        num_azim_src = len(azim_src_set)
        num_elev_src = len(elev_src_set)
        
        #
        # align HRIRs in time domain
        #
        if align_hrirs == True:
            # take 0 deg azim as reference
            index_peak_ref = np.argmax(np.abs(hrir_selected[base_elev_idx][0][0][:]))
            for elev in range(total_elev_hrir):
                for azim in range(total_azim_hrir):
                    azim_deg = int(azim*azim_nearest)
                    #take left channel if azim < 180 deg, otherwise take right channel
                    if azim_deg < 180:
                        index_peak_cur = np.argmax(np.abs(hrir_selected[elev][azim][0][:]))
                    else:
                        index_peak_cur = np.argmax(np.abs(hrir_selected[elev][azim][1][:]))    
                    hrir_shift = index_peak_ref-index_peak_cur
                    for chan in range(total_chan_hrir):
                        hrir_selected[elev][azim][chan][:] = np.roll(hrir_selected[elev][azim][chan][:],hrir_shift)
        
        
        #set number of output directions
        if out_directions == None:
            num_out_dirs=total_sets_air*total_chan_air#default value wil be sets * channels
        else:
            num_out_dirs=out_directions
        #limit max directions
        num_out_dirs=min(num_out_dirs,total_sets_air*total_chan_air)#cant have more out directions than air sets and channels
        #force minimum
        num_out_dirs=max(num_out_dirs,CN.MIN_REVERB_DIRS)
        
        log_string_a = 'num_out_dirs: ' + str(num_out_dirs)
        if CN.LOG_INFO == 1:
            logging.info(log_string_a)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string_a)
        
        #create numpy array for new BRIR dataset   
        brir_reverberation=np.zeros((CN.INTERIM_ELEVS,num_out_dirs,2,n_fft))
        
        #
        #convolve each input RIR with a sequence of HRIRs
        #
        curr_elev_id=base_elev_idx
        curr_azim_id=0
        curr_elev_in_set=0
        curr_azim_in_set=0
        curr_direction=0
        curr_polarity=1
        fip_polarity=1
        
        #hrir sections
        hrir_seg_len = hf.round_down_even(n_fft//source_directions)
        #create window for overlap add
        wind_r_fade_full=np.bartlett(hrir_seg_len)
        win_r_fade_out = np.split(wind_r_fade_full,2)[1]
        #twice as many overlap windows as source directions due to 100% overlap
        overlap_wins=source_directions*2

        log_string_a = 'BRIR estimation loop running'
        if CN.LOG_INFO == 1:
            logging.info(log_string_a)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string_a)
        
        for set_num in range(total_sets_air):
            #each set represents a new BRIR direction
            for chan_air in range(total_chan_air):
                if curr_direction < num_out_dirs:
                    #for each ir in set and ir channel, perform a convolution with a new HRIR
                    for ir_air in range(total_irs_air):
                        #for each segment in the ir, convolve with a new HRIR and overlapp add into brir array

                        #copy air
                        curr_air = np.copy(air_reverberation[set_num,ir_air,chan_air,:])

                        #grab hrir
                        curr_azim_deg = azim_src_set[curr_azim_in_set]
                        curr_elev_deg = elev_src_set[curr_elev_in_set]
                        curr_azim_id=int(curr_azim_deg/azim_nearest)
                        curr_elev_id=int((curr_elev_deg-elev_min)/elev_nearest)

                        curr_hrir_l=np.copy(hrir_selected[curr_elev_id][curr_azim_id][0][:])
                        curr_hrir_r=np.copy(hrir_selected[curr_elev_id][curr_azim_id][1][:])
                        
                        #convolve with windowed AIR
                        curr_brir_l = sp.signal.convolve(curr_air,curr_hrir_l, 'full', 'auto')
                        curr_brir_r = sp.signal.convolve(curr_air,curr_hrir_r, 'full', 'auto')
 
                        #crop result
                        #then add into output array
                        brir_reverberation[0,curr_direction,0,:] = np.add(brir_reverberation[0,curr_direction,0,:],curr_brir_l[0:n_fft])
                        brir_reverberation[0,curr_direction,1,:] = np.add(brir_reverberation[0,curr_direction,1,:],curr_brir_r[0:n_fft])
  
                        #increment elev and azim so that direction varies across IRs
                        if rand_sources == True:
                            curr_azim_in_set=random.randint(0, num_azim_src-1)
                            curr_elev_in_set=random.randint(0, num_elev_src-1)
                        else:
                            curr_azim_in_set=curr_azim_in_set+(1*curr_polarity)
                            if curr_azim_in_set >= num_azim_src:
                                curr_azim_in_set=0
                                curr_elev_in_set=curr_elev_in_set+1
                            if curr_azim_in_set < 0:
                                curr_azim_in_set=num_azim_src-1
                                curr_elev_in_set=curr_elev_in_set+1
                            if curr_elev_in_set >= num_elev_src:
                                curr_elev_in_set=0 
                                
                    if fip_polarity == 1:
                        curr_polarity=curr_polarity*-1
    
            #increment direction to store results in brir array
            curr_direction=curr_direction+1
        
        #window initial rise
        if ir_set not in CN.AC_SPACE_LIST_SUB:
            for direc in range(num_out_dirs):
                brir_reverberation[0,direc,0,:] = np.multiply(brir_reverberation[0,direc,0,:],direct_removal_win_b)
                brir_reverberation[0,direc,1,:] = np.multiply(brir_reverberation[0,direc,1,:],direct_removal_win_b)
        
        #window tail end of BRIR
        fade_start=int(44100*(n_fft/CN.N_FFT))
        
        l_fade_win_size=np.abs(fade_start-n_fft)*2
        hann_l_fade_full=np.hanning(l_fade_win_size)
        win_l_fade_out = np.split(hann_l_fade_full,2)[1]
        #
        #additional window to fade out tail end of late reflections
        #
        l_fade_out_win = data_pad_zeros.copy()
        l_fade_out_win[0:fade_start] = data_pad_ones[0:fade_start]
        l_fade_out_win[fade_start:] = win_l_fade_out
        for direc in range(num_out_dirs):
            brir_reverberation[0,direc,0,:] = np.multiply(brir_reverberation[0,direc,0,:],l_fade_out_win)
            brir_reverberation[0,direc,1,:] = np.multiply(brir_reverberation[0,direc,1,:],l_fade_out_win)
        
        #
        #set each direction to 0 level
        #
        fb_start=int(CN.SPECT_SNAP_F0*CN.N_FFT/samp_freq)
        fb_end=int(CN.SPECT_SNAP_F1*CN.N_FFT/samp_freq)
        for direc in range(num_out_dirs):
            data_fft = np.fft.fft(brir_reverberation[0,direc,0,0:CN.N_FFT])
            mag_fft=np.abs(data_fft)
            average_mag_l = np.mean(mag_fft[fb_start:fb_end])
            data_fft = np.fft.fft(brir_reverberation[0,direc,1,0:CN.N_FFT])
            mag_fft=np.abs(data_fft)
            average_mag_r = np.mean(mag_fft[fb_start:fb_end])
            average_mag=(average_mag_l+average_mag_r)/2
            for chan in range(total_chan_hrir):
                brir_reverberation[0,direc,chan,:] = np.divide(brir_reverberation[0,direc,chan,:],average_mag)
                
        #
        #optional: create compensation filter from average response and target, then equalise each BRIR
        #
        
        if mag_comp == True:
            
            print('FR Compensation enabled')
            if ir_set in CN.AC_SPACE_LIST_COMPMODE1:
                comp_mode=1#0 = compensate all directions together with a single filter, 1 = comp each direction separately
            else:
                comp_mode=0
            
            #level ends of spectrum
            high_freq=16000#16500,13500
            low_freq_in=40
            if ir_set in CN.AC_SPACE_LIST_SUB:
                high_freq=12000
                low_freq_in=46
            if ir_set == 'broadcast_studio_a':
                high_freq=15000
            
            #load target
            npy_file_name =  'reverb_target_mag_response.npy'
            reverb_folder = pjoin(CN.DATA_DIR_INT, 'reverberation')
            in_file_path = pjoin(reverb_folder,npy_file_name) 
            brir_fft_target_mag = np.load(in_file_path)
            low_freq=40
            if ir_set in CN.AC_SPACE_LIST_SUB:
                low_freq=125
            brir_fft_target_mag = hf.level_spectrum_ends(brir_fft_target_mag, low_freq, high_freq, smooth_win = 20)#150
            
            if comp_mode == 0:
                num_bairs_avg = 0
                brir_fft_avg_db = fr_flat.copy()
                
                #calculate average response
                for direc in range(num_out_dirs):
                    for chan in range(total_chan_hrir):
                        brir_current = np.copy(brir_reverberation[0,direc,chan,0:CN.N_FFT])#brir_out[elev][azim][chan][:]
                        brir_current_fft = np.fft.fft(brir_current)#
                        brir_current_mag_fft=np.abs(brir_current_fft)
                        brir_current_db_fft = hf.mag2db(brir_current_mag_fft)
                        
                        brir_fft_avg_db = np.add(brir_fft_avg_db,brir_current_db_fft)
                        
                        num_bairs_avg = num_bairs_avg+1
                
                #divide by total number of brirs
                brir_fft_avg_db = brir_fft_avg_db/num_bairs_avg
                #convert to mag
                brir_fft_avg_mag = hf.db2mag(brir_fft_avg_db)
                
                brir_fft_avg_mag = hf.level_spectrum_ends(brir_fft_avg_mag, low_freq_in, high_freq, smooth_win = 20)#150
                brir_fft_avg_mag_sm = hf.smooth_fft_octaves(data=brir_fft_avg_mag)
        
                #create compensation filter
                comp_mag = hf.db2mag(np.subtract(hf.mag2db(brir_fft_target_mag),hf.mag2db(brir_fft_avg_mag_sm)))
                comp_mag = hf.smooth_fft_octaves(data=comp_mag)
                #create min phase FIR
                comp_eq_fir = hf.mag_to_min_fir(comp_mag, crop=1)
     
                #equalise each brir with comp filter
                for direc in range(num_out_dirs):
                    for chan in range(total_chan_hrir):
                        #convolve BRIR with filters
                        brir_eq_b = np.copy(brir_reverberation[0,direc,chan,:])#
                        #apply DF eq
                        brir_eq_b = sp.signal.convolve(brir_eq_b,comp_eq_fir, 'full', 'auto')
                        brir_reverberation[0,direc,chan,:] = np.copy(brir_eq_b[0:n_fft])#
    
                if CN.PLOT_ENABLE == 1:
                    print(str(num_bairs_avg))
                    hf.plot_data(brir_fft_target_mag,'brir_fft_target_mag', normalise=0)
                    hf.plot_data(brir_fft_avg_mag,'brir_fft_avg_mag', normalise=0)
                    hf.plot_data(brir_fft_avg_mag_sm,'brir_fft_avg_mag_sm', normalise=0)  
                    hf.plot_data(comp_mag,'comp_mag', normalise=0) 
                    
            else:
         
                
                for direc in range(num_out_dirs):
                    
                    num_bairs_avg = 0
                    brir_fft_avg_db = fr_flat.copy()
                    
                    #calculate average response
                    for chan in range(total_chan_hrir):
                        brir_current = np.copy(brir_reverberation[0,direc,chan,0:CN.N_FFT])#brir_out[elev][azim][chan][:]
                        brir_current_fft = np.fft.fft(brir_current)#
                        brir_current_mag_fft=np.abs(brir_current_fft)
                        brir_current_db_fft = hf.mag2db(brir_current_mag_fft)
                        
                        brir_fft_avg_db = np.add(brir_fft_avg_db,brir_current_db_fft)
                        
                        num_bairs_avg = num_bairs_avg+1
                
                    #divide by total number of brirs
                    brir_fft_avg_db = brir_fft_avg_db/num_bairs_avg
                    #convert to mag
                    brir_fft_avg_mag = hf.db2mag(brir_fft_avg_db)
                    #level ends of spectrum
                    brir_fft_avg_mag = hf.level_spectrum_ends(brir_fft_avg_mag, low_freq_in, high_freq, smooth_win = 20)#150
                    brir_fft_avg_mag_sm = hf.smooth_fft_octaves(data=brir_fft_avg_mag)
            
                    #create compensation filter
                    comp_mag = hf.db2mag(np.subtract(hf.mag2db(brir_fft_target_mag),hf.mag2db(brir_fft_avg_mag_sm)))
                    comp_mag = hf.smooth_fft_octaves(data=comp_mag)
                    #create min phase FIR
                    comp_eq_fir = hf.mag_to_min_fir(comp_mag, crop=1)
     
                    #equalise each brir with comp filter
                    for chan in range(total_chan_hrir):
                        #convolve BRIR with filters
                        brir_eq_b = np.copy(brir_reverberation[0,direc,chan,:])#
                        #apply DF eq
                        brir_eq_b = sp.signal.convolve(brir_eq_b,comp_eq_fir, 'full', 'auto')
                        brir_reverberation[0,direc,chan,:] = np.copy(brir_eq_b[0:n_fft])#
    
                    if CN.PLOT_ENABLE == 1:
                        print(str(num_bairs_avg))
                        hf.plot_data(brir_fft_target_mag,str(direc)+'_brir_fft_target_mag', normalise=0)
                        hf.plot_data(brir_fft_avg_mag,str(direc)+'_brir_fft_avg_mag', normalise=0)
                        hf.plot_data(brir_fft_avg_mag_sm,str(direc)+'_brir_fft_avg_mag_sm', normalise=0)  
                        hf.plot_data(comp_mag,str(direc)+'_comp_mag', normalise=0) 

        
        #
        #set each direction to 0 level again
        #
        fb_start=int(CN.SPECT_SNAP_F0*CN.N_FFT/samp_freq)
        fb_end=int(CN.SPECT_SNAP_F1*CN.N_FFT/samp_freq)
        for direc in range(num_out_dirs):
            data_fft = np.fft.fft(brir_reverberation[0,direc,0,0:CN.N_FFT])
            mag_fft=np.abs(data_fft)
            average_mag_l = np.mean(mag_fft[fb_start:fb_end])
            data_fft = np.fft.fft(brir_reverberation[0,direc,1,0:CN.N_FFT])
            mag_fft=np.abs(data_fft)
            average_mag_r = np.mean(mag_fft[fb_start:fb_end])
            average_mag=(average_mag_l+average_mag_r)/2
            for chan in range(total_chan_hrir):
                brir_reverberation[0,direc,chan,:] = np.divide(brir_reverberation[0,direc,chan,:],average_mag)


        #
        #export wavs for testing
        #
        brir_out_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'est_brirs',ir_set)
        for direc in range(num_out_dirs):
            out_file_name = ir_set+'_'+str(direc)+'_est_brir.wav'
            out_file_path = pjoin(brir_out_folder,out_file_name)
            
            #create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            out_wav_array=np.zeros((n_fft,2))
            #grab BRIR
            out_wav_array[:,0] = np.copy(brir_reverberation[0,direc,0,:])#L
            out_wav_array[:,1] = np.copy(brir_reverberation[0,direc,1,:])#R
            
            if output_wavs == 1:
                hf.write2wav(file_name=out_file_path, data=out_wav_array, prevent_clipping=1, samplerate=samp_freq)
        
        #
        #save numpy array for later use in BRIR generation functions
        #
        if mag_comp == True or ir_set in CN.AC_SPACE_LIST_NOCOMP:
            npy_file_name =  'reverberation_dataset_' +ir_set+'.npy'
        else:
            npy_file_name =  'reverberation_dataset_nocomp_' +ir_set+'.npy'
    
        out_file_path = pjoin(brir_out_folder,npy_file_name)      
          
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        np.save(out_file_path,brir_reverberation)    
        
        log_string_a = 'Exported numpy file to: ' + out_file_path 
        if CN.LOG_INFO == 1:
            logging.info(log_string_a)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string_a)
   
    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to complete AIR to BRIR processing for: ' + ir_set 
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_error(log_string)
    
    
 
    
def raw_brirs_to_brir_set(ir_set='fw', df_comp=True, mag_comp=CN.MAG_COMP, lf_align=True,gui_logger=None):
    """
    function to convert RIRs to BRIRs
    :param ir_set: str, name of impulse response set. Must correspond to a folder in ASH-Toolset\data\interim\ir_data\<ir_group>
    :param df_comp: bool, true = apply diffuse field compensation, false = no compensation
    :param mag_comp: bool, true = apply target equalisation, false = no eq
    :param lf_align: bool, true = align brirs in time domain by peaks, false = no alignment
    :return: None
    """ 
    
    if ir_set in CN.AC_SPACE_LIST_LOWRT60:
        n_fft=CN.N_FFT
    else:
        n_fft=CN.N_FFT_L
    
    if ir_set in CN.AC_SPACE_LIST_NOROLL:
        lf_align=False
 
    
    
    samp_freq=44100
    output_wavs=1
    
    #windows
    data_pad_zeros=np.zeros(n_fft)
    data_pad_ones=np.ones(n_fft)
    total_chan_brir=2
    
    #direct sound window for 2nd phase
    direct_hanning_size=300#350
    direct_hanning_start=51#101
    hann_direct_full=np.hanning(direct_hanning_size)
    hann_direct = np.split(hann_direct_full,2)[0]
    direct_removal_win_b = data_pad_zeros.copy()
    direct_removal_win_b[direct_hanning_start:direct_hanning_start+int(direct_hanning_size/2)] = hann_direct
    direct_removal_win_b[direct_hanning_start+int(direct_hanning_size/2):]=data_pad_ones[direct_hanning_start+int(direct_hanning_size/2):]
    
    #impulse
    impulse=np.zeros(n_fft)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat = hf.mag2db(fr_flat_mag)
    #impulse
    impulse=np.zeros(CN.N_FFT)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat_s = hf.mag2db(fr_flat_mag)
    
    #loop through folders
    ir_in_folder = pjoin(CN.DATA_DIR_RAW, 'ir_data', 'raw_brirs',ir_set)
    
    try:
    
        #get number of IRs
        total_irs=0
        for root, dirs, files in os.walk(ir_in_folder):
            for filename in files:
                if '.wav' in filename:
                    total_irs=total_irs+1
          
        if total_irs == 5 or total_irs == 7 or total_irs > 7:
            num_out_sets=total_irs
        else:
            raise ValueError('Invalid number of input BRIRs')
            
        log_string_a = 'num_out_sets: ' + str(num_out_sets)
        if CN.LOG_INFO == 1:
            logging.info(log_string_a)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string_a)
          
        #numpy array, num sets x num irs in each set x 2 channels x NFFT max samples
        brir_reverberation=np.zeros((CN.INTERIM_ELEVS,num_out_sets,2,n_fft))
        
        set_counter=0
        for root, dirs, files in os.walk(ir_in_folder):
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
                    if samplerate != samp_freq:
                        fir_array = hf.resample_signal(fir_array, new_rate = samp_freq)
                        log_string_a = 'source samplerate: ' + str(samplerate) + ', resampled to 44100Hz'
                        if CN.LOG_INFO == 1:
                            logging.info(log_string_a)
                        if CN.LOG_GUI == 1 and gui_logger != None:
                            gui_logger.log_info(log_string_a)
 
                    extract_legth = min(n_fft,fir_length)
                    #load into numpy array
                    brir_reverberation[0,set_counter,0,0:extract_legth]=fir_array[0:extract_legth,0]#L
                    brir_reverberation[0,set_counter,1,0:extract_legth]=fir_array[0:extract_legth,1]#R
                    
                    set_counter=set_counter+1
        
 
        #crop raw IRs so that direct peak is at sample 50
        index_peak_ref = 40#50
        
        for set_num in range(num_out_sets):
            index_peak_cur = np.argmax(np.abs(brir_reverberation[0,set_num,0,:]))
            ir_shift = index_peak_ref-index_peak_cur
            brir_reverberation[0,set_num,0,:] = np.roll(brir_reverberation[0,set_num,0,:],ir_shift)
            brir_reverberation[0,set_num,1,:] = np.roll(brir_reverberation[0,set_num,1,:],ir_shift)

            #set end of array to zero to remove any data shifted to end of array
            if ir_shift < 0:
                brir_reverberation[0,set_num,0,ir_shift:] = brir_reverberation[0,set_num,0,ir_shift:]*0#left
                brir_reverberation[0,set_num,1,ir_shift:] = brir_reverberation[0,set_num,1,ir_shift:]*0#right
        

 
        
        #perform time domain synchronous averaging
        #align in low frequencies
        if lf_align == True:
    
            #contants for TD alignment of BRIRs
            t_shift_interval = CN.T_SHIFT_INTERVAL
            min_t_shift = CN.MIN_T_SHIFT_B
            max_t_shift = CN.MAX_T_SHIFT_B
            num_intervals = int(np.abs((max_t_shift-min_t_shift)/t_shift_interval))
            order=7#default 6
            delay_win_min_t = CN.DELAY_WIN_MIN_T
            delay_win_max_t = CN.DELAY_WIN_MAX_T
            delay_win_hop_size = CN.DELAY_WIN_HOP_SIZE
            delay_win_hops = CN.DELAY_WIN_HOPS
            cutoff_alignment = CN.CUTOFF_SUB#140
            #peak to peak within a sufficiently small sample window
            peak_to_peak_window = int(np.divide(samp_freq,cutoff_alignment)*0.95) #int(np.divide(samp_freq,cutoff_alignment)) 
            
            delay_eval_set = np.zeros((num_out_sets,num_intervals))
            

            #go through each room in the ordered list
            for ir in range(num_out_sets-1):#room in range(total_airs-1)
                this_air_orig_idx=ir
                next_air_orig_idx=ir+1

                #method 2: take sum of all prior rooms and this room
                rooms_to_add = 0
                this_air = data_pad_zeros.copy()
                for cum_air in range(ir+1):
                    cum_air_orig_idx = cum_air
                    rooms_to_add = rooms_to_add+1
                    this_air = np.add(this_air,brir_reverberation[0,cum_air_orig_idx,0,:])
                this_air = np.divide(this_air,rooms_to_add) 

                calc_delay = 1
                    
                if calc_delay == 1:
                    next_air = np.copy(brir_reverberation[0,next_air_orig_idx,0,:])
                    for delay in range(num_intervals):
                        
                        #shift next room BRIR
                        current_shift = min_t_shift+(delay*t_shift_interval)
                        n_air_shift = np.roll(next_air,current_shift)
                        #add current room BRIR to shifted next room BRIR
                        sum_ir = np.add(this_air,n_air_shift)
                        #calculate group delay
     
                        sum_ir_lp = hf.signal_lowpass_filter(sum_ir, cutoff_alignment, samp_freq, order)
                        peak_to_peak_iter=0
                        for hop_id in range(delay_win_hops):
                            samples = hop_id*delay_win_hop_size
                            peak_to_peak = np.abs(np.max(sum_ir_lp[delay_win_min_t+samples:delay_win_min_t+samples+peak_to_peak_window])-np.min(sum_ir_lp[delay_win_min_t+samples:delay_win_min_t+samples+peak_to_peak_window]))
                            #if this window has larger pk to pk, store in iter var
                            if peak_to_peak > peak_to_peak_iter:
                                peak_to_peak_iter = peak_to_peak
                        #store largest pk to pk distance of all windows into delay set
                        delay_eval_set[next_air_orig_idx,delay] = peak_to_peak_iter
                    
                    #shift next room by delay that has largest peak to peak distance (method 4 and 5)
                    index_shift = np.argmax(delay_eval_set[next_air_orig_idx,:])
    
                
                samples_shift=min_t_shift+(index_shift*t_shift_interval)
                brir_reverberation[0,next_air_orig_idx,0,:] = np.roll(brir_reverberation[0,next_air_orig_idx,0,:],samples_shift)#left
                brir_reverberation[0,next_air_orig_idx,1,:] = np.roll(brir_reverberation[0,next_air_orig_idx,1,:],samples_shift)#right
                
                #20240511: set end of array to zero to remove any data shifted to end of array
                if samples_shift < 0:
                    brir_reverberation[0,next_air_orig_idx,0,min_t_shift:] = brir_reverberation[0,next_air_orig_idx,0,min_t_shift:]*0#left
                    brir_reverberation[0,next_air_orig_idx,1,min_t_shift:] = brir_reverberation[0,next_air_orig_idx,1,min_t_shift:]*0#right
                        
     
        #
        #apply diffuse field compensation
        #
        if ir_set in CN.AC_SPACE_LIST_KU100:
            dummy_head='ku_100'
        else:
            dummy_head='other'
        
        if dummy_head == 'ku_100':
        
            #load compensation for ku100 BRIRs
            filename = 'ku_100_dummy_head_compensation.wav'
            wav_fname = pjoin(CN.DATA_DIR_INT, filename)
            samplerate, df_eq = wavfile.read(wav_fname)
            df_eq = df_eq / (2.**31)
        
        else:
  
            #determine magnitude response, assume 0 phase
            #set magnitude to 0dB
            #invert response
            #convert to IR
            #if using IFFT, result will be 0 phase and symmetric
            #if using ifftshift, result will be linear phase
            #get linear phase FIR, window, and convert to minimum phase
            #window min phase FIR again to remove artefacts
            
            num_bairs_avg = 0
            brir_fft_avg_db = fr_flat.copy()
            
            #get diffuse field spectrum
            for direc in range(num_out_sets):
                for chan in range(total_chan_brir):
                    brir_current = np.copy(brir_reverberation[0,direc,chan,0:CN.N_FFT])#brir_out[elev][azim][chan][:]
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
            brir_fft_avg_mag = hf.level_spectrum_ends(brir_fft_avg_mag, 40, 19000, smooth_win = 10)#150
        
            #smoothing
            if CN.SPECT_SMOOTH_MODE == 0:
                #6 stage process
                brir_fft_avg_mag_sm = hf.smooth_fft(brir_fft_avg_mag, 6729, 13, 1077)
                brir_fft_avg_mag_sm = hf.smooth_fft(brir_fft_avg_mag_sm, 673, 13, 538)
                brir_fft_avg_mag_sm = hf.smooth_fft(brir_fft_avg_mag_sm, 471, 13, 269)
                brir_fft_avg_mag_sm = hf.smooth_fft(brir_fft_avg_mag_sm, 269, 13, 135)
                brir_fft_avg_mag_sm = hf.smooth_fft(brir_fft_avg_mag_sm, 101, 27, 7)
                brir_fft_avg_mag_sm = hf.smooth_fft(brir_fft_avg_mag_sm, 20187, 13, 1077)
            else:
                #octave smoothing
                brir_fft_avg_mag_sm = hf.smooth_fft_octaves(data=brir_fft_avg_mag, n_fft=n_fft, win_size_base = 20)
            
            #invert response
            brir_fft_avg_mag_inv = hf.db2mag(hf.mag2db(brir_fft_avg_mag_sm)*-1)
            #create min phase FIR
            brir_df_inv_fir = hf.mag_to_min_fir(brir_fft_avg_mag_inv, crop=1)
            df_eq = brir_df_inv_fir
            
        if df_comp == True:
            #convolve with inverse filter
            for direc in range(num_out_sets):
                for chan in range(total_chan_brir):
                    brir_eq_b = np.copy(brir_reverberation[0,direc,chan,:])
                    #apply DF eq
                    brir_eq_b = sp.signal.convolve(brir_eq_b,df_eq, 'full', 'auto')
                    brir_reverberation[0,direc,chan,:] = np.copy(brir_eq_b[0:n_fft])
   
        #remove direction portion of signal
        for set_num in range(num_out_sets):
            for chan in range(total_chan_brir):
                # RIR has been shifted so apply fade in window again to remove any overlap with HRIR
                brir_reverberation[0,set_num,chan,:] = np.multiply(brir_reverberation[0,set_num,chan,:],direct_removal_win_b)

        #set each AIR to 0 level
        fb_start=int(CN.SPECT_SNAP_F0*CN.N_FFT/samp_freq)
        fb_end=int(CN.SPECT_SNAP_F1*CN.N_FFT/samp_freq)
        for set_num in range(num_out_sets):
            data_fft = np.fft.fft(brir_reverberation[0,set_num,0,0:CN.N_FFT])
            mag_fft=np.abs(data_fft)
            average_mag_l = np.mean(mag_fft[fb_start:fb_end])
            data_fft = np.fft.fft(brir_reverberation[0,set_num,1,0:CN.N_FFT])
            mag_fft=np.abs(data_fft)
            average_mag_r = np.mean(mag_fft[fb_start:fb_end])
            average_mag=(average_mag_l+average_mag_r)/2
            if average_mag > 0:
                for chan in range(total_chan_brir):
                    brir_reverberation[0,set_num,chan,:] = np.divide(brir_reverberation[0,set_num,chan,:],average_mag)
       
    
        #
        #optional: create compensation filter from average response and target, then equalise each BRIR
        #
        if mag_comp == True:
            
            num_bairs_avg = 0
            brir_fft_avg_db = fr_flat_s.copy()
            
            #calculate average response
            for direc in range(num_out_sets):
                for chan in range(total_chan_brir):
                    brir_current = np.copy(brir_reverberation[0,direc,chan,0:CN.N_FFT])#brir_out[elev][azim][chan][:]
                    brir_current_fft = np.fft.fft(brir_current)#
                    brir_current_mag_fft=np.abs(brir_current_fft)
                    brir_current_db_fft = hf.mag2db(brir_current_mag_fft)
                    
                    brir_fft_avg_db = np.add(brir_fft_avg_db,brir_current_db_fft)
                    
                    num_bairs_avg = num_bairs_avg+1
            
            #divide by total number of brirs
            brir_fft_avg_db = brir_fft_avg_db/num_bairs_avg
            #convert to mag
            brir_fft_avg_mag = hf.db2mag(brir_fft_avg_db)
            #level ends of spectrum
            brir_fft_avg_mag = hf.level_spectrum_ends(brir_fft_avg_mag, 40, 17000, smooth_win = 10)#150
            brir_fft_avg_mag_sm = hf.smooth_fft_octaves(data=brir_fft_avg_mag, win_size_base = 20)
            
            #load target
            npy_file_name =  'reverb_target_mag_response.npy'
            reverb_folder = pjoin(CN.DATA_DIR_INT, 'reverberation')
            in_file_path = pjoin(reverb_folder,npy_file_name) 
            brir_fft_target_mag = np.load(in_file_path)
            brir_fft_target_mag = hf.level_spectrum_ends(brir_fft_target_mag, 40, 17000, smooth_win = 10)#150
            
            #create compensation filter
            comp_mag = hf.db2mag(np.subtract(hf.mag2db(brir_fft_target_mag),hf.mag2db(brir_fft_avg_mag_sm)))
            comp_mag = hf.smooth_fft_octaves(data=comp_mag, win_size_base = 30)
            #create min phase FIR
            comp_eq_fir = hf.mag_to_min_fir(comp_mag, crop=1)
 
            #equalise each brir with comp filter
            for direc in range(num_out_sets):
                for chan in range(total_chan_brir):
                    #convolve BRIR with filters
                    brir_eq_b = np.copy(brir_reverberation[0,direc,chan,:])#
                    #apply DF eq
                    brir_eq_b = sp.signal.convolve(brir_eq_b,comp_eq_fir, 'full', 'auto')
                    brir_reverberation[0,direc,chan,:] = np.copy(brir_eq_b[0:n_fft])#

            if CN.PLOT_ENABLE == 1:
                print(str(num_bairs_avg))
                hf.plot_data(brir_fft_target_mag,'brir_fft_target_mag', normalise=0)
                hf.plot_data(brir_fft_avg_mag,'brir_fft_avg_mag', normalise=0)
                hf.plot_data(brir_fft_avg_mag_sm,'brir_fft_avg_mag_sm', normalise=0)  
                hf.plot_data(comp_mag,'comp_mag', normalise=0) 
        
    
    
    
        #
        #export wavs for testing
        #
        brir_out_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'comp_brirs',ir_set)
        for direc in range(num_out_sets):
            out_file_name = ir_set+'_'+str(direc)+'_comp_brir.wav'
            out_file_path = pjoin(brir_out_folder,out_file_name)
            
            #create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            out_wav_array=np.zeros((n_fft,2))
            #grab BRIR
            out_wav_array[:,0] = np.copy(brir_reverberation[0,direc,0,:])#L
            out_wav_array[:,1] = np.copy(brir_reverberation[0,direc,1,:])#R
            
            if output_wavs == 1:
                hf.write2wav(file_name=out_file_path, data=out_wav_array, prevent_clipping=1, samplerate=samp_freq)
        
        #
        #save numpy array for later use in BRIR generation functions
        #
        npy_file_name =  'reverberation_dataset_' +ir_set+'.npy'
    
        out_file_path = pjoin(brir_out_folder,npy_file_name)      
          
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        np.save(out_file_path,brir_reverberation)    
        
        log_string_a = 'Exported numpy file to: ' + out_file_path 
        if CN.LOG_INFO == 1:
            logging.info(log_string_a)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string_a)
       

    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to complete BRIR processing for: ' + ir_set 
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_error(log_string)
    
    
    
    
    
    
def loadIR(sessionPath):
    """Load impulse response (IR) data

    Parameters
    ------
    sessionPath: Path to IR folder

    Returns
    ------
    pos_mic: Microphone positions of shape (numMic, 3)
    pos_src: Source positions of shape (numSrc, 3)
    fullIR: IR data of shape (numSrc, numMic, irLen)
    """
    pos_mic = np.load(sessionPath.joinpath("pos_mic.npy"))
    pos_src = np.load(sessionPath.joinpath("pos_src.npy"))

    numMic = pos_mic.shape[0]
    
    allIR = []
    irIndices = []
    for f in sessionPath.iterdir():
        if not f.is_dir():
            if f.stem.startswith("ir_"):
                allIR.append(np.load(f))
                irIndices.append(int(f.stem.split("_")[-1]))

    assert(len(allIR) == numMic)
    numSrc = allIR[0].shape[0]
    irLen = allIR[0].shape[-1]
    fullIR = np.zeros((numSrc, numMic, irLen))
    for i, ir in enumerate(allIR):
        assert(ir.shape[0] == numSrc)
        assert(ir.shape[-1] == irLen)
        fullIR[:, irIndices[i], :] = ir

    return pos_mic, pos_src, fullIR    
    



def calc_avg_room_target_mag(gui_logger=None):
    """
    function to calculate average magnitude response of various rooms
    :return: None
    """ 
    
    air_ref_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'full_airs')
    #ir_data_folder = pjoin(CN.DATA_DIR_RAW, 'ir_data', 'prepped_airs')
    
    n_fft=CN.N_FFT
    #impulse
    impulse=np.zeros(n_fft)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat = hf.mag2db(fr_flat_mag)
    
    num_airs_avg = 0
    air_fft_avg_db = fr_flat.copy()
    
    log_string_a = 'Mag response estimation loop running'
    if CN.LOG_INFO == 1:
        logging.info(log_string_a)
    if CN.LOG_GUI == 1 and gui_logger != None:
        gui_logger.log_info(log_string_a)
            
    #loop through folders
    try:
        for root, dirs, files in os.walk(air_ref_folder):
            for filename in files:
                if '.npy' in filename and 'broadcast_studio_a' in filename:
                    #read npy files
                    air_file_path = pjoin(root, filename)
                    air_set = np.load(air_file_path)
                    
                    #get number of sets
                    total_sets_air = len(air_set)
                    total_irs_air =  len(air_set[0])
                    total_chan_air = len(air_set[0][0])
                    total_irs_sel=120
                    
                    for set_num in range(total_sets_air):
                        #each set represents a new BRIR direction
                        for chan_air in range(total_chan_air):
                            for ir_air in range(total_irs_sel):
                                curr_air = np.copy(air_set[set_num,ir_air,chan_air,0:n_fft])
                                if np.sum(np.abs(curr_air)) > 0:
                                    air_current_fft = np.fft.fft(curr_air)#
                                    air_current_mag_fft=np.abs(air_current_fft)
                                    air_current_db_fft = hf.mag2db(air_current_mag_fft)     
                                    air_fft_avg_db = np.add(air_fft_avg_db,air_current_db_fft)
                                    num_airs_avg = num_airs_avg+1
                                else:
                                    print(str(filename))
                                    print(str(set_num))
                                    print(str(chan_air))
                                    print(str(ir_air))
    
    
        #divide by total number of brirs
        air_fft_avg_db = air_fft_avg_db/num_airs_avg
        #convert to mag
        air_fft_avg_mag = hf.db2mag(air_fft_avg_db)
        #level ends of spectrum
        air_fft_avg_mag = hf.level_spectrum_ends(air_fft_avg_mag, 40, 18000, smooth_win = 10)#150
        
        #smoothing
        #octave smoothing
        air_fft_avg_mag_sm = hf.smooth_fft_octaves(data=air_fft_avg_mag, n_fft=n_fft)
        
        #create min phase FIR
        #avg_room_min_fir = hf.mag_to_min_fir(air_fft_avg_mag_sm, crop=1)

 
        if CN.PLOT_ENABLE == 1:
            print(str(num_airs_avg))
            hf.plot_data(air_fft_avg_mag,'air_fft_avg_mag', normalise=0)
            hf.plot_data(air_fft_avg_mag_sm,'air_fft_avg_mag_sm', normalise=0)    
  
    
        #todo:
        #load existing room targets from .mat
        #create new array including old and new target
        # #
        # #save numpy array for later use in BRIR generation functions
        # #
        # npy_file_name =  'avg_room_target_mag_response.npy'
        # brir_out_folder = pjoin(CN.DATA_DIR_INT, 'reverberation')
        # out_file_path = pjoin(brir_out_folder,npy_file_name)      
          
        # output_file = Path(out_file_path)
        # output_file.parent.mkdir(exist_ok=True, parents=True)
        
        # np.save(out_file_path,air_fft_avg_mag_sm)    
        
        # log_string_a = 'Exported numpy file to: ' + out_file_path 
        # if CN.LOG_INFO == 1:
        #     logging.info(log_string_a)
        # if CN.LOG_GUI == 1 and gui_logger != None:
        #     gui_logger.log_info(log_string_a)
    
    
    
    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to complete reverb response processing'
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_error(log_string)
            
            
            
    
def calc_reverb_target_mag(gui_logger=None):
    """
    function to calculate average magnitude response of BRIR sets
    :return: None
    """ 
    
    brir_ref_folder = pjoin(CN.DATA_DIR_INT, 'reverberation', 'ref_bin')
    
    n_fft=CN.N_FFT
    
    samp_freq=44100
    output_wavs=1
    
    #windows
    data_pad_zeros=np.zeros(n_fft)
    data_pad_ones=np.ones(n_fft)
    total_chan_brir=2
    
    #direct sound window for 2nd phase
    direct_hanning_size=300#350
    direct_hanning_start=51#101
    hann_direct_full=np.hanning(direct_hanning_size)
    hann_direct = np.split(hann_direct_full,2)[0]
    direct_removal_win_b = data_pad_zeros.copy()
    direct_removal_win_b[direct_hanning_start:direct_hanning_start+int(direct_hanning_size/2)] = hann_direct
    direct_removal_win_b[direct_hanning_start+int(direct_hanning_size/2):]=data_pad_ones[direct_hanning_start+int(direct_hanning_size/2):]
    
    #impulse
    impulse=np.zeros(n_fft)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat = hf.mag2db(fr_flat_mag)
    
    #loop through folders
    try:

        num_bairs_avg = 0
        brir_fft_avg_db = fr_flat.copy()
        
        for root, dirs, files in os.walk(brir_ref_folder):
            for filename in files:
                if '.npy' in filename:

                    #read npy files
                    air_file_path = pjoin(root, filename)
                    brir_reverberation = np.load(air_file_path)
                    
                    #get number of sets
                    total_elev_reverb = len(brir_reverberation)
                    total_azim_reverb = len(brir_reverberation[0])
                    total_chan_reverb = len(brir_reverberation[0][0])
                    total_samples_reverb = len(brir_reverberation[0][0][0])
      
                    #get diffuse field spectrum
                    for direc in range(total_azim_reverb):
                        for chan in range(total_chan_reverb):
                            brir_current = np.copy(brir_reverberation[0,direc,chan,0:n_fft])#brir_out[elev][azim][chan][:]
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
        brir_fft_avg_mag = hf.level_spectrum_ends(brir_fft_avg_mag, 40, 18000, smooth_win = 10)#150
        
        #smoothing
        #octave smoothing
        brir_fft_avg_mag_sm = hf.smooth_fft_octaves(data=brir_fft_avg_mag, n_fft=n_fft)
  
 
        if CN.PLOT_ENABLE == 1:
            print(str(num_bairs_avg))
            hf.plot_data(brir_fft_avg_mag,'brir_fft_avg_mag', normalise=0)
            hf.plot_data(brir_fft_avg_mag_sm,'brir_fft_avg_mag_sm', normalise=0)    
  
        #
        #save numpy array for later use in BRIR generation functions
        #
        npy_file_name =  'reverb_target_mag_response.npy'
        brir_out_folder = pjoin(CN.DATA_DIR_INT, 'reverberation')
        out_file_path = pjoin(brir_out_folder,npy_file_name)      
          
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        np.save(out_file_path,brir_fft_avg_mag_sm)    
        
        log_string_a = 'Exported numpy file to: ' + out_file_path 
        if CN.LOG_INFO == 1:
            logging.info(log_string_a)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string_a)
       

    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to complete BRIR reverb response processing'
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_error(log_string)
    
    
def calc_subrir(gui_logger=None):
    """
    function to calculate time domain average low frequency BRIR
    :return: None
    """ 
    n_fft=CN.N_FFT
    total_chan_brir=2
    output_wavs=1
    additional_filt=1
    samp_freq=44100
    variants=2#0=low reverb, 1=higher reverb
    
    #windows
    data_pad_zeros=np.zeros(n_fft)
    data_pad_ones=np.ones(n_fft)
    
    #initial rise window for sub
    initial_hanning_size=50
    initial_hanning_start=0#190
    hann_initial_full=np.hanning(initial_hanning_size)
    hann_initial = np.split(hann_initial_full,2)[0]
    initial_removal_win_sub = data_pad_zeros.copy()
    initial_removal_win_sub[initial_hanning_start:initial_hanning_start+int(initial_hanning_size/2)] = hann_initial
    initial_removal_win_sub[initial_hanning_start+int(initial_hanning_size/2):]=data_pad_ones[initial_hanning_start+int(initial_hanning_size/2):]
    
    #window for late reflections
    target_rt=400#ms
    n_fade_win_start = int(((target_rt)/1000)*CN.FS)
    n_fade_win_size=np.abs(8000)*2
    wind_n_fade_full=np.bartlett(n_fade_win_size)
    win_n_fade_out = np.split(wind_n_fade_full,2)[1]
    #additional window to fade out noise
    n_fade_out_win = data_pad_zeros.copy()
    n_fade_out_win[0:n_fade_win_start] = data_pad_ones[0:n_fade_win_start]
    n_fade_out_win[n_fade_win_start:n_fade_win_start+int(n_fade_win_size/2)] = win_n_fade_out
    
    #final window for late reflections
    target_rt=400#ms
    f_fade_win_start = int(((target_rt)/1000)*CN.FS)
    f_fade_win_size=np.abs(20000)*2
    wind_f_fade_full=np.bartlett(f_fade_win_size)
    win_f_fade_out = np.split(wind_f_fade_full,2)[1]
    #additional window to fade out noise
    f_fade_out_win = data_pad_zeros.copy()
    f_fade_out_win[0:f_fade_win_start] = data_pad_ones[0:f_fade_win_start]
    f_fade_out_win[f_fade_win_start:f_fade_win_start+int(f_fade_win_size/2)] = win_f_fade_out
    
    try:
        
 
        # load reference sub bass BRIR (FIR)
    
        selected_sub_brir= 'ash_sub_brir'       
        mat_fname = pjoin(CN.DATA_DIR_INT, 'sub_brir_dataset.mat')
        sub_brir_mat = mat73.loadmat(mat_fname)
        sub_brir_ir = np.zeros((2,n_fft))
        sub_brir_ir[0,0:CN.N_FFT] = sub_brir_mat[selected_sub_brir][0][0:CN.N_FFT]
        sub_brir_ir[1,0:CN.N_FFT] = sub_brir_mat[selected_sub_brir][1][0:CN.N_FFT]
        
        #variable crossover depending on RT60
        f_crossover_var=100#100
        peak_to_peak_window_sub = int(np.divide(samp_freq,f_crossover_var)*0.95)
        
        mag_range_a=int(30*(n_fft/CN.N_FFT))
        mag_range_b=int(290*(n_fft/CN.N_FFT))
        
        #set level of reference sub BRIR to 0 at low freqs
        data_fft = np.fft.fft(sub_brir_ir[0][:])
        mag_fft=np.abs(data_fft)
        average_mag = np.mean(mag_fft[mag_range_a:mag_range_b])
        if average_mag == 0:
            if CN.LOG_INFO == 1:
                logging.info('0 magnitude detected')
        for chan in range(CN.TOTAL_CHAN_BRIR):
            sub_brir_ir[chan][:] = np.divide(sub_brir_ir[chan][:],average_mag)
        
        #
        #section for loading estimated sub BRIR datasets and integrating into reference
        #
        
        num_sub_sets=4 
        
        #create numpy array for new BRIR dataset   
        subrir_sets=np.zeros((num_sub_sets,1,total_chan_brir,n_fft))
        subrir_sets_interim=np.zeros((num_sub_sets,1,total_chan_brir,n_fft))
        
        for sub_set_id in range(num_sub_sets):

            # load sub bass BRIR (estimated)
            if sub_set_id == 0:
                ir_set='sub_set_b'
            elif sub_set_id == 1:
                ir_set='sub_set_c'
            elif sub_set_id == 2:
                ir_set='sub_set_d'    
            elif sub_set_id == 3:
                ir_set='sub_set_e'
            if sub_set_id == 3:#wav
                brir_reverberation=np.zeros((CN.INTERIM_ELEVS,1,2,n_fft))
                #read wav files
                brir_out_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'est_brirs',ir_set)
                filename='BRIR_R32_C1_E0_A30_eq.wav'
                wav_fname = pjoin(brir_out_folder, filename)
                samplerate, data = wavfile.read(wav_fname)
                #samp_freq=samplerate
                fir_array = data / (2.**31)
                fir_length = len(fir_array)
                
                #resample if sample rate is not 44100
                if samplerate != samp_freq:
                    fir_array = hf.resample_signal(fir_array, new_rate = samp_freq)
                    log_string_a = 'source samplerate: ' + str(samplerate) + ', resampled to 44100Hz'
                    if CN.LOG_INFO == 1:
                        logging.info(log_string_a)
                    if CN.LOG_GUI == 1 and gui_logger != None:
                        gui_logger.log_info(log_string_a)

                extract_legth = min(n_fft,fir_length)
                #load into numpy array
                brir_reverberation[0,0,0,0:extract_legth]=fir_array[0:extract_legth,0]#L
                brir_reverberation[0,0,1,0:extract_legth]=fir_array[0:extract_legth,1]#R
  
            
            else:
                brir_out_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'est_brirs',ir_set)
                npy_file_name =  'reverberation_dataset_' +ir_set+'.npy'
                out_file_path = pjoin(brir_out_folder,npy_file_name)  
                brir_reverberation = np.load(out_file_path)
                
            

            #set level of subrir set est brir to 0 at low freqs
            #set each AIR to 0 level
            set_num=0
            data_fft = np.fft.fft(brir_reverberation[0,set_num,0,0:CN.N_FFT])
            mag_fft=np.abs(data_fft)
            average_mag_l = np.mean(mag_fft[mag_range_a:mag_range_b])
            data_fft = np.fft.fft(brir_reverberation[0,set_num,1,0:CN.N_FFT])
            mag_fft=np.abs(data_fft)
            average_mag_r = np.mean(mag_fft[mag_range_a:mag_range_b])
            average_mag=(average_mag_l+average_mag_r)/2
            if average_mag > 0:
                for chan in range(total_chan_brir):
                    brir_reverberation[0,set_num,chan,:] = np.divide(brir_reverberation[0,set_num,chan,:],average_mag)
      
            #
            #align in time domain
            #
            
            delay_eval_set_sub_p = np.zeros((CN.NUM_INTERVALS_S))
            delay_eval_set_sub_n = np.zeros((CN.NUM_INTERVALS_S))
            #section to calculate best delay for next ir to align with this ir
            brir_sample = np.copy(sub_brir_ir[0][:])
        
            subrir_sample_p = np.copy(brir_reverberation[0,0,0,:])#check first ir, first channel
            subrir_sample_n = np.multiply(np.copy(brir_reverberation[0,0,0,:]),-1)
        
            #run once for positive polarity
            for delay in range(CN.NUM_INTERVALS_S):
            
                #shift next ir (BRIR)
                current_shift = CN.MIN_T_SHIFT_S+(delay*CN.T_SHIFT_INTERVAL_S)
                subrir_shift_c = np.roll(subrir_sample_p,current_shift)
                
                #add current ir (SUBBRIR) to shifted next ir (BRIR)
                sum_ir_c = np.add(brir_sample,subrir_shift_c)
        
                #method 5: calculate distance from peak to peak within a 400 sample window
                sum_ir_lp = hf.signal_lowpass_filter(sum_ir_c, f_crossover_var, CN.FS, CN.ORDER)
                peak_to_peak_iter=0
                for hop_id in range(CN.DELAY_WIN_HOPS):
                    samples = hop_id*CN.DELAY_WIN_HOP_SIZE
                    local_max=np.max(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+peak_to_peak_window_sub])
                    local_min=np.min(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+peak_to_peak_window_sub])
                    peak_to_peak = np.abs(local_max)#now only looking at positive peak, was np.abs(local_max-local_min)
                    #if this window has larger pk to pk, store in iter var
                    if peak_to_peak > peak_to_peak_iter:
                        peak_to_peak_iter = peak_to_peak
                #store largest pk to pk distance of all windows into delay set
                delay_eval_set_sub_p[delay] = peak_to_peak_iter
        
            peak_to_peak_max_p = np.max(delay_eval_set_sub_p[:])
            index_shift_p = np.argmax(delay_eval_set_sub_p[:])
            
            #run once for negative polarity
            for delay in range(CN.NUM_INTERVALS_S):
            
                #shift next ir (BRIR)
                current_shift = CN.MIN_T_SHIFT_S+(delay*CN.T_SHIFT_INTERVAL_S)
                subrir_shift_c = np.roll(subrir_sample_n,current_shift)
                
                #add current ir (SUBBRIR) to shifted next ir (BRIR)
                sum_ir_c = np.add(brir_sample,subrir_shift_c)
        
                #method 5: calculate distance from peak to peak within a 400 sample window
                sum_ir_lp = hf.signal_lowpass_filter(sum_ir_c, f_crossover_var, CN.FS, CN.ORDER)
                peak_to_peak_iter=0
                for hop_id in range(CN.DELAY_WIN_HOPS):
                    samples = hop_id*CN.DELAY_WIN_HOP_SIZE
                    local_max=np.max(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+peak_to_peak_window_sub])
                    local_min=np.min(sum_ir_lp[CN.DELAY_WIN_MIN_T+samples:CN.DELAY_WIN_MIN_T+samples+peak_to_peak_window_sub])
                    peak_to_peak = np.abs(local_max)
                    #if this window has larger pk to pk, store in iter var
                    if peak_to_peak > peak_to_peak_iter:
                        peak_to_peak_iter = peak_to_peak
                #store largest pk to pk distance of all windows into delay set
                delay_eval_set_sub_n[delay] = peak_to_peak_iter
        
            peak_to_peak_max_n = np.max(delay_eval_set_sub_n[:])
            index_shift_n = np.argmax(delay_eval_set_sub_n[:])
            
            if peak_to_peak_max_p > peak_to_peak_max_n or CN.EVAL_POLARITY == False or sub_set_id == 3:
                index_shift=index_shift_p
                sub_polarity=1
            else:
                index_shift=index_shift_n
                sub_polarity=-1
            
            #shift next ir by delay that has largest peak to peak distance
            samples_shift=CN.MIN_T_SHIFT_S+(index_shift*CN.T_SHIFT_INTERVAL_S)
            
        
            for chan in range(CN.TOTAL_CHAN_BRIR):
                #roll subrir
                brir_reverberation[0,set_num,chan,:] = np.roll(brir_reverberation[0,set_num,chan,:],samples_shift)
                #change polarity if applicable
                brir_reverberation[0,set_num,chan,:] = np.multiply(brir_reverberation[0,set_num,chan,:],sub_polarity)
                #set end of array to zero to remove any data shifted to end of array
                if samples_shift < 0:
                    brir_reverberation[0,set_num,chan,samples_shift:] = brir_reverberation[0,set_num,chan,samples_shift:]*0#left
                    brir_reverberation[0,set_num,chan,:] = np.multiply(brir_reverberation[0,set_num,chan,:],initial_removal_win_sub)
                #also apply fade out window
                brir_reverberation[0,set_num,chan,:] = np.multiply(brir_reverberation[0,set_num,chan,:],n_fade_out_win)
            
            
            if CN.LOG_INFO == 1:
                logging.info('delay index = ' + str(index_shift))
                logging.info('sub polarity = ' + str(sub_polarity))
                logging.info('samples_shift = ' + str(samples_shift))
                logging.info('peak_to_peak_max_n = ' + str(peak_to_peak_max_n))
                logging.info('peak_to_peak_max_p = ' + str(peak_to_peak_max_p))
        
            #copy result into subrir_sets array
            for chan in range(CN.TOTAL_CHAN_BRIR):
                subrir_sets[sub_set_id,set_num,chan,:] = np.copy(brir_reverberation[0,set_num,chan,:])
        
        ir_set = 'sub_set_average'
        #folder for saving outputs
        brir_out_folder = pjoin(CN.DATA_DIR_INT, 'ir_data', 'est_brirs',ir_set)
        
        #
        #save wavs
        #
        
        out_file_name = 'pre_eq_sub_brir_orginal.wav'
        out_file_path = pjoin(brir_out_folder,out_file_name)
        
        #create dir if doesnt exist
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        out_wav_array=np.zeros((n_fft,2))
        #grab BRIR
        out_wav_array[:,0] = np.copy(sub_brir_ir[0][:])#L
        out_wav_array[:,1] = np.copy(sub_brir_ir[1][:])#R
        
        if output_wavs == 1:
            hf.write2wav(file_name=out_file_path, data=out_wav_array, prevent_clipping=1, samplerate=samp_freq)
    
        for sub_set_id in range(num_sub_sets):
            out_file_name = str(sub_set_id) + '_pre_eq_rolled_brir.wav'
            out_file_path = pjoin(brir_out_folder,out_file_name)
            
            #create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            out_wav_array=np.zeros((n_fft,2))
            #grab BRIR
            out_wav_array[:,0] = np.copy(subrir_sets[sub_set_id,set_num,0,:])#L
            out_wav_array[:,1] = np.copy(subrir_sets[sub_set_id,set_num,1,:])#R
            
            if output_wavs == 1:
                hf.write2wav(file_name=out_file_path, data=out_wav_array, prevent_clipping=1, samplerate=samp_freq)
        
        #
        # first stage of EQ
        #
        
        for sub_set_id in range(num_sub_sets):
            sub_brir_ir_new = np.zeros((2,n_fft))
            sub_brir_ir_new[0,0:CN.N_FFT] = np.copy(subrir_sets[sub_set_id,set_num,0,:])
            sub_brir_ir_new[1,0:CN.N_FFT] = np.copy(subrir_sets[sub_set_id,set_num,1,:])
             
            #optional additional filtering
            if additional_filt == 1:
                
                if sub_set_id == 0:
                    filter_type="peaking"
                    fc=18
                    sr=samp_freq
                    q=2
                    gain_db=-1
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
                    filter_type="peaking"
                    fc=38
                    sr=samp_freq
                    q=6
                    gain_db=-3
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=43
                    sr=samp_freq
                    q=4
                    gain_db=-1.0
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=56
                    sr=samp_freq
                    q=6
                    gain_db=-5
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
             
                    filter_type="peaking"
                    fc=72
                    sr=samp_freq
                    q=4.0
                    gain_db=-1
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=127
                    sr=samp_freq
                    q=7.0
                    gain_db=-2.5
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=149
                    sr=samp_freq
                    q=4.0
                    gain_db=-0.5
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=95
                    sr=samp_freq
                    q=2.0
                    gain_db=0.5
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                elif sub_set_id == 1:
                    filter_type="peaking"
                    fc=25
                    sr=samp_freq
                    q=5
                    gain_db=-7
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=38
                    sr=samp_freq
                    q=5
                    gain_db=-2.5
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=48
                    sr=samp_freq
                    q=6
                    gain_db=-5
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=76
                    sr=samp_freq
                    q=6
                    gain_db=-2
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=109
                    sr=samp_freq
                    q=6
                    gain_db=-3
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                elif sub_set_id == 2:
                    filter_type="peaking"
                    fc=48
                    sr=samp_freq
                    q=2.5
                    gain_db=-4
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=59
                    sr=samp_freq
                    q=4
                    gain_db=-7
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=82
                    sr=samp_freq
                    q=5
                    gain_db=-5
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=111
                    sr=samp_freq
                    q=6
                    gain_db=-9
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=130
                    sr=samp_freq
                    q=7
                    gain_db=-4
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=150
                    sr=samp_freq
                    q=2.5
                    gain_db=-2
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                
                elif sub_set_id == 3:
                    
                    filter_type="peaking"
                    fc=9
                    sr=samp_freq
                    q=2.5
                    gain_db=-5
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=14
                    sr=samp_freq
                    q=10
                    gain_db=-3
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=29
                    sr=samp_freq
                    q=2.5
                    gain_db=-1
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=61
                    sr=samp_freq
                    q=2.5
                    gain_db=-1
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=143
                    sr=samp_freq
                    q=9
                    gain_db=-1
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
                    filter_type="peaking"
                    fc=198
                    sr=samp_freq
                    q=9
                    gain_db=-1
                    pyquad = pyquadfilter.PyQuadFilter(sr)
                    pyquad.set_params(filter_type, fc, q, gain_db)
                    sub_brir_ir_pre = np.copy(sub_brir_ir_new)
                    sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
                    
            #save interim result
            out_file_name = str(sub_set_id) + '_after_eq_rolled_brir.wav'
            out_file_path = pjoin(brir_out_folder,out_file_name)
            
            #create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            out_wav_array=np.zeros((n_fft,2))
            #grab BRIR
            out_wav_array[:,0] = np.copy(sub_brir_ir_new[0][:])#L
            out_wav_array[:,1] = np.copy(sub_brir_ir_new[1][:])#R
            
            if output_wavs == 1:
                hf.write2wav(file_name=out_file_path, data=out_wav_array, prevent_clipping=1, samplerate=samp_freq) 
        
            #set level of subrir set est brir to 0 at low freqs
            #set each AIR to 0 level
            data_fft = np.fft.fft(sub_brir_ir_new[0][:])
            mag_fft=np.abs(data_fft)
            average_mag_l = np.mean(mag_fft[mag_range_a:mag_range_b])
            data_fft = np.fft.fft(sub_brir_ir_new[1][:])
            mag_fft=np.abs(data_fft)
            average_mag_r = np.mean(mag_fft[mag_range_a:mag_range_b])
            average_mag=(average_mag_l+average_mag_r)/2
            if average_mag > 0:
                for chan in range(total_chan_brir):
                    out_wav_array[:,chan] = np.divide(out_wav_array[:,chan],average_mag)
        
            #copy result into subrir_sets array
            for chan in range(CN.TOTAL_CHAN_BRIR):
                subrir_sets_interim[sub_set_id,set_num,chan,:] = np.copy(sub_brir_ir_new[chan][:])
  
        #ratios for merging
        ratio_list = [0.34,0.22,0.12,0.10,0.22]#

        sub_brir_ir_new = np.zeros((2,n_fft))
        #prepopulate with reference subrir
        sub_brir_ir_new[0,0:CN.N_FFT] = np.multiply(sub_brir_ir[0][:],ratio_list[0])
        sub_brir_ir_new[1,0:CN.N_FFT] = np.multiply(sub_brir_ir[1][:],ratio_list[0])
        for sub_set_id in range(num_sub_sets):
            #merge
            for chan in range(CN.TOTAL_CHAN_BRIR):
                sub_brir_ir_new[chan,0:CN.N_FFT] = np.add(np.multiply(subrir_sets_interim[sub_set_id,set_num,chan,:],ratio_list[sub_set_id+1]),np.copy(sub_brir_ir_new[chan,0:CN.N_FFT]))

        
        
        #save interim result
        out_file_name = 'pre_eq_after_merge_brir.wav'
        out_file_path = pjoin(brir_out_folder,out_file_name)
        
        #create dir if doesnt exist
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        out_wav_array=np.zeros((n_fft,2))
        #grab BRIR
        out_wav_array[:,0] = np.copy(sub_brir_ir_new[0][:])#L
        out_wav_array[:,1] = np.copy(sub_brir_ir_new[1][:])#R
        
        if output_wavs == 1:
            hf.write2wav(file_name=out_file_path, data=out_wav_array, prevent_clipping=1, samplerate=samp_freq) 
        
        #optional additional filtering
        if additional_filt == 1:
  
            filter_type="lowshelf"
            fc=16
            sr=samp_freq
            q=1.0
            gain_db=13.5#13.6
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            
            filter_type="peaking"
            fc=55
            sr=samp_freq
            q=6
            gain_db=-0.5
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            filter_type="peaking"
            fc=76
            sr=samp_freq
            q=6
            gain_db=-0.8
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            filter_type="peaking"
            fc=70
            sr=samp_freq
            q=2
            gain_db=-0.4
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            filter_type="peaking"
            fc=112
            sr=samp_freq
            q=8
            gain_db=-0.4
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            filter_type="peaking"
            fc=146
            sr=samp_freq
            q=8
            gain_db=-0.4
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
  
            filter_type="peaking"
            fc=113
            sr=samp_freq
            q=7.5
            gain_db=-0.4
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            filter_type="peaking"
            fc=133
            sr=samp_freq
            q=10.0
            gain_db=-0.2
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            filter_type="peaking"
            fc=144
            sr=samp_freq
            q=10.0
            gain_db=-0.5
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            filter_type="peaking"
            fc=162
            sr=samp_freq
            q=10.0
            gain_db=-0.25
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            filter_type="peaking"
            fc=38
            sr=samp_freq
            q=6
            gain_db=0.7
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            filter_type="peaking"
            fc=83
            sr=samp_freq
            q=6
            gain_db=0.2
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            #
            filter_type="peaking"
            fc=4
            sr=samp_freq
            q=2.5#3.0
            gain_db=-4.0
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            filter_type="peaking"
            fc=6
            sr=samp_freq
            q=3.0#3.5
            gain_db=-7.0
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            #
            filter_type="peaking"
            fc=3
            sr=samp_freq
            q=2.0#2.5
            gain_db=-3.0#-2.0
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
  
            #
            filter_type="peaking"
            fc=14
            sr=samp_freq
            q=2.5#3.0
            gain_db=-1.5
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            filter_type="peaking"
            fc=20
            sr=samp_freq
            q=5#3.0
            gain_db=-0.8
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)
            
            filter_type="peaking"
            fc=16
            sr=samp_freq
            q=6.0#
            gain_db=-1.0
            pyquad = pyquadfilter.PyQuadFilter(sr)
            pyquad.set_params(filter_type, fc, q, gain_db)
            sub_brir_ir_pre = np.copy(sub_brir_ir_new)
            sub_brir_ir_new = pyquad.filter(sub_brir_ir_pre)

            if CN.LOG_INFO == 1:
                logging.info('filter_type = ' + str(filter_type))
                logging.info('fc = ' + str(fc))
                logging.info('sr = ' + str(sr))
                logging.info('q = ' + str(q))
                logging.info('gain_db = ' + str(gain_db))
        
    
        #final windowing
        sub_brir_ir_new[0][:] = np.multiply(sub_brir_ir_new[0][:],f_fade_out_win)#L
        sub_brir_ir_new[1][:] = np.multiply(sub_brir_ir_new[1][:],f_fade_out_win)#R
    
        #
        #export wavs for testing
        #

        
        out_file_name = 'after_eq_sub_brir_new.wav'
        out_file_path = pjoin(brir_out_folder,out_file_name)
        
        #create dir if doesnt exist
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        out_wav_array=np.zeros((n_fft,2))
        #grab BRIR
        out_wav_array[:,0] = np.copy(sub_brir_ir_new[0][:])#L
        out_wav_array[:,1] = np.copy(sub_brir_ir_new[1][:])#R
        
        if output_wavs == 1:
            hf.write2wav(file_name=out_file_path, data=out_wav_array, prevent_clipping=1, samplerate=samp_freq)

        
        #
        #save numpy array for later use in BRIR generation functions
        #
        npy_file_name =  'sub_brir_new.npy'
        
        out_file_path = pjoin(brir_out_folder,npy_file_name)      
          
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        np.save(out_file_path,sub_brir_ir_new)    
        
        log_string_a = 'Exported numpy file to: ' + out_file_path 
        if CN.LOG_INFO == 1:
            logging.info(log_string_a)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string_a)
    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to complete sub BRIR processing'
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_error(log_string)
    
    

def calc_room_target_dataset(gui_logger=None):
    """
    function to calculate average magnitude response of various rooms
    :return: None
    """ 
    
    try:
        n_fft=CN.N_FFT
        #impulse
        impulse=np.zeros(n_fft)
        impulse[0]=1
        fr_flat_mag = np.abs(np.fft.fft(impulse))
        fr_flat = hf.mag2db(fr_flat_mag)
    
        log_string_a = 'Room target response calculation running'
        if CN.LOG_INFO == 1:
            logging.info(log_string_a)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string_a)
    
        #create new array for old and new targets
        num_targets=len(CN.ROOM_TARGET_LIST_FIRS)-1
        num_targets_new=num_targets*3
        fir_length=4096
        room_target_fir_arr=np.zeros((num_targets_new,fir_length))
        
        mag_range_a=1200
        mag_range_b=1800
        #load existing room targets from .mat
        #
        # load room target filter (FIR)
        
        mat_fname = pjoin(CN.DATA_DIR_INT, 'room_targets_firs.mat')
        room_target_mat = mat73.loadmat(mat_fname)
        
        target_id=0
        for room_target in CN.ROOM_TARGET_LIST_FIRS:
            if room_target != 'Flat':
                #load fir from mat struct
                room_target_fir = room_target_mat[room_target]
                
                #normalise to 0db at 1khz
                room_target_fir_t=np.zeros(CN.N_FFT)
                room_target_fir_t[0:fir_length] = np.copy(room_target_fir[0:fir_length])
                data_fft = np.fft.fft(room_target_fir_t)
                mag_fft=np.abs(data_fft)
                average_mag = np.mean(mag_fft[mag_range_a:mag_range_b])
                room_target_fir[0:fir_length] = np.divide(room_target_fir[0:fir_length],average_mag)

                #smooth spectrum
                room_target_fir_t=np.zeros(CN.N_FFT)
                room_target_fir_t[0:fir_length] = np.copy(room_target_fir[0:fir_length])
                data_fft = np.fft.fft(room_target_fir_t)
                room_target_mag=np.abs(data_fft)
                #level ends of spectrum
                room_target_mag_new = hf.level_spectrum_ends(room_target_mag, 20, 20000, smooth_win = 3)#150
                room_target_mag_new_sm = hf.smooth_fft_octaves(data=room_target_mag_new, n_fft=n_fft,fund_freq=150,win_size_base = 3)
                #create min phase FIR
                room_target_min_fir = hf.mag_to_min_fir(room_target_mag_new_sm, crop=1)


                #save into numpy array
                room_target_fir_arr[target_id,:] = np.copy(room_target_min_fir[0:fir_length])
                
                target_id=target_id+1
    
        #calculate new set with modified targets that are flat > 1000Hz
        for idx in range(num_targets):
            room_target_fir=np.zeros(CN.N_FFT)
            room_target_fir[0:fir_length] = np.copy(room_target_fir_arr[idx,:])
            data_fft = np.fft.fft(room_target_fir)
            room_target_mag=np.abs(data_fft)
            #level ends of spectrum
            room_target_mag_new = hf.level_spectrum_ends(room_target_mag, 20, 1000, smooth_win = 3)#150
            room_target_mag_new_sm = hf.smooth_fft(data=room_target_mag_new, n_fft=n_fft, crossover_f=2000, win_size_a = 3, win_size_b = 500)
            
            #create min phase FIR
            room_target_min_fir = hf.mag_to_min_fir(room_target_mag_new_sm, crop=1)
            
            #store into numpy array
            new_id=num_targets+idx
            room_target_fir_arr[new_id,:] = np.copy(room_target_min_fir[0:fir_length])
            
            if CN.PLOT_ENABLE == 1:
                print(str(idx))
                hf.plot_data(room_target_mag_new,'room_target_mag_new_'+str(idx), normalise=0)
                hf.plot_data(room_target_mag_new_sm,'room_target_mag_new_sm_'+str(idx), normalise=0)
                plot_name = 'room_target_min_fir_'+str(idx)
                hf.plot_td(room_target_min_fir[0:fir_length],plot_name)
            
        #calculate new set with modified targets that are flat > 5000Hz
        for idx in range(num_targets):
            room_target_fir=np.zeros(CN.N_FFT)
            room_target_fir[0:fir_length] = room_target_fir_arr[idx,:]
            data_fft = np.fft.fft(room_target_fir)
            room_target_mag=np.abs(data_fft)
            #level ends of spectrum
            room_target_mag_new = hf.level_spectrum_ends(room_target_mag, 20, 5000, smooth_win = 3)#150
            room_target_mag_new_sm = hf.smooth_fft(data=room_target_mag_new, n_fft=n_fft, crossover_f=2000, win_size_a = 3, win_size_b = 500)
            
            #create min phase FIR
            room_target_min_fir = hf.mag_to_min_fir(room_target_mag_new_sm, crop=1)
            
            #store into numpy array
            new_id=num_targets*2+idx
            room_target_fir_arr[new_id,:] = np.copy(room_target_min_fir[0:fir_length])
            
            if CN.PLOT_ENABLE == 1:
                print(str(idx))
                hf.plot_data(room_target_mag_new,'room_target_mag_new_'+str(idx), normalise=0)
                hf.plot_data(room_target_mag_new_sm,'room_target_mag_new_sm_'+str(idx), normalise=0)
                plot_name = 'room_target_min_fir_'+str(idx)
                hf.plot_td(room_target_min_fir[0:fir_length],plot_name)
            
      
        #create flat target and insert into start of numpy array
        #impulse
        impulse=np.zeros(fir_length)
        impulse[0]=1
        room_target_fir_arr_i=np.zeros((1,fir_length))
        room_target_fir_arr_i[0,:] = np.copy(impulse[0:fir_length])
        room_target_fir_arr_out=np.append(room_target_fir_arr_i,room_target_fir_arr,0)
        
    
        #
        #save numpy array for later use in BRIR generation functions
        #
        npy_file_name =  'room_targets_firs.npy'
        data_out_folder = CN.DATA_DIR_INT
        out_file_path = pjoin(data_out_folder,npy_file_name)      
          
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        np.save(out_file_path,room_target_fir_arr_out)    
        
        log_string_a = 'Exported numpy file to: ' + out_file_path 
        if CN.LOG_INFO == 1:
            logging.info(log_string_a)
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string_a)
        
        
    
    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to complete room target response processing'
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_error(log_string)
            
                