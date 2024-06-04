# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 14:02:51 2023

@author: Shanon
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
import soundfile as sf
from scipy.signal import butter
import scipy as sp
import datetime
from ash_toolset import constants as CN
from os.path import join as pjoin
import dearpygui.dearpygui as dpg

def print_message(message):
    """
    Function to print a message
    """
    current_time = datetime.datetime.now()
    print(current_time, ': ', message)


def round_to_multiple(number, multiple):
    """
    function to round to a multiple of a base
    """
    return multiple * round(number / multiple)


def plot_data(mag_response, title_name = 'Output', n_fft = 65536, samp_freq = 44100, y_lim_adjust = 0, save_plot=0, plot_path=CN.DATA_DIR_OUTPUT, normalise=1, plot_type=0):
    """
    Function takes a magnitude reponse array as an input and plots the reponse
    :param mag_response: numpy array 1d, magnitude reponse array
    :param title_name: string, name of plot
    :param n_fft: int, fft size
    :param samp_freq: int, sample frequency in Hz
    :param y_lim_adjust: int, 1 = adjust y axis to 30db range, 0 = no adjustment 
    :param save_plot: int, 1 = save plot to file, 0 = dont save plot
    :param plot_path: string, path to save plot 
    :param normalise: int, 1 = normalise low frequencies to 0db
    :param plot_type: int, 0 = matplotlib, 1 = dearpygui
    :return: None
    """

    nUniquePts = int(np.ceil((n_fft+1)/2.0))
    sampling_ratio = samp_freq/n_fft
    freqArray = np.arange(0, nUniquePts, 1.0) * sampling_ratio    
    
    #mag_response = mag_response / float(n_fft)
    mag_response = mag_response[0:nUniquePts]
    mag_response_log = 20*np.log10(mag_response)
    
    if normalise == 1:
        mag_response_log = mag_response_log-np.mean(mag_response_log[0:200])
    elif normalise == 2:
        mag_response_log = mag_response_log-np.mean(mag_response_log[1000:1300])
    
    if plot_type == 0:
        plt.figure()
        plt.plot(freqArray, mag_response_log, color='k', label="FR")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.xscale("log")
        plt.grid()
        plt.xlim([20, 20000])
        if y_lim_adjust == 1:
            plt.ylim([np.ceil(mag_response_log.max())-25, np.ceil(mag_response_log.max())+5])
        plt.title(title_name)
        plt.show()
        
        if save_plot == 1:
            out_file_name = title_name + '.png'
            out_file_path = pjoin(plot_path, out_file_name)
            plt.savefig(out_file_path)
    elif plot_type == 1:
        dpg.set_value('series_tag', [freqArray, mag_response_log])
        dpg.set_item_label('series_tag', title_name)



def plot_geq(geq_dict, title_name = 'Output', y_lim_adjust = 0, save_plot=0, plot_path=CN.DATA_DIR_OUTPUT):
    """
    Function takes a magnitude reponse dictionary as an input and plots the reponse
    :param geq_dict: dictionary, graphic eq dictionary with magnitude reponse
    :param title_name: string, name of plot
    :param y_lim_adjust: int, 1 = adjust y axis to 30db range, 0 = no adjustment 
    :param save_plot: int, 1 = save plot to file, 0 = dont save plot
    :param plot_path: string, path to save plot 
    :return: None
    """   
    
   
    # split dictionary into keys and values
    keys = list(geq_dict.keys())
    keys_float = [float(i) for i in keys]
    values = np.array(list(geq_dict.values()))
    
    plt.figure()
    plt.plot(keys_float, values, color='k', label="FR")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xscale("log")
    plt.grid()
    plt.xlim([20, 20000])
    if y_lim_adjust == 1:
        plt.ylim([np.ceil(values.max())-25, np.ceil(values.max())+5])
    plt.title(title_name)
    plt.show()
    
    if save_plot == 1:
        out_file_name = title_name + '.png'
        out_file_path = pjoin(plot_path, out_file_name)
        plt.savefig(out_file_path)
    

def plot_td(signal, title_name = 'Output', samp_freq = 44100):
    """
    Function takes a time domain signal as an input and plots the reponse
    :param signal: numpy array, time domain signal
    :param title_name: string, name of plot
    :param samp_freq: int, sample frequency in Hz
    :return: None
    """       

    Time = np.linspace(0, len(signal) / samp_freq, num=len(signal))

    plt.figure()
    plt.plot(Time, signal, color='k', label="amplitude")
    plt.xlabel("Time (S)")
    plt.ylabel("Amplitude")
    plt.grid()
    #plt.xlim([20, 20000])
    #plt.ylim([np.ceil(mag_response_log.max())-20, np.ceil(mag_response_log.max())+5])
    plt.title(title_name)
    plt.show()


def mag2db(mag):
    """Convert a magnitude to decibels (dB).

    If A is magnitude,

        db = 20 * log10(A)

    Parameters
    ----------
    mag : float or ndarray
        input magnitude or array of magnitudes

    Returns
    -------
    db : float or ndarray
        corresponding values in decibels

    Examples
    --------
    >>> ct.mag2db(10.0)                                         # doctest: +SKIP
    20.0

    >>> ct.mag2db(np.array([1, 0.01]))                          # doctest: +SKIP
    array([  0., -40.])

    """
    return 20. * np.log10(mag)

def db2mag(db):
    """Convert a gain in decibels (dB) to a magnitude.

    If A is magnitude,

        db = 20 * log10(A)

    Parameters
    ----------
    db : float or ndarray
        input value or array of values, given in decibels

    Returns
    -------
    mag : float or ndarray
        corresponding magnitudes

    Examples
    --------
    >>> ct.db2mag(-40.0)                                        # doctest: +SKIP
    0.01

    >>> ct.db2mag(np.array([0, -20]))                           # doctest: +SKIP
    array([1. , 0.1])

    """
    return 10. ** (db / 20.)



def list_diff(list1,list2):
    """
    function to get difference of 2 lists
    """
    
    result = []

    for i in range(len(list1)):
        result.append(list1[i] - list2[i])
        
    return result
  
# function to write wav file 
def write2wav(file_name, data, samplerate = 44100, prevent_clipping = 0):
    """
    Function takes a time domain signal as an input and writes wav file 
    :param data: numpy array, time domain signal
    :param file_name: string, name of wav
    :param samplerate: int, sample frequency in Hz
    :param prevent_clipping: int, 1 = reduce amplitude to prevent clipping
    :return: None
    """    
    
    #old method using scipy
    #write("example.wav", samplerate, data.astype(np.int32))
    
    #adjust gain
    if prevent_clipping == 1:
        max_amp = np.max(np.abs(data))
        if max_amp > 1:
            data = data/max_amp
    
    #new method using PySoundFile 
    #soundfile expects data in frames x channels, or one-dimensional data for mono files. librosa does it the other way around.
    sf.write(file_name, data, samplerate, 'PCM_24')
    
    
def resample_signal(clip, samplerate = 44100, new_rate = 48000):
    """
    function to resample a signal. By default will upsample from 44100Hz to 48000Hz
    """  
    
     # Resample data
    number_of_samples = round(len(clip) * float(new_rate) / samplerate)
    clip = sps.resample(clip, number_of_samples) 
    return clip
        
 # ==============================================================================
# Measure RT60
# sourced from: https://dsp.stackexchange.com/questions/86316/a-python-code-for-blind-estimation-of-rt60-from-recorded-audio
# ==============================================================================
def measure_rt60(h, fs=1, decay_db=60, plot=False, rt60_tgt=None):
    """
    Analyze the RT60 of an impulse response. Optionaly plots some useful information.

    Parameters
    ----------
    `h`: array_like
        The impulse response.
    `fs`: float or int, optional
        The sampling frequency of h (default to 1, i.e., samples).
    `decay_db`: float or int, optional
        The decay in decibels for which we actually estimate the time. Although
        we would like to estimate the RT60, it might not be practical. Instead,
        we measure the RT20 or RT30 and extrapolate to RT60.
    `plot`: bool, optional
        If set to ``True``, the power decay and different estimated values will
        be plotted (default False).
    `rt60_tgt`: float
        This parameter can be used to indicate a target RT60 to which we want
        to compare the estimated value.
    """

    h = np.array(h)
    fs = float(fs)

    # The power of the impulse response in dB
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    # remove the possibly all zero tail
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]

    # -5 dB headroom
    i_5db = np.min(np.where(-5 - energy_db > 0)[0])
    e_5db = energy_db[i_5db]
    t_5db = i_5db / fs

    # after decay
    i_decay = np.min(np.where(-5 - decay_db - energy_db > 0)[0])
    t_decay = i_decay / fs

    # compute the decay time
    decay_time = t_decay - t_5db
    est_rt60 = (60 / decay_db) * decay_time

    if plot:
        import matplotlib.pyplot as plt

        # Remove clip power below to minimum energy (for plotting purpose mostly)
        energy_min = energy[-1]
        energy_db_min = energy_db[-1]
        power[power < energy[-1]] = energy_min
        power_db = 10 * np.log10(power)
        power_db -= np.max(power_db)

        # time vector
        def get_time(x, fs):
            return np.arange(x.shape[0]) / fs - i_5db / fs

        T = get_time(power_db, fs)

        # plot power and energy
        plt.plot(get_time(energy_db, fs), energy_db, label="Energy")

        # now the linear fit
        plt.plot([0, est_rt60], [e_5db, -65], "--", label="Linear Fit")
        plt.plot(T, np.ones_like(T) * -60, "--", label="-60 dB")
        plt.vlines(
            est_rt60, energy_db_min, 0, linestyles="dashed", label="Estimated RT60"
        )

        if rt60_tgt is not None:
            plt.vlines(rt60_tgt, energy_db_min, 0, label="Target RT60")

        plt.legend()

        plt.show()

    return est_rt60



def signal_lowpass_filter(data, cutoff, fs, order=5, method=1):
    """
    Function takes a time domain signal as an input and applies low pass filter
    :param data: numpy array, time domain signal
    :param cutoff: int, cutoff frequency in Hz
    :param fs: int, sample frequency in Hz
    :param order: int, filter order
    :param method: int, 1=butter, 2=chevy1, 3=cheby2, 4=ellip, 5=bessel
    :return: numpy array, low pass filtered signal
    """   
    #Say I have a digital butterworth lowpass filter with an order of 3 and a cutoff of 4Hz at -3dB. this filter should have a roll-off of 6*3 = 18dB/Octave.
    #48db/octave = order of 8?
    
    if method == 1:
        #method 1
        sos = butter(order, cutoff, fs=fs, btype='low', output='sos', analog=False)
        #A forward-backward digital filter using cascaded second-order sections.
        y = sps.sosfilt(sos, data)
    elif method == 2:
        #method 2
        sos = sps.cheby1(order, 3, cutoff, btype='low', analog=False, output='sos', fs=fs)
        y = sps.sosfilt(sos, data)
    elif method == 3:
        #method 3
        sos = sps.cheby2(order, 40, cutoff, btype='low', analog=False, output='sos', fs=fs)
        y = sps.sosfilt(sos, data)
    elif method == 4:
        #method 4
        sos = sps.ellip(order, 5, 40, cutoff, btype='low', analog=False, output='sos', fs=fs)
        y = sps.sosfilt(sos, data)  
    else:
        #method 5
        sos = sps.bessel(order, cutoff, btype='low', analog=False, output='sos', fs=fs)
        y = sps.sosfilt(sos, data)  
        
        
    return y


def signal_highpass_filter(data, cutoff, fs, order=5, method=1):
    """
    Function takes a time domain signal as an input and applies high pass filter
    :param data: numpy array, time domain signal
    :param cutoff: int, cutoff frequency in Hz
    :param fs: int, sample frequency in Hz
    :param order: int, filter order
    :param method: int, 1=butter, 2=chevy1, 3=cheby2, 4=ellip, 5=bessel
    :return: numpy array, high pass filtered signal
    """  
    
    if method == 1:
        #method 1
        sos = butter(order, cutoff, fs=fs, btype='high', output='sos', analog=False)
        #A forward-backward digital filter using cascaded second-order sections.
        y = sps.sosfilt(sos, data)
    elif method == 2:
        #method 2
        sos = sps.cheby1(order, 3, cutoff, btype='high', analog=False, output='sos', fs=fs)
        y = sps.sosfilt(sos, data)
    elif method == 3:
        #method 3
        sos = sps.cheby2(order, 40, cutoff, btype='high', analog=False, output='sos', fs=fs)
        y = sps.sosfilt(sos, data)
    elif method == 4:
        #method 4
        sos = sps.ellip(order, 5, 40, cutoff, btype='high', analog=False, output='sos', fs=fs)
        y = sps.sosfilt(sos, data)
    else:
        #method 5
        sos = sps.bessel(order, cutoff, btype='high', analog=False, output='sos', fs=fs)
        y = sps.sosfilt(sos, data) 
    
    return y




def group_delay(sig):
    """
    function to calculate group delay
    """  
    
    b = np.fft.fft(sig)
    n_sig = np.multiply(sig, np.arange(len(sig)))
    br = np.fft.fft(n_sig)
    return np.divide(br, b + 0.01).real


def smooth_fft(data, crossover_fb=1500, win_size_a = 150, win_size_b = 750, n_fft=65536):
    """
    Function to perform smoothing of fft mag response
    :param data: numpy array, magnitude response of a signal
    :param crossover_fb: int, crossover frequency in Hz. Below this freq a smoothing window of win_size_a will be applied and win_size_b above this freq
    :param win_size_a: int, smoothing window size in Hz for lower frequencies
    :param win_size_b: int, smoothing window size in Hz for higher frequencies
    :param n_fft: int, fft size
    :return: numpy array, smoothed signal
    """  
    
    #yhat = sp.signal.savgol_filter(data, 20, 3)
    
    n_unique_pts = int(np.ceil((n_fft+1)/2.0))
    nyq_freq = n_unique_pts-1
    #apply win size a to low frequencies
    data_smooth_a = sp.ndimage.uniform_filter1d(data,size=win_size_a)
    data_smooth_b =np.zeros(n_fft)
    data_smooth_b[0:crossover_fb] = data_smooth_a[0:crossover_fb]
    #apply win size b to high frequencies
    data_smooth_b[crossover_fb:n_unique_pts] = sp.ndimage.uniform_filter1d(data_smooth_a,size=win_size_b)[crossover_fb:n_unique_pts]
    data_smooth_c = sp.ndimage.uniform_filter1d(data_smooth_b,size=10)
    
    #make conjugate symmetric
    for freq in range(n_fft):
        if freq>nyq_freq:
            dist_from_nyq = np.abs(freq-nyq_freq)
            data_smooth_c[freq]=data_smooth_c[nyq_freq-dist_from_nyq]
    
    return data_smooth_c


def mag_to_min_fir(data, n_fft=65536, out_win_size=4096):
    """
    Function to create min phase FIR from a fft mag response
    :param data: numpy array, magnitude response of a signal
    :param out_win_size: int, number of samples desired in output signal. Will crop signal
    :param n_fft: int, fft size
    :return: numpy array, time domain signal
    """  
    
    n_unique_pts = int(np.ceil((n_fft+1)/2.0))

    data_pad_zeros=np.zeros(n_fft)
    data_pad_ones=np.ones(n_fft)

    #create min phase FIR 
    data_ifft = np.fft.irfft(data[0:n_unique_pts])#zero phase and symmetric
    data_lin = np.fft.ifftshift(data_ifft)#linear phase
    #filter will have a magnitude response that approximates the square root of the original filter’s magnitude response.
    data_min = sp.signal.minimum_phase(data_lin[1:n_fft], 'homomorphic', n_fft)
    data_min_pad = data_pad_zeros.copy()
    #return to original mag response
    data_min_conv = sp.signal.convolve(data_min,data_min, 'full', 'direct')
    data_min_pad[0:n_fft-1] = data_min_conv[:]
    
    #apply window to result

    #fade out window
    fade_hanning_size=out_win_size*2
    fade_hanning_start=50
    hann_fade_full=np.hanning(fade_hanning_size)
    hann_fade = np.split(hann_fade_full,2)[1]
    fade_out_win = data_pad_ones.copy()
    fade_out_win[fade_hanning_start:fade_hanning_start+int(fade_hanning_size/2)] = hann_fade
    fade_out_win[fade_hanning_start+int(fade_hanning_size/2):]=data_pad_zeros[fade_hanning_start+int(fade_hanning_size/2):]

    #return result
    #result with original length and windowed
    data_out = np.multiply(data_min_pad[0:n_fft-1],fade_out_win[0:n_fft-1])
    return data_out
    
#modify spectrum to have flat mag response at low and high ends
def level_spectrum_ends(data, low_freq=20, high_freq=20000, n_fft=65536, fs=44100, smooth_win = 100):
    """
    Function to modify spectrum to have flat mag response at low and high ends
    :param data: numpy array, magnitude response of a signal
    :param low_freq: int, frequency in Hz below which will become flat
    :param high_freq: int, frequency in Hz above which will become flat
    :param n_fft: int, fft size
    :param fs: int, sample frequency in Hz
    :param smooth_win: int, smoothing window size in Hz to be applied after leveling ends
    :return: numpy array, spectrum with smooth ends
    """     
    n_unique_pts = int(np.ceil((n_fft+1)/2.0))
    nyq_freq = n_unique_pts-1
    
    data_mod = data.copy()
    low_freq_bin = int(low_freq*n_fft/fs)
    high_freq_bin = int(high_freq*n_fft/fs)
    data_mod[0:low_freq_bin] = data[low_freq_bin]
    data_mod[high_freq_bin:n_fft] = data[high_freq_bin]
    
    #apply slight smoothing
    data_smooth = sp.ndimage.uniform_filter1d(data_mod,size=smooth_win)
    
    #make conjugate symmetric
    for freq in range(n_fft):
        if freq>nyq_freq:
            dist_from_nyq = np.abs(freq-nyq_freq)
            data_smooth[freq]=data_smooth[nyq_freq-dist_from_nyq]
            
    return data_smooth


def padarray(A, size):
    """
    function to pad a numpy array with zeros to specified size
    """  
    #numpy.pad with constant mode, pass a tuple as second argument to tell how many zeros to pad on each size, a (2, 3) for instance will pad 2 zeros on the left side and 3 zeros on the right side:
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')
    