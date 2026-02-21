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
import librosa
import os
import re
import requests
import logging
from SOFASonix import SOFAFile
import sofar as sof
import csv
import random
import math
from scipy.signal import savgol_filter, medfilt, correlate
from scipy.ndimage import gaussian_filter1d
import threading
import time
import scipy.signal as signal
from scipy.stats import linregress
from os.path import exists
import concurrent.futures
import ast
import json
from pathlib import Path
from scipy.signal import resample_poly
from urllib.parse import urlsplit, urlunsplit, quote
from numpy import unwrap, diff
from scipy.signal import  windows
from numpy.fft import rfft, rfftfreq  
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import interp1d
import platform
import queue
from scipy.optimize import minimize

IS_WINDOWS = platform.system() == "Windows"
# Try to import sounddevice safely
try:
    import sounddevice as sd
    SD_AVAILABLE = True
except Exception:
    SD_AVAILABLE = False


def plot_data(
    mag_response,
    title_name='Output',
    n_fft=65536,
    samp_freq=44100,
    y_lim_adjust=0,
    y_lim_a=-25,
    y_lim_b=15,
    x_lim_adjust=0,
    x_lim_a=20,
    x_lim_b=20000,
    save_plot=0,
    plot_path=CN.DATA_DIR_OUTPUT,
    normalise=1,
    level_ends=0,
    plot_dest=0,  # kept for compatibility, ignored
):
    """
    Matplotlib-only plotting helper for debugging.

    This function previously supported DearPyGui plotting via plot_dest.
    plot_dest is now ignored but retained for backward compatibility.
    """

    # -----------------------------
    # Pre-processing
    # -----------------------------
    if level_ends == 1:
        mag_response = level_spectrum_ends(
            mag_response, 200, 19000, n_fft=n_fft
        )
        # mag_response = smooth_freq_octaves(
        #     data=mag_response, n_fft=n_fft
        # )
        mag_response = smooth_gaussian_octave(data=mag_response, n_fft=n_fft, fs=samp_freq, fraction=24)

    n_unique_pts = int(np.ceil((n_fft + 1) / 2.0))
    freq_array = np.arange(0, n_unique_pts) * (samp_freq / n_fft)

    mag_response = mag_response[:n_unique_pts]
    mag_response_log = 20 * np.log10(np.maximum(mag_response, 1e-12))

    # -----------------------------
    # Normalisation
    # -----------------------------
    if normalise == 1:
        mag_response_log -= np.mean(mag_response_log[:200])
    elif normalise == 2:
        mag_response_log -= np.mean(
            mag_response_log[
                CN.SPECT_SNAP_M_F0 : CN.SPECT_SNAP_M_F1
            ]
        )

    # -----------------------------
    # Matplotlib plotting
    # -----------------------------
    plt.figure()
    plt.plot(freq_array, mag_response_log, label="FR")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    if x_lim_adjust:
        plt.xlim(x_lim_a, x_lim_b)
    else:
        plt.xlim(20, 20000)

    if y_lim_adjust:
        plt.ylim(y_lim_a, y_lim_b)

    plt.title(title_name)
    plt.tight_layout()

    if save_plot == 1:
        out_file_name = f"{title_name}.png"
        out_file_path = pjoin(plot_path, out_file_name)
        plt.savefig(out_file_path, dpi=150)

    plt.show()

     

    
def _resolve_plot_tags(plot_dest):
    tag_map = {
        2: 'fde_series_tag',
        1: 'series_tag',
        3: 'ia_series_tag'
    }
    x_axis_map = {
        2: 'fde_x_axis',
        1: 'x_axis',
        3: 'ia_x_axis'
    }
    y_axis_map = {
        2: 'fde_y_axis',
        1: 'y_axis',
        3: 'ia_y_axis'
    }

    return (
        tag_map.get(plot_dest),
        x_axis_map.get(plot_dest),
        y_axis_map.get(plot_dest)
    )

def _extract_fir(fir_array):
    fir = np.asarray(fir_array)
    while fir.ndim > 1:
        fir = fir[0]
    return fir.astype(np.float64)

def plot_fir_generic(
    fir_array,
    view="magnitude",
    title_name="Output",
    samp_freq=CN.SAMP_FREQ,
    n_fft=CN.N_FFT,
    normalise=1,
    level_ends=0,
    decay_start_ms=0.0,
    decay_end_ms=1000.0,
    decay_step_ms=100.0,
    decay_win_ms=100.0,
    plot_dest=1,
    x_lim_adjust=0, x_lim_a=20, x_lim_b=20000,
    y_lim_adjust=0, y_lim_a=-25, y_lim_b=15,smooth_win_base=3
):
    """
    Plot a FIR using DearPyGui in multiple common views.
    
    Supports magnitude response, impulse response, group delay, and decay plots.
    The FIR may have any number of dimensions; only the first item is used and
    the last dimension is assumed to contain samples.
    
    Parameters
    ----------
    fir_array : np.ndarray
        FIR array with samples along the last dimension.
    
    view : {"Magnitude Response", "Impulse Response", "Group Delay", "Decay"}
        Plot type to render.
    
    samp_freq : int or float
        Sample rate in Hz.
    
    n_fft : int
        FFT size for frequency-domain plots.
    
    normalise : int
        Magnitude plot only. Normalise response if non-zero.
    
    level_ends : int
        Magnitude plot only. Apply end-level conditioning.
    
    decay_start_ms, decay_end_ms, decay_step_ms, decay_win_ms : float
        Decay plot only. Control time range and resolution of decay slices.
    
    plot_dest : int
        Selects which DearPyGui plot instance to update.
    
    x_lim_adjust, x_lim_a, x_lim_b :
        Optional manual x-axis limits.
    
    y_lim_adjust, y_lim_a, y_lim_b :
        Optional manual y-axis limits.
    """

    try:
        fir = _extract_fir(fir_array)
        
        # Always zero-pad FIR if n_fft is larger
        if n_fft > len(fir):
            padded = np.zeros(n_fft, dtype=fir.dtype)
            padded[:len(fir)] = fir
            fir = padded
    
        series_tag, x_axis_tag, y_axis_tag = _resolve_plot_tags(plot_dest)
        if series_tag is None:
            return
        
        #update user data dict with plot information, so that it can be regenerated when plot type changes
        if plot_dest == CN.TAB_QC_CODE:
            plot_type_tag = "plot_type"
        elif plot_dest == CN.TAB_QC_IA_CODE:
            plot_type_tag = "ia_plot_type"
        else:
            return
        
        plot_state = dpg.get_item_user_data(plot_type_tag)
        if plot_state is None:
            plot_state = {}
        plot_state.clear()
        plot_state.update(dict(
            fir=fir,
            title=title_name,
            samp_freq=samp_freq,
            n_fft=n_fft,
            normalise=normalise,
            level_ends=level_ends,
            decay_start_ms=decay_start_ms,
            decay_end_ms=decay_end_ms,
            decay_step_ms=decay_step_ms,
            decay_win_ms=decay_win_ms,
            x_lim_adjust=x_lim_adjust,
            x_lim_a=x_lim_a,
            x_lim_b=x_lim_b,
            plot_dest=plot_dest,
            y_lim_adjust=y_lim_adjust,
            y_lim_a=y_lim_a,
            y_lim_b=y_lim_b,
        ))
    
        # Clear existing plot data
        dpg.set_value(series_tag, [[], []])
        dpg.set_item_label(series_tag, title_name)
    
        if view == "Magnitude Response" or view == "Summary Response":
            _plot_magnitude_dpg(
                fir, samp_freq, n_fft,
                series_tag, x_axis_tag, y_axis_tag,
                normalise, level_ends,
                x_lim_adjust, x_lim_a, x_lim_b,
                y_lim_adjust, y_lim_a, y_lim_b,smooth_win_base=smooth_win_base
            )
    
        elif view == "Impulse Response":
            _plot_impulse_dpg(
                fir, samp_freq,
                series_tag, x_axis_tag, y_axis_tag,
                x_lim_adjust, x_lim_a, x_lim_b,
                y_lim_adjust, y_lim_a, y_lim_b
            )
    
        elif view == "Group Delay":
            _plot_group_delay_dpg(
                fir, samp_freq, n_fft,
                series_tag, x_axis_tag, y_axis_tag,
                x_lim_adjust, x_lim_a, x_lim_b,
                y_lim_adjust, y_lim_a, y_lim_b
            )
    
        elif view == "Decay":
            _plot_decay_dpg(
                fir, samp_freq, n_fft,
                series_tag, x_axis_tag, y_axis_tag,
                decay_start_ms, decay_end_ms,
                decay_step_ms, decay_win_ms,
                x_lim_adjust, x_lim_a, x_lim_b,
                y_lim_adjust, y_lim_a, y_lim_b
            )
    
        else:
            raise ValueError(f"Unsupported view: {view}")
            
    except Exception as ex:
        log_with_timestamp(f"Failed to generate plot: {ex}", log_type=2, exception=ex)

def _plot_magnitude_dpg(
    fir, samp_freq, n_fft,
    series_tag, x_axis_tag, y_axis_tag,
    normalise, level_ends,
    x_lim_adjust, x_lim_a, x_lim_b,
    y_lim_adjust, y_lim_a, y_lim_b,smooth_win_base=3
):
    
    _clear_decay_children(y_axis_tag, series_tag)
    n_fft = max(n_fft, len(fir))
    F = np.fft.rfft(fir, n=n_fft)
    mag = np.abs(F)

    if level_ends:
        mag = level_spectrum_ends(mag, 240, 19000, n_fft=n_fft)
    if x_lim_b > 200:
        #mag = smooth_freq_octaves(mag, n_fft=n_fft,fund_freq=100,win_size_base=smooth_win_base)
        mag = smooth_gaussian_octave(data=mag, n_fft=n_fft, fs=samp_freq, fraction=12)

    mag_db = 20 * np.log10(np.maximum(mag, 1e-12))

    if normalise == 1:#low frequencies
        mag_db -= np.mean(mag_db[:200])
    elif normalise == 2:#mid frequencies
        mag_db -= np.mean(mag_db[700:1300])

    freqs = np.fft.rfftfreq(n_fft, 1 / samp_freq)

    dpg.set_value(series_tag, [freqs, mag_db])
    dpg.set_item_label(y_axis_tag, "Magnitude (dB)")
    dpg.set_item_label(x_axis_tag, "Frequency (Hz)")
    dpg.configure_item(x_axis_tag, scale=dpg.mvPlotScale_Log10)

    _apply_axis_limits(
        x_axis_tag, y_axis_tag,
        x_lim_adjust, x_lim_a, x_lim_b,
        y_lim_adjust, y_lim_a, y_lim_b,
        series_tag=series_tag, mode="magnitude"
    )





def _plot_impulse_dpg(
    fir, samp_freq,
    series_tag, x_axis_tag, y_axis_tag,
    x_lim_adjust, x_lim_a, x_lim_b,
    y_lim_adjust, y_lim_a, y_lim_b,
    crop_samples=80,  # number of initial samples to crop
    tail_fraction=0.3, # fraction of IR tail to check
    zero_threshold=1e-6
):
    _clear_decay_children(y_axis_tag, series_tag)
    
    # ----------------- Determine if long IR -----------------
    tail_start = int(len(fir) * (1 - tail_fraction))
    tail_vals = fir[tail_start:]
    tail_nonzero = np.abs(tail_vals) >= zero_threshold

    if np.any(tail_nonzero):
        # Long IR: crop first N samples to avoid skewing scale
        fir = fir[crop_samples:]

    t_ms = np.arange(len(fir)) / samp_freq * 1000

    dpg.set_value(series_tag, [t_ms, fir])
    dpg.set_item_label(x_axis_tag, "Time (ms)")
    dpg.set_item_label(y_axis_tag, "Amplitude")
    dpg.configure_item(x_axis_tag, scale=dpg.mvPlotScale_Linear)

    _apply_axis_limits(
        x_axis_tag, y_axis_tag,
        x_adj=False, y_adj=False, series_tag=series_tag, mode="time"
    )
    
    

def _plot_group_delay_dpg(
    fir, samp_freq, n_fft,
    series_tag, x_axis_tag, y_axis_tag,
    x_lim_adjust, x_lim_a, x_lim_b,
    y_lim_adjust, y_lim_a, y_lim_b
):
    
    _clear_decay_children(y_axis_tag, series_tag)
    freqs, gd = calc_group_delay_from_ir(
        y=fir,
        sr=samp_freq
    )
    
    dpg.set_value(series_tag, [freqs, gd])
    dpg.set_item_label(x_axis_tag, "Frequency (Hz)")
    dpg.set_item_label(y_axis_tag, "Group Delay (ms)")
    dpg.configure_item(x_axis_tag, scale=dpg.mvPlotScale_Log10)

    _apply_axis_limits(
        x_axis_tag, y_axis_tag,
        x_adj=x_lim_adjust, x_a=x_lim_a, x_b=x_lim_b,
        y_adj=True, y_a=-100, y_b=100,
        series_tag=series_tag, mode="magnitude"
    )







def _plot_decay_dpg(
    fir, samp_freq, n_fft,
    series_tag, x_axis_tag, y_axis_tag,
    start_ms, end_ms, step_ms, win_ms,
    x_lim_adjust, x_lim_a, x_lim_b,
    y_lim_adjust, y_lim_a, y_lim_b,
    normalise=1  # 0 = no norm, 1 = normalize so reference = 0 dB
):
    """
    Plot REW-style decay curves with:
    - octave smoothing applied to ALL slices
    - shared normalization so reference curve = 0 dB
    """

    label_prefix = 'decay'

    # Prepare decay plot
    _prepare_decay_plot(y_axis_tag, series_tag)

    # ---------------- Full response ----------------
    n_fft_full = max(n_fft, len(fir))
    freqs_full = np.fft.rfftfreq(n_fft_full, 1 / samp_freq)

    F_full = np.fft.rfft(fir, n=n_fft_full)
    mag_full = np.abs(F_full)

    # ALWAYS smooth (matches magnitude plot)
    #mag_full = smooth_freq_octaves(mag_full, n_fft=n_fft_full,fund_freq=100,win_size_base=6)
    mag_full = smooth_gaussian_octave(data=mag_full, n_fft=n_fft, fs=samp_freq, fraction=12)

    mag_db_full = 20 * np.log10(np.maximum(mag_full, 1e-12))

    # Shared normalization reference
    if normalise:
        norm_offset = np.max(mag_db_full)
        mag_db_full -= norm_offset
    else:
        norm_offset = 0.0

    # Used for axis autoscaling
    dpg.set_value(series_tag, [freqs_full, mag_db_full])

    # ---------------- Time windows ----------------
    win_len = int(win_ms * samp_freq / 1000)
    times = np.arange(start_ms, end_ms + step_ms, step_ms)
    n_slices = len(times)

    # ---------------- Full response line ----------------
    full_series_id = dpg.add_line_series(
        freqs_full, mag_db_full,
        parent=y_axis_tag,
        label=f"{label_prefix}_full_response"
    )

    with dpg.theme() as full_theme:
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(
                dpg.mvPlotCol_Line,
                (255, 255, 255),
                category=dpg.mvThemeCat_Plots
            )
    dpg.bind_item_theme(full_series_id, full_theme)

    # ---------------- Slice colours ----------------
    cmap = plt.get_cmap('viridis')
    colors = [
        tuple(int(c * 255) for c in cmap((n_slices - 1 - i) / n_slices)[:3])
        for i in range(n_slices)
    ]

    # ---------------- Decay slices ----------------
    for i, t in enumerate(times):
        start = int(t * samp_freq / 1000)
        end = start + win_len
        if end > len(fir):
            break

        # Windowed segment
        seg = fir[start:end] * np.hanning(win_len)

        # FFT
        F_slice = np.fft.rfft(seg, n=n_fft)
        mag_slice = np.abs(F_slice)

        # ALWAYS smooth slice
        #mag_slice = smooth_freq_octaves(mag_slice, n_fft=n_fft,fund_freq=100,win_size_base=6)
        mag_slice = smooth_gaussian_octave(data=mag_slice, n_fft=n_fft, fs=samp_freq, fraction=12)

        mag_db_slice = 20 * np.log10(np.maximum(mag_slice, 1e-12))

        # Apply SAME normalization
        if normalise:
            mag_db_slice -= norm_offset

        center_ms = t + (win_ms / 2)
        display_label = f"{label_prefix}_{center_ms:.1f} ms"

        series_id = dpg.add_line_series(
            freqs_full, mag_db_slice,
            parent=y_axis_tag,
            label=display_label
        )

        with dpg.theme() as theme_id:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line,
                    colors[i],
                    category=dpg.mvThemeCat_Plots
                )
        dpg.bind_item_theme(series_id, theme_id)

    # ---------------- Axis setup ----------------
    dpg.set_item_label(x_axis_tag, "Frequency (Hz)")
    dpg.set_item_label(y_axis_tag, "Magnitude (dB)")
    dpg.configure_item(x_axis_tag, scale=dpg.mvPlotScale_Log10)

    # ---------------- Limits ----------------
    _apply_axis_limits(
        x_axis_tag, y_axis_tag,
        x_adj=x_lim_adjust, x_a=x_lim_a, x_b=x_lim_b,
        y_adj=True, y_a=-80, y_b=10,
        series_tag=series_tag,
        mode="decay"
    )

def _clear_decay_children(y_axis_tag, series_tag):
    """unhide main series and delete previous decay slices."""
    label_prefix = 'decay_'
    if dpg.does_item_exist(series_tag):
        dpg.configure_item(series_tag, show=True)

    if dpg.does_item_exist(y_axis_tag):
        children = dpg.get_item_children(y_axis_tag, slot=1)  # slot=1 = mvLineSeries
        if children:
            for child in children:
                # Only delete children that were previously created by decay plotting
                if label_prefix in dpg.get_item_label(child):
                    dpg.delete_item(child)
                    
def _prepare_decay_plot(y_axis_tag, series_tag):
    """Hide main series and delete previous decay slices for decay plotting."""
    label_prefix = 'decay_'
    # Hide the main series
    if dpg.does_item_exist(series_tag):
        dpg.configure_item(series_tag, show=False)

    # Delete previous decay lines under the y-axis
    if dpg.does_item_exist(y_axis_tag):
        children = dpg.get_item_children(y_axis_tag, slot=1)  # slot=1 = mvLineSeries
        if children:
            for child in children:
                # Only delete children that were previously created by decay plotting
                if label_prefix in dpg.get_item_label(child):
                    dpg.delete_item(child)
                    

                    
def _apply_axis_limits(
    x_axis_tag, y_axis_tag,
    x_adj=False, x_a=10, x_b=20000,
    y_adj=False, y_a=-10, y_b=10,
    series_tag=None,
    x_margin=0.0,
    y_margin=20.0,
    mode="magnitude",      # "magnitude" or "time"
    min_threshold=CN.THRESHOLD_CROP  # for time-domain
):
    if series_tag is None:
        dpg.set_axis_limits(x_axis_tag, x_a, x_b)
        dpg.set_axis_limits(y_axis_tag, y_a, y_b)
        return

    series_value = dpg.get_value(series_tag)
    if not series_value or len(series_value) != 2:
        dpg.set_axis_limits(x_axis_tag, x_a, x_b)
        dpg.set_axis_limits(y_axis_tag, y_a, y_b)
        return

    x_vals = np.array(series_value[0], dtype=float)
    y_vals = np.array(series_value[1], dtype=float)
    if len(x_vals) == 0 or len(y_vals) == 0:
        dpg.set_axis_limits(x_axis_tag, x_a, x_b)
        dpg.set_axis_limits(y_axis_tag, y_a, y_b)
        return

    # ----------------- Y-axis -----------------
    if mode in ("magnitude", "decay"):
        if mode == "magnitude":
            y_upper = 10.0
            y_lower = 10.0
        elif mode == "decay":
            y_upper = 10.0
            y_lower = 80.0

        # Restrict median calculation to x_a..x_b when x_adj is active
        if x_adj:
            mask = (x_vals >= x_a) & (x_vals <= x_b)
            y_subset = y_vals[mask]
            if y_subset.size > 0 and not np.all(np.isnan(y_subset)):
                y_med = np.nanmedian(y_subset)
            else:
                y_med = np.nanmedian(y_vals)
        else:
            y_med = np.nanmedian(y_vals)

        y_min = y_med - y_lower
        y_max = y_med + y_upper

    elif mode == "time":
        y_max = np.max(y_vals)
        y_min = np.min(y_vals)
            
    else:
        y_min, y_max = np.min(y_vals), np.max(y_vals)

    # Determine true min/max within mask
    if x_adj:
        mask = (x_vals >= x_a) & (x_vals <= x_b)
        y_subset = y_vals[mask]
        if y_subset.size == 0:
            y_subset = y_vals
    else:
        y_subset = y_vals
    
    y_true_min = np.nanmin(y_subset)
    y_true_max = np.nanmax(y_subset)
    
    # Apply Y-axis margin
    y_margin_val = (y_true_max - y_true_min) * (y_margin / 100)
    
    if y_adj:
        # Expand existing limits if data exceeds them
        new_y_a = min(y_a, y_true_min - y_margin_val)
        new_y_b = max(y_b, y_true_max + y_margin_val)
        dpg.set_axis_limits(y_axis_tag, new_y_a, new_y_b)
    else:
        #dpg.set_axis_limits(y_axis_tag, y_true_min - y_margin_val, y_true_max + y_margin_val)
        dpg.set_axis_limits(y_axis_tag, y_min - y_margin_val, y_max + y_margin_val)

    # ----------------- X-axis -----------------
    if x_adj:
        dpg.set_axis_limits(x_axis_tag, x_a, x_b)
    else:
        if mode == "magnitude":
            # X-limits where y is within ±y_upper / ±y_lower of median
            indices = np.where((y_vals >= y_min) & (y_vals <= y_max))[0]
        elif mode == "decay":
            # X-limits where y is within -y_lower / +y_upper of median
            indices = np.where((y_vals >= y_min) & (y_vals <= y_max))[0]
        elif mode == "time":
            # X-limits where y is above min_threshold
            indices = np.where(y_vals >= min_threshold)[0]
        else:
            indices = np.arange(len(x_vals))

        if len(indices) > 0:
            x_min = x_vals[indices[0]]
            x_max = x_vals[indices[-1]]
            x_range = x_max - x_min
            x_min -= x_range * x_margin
            x_max += x_range * x_margin
            # Clamp to reliable frequency range
            if mode != "time":
                x_min = max(x_min, x_a)
                x_max = min(x_max, x_b)
            
            # Guard against inverted range
            if x_min < x_max:
                dpg.set_axis_limits(x_axis_tag, float(x_min), float(x_max))
            else:
                dpg.set_axis_limits(x_axis_tag, x_a, x_b)
        else:
            # fallback if no indices
            dpg.set_axis_limits(x_axis_tag, x_vals[0], x_vals[-1])




def sort_names_by_values(names, values, descending=False):
    """
    Sorts a list of names based on corresponding integer values.

    Parameters:
        names (list of str): A list of names.
        values (list of int): A list of integer values corresponding to each name.
        descending (bool): If True, sorts in descending order. Defaults to False (ascending).

    Returns:
        list of str: A new list of names sorted according to the order of values.

    Raises:
        ValueError: If the input lists are not the same length.
        TypeError: If 'names' is not a list of strings or 'values' is not a list of integers.
    """
    # Validate input types
    if not isinstance(names, list) or not isinstance(values, list):
        raise TypeError("Both 'names' and 'values' must be lists.")
    
    if len(names) != len(values):
        raise ValueError("Both lists must be of the same length.")
    
    if not all(isinstance(name, str) for name in names):
        raise TypeError("All elements in 'names' must be strings.")
    
    if not all(isinstance(value, int) for value in values):
        raise TypeError("All elements in 'values' must be integers.")

    if not names:
        return []

    # Pair and sort based on the values
    paired_list = list(zip(names, values))
    sorted_pairs = sorted(paired_list, key=lambda pair: pair[1], reverse=descending)
    sorted_names = [name for name, _ in sorted_pairs]
    
    return sorted_names



def calculate_room_corner_centers(front_angle_deg):
    """
    Calculates 4 corner azimuths based on the angle of the front-right corner.
    
    Example: 
    Input 45 -> [45, 135, 225, 315] (Square)
    Input 40 -> [40, 140, 220, 320] (Slightly narrow/deep)
    """
    # 1. Front-Right (e.g., 40)
    fr = front_angle_deg
    
    # 2. Front-Left (e.g., 360 - 40 = 320)
    fl = 360 - front_angle_deg
    
    # 3. Rear-Right (e.g., 180 - 40 = 140)
    rr = 180 - front_angle_deg
    
    # 4. Rear-Left (e.g., 180 + 40 = 220)
    rl = 180 + front_angle_deg
    
    # Return sorted array for consistent processing
    return np.sort(np.array([fr, rr, rl, fl]))

def biased_spherical_coordinate_sampler(azim_src_set, elev_src_set, num_samples,
                                       biased_azimuth_centers=np.array([45, 135, 225, 315]),
                                       azimuth_spread=20, plot_distribution=False):
    """
    Samples spherical coordinates with a Gaussian bias towards specific azimuth centers.
    
    The bias is calculated using a circular wrapping distance. A smaller azimuth_spread 
    results in a stronger, narrower bias (tight clusters), while a larger spread 
    results in a weaker bias (more uniform distribution).

    Args:
        azim_src_set (np.ndarray): Sorted array of possible azimuth angles (degrees).
        elev_src_set (np.ndarray): Array of possible elevation angles (degrees).
        num_samples (int): Number of coordinate pairs to sample.
        biased_azimuth_centers (np.ndarray): Azimuths (degrees) to bias towards.
        azimuth_spread (float): Standard deviation of the bias. 
            Lower = Stronger bias (narrower peaks).
            Higher = Weaker bias (broader peaks/more uniform).
        plot_distribution (bool): If True, shows a histogram of results.

    Returns:
        tuple: (selected_azimuths, selected_elevations) as lists.
    """
    # Vectorized calculation of circular differences
    # Reshape to (num_azim, 1) and (1, num_centers) for broadcasting
    azim_col = azim_src_set[:, np.newaxis]
    centers_row = biased_azimuth_centers[np.newaxis, :]
    
    diff = np.abs(azim_col - centers_row)
    circular_diff = np.minimum(diff, 360 - diff)
    
    # Calculate Gaussian probabilities: P = sum( exp( -d^2 / (2 * sigma^2) ) )
    # Higher azimuth_spread (sigma) increases the denominator, flattening the curve.
    gauss_weights = np.exp(-(circular_diff**2) / (2 * azimuth_spread**2))
    probabilities = np.sum(gauss_weights, axis=1)
    probabilities /= np.sum(probabilities)

    # Sampling
    cdf = np.cumsum(probabilities)
    
    # Generate random values in bulk for speed
    rand_azims = [random.random() for _ in range(num_samples)]
    selected_azim_indices = np.searchsorted(cdf, rand_azims)
    selected_azimuths = azim_src_set[selected_azim_indices].tolist()

    # Elevation sampling (Uniform/Unbiased)
    selected_elevations = [random.choice(elev_src_set) for _ in range(num_samples)]

    if plot_distribution:
        plt.figure(figsize=(10, 5))
        plt.hist(selected_azimuths, bins=np.arange(0, 370, 10), edgecolor='black', alpha=0.7)
        plt.xlabel("Azimuth (degrees)")
        plt.ylabel("Frequency")
        plt.title(f"Azimuth Distribution (Spread={azimuth_spread})")
        plt.xticks(np.arange(0, 361, 45))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    return selected_azimuths, selected_elevations



def octave_smooth(freqs, values, fraction=1/3):
    """
    Efficiently applies fractional octave smoothing to a frequency-domain spectrum.

    Args:
        freqs (np.ndarray): Array of frequencies (Hz), must be sorted.
        values (np.ndarray): Array of corresponding values (e.g., group delay).
        fraction (float, optional): The fractional octave band (e.g., 1/3 for third-octave).
                                     Defaults to 1/3.

    Returns:
        np.ndarray: Array of smoothed values.
    """
    smoothed = np.copy(values)
    log_freqs = np.log2(freqs, where=freqs > 0, out=np.full_like(freqs, np.nan))
    fraction_half = fraction / 2

    for i, f_log in enumerate(log_freqs):
        if np.isnan(f_log):
            continue

        lower_log_f = f_log - fraction_half
        upper_log_f = f_log + fraction_half

        lower_idx = np.searchsorted(log_freqs, lower_log_f, side='left')
        upper_idx = np.searchsorted(log_freqs, upper_log_f, side='right')

        if lower_idx < upper_idx:
            smoothed[i] = np.nanmean(values[lower_idx:upper_idx])
        # If no frequencies fall within the band, the original value is kept (already initialized)

    return smoothed

def octave_smooth_slow(freqs, values, fraction=1/3):
    smoothed = np.zeros_like(values)
    for i, f in enumerate(freqs):
        if f <= 0:
            smoothed[i] = values[i]
            continue
        f1 = f * 2**(-fraction / 2)
        f2 = f * 2**(fraction / 2)
        idx = np.where((freqs >= f1) & (freqs <= f2))[0]
        if len(idx) > 0:
            smoothed[i] = np.nanmean(values[idx])
        else:
            smoothed[i] = values[i]
    return smoothed





def calc_group_delay_from_ir(
    y,
    sr=None,
    n_fft=CN.N_FFT,
    hop_length=512,
    smoothing_type='octave',
    smoothing_window=18,
    system_delay_ms=None,
    verbose=False
):
    """
    Calculate the group delay from a 1D impulse response with robust handling.
    Uses log_with_timestamp for all logging instead of print.
    """

    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError("Input 'y' must be a 1D NumPy array.")
    if not isinstance(sr, (int, float)) or sr <= 0:
        raise ValueError("Input 'sr' must be a positive number.")

    ir_len = len(y)

    # --- Zero-pad IR if shorter than n_fft ---
    if ir_len < n_fft:
        pad_len = n_fft - ir_len
        y = np.pad(y, (0, pad_len), mode="constant")
        if verbose:
            log_with_timestamp(
                f"Zero-padding IR: {ir_len} → {len(y)} samples"
            )

    padded_len = len(y)

    # --- Frame count (now always valid) ---
    n_frames = max((padded_len - n_fft) // hop_length + 1, 1)

    # Positive frequencies only
    freq_bins = rfftfreq(n_fft, 1 / sr)
    delta_f = freq_bins[1] - freq_bins[0]

    group_delays = []

    if verbose:
        log_with_timestamp(
            f"Group delay calculation: IR length={ir_len}, "
            f"padded_length={padded_len}, n_fft={n_fft}, "
            f"hop_length={hop_length}, n_frames={n_frames}"
        )

    # --- Compute group delay per frame ---
    for frame in range(n_frames):
        start_idx = frame * hop_length
        end_idx = start_idx + n_fft

        frame_data = y[start_idx:end_idx]

        # Safety (should not be needed now, but cheap)
        if len(frame_data) < n_fft:
            frame_data = np.pad(frame_data, (0, n_fft - len(frame_data)))

        yf = rfft(frame_data, n=n_fft)

        phase = np.angle(yf)
        unwrapped_phase = unwrap(phase)

        delta_phase = diff(unwrapped_phase)

        gd = -delta_phase / (2 * np.pi * delta_f)  # seconds
        gd_ms = gd * 1000                          # milliseconds

        group_delays.append(gd_ms)

    # --- Average across frames ---
    averaged_group_delay_ms = np.mean(group_delays, axis=0)

    # --- Smoothing ---
    smoothed_group_delay_ms = averaged_group_delay_ms.copy()

    if smoothing_type == 'savitzky-golay' and smoothing_window > 1:
        smoothed_group_delay_ms = savgol_filter(
            averaged_group_delay_ms, smoothing_window, 3
        )

    elif smoothing_type == 'hann' and smoothing_window > 1 and smoothing_window % 2 == 1:
        win = windows.hann(smoothing_window)
        win /= np.sum(win)
        smoothed_group_delay_ms = np.convolve(
            averaged_group_delay_ms, win, mode='same'
        )

    elif smoothing_type == 'gaussian' and smoothing_window > 0:
        smoothed_group_delay_ms = gaussian_filter1d(
            averaged_group_delay_ms, smoothing_window
        )

    elif smoothing_type == 'octave':
        freqs_smooth = freq_bins[1:]
        smoothed_group_delay_ms = octave_smooth(
            freqs_smooth,
            averaged_group_delay_ms,
            fraction=1 / smoothing_window
        )

    # --- System delay subtraction (bass-referenced) ---
    if system_delay_ms is None:
        # Bass reference range (Hz)
        bass_lo = 20.0
        bass_hi = 200.0

        freqs_ref = freq_bins[1:]  # matches GD vector
        bass_mask = (freqs_ref >= bass_lo) & (freqs_ref <= bass_hi)

        bass_slice = smoothed_group_delay_ms[bass_mask]

        if bass_slice.size and not np.all(np.isnan(bass_slice)):
            estimated_delay = np.nanmedian(bass_slice)

            if not np.isnan(estimated_delay):
                smoothed_group_delay_ms -= estimated_delay

                if verbose:
                    log_with_timestamp(
                        f"Estimated system delay (bass {bass_lo:.0f}–{bass_hi:.0f} Hz): "
                        f"{estimated_delay:.3f} ms"
                    )
    else:
        smoothed_group_delay_ms -= system_delay_ms
        if verbose:
            log_with_timestamp(
                f"Applied system delay: {system_delay_ms:.3f} ms"
            )

    if verbose:
        log_with_timestamp(
            f"Final group delay: min={np.nanmin(smoothed_group_delay_ms):.3f} ms, "
            f"max={np.nanmax(smoothed_group_delay_ms):.3f} ms, "
            f"mean={np.nanmean(smoothed_group_delay_ms):.3f} ms"
        )

    return freq_bins[1:], smoothed_group_delay_ms











def reshape_array_to_two_dims(arr: np.ndarray) -> np.ndarray:
    """
    Reshapes a NumPy array to meet specific dimension requirements.
    Useful for reshaping SRIRs, where we only want 2 dimensions

    Args:
        arr: The input NumPy array of arbitrary dimensions.

    Returns:
        A NumPy array with the following shape:
        - The last dimension is the longest dimension from the input array.
        - The first dimension is the product of all other dimensions 
          merged together.
    """

    original_shape = arr.shape
    num_dims = len(original_shape)

    if num_dims == 0:
        return arr  # Or raise an exception, depending on desired behavior for scalar input

    # Find the longest dimension and its index
    longest_dim = 0
    longest_dim_index = -1  # Initialize to -1 to handle empty shapes correctly
    for i, dim in enumerate(original_shape):
        if dim > longest_dim:
            longest_dim = dim
            longest_dim_index = i

    if num_dims == 1:
        return arr.reshape(1, original_shape[0])  # Handle 1D array case

    # Calculate the shape of the first dimension of the result
    other_dims_product = 1
    for i in range(num_dims):
        if i != longest_dim_index:
            other_dims_product *= original_shape[i]

    # Create the new shape tuple
    new_shape = (other_dims_product, longest_dim)

    # Move the longest axis to the last position using transpose
    if longest_dim_index != num_dims - 1:
        axes = list(range(num_dims))
        axes.pop(longest_dim_index)
        axes.append(longest_dim_index)
        arr = arr.transpose(axes)

    # Reshape the array
    reshaped_arr = arr.reshape(new_shape)

    return reshaped_arr



def reshape_array_to_two_dims_preserve_pairs(arr: np.ndarray) -> np.ndarray:
    """
    Reshapes a NumPy array to 2D while preserving pairs (length 2) if they exist.
    
    Logic:
    1. Identifies the longest dimension (samples).
    2. Checks for a 'pair' dimension (length 2).
    3. If a pair exists, it ensures the final 2D array interleaves them (L, R, L, R...).
    4. If no pair exists, it reverts to the original logic of flattening all 
       non-sample dimensions into the first axis.
    """
    original_shape = arr.shape
    num_dims = len(original_shape)

    if num_dims == 0:
        return arr
    if num_dims == 1:
        return arr.reshape(1, original_shape[0])

    # 1. Identify the longest dimension (Time/Samples)
    longest_dim_index = int(np.argmax(original_shape))
    longest_dim = original_shape[longest_dim_index]

    # 2. Identify pair axes (length 2) that are not the samples axis
    pair_axes = [i for i, s in enumerate(original_shape) if s == 2 and i != longest_dim_index]

    # Define the order of axes for transposition
    # 'others' preserves the original relative order of all remaining dimensions
    others = [i for i in range(num_dims) if i != longest_dim_index]

    if pair_axes:
        # We found a pair (e.g., Ears). 
        # To keep them together (L, R, L, R), the pair axis MUST be the last 
        # axis before the samples.
        pair_axis = pair_axes[0]
        others = [i for i in others if i != pair_axis]
        
        # New order: [Remaining..., Pair, Samples]
        new_axes_order = others + [pair_axis] + [longest_dim_index]
    else:
        # No pair found: Original logic.
        # New order: [Remaining..., Samples]
        new_axes_order = others + [longest_dim_index]

    # 3. Transpose to bring the dimensions into the correct priority
    arr = arr.transpose(new_axes_order)

    # 4. Calculate the product of all dimensions except the longest one
    other_dims_product = np.prod([original_shape[i] for i in range(num_dims) if i != longest_dim_index])
    
    # 5. Final reshape
    return arr.reshape(other_dims_product, longest_dim)


def reshape_array_to_three_dims(arr: np.ndarray) -> np.ndarray:
    """
    Reshapes a NumPy array to meet specific dimension requirements.
    useful for reshaping MultiSpeakerBRIR, where we want to keep an axis with 2 channels, 3 dimensions in total

    Args:
        arr: The input NumPy array of arbitrary dimensions.

    Returns:
        A NumPy array with the following shape:
        - The last dimension is the longest dimension from the input array.
        - The 2nd last dimension is the dimension from the input array with length 2.
        - The first dimension is the product of all other dimensions merged together.
    """

    original_shape = arr.shape
    num_dims = len(original_shape)

    if num_dims < 2:
        raise ValueError("Input array must have at least 2 dimensions.")

    # Find the longest dimension and its index
    longest_dim = 0
    longest_dim_index = -1
    for i, dim in enumerate(original_shape):
        if dim > longest_dim:
            longest_dim = dim
            longest_dim_index = i

    # Find the dimension with length 2 and its index
    dim_2_index = -1
    for i, dim in enumerate(original_shape):
        if dim == 2:
            dim_2_index = i
            break  # Only need to find the first dimension with length 2

    if dim_2_index == -1:
        raise ValueError("Input array must have a dimension with length 2.")

    # Calculate the shape of the first dimension of the result
    other_dims_product = 1
    for i in range(num_dims):
        if i != longest_dim_index and i != dim_2_index:
            other_dims_product *= original_shape[i]

    # Create the new shape tuple
    new_shape = (other_dims_product, 2, longest_dim)

    # Move the longest axis and axis with length 2 to the last positions using transpose
    if longest_dim_index != num_dims - 1 or dim_2_index != num_dims - 2:
        axes = list(range(num_dims))
        axes.pop(longest_dim_index)
        axes.pop(axes.index(dim_2_index)) #remove dim_2 after longest_dim is removed.
        axes.extend([dim_2_index, longest_dim_index])
        arr = arr.transpose(axes)

    # Reshape the array
    reshaped_arr = arr.reshape(new_shape)

    return reshaped_arr


def reshape_to_4d(measurement_array: np.ndarray, first_dim: int) -> np.ndarray:
    """
    Reshape a 2D array (measurements x samples) into a 4D array with shape (first_dim x N x 1 x samples),
    redistributing measurements across the first three dimensions. If needed, fills remaining
    positions with zeros.

    Parameters:
        measurement_array (np.ndarray): A 2D NumPy array of shape (measurements x samples)
        first_dim (int): The desired size of the first dimension (e.g., 7)

    Returns:
        np.ndarray: A 4D array of shape (first_dim x N x 1 x samples)
    """
    measurements, samples = measurement_array.shape

    # Compute the number of slots needed per group
    N = math.ceil(measurements / first_dim)

    # Initialize the output array with zeros
    output = np.zeros((first_dim, N, 1, samples), dtype=measurement_array.dtype)

    # Fill the array by distributing measurements across first_dim and N
    for idx in range(measurements):
        group = idx % first_dim  # Group index along the first dimension
        slot = idx // first_dim  # Slot index along the second dimension
        output[group, slot, 0, :] = measurement_array[idx]

    return output















def plot_average_measurements(file_path: str):
    """
    Loads a 2D NumPy array (measurements x samples) from the given file path,
    computes the average of all measurements, and plots the resulting 1D array.
    Also counts how many measurements consist entirely of zeros.

    Parameters:
        file_path (str): Path to the .npy file containing the 2D matrix.
    """
    # Load the matrix
    matrix = np.load(file_path)

    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D array, but got shape {matrix.shape}")

    # Count rows that are all zeros
    zero_rows = np.all(matrix == 0, axis=1)
    num_zero_rows = np.sum(zero_rows)
    print(f"Number of all-zero measurements: {num_zero_rows}")

    # Compute average across measurements (rows)
    average = np.mean(matrix, axis=0)

    # Plot the result
    plt.figure(figsize=(10, 4))
    plt.plot(average, label="Average Measurement")
    plt.title("Average of All Measurements")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


















def expand_measurements(
    measurement_array: np.ndarray,
    desired_measurements: int,
    fs: int = CN.SAMP_FREQ,
    pitch_range: tuple = (-12, 12),
    shuffle: bool = False,
    seed="random",#"random"
    num_threads: int = 4,
    plot_first_n: int = 10,
    plot: bool = CN.PLOT_ENABLE, 
    gui_logger=None,
    cancel_event=None,
    pitch_shift_comp=CN.AS_PS_COMP_LIST[1],
    comp_strength=1.0,
    ignore_ms: float = CN.IGNORE_MS,
    report_progress=0, 
    pitch_shift_mode: str = CN.AS_SPAT_EXP_LIST[1], #resample or librosa
    preserve_originals: bool = False #False due to compatibility issues with pitch shift

) -> tuple[np.ndarray, int]:
    """
    Expand a measurement array by generating pitch-shifted variants.
    Includes optional per-IR compensation, multithreading, and progress updates.

    Returns:
        output array (N, L), status code:
            0 = success
            1 = failure
            2 = cancelled
    """

    # ------------------------------------------------------------
    # Constants & Setup
    # ------------------------------------------------------------
    n_fft        = 4096#4096,8192,32768,16384
    truncate_len = 4096#512,8192,
    f_min        = 20
    f_max        = 20000

    measurement_array = measurement_array.astype(np.float64, copy=False)
    base_n, sample_len = measurement_array.shape
    

    output = np.empty((desired_measurements, sample_len), dtype=np.float64)
    status = 1
  
    # ------------------------------------------------------------
    # Precompute magnitude spectra of ORIGINAL IRs for compensation
    # ------------------------------------------------------------
    orig_mag = np.abs(np.fft.rfft(measurement_array, n=n_fft, axis=1))
    
    # ------------------------------------------------------------
    # === NEW: PER-IR AND GLOBAL REFERENCES ===
    # ------------------------------------------------------------
    orig_db = 20.0 * np.log10(np.maximum(orig_mag, 1e-12))
    
    # Precompute smoothed reference for EVERY original IR
    ref_db_per_ir = np.zeros_like(orig_db)
    for i in range(base_n):
        tmp_mag = 10.0 ** (orig_db[i] / 20.0)
        tmp_mag = smooth_gaussian_octave(data=tmp_mag, n_fft=n_fft, fs=fs, fraction=12)
        ref_db_per_ir[i] = 20.0 * np.log10(np.maximum(tmp_mag, 1e-12))

    # Precompute smoothed GLOBAL reference (mean)
    ref_db_global_raw = np.mean(orig_db, axis=0)
    ref_mag_g = 10.0 ** (ref_db_global_raw / 20.0)
    ref_mag_g = smooth_gaussian_octave(data=ref_mag_g, n_fft=n_fft, fs=fs, fraction=12)
    ref_db_global = 20.0 * np.log10(np.maximum(ref_mag_g, 1e-12))

    # ------------------------------------------------------------
    # Pitch shift value generator
    # ------------------------------------------------------------
    if abs(pitch_range[0]) < 1e-6 and abs(pitch_range[1]) < 1e-6:
        pitch_range = (-12, 12)
        log_with_timestamp("Invalid pitch range, resetting to default", gui_logger)
        

    if seed == "random":
        seed = np.random.SeedSequence().entropy

    rng = np.random.default_rng(seed)
    num_augmented = desired_measurements
    zero_ir = np.zeros(sample_len)
    
    if preserve_originals:
        num_augmented = max(desired_measurements - base_n, 0)
    
    pitch_shifts = rng.uniform(
        pitch_range[0],
        pitch_range[1],
        num_augmented
    )

    # Thread-safe plot queue
    plot_queue = queue.Queue()

    def apply_pitch_shift(ir, shift_val):
        if pitch_shift_mode == CN.AS_SPAT_EXP_LIST[1]:
            return apply_pitch_shift_librosa(ir, shift_val, fs)
    
        elif pitch_shift_mode == CN.AS_SPAT_EXP_LIST[2]:
            return apply_pitch_shift_resample(ir, shift_val, fs)
    
        else:
            raise ValueError(
                f"Invalid pitch_shift_mode '{pitch_shift_mode}'. "
                "Use 'librosa' or 'resample'."
            )
            
    # ------------------------------------------------------------
    # === CHANGED: compensation uses GLOBAL reference ===
    # ------------------------------------------------------------
    def build_compensation_filter(shifted_ir, ir_index, plot_idx=None):
        shifted_mag = np.abs(np.fft.rfft(shifted_ir, n=n_fft))
        shifted_mag = smooth_gaussian_octave(data=shifted_mag, n_fft=n_fft, fs=fs, fraction=6)#12
        shifted_db = 20.0 * np.log10(np.maximum(shifted_mag, 1e-12))

        # Choose between the average curve or the specific source curve
        target_ref_db = ref_db_global if pitch_shift_comp == CN.AS_PS_COMP_LIST[2]  else ref_db_per_ir[ir_index]
  
        comp_mag_raw_db = target_ref_db - shifted_db
        
        # ------------------------------------------------------------
        # NEW: limit compensation strength to ±15 dB
        # ------------------------------------------------------------
        comp_mag_raw_db = np.clip(comp_mag_raw_db, -15.0, 15.0)
        
        comp_mag = 10.0 ** (comp_mag_raw_db / 20.0)
        comp_mag = smooth_gaussian_octave(data=comp_mag, n_fft=n_fft,fs=fs, fraction=4)#6
        comp_mag_db = 20.0 * np.log10(np.maximum(comp_mag, 1e-12))

        # HF taper
        hf_start, hf_end = 6000.0, 11000.0
        bin_start = int(round(hf_start * n_fft / fs))
        bin_end   = int(round(hf_end   * n_fft / fs))

        n_bins = comp_mag_db.shape[-1]
        bin_start = np.clip(bin_start, 0, n_bins - 1)
        bin_end   = np.clip(bin_end, bin_start + 1, n_bins)

        length = bin_end - bin_start
        t = np.linspace(0.0, np.pi, length, endpoint=True)
        taper = 0.5 * (1.0 + np.cos(t))

        comp_mag_db[bin_start:bin_end] *= taper
        comp_mag_db[bin_end:] = 0.0

        comp_mag = 10.0 ** (comp_mag_db*comp_strength / 20.0)

        if plot and plot_idx is not None and plot_idx < plot_first_n:
            fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
            original_ir = measurement_array[ir_index]
    
            plot_queue.put({
                "idx": ir_index,
                "freqs": fft_freqs,
                "shifted_db": shifted_db,
                "comp_db": comp_mag_db,
                "ref_db": target_ref_db,
                "time_orig": original_ir,
                "time_shifted": shifted_ir
            })

        return build_min_phase_filter(comp_mag,fs=fs,n_fft=n_fft,truncate_len=truncate_len,f_min=f_min,f_max=f_max,band_limit=True)
 
    # ------------------------------------------------------------
    # Thread Process Function
    # ------------------------------------------------------------
    count_lock = threading.Lock()
    count = 0
    last_print_time = time.time()

    def process_one(i):
        nonlocal count, last_print_time
    
        if cancel_event and cancel_event.is_set():
            return None
    
        ir_index = i % base_n
        ir = measurement_array[ir_index]
    
        try:
            # ------------------------------------------------------------
            # Case 1: Preserve originals (no pitch shift, no compensation)
            # ------------------------------------------------------------
            if preserve_originals and i < base_n:
                shifted = ir.copy()
    
            # ------------------------------------------------------------
            # Case 2: Pitch-shifted augmentation
            # ------------------------------------------------------------
            else:
                shift_idx = i if not preserve_originals else (i - base_n)
                shift = pitch_shifts[shift_idx]
    
                shifted = apply_pitch_shift(ir, shift)
    
                # Ensure the shifted IR is the correct length before alignment
                if len(shifted) != sample_len:
                    if len(shifted) < sample_len:
                        temp = np.zeros(sample_len)
                        temp[:len(shifted)] = shifted
                        shifted = temp
                    else:
                        shifted = shifted[:sample_len]
   
                # Compensation only for pitch-shifted IRs
                if pitch_shift_comp != CN.AS_PS_COMP_LIST[0]:
                    comp = build_compensation_filter(shifted, ir_index, i)
                    conv = sp.signal.fftconvolve(shifted, comp, mode="full")
                    # trim to original IR length
                    shifted = conv[:sample_len]
      
            # ------------------------------------------------------------
            # Progress tracking
            # ------------------------------------------------------------
            with count_lock:
                count += 1
                now = time.time()
                if (count % 100 == 0) or (count == desired_measurements) or (now - last_print_time >= 10.0):
                    log_with_timestamp(
                        f"Expansion Progress: {count}/{desired_measurements} measurements processed.",
                        gui_logger
                    )
                    last_print_time = now
    
                    if report_progress > 0:
                        a, b = 0.1, 0.35
                        progress = a + (count / desired_measurements) * (b - a)
                        update_gui_progress(report_progress, progress=progress)
    
            return i, shifted
    
        except Exception as e:
            log_with_timestamp(f"Error processing measurement {i}: {e}", gui_logger)
            return i, None


        
    # ------------------------------------------------------------
    # ThreadPool execution (OPTIMISED)
    # ------------------------------------------------------------
    try:
        max_in_flight = max(num_threads * 2, 8)  # small buffer
        next_index = 0
        in_flight = set()
    
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    
            # Prime the queue
            while next_index < desired_measurements and len(in_flight) < max_in_flight:
                in_flight.add(executor.submit(process_one, next_index))
                next_index += 1
    
            while in_flight:
                done, in_flight = concurrent.futures.wait(
                    in_flight,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
    
                for f in done:
                    if cancel_event and cancel_event.is_set():
                        status = 2
                        log_with_timestamp("Expansion cancelled by user.", gui_logger)
                        return output, status
    
                    result = f.result()
                    if result is None:
                        status = 2
                        log_with_timestamp("Expansion cancelled by user.", gui_logger)
                        return output, status
    
                    idx, shifted = result
                    output[idx] = shifted if shifted is not None else zero_ir
    
                    # Submit next job
                    if next_index < desired_measurements:
                        in_flight.add(executor.submit(process_one, next_index))
                        next_index += 1
    
        status = 0
        log_with_timestamp("Expansion complete.", gui_logger)
    
        if shuffle:
            rng.shuffle(output)
    
    except Exception as e:
        log_with_timestamp(f"Exception in expansion: {e}", gui_logger)
        status = 1

    # ---------------------- Plot the queued IRs ----------------------
    if plot:
        while not plot_queue.empty():
            data = plot_queue.get()
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
            plt.subplots_adjust(hspace=0.4)
    
            # --- Subplot 1: Frequency Domain ---
            ax1.semilogx(data["freqs"], data["ref_db"], label='Reference (dB)', linestyle='--', color='black', alpha=0.7)
            ax1.semilogx(data["freqs"], data["shifted_db"], label='Shifted IR (dB)', color='tab:blue')
            ax1.semilogx(data["freqs"], data["comp_db"], label='Comp Filter (dB)', color='tab:red')
            ax1.axhline(0.0, linestyle='--', color='gray', linewidth=0.8)
            ax1.set_xlim(20, fs/2)
            ax1.set_ylim(-20, 20)
            ax1.set_title(f'Spectral Analysis: IR #{data["idx"]}')
            ax1.set_ylabel('Magnitude (dB)')
            ax1.grid(True, which='both', alpha=0.3)
            ax1.legend(loc='upper right', fontsize='small')
    
            # --- Subplot 2: Time Domain (Zoomed on Peak) ---
            time_axis = np.arange(len(data["time_orig"])) / fs * 1000 # in ms
            
            # Calculate a nice zoom window around the original peak
            peak_sample = np.argmax(np.abs(data["time_orig"]))
            start_ms = max(0, (peak_sample - 100) / fs * 1000)
            end_ms = min(len(data["time_orig"]) / fs * 1000, (peak_sample + 1000) / fs * 1000)
    
            ax2.plot(time_axis, data["time_orig"], label='Original', color='black', alpha=0.5)
            ax2.plot(time_axis, data["time_shifted"], label='Shifted + Aligned', color='tab:orange', linestyle='--')
            
            ax2.set_xlim(start_ms, end_ms) # Zoom into the first few milliseconds
            ax2.set_title(f'Time Domain Alignment (Zoomed View)')
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Amplitude')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right', fontsize='small')
    
            plt.show()
            plt.close()

    return output, status


def apply_ir_shift(arr, shift_amt, fade_len=10, circular=False):
    """
    Shifts an IR array and applies a tiny linear fade-in.
    
    Args:
        arr: The input impulse response array.
        shift_amt: Samples to shift (positive for delay, negative for advance).
        fade_len: Length of the linear fade-in ramp at the start of the buffer.
        circular: If True, uses np.roll (wrap-around). 
                  If False, uses zero-padding (linear shift).
    """
    if shift_amt == 0:
        result = arr.copy()
    elif circular:
        # --- Circular Shift ---
        result = np.roll(arr, shift_amt)
    else:
        # --- Linear Shift with Zero Padding (The "Safe" Way) ---
        result = np.zeros_like(arr)
        n = len(arr)
        if shift_amt > 0:  # Delay (Shift Right)
            s = min(shift_amt, n)
            result[s:] = arr[:n-s]
        elif shift_amt < 0:  # Advance (Shift Left)
            s = min(abs(shift_amt), n)
            result[:n-s] = arr[s:] # Take from sample S to the end, place at start

    # --- Apply Fade-In ---
    # Essential for circular shifts to kill wrap-around clicks,
    # and helpful for linear shifts to ensure a clean start.
    if fade_len > 0:
        actual_fade = min(fade_len, len(result))
        fade_curve = np.linspace(0.0, 1.0, actual_fade)
        result[:actual_fade] *= fade_curve
        
    return result



# ------------------------------------------------------------
# Helper: Pitch shift (librosa phase vocoder)
# ------------------------------------------------------------
def apply_pitch_shift_librosa(ir, shift_val, fs):
    return librosa.effects.pitch_shift(
        ir,
        sr=fs,
        n_steps=shift_val,
        res_type="kaiser_best"
    )


# ------------------------------------------------------------
# Helper: Pitch shift (fast resample-based)
# ------------------------------------------------------------
def apply_pitch_shift_resample(ir, shift_val, fs):
    """
    Fast pitch shift using librosa.resample with energy normalization.
    """
    factor = 2 ** (shift_val / 12.0)

    shifted = librosa.resample(
        ir,
        orig_sr=fs,
        target_sr=int(fs * factor),
        res_type="kaiser_best",
        axis=0,
        scale=True
    )

    # Pad or truncate
    if len(shifted) < len(ir):
        temp = np.zeros_like(ir)
        temp[:len(shifted)] = shifted
        shifted = temp
    else:
        shifted = shifted[:len(ir)]

    # Energy normalization (important for IRs)
    shifted *= np.linalg.norm(ir) / max(np.linalg.norm(shifted), 1e-12)

    return shifted







def equalize_brirs_parametric(
    brir_dataset,
    fs = CN.SAMP_FREQ,
    n_fft = 8192,
    truncate_len = None,
    num_filters = 24,
    low_freq_cut = 20.0,
    high_freq_cut = 6000.0,
    fraction_smooth_target = 1.2,
    fraction_smooth_avg = 12,
    diff_db_override = None,
    override_n_fft = None,
    plot = CN.PLOT_ENABLE
):
    """
    Optimized version: Fits parametric EQ via vectorized log-magnitude response
    and applies filtering via FFT convolution for massive speed gains.
    """
    # --- 1. Preparation ---
    brir_dataset = np.asarray(brir_dataset, dtype=np.float64)
    orig_shape = brir_dataset.shape
    n_samples = orig_shape[-1]
    if truncate_len is None: truncate_len = n_samples

    fft_freqs = np.fft.rfftfreq(n_fft, 1/fs)
    
    # --- 2. Difference Curve Logic (With Shortcut) ---
    if diff_db_override is not None:
        # SHORTCUT: Skip all averaging and smoothing
        diff_db = np.asarray(diff_db_override)
        if override_n_fft is not None:
            full_freqs = np.fft.rfftfreq(override_n_fft, 1/fs)
            diff_db = np.interp(fft_freqs, full_freqs, diff_db[:len(full_freqs)],
                                left=diff_db[0], right=diff_db[-1])
        elif len(diff_db) != len(fft_freqs):
            raise ValueError(f"diff_db_override length ({len(diff_db)}) must match n_fft/2 + 1 ({len(fft_freqs)})")
    else:
        # Standard logic: Only runs if no override is provided
        mag = np.abs(np.fft.rfft(brir_dataset.reshape(-1, n_samples), n=n_fft, axis=1))
        avg_db = 20 * np.log10(np.maximum(mag, 1e-12)).mean(axis=0)
        a_lo, a_hi = np.searchsorted(fft_freqs, [low_freq_cut, high_freq_cut])
        avg_db[:a_lo], avg_db[a_hi:] = avg_db[a_lo], avg_db[a_hi]

        # smooth_gaussian_octave is assumed to be defined in your namespace
        target_mag = smooth_gaussian_octave(10**(avg_db/20), n_fft=n_fft, fs=fs, fraction=fraction_smooth_target)
        avg_mag_cut = smooth_gaussian_octave(10**(avg_db/20), n_fft=n_fft, fs=fs, fraction=fraction_smooth_avg)
        diff_db = 20*np.log10(np.maximum(target_mag, 1e-12)) - 20*np.log10(np.maximum(avg_mag_cut, 1e-12))

    diff_db = np.clip(diff_db, -20, 20)

    # --- 3. Optimizer Setup (Subsampled for Speed) ---
    fit_idx = np.unique(np.geomspace(1, len(fft_freqs)-1, 512).astype(int))
    w_fit = 2 * np.pi * fft_freqs[fit_idx] / fs
    target_fit = diff_db[fit_idx]
    z1_fit = np.exp(-1j * w_fit)
    z2_fit = np.exp(-2j * w_fit)

    freqs_init = np.geomspace(low_freq_cut, high_freq_cut, num_filters)
    params0 = np.column_stack((freqs_init, np.interp(freqs_init, fft_freqs, diff_db), np.ones(num_filters))).flatten()
    bounds = [(low_freq_cut, high_freq_cut) if i%3==0 else (-15, 15) if i%3==1 else (0.5, 5) for i in range(num_filters*3)]

    def objective(params):
        p = params.reshape(-1, 3)
        f0, gain, Q = p[:, 0:1], p[:, 1:2], p[:, 2:3]
        A = 10**(gain/40); w0 = 2*np.pi*f0/fs; alpha = np.sin(w0)/(2*Q); cos_w0 = np.cos(w0)
        b0, b1, b2 = 1 + alpha*A, -2*cos_w0, 1 - alpha*A
        a0, a1, a2 = 1 + alpha/A, -2*cos_w0, 1 - alpha/A
        H = (b0 + b1*z1_fit + b2*z2_fit) / (a0 + a1*z1_fit + a2*z2_fit)
        resp_db = np.sum(20 * np.log10(np.abs(H) + 1e-12), axis=0)
        return np.sum((resp_db - target_fit)**2)

    res = minimize(objective, params0, bounds=bounds, method='L-BFGS-B', options={'maxiter': 120, 'ftol': 1e-4})
    opt = res.x.reshape(-1, 3)

    # --- 4. EQ-IR Kernel Generation ---
    sos = []
    for f0, gain, Q in opt:
        A = 10**(gain/40); w0 = 2*np.pi*f0/fs; alpha = np.sin(w0)/(2*Q)
        b0, b1, b2 = 1+alpha*A, -2*np.cos(w0), 1-alpha*A
        a0, a1, a2 = 1+alpha/A, -2*np.cos(w0), 1-alpha/A
        sos.append([b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0])
    
    # Create the filter kernel (Impulse Response)
    eq_ir = signal.sosfilt(np.array(sos), np.eye(1, n_fft)[0])

    # --- 5. Application via FFT Convolution ---
    flat_data = brir_dataset.reshape(-1, n_samples)
    output = signal.fftconvolve(flat_data, eq_ir[None, :], mode='full', axes=1)[:, :truncate_len]
    
    if plot:
        plt.figure(figsize=(10, 5))
        # High-res curve for plotting
        w_full = 2 * np.pi * fft_freqs / fs
        z1_f, z2_f = np.exp(-1j * w_full), np.exp(-2j * w_full)
        full_resp_db = np.zeros_like(fft_freqs)
        for f0, gain, Q in opt:
            A = 10**(gain/40); w0 = 2*np.pi*f0/fs; alpha = np.sin(w0)/(2*Q); cos_w0 = np.cos(w0)
            b0, b1, b2 = 1+alpha*A, -2*cos_w0, 1-alpha*A
            a0, a1, a2 = 1+alpha/A, -2*cos_w0, 1-alpha/A
            h = (b0 + b1*z1_f + b2*z2_f) / (a0 + a1*z1_f + a2*z2_f)
            full_resp_db += 20 * np.log10(np.abs(h) + 1e-12)
        
        plt.semilogx(fft_freqs, diff_db, 'r--', label='Target (Override)' if diff_db_override is not None else 'Target')
        plt.semilogx(fft_freqs, full_resp_db, 'k', linewidth=1.5, label='Fitted Parametric EQ')
        plt.grid(True, which='both', alpha=0.3); plt.legend(); plt.xlim(20, 20000); plt.ylim(-25, 25)
        plt.ylabel('Magnitude (dB)'); plt.xlabel('Frequency (Hz)'); plt.show()

    return output.reshape(orig_shape[:-1] + (truncate_len,))


def equalize_low_freqs_db(
    measurement_array: np.ndarray,
    fs: int = CN.SAMP_FREQ,
    n_fft: int = CN.N_FFT,
    truncate_len: int = 8192,
    num_threads: int = 4,
    plot_first_n: int = 1,
    plot: bool = CN.PLOT_ENABLE,
    gui_logger=None,
    cancel_event=None,
    report_progress=0
) -> tuple[np.ndarray, int]:
    """
    Equalizes low-frequency roll-off in a set of impulse responses.
    Averages magnitude across all IRs in dB, flattens below `low_freq_cut`,
    computes difference in dB, converts to min-phase FIR, and applies to IRs.
    """

    measurement_array = measurement_array.astype(np.float64, copy=False)
    base_n, sample_len = measurement_array.shape
    output = np.empty_like(measurement_array)
    status = 1
    
    target_low_freq_cut = 50.0
    avg_low_freq_cut    = 50.0

    target_high_freq_cut = 16000.0
    avg_high_freq_cut    = 16000.0

    # Thread-safe plot queue
    plot_queue = queue.Queue()

    # ------------------------------------------------------------
    # Step 1: Averaged magnitude spectrum
    # ------------------------------------------------------------
    mag = np.abs(np.fft.rfft(measurement_array, n=n_fft, axis=1))
    mag_db = 20 * np.log10(np.maximum(mag, 1e-12))
    avg_db = np.mean(mag_db, axis=0)

    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)

    # ------------------------------------------------------------
    # Step 2: Build TARGET (less smooth, flatter extremes)
    # ------------------------------------------------------------
    target_mag = 10 ** (avg_db / 20.0)
    target_mag = smooth_gaussian_octave(
        data=target_mag, n_fft=n_fft, fs=fs, fraction=1.3
    )

    target_db = 20 * np.log10(np.maximum(target_mag, 1e-12))

    # LF flatten
    t_lo = np.searchsorted(fft_freqs, target_low_freq_cut)
    if t_lo > 0:
        target_db[:t_lo] = target_db[t_lo]

    # HF flatten
    t_hi = np.searchsorted(fft_freqs, target_high_freq_cut)
    if t_hi < len(target_db):
        target_db[t_hi:] = target_db[t_hi]

    # ------------------------------------------------------------
    # Step 3: Build AVERAGE reference (much smoother)
    # ------------------------------------------------------------
    avg_mag_cut = 10 ** (avg_db / 20.0)
    avg_mag_cut = smooth_gaussian_octave(
        data=avg_mag_cut, n_fft=n_fft, fs=fs, fraction=12
    )

    avg_db_cut = 20 * np.log10(np.maximum(avg_mag_cut, 1e-12))

    # LF flatten
    a_lo = np.searchsorted(fft_freqs, avg_low_freq_cut)
    if a_lo > 0:
        avg_db_cut[:a_lo] = avg_db_cut[a_lo]

    # HF flatten
    a_hi = np.searchsorted(fft_freqs, avg_high_freq_cut)
    if a_hi < len(avg_db_cut):
        avg_db_cut[a_hi:] = avg_db_cut[a_hi]

    # ------------------------------------------------------------
    # Step 4: Difference curve (peak remover)
    # ------------------------------------------------------------
    diff_db = target_db - avg_db_cut
    diff_db = np.clip(diff_db, -9.0, 9.0)

    diff_mag = 10 ** (diff_db / 20.0)
    diff_mag = smooth_gaussian_octave(
        data=diff_mag, n_fft=n_fft, fs=fs, fraction=10
    )
    diff_db = 20 * np.log10(np.maximum(diff_mag, 1e-12))
    eq_fir = build_min_phase_filter(diff_mag, fs=fs, n_fft=n_fft, truncate_len=truncate_len,f_min=20, f_max=20000, band_limit=True)

    # ------------------------------------------------------------
    # Step 4: Threaded convolution
    # ------------------------------------------------------------
    count_lock = threading.Lock()
    count = 0
    last_print_time = time.time()

    def process_one(i):
        nonlocal count, last_print_time

        if cancel_event and cancel_event.is_set():
            return None

        ir = measurement_array[i]
        eq_ir = sp.signal.fftconvolve(ir, eq_fir, mode="full")
        # trim to original IR length

        with count_lock:
            count += 1
            now = time.time()
            if (count % 100 == 0) or (count == base_n) or (now - last_print_time >= 5.0):
                log_with_timestamp(f"Equalization Progress: {count}/{base_n} IRs processed.", gui_logger)
                last_print_time = now

                if report_progress > 0:
                    a, b = 0.1, 0.35
                    progress = a + (count / base_n) * (b - a)
                    update_gui_progress(report_progress, progress=progress)

        # Optional plotting
        if plot and i < plot_first_n:
            plot_queue.put((i, fft_freqs, avg_db_cut, target_db, diff_db))

        return i, eq_ir[:sample_len]

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_one, i) for i in range(base_n)]
            for f in concurrent.futures.as_completed(futures):
                if cancel_event and cancel_event.is_set():
                    status = 2
                    log_with_timestamp("Equalization cancelled by user.", gui_logger)
                    return output, status

                result = f.result()
                if result is None:
                    status = 2
                    log_with_timestamp("Equalization cancelled by user.", gui_logger)
                    return output, status

                idx, eq_ir = result
                output[idx] = eq_ir

        status = 0
        log_with_timestamp("Low-frequency equalization complete.", gui_logger)

    except Exception as e:
        log_with_timestamp(f"Exception in equalization: {e}", gui_logger)
        status = 1

    # ------------------------------------------------------------
    # Step 5: Plot queued results
    # ------------------------------------------------------------
    if plot:
        while not plot_queue.empty():
            i, freqs, avg_db_plot, target_db_plot, diff_db_plot = plot_queue.get()
            plt.figure(figsize=(10,5))
            plt.semilogx(freqs, avg_db_plot, label="Avg Magnitude (dB)")
            plt.semilogx(freqs, target_db_plot, label="Target Flattened (dB)")
            plt.semilogx(freqs, diff_db_plot, label="Difference (dB)")
            plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.title(f"Low-Frequency EQ IR #{i}")
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.xlim(20, fs/2)
            plt.show()
            plt.close()

    return output, status






def delete_all_files_in_directory(directory, verbose=True, gui_logger=None):
    """
    Deletes all files in the given directory and its subdirectories.
    
    Args:
        directory (str): The path to the directory.
        verbose (bool): If True, prints status messages. If False, suppresses output.
    """
    if not os.path.isdir(directory):
        if verbose:
            log_string = f"'{directory}' is not a valid directory. Unable to delete"
            log_with_timestamp(log_string, gui_logger=None)
        return

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                if verbose:
                    log_string = f"Deleted file: {file_path}"
                    log_with_timestamp(log_string, gui_logger=None)
            except Exception as e:
                if verbose:
                    log_string = f"Could not delete file: {file_path} (Reason: {e})"
                    log_with_timestamp(log_string, gui_logger=None)

def sofa_load_object(sofa_local_fname, gui_logger=None):
    """
    Loads a SOFA file and returns a dictionary of variables starting with 'sofa_'.
    
    Args:
        sofa_local_fname (str): Path to the local SOFA file.
        gui_logger (optional): Logger for GUI messages.
    
    Returns:
        dict: Dictionary of variables starting with 'sofa_'.
    """
    # Initialize an empty dictionary to store variables starting with 'sofa_'
    sofa_vars = {}
    
    try:
        #first try loading with SOFAsonix
        try:
            loadsofa = SOFAFile.load(sofa_local_fname, verbose=False)
            sofa_vars['sofa_data_ir'] = loadsofa.data_ir
            sofa_vars['sofa_samplerate'] = int(loadsofa.Data_SamplingRate[0])
            sofa_vars['sofa_source_positions'] = loadsofa.SourcePosition
            sofa_vars['sofa_convention_name'] = loadsofa.GLOBAL_SOFAConventions
            sofa_vars['sofa_version'] = loadsofa.GLOBAL_Version
            sofa_vars['sofa_convention_version'] = loadsofa.GLOBAL_SOFAConventionsVersion
            sofa_vars['sofa_source_position_type'] = loadsofa.SourcePosition_Type
            sofa_vars['sofa_source_position_units'] = loadsofa.SourcePosition_Units
        except:
            log_string = 'Unable to load SOFA file with SOFAsonix. Attempting to load with sofar'
            log_with_timestamp(log_string, gui_logger=None)
            
            try:
                #if fails, try loading with SOFAR
                loadsofa = sof.read_sofa(sofa_local_fname)#
                sofa_vars['sofa_data_ir'] = loadsofa.Data_IR
                sofa_vars['sofa_samplerate'] = int(loadsofa.Data_SamplingRate)
                sofa_vars['sofa_source_positions'] = loadsofa.SourcePosition
                sofa_vars['sofa_convention_name'] = loadsofa.GLOBAL_SOFAConventions
                sofa_vars['sofa_version'] = loadsofa.GLOBAL_Version
                sofa_vars['sofa_convention_version'] = loadsofa.GLOBAL_SOFAConventionsVersion
                sofa_vars['sofa_source_position_type'] = loadsofa.SourcePosition_Type
                sofa_vars['sofa_source_position_units'] = loadsofa.SourcePosition_Units
                
                log_string = 'Loaded Successfully with sofar'
                log_with_timestamp(log_string, gui_logger=None)
            except:
                try:
                    #if fails, try loading with SOFAR, with verify set to false
                    loadsofa = sof.read_sofa(sofa_local_fname, verify=False)#(sofa_local_fname, verify=False) verify=False ignores convention violations
                    sofa_vars['sofa_data_ir'] = loadsofa.Data_IR
                    sofa_vars['sofa_samplerate'] = int(loadsofa.Data_SamplingRate)
                    sofa_vars['sofa_source_positions'] = loadsofa.SourcePosition
                    sofa_vars['sofa_convention_name'] = loadsofa.GLOBAL_SOFAConventions
                    sofa_vars['sofa_version'] = loadsofa.GLOBAL_Version
                    sofa_vars['sofa_convention_version'] = loadsofa.GLOBAL_SOFAConventionsVersion
                    sofa_vars['sofa_source_position_type'] = loadsofa.SourcePosition_Type
                    sofa_vars['sofa_source_position_units'] = loadsofa.SourcePosition_Units
                    
                    log_string = 'Loaded Successfully with sofar'
                    log_with_timestamp(log_string, gui_logger=None)
                
                except:
                    log_string = 'Unable to load SOFA file. Likely due to unsupported convention version.'
                    log_with_timestamp(log_string, gui_logger=None)
            
                    raise ValueError('Unable to load SOFA file')
        
    
    except Exception as ex:

        log_string = 'SOFA load workflow failed'
        log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
        
    return sofa_vars
    

def load_convert_npy_to_float64(file_path: str, verbose: bool = False) -> np.ndarray:
    """
    Loads a .npy file, converts its data type from float32 to float64, and returns the converted array.
    
    Parameters:
    file_path (str): The path to the .npy file.
    verbose (bool): If True, prints information about the processing steps; otherwise, remains silent.
    
    Returns:
    np.ndarray: The converted array with dtype float64.
    """
    if verbose:
        log_string_a = f"Loading file: {file_path}"
        log_with_timestamp(log_string_a)

    # Load the .npy file
    data = np.load(file_path)
    
    if verbose:
        log_string_a = f"Original data type: {data.dtype}"
        log_with_timestamp(log_string_a)

    # Convert data to float64 if it is float32
    if data.dtype == np.float32:
        data = data.astype(np.float64)
        if verbose:
            log_string_a = "Data converted from float32 to float64."
            log_with_timestamp(log_string_a)
    else:
        if verbose:
            log_string_a = "Data type unchanged."
            log_with_timestamp(log_string_a)

    if verbose:
        log_string_a = "File processing complete."
        log_with_timestamp(log_string_a)

    return data



def get_array_memory_usage_mb(array: np.ndarray) -> None:
    """
    Calculates and prints the memory consumption of a NumPy array in megabytes,
    along with the data type.

    Args:
        array: The NumPy array to analyze.
    """
    memory_usage_bytes = array.nbytes
    memory_usage_mb = memory_usage_bytes / (1024 * 1024)  # Convert bytes to megabytes
    data_type = array.dtype

    print(f"Data type: {data_type}")
    print(f"Memory usage: {memory_usage_mb:.2f} MB")  # Print to 2 decimal places

def crop_array_last_dimension(array: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Crops the last dimension of a multi-dimensional NumPy array to a specified number of samples.

    Args:
        array: A multi-dimensional NumPy array.
        num_samples: The desired number of samples in the last dimension.

    Returns:
        A new NumPy array with the last dimension cropped to num_samples.

    Raises:
        ValueError: If num_samples is greater than the size of the last dimension.
    """

    if num_samples > array.shape[-1]:
        raise ValueError("num_samples cannot be greater than the size of the last dimension.")

    return array[..., :num_samples]



def get_crop_index(input_array, threshold=CN.THRESHOLD_CROP, tail_ignore=0):
    """
    Returns the index in a 1D NumPy array where the amplitude last exceeds a given threshold,
    ignoring a specified number of samples at the end (up to half the array length).
    """
    array_len = len(input_array)
    max_tail_ignore = array_len // 2
    tail_ignore = min(tail_ignore, max_tail_ignore)

    if tail_ignore >= array_len:
        return 0  # Redundant check, but safe

    # Work with the valid portion of the array
    valid_len = array_len - tail_ignore
    valid_range = input_array[:valid_len]
    abs_array = np.abs(valid_range)

    indices_above_threshold = np.where(abs_array > threshold)[0]

    if indices_above_threshold.size == 0:
        return 0

    last_idx = indices_above_threshold[-1]

    # NEW LOGIC:
    # If the signal stayed above threshold until the end of the valid range,
    # assume no cropping and return the end of the full array
    if tail_ignore > 0 and last_idx == valid_len - 1:
        return array_len - 1

    return last_idx

def get_crop_index_relative(input_array,threshold_db=-90.0,tail_ignore=0,head_ignore=0, min_floor_db=-140.0):
    """
    Returns the index in a 1D NumPy array where the amplitude last exceeds
    a threshold relative to the reflection decay level.

    Args:
        input_array (np.ndarray): 1D impulse response
        threshold_db (float): Threshold in dB relative to reflection peak (negative)
        tail_ignore (int): Number of samples to ignore at the end
        head_ignore (int): Number of samples to ignore at the beginning when
                           computing the reference level (e.g. direct sound)
        min_floor_db (float): Absolute safety floor relative to reference

    Returns:
        int: Crop index
    """
    array_len = len(input_array)
    if array_len == 0:
        return 0

    # --- Clamp ignores ---
    max_tail_ignore = array_len // 2
    tail_ignore = min(tail_ignore, max_tail_ignore)
    head_ignore = max(0, min(head_ignore, array_len - 1))

    valid_len = array_len - tail_ignore
    if valid_len <= 0:
        return 0

    # Full valid range used for crop search
    valid_range = input_array[:valid_len]
    abs_array = np.abs(valid_range)

    # --- Reference region (ignore direct sound) ---
    ref_start = head_ignore
    ref_end = valid_len

    if ref_start >= ref_end:
        return 0

    ref_array = abs_array[ref_start:ref_end]

    # Reference level (reflection peak)
    peak = np.max(ref_array)
    if peak <= 0:
        return 0

    # Convert dB threshold to linear (relative to reflection peak)
    threshold_db = max(threshold_db, min_floor_db)
    threshold_linear = peak * (10.0 ** (threshold_db / 20.0))

    indices_above = np.where(abs_array > threshold_linear)[0]

    if indices_above.size == 0:
        return 0

    last_idx = indices_above[-1]

    # If signal stayed above threshold until the valid end,
    # assume no cropping
    if tail_ignore > 0 and last_idx == valid_len - 1:
        return array_len - 1

    return last_idx


def crop_array(input_array, threshold):
    """
    Crops a 1D NumPy array based on a given threshold.

    Args:
        input_array (numpy.ndarray): The 1D NumPy array to crop.
        threshold (float): The threshold value used for cropping.

    Returns:
        numpy.ndarray: The cropped array.
    """
    reversed_array = input_array[::-1]
    try:
        crop_index = len(reversed_array) - np.argmax(np.abs(reversed_array) > threshold) - 1
        cropped_array = input_array[:crop_index]
    except ValueError:
        # np.argmax returns ValueError if the array is all False
        # In this case, return the original array
        cropped_array = input_array
    return cropped_array



def log_with_timestamp(log_string, gui_logger=None, log_type=0, exception=None):
    """
    Logs a message with an optional GUI logger and prefixes GUI logs with a timestamp.

    Args:
        log_string (str): The message to log.
        gui_logger (object, optional): A GUI logger object. Defaults to None.
        log_type (int): 0 = info, 1 = warning, 2 = error.
        exception (Exception, optional): An exception object to log. Defaults to None.
    """

    if CN.LOG_INFO == True:
        if exception:
            if log_type == 2:
                logging.error(log_string, exc_info=exception)
            else:
                logging.info(log_string, exc_info=exception) # If it's a warning or info
        else:
            logging.info(log_string)

    if CN.LOG_GUI == True and gui_logger is not None:
        timestamp = datetime.datetime.now().strftime("%I:%M %p")
        gui_log_string = f"[{timestamp}] {log_string}"
        if log_type == 0:
            gui_logger.log_info(gui_log_string)
        elif log_type == 1:
            gui_logger.log_warning(gui_log_string)
        else:
            gui_logger.log_error(gui_log_string)


def zero_pad_last_dimension(data, n_fft):
    """
    Zero-pads the last dimension of a NumPy array to a specified length.

    Args:
        data: The input NumPy array.
        n_fft: The desired length of the last dimension.

    Returns:
        A new NumPy array with the last dimension zero-padded to n_fft, or the original array if padding wasn't necessary.
        Returns None if there's an error (e.g., n_fft is smaller than the current last dimension).
    """

    try:
        original_shape = data.shape
        last_dim = data.shape[-1]

        if last_dim > n_fft:
            print("Warning: n_fft is smaller than the current last dimension. Data will be truncated.")
            padded_data = data[..., :n_fft]
            return padded_data


        if last_dim == n_fft:  # No padding needed
            #print("No padding needed")
            return data

        padding_length = n_fft - last_dim
        padding_shape = list(data.shape[:-1]) + [padding_length]  # Shape of the padding
        padding = np.zeros(padding_shape, dtype=data.dtype) # Preserve the original dtype
        padded_data = np.concatenate((data, padding), axis=-1)

        return padded_data

    except Exception as e:
        log_string = f"An error occurred: {e}"
        if CN.LOG_INFO == True:
            logging.error(log_string, exc_info=e)
        return None

def zero_pad_1d(arr, target_length):
    """
    Zero-pads a 1D NumPy array to a specified length.

    Args:
        arr (np.ndarray): A 1D NumPy array.
        target_length (int): The desired length of the output array.

    Returns:
        np.ndarray: A zero-padded 1D NumPy array of shape (target_length,).
    """

    current_length = arr.shape[0]

    # If the array is already the desired length or longer, return the original or truncated version
    if current_length >= target_length:
        return arr[:target_length]

    # Create a zero-padded array
    padded_arr = np.zeros(target_length, dtype=arr.dtype)
    
    # Copy the original array into the padded array
    padded_arr[:current_length] = arr

    return padded_arr




def combine_dims(a, start=0, count=2):
    """ Reshapes numpy array a by combining count dimensions, 
        starting at dimension index start 
        s[:start]: Keeps the dimensions before start.
        (-1,): This tells NumPy to infer the new dimension size.
        s[start+count:]: Keeps the dimensions after the combined ones
        example:
            a = np.random.rand(2, 3, 4, 5)  # Shape: (2, 3, 4, 5)
            b = combine_dims(a, start=1, count=2)  # Merge dimensions 1 and 2
            print(a.shape)  # (2, 3, 4, 5)
            print(b.shape)  # (2, 12, 5)  -> 3 * 4 = 12
        """
    s = a.shape
    return np.reshape(a, s[:start] + (-1,) + s[start+count:])


def azimuth_to_circular(angle):
    """
    Convert an azimuth angle from -180..180 (left negative, right positive)
    into circular CCW format 0..360 degrees.

    Examples:
        -30  -> 30
        30   -> 330
        180  -> 180
        -180 -> 180
        0    -> 0
    """
    return (360 - angle) % 360


def circular_to_azimuth(circ_angle):
    """
    Convert a circular CCW azimuth (0..360) back into -180..180 system.

    Examples:
        30   -> -30
        330  -> 30
        210  -> -150
        180  -> 180
        0    -> 0
    """
    # Normalize to 0–360 just in case
    circ_angle = circ_angle % 360

    # 0–180 maps to positive/right (mirrored)
    # 180–360 maps to negative/left when converted back
    angle = (360 - circ_angle) % 360

    # Convert 181–359 into -179..-1
    if angle > 180:
        angle -= 360

    return angle

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

def round_down_even(n):
    return 2 * int(n // 2) 





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
  

    
def write2wav(file_name, data, samplerate=CN.SAMP_FREQ, prevent_clipping=0, bit_depth='PCM_24', enable_resample=False, resample_mode='fast'):
    """
    Write a WAV file, optionally resampling the signal if samplerate != CN.SAMP_FREQ. data must be shaped (samples, channels); (channels, samples) will be auto-detected and transposed.

    :param data: numpy array, time domain signal
    :param file_name: output wav filename
    :param samplerate: int, requested output sample rate
    :param prevent_clipping: 1 to normalize to avoid clipping
    :param bit_depth: WAV subtype (default PCM_24)
    :param enable_resample: bool, if True resamples input to `samplerate` - disabled by default as resampling is handled elsewhere
    :param resample_mode: 'fast' | 'best' | etc., passed to resample_signal()
    """

    # Ensure numpy float32
    data = np.asarray(data, dtype=np.float32)
    
    # ---- Ensure shape = (samples, channels) ----
    if data.ndim == 2:
        n0, n1 = data.shape

        # Heuristic:
        # samples dimension should be much larger than channels
        # If not, assume (channels, samples) and transpose
        if n0 < n1 and n1 > 1024 and (n1 / max(n0, 1)) > 4:
            data = data.T

    # Optional resampling
    if enable_resample and samplerate != CN.SAMP_FREQ:
        data = resample_signal(
            signal=data,
            original_rate=CN.SAMP_FREQ,
            new_rate=samplerate,
            axis=0,
            scale=False,
            mode=resample_mode
        )

    # Optional clipping protection
    if prevent_clipping == 1:
        max_amp = np.max(np.abs(data))
        if max_amp > 1:
            data = data / max_amp

    # Write WAV
    sf.write(file_name, data, samplerate, subtype=bit_depth)
    
def read_wav_file(audiofilename):
    """
    Opens a WAV file and returns the sample rate and normalized audio samples.

    Uses `soundfile` for better support of various bit depths and formats.
    Always returns float32 or float64 in the range [-1.0, 1.0].

    Parameters:
        audiofilename (str): Path to the WAV file.

    Returns:
        samplerate (int): Sampling rate of the audio file.
        samples (np.ndarray): Normalized audio data (float32/float64).
    """
    samples, samplerate = sf.read(audiofilename, always_2d=False)
    return samplerate, samples


def wav_needs_update(path, samplerate, bit_depth, gui_logger=None):
    """
    Returns True if the WAV file does not exist OR if its metadata does not match.
    Uses hf.log_with_timestamp for all logs.
    """

    p = Path(path)

    # File does not exist
    if not p.is_file():
        log_with_timestamp(f"WAV missing → will write: {p.name}", gui_logger)
        return True

    # Try reading metadata
    try:
        info = sf.info(str(p))
    except Exception:
        log_with_timestamp(f"WAV unreadable → rewriting: {p.name}", gui_logger)
        return True

    # Check sample rate
    if info.samplerate != samplerate:
        log_with_timestamp(
            f"Sample rate mismatch for {p.name}: "
            f"file={info.samplerate}, expected={samplerate}",
            gui_logger
        )
        return True

    # Check bit depth (subtype)
    if info.subtype != bit_depth:
        log_with_timestamp(
            f"Bit depth mismatch for {p.name}: "
            f"file={info.subtype}, expected={bit_depth}",
            gui_logger
        )
        return True
    

    # All good → no update needed
    return False   


def resample_signal(signal, original_rate=CN.SAMP_FREQ, new_rate=48000,
                    axis=0, scale=False, mode='fast'):
    """
    Resample a signal and enforce the mathematically correct output length.
    """

    # No resampling needed
    if original_rate == new_rate:
        return signal

    # -------- Expected exact output length --------
    # length changes only along the resampling axis
    in_len = signal.shape[axis]
    target_len = round(in_len * float(new_rate) / float(original_rate))

    # -------- Perform resampling --------
    if mode == 'best' or mode == 'Quality':
        out = librosa.resample(
            signal,
            orig_sr=original_rate,
            target_sr=new_rate,
            res_type='kaiser_best',
            axis=axis,
            scale=scale
        )
    elif mode == 'fast' or mode == 'Performance':
        out = resample_poly(
            signal,
            new_rate,
            original_rate,
            axis=axis,
            window=('kaiser', 5.0)
        )
    else:
        out = resample_poly(
            signal,
            new_rate,
            original_rate,
            axis=axis,
            window=('kaiser', 14.0)
        )

    # -------- Enforce exact length --------
    out_len = out.shape[axis]

    # CROPPING
    if out_len > target_len:
        log_with_timestamp(
            f"Resample length correction: cropping from {out_len} to {target_len}"
        )
        slicer = [slice(None)] * out.ndim
        slicer[axis] = slice(0, target_len)
        out = out[tuple(slicer)]

    # PADDING
    elif out_len < target_len:
        log_with_timestamp(
            f"Resample length correction: padding from {out_len} to {target_len}"
        )
        pad_width = [(0, 0)] * out.ndim
        pad_width[axis] = (0, target_len - out_len)
        out = np.pad(out, pad_width)

    return out
   

    
def normalize_array(ir):
    """
    Normalizes an impulse response array to the range [-1, 1] based on its maximum absolute value.
    
    Parameters:
        ir (np.ndarray): Input impulse response array to normalize.
        
    Returns:
        np.ndarray: Normalized impulse response array.
    """
    max_val = np.max(np.abs(ir))
    if max_val > 0:
        ir = ir / max_val
    return ir       






def calculate_rt60(ir: np.ndarray, fs=CN.FS, bands=CN.OCTAVE_BANDS):
    """
    Calculate the average RT60 for given impulse response in 1/3 octave bands.
    
    Parameters:
        ir (np.ndarray): The impulse response (1D array)
        fs (int): The sampling frequency in Hz
        bands (np.ndarray): Array of center frequencies for the octave bands
    
    Returns:
        float: The average RT60 across the octave bands
    """
    rt60_values = []

    # For each octave band, calculate the RT60
    for band in bands:
        # Design the bandpass filter for the current band
        low_cut = band / (2**(1/6))  # Lower bound for 1/3 octave
        high_cut = band * (2**(1/6))  # Upper bound for 1/3 octave

        # Bandpass filter the impulse response
        nyquist = fs / 2
        low = low_cut / nyquist
        high = high_cut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_ir = signal.filtfilt(b, a, ir)

        # Calculate the energy decay curve (EDC) for the filtered signal
        squared_ir = filtered_ir ** 2
        edc = np.cumsum(squared_ir[::-1])[::-1]  # Integrate from the end

        # Find the decay time (RT60)
        initial_level = np.max(edc)
        final_level = initial_level / 1000  # 60 dB down
        crossing_index = np.where(edc <= final_level)[0][0]
        
        # Estimate RT60 by linear regression on the decay portion of the EDC
        time = np.arange(len(edc)) / fs
        rt60 = time[crossing_index]  # This is a simplified estimate
        
        rt60_values.append(rt60)

    # Return the average RT60 over all bands
    return np.mean(rt60_values)

# Example usage:
# ir = np.array([...])  # Your impulse response
# fs = 44100  # Sample rate of your IR
# average_rt60 = calculate_rt60(ir, fs)
# print("Average RT60 across 1/3-octave bands:", average_rt60)


def bandpass_filter(ir, fs, center_freq, fraction=3, order=4):
    """
    Band-pass filters the input IR to isolate a specific 1/3-octave band.
    
    Parameters:
        ir (np.array): Input impulse response.
        fs (int): Sample rate in Hz.
        center_freq (float): Center frequency of the band.
        fraction (int): Defines the fraction of the octave (default is 3 for 1/3-octave).
        order (int): Order of the Butterworth filter.
    
    Returns:
        np.array: Filtered impulse response.
    """
    factor = 2 ** (1 / (2 * fraction))  # Band edges scale factor
    low = center_freq / factor
    high = center_freq * factor
    sos = signal.butter(order, [low, high], btype='bandpass', fs=fs, output='sos')
    return signal.sosfilt(sos, ir)

def calculate_schroeder_curve(ir):
    """
    Calculates the Schroeder integration curve in dB.
    
    Parameters:
        ir (np.array): Filtered impulse response.
    
    Returns:
        np.array: Decay curve in dB (Schroeder curve).
    """
    energy = np.cumsum(ir[::-1] ** 2)[::-1]  # Reverse cumulative sum of energy
    energy /= np.max(energy)  # Normalize to peak = 1
    return 10 * np.log10(np.maximum(energy, 1e-12))  # Convert to dB (avoid log(0))

def linear_fit_decay(sch_db, fs, start_db, end_db):
    """
    Performs a linear regression on the Schroeder curve between two dB values
    to estimate RT60 via extrapolation.
    
    Parameters:
        sch_db (np.array): Schroeder curve in dB.
        fs (int): Sample rate in Hz.
        start_db (float): Start level in dB (e.g., -5).
        end_db (float): End level in dB (e.g., -25 or -35).
    
    Returns:
        float or None: Estimated RT60 (T20, T30) in seconds, or None if invalid.
    """
    start_idx = np.argmax(sch_db <= start_db)
    end_idx = np.argmax(sch_db <= end_db)
    if end_idx <= start_idx:
        return None  # Invalid range
    t = np.arange(end_idx - start_idx) / fs
    y = sch_db[start_idx:end_idx]
    slope, intercept, *_ = linregress(t, y)
    return -60 / slope if slope != 0 else None

def calculate_topt_from_schroeder(sch_db, fs):
    """
    Calculates Topt: the best linear fit over a variable decay range.
    Starts at -5 dB and searches for the best end point up to -45 dB.
    
    Parameters:
        sch_db (np.array): Schroeder curve in dB.
        fs (int): Sample rate in Hz.
    
    Returns:
        float or None: Estimated Topt in seconds.
    """
    best_fit_error = float("inf")
    best_rt = None
    start_idx = np.argmax(sch_db <= -5)
    
    for end_db in range(-10, -45, -1):
        end_idx = np.argmax(sch_db <= end_db)
        if end_idx <= start_idx:
            continue
        t = np.arange(end_idx - start_idx) / fs
        y = sch_db[start_idx:end_idx]
        slope, intercept, r_value, _, _ = linregress(t, y)
        residuals = np.sum((y - (slope * t + intercept)) ** 2)
        if residuals < best_fit_error and slope < 0:
            best_fit_error = residuals
            best_rt = -60 / slope  # RT60 from slope
    
    return best_rt

def compute_band_rt60s(ir, fs=CN.FS, bands=CN.OCTAVE_BANDS):
    """
    Calculates Topt-based RT60 estimates across all specified 1/3-octave bands.
    
    Parameters:
        ir (np.array): Raw impulse response.
        fs (int): Sample rate in Hz.
        bands (list): Center frequencies of bands.
    
    Returns:
        dict: Mapping of band center frequency to estimated RT60 (Topt).
    """
    results = {}
    for center_freq in bands:
        # Band-pass filter the IR at this frequency band
        filtered_ir = bandpass_filter(ir, fs, center_freq)
        
        # Compute Schroeder decay curve
        sch_db = calculate_schroeder_curve(filtered_ir)
        
        # Compute Topt
        rt = calculate_topt_from_schroeder(sch_db, fs)
        results[center_freq] = rt
    
    return results


    #Example Usage
    # # Load your impulse response as `ir` (e.g., from WAV) and define the sample rate `fs`
    # band_rt60s = compute_band_rt60s(ir, fs)
    # # Print estimated RT60s for each frequency band
    # for band, rt in band_rt60s.items():
    #     print(f"{band} Hz: Topt ≈ {rt:.2f} s" if rt is not None else f"{band} Hz: N/A")
    # Calculate average Topt across all bands
    # topt_values = band_rt60s["Topt"]
    # topt_mean = np.nanmean(topt_values)
    # print(f"Average Topt across bands: {topt_mean:.3f} seconds")


def signal_lowpass_filter(data, cutoff, fs, order=5, method=1, filtfilt=False):
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
    if filtfilt == True:#doubles the effective order of the filtering when compared to a simple forward filter. 
        order=int(order/2)
    
    if method == 1:
        #method 1
        sos = butter(order, cutoff, fs=fs, btype='low', output='sos', analog=False)

    elif method == 2:
        #method 2
        sos = sps.cheby1(order, 3, cutoff, btype='low', analog=False, output='sos', fs=fs)
    elif method == 3:
        #method 3
        sos = sps.cheby2(order, 40, cutoff, btype='low', analog=False, output='sos', fs=fs)
    elif method == 4:
        #method 4
        sos = sps.ellip(order, 5, 40, cutoff, btype='low', analog=False, output='sos', fs=fs)
    else:
        #method 5
        sos = sps.bessel(order, cutoff, btype='low', analog=False, output='sos', fs=fs)
        
    #A forward-backward digital filter using cascaded second-order sections.
    if filtfilt == True:
        y = sps.sosfiltfilt(sos, data, padtype='even', padlen=30000) #Uses sosfiltfilt instead of sosfilt → Ensures zero-phase distortion and more stable filtering.
    else:
        y = sps.sosfilt(sos, data)
        
        
    return y


def signal_highpass_filter(data, cutoff, fs, order=5, method=1, filtfilt=False):
    """
    Function takes a time domain signal as an input and applies high pass filter
    :param data: numpy array, time domain signal
    :param cutoff: int, cutoff frequency in Hz
    :param fs: int, sample frequency in Hz
    :param order: int, filter order
    :param method: int, 1=butter, 2=chevy1, 3=cheby2, 4=ellip, 5=bessel
    :return: numpy array, high pass filtered signal
    """  
    if filtfilt == True:#doubles the effective order of the filtering when compared to a simple forward filter. 
        order=int(order/2)
    
    if method == 1:
        #method 1
        sos = butter(order, cutoff, fs=fs, btype='high', output='sos', analog=False)
        
    elif method == 2:
        #method 2
        sos = sps.cheby1(order, 3, cutoff, btype='high', analog=False, output='sos', fs=fs)
    elif method == 3:
        #method 3
        sos = sps.cheby2(order, 40, cutoff, btype='high', analog=False, output='sos', fs=fs)
    elif method == 4:
        #method 4
        sos = sps.ellip(order, 5, 40, cutoff, btype='high', analog=False, output='sos', fs=fs)
    else:
        #method 5
        sos = sps.bessel(order, cutoff, btype='high', analog=False, output='sos', fs=fs)
        
    #A forward-backward digital filter using cascaded second-order sections.
    if filtfilt == True:
        y = sps.sosfiltfilt(sos, data, padtype='even', padlen=30000) #Uses sosfiltfilt instead of sosfilt → Ensures zero-phase distortion and more stable filtering.
    else:
        y = sps.sosfilt(sos, data)
    
    return y

def get_filter_sos( cutoff, fs, order=5, method=1, filtfilt=False, b_type='high'):
    """
    Function takes a time domain signal as an input and calculates SOS filter
    :param cutoff: int, cutoff frequency in Hz
    :param fs: int, sample frequency in Hz
    :param order: int, filter order
    :param method: int, 1=butter, 2=chevy1, 3=cheby2, 4=ellip, 5=bessel
    :param b_type: str, 'low' or 'high'
    :return: sos filter object
    """  
    if filtfilt == True:#doubles the effective order of the filtering when compared to a simple forward filter. 
        order=int(order/2)
        
    if cutoff < CN.MIN_FILT_FREQ:
        return None

    if method == 1:
        #method 1
        sos = butter(order, cutoff, fs=fs, btype=b_type, output='sos', analog=False)
        
    elif method == 2:
        #method 2
        sos = sps.cheby1(order, 3, cutoff, btype=b_type, analog=False, output='sos', fs=fs)
    elif method == 3:
        #method 3
        sos = sps.cheby2(order, 40, cutoff, btype=b_type, analog=False, output='sos', fs=fs)
    elif method == 4:
        #method 4
        sos = sps.ellip(order, 5, 40, cutoff, btype=b_type, analog=False, output='sos', fs=fs)
    else:
        #method 5
        sos = sps.bessel(order, cutoff, btype=b_type, analog=False, output='sos', fs=fs)
        

    return sos

def get_multiple_filter_sos(
    cutoffs: list,
    fs: float,
    order: int,
    filtfilt: bool = False,
    b_type: str = 'low'
) -> list:
    """
    Create a list of second-order section (SOS) filter objects based on multiple cutoff frequencies.

    Parameters:
        cutoffs (list): List of cutoff frequencies.
        fs (float): Sampling frequency.
        order (int): Filter order.
        filtfilt (bool): Whether zero-phase filtering is used (optional).
        b_type (str): Filter type ('low', 'high', 'bandpass', etc.).

    Returns:
        list: List of SOS filter objects (one per cutoff).
    """
    return [
        get_filter_sos(
            cutoff=cutoff,
            fs=fs,
            order=order,
            filtfilt=filtfilt,
            b_type=b_type
        )
        for cutoff in cutoffs
    ]


def apply_sos_filter(data, sos, filtfilt=False, axis=-1):
    """
    Applies a high or low pass filter to a time-domain signal using second-order sections (SOS).
    
    Parameters:
        data (np.ndarray): Input signal.
        sos (np.ndarray): Second-order sections filter coefficients.
        filtfilt (bool): If True, applies zero-phase filtering with filtfilt.
        axis (int): Axis along which to apply the filter.

    Returns:
        np.ndarray: Filtered signal.
    """
    data = np.asarray(data)
    
    if filtfilt:
        # Ensure padlen does not exceed the length of the signal along the axis
        signal_length = data.shape[axis]
        padlen = min(20000, signal_length - 1) if signal_length > 1 else 0

        y = sps.sosfiltfilt(sos, data, padtype='even', padlen=padlen, axis=axis)
    else:
        y = sps.sosfilt(sos, data, axis=axis)
    
    return y


def group_delay(sig):
    """
    function to calculate group delay
    """  
    
    b = np.fft.fft(sig)
    n_sig = np.multiply(sig, np.arange(len(sig)))
    br = np.fft.fft(n_sig)
    return np.divide(br, b + 0.01).real







def smooth_freq(
    data,
    crossover_f=1000,
    win_size_a=150,
    win_size_b=750,
    n_fft=CN.N_FFT, # Example default
    fs=CN.FS,    # Example default
    to_full=False,
    log_domain=True
):
    """
    Applies zero-phase two-stage smoothing to a magnitude FFT spectrum.
    """
    is_half = len(data) == n_fft // 2 + 1
    nyq_bin = n_fft // 2
    spectrum = data[:nyq_bin + 1].copy() if is_half else data.copy()

    if log_domain:
        # Avoid log(0)
        spectrum = 20 * np.log10(np.maximum(spectrum, 1e-12))

    # Convert Hz to bins
    crossover_bin = int(round(crossover_f * n_fft / fs))
    win_a_bins = max(1, int(round(win_size_a * n_fft / fs)))
    win_b_bins = max(1, int(round(win_size_b * n_fft / fs)))
    
    # Ensure window sizes are odd for perfect centering
    win_a_bins = win_a_bins if win_a_bins % 2 != 0 else win_a_bins + 1
    win_b_bins = win_b_bins if win_b_bins % 2 != 0 else win_b_bins + 1

    # uniform_filter1d with origin=0 is zero-phase (centered)
    # mode='reflect' handles the boundaries without adding a DC offset bias
    smooth_a = sp.ndimage.uniform_filter1d(spectrum, size=win_a_bins, mode='reflect', origin=0)
    smooth_b = sp.ndimage.uniform_filter1d(spectrum, size=win_b_bins, mode='reflect', origin=0)

    # Cross-fade between smooth_a and smooth_b to prevent "steps" at the crossover
    # We use a small 10-bin ramp for the transition
    ramp_width = 10 
    start = max(0, crossover_bin - ramp_width // 2)
    end = min(len(spectrum), crossover_bin + ramp_width // 2)
    
    combined = np.empty_like(spectrum)
    combined[:start] = smooth_a[:start]
    combined[end:] = smooth_b[end:]
    
    # Linear interpolation for the transition zone
    if end > start:
        alpha = np.linspace(0, 1, end - start)
        combined[start:end] = (1 - alpha) * smooth_a[start:end] + alpha * smooth_b[start:end]

    if log_domain:
        combined = 10**(combined / 20)

    # Format return
    if is_half:
        if to_full:
            full_spec = np.empty(n_fft, dtype=combined.dtype)
            full_spec[:nyq_bin + 1] = combined
            full_spec[nyq_bin + 1:] = combined[1:nyq_bin][::-1]
            return full_spec
        return combined
    else:
        result = np.empty_like(data)
        result[:nyq_bin + 1] = combined[:nyq_bin + 1]
        result[nyq_bin + 1:] = combined[1:nyq_bin][::-1]
        return result

def smooth_freq_octaves(
    data,
    fund_freq=120,
    win_size_base=15,
    n_fft=CN.N_FFT,
    fs=CN.FS,
    to_full=False,
    log_domain=True
):
    """
    Applies multi-band smoothing based on octave scaling of smoothing windows.
    This function performs a sequence of smoothing operations, increasing the smoothing
    window size with frequency (per-octave) to simulate logarithmic perceptual resolution.

    Parameters:
        data (np.ndarray): Magnitude spectrum (either full or half spectrum).
        fund_freq (float): Base frequency in Hz that defines the start of octave scaling.
        win_size_base (float): Base smoothing window size in Hz at the lowest octave.
        n_fft (int): FFT size used to calculate the spectrum.
        fs (int): Sampling rate in Hz.
        to_full (bool): If True and input is half-spectrum, mirror to full spectrum on return.
        log_domain (bool): If True, smoothing is applied in dB.

    Returns:
        np.ndarray: Octave-smoothed spectrum (same type as input unless to_full=True).
    """
    # Determine input format
    is_half = len(data) == n_fft // 2 + 1
    nyq_bin = n_fft // 2
    max_freq = fs / 2
    num_octaves = int(np.log2(max_freq / fund_freq))

    # Work on a copy
    smoothed = data.copy()

    # Iteratively smooth each octave band
    for i in range(num_octaves):
        factor = 2 ** i
        cutoff = fund_freq * factor
        win_a = win_size_base
        win_b = win_size_base * factor

        smoothed = smooth_freq(
            smoothed,
            crossover_f=cutoff,
            win_size_a=win_a,
            win_size_b=win_b,
            n_fft=n_fft,
            fs=fs,
            to_full=False,      # Keep half-spectrum during iteration
            log_domain=log_domain
        )

    # Return in desired format
    if is_half and to_full:
        full_spec = np.empty(n_fft, dtype=smoothed.dtype)
        full_spec[:nyq_bin + 1] = smoothed
        full_spec[nyq_bin + 1:] = smoothed[1:nyq_bin][::-1]
        return full_spec

    return smoothed
     



def smooth_fractional_octave(
    data,
    fraction=12,      # e.g., 3 for 1/3 octave, 12 for 1/12 octave
    n_fft=CN.N_FFT,
    fs=CN.FS,
    to_full=False,
    log_domain=True
):
    """
    Applies fractional octave smoothing to a magnitude spectrum.
    The smoothing window width increases proportionally with frequency.
    """
    is_half = len(data) == n_fft // 2 + 1
    nyq_bin = n_fft // 2
    spectrum = data[:nyq_bin + 1].copy() if is_half else data.copy()

    if log_domain:
        spectrum = 20 * np.log10(np.maximum(spectrum, 1e-12))

    num_bins = len(spectrum)
    bin_freqs = np.arange(num_bins) * (fs / n_fft)
    
    # Calculate the fractional octave width factor
    # For 1/N octave, the upper bound is f * 2^(1/2N) and lower is f * 2^(-1/2N)
    width_factor = 2**(1 / (2 * fraction))
    
    # Calculate bin boundaries for each frequency
    # We ensure at least a 1-bin wide window (no smoothing) at very low frequencies
    lower_bounds = np.round(np.arange(num_bins) / width_factor).astype(int)
    upper_bounds = np.round(np.arange(num_bins) * width_factor).astype(int)
    
    # Clip bounds to valid array indices
    lower_bounds = np.clip(lower_bounds, 0, num_bins - 1)
    upper_bounds = np.clip(upper_bounds, 0, num_bins - 1)

    # Fast sliding window average using cumulative sum
    # We pad the start to handle the boundary logic easily
    cumsum = np.cumsum(np.insert(spectrum, 0, 0))
    
    smoothed = np.empty_like(spectrum)
    for i in range(num_bins):
        low = lower_bounds[i]
        high = upper_bounds[i]
        # cumsum[high+1] is the sum up to index 'high'
        # cumsum[low] is the sum up to index 'low-1'
        smoothed[i] = (cumsum[high + 1] - cumsum[low]) / (high - low + 1)

    if log_domain:
        smoothed = 10**(smoothed / 20)

    # Return in correct format (Mirroring for full spectrum)
    if is_half:
        if to_full:
            full_spec = np.empty(n_fft, dtype=smoothed.dtype)
            full_spec[:nyq_bin + 1] = smoothed
            full_spec[nyq_bin + 1:] = smoothed[1:nyq_bin][::-1]
            return full_spec
        return smoothed
    else:
        result = np.empty_like(data)
        result[:nyq_bin + 1] = smoothed[:nyq_bin + 1]
        result[nyq_bin + 1:] = smoothed[1:nyq_bin][::-1]
        return result




def smooth_gaussian_octave(
    data,
    fraction=12,
    n_fft=CN.N_FFT,
    fs=CN.FS,
    to_full=False,
    log_domain=True
):
    """
    High-performance Gaussian fractional octave smoothing using log-resampling.
    Optimal for large N_FFT sizes.
    """
    is_half = len(data) == n_fft // 2 + 1
    nyq_bin = n_fft // 2
    spectrum = data[:nyq_bin + 1].copy() if is_half else data.copy()

    if log_domain:
        spectrum = 20 * np.log10(np.maximum(spectrum, 1e-12))

    num_bins = len(spectrum)
    
    # 1. Create a logarithmic frequency axis
    # We start from the first bin (>0 Hz) to avoid log(0)
    indices = np.arange(num_bins)
    # Use a high-density log-grid to preserve detail (e.g., 4x the original resolution)
    log_grid_size = num_bins 
    log_indices = np.geomspace(1, num_bins - 1, log_grid_size)
    
    # 2. Interpolate linear spectrum to logarithmic space
    # This "stretches" the low frequencies and "compresses" the high frequencies
    interp_func = interp1d(indices, spectrum, kind='linear', fill_value="extrapolate")
    log_spectrum = interp_func(log_indices)

    # 3. Apply a FIXED sigma Gaussian filter in log-space
    # In log-space, 1 octave is a constant distance. 
    # Points per octave = log_grid_size / (total octaves)
    num_octaves = np.log2((num_bins-1) / 1)
    points_per_octave = log_grid_size / num_octaves
    
    # Calculate sigma to match the fractional octave width
    # 2.5 is a scaling factor to match standard acoustic smoothing curves
    sigma = (points_per_octave / fraction) / 2.5
    
    # gaussian_filter1d is highly optimized C/Fortran code
    log_smoothed = gaussian_filter1d(log_spectrum, sigma=sigma, mode='reflect')

    # 4. Interpolate back to the original linear frequency bins
    back_interp_func = interp1d(log_indices, log_smoothed, kind='linear', 
                                fill_value="extrapolate", bounds_error=False)
    smoothed = back_interp_func(indices)
    
    # Fix the DC bin (index 0) which geomspace can't handle perfectly
    smoothed[0] = spectrum[0] 

    if log_domain:
        smoothed = 10**(smoothed / 20)

    # Return formatting
    if is_half:
        if to_full:
            full_spec = np.empty(n_fft, dtype=smoothed.dtype)
            full_spec[:nyq_bin + 1] = smoothed
            full_spec[nyq_bin + 1:] = smoothed[1:nyq_bin][::-1]
            return full_spec
        return smoothed
    else:
        result = np.empty_like(data)
        result[:nyq_bin + 1] = smoothed[:nyq_bin + 1]
        result[nyq_bin + 1:] = smoothed[1:nyq_bin][::-1]
        return result


def build_min_phase_filter( 
    smoothed_mag,
    freq_axis=None,
    fs=CN.FS,
    n_fft=CN.N_FFT,
    truncate_len=4096,
    f_min=20,
    f_max=20000,
    band_limit=False,
    normalize=False,
    norm_freq_range=(60, 300),
    norm_target=0.5,
    hf_relax_to_zero=False,
    hf_trust_freq=7000.0,
    hf_anchor_freq=20000.0,
    hf_decay_steps=5,          # number of synthetic HF points
    hf_decay_power=1.5,        # controls how fast it returns to 0 dB
    apply_smooth=False
):
    """
    Build a minimum-phase FIR filter from a magnitude spectrum (half or full).

    Optional HF relaxation:
    - Trust data up to hf_trust_freq
    - Insert a knee at hf_knee_freq
    - Force 0 dB at hf_anchor_freq
    - Smoothly interpolate in between using PCHIP
    """
    DEBUG = False
    
    try:
        if DEBUG:
            logging.info(f"smoothed_mag shape: {np.shape(smoothed_mag)}")

        fft_freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    
        # ------------------------------------------------------------
        # Optional HF relaxation (multi-point decay scaffold)
        # ------------------------------------------------------------
        if hf_relax_to_zero and freq_axis is not None:
            freq_axis = np.asarray(freq_axis, dtype=np.float64)
            smoothed_mag = np.asarray(smoothed_mag, dtype=np.float64)
        
            # Normalize first if requested
            if normalize:
                norm_mask = (
                    (freq_axis >= norm_freq_range[0]) &
                    (freq_axis <= norm_freq_range[1])
                )
                mean_val = np.mean(smoothed_mag[norm_mask])
                if mean_val != 0:
                    smoothed_mag = smoothed_mag / mean_val * norm_target
        
            # Keep trusted region
            keep_mask = freq_axis <= hf_trust_freq
            freqs_kept = freq_axis[keep_mask]
            mags_kept = smoothed_mag[keep_mask]
        
            mag_start = mags_kept[-1]
        
            # Generate decay scaffold (log-spaced freqs)
            decay_freqs = np.geomspace(
                hf_trust_freq,
                hf_anchor_freq,
                hf_decay_steps + 2
            )[1:-1]  # exclude endpoints
        
            # Normalized progress 0 → 1
            t = np.linspace(0, 1, hf_decay_steps + 2)[1:-1]
        
            # Power-law decay toward 1.0 (0 dB)
            decay_mags = 1.0 + (mag_start - 1.0) * (1.0 - t) ** hf_decay_power
        
            freqs_kept = np.concatenate([freqs_kept, decay_freqs, [hf_anchor_freq]])
            mags_kept = np.concatenate([mags_kept, decay_mags, [1.0]])
        
            freq_axis = freqs_kept
            smoothed_mag = mags_kept
    
        # ------------------------------------------------------------
        # Interpolate onto FFT bins
        # ------------------------------------------------------------
        if freq_axis is not None:
            interp_func = PchipInterpolator(freq_axis, smoothed_mag, extrapolate=False)
            mag_interp = interp_func(fft_freqs)
            if DEBUG:
                logging.info(f"mag_interp shape after interpolation: {np.shape(mag_interp)}")
    
            # Clamp edges
            mag_interp[fft_freqs < freq_axis[0]] = smoothed_mag[0]
            mag_interp[fft_freqs > freq_axis[-1]] = smoothed_mag[-1]
    
            # Octave smoothing
            #mag_interp = smooth_freq_octaves(data=mag_interp,n_fft=n_fft,win_size_base=7,fund_freq=100)
            mag_interp = smooth_gaussian_octave(data=mag_interp, n_fft=n_fft, fraction=12)
            if DEBUG:
                # After octave smoothing
                logging.info(f"mag_interp shape after smoothing: {np.shape(mag_interp)}")
        else:
            mag_interp = np.abs(smoothed_mag[:n_fft // 2 + 1])
        
    
        mag = mag_interp.copy()
        
        if apply_smooth:
            # Octave smoothing
            #mag = smooth_freq_octaves(data=mag,n_fft=n_fft,win_size_base=6,fund_freq=100)
            mag = smooth_gaussian_octave(data=mag, n_fft=n_fft, fraction=12)
    
        # ------------------------------------------------------------
        # Optional band limiting
        # ------------------------------------------------------------
        if band_limit:
            band_mask = (fft_freqs >= f_min) & (fft_freqs <= f_max)
            mag_band = np.zeros_like(mag)
            mag_band[band_mask] = mag[band_mask]
            mag_band[fft_freqs < f_min] = mag[np.argmax(fft_freqs >= f_min)]
            mag_band[fft_freqs > f_max] = mag[np.argmax(fft_freqs > f_max) - 1]
            mag = mag_band
            # Octave smoothing
            #mag = smooth_freq_octaves(data=mag,n_fft=n_fft,win_size_base=8,fund_freq=100)
            mag = smooth_gaussian_octave(data=mag, n_fft=n_fft, fraction=12)
    
        # ------------------------------------------------------------
        # Cepstrum → minimum phase
        # ------------------------------------------------------------
        log_mag = np.log(np.maximum(mag, 1e-8))
        # Before cepstrum
        if DEBUG:
            logging.info(f"log_mag shape: {np.shape(log_mag)}")
    
        cepstrum = np.fft.irfft(log_mag, n=n_fft)
        cepstrum[1:n_fft // 2] *= 2
        cepstrum[n_fft // 2 + 1:] = 0
    
        min_phase_spec = np.exp(np.fft.rfft(cepstrum, n=n_fft))
    
        impulse = np.fft.irfft(min_phase_spec, n=n_fft)
        impulse = impulse[:truncate_len]
    
        # Fade-out
        fade_start = truncate_len // 4
        fade_len = truncate_len - fade_start
        if fade_len > 0:
            fade_out = np.hanning(2 * fade_len)[fade_len:]
            impulse[fade_start:] *= fade_out
        
        if DEBUG:
            # Final impulse
            logging.info(f"impulse shape: {np.shape(impulse)}")
            
        return impulse

    except Exception as e:
        log_with_timestamp(f"Failed to convert to min phase FIR: {e}")


def build_min_phase_and_save(
    mag_response,
    file_path,
    fs=CN.FS,
    n_fft=CN.N_FFT,
    truncate_len=4096,
    f_min=20,
    f_max=20000,
    band_limit=False,
    prevent_clipping=1,
    bit_depth="PCM_24"
):
    """
    Build a minimum-phase FIR filter from a magnitude response and save it as a WAV file.

    Parameters:
        mag_response (np.ndarray): Magnitude response in *linear* scale.
        file_path (str): Output WAV path.
        fs (int): Sample rate.
        n_fft (int): FFT size used for MP reconstruction.
        truncate_len (int): Output IR length.
        f_min, f_max (float): Band-limit frequencies.
        band_limit (bool): Apply band-limiting before MP conversion.
        prevent_clipping (int): Apply amplitude normalization before exporting.
        bit_depth (str): WAV subtype (e.g., "PCM_16", "PCM_24").

    Returns:
        np.ndarray: The generated minimum-phase FIR filter.
    """

    # ---- Build Minimum-Phase Filter ----
    impulse = build_min_phase_filter(
        smoothed_mag=mag_response,
        fs=fs,
        n_fft=n_fft,
        truncate_len=truncate_len,
        f_min=f_min,
        f_max=f_max,
        band_limit=band_limit
    )

    # ---- Save to WAV ----
    write2wav(
        file_name=file_path,
        data=impulse,
        samplerate=fs,
        prevent_clipping=prevent_clipping,
        bit_depth=bit_depth
    )

    return impulse



def build_summary_response_fir(
    fir_array,
    fs=CN.FS,
    n_fft=CN.N_FFT,
    truncate_len=4096,
    eps=1e-12
):
    """
    Build a representative minimum-phase FIR from an array of FIRs by
    averaging magnitude responses in dB.

    Parameters
    ----------
    fir_array : np.ndarray
        Array of shape (..., samples). Last axis must be time samples.
    fs : int
        Sample rate.
    n_fft : int
        FFT size used for magnitude analysis.
    truncate_len : int
        Output FIR length after minimum-phase reconstruction.
    eps : float
        Numerical floor to avoid log(0).

    All remaining parameters map 1:1 to build_min_phase_filter().
    """

    fir_array = np.asarray(fir_array)

    if fir_array.ndim < 1:
        raise ValueError("fir_array must have at least one dimension")

    # ---- Flatten all measurement dimensions ----
    samples = fir_array.shape[-1]
    firs = fir_array.reshape(-1, samples)

    if firs.shape[0] == 0:
        raise ValueError("No FIRs found in array")

    # ---- Zero pad or truncate ----
    if samples < n_fft:
        pad_width = n_fft - samples
        firs = np.pad(firs, ((0, 0), (0, pad_width)))
    else:
        firs = firs[:, :n_fft]

    # ---- FFT magnitude ----
    mag = np.abs(np.fft.rfft(firs, axis=-1))

    # ---- Convert to dB (log domain averaging) ----
    mag_db = 20.0 * np.log10(np.maximum(mag, eps))

    # ---- Average across all measurements ----
    avg_mag_db = np.mean(mag_db, axis=0)

    # ---- Back to linear magnitude ----
    avg_mag = 10.0 ** (avg_mag_db / 20.0)

    # ---- Build minimum-phase FIR ----
    min_phase_fir = build_min_phase_filter(
        smoothed_mag=avg_mag,
        fs=fs,
        n_fft=n_fft,
        truncate_len=truncate_len
    )

    return min_phase_fir





def level_spectrum_ends(
    data,
    low_freq=20,
    high_freq=20000,
    n_fft=CN.N_FFT,
    fs=CN.FS,
    smooth_win=1,
    to_full=False
):
    """
    Modify a magnitude spectrum to flatten low and high frequency ends with smoothing.
    
    Supports both rfft (half) and fft (full) spectra.
    
    Parameters:
        data (np.ndarray): Magnitude spectrum (length n_fft or n_fft//2 + 1)
        low_freq (float): Frequency below which to flatten (Hz)
        high_freq (float): Frequency above which to flatten (Hz)
        n_fft (int): FFT size that produced the spectrum
        fs (int): Sampling frequency (Hz)
        smooth_win (float): Smoothing window size (Hz)
        to_full (bool): If True and input is half spectrum, output will be full spectrum

    Returns:
        np.ndarray: Modified spectrum, same format as input unless to_full=True
    """
    is_half_spectrum = len(data) == n_fft // 2 + 1
    freq_res = fs / n_fft

    # Define frequency bin bounds
    low_bin = int(low_freq / freq_res)
    high_bin = int(high_freq / freq_res)
    high_bin = min(high_bin, len(data) - 1)
    low_bin = min(low_bin, high_bin)

    # Leveling
    data_mod = data.copy()
    if low_bin > 0:
        data_mod[:low_bin] = data[low_bin]
    if high_bin < len(data_mod):
        data_mod[high_bin:] = data[high_bin - 1]

    # Smoothing
    if smooth_win >= 1:
        #data_mod = sp.ndimage.uniform_filter1d(data_mod, size=smooth_win_samples)
        data_mod = smooth_gaussian_octave(data=data_mod, n_fft=n_fft, fs=fs, fraction=12)

    if is_half_spectrum and to_full:
        # Convert to full (conjugate symmetric) real-valued spectrum
        full_spectrum = np.empty(n_fft, dtype=data_mod.dtype)
        full_spectrum[:n_fft // 2 + 1] = data_mod
        # Mirror the spectrum (exclude DC and Nyquist)
        full_spectrum[n_fft // 2 + 1:] = data_mod[1:n_fft // 2][::-1]
        return full_spectrum

    return data_mod



def padarray(A, size):
    """
    function to pad a numpy array with zeros to specified size
    """  
    #numpy.pad with constant mode, pass a tuple as second argument to tell how many zeros to pad on each size, a (2, 3) for instance will pad 2 zeros on the left side and 3 zeros on the right side:
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')
    



    
    
    


def summarize_array(arr):
    """
    Summarizes key information about the structure of a NumPy array.

    Args:
        arr: The NumPy array to summarize.

    Returns:
        A string summarizing the array's shape, dimensions, and data type.
    """

    if not isinstance(arr, np.ndarray):
        return "Input is not a NumPy array."

    summary = f"Array Summary:\n"
    summary += f"  Shape: {arr.shape}\n"
    summary += f"  Dimensions: {arr.ndim}\n"
    summary += f"  Data Type: {arr.dtype}\n"

    return summary    
    
# --- Safety fallback for empty or None lists ---
def safe_list(lst):
    """Return a list with one empty string if None or empty."""
    return lst if lst and isinstance(lst, list) else [""] 
    
def ensure_valid_selection(list_data, loaded_value):
    """
    Ensures the loaded_value is valid for GUI dropdowns.
    Returns a tuple (valid_list, valid_value).
    """
    # Replace None or empty list with a single empty string element
    if not list_data:
        return [""], ""

    # If loaded_value is not in the list, fall back to the first entry
    if loaded_value not in list_data:
        return list_data, list_data[0]

    return list_data, loaded_value

def sanitize_filename(filename):
    """
    Sanitizes a filename by replacing invalid Windows characters with underscores.

    Args:
        filename (str): The filename to sanitize.

    Returns:
        str: The sanitized filename.
    """
    # Characters invalid in Windows filenames: \ / : * ? " < > |
    invalid_chars = r'[\\/:*?"<>|]'
    sanitized_filename = re.sub(invalid_chars, '_', filename)
    return sanitized_filename    
    
def format_name_for_file(name: str) -> str:
    """Replace spaces with underscores for a filename-friendly string."""
    return name.replace(" ", "_")

def update_gui_progress(report_progress, progress=None, message=''):
    """
    function to update progress bar in gui
    """ 
    overlay_text=''
    if report_progress > 0:
        if report_progress == 3:#AS progress
            prev_progress = dpg.get_value('progress_bar_as')
            if progress is None:
                progress=prev_progress#use previous progress if not specified
            dpg.set_value("progress_bar_as", progress)
        elif report_progress == 2:#fde progress
            prev_progress = dpg.get_value('fde_progress_bar_brir')
            if progress is None:
                progress=prev_progress#use previous progress if not specified
            if progress == 0:
                overlay_text = str(message)
            else:
                if message == '':
                    overlay_text = str(int(progress*100))+'%'
                else:
                    overlay_text = str(int(progress*100))+'% - '+message
            dpg.set_value("fde_progress_bar_brir", progress)
            dpg.configure_item("fde_progress_bar_brir", overlay = overlay_text)
        else:#QC progress
            prev_progress = dpg.get_value('e_apo_progress_bar_brir')
            if progress is None:
                progress=prev_progress#use previous progress if not specified
            if progress == 0:
                overlay_text = str(message)
            else:
                if message == '':
                    overlay_text = str(int(progress*100))+'%'
                else:
                    overlay_text = str(int(progress*100))+'% - '+message
            dpg.set_value("e_apo_progress_bar_brir", progress)
            dpg.configure_item("e_apo_progress_bar_brir", overlay = overlay_text)
              
def check_stop_thread(gui_logger=None):
    """
    function to check if stop thread has been flagged and updates gui
    """ 
    
    #exit if stop thread flag is true
    stop_thread_1 = dpg.get_item_user_data("e_apo_progress_bar_brir")
    stop_thread_2 = dpg.get_item_user_data("fde_progress_bar_brir")
    if stop_thread_1 == True or stop_thread_2 == True:
        log_string = 'BRIR Processing cancelled by user'
        log_with_timestamp(log_string, gui_logger)
        return True
    
    return False
              
   
                 
              
def check_and_download_file(file_path, file_link, download=False, gui_logger=None):
    """
    Checks if the specified file exists, and if not, downloads it from the provided Google Drive link.
    
    Args:
        file_path (str): Full path to the file to check.
        file_link (str): Direct  download link.
        download (bool): Whether to attempt downloading the file if missing.
        gui_logger (callable, optional): Logger for GUI output.

    Returns:
        int: 0 = Success, 1 = Failure
    """
    status = 1

    try:
        log_with_timestamp(f"Checking for file: {file_path}", gui_logger)

        if exists(file_path):
            log_with_timestamp("File already exists.", gui_logger)
        elif download:
            log_with_timestamp("File not found. Starting download...", gui_logger)
            download_file(file_link, file_path, gui_logger=gui_logger)
            log_with_timestamp(f"File downloaded to: {file_path}", gui_logger)

        status = 0  # success

    except Exception as ex:
        log_with_timestamp("Failed to check or download file.", gui_logger, log_type=2, exception=ex)

    return status 


    

    
def _extract_gdrive_file_id(url: str) -> str | None:
    """
    Extract Google Drive file ID from common URL formats.
    """
    patterns = [
        r"https://drive\.google\.com/file/d/([^/]+)",
        r"https://drive\.google\.com/open\?id=([^&]+)",
        r"https://drive\.google\.com/uc\?id=([^&]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def _get_gdrive_confirm_token(response: requests.Response) -> str | None:
    """
    Google Drive sometimes requires a confirmation token for large files.
    """
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            return v
    return None



def clean_url_if_needed(url: str) -> str:
    """
    Cleans a URL only if it contains illegal characters (e.g. spaces).
    Preserves existing percent-encoding.
    """
    _ILLEGAL_URL_CHARS = re.compile(r"[ \t\n\r]")
    
    if not url:
        return url

    # Fast exit: URL already safe
    if not _ILLEGAL_URL_CHARS.search(url):
        return url

    split = urlsplit(url)

    cleaned_path = quote(split.path, safe="/%")

    return urlunsplit((
        split.scheme,
        split.netloc,
        cleaned_path,
        split.query,
        split.fragment
    ))

def download_file(url, save_location, gui_logger=None, overwrite=True):
    """
    Downloads a file from a URL and saves it to the specified location.

    Args:
        url (str): The URL of the file to download.
        save_location (str): Full path (including filename) to save the file.
        gui_logger: Optional GUI logger.
        overwrite (bool): If False and file exists, skip download.
    """

    os.makedirs(os.path.dirname(save_location), exist_ok=True)

    if os.path.exists(save_location) and not overwrite:
        log_with_timestamp(
            f"File already exists at: {save_location}",
            gui_logger
        )
        return True

    try:
        # ---- URL cleanup (only if needed) ----
        cleaned_url = clean_url_if_needed(url)
        if cleaned_url != url:
            log_with_timestamp(f"Cleaned URL for download:\n{url}\n→ {cleaned_url}",gui_logger)
        url = cleaned_url
        
        session = requests.Session()

        # ---- Google Drive handling ----
        if "drive.google" in url:
            file_id = _extract_gdrive_file_id(url)
            if not file_id:
                raise ValueError("Could not extract Google Drive file ID")

            download_url = "https://drive.google.com/uc?export=download"
            params = {"id": file_id}

            response = session.get(download_url, params=params, stream=True, timeout=30)
            response.raise_for_status()

            token = _get_gdrive_confirm_token(response)
            if token:
                params["confirm"] = token
                response = session.get(download_url, params=params, stream=True, timeout=30)
                response.raise_for_status()

        # ---- Normal HTTP(S) download ----
        else:
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()

        # ---- Write file ----
        with open(save_location, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        log_with_timestamp(
            f"File downloaded successfully and saved to: {save_location}",
            gui_logger
        )
        return True

    except requests.exceptions.RequestException as e:
        log_with_timestamp(
            log_string=f"Error downloading file: {e}",
            gui_logger=gui_logger,
            log_type=2,
            exception=e
        )
        return False

    except Exception as ex:
        log_with_timestamp(
            log_string="An unexpected error occurred",
            gui_logger=gui_logger,
            log_type=2,
            exception=ex
        )
        return False


def get_files_with_extension(directory, extension):
    """
    Returns a list of all files in a directory that have a specified extension,
    with the extension stripped out of the filenames.

    Args:
        directory (str): The path to the directory to search.
        extension (str): The file extension to filter by (e.g., "txt", "pdf").
                          It should NOT include a leading dot.

    Returns:
        list: A list of strings, where each string is the name of a file
              in the directory (without the extension) that ends with the 
              specified extension.
              Returns an empty list if no matching files are found 
              or if the directory is not valid.
    """

    file_list = [] # Corrected line

    if not os.path.isdir(directory):
        return file_list  # Or you might want to raise an exception here

    try:
        for filename in os.listdir(directory):
            if filename.lower().endswith("." + extension.lower()):
                #remove extension
                name_without_ext = os.path.splitext(filename)[0]
                file_list.append(name_without_ext)
    except Exception as e:
        print(f"An error occurred: {e}")
        return file_list # Or you might want to raise an exception here

    return file_list   


def load_csv_as_dicts(csv_dir, csv_name): 
    """
    Generic CSV loader that detects and converts column data types automatically.
    
    Parameters:
        csv_dir (str): Directory containing the CSV file.
        csv_name (str): Name of the CSV file.
    
    Returns:
        List[dict]: A list of dictionaries representing each row with inferred data types.
    """
    data_list = []
    filepath = pjoin(csv_dir, csv_name)

    try:
        with open(filepath, encoding='utf-8-sig', newline='') as inputfile:
            reader = csv.DictReader(inputfile)

            for row in reader:
                parsed_row = {}
                for key, value in row.items():
                    value = value.strip()

                    # Try to convert to int
                    try:
                        parsed_row[key] = int(value) if '.' not in value else float(value)
                    except ValueError:
                        # Try to convert to float
                        try:
                            float_value = float(value)
                            # Keep as float unless it's effectively an integer
                            if float_value.is_integer():
                                parsed_row[key] = int(float_value)  # Store as int if no fractional part
                            else:
                                parsed_row[key] = float_value  # Store as float
                        except ValueError:
                            # If neither int nor float, store as string
                            parsed_row[key] = value

                data_list.append(parsed_row)

    except Exception as e:
        print(f"Failed to load CSV '{csv_name}': {e}")

    return data_list

def extract_column(data, column, condition_key=None, condition_value=None):
    """
    Extracts a column from a list of dictionaries without auto-conversion.
    Optionally filters rows based on a condition.

    Parameters:
        data (list of dict): The metadata loaded from CSV.
        column (str): The key (column name) to extract values from.
        condition_key (str, optional): Key to filter rows.
        condition_value (any, optional): Value that condition_key must match.

    Returns:
        list: Values from the specified column.
    """
    if condition_key and condition_value is not None:
        return [row.get(column) for row in data if row.get(condition_key) == condition_value]
    
    return [row.get(column) for row in data]

def find_dict_by_value(data_list, key, value):
    """
    Search a list of dictionaries and return the first dictionary
    where the specified key matches the given value.

    Parameters:
        data_list (list): A list of dictionaries to search through.
        key (str): The key to look for in each dictionary.
        value (any): The value to match against.

    Returns:
        dict or None: The first matching dictionary, or None if not found.
    """
    for item in data_list:
        # Get the value associated with the key and compare it to the target value
        if item.get(key) == value:
            return item

    # Return None if no match was found
    return None

def get_value_from_matching_dict(data_list, match_key, match_value, return_key):
    """
    Search a list of dictionaries for the first dictionary where `match_key` equals `match_value`,
    and return the value associated with `return_key` in that dictionary.

    Parameters:
        data_list (list): A list of dictionaries to search through.
        match_key (str): The key to match on.
        match_value (any): The value to search for in `match_key`.
        return_key (str): The key whose value should be returned from the matched dictionary.

    Returns:
        any or None: The value corresponding to `return_key`, or None if not found or key missing.
    """
    for item in data_list:
        # Check if the current dictionary matches the given key-value pair
        if item.get(match_key) == match_value:
            # Return the value for the specified return_key (or None if it's missing)
            return item.get(return_key)

    # If no match is found, return None
    return None

def load_append_and_save_npy_matrices_recursive(
    directory: str,
    output_folder: str,
    output_filename: str = "combined_transform_matrix.npy",
    ignore_keyword: str = "combined"
) -> np.ndarray:
    """
    Recursively loads all .npy files from a directory and its subdirectories,
    assuming each file contains a 2D array of shape (measurements, samples),
    then appends them along the measurement axis and saves the result to a specified folder.

    Files whose names contain the 'ignore_keyword' are skipped during the search.

    Parameters:
        directory (str): Root directory to search for .npy files.
        output_folder (str): Directory where the combined output file should be saved.
        output_filename (str): Name of the output .npy file.
        ignore_keyword (str): Substring that, if found in a filename, will cause the file to be ignored.

    Returns:
        np.ndarray: A 2D array of stacked matrices.

    Raises:
        ValueError: If arrays are not 2D or their sample dimensions don't match.
    """
    appended = []
    sample_shape = None
    file_count = 0

    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            if fname.endswith(".npy") and ignore_keyword not in fname:
                path = os.path.join(root, fname)
                arr = np.load(path)

                if arr.ndim != 2:
                    raise ValueError(f"{path} is not a 2D array.")

                if sample_shape is None:
                    sample_shape = arr.shape[1]
                elif arr.shape[1] != sample_shape:
                    raise ValueError(
                        f"{path} has a different number of columns ({arr.shape[1]}) than expected ({sample_shape})."
                    )

                print(f"Loaded: {path} | shape: {arr.shape}")
                appended.append(arr)
                file_count += 1

    if not appended:
        raise ValueError("No valid .npy files were found or loaded.")

    combined = np.vstack(appended)
    
    #apply fading too
    #combined_faded = apply_fade_to_matrix(combined,fade_in_samples=150,fade_out_samples=150)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the combined matrix
    output_path = os.path.join(output_folder, output_filename)
    np.save(output_path, combined)

    print(f"\nTotal files loaded: {file_count}")
    print(f"Combined matrix shape: {combined.shape}")
    print(f"Saved combined matrix to: {output_path}")

    return combined

def apply_fade_to_matrix(
    matrix: np.ndarray,
    fade_in_samples: int = 100,
    fade_out_samples: int = 100
) -> np.ndarray:
    """
    Applies a linear fade-in and fade-out window to the second dimension (samples)
    of a 2D matrix with shape (measurements, samples).

    Parameters:
        matrix (np.ndarray): Input 2D matrix of shape (measurements, samples).
        fade_in_samples (int): Number of samples over which to apply the fade-in.
        fade_out_samples (int): Number of samples over which to apply the fade-out.

    Returns:
        np.ndarray: Matrix with fade-in and fade-out applied.
    
    Raises:
        ValueError: If the matrix is not 2D or fades exceed matrix length.
    """
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be 2D.")
    
    num_samples = matrix.shape[1]
    if fade_in_samples + fade_out_samples > num_samples:
        raise ValueError("Fade durations exceed the number of samples.")

    fade_in = np.linspace(0, 1, fade_in_samples)
    fade_out = np.linspace(1, 0, fade_out_samples)
    sustain = np.ones(num_samples - fade_in_samples - fade_out_samples)

    window = np.concatenate([fade_in, sustain, fade_out])
    faded_matrix = matrix * window  # Broadcasting applies fade to each measurement row

    return faded_matrix



def crop_samples(arr: np.ndarray, crop_length: int) -> np.ndarray:
    """
    Returns a cropped version of the input 2D array with specified sample length.

    Parameters:
        arr (np.ndarray): 2D array of shape (measurements x samples).
        crop_length (int): Number of samples to retain per row.

    Returns:
        np.ndarray: Cropped 2D array.
    """
    if arr.ndim != 2:
        raise ValueError("Input array must be 2D.")
    if crop_length > arr.shape[1]:
        raise ValueError("crop_length exceeds available samples.")
    return arr[:, :crop_length]

def crop_measurements(arr: np.ndarray, max_measurements: int) -> np.ndarray:
    """
    Returns a cropped version of the input 2D array with a limited number of measurements (rows).

    Parameters:
        arr (np.ndarray): 2D array of shape (measurements x samples).
        max_measurements (int): Maximum number of rows to keep.

    Returns:
        np.ndarray: Cropped 2D array with at most `max_measurements` rows.
    """
    if arr.ndim != 2:
        raise ValueError("Input array must be 2D.")
    if max_measurements > arr.shape[0]:
        raise ValueError("max_measurements exceeds number of available measurements.")
    return arr[:max_measurements, :]

def exponential_moving_average(x, alpha):
    """Applies exponential moving average smoothing to a 1D array."""
    ema = np.zeros_like(x)
    ema[0] = x[0]
    for i in range(1, len(x)):
        ema[i] = alpha * x[i] + (1 - alpha) * ema[i - 1]
    return ema

def plot_sample_value_distribution(data: np.ndarray, bins: int = 100):
    """
    Plot the distribution of sample values across all measurements in a 2D array.
    Highlights the mean and ±1 standard deviation.

    Parameters:
        data (np.ndarray): 2D array with shape (measurements, samples)
        bins (int): Number of histogram bins

    Returns:
        None (displays the plot)
    """
    if data.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array (measurements x samples)")

    # Flatten the data to a 1D array
    flattened = data.flatten()
    mean = np.mean(flattened)
    std = np.std(flattened)

    # Plot histogram
    plt.figure(figsize=(10, 4))
    counts, bin_edges, _ = plt.hist(flattened, bins=bins, density=True, color='skyblue',
                                    edgecolor='black', alpha=0.9, label='Histogram')

    # Overlay mean and standard deviation lines
    plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean:.3f}')
    plt.axvline(mean - std, color='orange', linestyle='--', linewidth=1.5, label=f'±1 Std = {std:.3f}')
    plt.axvline(mean + std, color='orange', linestyle='--', linewidth=1.5)

    # Formatting
    plt.title("Distribution of Sample Values")
    plt.xlabel("Sample Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def remove_leading_singletons(arr):
    """
    Removes leading singleton dimensions from a NumPy array.

    Parameters:
        arr (np.ndarray): An N-dimensional NumPy array.

    Returns:
        np.ndarray: A reshaped array with leading singleton dimensions removed.
    """
    shape = arr.shape
    first_non_singleton = 0

    # Find the index of the first non-singleton dimension
    for i, dim in enumerate(shape):
        if dim != 1:
            break
        first_non_singleton += 1

    # Slice off leading singleton dimensions
    if first_non_singleton > 0:
        return arr.reshape(shape[first_non_singleton:])
    return arr





def validate_choice(loaded_value, valid_list):
    """
    Ensure the loaded_value is present in valid_list.
    If not, return the first item in the list and log a fallback message.
    """
    # Ensure valid_list is iterable
    if not valid_list:
        msg = f"validate_choice: No valid list provided. Cannot validate '{loaded_value}'."
        log_with_timestamp(msg)
        return "Not Found"

    if loaded_value in valid_list:
        return loaded_value
    else:
        fallback = valid_list[0]
        msg = f"validate_choice: '{loaded_value}' is not valid. Falling back to '{fallback}'."
        log_with_timestamp(msg)
        return fallback




def safe_get(config, key, expected_type, default):
    """Safely get a value from config with fallback and type casting.
    Supports bool, list, dict, tuple via automatic parsing from JSON or Python literals.
    """
    try:
        val = config['DEFAULT'].get(key, default)
        
        # If the value already matches the expected type (and expected_type is a type), return it
        if isinstance(expected_type, type) and isinstance(val, expected_type):
            return val

        if expected_type == bool:
            if isinstance(val, str):
                lowered = val.strip().lower()
                if lowered in ('true', 'false'):
                    return lowered == 'true'
            return bool(ast.literal_eval(str(val)))

        if expected_type in (list, dict, tuple):
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    parsed = ast.literal_eval(val)
                return parsed if isinstance(parsed, expected_type) else default
            elif isinstance(val, expected_type):
                return val
            else:
                return default

        # Special case for ast.literal_eval as expected_type
        if expected_type == ast.literal_eval:
            if val is None or (isinstance(val, str) and val.strip() == ''):
                return default
            return ast.literal_eval(val)

        # For other types (int, float, str, etc.), cast
        if callable(expected_type):
            return expected_type(val)

        # Fallback for unexpected expected_type
        msg = f"safe_get: Expected type for key '{key}' is not callable. Using raw value."
        log_with_timestamp(msg)
        return val

    except Exception as e:
        msg = f"safe_get: Failed to load key '{key}' – {e}. Using default: {default}"
        log_with_timestamp(msg)
        return default
       
def check_write_permissions(path, gui_logger=None):
    """Check if the process has write access to the specified path."""
    try:
        test_file = Path(path) / ".perm_check.tmp"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.touch(exist_ok=True)
        test_file.unlink()  # remove the temp file
        return True
    except PermissionError:
        msg = f"Insufficient privileges to write to '{path}'. Please run as administrator or choose a different output folder."
        #print(msg)
        log_with_timestamp(msg, gui_logger)
        return False    
    
 
 
        
def get_default_output_info():
    """Return default output device name and sample rate, or Unknown if sounddevice fails."""
    if not SD_AVAILABLE:
        return "Unknown device", 0

    try:
        info = sd.query_devices(kind='output')  # default output device
        device_name = info.get('name', 'Unknown device')
        device_samplerate = int(info.get('default_samplerate', 0))
        return device_name, device_samplerate
    except Exception:
        return "Unknown device", 0


def update_default_output_text(reset_sd=True):
    """Update DPG text elements showing the default playback device and its sample rate.
    If sounddevice is unavailable or fails, show 'Unknown'."""
    try:
        # Reset PortAudio host API/device info on Windows
        if IS_WINDOWS and SD_AVAILABLE and reset_sd:
            sd._terminate()
            sd._initialize()

        # Get current default playback device info
        device_name, device_samplerate = get_default_output_info()

        # Convert to kHz for display
        sr_text = f"{device_samplerate / 1000:.1f} kHz" if device_samplerate != 0 else "Unknown"

        # Get currently selected WAV sample rate
        samp_freq_str = dpg.get_value('wav_sample_rate')
        samp_freq_int = CN.SAMPLE_RATE_DICT.get(samp_freq_str, 0)

        # Compare sample rates (both in Hz)
        if device_samplerate != 0 and samp_freq_int != 0 and samp_freq_int != device_samplerate:
            sr_text += " (Mismatch)"

        # Update GUI text elements
        if dpg.does_item_exist("def_pb_device_name"):
            dpg.set_value("def_pb_device_name", device_name)
        if dpg.does_item_exist("def_pb_device_sr"):
            dpg.set_value("def_pb_device_sr", sr_text)

    except Exception as e:
        # fallback text if anything goes wrong
        if dpg.does_item_exist("def_pb_device_name"):
            dpg.set_value("def_pb_device_name", "Default playback device: Unknown")
        if dpg.does_item_exist("def_pb_device_sr"):
            dpg.set_value("def_pb_device_sr", "Default sample rate: Unknown")
        msg = f"[Warning] Could not update default output info: {e}"
        log_with_timestamp(msg) 
    
 
        
def map_array_value_lookup(value, array_gui, array_internal, default=None):
    """
    Bi-directional mapping between GUI labels and internal values.
    Automatically detects the type of `value` and returns the corresponding value
    from the other array.

    Parameters
    ----------
    value : any
        The value to map. Can be a GUI label or internal value.
    array_gui : list
        List of GUI labels.
    array_internal : list
        List of corresponding internal values.
    default : any
        Value to return if `value` is not found. Defaults to None.

    Returns
    -------
    any
        Mapped value from the corresponding array, or `default` if not found.
    """
    try:
        if value in array_gui:
            return array_internal[array_gui.index(value)]
        elif value in array_internal:
            return array_gui[array_internal.index(value)]
        else:
            return default
    except Exception:
        return default        


    
def set_checkbox_and_sync_button(
    checkbox_tag,
    button_tag,
    value=None,
    on_texture=CN.BUTTON_IMAGE_ON,
    off_texture=CN.BUTTON_IMAGE_OFF,
    refresh=False
):
    """
    Set or refresh the value of a hidden checkbox and synchronise the corresponding image button.

    Parameters:
        checkbox_tag (str): tag of the hidden checkbox
        button_tag (str): tag of the image button
        value (bool | None): checkbox value to set (ignored if refresh=True)
        refresh (bool): if True, read current checkbox value and sync button only
        on_texture (str): texture tag for ON state
        off_texture (str): texture tag for OFF state
    """
    # Refresh mode: read authoritative state from checkbox
    if refresh:
        value = dpg.get_value(checkbox_tag)
    else:
        # Normal mode: explicitly set checkbox state
        if value is None:
            raise ValueError("value must be provided unless refresh=True")
        dpg.set_value(checkbox_tag, value)

    # Sync button image to checkbox state
    dpg.configure_item(
        button_tag,
        texture_tag=on_texture if value else off_texture
    )    
    
    
def pad_or_truncate_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    Truncate or zero-pad a 1D array to exactly target_len samples.

    Parameters
    ----------
    x : np.ndarray
        Input 1D array.
    target_len : int
        Desired output length.

    Returns
    -------
    np.ndarray
        Array of length target_len.
    """
    x = np.asarray(x).flatten()

    out = np.zeros(target_len, dtype=x.dtype)
    copy_len = min(len(x), target_len)
    out[:copy_len] = x[:copy_len]

    return out

def normalize_brir_band(
    ir_data,
    n_fft,
    fs,
    f_norm_start,
    f_norm_end,
    analysis_samples=None,
    eps=1e-12
):
    """
    Normalize IR/BRIR data by a single global scalar so that the
    average magnitude around a frequency band equals 0 dB.

    The magnitude is:
    - Measured using only the first `analysis_samples` (if provided)
    - Averaged in the log domain
    - Averaged across ALL measurements
    - Applied as a single scalar to the entire array

    Assumes last axis is time.
    """

    # -----------------------------
    # Determine analysis length
    # -----------------------------
    if analysis_samples is not None:
        analysis_len = min(analysis_samples, ir_data.shape[-1], n_fft)
    else:
        analysis_len = min(ir_data.shape[-1], n_fft)

    # -----------------------------
    # Frequency band → FFT bins
    # -----------------------------
    fb_start = int(f_norm_start * n_fft / fs)
    fb_end   = int(f_norm_end   * n_fft / fs)

    # -----------------------------
    # FFT of analysis window
    # -----------------------------
    fft_data = np.fft.rfft(
        ir_data[..., :analysis_len],
        n=n_fft,
        axis=-1
    )

    # -----------------------------
    # Magnitude in band
    # -----------------------------
    mag_band = np.abs(fft_data[..., fb_start:fb_end])

    # -----------------------------
    # Convert to dB
    # -----------------------------
    mag_db = 20.0 * np.log10(np.clip(mag_band, eps, None))

    # -----------------------------
    # Global log-domain average
    # -----------------------------
    avg_mag_db = np.mean(mag_db)

    # -----------------------------
    # Back to linear scalar
    # -----------------------------
    avg_mag = 10.0 ** (avg_mag_db / 20.0)
    avg_mag = max(avg_mag, eps)

    # -----------------------------
    # Apply single scalar
    # -----------------------------
    ir_data /= avg_mag

    return ir_data



