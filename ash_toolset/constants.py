# -*- coding: utf-8 -*-
"""
Main routine of ASH-Tools.

Created on Sun Apr 28 11:25:06 2024
@author: Shanon Pearce
License: (see /LICENSE)
"""

import numpy as np
from os.path import join as pjoin
from os import path
import sys
from pathlib import Path
from csv import DictReader
import csv
import operator
import os
from platformdirs import user_config_dir
import json

#Few helper functions to load constants
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



def load_user_reverb_csvs_recursive(directory, filename_key=None, filter_mode="include", match_mode="contains"):
    """
    Recursively searches a directory for CSV files and loads them as dicts,
    optionally including or excluding files based on a string match.

    Parameters:
        directory (str): Base directory to search in.
        filename_key (str, optional): Substring to filter filenames. If None, all CSVs are included.
        filter_mode (str): 'include' to load only matching files, 'exclude' to skip matching files.
        match_mode (str): One of 'contains', 'startswith', or 'endswith'.

    Returns:
        List[dict]: Parsed rows from all matching CSVs.
    """
    if match_mode not in {"contains", "startswith", "endswith"}:
        raise ValueError("match_mode must be 'contains', 'startswith', or 'endswith'")

    found_rows = []
    primary_key=METADATA_CSV_KEY

    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(".csv"):
                continue
            
            #check if file contains primary keyword
            match = True
            if primary_key is not None:
                match = primary_key in file
            if not match:
                continue
            
            #check if file matches secondary keyword
            match = True
            if filename_key is not None:
                if match_mode == "contains":
                    match = filename_key in file
                elif match_mode == "startswith":
                    match = file.startswith(filename_key)
                elif match_mode == "endswith":
                    match = file.endswith(filename_key)

                if filter_mode == "exclude":
                    match = not match

            if not match:
                continue

            path = os.path.join(root, file)
            try:
                with open(path, encoding='utf-8-sig', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        parsed_row = {}
                        for key, value in row.items():
                            value = value.strip()
                            try:
                                parsed_row[key] = int(value) if '.' not in value else float(value)
                            except ValueError:
                                try:
                                    float_value = float(value)
                                    parsed_row[key] = int(float_value) if float_value.is_integer() else float_value
                                except ValueError:
                                    parsed_row[key] = value
                        found_rows.append(parsed_row)
            except Exception as e:
                print(f"Failed to load user CSV {path}: {e}")

    return found_rows

def sort_dict_list(data, sort_key, reverse=False):
    """
    Sorts a list of dictionaries by the specified key (case-insensitive).

    Parameters:
        data (list): List of dictionaries to sort.
        sort_key (str): Dictionary key to sort by.
        reverse (bool): Sort descending if True, ascending if False.

    Returns:
        list: Sorted list of dictionaries.
    """
    return sorted(
        data,
        key=lambda x: str(x.get(sort_key, "")).strip().lower(),
        reverse=reverse
    )

def extract_column(data, column, condition_key=None, condition_value=None, condition_op='==', return_all_matches=False):
    """
    Extracts a value or list of values from a list of dictionaries.
    Can return a single value or multiple values that match a condition.

    Parameters:
        data (list of dict): The metadata loaded from CSV.
        column (str): The key (column name) to extract values from.
        condition_key (str, optional): Key to filter rows.
        condition_value (any, optional): Value that condition_key must satisfy.
        condition_op (str, optional): Comparison operator: '==', '!=', '>', '<', '>=', '<=' (default: '==').
        return_all_matches (bool): If True, return all matching values. If False, return first match only.

    Returns:
        list or str: If no condition is provided, returns a list of values.
                     If condition is provided:
                        - and return_all_matches is True → list of values (or empty list).
                        - and return_all_matches is False → first value (or empty string).
    """
    # Supported operators
    ops = {
        '==': operator.eq,
        '!=': operator.ne,
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '<=': operator.le
    }

    compare = ops.get(condition_op, operator.eq)

    if condition_key and condition_value is not None:
        matches = [row.get(column, "") for row in data if condition_key in row and compare(row.get(condition_key), condition_value)]
        if return_all_matches:
            return matches
        else:
            return matches[0] if matches else ""
    else:
        result = [row.get(column) for row in data]
        return result if result else []


def refresh_acoustic_space_metadata(): 
    """
    Reload and merge acoustic space metadata from internal and user sources,
    eliminate duplicates, sort, and populate global lists.
    """
    global reverb_data, REV_METADATA_FILE_NAME
    global AC_SPACE_LIST_GUI, AC_SPACE_LIST_SHORT, AC_SPACE_LIST_SRC
    global AC_SPACE_EST_R60, AC_SPACE_MEAS_R60, AC_SPACE_FADE_START
    global AC_SPACE_GAINS, AC_SPACE_DESCR, AC_SPACE_DATASET
    global AC_SPACE_LIST_NR, AC_SPACE_LIST_LOWRT60, AC_SPACE_LIST_HIRT60
    global AC_SPACE_LIST_COMP, AC_SPACE_LIST_MAX_SRC, AC_SPACE_LIST_ID_SRC

    # Load internal metadata
    csv_directory = pjoin(DATA_DIR_INT, 'reverberation')
    internal_data = load_csv_as_dicts(csv_directory, REV_METADATA_FILE_NAME)

    # Load user metadata
    user_csv_dir = DATA_DIR_AS_USER
    user_data = load_user_reverb_csvs_recursive(
        user_csv_dir, filename_key=USER_CSV_KEY, filter_mode="include", match_mode="contains"
    )

    # --- Normalize missing columns between internal and user CSVs ---
    all_keys = set()
    for row in internal_data + user_data:
        all_keys.update(row.keys())

    for row in internal_data + user_data:
        for key in all_keys:
            row.setdefault(key, "")
    # ---------------------------------------------------------------

    # Deduplicate based on 'file_name'
    existing_keys = set(row.get("file_name") for row in internal_data if "file_name" in row)
    unique_user_data = [row for row in user_data if row.get("file_name") not in existing_keys]

    # Merge and sort
    reverb_data = internal_data + unique_user_data
    reverb_data = sort_dict_list(reverb_data, sort_key='name_src', reverse=False)
    
    # Extract individual lists
    AC_SPACE_LIST_GUI = extract_column(reverb_data, 'name_gui')
    AC_SPACE_LIST_SHORT = extract_column(reverb_data, 'name_short')
    AC_SPACE_LIST_SRC = extract_column(reverb_data, 'name_src')
    AC_SPACE_EST_R60 = extract_column(reverb_data, 'est_rt60')
    AC_SPACE_MEAS_R60 = extract_column(reverb_data, 'meas_rt60')
    AC_SPACE_FADE_START = extract_column(reverb_data, 'fade_start')
    AC_SPACE_GAINS = extract_column(reverb_data, 'gain')
    AC_SPACE_DESCR = extract_column(reverb_data, 'description')
    AC_SPACE_DATASET = extract_column(reverb_data, 'source_dataset')
    AC_SPACE_LIST_NR = extract_column(
        reverb_data, 'name_src', condition_key='noise_reduce', condition_value='Yes', return_all_matches=True)
    AC_SPACE_LIST_LOWRT60 = extract_column(
        reverb_data, 'name_src', condition_key='low_rt60', condition_value='Yes', return_all_matches=True)
    AC_SPACE_LIST_HIRT60 = extract_column(
        reverb_data, 'name_src', condition_key='low_rt60', condition_value='No', return_all_matches=True)
    AC_SPACE_LIST_COMP = extract_column(
        reverb_data, 'name_src', condition_key='folder', condition_value='comp_bin', return_all_matches=True)
    AC_SPACE_LIST_MAX_SRC = extract_column(
        reverb_data, 'name_src', condition_key='source_hrtf_dataset', condition_value='max', return_all_matches=True)
    AC_SPACE_LIST_ID_SRC = extract_column(reverb_data, 'source_hrtf_id')




def get_settings_path():
    """
    Get the full path to the application's settings.ini file in the user's configuration directory.

    This function ensures that the user-specific configuration directory exists,
    then returns the path to the 'settings.ini' file within that directory.

    Returns:
        str: Full file path to the settings.ini file in the platform-appropriate user config folder.

    Note:
        The config directory location is platform-dependent:
        - Windows: %APPDATA%\\YourAppName
        - macOS: ~/Library/Application Support/YourAppName
        - Linux: ~/.config/YourAppName
    """
    appname = "ASH-Toolset"
    config_dir = user_config_dir(appname)
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "settings.ini")

def load_user_room_targets(data_dir_rt_user, room_targets_dict, room_target_list, room_target_list_short):
    """
    Load user-defined room target .npy files from a directory and append them to existing room target data.

    Parameters:
        data_dir_rt_user (str): Directory containing user-defined room target .npy files.
        room_targets_dict (dict): Existing dictionary mapping names to impulse responses and metadata.
        room_target_list (list): Existing list of full names to append to.
        room_target_list_short (list): Existing list of short names to append to.

    Returns:
        tuple: Updated (room_targets_dict, room_target_list, room_target_list_short)
    """
    for fname in os.listdir(data_dir_rt_user):
        if fname.lower().endswith(".npy"):
            try:
                path = os.path.join(data_dir_rt_user, fname)
                impulse_response = np.load(path)

                # Use filename (without extension) as base name
                base_name = os.path.splitext(fname)[0]
                full_name = base_name

                # Resolve name collision by adding suffix
                suffix = 1
                while full_name in room_targets_dict:
                    full_name = f"{base_name} ({suffix})"
                    suffix += 1

                room_targets_dict[full_name] = {
                    "short_name": full_name,
                    "impulse_response": impulse_response
                }
                room_target_list.append(full_name)
                room_target_list_short.append(full_name)

            except Exception as e:
                print(f"[WARN] Failed to load user room target '{fname}': {e}")

    return room_targets_dict, room_target_list, room_target_list_short

def refresh_room_targets():
    """
    Reloads room target arrays from internal and user sources.
    Rebuilds the global ROOM_TARGETS_* structures.
    """
    global ROOM_TARGETS_DICT, ROOM_TARGET_LIST, ROOM_TARGET_LIST_SHORT, ROOM_TARGET_KEYS, ROOM_TARGET_INDEX_MAP

    # Load the FIR array from .npy file (base targets)
    npy_fname = pjoin(DATA_DIR_INT, 'room_targets_firs.npy')
    room_target_arr = np.load(npy_fname)

    # Base lists
    ROOM_TARGET_LIST = ['Flat','ASH Target','Harman Target','HATS Target','Toole Target','rtings Target',
                        'ASH Target - Low End','Harman Target - Low End','HATS Target - Low End','Toole Target - Low End','rtings Target - Low End',
                        'ASH Target - Flat Highs','Harman Target - Flat Highs','HATS Target - Flat Highs','Toole Target - Flat Highs','rtings Target - Flat Highs']
    ROOM_TARGET_LIST_SHORT = ['Flat','ASH-Target','Harman-Target','HATS-Target','Toole-Target','rtings-Target',
                              'ASH-Target-Low-End','Harman-Target-Low-End','HATS-Target-Low-End','Toole-Target-Low-End','rtings-Target-Low-End',
                              'ASH-Target-Flat-Highs','Harman-Target-Flat-Highs','HATS-Target-Flat-Highs','Toole-Target-Flat-Highs','rtings-Target-Flat-Highs']

    # Build dict from base FIRs
    ROOM_TARGETS_DICT = {
        name: {
            "short_name": short,
            "impulse_response": room_target_arr[i]
        }
        for i, (name, short) in enumerate(zip(ROOM_TARGET_LIST, ROOM_TARGET_LIST_SHORT))
    }

    # Load and append user targets
    ROOM_TARGETS_DICT, ROOM_TARGET_LIST, ROOM_TARGET_LIST_SHORT = load_user_room_targets(
        DATA_DIR_RT_USER,
        ROOM_TARGETS_DICT,
        ROOM_TARGET_LIST,
        ROOM_TARGET_LIST_SHORT
    )

    # Rebuild access helpers
    ROOM_TARGET_KEYS = ROOM_TARGET_LIST
    ROOM_TARGET_INDEX_MAP = {name: idx for idx, name in enumerate(ROOM_TARGET_LIST)}






def refresh_sub_responses():
    """
    Reloads sub response arrays from internal and user sources.
    Rebuilds the global SUB_RESPONSE_* structures.
    """
    global sub_data, SUB_RESPONSE_LIST_GUI, SUB_RESPONSE_LIST_SHORT, SUB_RESPONSE_LIST_AS,SUB_RESPONSE_LIST_R60
    global SUB_RESPONSE_LIST_COMM, SUB_RESPONSE_LIST_RANGE, SUB_RESPONSE_LIST_TOL, SUB_RESPONSE_LIST_DATASET, SUB_RESPONSE_LIST_SOURCE, SUB_RESPONSE_LIST_LISTENER
    

    #new: load as dict lists
    csv_directory = DATA_DIR_SUB
    internal_data = load_csv_as_dicts(csv_directory,'lf_brir_metadata.csv')

    # Load user metadata
    user_csv_dir = DATA_DIR_AS_USER
    user_data = load_user_reverb_csvs_recursive(user_csv_dir, filename_key=SUB_USER_CSV_KEY, filter_mode="include", match_mode="contains")

    # Deduplicate based on 'file_name'
    existing_keys = set(row.get("file_name") for row in internal_data if "file_name" in row)
    unique_user_data = [row for row in user_data if row.get("file_name") not in existing_keys]

    # Merge and sort
    sub_data = internal_data + unique_user_data
    sub_data = sort_dict_list(sub_data, sort_key='name_src', reverse=False)
 
    SUB_RESPONSE_LIST_GUI = extract_column(data=sub_data, column='name_gui') 
    SUB_RESPONSE_LIST_SHORT = extract_column(data=sub_data, column='name_short') 
    SUB_RESPONSE_LIST_AS = extract_column(data=sub_data, column='acoustic_space') 
    SUB_RESPONSE_LIST_R60 = extract_column(data=sub_data, column='est_rt60') 
    SUB_RESPONSE_LIST_COMM = extract_column(data=sub_data, column='comments') 
    SUB_RESPONSE_LIST_RANGE = extract_column(data=sub_data, column='frequency_range') 
    SUB_RESPONSE_LIST_TOL = extract_column(data=sub_data, column='tolerance')
    SUB_RESPONSE_LIST_DATASET = extract_column(data=sub_data, column='dataset')
    SUB_RESPONSE_LIST_SOURCE = extract_column(data=sub_data, column='source_type')
    SUB_RESPONSE_LIST_LISTENER = extract_column(data=sub_data, column='listener_type')

 
    
def get_version(metadata_file_path):
    try:
        with open(metadata_file_path) as fp:
            return json.load(fp).get("version", "0.0")
    except Exception:
        return "0.0"



################################ 
#Constants


#commonly used
N_FFT = 65536
N_FFT_L = int(65536*2)
N_UNIQUE_PTS = int(np.ceil((N_FFT+1)/2.0))
SAMP_FREQ = 44100#sample rate for ash toolset
FS=SAMP_FREQ#keep for backwards compat
THRESHOLD_CROP = 0.0000005#0.0000005 -120db with reference level of 0.5


#Developer settings
PLOT_ENABLE = False#False
LOG_MEMORY=False #False
LOG_INFO=True     
LOG_GUI=True
HEAD_TRACK_RUNNING = False
PROCESS_BRIRS_RUNNING = False
SHOW_DEV_TOOLS=False#False
STOP_THREAD_FLAG = False
SHOW_AS_TAB = True
EXPORT_WAVS_DEFAULT=False#False

#control level of direct sound
DIRECT_SCALING_FACTOR = 1.0#reference level - approx 0db DRR. was 0.1

#crossover frequency for low frequency BRIR integration
ENABLE_SUB_INTEGRATION = True
ORDER=9#8
F_CROSSOVER = 120#120, default
EVAL_POLARITY=True#True
PEAK_MEAS_MODE=1#0=local max peak, 1 =peak to peak
FILTFILT_TDALIGN = False#apply forward reverse filtering in td alignment method?
FILTFILT_TDALIGN_AIR = False#apply forward reverse filtering in td alignment method?
FILTFILT_THRESH_F = 45
#size to hop peak to peak window
DELAY_WIN_HOP_SIZE = 25#10
DELAY_WIN_MIN_T = 0
DELAY_WIN_MAX_T = 1200#
DELAY_WIN_HOPS = int((DELAY_WIN_MAX_T-DELAY_WIN_MIN_T)/DELAY_WIN_HOP_SIZE)

#contants for TD alignment of BRIRs with other BRIRs
T_SHIFT_INTERVAL = 25#50
MIN_T_SHIFT = -1500
MAX_T_SHIFT = 0
NUM_INTERVALS = int(np.abs((MAX_T_SHIFT-MIN_T_SHIFT)/T_SHIFT_INTERVAL))

#sub alignment with reverb - neg pol
T_SHIFT_INTERVAL_N = 10#25
MIN_T_SHIFT_N = -230#-230
MAX_T_SHIFT_N = 50#50
NUM_INTERVALS_N = int(np.abs((MAX_T_SHIFT_N-MIN_T_SHIFT_N)/T_SHIFT_INTERVAL_N))
#sub alignment with reverb - pos pol
T_SHIFT_INTERVAL_P = 10#25
MIN_T_SHIFT_P = -230#-230
MAX_T_SHIFT_P = 50#50
NUM_INTERVALS_P = int(np.abs((MAX_T_SHIFT_P-MIN_T_SHIFT_P)/T_SHIFT_INTERVAL_P))


#reverb alignment with hrir
T_SHIFT_INTERVAL_R = 25
MIN_T_SHIFT_R = -500#-400
MAX_T_SHIFT_R = 100#100
NUM_INTERVALS_R = int(np.abs((MAX_T_SHIFT_P-MIN_T_SHIFT_P)/T_SHIFT_INTERVAL_P))

#hrir alignment with other hrirs
T_SHIFT_INTERVAL_H = 5
MIN_T_SHIFT_H = -30#
MAX_T_SHIFT_H = 50
NUM_INTERVALS_H = int(np.abs((MAX_T_SHIFT_H-MIN_T_SHIFT_H)/T_SHIFT_INTERVAL_H))
DELAY_WIN_MIN_H = 0
DELAY_WIN_MAX_H = 150
DELAY_WIN_HOPS_H = int((DELAY_WIN_MAX_H-DELAY_WIN_MIN_H)/DELAY_WIN_HOP_SIZE)

#AIR alignment
CUTOFF_ALIGNMENT_AIR = 110#110,100
CUTOFF_ALIGNMENT_TR_AIR = CUTOFF_ALIGNMENT_AIR#transformation applied cases, 140, 130, 120
PEAK_TO_PEAK_WINDOW_AIR = int(np.divide(SAMP_FREQ,CUTOFF_ALIGNMENT_AIR)*0.95) 
MIN_T_SHIFT_A = -1000#-1000
MAX_T_SHIFT_A = 250#250
DELAY_WIN_MIN_A = 0
DELAY_WIN_MAX_A = 1000#1000
DELAY_WIN_HOPS_A = int((DELAY_WIN_MAX_A-DELAY_WIN_MIN_A)/DELAY_WIN_HOP_SIZE)
#RAW BRIR alignment
CUTOFF_ALIGNMENT_BRIR = 110#110
MIN_T_SHIFT_B = -1000#700
MAX_T_SHIFT_B = 250#0
#AIR alignment with other AIR sets
MIN_T_SHIFT_D = -750#
MAX_T_SHIFT_D = 200#

#sub alignment in subbrir generation
SUB_EQ_MODE=1#0=manual peaking filters,1=auto
CUTOFF_ALIGNMENT_SUBRIR = 75
T_SHIFT_INTERVAL_S = 10
MIN_T_SHIFT_S = -1000#-900
MAX_T_SHIFT_S = 50#250
NUM_INTERVALS_S = int(np.abs((MAX_T_SHIFT_S-MIN_T_SHIFT_S)/T_SHIFT_INTERVAL_S))

ALIGN_LIMIT_AZIM = 1#option to limit azimuths for TD alignment
TOTAL_SAMPLES_HRIR=256
TOTAL_CHAN_HRIR = 2
TOTAL_CHAN_BRIR = 2
INTERIM_ELEVS = 1
NEAREST_AZ_BRIR_REVERB = 15
OUTPUT_AZIMS_REVERB = int(360/NEAREST_AZ_BRIR_REVERB)


#apply room sorting based on size
SORT_ROOMS_BY_SIZE=1
#enable shifting of rooms list
ROLL_ROOM = 0

#apply room weighting based on size
ROOM_WEIGHTING_DESC=1
SPECT_SNAP_F0=160
SPECT_SNAP_F1=3500#1500
SPECT_SNAP_M_F0=800#1200
SPECT_SNAP_M_F1=2100#1800
IGNORE_MS=0#50,100
#window for reverb shaping: 1=Hanning,2=Bartlett,3=blackman,4=hamming
WINDOW_TYPE=2#1
ALIGNMENT_METHOD = 5

#directories for app related data, relative to app directory
BASE_DIR_OS= path.abspath(path.dirname(path.dirname(__file__)))#using os.path
SCRIPT_DIR_OS = path.abspath(path.dirname(__file__))#using os.path
BASE_DIR_PATH = Path(__file__).resolve().parents[1] #using Pathlib
SCRIPT_DIR_PATH = Path(__file__).resolve().parent #using Pathlib
SCRIPT_DIRECTORY = path.dirname(path.abspath(sys.argv[0]))#old method using os.path
DATA_DIR_INT = pjoin(BASE_DIR_OS, 'data','interim')
DATA_DIR_SUB = pjoin(BASE_DIR_OS, 'data','interim','lf_brir')#sub
DATA_DIR_HRIR_NPY = pjoin(BASE_DIR_OS, 'data','interim','hrir')
DATA_DIR_HRIR_NPY_DH = pjoin(BASE_DIR_OS, 'data','interim','hrir','dh')
DATA_DIR_HRIR_NPY_DH_V = pjoin(BASE_DIR_OS, 'data','interim','hrir','dh','h','VIKING')
DATA_DIR_HRIR_NPY_HL = pjoin(BASE_DIR_OS, 'data','interim','hrir','hl')
DATA_DIR_HRIR_NPY_USER = pjoin(BASE_DIR_OS, 'data','interim','hrir','user')
DATA_DIR_HRIR_NPY_INTRP = pjoin(BASE_DIR_OS, 'data','interim','hrir','intrp')
DATA_DIR_EXT = pjoin(BASE_DIR_OS, 'data','external')
DATA_DIR_SOFA = pjoin(BASE_DIR_OS, 'data','external','SOFA')
DATA_DIR_SOFA_USER = pjoin(BASE_DIR_OS, 'data','user','SOFA')
DATA_DIR_IRS_USER = pjoin(BASE_DIR_OS, 'data','user','IRs')
DATA_DIR_AS_USER = pjoin(BASE_DIR_OS, 'data','interim','reverberation','user')
DATA_DIR_RT_USER = pjoin(BASE_DIR_OS, 'data','interim','room_targets','user')
DATA_DIR_ASSETS = pjoin(BASE_DIR_OS, 'data','external','assets')
DATA_DIR_RAW = pjoin(BASE_DIR_OS, 'data','raw')
DATA_DIR_RAW_HP_MEASRUEMENTS = pjoin(BASE_DIR_OS, 'data','raw','headphone_measurements')
DATA_DIR_ROOT = pjoin(BASE_DIR_OS, 'data')
DATA_DIR_OUTPUT = pjoin(BASE_DIR_OS, 'data','processed')
PROJECT_FOLDER = 'ASH-Outputs'   
PROJECT_FOLDER_SSD = 'ASH-Custom-Set' 
PROJECT_FOLDER_BRIRS = pjoin(PROJECT_FOLDER, 'BRIRs')  
PROJECT_FOLDER_BRIRS_SSD = pjoin(PROJECT_FOLDER_SSD, 'BRIRs')  
PROJECT_FOLDER_BRIRS_SOFA = pjoin(PROJECT_FOLDER, 'BRIRs', 'SOFA')  
PROJECT_FOLDER_HRIRS_SOFA = pjoin(PROJECT_FOLDER, 'HRIRs', 'SOFA')
PROJECT_FOLDER_CONFIGS = pjoin(PROJECT_FOLDER, 'E-APO-Configs') 
PROJECT_FOLDER_CONFIGS_BRIR = pjoin(PROJECT_FOLDER, 'E-APO-Configs','BRIR-Convolution') 
PROJECT_FOLDER_CONFIGS_HPCF = pjoin(PROJECT_FOLDER, 'E-APO-Configs','HpCF-Convolution') 
PROJECT_FOLDER_HPCFS = pjoin(PROJECT_FOLDER, 'HpCFs')  
PROJECT_FOLDER_HPCFS_SSD = pjoin(PROJECT_FOLDER, 'HpCFs')
ICON_LOCATION= pjoin(DATA_DIR_RAW, 'ash_icon_1.ico')  
METADATA_FILE = pjoin(BASE_DIR_OS, 'metadata.json')
FOLDER_BRIRS_LIVE = 'live_dataset'
SETTINGS_FILE_OLD = pjoin(BASE_DIR_OS, 'settings.ini')#prior to v3.3.0, was stored in app directory
SETTINGS_FILE_NEW = get_settings_path()#3.3.0 onwards is stored in user directory
# Initialize SETTINGS_FILE to None or new by default (you can override it in main)
SETTINGS_FILE = SETTINGS_FILE_NEW
SETTINGS_DIR = os.path.dirname(SETTINGS_FILE) 
__version__ = get_version(metadata_file_path=METADATA_FILE)

#constants for writing WAVs
NEAREST_AZ_WAV = 15
MIN_ELEV_WAV=0
MAX_ELEV_WAV=0
ELEV_OFFSET_WAV = np.abs(MIN_ELEV_WAV)
OUTPUT_AZIMS_WAV = int(360/NEAREST_AZ_WAV)

#GUI related
TOOLTIP_GAIN = 'Positive values may result in clipping'
TOOLTIP_ELEVATION = 'Positive values are above the listener while negative values are below the listener'
TOOLTIP_AZIMUTH = 'Positive values are to the right of the listener while negative values are to the left'
RADIUS=85
X_START=110
Y_START=100
#GUI
PROGRESS_FIN=' Active   '
PROGRESS_START=' Ready to Apply'
PROGRESS_START_HPCF='Ready to Apply Selection'
PROGRESS_START_BRIR='Ready to Apply Parameters'
PROGRESS_FIN_ALT=' Finished   '
PROGRESS_START_ALT=' Ready to Start'
PROCESS_BUTTON_BRIR='Apply Parameters'
PROCESS_BUTTON_HPCF='Apply Selection'


HP_COMP_LIST = ['In-Ear Headphones - High Strength','In-Ear Headphones - Low Strength','Over/On-Ear Headphones - High Strength','Over/On-Ear Headphones - Low Strength','None']
HP_COMP_LIST_SHORT = ['In-Ear-High','In-Ear-Low','Over+On-Ear-High','Over+On-Ear-Low','Hp-Comp-None']



#Room target related
ROOM_TARGETS_DICT = {}
ROOM_TARGET_LIST = []
ROOM_TARGET_LIST_SHORT = []
ROOM_TARGET_KEYS = []
ROOM_TARGET_INDEX_MAP = []

# Automatically refresh on import
refresh_room_targets()


#BRIR writing
#number of directions to grab to built HESUVI IR
NUM_SOURCE_AZIM_HE = 7
NUM_OUT_CHANNELS_HE = NUM_SOURCE_AZIM_HE*2
NUM_OUT_CHANNELS_TS = 4
NUM_OUT_CHANNELS_MC = 16

PRESET_16CH_LABELS = [
    # Standard home 7.1 front-back layout
    "FL|FR|FC|LFE|SL|SR|BL|BR",
    # Back channels swapped (common in some software stacks)
    "FL|FR|FC|LFE|BL|BR|SL|SR",
    # Cinema / Dolby-style front-heavy
    "FC|LFE|FL|FR|SL|SR|BL|BR",
    # Wider surround / side-priority
    "FL|FR|SL|SR|FC|LFE|BL|BR",
    # LFE between surrounds
    "FL|FR|FC|SL|SR|LFE|BL|BR",
    # Experimental / gaming style
    "FL|FR|BL|BR|FC|LFE|SL|SR"
]

#Equalizer APO constants
BRIR_EXPORT_ENABLE = True
AZIM_DICT = {'WIDE_BL':'-135','WIDE_BR':'135','NARROW_BL':'-150','NARROW_BR':'150','WIDEST_BL':'-120','WIDEST_BR':'120','SL':'-90','SR':'90','FL':'-30','FR':'30','FC':'0','WIDE_FL':'-35','WIDE_FR':'35','NARROW_FL':'-25','NARROW_FR':'25'}
CHANNEL_CONFIGS = [['2.0_Stereo','2.0','2.0 Stereo'],['2.0_Stereo_Narrow','2.0N','2.0 Stereo (narrow placement)'],['2.0_Stereo_Wide','2.0W','2.0 Stereo (wide placement)'],['7.1_Surround_Narrow_Back','7.1N','7.1 surround (narrow back placement)'],['7.1_Surround_Wide_Back','7.1W','7.1 surround (wide back placement)'],['5.1_Surround','5.1','5.1 surround']]
AUDIO_CHANNELS = ['2.0 Stereo','2.0 Stereo Upmix to 7.1','5.1 Surround','7.1 Surround','7.1 Downmix to Stereo']
UPMIXING_METHODS = ['Method A','Method B']
AUTO_GAIN_METHODS = ['Disabled','Prevent Clipping','Align Low Frequencies','Align Mid Frequencies']
NUM_SPEAK_CONFIGS = len(CHANNEL_CONFIGS)
ELEV_ANGLES_WAV_BK = [-30,0,30]
ELEV_ANGLES_WAV = [45,30,15,0,-15,-30,-45]#[-45,-30,-15,0,15,30,45]
ELEV_ANGLES_WAV_LOW = [30,15,0,-15,-30]
ELEV_ANGLES_WAV_MED = [45,30,15,0,-15,-30,-45]
ELEV_ANGLES_WAV_HI = [45,30,20,15,10,5,0,-5,-10,-15,-20,-30,-45]
ELEV_ANGLES_WAV_MAX = [40,30,20,15,10,5,0,-5,-10,-15,-20,-30,-40]
AZ_ANGLES_FL_WAV = [-90,-75,-60,-45,-40,-35,-30,-25,-20,-15,0]
AZ_ANGLES_FR_WAV = [90,75,60,45,40,35,30,25,20,15,0]
AZ_ANGLES_C_WAV = [-5,5,0]
AZ_ANGLES_SL_WAV = [-150,-135,-120,-105,-90,-75,-60]
AZ_ANGLES_SR_WAV = [150,135,120,105,90,75,60]
AZ_ANGLES_RL_WAV = [180,-165,-150,-135,-120,-105,-90]
AZ_ANGLES_RR_WAV = [180,165,150,135,120,105,90]
AZ_ANGLES_FL_WAV.reverse()
AZ_ANGLES_FR_WAV.reverse()
AZ_ANGLES_C_WAV.reverse()
AZ_ANGLES_SL_WAV.reverse()
AZ_ANGLES_SR_WAV.reverse()
AZ_ANGLES_RL_WAV.reverse()
AZ_ANGLES_RR_WAV.reverse()
AZIM_HORIZ_RANGE = {5,20,25,35,40,355,340,335,325,320}
AZIM_EXTRA_RANGE = {25,30,35,335,330,325}
EAPO_MUTE_GAIN=-70.0

#spatial resolution
SPATIAL_RES_LIST = ['Low','Medium','High']#0=low,1=med,2=high,3=max v3.6 was ['Low','Medium','High','Max']
SPATIAL_RES_LIST_LIM = ['Low','Medium','High']
SPATIAL_RES_ELEV_DESC = ['-30 to 30 degrees in 15 degree steps.','-45 to 45 degrees in 15 degree steps.',
                         '-50 to 50 degrees (WAV export) or -60 to 60 degrees (SOFA export) in 5 degree steps.','-40 to 40 degrees (WAV export) or -40 to 60 degrees (SOFA export) in 2 degree steps.']
SPATIAL_RES_AZIM_DESC = ['0 to 360 degrees in varying steps.','0 to 360 degrees in varying steps.',
                         '0 to 360 degrees in 5 degree steps.','0 to 360 degrees in 2 degree steps.']
SPATIAL_RES_ELEV_RNG = ['-30° to 30°','-45° to 45°',
                         '-50° to 50° (WAV) or -60° to 60° (SOFA)','-40° to 40° (WAV) or -40° to 60° (SOFA)']
SPATIAL_RES_AZIM_RNG = ['0° to 360°','0° to 360°',
                         '0° to 360°','0° to 360°']
SPATIAL_RES_ELEV_STP = ['15° steps','15° steps',
                         '5° steps','2° steps']
SPATIAL_RES_AZIM_STP = ['varying','varying',
                         '5° steps','2° steps']
NUM_SPATIAL_RES = len(SPATIAL_RES_LIST)
SPATIAL_RES_ELEV_MIN_IN=[-60, -60, -60, -40 ]#as per hrir dataset
SPATIAL_RES_ELEV_MAX_IN=[60, 60, 60, 60 ]#as per hrir dataset
SPATIAL_RES_ELEV_MIN_OUT=[-30, -45, -50, -40 ]#reduced set
SPATIAL_RES_ELEV_MAX_OUT=[30, 45, 50, 40 ]#reduced set
SPATIAL_RES_ELEV_NEAREST_IN=[5, 5, 5, 2]#as per hrir dataset
SPATIAL_RES_AZIM_NEAREST_IN=[5, 5, 5, 2]#as per hrir dataset
SPATIAL_RES_ELEV_NEAREST_PR=[15, 15, 5, 2]#3.1.0 increased processing resolution from [15, 15, 5, 2]
SPATIAL_RES_AZIM_NEAREST_PR=[5, 5, 5, 2]#
# SPATIAL_RES_ELEV_NEAREST_PR_R=[15, 15, 5, 2]#
# SPATIAL_RES_AZIM_NEAREST_PR_R=[5, 5, 5, 2]#
SPATIAL_RES_ELEV_NEAREST_OUT=[15, 15, 5, 2]#
SPATIAL_RES_AZIM_NEAREST_OUT=[5, 5, 5, 2]#



#hpcf related
NUM_ITER = 8 #was 4
PARAMS_PER_ITER = 4 #was 8
SENSITIVITY = 0.003 #0.003
FREQ_CUTOFF = 20000#f in bins
FREQ_CUTOFF_GEQ = 29000
EAPO_GAIN_ADJUST = 0.9*1.1*1.1*1.1
EAPO_QF_ADJUST = 0.5*0.8
HPCF_FIR_LENGTH = 384#was 1024, then 512
HPCF_SAMPLE_DEFAULT = 'Sample A'
HPCF_DATABASE_LIST = ['ASH Filters', 'Compilation']
# mapping: new_schema_col -> old_schema_col
HPCF_DB_SCHEMA_MAP = {
    "type": "brand",
    "headphone_name": "headphone",
    "source": "sample",
    "fir_json": "fir"
}#"source": "brand","headphone_name": "headphone","type": "sample",
# columns always expected by your app
HPCF_DB_STANDARD_COLUMNS = [
    "brand", "headphone", "sample", "sample_id",
    "fir", "graphic_eq", "graphic_eq_31", "graphic_eq_103", "created_on"
]

#retrieve geq frequency list as an array - 127 bands
GEQ_SET_F_127 = []
csv_fname = pjoin(DATA_DIR_RAW, 'wavelet_geq_freqs.csv')
with open(csv_fname, encoding='utf-8-sig', newline='') as inputfile:
    for row in csv.reader(inputfile):
        GEQ_SET_F_127.append(int(row[0]))
GEQ_SET_F_127 = np.array(GEQ_SET_F_127)
#retrieve geq frequency list as an array - 31 bands
GEQ_SET_F_31 = []
csv_fname = pjoin(DATA_DIR_RAW, '31_band_geq_freqs.csv')
with open(csv_fname, encoding='utf-8-sig', newline='') as inputfile:
    for row in csv.reader(inputfile):
        GEQ_SET_F_31.append(int(row[0]))
GEQ_SET_F_31 = np.array(GEQ_SET_F_31)
#retrieve geq frequency list as an array - 103 bands
GEQ_SET_F_103 = []
csv_fname = pjoin(DATA_DIR_RAW, 'hesuvi_geq_freqs.csv')
with open(csv_fname, encoding='utf-8-sig', newline='') as inputfile:
    for row in csv.reader(inputfile):
        GEQ_SET_F_103.append(int(row[0]))
GEQ_SET_F_103 = np.array(GEQ_SET_F_103)


#gui
DIRECT_GAIN_MAX=10#8
DIRECT_GAIN_MIN=-10#-8
EAPO_ERROR_CODE=501
ER_RISE_MAX=10#in ms
ER_RISE_MIN=0#

SAMPLE_RATE_LIST = ['44.1 kHz', '48 kHz', '96 kHz']
SAMPLE_RATE_DICT = {'44.1 kHz': 44100, '48 kHz': 48000, '96 kHz': 96000}  
BIT_DEPTH_LIST = ['24 bit', '32 bit']
BIT_DEPTH_DICT = {'24 bit': 'PCM_24', '32 bit': 'PCM_32'}  #{'24 bit': 'PCM_24', '32 bit': 'PCM_32'}  
HRTF_SYM_LIST = ['Disabled', 'Mirror Left Side', 'Mirror Right Side']
RESAMPLE_MODE_LIST = ['Performance', 'Quality']

#AIR and BRIR reverberation processing
LIMIT_REBERB_DIRS=True
MIN_REVERB_DIRS=1
MAX_IRS=3500#2520
MAX_IRS_TRANSFORM=2500#2520,1550,1050,2500 for combined
IR_MIN_THRESHOLD=800#before this number of IRs, transform will be applied to expand dataset
IR_MIN_THRESHOLD_FULLSET=1600
DIRECTION_MODE=3#0=one set of azimuths for each source, 1 = separate set of azimuths for left and right hem,2 = random using triangle distribution, 3 = biased_spherical_coordinate_sampler
MAG_COMP=True#enables DF compensation in AIR processing   True
RISE_WINDOW=True#enables windowing of initial rise in AIR processing

#deprecated
RT60_MIN=200
RT60_MAX_S=1250
RT60_MAX_L=2250

#Acoustic space related
AC_SPACE_LIST_COMP = ['generic_room_a', 'auditorium_a']
AC_SPACE_LIST_KU100 = ['concert_hall_a','music_chamber_a','theater_a']
AC_SPACE_LIST_SOFA_0 = ['audio_lab_a','office_a','studio_a','audio_lab_i']#sofa objects are divided evenly among all sets. (limited measurements per sofa)
AC_SPACE_LIST_SOFA_1 = ['']#sofa objects are divided evenly and distributed into specific sets, cycles sets (limited measurements per sofa)
AC_SPACE_LIST_SOFA_2 = ['']#sofa objects are divided among left and right hemispheres (limited measurements per sofa)
AC_SPACE_LIST_SOFA_3 = ['','broadcast_studio_a']#sofa objects are divided among left and right hemispheres based on sub folders (all measurements per sofa)
AC_SPACE_LIST_SOFA_4 = ['']#sofa IRs are divided among left and right hemispheres. 1st half of sofa obj goes to left hem, 2nd half goes to right. shared evenly across sofa objs (limited measurements per sofa) #'audio_lab_b'
AC_SPACE_LIST_SOFA_5 = ['','audio_lab_g','audio_lab_h','control_room_a','conference_room_a', 'audio_lab_b']#sofa IRs are divided among left and right hemispheres. 1st half of sofa obj goes to left hem, 2nd half goes to right (all measurements per sofa)
AC_SPACE_LIST_SOFA_6 = ['','office_b','studio_b','broadcast_studio_b']#sofa objects are divided evenly among all sets. (all measurements per sofa)
AC_SPACE_LIST_ISO = ['sub_set_a','sub_set_b']
AC_SPACE_LIST_ANU = ['studio_c','studio_d']#'studio_b'
AC_SPACE_LIST_44100 = ['outdoors_b']
AC_SPACE_LIST_SUBRIRDB = ['sub_set_c']
#AC_SPACE_LIST_NR = ['tatami_room_a','listening_room_a','listening_room_b','listening_room_c','listening_room_d','listening_room_g','courtyard_a','church_a']
AC_SPACE_LIST_CUTOFF = ['concert_hall_a']
AC_SPACE_LIST_LIM_CHANS = ['concert_hall_a','hall_a']
AC_SPACE_LIST_LIM_SETS = ['hall_a', 'outdoors_a','broadcast_studio_a','concert_hall_a', 'outdoors_b','seminar_room_a','seminar_room_b','lobby_a','seminar_room_c','broadcast_studio_b','office_b']
AC_SPACE_CHAN_LIMITED=6#3,4,5
AC_SPACE_LIST_NOROLL = ['auditorium_a', 'auditorium_b','office_c','seminar_room_e']
AC_SPACE_LIST_NOCOMP = ['audio_lab_d']
AC_SPACE_LIST_COMPMODE1 = ['control_room_a', 'tatami_room_a', 'office_a','studio_c','studio_d']#'studio_b'
AC_SPACE_LIST_WINDOW = ['hall_a', 'outdoors_a','seminar_room_a', 'outdoors_b']#,'broadcast_studio_a'
AC_SPACE_LIST_WINDOW_ALL = ['']
AC_SPACE_LIST_SLOWRISE = ['studio_c','studio_d','audio_lab_i','hall_a','small_room_a','medium_room_a','large_room_a','small_room_b','medium_room_b','large_room_b','small_room_c','small_room_d','small_room_e','small_room_f']
AC_SPACE_LIST_SUB = ['sub_set_a','sub_set_b','sub_set_c','sub_set_d','sub_set_e','sub_set_f','sub_set_g','sub_set_h','sub_set_i']
AC_SPACE_LIST_RWCP = ['audio_lab_f','conference_room_b', 'tatami_room_a']
AC_SPACE_LIST_VARIED_R = [' ']
AC_SPACE_LIST_AVG = ['audio_lab_a','audio_lab_b','audio_lab_d','control_room_a','conference_room_a','control_room_a','office_a','audio_lab_g', 'audio_lab_f','audio_lab_e','audio_lab_h']
AC_SPACE_LIST_LIM_AZ = [' ']
# #HRTF_LIST_NUM = ['01: Neumann KU 100 (SADIE)', '02: Neumann KU 100 (TH Köln)', '03: FABIAN HATS', '04: B&K Type 4128', '05: B&K Type 4128C (MMHR-HRIR)', '06: DADEC (MMHR-HRIR)', '07: HEAD acoustics HMSII.2 (MMHR-HRIR)', '08: KEMAR (MMHR-HRIR)', '09: KEMAR-N (MIT)', '10: KEMAR-L (MIT)', '11: KEMAR (SADIE)', '12: KEMAR-N (PKU-IOA)', '13: KEMAR-L (PKU-IOA)']
AC_SPACE_LIST_SORT_BY = ['Name','Reverberation Time']
AC_SPACE_LIST_HI_FC = []
AC_SPACE_LIST_MID_FC = []
AC_SPACE_LIST_LOW_FC = []

#load other lists from csv file
AC_SPACE_FIELD_NAMES = [
    'file_name', 'name_gui', 'name_short', 'name_src', 'est_rt60', 'meas_rt60', 'fade_start',
    'low_rt60', 'folder', 'version', 'gdrive_link', 'gain', 'f_crossover', 'order_crossover',
    'noise_reduce', 'source_hrtf_dataset', 'source_hrtf_id', 'description', 'notes',
    'source_dataset', 'citation', 'citation_link', 'high_fc', 'mid_fc', 'low_fc'
]

# Global variable declarations for clarity
reverb_data = []
AC_SPACE_LIST_GUI = []
AC_SPACE_LIST_SHORT = []
AC_SPACE_LIST_SRC = []
AC_SPACE_EST_R60 = []
AC_SPACE_MEAS_R60 = []
AC_SPACE_FADE_START = []
AC_SPACE_GAINS = []
AC_SPACE_DESCR = []
AC_SPACE_DATASET = []
AC_SPACE_LIST_NR = []
AC_SPACE_LIST_LOWRT60 = []
AC_SPACE_LIST_HIRT60 = []
AC_SPACE_LIST_COMP = []
AC_SPACE_LIST_MAX_SRC = []
AC_SPACE_LIST_ID_SRC = []

METADATA_CSV_KEY = "metadata"
USER_CSV_KEY = "_metadata"
ASI_USER_CSV_KEY = "_asi-metadata"
SUB_USER_CSV_KEY = "_sub-metadata"
REV_METADATA_FILE_NAME='reverberation_metadata_v2.csv'#was 'reverberation_metadata.csv' prior to 3.6
#section to load dict lists containing acoustic space metadata
#new: load as dict lists
refresh_acoustic_space_metadata()


# ## Accessing data example
# for entry in reverb_data:
#     print(entry['name_gui'], entry['est_rt60'])
# #
# AC_SPACE_LIST_GUI = extract_column(reverb_data, 'name_gui')                        # strings
# AC_SPACE_MEAS_R60 = extract_column(reverb_data, 'meas_rt60')                      # ints
# AC_SPACE_FADE_START = extract_column(reverb_data, 'fade_start')                   # ints
# AC_SPACE_GAINS = extract_column(reverb_data, 'gain')                              # floats
# AC_SPACE_LIST_LOWRT60 = extract_column(reverb_data, 'name_src', 'low_rt60', 'Yes')
# AC_SPACE_LIST_COMP = extract_column(reverb_data, 'name_src', 'folder', 'comp_bin')
# # All rows using 'max' HRTF dataset
# AC_SPACE_LIST_MAX_SRC = [row['name_src'] for row in reverb_data if row.get('source_hrtf_dataset') == 'max']



#SOFA related

#new: load as dict lists
csv_directory = DATA_DIR_SOFA
sofa_data = load_csv_as_dicts(csv_directory,'supported_conventions.csv')

SOFA_COMPAT_CONV = extract_column(data=sofa_data, column='Convention')  
SOFA_COMPAT_VERS = extract_column(data=sofa_data, column='Version') 
SOFA_COMPAT_CONVERS = extract_column(data=sofa_data, column='SOFAConventionsVersion')                       # strings
SOFA_OUTPUT_CONV = extract_column(data=sofa_data, column='Convention', condition_key='OutputFormat', condition_value='Yes', return_all_matches=True)
SOFA_OUTPUT_VERS = extract_column(data=sofa_data, column='Version', condition_key='OutputFormat', condition_value='Yes', return_all_matches=True)
SOFA_OUTPUT_CONVERS = extract_column(data=sofa_data, column='SOFAConventionsVersion', condition_key='OutputFormat', condition_value='Yes', return_all_matches=True)



#HRTF related - individual datasets
HRTF_A_GAIN_ADDIT = 2.5#
#Strings
HRTF_TYPE_LIST = ['Dummy Head / Head & Torso Simulator', 'Human Listener', 'User SOFA Input', 'Favourites']
HRTF_TYPE_LIST_FULL = ['Dummy Head / Head & Torso Simulator', 'Human Listener', 'User SOFA Input','Dummy Head - Max Resolution', 'Favourites']
HRTF_BASE_LIST_FAV = ['No favourites found']
#new: load as dict lists
csv_directory = DATA_DIR_HRIR_NPY
HRIR_METADATA_NAME='hrir_metadata.csv'
hrtf_data = load_csv_as_dicts(csv_directory,HRIR_METADATA_NAME)
HRTF_DATASET_LIST_INDV = extract_column(data=hrtf_data, column= 'dataset', condition_key='hrtf_type', condition_value='Human Listener', return_all_matches=True)
HRTF_DATASET_LIST_DUMMY = extract_column(data=hrtf_data, column='dataset', condition_key='hrtf_type', condition_value='Dummy Head / Head & Torso Simulator', return_all_matches=True)
HRTF_DATASET_LIST_DUMMY_MAX = extract_column(data=hrtf_data, column='dataset', condition_key='hrtf_index_max', condition_value=0, condition_op='!=', return_all_matches=True)
HRTF_TYPE_DEFAULT = extract_column(data=hrtf_data, column='hrtf_type', condition_key='default', condition_value='Yes', return_all_matches=False)
HRTF_DATASET_DEFAULT = extract_column(data=hrtf_data, column='dataset', condition_key='default', condition_value='Yes', return_all_matches=False)
HRTF_LISTENER_DEFAULT = extract_column(data=hrtf_data, column='name_gui', condition_key='default', condition_value='Yes', return_all_matches=False)

HRTF_DATASET_LIST_CUSTOM = ['N/A']
HRTF_DATASET_CUSTOM_DEFAULT = HRTF_DATASET_LIST_CUSTOM[0]
# Remove duplicates by converting to a set and back to a list
HRTF_DATASET_LIST_INDV = list(set(HRTF_DATASET_LIST_INDV))
# Sort the list alphabetically
HRTF_DATASET_LIST_INDV.sort()
# Remove duplicates by converting to a set and back to a list
HRTF_DATASET_LIST_DUMMY = list(set(HRTF_DATASET_LIST_DUMMY))
HRTF_DATASET_LIST_DUMMY_MAX = list(set(HRTF_DATASET_LIST_DUMMY_MAX))
# Sort the list alphabetically
HRTF_DATASET_LIST_DUMMY.sort()
HRTF_DATASET_LIST_DUMMY_MAX.sort()
#create a dictionary
HRTF_TYPE_DATASET_DICT = {
    HRTF_TYPE_LIST_FULL[0]: HRTF_DATASET_LIST_DUMMY,
    HRTF_TYPE_LIST_FULL[1]: HRTF_DATASET_LIST_INDV,
    HRTF_TYPE_LIST_FULL[2]: HRTF_DATASET_LIST_CUSTOM,
    HRTF_TYPE_LIST_FULL[3]: HRTF_DATASET_LIST_DUMMY_MAX,
    HRTF_TYPE_LIST_FULL[4]: HRTF_DATASET_LIST_CUSTOM
}
HRTF_POLARITY_LIST = ['Auto Select','Original','Reversed']

HRTF_DATASET_LIST_DEFAULT = HRTF_TYPE_DATASET_DICT.get(HRTF_TYPE_DEFAULT)
HRTF_LISTENER_LIST_DEFAULT = extract_column(data=hrtf_data, column='name_gui', condition_key='dataset', condition_value=HRTF_DATASET_DEFAULT, return_all_matches=True)
HRTF_AVERAGED_NAME_GUI = "Averaged HRTF"
HRTF_AVERAGED_NAME_FILE = "hrir_favourites_averaged"
HRTF_USER_SOFA_DEFAULT = 'No SOFA files found'
HRTF_USER_SOFA_PREFIX = 'u-'

#SUB Related
SUB_FC_SETTING_LIST = ['Auto Select', 'Custom Value']
SUB_FC_MIN=20
SUB_FC_MAX=150
SUB_FC_DEFAULT=120

# #new: load as dict lists

# Global variable declarations for clarity
sub_data=[]
SUB_RESPONSE_LIST_GUI=[]
SUB_RESPONSE_LIST_SHORT=[]
SUB_RESPONSE_LIST_AS=[]
SUB_RESPONSE_LIST_R60=[]
SUB_RESPONSE_LIST_COMM=[]
SUB_RESPONSE_LIST_RANGE=[]
SUB_RESPONSE_LIST_TOL=[]
SUB_RESPONSE_LIST_DATASET=[]
SUB_RESPONSE_LIST_SOURCE = []
SUB_RESPONSE_LIST_LISTENER = []

SUB_PLOT_LIST = ['Magnitude', 'Group Delay']
SUB_FIELD_NAMES = [
    'file_name', 'name_gui', 'name_short', 'acoustic_space', 'est_rt60', 'comments', 'frequency_range',
    'tolerance', 'folder','dataset'
]



refresh_sub_responses()




#plotting
IMPULSE=np.zeros(N_FFT)
IMPULSE[0]=1
FR_FLAT_MAG = np.abs(np.fft.fft(IMPULSE))


# Define the center frequencies of the 1/3-octave bands
OCTAVE_BANDS = np.array([50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 
                         500, 630, 800, 1000, 1250, 1600, 2000, 2500, 
                         3150, 4000, 5000, 6300, 8000, 10000])


# Global variable to track pending save
_SAVE_DEBOUNCE_TIME = 0.1  # seconds

#GUI Defaults



# === GUI Defaults ===

DEFAULTS = {
    # === Not directly GUI related, don't remove ===
    "path": "C:\\Program Files\\EqualizerAPO\\config",

    # === GUI ===
    "tab_bar": 0,
    
    # === Export tab HPCF ===
    "fde_hpcf_brand": "",#hpcf_brand, not known until db loaded
    "fde_hpcf_headphone": "",#hpcf_headphone, not known until db loaded
    "fde_hpcf_sample": "",#hpcf_sample, not known until db loaded
    "fir_hpcf_toggle": True,
    "fir_st_hpcf_toggle": False,
    "hesuvi_hpcf_toggle": False,
    "geq_hpcf_toggle": False,
    "geq_31_hpcf_toggle": False,
    "fde_hpcf_active_database": HPCF_DATABASE_LIST[0],

    # === Export tab HRTF / BRIR ===
    "fde_brir_hp_type": HP_COMP_LIST[2],#brir_hp_type
    "fde_hrtf_symmetry": HRTF_SYM_LIST[0],#hrtf_symmetry
    "fde_room_target": ROOM_TARGET_LIST[1],#room_target
    "fde_direct_gain": 2.0,#direct_gain
    "fde_direct_gain_slider": 2.0,
    "fde_acoustic_space": AC_SPACE_LIST_GUI[2],#acoustic_space
    "fde_brir_hrtf_type": HRTF_TYPE_DEFAULT,#brir_hrtf_type
    "fde_brir_hrtf_dataset": HRTF_DATASET_DEFAULT,#brir_hrtf_dataset
    "fde_brir_hrtf": HRTF_LISTENER_DEFAULT,#brir_hrtf
    "fde_brir_spat_res": SPATIAL_RES_LIST[2],#brir_spat_res, high as default
    "fde_crossover_f_mode": SUB_FC_SETTING_LIST[0],#crossover_f_mode
    "fde_crossover_f": SUB_FC_DEFAULT,#crossover_f
    "fde_sub_response": SUB_RESPONSE_LIST_GUI[3] if len(SUB_RESPONSE_LIST_GUI) > 3 else SUB_RESPONSE_LIST_GUI[0],#sub_response
    "fde_hp_rolloff_comp": False,#hp_rolloff_comp
    "fde_fb_filtering": False,#fb_filtering
    #"hrtf_default": HRTF_LISTENER_DEFAULT,
    
    "dir_brir_toggle": True,
    "ts_brir_toggle": False,
    "sofa_brir_toggle": False,
    "hesuvi_brir_toggle": False,
    "multi_chan_brir_toggle": False,
    
    "fde_wav_sample_rate": SAMPLE_RATE_LIST[0],#wav_sample_rate
    "fde_wav_bit_depth": BIT_DEPTH_LIST[0],#wav_bit_depth
    
    #default lists - used for listboxes
    "hrtf_list_favs": HRTF_BASE_LIST_FAV,
    "brir_hrtf_dataset_list": HRTF_DATASET_LIST_DEFAULT,
    "brir_hrtf_listener_list": HRTF_LISTENER_LIST_DEFAULT,
    


    # === QC Tab HPCF ===
    "qc_toggle_hpcf_history": False,
    "qc_hpcf_brand": "",#not known until db loaded
    "qc_hpcf_headphone": "",#not known until db loaded
    "qc_hpcf_sample": "",#not known until db loaded
    "qc_e_apo_curr_hpcf": "",#nothing generated to start with
    "qc_e_apo_sel_hpcf": "",#nothing generated to start with
    "qc_auto_apply_hpcf_sel": False,
    "e_apo_hpcf_conv": False,
    "qc_hpcf_active_database": HPCF_DATABASE_LIST[0],
    
    # === QC Tab BRIR ===
    "qc_brir_hp_type": HP_COMP_LIST[2],
    "qc_room_target": ROOM_TARGET_LIST[1],
    "qc_direct_gain": 2.0,
    "qc_direct_gain_slider": 2.0,
    "qc_acoustic_space": AC_SPACE_LIST_GUI[2],
    "qc_brir_hrtf_type": HRTF_TYPE_DEFAULT,
    "qc_brir_hrtf_dataset": HRTF_DATASET_DEFAULT,
    "qc_brir_hrtf": HRTF_LISTENER_DEFAULT,
    "qc_sub_response": SUB_RESPONSE_LIST_GUI[3] if len(SUB_RESPONSE_LIST_GUI) > 3 else SUB_RESPONSE_LIST_GUI[0],
    "qc_crossover_f_mode": SUB_FC_SETTING_LIST[0],
    "qc_crossover_f": SUB_FC_DEFAULT,
    "qc_hp_rolloff_comp": False,
    "qc_fb_filtering": False,
    "qc_lf_analysis_toggle": False,
    
    "qc_wav_sample_rate": SAMPLE_RATE_LIST[0],
    "qc_wav_bit_depth": BIT_DEPTH_LIST[0],
    
    "qc_e_apo_curr_brir_set": "",#nothing generated to start with
    "qc_e_apo_sel_brir_set": "",#nothing generated to start with
    "qc_e_apo_sel_brir_set_ts": "",#nothing generated to start with
    "e_apo_brir_conv": False,
    
    
    # === E-APO General ===
    "e_apo_audio_channels": AUDIO_CHANNELS[0],
    "e_apo_upmix_method": UPMIXING_METHODS[1],
    "e_apo_side_delay": 0,
    "e_apo_rear_delay": 10,
    "e_apo_mute": False,
    "e_apo_gain": 0.0,
    "e_apo_elev_angle": 0,
    "e_apo_az_angle_fl": -30,
    "e_apo_az_angle_fr": 30,
    "e_apo_az_angle_c": 0,
    "e_apo_az_angle_sl": -90,
    "e_apo_az_angle_sr": 90,
    "e_apo_az_angle_rl": -135,
    "e_apo_az_angle_rr": 135,
    #"e_apo_enable_hpcf": False,
    #"e_apo_enable_brir": False,
    "e_apo_autoapply_hpcf": False,
    "e_apo_prevent_clip": AUTO_GAIN_METHODS[1],


    # === E-APO Per-Channel Config ===
    # mute
    "e_apo_mute_fl": False,
    "e_apo_mute_fr": False,
    "e_apo_mute_c": False,
    "e_apo_mute_sl": False,
    "e_apo_mute_sr": False,
    "e_apo_mute_rl": False,
    "e_apo_mute_rr": False,
    # gain
    "e_apo_gain_oa": 0.0,
    "e_apo_gain_fl": 0.0,
    "e_apo_gain_fr": 0.0,
    "e_apo_gain_c": 0.0,
    "e_apo_gain_sl": 0.0,
    "e_apo_gain_sr": 0.0,
    "e_apo_gain_rl": 0.0,
    "e_apo_gain_rr": 0.0,
    # elevation
    "e_apo_elev_angle_fl": 0,
    "e_apo_elev_angle_fr": 0,
    "e_apo_elev_angle_c": 0,
    "e_apo_elev_angle_sl": 0,
    "e_apo_elev_angle_sr": 0,
    "e_apo_elev_angle_rl": 0,
    "e_apo_elev_angle_rr": 0,
    # === Hesuvi Per-Channel Angles ===
    "hesuvi_elev_angle_fl": 0,
    "hesuvi_elev_angle_fr": 0,
    "hesuvi_elev_angle_c": 0,
    "hesuvi_elev_angle_sl": 0,
    "hesuvi_elev_angle_sr": 0,
    "hesuvi_elev_angle_rl": 0,
    "hesuvi_elev_angle_rr": 0,
    "hesuvi_az_angle_fl": -30,
    "hesuvi_az_angle_fr": 30,
    "hesuvi_az_angle_c": 0,
    "hesuvi_az_angle_sl": -90,
    "hesuvi_az_angle_sr": 90,
    "hesuvi_az_angle_rl": -135,
    "hesuvi_az_angle_rr": 135,


    # === Misc ===
    "sofa_exp_conv": SOFA_OUTPUT_CONV[0],
    "check_updates_start_toggle": False,
    "hrtf_polarity_rev": HRTF_POLARITY_LIST[0],
    "force_hrtf_symmetry":HRTF_SYM_LIST[0],
    "er_delay_time":0.5,
    "export_resample_mode": RESAMPLE_MODE_LIST[0]
    

}






#mapping of ASH V3.5 settings keys to new keys (also new default keys)
#It’s structured as:"old_key": "new_key",
LEGACY_KEY_MAP = {
    # === General Paths & System ===
    "path": "path",
    "sampling_frequency": "fde_wav_sample_rate",#wav_sample_rate
    "bit_depth": "fde_wav_bit_depth",#wav_bit_depth
    "qc_wav_sample_rate": "qc_wav_sample_rate",
    "qc_wav_bit_depth": "qc_wav_bit_depth",

    # === BRIR / Room Simulation Parameters ===
    "brir_headphone_type": "fde_brir_hp_type",#brir_hp_type
    "brir_hrtf": "fde_brir_hrtf",#brir_hrtf
    "spatial_resolution": "fde_brir_spat_res",#brir_spat_res
    "brir_room_target": "fde_room_target",#room_target
    "brir_direct_gain": "fde_direct_gain",#direct_gain
    "brir_direct_gain_slider": "fde_direct_gain_slider",#not originally saved
    "acoustic_space": "fde_acoustic_space",#acoustic_space
    "brir_hrtf_type": "fde_brir_hrtf_type",#brir_hrtf_type
    "brir_hrtf_dataset": "fde_brir_hrtf_dataset",#brir_hrtf_dataset
    "crossover_f_mode": "fde_crossover_f_mode",#crossover_f_mode
    "crossover_f": "fde_crossover_f",#crossover_f
    "sub_response": "fde_sub_response",#sub_response
    "hp_rolloff_comp": "fde_hp_rolloff_comp",#hp_rolloff_comp
    "fb_filtering_mode": "fde_fb_filtering",#fb_filtering
    
    "sofa_exp_conv": "sofa_exp_conv",
    "hrtf_polarity": "hrtf_polarity_rev",

    # === Export Toggles ===
    "fir_hpcf_exp": "fir_hpcf_toggle",
    "fir_st_hpcf_exp": "fir_st_hpcf_toggle",
    "geq_hpcf_exp": "geq_hpcf_toggle",
    "geq_31_exp": "geq_31_hpcf_toggle",
    "hesuvi_hpcf_exp": "hesuvi_hpcf_toggle",
    "dir_brir_exp": "dir_brir_toggle",
    "ts_brir_exp": "ts_brir_toggle",
    "hesuvi_brir_exp": "hesuvi_brir_toggle",
    "multi_chan_brir_exp": "multi_chan_brir_toggle",
    "sofa_brir_exp": "sofa_brir_toggle",

    # === UI & Behavior ===
    "auto_check_updates": "check_updates_start_toggle",
    "force_hrtf_symmetry": "force_hrtf_symmetry",
    "er_delay_time": "er_delay_time",
    "show_hpcf_history": "qc_toggle_hpcf_history",
    "tab_selected": "tab_bar",
    "auto_apply_hpcf": "qc_auto_apply_hpcf_sel",
    "enable_hpcf": "e_apo_hpcf_conv",
    "enable_brir": "e_apo_brir_conv",
    "prevent_clip": "e_apo_prevent_clip",

    # === Current Selections ===
    "hpcf_current": "qc_e_apo_curr_hpcf",
    "hpcf_selected": "qc_e_apo_sel_hpcf",
    "brir_set_current": "qc_e_apo_curr_brir_set",
    "brir_set_selected": "qc_e_apo_sel_brir_set",

    # === Headphone & Sample Selections ===
    "qc_brand": "qc_hpcf_brand",
    "qc_headphone": "qc_hpcf_headphone",
    "qc_sample": "qc_hpcf_sample",

    # === BRIR Quick Config ===
    "qc_brir_headphone_type": "qc_brir_hp_type",
    "qc_brir_room_target": "qc_room_target",
    "qc_brir_direct_gain": "qc_direct_gain",
    "qc_brir_direct_gain_slider": "qc_direct_gain_slider",#not originally saved
    "qc_acoustic_space": "qc_acoustic_space",
    "qc_brir_hrtf": "qc_brir_hrtf",
    "qc_brir_hrtf_type": "qc_brir_hrtf_type",
    "qc_brir_hrtf_dataset": "qc_brir_hrtf_dataset",
    "qc_crossover_f_mode": "qc_crossover_f_mode",
    "qc_crossover_f": "qc_crossover_f",
    "qc_sub_response": "qc_sub_response",
    "qc_hp_rolloff_comp": "qc_hp_rolloff_comp",
    "qc_fb_filtering_mode": "qc_fb_filtering",

    # === Gain & Mute Controls (Channel Config) ===
    "mute_fl": "e_apo_mute_fl",
    "mute_fr": "e_apo_mute_fr",
    "mute_c": "e_apo_mute_c",
    "mute_sl": "e_apo_mute_sl",
    "mute_sr": "e_apo_mute_sr",
    "mute_rl": "e_apo_mute_rl",
    "mute_rr": "e_apo_mute_rr",

    "gain_oa": "e_apo_gain_oa",
    "gain_fl": "e_apo_gain_fl",
    "gain_fr": "e_apo_gain_fr",
    "gain_c": "e_apo_gain_c",
    "gain_sl": "e_apo_gain_sl",
    "gain_sr": "e_apo_gain_sr",
    "gain_rl": "e_apo_gain_rl",
    "gain_rr": "e_apo_gain_rr",

    "elev_fl": "e_apo_elev_angle_fl",
    "elev_fr": "e_apo_elev_angle_fr",
    "elev_c": "e_apo_elev_angle_c",
    "elev_sl": "e_apo_elev_angle_sl",
    "elev_sr": "e_apo_elev_angle_sr",
    "elev_rl": "e_apo_elev_angle_rl",
    "elev_rr": "e_apo_elev_angle_rr",

    "azim_fl": "e_apo_az_angle_fl",
    "azim_fr": "e_apo_az_angle_fr",
    "azim_c": "e_apo_az_angle_c",
    "azim_sl": "e_apo_az_angle_sl",
    "azim_sr": "e_apo_az_angle_sr",
    "azim_rl": "e_apo_az_angle_rl",
    "azim_rr": "e_apo_az_angle_rr",

    # === Upmix & Delays ===
    "upmix_method": "e_apo_upmix_method",
    "side_delay": "e_apo_side_delay",
    "rear_delay": "e_apo_rear_delay",
    "channel_config": "e_apo_audio_channels",

    # === HRTF Lists ===
    "hrtf_list_favs": "hrtf_list_favs",
}

#mapping of ASH V3.5 GUI keys to new GUI keys (also new default keys)
#It’s structured as:"old_key": "new_key",
LEGACY_GUI_KEY_MAP = {
    # === General Paths & System ===
    "path": "path",
    "wav_sample_rate": "fde_wav_sample_rate",#wav_sample_rate
    "wav_bit_depth": "fde_wav_bit_depth",#wav_bit_depth
    "qc_wav_sample_rate": "qc_wav_sample_rate",
    "qc_wav_bit_depth": "qc_wav_bit_depth",

    # === BRIR / Room Simulation Parameters ===
    "brir_hp_type": "fde_brir_hp_type",#brir_hp_type
    "brir_hrtf": "fde_brir_hrtf",#brir_hrtf
    "brir_spat_res": "fde_brir_spat_res",#brir_spat_res
    "rm_target_list": "fde_room_target",#room_target
    "direct_gain": "fde_direct_gain",#direct_gain
    "direct_gain_slider": "fde_direct_gain_slider",#direct_gain
    "acoustic_space_combo": "fde_acoustic_space",#acoustic_space
    "brir_hrtf_type": "fde_brir_hrtf_type",#brir_hrtf_type
    "brir_hrtf_dataset": "fde_brir_hrtf_dataset",#brir_hrtf_dataset
    "crossover_f_mode": "fde_crossover_f_mode",#crossover_f_mode
    "crossover_f": "fde_crossover_f",#crossover_f
    "sub_response": "fde_sub_response",#sub_response
    "hp_rolloff_comp": "fde_hp_rolloff_comp",#hp_rolloff_comp
    "fb_filtering_mode": "fde_fb_filtering",#fb_filtering
    
    "hrtf_polarity_rev": "hrtf_polarity_rev",
    "sofa_exp_conv": "sofa_exp_conv",
    
    # === Headphone & Sample Selections ===
    "brand_list": "fde_hpcf_brand",#hpcf_brand
    "headphone_list": "fde_hpcf_headphone",#hpcf_headphone
    "sample_list": "fde_hpcf_sample",#hpcf_sample

    # === Export Toggles ===
    "fir_hpcf_toggle": "fir_hpcf_toggle",
    "fir_st_hpcf_toggle": "fir_st_hpcf_toggle",
    "geq_hpcf_toggle": "geq_hpcf_toggle",
    "geq_31_hpcf_toggle": "geq_31_hpcf_toggle",
    "hesuvi_hpcf_toggle": "hesuvi_hpcf_toggle",
    "dir_brir_toggle": "dir_brir_toggle",
    "ts_brir_toggle": "ts_brir_toggle",
    "hesuvi_brir_toggle": "hesuvi_brir_toggle",
    "multi_chan_brir_toggle": "multi_chan_brir_toggle",
    "sofa_brir_toggle": "sofa_brir_toggle",

    # === UI & Behavior ===
    "check_updates_start_tag": "check_updates_start_toggle",
    "force_hrtf_symmetry": "force_hrtf_symmetry",
    "er_delay_time_tag": "er_delay_time",
    "qc_toggle_hpcf_history": "qc_toggle_hpcf_history",
    "tab_bar": "tab_bar",
    "auto_apply_hpcf": "qc_auto_apply_hpcf_sel",
    "e_apo_hpcf_conv": "e_apo_hpcf_conv",
    "e_apo_brir_conv": "e_apo_brir_conv",
    "e_apo_prevent_clip": "e_apo_prevent_clip",

    # === Current Selections ===
    "qc_e_apo_curr_hpcf": "qc_e_apo_curr_hpcf",
    "qc_e_apo_sel_hpcf": "qc_e_apo_sel_hpcf",
    "qc_e_apo_curr_brir_set": "qc_e_apo_curr_brir_set",
    "qc_e_apo_sel_brir_set": "qc_e_apo_sel_brir_set",

    # === Headphone & Sample Selections ===
    "qc_brand_list": "qc_hpcf_brand",
    "qc_headphone_list": "qc_hpcf_headphone",
    "qc_sample_list": "qc_hpcf_sample",

    # === BRIR Quick Config ===
    "qc_brir_hp_type": "qc_brir_hp_type",
    "qc_rm_target_list": "qc_room_target",
    "qc_direct_gain": "qc_direct_gain",
    "qc_direct_gain_slider": "qc_direct_gain_slider",
    "qc_acoustic_space_combo": "qc_acoustic_space",
    "qc_brir_hrtf": "qc_brir_hrtf",
    "qc_brir_hrtf_type": "qc_brir_hrtf_type",
    "qc_brir_hrtf_dataset": "qc_brir_hrtf_dataset",
    "qc_crossover_f_mode": "qc_crossover_f_mode",
    "qc_crossover_f": "qc_crossover_f",
    "qc_sub_response": "qc_sub_response",
    "qc_hp_rolloff_comp": "qc_hp_rolloff_comp",
    "qc_fb_filtering": "qc_fb_filtering",

    # === Gain & Mute Controls (Channel Config) ===
    "e_apo_mute_fl": "e_apo_mute_fl",
    "e_apo_mute_fr": "e_apo_mute_fr",
    "e_apo_mute_c": "e_apo_mute_c",
    "e_apo_mute_sl": "e_apo_mute_sl",
    "e_apo_mute_sr": "e_apo_mute_sr",
    "e_apo_mute_rl": "e_apo_mute_rl",
    "e_apo_mute_rr": "e_apo_mute_rr",

    "e_apo_gain_oa": "e_apo_gain_oa",
    "e_apo_gain_fl": "e_apo_gain_fl",
    "e_apo_gain_fr": "e_apo_gain_fr",
    "e_apo_gain_c": "e_apo_gain_c",
    "e_apo_gain_sl": "e_apo_gain_sl",
    "e_apo_gain_sr": "e_apo_gain_sr",
    "e_apo_gain_rl": "e_apo_gain_rl",
    "e_apo_gain_rr": "e_apo_gain_rr",

    "e_apo_elev_angle_fl": "e_apo_elev_angle_fl",
    "e_apo_elev_angle_fr": "e_apo_elev_angle_fr",
    "e_apo_elev_angle_c": "e_apo_elev_angle_c",
    "e_apo_elev_angle_sl": "e_apo_elev_angle_sl",
    "e_apo_elev_angle_sr": "e_apo_elev_angle_sr",
    "e_apo_elev_angle_rl": "e_apo_elev_angle_rl",
    "e_apo_elev_angle_rr": "e_apo_elev_angle_rr",

    "e_apo_az_angle_fl": "e_apo_az_angle_fl",
    "e_apo_az_angle_fr": "e_apo_az_angle_fr",
    "e_apo_az_angle_c": "e_apo_az_angle_c",
    "e_apo_az_angle_sl": "e_apo_az_angle_sl",
    "e_apo_az_angle_sr": "e_apo_az_angle_sr",
    "e_apo_az_angle_rl": "e_apo_az_angle_rl",
    "e_apo_az_angle_rr": "e_apo_az_angle_rr",

    # === Upmix & Delays ===
    "e_apo_upmix_method": "e_apo_upmix_method",
    "e_apo_side_delay": "e_apo_side_delay",
    "e_apo_rear_delay": "e_apo_rear_delay",
    "audio_channels_combo": "e_apo_audio_channels",

    # === HRTF Lists ===
    "hrtf_list_favs": "hrtf_list_favs",
}