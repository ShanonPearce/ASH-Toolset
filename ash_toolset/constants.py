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
import csv
import operator
import os
from platformdirs import user_config_dir
import json
import re

#Few helper functions to load constants

def sanitize_for_dpg(s: str) -> str:
    """
    Cleans a string for DearPyGui display:
    - Replaces smart quotes with ASCII quotes
    - Replaces common dashes and bullets with ASCII equivalents
    - Replaces non-breaking spaces with regular spaces
    - Removes other non-printable characters
    """
    if not s:
        return ""
    
    # Smart quotes → standard quotes
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    
    # Dashes → ASCII dash
    s = s.replace("–", "-").replace("—", "-")
    
    # Bullets → ASCII bullet
    s = s.replace("•", "*").replace("·", "*")
    
    # Non-breaking spaces
    s = s.replace("\u00A0", " ")
    
    # Remove other non-printable or problematic characters (keep ASCII 32-126)
    s = re.sub(r'[^\x20-\x7E]', '', s)
    
    return s.strip()

def load_ac_space_field_names_from_csv(
    metadata_filename,
    base_dir,
    fallback_fields=None,
    logger=None
):
    """
    Load CSV header fields from a metadata CSV file.
    Falls back to provided field list if file does not exist or is invalid.
    """

    csv_path = pjoin(base_dir, metadata_filename)

    try:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)

        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)

        if not header or not isinstance(header, list):
            raise ValueError("Invalid or empty CSV header")

        return [h.strip() for h in header if h.strip()]

    except Exception as e:
        if logger:
            logger(f"Metadata header load failed, using fallback: {e}")
        return fallback_fields or []
    
    


def load_csv_as_dicts(csv_dir, csv_name): 
    """
    Generic CSV loader that detects and converts column data types automatically.
    Columns containing 'name' or 'code' are always loaded as strings.
    
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

                    # Force string if key contains 'name' or 'code'
                    if 'name' in key.lower() or 'code' in key.lower():
                        parsed_row[key] = sanitize_for_dpg(value)
                        continue

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
                            parsed_row[key] = sanitize_for_dpg(value)

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
    global AC_SPACE_LIST_GUI, AC_SPACE_LIST_LABEL, AC_SPACE_LIST_ID
    global AC_SPACE_LIST_MEAS_R60
    global AC_SPACE_LIST_DESCR, AC_SPACE_LIST_DATASET
    global AC_SPACE_LIST_LOWRT60, AC_SPACE_LIST_COLLECTION1, AC_SPACE_LIST_COLLECTION2


    # Load internal metadata
    csv_directory = pjoin(DATA_DIR_INT, 'reverberation')
    internal_data = load_csv_as_dicts(csv_directory, REV_METADATA_FILE_NAME)

    # Load user metadata
    user_csv_dir = DATA_DIR_AS_USER
    user_data = load_user_reverb_csvs_recursive(
        user_csv_dir, filename_key=USER_CSV_KEY, filter_mode="include", match_mode="contains"
    )
    # NEW: only keep user rows that contain the new 'label' key
    user_data = [row for row in user_data if "label" in row]

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
    AC_SPACE_LIST_LABEL = extract_column(reverb_data, 'label')
    AC_SPACE_LIST_ID = extract_column(reverb_data, 'id')
    AC_SPACE_LIST_MEAS_R60 = extract_column(reverb_data, 'meas_rt60')
    AC_SPACE_LIST_DESCR = extract_column(reverb_data, 'description')
    AC_SPACE_LIST_DATASET = extract_column(reverb_data, 'source_dataset')
    AC_SPACE_LIST_LOWRT60 = extract_column(reverb_data, 'name_gui', condition_key='low_rt60', condition_value='Yes', return_all_matches=True)
    AC_SPACE_LIST_COLLECTION1 = extract_column(reverb_data, 'collection_1')
    AC_SPACE_LIST_COLLECTION2 = extract_column(reverb_data, 'collection_2')



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

def get_settings_dir():
    """
    Get the application's configuration directory path in the user's config location.

    This function ensures that the user-specific configuration directory exists,
    then returns the path to that directory.

    Returns:
        str: Full path to the platform-appropriate user config folder.

    Note:
        The config directory location is platform-dependent:
        - Windows: %APPDATA%\\ASH-Toolset
        - macOS: ~/Library/Application Support/ASH-Toolset
        - Linux: ~/.config/ASH-Toolset
    """
    appname = "ASH-Toolset"
    config_dir = user_config_dir(appname)
    os.makedirs(config_dir, exist_ok=True)
    return config_dir

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

    # ---- Read long and short names from CSV ----
    csv_fname = pjoin(DATA_DIR_INT, 'room_target_metadata.csv')
    ROOM_TARGET_LIST = []
    ROOM_TARGET_LIST_SHORT = []

    with open(csv_fname, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ROOM_TARGET_LIST.append(row['long_name'])
            ROOM_TARGET_LIST_SHORT.append(row['short_name'])

    # ---- Build dict from FIRs ----
    ROOM_TARGETS_DICT = {
        name: {
            "short_name": short,
            "impulse_response": room_target_arr[i]
        }
        for i, (name, short) in enumerate(zip(ROOM_TARGET_LIST, ROOM_TARGET_LIST_SHORT))
    }

    # ---- Load and append user targets ----
    ROOM_TARGETS_DICT, ROOM_TARGET_LIST, ROOM_TARGET_LIST_SHORT = load_user_room_targets(
        DATA_DIR_RT_USER,
        ROOM_TARGETS_DICT,
        ROOM_TARGET_LIST,
        ROOM_TARGET_LIST_SHORT
    )

    # ---- Rebuild access helpers ----
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
    SUB_RESPONSE_LIST_LISTENER = extract_column(data=sub_data, column='receiver_type')

def load_azimuth_lookup(csv_path):
    """
    Load azimuth/channel YES/NO table and return a dict of lists.
    Example return structure:
        {
            'fl': [-90, -75, ...],
            'fr': [90, 75, ...],
            ...
        }
    """
    # Initialize output dict
    channels = ["fl", "fr", "c", "sl", "sr", "rl", "rr"]
    result = {ch: [] for ch in channels}

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            az = int(row["azimuth"])
            for ch in channels:
                if row[ch].strip().lower() == "yes":
                    result[ch].append(az)

    return result 
    
def sort_azimuths_by_front_priority(angle_list):
    """
    Sorts azimuth angles so that angles closest to 0° come first.
    Example: [-90, 0, -30] → [0, -30, -90]
    """
    return sorted(angle_list, key=lambda a: abs(a))

def get_version(metadata_file_path):
    try:
        with open(metadata_file_path) as fp:
            return json.load(fp).get("version", "0.0")
    except Exception:
        return "0.0"

def to_circular(angle):
    return (360 - angle) % 360

################################ 
#Constants


########## commonly used for fitlering and transformations
N_FFT = 65536
N_FFT_L = int(65536*2)
N_UNIQUE_PTS = int(np.ceil((N_FFT+1)/2.0))
SAMP_FREQ = 44100#sample rate for ash toolset
FS=SAMP_FREQ#keep for backwards compat
THRESHOLD_CROP = 0.0000005#0.0000005 -120db with reference level of 0.5

TOTAL_SAMPLES_HRIR=256
TOTAL_CHAN_HRIR = 2
TOTAL_CHAN_BRIR = 2
INTERIM_ELEVS = 1
NEAREST_AZ_BRIR_REVERB = 15
OUTPUT_AZIMS_REVERB = int(360/NEAREST_AZ_BRIR_REVERB)

SPECT_SNAP_F0=600#160
SPECT_SNAP_F1=1400#3500
SPECT_SNAP_M_F0=800#1200
SPECT_SNAP_M_F1=2100#1800
IGNORE_MS=0#50,100
#window for reverb shaping: 1=Hanning,2=Bartlett,3=blackman,4=hamming
WINDOW_TYPE=2#1
ALIGNMENT_METHOD = 5

RT60_MAX_S=1250#reference time in ms to start fading out
#control level of direct sound
DIRECT_SCALING_FACTOR = 1.0#reference level - approx 0db DRR. was 0.1

#############  Developer settings
PLOT_ENABLE = False#False
LOG_MEMORY=False #False
LOG_INFO=True     
LOG_GUI=True
HEAD_TRACK_RUNNING = False
PROCESS_BRIRS_RUNNING = False
SHOW_DEV_TOOLS=False#False
STOP_THREAD_FLAG = False
EXPORT_WAVS_DEFAULT=False#False
BULK_AS_IMPORT=False
DEBUG_EXPORT = False  # ---- DEBUG WAV EXPORT HELPER ----        # flip to False to disable all exports


#
################ Low frequency extension and alignment constants
#

#crossover frequency for low frequency BRIR integration
ENABLE_SUB_INTEGRATION = True
ORDER=9#8
F_CROSSOVER = 120#120, default
EVAL_POLARITY=True#True
PEAK_MEAS_MODE=1#0=local max peak, 1 =peak to peak
FILTFILT_TDALIGN = True#apply forward reverse filtering in td alignment method for brir creation
FILTFILT_TDALIGN_AIR = True#apply forward reverse filtering in td alignment method for reverberation generation
FILTFILT_THRESH_F = 45
MIN_FILT_FREQ = 8#values below this will not be allowed in filter generation
#size to hop peak to peak window
DELAY_WIN_HOP_SIZE = 25#10
DELAY_WIN_MIN_T = 0
DELAY_WIN_MAX_T = 1200#
DELAY_WIN_HOPS = int((DELAY_WIN_MAX_T-DELAY_WIN_MIN_T)/DELAY_WIN_HOP_SIZE)

#contants for TD alignment of BRIRs with other BRIRs
T_SHIFT_INTERVAL = 25#50

#AIR alignment
CUTOFF_ALIGNMENT_AIR = 110#110,100
CUTOFF_ALIGNMENT_TR_AIR = CUTOFF_ALIGNMENT_AIR#transformation applied cases, 140, 130, 120
PEAK_TO_PEAK_WINDOW_AIR = int(np.divide(SAMP_FREQ,CUTOFF_ALIGNMENT_AIR)*0.95) 
MIN_T_SHIFT_A = -750#-1000,-1000  
MAX_T_SHIFT_A = 500#250,1000
DELAY_WIN_MIN_A = 0
DELAY_WIN_MAX_A = 1000#1000
DELAY_WIN_HOPS_A = int((DELAY_WIN_MAX_A-DELAY_WIN_MIN_A)/DELAY_WIN_HOP_SIZE)










############# directories for app related data, relative to app directory
#

BASE_DIR_OS= path.abspath(path.dirname(path.dirname(__file__)))#using os.path
SCRIPT_DIR_OS = path.abspath(path.dirname(__file__))#using os.path
BASE_DIR_PATH = Path(__file__).resolve().parents[1] #using Pathlib
SCRIPT_DIR_PATH = Path(__file__).resolve().parent #using Pathlib
SCRIPT_DIRECTORY = path.dirname(path.abspath(sys.argv[0]))#old method using os.path
DEBUG_DIR = pjoin(BASE_DIR_OS, 'tests','debug_wavs')    # relative to cwd
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
DATA_DIR_REVERB = pjoin(BASE_DIR_OS, 'data','interim','reverberation')
DATA_DIR_AS_USER = pjoin(BASE_DIR_OS, 'data','interim','reverberation','user')
DATA_DIR_RT_USER = pjoin(BASE_DIR_OS, 'data','interim','room_targets','user')
DATA_DIR_ASSETS = pjoin(BASE_DIR_OS, 'data','external','assets')
DATA_DIR_RAW = pjoin(BASE_DIR_OS, 'data','raw')
DATA_DIR_RAW_HP_MEASRUEMENTS = pjoin(BASE_DIR_OS, 'data','raw','headphone_measurements')
DATA_DIR_ROOT = pjoin(BASE_DIR_OS, 'data')
DATA_DIR_OUTPUT = pjoin(BASE_DIR_OS, 'data','processed')
DOCS_DIR_GUIDE = pjoin(BASE_DIR_OS, 'docs','user_guide')
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

################# Repository links
USER_GUIDE_URL = "https://raw.githubusercontent.com/ShanonPearce/ASH-Toolset/refs/heads/main/docs/user_guide/user_guide_v4.txt"
USER_GUIDE_NAME = "user_guide_v4.txt"
USER_GUIDE_PATH = pjoin(DOCS_DIR_GUIDE, USER_GUIDE_NAME)
AS_META_URL = "https://raw.githubusercontent.com/ShanonPearce/ASH-Toolset/refs/heads/main/data/interim/reverberation/reverberation_metadata_v3.csv"
MAIN_APP_META_URL = "https://raw.githubusercontent.com/ShanonPearce/ASH-Toolset/main/metadata.json"
TU_FAB_MAX_HRIR_GD_URL = "https://drive.google.com/file/d/1Q14yEBTv2JDu92pPloUQXf_SthSApQ8L/view?usp=drive_link"
TU_KU_MAX_HRIR_GD_URL = "https://drive.google.com/file/d/1vmpLYlH-BjBoFvziTD29WqxZGaoYsFuF/view?usp=drive_link"
TU_FAB_MAX_HRIR_GH_URL = ("https://raw.githubusercontent.com/ShanonPearce/ASH-Toolset/main/data/interim/hrir_dataset_comp_max_TU-FABIAN.npy")
TU_KU_MAX_HRIR_GH_URL = ("https://raw.githubusercontent.com/ShanonPearce/ASH-Toolset/main/data/interim/hrir_dataset_comp_max_THK-KU-100.npy")
TU_KU_MAX_HRIR_URLS = [TU_KU_MAX_HRIR_GH_URL, TU_KU_MAX_HRIR_GD_URL]
TU_FAB_MAX_HRIR_URLS = [TU_FAB_MAX_HRIR_GH_URL,TU_FAB_MAX_HRIR_GD_URL]
AVG_MAX_HRIR_URL = ""
ASH_FILT_DB_META_URL = "https://github.com/ShanonPearce/ASH-Toolset/raw/refs/heads/main/data/processed/hpcf_database_ash_metadata.json"
COMP_FILT_DB_META_URL = "https://github.com/ShanonPearce/ASH-Toolset/raw/refs/heads/main/data/processed/hpcf_database_compilation_metadata.json"
ASH_FILT_DB_URL = "https://github.com/ShanonPearce/ASH-Toolset/raw/refs/heads/main/data/processed/hpcf_database_ash.db"
COMP_FILT_DB_URL = "https://github.com/ShanonPearce/ASH-Toolset/raw/refs/heads/main/data/processed/hpcf_database_compilation.db"
HRTF_META_URL = "https://github.com/ShanonPearce/ASH-Toolset/raw/refs/heads/main/data/interim/hrir/hrir_metadata.csv"
############################ database naming
HPCF_DATABASE_ASH_NAME = 'hpcf_database_ash.db'
HPCF_DATABASE_COMP_NAME = 'hpcf_database_compilation.db'
HPCF_TARGETS_NAME = 'headphone_targets.db'
HPCF_META_ASH_NAME = 'hpcf_database_ash_metadata.json'
HPCF_META_COMP_NAME = 'hpcf_database_compilation_metadata.json'
HPCF_META_LAT_ASH_NAME = 'hpcf_database_ash_metadata_latest.json'
HPCF_META_LAT_COMP_NAME = 'hpcf_database_compilation_metadata_latest.json'
DATABASE_ASH_DIR = pjoin(DATA_DIR_OUTPUT,HPCF_DATABASE_ASH_NAME)
DATABASE_COMP_DIR = pjoin(DATA_DIR_OUTPUT,HPCF_DATABASE_COMP_NAME)
DATABASE_TARGET_DIR = pjoin(DATA_DIR_OUTPUT,HPCF_TARGETS_NAME)

########################    GUI related
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
PROCESS_BUTTON_HPCF='Apply Filter'
BUTTON_IMAGE_ON='on_blue_image'
BUTTON_IMAGE_OFF='off_image'

HP_COMP_LIST = ['In-Ear Headphones - High Strength','In-Ear Headphones - Low Strength','Over/On-Ear Headphones - High Strength','Over/On-Ear Headphones - Low Strength','None']
HP_COMP_LIST_SHORT = ['In-Ear-High','In-Ear-Low','Over+On-Ear-High','Over+On-Ear-Low','Hp-Comp-None']
HRTF_DF_CAL_MODE_LIST = ['Enable Calibration','Retain Diffuse-field','Retain and level spectrum ends']
PLOT_TYPE_LIST = ['Magnitude Response','Impulse Response','Group Delay', 'Decay']
PLOT_TYPE_LIST_IA = ['Magnitude Response','Impulse Response','Group Delay', 'Decay','Summary Response']
HPCF_TARGET_LIST = [
    'ASH Target',"5128 DF","5128 DF (Stock)","5128 DF 10dB","5128 FF","Adjusted Diffuse Field",
    "DF +10dB Tilt","Diffuse Field","Etymotic","Free Field","GRAS KEMAR DF",
    "Harman 2013","Harman 2015","Harman 2017","Harman 2018","Harman 2018 (KB501X)",
    "Harman 2018 (KB501X, No Bass)","Harman 2018 (Linear)","Harman 2018 (No Bass)",
    "Harman 2019","Harman 2019 V2","Harman 2025 MoA Average","Harman Beta 2024",
    "Harman Combined","Harman IE 2019","Harman IE 2019 (5128)","Harman IE 2019 V2",
    "Harman In-Room","Harman In-Room Flat","Harman Target (Generic)","IEC 60318-7 DF",
    "IEF 2025 IEC711","IEF Comp","IEF Neutral","IEF Neutral (Compensated)",
    "IEF Neutral 2023","IEF Neutral 2023 (Compensated)","ISO + Harman","ISO 11904-1 DF",
    "ISO 11904-2 DF","InnerFidelity ID","JM-1 10dB","JM-1 DF","KB501X DF",
    "KEMAR DF (KB006x)","KEMAR DF (KB006x) +10dB Tilt","KEMAR DF (KB500x) +10dB Tilt",
    "KEMAR DF (KB50xx)","KEMAR DF (KB50xx) +10dB Tilt","PEQdB OE","Rtings",
    "Simple DF Downslope","Sonarworks","SoundGuys"
]

#gui
DIRECT_GAIN_MAX=10#8
DIRECT_GAIN_MIN=-10#-8
EAPO_ERROR_CODE=501
ER_RISE_MAX=10#in ms
ER_RISE_MIN=0#
TAB_QC_CODE=1
#TAB_FDE_CODE=2#deprecated/obsolete
TAB_QC_IA_CODE=3

######################################### Room target related
ROOM_TARGETS_DICT = {}
ROOM_TARGET_LIST = []
ROOM_TARGET_LIST_SHORT = []
ROOM_TARGET_KEYS = []
ROOM_TARGET_INDEX_MAP = []

# Automatically refresh on import
refresh_room_targets()


#######################################   BRIR writing
#number of directions to grab to built HESUVI IR
NUM_SOURCE_AZIM_HE = 7
NUM_OUT_CHANNELS_HE = NUM_SOURCE_AZIM_HE*2
NUM_OUT_CHANNELS_TS = 4
NUM_OUT_CHANNELS_MC = 16
PRESET_16CH_LABELS = [
    # Standard home 7.1 front-back layout (PipeWire / ALSA default)
    "FL|FR|FC|LFE|SL|SR|BL|BR - Standard 7.1 (PipeWire compatible)",

    # Back channels before sides (seen in some engines / exports)
    "FL|FR|FC|LFE|BL|BR|SL|SR - 7.1 (Back-before-Side)",

    # Cinema / Dolby-style front-heavy ordering
    "FC|LFE|FL|FR|SL|SR|BL|BR - Cinema / Front-Priority",

    # Emphasise side surrounds before center/LFE
    "FL|FR|SL|SR|FC|LFE|BL|BR - Side-Priority Surround",

    # LFE placed between center and surrounds
    "FL|FR|FC|SL|SR|LFE|BL|BR - Center-First with Late LFE",

    # Non-standard / engine-oriented layout
    "FL|FR|BL|BR|FC|LFE|SL|SR - 7.1 (Back-before-Center)"
]

################ Equalizer APO constants
#
#
AUDIO_CHANNELS = ['2.0 Stereo','2.0 Stereo Upmix to 7.1','5.1 Surround','7.1 Surround','7.1 Downmix to Stereo']
UPMIXING_METHODS = ['A: Channel Duplication','B: Duplication + Separation']#['Method A','Method B']
AUTO_GAIN_METHODS = ['Disabled','Prevent All Clipping','Prevent Low-Frequency Clipping','Prevent Mid-Frequency Clipping'] #['Disabled','Prevent Clipping','Align Low Frequencies','Align Mid Frequencies']
EAPO_MUTE_GAIN=-80.0#custom gain level used to mute but not disable processing

################ Spatial Dataset and Resolution Definitions

AZIMUTH_ANGLES_ALL = [az for az in range(-175, 181, 5)]
CHANNELS_ALL = ["FL", "FR", "C", "SL", "SR", "RL", "RR"]
CHANNELS_HEAD = ["L","R"]
ELEV_ANGLES_WAV_LOW = [30,15,0,-15,-30]
ELEV_ANGLES_WAV_MED = [45,30,15,10,5,0,-5,-10,-15,-30,-45]
ELEV_ANGLES_WAV_HI = [50,45,40,35,30,25,20,15,10,5,0,-5,-10,-15,-20,-25,-30,-35,-40,-45,-50]
ELEV_ANGLES_WAV_ALL = [ELEV_ANGLES_WAV_LOW, ELEV_ANGLES_WAV_MED, ELEV_ANGLES_WAV_HI, []]
ELEV_ANGLES_WAV_GUI = [str(sorted(ELEV_ANGLES_WAV_LOW)), str(sorted(ELEV_ANGLES_WAV_MED)), str(sorted(ELEV_ANGLES_WAV_HI))]

#Reduced set of commonly used azimuth angles for reduced wav dataset export and E-APO configuration. Resolution must not exceed nearest 5 degrees.
#grab from csv
AZIMUTH_LOOKUP_CSV = pjoin(DATA_DIR_RAW, "channel_config_azimuths.csv")
# Load the table
AZ_LOOKUP = load_azimuth_lookup(AZIMUTH_LOOKUP_CSV)
# Now build the angled lists dynamically
AZ_ANGLES_FL_WAV = sort_azimuths_by_front_priority(AZ_LOOKUP["fl"])
AZ_ANGLES_FR_WAV = sort_azimuths_by_front_priority(AZ_LOOKUP["fr"])
AZ_ANGLES_C_WAV  = sort_azimuths_by_front_priority(AZ_LOOKUP["c"])
AZ_ANGLES_SL_WAV = sort_azimuths_by_front_priority(AZ_LOOKUP["sl"])
AZ_ANGLES_SR_WAV = sort_azimuths_by_front_priority(AZ_LOOKUP["sr"])
AZ_ANGLES_RL_WAV = sort_azimuths_by_front_priority(AZ_LOOKUP["rl"])
AZ_ANGLES_RR_WAV = sort_azimuths_by_front_priority(AZ_LOOKUP["rr"])
AZ_ANGLES_ALL_WAV = list(set(AZ_ANGLES_FL_WAV +AZ_ANGLES_FR_WAV +AZ_ANGLES_C_WAV +AZ_ANGLES_SL_WAV +AZ_ANGLES_SR_WAV +AZ_ANGLES_RL_WAV +AZ_ANGLES_RR_WAV))
AZ_ANGLES_ALL_WAV_SORTED = sorted(AZ_ANGLES_ALL_WAV)
AZ_ANGLES_ALL_WAV_CIRC = sorted({to_circular(a) for a in AZ_ANGLES_ALL_WAV})#convert to circular (CCW) numbering
AZIM_EXTRA_RANGE_CIRC = {25,30,35,335,330,325}#set of commonly used azimuths to always include in reduced export, in circular numbering
AZ_ANGLES_ALL_WAV_GUI = [str(sorted(AZ_ANGLES_ALL_WAV)), str(sorted(AZ_ANGLES_ALL_WAV)), '-180 to 180 degrees in 5 degree steps.']
#
#constants for writing WAVs
#below defines each level of spatial resolution. Each level will determine how many directions will be processed and saved to file. Lower levels are more suitable for WAV exports.
SPATIAL_RES_LIST = ['Low','Medium','High']#0=low,1=med,2=high,3=max v3.6 was ['Low','Medium','High','Max']
SPATIAL_RES_LIST_LIM = ['Low','Medium','High']
SPATIAL_RES_LIST_E_APO = ['Low','Medium']
SPATIAL_RES_ELEV_DESC = ['-30 to 30 degrees in 15 degree steps.','-45 to 45 degrees in 15 degree steps.','-50 to 50 degrees (WAV export) or -60 to 60 degrees (SOFA export) in 5 degree steps.','-40 to 40 degrees (WAV export) or -40 to 60 degrees (SOFA export) in 2 degree steps.']
SPATIAL_RES_AZIM_DESC = ['-180 to 180 or 0 to 360 degrees in varying steps.','-180 to 180 or 0 to 360 degrees in varying steps.','-180 to 180 or 0 to 360 degrees in 5 degree steps.','-180 to 180 or 0 to 360 degrees in 2 degree steps.']
SPATIAL_RES_ELEV_RNG = ['-30° to 30°','-45° to 45°','-50° to 50° (WAV) or -60° to 60° (SOFA)','-40° to 40° (WAV) or -40° to 60° (SOFA)']
SPATIAL_RES_AZIM_RNG = ['-180° to 180° (WAV) or 0° to 360° (SOFA)','-180° to 180° (WAV) or 0° to 360° (SOFA)','-180° to 180° (WAV) or 0° to 360° (SOFA)','-180° to 180° (WAV) or 0° to 360° (SOFA)']
SPATIAL_RES_ELEV_STP = ['15° steps','varying','5° steps','2° steps']
SPATIAL_RES_AZIM_STP = ['varying','varying','5° steps','2° steps']
NUM_SPATIAL_RES = len(SPATIAL_RES_LIST)
#below defines structure of internal spatial datasets at each spatial resolution
SPATIAL_RES_AZIM_MIN_IN=[0, 0, 0, 0 ]#lower limits (inclusive)
SPATIAL_RES_AZIM_MAX_IN=[355, 355, 355, 358 ]#upper limits (inclusive)
SPATIAL_RES_ELEV_MIN_IN=[-60, -60, -60, -40 ]#lower limits (inclusive)
SPATIAL_RES_ELEV_MAX_IN=[60, 60, 60, 60 ]#upper limits (inclusive)
SPATIAL_RES_ELEV_NEAREST_IN=[5, 5, 5, 2]#grid size (elevation)
SPATIAL_RES_AZIM_NEAREST_IN=[5, 5, 5, 2]#grid size (azimuth)
#below defines processing grid size, must not exceed internal grid size
SPATIAL_RES_ELEV_NEAREST_PR=[15, 5, 5, 2]#was [15, 15, 5, 2]
SPATIAL_RES_AZIM_NEAREST_PR=[5, 5, 5, 2]
#below defines resolution of WAV output datasets, must not exceed input limits and processing grid size
SPATIAL_RES_ELEV_MIN_WAV_OUT=[-30, -45, -50, -40 ]
SPATIAL_RES_ELEV_MAX_WAV_OUT=[30, 45, 50, 40 ]
SPATIAL_RES_ELEV_NEAREST_WAV_OUT=[15, 15, 5, 2]
SPATIAL_RES_AZIM_NEAREST_WAV_OUT=[15, 15, 5, 2]

HRTF_DIRECTION_FIX_LIST = [None,
                           "flip_azimuth",
                           "front_back",
                           "back_cw_start",
                           "left_start",
                           "right_start",
                           "invert_elevation" ,
                           ('offset', 5),
                           ('offset', -5),
                           ('offset', 10),
                           ('offset', -10), 
                           ('elevation_offset', 5),
                           ('elevation_offset', -5),
                           ('elevation_offset', 10),
                           ('elevation_offset', -10),
                           ('offsets', 5, 5),
                           ('offsets', 5, -5),
                           ('offsets', 5, 10),
                           ('offsets', 5, -10),
                           ('offsets', -5, 5),
                           ('offsets', -5, -5), 
                           ('offsets', -5, 10),
                           ('offsets', -5, -10)]
HRTF_DIRECTION_FIX_LIST_GUI = ["None",
                               "Reverse azimuths (CW <-> CCW)",
                               "Reference (0°) starts at back (180° shift)",
                               "Reference (0°) starts at back and reversed",
                               "Reference (0°) starts at left (90° shift CW)",
                               "Reference (0°) starts at right (90° shift CCW)",
                               "Flip elevation sign" ,
                               "Azimuth offset - shift source 5° CW",
                               "Azimuth offset - shift source 5° CCW",
                               "Azimuth offset - shift source 10° CW",
                               "Azimuth offset - shift source 10° CCW",
                               "Elevation offset - shift source 5° up",
                               "Elevation offset - shift source 5° down",
                               "Elevation offset - shift source 10° up",
                               "Elevation offset - shift source 10° down",
                               "Offsets - shift source 5° CW and 5° up",
                               "Offsets - shift source 5° CW and 5° down",
                               "Offsets - shift source 5° CW and 10° up",
                               "Offsets - shift source 5° CW and 10° down",
                               "Offsets - shift source 5° CCW and 5° up",
                               "Offsets - shift source 5° CCW and 5° down",
                               "Offsets - shift source 5° CCW and 10° up",
                               "Offsets - shift source 5° CCW and 10° down"]
HRTF_DIRECTION_FIX_LIST_DESC = ["No change to orientation",
                                "Reverses azimuth rotation direction (CW <-> CCW) without changing the front/back reference.",
                                "Rotates the coordinate system by 180°, swapping front and back.",
                                "Rotates the coordinate system by 180° and reverses azimuths.",
                                "Rotates the azimuth grid 90° clockwise.",
                                "Rotates the azimuth grid 90° counter-clockwise.",
                                "Flips elevation sign (positive <-> negative), inverting above/below.",
                                "Rotates the azimuth grid 5° clockwise.",
                                "Rotates the azimuth grid 5° counter-clockwise.",
                                "Rotates the azimuth grid 10° clockwise.",
                                "Rotates the azimuth grid 10° counter-clockwise.",
                                "Applies a +5° elevation offset (shifts sources upward).",
                                "Applies a -5° elevation offset (shifts sources downward).",
                                "Applies a +10° elevation offset (shifts sources upward).",
                                "Applies a -10° elevation offset (shifts sources downward).",
                                "Rotates the azimuth grid 5° clockwise & Applies a +5° elevation offset.",
                                "Rotates the azimuth grid 5° clockwise & Applies a -5° elevation offset.",
                                "Rotates the azimuth grid 5° clockwise & Applies a +10° elevation offset.",
                                "Rotates the azimuth grid 5° clockwise & Applies a -10° elevation offset.",
                                "Rotates the azimuth grid 5° counter-clockwise & Applies a +5° elevation offset.",
                                "Rotates the azimuth grid 5° counter-clockwise & Applies a -5° elevation offset.",
                                "Rotates the azimuth grid 5° counter-clockwise & Applies a +10° elevation offset.",
                                "Rotates the azimuth grid 5° counter-clockwise & Applies a -10° elevation offset."]



################ hpcf related
#
NUM_ITER = 8 #was 4
PARAMS_PER_ITER = 4 #was 8
SENSITIVITY = 0.003 #0.003
FREQ_CUTOFF = 20000#f in bins
FREQ_CUTOFF_GEQ = 29000
EAPO_GAIN_ADJUST = 0.9*1.1*1.1*1.1
EAPO_QF_ADJUST = 0.5*0.8
HPCF_FIR_LENGTH = 512#
HPCF_SAMPLE_DEFAULT = 'Sample A'
HPCF_DATABASE_LIST = ['ASH Filters', 'Compilation']
# mapping: new_schema_col -> old_schema_col
HPCF_DB_SCHEMA_MAP = {
    "type": "brand",
    "headphone_name": "headphone",
    "source": "sample",
    "mag_db": "mag_db"
}#"source": "brand","headphone_name": "headphone","type": "sample",
# columns always expected by your app
HPCF_DB_STANDARD_COLUMNS = [
    "brand", "headphone", "sample", "sample_id",
    "mag_db", "created_on", "type"
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




############### Export related constants
#
SAMPLE_RATE_LIST = ['44.1 kHz', '48 kHz', '96 kHz']
SAMPLE_RATE_DICT = {'44.1 kHz': 44100, '48 kHz': 48000, '96 kHz': 96000}  
BIT_DEPTH_LIST = ['24 bit', '32 bit']
BIT_DEPTH_DICT = {'24 bit': 'PCM_24', '32 bit': 'PCM_32'}  #{'24 bit': 'PCM_24', '32 bit': 'PCM_32'}  
HRTF_SYM_LIST = ['Disabled', 'Mirror Left Side', 'Mirror Right Side']
RESAMPLE_MODE_LIST = ['Performance', 'Quality']

#AIR and BRIR reverberation processing
LIMIT_REBERB_DIRS=True
MIN_REVERB_DIRS=1
MAX_IRS=5000#2520
IR_MIN_THRESHOLD=2000#below this number of IRs, transform will be applied to expand dataset
IR_MIN_THRESHOLD_DUPLICATE=400#below this number of IRs, IRs will be duplicated. Only used in low frequency mode
IR_MIN_THRESHOLD_FULLSET=1800#min number of IRs to create 7 brir sources, otherwise defaults to 5
DIRECTION_MODE=3#0=one set of azimuths for each source, 1 = separate set of azimuths for left and right hem,2 = random using triangle distribution, 3 = biased_spherical_coordinate_sampler
MAG_COMP=True#enables DF compensation in AIR processing   True
RISE_WINDOW=True#enables windowing of initial rise in AIR processing

# Reverberation tail crop threshold (dB)
REVERB_CROP_DB_MIN = -120.0
REVERB_CROP_DB_MAX = -60.0


############### Acoustic space related
#
AC_SPACE_CHAN_LIMITED=6#3,4,5
AC_SPACE_LIST_SORT_BY = ['Name','ID','Reverberation Time']
AC_SPACE_LIST_COLLECTIONS_BASE = ['All','User Imports','Favourites']
AS_BASE_LIST_FAV = ['No favourites found']
AS_TAIL_MODE_LIST = ['Short','Short Windowed','Long','Long Windowed']
AS_LISTENER_TYPE_LIST = ['FABIAN HATS','KU-100 Dummy Head','User Selection']
#load other lists from csv file

# Global variable declarations for clarity
reverb_data = []
AC_SPACE_LIST_GUI = []
AC_SPACE_LIST_LABEL = []
AC_SPACE_LIST_ID = []
AC_SPACE_LIST_MEAS_R60 = []
AC_SPACE_LIST_DESCR = []
AC_SPACE_LIST_DATASET = []
AC_SPACE_LIST_LOWRT60 = []
AC_SPACE_LIST_COLLECTION1 = []
AC_SPACE_LIST_COLLECTION2 = []


METADATA_CSV_KEY = "metadata"
USER_CSV_KEY = "_metadata"
ASI_USER_CSV_KEY = "_asi-metadata"
SUB_USER_CSV_KEY = "_sub-metadata"
REV_METADATA_FILE_NAME='reverberation_metadata_v3.csv'#was 'reverberation_metadata.csv' prior to 3.6
AS_USER_PREFIX="dataset_"
#section to load dict lists containing acoustic space metadata
#new: load as dict lists
refresh_acoustic_space_metadata()
# Dynamically loaded at startup
AC_SPACE_FIELD_NAMES = load_ac_space_field_names_from_csv(metadata_filename=REV_METADATA_FILE_NAME, base_dir=DATA_DIR_REVERB)
#get collections list
AC_SPACE_LIST_COLLECTIONS = sorted({x for x in AC_SPACE_LIST_COLLECTIONS_BASE + AC_SPACE_LIST_COLLECTION1 + AC_SPACE_LIST_COLLECTION2 if x}, key=str.lower)

############     SOFA related
#

#new: load as dict lists
csv_directory = DATA_DIR_ROOT
sofa_data = load_csv_as_dicts(csv_directory,'supported_conventions.csv')

SOFA_COMPAT_CONV = extract_column(data=sofa_data, column='Convention')  
SOFA_COMPAT_VERS = extract_column(data=sofa_data, column='Version') 
SOFA_COMPAT_CONVERS = extract_column(data=sofa_data, column='SOFAConventionsVersion')                       # strings
SOFA_OUTPUT_CONV = extract_column(data=sofa_data, column='Convention', condition_key='OutputFormat', condition_value='Yes', return_all_matches=True)
SOFA_OUTPUT_VERS = extract_column(data=sofa_data, column='Version', condition_key='OutputFormat', condition_value='Yes', return_all_matches=True)
SOFA_OUTPUT_CONVERS = extract_column(data=sofa_data, column='SOFAConventionsVersion', condition_key='OutputFormat', condition_value='Yes', return_all_matches=True)


############    Data summary table
#

csv_directory = DATA_DIR_ROOT
data_summary = load_csv_as_dicts(csv_directory,'data_summary.csv')

DATA_SUMMARY_TYPE = extract_column(data=data_summary, column='Type')  
DATA_SUMMARY_STAGE = extract_column(data=data_summary, column='Stage') 
DATA_SUMMARY_FORMAT = extract_column(data=data_summary, column='File format')
DATA_SUMMARY_STRUCT = extract_column(data=data_summary, column='Data Structure')
DATA_SUMMARY_COORD = extract_column(data=data_summary, column='Coordinate System')
DATA_SUMMARY_NOTES = extract_column(data=data_summary, column='Notes')

############### HRTF Dataset Related
#
#HRTF related - individual datasets
CTF_ADJUST_DIRS_THRESHOLD = 16 #number of directions in sofa dataset below which adjustment will be carried out
HRTF_A_GAIN_ADDIT = 2.5#
#Strings
HRTF_TYPE_LIST = ['Dummy Head / Head & Torso Simulator', 'Human Listener', 'User SOFA Input', 'Favourites']
HRTF_BASE_LIST_FAV = ['No favourites found']
#new: load as dict lists
csv_directory = DATA_DIR_HRIR_NPY
HRIR_METADATA_NAME='hrir_metadata.csv'
HRTF_DATA = load_csv_as_dicts(csv_directory,HRIR_METADATA_NAME)
HRTF_DATASET_LIST_INDV = extract_column(data=HRTF_DATA, column= 'dataset', condition_key='hrtf_type', condition_value='Human Listener', return_all_matches=True)
HRTF_DATASET_LIST_DUMMY = extract_column(data=HRTF_DATA, column='dataset', condition_key='hrtf_type', condition_value='Dummy Head / Head & Torso Simulator', return_all_matches=True)
HRTF_TYPE_DEFAULT = extract_column(data=HRTF_DATA, column='hrtf_type', condition_key='default', condition_value='Yes', return_all_matches=False)
HRTF_DATASET_DEFAULT = extract_column(data=HRTF_DATA, column='dataset', condition_key='default', condition_value='Yes', return_all_matches=False)
HRTF_LISTENER_DEFAULT = extract_column(data=HRTF_DATA, column='name_gui', condition_key='default', condition_value='Yes', return_all_matches=False)

#lists for attribution table
HRTF_META_TYPES = extract_column(data=HRTF_DATA, column='hrtf_type')
HRTF_META_DATASETS = extract_column(data=HRTF_DATA, column='dataset')
HRTF_META_ATTR = extract_column(data=HRTF_DATA, column='attribution')
# Step 1: Combine the lists into tuples per row
rows = list(zip(HRTF_META_TYPES, HRTF_META_DATASETS, HRTF_META_ATTR))
# Step 2: Keep only unique combinations of (dataset, attribution)
seen = set()
unique_rows = []
for row in rows:
    key = (row[1], row[2])  # dataset + attribution
    if key not in seen:
        seen.add(key)
        unique_rows.append(row)
# Step 3: Unpack the unique rows back into separate lists
HRTF_UNIQ_TYPES, HRTF_UNIQ_DATASETS, HRTF_UNIQ_ATTR = zip(*unique_rows)
# Convert to lists (optional, in case you want lists instead of tuples)
HRTF_UNIQ_TYPES = list(HRTF_UNIQ_TYPES)
HRTF_UNIQ_DATASETS = list(HRTF_UNIQ_DATASETS)
HRTF_UNIQ_ATTR = list(HRTF_UNIQ_ATTR)


HRTF_DATASET_LIST_USER = ['User Sourced']
HRTF_DATASET_LIST_FAVOURITES = ['User Favourites']
HRTF_DATASET_USER_DEFAULT = HRTF_DATASET_LIST_USER[0]
HRTF_DATASET_FAVOURITES_DEFAULT = HRTF_DATASET_LIST_FAVOURITES[0]
# Remove duplicates by converting to a set and back to a list
HRTF_DATASET_LIST_INDV = list(set(HRTF_DATASET_LIST_INDV))
# Sort the list alphabetically
HRTF_DATASET_LIST_INDV.sort()
# Remove duplicates by converting to a set and back to a list
HRTF_DATASET_LIST_DUMMY = list(set(HRTF_DATASET_LIST_DUMMY))
# Sort the list alphabetically
HRTF_DATASET_LIST_DUMMY.sort()
#create a dictionary
HRTF_TYPE_DATASET_DICT = {
    HRTF_TYPE_LIST[0]: HRTF_DATASET_LIST_DUMMY,
    HRTF_TYPE_LIST[1]: HRTF_DATASET_LIST_INDV,
    HRTF_TYPE_LIST[2]: HRTF_DATASET_LIST_USER,
    HRTF_TYPE_LIST[3]: HRTF_DATASET_LIST_FAVOURITES
}
HRTF_POLARITY_LIST = ['Auto Select','Original','Reversed']

HRTF_DATASET_LIST_DEFAULT = HRTF_TYPE_DATASET_DICT.get(HRTF_TYPE_DEFAULT)
HRTF_LISTENER_LIST_DEFAULT = extract_column(data=HRTF_DATA, column='name_gui', condition_key='dataset', condition_value=HRTF_DATASET_DEFAULT, return_all_matches=True)
HRTF_AVERAGED_NAME_GUI = "Averaged HRTF"
HRTF_AVERAGED_NAME_FILE = "hrir_favourites_averaged"
HRTF_USER_SOFA_DEFAULT = 'No SOFA files found'
HRTF_USER_SOFA_PREFIX = 'u-'


###############      SUB Related
#

SUB_FC_SETTING_LIST = ['Auto Select', 'Custom Value']
SUB_FC_MIN=0
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
# Time-domain unit impulse (for full FFTs or testing)
IMPULSE = np.eye(1, N_FFT, 0, dtype=np.float32)[0]
# Flat spectrum in dB — FULL FFT domain (length = N_FFT)
FR_FLAT_DB_FFT   = np.zeros(N_FFT, dtype=np.float32)
# Flat spectrum in dB — REAL FFT domain (length = N_FFT//2 + 1)
FR_FLAT_DB_RFFT  = np.zeros(N_FFT // 2 + 1, dtype=np.float32)

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
    "path": "",#"C:\\Program Files\\EqualizerAPO\\config"

    # === GUI ===
    "tab_bar": 0,
    
    # === Export tab HPCF ===
    "fde_fir_hpcf_toggle": True,
    "fde_fir_st_hpcf_toggle": False,
    "fde_hesuvi_hpcf_toggle": False,
    "fde_geq_hpcf_toggle": False,
    "fde_geq_31_hpcf_toggle": False,

    # === Export tab HRTF / BRIR ===
    "fde_brir_spat_res": SPATIAL_RES_LIST[0],#brir_spat_res, low as default
    
    "fde_dir_brir_toggle": True,
    "fde_ts_brir_toggle": False,
    "fde_sofa_brir_toggle": False,
    "fde_hesuvi_brir_toggle": False,
    "fde_multi_chan_brir_toggle": False,
  
    #default lists - used for listboxes
    "hrtf_list_favs": HRTF_BASE_LIST_FAV,
    "brir_hrtf_dataset_list": HRTF_DATASET_LIST_DEFAULT,
    "brir_hrtf_listener_list": HRTF_LISTENER_LIST_DEFAULT,
    "as_list_favs": AS_BASE_LIST_FAV,


    # === QC Tab HPCF ===
    "toggle_hpcf_history": False,
    "hpcf_brand": "",#not known until db loaded
    "hpcf_headphone": "",#not known until db loaded
    "hpcf_sample": "",#not known until db loaded
    "e_apo_curr_hpcf": "",#nothing generated to start with
    "e_apo_sel_hpcf": "",#nothing generated to start with
    "e_apo_auto_apply_hpcf_sel": False,
    "e_apo_hpcf_conv": False,
    "hpcf_active_database": HPCF_DATABASE_LIST[0],
    
    # === QC Tab BRIR ===
    "brir_hp_type": HP_COMP_LIST[2],
    "room_target": ROOM_TARGET_LIST[1],
    "direct_gain": 3.0,
    "direct_gain_slider": 3.0,
    "acoustic_space": AC_SPACE_LIST_GUI[0],
    "brir_hrtf_type": HRTF_TYPE_DEFAULT,
    "brir_hrtf_dataset": HRTF_DATASET_DEFAULT,
    "brir_hrtf": HRTF_LISTENER_DEFAULT,
    "sub_response": SUB_RESPONSE_LIST_GUI[12] if len(SUB_RESPONSE_LIST_GUI) > 12 else SUB_RESPONSE_LIST_GUI[0],
    "crossover_f_mode": SUB_FC_SETTING_LIST[0],
    "crossover_f": SUB_FC_DEFAULT,
    "hp_rolloff_comp": False,
    "fb_filtering": False,
    'hrtf_direction_misalign_comp': HRTF_DIRECTION_FIX_LIST_GUI[0],
    
    "wav_sample_rate": SAMPLE_RATE_LIST[0],
    "wav_bit_depth": BIT_DEPTH_LIST[0],
    
    "e_apo_curr_brir_set": "",#nothing generated to start with
    "e_apo_sel_brir_set": "",#nothing generated to start with
    "e_apo_sel_brir_set_ts": "",#nothing generated to start with
    "e_apo_brir_conv": False,
    
    
    # === E-APO General ===
    "e_apo_brir_spat_res": SPATIAL_RES_LIST_E_APO[0],#brir_spat_res, low as default
    "e_apo_audio_channels": AUDIO_CHANNELS[0],
    "e_apo_upmix_method": UPMIXING_METHODS[1],
    "e_apo_side_delay": 0,
    "e_apo_rear_delay": 10,
    "e_apo_mute": False,
    "e_apo_gain": 0.0,
    "e_apo_elev_angle": 0,
    "e_apo_autoapply_hpcf": False,
    "e_apo_prevent_clip": AUTO_GAIN_METHODS[1],
    # === E-APO Per-Channel Config ===
    "e_apo_az_angle_fl": -30,
    "e_apo_az_angle_fr": 30,
    "e_apo_az_angle_c": 0,
    "e_apo_az_angle_sl": -90,
    "e_apo_az_angle_sr": 90,
    "e_apo_az_angle_rl": -135,
    "e_apo_az_angle_rr": 135,
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
    # === export Per-Channel Angles ===
    "fde_elev_angle_fl": 0,
    "fde_elev_angle_fr": 0,
    "fde_elev_angle_c": 0,
    "fde_elev_angle_sl": 0,
    "fde_elev_angle_sr": 0,
    "fde_elev_angle_rl": 0,
    "fde_elev_angle_rr": 0,
    "fde_az_angle_fl": -30,
    "fde_az_angle_fr": 30,
    "fde_az_angle_c": 0,
    "fde_az_angle_sl": -90,
    "fde_az_angle_sr": 90,
    "fde_az_angle_rl": -135,
    "fde_az_angle_rr": 135,
    "fde_gain_oa": 0.0,
    "fde_gain_fl": 0.0,
    "fde_gain_fr": 0.0,
    "fde_gain_c": 0.0,
    "fde_gain_sl": 0.0,
    "fde_gain_sr": 0.0,
    "fde_gain_rl": 0.0,
    "fde_gain_rr": 0.0,
    "mapping_16ch_wav": PRESET_16CH_LABELS[0],
  
    # === Misc ===
    "sofa_exp_convention": SOFA_OUTPUT_CONV[1],
    "check_updates_start_toggle": False,
    "hrtf_polarity_rev": HRTF_POLARITY_LIST[0],
    "force_hrtf_symmetry":HRTF_SYM_LIST[0],
    "er_delay_time":0.0,
    "export_resample_mode": RESAMPLE_MODE_LIST[0],
    "hrtf_df_cal_mode": HRTF_DF_CAL_MODE_LIST[0],
    "hrtf_low_freq_suppression": True,
    "hpcf_fir_length":HPCF_FIR_LENGTH,
    "hpcf_target_curve": HPCF_TARGET_LIST[0],
    "hpcf_smooth_hf": False,
    "reverb_tail_crop_db": -90.0
    

}






#mapping of ASH V3.5 settings keys to new keys (also new default keys)
#It’s structured as:"old_key": "new_key",
LEGACY_KEY_MAP = {
    # === BRIR Quick Config 3.7.0 ===
    "qc_brir_hp_type": "brir_hp_type",
    "qc_room_target": "room_target",
    "qc_direct_gain": "direct_gain",
    "qc_direct_gain_slider": "direct_gain_slider",#not originally saved
    "qc_fb_filtering": "fb_filtering",
    # === Current Selections 3.7.0 === 
    "qc_e_apo_curr_hpcf": "e_apo_curr_hpcf",
    "qc_e_apo_sel_hpcf": "e_apo_sel_hpcf",
    "qc_e_apo_curr_brir_set": "e_apo_curr_brir_set",
    "qc_e_apo_sel_brir_set": "e_apo_sel_brir_set",
    # === UI & Behavior  3.7.0 ===
    "qc_toggle_hpcf_history": "toggle_hpcf_history",
    "qc_auto_apply_hpcf_sel": "e_apo_auto_apply_hpcf_sel",
    
    # === General Paths & System ===
    "path": "path",
    "qc_wav_sample_rate": "wav_sample_rate",
    "qc_wav_bit_depth": "wav_bit_depth",

    # === BRIR / Room Simulation Parameters ===
    "spatial_resolution": "fde_brir_spat_res",#brir_spat_res

    "hrtf_polarity": "hrtf_polarity_rev",

    # === Export Toggles ===
    "fir_hpcf_exp": "fde_fir_hpcf_toggle",
    "fir_st_hpcf_exp": "fde_fir_st_hpcf_toggle",
    "geq_hpcf_exp": "fde_geq_hpcf_toggle",
    "geq_31_exp": "fde_geq_31_hpcf_toggle",
    "hesuvi_hpcf_exp": "fde_hesuvi_hpcf_toggle",
    "dir_brir_exp": "fde_dir_brir_toggle",
    "ts_brir_exp": "fde_ts_brir_toggle",
    "hesuvi_brir_exp": "fde_hesuvi_brir_toggle",
    "multi_chan_brir_exp": "fde_multi_chan_brir_toggle",
    "sofa_brir_exp": "fde_sofa_brir_toggle",

    # === UI & Behavior ===
    "auto_check_updates": "check_updates_start_toggle",
    "force_hrtf_symmetry": "force_hrtf_symmetry",
    "er_delay_time": "er_delay_time",
    "show_hpcf_history": "toggle_hpcf_history",
    "tab_selected": "tab_bar",
    "auto_apply_hpcf": "e_apo_auto_apply_hpcf_sel",
    "enable_hpcf": "e_apo_hpcf_conv",
    "enable_brir": "e_apo_brir_conv",
    "prevent_clip": "e_apo_prevent_clip",
    

    # === Current Selections ===
    "hpcf_current": "e_apo_curr_hpcf",
    "hpcf_selected": "e_apo_sel_hpcf",
    "brir_set_current": "e_apo_curr_brir_set",
    "brir_set_selected": "e_apo_sel_brir_set",
    

    # === Headphone & Sample Selections ===
    "qc_brand": "hpcf_brand",
    "qc_headphone": "hpcf_headphone",
    "qc_sample": "hpcf_sample",
    "qc_hpcf_brand": "hpcf_brand",
    "qc_hpcf_headphone": "hpcf_headphone",
    "qc_hpcf_sample": "hpcf_sample",

    # === BRIR Quick Config ===
    "qc_brir_headphone_type": "brir_hp_type",
    "qc_brir_room_target": "room_target",
    "qc_brir_direct_gain": "direct_gain",
    "qc_brir_direct_gain_slider": "direct_gain_slider",#not originally saved
    "qc_acoustic_space": "acoustic_space",
    "qc_brir_hrtf": "brir_hrtf",
    "qc_brir_hrtf_type": "brir_hrtf_type",
    "qc_brir_hrtf_dataset": "brir_hrtf_dataset",
    "qc_crossover_f_mode": "crossover_f_mode",
    "qc_crossover_f": "crossover_f",
    "qc_sub_response": "sub_response",
    "qc_hp_rolloff_comp": "hp_rolloff_comp",
    "qc_fb_filtering_mode": "fb_filtering",
    

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
