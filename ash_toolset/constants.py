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

def load_user_reverb_csvs_recursive(directory, filename_key):
    """
    Recursively searches a directory for CSV files with names containing the filename_key,
    loads them as dicts, and returns a list of parsed rows.
    
    Parameters:
        directory (str): Base directory to search in.
        filename_key (str): Substring that must appear in CSV file names.

    Returns:
        List[dict]: Parsed rows from all matching CSVs.
    """
    found_rows = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv") and filename_key in file:
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
    global reverb_data
    global AC_SPACE_LIST_GUI, AC_SPACE_LIST_SHORT, AC_SPACE_LIST_SRC
    global AC_SPACE_EST_R60, AC_SPACE_MEAS_R60, AC_SPACE_FADE_START
    global AC_SPACE_GAINS, AC_SPACE_DESCR, AC_SPACE_DATASET
    global AC_SPACE_LIST_NR, AC_SPACE_LIST_LOWRT60, AC_SPACE_LIST_HIRT60
    global AC_SPACE_LIST_COMP, AC_SPACE_LIST_MAX_SRC, AC_SPACE_LIST_ID_SRC

    # Load internal metadata
    csv_directory = pjoin(DATA_DIR_INT, 'reverberation')
    internal_data = load_csv_as_dicts(csv_directory, 'reverberation_metadata.csv')

    # Load user metadata
    user_csv_dir = DATA_DIR_AS_USER
    #USER_CSV_KEY = "_metadata"#already global
    user_data = load_user_reverb_csvs_recursive(user_csv_dir, USER_CSV_KEY)

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


################################ 
#Constants

#commonly used
N_FFT = 65536
N_FFT_L = int(65536*2)
N_UNIQUE_PTS = int(np.ceil((N_FFT+1)/2.0))
SAMP_FREQ = 44100#sample rate for ash toolset
FS=SAMP_FREQ
THRESHOLD_CROP = 0.0000005#0.0000005 -120db with reference level of 0.5


#Developer settings
PLOT_ENABLE = False#False
LOG_INFO=True     
LOG_GUI=True
HEAD_TRACK_RUNNING = False
PROCESS_BRIRS_RUNNING = False
SHOW_DEV_TOOLS=False#False
STOP_THREAD_FLAG = False
SHOW_AS_TAB = True

#control level of direct sound
DIRECT_SCALING_FACTOR = 1.0#reference level - approx 0db DRR. was 0.1

#crossover frequency for low frequency BRIR integration
ENABLE_SUB_INTEGRATION = True
ORDER=9#8
F_CROSSOVER = 120#120, default
EVAL_POLARITY=True#True
PEAK_MEAS_MODE=1#0=local max peak, 1 =peak to peak

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
MIN_T_SHIFT_N = -230#-220
MAX_T_SHIFT_N = 50#100
NUM_INTERVALS_N = int(np.abs((MAX_T_SHIFT_N-MIN_T_SHIFT_N)/T_SHIFT_INTERVAL_N))
#sub alignment with reverb - pos pol
T_SHIFT_INTERVAL_P = 10#25
MIN_T_SHIFT_P = -230#-220
MAX_T_SHIFT_P = 50#100
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
CUTOFF_ALIGNMENT_AIR = 110#110
CUTOFF_ALIGNMENT_TR_AIR = 110#transformation applied cases, 140, 130, 120
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

#window for reverb shaping: 1=Hanning,2=Bartlett,3=blackman,4=hamming
WINDOW_TYPE=2#1
ALIGNMENT_METHOD = 5

#directories
BASE_DIR_OS= path.abspath(path.dirname(path.dirname(__file__)))#using os.path
SCRIPT_DIR_OS = path.abspath(path.dirname(__file__))#using os.path
BASE_DIR_PATH = Path(__file__).resolve().parents[1] #using Pathlib
SCRIPT_DIR_PATH = Path(__file__).resolve().parent #using Pathlib
SCRIPT_DIRECTORY = path.dirname(path.abspath(sys.argv[0]))#old method using os.path
DATA_DIR_INT = pjoin(BASE_DIR_OS, 'data','interim')
DATA_DIR_SUB = pjoin(BASE_DIR_OS, 'data','interim','sub')
DATA_DIR_HRIR_NPY = pjoin(BASE_DIR_OS, 'data','interim','hrir')
DATA_DIR_HRIR_NPY_DH = pjoin(BASE_DIR_OS, 'data','interim','hrir','dh')
DATA_DIR_HRIR_NPY_DH_V = pjoin(BASE_DIR_OS, 'data','interim','hrir','dh','h','VIKING')
DATA_DIR_HRIR_NPY_HL = pjoin(BASE_DIR_OS, 'data','interim','hrir','hl')
DATA_DIR_HRIR_NPY_USER = pjoin(BASE_DIR_OS, 'data','interim','hrir','user')
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

#HP_COMP_LIST = ['In-Ear Headphones - High Strength','In-Ear Headphones - Low Strength','Over-Ear Headphones - High Strength','Over-Ear Headphones - Low Strength','On-Ear Headphones - High Strength','On-Ear Headphones - Low Strength']
#HP_COMP_LIST_SHORT = ['In-Ear-High','In-Ear-Low','Over-Ear-High','Over-Ear-Low','On-Ear-High','On-Ear-Low']
HP_COMP_LIST = ['In-Ear Headphones - High Strength','In-Ear Headphones - Low Strength','Over/On-Ear Headphones - High Strength','Over/On-Ear Headphones - Low Strength']
HP_COMP_LIST_SHORT = ['In-Ear-High','In-Ear-Low','Over+On-Ear-High','Over+On-Ear-Low']



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
NUM_SOURCE_AZIM = 7
NUM_OUT_CHANNELS_HE = NUM_SOURCE_AZIM*2
NUM_OUT_CHANNELS_TS = 4

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
EAPO_MUTE_GAIN=-60.0

#spatial resolution
SPATIAL_RES_LIST = ['Low','Medium','High','Max']#0=low,1=med,2=high,3=max
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
SPATIAL_RES_ELEV_MIN=[-60, -60, -60, -40 ]#as per hrir dataset
SPATIAL_RES_ELEV_MAX=[60, 60, 60, 60 ]#as per hrir dataset
SPATIAL_RES_ELEV_MIN_OUT=[-30, -45, -50, -40 ]#reduced set
SPATIAL_RES_ELEV_MAX_OUT=[30, 45, 50, 40 ]#reduced set
SPATIAL_RES_ELEV_NEAREST=[5, 5, 5, 2]#as per hrir dataset
SPATIAL_RES_AZIM_NEAREST=[5, 5, 5, 2]#as per hrir dataset
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


#AIR and BRIR reverberation processing
LIMIT_REBERB_DIRS=True
MIN_REVERB_DIRS=1
MAX_IRS=3500#2520
MAX_IRS_TRANSFORM=2500#2520,1550,1050,2500 for combined
IR_MIN_THRESHOLD=500#before this number of IRs, transform will be applied to expand dataset
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
USER_CSV_KEY = "_metadata"

#section to load dict lists containing acoustic space metadata
#new: load as dict lists
refresh_acoustic_space_metadata()

# # Load internal metadata
# csv_directory = pjoin(DATA_DIR_INT, 'reverberation')
# reverb_data = load_csv_as_dicts(csv_directory, 'reverberation_metadata.csv')
# # Load additional user metadata
# user_csv_dir = DATA_DIR_AS_USER
# USER_CSV_KEY = "_metadata"
# user_reverb_data = load_user_reverb_csvs_recursive(user_csv_dir, USER_CSV_KEY)
# # Build a set of existing file_name values for duplicate detection
# existing_keys = set(row.get("file_name") for row in reverb_data if "file_name" in row)
# # Filter out duplicates from user data
# unique_user_reverb_data = [row for row in user_reverb_data if row.get("file_name") not in existing_keys]
# # Merge filtered user data into internal data
# reverb_data.extend(unique_user_reverb_data)
# #sort by name
# reverb_data = sort_dict_list(reverb_data, sort_key='name_src', reverse=False)
# #grab individual lists
# AC_SPACE_LIST_GUI = extract_column(data=reverb_data, column='name_gui')  
# AC_SPACE_LIST_SHORT = extract_column(data=reverb_data, column='name_short') 
# AC_SPACE_LIST_SRC = extract_column(data=reverb_data, column='name_src')                       #
# AC_SPACE_EST_R60 = extract_column(data=reverb_data, column='est_rt60')                      #
# AC_SPACE_MEAS_R60 = extract_column(data=reverb_data, column='meas_rt60')                      #
# AC_SPACE_FADE_START = extract_column(data=reverb_data, column='fade_start')                   #
# AC_SPACE_GAINS = extract_column(data=reverb_data, column='gain')                              #
# AC_SPACE_DESCR = extract_column(data=reverb_data, column='description')   
# AC_SPACE_DATASET = extract_column(data=reverb_data, column='source_dataset')   
# AC_SPACE_LIST_NR = extract_column(data=reverb_data, column='name_src', condition_key='noise_reduce', condition_value='Yes', return_all_matches=True)
# AC_SPACE_LIST_LOWRT60 = extract_column(data=reverb_data, column='name_src', condition_key='low_rt60', condition_value='Yes', return_all_matches=True)
# AC_SPACE_LIST_HIRT60 = extract_column(data=reverb_data, column='name_src', condition_key='low_rt60', condition_value='No', return_all_matches=True)
# AC_SPACE_LIST_COMP = extract_column(data=reverb_data, column='name_src', condition_key='folder', condition_value='comp_bin', return_all_matches=True)
# AC_SPACE_LIST_MAX_SRC = extract_column(data=reverb_data, column='name_src', condition_key='source_hrtf_dataset', condition_value='max', return_all_matches=True)
# AC_SPACE_LIST_ID_SRC = extract_column(data=reverb_data, column='source_hrtf_id')



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
HRTF_TYPE_LIST = ['Dummy Head / Head & Torso Simulator', 'Human Listener', 'User SOFA Input']
HRTF_TYPE_LIST_FULL = ['Dummy Head / Head & Torso Simulator', 'Human Listener', 'User SOFA Input','Dummy Head - Max Resolution']

#new: load as dict lists
csv_directory = DATA_DIR_HRIR_NPY
hrtf_data = load_csv_as_dicts(csv_directory,'hrir_metadata.csv')
HRTF_DATASET_LIST_INDV = extract_column(data=hrtf_data, column= 'dataset', condition_key='hrtf_type', condition_value='Human Listener', return_all_matches=True)
HRTF_DATASET_LIST_DUMMY = extract_column(data=hrtf_data, column='dataset', condition_key='hrtf_type', condition_value='Dummy Head / Head & Torso Simulator', return_all_matches=True)
HRTF_DATASET_LIST_DUMMY_MAX = extract_column(data=hrtf_data, column='dataset', condition_key='hrtf_index_max', condition_value=0, condition_op='!=', return_all_matches=True)
HRTF_TYPE_DEFAULT = extract_column(data=hrtf_data, column='hrtf_type', condition_key='default', condition_value='Yes', return_all_matches=False)
HRTF_DATASET_DEFAULT = extract_column(data=hrtf_data, column='dataset', condition_key='default', condition_value='Yes', return_all_matches=False)
HRTF_LISTENER_DEFAULT = extract_column(data=hrtf_data, column='name_gui', condition_key='default', condition_value='Yes', return_all_matches=False)
HRTF_DATASET_LIST_CUSTOM = ['N/A']
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
    HRTF_TYPE_LIST_FULL[3]: HRTF_DATASET_LIST_DUMMY_MAX
}


#SUB Related
SUB_FC_SETTING_LIST = ['Auto Select', 'Custom Value']
SUB_FC_MIN=40
SUB_FC_MAX=150
SUB_FC_DEFAULT=120

#new: load as dict lists
csv_directory = pjoin(DATA_DIR_INT, 'sub')
sub_data = load_csv_as_dicts(csv_directory,'sub_brir_metadata.csv')
SUB_RESPONSE_LIST_GUI = extract_column(data=sub_data, column='name_gui') 
SUB_RESPONSE_LIST_SHORT = extract_column(data=sub_data, column='name_short') 
SUB_RESPONSE_LIST_AS = extract_column(data=sub_data, column='acoustic_space') 
SUB_RESPONSE_LIST_R60 = extract_column(data=sub_data, column='est_rt60') 
SUB_RESPONSE_LIST_COMM = extract_column(data=sub_data, column='comments') 
SUB_RESPONSE_LIST_RANGE = extract_column(data=sub_data, column='frequency_range') 
SUB_RESPONSE_LIST_TOL = extract_column(data=sub_data, column='tolerance') 
SUB_PLOT_LIST = ['Magnitude', 'Group Delay']

#plotting
IMPULSE=np.zeros(N_FFT)
IMPULSE[0]=1
FR_FLAT_MAG = np.abs(np.fft.fft(IMPULSE))


# Define the center frequencies of the 1/3-octave bands
OCTAVE_BANDS = np.array([50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 
                         500, 630, 800, 1000, 1250, 1600, 2000, 2500, 
                         3150, 4000, 5000, 6300, 8000, 10000])