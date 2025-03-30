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

################################ 
#Constants


N_FFT = 65536
N_FFT_L = int(65536*2)
N_UNIQUE_PTS = int(np.ceil((N_FFT+1)/2.0))
SAMP_FREQ = 44100#sample rate for ash toolset
FS=SAMP_FREQ
THRESHOLD_CROP = 0.0000005#0.0000005 -120db with reference level of 0.5
SPECT_SMOOTH_MODE=1#0=use single pass smoothing, 1 = octave smoothing
APPLY_SUB_EQ = 0#1
APPLY_ROOM_TARGET = 1
APPLY_ADD_HP_EQ = 1
HRIR_MODE=1#0= load .mat dataset, 1=load .npy dataset
SUBRIR_MODE=1#0= load .mat dataset, 1=load .npy dataset
ROOM_TARGET_MODE=1#0= load .mat dataset, 1=load .npy dataset
PINNA_COMP_MODE=1#0= load .mat dataset, 1=load .npy dataset

PLOT_ENABLE = False#False
LOG_INFO=True     
LOG_GUI=True
HEAD_TRACK_RUNNING = False
PROCESS_BRIRS_RUNNING = False
SHOW_DEV_TOOLS=False
STOP_THREAD_FLAG = False

#control level of direct sound
DIRECT_SCALING_FACTOR = 1.0#reference level - approx 0db DRR. was 0.1

#crossover frequency for low frequency BRIR integration
ENABLE_SUB_INTEGRATION = True
F_CROSSOVER_HI = 140#140
F_CROSSOVER_MID = 130
F_CROSSOVER = 120#120, default
F_CROSSOVER_LOW = 85#110,85
CUTOFF_SUB = F_CROSSOVER
PEAK_TO_PEAK_WINDOW_SUB = int(np.divide(SAMP_FREQ,CUTOFF_SUB)*0.95)#np.divide(SAMP_FREQ,CUTOFF_SUB)  #peak to peak within a sufficiently small sample window
PEAK_MEAS_MODE=1#0=local max peak, 1 =peak to peak

#contants for TD alignment of BRIRs with other BRIRs
T_SHIFT_INTERVAL = 25#50
MIN_T_SHIFT = -1500
MAX_T_SHIFT = 0
NUM_INTERVALS = int(np.abs((MAX_T_SHIFT-MIN_T_SHIFT)/T_SHIFT_INTERVAL))
ORDER=8#8
DELAY_WIN_MIN_T = 0
DELAY_WIN_MAX_T = 1200#1500
GRP_DELAY_MIN_F = 30
GRP_DELAY_MAX_F = 150
CUTOFF_ALIGNMENT = 130#140
#peak to peak within a sufficiently small sample window
PEAK_TO_PEAK_WINDOW = int(np.divide(SAMP_FREQ,CUTOFF_ALIGNMENT)*0.95) 
#new constants
DELAY_WIN_HOP_SIZE = 5
DELAY_WIN_HOPS = int((DELAY_WIN_MAX_T-DELAY_WIN_MIN_T)/DELAY_WIN_HOP_SIZE)
EVAL_POLARITY=True#True

#constants to perform td alignment of BRIRs with sub brir
T_SHIFT_INTERVAL_C = 25
MIN_T_SHIFT_C = -250#-250
MAX_T_SHIFT_C = 50#150
NUM_INTERVALS_C = int(np.abs((MAX_T_SHIFT_C-MIN_T_SHIFT_C)/T_SHIFT_INTERVAL_C))
T_SHIFT_INTERVAL_S = 10
MIN_T_SHIFT_S = -370#-250
MAX_T_SHIFT_S = 250#250
NUM_INTERVALS_S = int(np.abs((MAX_T_SHIFT_S-MIN_T_SHIFT_S)/T_SHIFT_INTERVAL_S))
T_SHIFT_INTERVAL_N = 25
MIN_T_SHIFT_N = -225#-250
MAX_T_SHIFT_N = 100#100
NUM_INTERVALS_N = int(np.abs((MAX_T_SHIFT_N-MIN_T_SHIFT_N)/T_SHIFT_INTERVAL_N))
T_SHIFT_INTERVAL_P = 25
MIN_T_SHIFT_P = -225#-200
MAX_T_SHIFT_P = 100#50
NUM_INTERVALS_P = int(np.abs((MAX_T_SHIFT_P-MIN_T_SHIFT_P)/T_SHIFT_INTERVAL_P))
#reverb alignment
T_SHIFT_INTERVAL_R = 25
MIN_T_SHIFT_R = -500#-400
MAX_T_SHIFT_R = 100#100
NUM_INTERVALS_R = int(np.abs((MAX_T_SHIFT_P-MIN_T_SHIFT_P)/T_SHIFT_INTERVAL_P))
#hrir alignment
T_SHIFT_INTERVAL_H = 5
MIN_T_SHIFT_H = -30#
MAX_T_SHIFT_H = 50
NUM_INTERVALS_H = int(np.abs((MAX_T_SHIFT_H-MIN_T_SHIFT_H)/T_SHIFT_INTERVAL_H))
DELAY_WIN_MIN_H = 0
DELAY_WIN_MAX_H = 150
DELAY_WIN_HOPS_H = int((DELAY_WIN_MAX_H-DELAY_WIN_MIN_H)/DELAY_WIN_HOP_SIZE)
#AIR alignment
MIN_T_SHIFT_A = -1000#-1000,-800
MAX_T_SHIFT_A = 250#50,250
DELAY_WIN_MIN_A = 0
DELAY_WIN_MAX_A = 1000#1500
DELAY_WIN_HOPS_A = int((DELAY_WIN_MAX_A-DELAY_WIN_MIN_A)/DELAY_WIN_HOP_SIZE)
MIN_T_SHIFT_B = -3000#for longer delays
MAX_T_SHIFT_B = 0#
MIN_T_SHIFT_D = -750#
MAX_T_SHIFT_D = 200#
CUTOFF_ALIGNMENT_AIR = F_CROSSOVER_LOW#F_CROSSOVER
PEAK_TO_PEAK_WINDOW_AIR = int(np.divide(SAMP_FREQ,CUTOFF_ALIGNMENT)*0.95) 


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
DATA_DIR_HRIR_NPY = pjoin(BASE_DIR_OS, 'data','interim','hrir')
DATA_DIR_HRIR_NPY_DH = pjoin(BASE_DIR_OS, 'data','interim','hrir','dh')
DATA_DIR_HRIR_NPY_HL = pjoin(BASE_DIR_OS, 'data','interim','hrir','hl')
DATA_DIR_HRIR_NPY_USER = pjoin(BASE_DIR_OS, 'data','interim','hrir','user')
DATA_DIR_EXT = pjoin(BASE_DIR_OS, 'data','external')
DATA_DIR_SOFA = pjoin(BASE_DIR_OS, 'data','external','SOFA')
DATA_DIR_SOFA_USER = pjoin(BASE_DIR_OS, 'data','user','SOFA')
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
SETTINGS_FILE = pjoin(BASE_DIR_OS, 'settings.ini')
METADATA_FILE = pjoin(BASE_DIR_OS, 'metadata.json')
FOLDER_BRIRS_LIVE = 'live_dataset'

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


HP_COMP_LIST = ['In-Ear Headphones - High Strength','In-Ear Headphones - Low Strength','Over/On-Ear Headphones - High Strength','Over/On-Ear Headphones - Low Strength']
HP_COMP_LIST_SHORT = ['In-Ear-High','In-Ear-Low','Over+On-Ear-High','Over+On-Ear-Low']

ROOM_TARGET_LIST = ['Flat','ASH Target','Harman Target','HATS Target','Toole Target','rtings Target',
                    'ASH Target - Low End','Harman Target - Low End','HATS Target - Low End','Toole Target - Low End','rtings Target - Low End',
                    'ASH Target - Flat Highs','Harman Target - Flat Highs','HATS Target - Flat Highs','Toole Target - Flat Highs','rtings Target - Flat Highs']
ROOM_TARGET_LIST_SHORT = ['Flat','ASH-Target','Harman-Target','HATS-Target','Toole-Target','rtings-Target',
                          'ASH-Target-Low-End','Harman-Target-Low-End','HATS-Target-Low-End','Toole-Target-Low-End','rtings-Target-Low-End',
                          'ASH-Target-Flat-Highs','Harman-Target-Flat-Highs','HATS-Target-Flat-Highs','Toole-Target-Flat-Highs','rtings-Target-Flat-Highs']
ROOM_TARGET_LIST_FIRS = ['Flat','ash_room_target_fir','harman_b_room_target_fir','hats_room_target_fir','toole_trained_room_target_fir','rtings_target_fir']

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

#Head tracking related
THREAD_FPS = 10
THREAD_FPS_MIN=3
THREAD_FPS_MAX=30
PROVIDERS_CUDA=['CUDAExecutionProvider', 'CPUExecutionProvider']
PROVIDERS_DML=['DmlExecutionProvider', 'CPUExecutionProvider']
TRACK_SENSITIVITY = 4
BRIR_SEL_WIN_WIDTH_RED=390
BRIR_SEL_WIN_WIDTH_FULL=520
BRIR_SEL_LIST_WIDTH_RED=370
BRIR_SEL_LIST_WIDTH_FULL=500
HP_SEL_WIN_WIDTH_RED=360
HP_SEL_WIN_WIDTH_FULL=420
HP_SEL_LIST_WIDTH_RED=217
HP_SEL_LIST_WIDTH_FULL=277

#hpcf related
NUM_ITER = 8 #was 4
PARAMS_PER_ITER = 4 #was 8
SENSITIVITY = 0.003 #0.003
FREQ_CUTOFF = 20000#f in bins
FREQ_CUTOFF_GEQ = 29000
EAPO_GAIN_ADJUST = 0.9*1.1*1.1*1.1
EAPO_QF_ADJUST = 0.5*0.8
HPCF_FIR_LENGTH = 384#was 1024, then 512
#gui
DIRECT_GAIN_MAX=10#8
DIRECT_GAIN_MIN=-10#-8
EAPO_ERROR_CODE=501
ER_RISE_MAX=10#in ms
ER_RISE_MIN=0#

SAMPLE_RATE_LIST = ['44.1 kHz', '48 kHz', '96 kHz']
SAMPLE_RATE_DICT = {'44.1 kHz': 44100, '48 kHz': 48000, '96 kHz': 96000}  
BIT_DEPTH_LIST = ['24 bit', '32 bit']
BIT_DEPTH_DICT = {'24 bit': 'PCM_24', '32 bit': 'PCM_32'}  
HRTF_SYM_LIST = ['Disabled', 'Mirror Left Side', 'Mirror Right Side']


#AIR and BRIR reverberation processing
LIMIT_REBERB_DIRS=True
MIN_REVERB_DIRS=1
MAX_IRS=2520#1260
DIRECTION_MODE=1#0=one set of azimuths for each source, 1 = separate set of azimuths for left and right hem,2 = random using triangle distribution
MAG_COMP=True#enables DF compensation in AIR processing
RISE_WINDOW=True#enables windowing of initial rise in AIR processing

#deprecated
RT60_MIN=200
RT60_MAX_S=1250
RT60_MAX_L=2250

#Acoustic space related
AC_SPACE_LIST_COMP = ['generic_room_a', 'auditorium_a']
AC_SPACE_LIST_KU100 = ['concert_hall_a','music_chamber_a','theater_a']
AC_SPACE_LIST_SOFA_0 = ['audio_lab_a','office_a','studio_a']
AC_SPACE_LIST_SOFA_1 = ['']
AC_SPACE_LIST_SOFA_2 = ['']
AC_SPACE_LIST_SOFA_3 = ['','broadcast_studio_a']
AC_SPACE_LIST_SOFA_4 = ['audio_lab_g','audio_lab_h', 'audio_lab_b','sub_set_d','audio_lab_i']
AC_SPACE_LIST_SOFA_5 = ['','control_room_a','conference_room_a']
AC_SPACE_LIST_ISO = ['sub_set_a','sub_set_b']
AC_SPACE_LIST_ANU = ['studio_b']
AC_SPACE_LIST_44100 = ['outdoors_b']
AC_SPACE_LIST_SUBRIRDB = ['sub_set_c']
AC_SPACE_LIST_NR = ['tatami_room_a']
AC_SPACE_LIST_CUTOFF = ['concert_hall_a']
AC_SPACE_LIST_LIM_CHANS = ['concert_hall_a','hall_a']
AC_SPACE_LIST_LIM_SETS = ['hall_a', 'outdoors_a','broadcast_studio_a','concert_hall_a', 'outdoors_b','seminar_room_a']
AC_SPACE_CHAN_LIMITED=5#3,4
AC_SPACE_LIST_NOROLL = ['auditorium_a', 'auditorium_b']
AC_SPACE_LIST_NOCOMP = ['audio_lab_d']
AC_SPACE_LIST_COMPMODE1 = ['control_room_a', 'tatami_room_a', 'office_a','studio_b']
AC_SPACE_LIST_WINDOW = ['hall_a', 'outdoors_a','seminar_room_a','broadcast_studio_a', 'outdoors_b']
AC_SPACE_LIST_WINDOW_ALL = ['']
AC_SPACE_LIST_SLOWRISE = ['studio_b','audio_lab_i','hall_a','small_room_a','medium_room_a','large_room_a','small_room_b','medium_room_b','large_room_b','small_room_c','small_room_d']
AC_SPACE_LIST_SUB = ['sub_set_a','sub_set_b','sub_set_c','sub_set_d','sub_set_e','sub_set_f']
AC_SPACE_LIST_RWCP = ['audio_lab_f','conference_room_b', 'tatami_room_a']
AC_SPACE_LIST_VARIED_R = [' ']
#AC_SPACE_LIST_HI_FC = [ 'audio_lab_c','control_room_a', 'auditorium_a', 'tatami_room_a','seminar_room_a','conference_room_a','concert_hall_a']
#AC_SPACE_LIST_MID_FC = ['audio_lab_a','hall_a','office_a']
#AC_SPACE_LIST_LOW_FC = ['small_room_a','small_room_c','small_room_d','large_room_a','large_room_b']#,'medium_room_a','medium_room_b','large_room_a','large_room_b','small_room_a','small_room_b',
AC_SPACE_LIST_AVG = ['audio_lab_a','audio_lab_b','audio_lab_d','control_room_a','conference_room_a','control_room_a','office_a','audio_lab_g', 'audio_lab_f','audio_lab_e','audio_lab_h']
AC_SPACE_LIST_HRTF_1 = ['small_room_b','small_room_d','medium_room_b','large_room_b']#
AC_SPACE_LIST_ALL_AZ = ['small_room_a','small_room_b','small_room_c','small_room_d','small_room_e','large_room_a','large_room_b']

#load other lists from csv file
AC_SPACE_LIST_GUI = []
AC_SPACE_LIST_SHORT = []
AC_SPACE_LIST_SRC = []
#estimated RT60 in ms, rough estimate only
AC_SPACE_EST_R60 = []
#estimated RT60 in ms, measured
AC_SPACE_MEAS_R60 = []
#estimated time in ms to start fade out
AC_SPACE_FADE_START = []
AC_SPACE_LIST_LOWRT60 = []
AC_SPACE_GAINS = []
AC_SPACE_LIST_HI_FC = []
AC_SPACE_LIST_MID_FC = []
AC_SPACE_LIST_LOW_FC = []

try:
    #directories
    csv_directory = pjoin(DATA_DIR_INT, 'reverberation')
    #read metadata from csv. Expects reverberation_metadata.csv 
    metadata_file_name = 'reverberation_metadata.csv'
    metadata_file = pjoin(csv_directory, metadata_file_name)
    with open(metadata_file, encoding='utf-8-sig', newline='') as inputfile:
        reader = DictReader(inputfile)
        for row in reader:#rows 2 and onward
            #store each row as a dictionary
            #append to list of dictionaries
            AC_SPACE_LIST_GUI.append(row.get('name_gui'))  
            AC_SPACE_LIST_SHORT.append(row.get('name_short'))  
            AC_SPACE_LIST_SRC.append(row.get('name_src'))  
            AC_SPACE_EST_R60.append(int(row.get('est_rt60')))  
            AC_SPACE_MEAS_R60.append(int(row.get('meas_rt60')))  
            AC_SPACE_FADE_START.append(int(row.get('fade_start')))
            AC_SPACE_GAINS.append(float(row.get('gain'))) 
            low_rt_flag = row.get('low_rt60')
            name_src = row.get('name_src')
            if low_rt_flag == "Yes":
                AC_SPACE_LIST_LOWRT60.append(name_src)
            # low_fc_flag = row.get('low_fc')
            # mid_fc_flag = row.get('mid_fc')
            # hi_fc_flag = row.get('high_fc')
            # if low_fc_flag == "Yes":
            #     AC_SPACE_LIST_LOW_FC.append(name_src)
            # if mid_fc_flag == "Yes":
            #     AC_SPACE_LIST_MID_FC.append(name_src)
            # if hi_fc_flag == "Yes":
            #     AC_SPACE_LIST_HI_FC.append(name_src)

except Exception:
    pass


#SOFA related
SOFA_COMPAT_CONV = []
SOFA_COMPAT_VERS = []
SOFA_COMPAT_CONVERS = []
SOFA_OUTPUT_CONV = []
SOFA_OUTPUT_VERS = []
SOFA_OUTPUT_CONVERS = []

try:
    #directories
    #read metadata from csv. Expects reverberation_metadata.csv 
    metadata_file_name = 'supported_conventions.csv'
    metadata_file = pjoin(DATA_DIR_SOFA, metadata_file_name)
    with open(metadata_file, encoding='utf-8-sig', newline='') as inputfile:
        reader = DictReader(inputfile)
        for row in reader:#rows 2 and onward
            #store each row as a dictionary
            #append to list of dictionaries
            SOFA_COMPAT_CONV.append(row.get('Convention'))  
            SOFA_COMPAT_VERS.append(row.get('Version'))  
            SOFA_COMPAT_CONVERS.append(row.get('SOFAConventionsVersion'))  
            out_flag = row.get('OutputFormat')
            if out_flag == "Yes":
                SOFA_OUTPUT_CONV.append(row.get('Convention'))  
                SOFA_OUTPUT_VERS.append(row.get('Version'))  
                SOFA_OUTPUT_CONVERS.append(row.get('SOFAConventionsVersion'))  

except Exception:
    pass



#HRTF related - individual datasets
HRTF_A_GAIN_ADDIT = 2.5#
#Strings
HRTF_TYPE_LIST = ['Dummy Head / Head & Torso Simulator', 'Human Listener', 'User SOFA Input']
HRTF_TYPE_LIST_FULL = ['Dummy Head / Head & Torso Simulator', 'Human Listener', 'User SOFA Input','Dummy Head - Max Resolution']
HRTF_DATASET_LIST_DUMMY = []
HRTF_DATASET_LIST_DUMMY_MAX = []
HRTF_DATASET_LIST_INDV = []
HRTF_DATASET_LIST_CUSTOM = ['N/A']
HRTF_TYPE_DEFAULT=''
HRTF_DATASET_DEFAULT=''
HRTF_LISTENER_DEFAULT=''
#load lists from csv file
try:

    #directories
    csv_directory = DATA_DIR_HRIR_NPY
    #read metadata from csv. Expects reverberation_metadata.csv 
    metadata_file_name = 'hrir_metadata.csv'
    metadata_file = pjoin(csv_directory, metadata_file_name)
    with open(metadata_file, encoding='utf-8-sig', newline='') as inputfile:
        reader = DictReader(inputfile)
        for row in reader:#rows 2 and onward
            #store each row as a dictionary
            #append to list of dictionaries
            if row.get('hrtf_type') == 'Human Listener':
                HRTF_DATASET_LIST_INDV.append(row.get('dataset')) 
            if row.get('hrtf_type') == 'Dummy Head / Head & Torso Simulator':
                HRTF_DATASET_LIST_DUMMY.append(row.get('dataset')) 
            if row.get('hrtf_index_max') != '':
                HRTF_DATASET_LIST_DUMMY_MAX.append(row.get('dataset')) 
            if row.get('default') == 'Yes':
                HRTF_TYPE_DEFAULT = row.get('hrtf_type')
                HRTF_DATASET_DEFAULT = row.get('dataset')
                HRTF_LISTENER_DEFAULT = row.get('name_gui')
                
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
    
 
except Exception:
    pass
#create a dictionary
HRTF_TYPE_DATASET_DICT = {
    HRTF_TYPE_LIST_FULL[0]: HRTF_DATASET_LIST_DUMMY,
    HRTF_TYPE_LIST_FULL[1]: HRTF_DATASET_LIST_INDV,
    HRTF_TYPE_LIST_FULL[2]: HRTF_DATASET_LIST_CUSTOM,
    HRTF_TYPE_LIST_FULL[3]: HRTF_DATASET_LIST_DUMMY_MAX
}


#GUI
PROGRESS_FIN=' Active   '
PROGRESS_START=' Ready to Apply'
PROGRESS_START_HPCF='Ready to Apply Selection'
PROGRESS_START_BRIR='Ready to Apply Parameters'
PROGRESS_FIN_ALT=' Finished   '
PROGRESS_START_ALT=' Ready to Start'
PROCESS_BUTTON_BRIR='Apply Parameters'
PROCESS_BUTTON_HPCF='Apply Selection'