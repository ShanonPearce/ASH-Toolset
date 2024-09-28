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


################################ 
#Constants

PLOT_ENABLE = 0
LOG_INFO=1     
LOG_GUI=1

N_FFT = 65536
N_FFT_L = int(65536*2)
N_UNIQUE_PTS = int(np.ceil((N_FFT+1)/2.0))
SAMP_FREQ = 44100
FS=SAMP_FREQ

SPECT_SMOOTH_MODE=1#0=use single pass smoothing, 1 = octave smoothing
APPLY_SUB_EQ = 0#1
APPLY_ROOM_TARGET = 1
APPLY_ADD_HP_EQ = 1
HRIR_MODE=1#0= load .mat dataset, 1=load .npy dataset
SUBRIR_MODE=1#0= load .mat dataset, 1=load .npy dataset
ROOM_TARGET_MODE=1#0= load .mat dataset, 1=load .npy dataset

#control level of direct sound
DIRECT_SCALING_FACTOR = 1.0#reference level - approx 0db DRR. was 0.1

#crossover frequency for low frequency BRIR integration
ENABLE_SUB_INTEGRATION = 1
F_CROSSOVER = 120#125
F_CROSSOVER_HI = 140#140
F_CROSSOVER_MID = 130
F_CROSSOVER_LOW = 100#110
CUTOFF_SUB = F_CROSSOVER
PEAK_TO_PEAK_WINDOW_SUB = int(np.divide(SAMP_FREQ,CUTOFF_SUB)*0.95)#np.divide(SAMP_FREQ,CUTOFF_SUB)  #peak to peak within a sufficiently small sample window
PEAK_MEAS_MODE=1#0=local max peak, 1 =peak to peak

#contants for TD alignment of BRIRs with other BRIRs
T_SHIFT_INTERVAL = 25#50
MIN_T_SHIFT = -1500
MAX_T_SHIFT = 0
NUM_INTERVALS = int(np.abs((MAX_T_SHIFT-MIN_T_SHIFT)/T_SHIFT_INTERVAL))
ORDER=7#default 6
DELAY_WIN_MIN_T = 0
DELAY_WIN_MAX_T = 1200#1500
GRP_DELAY_MIN_F = 30
GRP_DELAY_MAX_F = 150
CUTOFF_ALIGNMENT = 130#140
#peak to peak within a sufficiently small sample window
PEAK_TO_PEAK_WINDOW = int(np.divide(SAMP_FREQ,CUTOFF_ALIGNMENT)*0.95) 

#constants to perform td alignment of BRIRs with sub brir
T_SHIFT_INTERVAL_C = 25
MIN_T_SHIFT_C = -250#-250
MAX_T_SHIFT_C = 50#150
NUM_INTERVALS_C = int(np.abs((MAX_T_SHIFT_C-MIN_T_SHIFT_C)/T_SHIFT_INTERVAL_C))
T_SHIFT_INTERVAL_S = 10
MIN_T_SHIFT_S = -250#-250
MAX_T_SHIFT_S = 250#150
NUM_INTERVALS_S = int(np.abs((MAX_T_SHIFT_S-MIN_T_SHIFT_S)/T_SHIFT_INTERVAL_S))
T_SHIFT_INTERVAL_N = 25
MIN_T_SHIFT_N = -250#-250
MAX_T_SHIFT_N = 100#150
NUM_INTERVALS_N = int(np.abs((MAX_T_SHIFT_N-MIN_T_SHIFT_N)/T_SHIFT_INTERVAL_N))
T_SHIFT_INTERVAL_P = 25
MIN_T_SHIFT_P = -250#-250
MAX_T_SHIFT_P = 50#150
NUM_INTERVALS_P = int(np.abs((MAX_T_SHIFT_P-MIN_T_SHIFT_P)/T_SHIFT_INTERVAL_P))

#new constants
DELAY_WIN_HOP_SIZE = 5
DELAY_WIN_HOPS = int((DELAY_WIN_MAX_T-DELAY_WIN_MIN_T)/DELAY_WIN_HOP_SIZE)
EVAL_POLARITY=True

#AIR alignment
MIN_T_SHIFT_A = -1000#-1000,-800
MAX_T_SHIFT_A = 250#50,250
DELAY_WIN_MIN_A = 0
DELAY_WIN_MAX_A = 1000#1500
DELAY_WIN_HOPS_A = int((DELAY_WIN_MAX_A-DELAY_WIN_MIN_A)/DELAY_WIN_HOP_SIZE)
MIN_T_SHIFT_B = -3000#for longer delays
MAX_T_SHIFT_B = 0#
CUTOFF_ALIGNMENT_AIR = F_CROSSOVER#140
PEAK_TO_PEAK_WINDOW_AIR = int(np.divide(SAMP_FREQ,CUTOFF_ALIGNMENT)*0.95) 

#option to limit azimuths for TD alignment
ALIGN_LIMIT_AZIM = 1

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
DATA_DIR_EXT = pjoin(BASE_DIR_OS, 'data','external')
DATA_DIR_SOFA = pjoin(BASE_DIR_OS, 'data','external','SOFA')
DATA_DIR_SOFA_USER = pjoin(BASE_DIR_OS, 'data','external','SOFA','user')
DATA_DIR_ASSETS = pjoin(BASE_DIR_OS, 'data','external','assets')
DATA_DIR_RAW = pjoin(BASE_DIR_OS, 'data','raw')
DATA_DIR_RAW_HP_MEASRUEMENTS = pjoin(BASE_DIR_OS, 'data','raw','headphone_measurements')
DATA_DIR_ROOT = pjoin(BASE_DIR_OS, 'data')
DATA_DIR_OUTPUT = pjoin(BASE_DIR_OS, 'data','processed')
PROJECT_FOLDER = 'ASH-Custom-Set'    
PROJECT_FOLDER_BRIRS = pjoin(PROJECT_FOLDER, 'BRIRs')  
PROJECT_FOLDER_BRIRS_SOFA = pjoin(PROJECT_FOLDER, 'BRIRs', 'SOFA')  
PROJECT_FOLDER_CONFIGS = pjoin(PROJECT_FOLDER, 'E-APO-Configs') 
PROJECT_FOLDER_CONFIGS_BRIR = pjoin(PROJECT_FOLDER, 'E-APO-Configs','BRIR-Convolution') 
PROJECT_FOLDER_CONFIGS_HPCF = pjoin(PROJECT_FOLDER, 'E-APO-Configs','HpCF-Convolution') 
PROJECT_FOLDER_HPCFS = pjoin(PROJECT_FOLDER, 'HpCFs')  
ICON_LOCATION= pjoin(DATA_DIR_RAW, 'ash_icon_1.ico')  
SETTINGS_FILE = pjoin(BASE_DIR_OS, 'settings.ini')
METADATA_FILE = pjoin(BASE_DIR_OS, 'metadata.json')

#constants for writing WAVs
NEAREST_AZ_WAV = 15
MIN_ELEV_WAV=0
MAX_ELEV_WAV=0
ELEV_OFFSET_WAV = np.abs(MIN_ELEV_WAV)
OUTPUT_AZIMS_WAV = int(360/NEAREST_AZ_WAV)

#Strings
#HRTF_LIST_NUM = ['01: Neumann KU 100 (SADIE)', '02: Neumann KU 100 (TH Köln)', '03: FABIAN HATS', '04: B&K Type 4128', '05: B&K Type 4128C (MMHR-HRIR)', '06: DADEC (MMHR-HRIR)', '07: HEAD acoustics HMSII.2 (MMHR-HRIR)', '08: KEMAR (MMHR-HRIR)', '09: KEMAR-N (MIT)', '10: KEMAR-L (MIT)', '11: KEMAR (SADIE)', '12: KEMAR-N (PKU-IOA)', '13: KEMAR-L (PKU-IOA)']
HRTF_LIST_NUM = ['Neumann KU 100 (SADIE)', 'Neumann KU 100 (TH Köln)', 'FABIAN HATS', 'B&K Type 4128', 'B&K Type 4128C (MMHR-HRIR)', 'DADEC (MMHR-HRIR)', 'HEAD acoustics HMSII.2 (MMHR-HRIR)', 'KEMAR (MMHR-HRIR)', 'KEMAR-N (MIT)', 'KEMAR-L (MIT)', 'KEMAR (SADIE)', 'KEMAR-N (PKU-IOA)', 'KEMAR-L (PKU-IOA)']
HRTF_LIST_SHORT = ['KU_100_SADIE', 'KU_100_THK', 'FABIAN', 'B&K_4128', 'B&K_4128C_MMHR', 'DADEC_MMHR', 'HMSII.2_MMHR', 'KEMAR_MMHR', 'KEMAR-N_MIT', 'KEMAR-L_MIT', 'KEMAR_SADIE', 'KEMAR-N_PKU', 'KEMAR-L_PKU']
HRTF_GAIN_LIST = ['-11.0 dB', '-12.0 dB', '-11.0 dB', '-9.0 dB', '-8.0 dB', '-10.5 dB', '-9.5 dB', '-8.0 dB', '-12.0 dB', '-12.9 dB', '-10.8 dB', '-7.8 dB', '-11.0 dB']
HRTF_GAIN_LIST_NUM = [-7.0, -9.0, -8.0, -3.8, -2.3,-6.0,-5.5,-3.0,-7.5,-8.3,-5.0,-7.8,-7.8]
HRTF_GAIN_ADDIT = 3

HRTF_LIST_FULL_RES_NUM = ['Neumann KU 100 (TH Köln)', 'FABIAN HATS']
HRTF_LIST_FULL_RES_SHORT = ['KU_100_THK', 'FABIAN']
HRTF_GAIN_LIST_FULL_RES_NUM = [-9.0, -8.0]

HP_COMP_LIST = ['In-Ear Headphones - High Strength','In-Ear Headphones - Low Strength','Over/On-Ear Headphones - High Strength','Over/On-Ear Headphones - Low Strength']
HP_COMP_LIST_SHORT = ['In-Ear-High','In-Ear-Low','Over+On-Ear-High','Over+On-Ear-Low']

# ROOM_TARGET_LIST = ['Flat','ASH Target','Harman Target','HATS Target','Toole Target','rtings Target']
# ROOM_TARGET_LIST_SHORT = ['Flat','ASH-Target','Harman-Target','HATS-Target','Toole-Target','rtings-Target']
# ROOM_TARGET_LIST_FIRS = ['Flat','ash_room_target_fir','harman_b_room_target_fir','hats_room_target_fir','toole_trained_room_target_fir','rtings_target_fir']
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
BRIR_EXPORT_ENABLE = 1
AZIM_DICT = {'WIDE_BL':'-135','WIDE_BR':'135','NARROW_BL':'-150','NARROW_BR':'150','WIDEST_BL':'-120','WIDEST_BR':'120','SL':'-90','SR':'90','FL':'-30','FR':'30','FC':'0','WIDE_FL':'-35','WIDE_FR':'35','NARROW_FL':'-25','NARROW_FR':'25'}
CHANNEL_CONFIGS = [['2.0_Stereo','2.0','2.0 Stereo'],['2.0_Stereo_Narrow','2.0N','2.0 Stereo (narrow placement)'],['2.0_Stereo_Wide','2.0W','2.0 Stereo (wide placement)'],['7.1_Surround_Narrow_Back','7.1N','7.1 surround (narrow back placement)'],['7.1_Surround_Wide_Back','7.1W','7.1 surround (wide back placement)'],['5.1_Surround','5.1','5.1 surround']]
AUDIO_CHANNELS = ['2.0 Stereo','5.1 Surround','7.1 Surround','7.1 Downmix to Stereo']
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
AZ_ANGLES_SL_WAV = [-120,-105,-90,-75,-60]
AZ_ANGLES_SR_WAV = [120,105,90,75,60]
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

#spatial resolution
SPATIAL_RES_LIST = ['Low','Medium','High','Max']
NUM_SPATIAL_RES = len(SPATIAL_RES_LIST)
SPATIAL_RES_ELEV_MIN=[-60, -60, -60, -40 ]#as per hrir dataset
SPATIAL_RES_ELEV_MAX=[60, 60, 60, 60 ]#as per hrir dataset
SPATIAL_RES_ELEV_MIN_OUT=[-30, -45, -50, -40 ]#reduced set
SPATIAL_RES_ELEV_MAX_OUT=[30, 45, 50, 40 ]#reduced set
SPATIAL_RES_ELEV_NEAREST=[5, 5, 5, 2]#as per hrir dataset
SPATIAL_RES_AZIM_NEAREST=[5, 5, 5, 2]#as per hrir dataset
SPATIAL_RES_ELEV_NEAREST_PR=[15, 15, 5, 2]#reduced set
SPATIAL_RES_AZIM_NEAREST_PR=[5, 5, 5, 2]#reduced set

#Head tracking related
THREAD_FPS = 10
THREAD_FPS_MIN=3
THREAD_FPS_MAX=30
PROVIDERS_CUDA=['CUDAExecutionProvider', 'CPUExecutionProvider']
PROVIDERS_DML=['DmlExecutionProvider', 'CPUExecutionProvider']
TRACK_SENSITIVITY = 4
BRIR_SEL_WIN_WIDTH_RED=390
BRIR_SEL_WIN_WIDTH_FULL=500
BRIR_SEL_LIST_WIDTH_RED=370
BRIR_SEL_LIST_WIDTH_FULL=480
HP_SEL_WIN_WIDTH_RED=360
HP_SEL_WIN_WIDTH_FULL=420
HP_SEL_LIST_WIDTH_RED=217
HP_SEL_LIST_WIDTH_FULL=277

#hpcf related
NUM_ITER = 8 #4
PARAMS_PER_ITER = 4 #8
SENSITIVITY = 0.003 #0.003
FREQ_CUTOFF = 20000#f in bins
FREQ_CUTOFF_GEQ = 29000
EAPO_GAIN_ADJUST = 0.9*1.1*1.1*1.1
EAPO_QF_ADJUST = 0.5*0.8
HPCF_FIR_LENGTH = 1024
DIRECT_GAIN_MAX=8#6
DIRECT_GAIN_MIN=-8#-6

SAMPLE_RATE_LIST = ['44.1 kHz', '48 kHz', '96 kHz']
SAMPLE_RATE_DICT = {'44.1 kHz': 44100, '48 kHz': 48000, '96 kHz': 96000}  
BIT_DEPTH_LIST = ['24 bit', '32 bit']
BIT_DEPTH_DICT = {'24 bit': 'PCM_24', '32 bit': 'PCM_32'}  

HEAD_TRACK_RUNNING = False
PROCESS_BRIRS_RUNNING = False
SHOW_DEV_TOOLS=True
STOP_THREAD_FLAG = False

#AIR and BRIR reverberation processing
LIMIT_REBERB_DIRS=True
MIN_REVERB_DIRS=1
MAX_IRS=1050#980
AC_SPACE_LIST_GUI = [ 'Audio Lab A','Audio Lab B','Audio Lab C','Audio Lab D','Audio Lab E','Audio Lab F','Audio Lab G','Audio Lab H', 'Auditorium','Broadcast Studio',
                     'Concert Hall','Conference Room','Control Room','Hall','Office', 'Outdoors', 'Seminar Room', 'Studio', 'Tatami Room']

AC_SPACE_LIST_SHORT = [ 'Audio-Lab-A','Audio-Lab-B','Audio-Lab-C','Audio-Lab-D','Audio-Lab-E','Audio-Lab-F','Audio-Lab-G','Audio-Lab-H', 'Auditorium','Bc-Studio', 
                     'Concert-Hall','Conference-Rm','Control-Rm','Hall','Office', 'Outdoors', 'Seminar-Rm', 'Studio', 'Tatami-Rm']

AC_SPACE_LIST_SRC = [ 'audio_lab_a','audio_lab_b','audio_lab_c','audio_lab_d','audio_lab_e','audio_lab_f','audio_lab_g','audio_lab_h', 'auditorium_a','broadcast_studio_a',
                     'concert_hall_a','conference_room_a','control_room_a','hall_a','office_a', 'outdoors_a', 'seminar_room_a', 'studio_a', 'tatami_room_a']

#estimated RT60 in ms, rough estimate only
AC_SPACE_EST_R60 = [ 300,600,500,300,300,900,400,600, 1500,1400,
                     1500,500,300,1500,300, 1500, 1000, 300,500]
#estimated time in ms to start fade out
AC_SPACE_FADE_START = [ 320,700,530,400,410,0,450,700, 0,1400,
                     0,560,400,0,520, 0, 1100, 550,700]


AC_SPACE_LIST_LOWRT60 = ['audio_lab_a','audio_lab_b', 'audio_lab_c', 'audio_lab_d','audio_lab_e','audio_lab_f','audio_lab_g','audio_lab_h', 'auditorium_a',
                         'conference_room_a','control_room_a','office_a','seminar_room_a', 'studio_a','sub_set_a','sub_set_b','sub_set_c','sub_set_d', 'tatami_room_a']

AC_SPACE_LIST_COMP = ['generic_room_a', 'auditorium_a']
AC_SPACE_LIST_KU100 = ['concert_hall_a','music_chamber_a','theater_a']
AC_SPACE_LIST_SOFA_0 = ['audio_lab_a','office_a','control_room_a', 'control_room_b']
AC_SPACE_LIST_SOFA_1 = ['']
AC_SPACE_LIST_SOFA_2 = ['']
AC_SPACE_LIST_SOFA_3 = ['conference_room_a']
AC_SPACE_LIST_SOFA_4 = ['audio_lab_g','audio_lab_h', 'audio_lab_b','broadcast_studio_a','sub_set_d']
AC_SPACE_LIST_SOFA_5 = ['']
AC_SPACE_LIST_ISO = ['studio_b','sub_set_a','sub_set_b']
AC_SPACE_LIST_SUBRIRDB = ['sub_set_c']
AC_SPACE_LIST_NR = ['tatami_room_a']
AC_SPACE_LIST_CUTOFF = ['concert_hall_a']
AC_SPACE_LIST_LIM_CHANS = ['concert_hall_a']
AC_SPACE_LIST_LIM_SETS = ['hall_a', 'outdoors_a']
AC_SPACE_CHAN_LIMITED=5#3,4
AC_SPACE_LIST_NOROLL = ['auditorium_a', 'auditorium_b']
AC_SPACE_LIST_NOCOMP = ['audio_lab_d']
AC_SPACE_LIST_WINDOW = ['hall_b']
AC_SPACE_LIST_SUB = ['sub_set_a','sub_set_b','sub_set_c','sub_set_d']
AC_SPACE_LIST_RWCP = ['audio_lab_f','conference_room_b', 'tatami_room_a']
AC_SPACE_LIST_VARIED_R = [' ']
AC_SPACE_LIST_HI_FC = [ 'audio_lab_c','control_room_a', 'auditorium_a']
AC_SPACE_LIST_MID_FC = ['audio_lab_a', 'studio_a','hall_a','office_a']
AC_SPACE_LIST_LOW_FC = ['broadcast_studio_a', 'outdoors_a','concert_hall_a']
AC_SPACE_LIST_AVG = ['audio_lab_a','audio_lab_b','audio_lab_d','control_room_a','conference_room_a','control_room_a','office_a','audio_lab_g', 'audio_lab_f','audio_lab_e','audio_lab_h']

RT60_MIN=200
RT60_MAX_S=1250
RT60_MAX_L=2250
MAG_COMP=True#enables DF compensation in AIR processing
RISE_WINDOW=True#enables windowing of initial rise in AIR processing