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

#control level of direct sound
DIRECT_SCALING_FACTOR = 1.0#reference level - approx 0db DRR. was 0.1

#crossover frequency for low frequency BRIR integration
ENABLE_SUB_INTEGRATION = 1
F_CROSSOVER = 145#160 default,120 good with 7 order, 155
APPLY_SUB_EQ = 1

#apply any room target
APPLY_ROOM_TARGET = 1
APPLY_ADD_HP_EQ = 1

NEAREST_AZ_BRIR = 15
NEAREST_AZ_HRIR = 5
NEAREST_ELEV=15
N_FFT = 65536
SAMP_FREQ = 44100
FS=SAMP_FREQ
MIN_ELEV=-60
MAX_ELEV=60

ELEV_OFFSET = np.abs(MIN_ELEV)

#apply room sorting based on size
SORT_ROOMS_BY_SIZE=1

#enable shifting of rooms list
ROLL_ROOM = 0

#apply room weighting based on size
ROOM_WEIGHTING_DESC=1

#window for reverb shaping: 1=Hanning,2=Bartlett,3=blackman,4=hamming
WINDOW_TYPE=1#2
ALIGNMENT_METHOD = 5

TOTAL_CHAN_BRIR = 2

OUTPUT_AZIMS = int(360/NEAREST_AZ_HRIR)
INTERIM_ELEVS = 1
OUTPUT_ELEVS = int((MAX_ELEV-MIN_ELEV)/NEAREST_ELEV +1)
N_UNIQUE_PTS = int(np.ceil((N_FFT+1)/2.0))

#contants for TD alignment of BRIRs
T_SHIFT_INTERVAL = 50
MIN_T_SHIFT = -1500
MAX_T_SHIFT = 0
NUM_INTERVALS = int(np.abs((MAX_T_SHIFT-MIN_T_SHIFT)/T_SHIFT_INTERVAL))
ORDER=7#default 6
DELAY_WIN_MIN_T = 0
DELAY_WIN_MAX_T = 1500
GRP_DELAY_MIN_F = 30
GRP_DELAY_MAX_F = 150
CUTOFF_ALIGNMENT = 155
#peak to peak within a sufficiently small sample window
PEAK_TO_PEAK_WINDOW = int(np.divide(SAMP_FREQ,CUTOFF_ALIGNMENT)) 

#option to limit azimuths for TD alignment
ALIGN_LIMIT_AZIM = 1

#directories
BASE_DIR_OS= path.abspath(path.dirname(path.dirname(__file__)))#using os.path
SCRIPT_DIR_OS = path.abspath(path.dirname(__file__))#using os.path
BASE_DIR_PATH = Path(__file__).resolve().parents[1] #using Pathlib
SCRIPT_DIR_PATH = Path(__file__).resolve().parent #using Pathlib
SCRIPT_DIRECTORY = path.dirname(path.abspath(sys.argv[0]))#old method using os.path
DATA_DIR_INT = pjoin(BASE_DIR_OS, 'data','interim')
DATA_DIR_EXT = pjoin(BASE_DIR_OS, 'data','external')
DATA_DIR_RAW = pjoin(BASE_DIR_OS, 'data','raw')
DATA_DIR_RAW_HP_MEASRUEMENTS = pjoin(BASE_DIR_OS, 'data','raw','headphone_measurements')
DATA_DIR_ROOT = pjoin(BASE_DIR_OS, 'data')
DATA_DIR_OUTPUT = pjoin(BASE_DIR_OS, 'data','processed')
PROJECT_FOLDER = 'ASH-Custom-Set'    
PROJECT_FOLDER_BRIRS = pjoin(PROJECT_FOLDER, 'BRIRs')  
PROJECT_FOLDER_CONFIGS = pjoin(PROJECT_FOLDER, 'E-APO-Configs') 
PROJECT_FOLDER_CONFIGS_BRIR = pjoin(PROJECT_FOLDER, 'E-APO-Configs','BRIR-Convolution') 
PROJECT_FOLDER_CONFIGS_HPCF = pjoin(PROJECT_FOLDER, 'E-APO-Configs','HpCF-Convolution') 
PROJECT_FOLDER_HPCFS = pjoin(PROJECT_FOLDER, 'HpCFs')  
ICON_LOCATION= pjoin(DATA_DIR_RAW, 'ash_icon_1.ico')  
SETTINGS_FILE = pjoin(BASE_DIR_OS, 'settings.ini')
METADATA_FILE = pjoin(BASE_DIR_OS, 'metadata.json')

#constants to perform td alignment of BRIRs with sub brir
T_SHIFT_INTERVAL_C = 25
MIN_T_SHIFT_C = -175
MAX_T_SHIFT_C = 125
NUM_INTERVALS_C = int(np.abs((MAX_T_SHIFT_C-MIN_T_SHIFT_C)/T_SHIFT_INTERVAL_C))

CUTOFF_SUB = F_CROSSOVER
PEAK_TO_PEAK_WINDOW_SUB = int(np.divide(SAMP_FREQ,CUTOFF_SUB)) #peak to peak within a sufficiently small sample window

#constants for writing WAVs
NEAREST_AZ_WAV = 15
MIN_ELEV_WAV=0
MAX_ELEV_WAV=0
ELEV_OFFSET_WAV = np.abs(MIN_ELEV_WAV)
OUTPUT_AZIMS_WAV = int(360/NEAREST_AZ_WAV)

#Strings
HRTF_LIST = ['Neumann KU 100', 'KEMAR Large Pinna', 'KEMAR Normal Pinna', 'B&K Type 4128 HATS', 'DADEC', 'HEAD acoustics HMSII.2', 'KEMAR', 'B&K Type 4128C HATS']
HRTF_LIST_NUM = ['01: Neumann KU 100', '02: KEMAR Large Pinna', '03: KEMAR Normal Pinna', '04: B&K Type 4128 HATS', '05: DADEC', '06: HEAD acoustics HMSII.2', '07: KEMAR', '08: B&K Type 4128C HATS']
HRTF_LIST_SHORT = ['KU_100', 'KEMAR_Large', 'KEMAR_Normal', 'B&K_4128', 'DADEC', 'HMSII.2', 'KEMAR', 'B&K_4128C']
HP_COMP_LIST = ['In-Ear Headphones','Over-Ear/On-Ear Headphones']
HP_COMP_LIST_SHORT = ['In-Ear','Over-Ear&On-Ear']

ROOM_TARGET_LIST = ['Flat','ASH Target','Harman Target','HATS Target','Toole Target','rtings Target']
ROOM_TARGET_LIST_SHORT = ['Flat','ASH_Target','Harman_Target','HATS_Target','Toole_Target','rtings_Target']
ROOM_TARGET_LIST_FIRS = ['Flat','ash_room_target_fir','harman_b_room_target_fir','hats_room_target_fir','toole_trained_room_target_fir','rtings_target_fir']

#BRIR writing
#number of directions to grab to built HESUVI IR
NUM_SOURCE_AZIM = 7
NUM_OUT_CHANNELS_HE = NUM_SOURCE_AZIM*2
NUM_OUT_CHANNELS_TS = 4


#Equalizer APO constants
AZIM_DICT = {'WIDE_BL':'-135','WIDE_BR':'135','NARROW_BL':'-150','NARROW_BR':'150','WIDEST_BL':'-120','WIDEST_BR':'120','SL':'-90','SR':'90','FL':'-30','FR':'30','FC':'0','WIDE_FL':'-35','WIDE_FR':'35','NARROW_FL':'-25','NARROW_FR':'25'}
CHANNEL_CONFIGS = [['2.0_Stereo','2.0','2.0 Stereo'],['2.0_Stereo_Narrow','2.0N','2.0 Stereo (narrow placement)'],['2.0_Stereo_Wide','2.0W','2.0 Stereo (wide placement)'],['7.1_Surround_Narrow_Back','7.1N','7.1 surround (narrow back placement)'],['7.1_Surround_Wide_Back','7.1W','7.1 surround (wide back placement)'],['5.1_Surround','5.1','5.1 surround']]
NUM_SPEAK_CONFIGS = len(CHANNEL_CONFIGS)

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