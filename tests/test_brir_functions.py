# -*- coding: utf-8 -*-


"""
Test routine of ASH-Tools.

Created on Sun Apr 28 11:25:06 2024
@author: Shanon Pearce
License: (see /LICENSE)
"""


import logging.config
from os.path import join as pjoin
from ash_tools import brir_generation
from ash_tools import brir_export
from ash_tools import e_apo_config_creation
from ash_tools import constants as CN
from ash_tools import hpcf_functions

#logging
logging.basicConfig(
handlers=[
    logging.StreamHandler(),
    logging.FileHandler("log.log", mode="w"),
],
level=logging.DEBUG,
format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info('Started')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

#
#program code
#

#
#set variables for testing
#
hrtf_type=1#4
direct_gain_db=-3#0
room_target = 1
apply_pinna_comp = 1
target_rt60 = 750#400
primary_path = 'C:\Program Files\EqualizerAPO'
hesuvi_path = 'C:\Program Files\EqualizerAPO\config\HeSuVi'





"""
#Run BRIR reverb synthesis
"""
brir_generation.generate_reverberant_brir()

"""
#Run BRIR integration
"""

brir_gen = brir_generation.generate_integrated_brir(hrtf_type, direct_gain_db, room_target, apply_pinna_comp, target_rt60)

"""
#Run BRIR export
"""
#calculate name
brir_name = CN.HRTF_LIST_SHORT[hrtf_type-1] +'_'+ str(target_rt60) + 'ms_' + str(direct_gain_db) + 'dB_' + CN.ROOM_TARGET_LIST_SHORT[room_target] + '_' + CN.HP_COMP_LIST_SHORT[apply_pinna_comp]
brir_export.export_brir(brir_arr=brir_gen, hrtf_type=hrtf_type, target_rt60=target_rt60, brir_name=brir_name, primary_path=primary_path, hesuvi_path=hesuvi_path)

"""
#Run E-APO Config creator for BRIR convolution
"""
e_apo_config_creation.write_e_apo_configs_brirs(brir_name=brir_name, primary_path=primary_path, hrtf_type=hrtf_type)















logging.info('Finished') 

