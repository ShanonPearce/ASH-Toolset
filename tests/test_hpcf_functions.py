# -*- coding: utf-8 -*-


"""
Test routine of ASH-Tools.

Created on Sun Apr 28 11:25:06 2024
@author: Shanon Pearce
License: (see /LICENSE)
"""

import logging.config
from os.path import join as pjoin
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

primary_path = 'C:\Program Files\EqualizerAPO'
hesuvi_path = 'C:\Program Files\EqualizerAPO\config\HeSuVi'




  





"""
#Run HpCF functions here
"""

database = pjoin(CN.DATA_DIR_OUTPUT,'hpcf_database.db')
# create a database connection
conn = hpcf_functions.create_connection(database)

#add hpcfs into database
hpcf_functions.hpcf_wavs_to_database(conn)

#TEST: get list of brands
brands_list = hpcf_functions.get_brand_list(conn)

#TEST: get list of headphones
headphones_list = hpcf_functions.get_all_headphone_list(conn)

#TEST: get list of headphones for a brand
brand = 'Beyerdynamic'
hp_list_specific = hpcf_functions.get_headphone_list(conn, brand)

#TEST: get brand for a specific headphone
headphone = '1MORE Quad Driver'
brand_b = hpcf_functions.get_brand(conn, headphone)

#TEST: get all samples for a headphone
headphone = '1MORE Quad Driver'
sample_list_a = hpcf_functions.get_hpcf_samples(conn, headphone)

#TEST: get all samples for a headphone
headphone = '1MORE Quad Driver'
sample_list_b = hpcf_functions.get_hpcf_samples_dicts(conn, headphone)
for s in sample_list_b:
    sample_dict = dict(s)

#TEST: get hpcf for a specific headphone and sample
headphone = '1MORE Quad Driver'
sample = 'Sample B'
sample_list_c = hpcf_functions.get_hpcf_headphone_sample_dict(conn, headphone, sample)
sample_dict_b = dict(sample_list_c)

#TEST: get sample id from name
sample = 'Sample D'
sample_id = hpcf_functions.hpcf_sample_to_id(sample)

#TEST: get sample name from id 
sample_name = hpcf_functions.hpcf_sample_id_to_name(sample_id)



#TEST: export hpcfs
headphone = '1MORE Quad Driver'
hpcf_functions.hpcf_to_file_bulk(conn, primary_path, headphone, hesuvi_path, hesuvi_export=1)

#TEST: export plot
headphone = '1MORE Quad Driver'
sample = 'Sample B'
hpcf_functions.hpcf_to_plot(conn, headphone, sample, primary_path, save_to_file=0)




#TEST: generate averages
hpcf_functions.hpcf_generate_averages(conn)


#TEST: delete hpcf
headphone = '1MORE Triple Driver'
hpcf_functions.delete_headphone(conn, headphone)

headphone = '1MORE Quad Driver'
sample = 'Sample B'
hpcf_functions.delete_headphone_sample(conn, headphone, sample)

#TEST: rename hpcf

headphone_old = 'AKG K400'
headphone_new = 'AKG K401'
hpcf_functions.rename_hpcf_headphone(conn, headphone_old, headphone_new)

headphone_old = 'AKG K420'
sample_old = 'Sample A'
headphone_new = 'AKG K44'
hpcf_functions.rename_hpcf_headphone_and_sample(conn, headphone_old, sample_old, headphone_new)

headphone_old = 'AKG K420'
sample_old = 'Sample A'
headphone_new = 'AKG K44'
hpcf_functions.rename_hpcf_headphone_and_sample(conn, headphone_old, sample_old, headphone_new)


hpcf_functions.renumber_headphone_samples(conn)




#TEST: Generate HpCF script

measurement_folder_name = 'TEST_PAYLOAD'

#generate summary
hpcf_functions.generate_hp_summary_sheet(conn, measurement_folder_name)

#calculate hpcfs
hpcf_functions.calculate_new_hpcfs(conn, measurement_folder_name, in_ear_set = 0)

#TEST: generate averages
hpcf_functions.hpcf_generate_averages(conn)

headphone = 'Creative Zen Hybrid Pro'
hpcf_functions.hpcf_to_file_bulk(conn, primary_path, headphone, hesuvi_path, hesuvi_export=1)
    


#finally close the connection
conn.close()









logging.info('Finished') 

