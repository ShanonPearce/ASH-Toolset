# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:55:01 2024

@author: Shanon
"""

# import packages
from os.path import join as pjoin
import os
from pathlib import Path
import time
import logging
from datetime import date
from datetime import datetime
from ash_toolset import constants as CN
import sqlite3
from sqlite3 import Error
from scipy.io import wavfile
import json
from csv import DictReader
import numpy as np
import pyfar as pf
from scipy.signal import find_peaks
from ash_toolset import helper_functions as hf
import csv
import string
from ash_toolset import e_apo_config_creation
from scipy.interpolate import CubicSpline
import dearpygui.dearpygui as dpg
import gdown
import urllib.request
import shutil

today = str(date.today())


logger = logging.getLogger(__name__)
log_info=1




def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """

    conn = None
    try:
        conn = sqlite3.connect(db_file, check_same_thread=False)
        return conn
    except Error as e:
        logging.error("Error occurred", exc_info = e)

    return conn


def create_table(conn, table_name, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return: None
    """
    
    try:
        c = conn.cursor()
        
        #drop table if already exists
        drop_table_sql = "drop table if exists " + table_name
        c.execute(drop_table_sql)
        
        #create new table
        c.execute(create_table_sql)
        
        c.close()
    except Error as e:
        logging.error("Error occurred", exc_info = e)


def create_hpcf_row(conn, hpcf_data, brand, headphone, sample, gui_logger=None):
    """
    Create a new hpcf row
    :param conn: Connection object
    :param hpcf_data: tuple containing brand,headphone,sample,sample_id,fir,graphic_eq,graphic_eq_31,graphic_eq_103,created_on
    :param brand: string, headphone brand
    :param headphone: string, name of headphone
    :param sample: string, name of sample
    :param gui_logger: gui logger object for dearpygui
    :return: None
    """
    
    try:
        sql = ''' INSERT INTO hpcf_table(brand,headphone,sample,sample_id,fir,graphic_eq,graphic_eq_31,graphic_eq_103,created_on)
                  VALUES(?,?,?,?,?,?,?,?,?) '''
        cur = conn.cursor()
        cur.execute(sql, hpcf_data)
        #conn.commit()
        
        #log results
        log_string = 'HpCF inserted into database.' + ' Brand: ' + str(brand) + ' Headphone: ' + str(headphone) + ' Sample: ' + str(sample)
        hf.log_with_timestamp(log_string, gui_logger)
        
        
        cur.close()
    
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None   


def get_all_headphone_list(conn):
    """
    Function retrieves list of headphones from database
    """
    try:
        sql = 'select DISTINCT headphone from hpcf_table'
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        if rows:
            rows_list = [i[0] for i in rows]#convert from list of tuples to list
            return sorted(rows_list, key=str.casefold) #sort list before returning
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None     


def get_brand_list(conn):
    """
    Function retrieves list of brands from database
    """
    try:
        sql = 'select DISTINCT brand from hpcf_table'
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        if rows:
            rows_list = [i[0] for i in rows]#convert from list of tuples to list
            return sorted(rows_list, key=str.casefold) #sort list before returning
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None     
    
def search_brand_list(conn,search_str=None):
    """
    Function retrieves list of brands from database based on search string
    """
    try:
        if search_str == None or not search_str:
            return []
        else:
            brand_tuple = ('%'+search_str+'%',)
            sql = "select DISTINCT brand from hpcf_table where brand like ?"
            cur = conn.cursor()
            cur.execute(sql, brand_tuple)
 
            rows = cur.fetchall()
            cur.close()
            if rows:
                rows_list = [i[0] for i in rows]#convert from list of tuples to list
                return sorted(rows_list, key=str.casefold) #sort list before returning
            
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None     
 
def search_headphone_list(conn,search_str=None):
    """
    Function retrieves list of headphones from database based on search string
    """
    try:

        if search_str == None or not search_str:
            return []
        else:
            brand_tuple = ('%'+search_str+'%',)
            sql = "select DISTINCT headphone from hpcf_table where headphone like ?"
            cur = conn.cursor()
            cur.execute(sql, brand_tuple)
 
            rows = cur.fetchall()
            cur.close()
            if rows:
                rows_list = [i[0] for i in rows]#convert from list of tuples to list
                return sorted(rows_list, key=str.casefold) #sort list before returning
 
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None     
        
def search_headphones_in_list(conn,search_list=None):
    """
    Function retrieves list of headphones from database based on list of search strings
    """
    try:

        if search_list == None or not search_list:
            return []
        else:
            sql = f"select DISTINCT headphone from hpcf_table where headphone in ({','.join(['?']*len(search_list))})"
            cur = conn.cursor()
            cur.execute(sql, search_list)
 
            rows = cur.fetchall()
            cur.close()
            if rows:
                rows_list = [i[0] for i in rows]#convert from list of tuples to list
                return sorted(rows_list, key=str.casefold) #sort list before returning
 
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None            

def get_headphone_list(conn, brand):
    """
    Function retrieves list of headphones from database for a specified brand
    """ 
    try:
        brand_tuple = (brand,)
        sql = 'select DISTINCT headphone from hpcf_table where brand =?'
        cur = conn.cursor()
        cur.execute(sql, brand_tuple)
        rows = cur.fetchall()
        cur.close()
        if rows:
            rows_list = [i[0] for i in rows]#convert from list of tuples to list
            return sorted(rows_list, key=str.casefold) #sort list before returning
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None     
    

def get_brand(conn, headphone):
    """
    Function retrieves brand from database for a specified headphone
    """
    try:
        headphone_tuple = (headphone,)
        sql = 'select DISTINCT brand from hpcf_table where headphone =?'
        cur = conn.cursor()
        cur.execute(sql, headphone_tuple)
        rows = cur.fetchall()
        cur.close()
        if rows:
            rows_string = rows[0][0]#convert from tuple to string
            return rows_string
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None     
    

def get_samples_list(conn, headphone):
    """
    Function retrieves samples from database for a specified headphone
    """
    try:
        headphone_tuple = (headphone,)
        sql = 'select DISTINCT sample from hpcf_table where headphone =?'
        cur = conn.cursor()
        cur.execute(sql, headphone_tuple)
        rows = cur.fetchall()
        cur.close()
        if rows:
            rows_list = [i[0] for i in rows]#convert from list of tuples to list
            return sorted(rows_list, key=str.casefold)
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None     
    


def get_hpcf_samples(conn, headphone):
    """
    Function retrieves filter data (all samples) from database for a specified headphone 
    """  
    try:
        headphone_tuple = (headphone,)
        sql = 'select brand,headphone,sample,sample_id,fir,graphic_eq,graphic_eq_31,graphic_eq_103,created_on from hpcf_table where headphone =?'
        cur = conn.cursor()
        cur.execute(sql, headphone_tuple)
        rows = cur.fetchall()
        cur.close()
        if rows:
            return rows
        else:
            return None
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None     
       

def get_hpcf_samples_dicts(conn, headphone):
    """
    Function retrieves filter data (all samples) for a specified headphone from database 
    Returns list of dict like objects
    use conn.row_factory = sqlite3.Row, but the results are not directly dictionaries. One has to add an additional "cast" to dict
    """
        
    try:
        headphone_tuple = (headphone,)
        sql = 'select brand,headphone,sample,sample_id,fir,graphic_eq,graphic_eq_31,graphic_eq_103,created_on from hpcf_table where headphone =?'
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql, headphone_tuple)
        rows = cur.fetchall()
        cur.close()
        if rows:
            return rows
        else:
            return None
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None     
        


def get_hpcf_sample(conn, hpcf_id):
    """
    Function retrieves filter data from database for a specified hpcf
    """ 
    try:
        headphone_tuple = (hpcf_id,)
        sql = 'select brand,headphone,sample,sample_id,fir,graphic_eq,graphic_eq_31,graphic_eq_103,created_on from hpcf_table where id =?'
        cur = conn.cursor()
        cur.execute(sql, headphone_tuple)
        rows = cur.fetchall()
        cur.close()
        if rows:
            return rows
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None     
    
    

    
def get_hpcf_headphone_sample_dict(conn, headphone, sample):
    """
    Function retrieves filter data for a specified headphone and sample from database 
    Returns dict like object
    use conn.row_factory = sqlite3.Row, but the results are not directly dictionaries. One has to add an additional "cast" to dict
    """
    try:
        headphone_tuple = (headphone, sample)
        sql = 'select brand,headphone,sample,sample_id,fir,graphic_eq,graphic_eq_31,graphic_eq_103,created_on from hpcf_table where headphone =? AND sample =?'
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql, headphone_tuple)
        rows = cur.fetchall()
        cur.close()
        if rows:
            return rows[0]
        else:
            return None
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None      
    
def get_hpcf_date_range_dicts(conn, date_str='2024-06-21'):
    """
    Function retrieves filter data for a specified headphone and sample from database 
    Returns dict like object
    use conn.row_factory = sqlite3.Row, but the results are not directly dictionaries. One has to add an additional "cast" to dict
    """
    try:
        headphone_tuple = (date_str,)
        sql = 'select brand,headphone,sample,created_on from hpcf_table where date(created_on) >= date(?)'
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql, headphone_tuple)
        rows = cur.fetchall()
        cur.close()
        if rows:
            return rows
        else:
            return None
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None      
    

def replace_hpcf_filter_data(conn, hpcf_data, headphone, sample, gui_logger=None):
    """
    Function replaces filter data for a particular hpcf
    :param conn: Connection object
    :param hpcf_data: tuple containing parameters for sql query in same order as query
    :param headphone: string, name of headphone
    :param sample: string, name of sample
    :param gui_logger: gui logger object for dearpygui
    :return: None
    """
    
    try:
        #create tuple with all params
        headphone_data_l = list(hpcf_data)
        headphone_data_l.append(headphone)
        headphone_data_l.append(sample)
        headphone_data_t = tuple(headphone_data_l)
        
        sql = 'UPDATE hpcf_table SET fir = ?, graphic_eq = ?, graphic_eq_31 = ?, graphic_eq_103 = ?, created_on = ? WHERE headphone = ? AND sample = ?'
        cur = conn.cursor()
        cur.execute(sql, headphone_data_t)
        #conn.commit()
        cur.close()
        
        #log results
        log_string = 'HpCF Updated in database. Headphone: ' + str(headphone) + ' Sample: ' + str(sample)
        hf.log_with_timestamp(log_string, gui_logger)
            
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None      
    

   
def delete_headphone(conn, headphone, gui_logger=None):
    """
    Function removes rows in database for a specified headphone
    """ 
    
    try:
        cur = conn.cursor()
        headphone_data = (headphone,)
        delete_stmt = 'DELETE FROM hpcf_table WHERE headphone = ?'
        cur.execute(delete_stmt, headphone_data)
        conn.commit()
        cur.close()
        
        #log results
        log_string = 'HpCFs deleted from database for: ' + str(headphone)
        hf.log_with_timestamp(log_string, gui_logger)
        
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None      

    

  
def delete_headphone_sample(conn, headphone, sample, gui_logger=None):
    """
    Function removes row in database for a specified headphone and sample
    """  
    
    try:
        cur = conn.cursor()
        headphone_data = (headphone,sample)
        delete_stmt = 'DELETE FROM hpcf_table WHERE headphone = ? AND sample = ?'
        cur.execute(delete_stmt, headphone_data)
        conn.commit()
        cur.close()
        
        #log results
        log_string = 'HpCFs deleted from database for: ' + str(headphone) + ' ' + str(sample)
        hf.log_with_timestamp(log_string, gui_logger)
        
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None   

    



def rename_hpcf_sample_field(conn, headphone, sample_id_old, sample_id_new_min=1, fill_gaps=0, gui_logger=None):
    """
    Function renames sample field and updates sample id in the database for a specified headphone and sample id
    use this function to modify sample name/id for a sample within the same headphone
    :param conn: Connection object
    :param headphone: string, headphone name
    :param sample_old: string, sample name to be renamed
    :param sample_id_new_min: string, smallest sample id desired
    :param fill_gaps: int, set to 1 for fill gaps mode. This will search for next available gap and set sample to that gap if lower than current sample
    :return: None
    """    
    
    try:
        
        
        #get samples for specified headphone
        hpcfs_headphone = get_hpcf_samples_dicts(conn, headphone)
        
        #find gaps
        if fill_gaps == 1:
            smallest_gap_id = 99#start with large number
            previous_sample_id=0
            current_sample_id=0
            
            #find smallest sample id of gaps (find first gap)
            for s in hpcfs_headphone:
                sample_dict = dict(s)
                previous_sample_id = current_sample_id
                current_sample_id = sample_dict['sample_id'] 
                if current_sample_id-previous_sample_id>1:
                    smallest_gap_id=previous_sample_id+1
                    break
            #set new sample id to the first gap if found and lower id than current sample
            if smallest_gap_id < sample_id_old:
                sample_id_new=smallest_gap_id
            else:
                sample_id_new=sample_id_old
        else:
            #find largest ID
            largest_id = 0
            for s in hpcfs_headphone:
                sample_dict = dict(s)
                sample_id = sample_dict['sample_id']
                if sample_id > largest_id:
                    largest_id = sample_id
                    
            #check for cases where samples already exist from desired ID onwards
            if sample_id_new_min <= largest_id:
                #keep original id if sample already exists within acceptable range
                if sample_id_old >= sample_id_new_min:
                    sample_id_new = sample_id_old
                #otherwise increment largest to give new sample ID 
                else:
                    sample_id_new = largest_id+1
            else:
                sample_id_new = max(sample_id_new_min,largest_id+1)
            
        sample_old = hpcf_sample_id_to_name(sample_id_old) 
        sample_new = hpcf_sample_id_to_name(sample_id_new) 
        
        #only update if new sample ID was calculated
        if sample_id_old != sample_id_new:
            #then rename headphone field for all HPCFs belonging to headphone_old to headphone_new
            headphone_data = (sample_new,sample_id_new,headphone,sample_id_old)
            sql = 'UPDATE hpcf_table SET sample = ?, sample_id = ? WHERE headphone = ? AND sample_id = ?'
            cur = conn.cursor()
            cur.execute(sql, headphone_data)
            conn.commit()
            cur.close()
        
        #log results
        if CN.LOG_INFO == True and sample_id_old != sample_id_new:
            log_string = 'Sample name and ID updated in database. Headphone: ' + str(headphone) + ' sample: ' + str(sample_old) + ' sample id: ' + str(sample_id_old) + ' renamed to: ' + str(sample_new) + ' new sample id: ' + str(sample_id_new) 
            hf.log_with_timestamp(log_string, gui_logger)
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None      



def rename_hpcf_sample_name(conn, headphone, sample_name_old, sample_name_new, gui_logger=None):
    """
    Function renames sample field and updates sample id in the database for a specified headphone and sample name
    :param conn: Connection object
    :param headphone: string, headphone name
    :param sample_name_old: string, sample name to be renamed
    :param sample_name_new: string, new sample name
    :return: None
    """    
    
    try:

        #only update if new sample ID was calculated
        if sample_name_old != sample_name_new and sample_name_new != None:
            
            sample_id_new = hpcf_sample_to_id(sample_name_new)
            #then rename headphone field for all HPCFs belonging to headphone_old to headphone_new
            headphone_data = (sample_name_new,sample_id_new,headphone,sample_name_old)
            sql = 'UPDATE hpcf_table SET sample = ?, sample_id = ? WHERE headphone = ? AND sample = ?'
            cur = conn.cursor()
            cur.execute(sql, headphone_data)
            conn.commit()
            cur.close()
        
            #log results
            if CN.LOG_INFO == True:
                log_string = 'Sample name and ID updated in database. Headphone: ' + str(headphone) + ' sample: ' + str(sample_name_old)  + ' renamed to: ' + str(sample_name_new) + ' new sample id: ' + str(sample_id_new) 
                hf.log_with_timestamp(log_string, gui_logger)
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None      


   
def rename_hpcf_sample_name_bulk(conn, sample_name_old, sample_name_new, gui_logger=None):
    """
    Function renames sample field and updates sample id in the database for a all headphones with specified sample name
    :param conn: Connection object
    :param sample_name_old: string, sample name to be renamed
    :param sample_name_new: string, new sample name
    :return: None
    """ 
    try:

        #only update if new sample ID was calculated
        if sample_name_old != sample_name_new and sample_name_new != None:
            
            sample_id_new = hpcf_sample_to_id(sample_name_new)
            #then rename headphone field for all HPCFs belonging to headphone_old to headphone_new
            headphone_data = (sample_name_new,sample_id_new,sample_name_old)
            sql = 'UPDATE hpcf_table SET sample = ?, sample_id = ? WHERE sample = ?'
            cur = conn.cursor()
            cur.execute(sql, headphone_data)
            conn.commit()
            cur.close()
        
            #log results
            if CN.LOG_INFO == True:
                log_string = 'Sample name and ID updated in database. Sample: ' + str(sample_name_old)  + ' renamed to: ' + str(sample_name_new) + ' new sample id: ' + str(sample_id_new) 
                hf.log_with_timestamp(log_string, gui_logger)
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None      


   
def rename_hpcf_headphone(conn, headphone_old, headphone_new, gui_logger=None):
    """
    Function renames headphone field in the database for a specified headphone. Also updates sample name and ID in cases where duplicates occur
    Use this function to move all hpcfs for one headphone to another headphone or a new headphone
    :param conn: Connection object
    :param headphone_old: string, name of headphone to be renamed
    :param headphone_new: string, new name for headphone
    :return: None
    """ 
    
    try:
        
        now_datetime = datetime.now()
        
        #only proceed if old headphone actually has data
        hpcfs_old_headphone = get_hpcf_samples_dicts(conn, headphone_old)
        if hpcfs_old_headphone != None:
        
            #first update sample names and IDs of HPCFs belonging to headphone_old to avoid duplicates when renaming headphone field
            
            #get samples for new headphone
            hpcfs_new_headphone = get_hpcf_samples_dicts(conn, headphone_new)
            #proceed if new headphone name already has samples
            if hpcfs_new_headphone != None:
                #find largest ID
                largest_id = 0
                for s in hpcfs_new_headphone:
                    sample_dict = dict(s)
                    sample_id = sample_dict['sample_id']
                    if sample_id > largest_id:
                        largest_id = sample_id
                #calculate new minimum sample ID        
                new_sample_id_min = 1 + largest_id
                
                #get samples for old headphone, calculate new sample name and id, and update in database
                hpcfs_old_headphone = get_hpcf_samples_dicts(conn, headphone_old)
                for s in hpcfs_old_headphone:
                    sample_dict = dict(s)
                    sample_id_old = sample_dict['sample_id']
                    if sample_id_old > 0:
                        #update sample name and id in database
                        rename_hpcf_sample_field(conn, headphone_old, sample_id_old, new_sample_id_min)
                        
            
            
            #then rename headphone field for all HPCFs belonging to headphone_old to headphone_new
            headphone_data = (headphone_new,now_datetime,headphone_old,'Average')
            sql = 'UPDATE hpcf_table SET headphone = ?, created_on = ? WHERE headphone = ? AND sample != ?'
            cur = conn.cursor()
            cur.execute(sql, headphone_data)
            conn.commit()
            cur.close()
            
            #log results
            log_string = 'HpCF Updated in database. Headphone: ' + str(headphone_old) + ' renamed to ' + str(headphone_new)
            hf.log_with_timestamp(log_string, gui_logger)
                
            #finally remove any remaining hpcfs for the old headphone (e.g. old averages)
            delete_headphone(conn, headphone_old)
        else:
            #log results
            if CN.LOG_INFO == True:
                logging.info('Headphone ' + headphone_old +' not found')
        
        
        
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None      

    
    

def rename_hpcf_headphone_and_sample(conn, headphone_old, sample_old, headphone_new, gui_logger=None):
    """
    Function renames headphone field and updates sample and sample ID field for a specified headphone and sample in the database 
    Use this function to move a specific hpcf to a different headphone
    :param conn: Connection object
    :param headphone_old: string, name of headphone to be renamed
    :param sample_old: string, name of selected sample to be renamed
    :param headphone_new: string, new name for headphone
    :return: None
    """
    
    try:
        
        now_datetime = datetime.now()
        
        #only proceed if old headphone actually has data
        hpcf_old_headphone = get_hpcf_headphone_sample_dict(conn, headphone_old, sample_old)
        if hpcf_old_headphone != None:
            #get samples for new headphone
            hpcfs_new_headphone = get_hpcf_samples_dicts(conn, headphone_new)
            #proceed if new headphone name already has samples
            if hpcfs_new_headphone != None:
                #find largest ID
                largest_id = 0
                for s in hpcfs_new_headphone:
                    sample_dict = dict(s)
                    sample_id = sample_dict['sample_id']
                    if sample_id > largest_id:
                        largest_id = sample_id
                
                #calculate new sample ID
                sample_id_new = 1 + largest_id
                sample_new = hpcf_sample_id_to_name(sample_id_new) 
            else:
                largest_id=0
                sample_id_new = 1 + largest_id
                sample_new = hpcf_sample_id_to_name(sample_id_new) 
                
            #then rename headphone and sample field for all HPCFs belonging to headphone_old to headphone_new
            headphone_data = (headphone_new,sample_new,sample_id_new,now_datetime,headphone_old,sample_old)
            sql = 'UPDATE hpcf_table SET headphone = ?, sample = ?, sample_id = ?, created_on = ? WHERE headphone = ? AND sample = ?'
            cur = conn.cursor()
            cur.execute(sql, headphone_data)
            conn.commit()
            cur.close()
   
            #log results
            log_string = 'HpCF Updated in database. Headphone: ' + str(headphone_old) + ' renamed to ' + str(headphone_new) + ' and Sample: ' + str(sample_old) + ' renamed to ' + str(sample_new)
            hf.log_with_timestamp(log_string, gui_logger)
        else:
            #log results
            log_string = 'Headphone ' + headphone_old + ' sample ' + sample_old +' not found'
            hf.log_with_timestamp(log_string, gui_logger)
    
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info = e)
        return None      

    
  




  
def renumber_headphone_samples(conn, gui_logger=None):
    """
    Function sorts and renumbers all HPCF samples in the database
    """  
   
    try:
        #get list of all headphones in the DB
        headphone_list = get_all_headphone_list(conn)
        
        #for each headphone, grab all samples
        for h in headphone_list:

            sample_list = get_hpcf_samples_dicts(conn, h)
            
            #run rename hpcf sample field function for every sample
            for s in sample_list:
                sample_dict = dict(s)
                sample_id_current = sample_dict['sample_id']
                #only run if more than 1 sample for the headphone
                if sample_id_current > 1:
                    rename_hpcf_sample_field(conn, h, sample_id_current, sample_id_new_min=1, fill_gaps=1, gui_logger=gui_logger)

    except Error as e:
        logging.error("Error occurred", exc_info = e)


    #log results
    log_string = 'HpCF sample renumbering completed'
    hf.log_with_timestamp(log_string, gui_logger)




def hpcf_retrieve_metadata(csv_location, wav_name): 
    """
    Function reads hpcf_wav_summary and returns metadata for a particular wav file
    """
    
    try:
        with open(csv_location, 'r', encoding='utf-8-sig') as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                # check the arguments against the row
                if row['Wav Name'] == wav_name: 
                   return dict(row)
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        return None    
        

def hpcf_retrieve_set_geq_freqs(f_set=1):     
    """
    Function reads wavelet_geq_freqs and returns frequency list
    """
    
    try:
        #retrieve geq frequency list as an array
        if f_set == 1:
            csv_fname = pjoin(CN.DATA_DIR_RAW, 'wavelet_geq_freqs.csv')
        elif f_set == 2:
            csv_fname = pjoin(CN.DATA_DIR_RAW, '31_band_geq_freqs.csv')
        elif f_set == 3:
            csv_fname = pjoin(CN.DATA_DIR_RAW, 'hesuvi_geq_freqs.csv')    
        else:
            csv_fname = pjoin(CN.DATA_DIR_RAW, 'wavelet_geq_freqs.csv')
        geq_set_f = []
        with open(csv_fname, encoding='utf-8-sig', newline='') as inputfile:
            for row in csv.reader(inputfile):
                geq_set_f.append(int(row[0]))
        geq_set_f = np.array(geq_set_f)
        return geq_set_f
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        return None  
    

def hpcf_sample_to_id(sample):
    """
    Function takes sample name and returns index in alphabet
    """
    
    try:
        #averages will be considered as index 0
        if 'Average' in sample or 'average' in sample:
            return 0
        #get index of last character
        last_char = sample[-1]
        letter = last_char.lower()
        sample_id = list(string.ascii_lowercase).index(letter) + 1
        return sample_id

    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        

def hpcf_sample_id_to_name(sample_id):
    """
    Function takes sample id and returns sample name from alphabet
    """
    
    try:
        #averages will be considered as index 0
        if sample_id == 0:
            return 'Average'
        #get index of last character
        sample_name = 'Sample ' + chr(ord('@')+sample_id)
        return sample_name

    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        
        
        

    
        

def hpcf_fir_to_geq(fir_array, geq_mode=1, sample_rate=CN.SAMP_FREQ, geq_freq_arr = np.array([]), output_string_type = 2):
    """
    Function takes an FIR as an input and returns EQ filter. Supports Graphic EQ Full, Graphic EQ
    :param fir_array: numpy array 1d, contains time domain FIR filter at 44100Hz
    :param geq_mode: int, 1 = graphic eq 32 bands based on prominent peaks, 2 = graphic eq with variable bands (hesuvi, wavelet, 31 band etc.)
    :param sample_rate: int, sample frequency in Hz
    :param geq_freq_arr: numpy array, contains list of frequencies that form filter
    :param output_string_type: int, set output string type, 1 = graphic EQ string (directly E-APO compatible), 2 = JSON string (to be converted to above format when exporting to file)
    """

    try:
        #set variables
        geq_params = 32#32 params unless geq_mode is 2
        samp_freq=sample_rate
        sampling_ratio = samp_freq/(CN.N_FFT-1)
        
        #get list of freqs for Hesuvi geq from CSV if array not provided
        if geq_mode == 2:
            if geq_freq_arr.size == 0:
                geq_set_f = hpcf_retrieve_set_geq_freqs()
                #convert to freq bins
                geq_set_fb = np.round(geq_set_f/sampling_ratio)
            else:
                geq_set_f = geq_freq_arr
                #convert to freq bins
                geq_set_fb = np.round(geq_set_f/sampling_ratio)
            #num frequencies: 103 for hesuvi, 127 for wavelet, 31 for 31 band
            geq_params = geq_set_f.size 
            
        
        #arrays
        error =np.zeros(CN.N_UNIQUE_PTS)
        peaks_to_check = np.empty(CN.NUM_ITER, dtype=int)
        hpcf_gains_evo=np.zeros((CN.NUM_ITER,CN.PARAMS_PER_ITER))
        hpcf_qFactor_evo=np.zeros((CN.NUM_ITER,CN.PARAMS_PER_ITER))
        hpcf_centerFreq_evo=np.zeros((CN.NUM_ITER,CN.PARAMS_PER_ITER))
        hpcf_gains_geq=np.zeros(geq_params)
        hpcf_centerFreq_geq=np.zeros(geq_params)
        filter_evo = pf.signals.impulse(CN.N_FFT)
    
    
        
    
        #grab FIR from input param
        data = fir_array

        data_pad=np.zeros(65536)
        data_pad[0:(CN.HPCF_FIR_LENGTH-1)]=data[0:(CN.HPCF_FIR_LENGTH-1)]
        data_fft = np.fft.fft(data_pad)
        hpcf_target_fr = np.abs(data_fft)
        
        
        
        #normalise to 0db
        #hpcf_target_fr = np.divide(hpcf_target_fr,hpcf_target_fr.max())
        #normalise to 0db in low freqs
        hpcf_target_fr = np.divide(hpcf_target_fr,np.mean(hpcf_target_fr[0:200]))
        if CN.PLOT_ENABLE == True:
            hf.plot_data(hpcf_target_fr,'HpCF plot',CN.N_FFT,samp_freq)
        target_filter = hpcf_target_fr
        #truncate
        target_filter = target_filter[0:CN.N_UNIQUE_PTS]
        #handle divide by zero
        target_filter  = np.where(target_filter > 1.0e-10, target_filter, 1.0e-10)
        target_filter_db=hf.mag2db(target_filter)
        
        hpcf_gains_geq=np.zeros(geq_params)
        hpcf_centerFreq_geq=np.zeros(geq_params)
        
        
        for m in range(CN.NUM_ITER):
        
            #set current target
            if m == 0:
                current_target_filter = target_filter
            else:
                current_target_filter = error    
                    
            #use error evo as targets for future iterations, generate new params based on previous error
            
        
            #
            #find local minima and local maxima
            #
            
            # for local maxima
            local_maxima_fb, local_maxima_details = find_peaks(current_target_filter,prominence=0.0001)
            local_maxima_prom = local_maxima_details["prominences"]
            # for local minima
            local_minima_fb, local_minima_details = find_peaks(-current_target_filter,prominence=0.0001)
            local_minima_prom = local_minima_details["prominences"]
            # combine minima and maxima then form 2d vector with prominance values for ranking
            local_min_max_fb_com = np.concatenate((local_maxima_fb,local_minima_fb))
            local_min_max_prom_com = np.concatenate((local_maxima_prom,local_minima_prom))
            local_min_max_fb_p = np.vstack((local_min_max_fb_com, local_min_max_prom_com)).T
            #filter only significant prom values 
            local_min_max_fb_p = local_min_max_fb_p[np.all(local_min_max_fb_p >CN.SENSITIVITY, axis = 1),:]
            #create copy for geq
            local_min_max_fb_p_g = local_min_max_fb_p.copy()
            # if in first x iterations, filter only below y kHz
            if m <= 5:
                local_min_max_fb_p = local_min_max_fb_p[np.all(local_min_max_fb_p <CN.FREQ_CUTOFF, axis = 1),:]
                local_min_max_fb_p_g = local_min_max_fb_p_g[np.all(local_min_max_fb_p_g <CN.FREQ_CUTOFF_GEQ, axis = 1),:]
            # sort by largest prom values
            Index = 1
            local_min_max_fb_p_sort_p = np.flip(local_min_max_fb_p[local_min_max_fb_p[:,Index].argsort()],0)
            #do same for geq
            local_min_max_fb_p_sort_p_g = np.flip(local_min_max_fb_p_g[local_min_max_fb_p_g[:,Index].argsort()],0)
            total_peaks_geq = local_min_max_fb_p_sort_p_g.shape[0]
            
            # if first iteration, include 1 low freq and 1 high freq in the peaks
            if m == 0:
                try:
                    #update array used for peq
                    local_min_max_fb_p_sort_p[CN.PARAMS_PER_ITER-2,0]=30
                    local_min_max_fb_p_sort_p[CN.PARAMS_PER_ITER-2,1]=0.1
                    local_min_max_fb_p_sort_p[CN.PARAMS_PER_ITER-1,0]=31000
                    local_min_max_fb_p_sort_p[CN.PARAMS_PER_ITER-1,1]=0.1
                    
                    #update array used for geq params from peaks
                    new_entries = np.array([30,0.1])
                    new_entries_mat = new_entries[np.newaxis, :]
                    local_min_max_fb_p_sort_p_g = np.append(local_min_max_fb_p_sort_p_g,new_entries_mat,0)
        
                    total_peaks_geq_mod = total_peaks_geq+1;#add 1 since we manually added 1 more peaks
                except:
                    pass
        
        
            #convert freq bin to freqs;
            local_min_max_fr_p_sort_p = local_min_max_fb_p_sort_p.copy()
            local_min_max_fr_p_sort_p[:,0] = local_min_max_fr_p_sort_p[:,0]*sampling_ratio
            #create copy for geq
            local_min_max_fr_p_sort_p_g = local_min_max_fb_p_sort_p_g.copy()
            local_min_max_fr_p_sort_p_g[:,0] = local_min_max_fr_p_sort_p_g[:,0]*sampling_ratio
            
            # sort smallest to largest - by freq
            Index = 0
            local_min_max_fr_p_sort_f = local_min_max_fr_p_sort_p[local_min_max_fr_p_sort_p[:,Index].argsort()]
            
            
            
            # if first iteration, grab geq params
            if m == 0:
                if geq_mode == 1:#grab params from peaks
                    for n in range(total_peaks_geq_mod):
                        try:
                            #gain is as per target filter but converted to db
                            mag = current_target_filter[int(local_min_max_fb_p_sort_p_g[n,0])]
                            #handle divide by zero cases
                            mag  = mag if mag > 1.0e-5 else 1.0e-5
                            gain = hf.mag2db(mag)
                            #center freq
                            cf = local_min_max_fr_p_sort_p_g[n,0]
                            #store params in array
                            hpcf_gains_geq[n] = gain
                            hpcf_centerFreq_geq[n] = cf
                        except:
                            pass
                elif geq_mode == 2:#grab params from set freq list
                     for n in range(geq_params):
                         #gain is as per target filter but converted to db
                         mag = current_target_filter[int(geq_set_fb[n])]
                         #handle divide by zero cases
                         mag  = mag if mag > 1.0e-5 else 1.0e-5
                         gain = hf.mag2db(mag)
                         #center freq
                         cf = geq_set_f[n]
                         #store params in array
                         hpcf_gains_geq[n] = gain
                         hpcf_centerFreq_geq[n] = cf  
                
            total_peaks = local_min_max_fr_p_sort_p.shape[0]
            peaks_to_check[m] = np.minimum(total_peaks,CN.PARAMS_PER_ITER)
            
            #
            #loop through minima/maxima, calculate params, create peq using above params, filter signal progressively
            #
            for n in range(peaks_to_check[m]):
                
                #calculate Q factor, bandwidth, gains for each freq
                #gain is as per target filter but converted to db
                mag = current_target_filter[int(local_min_max_fb_p_sort_p[n,0])]
                #handle divide by zero cases
                mag  = mag if mag > 1.0e-5 else 1.0e-5
                gain = hf.mag2db(mag)*0.80#0.75,0.80
                
                
                #center freq
                cf = local_min_max_fr_p_sort_p[n,0]
                #find index in array sorted by freq
                i = np.where(local_min_max_fr_p_sort_f[:,0] == local_min_max_fr_p_sort_p[n,0])[0][0]
                #then calc bandwidth
                if i == total_peaks-1:
                    bw = np.abs(local_min_max_fr_p_sort_f[i,0]-local_min_max_fr_p_sort_f[i-1,0])
                else:
                    bw = np.abs(local_min_max_fr_p_sort_f[i,0]-local_min_max_fr_p_sort_f[i+1,0])
                #bw=bw[0]#no longer required
                #convert to qFactor
               
                qf = np.divide(cf,bw)
        
                #build parametric eq using above parameters
                #filter progressively
                filter_evo = pf.dsp.filter.bell(filter_evo,cf,gain,qf)
                
                #store params in array
                hpcf_gains_evo[m,n] = gain*CN.EAPO_GAIN_ADJUST#adjust for E-APO?
                hpcf_qFactor_evo[m,n] = qf*CN.EAPO_QF_ADJUST#adjust for E-APO?
                hpcf_centerFreq_evo[m,n] = cf
        
            #calculate error
            #convert from cmplx to mag
            filter_evo_mag = np.abs(filter_evo.freq).transpose()
            filter_evo_mag = np.squeeze(filter_evo_mag)
            filter_evo_db=hf.mag2db(filter_evo_mag)
            diff_db = np.subtract(target_filter_db,filter_evo_db)
            error = hf.db2mag(diff_db)
            
            if CN.PLOT_ENABLE == True:
                hf.plot_data(error,'error',CN.N_FFT,samp_freq)
                hf.plot_data(filter_evo_mag,'filter_evo_mag',CN.N_FFT,samp_freq)
        
        if CN.PLOT_ENABLE == True:
            pf.plot.freq(filter_evo)
        
        
         
        #graphic eq
        if geq_mode == 1:
            total_peaks_geq_exp = np.minimum(geq_params,total_peaks_geq_mod)
        elif geq_mode == 2:
            total_peaks_geq_exp = geq_params
        # sort smallest to largest - by freq
        hpcf_centerFreq_geq_sort_f = hpcf_centerFreq_geq[hpcf_centerFreq_geq[0:total_peaks_geq_exp].argsort()]
        hpcf_gains_geq_sort_f = hpcf_gains_geq[hpcf_centerFreq_geq[0:total_peaks_geq_exp].argsort()]
        
        
        
        out_string = ''
        #1 = graphic EQ string (directly E-APO compatible)
        if output_string_type == 1:
            #store as a string
            geq_string = 'GraphicEQ: '
            for m in range(total_peaks_geq_exp):
                    geq_string =geq_string + str(round(hpcf_centerFreq_geq_sort_f[m])) + ' ' + str(round(hpcf_gains_geq_sort_f[m],1))
                    if m < total_peaks_geq_exp-1:
                        geq_string = geq_string + '; '
            out_string = geq_string
        #2 = JSON string (to be converted to above format when exporting to file)    
        elif output_string_type == 2:
            dict_keys = np.round(hpcf_centerFreq_geq_sort_f[0:total_peaks_geq_exp])
            dict_values = np.round(hpcf_gains_geq_sort_f[0:total_peaks_geq_exp],1)
            dictionary = dict(zip(dict_keys, dict_values))
            json_string = json.dumps(dictionary)
            out_string = json_string
        #invalid  
        else:
             logging.error("invalid string type selected for GEQ") 
        
        
        return out_string

    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        return None  



def hpcf_wavs_to_database(conn, gui_logger=None):
    """
    Function reads HpCF WAV dataset and populates a new database
    """
    
    # get the start time
    st = time.time()
    
    
    
    try:
        
        current_list = get_all_headphone_list(conn)
        if current_list == None:
            raise ValueError('Unable to create new database table: hpcf_table. Table already exists. Delete existing table or database before proceeding.')
    
        #retrieve geq frequency list as an array - 127 bands
        geq_set_f_127 = hpcf_retrieve_set_geq_freqs(f_set=1)
        #retrieve geq frequency list as an array - 31 bands
        geq_set_f_31 = hpcf_retrieve_set_geq_freqs(f_set=2)
        #retrieve geq frequency list as an array - 103 bands
        geq_set_f_103 = hpcf_retrieve_set_geq_freqs(f_set=3)
    
        now_datetime = datetime.now()
        # # create a database connection

        table_name = 'hpcf_table'
        sql_create_tasks_table = """CREATE TABLE IF NOT EXISTS hpcf_table (
                                    id integer PRIMARY KEY,
                                    brand text,
                                    headphone text,
                                    sample text,
                                    sample_id INT,
                                    fir text,
                                    graphic_eq text,
                                    graphic_eq_31 text,
                                    graphic_eq_103 text,
                                    created_on text
                                );"""
    

        # create tables
        if conn is not None:

            # create tasks table
            create_table(conn, table_name=table_name, create_table_sql=sql_create_tasks_table)
        else:
            logging.error("Error! cannot create the database connection.")
            return False
            
        #read hpcf WAVs one by one
        #iterate over files in the input directory
        
        hpcf_input_path = pjoin(CN.DATA_DIR_INT, 'hpcf_wavs')
        hpcf_summary_file_path = pjoin(CN.DATA_DIR_INT, 'hpcf_wav_summary.csv')
        
        for root, dirs, files in os.walk(hpcf_input_path):
            for filename in files:
                if '.wav' in filename:
                    wav_fname = pjoin(root, filename)
                    samplerate, data = wavfile.read(wav_fname)
                    fir_array = data / (2.**31)
                    #wav_length = data.shape[0]
                    #length_time = data.shape[0] / samplerate
        
        
                    #get metadata (brand, headphone name, sample name)
                    metadata_dict = hpcf_retrieve_metadata(csv_location=hpcf_summary_file_path, wav_name=filename)
                    brand_str = metadata_dict.get('Brand')
                    headphone_str = metadata_dict.get('Name')
                    sample_str = metadata_dict.get('Sample')
                    sample_id = hpcf_sample_to_id(sample_str)
                    
                    ### get fir
                    # convert from array to list
                    fir_list = fir_array.tolist()
                    # convert from list to Json
                    fir_json_str = json.dumps(fir_list)
                    

                    #get graphic eq filter 127 band
                    geq_str = hpcf_fir_to_geq(fir_array=fir_array,geq_mode=2,sample_rate=samplerate,geq_freq_arr=geq_set_f_127)
                    
                    #get graphic eq filter 31 band
                    geq_31_str = hpcf_fir_to_geq(fir_array=fir_array,geq_mode=2,sample_rate=samplerate,geq_freq_arr=geq_set_f_31)
                    
                    #obsolete
                    ##get graphic eq 32 band filter
                    #geq_32_str = hpcf_fir_to_geq(fir_array=fir_array,geq_mode=1,sample_rate=samplerate)
                    
                    #get graphic eq 103 band filter
                    geq_103_str = hpcf_fir_to_geq(fir_array=fir_array,geq_mode=2,sample_rate=samplerate,geq_freq_arr=geq_set_f_103)
                    
                    

                    #last modified text
                    created_on = now_datetime
     
                    #create hpcf tuple for this hpcf
                    # tasks
                    hpcf_to_insert = (brand_str, headphone_str, sample_str, sample_id, fir_json_str, geq_str, geq_31_str, geq_103_str, created_on)
                    
                    # create entry
                    create_hpcf_row(conn, hpcf_to_insert, brand_str, headphone_str, sample_str, gui_logger=gui_logger)
            
        conn.commit()
    
    
        log_string = 'Database created'
        hf.log_with_timestamp(log_string, gui_logger)
                
        
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
            
    # get the end time
    et = time.time()
    
    # get the execution time
    elapsed_time = et - st
    if CN.LOG_INFO == True:
        logging.info('Execution time:' + str(elapsed_time) + ' seconds')
        
        




def hpcf_to_file(hpcf_dict, primary_path, fir_export = True, fir_stereo_export = True, geq_export = True, geq_31_export = True, geq_103_export = False, hesuvi_export = True, geq_json = True, eapo_export=True, gui_logger=None, samp_freq=CN.SAMP_FREQ, bit_depth='PCM_24', force_output=False):
    """
    Function exports filter to a wav or txt file
    To be run on a hpcf dictionary, call once for every headphone sample. For a given hpcf, exports: FIR-Mono, FIR-Stereo, Graphic EQ full bands, Graphic EQ 31 Band, Graphic EQ 103 Band
    :param hpcf_dict: dictionary of hpcf data from database
    :param primary_path: string, output directory
    :param fir_export: int, 1 = export mono fir files
    :param fir_stereo_export: int, 1 = export stereo fir files
    :param geq_export: int, 1 = export graphic eq (full band) files
    :param geq_31_export: int, 1 = export graphic eq (31 band) files
    :param geq_103_export: int, 1 = export graphic eq (103 band) files
    :param hesuvi_export: int, 1 = export hesuvi files
    :param eapo_export: int, 1 = export equalizer apo config files for hpcf fir convolution
    :param geq_json: int, 1 = geq is a JSON string to be converted, 0 = dont convert geq string
    """
    
    try:


        #hesuvi path
        if 'EqualizerAPO' in primary_path:
            hesuvi_path = pjoin(primary_path,'HeSuVi')   
        else:
            hesuvi_path = pjoin(primary_path, CN.PROJECT_FOLDER,'HeSuVi')   
        
        #Directories
        brand_folder = hpcf_dict.get('brand')
        brand = brand_folder
        headphone = hpcf_dict.get('headphone')
        sample = hpcf_dict.get('sample')
        
        #output directories
        out_file_dir_wav = pjoin(primary_path, CN.PROJECT_FOLDER_HPCFS,'FIRs',brand_folder)
        out_file_dir_st_wav = pjoin(primary_path, CN.PROJECT_FOLDER_HPCFS,'FIRs_Stereo',brand_folder)
        out_file_dir_geq = pjoin(primary_path, CN.PROJECT_FOLDER_HPCFS,'Graphic_EQ_127_band',brand_folder)
        out_file_dir_geq_31 = pjoin(primary_path, CN.PROJECT_FOLDER_HPCFS,'Graphic_EQ_31_band',brand_folder)
        out_file_dir_geq_103 = pjoin(primary_path, CN.PROJECT_FOLDER_HPCFS,'Graphic_EQ_103_band',brand_folder)
        
        #full hpcf name
        hpcf_name = headphone + ' ' + sample
        hpcf_name_wav = headphone + ' ' + sample + '.wav'
        hpcf_name_wav = hpcf_name_wav.replace(" ", "_")
        hpcf_name_wav=hf.sanitize_filename(hpcf_name_wav)#sanitize file name in case of invalid windows characters
        hpcf_name_geq = headphone + ' ' + sample + '.txt'
        hpcf_name_geq = hpcf_name_geq.replace(" ", "_")
        hpcf_name_geq=hf.sanitize_filename(hpcf_name_geq)#sanitize file name in case of invalid windows characters
        
        #
        #save FIR to wav
        #
        hpcf_fir_json = hpcf_dict.get('fir')
        hpcf_fir_list = json.loads(hpcf_fir_json) 
        hpcf_fir = np.array(hpcf_fir_list)
        
        #resample if samp_freq is not 44100
        if samp_freq != CN.SAMP_FREQ:
            hpcf_fir = hf.resample_signal(hpcf_fir, new_rate = samp_freq)
        
        out_file_path = pjoin(out_file_dir_wav, hpcf_name_wav)
        
        if fir_export == True:
            #create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            #dont write if file already exists
            if force_output == True or not output_file.is_file():
                hf.write2wav(file_name=out_file_path, data=hpcf_fir, prevent_clipping=1, bit_depth=bit_depth, samplerate=samp_freq)
            
                log_string = 'HpCF (WAV FIR): ' + hpcf_name + ' saved to: ' + str(out_file_dir_wav)
                hf.log_with_timestamp(log_string, gui_logger)
            
        
        #also create E-APO config for this sample
        if eapo_export == True:
            e_apo_config_creation.write_e_apo_configs_hpcfs(brand, headphone, sample, primary_path)
        
        #
        #save FIR to stereo WAV
        #
        output_wav_s = np.empty((CN.HPCF_FIR_LENGTH, 2))
        
        output_wav_s[0:CN.HPCF_FIR_LENGTH,0] = hpcf_fir[0:CN.HPCF_FIR_LENGTH]
        output_wav_s[0:CN.HPCF_FIR_LENGTH,1] = hpcf_fir[0:CN.HPCF_FIR_LENGTH]

        out_file_path = pjoin(out_file_dir_st_wav, hpcf_name_wav)
        
        if fir_stereo_export == True:
            #create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)

            hf.write2wav(file_name=out_file_path, data=output_wav_s, prevent_clipping=1, bit_depth=bit_depth, samplerate=samp_freq)
            
            
            log_string = 'HpCF (WAV Stereo FIR): ' + hpcf_name + ' saved to: ' + str(out_file_dir_st_wav)
            hf.log_with_timestamp(log_string, gui_logger)
                
        #
        #save Graphic EQ to txt file
        #
        

        out_file_path = pjoin(out_file_dir_geq, hpcf_name_geq)
        
        if geq_export == True:
            #hpcf_geq = hpcf_dict.get('graphic_eq')
            #20250408: calculate instead of storing in database
            hpcf_geq = hpcf_fir_to_geq(fir_array=hpcf_fir,geq_mode=2,sample_rate=CN.SAMP_FREQ,geq_freq_arr=CN.GEQ_SET_F_127)
            
            #create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
        
            #if geq string type is JSON string, convert to GEQ string
            if geq_json == True:
                dictionary = json.loads(hpcf_geq)
                # split dictionary into keys and values
                keys = list(dictionary.keys())
                values = list(dictionary.values())
                list_length = len(values)
                #convert to a string
                geq_string = 'GraphicEQ: '
                for m in range(list_length):
                        geq_string =geq_string + keys[m] + ' ' + str(values[m])
                        if m < list_length-1:
                            geq_string = geq_string + '; '
                hpcf_geq = geq_string
            
            #save as a file    
            with open(out_file_path, 'w') as f:
                f.write(hpcf_geq)    
            log_string = 'HpCF (Graphic EQ): ' + hpcf_name + ' saved to: ' + str(out_file_dir_geq)
            hf.log_with_timestamp(log_string, gui_logger)
        
        #
        #save Graphic EQ 31 band to txt file
        #
        

        out_file_path = pjoin(out_file_dir_geq_31, hpcf_name_geq)
        
        if geq_31_export == True:
            #hpcf_geq_31 = hpcf_dict.get('graphic_eq_31')
            #20250408: calculate instead of storing in database
            hpcf_geq_31 = hpcf_fir_to_geq(fir_array=hpcf_fir,geq_mode=2,sample_rate=CN.SAMP_FREQ,geq_freq_arr=CN.GEQ_SET_F_31)
            
            #create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
        
            #if geq string type is JSON string, convert to GEQ string
            if geq_json == True:
                dictionary = json.loads(hpcf_geq_31)
                # split dictionary into keys and values
                keys = list(dictionary.keys())
                values = list(dictionary.values())
                list_length = len(values)
                #convert to a string
                geq_string = 'GraphicEQ: '
                for m in range(list_length):
                        geq_string =geq_string + keys[m] + ' ' + str(values[m])
                        if m < list_length-1:
                            geq_string = geq_string + '; '
                hpcf_geq_31 = geq_string
            
            #save as a file    
            with open(out_file_path, 'w') as f:
                f.write(hpcf_geq_31)
            log_string = 'HpCF (Graphic EQ 31 band): '+ hpcf_name +' saved to: ' + str(out_file_dir_geq_31)
            hf.log_with_timestamp(log_string, gui_logger)
        
        #
        #save Graphic EQ 103 band to txt file
        #
        if geq_103_export == True or hesuvi_export == True:
            #hpcf_geq_103 = hpcf_dict.get('graphic_eq_103')
            #20250408: calculate instead of storing in database
            hpcf_geq_103 = hpcf_fir_to_geq(fir_array=hpcf_fir,geq_mode=2,sample_rate=CN.SAMP_FREQ,geq_freq_arr=CN.GEQ_SET_F_103)

        out_file_path = pjoin(out_file_dir_geq_103, hpcf_name_geq)
        
        if geq_103_export == True:
            #create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
        
            #if geq string type is JSON string, convert to GEQ string
            if geq_json == True:
                dictionary = json.loads(hpcf_geq_103)
                # split dictionary into keys and values
                keys = list(dictionary.keys())
                values = list(dictionary.values())
                list_length = len(values)
                #convert to a string
                geq_string = 'GraphicEQ: '
                for m in range(list_length):
                        geq_string =geq_string + keys[m] + ' ' + str(values[m])
                        if m < list_length-1:
                            geq_string = geq_string + '; '
                hpcf_geq_103 = geq_string
            
            #save as a file    
            with open(out_file_path, 'w') as f:
                f.write(hpcf_geq_103)
            log_string = 'HpCF (Graphic EQ 103 band): '+ hpcf_name +' saved to: ' + str(out_file_dir_geq_103)
            hf.log_with_timestamp(log_string, gui_logger)
        

        #also export graphic eq to hesuvi folder
        if hesuvi_export == True:
            out_file_folder = pjoin(hesuvi_path, 'eq','_HpCFs',brand_folder)
            out_file_path = pjoin(out_file_folder,hpcf_name_geq)
            
            #create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            #if geq string type is JSON string, convert to GEQ string
            if geq_json == True:
                dictionary = json.loads(hpcf_geq_103)
                # split dictionary into keys and values
                keys = list(dictionary.keys())
                values = list(dictionary.values())
                list_length = len(values)
                #convert to a string
                geq_string = 'GraphicEQ: '
                for m in range(list_length):
                        geq_string =geq_string + str(int(float(keys[m]))) + ' ' + str(values[m])
                        if m < list_length-1:
                            geq_string = geq_string + '; '
                hpcf_geq_103 = geq_string
            
            with open(out_file_path, 'w') as f:
                f.write(hpcf_geq_103)#f.write(hpcf_geq)
            log_string = 'HpCF (HeSuVi GEQ): ' + hpcf_name + ' saved to: ' + str(out_file_folder)
            hf.log_with_timestamp(log_string, gui_logger)


    except Exception as ex:
        log_string = 'Failed to export HpCFs'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error



def hpcf_to_file_bulk(conn, primary_path, headphone=None, fir_export = True, fir_stereo_export = True, geq_export = True, geq_31_export = True, geq_103_export = False, hesuvi_export = True, eapo_export=True, report_progress=0, gui_logger=None, samp_freq=CN.SAMP_FREQ, bit_depth='PCM_24', force_output=False):
    """
    Function bulk exports all filters to wav or txt files
    calls above function on each headphone/sample combination in the database
    :param headphone: string, name of headphone to export to file
    :param fir_export: int, 1 = export mono fir files
    :param fir_stereo_export: int, 1 = export stereo fir files
    :param geq_export: int, 1 = export graphic eq (full band) files
    :param geq_31_export: int, 1 = export graphic eq (31 band) files
    :param geq_103_export: int, 1 = export graphic eq (103 band) files
    :param hesuvi_export: int, 1 = export hesuvi files
    :param eapo_export: int, 1 = export equalizer apo config files for hpcf fir convolution
    :param report_progress: bool, True = report progress to dearpygui progress bar
    """
    
    try:
        
        #if no headphone specified,
        if headphone == None:
            #get list of all headphones in the DB
            headphone_list = get_all_headphone_list(conn)
        else:
            headphone_list = [headphone]
        
        #for each headphone
        for h in headphone_list:
            
            #get list of samples
            sample_list = get_hpcf_samples_dicts(conn, h)

            num_samples = len(sample_list)
            #export each sample
            for index, s in enumerate(sample_list):
            #for s in sample_list:
                sample_dict = dict(s)
                hpcf_to_file(sample_dict, primary_path=primary_path, fir_export=fir_export, fir_stereo_export=fir_stereo_export, geq_export=geq_export, 
                             geq_31_export=geq_31_export, geq_103_export=geq_103_export, hesuvi_export=hesuvi_export, eapo_export=eapo_export, gui_logger=gui_logger, samp_freq=samp_freq, bit_depth=bit_depth, force_output=force_output)

                if report_progress > 0:
                    progress = ((index+1)/num_samples)
                    if report_progress == 2:
                        dpg.set_value("progress_bar_hpcf", progress)
                        dpg.configure_item("progress_bar_hpcf", overlay = str(int(progress*100))+'%')
                    else:
                        dpg.set_value("qc_progress_bar_hpcf", progress)
                        dpg.configure_item("qc_progress_bar_hpcf", overlay = str(int(progress*100))+'%')
                        if progress == 1:
                            dpg.configure_item("qc_progress_bar_hpcf", overlay = CN.PROGRESS_FIN)
                    

    except Error as e:
        logging.error("Error occurred", exc_info = e)


        




def hpcf_generate_averages(conn, gui_logger=None):
    """
    Function hpcf_generate_averages
        1) searches DB for HpCFs without averages then generates averages and inserts into database
        2) searches DB for HpCFs with averages last modified before latest sample then generates averages and updates average in database
    """
    
    try:

        #retrieve geq frequency list as an array - 127 bands
        geq_set_f_127 = hpcf_retrieve_set_geq_freqs(f_set=1)
        #retrieve geq frequency list as an array - 31 bands
        geq_set_f_31 = hpcf_retrieve_set_geq_freqs(f_set=2)
        #retrieve geq frequency list as an array - 103 bands
        geq_set_f_103 = hpcf_retrieve_set_geq_freqs(f_set=3)
        
        now_datetime = datetime.now()
        
        #impulse
        impulse=np.zeros(CN.N_FFT)
        impulse[0]=1
        fr_flat_mag = np.abs(np.fft.fft(impulse))
        fr_flat = hf.mag2db(fr_flat_mag)
        
        #get list of all headphones in the DB
        headphone_list = get_all_headphone_list(conn)
        
        #for each headphone, grab all samples
        for h in headphone_list:
            update_average=0
            insert_average=0
            
            sample_list = get_hpcf_samples_dicts(conn, h)
            
            #get last modified of averaged sample
            has_average = 0
            num_samples=0
            average_last_mod = datetime(2000, 1, 1)
            samples_last_mod = datetime(2000, 1, 1)
            for s in sample_list:
                sample_dict = dict(s)
                sample_name = sample_dict['sample']
                if sample_name == 'Average':
                    has_average=1
                    average_last_mod_str = sample_dict['created_on']
                    average_last_mod = datetime.strptime(average_last_mod_str, "%Y-%m-%d %H:%M:%S.%f")
                    
                #check last modified of this sample, if greater than previous latest date, update latest date of samples
                sample_last_mod_str = sample_dict['created_on']
                sample_last_mod = datetime.strptime(sample_last_mod_str, "%Y-%m-%d %H:%M:%S.%f")
                if sample_last_mod > samples_last_mod:
                    samples_last_mod = sample_last_mod
                #get number of samples
                sample_id = sample_dict['sample_id']
                if sample_id > num_samples:
                    num_samples = sample_id
            #compare last modified date of average with last modified date of samples
            #update average if average is outdated
            if samples_last_mod > average_last_mod and num_samples > 1 and has_average == 1:
                update_average=1
                insert_average=0
                
            #insert average if no sample exists with average name (has_average == 0)
            if has_average == 0 and num_samples > 1:
                update_average=0
                insert_average=1
                
            #generate average if either of above are required and there are more than 1 samples
            if (update_average == 1 or insert_average == 1):
                num_samples_avg = 0
                
                hpcf_fft_avg_db = fr_flat.copy()

                #generate average spectrum of each sample filter
                for s in sample_list:
                    sample_dict = dict(s)
                    sample_id = sample_dict['sample_id']
                    #only consider samples that are non averages
                    if sample_id > 0:
                        sample_fir_json = sample_dict['fir']
                        sample_fir_list = json.loads(sample_fir_json) 
                        sample_fir = np.array(sample_fir_list)
                        
                        #zero pad
                        data_pad=np.zeros(65536)
                        data_pad[0:(CN.HPCF_FIR_LENGTH-1)]=sample_fir[0:(CN.HPCF_FIR_LENGTH-1)]
                    
                        hpcf_current = data_pad
                        hpcf_current_fft = np.fft.fft(hpcf_current)
                        hpcf_current_mag_fft=np.abs(hpcf_current_fft)
                        hpcf_current_db_fft = hf.mag2db(hpcf_current_mag_fft)
                        
                        hpcf_fft_avg_db = np.add(hpcf_fft_avg_db,hpcf_current_db_fft)
                        
                        num_samples_avg = num_samples_avg+1
                
                
                #divide by total number of samples
                hpcf_fft_avg_db = hpcf_fft_avg_db/num_samples_avg
                #convert to mag
                hpcf_fft_avg_mag = hf.db2mag(hpcf_fft_avg_db)
                #create min phase FIR
                hpcf_avg_fir_full = hf.mag_to_min_fir(hpcf_fft_avg_mag, out_win_size=CN.HPCF_FIR_LENGTH)
                #store in cropped array
                hpcf_avg_fir_array=np.zeros((CN.HPCF_FIR_LENGTH))
                hpcf_avg_fir_array[:] = np.copy(hpcf_avg_fir_full[0:CN.HPCF_FIR_LENGTH])
                

                ### get fir
                # convert from array to list
                fir_list = hpcf_avg_fir_array.tolist()
                # convert from list to Json
                fir_json_str = json.dumps(fir_list)
 
                
                #get graphic eq filter 127 band
               # geq_str = hpcf_fir_to_geq(fir_array=hpcf_avg_fir_array,geq_mode=2,sample_rate=CN.SAMP_FREQ,geq_freq_arr=geq_set_f_127)
                #20250408: no longer required to store geq data. Replace with empty string
                geq_str = ''
                
                #get graphic eq filter 31 band
                #geq_31_str = hpcf_fir_to_geq(fir_array=hpcf_avg_fir_array,geq_mode=2,sample_rate=CN.SAMP_FREQ,geq_freq_arr=geq_set_f_31)
                #20250408: no longer required to store geq data. Replace with empty string
                geq_31_str = ''
                
                ##get graphic eq 32 band filter
                #geq_32_str = hpcf_fir_to_geq(fir_array=hpcf_avg_fir_array,geq_mode=1,sample_rate=CN.SAMP_FREQ)
                
                #get graphic eq filter 103 band
                #geq_103_str = hpcf_fir_to_geq(fir_array=hpcf_avg_fir_array,geq_mode=2,sample_rate=CN.SAMP_FREQ,geq_freq_arr=geq_set_f_103)
                #20250408: no longer required to store geq data. Replace with empty string
                geq_103_str = ''

                #last modified text
                created_on = now_datetime

                #case for updating existing average
                if update_average == 1:
                    hpcf_data = (fir_json_str,geq_str,geq_31_str,geq_103_str,created_on)
                    replace_hpcf_filter_data(conn,hpcf_data,h,'Average', gui_logger=gui_logger)
                
                
                #case for inserting new average
                if insert_average == 1:
                    #get brand
                    brand = get_brand(conn, h)
                    #create tuple
                    hpcf_to_insert = (brand, h, 'Average', 0, fir_json_str, geq_str, geq_31_str, geq_103_str, created_on)
                    
                    # create entry
                    create_hpcf_row(conn, hpcf_to_insert, brand, h, 'Average', gui_logger=gui_logger)
        
                
        conn.commit()
        
    except Error as e:
        logging.error("Error occurred", exc_info = e)

    
    
    


def hpcf_to_plot(conn, headphone, sample, primary_path=CN.DATA_DIR_OUTPUT, save_to_file=0, plot_type=0):
    """
    Function returns a plot of freq response for a specified hpCF, also plot graphic eq and graphic eq 32 band filters
    :param headphone: string, name of headphone to plot
    :param sample: string, name of sample to plot
    :param primary_path: string, path to save plots to if save_to_file == 1
    :param save_to_file: int, 1 = save plot to a file
    :param plot_type: int, 0 = matplotlib, 1 = dearpygui (series 1 - filter export), 1 = dearpygui (series 2 - quick config)
    """
    
    try:
        
        #get dictionary of hpcf data
        hpcf_dict = dict(get_hpcf_headphone_sample_dict(conn, headphone, sample))
        
        #Directories
        brand_folder = hpcf_dict.get('brand')
        hpcf_name = hpcf_dict.get('headphone') + ' ' + hpcf_dict.get('sample')
        
        #output directories
        out_file_dir_plot = pjoin(primary_path, CN.PROJECT_FOLDER_HPCFS,'Plots',brand_folder)
        
        hpcf_fir_json = hpcf_dict.get('fir')
        hpcf_fir_list = json.loads(hpcf_fir_json) 
        hpcf_fir = np.array(hpcf_fir_list)
        
        #get freq response
        data_pad=np.zeros(65536)
        data_pad[0:(CN.HPCF_FIR_LENGTH-1)]=hpcf_fir[0:(CN.HPCF_FIR_LENGTH-1)]
        data_fft = np.fft.fft(data_pad)
        hpcf_fr_mag = np.abs(data_fft)
        
        #run plot
        #plot_tile = hpcf_name + ' frequency response'
        plot_tile = hpcf_name 
        hf.plot_data(hpcf_fr_mag, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=save_to_file, plot_path=out_file_dir_plot, plot_type=plot_type)
        
  
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        
    

def generate_hp_summary_sheet(conn, measurement_folder_name, in_ear_set=0, gui_logger=None):
    """
    Function generates a csv summary sheet of headphone measurements within a folder
    """

    try:

        if in_ear_set == 1:
            hp_folder = 'in_ear'
        else:
            hp_folder = 'over_ear'

        # delimiters
        delimiter_l = ' L.txt'
        delimiter_r = ' R.txt'

        # get master list of current headphone names
        headphone_list = get_all_headphone_list(conn)
        brands_list = get_brand_list(conn)
        
        # directories
        measurement_directory = pjoin(CN.DATA_DIR_RAW_HP_MEASRUEMENTS, hp_folder, measurement_folder_name)

        output_directory = CN.DATA_DIR_RAW_HP_MEASRUEMENTS
        out_file_name = measurement_folder_name + '_summary.csv'
        output_file = pjoin(output_directory, out_file_name)

        # read models from metadata csv. Expects metadata.csv in the specific measurement folder
        metadata_file_name = 'metadata.csv'
        metadata_file = pjoin(measurement_directory, metadata_file_name)
        models_list = []
        filename_mapping = {}  # Dictionary to store Filename to Models mapping
        filename_brand_mapping = {} # Dictionary to store Filename to Brands mapping

        with open(metadata_file, encoding='utf-8-sig', newline='') as inputfile:
            reader = csv.DictReader(inputfile)
            for row in reader:  # rows 2 and onward
                model = row.get('Models')
                models_list.append(model)
                filename = row.get('Filename')
                brand = row.get('Brands')
                if filename:
                    filename_mapping[filename] = model
                    filename_brand_mapping[filename] = brand

        # logging.info(str(models_list))

        # list all names in path
        dir_list = os.listdir(measurement_directory)

        # for each txt file in folder
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            # headings
            writer.writerow(['file_name', 'extracted_name', 'original_name', 'original_brand', 'closest_brand', 'suggested_brand', 'suggested_name', 'suggested_name_alternate', 'chosen_brand', 'chosen_name', 'include'])
            # write row for each input file
            for txt_file in dir_list:

                file_name = txt_file
                logging.info("Processing:" + str(file_name))

                if 'txt' in file_name:

                    include_hp = 'Y'
                    
                    file_name_converted = file_name.replace("_Y_", " ") 
                    name_before_l = file_name_converted.split(delimiter_l)[0]
                    name_before_l_r = name_before_l.split(delimiter_r)[0]
                    extracted_name = name_before_l_r

                    # Check if a filename match exists in metadata
                    if filename_mapping:
                        filename_matches = hf.get_close_matches_fuzz(file_name, list(filename_mapping.keys()))
                        if filename_matches:
                            original_name = filename_mapping[filename_matches[0][0]]
                            original_brand = filename_brand_mapping[filename_matches[0][0]]
                        else:
                            #find full name from model list if no filename match
                            original_name_matches = hf.get_close_matches_fuzz(extracted_name, models_list)

                            if original_name_matches == None or not original_name_matches:
                                original_name = extracted_name
                                original_brand = ''
                                include_hp = 'N'
                            else:
                                original_name = original_name_matches[0][0]
                                original_brand = ''
                    else:
                        #find full name from model list if no filename mapping
                        original_name_matches = hf.get_close_matches_fuzz(extracted_name, models_list)

                        if original_name_matches == None or not original_name_matches:
                            original_name = extracted_name
                            original_brand = ''
                            include_hp = 'N'
                        else:
                            original_name = original_name_matches[0][0]
                            original_brand = ''

                    # calculate a suggested name based on closest match in database
                    suggested_name_matches = hf.get_close_matches_fuzz(original_name, headphone_list)

                    if suggested_name_matches == None or not suggested_name_matches:
                        suggested_name = ''
                        suggested_name_alt = ''
                    else:
                        suggested_name = suggested_name_matches[0][0]
                        if len(suggested_name_matches) > 1:
                            suggested_name_alt = suggested_name_matches[1][0]
                        else:
                            suggested_name_alt = ''

                    # also calculate a closest matching brand based on original brand (if present)
                    suggested_name_matches = hf.get_close_matches_fuzz(original_brand, brands_list)
                    if suggested_name_matches == None or not suggested_name_matches:
                        closest_brand = ''
                    else:
                        closest_brand = suggested_name_matches[0][0]
                    
                    

                    # grab the brand for the suggested name
                    suggested_brand = get_brand(conn, suggested_name)

                    # write new row to file
                    writer.writerow([file_name, extracted_name, original_name, original_brand, closest_brand, suggested_brand, suggested_name, suggested_name_alt, '', '', include_hp])

        # log results
        log_string = 'Summary sheet saved: ' + output_file
        hf.log_with_timestamp(log_string, gui_logger)

    except Error as e:
        logging.error("Error occurred", exc_info=e)


def generate_hp_summary_sheet_basic(conn, measurement_folder_name, in_ear_set = 0, gui_logger=None):
    """
    Function generates a csv summary sheet of headphone measurements within a folder
    """
    
    try:
        
        if in_ear_set == 1:
            hp_folder = 'in_ear'
        else:
            hp_folder = 'over_ear'   
        
        #delimiters
        delimiter_l = ' L.txt'
        delimiter_r = ' R.txt'
        
        #get master list of current headphone names
        headphone_list = get_all_headphone_list(conn)
        
        #directories
        measurement_directory = pjoin(CN.DATA_DIR_RAW_HP_MEASRUEMENTS, hp_folder, measurement_folder_name)
        
        output_directory = CN.DATA_DIR_RAW_HP_MEASRUEMENTS
        out_file_name = measurement_folder_name + '_summary.csv'
        output_file = pjoin(output_directory, out_file_name)

        #read models from metadata csv. Expects metadata.csv in the specific measurement folder
        metadata_file_name = 'metadata.csv'
        metadata_file = pjoin(measurement_directory, metadata_file_name)
        models_list = []
        with open(metadata_file, encoding='utf-8-sig', newline='') as inputfile:
            reader = DictReader(inputfile)
            for row in reader:#rows 2 and onward
                #store each row as a dictionary
                #append to list of dictionaries
                model = row.get('Models')
                models_list.append(model)  
        
        
        #logging.info(str(models_list))
        
        #list all names in path
        dir_list = os.listdir(measurement_directory)

        #for each txt file in folder
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            #headings
            writer.writerow(['file_name', 'extracted_name', 'original_name', 'suggested_brand', 'suggested_name', 'suggested_name_alternate','chosen_brand','chosen_name','include'])
            #write row for each input file
            for txt_file in dir_list:
                
                file_name = txt_file
                
                if 'txt' in file_name:
                    
                    include_hp = 'Y'

                    name_before_l = file_name.split(delimiter_l)[0]
                    name_before_l_r = name_before_l.split(delimiter_r)[0]
                    extracted_name = name_before_l_r

                    
                    #find full name from model list
                    original_name_matches = hf.get_close_matches_fuzz(name_before_l_r,models_list)
                    
                    if original_name_matches == None or not original_name_matches:
                        original_name=extracted_name
                        include_hp='N'
                    else:
                        original_name = original_name_matches[0][0]
  
                    #calculate a suggested name based on closest match in database
                    suggested_name_matches = hf.get_close_matches_fuzz(original_name,headphone_list)
                    
                    if suggested_name_matches == None or not suggested_name_matches:
                        suggested_name=''
                        suggested_name_alt ='' 
                    else:
                        suggested_name = suggested_name_matches[0][0]
                        if len(suggested_name_matches) > 1:
                            suggested_name_alt = suggested_name_matches[1][0]
                        else:
                            suggested_name_alt ='' 
                        
      
                    #grab the brand for the suggested name
                    suggested_brand = get_brand(conn, suggested_name)
                    
                    #write new row to file
                    writer.writerow([file_name, extracted_name, original_name, suggested_brand, suggested_name,suggested_name_alt, '', '',include_hp])
                
        #log results
        log_string = 'Summary sheet saved: ' + output_file
        hf.log_with_timestamp(log_string, gui_logger)
        
    except Error as e:
        logging.error("Error occurred", exc_info = e)


def calculate_new_hpcfs(conn, measurement_folder_name, in_ear_set = 0, gui_logger=None):
    """
    Function calculates new HpCFs from RAW data and populates new row in database
    """
    
    try:
        
        if in_ear_set == 1:
            hp_folder = 'in_ear'
        else:
            hp_folder = 'over_ear'   
        
        now_datetime = datetime.now()
        
        #directories
        data_raw_directory = CN.DATA_DIR_RAW_HP_MEASRUEMENTS
        measurement_directory = pjoin(CN.DATA_DIR_RAW_HP_MEASRUEMENTS, hp_folder, measurement_folder_name)
        
        #impulse
        impulse=np.zeros(CN.N_FFT)
        impulse[0]=1
        fr_flat_mag = np.abs(np.fft.fft(impulse))
        fr_flat = hf.mag2db(fr_flat_mag)
        
        #retrieve geq frequency list as an array - 127 bands
        geq_set_f_127 = hpcf_retrieve_set_geq_freqs(f_set=1)
        #retrieve geq frequency list as an array - 31 bands
        geq_set_f_31 = hpcf_retrieve_set_geq_freqs(f_set=2)
        #retrieve geq frequency list as an array - 103 bands
        geq_set_f_103 = hpcf_retrieve_set_geq_freqs(f_set=3)
        
        #create array for new x axis for interpolation
        freq_max = int(CN.SAMP_FREQ/2)
        #in np fft array where nfft = 65536, id 0 and id 32768 are unique. Create 32769 unique values
        num_samples = int(CN.N_FFT/2)+1
        xnew = np.linspace(0, freq_max, num=num_samples)
        
        #read in-ear headphone equalisation from WAV file
        if CN.APPLY_ADD_HP_EQ > 0:
            filename = 'diffuse_field_eq_for_in_ear_headphones.wav'
            wav_fname = pjoin(CN.DATA_DIR_INT, filename)
            samplerate, data_addit_eq = wavfile.read(wav_fname)
            data_addit_eq = data_addit_eq / (2.**31)
            #convert to mag response
            in_ear_eq_fir=np.zeros(CN.N_FFT)
            in_ear_eq_fir[0:1024] = data_addit_eq[0:1024]
            data_fft = np.fft.fft(in_ear_eq_fir)#
            in_ear_eq_mag=np.abs(data_fft)
            in_ear_eq_db=hf.mag2db(in_ear_eq_mag)
        
        #variable to identify if set contains left and right samples
        left_and_right_set = 0
        left_meas_count = 0
        right_meas_count = 0
        
        #
        #read metadata from csv
        #
        metadata_file_name = measurement_folder_name + '_summary.csv'
        metadata_file = pjoin(data_raw_directory, metadata_file_name)
        metadata_dict_list = []
        with open(metadata_file, encoding='utf-8-sig', newline='') as inputfile:
            reader = DictReader(inputfile)
            for row in reader:#rows 2 and onward
                #store each row as a dictionary

                #append to list of dictionaries
                metadata_dict_list.append(row)  
                
                #check if set contains left and right samples
                file_name = row.get('file_name')
                if 'L.txt' in file_name or 'l.txt' in file_name or 'l1.txt' in file_name or 'L1.txt' in file_name:
                    left_meas_count = left_meas_count+1
                if 'R.txt' in file_name or 'r.txt' in file_name or 'r1.txt' in file_name or 'R1.txt' in file_name:
                    right_meas_count = right_meas_count+1
                
        #check if set contains left and right samples
        if left_meas_count > 0 and right_meas_count > 0:
            left_and_right_set=1
        
        #create array to fill with frequency data
        hpcf_data_dict_list = metadata_dict_list
        
        #
        #read measurements from txt
        #
        for root, dirs, files in os.walk(measurement_directory):
            for filename in sorted(files):
                for hpcf_dict in hpcf_data_dict_list:
                    file_name_hpcf = hpcf_dict.get('file_name')
                    
                    #logging.info(filename)
                    
                    if file_name_hpcf == filename:
                        #read txt file and store data into array
                        txt_fname = pjoin(root, filename)
                        try:
                            data = np.loadtxt(txt_fname, comments='*')
                        except:
                            try:
                                data = np.loadtxt(txt_fname, comments='*', delimiter=';')
                            except:
                                data = np.loadtxt(txt_fname, comments='*', delimiter=',')
                 
                        #1-D data interpolation
                        data_x = data[:,0]
                        data_y = data[:,1]
                        spl = CubicSpline(data_x, data_y)
                        data_y_new = spl(xnew)

                        #fill entire spectrum
                        data_y_full=np.zeros(CN.N_FFT)
                        reversed_arr = np.flipud(data_y_new)
                        neg_f_ind_start = int(CN.N_FFT/2)+1
                        rev_arr_ind_start = 1#ignore nyquist f
                        rev_arr_ind_end = int(CN.N_FFT/2) #ignore DC
                        data_y_full[0:neg_f_ind_start] = data_y_new
                        data_y_full[neg_f_ind_start:] = reversed_arr[1:rev_arr_ind_end]
               
                        #add to current dictionary
                        hpcf_dict['meas_resp_db_intrp'] = data_y_full
  
        #condense list if this is a left/right set
        if left_and_right_set == 1:
            #create new dict list for condensed set
            hpcf_data_dict_list_c = []
            #iterate through full list
            for hpcf_dict_l in hpcf_data_dict_list:
                extract_name = hpcf_dict_l.get('extracted_name')
                file_name_l = hpcf_dict_l.get('file_name')
       
                #continue if this is a L measurement
                if 'L.txt' in file_name_l or 'l.txt' in file_name or 'l1.txt' in file_name_l or 'L1.txt' in file_name_l:
                    #grab response for L measurement
                    resp_db_l = hpcf_dict_l.get('meas_resp_db_intrp')
                    #find R measurement
                    for hpcf_dict_r in hpcf_data_dict_list:
                        extract_name_r = hpcf_dict_r.get('extracted_name')
                        file_name_r = hpcf_dict_r.get('file_name')
                        #continue if this is a R measurement
                        if ('R.txt' in file_name_r or 'r.txt' in file_name_r or 'r1.txt' in file_name or 'R1.txt' in file_name) and (extract_name_r == extract_name):
                            #grab response for R measurement
                            resp_db_r = hpcf_dict_r.get('meas_resp_db_intrp')
                    #calculate average response
                    resp_db_avg = np.divide(np.add(resp_db_l,resp_db_r),2)
                    #create new dictionary
                    dict_new = hpcf_dict_l.copy()
                    dict_new.update({"meas_resp_db_intrp": resp_db_avg})
                    #add to new dict list
                    hpcf_data_dict_list_c.append(dict_new)
            
        else:
            hpcf_data_dict_list_c=hpcf_data_dict_list
        
        #
        #calculate target headphone
        #
        hpcf_fft_avg_db = fr_flat.copy()
        num_headphones = 0
        for hpcf_dict in hpcf_data_dict_list_c:
            #check if this row should be included
            include_file = hpcf_dict.get('include')
            if include_file == 'Y':
                #get db response
                resp_db = hpcf_dict.get('meas_resp_db_intrp')
                #add db response to average
                hpcf_fft_avg_db = np.add(hpcf_fft_avg_db,resp_db)
                
                num_headphones = num_headphones+1
            
        #divide by total number of samples
        hpcf_fft_avg_db = hpcf_fft_avg_db/num_headphones
        hpcf_target_db = hpcf_fft_avg_db.copy()
        
        #apply in ear compensation to target if in_ear_set
        if in_ear_set == 1:
            #apply additional EQ for in-ear headphones
            hpcf_target_db_comp = np.add(hpcf_target_db,in_ear_eq_db)
        else:
            hpcf_target_db_comp = hpcf_target_db.copy()
            
        #apply smoothing
        #convert to mag
        hpcf_target_mag = hf.db2mag(hpcf_target_db_comp)
        #level ends of spectrum
        hpcf_target_mag = hf.level_spectrum_ends(hpcf_target_mag, 50, 19000, smooth_win = 7)#
        #smoothing
        if CN.SPECT_SMOOTH_MODE == 0:
            hpcf_target_mag = hf.smooth_fft(hpcf_target_mag, 1124, 20, 101)
            hpcf_target_mag = hf.smooth_fft(hpcf_target_mag, 8075, 20, 505)
            hpcf_target_mag = hf.smooth_fft(hpcf_target_mag, 20187, 20, 20)  
        else:
            hpcf_target_mag = hf.smooth_fft_octaves(hpcf_target_mag)
        
        #back to db
        hpcf_target_db_comp=hf.mag2db(hpcf_target_mag)
            
        #
        #calculate HpCFs
        #
        for hpcf_dict in hpcf_data_dict_list_c:
            
            #check if this row should be included
            include_file = hpcf_dict.get('include')
            if include_file == 'Y':
                #get metadata
                brand = hpcf_dict.get('chosen_brand')
                headphone = hpcf_dict.get('chosen_name')
                
                #get db response
                resp_db = hpcf_dict.get('meas_resp_db_intrp')
                #subtract from target
                hpcf_db = np.subtract(hpcf_target_db_comp,resp_db)
                
                #
                #convert to time domain (FIR)
                #
                #convert to mag
                hpcf_fft_out_mag = hf.db2mag(hpcf_db)
  
                #level ends of spectrum
                hpcf_fft_out_mag = hf.level_spectrum_ends(hpcf_fft_out_mag, 50, 19000, smooth_win = 7)#
  
                #smoothing
                if CN.SPECT_SMOOTH_MODE == 0:
                    hpcf_fft_out_mag = hf.smooth_fft(hpcf_fft_out_mag, 1124, 20, 101)
                    hpcf_fft_out_mag = hf.smooth_fft(hpcf_fft_out_mag, 5383, 20, 336)
                    hpcf_fft_out_mag = hf.smooth_fft(hpcf_fft_out_mag, 7402, 20, 673)
                    hpcf_fft_out_mag = hf.smooth_fft(hpcf_fft_out_mag, 20187, 20, 20)
                else:
                    hpcf_fft_out_mag = hf.smooth_fft_octaves(hpcf_fft_out_mag)
                
                
                #normalise to -6db in low frequencies
                hpcf_fft_out_mag = np.divide(hpcf_fft_out_mag,np.mean(hpcf_fft_out_mag[60:300]))
                hpcf_fft_out_mag = hpcf_fft_out_mag*0.5
                
                #create min phase FIR
                hpcf_out_fir_full = hf.mag_to_min_fir(hpcf_fft_out_mag, out_win_size=CN.HPCF_FIR_LENGTH)
                
                #store in cropped array
                hpcf_out_fir_array=np.zeros((CN.HPCF_FIR_LENGTH))
                hpcf_out_fir_array[:] = np.copy(hpcf_out_fir_full[0:CN.HPCF_FIR_LENGTH])
                
    
                
                #
                #check for duplicate FIR
                #
                #get list of hpcfs for the selected headphone
                #get samples for specified headphone
                hpcfs_headphone = get_hpcf_samples_dicts(conn, headphone)
                #check fir data and compare with new fir data
                duplicate_found=0
                #find largest ID
                largest_id = 0
                if hpcfs_headphone and hpcfs_headphone != None:
                    for s in hpcfs_headphone:
                        sample_dict = dict(s)
                        hpcf_fir_json = sample_dict.get('fir')
                        hpcf_fir_list = json.loads(hpcf_fir_json) 
                        hpcf_fir = np.array(hpcf_fir_list)
                        if np.array_equal(hpcf_fir, hpcf_out_fir_array):
                            duplicate_found=1;
                        sample_id = sample_dict['sample_id']
                        if sample_id > largest_id:
                            largest_id = sample_id
                        
                if duplicate_found == 1:
                    log_string = 'Duplicate HpCF generated. Not inserted into DB. Brand: ' + brand + ' Headphone: ' + headphone
                    hf.log_with_timestamp(log_string, gui_logger)
                
                #add each HPCF to database if not duplicate
                else:
                    
                    # convert from array to list
                    fir_list = hpcf_out_fir_array.tolist()
                    # convert from list to Json
                    fir_json_str = json.dumps(fir_list)
                    
                    
                    #get graphic eq filter 127 band
                    #geq_str = hpcf_fir_to_geq(fir_array=hpcf_out_fir_array,geq_mode=2,sample_rate=samplerate,geq_freq_arr=geq_set_f_127)
                    #20250408: no longer required to store geq data. Replace with empty string
                    geq_str = ''
                    
                    #get graphic eq filter 31 band
                    #geq_31_str = hpcf_fir_to_geq(fir_array=hpcf_out_fir_array,geq_mode=2,sample_rate=samplerate,geq_freq_arr=geq_set_f_31)
                    #20250408: no longer required to store geq data. Replace with empty string
                    geq_31_str = ''
                
                    ##get graphic eq 32 band filter
                    #geq_32_str = hpcf_fir_to_geq(fir_array=hpcf_out_fir_array,geq_mode=1,sample_rate=samplerate)
                       
                    #get graphic eq filter 31 band
                    #geq_103_str = hpcf_fir_to_geq(fir_array=hpcf_out_fir_array,geq_mode=2,sample_rate=samplerate,geq_freq_arr=geq_set_f_103)
                    #20250408: no longer required to store geq data. Replace with empty string
                    geq_103_str = ''
    
    
                    #calculate sample id
                    sample_id = largest_id+1
                    #calculate sample
                    sample_name = hpcf_sample_id_to_name(sample_id)
                    #last modified text
                    created_on = now_datetime
     
                    #create hpcf tuple for this hpcf
                    # tasks
                    hpcf_to_insert = (brand, headphone, sample_name, sample_id, fir_json_str, geq_str, geq_31_str, geq_103_str, created_on)
                    
                    log_string = 'HpCF generated. Brand: ' + brand + ' Headphone: ' + headphone + ' Sample: ' + sample_name + ' Sample id: ' + str(sample_id)
                    hf.log_with_timestamp(log_string, gui_logger)
                    
                    if sample_id > 26:
                        log_string = 'HpCF not inserted into dB due to sample ID exceeding max sample ID (26) for this headphone'
                        hf.log_with_timestamp(log_string, gui_logger)
                    elif headphone == '' or headphone == None or brand == '' or brand == None:
                        log_string = 'HpCF not inserted into dB due to missing brand or headphone name'
                        hf.log_with_timestamp(log_string, gui_logger)
                    else:
                        # create entry
                        create_hpcf_row(conn, hpcf_to_insert, brand, headphone, sample_name)
       
                    
        
        conn.commit()
        log_string = 'Finished processing measurements'
        hf.log_with_timestamp(log_string, gui_logger)
        
        
    except Error as e:

        log_string = 'Failed to generate new HpCFs'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=e)#log error
            
        return False
        
        


            
            

def hpcf_generate_variants(conn, gui_logger=None):
    """
    Function hpcf_generate_variants
        searches DB for HpCFs with multiple frequency response variants then generates averages for each variant and inserts into database
    """
    
    try:

        log_string = 'Searching HpCF database for variants'
        hf.log_with_timestamp(log_string, gui_logger)

        #retrieve geq frequency list as an array - 127 bands
        geq_set_f_127 = hpcf_retrieve_set_geq_freqs(f_set=1)
        #retrieve geq frequency list as an array - 31 bands
        geq_set_f_31 = hpcf_retrieve_set_geq_freqs(f_set=2)
        #retrieve geq frequency list as an array - 103 bands
        geq_set_f_103 = hpcf_retrieve_set_geq_freqs(f_set=3)
        #impulse
        impulse=np.zeros(CN.N_FFT)
        impulse[0]=1
        fr_flat_mag = np.abs(np.fft.fft(impulse))
        fr_flat = hf.mag2db(fr_flat_mag)
        
        now_datetime = datetime.now()
 
        #get list of all headphones in the DB
        headphone_list = get_all_headphone_list(conn)
        
        #check one brand for testing
        #headphone_list=get_headphone_list(conn, brand='Beyerdynamic')
        
        #for each headphone, grab all samples
        for h in headphone_list:

            sample_list = get_hpcf_samples_dicts(conn, h)
            #loop through samples to get number of samples
            num_samples=0
            for s in sample_list:
                sample_dict = dict(s)
                #get number of samples
                sample_id = sample_dict['sample_id']
                if sample_id > num_samples:
                    num_samples = sample_id

            #continue if there are more than 1 samples for this headphone
            if num_samples > 1:

                #create new dict list for condensed set
                hpcf_variant_dict_list = []
                hpcf_data_dict_list = []

                #loop through each sample filter and identify variants
                for s in sample_list:
                    sample_dict = dict(s)
                    sample_id = sample_dict['sample_id']
                    #only consider samples that are non averages
                    if sample_id > 0:
 
                        sample_fir_json = sample_dict['fir']
                        sample_fir_list = json.loads(sample_fir_json) 
                        sample_fir = np.array(sample_fir_list)
                        
                        #get frequency response of current sample
                        #zero pad
                        data_pad=np.zeros(65536)
                        data_pad[0:(CN.HPCF_FIR_LENGTH-1)]=sample_fir[0:(CN.HPCF_FIR_LENGTH-1)]
                        hpcf_current = data_pad
                        hpcf_current_fft = np.fft.fft(hpcf_current)
                        hpcf_current_mag_fft=np.abs(hpcf_current_fft)
                        hpcf_current_db_fft = hf.mag2db(hpcf_current_mag_fft)
                        db_response_curr = hpcf_current_db_fft

                        #reset variant to 0
                        variant=0
                        max_variant=1
                        max_diff_list = []
                        
                        #sample 1 will aways be variant 1
                        if sample_id == 1:
                            variant=1
                        else:    
                            #objective: find a previous sample where no significant differences exist in response
                            #loop through all previous samples
                            for p in hpcf_data_dict_list:
                                db_response_prev = p['db_response']
                                variant_prev = p['variant']
                                #find the largest variant ID
                                if variant_prev>max_variant:
                                    max_variant=variant_prev
        
                                #subtract db response of prev sample from current sample
                                db_response_diff = np.subtract(db_response_curr,db_response_prev)
                                
                                #keep only frequencies below a cutoff
                                lf_cutoff = 200
                                hf_cutoff = 13500
                                lf_cutoff_fb = round(lf_cutoff*65536/44100)
                                hf_cutoff_fb = round(hf_cutoff*65536/44100)
                                db_response_diff_c=np.zeros(65536)
                                db_response_diff_c[lf_cutoff_fb:hf_cutoff_fb] = db_response_diff[lf_cutoff_fb:hf_cutoff_fb]
             
                                max_diff = np.max(np.abs(db_response_diff_c))
                                
                                #if largest peak is less than x dB, consider this a similar shape and same variant
                                if max_diff < 10:
                                    diff_dict = {'variant': variant_prev, 'max_diff': max_diff}
                                    max_diff_list.append(diff_dict)
                                    
                                #if multiple similar samples were found, pick the one with smallest difference
                                smallest_diff=100
                                for p in max_diff_list:
                                    max_diff = p['max_diff']
                                    variant_prev = p['variant']
                                    if max_diff < smallest_diff:
                                        smallest_diff=max_diff
                                        variant=variant_prev
       
                            #a similar sample was not found. Create new variant
                            if variant == 0:
                                variant = max_variant+1
                                                 
                        variant_dict = {'sample_id': sample_id, 'variant': variant}
                        data_dict = {'sample_id': sample_id, 'variant': variant, 'db_response':db_response_curr}
                        #add to new dict list
                        hpcf_variant_dict_list.append(variant_dict)    
                        hpcf_data_dict_list.append(data_dict)    
 
                #finished compiling variant data into dict list for this headphone
                variant_array = np.zeros(30)
                
                #loop through variant dict list to get sample count of each variant
                for v in hpcf_variant_dict_list:
                    # json_string = json.dumps(v)
                    # log_string = 'Headphone: ' + h + ' Results: ' + json_string
                    # if CN.LOG_INFO == True:
                    #     logging.info(log_string)
                    # if CN.LOG_GUI == True and gui_logger != None:
                    #     gui_logger.log_info(log_string)
                    
                    variant = v['variant']
                    #increase count
                    variant_array[variant]=variant_array[variant]+1
 
                num_val_variants=0
                min_var_count=2
                min_val_variants=2

                #loop through variant array to determine how many valid variants exist
                for v in variant_array:
                    #v is count of variants, index is variant
                    if v >= min_var_count:
                        num_val_variants=num_val_variants+1
                
                #only proceed if there are at least 2 valid variants
                if num_val_variants >= min_val_variants:
                    #for each valid variant
                    for idx, v in enumerate(variant_array):
                        #v is count of variants, index is variant
                        if v >= min_var_count:

                            #loop through samples to generate new average hpcf
                            num_samples_avg = 0
                            hpcf_fft_avg_db = fr_flat.copy()
            
                            #generate average spectrum of each relevant sample filter
                            for s in sample_list:
                                sample_dict = dict(s)
                                sample_id = sample_dict['sample_id']
                                sample_variant=0
                                #only consider samples that are non averages
                                if sample_id > 0:
                                    #grab variant for this sample
                                    for v in hpcf_variant_dict_list:
                                        s_id=v['sample_id']
                                        if s_id == sample_id:
                                            sample_variant = v['variant']
                                    #proceed if this sample belongs to this variant
                                    if sample_variant == idx:
                                        sample_fir_json = sample_dict['fir']
                                        sample_fir_list = json.loads(sample_fir_json) 
                                        sample_fir = np.array(sample_fir_list)
                                        #zero pad
                                        data_pad=np.zeros(65536)
                                        data_pad[0:(CN.HPCF_FIR_LENGTH-1)]=sample_fir[0:(CN.HPCF_FIR_LENGTH-1)]
                                        hpcf_current = data_pad
                                        hpcf_current_fft = np.fft.fft(hpcf_current)
                                        hpcf_current_mag_fft=np.abs(hpcf_current_fft)
                                        hpcf_current_db_fft = hf.mag2db(hpcf_current_mag_fft)
                                        hpcf_fft_avg_db = np.add(hpcf_fft_avg_db,hpcf_current_db_fft)
                                        num_samples_avg = num_samples_avg+1
             
                            #divide by total number of samples
                            hpcf_fft_avg_db = hpcf_fft_avg_db/num_samples_avg
                            #convert to mag
                            hpcf_fft_avg_mag = hf.db2mag(hpcf_fft_avg_db)
                            #create min phase FIR
                            hpcf_avg_fir_full = hf.mag_to_min_fir(hpcf_fft_avg_mag, out_win_size=CN.HPCF_FIR_LENGTH)
                            #store in cropped array
                            hpcf_avg_fir_array=np.zeros((CN.HPCF_FIR_LENGTH))
                            hpcf_avg_fir_array[:] = np.copy(hpcf_avg_fir_full[0:CN.HPCF_FIR_LENGTH])

                            ### get fir
                            # convert from array to list
                            fir_list = hpcf_avg_fir_array.tolist()
                            # convert from list to Json
                            fir_json_str = json.dumps(fir_list)
                                         
                            #get graphic eq filter 127 band
                            #geq_str = hpcf_fir_to_geq(fir_array=hpcf_avg_fir_array,geq_mode=2,sample_rate=CN.SAMP_FREQ,geq_freq_arr=geq_set_f_127)
                            #20250408: no longer required to store geq data. Replace with empty string
                            geq_str = ''
                            #get graphic eq filter 31 band
                            #geq_31_str = hpcf_fir_to_geq(fir_array=hpcf_avg_fir_array,geq_mode=2,sample_rate=CN.SAMP_FREQ,geq_freq_arr=geq_set_f_31)
                            #20250408: no longer required to store geq data. Replace with empty string
                            geq_31_str = ''
                            #get graphic eq filter 103 band
                            #geq_103_str = hpcf_fir_to_geq(fir_array=hpcf_avg_fir_array,geq_mode=2,sample_rate=CN.SAMP_FREQ,geq_freq_arr=geq_set_f_103)
                            #20250408: no longer required to store geq data. Replace with empty string
                            geq_103_str = ''
                            #last modified text
                            created_on = now_datetime
                    
                            #average_variant hpcf is now created
                            #calc new sample name
                            sample_name_var = 'Average Variant '+str(idx)
                            #delete existing one if exists
                            delete_headphone_sample(conn, h, sample_name_var, gui_logger=None)
                            #insert new hpcf into database
                            #get brand
                            brand = get_brand(conn, h)
                            #create tuple
                            hpcf_to_insert = (brand, h, sample_name_var, 0, fir_json_str, geq_str, geq_31_str, geq_103_str, created_on)
                            # create entry
                            create_hpcf_row(conn, hpcf_to_insert, brand, h, sample_name_var, gui_logger=gui_logger)
                            
                
        log_string = 'Completed Search'
        hf.log_with_timestamp(log_string, gui_logger)
                
        conn.commit()
        
    except Error as e:
        logging.error("Error occurred", exc_info = e)

    


def get_recent_hpcfs(conn, date_str ='2025-01-01', gui_logger=None):
    """
    Function get_recent_hpcfs
        searches DB for HpCFs created after a certain date
    """
    
    try:

        #get list of all headphones in the DB
        hpcf_list = get_hpcf_date_range_dicts(conn, date_str)
        if hpcf_list == None:
            log_string = 'No HpCFs found'
        else:
            #for each headphone, grab all samples
            for h in hpcf_list:
     
                hpcf_dict = dict(h)
                
                json_string = json.dumps(hpcf_dict)
                
                log_string = 'New HpCF: ' + json_string
                hf.log_with_timestamp(log_string, gui_logger)

    
    except Error as e:
        logging.error("Error occurred", exc_info = e)


def check_for_database_update(conn, gui_logger=None):
    """ 
    Function finds version of latest database, compares with current version
    """
    
    try:
        
        #get version of local database
        json_fname = pjoin(CN.DATA_DIR_OUTPUT, 'hpcf_database_metadata.json')
        with open(json_fname) as fp:
            _info = json.load(fp)
        local_db_version = _info['version']
        
        #log results
        log_string = 'Checking for Headphone Correction Filter dataset updates'
        hf.log_with_timestamp(log_string, gui_logger)
        
        #log results
        log_string = 'Current dataset version: ' + str(local_db_version)
        hf.log_with_timestamp(log_string, gui_logger)
            
        #log results
        log_string = 'Finding latest version'
        hf.log_with_timestamp(log_string, gui_logger)
            
        #get version of online database
        url = "https://drive.google.com/file/d/1lcJrNhusYq1g-M8As1JHwHIYYAXRhr-X/view?usp=drive_link"
        output = pjoin(CN.DATA_DIR_OUTPUT, 'hpcf_database_metadata_latest.json')
        gdown.download(url, output, fuzzy=True)
        
        #read json
        json_fname = output
        with open(json_fname) as fp:
            _info = json.load(fp)
        web_db_version = _info['version']
        
        #log results
        log_string = 'Latest dataset version: ' + str(web_db_version)
        hf.log_with_timestamp(log_string, gui_logger)
            
        if local_db_version == web_db_version:
            #log results
            log_string = 'No update available'
        else:
            log_string = "New version available. Click 'Download Latest Dataset' to update"
        hf.log_with_timestamp(log_string, gui_logger)
        
        return True
    

        
    except Error as e:

        log_string = 'Failed to check HpCF versions'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=e)#log error
            
        return False


def downlod_latest_database(conn, gui_logger=None):
    """ 
    Function downloads and replaces current database
    """
    
    try:
  
        log_string = 'Downloading latest HpCF database'
        hf.log_with_timestamp(log_string, gui_logger)
  
        #download latest version of database
        url = "https://drive.google.com/file/d/1car3DqHNqziJbgduV4VsVngyBgaTTY2I/view?usp=drive_link"
        output = pjoin(CN.DATA_DIR_OUTPUT,'hpcf_database.db')
        gdown.download(url, output, fuzzy=True)

        #also download metadata
        url = "https://drive.google.com/file/d/1lcJrNhusYq1g-M8As1JHwHIYYAXRhr-X/view?usp=drive_link"
        output = pjoin(CN.DATA_DIR_OUTPUT, 'hpcf_database_metadata.json')
        gdown.download(url, output, fuzzy=True)

        #log results
        log_string = 'Latest HpCF database downloaded to: ' + str(output)
        hf.log_with_timestamp(log_string, gui_logger)

        return True
    
    except Error as e:
 
        log_string = 'Failed to download latest HpCF database'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=e)#log error
            
        return False
            
            
            

def check_for_app_update(gui_logger=None):
    """ 
    Function finds version of latest app, compares with current version
    """
    
    try:
        
        with open(CN.METADATA_FILE) as fp:
            _info = json.load(fp)
        __version__ = _info['version']
        
        #log results
        log_string = 'Checking for app updates'
        hf.log_with_timestamp(log_string, gui_logger)
        
        #log results
        log_string = 'Current ASH Toolset version: ' + str(__version__)
        hf.log_with_timestamp(log_string, gui_logger)
            
        #log results
        log_string = 'Finding latest version...'
        hf.log_with_timestamp(log_string, gui_logger)
            
        #get version of online database
        url = "https://raw.githubusercontent.com/ShanonPearce/ASH-Toolset/main/metadata.json"
        output = pjoin(CN.DATA_DIR_EXT, 'metadata_latest.json')
        urllib.request.urlretrieve(url, output)
   
        #read json
        json_fname = output
        with open(json_fname) as fp:
            _info = json.load(fp)
        web_app_version = _info['version']
        
        #log results
        log_string = 'Latest ASH Toolset version: ' + str(web_app_version)
        hf.log_with_timestamp(log_string, gui_logger)
            
        if __version__ == web_app_version:
            #log results
            log_string = 'No update required'
        else:
            log_string = "New version available at https://sourceforge.net/projects/ash-toolset/"
        hf.log_with_timestamp(log_string, gui_logger)
        
        return True
    
    except Error as e:
        
        log_string = 'Failed to check app versions'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=e)#log error
            
        return False

def remove_hpcfs(primary_path, gui_logger=None):
    """
    Function deletes HpCFs and E-APO configs stored in a specified directory
    """
    out_file_dir_hpcf = pjoin(primary_path, CN.PROJECT_FOLDER_HPCFS)
    output_config_path = pjoin(primary_path, CN.PROJECT_FOLDER_CONFIGS_HPCF)
    
    try:
        
        if os.path.exists(out_file_dir_hpcf):
            shutil.rmtree(out_file_dir_hpcf)
            log_string = 'Deleted folder and contents: ' + out_file_dir_hpcf 
            hf.log_with_timestamp(log_string, gui_logger)
                
        if os.path.exists(output_config_path):
            shutil.rmtree(output_config_path)
            log_string = 'Deleted folder and contents: ' + output_config_path
            hf.log_with_timestamp(log_string, gui_logger)    
                
                
    
    except Exception as ex:
        log_string = 'Failed to delete folders: ' + out_file_dir_hpcf + ' & ' + output_config_path
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
            
def remove_select_hpcfs(primary_path, headphone, gui_logger=None):
    """
    Function deletes HpCFs stored in a specified directory
    """
    out_file_dir_hpcf = pjoin(primary_path, CN.PROJECT_FOLDER_HPCFS)
    original_hp_name =  headphone.replace(" ", "_")
    
    #delimiters
    delimiter_a = '_Sample'
    delimiter_b = '_Average'
    
    try:
 
        for root, dirs, file in os.walk(out_file_dir_hpcf):
            for filename in file:
                
                name_before_a = filename.split(delimiter_a)[0]
                name_before_sample = name_before_a.split(delimiter_b)[0]
                if original_hp_name in filename and name_before_sample == original_hp_name:
                    file_path=os.path.join(root, filename)
                    os.remove(file_path) # delete file based on their name
                    log_string = 'Deleted file: ' + file_path 
                    hf.log_with_timestamp(log_string, gui_logger)   
        
 
  
    except Exception as ex:
        log_string = 'Failed to delete folder: ' + out_file_dir_hpcf
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
            
 
    
def crop_hpcf_firs(conn, gui_logger=None):
    """
    Function iterates through all FIRs in the database and crops to X samples
    """
    
    try:
        #fade out window
        data_pad_ones=np.ones(CN.HPCF_FIR_LENGTH)
        data_pad_zeros=np.zeros(CN.HPCF_FIR_LENGTH)
        out_win_size=50
        fade_hanning_size=out_win_size*2
        fade_hanning_start=CN.HPCF_FIR_LENGTH-60
        hann_fade_full=np.hanning(fade_hanning_size)
        hann_fade = np.split(hann_fade_full,2)[1]
        fade_out_win = data_pad_ones.copy()
        fade_out_win[fade_hanning_start:fade_hanning_start+int(fade_hanning_size/2)] = hann_fade
        fade_out_win[fade_hanning_start+int(fade_hanning_size/2):]=data_pad_zeros[fade_hanning_start+int(fade_hanning_size/2):]
        
        #for each hpcf in database

        #get list of all headphones in the DB
        headphone_list = get_all_headphone_list(conn)
      
        #for each headphone, grab all samples
        for h in headphone_list:
            sample_list = get_hpcf_samples_dicts(conn, h)
            #for each sample
            for s in sample_list:
                
                sample_dict = dict(s)
                #grab fir
                sample_fir_json = sample_dict['fir']
                sample_name = sample_dict['sample']
                created_on = sample_dict['created_on']
                geq_str = sample_dict['graphic_eq']
                geq_31_str = sample_dict['graphic_eq_31']
                geq_103_str = sample_dict['graphic_eq_103']
                sample_fir_list = json.loads(sample_fir_json) 
                sample_fir = np.array(sample_fir_list)

                #check length
                if sample_fir.size > CN.HPCF_FIR_LENGTH:
 
                    #crop to 512 samples
                    hpcf_new_fir_array=np.zeros((CN.HPCF_FIR_LENGTH))
                    hpcf_new_fir_array[:] = np.copy(sample_fir[0:CN.HPCF_FIR_LENGTH])
        
                    #apply window
                    hpcf_new_fir_array = np.multiply(hpcf_new_fir_array,fade_out_win)
        
                    #replace existing hpcf
                    # convert from array to list
                    fir_list = hpcf_new_fir_array.tolist()
                    # convert from list to Json
                    fir_json_str = json.dumps(fir_list)
   
                    #case for updating existing hpcf
                    hpcf_data = (fir_json_str,geq_str,geq_31_str,geq_103_str,created_on)
                    replace_hpcf_filter_data(conn,hpcf_data,h,sample_name, gui_logger=gui_logger)
                    
        conn.commit()
    
    except Error as e:
        logging.error("Error occurred", exc_info = e)
    

def process_mono_hp_cues(conn, measurement_folder_name, in_ear_set = 0, gui_logger=None):
    """
    Function performs statistical analysis on monoaural cues from headphone measurements
    """
    
    try:
        
        log_string = 'Starting hp cues analysis'
        hf.log_with_timestamp(log_string, gui_logger)
        
        if in_ear_set == 1:
            hp_folder = 'in_ear'
        else:
            hp_folder = 'over_ear'   
  
        #directories
        data_raw_directory = CN.DATA_DIR_RAW_HP_MEASRUEMENTS
        measurement_directory = pjoin(CN.DATA_DIR_RAW_HP_MEASRUEMENTS, hp_folder, measurement_folder_name)
        
        #impulse
        impulse=np.zeros(CN.N_FFT)
        impulse[0]=1
        fr_flat_mag = np.abs(np.fft.fft(impulse))
        fr_flat = hf.mag2db(fr_flat_mag)
    
        
        #create array for new x axis for interpolation
        freq_max = int(CN.SAMP_FREQ/2)
        #in np fft array where nfft = 65536, id 0 and id 32768 are unique. Create 32769 unique values
        num_samples = int(CN.N_FFT/2)+1
        xnew = np.linspace(0, freq_max, num=num_samples)
        
        #read in-ear headphone equalisation from WAV file
        if CN.APPLY_ADD_HP_EQ > 0:
            filename = 'diffuse_field_eq_for_in_ear_headphones.wav'
            wav_fname = pjoin(CN.DATA_DIR_INT, filename)
            samplerate, data_addit_eq = wavfile.read(wav_fname)
            data_addit_eq = data_addit_eq / (2.**31)
            #convert to mag response
            in_ear_eq_fir=np.zeros(CN.N_FFT)
            in_ear_eq_fir[0:1024] = data_addit_eq[0:1024]
            data_fft = np.fft.fft(in_ear_eq_fir)#
            in_ear_eq_mag=np.abs(data_fft)
            in_ear_eq_db=hf.mag2db(in_ear_eq_mag)
        
        #variable to identify if set contains left and right samples
        left_and_right_set = 0
        left_meas_count = 0
        right_meas_count = 0
        
        #
        #read metadata from csv
        #
        metadata_file_name = measurement_folder_name + '_summary.csv'
        metadata_file = pjoin(data_raw_directory, metadata_file_name)
        metadata_dict_list = []
        with open(metadata_file, encoding='utf-8-sig', newline='') as inputfile:
            reader = DictReader(inputfile)
            for row in reader:#rows 2 and onward
                #store each row as a dictionary

                #append to list of dictionaries
                metadata_dict_list.append(row)  
                
                #check if set contains left and right samples
                file_name = row.get('file_name')
                if 'L.txt' in file_name or 'l.txt' in file_name or 'l1.txt' in file_name or 'L1.txt' in file_name:
                    left_meas_count = left_meas_count+1
                if 'R.txt' in file_name or 'r.txt' in file_name or 'r1.txt' in file_name or 'R1.txt' in file_name:
                    right_meas_count = right_meas_count+1
                
        #check if set contains left and right samples
        if left_meas_count > 0 and right_meas_count > 0:
            left_and_right_set=1
        
        #create array to fill with frequency data
        hpcf_data_dict_list = metadata_dict_list
        
        #
        #read measurements from txt
        #
        for root, dirs, files in os.walk(measurement_directory):
            for filename in sorted(files):
                for hpcf_dict in hpcf_data_dict_list:
                    file_name_hpcf = hpcf_dict.get('file_name')
                    
                    #logging.info(filename)
                    
                    if file_name_hpcf == filename:
                        #read txt file and store data into array
                        txt_fname = pjoin(root, filename)
                        try:
                            data = np.loadtxt(txt_fname, comments='*')
                        except:
                            try:
                                data = np.loadtxt(txt_fname, comments='*', delimiter=';')
                            except:
                                data = np.loadtxt(txt_fname, comments='*', delimiter=',')
                 
                        #1-D data interpolation
                        data_x = data[:,0]
                        data_y = data[:,1]
                        spl = CubicSpline(data_x, data_y)
                        data_y_new = spl(xnew)

                        #fill entire spectrum
                        data_y_full=np.zeros(CN.N_FFT)
                        reversed_arr = np.flipud(data_y_new)
                        neg_f_ind_start = int(CN.N_FFT/2)+1
                        rev_arr_ind_start = 1#ignore nyquist f
                        rev_arr_ind_end = int(CN.N_FFT/2) #ignore DC
                        data_y_full[0:neg_f_ind_start] = data_y_new
                        data_y_full[neg_f_ind_start:] = reversed_arr[1:rev_arr_ind_end]
               
                        #add to current dictionary
                        hpcf_dict['meas_resp_db_intrp'] = data_y_full
  
        #condense list if this is a left/right set
        if left_and_right_set == 1:
            #create new dict list for condensed set
            hpcf_data_dict_list_c = []
            #iterate through full list
            for hpcf_dict_l in hpcf_data_dict_list:
                extract_name = hpcf_dict_l.get('extracted_name')
                file_name_l = hpcf_dict_l.get('file_name')
       
                #continue if this is a L measurement
                if 'L.txt' in file_name_l or 'l.txt' in file_name or 'l1.txt' in file_name_l or 'L1.txt' in file_name_l:
                    #grab response for L measurement
                    resp_db_l = hpcf_dict_l.get('meas_resp_db_intrp')
                    #find R measurement
                    for hpcf_dict_r in hpcf_data_dict_list:
                        extract_name_r = hpcf_dict_r.get('extracted_name')
                        file_name_r = hpcf_dict_r.get('file_name')
                        #continue if this is a R measurement
                        if ('R.txt' in file_name_r or 'r.txt' in file_name_r or 'r1.txt' in file_name or 'R1.txt' in file_name) and (extract_name_r == extract_name):
                            #grab response for R measurement
                            resp_db_r = hpcf_dict_r.get('meas_resp_db_intrp')
                    #calculate average response
                    resp_db_avg = np.divide(np.add(resp_db_l,resp_db_r),2)
                    #create new dictionary
                    dict_new = hpcf_dict_l.copy()
                    dict_new.update({"meas_resp_db_intrp": resp_db_avg})
                    #add to new dict list
                    hpcf_data_dict_list_c.append(dict_new)
            
        else:
            hpcf_data_dict_list_c=hpcf_data_dict_list
        
        #
        #calculate target headphone
        #
        hpcf_fft_avg_db = fr_flat.copy()
        num_headphones = 0
        for hpcf_dict in hpcf_data_dict_list_c:
            #get db response
            resp_db = hpcf_dict.get('meas_resp_db_intrp')
            #add db response to average
            hpcf_fft_avg_db = np.add(hpcf_fft_avg_db,resp_db)
            
            num_headphones = num_headphones+1
            
        #divide by total number of samples
        hpcf_fft_avg_db = hpcf_fft_avg_db/num_headphones
        hpcf_target_db = hpcf_fft_avg_db.copy()
        
        #apply in ear compensation to target if in_ear_set
        if in_ear_set == 1:
            #apply additional EQ for in-ear headphones
            hpcf_target_db_comp = np.add(hpcf_target_db,in_ear_eq_db)
        else:
            hpcf_target_db_comp = hpcf_target_db.copy()
            
        #apply smoothing
        #convert to mag
        hpcf_target_mag = hf.db2mag(hpcf_target_db_comp)
        #level ends of spectrum
        hpcf_target_mag = hf.level_spectrum_ends(hpcf_target_mag, 50, 19000, smooth_win = 7)#
        #smoothing
        hpcf_target_mag = hf.smooth_fft_octaves(hpcf_target_mag)
            
        if CN.PLOT_ENABLE == True:
            hf.plot_data(hpcf_target_mag,'hpcf_target_mag ' + measurement_folder_name, normalise=0)

        #
        #save numpy array for later use
        #
        ir_set='hp_cues'
        data_out_folder = pjoin(CN.DATA_DIR_INT, 'mono_cues', ir_set,hp_folder)
        npy_file_name = hp_folder + '_' + measurement_folder_name + '_mean_hp.npy'
        out_file_path = pjoin(data_out_folder,npy_file_name)      
          
        output_file = Path(out_file_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        np.save(out_file_path,hpcf_target_mag)    
        
        log_string = 'Exported numpy file to: ' + out_file_path 
        hf.log_with_timestamp(log_string, gui_logger)




    except Exception as ex:
        log_string = 'Failed to complete analysis'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error