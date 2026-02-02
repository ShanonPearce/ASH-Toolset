# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:55:01 2024

@author: Shanon
"""

# import packages
from os.path import join as pjoin
import os
from pathlib import Path
import logging
from datetime import date
from datetime import datetime
from ash_toolset import constants as CN
import sqlite3
from sqlite3 import Error
import json
from csv import DictReader
import numpy as np
from ash_toolset import helper_functions as hf
import csv
import string
import dearpygui.dearpygui as dpg
import shutil
import re
import soundfile as sf
from collections import defaultdict
from collections import Counter
import itertools
import difflib

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

#
##### database management
#





##### new retrieval functions, compatible with both DB schemas

def detect_schema(conn):
    """Detect whether we're using the new or old schema."""
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(hpcf_table)")
    cols = [r[1] for r in cur.fetchall()]
    cur.close()
    return "source" in cols  # True if new schema

def map_row_to_standard_dict(row, new_schema=False):
    """Return a dict with guaranteed standard keys."""
    d = dict(row)
    if new_schema:
        # rename keys per SCHEMA_MAP
        mapped = {CN.HPCF_DB_SCHEMA_MAP.get(k, k): v for k, v in d.items()}
    else:
        mapped = d

    # ensure all STANDARD_COLUMNS exist
    for key in CN.HPCF_DB_STANDARD_COLUMNS:
        if key not in mapped:
            mapped[key] = None
    return mapped

# ---------------------------------
# List retrieval functions using HPCF_DB_SCHEMA_MAP with schema detection
# ---------------------------------



def get_all_headphone_list(conn):
    """Retrieve list of all headphones."""
    try:
        is_new = detect_schema(conn)
        col = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "headphone"][0] if is_new else "headphone"
        sql = f"SELECT DISTINCT {col} FROM hpcf_table"
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return sorted([r[0] for r in rows], key=str.casefold) if rows else []
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info=e)
        return []


def get_brand_list(conn):
    """Retrieve list of all brands or sources."""
    try:
        is_new = detect_schema(conn)
        col = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "brand"][0] if is_new else "brand"
        sql = f"SELECT DISTINCT {col} FROM hpcf_table"
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return sorted([r[0] for r in rows], key=str.casefold) if rows else []
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info=e)
        return []


def search_brand_list(conn, search_str=None):
    """Retrieve list of brands or sources matching a search string."""
    try:
        if not search_str:
            return []
        is_new = detect_schema(conn)
        col = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "brand"][0] if is_new else "brand"
        sql = f"SELECT DISTINCT {col} FROM hpcf_table WHERE {col} LIKE ?"
        cur = conn.cursor()
        cur.execute(sql, (f"%{search_str}%",))
        rows = cur.fetchall()
        cur.close()
        return sorted([r[0] for r in rows], key=str.casefold) if rows else []
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info=e)
        return []


def search_headphone_list(conn, search_str=None):
    """Retrieve list of headphones matching a search string."""
    try:
        if not search_str:
            return []
        is_new = detect_schema(conn)
        col = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "headphone"][0] if is_new else "headphone"
        sql = f"SELECT DISTINCT {col} FROM hpcf_table WHERE {col} LIKE ?"
        cur = conn.cursor()
        cur.execute(sql, (f"%{search_str}%",))
        rows = cur.fetchall()
        cur.close()
        return sorted([r[0] for r in rows], key=str.casefold) if rows else []
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info=e)
        return []


def search_headphones_in_list(conn, search_list=None):
    """Retrieve headphones from a given list."""
    try:
        if not search_list:
            return []
        is_new = detect_schema(conn)
        col = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "headphone"][0] if is_new else "headphone"
        placeholders = ",".join(["?"] * len(search_list))
        sql = f"SELECT DISTINCT {col} FROM hpcf_table WHERE {col} IN ({placeholders})"
        cur = conn.cursor()
        cur.execute(sql, search_list)
        rows = cur.fetchall()
        cur.close()
        return sorted([r[0] for r in rows], key=str.casefold) if rows else []
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info=e)
        return []


def get_headphone_list(conn, brand, sample=None):
    """Retrieve list of headphones for a given brand and optional sample/type."""
    try:
        is_new = detect_schema(conn)
        col_brand = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "brand"][0] if is_new else "brand"
        col_headphone = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "headphone"][0] if is_new else "headphone"
        col_sample = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "sample"][0] if is_new else "sample"

        cur = conn.cursor()
        if sample:
            sql = f"SELECT DISTINCT {col_headphone} FROM hpcf_table WHERE {col_brand}=? AND {col_sample}=?"
            cur.execute(sql, (brand, sample))
        else:
            sql = f"SELECT DISTINCT {col_headphone} FROM hpcf_table WHERE {col_brand}=?"
            cur.execute(sql, (brand,))
        rows = cur.fetchall()
        cur.close()
        return sorted([r[0] for r in rows], key=str.casefold) if rows else []
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info=e)
        return []


def get_brand(conn, headphone):
    """Retrieve brand for a specified headphone."""
    try:
        is_new = detect_schema(conn)
        col_brand = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "brand"][0] if is_new else "brand"
        col_headphone = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "headphone"][0] if is_new else "headphone"
        sql = f"SELECT DISTINCT {col_brand} FROM hpcf_table WHERE {col_headphone}=?"
        cur = conn.cursor()
        cur.execute(sql, (headphone,))
        rows = cur.fetchall()
        cur.close()
        return rows[0][0] if rows else ""
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info=e)
        return ""


def get_samples_list(conn, headphone):
    """Retrieve samples for a specified headphone."""
    try:
        is_new = detect_schema(conn)
        col_headphone = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "headphone"][0] if is_new else "headphone"
        col_sample = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "sample"][0] if is_new else "sample"
        sql = f"SELECT DISTINCT {col_sample} FROM hpcf_table WHERE {col_headphone}=?"
        cur = conn.cursor()
        cur.execute(sql, (headphone,))
        rows = cur.fetchall()
        cur.close()
        return sorted([r[0] for r in rows], key=str.casefold) if rows else []
    except sqlite3.Error as e:
        logging.error("Error occurred", exc_info=e)
        return []

def get_all_samples_list(conn):
    """Retrieve a distinct list of all samples from the database."""
    try:
        is_new = detect_schema(conn)
        col_sample = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "sample"][0] if is_new else "sample"

        sql = f"SELECT DISTINCT {col_sample} FROM hpcf_table"
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()

        if rows:
            return sorted([r[0] for r in rows if r[0] not in (None, "")], key=str.casefold)

    except sqlite3.Error as e:
        logging.error("Error retrieving sample list", exc_info=e)

    return []

# ---------------------------------
# Filter data retrieval
# ---------------------------------



def apply_target_curve_to_mag_db(
    mag_db: np.ndarray,
    freq_axis: np.ndarray,
    target: str,
    headphone_targets_db_path=CN.DATABASE_TARGET_DIR
) -> np.ndarray:
    """
    Apply a target curve from headphone_targets.db to the input mag_db.

    Args:
        mag_db (np.ndarray): Original magnitude response in dB
        freq_axis (np.ndarray): Frequency axis corresponding to mag_db
        target (str): Target name (matches target_name in target_difference_averages)
        headphone_targets_db_path (str): Path to headphone_targets.db

    Returns:
        np.ndarray: mag_db with target curve applied (dB)
    """


    # Return original if target is default
    if target == CN.HPCF_TARGET_LIST[0]:
        return mag_db.copy()

    if not isinstance(mag_db, np.ndarray):
        mag_db = np.array(mag_db, dtype=np.float64)

    conn = sqlite3.connect(headphone_targets_db_path)
    cur = conn.cursor()

    # ------------------------------------------------------------
    # 1. Load frequency axis from DB
    # ------------------------------------------------------------
    cur.execute("SELECT frequency FROM frequency_axis ORDER BY idx")
    db_freq_axis = np.array([row[0] for row in cur.fetchall()], dtype=np.float64)

    if len(db_freq_axis) != len(freq_axis) or not np.allclose(db_freq_axis, freq_axis, atol=1e-6):
        conn.close()
        raise ValueError(
            f"Frequency axis mismatch with headphone_targets.db "
            f"(len={len(freq_axis)} vs {len(db_freq_axis)})"
        )

    # ------------------------------------------------------------
    # 2. Load target difference curve
    # ------------------------------------------------------------
    cur.execute(
        "SELECT avg_diff_db FROM target_difference_averages WHERE target_name = ?",
        (target,)
    )
    row = cur.fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"Target '{target}' not found in target_difference_averages")

    target_curve = np.array(json.loads(row[0]), dtype=np.float64)

    if len(target_curve) != len(freq_axis):
        conn.close()
        raise ValueError(
            f"Target curve length {len(target_curve)} does not match freq_axis length {len(freq_axis)}"
        )

    conn.close()

    # ------------------------------------------------------------
    # 3. Apply target curve (dB addition)
    # ------------------------------------------------------------
    result_mag_db = mag_db + target_curve

    return result_mag_db




def get_hpcf_samples_dicts(conn, headphone, brand=None, build_fir=True,
                           fs=CN.FS, n_fft=CN.N_FFT, truncate_len=CN.HPCF_FIR_LENGTH, target=CN.HPCF_TARGET_LIST[0], smooth_hf=False):
    """
    Retrieve all sample filters for a given headphone.
    Optionally filter by brand to reduce duplicates.
    Returns a list of dicts with mag_db, frequency axis, type, and optionally FIR.
    For old schema, also includes sample_id.
    Optimized so that frequency axis is fetched only once.
    """
    try:
        if not headphone:
            return []

        is_new = detect_schema(conn)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Build query
        if is_new:
            col_brand = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "brand"][0]
            col_headphone = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "headphone"][0]
            col_sample = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "sample"][0]

            if brand:
                sql = f"""
                    SELECT {col_brand}, {col_headphone}, {col_sample}, type, mag_db
                    FROM hpcf_table
                    WHERE {col_headphone}=? AND {col_brand}=?
                """
                cur.execute(sql, (headphone, brand))
            else:
                sql = f"""
                    SELECT {col_brand}, {col_headphone}, {col_sample}, type, mag_db
                    FROM hpcf_table
                    WHERE {col_headphone}=?
                """
                cur.execute(sql, (headphone,))
        else:
            # old schema fallback
            if brand:
                sql = "SELECT brand, headphone, sample, sample_id, type, mag_db, created_on FROM hpcf_table WHERE headphone=? AND brand=?"
                cur.execute(sql, (headphone, brand))
            else:
                sql = "SELECT brand, headphone, sample, sample_id, type, mag_db, created_on FROM hpcf_table WHERE headphone=?"
                cur.execute(sql, (headphone,))

        rows = cur.fetchall()
        if not rows:
            cur.close()
            return []

        # Fetch frequency axis once
        cur.execute("SELECT frequency FROM frequency_axis ORDER BY idx ASC")
        freq_axis = np.array([row[0] for row in cur.fetchall()], dtype=np.float64)
        cur.close()

        results = []
        for r in rows:
            d = map_row_to_standard_dict(r, new_schema=is_new)
            d["freqs"] = freq_axis

            # Ensure type exists
            if "type" not in d and "type" in r.keys():
                d["type"] = r["type"]

            # Only include sample_id for old schema
            if not is_new and "sample_id" in r.keys():
                d["sample_id"] = r["sample_id"]
            elif is_new:
                d["sample_id"] = None

            if build_fir and "mag_db" in d:
                # Parse JSON string safely and convert to numeric array
                mag_db = np.array(json.loads(d["mag_db"]), dtype=np.float64)
                
                # ----------------------------------------------------
                # Apply target curve using helper
                # ----------------------------------------------------
                mag_db = apply_target_curve_to_mag_db(
                    mag_db=mag_db,
                    freq_axis=freq_axis,
                    target=target
                )
                
                # Convert dB to linear scale
                mag_linear = 10 ** (mag_db / 20.0)
          
                # Build FIR using pre-fetched freq_axis
                d["fir"] = hf.build_min_phase_filter(
                    smoothed_mag=mag_linear,
                    freq_axis=freq_axis,
                    fs=fs,
                    n_fft=n_fft,
                    truncate_len=truncate_len, hf_relax_to_zero=smooth_hf
                )

            results.append(d)

        return results

    except sqlite3.Error as e:
        logging.error("Error occurred in get_hpcf_samples_dicts", exc_info=e)

    return []


    
def get_hpcf_headphone_sample_dict(conn, headphone, sample, brand=None, build_fir=True,
                                   fs=CN.FS, n_fft=CN.N_FFT, truncate_len=CN.HPCF_FIR_LENGTH, target=CN.HPCF_TARGET_LIST[0], smooth_hf=False):
    """
    Retrieve filter data for a given headphone+sample pair.
    Optionally filter by brand to reduce duplicates.
    Returns a dict with mag_db, frequency axis, type, sample_id (old schema), and optionally FIR.
    """
    try:
        if not headphone or not sample:
            return {}

        is_new = detect_schema(conn)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        if is_new:
            col_brand = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "brand"][0]
            col_headphone = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "headphone"][0]
            col_sample = [k for k, v in CN.HPCF_DB_SCHEMA_MAP.items() if v == "sample"][0]

            if brand:
                sql = f"""
                    SELECT {col_brand}, {col_headphone}, {col_sample}, type, mag_db
                    FROM hpcf_table
                    WHERE {col_headphone}=? AND {col_sample}=? AND {col_brand}=?
                """
                cur.execute(sql, (headphone, sample, brand))
            else:
                sql = f"""
                    SELECT {col_brand}, {col_headphone}, {col_sample}, type, mag_db
                    FROM hpcf_table
                    WHERE {col_headphone}=? AND {col_sample}=?
                """
                cur.execute(sql, (headphone, sample))
        else:
            # old schema fallback includes sample_id
            if brand:
                sql = "SELECT brand, headphone, sample, sample_id, type, mag_db, created_on FROM hpcf_table WHERE headphone=? AND sample=? AND brand=?"
                cur.execute(sql, (headphone, sample, brand))
            else:
                sql = "SELECT brand, headphone, sample, sample_id, type, mag_db, created_on FROM hpcf_table WHERE headphone=? AND sample=?"
                cur.execute(sql, (headphone, sample))

        rows = cur.fetchall()
        if not rows:
            cur.close()
            return {}

        # Fetch frequency axis once
        cur.execute("SELECT frequency FROM frequency_axis ORDER BY idx ASC")
        freq_axis = np.array([row[0] for row in cur.fetchall()], dtype=np.float64)
        cur.close()

        d = map_row_to_standard_dict(rows[0], new_schema=is_new)
        d["freqs"] = freq_axis

        # Ensure type is included
        if "type" not in d and "type" in rows[0].keys():
            d["type"] = rows[0]["type"]

        # Include sample_id for old schema, None for new schema
        if not is_new and "sample_id" in rows[0].keys():
            d["sample_id"] = rows[0]["sample_id"]
        else:
            d["sample_id"] = None

        if build_fir and "mag_db" in d:
            # Parse JSON string safely and convert to numeric array
            mag_db = np.array(json.loads(d["mag_db"]), dtype=np.float64)
            
            # ----------------------------------------------------
            # Apply target curve using helper
            # ----------------------------------------------------
            mag_db = apply_target_curve_to_mag_db(
                mag_db=mag_db,
                freq_axis=freq_axis,
                target=target
            )
            
            # Convert dB to linear scale
            mag_linear = 10 ** (mag_db / 20.0)

            # Build FIR using pre-fetched freq_axis
            d["fir"] = hf.build_min_phase_filter(
                smoothed_mag=mag_linear,
                freq_axis=freq_axis,
                fs=fs,
                n_fft=n_fft,
                truncate_len=truncate_len, hf_relax_to_zero=smooth_hf
            )

        return d

    except sqlite3.Error as e:
        logging.error("Error occurred in get_hpcf_headphone_sample_dict", exc_info=e)

    return {}
    
   
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
##### misc functions     

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
        
        
        

#   
## data transformations
#        

def hpcf_fir_to_geq(fir_array, geq_mode=2, sample_rate=CN.SAMP_FREQ,
                    geq_freq_arr=None, output_string_type=2):
    """
    Converts an FIR to a graphic EQ using interpolation (mode 2 only).
    Obsolete mode 1 removed.
    
    :param fir_array: numpy array, FIR impulse response
    :param geq_mode: int, only mode 2 supported
    :param sample_rate: int, sampling frequency
    :param geq_freq_arr: numpy array of target GEQ center frequencies
    :param output_string_type: int, 1=E-APO string, 2=JSON string
    :return: string, GEQ data
    """
    try:
        if geq_mode != 2:
            raise ValueError("Only geq_mode=2 is supported now.")

        n_fft = len(fir_array) * 2  # zero-pad
        fir_padded = np.zeros(n_fft)
        fir_padded[:len(fir_array)] = fir_array
        fir_fft = np.fft.fft(fir_padded)
        mag = np.abs(fir_fft[:n_fft // 2 + 1])
        mag /= np.mean(mag[:200])
        mag_db = hf.mag2db(mag)
        freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)

        # Get GEQ frequency array
        if geq_freq_arr is None or len(geq_freq_arr) == 0:
            geq_freq_arr = hpcf_retrieve_set_geq_freqs()

        # Interpolate target magnitude
        geq_gains_db = np.interp(geq_freq_arr, freqs, mag_db)

        # Output string
        if output_string_type == 1:  # E-APO string
            geq_string = "GraphicEQ: " + "; ".join(
                f"{int(f)} {round(g,1)}" for f, g in zip(geq_freq_arr, geq_gains_db)
            )
            return geq_string
        elif output_string_type == 2:  # JSON string
            # Convert keys and values to native Python types
            dictionary = {float(f): float(round(g, 1)) for f, g in zip(geq_freq_arr, geq_gains_db)}
            return json.dumps(dictionary)
        else:
            logging.error("Invalid string type selected for GEQ")
            return None

    except Exception as ex:
        logging.error("Error occurred in hpcf_fir_to_geq", exc_info=ex)
        return None







def hpcf_to_file(hpcf_dict, primary_path, fir_export = True, fir_stereo_export = True, geq_export = True, geq_31_export = True, geq_103_export = False, hesuvi_export = True, geq_json = True, eapo_export=False, gui_logger=None, samp_freq=CN.SAMP_FREQ, bit_depth='PCM_24', resample_mode=CN.RESAMPLE_MODE_LIST[0], force_output=False):
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
        
        success = True   # <-- result flag


        #hesuvi path
        if 'EqualizerAPO' in primary_path:
            hesuvi_path = pjoin(primary_path,'HeSuVi')   
        else:
            hesuvi_path = pjoin(primary_path, CN.PROJECT_FOLDER,'HeSuVi')   
        
        #Directories
        brand = hpcf_dict.get('brand')
        brand_folder = brand.replace(" ", "_")
        
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
        hpcf_fir = hpcf_dict.get('fir')
        
        #resample if samp_freq is not 44100
        if samp_freq != CN.SAMP_FREQ:
            hpcf_fir = hf.resample_signal(hpcf_fir, new_rate = samp_freq, mode=resample_mode)
        
        out_file_path = pjoin(out_file_dir_wav, hpcf_name_wav)
        
        if fir_export == True:
            
            if not hf.check_write_permissions(out_file_dir_wav, gui_logger):
                # Skip exporting or raise exception
                return
            #create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            #dont write if file already exists          
            if force_output or hf.wav_needs_update(out_file_path, samp_freq, bit_depth):
                hf.write2wav(file_name=out_file_path, data=hpcf_fir, prevent_clipping=1, bit_depth=bit_depth, samplerate=samp_freq)
                log_string = 'HpCF (WAV FIR): ' + hpcf_name + ' saved to: ' + str(out_file_dir_wav)
                hf.log_with_timestamp(log_string, gui_logger)
            else:
                hf.log_with_timestamp(f"Skipped writing (unchanged wav file): {hpcf_name_wav}")
            
        

        
        #
        #save FIR to stereo WAV
        #

        out_file_path = pjoin(out_file_dir_st_wav, hpcf_name_wav)
        
        if fir_stereo_export == True:
        
            # build stereo array with real length
            fir_len = len(hpcf_fir)
            output_wav_s = np.column_stack((hpcf_fir, hpcf_fir))  # shape (N, 2)
        
            out_file_path = pjoin(out_file_dir_st_wav, hpcf_name_wav)
        
            if not hf.check_write_permissions(out_file_dir_st_wav, gui_logger):
                return
        
            # create dir if doesnt exist
            output_file = Path(out_file_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
        
            hf.write2wav(
                file_name=out_file_path,
                data=output_wav_s,
                prevent_clipping=1,
                bit_depth=bit_depth,
                samplerate=samp_freq
            )
        
            log_string = f'HpCF (WAV Stereo FIR): {hpcf_name} saved to: {out_file_dir_st_wav}'
            hf.log_with_timestamp(log_string, gui_logger)
                
        #
        #save Graphic EQ to txt file
        #
        

        out_file_path = pjoin(out_file_dir_geq, hpcf_name_geq)
        
        if geq_export == True:
            if not hf.check_write_permissions(out_file_dir_geq, gui_logger):
                # Skip exporting or raise exception
                return
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
            if not hf.check_write_permissions(out_file_dir_geq_31, gui_logger):
                # Skip exporting or raise exception
                return
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
            if not hf.check_write_permissions(out_file_dir_geq_103, gui_logger):
                # Skip exporting or raise exception
                return
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
            if not hf.check_write_permissions(out_file_folder, gui_logger):
                # Skip exporting or raise exception
                return
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
        success = False
        log_string = 'Failed to export HpCFs'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=ex)#log error
        
    return success   # <-- return the flag





        
def hpcf_to_file_bulk(conn, primary_path, headphone=None, fir_export = True, fir_stereo_export = True, geq_export = True, geq_31_export = True, geq_103_export = False, hesuvi_export = True, eapo_export=False, 
                      report_progress=0, gui_logger=None, samp_freq=CN.SAMP_FREQ, bit_depth='PCM_24', resample_mode=CN.RESAMPLE_MODE_LIST[0], force_output=False, fir_length=CN.HPCF_FIR_LENGTH):
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
  
    Bulk-export all filters for one or all headphones in the database.
    Returns True if all exports succeed, False otherwise.
    """

    try:
        any_failures = False

        # If no headphone specified, export all headphones
        if headphone is None:
            headphone_list = get_all_headphone_list(conn)
        else:
            headphone_list = [headphone]

        # Process each headphone
        for h in headphone_list:
            
            target = dpg.get_value("hpcf_target_curve")
            smooth_hf = dpg.get_value("hpcf_smooth_hf")
            sample_list = get_hpcf_samples_dicts(conn, h, truncate_len=fir_length, target=target, smooth_hf=smooth_hf)
            num_samples = len(sample_list)

            for index, s in enumerate(sample_list):
                sample_dict = dict(s)

                result = hpcf_to_file(sample_dict, primary_path=primary_path, fir_export=fir_export, fir_stereo_export=fir_stereo_export, geq_export=geq_export, resample_mode=resample_mode, 
                             geq_31_export=geq_31_export, geq_103_export=geq_103_export, hesuvi_export=hesuvi_export, eapo_export=eapo_export, gui_logger=gui_logger, samp_freq=samp_freq, 
                             bit_depth=bit_depth, force_output=force_output)

                if not result:
                    any_failures = True

                # Progress bar
                if report_progress > 0:
                    progress = (index + 1) / num_samples
                    if report_progress == 2:
                        dpg.set_value("fde_progress_bar_hpcf", progress)
                        dpg.configure_item("fde_progress_bar_hpcf", overlay=f"{int(progress*100)}%")
                    else:
                        dpg.set_value("e_apo_progress_bar_hpcf", progress)
                        dpg.configure_item("e_apo_progress_bar_hpcf", overlay=f"{int(progress*100)}%")
                        if progress == 1:
                            dpg.configure_item("e_apo_progress_bar_hpcf", overlay=CN.PROGRESS_FIN)

            # Write metadata (only once per headphone)
            if fir_export:
                brand = sample_dict.get("brand")
                headphone_name = sample_dict.get("headphone")

                brand_folder = brand.replace(" ", "_")
                out_file_dir_wav = pjoin(primary_path, CN.PROJECT_FOLDER_HPCFS, "FIRs", brand_folder)

                os.makedirs(out_file_dir_wav, exist_ok=True)

                metadata = {
                    "brand_or_type": brand,
                    "headphone": headphone_name,
                    "samp_freq": samp_freq,
                    "bit_depth": bit_depth,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "num_samples": num_samples,
                    "fir_length": fir_length,
                }

                json_filename = f"{headphone_name.replace(' ', '_')}_metadata.json"
                json_path = pjoin(out_file_dir_wav, json_filename)

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=4)

        # Summary log
        if any_failures:
            hf.log_with_timestamp("WARNING: One or more HpCF files failed to export.", gui_logger, log_type=1)

        # Return True = success / False = failure
        return not any_failures

    except Error as e:
        logging.error("Error occurred", exc_info=e)
        return False




    
    
    




def hpcf_to_plot(
    conn,
    headphone,
    sample,
    primary_path=CN.DATA_DIR_OUTPUT,
    save_to_file=0,
    plot_dest=0,
    brand=None,
    gui_logger=None
):
    """
    Function plots a FIR for a specified HpCF using DearPyGui.

    :param conn: SQLite connection
    :param headphone: string, name of headphone to plot
    :param sample: string, name of sample to plot
    :param primary_path: string, path to save plots to if save_to_file == 1
    :param save_to_file: int, 1 = save plot to a file
    :param plot_dest: int, 0 = matplotlib, 1 = DPG (series 1), 2 = DPG (series 2)
    :param brand: optional string, filter by brand/source
    :param gui_logger: optional GUI logger reference
    :param view: string, type of plot ("Magnitude Response", "Impulse Response", "Group Delay", "Decay")
    """

    try:
        # Get dictionary of HpCF data
        target = dpg.get_value("hpcf_target_curve")
        smooth_hf = dpg.get_value("hpcf_smooth_hf")
        hpcf_dict = get_hpcf_headphone_sample_dict(conn, headphone, sample, brand=brand, target=target, smooth_hf=smooth_hf)
        if hpcf_dict is None:
            hf.log_with_timestamp(f"No HpCF data found for {headphone} / {sample}", gui_logger)
            return

        # FIR array
        fir_to_plot = hpcf_dict.get('fir')

        # Plot title
        plot_title = f"{hpcf_dict.get('headphone')} {hpcf_dict.get('sample')} - {target}"
        
        view = dpg.get_value("plot_type")

        # Plot using the new generic function
        hf.plot_fir_generic(
            fir_array=fir_to_plot,
            view=view,
            title_name=plot_title,
            samp_freq=CN.SAMP_FREQ,
            n_fft=CN.N_FFT,
            normalise=1,
            level_ends=0,
            x_lim_adjust=True,
            x_lim_a=20,
            x_lim_b=20000,
            y_lim_adjust=True,
            y_lim_a=-20,
            y_lim_b=15,
            plot_dest=plot_dest
        )

        hf.log_with_timestamp(f"Plotted HpCF: {plot_title}", gui_logger)

    except Exception as ex:
        hf.log_with_timestamp(f"Error plotting HpCF for {headphone} / {sample}: {ex}", gui_logger)
        
        


def extract_base_name(file_name: str) -> str:
    """
    Strips common L/R suffixes (txt/csv, case-insensitive) 
    from headphone measurement filenames.
    Returns the base headphone name.
    """
    suffixes = [' l.txt', ' r.txt', ' l.csv', ' r.csv']
    file_name_lower = file_name.lower()
    for suffix in suffixes:
        if file_name_lower.endswith(suffix):
            return file_name[: -len(suffix)]  # preserve original case except suffix
    return file_name  # no suffix found



      



def get_recent_hpcfs(conn, date_str='2025-01-01', gui_logger=None, output_file="hpcf_list.txt"):
    """
    Function get_recent_hpcfs
        searches DB for HpCFs created after a certain date
        logs them and saves to a text file in the requested format
    """
    try:
        hpcf_list = get_hpcf_date_range_dicts(conn, date_str)
        if not hpcf_list:
            log_string = 'No HpCFs found'
            hf.log_with_timestamp(log_string, gui_logger)
            return

        with open(output_file, 'w', encoding='utf-8') as f:
            for h in hpcf_list:
                hpcf_dict = dict(h)
                
                # Convert dict to string without braces
                inner_string = ', '.join(f'"{k}": "{v}"' for k, v in hpcf_dict.items())
                log_string = f'New HpCF: {inner_string}'
                
                # Log
                hf.log_with_timestamp(log_string, gui_logger)
                
                # Write to file
                f.write(log_string + '\n')

    except Error as e:
        logging.error("Error occurred", exc_info=e)




def check_for_database_update(
    conn,
    gui_logger=None,
    download_if_update=False
):
    """
    Check both Main and Compilation HpCF database versions against their
    latest online versions.

    If download_if_update is True, downloads updated databases automatically.
    """
    try:
        databases = [
            {
                "name": "ASH Filters Database",
                "local_meta": pjoin(CN.DATA_DIR_OUTPUT, CN.HPCF_META_ASH_NAME),
                "remote_meta_url": CN.ASH_FILT_DB_META_URL,
                "remote_meta_local": pjoin(CN.DATA_DIR_OUTPUT,CN.HPCF_META_LAT_ASH_NAME),
            },
            {
                "name": "Compilation Database",
                "local_meta": pjoin(CN.DATA_DIR_OUTPUT,CN.HPCF_META_COMP_NAME),
                "remote_meta_url": CN.COMP_FILT_DB_META_URL,
                "remote_meta_local": pjoin(CN.DATA_DIR_OUTPUT,CN.HPCF_META_LAT_COMP_NAME),
            },
        ]

        updates_available = False

        for db in databases:
            hf.log_with_timestamp(
                f"Checking for {db['name']} dataset updates",
                gui_logger
            )

            # --- read local metadata ---
            if not os.path.exists(db["local_meta"]):
                hf.log_with_timestamp(
                    f"No local metadata found for {db['name']}",
                    gui_logger,
                    log_type=2
                )
                local_version = None
            else:
                with open(db["local_meta"], encoding="utf-8") as fp:
                    local_info = json.load(fp)
                local_version = local_info.get("version")

            hf.log_with_timestamp(
                f"Current {db['name']} version: {local_version}",
                gui_logger
            )

            # --- download remote metadata (always to file) ---
            response = hf.download_file(
                url=db["remote_meta_url"],
                save_location=db["remote_meta_local"],
                gui_logger=gui_logger
            )
            if response is not True:
                hf.log_with_timestamp(
                    f"Failed to download metadata for {db['name']}",
                    gui_logger,
                    log_type=2
                )
                continue

            with open(db["remote_meta_local"], encoding="utf-8") as fp:
                web_info = json.load(fp)
            web_version = web_info.get("version")

            hf.log_with_timestamp(
                f"Latest {db['name']} version: {web_version}",
                gui_logger
            )

            if local_version == web_version:
                hf.log_with_timestamp(
                    f"No update available for {db['name']}",
                    gui_logger
                )
            else:
                updates_available = True
                hf.log_with_timestamp(
                    f"New version available for {db['name']}",
                    gui_logger, log_type=1
                )

        # --- optionally download updates ---
        if updates_available and download_if_update:
            hf.log_with_timestamp(
                "Downloading updated HpCF databases...",
                gui_logger
            )
            return download_latest_database(conn, gui_logger)

        return updates_available

    except Error as e:
        hf.log_with_timestamp(
            log_string="Failed to check HpCF versions",
            gui_logger=gui_logger,
            log_type=2,
            exception=e
        )
        return False

def download_latest_database(conn, gui_logger=None):
    """
    Download and replace both Main and Compilation HpCF databases.
    """
    try:
        datasets = [
            {
                "name": "ASH Filters",
                "db_url": CN.ASH_FILT_DB_URL,
                "meta_url": CN.ASH_FILT_DB_META_URL,
                "db_local": pjoin(CN.DATA_DIR_OUTPUT, CN.HPCF_DATABASE_ASH_NAME),
                "meta_local": pjoin(CN.DATA_DIR_OUTPUT, CN.HPCF_META_ASH_NAME),
            },
            {
                "name": "Compilation",
                "db_url": CN.COMP_FILT_DB_URL,
                "meta_url": CN.COMP_FILT_DB_META_URL,
                "db_local": pjoin(CN.DATA_DIR_OUTPUT,CN.HPCF_DATABASE_COMP_NAME),
                "meta_local": pjoin(CN.DATA_DIR_OUTPUT,CN.HPCF_META_COMP_NAME),
            },
        ]

        for ds in datasets:
            hf.log_with_timestamp(
                f"Downloading latest {ds['name']} database...",
                gui_logger
            )

            response = hf.download_file(
                url=ds["db_url"],
                save_location=ds["db_local"],
                gui_logger=gui_logger
            )
            if response is not True:
                hf.log_with_timestamp(
                    f"{ds['name']} database download failed",
                    gui_logger,
                    log_type=2
                )
                return False

            response = hf.download_file(
                url=ds["meta_url"],
                save_location=ds["meta_local"],
                gui_logger=gui_logger
            )
            if response is not True:
                hf.log_with_timestamp(
                    f"{ds['name']} metadata download failed",
                    gui_logger,
                    log_type=2
                )
                return False

            hf.log_with_timestamp(
                f"{ds['name']} database updated successfully",
                gui_logger
            )

        hf.log_with_timestamp(
            "All available databases updated successfully.",
            gui_logger
        )
        return True

    except Error as e:
        hf.log_with_timestamp(
            log_string="Failed to download latest HpCF databases",
            gui_logger=gui_logger,
            log_type=2,
            exception=e
        )
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
            
 
   
    







    
    
    
    
    
    
    
    
    


def create_weighted_frequency_table_in_db(
    db_path,
    table_name,
    min_freq=20,
    max_freq=20000,
    total_points=190,
    rounding="int",
    replace_table=True
):
    """
    Create a weighted log-spaced frequency table with higher HF resolution.
    """

    if min_freq <= 0 or max_freq <= 0:
        raise ValueError("Frequencies must be > 0")

    if min_freq >= max_freq:
        raise ValueError("min_freq must be < max_freq")

    # Band split (Hz)
    bands = [
        (min_freq, 100, int(total_points * 0.15)),
        (100, 1000, int(total_points * 0.30)),
        (1000, max_freq, total_points)
    ]

    # Fix last band count so total matches exactly
    bands[2] = (
        bands[2][0],
        bands[2][1],
        total_points - bands[0][2] - bands[1][2]
    )

    freqs = []

    for f_min, f_max, n in bands:
        if n <= 0:
            continue
        band_freqs = np.logspace(
            np.log10(f_min),
            np.log10(f_max),
            num=n,
            endpoint=False
        )
        freqs.append(band_freqs)

    freqs = np.concatenate(freqs)

    # Optional rounding
    if rounding == "int":
        freqs = np.round(freqs).astype(int)
    elif rounding == "2dp":
        freqs = np.round(freqs, 2)

    # Remove duplicates & sort
    freqs = np.unique(freqs)

    # Write to DB
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if replace_table:
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")

    cur.execute(f"""
        CREATE TABLE {table_name} (
            idx INTEGER PRIMARY KEY,
            frequency REAL
        )
    """)

    cur.executemany(
        f"INSERT INTO {table_name} (frequency) VALUES (?)",
        [(float(f),) for f in freqs]
    )

    conn.commit()
    conn.close()

    print(
        f"Created weighted frequency table '{table_name}' "
        f"with {len(freqs)} frequencies in {db_path}"
    )
    
def create_diffuse_field_curves(
    wav_path,
    db_path,
    freq_table_name,
    table_name="diffuse_field_eq",
    sample_rate=44100,
    fft_size=65536,
    rounding=4
):
    """
    Load a min-phase WAV FIR, convert to magnitude (dB),
    sample at freq_table_name, and store 3 scaled curves in the database:
    - df_curve_db_partial (original strength = 0.67)
    - df_curve_db_full (1.0 strength)
    - df_curve_db_remainder (0.33 strength)
    """

    # Load FIR from WAV
    fir, sr = sf.read(wav_path)
    if fir.ndim > 1:
        fir = fir[:, 0]  # take first channel if stereo

    if sr != sample_rate:
        raise ValueError(f"Expected sample rate {sample_rate}, got {sr}")

    # FFT
    H = np.fft.rfft(fir, n=fft_size)
    mag = np.abs(H)
    mag[mag < 1e-12] = 1e-12
    mag_db = 20 * np.log10(mag)

    # Load frequency axis from DB
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    freqs = np.array([row[0] for row in cur.execute(
        f"SELECT frequency FROM {freq_table_name} ORDER BY idx"
    )])

    fft_freqs = np.fft.rfftfreq(fft_size, 1 / sample_rate)

    # Interpolate magnitude to freq axis
    mag_db_interp = np.interp(freqs, fft_freqs, mag_db)

    # Define scaling factors
    scales = {
        "df_curve_db_partial": 1.0,   # original strength
        "df_curve_db_full": 1.4925,      # 100% strength
        "df_curve_db_remainder": 0.5  # 33% strength
    }

    # Drop table if exists
    cur.execute(f"DROP TABLE IF EXISTS {table_name}")

    # Create new table
    cur.execute(f"""
        CREATE TABLE {table_name} (
            idx INTEGER PRIMARY KEY,
            df_curve_db_partial TEXT,
            df_curve_db_full TEXT,
            df_curve_db_remainder TEXT
        )
    """)

    # Apply scaling and rounding
    partial = np.round(mag_db_interp * scales["df_curve_db_partial"], rounding)
    full    = np.round(mag_db_interp * scales["df_curve_db_full"], rounding)
    remainder = np.round(mag_db_interp * scales["df_curve_db_remainder"], rounding)

    # Store as a single row (JSON arrays)
    cur.execute(f"""
        INSERT INTO {table_name} (df_curve_db_partial, df_curve_db_full, df_curve_db_remainder)
        VALUES (?, ?, ?)
    """, (
        json.dumps(partial.tolist()),
        json.dumps(full.tolist()),
        json.dumps(remainder.tolist())
    ))

    conn.commit()
    conn.close()

    print(f"Created table '{table_name}' with 3 scaled DF curves")
    

    
def apply_in_ear_diffuse_field_eq(
    db_path,
    diffuse_field_table="diffuse_field_eq",
    hpcf_table="hpcf_table",
    group_targets_table="group_targets",
    eq_column="df_curve_db_full",
    target_types=("in-ear", "earbud"),
    freq_table_name="frequency_axis",
    rounding=4
):
    """
    Adds the specified diffuse field curve to:
      1) mag_db in hpcf_table
      2) target_db in group_targets

    where type is in target_types.

    Both curves are in dB (log domain).
    After addition, each curve is normalized to 0 dB at 1 kHz.
    """

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # ------------------------------------------------------------
    # 1. Load diffuse field curve (single row)
    # ------------------------------------------------------------
    cur.execute(f"""
        SELECT {eq_column}
        FROM {diffuse_field_table}
        LIMIT 1
    """)
    row = cur.fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"No row found in table {diffuse_field_table}")

    eq_curve = np.array(json.loads(row[0]), dtype=np.float64)

    # ------------------------------------------------------------
    # 2. Load frequency axis (once)
    # ------------------------------------------------------------
    cur.execute(f"""
        SELECT frequency
        FROM {freq_table_name}
        ORDER BY idx
    """)
    freq_axis = np.array([r[0] for r in cur.fetchall()], dtype=np.float64)

    idx_1khz = np.where((freq_axis >= 950) & (freq_axis <= 1050))[0]
    if idx_1khz.size == 0:
        conn.close()
        raise RuntimeError("No frequency bins found for 1 kHz normalization")

    # ------------------------------------------------------------
    # 3. Apply DF EQ to hpcf_table
    # ------------------------------------------------------------
    for target_type in target_types:
        cur.execute(
            f"SELECT id, mag_db FROM {hpcf_table} WHERE type = ?",
            (target_type,)
        )
        rows = cur.fetchall()

        for row_id, mag_db_str in rows:
            if not mag_db_str:
                continue

            mag_db = np.array(json.loads(mag_db_str), dtype=np.float64)

            # Add DF curve
            new_mag_db = mag_db + eq_curve

            # 1 kHz normalization
            new_mag_db -= np.mean(new_mag_db[idx_1khz])

            # Rounding
            new_mag_db = np.round(new_mag_db, rounding)

            # Update
            cur.execute(
                f"UPDATE {hpcf_table} SET mag_db = ? WHERE id = ?",
                (json.dumps(new_mag_db.tolist()), row_id)
            )

    # ------------------------------------------------------------
    # 4. Apply DF EQ to group_targets.target_db
    # ------------------------------------------------------------
    for target_type in target_types:
        cur.execute(f"""
            SELECT id, target_db
            FROM {group_targets_table}
            WHERE type = ?
        """, (target_type,))
        rows = cur.fetchall()

        for row_id, target_db_str in rows:
            if not target_db_str:
                continue

            target_db = np.array(json.loads(target_db_str), dtype=np.float64)

            # Add DF curve
            new_target_db = target_db + eq_curve

            # 1 kHz normalization
            new_target_db -= np.mean(new_target_db[idx_1khz])

            # Rounding
            new_target_db = np.round(new_target_db, rounding)

            # Update
            cur.execute(
                f"""
                UPDATE {group_targets_table}
                SET target_db = ?
                WHERE id = ?
                """,
                (json.dumps(new_target_db.tolist()), row_id)
            )

    conn.commit()
    conn.close()

    print(
        f"Applied diffuse field '{eq_column}' to "
        f"hpcf_table and group_targets, normalized at 1 kHz "
        f"for target types: {target_types}"
    )




def average_headphone_mag_db_inplace(
    db_path,
    hpcf_table="hpcf_table",
    freq_table="frequency_axis",
    rounding=4
):
    """
    Create averaged mag_db curves per (type, headphone_name)
    directly inside the existing hpcf_table.
    Only averages if there are >=2 rows per group.
    """

    import sqlite3
    import json
    import numpy as np
    from collections import defaultdict

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # --- Load frequency axis ---
    cur.execute(f"SELECT frequency FROM {freq_table} ORDER BY idx")
    freq_axis = np.array([r[0] for r in cur.fetchall()])

    # --- Load all rows ---
    cur.execute(f"""
        SELECT source, type, rig, headphone_name, mag_db
        FROM {hpcf_table}
    """)
    rows = cur.fetchall()

    # --- Group by (type, headphone_name) ---
    grouped = defaultdict(list)
    for source, hp_type, rig, headphone_name, mag_db_json in rows:
        mag_db = np.array(json.loads(mag_db_json), dtype=np.float64)
        key = (hp_type, headphone_name)
        grouped[key].append(mag_db)

    inserted = 0

    # --- Process averages ---
    for (hp_type, headphone_name), mag_list in grouped.items():
        if len(mag_list) < 2:
            continue  # only average if >=2 rows

        # Average directly in dB
        mag_avg = np.mean(np.stack(mag_list), axis=0)

        # Optional: normalize to 0 dB at 1 kHz
 

        mag_avg = np.round(mag_avg, rounding)

        # Insert averaged row
        cur.execute(f"""
            INSERT INTO {hpcf_table}
            (source, type, rig, headphone_name, mag_db)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "Averaged Measurements",
            hp_type,
            None,  # source/rig not relevant here
            headphone_name,
            json.dumps(mag_avg.tolist())
        ))

        inserted += 1

    conn.commit()
    conn.close()

    print(f"Inserted {inserted} averaged headphone responses into {hpcf_table}")
    
    
    
    
    
    
    


def build_hpcf_db_from_csv( 
    root_dir,
    db_path="headphone_compilation.db",
    freq_table_name="frequency_axis",
    correction_wav=None,
    sample_rate=44100,
    fft_size=65536,
    rounding=4,
    min_freq=20,
    max_freq=20000,
    total_points=190,
    include_original=False  # NEW PARAMETER
):
    """
    Build hpcf_table database with optional column for original measurements.
    """


    # ------------------------------------------------------------
    # Fresh DB
    # ------------------------------------------------------------
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Deleted existing database: {db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # ------------------------------------------------------------
    # 1. Frequency axis
    # ------------------------------------------------------------
    create_weighted_frequency_table_in_db(
        db_path=db_path,
        table_name=freq_table_name,
        min_freq=min_freq,
        max_freq=max_freq,
        total_points=total_points,
        rounding="int",
        replace_table=True
    )

    cur.execute(f"SELECT frequency FROM {freq_table_name} ORDER BY idx")
    freq_axis = np.array([r[0] for r in cur.fetchall()], dtype=np.float64)
    #for normalization
    idx_1khz = np.where((freq_axis >= 950) & (freq_axis <= 1050))[0]

    # ------------------------------------------------------------
    # 2. Collect measurements
    # ------------------------------------------------------------
    groups = {}  # key = (source, type, rig), value = dict(headphone_name -> list of measurements)

    for source in os.listdir(root_dir):
        source_path = os.path.join(root_dir, source)
        data_path = os.path.join(source_path, "data")
        if not os.path.isdir(data_path):
            continue

        for hp_type in os.listdir(data_path):
            type_path = os.path.join(data_path, hp_type)
            if not os.path.isdir(type_path):
                continue

            rig_dirs = [d for d in os.listdir(type_path) if os.path.isdir(os.path.join(type_path, d))]
            if not rig_dirs:
                rig_dirs = [None]

            for rig in rig_dirs:
                base_path = os.path.join(type_path, rig) if rig else type_path

                group_key = (source, hp_type, rig)
                if group_key not in groups:
                    groups[group_key] = {}

                for csv_file in os.listdir(base_path):
                    if not csv_file.lower().endswith(".csv"):
                        continue

                    csv_path = os.path.join(base_path, csv_file)
                    freqs, raws = [], []

                    with open(csv_path, newline="") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            freqs.append(float(row["frequency"]))
                            raws.append(float(row["raw"]))

                    freqs = np.array(freqs, dtype=np.float64)
                    raws = np.array(raws, dtype=np.float64)

                    headphone_name = csv_file.replace(".csv", "")
                    if headphone_name not in groups[group_key]:
                        groups[group_key][headphone_name] = []

                    groups[group_key][headphone_name].append((freqs, raws))

    # ------------------------------------------------------------
    # 3. Create hpcf_table
    # ------------------------------------------------------------
    create_cols = """
        id INTEGER PRIMARY KEY,
        source TEXT,
        type TEXT,
        rig TEXT,
        headphone_name TEXT,
        mag_db TEXT
    """
    if include_original:
        create_cols += ", measurement TEXT"  # optional original curve

    cur.execute(f"CREATE TABLE hpcf_table ({create_cols})")

    # ------------------------------------------------------------
    # 4. Build mag_db responses
    # ------------------------------------------------------------
    for group_key, headphones in groups.items():
        source, hp_type, rig = group_key

        all_raws = []
        for meas_list in headphones.values():
            for freqs, raw in meas_list:
                all_raws.append((freqs, raw))

        if len(all_raws) < 2:
            continue  # skip if insufficient data

        base_freqs = all_raws[0][0]
        for freqs, _ in all_raws[1:]:
            if not np.allclose(freqs, base_freqs):
                raise ValueError(f"Frequency axis mismatch in group {group_key}")

        raw_stack = np.stack([raw for _, raw in all_raws])
        group_target = np.mean(raw_stack, axis=0)

        for headphone_name, meas_list in headphones.items():
            hp_stack = np.stack([raw for _, raw in meas_list])
            hp_avg = np.mean(hp_stack, axis=0)
        
            # --- Interpolate hp_avg to freq_axis, normalize and round ---
            #hp_avg_interp = np.interp(freq_axis, base_freqs, hp_avg, left=hp_avg[0], right=hp_avg[-1])
            hp_avg_interp = safe_interp(freq_axis, base_freqs, hp_avg)
            hp_avg_interp -= np.mean(hp_avg_interp[idx_1khz])
            hp_avg_interp = np.round(hp_avg_interp, rounding)
        
            # --- Compute difference to target (mag_db) ---
            diff = group_target - hp_avg
            #mag_db = np.interp(freq_axis, base_freqs, diff, left=diff[0], right=diff[-1])
            mag_db = safe_interp(freq_axis, base_freqs, diff)
            mag_db -= np.mean(mag_db[idx_1khz])
            mag_db = np.round(mag_db, rounding)
        
            if include_original:
                cur.execute("""
                    INSERT INTO hpcf_table (source, type, rig, headphone_name, mag_db, measurement)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    source, hp_type, rig, headphone_name,
                    json.dumps(mag_db.tolist()),
                    json.dumps(hp_avg_interp.tolist())  # store normalized & interpolated original
                ))
            else:
                cur.execute("""
                    INSERT INTO hpcf_table (source, type, rig, headphone_name, mag_db)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    source, hp_type, rig, headphone_name,
                    json.dumps(mag_db.tolist())
                ))

    conn.commit()

    # ------------------------------------------------------------
    # 5. Diffuse-field EQ
    # ------------------------------------------------------------
    if correction_wav:
        create_diffuse_field_curves(
            wav_path=correction_wav,
            db_path=db_path,
            freq_table_name=freq_table_name,
            table_name="diffuse_field_eq",
            sample_rate=sample_rate,
            fft_size=fft_size,
            rounding=rounding
        )
        apply_in_ear_diffuse_field_eq(
            db_path=db_path,
            diffuse_field_table="diffuse_field_eq",
            hpcf_table="hpcf_table",
            eq_column="df_curve_db_full",
            target_types=("in-ear", "earbud"),
            rounding=rounding
        )

    # ------------------------------------------------------------
    # 6. In-place averaging
    # ------------------------------------------------------------
    average_headphone_mag_db_inplace(
        db_path=db_path,
        hpcf_table="hpcf_table",
        freq_table=freq_table_name,
        rounding=rounding
    )

    conn.close()
    print(f"Database build complete: {db_path}")


    


def build_hpcf_db_from_meas_db(
    measurements_db_path,
    filters_db_path,
    freq_table_name="frequency_axis",
    min_freq=20,
    max_freq=20000,
    total_points=190,
    rounding=4,
    correction_wav=None,
    sample_rate=44100,
    fft_size=65536,
    cutoff_date=None
):
    """
    Build hpcf_table from measurements.db and store one group target per
    (source, type, rig) in a separate group_targets table.
    """

    # ------------------------------------------------------------
    # Fresh filters DB
    # ------------------------------------------------------------
    if os.path.exists(filters_db_path):
        os.remove(filters_db_path)
        print(f"[INFO] Deleted existing database: {filters_db_path}")

    conn = sqlite3.connect(filters_db_path)
    cur = conn.cursor()

    # ------------------------------------------------------------
    # 1. Frequency axis
    # ------------------------------------------------------------
    create_weighted_frequency_table_in_db(
        db_path=filters_db_path,
        table_name=freq_table_name,
        min_freq=min_freq,
        max_freq=max_freq,
        total_points=total_points,
        rounding="int",
        replace_table=True
    )

    cur.execute(f"SELECT frequency FROM {freq_table_name} ORDER BY idx")
    freq_axis = np.array([r[0] for r in cur.fetchall()], dtype=np.float64)

    # ------------------------------------------------------------
    # 2. Tables
    # ------------------------------------------------------------
    cur.execute("""
        CREATE TABLE group_targets (
            id INTEGER PRIMARY KEY,
            source TEXT,
            type TEXT,
            rig TEXT,
            target_db TEXT,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE hpcf_table (
            id INTEGER PRIMARY KEY,
            source TEXT,
            type TEXT,
            rig TEXT,
            brand TEXT,
            headphone_name TEXT,
            mag_db TEXT,
            group_target_id INTEGER,
            FOREIGN KEY(group_target_id) REFERENCES group_targets(id)
        )
    """)

    # ------------------------------------------------------------
    # 3. Gather measurements
    # ------------------------------------------------------------
    meas_conn = sqlite3.connect(measurements_db_path)
    meas_cur = meas_conn.cursor()

    meas_cur.execute("""
        SELECT source, headphone_type, brand, model,
               frequency_hz, spl_db, created_at
        FROM measurements
    """)
    rows = meas_cur.fetchall()
    meas_conn.close()

    groups = {}
    for source, hp_type, brand, model, freq_json, spl_json, created_at in rows:
        freqs = np.array(json.loads(freq_json), dtype=np.float64)
        spls  = np.array(json.loads(spl_json), dtype=np.float64)
        key = (source, hp_type, "rig1")
        groups.setdefault(key, {}).setdefault((brand, model), []).append(
            (freqs, spls, created_at)
        )

    # ------------------------------------------------------------
    # 4. Build targets + HpCFs
    # ------------------------------------------------------------
    for group_key, headphones in groups.items():
        source, hp_type, rig = group_key

        all_raws = [
            (freqs, spls)
            for meas_list in headphones.values()
            for freqs, spls, _ in meas_list
        ]
        if len(all_raws) < 2:
            continue

        # interpolated = [
        #     np.interp(freq_axis, freqs, spls, left=spls[0], right=spls[-1])
        #     for freqs, spls in all_raws
        # ]
        interpolated = [
            safe_interp(freq_axis, freqs, spls)
            for freqs, spls in all_raws
        ]
        
        group_target = np.mean(np.stack(interpolated), axis=0)
        group_target = np.round(group_target, rounding)

        # Insert ONE group target
        cur.execute("""
            INSERT INTO group_targets (source, type, rig, target_db, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            source,
            hp_type,
            rig,
            json.dumps(group_target.tolist()),
            datetime.utcnow().isoformat()
        ))
        group_target_id = cur.lastrowid

        # Process headphones
        for (brand, headphone_name), meas_list in headphones.items():
            if headphone_name.lower() == "unknown":
                continue

            if cutoff_date:
                meas_list = [
                    (f, s) for f, s, created_at in meas_list
                    if created_at >= cutoff_date
                ]
            else:
                meas_list = [(f, s) for f, s, _ in meas_list]

            if not meas_list:
                continue

            # hp_interp = [
            #     np.interp(freq_axis, freqs, spls, left=spls[0], right=spls[-1])
            #     for freqs, spls in meas_list
            # ]
            hp_interp = [
                safe_interp(freq_axis, freqs, spls)
                for freqs, spls in meas_list
            ]
            hp_avg = np.mean(np.stack(hp_interp), axis=0)

            mag_db = group_target - hp_avg
            idx_1khz = np.where((freq_axis >= 950) & (freq_axis <= 1050))[0]
            mag_db -= np.mean(mag_db[idx_1khz])
            mag_db = np.round(mag_db, rounding)

            cur.execute("""
                INSERT INTO hpcf_table
                (source, type, rig, brand, headphone_name, mag_db, group_target_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                source,
                hp_type,
                rig,
                brand,
                headphone_name,
                json.dumps(mag_db.tolist()),
                group_target_id
            ))

    conn.commit()

    # ------------------------------------------------------------
    # 5. Diffuse-field EQ
    # ------------------------------------------------------------
    if correction_wav:
        create_diffuse_field_curves(
            wav_path=correction_wav,
            db_path=filters_db_path,
            freq_table_name=freq_table_name,
            table_name="diffuse_field_eq",
            sample_rate=sample_rate,
            fft_size=fft_size,
            rounding=rounding
        )
        apply_in_ear_diffuse_field_eq(
            db_path=filters_db_path,
            diffuse_field_table="diffuse_field_eq",
            hpcf_table="hpcf_table",
            eq_column="df_curve_db_full",
            target_types=("in-ear", "earbud"),
            rounding=rounding
        )

    conn.close()
    print(f"[INFO] HPCF database built successfully at {filters_db_path}")

def safe_interp(
    x_new: np.ndarray,
    freqs: np.ndarray,
    values: np.ndarray,
    *,
    deduplicate: bool = True
) -> np.ndarray:
    """
    Safely interpolate values onto a new axis by sorting frequencies first.

    Parameters
    ----------
    x_new : np.ndarray
        Target axis to interpolate onto (e.g. freq_axis).
    freqs : np.ndarray
        Original frequency axis (may be unordered).
    values : np.ndarray
        Values corresponding to freqs (e.g. SPL).
    deduplicate : bool, optional
        If True, remove duplicate frequencies (keeps first occurrence).

    Returns
    -------
    np.ndarray
        Interpolated values aligned to x_new.
    """

    freqs = np.asarray(freqs, dtype=float)
    values = np.asarray(values, dtype=float)

    if freqs.shape != values.shape:
        raise ValueError("freqs and values must have the same shape")

    # Sort by frequency
    order = np.argsort(freqs)
    freqs = freqs[order]
    values = values[order]

    # Optionally remove duplicate frequencies
    if deduplicate:
        freqs, unique_idx = np.unique(freqs, return_index=True)
        values = values[unique_idx]

    # Perform interpolation
    return np.interp(
        x_new,
        freqs,
        values,
        left=values[0],
        right=values[-1],
    )
    
def integrate_filters_into_master(
    filters_db_path,
    master_db_path,
    brand_threshold=90,
    headphone_token_threshold=95,
    headphone_seq_threshold=98
):
    """
    Integrate rows from filters.db into master hpcf_table database using fuzzy matching.

    Hybrid headphone matching:
        - First token-based similarity
        - Then sequence similarity
        - Only accept if both exceed thresholds

    Sample IDs are limited to 26 (A-Z).
    Logs all matches, and warns when filter headphone does not match exactly.
    """
    # Internal tool — requires rapidfuzz, not a user dependency
    from rapidfuzz import process, fuzz

    def sample_name_from_id(sample_id):
        if sample_id > 26:
            return None
        return f"Sample {string.ascii_uppercase[sample_id-1]}"

    def normalize_name(name):
        # Lowercase, remove non-alphanumeric chars
        return re.sub(r'[^a-z0-9]', '', name.lower())

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    f_conn = sqlite3.connect(filters_db_path)
    f_cur = f_conn.cursor()
    m_conn = sqlite3.connect(master_db_path)
    m_cur = m_conn.cursor()

    m_cur.execute("SELECT DISTINCT brand FROM hpcf_table")
    master_brands = [r[0] for r in m_cur.fetchall()]

    m_cur.execute("SELECT brand, headphone FROM hpcf_table")
    master_rows = m_cur.fetchall()

    f_cur.execute("SELECT brand, headphone_name, mag_db, source, type FROM hpcf_table")
    filter_rows = f_cur.fetchall()

    for idx, (f_brand, f_headphone, mag_db_str, source, hp_type) in enumerate(filter_rows, 1):
        # --- Brand matching ---
        f_brand_norm = normalize_name(f_brand)
        normalized_brands = {b: normalize_name(b) for b in master_brands}
        best_norm_match, brand_score, _ = process.extractOne(f_brand_norm, normalized_brands.values(), scorer=fuzz.token_sort_ratio)
        # Map back to original brand name
        original_brand_match = [k for k, v in normalized_brands.items() if v == best_norm_match][0]
        
        if brand_score >= brand_threshold:
            master_brand = original_brand_match
            brand_status = "existing"
        else:
            master_brand = f_brand
            brand_status = "new"

        # --- Headphone matching ---
        candidate_headphones = [h for b, h in master_rows if b == master_brand]
        master_headphone = f_headphone
        next_sample_id = 1
        headphone_status = "new"

        if candidate_headphones:
            # Step 1: exact match (case-insensitive)
            exact_matches = [h for h in candidate_headphones if h.lower() == f_headphone.lower()]
            if exact_matches:
                master_headphone = exact_matches[0]
                m_cur.execute(
                    "SELECT MAX(sample_id) FROM hpcf_table WHERE brand=? AND headphone=?",
                    (master_brand, master_headphone)
                )
                row = m_cur.fetchone()
                next_sample_id = (row[0] or 0) + 1
                headphone_status = "existing"
            else:
                # Step 2: normalized hybrid matching
                f_norm = normalize_name(f_headphone)
                candidate_norm = {h: normalize_name(h) for h in candidate_headphones}
                best_match, token_score, _ = process.extractOne(f_norm, candidate_norm.values(), scorer=fuzz.token_sort_ratio)
                # Map back to original headphone name
                original_match = [k for k, v in candidate_norm.items() if v == best_match][0]
                seq_score = fuzz.ratio(f_norm, best_match)

                if token_score >= headphone_token_threshold and seq_score >= headphone_seq_threshold:
                    master_headphone = original_match
                    m_cur.execute(
                        "SELECT MAX(sample_id) FROM hpcf_table WHERE brand=? AND headphone=?",
                        (master_brand, master_headphone)
                    )
                    row = m_cur.fetchone()
                    next_sample_id = (row[0] or 0) + 1
                    headphone_status = "existing"
                    if master_headphone != f_headphone:
                        logging.warning(f"Filter headphone '{f_headphone}' matched to existing '{master_headphone}' "
                                        f"(token={token_score:.1f}, seq={seq_score:.1f})")
                else:
                    headphone_status = f"new (token={token_score:.1f}, seq={seq_score:.1f})"
                    logging.info(f"Filter headphone '{f_headphone}' did NOT match any existing headphone -> new entry")

        # Limit sample ID
        if next_sample_id > 26:
            logging.warning(f"Skipping '{f_headphone}' ({f_brand}): sample_id exceeds 26")
            continue

        sample_name = sample_name_from_id(next_sample_id)
        created_on = datetime.now().isoformat(timespec='seconds')

        m_cur.execute("""
            INSERT INTO hpcf_table (brand, headphone, sample, sample_id, created_on, type, mag_db)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (master_brand, master_headphone, sample_name, next_sample_id, created_on, hp_type, mag_db_str))

        master_rows.append((master_brand, master_headphone))
        if master_brand not in master_brands:
            master_brands.append(master_brand)

        logging.info(f"[{idx}/{len(filter_rows)}] Brand: {f_brand} ({brand_status}), "
                     f"Headphone: {f_headphone} -> {master_headphone} ({headphone_status}), "
                     f"Sample: {sample_name}, Type: {hp_type}")

    m_conn.commit()
    f_conn.close()
    m_conn.close()
    logging.info(f"Integrated {len(filter_rows)} filter rows into master database")
    
    

def normalize_created_on_to_ymd(conn):
    """
    Normalize all created_on timestamps in hpcf_table to YYYY-MM-DD format.
    Only keeps year, month, and day.
    """
    cur = conn.cursor()
    # Keep only first 10 characters (YYYY-MM-DD)
    cur.execute("""
        UPDATE hpcf_table
        SET created_on = SUBSTR(created_on, 1, 10)
        WHERE created_on IS NOT NULL
    """)
    conn.commit()
    logging.info("[INFO] Normalized all created_on timestamps to YYYY-MM-DD.")

def hpcf_generate_averages(db_path):
    """
    Generate or update 'Average' mag_db for each headphone if more than one non-Average sample exists.
    Also auto-populates 'id' for new rows.
    """
    now_datetime = datetime.now()

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            
            # Normalize timestamps to YYYY-MM-DD before generating averages
            normalize_created_on_to_ymd(conn)

            # Fetch all rows once
            cur.execute("SELECT * FROM hpcf_table ORDER BY headphone, sample_id")
            rows = cur.fetchall()

            # Organize by headphone
            headphones = {}
            max_id = 0
            for r in rows:
                h = r['headphone']
                headphones.setdefault(h, []).append(r)
                if r['id'] is not None and r['id'] > max_id:
                    max_id = r['id']
            next_id = max_id + 1

            for h, samples in headphones.items():
                # Track average info
                has_average = False
                average_last_mod = datetime(2000, 1, 1)
                samples_last_mod = datetime(2000, 1, 1)
                detected_type = None
                non_avg_count = 0

                for s in samples:
                    created_on = datetime.fromisoformat(s['created_on'])
                    if s['sample'] == 'Average':
                        has_average = True
                        average_last_mod = created_on
                    else:
                        non_avg_count += 1
                        if not detected_type and s['type']:
                            detected_type = s['type']
                        if created_on > samples_last_mod:
                            samples_last_mod = created_on

                # Skip if there is 0 or 1 non-Average sample
                if non_avg_count <= 1:
                    continue

                # Determine if update or insert
                update_average = has_average and (samples_last_mod > average_last_mod or average_last_mod.date() == now_datetime.date())
                
                insert_average = not has_average

                if not (update_average or insert_average):
                    continue

                # Build average
                mag_stack = []
                for s in samples:
                    if s['sample'] != 'Average':  # only non-Average samples
                        try:
                            mag_stack.append(np.array(json.loads(s['mag_db']), dtype=np.float64))
                        except Exception as e:
                            logging.warning(f"Skipping malformed mag_db for {s['sample']} of {h}: {e}")

                if len(mag_stack) < 2:
                    continue  # just in case, skip if fewer than 2 valid samples

                min_len = min(arr.shape[0] for arr in mag_stack)
                mag_stack_trimmed = [arr[:min_len] for arr in mag_stack]
                mag_avg_db = np.mean(np.stack(mag_stack_trimmed), axis=0)
                mag_avg_db_json = json.dumps(np.round(mag_avg_db, 4).tolist())

                if update_average:
                    cur.execute("""
                        UPDATE hpcf_table
                        SET mag_db = ?, created_on = ?, type = ?
                        WHERE headphone = ? AND sample = 'Average'
                    """, (mag_avg_db_json, now_datetime.isoformat(), detected_type, h))
                    logging.info(f"[UPDATE] Average updated for {h}")

                if insert_average:
                    # Get brand from first sample
                    brand = samples[0]['brand'] or "Unknown"
                    cur.execute("""
                        INSERT INTO hpcf_table (id, brand, headphone, sample, sample_id, created_on, type, mag_db)
                        VALUES (?, ?, ?, 'Average', 0, ?, ?, ?)
                    """, (next_id, brand, h, now_datetime.isoformat(), detected_type, mag_avg_db_json))
                    next_id += 1
                    logging.info(f"[INSERT] Average inserted for {h}")

            conn.commit()

    except Exception:
        logging.error("Error in hpcf_generate_averages_fast", exc_info=True)
        
        
        

def populate_unique_ids(db_path):
    """
    Populate the `id` column in hpcf_table with unique integers.
    If any ids already exist, continue from the max id + 1.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Ensure id column exists
    cur.execute("PRAGMA table_info(hpcf_table)")
    columns = [col[1] for col in cur.fetchall()]
    if "id" not in columns:
        raise ValueError("hpcf_table does not have an 'id' column.")

    # Find current max id
    cur.execute("SELECT MAX(id) FROM hpcf_table")
    max_id = cur.fetchone()[0] or 0
    next_id = max_id + 1

    # Update rows with NULL id or duplicate ids
    cur.execute("SELECT rowid FROM hpcf_table ORDER BY rowid")
    rows = cur.fetchall()
    for row in rows:
        cur.execute("UPDATE hpcf_table SET id = ? WHERE rowid = ?", (next_id, row[0]))
        next_id += 1

    conn.commit()
    conn.close()
    print(f"[INFO] Populated hpcf_table.id with unique ids starting from {max_id + 1}.")
        
def print_cross_brand_duplicate_headphones(db_path):
    """
    Print all rows where the same headphone name exists under multiple brands.
    Duplicate headphones under the SAME brand are allowed.
    mag_db is intentionally excluded from output.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT
            id,
            brand,
            headphone,
            sample,
            sample_id,
            created_on,
            type
        FROM hpcf_table
        WHERE headphone IN (
            SELECT headphone
            FROM hpcf_table
            GROUP BY headphone
            HAVING COUNT(DISTINCT brand) > 1
        )
        ORDER BY headphone, brand, sample
    """)

    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("[INFO] No cross-brand duplicate headphone names found.")
        return

    print("[WARNING] Cross-brand duplicate headphone names detected:\n")

    for row in rows:
        print(row)
        
        
def fix_cross_brand_duplicate_headphones(db_path, apply_fix=True):
    """
    Detect headphones that appear under multiple brands.
    Optionally auto-correct brands by choosing the most common one.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    apply_fix : bool
        If True, updates the database.
        If False, performs a dry run and only prints actions.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Find headphones with multiple brands
    cur.execute("""
        SELECT headphone
        FROM hpcf_table
        GROUP BY headphone
        HAVING COUNT(DISTINCT brand) > 1
    """)
    headphones = [r[0] for r in cur.fetchall()]

    if not headphones:
        print("[INFO] No cross-brand duplicate headphone names found.")
        conn.close()
        return

    print(f"[WARNING] Found {len(headphones)} headphones with cross-brand duplicates\n")

    for headphone in headphones:
        # Fetch rows for this headphone
        cur.execute("""
            SELECT id, brand
            FROM hpcf_table
            WHERE headphone = ?
        """, (headphone,))
        rows = cur.fetchall()

        brands = [b for _, b in rows if b is not None]
        brand_counts = Counter(brands)

        # Pick most common brand
        canonical_brand, count = brand_counts.most_common(1)[0]

        print(f"Headphone: {headphone}")
        print(f"  Brand counts: {dict(brand_counts)}")
        print(f"  → Selected canonical brand: '{canonical_brand}'\n")

        # Apply fix
        for row_id, brand in rows:
            if brand != canonical_brand:
                if apply_fix:
                    cur.execute("""
                        UPDATE hpcf_table
                        SET brand = ?
                        WHERE id = ?
                    """, (canonical_brand, row_id))
                    print(f"    [FIXED] Row {row_id}: '{brand}' → '{canonical_brand}'")
                else:
                    print(f"    [DRY-RUN] Row {row_id}: '{brand}' → '{canonical_brand}'")

        print()

    if apply_fix:
        conn.commit()
        print("[INFO] Database updated successfully.")
    else:
        print("[INFO] Dry run complete. No changes written.")

    conn.close()        
    
def find_similar_brands(
    db_path,
    similarity_threshold=0.85,
    min_length=3
):
    """
    Find brand names that are highly similar and may refer to the same brand.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    similarity_threshold : float
        Similarity ratio (0–1). Higher = stricter.
        0.85 is a good default.
    min_length : int
        Ignore very short brand names to reduce false positives.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT brand
        FROM hpcf_table
        WHERE brand IS NOT NULL AND TRIM(brand) != ''
    """)
    brands = sorted({b.strip() for (b,) in cur.fetchall()})

    conn.close()

    print(f"[INFO] Checking {len(brands)} unique brand names\n")

    matches = []

    for b1, b2 in itertools.combinations(brands, 2):
        if len(b1) < min_length or len(b2) < min_length:
            continue

        # Case-insensitive similarity
        ratio = difflib.SequenceMatcher(
            None, b1.lower(), b2.lower()
        ).ratio()

        if ratio >= similarity_threshold:
            matches.append((b1, b2, ratio))

    if not matches:
        print("[INFO] No highly similar brand names found.")
        return

    print("[WARNING] Potential duplicate / alias brand names:\n")

    for b1, b2, ratio in sorted(matches, key=lambda x: -x[2]):
        print(f"  '{b1}'  <->  '{b2}'   (similarity={ratio:.3f})")
        
def autocorrect_case_only_brands(db_path, dry_run=False):
    """
    Auto-correct brand names that differ ONLY by case.
    Picks the most common casing and updates others.

    Parameters
    ----------
    db_path : str
        Path to SQLite database
    dry_run : bool
        If True, prints changes without modifying DB
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT brand
        FROM hpcf_table
        WHERE brand IS NOT NULL AND TRIM(brand) != ''
    """)
    brands = [b[0] for b in cur.fetchall()]

    # Group by lowercase
    groups = {}
    for b in brands:
        groups.setdefault(b.lower(), []).append(b)

    corrections = []

    for lower, variants in groups.items():
        unique_variants = set(variants)
        if len(unique_variants) <= 1:
            continue  # no conflict

        # Only allow case-only differences
        if any(v.lower() != lower for v in unique_variants):
            continue

        counts = Counter(variants)
        canonical = counts.most_common(1)[0][0]

        for v in unique_variants:
            if v != canonical:
                corrections.append((v, canonical, counts[v]))

    if not corrections:
        print("[INFO] No case-only brand corrections needed.")
        conn.close()
        return

    print("[WARNING] Case-only brand corrections detected:\n")

    for old, new, count in corrections:
        print(f"  '{old}'  →  '{new}'   ({count} rows)")

    if dry_run:
        print("\n[DRY RUN] No changes were written to the database.")
        conn.close()
        return

    # Apply updates
    for old, new, _ in corrections:
        cur.execute(
            "UPDATE hpcf_table SET brand = ? WHERE brand = ?",
            (new, old)
        )

    conn.commit()
    conn.close()
    print("\n[INFO] Case-only brand corrections applied successfully.")
    
    
def sample_label_from_id(sample_id):
    if sample_id <= 0:
        return "Average"
    return f"Sample {chr(ord('A') + sample_id - 1)}"


def fix_duplicate_samples(db_path, dry_run=False):
    """
    Fix duplicate sample_id or sample within each (brand, headphone) group.
    Assigns the next available sample_id and sample name.
    """

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT id, brand, headphone, sample, sample_id
        FROM hpcf_table
        WHERE brand IS NOT NULL
          AND headphone IS NOT NULL
        ORDER BY brand, headphone, sample_id, id
    """)
    rows = cur.fetchall()

    # Group by (brand, headphone)
    groups = defaultdict(list)
    for row in rows:
        groups[(row[1], row[2])].append(row)

    total_fixes = 0

    for (brand, headphone), entries in groups.items():
        used_ids = set()
        used_samples = set()
        duplicates = []

        # Pass 1: identify duplicates
        for row_id, _, _, sample, sample_id in entries:
            sample_key = (sample or "").strip().lower()

            # Average is special, never touched
            if sample_id == 0:
                used_ids.add(0)
                used_samples.add(sample_key)
                continue

            if sample_id not in used_ids and sample_key not in used_samples:
                used_ids.add(sample_id)
                used_samples.add(sample_key)
            else:
                duplicates.append(row_id)

        if not duplicates:
            continue  # nothing to fix for this headphone

        # Find next available sample_id for THIS headphone
        next_id = 1
        while next_id in used_ids:
            next_id += 1

        for row_id in duplicates:
            new_sample_id = next_id
            new_sample = sample_label_from_id(new_sample_id)

            print(
                f"[FIX] {brand} | {headphone} | row_id={row_id} → "
                f"{new_sample} (sample_id={new_sample_id})"
            )

            if not dry_run:
                cur.execute("""
                    UPDATE hpcf_table
                    SET sample_id = ?, sample = ?
                    WHERE id = ?
                """, (new_sample_id, new_sample, row_id))

            used_ids.add(new_sample_id)
            used_samples.add(new_sample.lower())
            next_id += 1
            total_fixes += 1

    if dry_run:
        print(f"\n[DRY RUN] {total_fixes} duplicate samples detected. No changes written.")
    else:
        conn.commit()
        print(f"\n[INFO] {total_fixes} duplicate samples corrected successfully.")

    conn.close()
    
def export_headphones_by_date(db_path, output_txt_path, date_str):
    """
    Export a list of headphones where rows in the database were created
    on or after the specified date, sorted by headphone name, and print each row
    in verbose format.

    Args:
        db_path (str or Path): Path to the SQLite database.
        output_txt_path (str or Path): Path to save the output TXT file.
        date_str (str): Date string in "YYYY-MM-DD" format.
    """
    import sqlite3
    from datetime import datetime
    from pathlib import Path
    import json

    date_filter = datetime.strptime(date_str, "%Y-%m-%d").date()
    output_txt_path = Path(output_txt_path)

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # Fetch all rows
            cur.execute("SELECT brand, headphone, sample, created_on FROM hpcf_table")
            rows = cur.fetchall()

            # Filter and prepare rows
            filtered = []
            for r in rows:
                created_date = datetime.fromisoformat(r['created_on']).date()
                if created_date >= date_filter:
                    filtered.append({
                        "brand": r['brand'],
                        "headphone": r['headphone'],
                        "sample": r['sample'],
                        "created_on": r['created_on']
                    })

            # Sort by headphone name
            filtered.sort(key=lambda x: x['headphone'].lower())

            output_lines = []
            for row in filtered:
                line_str = f'New HpCF: {json.dumps(row)}'
                print(line_str)
                output_lines.append(line_str)

            # Save to TXT
            with open(output_txt_path, "w", encoding="utf-8") as f:
                for line in output_lines:
                    f.write(line + "\n")

            print(f"[INFO] Saved {len(output_lines)} entries to {output_txt_path}")

    except Exception as e:
        print(f"[ERROR] Failed to export headphones: {e}")