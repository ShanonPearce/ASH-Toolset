# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 19:16:49 2025

@author: Shanon
"""

import logging.config
from os.path import join as pjoin
from pathlib import Path
import configparser
from ash_toolset import brir_generation
from ash_toolset import brir_export
from ash_toolset import e_apo_config_creation
from ash_toolset import constants as CN
from ash_toolset import hpcf_functions
from ash_toolset import helper_functions as hf
from ash_toolset import air_processing
from ash_toolset import hrir_processing
from scipy.io import wavfile
import dearpygui.dearpygui as dpg
import numpy as np
import ast
import math
from time import sleep
import threading
import scipy as sp
import os
import subprocess
import sys
import os
import time
import csv
from datetime import datetime
import shutil
import scipy.signal as signal
import platform
import subprocess
import json
import urllib.request

#
#
## Settings related
#
#



def reset_settings(sender=None, app_data=None, user_data=None):
    """
    GUI callback to reset all settings to their default values.
    
    Expects `user_data` to be a dict with default lists and values:
        - 'brands_list', 'hp_list_default', 'sample_list_default',
          'brand_default', 'headphone_default', 'sample_default'
    """
    try:
        # --- START OF FUNCTION LOGIC ---
        hpcf_defaults = dpg.get_item_user_data("qc_e_apo_sel_hpcf")
        logz = dpg.get_item_user_data("console_window")
    
        # Reset all GUI items from CN.DEFAULTS
        for key, default_value in CN.DEFAULTS.items():
            if dpg.does_item_exist(key):
                try:
                    dpg.set_value(key, default_value)
                except Exception as e:
                    logging.warning(f"Failed to reset GUI element '{key}': {e}")
    
        # Reset HpCF listboxes and selections
        for suffix in ("fde_", "qc_"):
            if 'brands_list_default' in hpcf_defaults:
                dpg.configure_item(f"{suffix}hpcf_brand", items=hpcf_defaults['brands_list_default'])
                dpg.set_value(f"{suffix}hpcf_brand", hpcf_defaults['brand_default'])
            if 'hp_list_default' in hpcf_defaults:
                dpg.configure_item(f"{suffix}hpcf_headphone", items=hpcf_defaults['hp_list_default'])
                dpg.set_value(f"{suffix}hpcf_headphone", hpcf_defaults['headphone_default'])
            if 'sample_list_default' in hpcf_defaults:
                dpg.configure_item(f"{suffix}hpcf_sample", items=hpcf_defaults['sample_list_default'])
                dpg.set_value(f"{suffix}hpcf_sample", hpcf_defaults['sample_default'])
        
        #reset hpcf database, both tabs will be reset to ASH db
        change_hpcf_database(selected_db=CN.DEFAULTS["qc_hpcf_active_database"],use_previous_settings=False)
        dpg.configure_item('qc_hrtf_add_favourite',show=True)
        dpg.configure_item('qc_hrtf_remove_favourite',show=False)
        dpg.configure_item('qc_hrtf_average_favourite',show=False)
        dpg.configure_item('qc_open_user_sofa_folder',show=False)
        dpg.configure_item('hrtf_add_favourite',show=True)
        dpg.configure_item('hrtf_remove_favourite',show=False)
        dpg.configure_item('hrtf_average_favourite',show=False)
        dpg.configure_item('open_user_sofa_folder',show=False)
        updated_fav_list = CN.HRTF_BASE_LIST_FAV
        # Replace user data
        dpg.configure_item('hrtf_add_favourite', user_data=updated_fav_list)
    
        # Reset HRTF selection items
        dpg.configure_item('fde_brir_hrtf', items=CN.DEFAULTS["brir_hrtf_listener_list"])
        dpg.configure_item('qc_brir_hrtf', items=CN.DEFAULTS["brir_hrtf_listener_list"])
        dpg.configure_item('fde_brir_hrtf_dataset', items=CN.DEFAULTS["brir_hrtf_dataset_list"])
        dpg.configure_item('qc_brir_hrtf_dataset', items=CN.DEFAULTS["brir_hrtf_dataset_list"])
    
        # Reset progress bars and GUI toggles
        reset_hpcf_progress()
        reset_brir_progress()
        qc_reset_progress()
        e_apo_toggle_hpcf_gui(app_data=False)
        e_apo_toggle_brir_gui(app_data=False)
    
        # Reset output directories
        e_apo_path = dpg.get_value('e_apo_program_path')
        if e_apo_path:
            primary_path = pjoin(e_apo_path, "config")
            primary_ash_path = pjoin(primary_path, CN.PROJECT_FOLDER)
        else:
            primary_path = 'C:\\Program Files\\EqualizerAPO\\config'
            primary_ash_path = pjoin(primary_path, CN.PROJECT_FOLDER)
    
        dpg.set_value('selected_folder_base', primary_path)
        dpg.set_value('selected_folder_ash', primary_ash_path)
        dpg.set_value('selected_folder_ash_tooltip', primary_ash_path)
    
        # Hesuvi path
        if 'EqualizerAPO' in primary_path:
            hesuvi_path_selected = pjoin(primary_path, 'HeSuVi')
        else:
            hesuvi_path_selected = pjoin(primary_path, CN.PROJECT_FOLDER, 'HeSuVi')
    
        dpg.set_value('selected_folder_hesuvi', hesuvi_path_selected)
        dpg.set_value('selected_folder_hesuvi_tooltip', hesuvi_path_selected)
    
        # Reset channel config
        reset_channel_config(sender, app_data, user_data)
    
        # Save settings after reset
        save_settings(update_hpcf_pars=True, update_brir_pars=True)
    
        hf.log_with_timestamp('Settings have been reset to default', logz)
        
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)

def reset_channel_config(sender=None, app_data=None, user_data=None):
    """
    GUI callback to reset channel config in E-APO config section.
    
    Expects CN.DEFAULTS to contain default values for relevant keys.
    """
    try:
        
        allowed_prefixes = (
            "e_apo_upmix_method",
            "e_apo_side_delay",
            "e_apo_rear_delay",
            "e_apo_mute_",
            "e_apo_gain_",
            "e_apo_elev_angle_",
            "e_apo_az_angle_",
            "hesuvi_elev_angle_",
            "hesuvi_az_angle_",
            "e_apo_prevent_clip",
            "e_apo_audio_channels",
        )
    
        for key, default_value in CN.DEFAULTS.items():
            if any(key.startswith(prefix) for prefix in allowed_prefixes):
                if dpg.does_item_exist(key):
                    try:
                        dpg.set_value(key, default_value)
                    except Exception as e:
                        logging.warning(f"Failed to reset GUI element '{key}': {e}")
    
        # Reapply channel configuration and refresh
        e_apo_select_channels(app_data=CN.DEFAULTS["e_apo_audio_channels"], aquire_config=False)
        e_apo_activate_direction()
        e_apo_config_acquire()

    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)



def load_settings(defaults=None, preset_name=None, version=None,  
                  migrate_func=None, logz=None, set_gui_values=False,
                  legacy_mapping=None):
    """
    Load settings from the main config file or a preset INI file.
    Falls back to defaults if the file is missing or incomplete.
    """
    defaults = defaults or CN.DEFAULTS
    base_settings_file = CN.SETTINGS_FILE
    version = version or getattr(CN, "__version__", "0.0")
    legacy_mapping = legacy_mapping or getattr(CN, "LEGACY_KEY_MAP", None)

    # --- Derive file path based on preset ---
    settings_file = (
        os.path.join(CN.SETTINGS_DIR, f"{preset_name}.ini")
        if preset_name else base_settings_file
    )

    loaded_values = defaults.copy()

    try:
        # --- Optional migration step ---
        try:
            if migrate_func:
                migrate_func()
            else:
                migrate_settings()
        except Exception as e:
            if logz:
                hf.log_with_timestamp(f"Warning: Migration step failed – {e}", logz)

        config = configparser.ConfigParser()
        config.read(settings_file)

        # --- Version check ---
        version_loaded = hf.safe_get(config, "version", str, version)
        if version_loaded != version and logz:
            hf.log_with_timestamp(
                f"Settings version mismatch: file={version_loaded}, expected={version}. Loading anyway.", logz
            )

        # --- Apply loaded or default values ---
        for key, default_value in defaults.items():
            value_type = type(default_value)
            try:
                if config.has_option('DEFAULT', key):
                    loaded_values[key] = hf.safe_get(config, key, value_type, default_value)
                elif legacy_mapping:
                    for old_key, new_key in legacy_mapping.items():
                        if new_key == key and config.has_option('DEFAULT', old_key):
                            loaded_values[key] = hf.safe_get(config, old_key, value_type, default_value)
                            if logz:
                                hf.log_with_timestamp(f"Loaded legacy key '{old_key}' → '{key}'", logz)
                            break
            except Exception as e:
                if logz:
                    hf.log_with_timestamp(f"Failed to load key '{key}' from {settings_file}: {e}", logz)

        # --- Optionally update GUI ---
        if set_gui_values:
            try:
                
                #refresh brand, headphone, sample and database gui elements
                change_hpcf_database(selected_db=loaded_values.get("qc_hpcf_active_database"), use_previous_settings=False)
                hpcf_db_dict = dpg.get_item_user_data("qc_e_apo_sel_hpcf")
                if not hpcf_db_dict or "conn" not in hpcf_db_dict:
                    raise RuntimeError("No valid database connection found in qc_e_apo_sel_hpcf user data.")
                conn = hpcf_db_dict["conn"]
                headphone_selected = loaded_values.get("qc_hpcf_headphone")
                sample_selected = loaded_values.get("qc_hpcf_sample")
                brand_selected = hpcf_functions.get_brand(conn, headphone_selected) if headphone_selected else None
                #update qc hpcf lists
                handle_hpcf_update(tab_prefix="qc", update_type="full_update", brand=brand_selected, headphone=headphone_selected, sample=sample_selected)

                    
                if logz:
                    hf.log_with_timestamp("Refreshed HpCF GUI elements.")

                #update all gui values
                applied = 0
                for key, value in loaded_values.items():
                    if dpg.does_item_exist(key):
                        try:
                            dpg.set_value(key, value)
                            applied += 1
                        except Exception as e:
                            hf.log_with_timestamp(f"Failed to set GUI element '{key}': {e}", logz)
                qc_reset_progress()
                if logz:
                    hf.log_with_timestamp(f"Loaded {settings_file} and applied {applied} loaded settings to GUI elements.", logz)

            except Exception as e:
                if logz:
                    hf.log_with_timestamp(f"Error applying loaded settings to GUI: {e}", logz)

    except Exception as e:
        if logz:
            hf.log_with_timestamp(f"Error loading saved configuration ({settings_file}): {e}", logz)
            hf.log_with_timestamp("Falling back to default values.", logz)

    return loaded_values



                
def migrate_settings():
    """
    Migrate the settings file from the old application directory to the new user-specific config directory.

    This function checks if the new settings file already exists; if so, no action is taken.
    If the new settings file does not exist but the old settings file does, the old settings file
    is copied to the new location to preserve user configuration. After a successful copy,
    the old settings file is deleted to avoid confusion.

    This migration ensures that settings persist across application versions that changed
    the default storage location for the settings file.

    Note:
        - Assumes SETTINGS_FILE_NEW and SETTINGS_FILE_OLD are defined globally or imported.
        - Does not return any value.
    """
    try:
        if os.path.exists(CN.SETTINGS_FILE_NEW):
            return

        if os.path.exists(CN.SETTINGS_FILE_OLD):
            new_dir = os.path.dirname(CN.SETTINGS_FILE_NEW)
            os.makedirs(new_dir, exist_ok=True)
            logging.info(f"[INFO] Migrating settings from old to new path: {CN.SETTINGS_FILE_OLD} → {CN.SETTINGS_FILE_NEW}")
            shutil.copy2(CN.SETTINGS_FILE_OLD, CN.SETTINGS_FILE_NEW)

            # If copy successful, delete old settings file
            os.remove(CN.SETTINGS_FILE_OLD)
            logging.info(f"[INFO] Deleted old settings file: {CN.SETTINGS_FILE_OLD}")

    except Exception as e:
        logging.error(f"Failed to migrate settings file: {e}")

    


# Global variable to track pending save
_save_timer = None
_save_state = {"update_hpcf_pars": False, "update_brir_pars": False, "preset_name": None}

def save_settings_debounced(update_hpcf_pars=False, update_brir_pars=False, preset_name=None):
    """
    Debounced save_settings that accumulates parameters from multiple rapid calls.
    Ensures no updates are lost.
    """
    global _save_timer, _save_state

    # --- Merge incoming parameters ---
    _save_state["update_hpcf_pars"] = _save_state["update_hpcf_pars"] or update_hpcf_pars
    _save_state["update_brir_pars"] = _save_state["update_brir_pars"] or update_brir_pars
    _save_state["preset_name"] = preset_name or _save_state["preset_name"]

    def do_save():
        global _save_timer, _save_state
        save_settings(
            update_hpcf_pars=_save_state["update_hpcf_pars"],
            update_brir_pars=_save_state["update_brir_pars"],
            preset_name=_save_state["preset_name"]
        )
        # Reset state
        _save_state = {"update_hpcf_pars": False, "update_brir_pars": False, "preset_name": None}
        _save_timer = None

    # Cancel any previously scheduled save
    if _save_timer:
        _save_timer.cancel()

    # Schedule a new save after the debounce interval
    _save_timer = threading.Timer(CN._SAVE_DEBOUNCE_TIME, do_save)
    _save_timer.start()


def save_settings(update_hpcf_pars=False, update_brir_pars=False, preset_name=None):
    """
    Save current GUI settings to the main config file or a preset INI file.

    Args:
        update_hpcf_pars (bool, optional): If True, include HPCF-related settings.
        update_brir_pars (bool, optional): If True, include BRIR-related settings.
        preset_name (str, optional): Optional preset name (saves to <preset_name>.ini).
    """
    __version__ = dpg.get_item_user_data("log_text")
    logz = dpg.get_item_user_data("console_window")

    # --- Determine file path ---
    settings_file = CN.SETTINGS_FILE
    if preset_name:
        settings_dir = CN.SETTINGS_DIR
        settings_file = os.path.join(settings_dir, f"{preset_name}.ini")


    # --- Auto-update HPCF if enabled ---
    if dpg.get_value('qc_auto_apply_hpcf_sel'):
        update_hpcf_pars = True

    # --- Load existing config or initialize ---
    config = configparser.ConfigParser()
    if os.path.exists(settings_file):
        config.read(settings_file)
    else:
        config['DEFAULT'] = {}

    # --- Core metadata ---
    #not stored directly in gui elements
    config['DEFAULT']['version'] = __version__
    config['DEFAULT']['hrtf_list_favs'] = json.dumps(dpg.get_item_user_data("hrtf_add_favourite") or [])
    config['DEFAULT']['path'] = dpg.get_value('selected_folder_base')

    # --- Define deferred keys ---
    hpcf_keys = {'qc_hpcf_brand', 'qc_hpcf_headphone', 'qc_hpcf_sample'}
    brir_keys = {
        'qc_brir_hp_type', 'qc_room_target', 'qc_direct_gain', 'qc_direct_gain_slider',
        'qc_acoustic_space', 'qc_brir_hrtf', 'qc_brir_hrtf_type',
        'qc_brir_hrtf_dataset', 'qc_crossover_f_mode', 'qc_crossover_f',
        'qc_sub_response', 'qc_hp_rolloff_comp', 'qc_fb_filtering_mode'
    }

    # --- Save all keys whose GUI items exist ---
    for key in CN.DEFAULTS.keys():
        try:
            # Skip deferred HPCF keys unless flagged
            if key in hpcf_keys and not update_hpcf_pars:
                continue

            # Skip deferred BRIR keys unless flagged
            if key in brir_keys and not update_brir_pars:
                continue

            # Only save if GUI item exists
            if dpg.does_item_exist(key):
                config['DEFAULT'][key] = str(dpg.get_value(key))

        except Exception as e:
            logging.warning(f"Failed to save key '{key}': {e}")

    # --- Write updated config back to file ---
    try:
        with open(settings_file, 'w') as configfile:
            config.write(configfile)
    
        # log_string = f"Settings successfully saved to '{settings_file}'."
        # if settings_file == CN.SETTINGS_FILE:
        #     hf.log_with_timestamp(log_string=log_string)
        # else:
        #     hf.log_with_timestamp(log_string=log_string, gui_logger=logz)
            
        log_string = f"Preset successfully saved to '{settings_file}'."
        if settings_file != CN.SETTINGS_FILE:
            hf.log_with_timestamp(log_string=log_string, gui_logger=logz)
            
        if CN.LOG_MEMORY:
            hf.log_memory_usage()
    
    except Exception as e:
        log_string = f"Error: Failed to write settings to '{settings_file}' — {e}"
        logging.error(log_string)
        hf.log_with_timestamp(log_string=log_string, gui_logger=logz)







#
#
## GUI Functions - HPCFs
#
#

def get_hpcf_dict():
    """
    Retrieves current GUI selections and available lists for both QC and FDE tabs.
    Uses the active SQLite connection from 'qc_e_apo_sel_hpcf' user data.

    Returns:
        dict: {
            "qc": {
                "brand": str or None,
                "headphone": str or None,
                "sample": str or None,
                "brands_list": list,
                "headphones_list": list,
                "samples_list": list
            },
            "fde": {
                "brand": str or None,
                "headphone": str or None,
                "sample": str or None,
                "brands_list": list,
                "headphones_list": list,
                "samples_list": list
            }
        }
        Example usage
        
        gui_state = get_current_hpcf_gui_state()

        # Access QC tab values
        qc_brand = gui_state["qc"]["brand"]
        qc_headphones = gui_state["qc"]["headphones_list"]
        
        # Access FDE tab values
        fde_sample = gui_state["fde"]["sample"]
    """
    
    # Retrieve shared database connection
    hpcf_db_dict = dpg.get_item_user_data("qc_e_apo_sel_hpcf")
    if not hpcf_db_dict or "conn" not in hpcf_db_dict:
        raise RuntimeError("No valid database connection found in qc_e_apo_sel_hpcf user data.")
    conn = hpcf_db_dict["conn"]

    result = {}

    try:
        for prefix in ("qc", "fde"):
            # Read current GUI selections
            headphone_selected = dpg.get_value(f"{prefix}_hpcf_headphone")
            sample_selected = dpg.get_value(f"{prefix}_hpcf_sample")
    
            # Resolve brand name from headphone (returns None if not found)
            # there should only be one brand value per headphone value so can be derived from headphone
            brand_selected = hpcf_functions.get_brand(conn, headphone_selected) if headphone_selected else None
    
            # Get available lists safely
            brands_list = hpcf_functions.get_brand_list(conn) or []
            hp_list = hpcf_functions.get_headphone_list(conn, brand_selected) or []
            sample_list = hpcf_functions.get_samples_list(conn, headphone_selected) or []
            sample_list_sorted = sorted(sample_list) if sample_list else []
    
            result[prefix] = {
                "brand": brand_selected,
                "headphone": headphone_selected,
                "sample": sample_selected,
                "brands_list": brands_list,
                "headphones_list": hp_list,
                "samples_list": sample_list_sorted
            }
            
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)

    return result



def change_hpcf_database_callback(sender=None, app_data=None, user_data=None):
    #callback for below function
    change_hpcf_database(selected_db=app_data,use_previous_settings=True)

def change_hpcf_database(selected_db,use_previous_settings=False):
    """
    GUI callback to update headphone related gui items based on selected database. 
    This will update titles and list values. The same gui elements are reused for each database but with different labels
    """
    
    try:
        #update conn reference in db dict user data to point to active database
        hpcf_db_dict = dpg.get_item_user_data("qc_e_apo_sel_hpcf")
        
        #ensure both tabs have the same database active at the same time as db dict considers one active db
        dpg.set_value('qc_hpcf_active_database', selected_db)
        dpg.set_value('fde_hpcf_active_database', selected_db)
        # --- update text labels based on schema ---
        #also update active database connection
        if selected_db == CN.HPCF_DATABASE_LIST[0]:#ash filters
            dpg.set_value('qc_hpcf_brand_search_title', "Search Brand:")
            dpg.configure_item('qc_hpcf_brand_search_title', show=True)
            dpg.configure_item('qc_hpcf_brand_search', show=True)
            dpg.set_value('qc_hpcf_headphone_search_title', "Search Headphone:")
            dpg.set_value('qc_hpcf_brand_title', "Brand")
            dpg.set_value('qc_hpcf_headphone_title', "Headphone")
            dpg.set_value('qc_hpcf_sample_title', "Sample")
            dpg.configure_item('qc_hpcf_brand', width=135)
            dpg.configure_item('qc_hpcf_headphone', width=250)
            dpg.configure_item('qc_hpcf_sample', width=115)
            
            dpg.set_value('fde_hpcf_brand_search_title', "Search Brand:")
            dpg.configure_item('fde_hpcf_brand_search_title', show=True)
            dpg.configure_item('fde_hpcf_brand_search', show=True)
            dpg.set_value('fde_hpcf_headphone_search_title', "Search Headphone:")
            dpg.set_value('fde_hpcf_brand_title', "Brand")
            dpg.set_value('fde_hpcf_headphone_title', "Headphone")
            dpg.set_value('fde_hpcf_sample_title', "Sample")
            dpg.configure_item('fde_hpcf_brand', width=135)
            dpg.configure_item('fde_hpcf_headphone', width=250)
            dpg.configure_item('fde_hpcf_sample', width=115)
            
            hpcf_db_dict['conn'] = hpcf_db_dict.get('conn_ash') or hpcf_db_dict['conn']
            hpcf_db_dict['database'] = hpcf_db_dict.get('database_ash') or hpcf_db_dict['database']
        elif selected_db == CN.HPCF_DATABASE_LIST[1]:#compilation database
            dpg.set_value('qc_hpcf_brand_search_title', "Search Type:")
            dpg.configure_item('qc_hpcf_brand_search_title', show=False)
            dpg.configure_item('qc_hpcf_brand_search', show=False)
            dpg.set_value('qc_hpcf_headphone_search_title', "Search Headphone:")
            dpg.set_value('qc_hpcf_brand_title', "Type")
            dpg.set_value('qc_hpcf_headphone_title', "Headphone")
            dpg.set_value('qc_hpcf_sample_title', "Dataset")
            dpg.configure_item('qc_hpcf_brand', width=80)
            dpg.configure_item('qc_hpcf_headphone', width=290)
            dpg.configure_item('qc_hpcf_sample', width=140)
            
            dpg.set_value('fde_hpcf_brand_search_title', "Search Type:")
            dpg.configure_item('fde_hpcf_brand_search_title', show=False)
            dpg.configure_item('fde_hpcf_brand_search', show=False)
            dpg.set_value('fde_hpcf_headphone_search_title', "Search Headphone:")
            dpg.set_value('fde_hpcf_brand_title', "Type")
            dpg.set_value('fde_hpcf_headphone_title', "Headphone")
            dpg.set_value('fde_hpcf_sample_title', "Dataset")
            dpg.configure_item('fde_hpcf_brand', width=80)
            dpg.configure_item('fde_hpcf_headphone', width=290)
            dpg.configure_item('fde_hpcf_sample', width=140)
            
            hpcf_db_dict['conn'] = hpcf_db_dict.get('conn_comp') or hpcf_db_dict['conn']
            hpcf_db_dict['database'] = hpcf_db_dict.get('database_comp') or hpcf_db_dict['database']
        else:
            logging.error(f"Unknown database selection: {selected_db}")
            return
    
        handle_hpcf_update(tab_prefix="qc", update_type="full_reset")
        handle_hpcf_update(tab_prefix="fde", update_type="full_reset")
    
        #attempt to default to previously saved settings for convenience, in case db was switched back to original db. QC tab only.
        if use_previous_settings == True:
            loaded_values = load_settings()
            conn = hpcf_db_dict['conn']
            brand_applied = loaded_values['qc_hpcf_brand']
            headphone_applied = loaded_values['qc_hpcf_headphone']
            sample_applied = loaded_values['qc_hpcf_sample']
            # Ensure lists are not None and not empty
            brands_list = hpcf_functions.get_brand_list(conn) or []
            hp_list = hpcf_functions.get_headphone_list(conn, brand_applied) or []
            sample_list_specific = hpcf_functions.get_samples_list(conn, headphone_applied) or []
            sample_list_sorted = sorted(sample_list_specific)
            # Only proceed if all lists have items and saved values are valid
            if brands_list and hp_list and sample_list_sorted:
                if brand_applied in brands_list and headphone_applied in hp_list and sample_applied in sample_list_sorted:
                    # update brand list
                    dpg.configure_item('qc_hpcf_brand', items=brands_list)
                    dpg.set_value("qc_hpcf_brand", brand_applied)
                    # update headphone list
                    dpg.configure_item('qc_hpcf_headphone', items=hp_list)
                    dpg.set_value("qc_hpcf_headphone", headphone_applied)
                    # update sample list
                    dpg.configure_item('qc_hpcf_sample', items=sample_list_sorted)
                    dpg.set_value("qc_hpcf_sample", sample_applied)
                    # update plot
                    hpcf_functions.hpcf_to_plot(conn, headphone_applied, sample_applied, plot_type=2)
        
        #reset search bars
        dpg.set_value("qc_hpcf_brand_search", "")
        dpg.set_value("qc_hpcf_headphone_search", "")
        dpg.set_value("fde_hpcf_brand_search", "")
        dpg.set_value("fde_hpcf_headphone_search", "")
        #will reset progress bars  
        dpg.set_value("qc_toggle_hpcf_history", False)
        qc_reset_progress()
        reset_hpcf_progress()
        
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)    

def filter_hpcf_lists(tab_prefix, search_type, app_data):
    """
    Update brand/headphone/sample lists for a single tab.

    Parameters
    ----------
    tab_prefix : str
        'fde' or 'qc'
    search_type : str
        'brand' or 'headphone'
    app_data : str
        The search string entered by the user.
    """
    
    try:
        hpcf_db_dict = dpg.get_item_user_data("qc_e_apo_sel_hpcf")
        conn = hpcf_db_dict["conn"]
        brands_list = hpcf_functions.get_brand_list(conn)
        
        # --- UI element tags ---
        brand_combo = f"{tab_prefix}_hpcf_brand"
        headphone_combo = f"{tab_prefix}_hpcf_headphone"
        sample_combo = f"{tab_prefix}_hpcf_sample"
        plot_type = 1 if tab_prefix == "fde" else 2
        hpcf_database_sel = dpg.get_value('fde_hpcf_active_database') if tab_prefix == "fde" else dpg.get_value('qc_hpcf_active_database')
    
        search_str = app_data.strip() if app_data else ""
    
        # --- Determine search results ---
        if search_type == "brand":
            search_results = hpcf_functions.search_brand_list(conn, search_str)
        elif search_type == "headphone":
            search_results = hpcf_functions.search_headphone_list(conn, search_str)
        else:
            return
    
        # --- Determine what to display ---
        if search_results:
            if search_type == "brand":
                brand_list = search_results.copy()
                headphone_list = hpcf_functions.get_headphone_list(conn, brand_list[0])
            else:  # headphone search
                brand_list = []  # clear brand filter
                headphone_list = search_results.copy()
        else:
            # fallback to default full list
            brand_list = brands_list.copy()
            headphone_list = hpcf_functions.get_headphone_list(conn, brand_list[0])
    
        # --- Dependent sample list ---
        headphone = headphone_list[0]
        sample_list = sorted(hpcf_functions.get_samples_list(conn, headphone))
        sample_default = CN.HPCF_SAMPLE_DEFAULT if hpcf_database_sel == CN.HPCF_DATABASE_LIST[0] else sample_list[0]
        #force refresh of brand value to ensure it is up to date and synchronised with headphone
        current_brand=hpcf_functions.get_brand(conn, headphone)
        dpg.set_value(brand_combo, current_brand)
        
        # --- Update brand combo ---
        dpg.configure_item(brand_combo, items=brand_list)
        if brand_list:
            dpg.set_value(brand_combo, brand_list[0])
    
        # --- Update headphone combo ---
        dpg.configure_item(headphone_combo, items=headphone_list)
        dpg.set_value(brand_combo, headphone)
    
        # --- Update sample combo ---
        dpg.configure_item(sample_combo, items=sample_list)
        dpg.set_value(sample_combo, sample_default)
    
        # --- Update plot ---
        hpcf_functions.hpcf_to_plot(conn, headphone, sample_default, plot_type=plot_type)
    
        # --- Reset progress ---
        if tab_prefix == "qc":
            dpg.set_value("qc_toggle_hpcf_history", False)
            qc_reset_progress()
        else:
            reset_hpcf_progress()
    
        save_settings()
        
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)
        
        

def filter_brand_list(sender, app_data):
    """Brand search callback (updates only the relevant tab)."""
    if sender.startswith("qc_"):
        tab_prefix = "qc"
    elif sender.startswith("fde_"):
        tab_prefix = "fde"
    else:
        return

    filter_hpcf_lists(tab_prefix, "brand", app_data)


def filter_headphone_list(sender, app_data):
    """Headphone search callback (updates only the relevant tab)."""
    if sender.startswith("qc_"):
        tab_prefix = "qc"
    elif sender.startswith("fde_"):
        tab_prefix = "fde"
    else:
        return

    filter_hpcf_lists(tab_prefix, "headphone", app_data)



  
    
  
def qc_show_hpcf_history(sender=None, app_data=None):
    """ 
    GUI function to update list of headphone based on exported hpcf files
    """
    
    try:
        hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
        conn = hpcf_db_dict['conn']
        brands_list = hpcf_functions.get_brand_list(conn)
        
        output_path = dpg.get_value('qc_selected_folder_base')
        samples_list_all_distinct =hpcf_functions.get_all_samples_list(conn)
        hp_list_out_latest = e_apo_config_creation.get_exported_hp_list(output_path,samples_list_all_distinct)
        search_str = hp_list_out_latest
        #update brand list with filtered set
        headphone_list_saved = hpcf_functions.search_headphones_in_list(conn, search_str)
        
        
        #attempt to apply previously saved settings for convenience, in case db was switched back to original db. QC tab only
        loaded_values = load_settings()
        conn = hpcf_db_dict['conn']
        brand_applied = loaded_values['qc_hpcf_brand']
        headphone_applied = loaded_values['qc_hpcf_headphone']
        sample_applied = loaded_values['qc_hpcf_sample']
        
        #set list values to previous values if toggle disabled
        #brand=dpg.get_value('qc_hpcf_brand')
        headphone_selected = headphone_applied#dpg.get_value('qc_hpcf_headphone')
        sample_selected = sample_applied#dpg.get_value('qc_hpcf_sample')
        brand_selected=hpcf_functions.get_brand(conn, headphone_selected)
        hp_list_selected = hpcf_functions.get_headphone_list(conn, brand_selected)
        hpcf_database_sel = dpg.get_value('qc_hpcf_active_database')
        
        
        #toggled on and there are saved filters -> populate lists with saved filters
        if headphone_list_saved and headphone_list_saved != None and app_data == True:
            #if selected headphone is in history, set default to selected, otherwise pick first value
            default_headphone = headphone_selected if headphone_selected in headphone_list_saved else headphone_list_saved[0]
        
            #clear out brand list
            dpg.configure_item('qc_hpcf_brand',items=[])
            #force update of brand value to ensure it is up to date and synchronised despite list being blank
            default_brand=hpcf_functions.get_brand(conn, default_headphone)
            dpg.set_value("qc_hpcf_brand", default_brand)
            
            #update headphone list
            dpg.configure_item('qc_hpcf_headphone',items=headphone_list_saved)
            
            #reset headphone value to first headphone
            dpg.set_value("qc_hpcf_headphone", default_headphone)
            
            #also update sample list
            #headphone = headphone_list_specific[0]
            sample_list_specific = hpcf_functions.get_samples_list(conn, default_headphone)
            sample_list_sorted = (sorted(sample_list_specific))
            dpg.configure_item('qc_hpcf_sample',items=sample_list_sorted)
            sample_default = CN.HPCF_SAMPLE_DEFAULT if hpcf_database_sel == CN.HPCF_DATABASE_LIST[0] else sample_list_sorted[0]
            sample_new = sample_selected if headphone_selected in headphone_list_saved else sample_default
      
            #also update plot to Sample A
            #sample = CN.HPCF_SAMPLE_DEFAULT
            hpcf_functions.hpcf_to_plot(conn, default_headphone, sample_new, plot_type=2)
            
            #reset sample list to Sample A
            dpg.set_value("qc_hpcf_sample", sample_new)
    
        #toggled off -> populate list with previously applied filters
        else:
            #validate lists in case of misalignment between applied filter and current DB, will reset
            brands_list, brand_selected = hf.ensure_valid_selection( brands_list, brand_selected)
            hp_list_selected = hpcf_functions.get_headphone_list(conn, brand_selected)#
            hp_list_selected, headphone_selected = hf.ensure_valid_selection(hp_list_selected, headphone_selected)
            sample_list_specific = hpcf_functions.get_samples_list(conn, headphone_selected)
            sample_list_sorted = (sorted(sample_list_specific))
            sample_list_sorted, sample_selected = hf.ensure_valid_selection(sample_list_sorted, sample_selected)
        
            #reset brand list
            dpg.configure_item('qc_hpcf_brand',items=brands_list)
            #reset brand value to first brand
            dpg.set_value("qc_hpcf_brand", brand_selected)
            #update headphone list
            
            dpg.configure_item('qc_hpcf_headphone',items=hp_list_selected)
            dpg.set_value("qc_hpcf_headphone", headphone_selected)
            #also update sample list

            dpg.configure_item('qc_hpcf_sample',items=sample_list_sorted)
            #also update plot
            hpcf_functions.hpcf_to_plot(conn, headphone_selected, sample_selected, plot_type=2)
            #reset sample list
            dpg.set_value("qc_hpcf_sample", sample_selected)
            
        
        dpg.configure_item("qc_clear_history_popup", show=False)
        #reset search bars
        dpg.set_value("qc_hpcf_brand_search", "")
        dpg.set_value("qc_hpcf_headphone_search", "")
        #reset progress
        qc_reset_progress()
        save_settings()

    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)



    
       
def handle_hpcf_update(tab_prefix, update_type, changed_item=None, brand=None, headphone=None, sample=None):
    """
    Unified handler for both FDE and QC tab callbacks.
    Handles ASH and Compilation database hierarchies with dynamic sample selection.

    Parameters:
        tab_prefix (str): 'fde' or 'qc'
        update_type (str): 'full_reset', 'brand', 'headphone', 'sample', or 'full_update'
    """
    
    try:
        # Retrieve DB connection
        hpcf_db_dict = dpg.get_item_user_data("qc_e_apo_sel_hpcf")
        conn = hpcf_db_dict['conn']
    
        # Try to get GUI logger
        # try:
        #     logz = dpg.get_item_user_data("console_window")
        # except Exception:
        #     logz = None
        logz = None
    
        plot_type = 1 if tab_prefix == "fde" else 2
        hpcf_database_sel = dpg.get_value(f'{tab_prefix}_hpcf_active_database')
        reset_func = reset_hpcf_progress if tab_prefix == "fde" else qc_reset_progress
        auto_apply_id = f"{tab_prefix}_auto_apply_hpcf_sel"
        brand_id = f"{tab_prefix}_hpcf_brand"
        headphone_id = f"{tab_prefix}_hpcf_headphone"
        sample_id = f"{tab_prefix}_hpcf_sample"
    
        def refresh_dropdown(tag, items, value):
            if not items:
                return
            dpg.configure_item(tag, show=False)
            dpg.configure_item(tag, items=items)
            dpg.set_value(tag, value)
            dpg.configure_item(tag, show=True)
    
        def update_samples_and_plot(headphone, default_sample=None):
            sample_list = hpcf_functions.get_samples_list(conn, headphone) or []
            sample_list_sorted = sorted(sample_list)
            if default_sample and default_sample in sample_list_sorted:
                sample_to_use = default_sample
            elif sample_list_sorted:
                sample_to_use = sample_list_sorted[0]
            else:
                sample_to_use = None
    
            if sample_to_use:
                hf.log_with_timestamp(f"Plotting HPCF: {headphone} | {sample_to_use}", logz)
                hpcf_functions.hpcf_to_plot(conn, headphone, sample_to_use, plot_type=plot_type)
    
            if sample_list_sorted:
                refresh_dropdown(sample_id, sample_list_sorted, sample_to_use)
    
            current_brand = hpcf_functions.get_brand(conn, headphone)
            if current_brand:
                dpg.set_value(brand_id, current_brand)
    
        # --- MAIN HANDLING LOGIC ---
        if update_type == "full_update":
            hf.log_with_timestamp(f"Performing full HPCF update for tab '{tab_prefix}'", logz)
            if not all([brand, headphone, sample]):
                hf.log_with_timestamp("Full update skipped: missing brand/headphone/sample", logz, log_type=1)
                return
            refresh_dropdown(brand_id, hpcf_functions.get_brand_list(conn) or [], brand)
            refresh_dropdown(headphone_id, hpcf_functions.get_headphone_list(conn, brand) or [], headphone)
            refresh_dropdown(sample_id, sorted(hpcf_functions.get_samples_list(conn, headphone) or []), sample)
            hf.log_with_timestamp(f"Full update set: brand={brand}, headphone={headphone}, sample={sample}", logz)
            hpcf_functions.hpcf_to_plot(conn, headphone, sample, plot_type=plot_type)
    
        elif update_type in ("full_reset", "brand", "headphone"):
            if update_type == "full_reset":
                hf.log_with_timestamp(f"Resetting HPCF selection for tab '{tab_prefix}'", logz)
                brands_list_default = hpcf_functions.get_brand_list(conn) or []
                if not brands_list_default:
                    reset_func()
                    return
                brand_default = brands_list_default[0]
                refresh_dropdown(brand_id, brands_list_default, brand_default)
                brand_to_use = brand_default
            elif update_type == "brand":
                brand_to_use = changed_item
                hf.log_with_timestamp(f"Brand changed: {brand_to_use}", logz)
            else:
                brand_to_use = None
    
            if update_type in ("full_reset", "brand"):
                hp_list = hpcf_functions.get_headphone_list(conn, brand_to_use) or []
                if not hp_list:
                    reset_func()
                    return
                headphone_to_use = hp_list[0]
                refresh_dropdown(headphone_id, hp_list, headphone_to_use)
                hf.log_with_timestamp(f"Headphone list refreshed for brand: {brand_to_use}", logz)
            elif update_type == "headphone":
                headphone_to_use = changed_item
                hf.log_with_timestamp(f"Headphone changed: {headphone_to_use}", logz)
            else:
                headphone_to_use = None
    
            default_sample = CN.HPCF_SAMPLE_DEFAULT if hpcf_database_sel == CN.HPCF_DATABASE_LIST[0] else None
            update_samples_and_plot(headphone_to_use, default_sample=default_sample)
    
        elif update_type == "sample":
            headphone_selected = dpg.get_value(headphone_id)
            sample_to_use = changed_item
            if sample_to_use and headphone_selected:
                hf.log_with_timestamp(f"Sample changed: {sample_to_use} for headphone {headphone_selected}", logz)
                hpcf_functions.hpcf_to_plot(conn, headphone_selected, sample_to_use, plot_type=plot_type)
    
        # Reset progress and save
        reset_func()
        save_settings()
    
        if tab_prefix == "qc" and dpg.get_value(auto_apply_id):
            hf.log_with_timestamp("Auto-applying HPCF in QC tab", logz)
            qc_process_hpcfs()    
            
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)

# FDE tab
def update_headphone_list(sender, app_data):
    handle_hpcf_update("fde", "brand", app_data)

def update_sample_list(sender, app_data):
    handle_hpcf_update("fde", "headphone",  app_data)

def plot_sample(sender, app_data, user_data):
    handle_hpcf_update("fde", "sample",  app_data)

# QC tab
def qc_update_headphone_list(sender, app_data):
    handle_hpcf_update("qc", "brand",  app_data)

def qc_update_sample_list(sender, app_data):
    handle_hpcf_update("qc", "headphone",  app_data)

def qc_plot_sample(sender, app_data, user_data):
    handle_hpcf_update("qc", "sample",  app_data)    
    


 
def export_hpcf_file_toggle(sender, app_data):
    """ 
    GUI function to trigger save and refresh
    """

    save_settings()
    #reset progress
    reset_hpcf_progress()


    
def process_hpcfs(sender=None, app_data=None, user_data=None):
    """GUI function to process HPCFs (FDE tab) safely with validation and logging."""

    try:
        # Retrieve logger and DB connection
        logz = dpg.get_item_user_data("console_window")  # contains logger
        hpcf_db_dict = dpg.get_item_user_data("qc_e_apo_sel_hpcf")
    
        if not hpcf_db_dict or "conn" not in hpcf_db_dict:
            hf.log_with_timestamp("No valid database connection found — skipping FDE HPCF processing.", logz)
            return
        conn = hpcf_db_dict["conn"]
    
        # Get GUI state for both tabs (we only need FDE)
        gui_state = get_hpcf_dict()
        fde_state = gui_state["fde"]
    
        brand = fde_state["brand"]
        headphone = fde_state["headphone"]
        sample = fde_state["sample"]
        brands_list = fde_state["brands_list"]
        hp_list = fde_state["headphones_list"]
        sample_list_sorted = fde_state["samples_list"]
    
        # Validate selections
        if not (brands_list and hp_list and sample_list_sorted):
            hf.log_with_timestamp(f"One or more lists are empty — cannot export HPCFs safely. Brands: {len(brands_list)}, Headphones: {len(hp_list)}, Samples: {len(sample_list_sorted)}", logz)
            return
        if brand not in brands_list or headphone not in hp_list or sample not in sample_list_sorted:
            hf.log_with_timestamp(f"Invalid selection detected — brand={brand}, headphone={headphone}, sample={sample}. Skipping export.", logz)
            return
    
        # Retrieve export options
        output_path = dpg.get_value("selected_folder_base")
    
        fir_export = dpg.get_value("fir_hpcf_toggle")
        fir_stereo_export = dpg.get_value("fir_st_hpcf_toggle")
        geq_export = dpg.get_value("geq_hpcf_toggle")
        geq_31_export = dpg.get_value("geq_31_hpcf_toggle")
        geq_103_export = False
        hesuvi_export = dpg.get_value("hesuvi_hpcf_toggle")
    
        samp_freq_str = dpg.get_value("fde_wav_sample_rate")
        samp_freq_int = CN.SAMPLE_RATE_DICT.get(samp_freq_str)
        bit_depth_str = dpg.get_value("fde_wav_bit_depth")
        bit_depth = CN.BIT_DEPTH_DICT.get(bit_depth_str)
    
        # Perform export
        hpcf_functions.hpcf_to_file_bulk(
            conn,
            primary_path=output_path,
            headphone=headphone,
            fir_export=fir_export,
            fir_stereo_export=fir_stereo_export,
            geq_export=geq_export,
            samp_freq=samp_freq_int,
            bit_depth=bit_depth,
            geq_31_export=geq_31_export,
            geq_103_export=geq_103_export,
            hesuvi_export=hesuvi_export,
            gui_logger=logz,
            report_progress=2,
            force_output=True
        )
    
        save_settings()
        
        # everything is valid and completed
        hf.log_with_timestamp(f" Export filters for {brand} / {headphone}", logz)
        
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)
    
   
def qc_apply_hpcf_params(sender=None, app_data=None):
    """ 
    GUI function to apply hpcf parameters
    """
    force_output=False
    #check if saved hpcf set name is matching with currently selected params
    hpcf_name_full = calc_hpcf_name(full_name=True)
    hpcf_name = calc_hpcf_name(full_name=False)
    sel_hpcf_set=dpg.get_value('qc_e_apo_sel_hpcf')

    if hpcf_name in sel_hpcf_set:#if only sample rate or bit depth changed, force write output
        force_output=True
    #if matching, enable hpcf conv in config
    if hpcf_name_full == sel_hpcf_set:# this is when it was previously disabled but no selection was changed before applying
        dpg.set_value("e_apo_hpcf_conv", True)
        dpg.set_value("qc_e_apo_curr_hpcf", hpcf_name)
        dpg.set_value("qc_progress_bar_hpcf", 1)
        dpg.configure_item("qc_progress_bar_hpcf", overlay = CN.PROGRESS_FIN)
        e_apo_config_acquire()
    else:#else run hpcf processing from scratch
        qc_process_hpcfs(force_output=force_output)
            
 
   
    
def qc_process_hpcfs(force_output=False):
    """ 
    GUI function to process HPCFs safely with validation and logging.
    """

    # Retrieve logger and database connection
    logz = dpg.get_item_user_data("console_window")  # contains logger
    hpcf_db_dict = dpg.get_item_user_data("qc_e_apo_sel_hpcf")
    if not hpcf_db_dict or "conn" not in hpcf_db_dict:
        hf.log_with_timestamp(" No valid database connection found — skipping HPCF processing.", logz)
        return
    conn = hpcf_db_dict["conn"]

    # Get current GUI state
    gui_state = get_hpcf_dict()
    qc_state = gui_state["qc"]

    brand = qc_state["brand"]
    headphone = qc_state["headphone"]
    sample = qc_state["sample"]
    brands_list = qc_state["brands_list"]
    hp_list = qc_state["headphones_list"]
    sample_list_sorted = qc_state["samples_list"]

    # Validate selections
    if not (brands_list and hp_list and sample_list_sorted):
        hf.log_with_timestamp(
            f" One or more lists are empty, cannot process HPCFs safely. Brands: {len(brands_list)}, Headphones: {len(hp_list)}, Samples: {len(sample_list_sorted)}, brand={brand}, headphone={headphone}, sample={sample}",
            logz,
        )
        return
    # Ensure selections are valid within lists
    if brand not in brands_list or headphone not in hp_list or sample not in sample_list_sorted:
        hf.log_with_timestamp(
            f" Invalid selection: brand={brand}, headphone={headphone}, sample={sample}. Skipping processing.",
            logz,
        )
        return

    output_path = dpg.get_value("qc_selected_folder_base")
    fir_export = True
    fir_stereo_export = False
    geq_export = False
    geq_31_export = False
    geq_103_export = False
    hesuvi_export = False
    eapo_export = False

    samp_freq_str = dpg.get_value("qc_wav_sample_rate")
    samp_freq_int = CN.SAMPLE_RATE_DICT.get(samp_freq_str)
    bit_depth_str = dpg.get_value("qc_wav_bit_depth")
    bit_depth = CN.BIT_DEPTH_DICT.get(bit_depth_str)

    # Call processing function
    hpcf_functions.hpcf_to_file_bulk(
        conn,
        primary_path=output_path,
        headphone=headphone,
        fir_export=fir_export,
        fir_stereo_export=fir_stereo_export,
        geq_export=geq_export,
        samp_freq=samp_freq_int,
        bit_depth=bit_depth,
        geq_31_export=geq_31_export,
        geq_103_export=geq_103_export,
        hesuvi_export=hesuvi_export,
        eapo_export=eapo_export,
        gui_logger=logz,
        report_progress=1,
        force_output=force_output,
    )

    # Update configuration
    dpg.set_value("e_apo_hpcf_conv", True)
    e_apo_config_acquire()

    # Update displayed names and save settings
    filter_name = calc_hpcf_name(full_name=False)
    filter_name_full = calc_hpcf_name(full_name=True)
    dpg.set_value("qc_e_apo_curr_hpcf", filter_name)
    dpg.set_value("qc_e_apo_sel_hpcf", filter_name_full)

    save_settings(update_hpcf_pars=True)
    
    # everything is valid and completed
    hf.log_with_timestamp(f" Applied filter for {brand} / {headphone} / {sample}", logz)
  
   
    
#
#
## GUI Functions - BRIRs
#
#




def select_spatial_resolution(sender, app_data):
    """ 
    GUI function to update spatial resolution based on input
    """
    try:
    
        fde_brir_hrtf_type = dpg.get_value('fde_brir_hrtf_type')
        
        #update hrtf list based on spatial resolution
        #also update file format selection based on spatial resolution
        #set some to false and hide irrelevant options
        if app_data == 'High':
    
            dpg.configure_item("ts_brir_toggle", show=True)
            dpg.configure_item("hesuvi_brir_toggle", show=True)
            dpg.configure_item("multi_chan_brir_toggle", show=True)
            dpg.configure_item("sofa_brir_toggle", show=True)
     
            dpg.configure_item("ts_brir_tooltip", show=True)
            dpg.configure_item("hesuvi_brir_tooltip", show=True)
            dpg.configure_item("multi_chan_brir_tooltip", show=True)
            dpg.configure_item("sofa_brir_tooltip", show=True)
            
        else:
    
            dpg.set_value("sofa_brir_toggle", False)
    
            dpg.configure_item("ts_brir_toggle", show=True)
            dpg.configure_item("hesuvi_brir_toggle", show=True)
            dpg.configure_item("multi_chan_brir_toggle", show=True)
            dpg.configure_item("sofa_brir_toggle", show=False)
    
            dpg.configure_item("ts_brir_tooltip", show=True)
            dpg.configure_item("hesuvi_brir_tooltip", show=True)
            dpg.configure_item("multi_chan_brir_tooltip", show=True)
            dpg.configure_item("sofa_brir_tooltip", show=False)
        
      
        if fde_brir_hrtf_type == CN.HRTF_TYPE_LIST[0]:
            brir_hrtf_dataset_list_new = CN.HRTF_TYPE_DATASET_DICT.get(fde_brir_hrtf_type)
            dpg.configure_item('fde_brir_hrtf_dataset',items=brir_hrtf_dataset_list_new)
            brir_hrtf_dataset_new = dpg.get_value('fde_brir_hrtf_dataset')
            hrtf_list_new = hrir_processing.get_listener_list(listener_type=fde_brir_hrtf_type, dataset_name=brir_hrtf_dataset_new)
            dpg.configure_item('fde_brir_hrtf',items=hrtf_list_new)
                
        #reset progress bar
        update_brir_param()
        
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)



    



def select_hp_comp(sender, app_data):
    """ 
    GUI function to update brir based on input
    """
    impulse=CN.IMPULSE
    hp_type = dpg.get_value("fde_brir_hp_type")
    pinna_comp_int = CN.HP_COMP_LIST.index(hp_type)
    pinna_comp = pinna_comp_int
    
    #run plot
    try:
        # load pinna comp filter (FIR)
        npy_fname = pjoin(CN.DATA_DIR_INT, 'headphone_ear_comp_dataset.npy')
        ear_comp_fir_dataset = hf.load_convert_npy_to_float64(npy_fname)
        pinna_comp_fir_loaded = ear_comp_fir_dataset[pinna_comp,:]
        


        pinna_comp_fir=np.zeros(CN.N_FFT)
        pinna_comp_fir[0:1024] = pinna_comp_fir_loaded[0:1024]
        data_fft = np.fft.fft(pinna_comp_fir)
        mag_response=np.abs(data_fft)
        plot_tile = 'Headphone Compensation: ' + hp_type
        hf.plot_data(mag_response, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1,y_lim_a=-10, y_lim_b=10, save_plot=0, normalise=2, level_ends=1, plot_type=1)

    except:
        pass

    #reset progress bar
    update_brir_param()




def update_direct_gain(sender, app_data):
    """ 
    GUI function to update brir based on input
    """
    d_gain=app_data
    dpg.set_value("fde_direct_gain_slider", d_gain)

    #reset progress bar
    update_brir_param()

def update_direct_gain_slider(sender, app_data):
    """ 
    GUI function to update brir based on input
    """
    
    d_gain=app_data
    dpg.set_value("fde_direct_gain", d_gain)

    #reset progress bar
    update_brir_param()
 
def update_crossover_f(sender=None, app_data=None, user_data=None):
    """ 
    GUI function to make updates to freq crossover parameters
    """
    if app_data == CN.SUB_FC_SETTING_LIST[0]:
        ac_space = dpg.get_value("fde_acoustic_space")
        ac_space_int = CN.AC_SPACE_LIST_GUI.index(ac_space)
        ac_space_src = CN.AC_SPACE_LIST_SRC[ac_space_int]
        f_crossover_var,order_var= brir_generation.get_ac_f_crossover(name_src=ac_space_src)
        dpg.set_value("fde_crossover_f", f_crossover_var)
        
    if type(app_data) is int:
        dpg.set_value("fde_crossover_f_mode", CN.SUB_FC_SETTING_LIST[1])#set mode to custom
        
    update_brir_param()

def update_ac_space(sender, app_data, user_data):
    """ 
    GUI function to make updates to acoustic spaces
    """
    crossover_mode = dpg.get_value("fde_crossover_f_mode")
    if crossover_mode == CN.SUB_FC_SETTING_LIST[0]:
        update_crossover_f(app_data=crossover_mode)
        
    update_brir_param()

 
def select_room_target(sender, app_data):
    """Callback for room target tab 1, uses plot_type=1."""
    plot_room_target(app_data, plot_type=1)
    update_brir_param()

def qc_select_room_target(sender, app_data):
    """Callback for room target tab 2 (quick config), uses plot_type=2."""
    plot_room_target(app_data, plot_type=2)
    qc_update_brir_param()    
 
    
def plot_room_target(target_sel: str, plot_type: int):
    """
    Shared function to plot a room target frequency response.

    Parameters:
    - target_sel: str, key name of the room target in ROOM_TARGETS_DICT
    - plot_type: int, different plot styles (1 or 2)
    """
    try:
        target_data = CN.ROOM_TARGETS_DICT.get(target_sel)
        if not target_data:
            logging.warning(f"Room target '{target_sel}' not found in ROOM_TARGETS_DICT.")
            return

        fir = target_data["impulse_response"]

        room_target_fir = np.zeros(CN.N_FFT)
        room_target_fir[:len(fir)] = fir

        data_fft = np.fft.fft(room_target_fir)
        room_target_mag = np.abs(data_fft)

        plot_title = f"{target_sel} frequency response"
        hf.plot_data(
            room_target_mag,
            title_name=plot_title,
            n_fft=CN.N_FFT,
            samp_freq=CN.SAMP_FREQ,
            y_lim_adjust=1,
            y_lim_a=-12,
            y_lim_b=12,
            save_plot=0,
            normalise=2,
            plot_type=plot_type
        )

    except Exception as e:
        logging.error(f"Failed to plot room target '{target_sel}': {e}")
    
 
    
 
def _select_hrtf_common(mode):
    """
    Shared core for both FDE and QC HRTF selection.
    mode: 'fde' or 'qc'
    """
    fr_flat_mag = CN.FR_FLAT_MAG
    brir_dict = get_brir_dict()

    # Map per-mode settings
    if mode == 'fde':
        prefix = 'fde'
        plot_type = 1
        update_func = update_brir_param
    elif mode == 'qc':
        prefix = 'qc'
        plot_type = 2
        update_func = qc_update_brir_param
    else:
        raise ValueError(f"Invalid mode: {mode}")

    brir_hrtf_type = brir_dict.get(f'{prefix}_brir_hrtf_type')
    brir_hrtf_dataset = brir_dict.get(f'{prefix}_brir_hrtf_dataset')
    brir_hrtf_gui = brir_dict.get(f'{prefix}_brir_hrtf')
    brir_hrtf_short = brir_dict.get(f'{prefix}_brir_hrtf_short')

    try:

            
        brir_hrtf_type, brir_hrtf_dataset, brir_hrtf_gui, brir_hrtf_short = brir_generation.brir_hrtf_param_cleaning(
            brir_hrtf_type,
            brir_hrtf_dataset,
            brir_hrtf_gui,
            brir_hrtf_short
        )
        
        spat_res_int = 0

        # --- Construct file path ---
        npy_fname = brir_generation.get_hrir_file_path(
            brir_hrtf_type=brir_hrtf_type,
            brir_hrtf_gui=brir_hrtf_gui,
            brir_hrtf_dataset=brir_hrtf_dataset,
            brir_hrtf_short=brir_hrtf_short,
            spatial_res=spat_res_int
        )

        if os.path.exists(npy_fname):
            hf.log_with_timestamp(f"Loading HRIR file: {npy_fname}")

            # --- Load HRIR .npy file ---
            hrir_list = np.load(npy_fname)
            if not isinstance(hrir_list, np.ndarray):
                raise TypeError(f"Loaded HRIR object is not a NumPy array: {type(hrir_list)}")

            # --- Select first dimension ---
            hrir_selected = hrir_list[0]
            total_elev_hrir = len(hrir_selected)
            total_azim_hrir = len(hrir_selected[0])
            total_chan_hrir = len(hrir_selected[0][0])
            total_samples_hrir = len(hrir_selected[0][0][0])

            elev_min = CN.SPATIAL_RES_ELEV_MIN_IN[spat_res_int]
            elev_nearest = CN.SPATIAL_RES_ELEV_NEAREST_IN[spat_res_int]
            azim_nearest = CN.SPATIAL_RES_AZIM_NEAREST_IN[spat_res_int]

            hf.log_with_timestamp(
                f"Loaded HRIR shape={hrir_selected.shape}, "
                f"elev_min={elev_min}, elev_step={elev_nearest}, azim_step={azim_nearest}"
            )

            # --- Extract HRIR for 0° elevation, 330° azimuth, right ear ---
            hrtf_mag = None
            for elev in range(total_elev_hrir):
                elev_deg = int(elev_min + elev * elev_nearest)
                for azim in range(total_azim_hrir):
                    azim_deg = int(azim * azim_nearest)
                    if elev_deg == 0 and azim_deg == 330:
                        chan = 1
                        hrir = np.zeros(CN.N_FFT)
                        hrir[:total_samples_hrir] = hrir_selected[elev][azim][chan][:total_samples_hrir]
                        data_fft = np.fft.fft(hrir)
                        hrtf_mag = np.abs(data_fft)
                        break
                if hrtf_mag is not None:
                    break

            if hrtf_mag is None:
                raise ValueError("No HRIR found at elevation=0°, azimuth=330°.")

            # --- Plot ---
            plot_title = f"HRTF sample: {brir_hrtf_short} 0° elevation, 30° azimuth, right ear"
            hf.plot_data(
                hrtf_mag,
                title_name=plot_title,
                n_fft=CN.N_FFT,
                samp_freq=CN.SAMP_FREQ,
                y_lim_adjust=1,
                save_plot=0,
                normalise=2,
                level_ends=1,
                plot_type=plot_type
            )

    except Exception as e:
        hf.log_with_timestamp(f"{prefix}_select_hrtf failed: {e}")
        hf.plot_data(
            fr_flat_mag,
            title_name='No Preview Available',
            n_fft=CN.N_FFT,
            samp_freq=CN.SAMP_FREQ,
            y_lim_adjust=1,
            save_plot=0,
            normalise=2,
            plot_type=plot_type
        )

    # --- Reset progress bar ---
    update_func()


# --- Wrapper callbacks ---
def select_hrtf(sender=None, app_data=None):
    _select_hrtf_common('fde')

def qc_select_hrtf(sender=None, app_data=None):
    _select_hrtf_common('qc')
    
    

    
    
    
    

def qc_select_hp_comp(sender, app_data):
    """ 
    GUI function to update brir based on input
    """
    impulse=CN.IMPULSE
    hp_type = dpg.get_value("qc_brir_hp_type")
    pinna_comp_int = CN.HP_COMP_LIST.index(hp_type)
    pinna_comp = pinna_comp_int
    
    #run plot
    try:
   
        # load pinna comp filter (FIR)
        npy_fname = pjoin(CN.DATA_DIR_INT, 'headphone_ear_comp_dataset.npy')
        ear_comp_fir_dataset = hf.load_convert_npy_to_float64(npy_fname)
        pinna_comp_fir_loaded = ear_comp_fir_dataset[pinna_comp,:]
        
        pinna_comp_fir=np.zeros(CN.N_FFT)
        pinna_comp_fir[0:1024] = pinna_comp_fir_loaded[0:1024]
        data_fft = np.fft.fft(pinna_comp_fir)
        mag_response=np.abs(data_fft)
        plot_tile = 'Headphone Compensation: ' + hp_type
        hf.plot_data(mag_response, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, y_lim_a=-10, y_lim_b=10, save_plot=0, normalise=2, level_ends=1, plot_type=2)

    except:
        pass

    #reset progress bar
    qc_update_brir_param()
    
     
def qc_update_crossover_f(sender=None, app_data=None, user_data=None):
    """ 
    GUI function to make updates to freq crossover parameters
    """
    if app_data == CN.SUB_FC_SETTING_LIST[0]:
        ac_space = dpg.get_value("qc_acoustic_space")
        ac_space_int = CN.AC_SPACE_LIST_GUI.index(ac_space)
        ac_space_src = CN.AC_SPACE_LIST_SRC[ac_space_int]
        f_crossover_var,order_var= brir_generation.get_ac_f_crossover(name_src=ac_space_src)
        dpg.set_value("qc_crossover_f", f_crossover_var)
        
    if type(app_data) is int:
        dpg.set_value("qc_crossover_f_mode", CN.SUB_FC_SETTING_LIST[1])#set mode to custom
        
    qc_update_brir_param()
    
def qc_lf_analyse_toggle(sender=None, app_data=None, user_data=None):
    """ 
    GUI function to make updates to LF analysis panel
    """
    #if toggled
    if app_data == True:
        lfa_type = dpg.get_value("qc_lf_analysis_type")
        if lfa_type == 'Magnitude':
            qc_lf_analyse_mag()
        if lfa_type == 'Group Delay':
            qc_lf_analyse_gd()
    else:
        #initial plot
        impulse=np.zeros(CN.N_FFT)
        impulse[0]=1
        fr_flat_mag = np.abs(np.fft.fft(impulse))
        hf.plot_data(fr_flat_mag, title_name='', n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=2, plot_type=3)
            
    qc_update_brir_param()
        
def qc_lf_analyse_change_type(sender=None, app_data=None, user_data=None):
    """ 
    GUI function to make updates to LF analysis panel
    """
    #if toggled
    activated = dpg.get_value("qc_lf_analysis_toggle")
    if activated == True:
        if app_data == 'Magnitude':
            qc_lf_analyse_mag()
        if app_data == 'Group Delay':
            qc_lf_analyse_gd()
            
    qc_update_brir_param()
    
def qc_lf_analyse_mag():
    """ 
    GUI function to plot magnitude response of exported BRIR in LFs
    """
    n_fft=CN.N_FFT
    peak_gain=0#default value
    impulse=np.zeros(n_fft)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat = hf.mag2db(fr_flat_mag)
    brir_in_arr=np.zeros(n_fft)
    
    primary_path=dpg.get_value('qc_selected_folder_base')
    brir_set=CN.FOLDER_BRIRS_LIVE
    brir_set_formatted = brir_set.replace(" ", "_")
    brirs_path = pjoin(primary_path, CN.PROJECT_FOLDER_BRIRS, brir_set_formatted)
    elev=0
    azim_fl=-30
    azim_fr=30
    total_irs=0
    brir_name_wav_l = 'BRIR' + '_E' + str(elev) + '_A' + str(azim_fl) + '.wav'
    brir_name_wav_r = 'BRIR' + '_E' + str(elev) + '_A' + str(azim_fr) + '.wav'
    #brir_name = calc_brir_set_name(full_name=False)
    brir_name = dpg.get_value('qc_e_apo_curr_brir_set')
    
    try:
        #sum 4x channels
        input_chan=0
        total_irs=total_irs+1
        wav_fname = pjoin(brirs_path, brir_name_wav_l)
        samplerate, fir_array = hf.read_wav_file(wav_fname)
        fir_length = min(len(fir_array),n_fft)
        brir_in_arr[0:fir_length]=np.add(brir_in_arr[0:fir_length],fir_array[0:fir_length,input_chan])
        input_chan=1
        total_irs=total_irs+1
        brir_in_arr[0:fir_length]=np.add(brir_in_arr[0:fir_length],fir_array[0:fir_length,input_chan])
        
        input_chan=0
        total_irs=total_irs+1
        wav_fname = pjoin(brirs_path, brir_name_wav_r)
        samplerate, fir_array = hf.read_wav_file(wav_fname)
        fir_length = min(len(fir_array),n_fft)
        brir_in_arr[0:fir_length]=np.add(brir_in_arr[0:fir_length],fir_array[0:fir_length,input_chan])
        input_chan=1
        total_irs=total_irs+1
        brir_in_arr[0:fir_length]=np.add(brir_in_arr[0:fir_length],fir_array[0:fir_length,input_chan])
        
        brir_in_arr=np.divide(brir_in_arr,total_irs)
        
        data_fft = np.fft.fft(brir_in_arr[0:CN.N_FFT])
        mag_fft=np.abs(data_fft)
    
        plot_title = brir_name 
        
        hf.plot_data_generic(mag_fft, title_name=plot_title, data_type='magnitude', n_fft=n_fft, samp_freq=samplerate, y_lim_adjust = 1, y_lim_a=-10, y_lim_b=10, x_lim_adjust = 1,x_lim_a=10, x_lim_b=200, save_plot=0, normalise=1,  plot_type=3)
        
    except Exception as ex:
        
        logging.error("Error occurred", exc_info = ex) 
        
        hf.plot_data(fr_flat_mag, title_name='File not found. Apply parameters to generate dataset', n_fft=n_fft, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=1, plot_type=3)
    
 
def qc_lf_analyse_gd():
    """ 
    GUI function to plot group delay response of exported BRIR in LFs
    """
    n_fft=CN.N_FFT
    peak_gain=0#default value
    impulse=np.zeros(n_fft)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    fr_flat = hf.mag2db(fr_flat_mag)
    brir_in_arr_l=np.zeros(n_fft)
    
    primary_path=dpg.get_value('qc_selected_folder_base')
    brir_set=CN.FOLDER_BRIRS_LIVE
    brir_set_formatted = brir_set.replace(" ", "_")
    brirs_path = pjoin(primary_path, CN.PROJECT_FOLDER_BRIRS, brir_set_formatted)
    elev_fl=0
    azim_fl=-30
    brir_name_wav = 'BRIR' + '_E' + str(elev_fl) + '_A' + str(azim_fl) + '.wav'
    #brir_name = calc_brir_set_name(full_name=False)
    brir_name = dpg.get_value('qc_e_apo_curr_brir_set')
    
    try:

        wav_fname = pjoin(brirs_path, brir_name_wav)
        samplerate, fir_array = hf.read_wav_file(wav_fname)
        fir_length = min(len(fir_array),n_fft)
        brir_in_arr_l[0:fir_length]=fir_array[0:fir_length,0]
  
        freqs, gd = hf.calc_group_delay_from_ir(y=brir_in_arr_l, sr=samplerate, n_fft=n_fft, hop_length=512, smoothing_window=18, smoothing_type = 'octave', system_delay_ms=None)
   
        plot_title = brir_name 

        hf.plot_data_generic(gd, freqs=freqs, title_name=plot_title, data_type='group_delay',n_fft=n_fft, samp_freq=samplerate, y_lim_adjust = 1, y_lim_a=-150, y_lim_b=150, x_lim_adjust = 1,x_lim_a=10, x_lim_b=200,  plot_type=3)
        
    except Exception as ex:
        
        logging.error("Error occurred", exc_info = ex) 
        
        hf.plot_data(fr_flat_mag, title_name='File not found. Apply parameters to generate dataset', n_fft=n_fft, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, save_plot=0, normalise=1, plot_type=3)
    
     
 
    

def qc_update_ac_space(sender, app_data, user_data):
    """ 
    GUI function to make updates to acoustic spaces
    """
    crossover_mode = dpg.get_value("qc_crossover_f_mode")
    if crossover_mode == CN.SUB_FC_SETTING_LIST[0]:
        qc_update_crossover_f(app_data=crossover_mode)
        
    qc_update_brir_param()


def qc_sort_ac_space(sender, app_data, user_data):
    """ 
    GUI function to make updates to acoustic spaces
    """
    if app_data == 'Name':
        dpg.configure_item('qc_acoustic_space',items=CN.AC_SPACE_LIST_GUI)
    elif app_data == 'Reverberation Time':
        unsorted_list = CN.AC_SPACE_LIST_GUI
        values_list = CN.AC_SPACE_MEAS_R60
        sorted_list = hf.sort_names_by_values(names=unsorted_list, values=values_list, descending=False)
        dpg.configure_item('qc_acoustic_space',items=sorted_list)
  
def sort_ac_space(sender, app_data, user_data):
    """ 
    GUI function to make updates to acoustic spaces
    """
    if app_data == 'Name':
        dpg.configure_item('fde_acoustic_space',items=CN.AC_SPACE_LIST_GUI)
    elif app_data == 'Reverberation Time':
        unsorted_list = CN.AC_SPACE_LIST_GUI
        values_list = CN.AC_SPACE_MEAS_R60
        sorted_list = hf.sort_names_by_values(names=unsorted_list, values=values_list, descending=False)
        dpg.configure_item('fde_acoustic_space',items=sorted_list)
  




def qc_update_direct_gain(sender, app_data):
    """ 
    GUI function to update brir based on input
    """
    d_gain=app_data
    dpg.set_value("qc_direct_gain_slider", d_gain)

    #reset progress bar
    qc_update_brir_param()
    

def qc_update_direct_gain_slider(sender, app_data):
    """ 
    GUI function to update brir based on input
    """
    
    d_gain=app_data
    dpg.set_value("qc_direct_gain", d_gain)

    #reset progress bar
    qc_update_brir_param()
    
def select_sub_brir(sender, app_data):
    """ 
    GUI function to update brir based on input
    """

    logz=dpg.get_item_user_data("console_window")#contains logger
    fde_sub_response=app_data
    
    #run plot
    plot_sub_brir(fde_sub_response,1)

    #reset progress bar
    update_brir_param()
    
def qc_select_sub_brir(sender, app_data):
    """ 
    GUI function to update brir based on input
    """

    logz=dpg.get_item_user_data("console_window")#contains logger
    fde_sub_response=app_data
    
    #run plot
    plot_sub_brir(fde_sub_response,2)

    #reset progress bar
    qc_update_brir_param()
    
def plot_sub_brir(name, plot_type):
    
    #run plot
    try:
        
        sub_data=CN.sub_data
        sub_file_name = CN.extract_column(data=sub_data, column='file_name', condition_key='name_gui', condition_value=name, return_all_matches=False)
        sub_folder = CN.extract_column(data=sub_data, column='folder', condition_key='name_gui', condition_value=name, return_all_matches=False)
        #file_name = get_sub_f_name(fde_sub_response=fde_sub_response, gui_logger=gui_logger)
        if sub_folder == 'sub' or sub_folder == 'lf_brir':#default sub responses
            npy_fname = pjoin(CN.DATA_DIR_SUB, sub_file_name+'.npy')
        else:#user sub response
            file_folder = pjoin(CN.DATA_DIR_AS_USER,name)
            npy_fname = pjoin(file_folder, sub_file_name+'.npy')
        
        sub_brir_npy = hf.load_convert_npy_to_float64(npy_fname)
        sub_brir_ir = np.zeros(CN.N_FFT)
        available_samples = min(CN.N_FFT, sub_brir_npy.shape[-1])
        sub_brir_ir[:available_samples] = sub_brir_npy[0, :available_samples]
        #sub_brir_ir[0:CN.N_FFT] = sub_brir_npy[0][0:CN.N_FFT]
        data_fft = np.fft.fft(sub_brir_ir)
        sub_mag=np.abs(data_fft)
    
        plot_tile = name
        hf.plot_data(sub_mag, title_name=plot_tile, n_fft=CN.N_FFT, samp_freq=CN.SAMP_FREQ, y_lim_adjust = 1, y_lim_a=-8, y_lim_b=8, x_lim_adjust = 1,x_lim_a=10, x_lim_b=150, save_plot=0, normalise=1, plot_type=plot_type)


    except:
        pass
    
    
    


def update_brir_param(sender=None, app_data=None):
    """ 
    GUI function to update brir based on input
    """

    #reset progress bar
    reset_brir_progress()
    save_settings()
    

def qc_update_brir_param(sender=None, app_data=None):
    """ 
    GUI function to update brir based on input
    """

    #reset progress bar
    qc_reset_progress()
    save_settings()
    
def export_brir_toggle(sender, app_data):
    """ 
    GUI function to update settings based on toggle
    """
    
    #reset progress bar
    reset_brir_progress()
    save_settings()

def sync_wav_sample_rate(sender, app_data):
    """ 
    GUI function to update settings based on toggle
    """
    dpg.set_value("fde_wav_sample_rate", app_data)
    dpg.set_value("qc_wav_sample_rate", app_data)
    
    #reset progress bar
    qc_reset_progress()
    reset_brir_progress()
    reset_hpcf_progress()
    
    save_settings()

def sync_wav_bit_depth(sender, app_data):
    """ 
    GUI function to update settings based on toggle
    """
    dpg.set_value("fde_wav_bit_depth", app_data)
    dpg.set_value("qc_wav_bit_depth", app_data)
    
    #reset progress bar
    qc_reset_progress()
    reset_brir_progress()
    reset_hpcf_progress()
    
    save_settings()

 
 
def update_hrtf_dataset_list(sender, app_data):
    """ 
    GUI function to update list of hrtf datasets based on selected hrtf type
    """
    
    try:
        if app_data != None:
            brir_hrtf_type_new= app_data
            
            if brir_hrtf_type_new == 'Favourites':
                dpg.configure_item('hrtf_add_favourite',show=False)
                dpg.configure_item('hrtf_remove_favourite',show=True)
                dpg.configure_item('hrtf_average_favourite',show=True)
                dpg.configure_item('open_user_sofa_folder',show=False)
            elif brir_hrtf_type_new == 'User SOFA Input':
                dpg.configure_item('hrtf_add_favourite',show=True)
                dpg.configure_item('hrtf_remove_favourite',show=False)
                dpg.configure_item('hrtf_average_favourite',show=False)
                dpg.configure_item('open_user_sofa_folder',show=True)
            else:
                dpg.configure_item('hrtf_add_favourite',show=True)
                dpg.configure_item('hrtf_remove_favourite',show=False)
                dpg.configure_item('hrtf_average_favourite',show=False)
                dpg.configure_item('open_user_sofa_folder',show=False)
                
            #update spatial res to valid list
            dpg.configure_item('fde_brir_spat_res',items=CN.SPATIAL_RES_LIST)
        
            brir_hrtf_dataset_list_new = CN.HRTF_TYPE_DATASET_DICT.get(brir_hrtf_type_new)
            #update dataset list with filtered type
            dpg.configure_item('fde_brir_hrtf_dataset',items=brir_hrtf_dataset_list_new)
            brir_hrtf_dataset_new=brir_hrtf_dataset_list_new[0]
            #reset dataset value to first dataset
            dpg.configure_item('fde_brir_hrtf_dataset',show=False)
            dpg.configure_item('fde_brir_hrtf_dataset',default_value=brir_hrtf_dataset_new)
            dpg.configure_item('fde_brir_hrtf_dataset',show=True)
            
            #hrtf list based on dataset and hrtf type
            hrtf_list_new = hrir_processing.get_listener_list(listener_type=brir_hrtf_type_new, dataset_name=brir_hrtf_dataset_new)
            dpg.configure_item('fde_brir_hrtf',items=hrtf_list_new)
            brir_hrtf_new=hrtf_list_new[0]
            #reset listener value to first listener
            dpg.configure_item('fde_brir_hrtf',show=False)
            dpg.configure_item('fde_brir_hrtf',default_value=brir_hrtf_new)
            dpg.configure_item('fde_brir_hrtf',show=True)
            select_hrtf()
            #reset progress bar
            reset_brir_progress()
            save_settings()
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)
            
    
def update_hrtf_list(sender, app_data):
    """ 
    GUI function to update list of hrtfs based on selected dataset
    """
    try:
        if app_data != None:
            brir_hrtf_type_new = (dpg.get_value("fde_brir_hrtf_type"))
            brir_hrtf_dataset_new= app_data
    
            #hrtf list based on dataset and hrtf type
            hrtf_list_new = hrir_processing.get_listener_list(listener_type=brir_hrtf_type_new, dataset_name=brir_hrtf_dataset_new)
            dpg.configure_item('fde_brir_hrtf',items=hrtf_list_new)
            if hrtf_list_new:
                brir_hrtf_new = hrtf_list_new[0]
            else:
                brir_hrtf_new = 'No HRTFs Found'
            #reset listener value to first listener
            dpg.configure_item('fde_brir_hrtf',show=False)
            dpg.configure_item('fde_brir_hrtf',default_value=brir_hrtf_new)
            dpg.configure_item('fde_brir_hrtf',show=True)
            select_hrtf()
            #reset progress bar
            reset_brir_progress()
            save_settings()
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e) 
        
def qc_update_hrtf_dataset_list(sender, app_data):
    """ 
    GUI function to update list of hrtf datasets based on selected hrtf type
    """
    try:
        if app_data != None:
            qc_brir_hrtf_type_new= app_data
            
            if qc_brir_hrtf_type_new == 'Favourites':
                dpg.configure_item('qc_hrtf_add_favourite',show=False)
                dpg.configure_item('qc_hrtf_remove_favourite',show=True)
                dpg.configure_item('qc_hrtf_average_favourite',show=True)
                dpg.configure_item('qc_open_user_sofa_folder',show=False)
            elif qc_brir_hrtf_type_new == 'User SOFA Input':
                dpg.configure_item('qc_hrtf_add_favourite',show=True)
                dpg.configure_item('qc_hrtf_remove_favourite',show=False)
                dpg.configure_item('qc_hrtf_average_favourite',show=False)
                dpg.configure_item('qc_open_user_sofa_folder',show=True)
            else:
                dpg.configure_item('qc_hrtf_add_favourite',show=True)
                dpg.configure_item('qc_hrtf_remove_favourite',show=False)
                dpg.configure_item('qc_hrtf_average_favourite',show=False)
                dpg.configure_item('qc_open_user_sofa_folder',show=False)
            
            qc_brir_hrtf_dataset_list_new = CN.HRTF_TYPE_DATASET_DICT.get(qc_brir_hrtf_type_new)
            #update dataset list with filtered type
            dpg.configure_item('qc_brir_hrtf_dataset',items=qc_brir_hrtf_dataset_list_new)
            qc_brir_hrtf_dataset_new=qc_brir_hrtf_dataset_list_new[0]
            #reset dataset value to first dataset
            dpg.configure_item('qc_brir_hrtf_dataset',show=False)
            dpg.configure_item('qc_brir_hrtf_dataset',default_value=qc_brir_hrtf_dataset_new)
            dpg.configure_item('qc_brir_hrtf_dataset',show=True)
            
            #qc hrtf list based on dataset and hrtf type
            qc_hrtf_list_new = hrir_processing.get_listener_list(listener_type=qc_brir_hrtf_type_new, dataset_name=qc_brir_hrtf_dataset_new)
            dpg.configure_item('qc_brir_hrtf',items=qc_hrtf_list_new)
            qc_brir_hrtf_new=qc_hrtf_list_new[0]
            #reset value to first hrtf
            dpg.configure_item('qc_brir_hrtf',show=False)
            dpg.configure_item('qc_brir_hrtf',default_value=qc_brir_hrtf_new)
            dpg.configure_item('qc_brir_hrtf',show=True)
            qc_select_hrtf()
            #reset progress bar
            qc_reset_progress()
            save_settings()
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)        

def qc_update_hrtf_list(sender, app_data):
    """ 
    GUI function to update list of hrtfs based on selected dataset
    """
    try:
        if app_data != None:
            qc_brir_hrtf_type_new = (dpg.get_value("qc_brir_hrtf_type"))
            qc_brir_hrtf_dataset_new= app_data
    
            #qc hrtf list based on dataset and hrtf type
            qc_hrtf_list_new = hrir_processing.get_listener_list(listener_type=qc_brir_hrtf_type_new, dataset_name=qc_brir_hrtf_dataset_new)
            dpg.configure_item('qc_brir_hrtf',items=qc_hrtf_list_new)
            if qc_hrtf_list_new:
                qc_brir_hrtf_new = qc_hrtf_list_new[0]
            else:
                qc_brir_hrtf_new = 'No HRTFs Found'
            #reset listener value to first listener
            dpg.configure_item('qc_brir_hrtf',show=False)
            dpg.configure_item('qc_brir_hrtf',default_value=qc_brir_hrtf_new)
            dpg.configure_item('qc_brir_hrtf',show=True)
            
            qc_select_hrtf()
            #reset progress bar
            qc_reset_progress()
            save_settings()
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)        
        
def start_process_brirs(sender, app_data, user_data):
    """ 
    GUI function to start or stop head tracking thread
    """
    #thread bool
    process_brirs_running=dpg.get_item_user_data("brir_tag")
    
    if process_brirs_running == False:

        #set thread running flag
        process_brirs_running = True
        #update user data
        dpg.configure_item('brir_tag',user_data=process_brirs_running)
        dpg.configure_item('brir_tag',label="Cancel")
        
        #set stop thread flag flag
        stop_thread_flag = False
        #update user data
        dpg.configure_item('progress_bar_brir',user_data=stop_thread_flag)
        
        #start thread
        thread = threading.Thread(target=process_brirs, args=(), daemon=True)
        thread.start()
   
    else:
        
        #set stop thread flag flag
        stop_thread_flag = True
        #update user data
        dpg.configure_item('progress_bar_brir',user_data=stop_thread_flag)

def process_brirs(sender=None, app_data=None, user_data=None):
    """ 
    GUI function to process BRIRs
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    #grab parameters
    brir_directional_export = (dpg.get_value("dir_brir_toggle"))
    brir_ts_export = (dpg.get_value("ts_brir_toggle"))
    hesuvi_export = (dpg.get_value("hesuvi_brir_toggle"))
    multichan_export = (dpg.get_value("multi_chan_brir_toggle"))
    multichan_mapping = (dpg.get_value("mapping_16ch_wav"))
    sofa_export = (dpg.get_value("sofa_brir_toggle"))
    spat_res = dpg.get_value("fde_brir_spat_res")
    spat_res_int = CN.SPATIAL_RES_LIST.index(spat_res)
    output_path = dpg.get_value('selected_folder_base')
    sofa_conv=dpg.get_value("sofa_exp_conv")
    
    #grab parameters
    brir_dict_params=get_brir_dict()
    
    #calculate name

    #20250622: fix name related bug
    brir_name = calc_brir_set_name(full_name=False,tab=1)
    

    """
    #Run BRIR integration
    """
    
    brir_gen, status = brir_generation.generate_integrated_brir(brir_name=brir_name, spatial_res=spat_res_int, report_progress=2, gui_logger=logz, brir_dict=brir_dict_params)
    
    """
    #Run BRIR export
    """
   
    if brir_gen.size != 0 and status == 0:
    
        brir_export.export_brir(brir_arr=brir_gen, brir_name=brir_name, primary_path=output_path, gui_logger=logz, spatial_res=spat_res_int,
                             brir_dir_export=brir_directional_export, brir_ts_export=brir_ts_export, hesuvi_export=hesuvi_export, 
                               sofa_export=sofa_export, multichan_export=multichan_export, multichan_mapping=multichan_mapping, brir_dict=brir_dict_params, sofa_conv=sofa_conv)
        
        #set progress to 100 as export is complete (assume E-APO export time is negligible)
        progress = 100/100
        hf.update_gui_progress(report_progress=2, progress=progress, message='Processed')
    elif status == 1:
        progress = 0/100
        hf.update_gui_progress(report_progress=2, progress=progress, message='Failed to generate dataset. Refer to log')
    elif status == 2:
        progress = 0/100
        hf.update_gui_progress(report_progress=2, progress=progress, message='Cancelled')
        


    
    #also reset thread running flag
    process_brirs_running = False
    #update user data
    dpg.configure_item('brir_tag',user_data=process_brirs_running)
    dpg.configure_item('brir_tag',label="Process")
    #set stop thread flag flag
    stop_thread_flag = False
    #update user data
    dpg.configure_item('progress_bar_brir',user_data=stop_thread_flag)

def e_apo_toggle_brir_gui(sender=None, app_data=None):
    """ 
    GUI function to toggle brir convolution
    app_data is the toggle
    
    """
    aquire_config=True
    use_stored_brirs=False
    force_run_process=False

    e_apo_toggle_brir_custom(activate=app_data, aquire_config=aquire_config, use_stored_brirs=use_stored_brirs, force_run_process=force_run_process)


def e_apo_toggle_brir_custom(activate=False, aquire_config=True, use_stored_brirs=False, force_run_process=False):
    """ 
    GUI function to toggle brir convolution - with custom parameters passed in
    app_data is the toggle
    
    """
    
    process_brirs_running=dpg.get_item_user_data("qc_brir_tag")
    if activate == False:#toggled off
        dpg.set_value("qc_e_apo_curr_brir_set", '')
        #call main config writer function
        if aquire_config==True:#custom parameter will be none if called by gui
            e_apo_config_acquire()
        
        if process_brirs_running == True:
            #stop processing if already processing brirs
            qc_stop_process_brirs()
        else:
            #reset progress
            dpg.set_value("qc_progress_bar_brir", 0)
            dpg.configure_item("qc_progress_bar_brir", overlay = CN.PROGRESS_START)
        
    else:#toggled on
        #check if saved brir set name is matching with currently selected params
        brir_name_full = calc_brir_set_name(full_name=True)
        brir_name = calc_brir_set_name(full_name=False)
        sel_brir_set=dpg.get_value('qc_e_apo_sel_brir_set')
        #if matching and not forced to run, enable brir conv in config
        if brir_name_full == sel_brir_set and force_run_process==False:#custom parameter will be none if called by gui
            dpg.set_value("e_apo_brir_conv", True)
            dpg.set_value("qc_e_apo_curr_brir_set", brir_name)
            dpg.set_value("qc_progress_bar_brir", 1)
            dpg.configure_item("qc_progress_bar_brir", overlay = CN.PROGRESS_FIN)
            e_apo_activate_direction(force_reset=True)#run in case direction not found in case of reduced dataset
            if aquire_config==True:#custom parameter will be none if called by gui
                e_apo_config_acquire()
        else:#else run brir processing from scratch
            if process_brirs_running == False:#only start if not already running
                qc_start_process_brirs(use_stored_brirs=use_stored_brirs)




def qc_apply_brir_params(sender=None, app_data=None):
    """ 
    GUI function to apply brir parameters, used for button press
    """

    #check if saved brir set name is matching with currently selected params
    brir_name = calc_brir_set_name(full_name=False)
    brir_name_full = calc_brir_set_name(full_name=True)
    sel_brir_set=dpg.get_value('qc_e_apo_sel_brir_set')
    #if matching, enable brir conv in config
    if brir_name_full == sel_brir_set:
        dpg.set_value("e_apo_brir_conv", True)
        dpg.set_value("qc_e_apo_curr_brir_set", brir_name)
        dpg.set_value("qc_progress_bar_brir", 1)
        dpg.configure_item("qc_progress_bar_brir", overlay = CN.PROGRESS_FIN)
        e_apo_activate_direction(force_reset=True)#run in case direction not found due to reduced dataset
        e_apo_config_acquire()
    else:#else run brir processing from scratch
        qc_start_process_brirs()#this may trigger a cancel if already running
        
   
def qc_start_process_brirs(use_stored_brirs=False):
    """ 
    GUI function to start or stop processing of BRIRs thread
    """
    #thread bool
    process_brirs_running=dpg.get_item_user_data("qc_brir_tag")
    
    if process_brirs_running == False:#if not already running

        #set thread running flag
        process_brirs_running = True
        #update user data
        dpg.configure_item('qc_brir_tag',user_data=process_brirs_running)
        dpg.configure_item('qc_brir_tag',label="Cancel")
        
        #reset stop thread flag flag
        stop_thread_flag = False
        #update user data
        dpg.configure_item('qc_progress_bar_brir',user_data=stop_thread_flag)
        
        #start thread
        thread = threading.Thread(target=qc_process_brirs, args=(use_stored_brirs,), daemon=True)
        thread.start()
   
    else:#if already running, cancel it
        
        qc_stop_process_brirs()

def qc_stop_process_brirs():
    """ 
    GUI function to stop processing of BRIRs thread
    """
    #set stop thread flag flag
    stop_thread_flag = True
    #update user data
    dpg.configure_item('qc_progress_bar_brir',user_data=stop_thread_flag)
    

def qc_process_brirs(use_stored_brirs=False):
    """ 
    GUI function to process BRIRs
    use_stored_brirs: bool, true = this function should use a prepopulated dict list containing BRIR data, false = generate new brir dataset
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    #preset
    brir_directional_export = True
    brir_ts_export = False
    hesuvi_export = False
    sofa_export = False
    spat_res_int = 0
    out_dataset_name = CN.FOLDER_BRIRS_LIVE 
    brir_name = calc_brir_set_name(full_name=False)
    brir_name_full = calc_brir_set_name(full_name=True)
    reduce_dataset = True
    output_path = dpg.get_value('qc_selected_folder_base')
    
    #grab parameters
    brir_dict_params=get_brir_dict()
 
    log_string = 'Processing: ' + brir_name
    hf.log_with_timestamp(log_string, logz)
    

    #contains previously processed brirs
    brir_dict_list = dpg.get_item_user_data("e_apo_brir_conv") or []
    #flag to indicate stored directions should be used in output, cases where direction was missing and needs to be written before applying direction
    force_use_brir_dict = dpg.get_item_user_data("qc_e_apo_curr_brir_set") or False
    
    # Decide which brir dict to use for output directions
    if use_stored_brirs == True or force_use_brir_dict == True:
        brir_dict_out_config = dpg.get_item_user_data("qc_e_apo_sel_brir_set") or {}#grab desired directions from previously stored values, in case of direction change where brirs dont exist
    else:
        brir_dict_out_config = get_brir_dict()#grab relevant config data from current gui elements
    
    
    """
    #Run BRIR integration
    """
    #only generate if prepop brirs not provided
    if not use_stored_brirs or not brir_dict_list:
        brir_gen, status = brir_generation.generate_integrated_brir(brir_name=out_dataset_name, spatial_res=spat_res_int, report_progress=1, gui_logger=logz, brir_dict=brir_dict_params)
    else:
        brir_gen = np.array([])
        status = 0
    
    
    """
    #Run BRIR export
    """
    if (brir_gen.size != 0 and status == 0) or (use_stored_brirs == True and brir_dict_list):#either brir dataset was generated or dict list was provided
        #mute and disable conv before exporting files -> avoids conflicts
        gain_oa_selected=dpg.get_value('e_apo_gain_oa')
        dpg.set_value("e_apo_gain_oa", CN.EAPO_MUTE_GAIN)
        dpg.set_value("e_apo_brir_conv", False)
        e_apo_config_acquire(estimate_gain=False)
        #run export function
        brir_dict_list_new = brir_export.export_brir(brir_arr=brir_gen, brir_name=out_dataset_name, primary_path=output_path, 
                             brir_dir_export=brir_directional_export, brir_ts_export=brir_ts_export, hesuvi_export=hesuvi_export, 
                            gui_logger=logz,  spatial_res=spat_res_int, sofa_export=sofa_export, reduce_dataset=reduce_dataset, brir_dict=brir_dict_out_config,
                            use_stored_brirs=use_stored_brirs, brir_dict_list=brir_dict_list)
    
        #set progress to 100 as export is complete (assume E-APO export time is negligible)
        progress = 100/100
        hf.update_gui_progress(report_progress=1, progress=progress, message='Processed')
        #rewrite config file
        dpg.set_value("e_apo_brir_conv", True)
        #save dict list within gui element
        if brir_dict_list_new:#only update if not empty list
            # Prevent memory leak by clearing old references
            old_data = dpg.get_item_user_data("e_apo_brir_conv") or []
            for item in old_data:
                if isinstance(item, dict):
                    item.clear()
            old_data.clear()
            dpg.set_item_user_data("e_apo_brir_conv", None)# Ensure no lingering references
            dpg.set_item_user_data("e_apo_brir_conv", brir_dict_list_new)#use later if changing directions
                        
            
        if use_stored_brirs == False:#only update if normal process
            #update current brir set text
            dpg.set_value("qc_e_apo_curr_brir_set", brir_name)
            dpg.set_value("qc_e_apo_sel_brir_set", brir_name_full)
        #unmute before writing configs once more
        dpg.set_value("e_apo_gain_oa", gain_oa_selected)
        dpg.set_value("e_apo_brir_conv", True)
        #wait before updating config
        sleep(0.1)
        #update directions to previously stored desired values if flagged
        if use_stored_brirs == True or force_use_brir_dict == True:
            e_apo_update_direction(aquire_config=False, brir_dict_new=brir_dict_out_config)
        #Reset user data flag, dont use stored config dict for next output unless flagged again
        dpg.configure_item('qc_e_apo_curr_brir_set',user_data=False)
        #rewrite config file, will also save settings
        e_apo_config_acquire()
        #only update settings if following normal process, new brirs were generated which means those parameters should be saved
        if use_stored_brirs == False:
            save_settings(update_brir_pars=True)
            #if live set, also write a file containing name of dataset
            out_file_path = pjoin(output_path, CN.PROJECT_FOLDER_BRIRS,out_dataset_name,'dataset_name.txt')
            with open(out_file_path, 'w') as file:
                file.write(brir_name)

            
    elif status == 1:
        progress = 0/100
        hf.update_gui_progress(report_progress=1, progress=progress, message='Failed to generate dataset. Refer to log')
    elif status == 2:
        progress = 0/100
        hf.update_gui_progress(report_progress=1, progress=progress, message='Cancelled')
  
    #also reset thread running flag
    process_brirs_running = False
    #update user data
    dpg.configure_item('qc_brir_tag',user_data=process_brirs_running)
    dpg.configure_item('qc_brir_tag',label=CN.PROCESS_BUTTON_BRIR)
    #set stop thread flag flag
    stop_thread_flag = False
    #update user data
    dpg.configure_item('qc_progress_bar_brir',user_data=stop_thread_flag)
    if status == 0:
        #reset progress in case of changed settings
        qc_reset_progress()
    #plot LF analysis if toggled
    activated = dpg.get_value("qc_lf_analysis_toggle")
    if activated:
        qc_lf_analyse_toggle(app_data=activated)

def calc_brir_set_name(full_name=True,tab=0):
    """ 
    GUI function to calculate brir set name from currently selected parameters
    """
    brir_dict=get_brir_dict()
    if tab == 0:#quick config tab
        
        
        qc_room_target_name = brir_dict.get("qc_room_target")
        qc_target_name_short = CN.ROOM_TARGETS_DICT[qc_room_target_name]["short_name"]
        
        qc_direct_gain_db = brir_dict.get("qc_direct_gain_db")
        qc_ac_space_short = brir_dict.get("qc_ac_space_short")
        qc_pinna_comp = brir_dict.get("qc_pinna_comp")
        qc_sample_rate = brir_dict.get("qc_samp_freq_str")
        qc_bit_depth = brir_dict.get("qc_bit_depth_str")
        qc_brir_hrtf_short=brir_dict.get('qc_brir_hrtf_short')
        
        qc_crossover_f=brir_dict.get('qc_crossover_f')
        qc_sub_response=brir_dict.get('qc_sub_response')
        qc_sub_response_short=brir_dict.get('qc_sub_response_short')
        qc_hp_rolloff_comp=brir_dict.get('qc_hp_rolloff_comp')
        qc_fb_filtering=brir_dict.get('qc_fb_filtering')
        
        hrtf_symmetry = brir_dict.get("hrtf_symmetry")
        er_delay_time = brir_dict.get("er_delay_time")
        hrtf_polarity = brir_dict.get("hrtf_polarity")
        
        averaged_last_saved = get_avg_hrtf_timestamp()
        
     
        if full_name==True:
            brir_name = (qc_brir_hrtf_short + ' '+qc_ac_space_short + ' ' + str(qc_direct_gain_db) + 'dB ' + qc_target_name_short + ' ' + CN.HP_COMP_LIST_SHORT[qc_pinna_comp] 
                         + ' ' + qc_sample_rate + ' ' + qc_bit_depth + ' ' + hrtf_symmetry + ' ' + str(er_delay_time) + ' ' + str(qc_crossover_f) + ' ' + str(qc_sub_response) 
                         + ' ' + str(qc_hp_rolloff_comp) + ' ' + str(qc_fb_filtering) + ' ' + hrtf_polarity + ' ' + averaged_last_saved )
        else:
            brir_name = (qc_brir_hrtf_short + ', '+qc_ac_space_short + ', ' + str(qc_direct_gain_db) + 'dB, ' + qc_target_name_short + ', ' + qc_sub_response_short
                         + '-' +str(qc_crossover_f) + ', ' + CN.HP_COMP_LIST_SHORT[qc_pinna_comp] )

    else:#filter and dataset tab

        fde_room_target_name = brir_dict.get("fde_room_target")
        fde_target_name_short = CN.ROOM_TARGETS_DICT[fde_room_target_name]["short_name"]
        
        fde_direct_gain_db = brir_dict.get("fde_direct_gain_db")
        fde_ac_space_short = brir_dict.get("fde_ac_space_short")
        fde_pinna_comp = brir_dict.get("fde_pinna_comp")
        fde_sample_rate = brir_dict.get("fde_samp_freq_str")
        fde_bit_depth = brir_dict.get("fde_bit_depth_str")
        fde_brir_hrtf_short=brir_dict.get('fde_brir_hrtf_short')
        
        hrtf_symmetry = brir_dict.get("hrtf_symmetry")
        er_delay_time = brir_dict.get("er_delay_time")
        
        fde_crossover_f=brir_dict.get('fde_crossover_f')
        fde_sub_response=brir_dict.get('fde_sub_response')
        fde_sub_response_short=brir_dict.get('fde_sub_response_short')
        fde_hp_rolloff_comp=brir_dict.get('fde_hp_rolloff_comp')
        fde_fb_filtering=brir_dict.get('fde_fb_filtering')
        
        brir_name = (fde_brir_hrtf_short + ', '+fde_ac_space_short + ', ' + str(fde_direct_gain_db) + 'dB, ' + fde_target_name_short + ', ' + fde_sub_response_short
                     + '-' +str(fde_crossover_f) + ', ' + CN.HP_COMP_LIST_SHORT[fde_pinna_comp] )

        

    return brir_name

def calc_hpcf_name(full_name=True):
    """ 
    GUI function to calculate hpcf from currently selected parameters
    """
    headphone = dpg.get_value('qc_hpcf_headphone')
    sample = dpg.get_value('qc_hpcf_sample')
    sample_rate = dpg.get_value('qc_wav_sample_rate')
    bit_depth = dpg.get_value('qc_wav_bit_depth')
    if full_name==True:
        filter_name = headphone + ', ' + sample + ', ' + sample_rate + ', ' + bit_depth
    else:
        filter_name = headphone + ', ' + sample


    return filter_name


#
#
## misc tools and settings
#
#


def show_selected_folder(sender, files, cancel_pressed):
    """ 
    GUI function to process selected folder
    """
    if not cancel_pressed:
        base_folder_selected=files[0]
        ash_folder_selected=pjoin(base_folder_selected, CN.PROJECT_FOLDER)
        dpg.set_value('selected_folder_base', base_folder_selected)
        dpg.set_value('selected_folder_ash', ash_folder_selected)
        dpg.set_value('selected_folder_ash_tooltip', ash_folder_selected)
        #hesuvi path
        if 'EqualizerAPO' in base_folder_selected:
            hesuvi_path_selected = pjoin(base_folder_selected,'HeSuVi')#stored outside of project folder (within hesuvi installation)
        else:
            hesuvi_path_selected = pjoin(base_folder_selected, CN.PROJECT_FOLDER,'HeSuVi')#stored within project folder
        dpg.set_value('selected_folder_hesuvi', hesuvi_path_selected)
        dpg.set_value('selected_folder_hesuvi_tooltip', hesuvi_path_selected)
        save_settings()
       
    



def remove_brirs(sender, app_data, user_data):
    """ 
    GUI function to delete generated BRIRs
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    base_folder_selected=dpg.get_value('selected_folder_base')
    brir_export.remove_brirs(base_folder_selected, gui_logger=logz)    
    base_folder_selected=dpg.get_value('qc_selected_folder_base')
    brir_export.remove_brirs(base_folder_selected, gui_logger=logz)  
    #disable brir convolution
    dpg.set_value("qc_e_apo_sel_brir_set", 'Deleted')
    dpg.set_value("e_apo_brir_conv", False)
    e_apo_toggle_brir_gui(app_data=False)
    
    
    
def remove_hpcfs(sender, app_data, user_data):
    """ 
    GUI function to remove generated HpCFs
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    base_folder_selected=dpg.get_value('selected_folder_base')
    hpcf_functions.remove_hpcfs(base_folder_selected, gui_logger=logz) 
    base_folder_selected=dpg.get_value('qc_selected_folder_base')
    hpcf_functions.remove_hpcfs(base_folder_selected, gui_logger=logz) 
    #disable hpcf convolution
    dpg.set_value("e_apo_hpcf_conv", False)
    dpg.set_value("qc_e_apo_sel_hpcf", 'Deleted')
    e_apo_toggle_hpcf_gui(app_data=False)
    
    #disable show history
    dpg.set_value("qc_toggle_hpcf_history", False)
    qc_show_hpcf_history(app_data=False)
    

#
# Equalizer APO configuration functions
#

def e_apo_auto_apply_hpcf(sender, app_data, user_data):
    """ 
    GUI function to toggle auto apply hpcf convolution
    """
    if app_data == True:
        e_apo_toggle_hpcf_gui(app_data=True)
    

def e_apo_toggle_hpcf_gui(sender=None, app_data=True):
    """ 
    GUI function to toggle hpcf convolution
    """
    aquire_config=True
    e_apo_toggle_hpcf_custom(activate=app_data, aquire_config=aquire_config)
        
def e_apo_toggle_hpcf_custom(activate=False, aquire_config=True):
    """ 
    GUI function to toggle hpcf convolution
    """
    force_output=False
    if activate == False:
        dpg.set_value("qc_e_apo_curr_hpcf", '')
        #call main config writer function
        if aquire_config==True or aquire_config==None:#custom parameter will be none if called by gui
            e_apo_config_acquire()
        #reset progress
        dpg.set_value("qc_progress_bar_hpcf", 0)
        dpg.configure_item("qc_progress_bar_hpcf", overlay = CN.PROGRESS_START)
    else:
        #check if saved hpcf set name is matching with currently selected params
        hpcf_name_full = calc_hpcf_name(full_name=True)
        hpcf_name = calc_hpcf_name(full_name=False)
        if hpcf_name in hpcf_name_full:#if only sample rate or bit depth changed, force write output
            force_output=True
        sel_hpcf_set=dpg.get_value('qc_e_apo_sel_hpcf')#most recently used hpcf name
        #if matching, enable hpcf conv in config
        if hpcf_name_full == sel_hpcf_set:#this is when user toggles off and on but didnt change selection
            dpg.set_value("e_apo_hpcf_conv", True)
            dpg.set_value("qc_e_apo_curr_hpcf", hpcf_name)
            dpg.set_value("qc_progress_bar_hpcf", 1)
            dpg.configure_item("qc_progress_bar_hpcf", overlay = CN.PROGRESS_FIN)
            if aquire_config==True or aquire_config==None:#custom parameter will be none if called by gui
                e_apo_config_acquire()
        else:#else run hpcf processing from scratch
            qc_process_hpcfs(force_output=force_output)

def get_brir_dict():
    """ 
    GUI function to get inputs relating to brirs and store in a dict, returns a dict
    """
            
    #brir related selections
    enable_brir_selected=dpg.get_value('e_apo_brir_conv')
    brir_set_folder=CN.FOLDER_BRIRS_LIVE
    brir_set_name=dpg.get_value('qc_e_apo_sel_brir_set')
    mute_fl_selected=dpg.get_value('e_apo_mute_fl')
    mute_fr_selected=dpg.get_value('e_apo_mute_fr')
    mute_c_selected=dpg.get_value('e_apo_mute_c')
    mute_sl_selected=dpg.get_value('e_apo_mute_sl')
    mute_sr_selected=dpg.get_value('e_apo_mute_sr')
    mute_rl_selected=dpg.get_value('e_apo_mute_rl')
    mute_rr_selected=dpg.get_value('e_apo_mute_rr')
    gain_oa_selected=dpg.get_value('e_apo_gain_oa')
    gain_fl_selected=dpg.get_value('e_apo_gain_fl')
    gain_fr_selected=dpg.get_value('e_apo_gain_fr')
    gain_c_selected=dpg.get_value('e_apo_gain_c')
    gain_sl_selected=dpg.get_value('e_apo_gain_sl')
    gain_sr_selected=dpg.get_value('e_apo_gain_sr')
    gain_rl_selected=dpg.get_value('e_apo_gain_rl')
    gain_rr_selected=dpg.get_value('e_apo_gain_rr')
    elev_fl_selected=dpg.get_value('e_apo_elev_angle_fl')
    elev_fr_selected=dpg.get_value('e_apo_elev_angle_fr')
    elev_c_selected=dpg.get_value('e_apo_elev_angle_c')
    elev_sl_selected=dpg.get_value('e_apo_elev_angle_sl')
    elev_sr_selected=dpg.get_value('e_apo_elev_angle_sr')
    elev_rl_selected=dpg.get_value('e_apo_elev_angle_rl')
    elev_rr_selected=dpg.get_value('e_apo_elev_angle_rr')
    azim_fl_selected=dpg.get_value('e_apo_az_angle_fl')
    azim_fr_selected=dpg.get_value('e_apo_az_angle_fr')
    azim_c_selected=dpg.get_value('e_apo_az_angle_c')
    azim_sl_selected=dpg.get_value('e_apo_az_angle_sl')
    azim_sr_selected=dpg.get_value('e_apo_az_angle_sr')
    azim_rl_selected=dpg.get_value('e_apo_az_angle_rl')
    azim_rr_selected=dpg.get_value('e_apo_az_angle_rr')
    
    #hesuvi brir related selections
    h_elev_fl_selected=dpg.get_value('hesuvi_elev_angle_fl')
    h_elev_fr_selected=dpg.get_value('hesuvi_elev_angle_fr')
    h_elev_c_selected=dpg.get_value('hesuvi_elev_angle_c')
    h_elev_sl_selected=dpg.get_value('hesuvi_elev_angle_sl')
    h_elev_sr_selected=dpg.get_value('hesuvi_elev_angle_sr')
    h_elev_rl_selected=dpg.get_value('hesuvi_elev_angle_rl')
    h_elev_rr_selected=dpg.get_value('hesuvi_elev_angle_rr')
    h_azim_fl_selected=dpg.get_value('hesuvi_az_angle_fl')
    h_azim_fr_selected=dpg.get_value('hesuvi_az_angle_fr')
    h_azim_c_selected=dpg.get_value('hesuvi_az_angle_c')
    h_azim_sl_selected=dpg.get_value('hesuvi_az_angle_sl')
    h_azim_sr_selected=dpg.get_value('hesuvi_az_angle_sr')
    h_azim_rl_selected=dpg.get_value('hesuvi_az_angle_rl')
    h_azim_rr_selected=dpg.get_value('hesuvi_az_angle_rr')
    
    #hrtf type, dataset, hrtf name
    fde_brir_hrtf_type_selected=dpg.get_value('fde_brir_hrtf_type')
    #print(brir_hrtf_type_selected)
    fde_brir_hrtf_dataset_selected=dpg.get_value('fde_brir_hrtf_dataset')
    fde_brir_hrtf_selected=dpg.get_value('fde_brir_hrtf')
    fde_brir_hrtf_short = hrir_processing.get_name_short(listener_type=fde_brir_hrtf_type_selected, dataset_name=fde_brir_hrtf_dataset_selected, name_gui=fde_brir_hrtf_selected)
    qc_brir_hrtf_type_selected=dpg.get_value('qc_brir_hrtf_type')
    qc_brir_hrtf_dataset_selected=dpg.get_value('qc_brir_hrtf_dataset')
    qc_brir_hrtf_selected=dpg.get_value('qc_brir_hrtf')
    qc_brir_hrtf_short = hrir_processing.get_name_short(listener_type=qc_brir_hrtf_type_selected, dataset_name=qc_brir_hrtf_dataset_selected, name_gui=qc_brir_hrtf_selected)
    
    
    elev_list = [
         int(elev_fl_selected),
         int(elev_fr_selected),
         int(elev_c_selected),
         int(elev_sl_selected),
         int(elev_sr_selected),
         int(elev_rl_selected),
         int(elev_rr_selected),
     ]
    azim_list = [
         int(azim_fl_selected),
         int(azim_fr_selected),
         int(azim_c_selected),
         int(azim_sl_selected),
         int(azim_sr_selected),
         int(azim_rl_selected),
         int(azim_rr_selected),
     ]
    
    #QC params
    # qc_target = dpg.get_value("qc_room_target")
    # qc_room_target_int = CN.ROOM_TARGET_LIST.index(qc_target)
    # qc_room_target = qc_room_target_int
    qc_target = dpg.get_value("qc_room_target")
    qc_room_target_int = CN.ROOM_TARGET_INDEX_MAP.get(qc_target, -1)  # -1 or suitable default/error handling
    qc_room_target = qc_target  # for BRIR params
    
    qc_direct_gain_db = dpg.get_value("qc_direct_gain")
    qc_direct_gain_db = round(qc_direct_gain_db,1)#round to nearest .1 dB
    qc_ac_space = dpg.get_value("qc_acoustic_space")
    qc_ac_space_int = CN.AC_SPACE_LIST_GUI.index(qc_ac_space)
    qc_ac_space_short = CN.AC_SPACE_LIST_SHORT[qc_ac_space_int]
    qc_ac_space_src = CN.AC_SPACE_LIST_SRC[qc_ac_space_int]
    qc_hp_type = dpg.get_value("qc_brir_hp_type")
    qc_pinna_comp_int = CN.HP_COMP_LIST.index(qc_hp_type)
    qc_pinna_comp = qc_pinna_comp_int
    qc_samp_freq_str = dpg.get_value('qc_wav_sample_rate')
    qc_samp_freq_int = CN.SAMPLE_RATE_DICT.get(qc_samp_freq_str)
    qc_bit_depth_str = dpg.get_value('qc_wav_bit_depth')
    qc_bit_depth = CN.BIT_DEPTH_DICT.get(qc_bit_depth_str)
    
    #brir params
    # target = dpg.get_value("fde_room_target")
    # room_target_int = CN.ROOM_TARGET_LIST.index(target)
    # fde_room_target = room_target_int
    fde_target = dpg.get_value("fde_room_target")
    fde_room_target_int = CN.ROOM_TARGET_INDEX_MAP.get(fde_target, -1)
    fde_room_target = fde_target
    
    fde_direct_gain_db = dpg.get_value("fde_direct_gain")
    fde_direct_gain_db = round(fde_direct_gain_db,1)#round to nearest .1 dB
    fde_ac_space = dpg.get_value("fde_acoustic_space")
    fde_ac_space_int = CN.AC_SPACE_LIST_GUI.index(fde_ac_space)
    fde_ac_space_short = CN.AC_SPACE_LIST_SHORT[fde_ac_space_int]
    fde_ac_space_src = CN.AC_SPACE_LIST_SRC[fde_ac_space_int]
    fde_hp_type = dpg.get_value("fde_brir_hp_type")
    fde_pinna_comp_int = CN.HP_COMP_LIST.index(fde_hp_type)
    fde_pinna_comp = fde_pinna_comp_int
    fde_samp_freq_str = dpg.get_value('fde_wav_sample_rate')
    fde_samp_freq_int = CN.SAMPLE_RATE_DICT.get(fde_samp_freq_str)
    fde_bit_depth_str = dpg.get_value('fde_wav_bit_depth')
    fde_bit_depth = CN.BIT_DEPTH_DICT.get(fde_bit_depth_str)
    fde_spat_res = dpg.get_value("fde_brir_spat_res")
    fde_spat_res_int = CN.SPATIAL_RES_LIST.index(fde_spat_res)
    
    #misc
    hrtf_symmetry = dpg.get_value('force_hrtf_symmetry')
    er_delay_time = dpg.get_value('er_delay_time')
    er_delay_time = round(er_delay_time,1)#round to nearest .1 dB
    hrtf_polarity = dpg.get_value('hrtf_polarity_rev')
    
    #low freq
    qc_crossover_f_mode = dpg.get_value('qc_crossover_f_mode')
    qc_crossover_f = dpg.get_value('qc_crossover_f')
    qc_sub_response = dpg.get_value('qc_sub_response')
    qc_sub_response_int = CN.SUB_RESPONSE_LIST_GUI.index(qc_sub_response)
    qc_sub_response_short = CN.SUB_RESPONSE_LIST_SHORT[qc_sub_response_int]
    qc_hp_rolloff_comp = dpg.get_value('qc_hp_rolloff_comp')
    qc_fb_filtering = dpg.get_value('qc_fb_filtering')
    fde_crossover_f_mode = dpg.get_value('fde_crossover_f_mode')
    fde_crossover_f = dpg.get_value('fde_crossover_f')
    fde_sub_response = dpg.get_value('fde_sub_response')
    fde_sub_response_int = CN.SUB_RESPONSE_LIST_GUI.index(fde_sub_response)
    fde_sub_response_short = CN.SUB_RESPONSE_LIST_SHORT[fde_sub_response_int]
    fde_hp_rolloff_comp = dpg.get_value('fde_hp_rolloff_comp')
    fde_fb_filtering = dpg.get_value('fde_fb_filtering')
    
    

    brir_dict = {
        'enable_conv': enable_brir_selected, 'brir_set_folder': brir_set_folder, 
        'brir_set_name': brir_set_name, 'mute_fl': mute_fl_selected, 'mute_fr': mute_fr_selected, 
        'mute_c': mute_c_selected, 'mute_sl': mute_sl_selected, 'mute_sr': mute_sr_selected, 
        'mute_rl': mute_rl_selected, 'mute_rr': mute_rr_selected, 'gain_oa': gain_oa_selected, 
        'gain_fl': gain_fl_selected, 'gain_fr': gain_fr_selected, 'gain_c': gain_c_selected, 
        'gain_sl': gain_sl_selected, 'gain_sr': gain_sr_selected, 'gain_rl': gain_rl_selected, 
        'gain_rr': gain_rr_selected, 
    
        # Individual Elevation Variables
        'elev_fl': elev_list[0], 'elev_fr': elev_list[1], 'elev_c': elev_list[2], 
        'elev_sl': elev_list[3], 'elev_sr': elev_list[4], 'elev_rl': elev_list[5], 
        'elev_rr': elev_list[6], 
    
        # Individual Azimuth Variables
        'azim_fl': azim_list[0], 'azim_fr': azim_list[1], 'azim_c': azim_list[2], 
        'azim_sl': azim_list[3], 'azim_sr': azim_list[4], 'azim_rl': azim_list[5], 
        'azim_rr': azim_list[6], 
    
        # Elevation and Azimuth Lists
        'elev_list': elev_list, 'azim_list': azim_list, 
    
        # HRTF selections
        'fde_brir_hrtf_type': fde_brir_hrtf_type_selected, 'fde_brir_hrtf_dataset': fde_brir_hrtf_dataset_selected, 
        'fde_brir_hrtf': fde_brir_hrtf_selected, 'fde_brir_hrtf_short': fde_brir_hrtf_short, 
        'qc_brir_hrtf_type': qc_brir_hrtf_type_selected, 'qc_brir_hrtf_dataset': qc_brir_hrtf_dataset_selected, 
        'qc_brir_hrtf': qc_brir_hrtf_selected, 'qc_brir_hrtf_short': qc_brir_hrtf_short, 
    
        # Hesuvi (h_) Elevation Variables
        'h_elev_fl': h_elev_fl_selected, 'h_elev_fr': h_elev_fr_selected, 
        'h_elev_c': h_elev_c_selected, 'h_elev_sl': h_elev_sl_selected, 
        'h_elev_sr': h_elev_sr_selected, 'h_elev_rl': h_elev_rl_selected, 
        'h_elev_rr': h_elev_rr_selected, 
    
        # Hesuvi (h_) Azimuth Variables
        'h_azim_fl': h_azim_fl_selected, 'h_azim_fr': h_azim_fr_selected, 
        'h_azim_c': h_azim_c_selected, 'h_azim_sl': h_azim_sl_selected, 
        'h_azim_sr': h_azim_sr_selected, 'h_azim_rl': h_azim_rl_selected, 
        'h_azim_rr': h_azim_rr_selected, 
    
        # QC room and acoustic space selections
        'qc_room_target': qc_room_target, 'qc_room_target_int': qc_room_target_int, 'qc_direct_gain_db': qc_direct_gain_db, 
        'qc_ac_space_short': qc_ac_space_short, 'qc_ac_space_src': qc_ac_space_src, 
        'qc_pinna_comp': qc_pinna_comp, 'qc_samp_freq_int': qc_samp_freq_int, 'qc_samp_freq_str': qc_samp_freq_str,   
        'qc_bit_depth': qc_bit_depth, 'qc_bit_depth_str': qc_bit_depth_str, 
    
        # General BRIR selections
        'fde_room_target': fde_room_target, 'fde_room_target_int': fde_room_target_int, 'fde_direct_gain_db': fde_direct_gain_db, 
        'fde_ac_space_short': fde_ac_space_short, 'fde_ac_space_src': fde_ac_space_src, 
        'fde_pinna_comp': fde_pinna_comp, 'fde_samp_freq_int': fde_samp_freq_int, 'fde_samp_freq_str': fde_samp_freq_str, 
        'fde_bit_depth': fde_bit_depth, 'fde_bit_depth_str': fde_bit_depth_str, 'fde_spat_res_int': fde_spat_res_int, 
    
        # Additional variables
        'hrtf_symmetry': hrtf_symmetry, 'er_delay_time': er_delay_time,
        
        # Additional variables
        'qc_crossover_f_mode': qc_crossover_f_mode, 'qc_crossover_f': qc_crossover_f, 'qc_sub_response': qc_sub_response, 'qc_sub_response_short': qc_sub_response_short, 'qc_hp_rolloff_comp': qc_hp_rolloff_comp,
        'qc_fb_filtering': qc_fb_filtering, 'fde_crossover_f_mode': fde_crossover_f_mode, 'fde_crossover_f': fde_crossover_f, 'fde_sub_response': fde_sub_response, 'fde_sub_response_short': fde_sub_response_short, 
        'fde_hp_rolloff_comp': fde_hp_rolloff_comp, 'fde_fb_filtering': fde_fb_filtering, 'hrtf_polarity': hrtf_polarity
    }



    return brir_dict
    


def e_apo_config_acquire(estimate_gain=True):
    """ 
    GUI function to acquire lock on function to write updates to custom E-APO config
    """
    e_apo_conf_lock=dpg.get_item_user_data("e_apo_hpcf_conv")#contains lock
    e_apo_conf_lock.acquire()
    e_apo_config_write(estimate_gain=estimate_gain)
    e_apo_conf_lock.release()
    
def e_apo_config_acquire_gui(sender=None, app_data=None):
    """ 
    GUI function to acquire lock on function to write updates to custom E-APO config
    """
    e_apo_conf_lock=dpg.get_item_user_data("e_apo_hpcf_conv")#contains lock
    estimate_gain=True
    e_apo_conf_lock.acquire()
    e_apo_config_write(estimate_gain=estimate_gain)
    e_apo_conf_lock.release()
      

def e_apo_config_write(estimate_gain=True):
    """ 
    GUI function to write updates to custom E-APO config
    """
    
    try:
        hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
        conn = hpcf_db_dict['conn']
        logz=dpg.get_item_user_data("console_window")#contains logger
        base_folder_selected=dpg.get_value('qc_selected_folder_base')
        
        #hpcf related selections
        enable_hpcf_selected=dpg.get_value('e_apo_hpcf_conv')
        headphone_selected=dpg.get_value('qc_hpcf_headphone')
        sample_selected=dpg.get_value('qc_hpcf_sample')
        #brand_selected=dpg.get_value('qc_hpcf_brand')
        brand_selected = hpcf_functions.get_brand(conn, headphone_selected)
        
        hpcf_dict = {'enable_conv': enable_hpcf_selected, 'brand': brand_selected, 'headphone': headphone_selected, 'sample': sample_selected}
        
        #brir related selections
        enable_brir_selected=dpg.get_value('e_apo_brir_conv')
        brir_set_folder=CN.FOLDER_BRIRS_LIVE
        gain_oa_selected=dpg.get_value('e_apo_gain_oa')
    
        brir_dict = get_brir_dict()
      
        audio_channels=dpg.get_value('e_apo_audio_channels')
        upmix_method=dpg.get_value('e_apo_upmix_method')
        side_delay=dpg.get_value('e_apo_side_delay')
        rear_delay=dpg.get_value('e_apo_rear_delay')
        
        #get spatial resolution for this brir set
        spatial_res_sel = 0
        
        #run function to write custom config
        gain_conf = e_apo_config_creation.write_ash_e_apo_config(primary_path=base_folder_selected, hpcf_dict=hpcf_dict, brir_dict=brir_dict, audio_channels=audio_channels, gui_logger=logz, spatial_res=spatial_res_sel, upmix_method=upmix_method, side_delay=side_delay, rear_delay=rear_delay)
     
        #run function to load the custom config file in config.txt
        gain_oa_selected=dpg.get_value('e_apo_gain_oa')
        if enable_hpcf_selected == True or enable_brir_selected == True or gain_oa_selected == CN.EAPO_MUTE_GAIN:
            load_config = True
        else:
            load_config = False
            
        prevent_clipping = dpg.get_value('e_apo_prevent_clip')
      
        #also update estimated peak gain
        if estimate_gain == True:
            #average gain
            est_pk_gain_2 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '2.0 Stereo', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder, calc_mode=1))
            dpg.set_value("e_apo_gain_avg_2_0", str(est_pk_gain_2))
            est_pk_gain_5 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '5.1 Surround', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder, calc_mode=1))
            dpg.set_value("e_apo_gain_avg_5_1", str(est_pk_gain_5))
            est_pk_gain_7 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '7.1 Surround', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder, calc_mode=1))
            dpg.set_value("e_apo_gain_avg_7_1", str(est_pk_gain_7))
            #peak gain
            est_pk_gain_2 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '2.0 Stereo', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder))
            dpg.set_value("e_apo_gain_peak_2_0", str(est_pk_gain_2))
            est_pk_gain_5 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '5.1 Surround', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder))
            dpg.set_value("e_apo_gain_peak_5_1", str(est_pk_gain_5))
            est_pk_gain_7 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '7.1 Surround', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder))
            dpg.set_value("e_apo_gain_peak_7_1", str(est_pk_gain_7))
            
            #clipping prevention
            one_chan_mute=False
            mute_fl=brir_dict.get('mute_fl')
            mute_fr=brir_dict.get('mute_fr')
            if (mute_fl == True or mute_fr == True):#if at least one channel is muted, dont adjust gain
                one_chan_mute=True
            #if clipping prevention enabled, grab 2.0 peak gain, calc new gain and rewrite the custom config with gain override
            if prevent_clipping != CN.AUTO_GAIN_METHODS[0] and load_config == True and one_chan_mute == False:
                constant_reduction=0.0
                if prevent_clipping == 'Align Low Frequencies':
                    #peak gain - low frequencies
                    est_pk_gain_reference = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '2.0 Stereo', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder,freq_mode = 'Align Low Frequencies'))
                elif prevent_clipping == 'Align Mid Frequencies':
                    est_pk_gain_reference = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '2.0 Stereo', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder,freq_mode = 'Align Mid Frequencies'))
                else:
                    est_pk_gain_reference = est_pk_gain_2
                gain_adjustment = float(est_pk_gain_reference*-1)-constant_reduction#change polarity and reduce slightly
                gain_adjustment = min(gain_adjustment, 20.0)#limit to max of 20db
                dpg.set_value("e_apo_gain_oa", gain_oa_selected+gain_adjustment)
                #run function to write custom config
                gain_conf = e_apo_config_creation.write_ash_e_apo_config(primary_path=base_folder_selected, hpcf_dict=hpcf_dict, brir_dict=brir_dict, audio_channels=audio_channels, gui_logger=logz, spatial_res=spatial_res_sel, gain_adjustment=gain_adjustment, upmix_method=upmix_method, side_delay=side_delay, rear_delay=rear_delay)
                #run function to load the custom config file in config.txt
                #if true, edit config.txt to include the custom config
                e_apo_config_creation.include_ash_e_apo_config(primary_path=base_folder_selected, enabled=load_config)
                #average gain
                est_pk_gain_2 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '2.0 Stereo', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder, calc_mode=1))
                dpg.set_value("e_apo_gain_avg_2_0", str(est_pk_gain_2))
                est_pk_gain_5 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '5.1 Surround', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder, calc_mode=1))
                dpg.set_value("e_apo_gain_avg_5_1", str(est_pk_gain_5))
                est_pk_gain_7 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '7.1 Surround', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder, calc_mode=1))
                dpg.set_value("e_apo_gain_avg_7_1", str(est_pk_gain_7))
                #peak gain
                est_pk_gain_2 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '2.0 Stereo', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder))
                dpg.set_value("e_apo_gain_peak_2_0", str(est_pk_gain_2))
                est_pk_gain_5 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '5.1 Surround', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder))
                dpg.set_value("e_apo_gain_peak_5_1", str(est_pk_gain_5))
                est_pk_gain_7 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '7.1 Surround', hpcf_dict=hpcf_dict, brir_dict=brir_dict, brir_set=brir_set_folder))
                dpg.set_value("e_apo_gain_peak_7_1", str(est_pk_gain_7))
            else:
                e_apo_config_creation.include_ash_e_apo_config(primary_path=base_folder_selected, enabled=load_config)
        else:
            #edit config.txt to include the custom config - only run once
            e_apo_config_creation.include_ash_e_apo_config(primary_path=base_folder_selected, enabled=load_config)
            
        #also save settings
        save_settings()  
        
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)
        

def e_apo_adjust_preamp(sender=None, app_data=None):
    """ 
    GUI function to process updates in E-APO config section
    """
    #turn off auto adjust preamp option
    dpg.set_value("e_apo_prevent_clip", CN.AUTO_GAIN_METHODS[0])
    e_apo_config_acquire()




def e_apo_update_direction(sender=None, app_data=None, aquire_config=False, brir_dict_new={}):
    """ 
    GUI function to process updates to directions in E-APO config section
    """
    
    try:
        if brir_dict_new:
            #if brir dict is provided, grab values
            elev_fl=brir_dict_new.get('elev_fl')
            elev_fr=brir_dict_new.get('elev_fr')
            elev_c=brir_dict_new.get('elev_c')
            elev_sl=brir_dict_new.get('elev_sl')
            elev_sr=brir_dict_new.get('elev_sr')
            elev_rl=brir_dict_new.get('elev_rl')
            elev_rr=brir_dict_new.get('elev_rr')
            azim_fl=brir_dict_new.get('azim_fl')
            azim_fr=brir_dict_new.get('azim_fr')
            azim_c=brir_dict_new.get('azim_c')
            azim_sl=brir_dict_new.get('azim_sl')
            azim_sr=brir_dict_new.get('azim_sr')
            azim_rl=brir_dict_new.get('azim_rl')
            azim_rr=brir_dict_new.get('azim_rr')
            #then update gui values to align
            dpg.set_value("e_apo_elev_angle_fl", elev_fl)
            dpg.set_value("e_apo_elev_angle_fr", elev_fr)
            dpg.set_value("e_apo_elev_angle_c", elev_c)
            dpg.set_value("e_apo_elev_angle_sl", elev_sl)
            dpg.set_value("e_apo_elev_angle_sr", elev_sr)
            dpg.set_value("e_apo_elev_angle_rl", elev_rl)
            dpg.set_value("e_apo_elev_angle_rr", elev_rr)
            dpg.set_value("e_apo_az_angle_fl", azim_fl)
            dpg.set_value("e_apo_az_angle_fr", azim_fr)
            dpg.set_value("e_apo_az_angle_c", azim_c)
            dpg.set_value("e_apo_az_angle_sl", azim_sl)
            dpg.set_value("e_apo_az_angle_sr", azim_sr)
            dpg.set_value("e_apo_az_angle_rl", azim_rl)
            dpg.set_value("e_apo_az_angle_rr", azim_rr)
        else:
            #else grab azimuths from gui elements and proceed. No update required to gui values
            brir_dict_new=get_brir_dict()
            azim_fl=brir_dict_new.get('azim_fl')
            azim_fr=brir_dict_new.get('azim_fr')
            azim_c=brir_dict_new.get('azim_c')
            azim_sl=brir_dict_new.get('azim_sl')
            azim_sr=brir_dict_new.get('azim_sr')
            azim_rl=brir_dict_new.get('azim_rl')
            azim_rr=brir_dict_new.get('azim_rr')
        
        #update gui elements and write new config 
      
        #update each azimuth based drawing
        azimuth=int(azim_fl)
        dpg.apply_transform("fl_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("fl_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_fr)
        dpg.apply_transform("fr_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("fr_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_c)
        dpg.apply_transform("c_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("c_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_sl)
        dpg.apply_transform("sl_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("sl_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_sr)
        dpg.apply_transform("sr_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("sr_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_rl)
        dpg.apply_transform("rl_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("rl_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_rr)
        dpg.apply_transform("rr_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("rr_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
            
        #finally rewrite config file
        if aquire_config == True or aquire_config == None:#custom parameter will be none if called by gui
            e_apo_config_acquire()
            
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)

def e_apo_activate_direction_gui(sender=None, app_data=None):
    """ 
    GUI function to process updates to directions in E-APO config section
    """
    aquire_config=True 
    force_reset=False
    e_apo_activate_direction(aquire_config=aquire_config, force_reset=force_reset)

def e_apo_activate_direction(aquire_config=False, force_reset=False):
    """ 
    GUI function to process updates to directions in E-APO config section
    """
    
    try:    
        __version__=dpg.get_item_user_data("log_text")#contains version
        logz=dpg.get_item_user_data("console_window")#contains logger
        
        #get current selections
        base_folder_selected=dpg.get_value('qc_selected_folder_base')
        channels_selected=dpg.get_value('e_apo_audio_channels')
        brir_set_folder=CN.FOLDER_BRIRS_LIVE
        brir_dict=get_brir_dict()
        #print(brir_dict)
        
        # Explicitly clear any previous BRIRs stored in GUI to avoid leaks
        dpg.configure_item('qc_e_apo_sel_brir_set', user_data=None)
        #Get brir_dict for desired directions, store in a gui element for later
        dpg.configure_item('qc_e_apo_sel_brir_set',user_data=brir_dict)
        
        #run function to check if all brirs currently exist (returns true if brirs are disabled)
        all_brirs_found = e_apo_config_creation.dataset_all_brirs_found(primary_path=base_folder_selected, brir_set=brir_set_folder, brir_dict=brir_dict, channel_config = channels_selected)
        
        #if some files are missing (due to reduced dataset size)
        if all_brirs_found == False:
            #this means we cannot use current selections
            try:
                #load previous settings
                config = configparser.ConfigParser()
                config.read(CN.SETTINGS_FILE)
                version_loaded = config['DEFAULT']['version']
                e_apo_elev_angle_fl_loaded=int(config['DEFAULT']['elev_fl'])
                e_apo_elev_angle_fr_loaded=int(config['DEFAULT']['elev_fr'])
                e_apo_elev_angle_c_loaded=int(config['DEFAULT']['elev_c'])
                e_apo_elev_angle_sl_loaded=int(config['DEFAULT']['elev_sl'])
                e_apo_elev_angle_sr_loaded=int(config['DEFAULT']['elev_sr'])
                e_apo_elev_angle_rl_loaded=int(config['DEFAULT']['elev_rl'])
                e_apo_elev_angle_rr_loaded=int(config['DEFAULT']['elev_rr'])
                e_apo_az_angle_fl_loaded=int(config['DEFAULT']['azim_fl'])
                e_apo_az_angle_fr_loaded=int(config['DEFAULT']['azim_fr'])  
                e_apo_az_angle_c_loaded=int(config['DEFAULT']['azim_c'] )
                e_apo_az_angle_sl_loaded=int(config['DEFAULT']['azim_sl'])
                e_apo_az_angle_sr_loaded=int(config['DEFAULT']['azim_sr'] )
                e_apo_az_angle_rl_loaded=int(config['DEFAULT']['azim_rl'])
                e_apo_az_angle_rr_loaded=int(config['DEFAULT']['azim_rr'])
                    
                #update directions to previously saved values
                dpg.set_value("e_apo_elev_angle_fl", e_apo_elev_angle_fl_loaded)
                dpg.set_value("e_apo_elev_angle_fr", e_apo_elev_angle_fr_loaded)
                dpg.set_value("e_apo_elev_angle_c", e_apo_elev_angle_c_loaded)
                dpg.set_value("e_apo_elev_angle_sl", e_apo_elev_angle_sl_loaded)
                dpg.set_value("e_apo_elev_angle_sr", e_apo_elev_angle_sr_loaded)
                dpg.set_value("e_apo_elev_angle_rl", e_apo_elev_angle_rl_loaded)
                dpg.set_value("e_apo_elev_angle_rr", e_apo_elev_angle_rr_loaded)
                dpg.set_value("e_apo_az_angle_fl", e_apo_az_angle_fl_loaded)
                dpg.set_value("e_apo_az_angle_fr", e_apo_az_angle_fr_loaded)
                dpg.set_value("e_apo_az_angle_c", e_apo_az_angle_c_loaded)
                dpg.set_value("e_apo_az_angle_sl", e_apo_az_angle_sl_loaded)
                dpg.set_value("e_apo_az_angle_sr", e_apo_az_angle_sr_loaded)
                dpg.set_value("e_apo_az_angle_rl", e_apo_az_angle_rl_loaded)
                dpg.set_value("e_apo_az_angle_rr", e_apo_az_angle_rr_loaded)
                
            except:
                pass
            
            log_string = 'Selected direction was not found'
            hf.log_with_timestamp(log_string, logz)
            
            #is there already a brir dataset in memory?
            brir_dict_list=dpg.get_item_user_data("e_apo_brir_conv")
            
            #reset progress and disable brir conv as not started yet
            if force_reset == True and not brir_dict_list:#only when triggered by apply button or toggle and no stored brir data
                dpg.set_value("qc_e_apo_curr_brir_set", '')
                dpg.set_value("qc_progress_bar_brir", 0)
                dpg.configure_item("qc_progress_bar_brir", overlay = CN.PROGRESS_START)
                dpg.set_value("e_apo_brir_conv", False)
            
            #If wav files not found, check if dict list is not empty, 
            if brir_dict_list:#list is not empty
                #if list populated, export new wavs from dict list for matching directions 
                log_string = 'Exporting missing direction(s)'
                hf.log_with_timestamp(log_string, logz)
                #attempt to export wavs
                e_apo_toggle_brir_custom(activate=True, use_stored_brirs=True, force_run_process=True)#do not use button due to cancellation logic
            else:
                #list is not populated, 
                #Update user data to flag that stored BRIR dict should be used for azimuths
                dpg.configure_item('qc_e_apo_curr_brir_set',user_data=True)
                log_string = 'Processing and exporting missing direction(s)'
                hf.log_with_timestamp(log_string, logz)
                #If list not populated, trigger new brir processing (likely restarted app)
                e_apo_toggle_brir_custom(activate=True, use_stored_brirs=False, force_run_process=True)#do not use button due to cancellation logic
                
      
        #use updated azimuths
        brir_dict_new=get_brir_dict()
        azim_fl_selected=brir_dict_new.get('azim_fl')
        azim_fr_selected=brir_dict_new.get('azim_fr')
        azim_c_selected=brir_dict_new.get('azim_c')
        azim_sl_selected=brir_dict_new.get('azim_sl')
        azim_sr_selected=brir_dict_new.get('azim_sr')
        azim_rl_selected=brir_dict_new.get('azim_rl')
        azim_rr_selected=brir_dict_new.get('azim_rr')
        
        #update gui elements and write new config 
      
        #update each azimuth based drawing
        azimuth=int(azim_fl_selected)
        dpg.apply_transform("fl_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("fl_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_fr_selected)
        dpg.apply_transform("fr_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("fr_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_c_selected)
        dpg.apply_transform("c_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("c_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_sl_selected)
        dpg.apply_transform("sl_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("sl_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_sr_selected)
        dpg.apply_transform("sr_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("sr_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_rl_selected)
        dpg.apply_transform("rl_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("rl_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
        azimuth=int(azim_rr_selected)
        dpg.apply_transform("rr_drawing", dpg.create_rotation_matrix(math.pi*(90.0+(azimuth*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
        dpg.apply_transform("rr_drawing_inner", dpg.create_rotation_matrix(math.pi*(90.0+180-(azimuth*-1))/180.0 , [0, 0, -1]))
            
        #finally rewrite config file
        if aquire_config == True:#custom parameter will be none if called by gui
            e_apo_config_acquire()
            
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)



def mute_all_chans():
    """ 
    GUI function to mute channel config in E-APO config section
    """
    dpg.set_value("e_apo_mute_fl", True)
    dpg.set_value("e_apo_mute_fr", True)
    dpg.set_value("e_apo_mute_c", True)
    dpg.set_value("e_apo_mute_sl", True)
    dpg.set_value("e_apo_mute_sr", True)
    dpg.set_value("e_apo_mute_rl", True)
    dpg.set_value("e_apo_mute_rr", True)

  
def e_apo_select_channels_gui(sender=None, app_data=None):
    """ 
    GUI function to process updates in E-APO config section
    """
    aquire_config=True
    e_apo_select_channels(app_data=app_data, aquire_config=aquire_config)  
  
def e_apo_select_channels(app_data=None, aquire_config=True):
    """ 
    GUI function to process updates in E-APO config section
    """

    try:
        if app_data == '2.0 Stereo Upmix to 7.1':
            dpg.configure_item("upmixing_table", show=True)
        else:
            dpg.configure_item("upmixing_table", show=False)
            
        if app_data == '2.0 Stereo':  
            dpg.configure_item("e_apo_mute_c", show=False)
            dpg.configure_item("e_apo_mute_sl", show=False)
            dpg.configure_item("e_apo_mute_sr", show=False)
            dpg.configure_item("e_apo_mute_rl", show=False)
            dpg.configure_item("e_apo_mute_rr", show=False) 
            dpg.configure_item("e_apo_gain_c", show=False)
            dpg.configure_item("e_apo_gain_sl", show=False)
            dpg.configure_item("e_apo_gain_sr", show=False)
            dpg.configure_item("e_apo_gain_rl", show=False)
            dpg.configure_item("e_apo_gain_rr", show=False)
            dpg.configure_item("e_apo_elev_angle_c", show=False)
            dpg.configure_item("e_apo_elev_angle_sl", show=False)
            dpg.configure_item("e_apo_elev_angle_sr", show=False)
            dpg.configure_item("e_apo_elev_angle_rl", show=False)
            dpg.configure_item("e_apo_elev_angle_rr", show=False)
            dpg.configure_item("e_apo_az_angle_c", show=False)
            dpg.configure_item("e_apo_az_angle_sl", show=False)
            dpg.configure_item("e_apo_az_angle_sr", show=False)
            dpg.configure_item("e_apo_az_angle_rl", show=False)
            dpg.configure_item("e_apo_az_angle_rr", show=False)
            dpg.configure_item("c_drawing", show=False)
            dpg.configure_item("c_drawing_inner", show=False)
            dpg.configure_item("sl_drawing", show=False)
            dpg.configure_item("sl_drawing_inner", show=False)
            dpg.configure_item("sr_drawing", show=False)
            dpg.configure_item("sr_drawing_inner", show=False)
            dpg.configure_item("rl_drawing", show=False)
            dpg.configure_item("rl_drawing_inner", show=False)
            dpg.configure_item("rr_drawing", show=False)
            dpg.configure_item("rr_drawing_inner", show=False)
            dpg.configure_item("e_apo_gain_peak_5_1", show=False)
            dpg.configure_item("e_apo_gain_peak_7_1", show=False)
            dpg.configure_item("e_apo_gain_avg_5_1", show=False)
            dpg.configure_item("e_apo_gain_avg_7_1", show=False)

        elif app_data == '5.1 Surround':  
            dpg.configure_item("e_apo_mute_c", show=True)
            dpg.configure_item("e_apo_mute_sl", show=False)
            dpg.configure_item("e_apo_mute_sr", show=False)
            dpg.configure_item("e_apo_mute_rl", show=True)
            dpg.configure_item("e_apo_mute_rr", show=True)
            dpg.configure_item("e_apo_gain_c", show=True)
            dpg.configure_item("e_apo_gain_sl", show=False)
            dpg.configure_item("e_apo_gain_sr", show=False)
            dpg.configure_item("e_apo_gain_rl", show=True)
            dpg.configure_item("e_apo_gain_rr", show=True)
            dpg.configure_item("e_apo_elev_angle_c", show=True)
            dpg.configure_item("e_apo_elev_angle_sl", show=False)
            dpg.configure_item("e_apo_elev_angle_sr", show=False)
            dpg.configure_item("e_apo_elev_angle_rl", show=True)
            dpg.configure_item("e_apo_elev_angle_rr", show=True)
            dpg.configure_item("e_apo_az_angle_c", show=True)
            dpg.configure_item("e_apo_az_angle_sl", show=False)
            dpg.configure_item("e_apo_az_angle_sr", show=False)
            dpg.configure_item("e_apo_az_angle_rl", show=True)
            dpg.configure_item("e_apo_az_angle_rr", show=True)
            dpg.configure_item("c_drawing", show=True)
            dpg.configure_item("c_drawing_inner", show=True)
            dpg.configure_item("sl_drawing", show=False)
            dpg.configure_item("sl_drawing_inner", show=False)
            dpg.configure_item("sr_drawing", show=False)
            dpg.configure_item("sr_drawing_inner", show=False)
            dpg.configure_item("rl_drawing", show=True)
            dpg.configure_item("rl_drawing_inner", show=True)
            dpg.configure_item("rr_drawing", show=True)
            dpg.configure_item("rr_drawing_inner", show=True)
            dpg.configure_item("e_apo_gain_peak_5_1", show=True)
            dpg.configure_item("e_apo_gain_peak_7_1", show=False)
            dpg.configure_item("e_apo_gain_avg_5_1", show=True)
            dpg.configure_item("e_apo_gain_avg_7_1", show=False)
        
        elif app_data == '7.1 Surround' or app_data == '2.0 Stereo Upmix to 7.1':
            dpg.configure_item("e_apo_mute_c", show=True)
            dpg.configure_item("e_apo_mute_sl", show=True)
            dpg.configure_item("e_apo_mute_sr", show=True)
            dpg.configure_item("e_apo_mute_rl", show=True)
            dpg.configure_item("e_apo_mute_rr", show=True)
            dpg.configure_item("e_apo_gain_c", show=True)
            dpg.configure_item("e_apo_gain_sl", show=True)
            dpg.configure_item("e_apo_gain_sr", show=True)
            dpg.configure_item("e_apo_gain_rl", show=True)
            dpg.configure_item("e_apo_gain_rr", show=True)
            dpg.configure_item("e_apo_elev_angle_c", show=True)
            dpg.configure_item("e_apo_elev_angle_sl", show=True)
            dpg.configure_item("e_apo_elev_angle_sr", show=True)
            dpg.configure_item("e_apo_elev_angle_rl", show=True)
            dpg.configure_item("e_apo_elev_angle_rr", show=True)
            dpg.configure_item("e_apo_az_angle_c", show=True)
            dpg.configure_item("e_apo_az_angle_sl", show=True)
            dpg.configure_item("e_apo_az_angle_sr", show=True)
            dpg.configure_item("e_apo_az_angle_rl", show=True)
            dpg.configure_item("e_apo_az_angle_rr", show=True) 
            dpg.configure_item("c_drawing", show=True)
            dpg.configure_item("c_drawing_inner", show=True)
            dpg.configure_item("sl_drawing", show=True)
            dpg.configure_item("sl_drawing_inner", show=True)
            dpg.configure_item("sr_drawing", show=True)
            dpg.configure_item("sr_drawing_inner", show=True)
            dpg.configure_item("rl_drawing", show=True)
            dpg.configure_item("rl_drawing_inner", show=True)
            dpg.configure_item("rr_drawing", show=True)
            dpg.configure_item("rr_drawing_inner", show=True)
            dpg.configure_item("e_apo_gain_peak_5_1", show=True)
            dpg.configure_item("e_apo_gain_peak_7_1", show=True)
            dpg.configure_item("e_apo_gain_avg_5_1", show=True)
            dpg.configure_item("e_apo_gain_avg_7_1", show=True)
            
        elif app_data == '7.1 Downmix to Stereo':
            dpg.configure_item("e_apo_mute_c", show=True)
            dpg.configure_item("e_apo_mute_sl", show=True)
            dpg.configure_item("e_apo_mute_sr", show=True)
            dpg.configure_item("e_apo_mute_rl", show=True)
            dpg.configure_item("e_apo_mute_rr", show=True)
            dpg.configure_item("e_apo_gain_c", show=True)
            dpg.configure_item("e_apo_gain_sl", show=True)
            dpg.configure_item("e_apo_gain_sr", show=True)
            dpg.configure_item("e_apo_gain_rl", show=True)
            dpg.configure_item("e_apo_gain_rr", show=True)
            dpg.configure_item("e_apo_elev_angle_c", show=False)
            dpg.configure_item("e_apo_elev_angle_sl", show=False)
            dpg.configure_item("e_apo_elev_angle_sr", show=False)
            dpg.configure_item("e_apo_elev_angle_rl", show=False)
            dpg.configure_item("e_apo_elev_angle_rr", show=False)
            dpg.configure_item("e_apo_az_angle_c", show=False)
            dpg.configure_item("e_apo_az_angle_sl", show=False)
            dpg.configure_item("e_apo_az_angle_sr", show=False)
            dpg.configure_item("e_apo_az_angle_rl", show=False)
            dpg.configure_item("e_apo_az_angle_rr", show=False)
            dpg.configure_item("c_drawing", show=False)
            dpg.configure_item("c_drawing_inner", show=False)
            dpg.configure_item("sl_drawing", show=False)
            dpg.configure_item("sl_drawing_inner", show=False)
            dpg.configure_item("sr_drawing", show=False)
            dpg.configure_item("sr_drawing_inner", show=False)
            dpg.configure_item("rl_drawing", show=False)
            dpg.configure_item("rl_drawing_inner", show=False)
            dpg.configure_item("rr_drawing", show=False)
            dpg.configure_item("rr_drawing_inner", show=False)
            dpg.configure_item("e_apo_gain_peak_5_1", show=True)
            dpg.configure_item("e_apo_gain_peak_7_1", show=True)
            dpg.configure_item("e_apo_gain_avg_5_1", show=True)
            dpg.configure_item("e_apo_gain_avg_7_1", show=True)

    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)
        
        
    #finally rewrite config file
    if aquire_config == True:#custom parameter will be none if called by gui
        e_apo_config_acquire()
        

 
def reset_brir_progress():
    """ 
    GUI function to reset progress bar
    """
    #if not already running
    #thread bool
    process_brirs_running=dpg.get_item_user_data("brir_tag")
    if process_brirs_running == False:
        #reset progress bar
        dpg.set_value("progress_bar_brir", 0)
        dpg.configure_item("progress_bar_brir", overlay = CN.PROGRESS_START_ALT)

def reset_hpcf_progress():
    """ 
    GUI function to reset progress bar
    """
    dpg.set_value("progress_bar_hpcf", 0)
    dpg.configure_item("progress_bar_hpcf", overlay = CN.PROGRESS_START_ALT)

def qc_reset_progress():
    """ 
    GUI function to reset progress bar
    """
    
    try:
        #reset brir progress if applicable
        process_brirs_running=dpg.get_item_user_data("qc_brir_tag")
        brir_conv_enabled=dpg.get_value('e_apo_brir_conv')
        #if not already running
        #thread bool
        if process_brirs_running == False:
            #check if saved brir set name is matching with currently selected params
            brir_name = calc_brir_set_name(full_name=True)
            sel_brir_set=dpg.get_value('qc_e_apo_sel_brir_set')
            if brir_name != sel_brir_set:
                #reset progress bar
                dpg.set_value("qc_progress_bar_brir", 0)
                dpg.configure_item("qc_progress_bar_brir", overlay = CN.PROGRESS_START)
            elif brir_name == sel_brir_set and sel_brir_set !='' and brir_conv_enabled==True:
                dpg.set_value("qc_progress_bar_brir", 1)
                dpg.configure_item("qc_progress_bar_brir", overlay = CN.PROGRESS_FIN)
            else:
                #reset progress bar
                dpg.set_value("qc_progress_bar_brir", 0)
                dpg.configure_item("qc_progress_bar_brir", overlay = CN.PROGRESS_START)
    
        #reset hpcf progress if applicable
        filter_name = calc_hpcf_name()
        sel_hpcf = dpg.get_value('qc_e_apo_sel_hpcf')
        hpcf_conv_enabled=dpg.get_value('e_apo_hpcf_conv')
        if filter_name != sel_hpcf:
            #reset progress bar
            dpg.set_value("qc_progress_bar_hpcf", 0)
            dpg.configure_item("qc_progress_bar_hpcf", overlay = CN.PROGRESS_START)
        elif filter_name == sel_hpcf and sel_hpcf !='' and hpcf_conv_enabled==True:
            dpg.set_value("qc_progress_bar_hpcf", 1)
            dpg.configure_item("qc_progress_bar_hpcf", overlay = CN.PROGRESS_FIN)
        else:
            #reset progress bar
            dpg.set_value("qc_progress_bar_hpcf", 0)
            dpg.configure_item("qc_progress_bar_hpcf", overlay = CN.PROGRESS_START)
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)

#
## GUI Functions - Additional DEV tools
#    

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
    
    except Exception as e:
        
        log_string = 'Failed to check app versions'
        hf.log_with_timestamp(log_string=log_string, gui_logger=gui_logger, log_type = 2, exception=e)#log error
            
        return False


def check_app_version(sender, app_data, user_data):
    """ 
    GUI function to check app version
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    check_for_app_update(gui_logger=logz)

def check_db_version(sender, app_data, user_data):
    """ 
    GUI function to check db version
    """
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
    hpcf_functions.check_for_database_update(conn=conn, gui_logger=logz)

def download_latest_db(sender, app_data, user_data):
    """ 
    GUI function to download latest db
    """
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
    hpcf_functions.download_latest_database(conn=conn, gui_logger=logz)
    
def check_as_versions(sender, app_data, user_data):
    """ 
    GUI function to check acoustic space versions
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    air_processing.acoustic_space_updates(download_updates=False, gui_logger=logz)

def download_latest_as_sets(sender, app_data, user_data):
    """ 
    GUI function to download latest acoustic spaces
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    air_processing.acoustic_space_updates(download_updates=True, gui_logger=logz)

def check_hrtf_versions(sender, app_data, user_data):
    """ 
    GUI function to check hrtf versions
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    hrir_processing.hrir_metadata_updates(download_updates=False, gui_logger=logz)

def download_latest_hrtf_sets(sender, app_data, user_data):
    """ 
    GUI function to download latest hrtf metadata
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    hrir_processing.hrir_metadata_updates(download_updates=True, gui_logger=logz)
    
def check_all_updates():
    """ 
    GUI function to check for all updates
    """
    hpcf_db_dict=dpg.get_item_user_data("qc_e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
    check_for_app_update(gui_logger=logz)
    hpcf_functions.check_for_database_update(conn=conn, gui_logger=logz)
    air_processing.acoustic_space_updates(download_updates=False, gui_logger=logz)
    hrir_processing.hrir_metadata_updates(download_updates=False, gui_logger=logz)




#
#
#### Acoustic space processing functions
#
#
 

def get_ir_folders():
    DATA_DIR_IR_USER = CN.DATA_DIR_IRS_USER
    """Returns a list of subdirectories in the IR user folder"""
    if not os.path.isdir(DATA_DIR_IR_USER):
        return []
    return [f for f in os.listdir(DATA_DIR_IR_USER) if os.path.isdir(os.path.join(DATA_DIR_IR_USER, f))]

def update_ir_folder_list():
    folders = get_ir_folders()
    if not folders:
        dpg.set_value("selected_folder_display", "")
        dpg.configure_item("ir_folder_list", items=["<No folders found>"], enabled=False)
    else:
        dpg.configure_item("ir_folder_list", items=folders, enabled=True,default_value=folders[0])
        dpg.set_value("selected_folder_display", folders[0])

def folder_selected_callback(sender, app_data, user_data):
    dpg.set_value("selected_folder_display", app_data)


def launch_processing_thread():
    user_data = dpg.get_item_user_data("start_processing_btn")
    if user_data.get("ir_processing_running", False):
        hf.log_with_timestamp("Processing already in progress. Please wait.")
        return

    user_data["ir_processing_running"] = True  # Set flag to block new launches
    dpg.set_item_user_data("start_processing_btn", user_data)

    def wrapper():
        try:
            start_processing_callback()
        finally:
            user_data["ir_processing_running"] = False  # Reset when done
            dpg.set_item_user_data("start_processing_btn", user_data)

    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()

def start_processing_callback():
    """ 
    GUI function to handle AIR processing and AIR to BRIR conversion.
    """
    logger_obj = dpg.get_item_user_data("import_console_window")
    # Get the cancel flag
    cancel_event = dpg.get_item_user_data("cancel_processing_button")["cancel_event"]
    cancel_event.clear()  # Reset at start
    try:
        hf.update_gui_progress(report_progress=3, progress=0.0) # Start
        selected_folder = dpg.get_value("selected_folder_display")
        name = dpg.get_value("space_name")
        description = dpg.get_value("space_description")
        directions = dpg.get_value("unique_directions")
        noise_reduction_mode = dpg.get_value("noise_reduction_mode")
        pitch_low = dpg.get_value("pitch_range_low")
        pitch_high = dpg.get_value("pitch_range_high")
        long_reverb_mode = dpg.get_value("long_tail_mode")
        pitch_shift_comp = dpg.get_value("pitch_shift_comp")
        alignment_freq = dpg.get_value("alignment_freq")
        rise_time = dpg.get_value("as_rise_time")
        as_subwoofer_mode = dpg.get_value("as_subwoofer_mode")
        __version__ = dpg.get_item_user_data("log_text")  # contains version

        # Input validation
        if not selected_folder:
            hf.log_with_timestamp("No folder selected.", log_type=2, gui_logger=logger_obj)
            return
        if pitch_low > pitch_high:
            hf.log_with_timestamp("Invalid pitch range: lower bound exceeds upper bound.", log_type=2, gui_logger=logger_obj)
            return

        as_name = selected_folder if name == '' else name
        name_formatted = as_name.replace(" ", "_")
        #change naming convention for subwoofer cases
        # if as_subwoofer_mode == True:
        #     name_formatted="(SUB)-"+name_formatted
        file_name = "reverberation_dataset_" + name_formatted
        name_gui = as_name
        name_short = as_name
        name_src = name_formatted
        hf.log_with_timestamp("Started processing...", gui_logger=logger_obj)

        # Step 1: Check or download HRIR dataset
        hf.log_with_timestamp("Step 1: Checking HRIR dataset...")
        try:
            npy_fname = pjoin(CN.DATA_DIR_INT, 'hrir_dataset_comp_max_TU-FABIAN.npy')#pjoin(CN.DATA_DIR_INT, 'hrir_dataset_compensated_max.npy')
            max_data_set_dl_link = "https://drive.google.com/file/d/1Q14yEBTv2JDu92pPloUQXf_SthSApQ8L/view?usp=drive_link"
            status_code = hf.check_and_download_file(npy_fname, max_data_set_dl_link, download=True, gui_logger=logger_obj)
        except:
            npy_fname = pjoin(CN.DATA_DIR_INT, 'hrir_dataset_comp_max_THK-KU-100.npy')#pjoin(CN.DATA_DIR_INT, 'hrir_dataset_compensated_max.npy')
            max_data_set_dl_link = "https://drive.google.com/file/d/1vmpLYlH-BjBoFvziTD29WqxZGaoYsFuF/view?usp=drive_link"
            status_code = hf.check_and_download_file(npy_fname, max_data_set_dl_link, download=True, gui_logger=logger_obj)
        if status_code != 0:
            hf.update_gui_progress(report_progress=3, progress=0.0) # Reset on failure
        if status_code == 1:
            hf.log_with_timestamp("Error: Failed to download or check HRIR datasets.", log_type=2, gui_logger=logger_obj)
            return
        elif status_code == 2:
            hf.log_with_timestamp("Process cancelled by user.", log_type=1, gui_logger=logger_obj)
            return
        # Insert checks at key points:
        if cancel_event.is_set():
            hf.log_with_timestamp("Process cancelled by user.", log_type=1, gui_logger=logger_obj)
            return
        hf.update_gui_progress(report_progress=3, progress=0.05)

        # Step 2: Prepare IR dataset
        hf.log_with_timestamp("Step 2: Preparing IR dataset...", gui_logger=logger_obj)
        desired_measurements = directions
        pitch_range = (pitch_low, pitch_high)
        air_dataset, status_code = air_processing.prepare_air_dataset(
            ir_set=name_src, input_folder=selected_folder, gui_logger=logger_obj,
            wav_export=False, use_user_folder=True, save_npy=False,
            desired_measurements=desired_measurements,
            pitch_range=pitch_range, long_mode=long_reverb_mode,
            cancel_event=cancel_event, report_progress=3, noise_reduction_mode=noise_reduction_mode, f_alignment = alignment_freq, 
            pitch_shift_comp=pitch_shift_comp,subwoofer_mode=as_subwoofer_mode
        )
        #print(str(status_code))
        if status_code != 0:
            hf.update_gui_progress(report_progress=3, progress=0.0) # Reset on failure
        if status_code == 1:
            hf.log_with_timestamp("Error: Failed to prepare IR dataset.", log_type=2, gui_logger=logger_obj)
            return
        elif status_code == 2:
            hf.log_with_timestamp("IR preparation cancelled by user.", log_type=1, gui_logger=logger_obj)
            return
        if air_dataset.size == 0:
            hf.log_with_timestamp("Invalid or empty IR dataset.", log_type=2, gui_logger=logger_obj)
            return
        # Insert checks at key points:
        if cancel_event.is_set():
            hf.log_with_timestamp("Process cancelled by user.", log_type=1, gui_logger=logger_obj)
            return
        hf.update_gui_progress(report_progress=3, progress=0.70)

        # Step 3: Convert IRs to BRIRs
        hf.log_with_timestamp("Step 3: Converting IRs to BRIRs...", gui_logger=logger_obj)
        brir_reverberation, status_code = air_processing.convert_airs_to_brirs(
            ir_set=name_src, ir_group='user',
            air_dataset=air_dataset, gui_logger=logger_obj,
             long_mode=long_reverb_mode, use_user_folder=True,
            cancel_event=cancel_event, report_progress=3, rise_time=rise_time,subwoofer_mode=as_subwoofer_mode
        )
        if status_code != 0:
            hf.update_gui_progress(report_progress=3, progress=0.0) # Reset on failure
        if status_code == 1:
            hf.log_with_timestamp("Error: Failed to convert IRs to BRIRs.", log_type=2, gui_logger=logger_obj)
            return
        elif status_code == 2:
            hf.log_with_timestamp("BRIR conversion cancelled by user.", log_type=1, gui_logger=logger_obj)
            return
        # Insert checks at key points:
        if cancel_event.is_set():
            hf.log_with_timestamp("Process cancelled by user.", log_type=1, gui_logger=logger_obj)
            return
        hf.update_gui_progress(report_progress=3, progress=0.90)

        # Step 4: Write metadata CSV
        hf.log_with_timestamp("Step 4: Writing metadata...", gui_logger=logger_obj)

        # Timestamp for notes
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #notes = f"Created with ASH Toolset (AS Import tool) {__version__} on {timestamp_str}"
        notes = (
            f"Created with ASH Toolset {__version__} on {timestamp_str} | "
            f"low-frequency_mode={as_subwoofer_mode}, "
            f"noise_reduction_mode={noise_reduction_mode}, "
            f"rise_time={rise_time}ms, "
            f"pitch_range=({pitch_low}, {pitch_high}), "
            f"pitch_shift_comp={pitch_shift_comp}, "
            f"alignment_freq={alignment_freq}Hz, "
            f"directions={directions}, "
            f"long_reverb_mode={long_reverb_mode}"
        )
        description_full = notes if not description.strip() else f"{description}, {notes}"
        noise_reduction = "Yes" if noise_reduction_mode else "No"
        low_rt60 = "Yes" if not long_reverb_mode else "No"
        folder="user"

        # Calculate RT60 robustly
        try:
            # Always get the last axis as the sample array
            squeezed = np.squeeze(brir_reverberation)
            if squeezed.ndim == 1:
                sample_reverb = squeezed
            elif squeezed.ndim >= 2:
                sample_reverb = squeezed[0] if squeezed.ndim == 2 else squeezed[0, 0]
            # Ensure we're dealing with a 1D array now
            sample_reverb = np.asarray(sample_reverb).squeeze()
            band_rt60s = hf.compute_band_rt60s(sample_reverb)
            topt_values = [v for v in band_rt60s.values() if not np.isnan(v)]
            if len(topt_values) == 0:
                raise ValueError("Topt values missing or all NaN.")
            topt_mean = np.mean(topt_values)
            topt_ms = topt_mean * 1000  # Convert to milliseconds
            est_rt60 = int(np.ceil(topt_ms / 50.0) * 50)  # Round up to nearest 50
            meas_rt60 = int(round(topt_ms))               # Regular rounding
        except Exception as rt60_ex:
            hf.log_with_timestamp(f"RT60 estimation failed: {rt60_ex}", gui_logger=logger_obj, log_type=1)
            est_rt60 = 600
            meas_rt60 = 600

        rows = [{
            "file_name": file_name,
            "name_gui": name_gui,
            "name_short": name_short,
            "name_src": name_src,
            "est_rt60": est_rt60,
            "meas_rt60": meas_rt60,
            "fade_start": 0,
            "gain":0,
            "low_rt60": low_rt60,
            "folder": folder,
            "version": "1.0.0",
            "f_crossover": 110,
            "order_crossover": 9,
            "noise_reduce": noise_reduction,
            "description": description_full,
            "notes": notes,
            "source_dataset": selected_folder
        }]
        air_processing.write_as_metadata_csv(ir_set=name_src, data_rows=rows, sub_mode=as_subwoofer_mode, gui_logger=logger_obj)
        hf.update_gui_progress(report_progress=3, progress=1.0)

        hf.log_with_timestamp("Processing complete", gui_logger=logger_obj)
        #hf.log_with_timestamp("Note: Restart is required for new acoustic space to be displayed in other tabs", gui_logger=logger_obj, log_type=1)
   
        #also save subwoofer metadata file if subwoofer mode is enabled
        if as_subwoofer_mode == True:
            file_name_sub=file_name
            name_gui_sub=name_gui
            name_short_sub=name_short
            acoustic_space_sub=description
            est_rt60_sub=meas_rt60
            comments_sub=f"Created with ASH Toolset {__version__} on {timestamp_str}"
            frequency_range_sub=""
            tolerance_sub=""
            dataset_sub=""
            folder_sub=folder
            rows = [{
                "file_name": file_name_sub,
                "name_gui": name_gui_sub,
                "name_short": name_short_sub,
                "acoustic_space": acoustic_space_sub,
                "est_rt60": est_rt60_sub,
                "comments": comments_sub,
                "frequency_range": frequency_range_sub,
                "tolerance":tolerance_sub,
                "folder": folder_sub,
                "dataset": dataset_sub
            }]
            air_processing.write_sub_metadata_csv(ir_set=name_src, data_rows=rows, gui_logger=logger_obj)
            
        time.sleep(0.1)
        update_as_table_from_csvs()
        

    except Exception as ex:
        hf.log_with_timestamp(f"Unexpected error during processing: {ex}", log_type=2, gui_logger=logger_obj)

    
    
def cancel_processing_callback(sender, app_data, user_data):
    cancel_event = user_data["cancel_event"]

    # Only proceed if the flag hasn't been set already
    if not cancel_event.is_set():
        cancel_event.set()
        # logger_obj = dpg.get_item_user_data("import_console_window")
        # logger_obj.log_info("Cancellation flag set.")
    

    
 
    
def open_user_int_as_folder(sender, app_data, user_data):
    """
    Opens the specified folder in the OS file explorer and logs the result.

    """
    logger_obj = dpg.get_item_user_data("import_console_window")
    selected_folder = CN.DATA_DIR_AS_USER
    open_folder_in_explorer(selected_folder, gui_logger=logger_obj)
    
def open_user_input_as_folder(sender, app_data, user_data):
    """
    Opens the specified folder in the OS file explorer and logs the result.

    """
    logger_obj = dpg.get_item_user_data("import_console_window")
    selected_folder = CN.DATA_DIR_IRS_USER
    open_folder_in_explorer(selected_folder, gui_logger=logger_obj)
    
def open_folder_in_explorer(folder_path, gui_logger=None):
    """
    Opens the specified folder in the OS file explorer and logs the result.

    :param folder_path: str - Absolute or relative path to the folder to open.
    """

    # # Get the logger object associated with the import console window
    # logger_obj = dpg.get_item_user_data("import_console_window")
    # if use_main_log == True:
    #     logz=dpg.get_item_user_data("console_window")#contains logger
    

    # Check if the provided path is a valid directory
    if not os.path.isdir(folder_path):
        hf.log_with_timestamp(f"Folder does not exist: {folder_path}", gui_logger)
        return

    try:
        # Attempt to open the folder based on the current operating system
        if sys.platform == "win32":
            os.startfile(folder_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", folder_path], check=True)
        else:
            subprocess.run(["xdg-open", folder_path], check=True)

        # Log success
        hf.log_with_timestamp(f" Opened folder: {folder_path}", gui_logger)


    except Exception as e:
        # Log any exceptions encountered during the attempt
        hf.log_with_timestamp(f" Failed to open folder: {e}", gui_logger)

def open_output_folder(sender, app_data, user_data):
    """ 
    GUI function to open output folder in windows explorer
    """
    ash_folder_selected=dpg.get_value('selected_folder_ash')
    base_folder_selected=dpg.get_value('selected_folder_base')
    try:
        os.startfile(ash_folder_selected)
    except Exception: 
        try:
            os.startfile(base_folder_selected)
        except Exception as e: 
            #print(e)
            #print('Failed')
            pass



def delete_selected_callback():
    """
    Deletes all selected rows and associated folders based on user_data of 'selected_ir_rows'.
    """
    logger_obj = dpg.get_item_user_data("import_console_window")
    selected_indices = dpg.get_item_user_data("selected_ir_rows")

    if not selected_indices:
        hf.log_with_timestamp("No rows selected to delete.", logger_obj)
        dpg.configure_item("del_processed_popup", show=False)
        return

    rows = dpg.get_item_children("processed_irs_table", slot=1)

    for selected_idx in sorted(selected_indices, reverse=True):
        if selected_idx < 0 or selected_idx >= len(rows):
            hf.log_with_timestamp(f"Row index {selected_idx} out of range.", logger_obj)
            continue

        row = rows[selected_idx]
        row_children = dpg.get_item_children(row, slot=1)

        if not row_children or len(row_children) < 1:
            hf.log_with_timestamp(f"Row structure unexpected at index {selected_idx}.", logger_obj)
            continue

        selectable = row_children[0]
        dataset = dpg.get_item_user_data(selectable)
        ir_set_folder = pjoin(CN.DATA_DIR_AS_USER, dataset)

        if os.path.exists(ir_set_folder) and os.path.isdir(ir_set_folder):
            try:
                shutil.rmtree(ir_set_folder)
                hf.log_with_timestamp(f"Deleted folder: {ir_set_folder}", logger_obj)
            except Exception as e:
                hf.log_with_timestamp(f"Failed to delete folder {ir_set_folder}: {e}", logger_obj, log_type=2)
        else:
            hf.log_with_timestamp(f"Folder does not exist: {ir_set_folder}", logger_obj)

    # Refresh table and clear selection
    time.sleep(0.1)
    update_as_table_from_csvs()
    dpg.set_item_user_data("selected_ir_rows", [])
    dpg.configure_item("del_processed_popup", show=False)


        
def on_ir_row_selected(sender, app_data, user_data):
    """
    Tracks multiple selected rows by updating user_data on a hidden UI element.
    """
    rows = dpg.get_item_children("processed_irs_table", 1)
    selected = dpg.get_item_user_data("selected_ir_rows")

    for i, row in enumerate(rows):
        row_children = dpg.get_item_children(row, 1)
        if row_children and row_children[0] == sender:
            if app_data:  # Selected
                if i not in selected:
                    selected.append(i)
            else:  # Deselected
                if i in selected:
                    selected.remove(i)
            break

    dpg.set_item_user_data("selected_ir_rows", selected)


def update_as_table_from_csvs(): 
    """
    Search directory and subdirectories for CSV files with a matching key in the name,
    load contents, extract columns, and update the DPG table.
    """
    try:
        logger_obj = dpg.get_item_user_data("import_console_window")
        directory = CN.DATA_DIR_AS_USER
        filename_key = CN.USER_CSV_KEY#include user AS metadata
        alt_filename_key = CN.ASI_USER_CSV_KEY
        exclude_key = CN.SUB_USER_CSV_KEY#ignore sub metadata in as table
        matching_rows = []
    
        hf.log_with_timestamp("Starting search for saved acoustic spaces...", logger_obj)
    
        # Walk through directory
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv") and (filename_key in file or alt_filename_key in file) and exclude_key not in file:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
                            reader = csv.DictReader(csvfile)
                            for row in reader:
                                name = row.get("name_gui", "")
                                rt60 = row.get("est_rt60", "")
                                description = row.get("description", "")
                                name_src = row.get("name_src", "")
                                matching_rows.append((name, rt60, description, name_src))
                        hf.log_with_timestamp(f"Loaded: {file_path}", logger_obj)
                    except Exception as e:
                        hf.log_with_timestamp(f"Failed to load {file_path}: {e}", logger_obj, log_type=2)
    
        # Clear existing rows in the table
        existing_rows = dpg.get_item_children("processed_irs_table", 1)
        if existing_rows:
            for row_id in existing_rows:
                dpg.delete_item(row_id)
    
        # Add new rows with selectable in the first column and dataset stored in user_data
        for idx, (name, rt60, description, name_src) in enumerate(matching_rows):
            with dpg.table_row(parent="processed_irs_table"):
                dpg.add_selectable(
                    label=name,
                    callback=on_ir_row_selected,
                    span_columns=True,
                    user_data=name_src
                )
                dpg.add_text(str(rt60))
                dpg.add_text(description)
    
        # Reset multi-selection tracking
        dpg.set_item_user_data("selected_ir_rows", [])
    
        hf.log_with_timestamp(f"Table updated with {len(matching_rows)} entries.", logger_obj)
        
        #Update listboxes by refreshing AS constants
        CN.refresh_acoustic_space_metadata()
        dpg.configure_item("fde_acoustic_space", items=CN.AC_SPACE_LIST_GUI)
        dpg.configure_item("qc_acoustic_space", items=CN.AC_SPACE_LIST_GUI)
        
        CN.refresh_sub_responses()#also update sub responses in case sub mode was enabled
        dpg.configure_item("fde_sub_response", items=CN.SUB_RESPONSE_LIST_GUI)
        dpg.configure_item("qc_sub_response", items=CN.SUB_RESPONSE_LIST_GUI)
        
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)


#
#
#### Room target processing functions
#
#

def update_room_target_name_display(sender, app_data, user_data):
    dpg.set_value("room_target_name_display", app_data if app_data else "")
    
    

def generate_room_target_callback(sender, app_data, user_data):
    """
    Callback function to generate a room target filter from UI parameters.
    Used both for interactive preview and final target creation.

    Parameters:
    - sender: ID of the UI element triggering the callback.
    - app_data: Data passed by the UI (unused).
    - user_data: Dictionary, must contain 'generation_running' (bool) and optionally 'save_to_file' (bool).
    """
    #logger_obj = dpg.get_item_user_data("rt_console_window")
    logger_obj = None

    if user_data.get("generation_running"):
        hf.log_with_timestamp("Generation already running, skipping...", logger_obj)
        return
    user_data["generation_running"] = True

    try:
        # --- Extract parameters ---
        name = dpg.get_value("target_name").strip()
        ls_freq = dpg.get_value("low_shelf_freq")
        ls_gain = dpg.get_value("low_shelf_gain")
        ls_q = dpg.get_value("low_shelf_q")

        hs_freq = dpg.get_value("high_shelf_freq")
        hs_gain = dpg.get_value("high_shelf_gain")
        hs_q = dpg.get_value("high_shelf_q")

        def safe_gain_str(g: float) -> str:
            return f"{g:.1f}dB"  # "-" retained, "+" omitted

        if not name:
            name = (
                f"LS_{int(ls_freq)}_{safe_gain_str(ls_gain)}_Q{ls_q:.2f}_"
                f"HS_{int(hs_freq)}_{safe_gain_str(hs_gain)}_Q{hs_q:.2f}"
            )

        dpg.set_value("room_target_name_display", name)
        if user_data.get("save_to_file", False):
            hf.log_with_timestamp(f"Generating room target '{name}'...", logger_obj)

        # --- Create combined magnitude response ---
        fs = CN.FS
        n_fft = 65536
        freqs = np.fft.rfftfreq(n_fft, 1 / fs)

        sos_ls = _design_shelf_filter(fs, ls_freq, ls_gain, ls_q, btype='low')
        _, h_ls = signal.sosfreqz(sos_ls, worN=freqs, fs=fs)

        sos_hs = _design_shelf_filter(fs, hs_freq, hs_gain, hs_q, btype='high')
        _, h_hs = signal.sosfreqz(sos_hs, worN=freqs, fs=fs)

        combined_mag = np.abs(h_ls * h_hs)

        # --- Convert to minimum-phase FIR ---
        #fir = hf.mag_to_min_fir(combined_mag, n_fft=n_fft, out_win_size=4096, crop=1)
        fir = hf.build_min_phase_filter(combined_mag, n_fft=n_fft, truncate_len=4096)

        # --- Save only if explicitly requested ---
        if user_data.get("save_to_file", False):
            target_folder = CN.DATA_DIR_RT_USER
            os.makedirs(target_folder, exist_ok=True)
            out_path = os.path.join(target_folder, f"{name}.npy")
            np.save(out_path, fir)
            hf.log_with_timestamp(f"Saved FIR filter to '{out_path}'", logger_obj)

            # Add to preset list
            update_room_target_list(None, None, user_data)
            #update progress bar to 100%
            dpg.set_value("progress_bar_target_gen", 1.0)
        else:
            #not writing to file means no progress considered
            dpg.set_value("progress_bar_target_gen", 0.0)
            
        # --- Update plot ---
        db_mag = 20 * np.log10(np.maximum(combined_mag, 1e-12))  # Avoid log(0)
        dpg.set_value("rt_plot_series", [freqs.tolist(), db_mag.tolist()])

        plot_title =  "Room Target Preview - Unsaved Target"
        dpg.set_value("target_plot_title", plot_title)
        dpg.set_item_label("rt_plot_series", name)
 
    except Exception as e:
        hf.log_with_timestamp(f"Error generating room target: {e}", logger_obj)
        dpg.set_value("progress_bar_target_gen", 0.0)

    finally:
        
        user_data["generation_running"] = False


def _design_shelf_filter(fs, freq, gain_db, Q, btype='low'):
    """
    Designs a biquad shelf filter using RBJ cookbook formula approximation.
    """
    gain = 10 ** (gain_db / 40)  # sqrt version
    omega = 2 * np.pi * freq / fs
    alpha = np.sin(omega) / (2 * Q)
    A = gain

    cos_omega = np.cos(omega)

    if btype == 'low':
        b0 =    A*((A+1) - (A-1)*cos_omega + 2*np.sqrt(A)*alpha)
        b1 =  2*A*((A-1) - (A+1)*cos_omega)
        b2 =    A*((A+1) - (A-1)*cos_omega - 2*np.sqrt(A)*alpha)
        a0 =        (A+1) + (A-1)*cos_omega + 2*np.sqrt(A)*alpha
        a1 =   -2*((A-1) + (A+1)*cos_omega)
        a2 =        (A+1) + (A-1)*cos_omega - 2*np.sqrt(A)*alpha

    elif btype == 'high':
        b0 =    A*((A+1) + (A-1)*cos_omega + 2*np.sqrt(A)*alpha)
        b1 = -2*A*((A-1) + (A+1)*cos_omega)
        b2 =    A*((A+1) + (A-1)*cos_omega - 2*np.sqrt(A)*alpha)
        a0 =        (A+1) - (A-1)*cos_omega + 2*np.sqrt(A)*alpha
        a1 =    2*((A-1) - (A+1)*cos_omega)
        a2 =        (A+1) - (A-1)*cos_omega - 2*np.sqrt(A)*alpha

    b = np.array([b0, b1, b2])
    a = np.array([a0, a1, a2])
    return signal.tf2sos(b / a0, a / a0)

def update_room_target_list(sender=None, app_data=None, user_data=None):
    """
    Callback to refresh the list of saved room target filters in the GUI.
    
    It scans the user room target directory for .npy files and updates the listbox
    to include any newly added targets that aren't already listed.

    Parameters:
    - sender: The UI element that triggered the callback (unused).
    - app_data: Additional data passed by the UI (unused).
    - user_data: Dictionary for shared app state (optional).
    """
    #logger_obj = dpg.get_item_user_data("rt_console_window")
    logger_obj = None

    try:
        target_folder = CN.DATA_DIR_RT_USER
        if not os.path.isdir(target_folder):
            hf.log_with_timestamp("Room target directory does not exist yet.", logger_obj)
            return

        # Get all .npy file names (without extension)
        file_names = [
            os.path.splitext(f)[0]
            for f in os.listdir(target_folder)
            if f.endswith(".npy")
        ]

        file_names.sort()  # Optional: sort alphabetically

        dpg.configure_item("room_target_listbox", items=file_names)
        hf.log_with_timestamp(f"Updated room target list: {len(file_names)} entries found.", logger_obj)
        
        #also update master list in constants and gui listbox
        CN.refresh_room_targets()
        dpg.configure_item("fde_room_target", items=CN.ROOM_TARGET_KEYS)
        dpg.configure_item("qc_room_target", items=CN.ROOM_TARGET_KEYS)

    except Exception as e:
        hf.log_with_timestamp(f"Error refreshing room target list: {e}", logger_obj)

def open_user_rt_folder(sender, app_data, user_data):
    """
    Opens the specified folder in the OS file explorer and logs the result.

    """
    logger_obj = dpg.get_item_user_data("rt_console_window")
    selected_folder = CN.DATA_DIR_RT_USER
    open_folder_in_explorer(selected_folder, gui_logger=logger_obj)
 
def delete_selected_target_callback(sender, app_data, user_data):
    """
    Callback for deleting the selected room target filter from disk.

    This function:
    - Gets the currently selected item from the room target listbox.
    - Deletes the corresponding .npy file from the user data directory.
    - Hides the confirmation popup.
    - Refreshes the room target list in the GUI.
    """
    #logger_obj = dpg.get_item_user_data("rt_console_window")
    logger_obj = None

    try:
        selected = dpg.get_value("room_target_listbox")
        if not selected:
            hf.log_with_timestamp("No target selected for deletion.", logger_obj)
            return

        file_path = os.path.join(CN.DATA_DIR_RT_USER, f"{selected}.npy")
        if os.path.isfile(file_path):
            os.remove(file_path)
            hf.log_with_timestamp(f"Deleted room target: {selected}", logger_obj)
        else:
            hf.log_with_timestamp(f"File not found for deletion: {file_path}", logger_obj)
  
    except Exception as e:
        hf.log_with_timestamp(f"Error deleting room target: {e}", logger_obj)

    finally:
        dpg.configure_item("del_target_popup", show=False)
        update_room_target_list(None, None, user_data)
        # Inside finally block or after deletion
        clear_room_target_plot()
 
def on_room_target_selected(sender, app_data, user_data):
    """
    Callback when a room target is selected from the listbox.
    Loads the FIR file, computes magnitude response, and updates the plot.
    
    Parameters:
    - sender: ID of the listbox
    - app_data: selected item (filename without extension)
    - user_data: dictionary, may contain plot settings like 'plot_title'
    """
    #logger_obj = dpg.get_item_user_data("rt_console_window")
    logger_obj = None

    try:
        selected = app_data  # Selected filename string
        if not selected:
            hf.log_with_timestamp("No room target selected.", logger_obj)
            return

        file_path = os.path.join(CN.DATA_DIR_RT_USER, f"{selected}.npy")
        if not os.path.isfile(file_path):
            hf.log_with_timestamp(f"Room target file not found: {file_path}", logger_obj)
            return

        fir = np.load(file_path)
        hf.log_with_timestamp(f"Loaded room target '{selected}' from {file_path}", logger_obj)

        # Compute frequency response
        fs = CN.FS
        n_fft = 65536
        freqs = np.fft.rfftfreq(n_fft, 1 / fs)
        H = np.fft.rfft(fir, n=n_fft)
        mag = np.abs(H)

        # Convert to dB
        db_mag = 20 * np.log10(np.maximum(mag, 1e-12))

        # Update plot data
        dpg.set_value("rt_plot_series", [freqs.tolist(), db_mag.tolist()])

        # Update plot title and legend label
        plot_title = "Room Target Preview - Saved Target"
        dpg.set_value("target_plot_title", plot_title)
        dpg.set_item_label("rt_plot_series", selected)

        hf.log_with_timestamp(f"Updated plot for room target '{selected}'", logger_obj)

    except Exception as e:
        hf.log_with_timestamp(f"Error loading room target '{app_data}': {e}", logger_obj)  

def clear_room_target_plot():
    """
    Clears the room target plot by showing a flat 0 dB line and resetting title/label.
    """
    try:
        fs = CN.FS
        n_fft = 65536
        freqs = np.fft.rfftfreq(n_fft, 1 / fs)
        flat_db = np.zeros_like(freqs)

        # Update plot with flat line
        dpg.set_value("rt_plot_series", [freqs.tolist(), flat_db.tolist()])

        # Clear title and legend label
        dpg.set_value("target_plot_title", "")
        dpg.set_item_label("rt_plot_series", "")
        
    except Exception as e:
        hf.log_with_timestamp(f"Error clearing plot: {e}")    

def open_sound_control_panel():
    """
    Opens the Windows legacy Sound Control Panel (mmsys.cpl) if the app is running on Windows.

    This function checks the operating system and attempts to launch the classic
    Sound settings dialog used to manage playback and recording devices.

    Logs an info message on non-Windows systems or an error if the subprocess fails.
    """
    if platform.system() == "Windows":
        try:
            result = subprocess.run(["control", "mmsys.cpl"], shell=True)
            logging.info(f"Sound Control Panel command exited with code {result.returncode}.")
        except Exception as e:
            logging.error(f"Exception occurred while trying to open Sound Control Panel: {e}")
    else:
        logging.info("Sound Control Panel is only available on Windows.")
       



#
#### HRTF favourites
#
       

def add_hrtf_favourite_callback(sender, app_data):
    """ 
    GUI function to add hrtf to favourites list
    """        
    update_hrtf_favourite('Export', 'add')
    
def add_hrtf_favourite_qc_callback(sender, app_data):
    """ 
    GUI function to add hrtf to favourites list
    """        
    update_hrtf_favourite('QC', 'add')


    
def remove_hrtf_favourite_callback(sender, app_data):
    """ 
    GUI function to add hrtf to favourites list
    """        
    update_hrtf_favourite('Export', 'remove')
    

    
def remove_hrtf_favourite_qc_callback(sender, app_data):
    """ 
    GUI function to add hrtf to favourites list
    """        
    update_hrtf_favourite('QC', 'remove')
   
  
def update_hrtf_favourite(tab, action): 
    """ 
    GUI function to add or remove an HRTF from the favourites list.

    Args:
        tab (str): 'QC' or other tab identifier.
        action (str): 'add' or 'remove'.
    """
    try:
        # Get current list of favourites (string list)
        current_fav_list = dpg.get_item_user_data("hrtf_add_favourite") or []
    
        # Get current selected hrtf (string) from GUI element
        brir_dict = get_brir_dict()
        if tab == 'QC':
            brir_hrtf_type = brir_dict.get('qc_brir_hrtf_type')
            brir_hrtf_dataset = brir_dict.get('qc_brir_hrtf_dataset')
            brir_hrtf = brir_dict.get('qc_brir_hrtf')
            brir_hrtf_short = brir_dict.get('qc_brir_hrtf_short')
        else:
            brir_hrtf_type = brir_dict.get('fde_brir_hrtf_type')
            brir_hrtf_dataset = brir_dict.get('fde_brir_hrtf_dataset')
            brir_hrtf = brir_dict.get('fde_brir_hrtf')
            brir_hrtf_short = brir_dict.get('fde_brir_hrtf_short')
            
        # print(brir_hrtf_type)
        # print(brir_hrtf_dataset)
        # print(brir_hrtf)
        # print(brir_hrtf_short)
        
        if brir_hrtf_type in ('Dummy Head / Head & Torso Simulator', 'Human Listener', 'Favourites', 'User SOFA Input') and brir_hrtf_short != CN.HRTF_USER_SOFA_DEFAULT:
            # Modify list based on action
            if action == "add":
                # Remove 'No favourites found' if it's present before adding new favourites
                updated_fav_list = [f for f in current_fav_list if f != 'No favourites found']
                if brir_hrtf_type == 'User SOFA Input':
                    #prefix with 'U-'
                    brir_hrtf_new = CN.HRTF_USER_SOFA_PREFIX+brir_hrtf_short
                else:
                    brir_hrtf_new=brir_hrtf_short
                if brir_hrtf_new and brir_hrtf_new not in updated_fav_list:
                    updated_fav_list.append(brir_hrtf_new)
        
            elif action == "remove":
                updated_fav_list = [item for item in current_fav_list if item != brir_hrtf_short]
        
            else:
                updated_fav_list = current_fav_list
    
            # If updated list is empty, replace with placeholder
            if not updated_fav_list:
                updated_fav_list = CN.HRTF_BASE_LIST_FAV
    
            # Replace user data
            dpg.configure_item('hrtf_add_favourite', user_data=updated_fav_list)
            
            # Also refresh list
            if brir_dict.get('fde_brir_hrtf_type') == 'Favourites':
                update_hrtf_list(sender=None, app_data=brir_dict.get('fde_brir_hrtf_dataset'))
            if brir_dict.get('qc_brir_hrtf_type') == 'Favourites':
                qc_update_hrtf_list(sender=None, app_data=brir_dict.get('qc_brir_hrtf_dataset'))
        
            # Save settings so that changes are saved to file
            save_settings()
            
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)   
    
   

       
    

        
 # --- Global flag to prevent re-entry ---
_is_creating_hrtf_avg = False

def create_hrtf_favourite_avg(sender, app_data):
    """
    GUI callback function to create an averaged HRTF and add it to favourites.
    Prevents multiple concurrent executions.
    """
    global _is_creating_hrtf_avg
    if _is_creating_hrtf_avg:
        logz = dpg.get_item_user_data("console_window")
        hf.log_with_timestamp("Averaged HRTF creation already in progress. Please wait...", logz)
        return

    _is_creating_hrtf_avg = True
    logz = dpg.get_item_user_data("console_window")

    def task():
        try:
            hf.log_with_timestamp("Starting averaged HRTF creation...", logz)

            dpg.configure_item("hrtf_average_fav_load_ind", show=True)  # loading indicators
            dpg.configure_item("qc_hrtf_average_fav_load_ind", show=True)

            # --- 1. Get and validate favourites list ---
            current_fav_list = dpg.get_item_user_data("hrtf_add_favourite") or []
            filtered_fav_list = [
                f for f in current_fav_list
                if f not in ('No favourites found', CN.HRTF_AVERAGED_NAME_GUI)
            ]

            if len(filtered_fav_list) < 2:
                hf.log_with_timestamp(
                    f"Insufficient favourites ({len(filtered_fav_list)} found). "
                    f"Need at least 2 to build average.", logz
                )
                dpg.configure_item('hrtf_add_favourite', user_data=current_fav_list)
                return

            hf.log_with_timestamp(
                f"Loading {len(filtered_fav_list)} HRIR sets for averaging...", logz
            )

            # --- 2. Load selected HRIR datasets ---
            hrir_list_favs_loaded, status = hrir_processing.load_hrirs_list(
                hrtf_list=filtered_fav_list, gui_logger=logz
            )

            if status != 0 or not hrir_list_favs_loaded:
                hf.log_with_timestamp("Failed to load HRIR datasets or too few loaded.", logz)
                return

            hf.log_with_timestamp(
                f"Loaded HRIR list structure:\n{[type(x) for x in hrir_list_favs_loaded]}",
                logz
            )
            for i, hrir in enumerate(hrir_list_favs_loaded):
                if isinstance(hrir, np.ndarray):
                    hf.log_with_timestamp(f"[{i}] NumPy array with shape {hrir.shape}", logz)
                elif isinstance(hrir, dict):
                    hf.log_with_timestamp(f"[{i}] Dict with keys: {list(hrir.keys())}", logz)
                else:
                    hf.log_with_timestamp(f"[{i}] Type: {type(hrir)}", logz)

            # --- 3. Perform interpolation & averaging ---
            hrir_avg = hrir_processing.build_averaged_listener_from_sets(
                hrir_list_favs_loaded,
                gui_logger=logz
            )

            if hrir_avg is None:
                hf.log_with_timestamp("Averaging failed: HRIR result is None.", logz)
                return

            # --- 4. Save as SOFA file ---
            output_path = dpg.get_value('qc_selected_folder_base')
            qc_samp_freq_str = dpg.get_value('qc_wav_sample_rate')
            qc_samp_freq_int = CN.SAMPLE_RATE_DICT.get(qc_samp_freq_str)

            brir_export.export_sofa_ir(
                primary_path=output_path,
                ir_arr=hrir_avg,
                ir_set_name=CN.HRTF_AVERAGED_NAME_FILE,
                samp_freq=qc_samp_freq_int,
                gui_logger=logz
            )

            # --- 5. Write metadata file ---
            try:
                metadata = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "included_hrtf_sets": filtered_fav_list,
                    "sample_rate": qc_samp_freq_int,
                    "interpolation_mode": "auto"
                }
                metadata_path = os.path.join(CN.DATA_DIR_HRIR_NPY_INTRP, "averaged_hrtf_metadata.json")
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=4)
                hf.log_with_timestamp(f"Metadata written to {metadata_path}", logz)
            except Exception as meta_e:
                hf.log_with_timestamp(f"Warning: Failed to write metadata file: {meta_e}", logz)

            # --- 6. Update favourites list ---
            updated_fav_list = filtered_fav_list + [CN.HRTF_AVERAGED_NAME_GUI]
            hf.log_with_timestamp("Averaged HRTF successfully created.", logz)

            # --- 7. Update GUI lists ---
            dpg.configure_item('hrtf_add_favourite', user_data=updated_fav_list)
            dpg.set_value("fde_brir_hrtf_type", 'Favourites')
            dpg.set_value("qc_brir_hrtf_type", 'Favourites')
            update_hrtf_dataset_list(sender=None, app_data='Favourites')
            qc_update_hrtf_dataset_list(sender=None, app_data='Favourites')
            dpg.set_value("fde_brir_hrtf_dataset", CN.HRTF_DATASET_CUSTOM_DEFAULT)
            dpg.set_value("qc_brir_hrtf_dataset", CN.HRTF_DATASET_CUSTOM_DEFAULT)
            update_hrtf_list(sender=None, app_data=CN.HRTF_DATASET_CUSTOM_DEFAULT)
            qc_update_hrtf_list(sender=None, app_data=CN.HRTF_DATASET_CUSTOM_DEFAULT)
            hf.log_with_timestamp("HRTF favourites list updated.", logz)
            qc_reset_progress()

        except Exception as e:
            hf.log_with_timestamp(f"Error in HRTF averaging callback: {e}", logz)

        finally:
            dpg.configure_item("hrtf_average_fav_load_ind", show=False)
            dpg.configure_item("qc_hrtf_average_fav_load_ind", show=False)
            global _is_creating_hrtf_avg
            _is_creating_hrtf_avg = False  # unlock after finish

    # run the heavy work in a background thread
    threading.Thread(target=task, daemon=True).start()       
        
        
def get_avg_hrtf_timestamp(metadata_dir=CN.DATA_DIR_HRIR_NPY_INTRP):
    """Return the timestamp string from the averaged HRTF metadata file."""
    metadata_path = os.path.join(metadata_dir, "averaged_hrtf_metadata.json")
    fallback = "no_averaged_meta"
    if not os.path.exists(metadata_path):
        return fallback  # No metadata file found

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata.get("timestamp", fallback)
    except Exception as e:
        hf.log_with_timestamp(f"Error reading metadata file: {e}")
        return fallback      
  
#
#### Misc
#    
    
def open_user_settings_folder(sender, app_data, user_data):
    """
    Opens the specified folder in the OS file explorer and logs the result.

    """
    selected_folder = CN.SETTINGS_DIR
    open_folder_in_explorer(selected_folder)
 
def open_user_sofa_folder(sender, app_data, user_data):
    """
    Opens the specified folder in the OS file explorer and logs the result.

    """
    selected_folder = CN.DATA_DIR_SOFA_USER
    open_folder_in_explorer(selected_folder)
     
 
    
 
#
#### QC Presets
# 
    

# --- Helper functions ---
def qc_get_selected_preset():
    """Return currently selected preset from listbox, or None."""
    selection = dpg.get_value("qc_preset_list")
    return selection if selection else None


    
def qc_refresh_preset_list(): 
    """Update listbox with Current, Default, and other presets (excluding settings.ini)."""
    # Start with fixed items
    presets = ["Current Parameters", "Default Settings"]

    # Add any other preset files from the folder, excluding 'settings.ini'
    other_presets = [
        f[:-4] for f in os.listdir(CN.SETTINGS_DIR)
        if f.endswith(".ini") and f[:-4] not in presets and f.lower() != "settings.ini"
    ]
    presets.extend(sorted(other_presets))  # optional: sort alphabetically

    # Update listbox
    dpg.configure_item("qc_preset_list", items=presets)

# --- Load callback ---
def qc_load_selected_preset_callback(sender, app_data):
    """Load the selected preset or apply default/current parameters."""
    logz = dpg.get_item_user_data("console_window")
    preset_name = qc_get_selected_preset()
    if not preset_name:
        hf.log_with_timestamp("No preset selected to load.")
        return

    if preset_name == "Current Parameters":
        hf.log_with_timestamp("Viewing current parameters; no changes applied.")
        return
    elif preset_name == "Default Settings":
        reset_settings()
        hf.log_with_timestamp("Loaded: Default Settings")
    else:
        try:
            #grab metadata of previously active filters/brirs
            qc_e_apo_curr_brir_set_prev = dpg.get_value('qc_e_apo_curr_brir_set')
            qc_e_apo_sel_brir_set_prev = dpg.get_value('qc_e_apo_sel_brir_set')
            qc_e_apo_curr_hpcf_prev = dpg.get_value('qc_e_apo_curr_hpcf')
            qc_e_apo_sel_hpcf_prev = dpg.get_value('qc_e_apo_sel_hpcf')
            
            #load settings AND apply to gui elements
            load_settings( preset_name=preset_name,  set_gui_values=True, logz=logz)
            hf.log_with_timestamp(f"Loaded: {preset_name}")
            
            #Apply hpcf parameters if enabled
            enable_hpcf_selected=dpg.get_value('e_apo_hpcf_conv')
            if enable_hpcf_selected == True:
                dpg.set_value("qc_e_apo_curr_hpcf", qc_e_apo_curr_hpcf_prev)
                dpg.set_value("qc_e_apo_sel_hpcf", qc_e_apo_sel_hpcf_prev) 
                qc_process_hpcfs()
            #Apply brir parameters if enabled
            enable_brir_selected=dpg.get_value('e_apo_brir_conv')
            if enable_brir_selected == True:
                dpg.set_value("qc_e_apo_curr_brir_set", qc_e_apo_curr_brir_set_prev)
                dpg.set_value("qc_e_apo_sel_brir_set", qc_e_apo_sel_brir_set_prev)
                qc_start_process_brirs()
            #will also save settings (main config) to ensure they are applied and saved
            #disable show history
            dpg.set_value("qc_toggle_hpcf_history", False)
            
            
        except Exception as e:
            hf.log_with_timestamp(f"Failed to load preset '{preset_name}': {e}")

# --- rename callback ---
def qc_rename_selected_preset_callback(sender, app_data):
    """Rename the selected preset file."""
    logz = dpg.get_item_user_data("console_window")
    preset_name = qc_get_selected_preset()
    
    if not preset_name:
        hf.log_with_timestamp("No preset selected to rename.", logz)
        return

    if preset_name in ("Current Parameters", "Default Settings"):
        hf.log_with_timestamp("Viewing current parameters; no changes applied.", logz)
        return

    try:
        # --- 1. Build full file path of current preset ---
        settings_dir = CN.SETTINGS_DIR
        old_path = os.path.join(settings_dir, f"{preset_name}.ini")

        # --- 2. Get new name from input field ---
        new_preset_name = dpg.get_value("qc_rename_selected_preset_text").strip()

        # --- 3. Validate new name ---
        if not new_preset_name:
            hf.log_with_timestamp("New preset name cannot be blank.", logz)
            return

        if new_preset_name == preset_name:
            hf.log_with_timestamp("New name is the same as the current name. No changes made.", logz)
            return

        new_path = os.path.join(settings_dir, f"{new_preset_name}.ini")

        # --- 4. Check existence ---
        if not os.path.exists(old_path):
            hf.log_with_timestamp(f"Preset file '{preset_name}.ini' does not exist.", logz)
            return

        if os.path.exists(new_path):
            hf.log_with_timestamp(f"A preset named '{new_preset_name}' already exists.", logz)
            return

        # --- 5. Rename the file ---
        os.rename(old_path, new_path)
        hf.log_with_timestamp(f"Preset renamed: '{preset_name}' → '{new_preset_name}'", logz)

        # --- 6. Refresh preset list in GUI ---
        qc_refresh_preset_list()

    except Exception as e:
        hf.log_with_timestamp(f"Failed to rename preset '{preset_name}': {e}", logz)


# --- Save callback ---
def qc_save_preset_callback(sender, app_data):
    """Save current parameters as a new preset file."""
    logz = dpg.get_item_user_data("console_window")
    enable_hpcf_selected=dpg.get_value('e_apo_hpcf_conv')
    enable_brir_selected=dpg.get_value('e_apo_brir_conv')
    hpcf_name = calc_hpcf_name(full_name=False)
    brir_name = calc_brir_set_name(full_name=False,tab=0)
    hpcf_name = hpcf_name.replace(",", "")
    hpcf_name = hpcf_name.replace("Sample ", "Samp.")
    brir_name = brir_name.replace(",", "")
    brir_name = brir_name.replace("Ear-High", "Ear-Hi")
    brir_name = brir_name.replace("Ear-Low", "Ear-Lo")
    if enable_hpcf_selected == True and enable_brir_selected == True:
        preset_name = hpcf_name +", " + brir_name  
    elif enable_hpcf_selected == True: #only hpcf enabled, assume only want to save hpcf
        preset_name = hpcf_name 
    elif enable_brir_selected == True: #only brir enabled, assume only want to save brir
        preset_name = brir_name 
    else:
        preset_name = hpcf_name +", " + brir_name  
    
    
    if not preset_name or preset_name in ["Current Parameters", "Default Settings"]:
        hf.log_with_timestamp("Invalid preset name.", gui_logger=logz)
        return

    try:
        save_settings(update_hpcf_pars=True, update_brir_pars=True, preset_name=preset_name)
        # Schedule the list refresh slightly after the debounce interval
        threading.Timer(CN._SAVE_DEBOUNCE_TIME + 0.2, qc_refresh_preset_list).start()
        hf.log_with_timestamp(f"Saved: {preset_name}", gui_logger=logz)
    except Exception as e:
        hf.log_with_timestamp(f"Failed to save preset '{preset_name}': {e}", gui_logger=logz)

# --- Delete callback ---
def qc_delete_selected_preset_callback(sender, app_data):
    """Delete the selected preset file from disk."""
    logz = dpg.get_item_user_data("console_window")
    preset_name = qc_get_selected_preset()
    if not preset_name:
        hf.log_with_timestamp("No preset selected to delete.", gui_logger=logz)
        dpg.configure_item("qc_del_preset_popup", show=False)
        return

    if preset_name in ["Current Parameters", "Default Settings"]:
        hf.log_with_timestamp(f"Cannot delete special preset: {preset_name}", gui_logger=logz)
        dpg.configure_item("qc_del_preset_popup", show=False)
        return

    preset_file = os.path.join(CN.SETTINGS_DIR, f"{preset_name}.ini")
    if os.path.exists(preset_file):
        try:
            os.remove(preset_file)
            qc_refresh_preset_list()
            hf.log_with_timestamp(f"Deleted: {preset_name}", gui_logger=logz)
        except Exception as e:
            hf.log_with_timestamp(f"Failed to delete preset '{preset_name}': {e}", gui_logger=logz)
    dpg.configure_item("qc_del_preset_popup", show=False)
