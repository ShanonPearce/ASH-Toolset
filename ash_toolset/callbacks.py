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
import dearpygui.dearpygui as dpg
import numpy as np
import math
from time import sleep
import threading
import os
import subprocess
import sys
import time
import csv
from datetime import datetime
import shutil
import scipy.signal as signal
import platform
import json
import urllib.request
from os.path import join as pjoin, expanduser, normpath


############################################### Settings and preset related
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
        hpcf_defaults = dpg.get_item_user_data("e_apo_sel_hpcf")
        logz = dpg.get_item_user_data("console_window")
    
        # Reset all GUI items from CN.DEFAULTS
        for key, default_value in CN.DEFAULTS.items():
            if dpg.does_item_exist(key):
                try:
                    dpg.set_value(key, default_value)
                except Exception as e:
                    logging.warning(f"Failed to reset GUI element '{key}': {e}")
    
        # Reset HpCF listboxes and selections
        if 'brands_list_default' in hpcf_defaults:
            dpg.configure_item("hpcf_brand", items=hpcf_defaults['brands_list_default'])
            dpg.set_value("hpcf_brand", hpcf_defaults['brand_default'])
        if 'hp_list_default' in hpcf_defaults:
            dpg.configure_item("hpcf_headphone", items=hpcf_defaults['hp_list_default'])
            dpg.set_value("hpcf_headphone", hpcf_defaults['headphone_default'])
        if 'sample_list_default' in hpcf_defaults:
            dpg.configure_item("hpcf_sample", items=hpcf_defaults['sample_list_default'])
            dpg.set_value("hpcf_sample", hpcf_defaults['sample_default'])
        
        #reset hpcf database, both tabs will be reset to ASH db
        change_hpcf_database(selected_db=CN.DEFAULTS["hpcf_active_database"],use_previous_settings=False)
        
        dpg.configure_item('hrtf_add_favourite',show=True)
        dpg.configure_item('hrtf_remove_favourite',show=False)
        dpg.configure_item('hrtf_average_favourite',show=False)
        dpg.configure_item('open_user_sofa_folder',show=False)
        updated_fav_list = CN.HRTF_BASE_LIST_FAV
        # Replace user data
        dpg.configure_item('hrtf_add_favourite', user_data=updated_fav_list)
        # Reset HRTF selection items
        dpg.configure_item('brir_hrtf', items=CN.DEFAULTS["brir_hrtf_listener_list"])
        dpg.configure_item('brir_hrtf_dataset', items=CN.DEFAULTS["brir_hrtf_dataset_list"])
        
        #sort acoustic spaces
        dpg.set_value("as_collection", CN.AC_SPACE_LIST_COLLECTIONS[0])
        updated_fav_list = CN.AS_BASE_LIST_FAV
        dpg.configure_item('as_add_favourite', user_data=updated_fav_list)
        update_ac_space_display()
        update_ac_space_info()
    
        # Reset progress bars and GUI toggles
        fde_reset_hpcf_progress()
        fde_reset_brir_progress()
        e_apo_reset_progress()
        e_apo_toggle_hpcf_gui(app_data=False)
        e_apo_toggle_brir_gui(app_data=False)
    
        # Reset output directories
        export_base_path = pjoin(expanduser("~"), "ASH-Toolset")
        export_ash_path = pjoin(export_base_path, CN.PROJECT_FOLDER)
        dpg.set_value('selected_folder_base', export_base_path)
        dpg.set_value('output_folder_fde', export_ash_path)
        dpg.set_value('output_folder_fde_tooltip', export_ash_path)
    
        #refresh direction fix
        refresh_direction_fix_selection()
    
        # Reset channel config
        reset_channel_config()
        e_apo_update_elev_list()
        fde_update_angle_list()
    
        # Save settings after reset
        save_settings(update_hpcf_pars=True, update_brir_pars=True)
        
        dpg.configure_item("reset_settings_popup", show=False)
    
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
            "fde_gain_",
            "fde_elev_angle_",
            "fde_az_angle_",
            "e_apo_prevent_clip",
            "e_apo_audio_channels", "e_apo_brir_spat_res"
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
        e_apo_update_elev_list()
        e_apo_reset_progress()
        e_apo_config_acquire(caller='configure_brir')
        fde_update_angle_list()

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
            hf.log_with_timestamp(f"Warning: Migration step failed – {e}", logz)

        config = configparser.ConfigParser()
        config.read(settings_file)

        # --- Version check ---
        version_loaded = hf.safe_get(config, "version", str, version)
        if version_loaded != version:
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
                            hf.log_with_timestamp(f"Loaded legacy key '{old_key}' → '{key}'", logz)
                            break
            except Exception as e:
                hf.log_with_timestamp(f"Failed to load key '{key}' from {settings_file}: {e}", logz)

        hf.log_with_timestamp(f"Loaded settings file: {settings_file}.", logz)
        # --- Optionally update GUI ---
        if set_gui_values:
            try:
                
                #refresh brand, headphone, sample and database gui elements. Will also #update qc hpcf lists and #set history view
                change_hpcf_database(selected_db=loaded_values.get("hpcf_active_database"), use_previous_settings=True)
                hpcf_db_dict = dpg.get_item_user_data("e_apo_sel_hpcf")
                if not hpcf_db_dict or "conn" not in hpcf_db_dict:
                    raise RuntimeError("No valid database connection found in e_apo_sel_hpcf user data.")
           
                #display all acoustic spaces
                dpg.set_value("as_collection", CN.AC_SPACE_LIST_COLLECTIONS[0])
                update_ac_space_display()
                
                #update dataset and listener lists
                brir_hrtf_type = loaded_values.get('brir_hrtf_type')
                brir_hrtf_dataset = loaded_values.get('brir_hrtf_dataset')
                _handle_hrtf_dataset_update(brir_hrtf_type,update_lists_only=True)
                _handle_hrtf_list_update(brir_hrtf_dataset,selected_type=brir_hrtf_type,update_lists_only=True)
 
                #update all gui values
                applied = 0
                for key, value in loaded_values.items():
                    if dpg.does_item_exist(key):
                        try:
                            dpg.set_value(key, value)
                            applied += 1
                        except Exception as e:
                            hf.log_with_timestamp(f"Failed to set GUI element '{key}': {e}", logz)
                            
                e_apo_reset_progress()
                fde_reset_brir_progress()
                fde_reset_hpcf_progress()
                e_apo_update_elev_list()
                fde_update_angle_list()
                #sort acoustic spaces
                update_ac_space_display()
                update_ac_space_info()
                #refresh direction fix
                refresh_direction_fix_selection()
                #refresh on off buttons
                hf.set_checkbox_and_sync_button(checkbox_tag="e_apo_hpcf_conv",button_tag="e_apo_hpcf_conv_btn",refresh=True)
                hf.set_checkbox_and_sync_button(checkbox_tag="e_apo_brir_conv",button_tag="e_apo_brir_conv_btn",refresh=True)
                
                hf.log_with_timestamp("Refreshed HpCF GUI elements.")
                
                hf.log_with_timestamp(f"Applied {applied} loaded settings to GUI elements.", logz)

            except Exception as e:
                hf.log_with_timestamp(f"Error applying loaded settings to GUI: {e}", logz)

    except Exception as e:
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
    if dpg.get_value('e_apo_auto_apply_hpcf_sel'):
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
    config['DEFAULT']['as_list_favs'] = json.dumps(dpg.get_item_user_data("as_add_favourite") or [])
    config['DEFAULT']['path'] = dpg.get_value('selected_folder_base')

    # --- Define deferred keys ---
    hpcf_keys = {'hpcf_brand', 'hpcf_headphone', 'hpcf_sample','hpcf_active_database'}
    brir_keys = {
        'brir_hp_type', 'room_target', 'direct_gain', 'direct_gain_slider',
        'acoustic_space', 'brir_hrtf', 'brir_hrtf_type','e_apo_brir_spat_res',
        'brir_hrtf_dataset', 'crossover_f_mode', 'crossover_f',
        'sub_response', 'hp_rolloff_comp', 'fb_filtering_mode','hrtf_direction_misalign_comp'
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
    
 
        log_string = f"Preset successfully saved to '{settings_file}'."
        if settings_file != CN.SETTINGS_FILE:
            hf.log_with_timestamp(log_string=log_string, gui_logger=logz)
            
        if CN.LOG_MEMORY:
            hf.log_memory_usage()
    
    except Exception as e:
        log_string = f"Error: Failed to write settings to '{settings_file}' — {e}"
        logging.error(log_string)
        hf.log_with_timestamp(log_string=log_string, gui_logger=logz)

    #rerun audio device text update for regular refresh
    hf.update_default_output_text()



######################################  GUI Functions - HPCFs
#
#

def update_hpcf_gui_labels():
    """Update GUI labels and widths based on the currently active HPCF database."""
    active_database = dpg.get_value('hpcf_active_database')

    if active_database == CN.HPCF_DATABASE_LIST[0]:  # Ash filters
        # Brand search
        dpg.set_value('hpcf_brand_search_title', "Search Brand:")
        dpg.configure_item('hpcf_brand_search_title', show=True)
        dpg.configure_item('hpcf_brand_search', show=True)

        # Headphone search
        dpg.set_value('hpcf_headphone_search_title', "Search Headphone:")

        # Table headers
        dpg.set_value('hpcf_brand_title', "Brand")
        dpg.set_value('hpcf_headphone_title', "Headphone")
        dpg.set_value('hpcf_sample_title', "Sample")

        # Column widths
        dpg.configure_item('hpcf_brand', width=135)
        dpg.configure_item('hpcf_headphone', width=250)
        dpg.configure_item('hpcf_sample', width=120)

    else:  # Compilation database
        # Brand search
        dpg.set_value('hpcf_brand_search_title', "Search Type:")
        dpg.configure_item('hpcf_brand_search_title', show=False)
        dpg.configure_item('hpcf_brand_search', show=False)

        # Headphone search
        dpg.set_value('hpcf_headphone_search_title', "Search Headphone:")

        # Table headers
        dpg.set_value('hpcf_brand_title', "Type")
        dpg.set_value('hpcf_headphone_title', "Headphone")
        dpg.set_value('hpcf_sample_title', "Source")

        # Column widths
        dpg.configure_item('hpcf_brand', width=80)
        dpg.configure_item('hpcf_headphone', width=290)
        dpg.configure_item('hpcf_sample', width=140)

def get_hpcf_dict(caller='apply_hpcf'):
    """
    Retrieves current GUI selections and available lists for both QC and FDE tabs.
    Uses the active SQLite connection from 'e_apo_sel_hpcf' user data.

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
        brand = gui_state["qc"]["brand"]
        headphones = gui_state["qc"]["headphones_list"]
        
        # Access FDE tab values
        fde_sample = gui_state["fde"]["sample"]
    """
    
    # Retrieve shared database connection
    hpcf_db_dict = dpg.get_item_user_data("e_apo_sel_hpcf")
    if not hpcf_db_dict or "conn" not in hpcf_db_dict:
        raise RuntimeError("No valid database connection found in e_apo_sel_hpcf user data.")
    conn = hpcf_db_dict["conn"]

    result = {}

    try:

        # Read current GUI selections
        enable_hpcf_selected=dpg.get_value('e_apo_hpcf_conv')
        if caller=='apply_hpcf':#default flow - assume function was called by applying hpcf selection, grab from gui because gui values are applied headphone
            headphone_selected = dpg.get_value("hpcf_headphone")
            sample_selected = dpg.get_value("hpcf_sample")
        else:#fallback - assume function was called elsewhere, grab from settings in case changes havent been applied, so that the config only contains applied headphone
            loaded_values = load_settings()
            headphone_selected = loaded_values['hpcf_headphone']
            sample_selected = loaded_values['hpcf_sample']
            

        # Resolve brand name from headphone (returns None if not found)
        # there should only be one brand value per headphone value so can be derived from headphone
        brand_selected = hpcf_functions.get_brand(conn, headphone_selected) if headphone_selected else None

        # Get available lists safely
        brands_list = hpcf_functions.get_brand_list(conn) or []
        hp_list = hpcf_functions.get_headphone_list(conn, brand_selected) or []
        sample_list = hpcf_functions.get_samples_list(conn, headphone_selected) or []
        sample_list_sorted = sorted(sample_list) if sample_list else []

        result = {
            "brand": brand_selected,
            "headphone": headphone_selected,
            "sample": sample_selected,
            "brands_list": brands_list,
            "headphones_list": hp_list,
            "samples_list": sample_list_sorted,
            'enable_conv': enable_hpcf_selected
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
        hpcf_db_dict = dpg.get_item_user_data("e_apo_sel_hpcf")
        
        #ensure both tabs have the same database active at the same time as db dict considers one active db
        dpg.set_value('hpcf_active_database', selected_db)
        
        #function will update text labels based on schema ---
        update_hpcf_gui_labels()
     
        #also update active database connection
        if selected_db == CN.HPCF_DATABASE_LIST[0]:#ash filters
            hpcf_db_dict['conn'] = hpcf_db_dict.get('conn_ash') or hpcf_db_dict['conn']
            hpcf_db_dict['database'] = hpcf_db_dict.get('database_ash') or hpcf_db_dict['database']
        elif selected_db == CN.HPCF_DATABASE_LIST[1]:#compilation database
            hpcf_db_dict['conn'] = hpcf_db_dict.get('conn_comp') or hpcf_db_dict['conn']
            hpcf_db_dict['database'] = hpcf_db_dict.get('database_comp') or hpcf_db_dict['database']
        else:
            logging.error(f"Unknown database selection: {selected_db}")
            return
    
        handle_hpcf_update( update_type="full_reset")
  
        #attempt to default to previously saved settings for convenience, in case db was switched back to original db.
        if use_previous_settings == True:
            loaded_values = load_settings()
            brand_applied = loaded_values['hpcf_brand']
            headphone_applied = loaded_values['hpcf_headphone']
            sample_applied = loaded_values['hpcf_sample']
            
            handle_hpcf_update( update_type="full_update", brand=brand_applied, headphone=headphone_applied, sample=sample_applied)
            
      
        #reset search bars
        dpg.set_value("hpcf_brand_search", "")
        #dpg.set_value("hpcf_headphone_search", "")
        current_search = dpg.get_value('hpcf_headphone_search')
        if current_search != "":
            filter_hpcf_lists( "headphone", current_search)

        #will reset progress bars  
        #dpg.set_value("toggle_hpcf_history", False)
        hpcf_hist_toggled = dpg.get_value('toggle_hpcf_history')
        if hpcf_hist_toggled:
            show_hpcf_history(app_data=hpcf_hist_toggled)
            
        e_apo_reset_progress()
        fde_reset_hpcf_progress()
        
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)    

def filter_hpcf_lists(search_type, app_data):
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
        hpcf_db_dict = dpg.get_item_user_data("e_apo_sel_hpcf")
        conn = hpcf_db_dict["conn"]
        brands_list = hpcf_functions.get_brand_list(conn)
        
        # --- UI element tags ---
        brand_combo = "hpcf_brand"
        headphone_combo = "hpcf_headphone"
        sample_combo = "hpcf_sample"
        plot_dest = 1
        hpcf_database_sel =  dpg.get_value('hpcf_active_database')
    
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
        hpcf_functions.hpcf_to_plot(conn, headphone, sample_default, plot_dest=plot_dest)
    
        # --- Reset progress ---
        dpg.set_value("toggle_hpcf_history", False)
        e_apo_reset_progress()
        fde_reset_hpcf_progress()
    
        save_settings()
        
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)
        
        

def filter_brand_list(sender, app_data):
    """Brand search callback (updates only the relevant tab)."""
  

    filter_hpcf_lists( "brand", app_data)


def filter_headphone_list(sender, app_data):
    """Headphone search callback (updates only the relevant tab)."""


    filter_hpcf_lists( "headphone", app_data)



  
    
  
def show_hpcf_history(sender=None, app_data=None):
    """ 
    GUI function to update list of headphone based on exported hpcf files
    """
    
    try:
        hpcf_db_dict=dpg.get_item_user_data("e_apo_sel_hpcf")#dict contains db connection object
        conn = hpcf_db_dict['conn']
        brands_list = hpcf_functions.get_brand_list(conn)
        
        output_path = dpg.get_value('e_apo_config_folder')
        samples_list_all_distinct =hpcf_functions.get_all_samples_list(conn)
        hp_list_out_latest = e_apo_config_creation.get_exported_hp_list(output_path,samples_list_all_distinct)
        search_str = hp_list_out_latest
        #update brand list with filtered set
        headphone_list_saved = hpcf_functions.search_headphones_in_list(conn, search_str)
        
 

        hpcf_database_sel = dpg.get_value('hpcf_active_database')
        
        #attempt to apply previously selected settings if switched off
        gui_state = get_hpcf_dict()
        brand_selected = gui_state["brand"]
        headphone_selected = gui_state["headphone"]
        sample_selected = gui_state["sample"]
        brands_list = gui_state["brands_list"]
        hp_list_selected = gui_state["headphones_list"]
        samples_list = gui_state["samples_list"]
        
        #toggled on and there are saved filters -> populate lists with saved filters
        if headphone_list_saved and headphone_list_saved != None and app_data == True:
            #if selected headphone is in history, set default to selected, otherwise pick first value
            default_headphone = headphone_selected if headphone_selected in headphone_list_saved else headphone_list_saved[0]
        
            #clear out brand list
            dpg.configure_item('hpcf_brand',items=[])
            #force update of brand value to ensure it is up to date and synchronised despite list being blank
            default_brand=hpcf_functions.get_brand(conn, default_headphone)
            dpg.set_value("hpcf_brand", default_brand)
            
            #update headphone list
            dpg.configure_item('hpcf_headphone',items=headphone_list_saved)
            
            #reset headphone value to first headphone
            dpg.set_value("hpcf_headphone", default_headphone)
            
            #also update sample list
            #headphone = headphone_list_specific[0]
            sample_list_specific = hpcf_functions.get_samples_list(conn, default_headphone)
            sample_list_sorted = (sorted(sample_list_specific))
            dpg.configure_item('hpcf_sample',items=sample_list_sorted)
            sample_default = CN.HPCF_SAMPLE_DEFAULT if hpcf_database_sel == CN.HPCF_DATABASE_LIST[0] else sample_list_sorted[0]
            sample_new = sample_selected if headphone_selected in headphone_list_saved else sample_default
      
            #also update plot to Sample A
            #sample = CN.HPCF_SAMPLE_DEFAULT
            hpcf_functions.hpcf_to_plot(conn, default_headphone, sample_new, plot_dest=CN.TAB_QC_CODE)
            
            #reset sample list to Sample A
            dpg.set_value("hpcf_sample", sample_new)
    
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
            dpg.configure_item('hpcf_brand',items=brands_list)
            #reset brand value to first brand
            dpg.set_value("hpcf_brand", brand_selected)
            #update headphone list
            
            dpg.configure_item('hpcf_headphone',items=hp_list_selected)
            dpg.set_value("hpcf_headphone", headphone_selected)
            #also update sample list

            dpg.configure_item('hpcf_sample',items=sample_list_sorted)
            #also update plot
            hpcf_functions.hpcf_to_plot(conn, headphone_selected, sample_selected, plot_dest=CN.TAB_QC_CODE)
            #reset sample list
            dpg.set_value("hpcf_sample", sample_selected)
            
        
        dpg.configure_item("clear_history_popup", show=False)
        #reset search bars
        dpg.set_value("hpcf_brand_search", "")
        dpg.set_value("hpcf_headphone_search", "")
        #reset progress
        e_apo_reset_progress()
        save_settings()

    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)



    
       
def handle_hpcf_update( update_type, changed_item=None, brand=None, headphone=None, sample=None):
    """
    Unified handler for both FDE and QC tab callbacks.
    Handles ASH and Compilation database hierarchies with dynamic sample selection.

    Parameters:
        tab_prefix (str): 'fde' or 'qc'
        update_type (str): 'full_reset', 'brand', 'headphone', 'sample', or 'full_update'
    """
    
    try:
        # Retrieve DB connection
        hpcf_db_dict = dpg.get_item_user_data("e_apo_sel_hpcf")
        conn = hpcf_db_dict['conn']
    
        # Try to get GUI logger
        logz = None
    
        plot_dest =  1
        hpcf_database_sel = dpg.get_value('hpcf_active_database')
        auto_apply_id = "e_apo_auto_apply_hpcf_sel"
        brand_id = "hpcf_brand"
        headphone_id = "hpcf_headphone"
        sample_id = "hpcf_sample"
    
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
                hpcf_functions.hpcf_to_plot(conn, headphone, sample_to_use, plot_dest=plot_dest)
    
            if sample_list_sorted:
                refresh_dropdown(sample_id, sample_list_sorted, sample_to_use)
    
            current_brand = hpcf_functions.get_brand(conn, headphone)
            if current_brand:
                dpg.set_value(brand_id, current_brand)
    
        # --- MAIN HANDLING LOGIC ---
        if update_type == "full_update":
            hf.log_with_timestamp("Performing full HPCF update", logz)
            if not all([brand, headphone, sample]):
                hf.log_with_timestamp("Full update skipped: missing brand/headphone/sample", logz, log_type=1)
                return
            refresh_dropdown(brand_id, hpcf_functions.get_brand_list(conn) or [], brand)
            refresh_dropdown(headphone_id, hpcf_functions.get_headphone_list(conn, brand) or [], headphone)
            refresh_dropdown(sample_id, sorted(hpcf_functions.get_samples_list(conn, headphone) or []), sample)
            hf.log_with_timestamp(f"Full update set: brand={brand}, headphone={headphone}, sample={sample}", logz)
            hpcf_functions.hpcf_to_plot(conn, headphone, sample, plot_dest=plot_dest)
    
        elif update_type in ("full_reset", "brand", "headphone"):
            if update_type == "full_reset":
                hf.log_with_timestamp("Resetting HPCF selection", logz)
                brands_list_default = hpcf_functions.get_brand_list(conn) or []
                if not brands_list_default:
                    fde_reset_hpcf_progress()
                    e_apo_reset_progress()
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
                    fde_reset_hpcf_progress()
                    e_apo_reset_progress()
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
                hpcf_functions.hpcf_to_plot(conn, headphone_selected, sample_to_use, plot_dest=plot_dest)
    
        # Reset progress and save
        fde_reset_hpcf_progress()
        e_apo_reset_progress()
        save_settings()
    
        if dpg.get_value(auto_apply_id):
            hf.log_with_timestamp("Auto-applying HPCF in QC tab", logz)
            qc_process_hpcfs()    
            
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)


# QC tab
def update_headphone_list(sender, app_data):
    handle_hpcf_update( "brand",  app_data)

def update_sample_list(sender, app_data):
    handle_hpcf_update( "headphone",  app_data)

def plot_sample(sender, app_data, user_data):
    handle_hpcf_update( "sample",  app_data)    
    


 
def fde_export_hpcf_file_toggle(sender, app_data):
    """ 
    GUI function to trigger save and refresh
    """

    save_settings()
    #reset progress
    fde_reset_hpcf_progress()


    
def fde_process_hpcfs(sender=None, app_data=None, user_data=None):
    """GUI function to process HPCFs (FDE tab) safely with validation and logging."""

    try:
        # Retrieve logger and DB connection
        logz = dpg.get_item_user_data("console_window")  # contains logger
        hpcf_db_dict = dpg.get_item_user_data("e_apo_sel_hpcf")

        if not hpcf_db_dict or "conn" not in hpcf_db_dict:
            hf.log_with_timestamp("No valid database connection found — skipping FDE HPCF processing.", logz)
            return
        conn = hpcf_db_dict["conn"]
    
        # Get GUI state for both tabs (we only need FDE)
        gui_state = get_hpcf_dict()
        fde_state = gui_state
    
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
    
        fir_export = dpg.get_value("fde_fir_hpcf_toggle")
        fir_stereo_export = dpg.get_value("fde_fir_st_hpcf_toggle")
        geq_export = dpg.get_value("fde_geq_hpcf_toggle")
        geq_31_export = dpg.get_value("fde_geq_31_hpcf_toggle")
        geq_103_export = False
        hesuvi_export = dpg.get_value("fde_hesuvi_hpcf_toggle")
    
        samp_freq_str = dpg.get_value("wav_sample_rate")
        samp_freq_int = CN.SAMPLE_RATE_DICT.get(samp_freq_str)
        bit_depth_str = dpg.get_value("wav_bit_depth")
        bit_depth = CN.BIT_DEPTH_DICT.get(bit_depth_str)
        resample_mode=CN.RESAMPLE_MODE_LIST[0]
        fir_length = dpg.get_value("hpcf_fir_length")
    
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
            gui_logger=logz, resample_mode=resample_mode,
            report_progress=2,
            force_output=True, fir_length=fir_length
        )
    
        save_settings(update_hpcf_pars=True)
        
        # everything is valid and completed
        hf.log_with_timestamp(f" Export filters for {brand} / {headphone}", logz)
        
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)
    
   
def e_apo_apply_hpcf_params(sender=None, app_data=None):
    """ 
    GUI function to apply hpcf parameters
    """
    force_output=False

        
    qc_process_hpcfs(force_output=force_output)
            
 
   
    
def qc_process_hpcfs(force_output=False):
    """ 
    GUI function to process HPCFs safely with validation and logging.
    """
    
    # ---- Non-Windows behaviour (Linux/macOS for now) ----
    if sys.platform != "win32":
        hf.log_with_timestamp("Unable to apply filters in Equalizer APO due to incompatible OS")
        save_settings()
        return

    # Retrieve logger and database connection
    logz = dpg.get_item_user_data("console_window")  # contains logger
    hpcf_db_dict = dpg.get_item_user_data("e_apo_sel_hpcf")
    if not hpcf_db_dict or "conn" not in hpcf_db_dict:
        hf.log_with_timestamp(" No valid database connection found — skipping HPCF processing.", logz)
        return
    conn = hpcf_db_dict["conn"]

    # Get current GUI state
    gui_state = get_hpcf_dict()
    state = gui_state

    brand = state["brand"]
    headphone = state["headphone"]
    sample = state["sample"]
    brands_list = state["brands_list"]
    hp_list = state["headphones_list"]
    sample_list_sorted = state["samples_list"]

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

    output_path = dpg.get_value("e_apo_config_folder")
    fir_export = True
    fir_stereo_export = False
    geq_export = False
    geq_31_export = False
    geq_103_export = False
    hesuvi_export = False
    eapo_export = False

    samp_freq_str = dpg.get_value("wav_sample_rate")
    samp_freq_int = CN.SAMPLE_RATE_DICT.get(samp_freq_str)
    bit_depth_str = dpg.get_value("wav_bit_depth")
    bit_depth = CN.BIT_DEPTH_DICT.get(bit_depth_str)
    resample_mode=CN.RESAMPLE_MODE_LIST[0]
    fir_length = dpg.get_value("hpcf_fir_length")

    # Call processing function
    success = hpcf_functions.hpcf_to_file_bulk(
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
        gui_logger=logz, resample_mode=resample_mode,
        report_progress=1,
        force_output=force_output, fir_length=fir_length
    )
    
    if not success:
        hf.log_with_timestamp("HPCF bulk export encountered errors.", logz, log_type=1)
    else:
        # Update displayed names and save settings
        filter_name = calc_hpcf_name(full_name=False)
        filter_name_full = calc_hpcf_name(full_name=True)
        dpg.set_value("e_apo_curr_hpcf", filter_name)
        dpg.set_value("e_apo_sel_hpcf", filter_name_full)
        
        # Update configuration
        #dpg.set_value("e_apo_hpcf_conv", True)
        hf.set_checkbox_and_sync_button(checkbox_tag="e_apo_hpcf_conv",button_tag="e_apo_hpcf_conv_btn",value=True)
        e_apo_config_acquire(caller='apply_hpcf')
  
        save_settings(update_hpcf_pars=True)

        
        # everything is valid and completed
        hf.log_with_timestamp(f" Applied filter for {brand} / {headphone} / {sample}", logz)
  
   
    
#
#
################# GUI Functions - BRIR parameter Related Callbacks
#
#


def refresh_direction_fix_selection():
    """
    Update the selection of the direction-fix GUI element based on the
    previously applied fix stored in the HRIR metadata file.

    If metadata is missing or fails to load, the GUI will be set to the default value.
    """
    try:
        # Get currently selected BRIR info
        brir_meta_dict = get_brir_dict()
        brir_hrtf_type = brir_meta_dict.get('brir_hrtf_type')
        brir_hrtf_dataset = brir_meta_dict.get('brir_hrtf_dataset')
        brir_hrtf_gui = brir_meta_dict.get('brir_hrtf')
        brir_hrtf_short = brir_meta_dict.get('brir_hrtf_short')

        # Build HRTF metadata dictionary
        hrtf_dict_list = [
            {
                "hrtf_type": brir_hrtf_type,
                "hrtf_dataset": brir_hrtf_dataset,
                "hrtf_gui": brir_hrtf_gui,
                "hrtf_short": brir_hrtf_short
            }
        ]

        # Load metadata only (fast, avoids loading full HRIR arrays)
        _, status, hrir_metadata_list = hrir_processing.load_hrirs_list(hrtf_dict_list=hrtf_dict_list,metadata_only=True)

        # If metadata loaded successfully, update GUI with previously applied direction fix
        if status == 0 and hrir_metadata_list:
            hrir_metadata = hrir_metadata_list[0]
            direction_fix_applied = hrir_metadata.get("direction_fix_gui")
            if direction_fix_applied is not None:
                dpg.set_value("hrtf_direction_misalign_comp", direction_fix_applied)
            else:
                # fallback to default if not present in metadata
                dpg.set_value("hrtf_direction_misalign_comp", CN.HRTF_DIRECTION_FIX_LIST_GUI[0])
        else:
            # fallback to default if loading failed
            dpg.set_value("hrtf_direction_misalign_comp", CN.HRTF_DIRECTION_FIX_LIST_GUI[0])

    except Exception as e:
        # log the error and reset GUI to default
        hf.log_with_timestamp(f"Error refreshing direction fix selection: {e}", log_type=2, exception=e)
        dpg.set_value("hrtf_direction_misalign_comp", CN.HRTF_DIRECTION_FIX_LIST_GUI[0])
    

def fde_select_spatial_resolution(sender, app_data):
    """ 
    GUI function to update spatial resolution based on input
    """
    try:

        #update hrtf list based on spatial resolution
        #also update file format selection based on spatial resolution
        #set some to false and hide irrelevant options
        # if app_data == 'High':

        #     dpg.configure_item("fde_sofa_brir_toggle", show=True)
 
        #     dpg.configure_item("fde_sofa_brir_tooltip", show=True)
            
        # else:
    
        #     dpg.set_value("fde_sofa_brir_toggle", False)
    
        #     dpg.configure_item("fde_sofa_brir_toggle", show=False)
    
        #     dpg.configure_item("fde_sofa_brir_tooltip", show=False)
       
        #reset progress bar
        fde_reset_brir_progress()
        
        fde_update_angle_list()
        
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)



def fde_update_angle_list():
    """ 
    GUI function to process updates in export config section.
    This updates the elevation and azimuth lists based on saved spatial resolution.
    """

    # --- Get saved spatial resolution ---
    spat_res = dpg.get_value("fde_brir_spat_res")

    if spat_res not in CN.SPATIAL_RES_LIST:
        return  # safety guard

    spat_res_int = CN.SPATIAL_RES_LIST.index(spat_res)

    # --- Get new elevation list ---
    elev_list_new = CN.ELEV_ANGLES_WAV_ALL[spat_res_int]

    # --- Elevation combo item tags ---
    elev_items = [
        "fde_elev_angle_fl",
        "fde_elev_angle_fr",
        "fde_elev_angle_c",
        "fde_elev_angle_sl",
        "fde_elev_angle_sr",
        "fde_elev_angle_rl",
        "fde_elev_angle_rr",
    ]

    # --- Update elevation combos and validate current values ---
    for item in elev_items:
        dpg.configure_item(item, items=elev_list_new)

        try:
            current_val = int(dpg.get_value(item))
        except (TypeError, ValueError):
            current_val = None

        if current_val not in elev_list_new:
            fallback = 0 if 0 in elev_list_new else elev_list_new[0]
            dpg.set_value(item, fallback)

    # --- Base azimuth lists (copy to avoid mutating constants) ---
    az_lists = {
        "fde_az_angle_fl": CN.AZ_ANGLES_FL_WAV.copy(),
        "fde_az_angle_fr": CN.AZ_ANGLES_FR_WAV.copy(),
        "fde_az_angle_c":  CN.AZ_ANGLES_C_WAV.copy(),
        "fde_az_angle_sl": CN.AZ_ANGLES_SL_WAV.copy(),
        "fde_az_angle_sr": CN.AZ_ANGLES_SR_WAV.copy(),
        "fde_az_angle_rl": CN.AZ_ANGLES_RL_WAV.copy(),
        "fde_az_angle_rr": CN.AZ_ANGLES_RR_WAV.copy(),
    }

    # --- High spatial resolution expansion ---
    if spat_res_int > 1:
        # RL special-case adjustment
        if 180 in az_lists["fde_az_angle_rl"]:
            az_lists["fde_az_angle_rl"].remove(180)
            az_lists["fde_az_angle_rl"].append(-175)

        def expand_az_list(az_list, step=5):
            if not az_list:
                return az_list

            az_min = min(az_list)
            az_max = max(az_list)

            expanded = list(range(az_min, az_max + step, step))
            expanded.sort(key=lambda x: (abs(x), x))  # center-first ordering
            return expanded

        for key in az_lists:
            az_lists[key] = expand_az_list(az_lists[key])

    # --- Update azimuth combos and validate current values ---
    for item, az_list in az_lists.items():
        dpg.configure_item(item, items=az_list)

        try:
            current_val = int(dpg.get_value(item))
        except (TypeError, ValueError):
            current_val = None

        if current_val not in az_list:
            fallback = 0 if 0 in az_list else az_list[0]
            dpg.set_value(item, fallback)



def select_hp_comp(sender, app_data):
    _handle_hp_comp_selection()

def _handle_hp_comp_selection():
    """
    Unified handler for headphone compensation selection.
    """

    # Get selected HP type depending on tab
    hp_type = dpg.get_value("brir_hp_type")

    pinna_comp = CN.HP_COMP_LIST.index(hp_type)

    try:
        # Load FIR dataset
        npy_fname = pjoin(CN.DATA_DIR_INT, 'headphone_ear_comp_dataset.npy')
        ear_comp_fir_dataset = hf.load_convert_npy_to_float64(npy_fname)
        pinna_comp_fir_loaded = ear_comp_fir_dataset[pinna_comp, :]

        # Zero-pad / place FIR into FFT buffer
        pinna_comp_fir = np.zeros(CN.N_FFT)
        pinna_comp_fir[:1024] = pinna_comp_fir_loaded[:1024]

        # Plot
        plot_title = f"Headphone Compensation: {hp_type}"

        view = dpg.get_value("plot_type")

        # Plot using the new generic function
        hf.plot_fir_generic(
            fir_array=pinna_comp_fir,
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
            y_lim_a=-10,
            y_lim_b=10,
            plot_dest=CN.TAB_QC_CODE
        )

    except Exception as e:
        hf.log_with_timestamp(f"_handle_hp_comp_selection failed: {e}")

    # Tab-specific BRIR update
    fde_reset_brir_progress()
    update_brir_param()
        
        

 
def update_direct_gain(sender, app_data):
    """ 
    GUI function to update brir based on input
    """
    d_gain=app_data
    dpg.set_value("direct_gain_slider", d_gain)

    #reset progress bar
    update_brir_param()
    

def update_direct_gain_slider(sender, app_data):
    """ 
    GUI function to update brir based on input
    """
    
    d_gain=app_data
    dpg.set_value("direct_gain", d_gain)

    #reset progress bar
    update_brir_param()    
 
    
 


def update_crossover_f(sender=None, app_data=None, user_data=None):
    """ 
    GUI function to make updates to freq crossover parameters
    """
    if app_data == CN.SUB_FC_SETTING_LIST[0]:
        ac_space = dpg.get_value("acoustic_space")

        reverb_data=CN.reverb_data
        f_crossover_var = int(CN.extract_column(data=reverb_data, column='f_crossover', condition_key='name_gui', condition_value=ac_space, return_all_matches=False))
        dpg.set_value("crossover_f", f_crossover_var)
        
    if type(app_data) is int:
        dpg.set_value("crossover_f_mode", CN.SUB_FC_SETTING_LIST[1])#set mode to custom
        
    update_brir_param()




def update_ac_space(sender, app_data, user_data):
    """ 
    GUI function to make updates to acoustic spaces
    """
    crossover_mode = dpg.get_value("crossover_f_mode")
    if crossover_mode == CN.SUB_FC_SETTING_LIST[0]:
        update_crossover_f(app_data=crossover_mode)
        
    ac_space = app_data

    plot_ac_space(ac_space_gui=ac_space, plot_dest=CN.TAB_QC_CODE)
    
    update_ac_space_info()
    
    update_brir_param()

def plot_ac_space(ac_space_gui, plot_dest):
    """
    Shared function to plot an acoustic space response.

    Parameters:
    - ac_space_gui: str, gui name of acoustic space
    - ac_space_src: str, source name of acoustic space
    - plot_dest: int, different plot styles (1 or 2)
    """
    try:
  
        reverb_data=CN.reverb_data
        as_file_name = CN.extract_column(data=reverb_data, column='file_name', condition_key='name_gui', condition_value=ac_space_gui, return_all_matches=False)
        as_folder = CN.extract_column(data=reverb_data, column='folder', condition_key='name_gui', condition_value=ac_space_gui, return_all_matches=False)
        as_id = CN.extract_column(data=reverb_data, column='id', condition_key='name_gui', condition_value=ac_space_gui, return_all_matches=False)
        if as_folder == 'user':
            brir_rev_folder = pjoin(CN.DATA_DIR_INT, 'reverberation', as_folder, as_id)
        else:
            brir_rev_folder = pjoin(CN.DATA_DIR_INT, 'reverberation', as_folder)
        npy_file_name = as_file_name + '.npy'
        npy_file_path = pjoin(brir_rev_folder, npy_file_name)
        # --- Try loading from local file ---
        brir_reverberation = hf.load_convert_npy_to_float64(npy_file_path)
        if brir_reverberation is None or len(brir_reverberation) == 0:
            hf.log_with_timestamp('Reverberation dataset is empty. Unable to plot.')
            return
  
        fir = brir_reverberation

        plot_title = f"{ac_space_gui}"
  
        view = dpg.get_value("plot_type")
        # Plot using the new generic function
        hf.plot_fir_generic(
            fir_array=fir,
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
            y_lim_b=10,
            plot_dest=plot_dest
        )

    except Exception as e:
        hf.log_with_timestamp(f"Failed to plot acoustic space '{ac_space_gui}': {e}")
    

def update_ac_space_info():
    """ 
    GUI function to make updates to acoustic space info text
    """
    
    ac_space = dpg.get_value("acoustic_space")

    reverb_data=CN.reverb_data
    ac_id = CN.extract_column(data=reverb_data, column='id', condition_key='name_gui', condition_value=ac_space, return_all_matches=False)
    ac_label = CN.extract_column(data=reverb_data, column='label', condition_key='name_gui', condition_value=ac_space, return_all_matches=False)
    ac_rt60 = CN.extract_column(data=reverb_data, column='meas_rt60', condition_key='name_gui', condition_value=ac_space, return_all_matches=False)
    ac_desc = CN.extract_column(data=reverb_data, column='description', condition_key='name_gui', condition_value=ac_space, return_all_matches=False)
    ac_col = CN.extract_column(data=reverb_data, column='collection_1', condition_key='name_gui', condition_value=ac_space, return_all_matches=False)
    dpg.set_value("acoustic_space_id_text", ac_id)
    dpg.set_value("acoustic_space_name_text", ac_label)
    dpg.set_value("acoustic_space_rt60_text", ac_rt60)
    if ac_col == 'User Imports':
        dpg.set_value("acoustic_space_desc_text", "")
    else:
        dpg.set_value("acoustic_space_desc_text", ac_desc)
            


  


def update_ac_space_display(sender=None, app_data=None, user_data=None):
    """
    Central function to handle both filtering and sorting logic 
    to ensure they respect each other.
    """
    try:
        # 1. Get current states from GUI
        selected_collection = dpg.get_value("as_collection") # Ensure this tag matches your filter dropdown
        sort_criterion = dpg.get_value("sort_by_as")
        
        # 2. Start with the "Master Data" - Zip everything together to keep rows aligned
        # (name, id, r60)
        master_list = list(zip(
            CN.AC_SPACE_LIST_GUI, 
            CN.AC_SPACE_LIST_ID, 
            CN.AC_SPACE_LIST_MEAS_R60
        ))

        # 3. APPLY FILTERING
        filtered_data = []
        if not selected_collection or selected_collection == "All":
            filtered_data = master_list
        elif selected_collection == "Favourites":
            favourites = dpg.get_item_user_data("as_add_favourite") or []
            for item in master_list:
                name_gui, space_id, r60 = item
                if name_gui in favourites:
                    filtered_data.append(item)
        else:
            for item in master_list:
                name_gui, space_id, r60 = item
                
                # Check metadata collections using your existing logic
                col1 = CN.extract_column(CN.reverb_data, "collection_1", "name_gui", name_gui, False)
                col2 = CN.extract_column(CN.reverb_data, "collection_2", "name_gui", name_gui, False)

                if selected_collection in (col1, col2):
                    filtered_data.append(item)

        # 4. APPLY SORTING to the filtered result
        if sort_criterion == 'Name':
            # Sort by name_gui (index 0)
            filtered_data.sort(key=lambda x: x[0].lower() if isinstance(x[0], str) else x[0])
            
        elif sort_criterion == 'Reverberation Time':
            # Sort by R60 (index 2)
            filtered_data.sort(key=lambda x: x[2])
            
        elif sort_criterion == 'ID':
            # Sort by ID (index 1)
            filtered_data.sort(key=lambda x: x[1].lower() if isinstance(x[1], str) else x[1])

        # 5. Push final list to UI
        final_names = [item[0] for item in filtered_data]
        dpg.configure_item('acoustic_space', items=final_names)
        
        # Configure favourite/user buttons based on type
        if selected_collection == "Favourites":
            dpg.configure_item("as_add_favourite", show=False)
            dpg.configure_item("as_remove_favourite", show=True)
        else:
            dpg.configure_item("as_add_favourite", show=True)
            dpg.configure_item("as_remove_favourite", show=False)
        
        update_ac_space_info()
        e_apo_reset_progress()
        fde_reset_brir_progress()

    except Exception as e:
        hf.log_with_timestamp(f"Update failed: {e}")


def add_as_favourite_callback(sender, app_data):
    """ 
    GUI function to add hrtf to favourites list
    """        
    update_as_favourite( 'add')


    
def remove_as_favourite_callback(sender, app_data):
    """ 
    GUI function to add hrtf to favourites list
    """        
    update_as_favourite( 'remove')
   
  
def update_as_favourite( action): 
    """ 
    GUI function to add or remove an HRTF from the favourites list.

    Args:
        tab (str): 'QC' or other tab identifier.
        action (str): 'add' or 'remove'.
    """
    try:
        # Get current list of favourites (string list)
        current_fav_list = dpg.get_item_user_data("as_add_favourite") or []
    
        # Get current selected AS from GUI element
        as_selected = dpg.get_value("acoustic_space")


        # Modify list based on action
        if action == "add":
            # Remove 'No favourites found' if it's present before adding new favourites
            updated_fav_list = [f for f in current_fav_list if f != 'No favourites found']

            if as_selected not in updated_fav_list:
                updated_fav_list.append(as_selected)
    
        elif action == "remove":
            updated_fav_list = [item for item in current_fav_list if item != as_selected]
    
        else:
            updated_fav_list = current_fav_list

        # If updated list is empty, replace with placeholder
        if not updated_fav_list:
            updated_fav_list = CN.AS_BASE_LIST_FAV

        # Replace user data
        dpg.configure_item('as_add_favourite', user_data=updated_fav_list)
        
        # Also refresh list
        update_ac_space_display()
    
        # Save settings so that changes are saved to file
        save_settings()
            
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)   
    



def select_room_target(sender, app_data):
    """Callback for room target"""
    plot_room_target(app_data, plot_dest=CN.TAB_QC_CODE)
    update_brir_param()    
 
    
def plot_room_target(target_sel: str, plot_dest: int):
    """
    Shared function to plot a room target frequency response.

    Parameters:
    - target_sel: str, key name of the room target in ROOM_TARGETS_DICT
    - plot_dest: int, different plot styles (1 or 2)
    """
    try:
        target_data = CN.ROOM_TARGETS_DICT.get(target_sel)
        if not target_data:
            hf.log_with_timestamp(f"Room target '{target_sel}' not found in ROOM_TARGETS_DICT.")
            return

        fir = target_data["impulse_response"]

        room_target_fir = np.zeros(CN.N_FFT)
        room_target_fir[:len(fir)] = fir

        plot_title = f"{target_sel}"
  
        view = dpg.get_value("plot_type")
        # Plot using the new generic function
        hf.plot_fir_generic(
            fir_array=room_target_fir,
            view=view,
            title_name=plot_title,
            samp_freq=CN.SAMP_FREQ,
            n_fft=CN.N_FFT,
            normalise=2,
            level_ends=0,
            x_lim_adjust=True,
            x_lim_a=20,
            x_lim_b=20000,
            y_lim_adjust=True,
            y_lim_a=-12,
            y_lim_b=12,
            plot_dest=plot_dest
        )

    except Exception as e:
        hf.log_with_timestamp(f"Failed to plot room target '{target_sel}': {e}")
    
 
# --- Wrapper callbacks ---


def update_hrtf_dataset_list(sender, app_data):
    _handle_hrtf_dataset_update(app_data)     


def update_hrtf_list(sender, app_data):
    _handle_hrtf_list_update(app_data)    

def _handle_hrtf_dataset_update(selected_type,update_lists_only=False):
    """
    Unified handler for updating HRTF dataset list based on HRTF type.
    """
    try:
        if selected_type is None:
            return

        # Configure favourite/user buttons based on type
        if selected_type == "Favourites":
            dpg.configure_item("hrtf_add_favourite", show=False)
            dpg.configure_item("hrtf_remove_favourite", show=True)
            dpg.configure_item("hrtf_average_favourite", show=True)
            dpg.configure_item("open_user_sofa_folder", show=False)
        elif selected_type == "User SOFA Input":
            dpg.configure_item("hrtf_add_favourite", show=True)
            dpg.configure_item("hrtf_remove_favourite", show=False)
            dpg.configure_item("hrtf_average_favourite", show=False)
            dpg.configure_item("open_user_sofa_folder", show=True)
        else:
            dpg.configure_item("hrtf_add_favourite", show=True)
            dpg.configure_item("hrtf_remove_favourite", show=False)
            dpg.configure_item("hrtf_average_favourite", show=False)
            dpg.configure_item("open_user_sofa_folder", show=False)

        # Update dataset list
        dataset_list = CN.HRTF_TYPE_DATASET_DICT.get(selected_type, [])
        dpg.configure_item("brir_hrtf_dataset", items=dataset_list)
        if dataset_list:
            dataset_new = dataset_list[0]
            dpg.configure_item("brir_hrtf_dataset", show=False)
            dpg.configure_item("brir_hrtf_dataset", default_value=dataset_new)
            dpg.configure_item("brir_hrtf_dataset", show=True)
        else:
            dataset_new = None

        # Update HRTF list for first dataset
        if dataset_new:
            hrtf_list = hrir_processing.get_listener_list(listener_type=selected_type, dataset_name=dataset_new)
            
            #if favourites selected and averaged HRTF is in the listener list, check if npy file exists. If not exists, remove averaged HRTF from listener list
            if selected_type == "Favourites" and hrtf_list and CN.HRTF_AVERAGED_NAME_GUI in hrtf_list:
                hrtf_type = selected_type
                hrtf_dataset = dataset_new
                hrtf_gui = CN.HRTF_AVERAGED_NAME_GUI
                hrtf_short = CN.HRTF_AVERAGED_NAME_GUI
                #translate hrtf parameters
                hrtf_type, hrtf_dataset, hrtf_gui, hrtf_short = hrir_processing.hrtf_param_cleaning(hrtf_type,hrtf_dataset,hrtf_gui,hrtf_short)
                # --- Construct file path ---
                npy_fname = hrir_processing.get_hrir_file_path(hrtf_type=hrtf_type,hrtf_gui=hrtf_gui,hrtf_dataset=hrtf_dataset,hrtf_short=hrtf_short)
                # Remove averaged entry if file missing
                if not Path(npy_fname).exists():
                    hrtf_list.remove(CN.HRTF_AVERAGED_NAME_GUI)
            
            #proceed to update listner list and default item
            dpg.configure_item("brir_hrtf", items=hrtf_list)
            hrtf_new = hrtf_list[0] if hrtf_list else "No HRTFs Found"
            dpg.configure_item("brir_hrtf", show=False)
            dpg.configure_item("brir_hrtf", default_value=hrtf_new)
            dpg.configure_item("brir_hrtf", show=True)
            
            
        #knock on effects for changing type
        if update_lists_only == False:
            # Tab-specific selection callback and reset
            select_hrtf()
            e_apo_reset_progress()
            fde_reset_brir_progress()
      
            save_settings()

    except Exception as e:
        hf.log_with_timestamp(f"Error updating HRTF dataset list: {e}", log_type=2, exception=e)

def _handle_hrtf_list_update(selected_dataset,selected_type=None,update_lists_only=False):
    """
    Unified handler for updating HRTF list based on selected dataset.
    """
    try:
        if selected_dataset is None:
            return
        if selected_type is None:
            listener_type = dpg.get_value("brir_hrtf_type")
        else:
            listener_type = selected_type

        # Update HRTF list
        hrtf_list = hrir_processing.get_listener_list(listener_type=listener_type, dataset_name=selected_dataset)
        dpg.configure_item("brir_hrtf", items=hrtf_list)
        hrtf_new = hrtf_list[0] if hrtf_list else "No HRTFs Found"

        # Reset selection to first HRTF
        dpg.configure_item("brir_hrtf", show=False)
        dpg.configure_item("brir_hrtf", default_value=hrtf_new)
        dpg.configure_item("brir_hrtf", show=True)

        #knock on effects for changing type
        if update_lists_only == False:
            # Tab-specific selection callback and reset
            select_hrtf()
            e_apo_reset_progress()
            fde_reset_brir_progress()

            save_settings()

    except Exception as e:
        hf.log_with_timestamp(f"Error updating HRTF list: {e}", log_type=2, exception=e)


def select_hrtf(sender=None, app_data=None):
    _select_hrtf_common()
    
    
 


def _load_hrir_selection(skip_sofa=True):
    """
    Helper function: loads and returns the HRIR dataset and its metadata
    for the current selected parameters.

    Returns:
        (hrir_selected, hrir_metadata)
        - hrir_selected: np.ndarray or None
        - hrir_metadata: dict or None
    """
    hrir_selected = None
    hrir_metadata = None

    brir_meta_dict = get_brir_dict()

    # Extract HRTF identifiers from BRIR dictionary
    brir_hrtf_type = brir_meta_dict.get('brir_hrtf_type')
    brir_hrtf_dataset = brir_meta_dict.get('brir_hrtf_dataset')
    brir_hrtf_gui = brir_meta_dict.get('brir_hrtf')
    brir_hrtf_short = brir_meta_dict.get('brir_hrtf_short')
    hrtf_direction_misalign_comp = brir_meta_dict.get("hrtf_direction_misalign_comp")
    hrtf_low_freq_suppression = brir_meta_dict.get('hrtf_low_freq_suppression')

    # Build HRTF metadata dictionary
    hrtf_dict_list = [
        {
            "hrtf_type": brir_hrtf_type,
            "hrtf_dataset": brir_hrtf_dataset,
            "hrtf_gui": brir_hrtf_gui,
            "hrtf_short": brir_hrtf_short
        }
    ]

    try:
        hf.log_with_timestamp(f"Loading HRIR: {brir_hrtf_gui}")

        hrir_list, status, hrir_metadata_list = hrir_processing.load_hrirs_list(
            hrtf_dict_list=hrtf_dict_list,
            force_skip_sofa=skip_sofa,
            direction_fix_gui=hrtf_direction_misalign_comp,
            apply_lf_suppression=hrtf_low_freq_suppression
        )

        if status != 0:
            raise RuntimeError(f"HRIR loader returned non-zero status ({status})")

        if not hrir_list:
            raise ValueError(f"No HRIRs returned for dataset: {brir_hrtf_short}")

        # Extract HRIR and metadata (single selection)
        hrir_selected = hrir_list[0]
        if hrir_metadata_list:
            hrir_metadata = hrir_metadata_list[0]

    except Exception as e:
        hf.log_with_timestamp(f"_load_hrir_selection failed: {e}", log_type=2)

    return hrir_selected, hrir_metadata
    
def _select_hrtf_common():
    """
    Shared core for HRTF selection: loads the HRIR, extracts a representative FIR
    (0° elevation, 30° azimuth, right ear), optionally applies diffuse-field 
    calibration reversal, plots it, and refreshes GUI elements.
    """
    brir_meta_dict = get_brir_dict()

    # Settings
    plot_dest = 1
    spat_res_int = 2

    # Extract HRTF identifiers from BRIR dictionary
    brir_hrtf_gui = brir_meta_dict.get('brir_hrtf')
    brir_hrtf_short = brir_meta_dict.get('brir_hrtf_short')

    hrtf_df_cal_mode = brir_meta_dict.get("hrtf_df_cal_mode")

    view = dpg.get_value("plot_type")
    
    fir_array = CN.IMPULSE
    plot_title = f"HRTF sample: {brir_hrtf_short} (no preview available)"

    try:
        hf.log_with_timestamp(f"Loading HRIR: {brir_hrtf_gui}")

        # ------------------------------------------------------------
        # Load HRIR and metadata via helper
        # ------------------------------------------------------------
        hrir_selected, hrir_metadata = _load_hrir_selection()
        if hrir_selected is None:
            raise RuntimeError("HRIR loading failed")
        
        total_elev_hrir, total_azim_hrir, total_chan_hrir, total_samples_hrir = hrir_selected.shape

        # Spatial resolution
        elev_min = CN.SPATIAL_RES_ELEV_MIN_IN[spat_res_int]
        elev_step = CN.SPATIAL_RES_ELEV_NEAREST_IN[spat_res_int]
        azim_step = CN.SPATIAL_RES_AZIM_NEAREST_IN[spat_res_int]

        hf.log_with_timestamp(
            f"Loaded HRIR shape={hrir_selected.shape}, "
            f"elev_min={elev_min}, elev_step={elev_step}, azim_step={azim_step}"
        )


        # ------------------------------------------------------------
        # Retrieve preview direction + channel from GUI
        # ------------------------------------------------------------
        try:
            target_elev_deg = float(dpg.get_value("filter_prev_elev"))
            target_azim_deg = float(dpg.get_value("filter_prev_azim"))
            target_azim_deg = hf.azimuth_to_circular(target_azim_deg)
        except Exception:
            raise ValueError("Invalid elevation or azimuth selection from GUI")
        
        channel_gui = dpg.get_value("filter_prev_channel")
        
        # Map GUI channel → index
        if channel_gui in ("R", "Right", "Right Ear"):
            chan = 1
        elif channel_gui in ("L", "Left", "Left Ear"):
            chan = 0
        else:
            raise ValueError(f"Unknown channel selection: {channel_gui}")

        elev_idx = round((target_elev_deg - elev_min) / elev_step)
        azim_idx = round(target_azim_deg / azim_step)

        if elev_idx < 0 or elev_idx >= total_elev_hrir or azim_idx < 0 or azim_idx >= total_azim_hrir:
            raise ValueError(f"No HRIR found at elevation={target_elev_deg}°, azimuth={target_azim_deg}°.")

        # Extract only the valid FIR samples (no zero-padding)
        hrir = hrir_selected[elev_idx, azim_idx, chan, :total_samples_hrir]

        # --- Optional diffuse field reversal ---
        df_cal_reversal = (hrtf_df_cal_mode != CN.HRTF_DF_CAL_MODE_LIST[0] and hrir_metadata is not None)
        if df_cal_reversal:
            #ctf_file is original ctf, including any roll off. ctf_le_file has leveled ends, reverses DF correction exactly and wont have roll off
            desired_ctf = "ctf_le_file" if hrtf_df_cal_mode == CN.HRTF_DF_CAL_MODE_LIST[2] else "ctf_file"
            ctf_path = hrir_metadata.get(desired_ctf, None)

            if not ctf_path or not Path(ctf_path).exists():
                reason = "Metadata missing CTF key" if not ctf_path else "CTF file not found"
                hf.log_with_timestamp(f"{reason}: '{desired_ctf}' for {brir_hrtf_short}", log_type=1)
            else:
                # Read CTF and ensure mono
                _, ctf_data = hf.read_wav_file(ctf_path)
                if ctf_data.ndim > 1:
                    ctf_data = ctf_data[:, 0]

                # --- Convolve only valid portions ---
                hrir = np.convolve(hrir, ctf_data, mode="full")
                hf.log_with_timestamp(f"Applied DF calibration reversal via convolution ({desired_ctf})")

        # --- Plot FIR ---
        #plot_title = f"HRTF sample: {brir_hrtf_short} 0° elevation, -30° azimuth, left ear"
        plot_title = f"HRTF sample: {brir_hrtf_short} {target_elev_deg:.0f}° elevation, {target_azim_deg:.0f}° azimuth, {channel_gui} ear"
        fir_array = hrir
        

    except Exception as e:
        hf.log_with_timestamp(f"select_hrtf failed: {e}")

    hf.plot_fir_generic(
        fir_array=fir_array,
        view=view,
        title_name=plot_title,
        samp_freq=CN.SAMP_FREQ,
        n_fft=CN.N_FFT,
        normalise=2,
        level_ends=False,#True
        x_lim_adjust=True,
        x_lim_a=200,#200
        x_lim_b=20000,
        y_lim_adjust=True,
        y_lim_a=-10,
        y_lim_b=15,
        plot_dest=plot_dest,
        smooth_win_base=8
    )

    # --- Reset progress bars ---
    fde_reset_brir_progress()
    fde_reset_hpcf_progress()
    e_apo_reset_progress()
    # --- Refresh direction fix GUI ---
    refresh_direction_fix_selection()    
    
    
    
def select_sub_brir(sender, app_data):
    """ 
    GUI handler for sub BRIR selection
    """

    logz=dpg.get_item_user_data("console_window")#contains logger
    sub_response=app_data
    plot_type = dpg.get_value("plot_type")
    #run plot
    plot_sub_brir(sub_response,CN.TAB_QC_CODE,plot_type)
    #reset progress bar
    update_brir_param()
 

    
def plot_sub_brir(name, plot_dest,plot_type):
    """
    Callback to plot the magnitude response of a subwoofer BRIR.

    Parameters
    ----------
    name : str
        GUI name of the subwoofer BRIR to plot.
    plot_dest : int
        Specifies which DearPyGui plot instance to update.
    """
    try:
        # --- Load subwoofer data ---
        sub_data = CN.sub_data
        sub_file_name = CN.extract_column(
            data=sub_data,
            column='file_name',
            condition_key='name_gui',
            condition_value=name,
            return_all_matches=False
        )
        sub_folder = CN.extract_column(
            data=sub_data,
            column='folder',
            condition_key='name_gui',
            condition_value=name,
            return_all_matches=False
        )

        if sub_folder in ('sub', 'lf_brir'):  # default sub responses
            npy_fname = pjoin(CN.DATA_DIR_SUB, sub_file_name + '.npy')
        else:  # user-provided sub response
            file_folder = pjoin(CN.DATA_DIR_AS_USER, name)
            npy_fname = pjoin(file_folder, sub_file_name + '.npy')

        sub_brir_npy = hf.load_convert_npy_to_float64(npy_fname)

        # --- Plot using generic FIR function ---
        hf.plot_fir_generic(
            fir_array=sub_brir_npy,
            view=plot_type,
            title_name=name,
            samp_freq=CN.SAMP_FREQ,
            n_fft=CN.N_FFT,
            normalise=1,
            x_lim_adjust=True,
            x_lim_a=10,
            x_lim_b=150,
            plot_dest=plot_dest
        )

    except Exception as e:
        logging.error(f"Failed to plot sub BRIR '{name}': {e}", exc_info=True)

     




def load_ia_brir_fir(
    mode="single_direction",
    elev=0,
    azim=-30,
    channel=0,
    enforce_azim_symmetry=True
):
    """
    Load FIR for integrated analysis.

    Modes
    -----
    single_direction :
        Load first-channel FIR from a specific BRIR WAV
        (default: elev=0, azim=-30).

    summary :
        Build a summary FIR from all previously processed BRIRs
        stored in user_data ("e_apo_brir_conv").

    Returns
    -------
    fir : np.ndarray
        1D FIR array.
    samplerate : int
        Sample rate.
    title : str
        Plot / display title.

    Raises
    ------
    RuntimeError on failure.
    """

    n_fft = CN.N_FFT
    samplerate = CN.SAMP_FREQ
    brir_name = dpg.get_value("e_apo_curr_brir_set")

    # ------------------------------------------------------------
    # MODE 1: SINGLE DIRECTION (existing behaviour)
    # ------------------------------------------------------------
    if mode == "single_direction":

        primary_path = dpg.get_value("e_apo_config_folder")
        brir_set = CN.FOLDER_BRIRS_LIVE.replace(" ", "_")
        brirs_path = pjoin(primary_path, CN.PROJECT_FOLDER_BRIRS, brir_set)

        wav_fname = pjoin(brirs_path, f"BRIR_E{elev}_A{azim}.wav")

        if not Path(wav_fname).exists():
            raise RuntimeError(f"BRIR file not found: {wav_fname}")

        samplerate, wav_data = hf.read_wav_file(wav_fname)

        fir = np.zeros(n_fft)
        length = min(len(wav_data), n_fft)
        fir[:length] = wav_data[:length, channel]  # FIRST CHANNEL ONLY

        title = f"{brir_name} (E{elev}, A{azim})"
        return fir, samplerate, title

    # ------------------------------------------------------------
    # MODE 2: SUMMARY RESPONSE
    # ------------------------------------------------------------
    elif mode == "summary":
    
        hf.log_with_timestamp("Building BRIR summary response")
    
        brir_dict_list = dpg.get_item_user_data("e_apo_brir_conv") or []
    
        fir_list = []
    
        for data_dict in brir_dict_list:
    
            # ----------------------------------------------------
            # Optional azimuth symmetry enforcement
            # ----------------------------------------------------
            if enforce_azim_symmetry:
                azim_deg = data_dict.get("azim_deg_wav")
                if azim_deg is None:
                    continue
                if azim_deg % 15 != 0:
                    continue
    
            out_wav_array = data_dict.get("out_wav_array")
    
            if out_wav_array is None or out_wav_array.size == 0:
                continue
    
            # Expected shape: (samples, channels)
            if out_wav_array.ndim == 1:
                # Mono fallback
                fir_list.append(out_wav_array)
    
            elif out_wav_array.ndim == 2:
                # Explicitly grab BOTH channels
                for ch in range(out_wav_array.shape[1]):
                    fir_list.append(out_wav_array[:, ch])
    
            else:
                hf.log_with_timestamp(
                    f"Skipping BRIR with unexpected shape {out_wav_array.shape}",
                    log_type=1
                )
    
        # --------------------------------------------------------
        # If nothing usable found → return impulse FIR
        # --------------------------------------------------------
        if not fir_list:
            fir = np.zeros(4096)
            fir[0] = 1.0  # unit impulse
    
            title = "Process new dataset to generate summary"
            return fir, samplerate, title
    
        # Stack into (measurements, samples)
        brir_array = np.asarray(fir_list)
    
        fir = hf.build_summary_response_fir(
            brir_array,
            fs=CN.SAMP_FREQ,
            n_fft=n_fft,
            truncate_len=16384,
        )
    
        title = f"Diffuse-field Summary: {brir_name}"
        return fir, samplerate, title

    # ------------------------------------------------------------
    # INVALID MODE
    # ------------------------------------------------------------
    else:
        raise ValueError(f"Unknown mode '{mode}'")




def ia_new_plot(sender=None, app_data=None, user_data=None):
    """
    Callback for integrated plot-type combo.

    sender   : combo tag
    app_data : selected plot type string
    user_data: plot_state dict
    """

    plot_state = {}

    try:
        plot_type = dpg.get_value("ia_plot_type")
        # ------------------------------------------------------------
        # Retrieve preview direction + channel from GUI
        # ------------------------------------------------------------
        try:
            target_elev_deg = int(dpg.get_value("ia_plot_elev"))
            target_azim_deg = int(dpg.get_value("ia_plot_azim"))
        except Exception:
            raise ValueError("Invalid elevation or azimuth selection from GUI")
        
        channel_gui = dpg.get_value("ia_plot_channel")
        
        # Map GUI channel → index
        if channel_gui in ("R", "Right", "Right Ear"):
            chan = 1
        elif channel_gui in ("L", "Left", "Left Ear"):
            chan = 0
        else:
            raise ValueError(f"Unknown channel selection: {channel_gui}")
            
        load_brir_mode = 'summary' if plot_type == 'Summary Response' else 'single_direction'
        fir, samplerate, title = load_ia_brir_fir(mode=load_brir_mode,elev=target_elev_deg,azim=target_azim_deg,channel=chan)

        # Update shared plot state
        plot_state.update({
            "fir": fir,
            "samp_freq": samplerate,
            "title": title,
            "n_fft": CN.N_FFT,
            "normalise": 1,
            "level_ends": 0,
            # decay defaults (safe even if unused)
            "decay_start_ms": 0.0,
            "decay_end_ms": 1000.0,
            "decay_step_ms": 100.0,
            "decay_win_ms": 100.0,
            # LF-friendly defaults
            "x_lim_adjust": True,
            "x_lim_a": 20,
            "x_lim_b": 20000,
            "y_lim_adjust": False,
            "y_lim_a": -150,
            "y_lim_b": 150,
        })

        hf.plot_fir_generic(
            fir_array=fir,
            view=plot_type,
            title_name=title,
            samp_freq=samplerate,
            n_fft=CN.N_FFT,
            normalise=plot_state["normalise"],
            level_ends=plot_state["level_ends"],
            decay_start_ms=plot_state["decay_start_ms"],
            decay_end_ms=plot_state["decay_end_ms"],
            decay_step_ms=plot_state["decay_step_ms"],
            decay_win_ms=plot_state["decay_win_ms"],
            plot_dest=CN.TAB_QC_IA_CODE,
            x_lim_adjust=plot_state["x_lim_adjust"],
            x_lim_a=plot_state["x_lim_a"],
            x_lim_b=plot_state["x_lim_b"],
            y_lim_adjust=plot_state["y_lim_adjust"],
            y_lim_a=plot_state["y_lim_a"],
            y_lim_b=plot_state["y_lim_b"],
        )

    except Exception as ex:
        hf.log_with_timestamp(f"LFA plot failed: {ex}")

        # fallback: flat impulse
        impulse = np.zeros(CN.N_FFT)
        impulse[0] = 1.0

        hf.plot_fir_generic(
            fir_array=impulse,
            view=CN.PLOT_TYPE_LIST_IA[0],
            title_name="Direction not found in output",
            samp_freq=CN.SAMP_FREQ,
            n_fft=CN.N_FFT,
            plot_dest=CN.TAB_QC_IA_CODE,
        )











    
def plot_type_changed(sender, app_data, user_data):
    """
    Callback for plot type combo.

    sender   : combo tag (e.g. 'plot_type')
    app_data : newly selected plot type string
    user_data: plot_state dict stored on the combo
    """

    plot_state = user_data

    if not isinstance(plot_state, dict):
        return
    if "fir" not in plot_state:
        return

    try:
        hf.plot_fir_generic(
            fir_array=plot_state["fir"],
            view=app_data,
            title_name=plot_state.get("title", "Output"),
            samp_freq=plot_state.get("samp_freq", CN.SAMP_FREQ),
            n_fft=plot_state.get("n_fft", CN.N_FFT),
            normalise=plot_state.get("normalise", 1),
            level_ends=plot_state.get("level_ends", 0),
            decay_start_ms=plot_state.get("decay_start_ms", 0.0),
            decay_end_ms=plot_state.get("decay_end_ms", 1000.0),
            decay_step_ms=plot_state.get("decay_step_ms", 100.0),
            decay_win_ms=plot_state.get("decay_win_ms", 100.0),
            plot_dest=plot_state.get("plot_dest", CN.TAB_QC_CODE),
            x_lim_adjust=plot_state.get("x_lim_adjust", False),
            x_lim_a=plot_state.get("x_lim_a", 20),
            x_lim_b=plot_state.get("x_lim_b", 20000),
            y_lim_adjust=plot_state.get("y_lim_adjust", False),
            y_lim_a=plot_state.get("y_lim_a", -25),
            y_lim_b=plot_state.get("y_lim_b", 15),
        )

    except Exception as e:
        hf.log_with_timestamp(f"plot type update failed: {e}")





    

def update_brir_param(sender=None, app_data=None):
    """ 
    GUI function to update brir based on input
    """

    #reset progress bar
    e_apo_reset_progress()
    fde_reset_brir_progress()
    
    save_settings()
    


def sync_wav_sample_rate(sender, app_data):
    """ 
    GUI function to update settings based on toggle
    """
    dpg.set_value("wav_sample_rate", app_data)
    
    #reset progress bar
    e_apo_reset_progress()
    fde_reset_brir_progress()
    fde_reset_hpcf_progress()
    
    save_settings()

def sync_wav_bit_depth(sender, app_data):
    """ 
    GUI function to update settings based on toggle
    """
    dpg.set_value("wav_bit_depth", app_data)
    
    #reset progress bar
    e_apo_reset_progress()
    fde_reset_brir_progress()
    fde_reset_hpcf_progress()
    
    save_settings()

 
 




#  
#
################# BRIR Processing related callbacks
#    
#  
    
def fde_start_process_brirs(sender, app_data, user_data):
    """ 
    GUI function to start or stop head tracking thread
    """
    #thread bool
    process_brirs_running=dpg.get_item_user_data("fde_brir_tag")
    
    if process_brirs_running == False:

        #set thread running flag
        process_brirs_running = True
        #update user data
        dpg.configure_item('fde_brir_tag',user_data=process_brirs_running)
        dpg.configure_item('fde_brir_tag',label="Cancel")
        
        #set stop thread flag flag
        stop_thread_flag = False
        #update user data
        dpg.configure_item('fde_progress_bar_brir',user_data=stop_thread_flag)
        
        #start thread
        thread = threading.Thread(target=fde_process_brirs, args=(), daemon=True)
        thread.start()
   
    else:
        
        #set stop thread flag flag
        stop_thread_flag = True
        #update user data
        dpg.configure_item('fde_progress_bar_brir',user_data=stop_thread_flag)

def fde_process_brirs(sender=None, app_data=None, user_data=None):
    """ 
    GUI function to process BRIRs
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    #grab parameters
    brir_directional_export = (dpg.get_value("fde_dir_brir_toggle"))
    brir_ts_export = (dpg.get_value("fde_ts_brir_toggle"))
    hesuvi_export = (dpg.get_value("fde_hesuvi_brir_toggle"))
    multichan_export = (dpg.get_value("fde_multi_chan_brir_toggle"))
    sofa_export = (dpg.get_value("fde_sofa_brir_toggle"))
    spat_res = dpg.get_value("fde_brir_spat_res")
    spat_res_int = CN.SPATIAL_RES_LIST.index(spat_res)
    output_path = dpg.get_value('selected_folder_base')
    hesuvi_path = dpg.get_value('output_folder_hesuvi')
    sofa_conv=dpg.get_value("sofa_exp_convention")
    resample_mode=CN.RESAMPLE_MODE_LIST[0]
    
    #grab parameters
    brir_dict_params=get_brir_dict()
    
    #calculate name

    #20250622: fix name related bug
    brir_name = calc_brir_set_name(full_name=False)
    

    """
    #Run BRIR integration
    """
    
    brir_gen, status = brir_generation.generate_integrated_brir(brir_name=brir_name, spatial_res=spat_res_int, report_progress=2, gui_logger=logz, brir_meta_dict=brir_dict_params)
    
    """
    #Run BRIR export
    """
   
    if brir_gen.size != 0 and status == 0:
    
        brir_export.export_brir(brir_arr=brir_gen, brir_name=brir_name, primary_path=output_path, hesuvi_path=hesuvi_path, gui_logger=logz, spatial_res=spat_res_int,
                             brir_dir_export=brir_directional_export, brir_ts_export=brir_ts_export, hesuvi_export=hesuvi_export, resample_mode=resample_mode,
                               sofa_export=sofa_export, multichan_export=multichan_export, brir_meta_dict=brir_dict_params, sofa_conv=sofa_conv)
        
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
    dpg.configure_item('fde_brir_tag',user_data=process_brirs_running)
    dpg.configure_item('fde_brir_tag',label="Process")
    #set stop thread flag flag
    stop_thread_flag = False
    #update user data
    dpg.configure_item('fde_progress_bar_brir',user_data=stop_thread_flag)
    save_settings(update_brir_pars=True)


def e_apo_apply_brir_params(sender=None, app_data=None):
    """ 
    GUI function to apply brir parameters, used for button press
    """

    qc_start_process_brirs()#this may trigger a cancel if already running
        
   
def qc_start_process_brirs(use_stored_brirs=False):
    """ 
    GUI function to start or stop processing of BRIRs thread
    """
    
    if sys.platform != "win32":
        hf.log_with_timestamp(
            "Unable to apply binaural simulation in Equalizer APO due to incompatible OS"
        )
        save_settings()
        return
    
    #thread bool
    process_brirs_running=dpg.get_item_user_data("e_apo_brir_tag")
    
    if process_brirs_running == False:#if not already running

        #set thread running flag
        process_brirs_running = True
        #update user data
        dpg.configure_item('e_apo_brir_tag',user_data=process_brirs_running)
        dpg.configure_item('e_apo_brir_tag',label="Cancel")
        
        #reset stop thread flag flag
        stop_thread_flag = False
        #update user data
        dpg.configure_item('e_apo_progress_bar_brir',user_data=stop_thread_flag)
        
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
    dpg.configure_item('e_apo_progress_bar_brir',user_data=stop_thread_flag)
    

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
    out_dataset_name = CN.FOLDER_BRIRS_LIVE 
    brir_name = calc_brir_set_name(full_name=False)
    brir_name_full = calc_brir_set_name(full_name=True)
    reduce_dataset = True
    output_path = dpg.get_value('e_apo_config_folder')
    resample_mode=CN.RESAMPLE_MODE_LIST[0]
    spat_res = dpg.get_value("e_apo_brir_spat_res")
    spat_res_int = CN.SPATIAL_RES_LIST.index(spat_res)
    
    #grab parameters
    brir_dict_params=get_brir_dict()
 
    log_string = 'Processing: ' + brir_name
    hf.log_with_timestamp(log_string, logz)
    

    #contains previously processed brirs
    brir_dict_list = dpg.get_item_user_data("e_apo_brir_conv") or []
    #flag to indicate stored directions should be used in output, cases where direction was missing and needs to be written before applying direction
    force_use_brir_dict = dpg.get_item_user_data("e_apo_curr_brir_set") or False
    
    # Decide which brir dict to use for output directions
    if use_stored_brirs == True or force_use_brir_dict == True:
        #grab desired directions from previously stored values, in case of direction change where brirs dont exist. Since directions were reverted temporarily, use stored copy
        brir_dict_out_config = dpg.get_item_user_data("e_apo_sel_brir_set") or {}
    else:
        brir_dict_out_config = get_brir_dict()#grab relevant config data from current gui elements
    
    
    """
    #Run BRIR integration
    """
    #only generate if prepop brirs not provided
    if not use_stored_brirs or not brir_dict_list:
        brir_gen, status = brir_generation.generate_integrated_brir(brir_name=out_dataset_name, spatial_res=spat_res_int, report_progress=1, gui_logger=logz, brir_meta_dict=brir_dict_params)
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
        #dpg.set_value("e_apo_brir_conv", False)
        hf.set_checkbox_and_sync_button(checkbox_tag="e_apo_brir_conv",button_tag="e_apo_brir_conv_btn",value=False)
        e_apo_config_acquire(estimate_gain=False,caller='apply_brir')
        #run export function
        brir_dict_list_new = brir_export.export_brir(brir_arr=brir_gen, brir_name=out_dataset_name, primary_path=output_path, 
                             brir_dir_export=brir_directional_export, brir_ts_export=brir_ts_export, hesuvi_export=hesuvi_export, resample_mode=resample_mode,
                            gui_logger=logz,  spatial_res=spat_res_int, sofa_export=sofa_export, reduce_dataset=reduce_dataset, brir_meta_dict=brir_dict_out_config,
                            use_stored_brirs=use_stored_brirs, brir_dict_list=brir_dict_list)
    
        #set progress to 100 as export is complete (assume E-APO export time is negligible)
        progress = 100/100
        hf.update_gui_progress(report_progress=1, progress=progress, message='Processed')
        #rewrite config file
        #dpg.set_value("e_apo_brir_conv", True)
        hf.set_checkbox_and_sync_button(checkbox_tag="e_apo_brir_conv",button_tag="e_apo_brir_conv_btn",value=True)
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
            #update gui elements to store current brir set text and applied brir set name
            dpg.set_value("e_apo_curr_brir_set", brir_name)
            dpg.set_value("e_apo_sel_brir_set", brir_name_full)           
            # store the timestamp of this run
            run_timestamp = datetime.now().isoformat()
            dpg.set_value("e_apo_sel_brir_set_ts", run_timestamp)
            
        #unmute before writing configs once more
        dpg.set_value("e_apo_gain_oa", gain_oa_selected)
        #dpg.set_value("e_apo_brir_conv", True)
        hf.set_checkbox_and_sync_button(checkbox_tag="e_apo_brir_conv",button_tag="e_apo_brir_conv_btn",value=True)

        #wait before updating config
        sleep(0.1)
        #update directions to previously stored desired values if flagged
        if use_stored_brirs == True or force_use_brir_dict == True:
            e_apo_update_direction(aquire_config=False, brir_dict_new=brir_dict_out_config)
        #Reset user data flag, dont use stored config dict for next output unless flagged again
        dpg.configure_item('e_apo_curr_brir_set',user_data=False)
        #rewrite config file, will also save settings
        e_apo_config_acquire(caller='apply_brir')
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
    dpg.configure_item('e_apo_brir_tag',user_data=process_brirs_running)
    dpg.configure_item('e_apo_brir_tag',label=CN.PROCESS_BUTTON_BRIR)
    #set stop thread flag flag
    stop_thread_flag = False
    #update user data
    dpg.configure_item('e_apo_progress_bar_brir',user_data=stop_thread_flag)      
    #update elevation list from saved spatial res
    e_apo_update_elev_list()
    if status == 0:
        #reset progress in case of changed settings
        e_apo_reset_progress()
    #plot to integrated analysis
    ia_new_plot()
  

def check_brir_dataset_exported():
    """
    Check whether the currently selected BRIR dataset
    has already been exported.

    Returns
    -------
    tuple[bool, str | None]
        (already_exported, matching_folder_name)
    """
    ash_folder = dpg.get_value("selected_folder_base")
    out_dir_brir_sets = pjoin(ash_folder, CN.PROJECT_FOLDER_BRIRS)

    brir_name = calc_brir_set_name(full_name=False)
  
    if not os.path.isdir(out_dir_brir_sets):
        return False, None

    for name in os.listdir(out_dir_brir_sets):
        if name == brir_name and os.path.isdir(pjoin(out_dir_brir_sets, name)):
            return True, name

    return False, None

def fde_reset_brir_progress():
    """
    GUI function to reset BRIR export progress bar.

    If the current BRIR dataset has already been exported,
    leave the progress bar at 100%.
    """
    # thread running flag
    process_brirs_running = dpg.get_item_user_data("fde_brir_tag")

    if process_brirs_running is False:
        already_exported, _ = check_brir_dataset_exported()

        if already_exported:
            # leave progress at 100% to indicate completed export
            dpg.set_value("fde_progress_bar_brir", 1.0)
            dpg.configure_item(
                "fde_progress_bar_brir",
                overlay="Dataset exported"
            )
        else:
            # reset progress bar
            dpg.set_value("fde_progress_bar_brir", 0.0)
            dpg.configure_item(
                "fde_progress_bar_brir",
                overlay=CN.PROGRESS_START_ALT
            )

def check_hpcf_headphone_exported(headphone=None):
    """
    Check whether the currently selected headphone has already been exported.

    Parameters
    ----------
    headphone : str | None
        Headphone name to check. If None, reads from GUI state.

    Returns
    -------
    bool
        True if the headphone has already been exported, False otherwise.
    """
    # Get headphone from GUI if not provided
    if headphone is None:
        gui_state = get_hpcf_dict()
        headphone = gui_state.get("headphone")
 
    if not headphone:
        return False

    # Get output directories
    primary_path = dpg.get_value("selected_folder_base")
    out_dirs = [
        pjoin(primary_path, CN.PROJECT_FOLDER_HPCFS, 'FIRs'),
        pjoin(primary_path, CN.PROJECT_FOLDER_HPCFS, 'FIRs_Stereo')
    ]

    exported_headphones = set()

    for out_dir in out_dirs:
        if not os.path.isdir(out_dir):
            continue
        # Walk all subdirectories
        for root, _, files in os.walk(out_dir):
            for f in files:
                if f.lower().endswith(".json"):
                    json_path = pjoin(root, f)
                    try:
                        with open(json_path, "r", encoding="utf-8") as jf:
                            data = json.load(jf)
                            hp = data.get("headphone")
                            if hp:
                                exported_headphones.add(hp)
                    except Exception:
                        continue  # skip invalid JSON

    return headphone in exported_headphones


def fde_reset_hpcf_progress():
    """
    GUI function to reset HPCF export progress bar.

    Leaves progress bar at 100% if the headphone is already exported.
    """

    if check_hpcf_headphone_exported():
        dpg.set_value("fde_progress_bar_hpcf", 1.0)
        dpg.configure_item(
            "fde_progress_bar_hpcf",
            overlay="Headphone exported"
        )
    else:
        dpg.set_value("fde_progress_bar_hpcf", 0.0)
        dpg.configure_item(
            "fde_progress_bar_hpcf",
            overlay=CN.PROGRESS_START_ALT
        )


def e_apo_reset_progress():
    """ 
    GUI function to reset progress bar
    """
    
    try:
        
        #reset brir progress if applicable
        process_brirs_running=dpg.get_item_user_data("e_apo_brir_tag")
        brir_conv_enabled=dpg.get_value('e_apo_brir_conv')
        #if not already running
        #thread bool
        if process_brirs_running == False:
            #check if saved brir set name is matching with currently selected params
            brir_name = calc_brir_set_name(full_name=True)
            sel_brir_set=dpg.get_value('e_apo_sel_brir_set')
            if brir_name != sel_brir_set:
                #reset progress bar
                dpg.set_value("e_apo_progress_bar_brir", 0)
                dpg.configure_item("e_apo_progress_bar_brir", overlay = CN.PROGRESS_START)
            elif brir_name == sel_brir_set and sel_brir_set !='' and brir_conv_enabled==True:
                dpg.set_value("e_apo_progress_bar_brir", 1)
                dpg.configure_item("e_apo_progress_bar_brir", overlay = CN.PROGRESS_FIN)
            else:
                #reset progress bar
                dpg.set_value("e_apo_progress_bar_brir", 0)
                dpg.configure_item("e_apo_progress_bar_brir", overlay = CN.PROGRESS_START)
    
        #reset hpcf progress if applicable
        filter_name = calc_hpcf_name()
        sel_hpcf = dpg.get_value('e_apo_sel_hpcf')
        hpcf_conv_enabled=dpg.get_value('e_apo_hpcf_conv')
        if filter_name != sel_hpcf:
            #reset progress bar
            dpg.set_value("e_apo_progress_bar_hpcf", 0)
            dpg.configure_item("e_apo_progress_bar_hpcf", overlay = CN.PROGRESS_START)
        elif filter_name == sel_hpcf and sel_hpcf !='' and hpcf_conv_enabled==True:
            dpg.set_value("e_apo_progress_bar_hpcf", 1)
            dpg.configure_item("e_apo_progress_bar_hpcf", overlay = CN.PROGRESS_FIN)
        else:
            #reset progress bar
            dpg.set_value("e_apo_progress_bar_hpcf", 0)
            dpg.configure_item("e_apo_progress_bar_hpcf", overlay = CN.PROGRESS_START)
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)



################################### Naming and parameter retrieval related
#

def calc_brir_set_name(full_name=True):
    """ 
    GUI function to calculate brir set name from currently selected parameters
    """
    brir_meta_dict=get_brir_dict()
    
    #common items
    hrtf_symmetry = brir_meta_dict.get("hrtf_symmetry")
    er_delay_time = brir_meta_dict.get("er_delay_time")
    hrtf_polarity = brir_meta_dict.get("hrtf_polarity")
    hrtf_df_cal_mode = brir_meta_dict.get("hrtf_df_cal_mode")
    
 
    room_target_name = brir_meta_dict.get("room_target")
    target_name_short = CN.ROOM_TARGETS_DICT[room_target_name]["short_name"]
    
    direct_gain_db = brir_meta_dict.get("direct_gain_db")
    ac_space_short = brir_meta_dict.get("ac_space_short")
    pinna_comp = brir_meta_dict.get("pinna_comp")
    sample_rate = brir_meta_dict.get("samp_freq_str")
    bit_depth = brir_meta_dict.get("bit_depth_str")
    brir_hrtf_short=brir_meta_dict.get('brir_hrtf_short')
    hrtf_direction_misalign_comp = brir_meta_dict.get("hrtf_direction_misalign_comp")
    crossover_f=brir_meta_dict.get('crossover_f')
    sub_response=brir_meta_dict.get('sub_response')
    sub_response_short=brir_meta_dict.get('sub_response_short')
    hp_rolloff_comp=brir_meta_dict.get('hp_rolloff_comp')
    fb_filtering=brir_meta_dict.get('fb_filtering')
    
    e_apo_brir_spat_res=brir_meta_dict.get('e_apo_brir_spat_res')
    fde_brir_spat_res=brir_meta_dict.get('fde_brir_spat_res')

    averaged_last_saved = get_avg_hrtf_timestamp()
    
 
    if full_name==True:
        brir_name = (brir_hrtf_short + ' '+ac_space_short + ' ' + str(direct_gain_db) + 'dB ' + target_name_short + ' ' + CN.HP_COMP_LIST_SHORT[pinna_comp] 
                     + ' ' + sample_rate + ' ' + bit_depth + ' ' + hrtf_symmetry + ' ' + str(er_delay_time) + ' ' + str(crossover_f) + ' ' + str(sub_response) 
                     + ' ' + str(hp_rolloff_comp) + ' ' + str(fb_filtering) + ' ' + hrtf_polarity + ' ' + averaged_last_saved + ' ' + hrtf_df_cal_mode 
                     + ' ' + hrtf_direction_misalign_comp + ' ' + e_apo_brir_spat_res)
    else:
        brir_name = (ac_space_short + ', ' + str(direct_gain_db) + 'dB, '  + brir_hrtf_short + ', ' + target_name_short + ', ' + CN.HP_COMP_LIST_SHORT[pinna_comp] + ', ' +   sub_response_short + '-' +str(crossover_f) + 'Hz'  )


    return brir_name

def calc_hpcf_name(full_name=True):
    """ 
    GUI function to calculate hpcf from currently selected parameters
    """
    headphone = dpg.get_value('hpcf_headphone')
    sample = dpg.get_value('hpcf_sample')
    sample_rate = dpg.get_value('wav_sample_rate')
    bit_depth = dpg.get_value('wav_bit_depth')
    if full_name==True:
        filter_name = headphone + ', ' + sample + ', ' + sample_rate + ', ' + bit_depth
    else:
        filter_name = headphone + ', ' + sample


    return filter_name


def get_brir_dict():
    """ 
    GUI function to get inputs relating to brirs and store in a dict, returns a dict
    """
            
    #brir related selections
    enable_brir_selected=dpg.get_value('e_apo_brir_conv')
    brir_set_folder=CN.FOLDER_BRIRS_LIVE
    brir_set_name=dpg.get_value('e_apo_sel_brir_set')
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
    
    #export brir related channel config selections
    fde_elev_fl_selected=dpg.get_value('fde_elev_angle_fl')
    fde_elev_fr_selected=dpg.get_value('fde_elev_angle_fr')
    fde_elev_c_selected=dpg.get_value('fde_elev_angle_c')
    fde_elev_sl_selected=dpg.get_value('fde_elev_angle_sl')
    fde_elev_sr_selected=dpg.get_value('fde_elev_angle_sr')
    fde_elev_rl_selected=dpg.get_value('fde_elev_angle_rl')
    fde_elev_rr_selected=dpg.get_value('fde_elev_angle_rr')
    fde_azim_fl_selected=dpg.get_value('fde_az_angle_fl')
    fde_azim_fr_selected=dpg.get_value('fde_az_angle_fr')
    fde_azim_c_selected=dpg.get_value('fde_az_angle_c')
    fde_azim_sl_selected=dpg.get_value('fde_az_angle_sl')
    fde_azim_sr_selected=dpg.get_value('fde_az_angle_sr')
    fde_azim_rl_selected=dpg.get_value('fde_az_angle_rl')
    fde_azim_rr_selected=dpg.get_value('fde_az_angle_rr')
    fde_gain_fl_selected=dpg.get_value('fde_gain_fl')
    fde_gain_fr_selected=dpg.get_value('fde_gain_fr')
    fde_gain_c_selected=dpg.get_value('fde_gain_c')
    fde_gain_sl_selected=dpg.get_value('fde_gain_sl')
    fde_gain_sr_selected=dpg.get_value('fde_gain_sr')
    fde_gain_rl_selected=dpg.get_value('fde_gain_rl')
    fde_gain_rr_selected=dpg.get_value('fde_gain_rr')
    multichan_mapping = dpg.get_value("mapping_16ch_wav")
    
    #hrtf type, dataset, hrtf name
    brir_hrtf_type_selected=dpg.get_value('brir_hrtf_type')
    brir_hrtf_dataset_selected=dpg.get_value('brir_hrtf_dataset')
    brir_hrtf_selected=dpg.get_value('brir_hrtf')
    brir_hrtf_short = hrir_processing.get_name_short(listener_type=brir_hrtf_type_selected, dataset_name=brir_hrtf_dataset_selected, name_gui=brir_hrtf_selected)
    
    
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
    target = dpg.get_value("room_target")
    room_target_int = CN.ROOM_TARGET_INDEX_MAP.get(target, -1)  # -1 or suitable default/error handling
    room_target = target  # for BRIR params
    
    direct_gain_db = dpg.get_value("direct_gain")
    direct_gain_db = round(direct_gain_db,1)#round to nearest .1 dB
    
    reverb_data=CN.reverb_data
    ac_space_gui = dpg.get_value("acoustic_space")
    ac_space_short = CN.extract_column(data=reverb_data, column='name_short', condition_key='name_gui', condition_value=ac_space_gui, return_all_matches=False)

    
    hp_type = dpg.get_value("brir_hp_type")
    pinna_comp_int = CN.HP_COMP_LIST.index(hp_type)
    pinna_comp = pinna_comp_int
    samp_freq_str = dpg.get_value('wav_sample_rate')
    samp_freq_int = CN.SAMPLE_RATE_DICT.get(samp_freq_str)
    bit_depth_str = dpg.get_value('wav_bit_depth')
    bit_depth = CN.BIT_DEPTH_DICT.get(bit_depth_str)
    
    #brir params

    
    #misc
    hrtf_symmetry = dpg.get_value('force_hrtf_symmetry')
    er_delay_time = dpg.get_value('er_delay_time')
    er_delay_time = round(er_delay_time,1)#round to nearest .1 dB
    reverb_tail_crop_db = dpg.get_value('reverb_tail_crop_db')
    reverb_tail_crop_db = round(reverb_tail_crop_db,1)#round to nearest .1 dB
    hrtf_polarity = dpg.get_value('hrtf_polarity_rev')
    hrtf_polarity = dpg.get_value('hrtf_polarity_rev')
    hrtf_direction_misalign_comp = dpg.get_value('hrtf_direction_misalign_comp')
    hrtf_df_cal_mode = dpg.get_value('hrtf_df_cal_mode')
    e_apo_brir_spat_res=dpg.get_value('e_apo_brir_spat_res')
    fde_brir_spat_res=dpg.get_value('fde_brir_spat_res')
    hrtf_low_freq_suppression=dpg.get_value('hrtf_low_freq_suppression')
    
    #low freq
    crossover_f_mode = dpg.get_value('crossover_f_mode')
    crossover_f = dpg.get_value('crossover_f')
    sub_response = dpg.get_value('sub_response')
    sub_response_int = CN.SUB_RESPONSE_LIST_GUI.index(sub_response)
    sub_response_short = CN.SUB_RESPONSE_LIST_SHORT[sub_response_int]
    hp_rolloff_comp = dpg.get_value('hp_rolloff_comp')
    fb_filtering = dpg.get_value('fb_filtering')

    

    brir_meta_dict = {
        
        'enable_conv': enable_brir_selected, 'brir_set_folder': brir_set_folder, 
        'brir_set_name': brir_set_name, 
        
        # (QC) channel Variables
        'mute_fl': mute_fl_selected, 'mute_fr': mute_fr_selected, 
        'mute_c': mute_c_selected, 'mute_sl': mute_sl_selected, 'mute_sr': mute_sr_selected, 
        'mute_rl': mute_rl_selected, 'mute_rr': mute_rr_selected, 'gain_oa': gain_oa_selected, 
        'gain_fl': gain_fl_selected, 'gain_fr': gain_fr_selected, 'gain_c': gain_c_selected, 
        'gain_sl': gain_sl_selected, 'gain_sr': gain_sr_selected, 'gain_rl': gain_rl_selected, 
        'gain_rr': gain_rr_selected, 
    
        # (QC) Individual Elevation Variables
        'elev_fl': elev_list[0], 'elev_fr': elev_list[1], 'elev_c': elev_list[2], 
        'elev_sl': elev_list[3], 'elev_sr': elev_list[4], 'elev_rl': elev_list[5], 
        'elev_rr': elev_list[6], 
    
        # (QC) Individual Azimuth Variables
        'azim_fl': azim_list[0], 'azim_fr': azim_list[1], 'azim_c': azim_list[2], 
        'azim_sl': azim_list[3], 'azim_sr': azim_list[4], 'azim_rl': azim_list[5], 
        'azim_rr': azim_list[6], 
    
        # Elevation and Azimuth Lists
        'elev_list': elev_list, 'azim_list': azim_list, 
    
        # HRTF selections
        'brir_hrtf_type': brir_hrtf_type_selected, 'brir_hrtf_dataset': brir_hrtf_dataset_selected, 
        'brir_hrtf': brir_hrtf_selected, 'brir_hrtf_short': brir_hrtf_short, 
    
        # export (h_) Elevation Variables
        'fde_elev_fl': fde_elev_fl_selected, 'fde_elev_fr': fde_elev_fr_selected, 
        'fde_elev_c': fde_elev_c_selected, 'fde_elev_sl': fde_elev_sl_selected, 
        'fde_elev_sr': fde_elev_sr_selected, 'fde_elev_rl': fde_elev_rl_selected, 
        'fde_elev_rr': fde_elev_rr_selected, 
    
        # export (fde_) Azimuth Variables
        'fde_azim_fl': fde_azim_fl_selected, 'fde_azim_fr': fde_azim_fr_selected, 
        'fde_azim_c': fde_azim_c_selected, 'fde_azim_sl': fde_azim_sl_selected, 
        'fde_azim_sr': fde_azim_sr_selected, 'fde_azim_rl': fde_azim_rl_selected, 
        'fde_azim_rr': fde_azim_rr_selected, 
        
        # export (fde_) gain Variables
        'fde_gain_fl': fde_gain_fl_selected, 'fde_gain_fr': fde_gain_fr_selected, 
        'fde_gain_c': fde_gain_c_selected, 'fde_gain_sl': fde_gain_sl_selected, 
        'fde_gain_sr': fde_gain_sr_selected, 'fde_gain_rl': fde_gain_rl_selected, 
        'fde_gain_rr': fde_gain_rr_selected, 
    
        # QC room and acoustic space selections
        'room_target': room_target, 'room_target_int': room_target_int, 'direct_gain_db': direct_gain_db, 
        'ac_space_short': ac_space_short,   'ac_space_gui': ac_space_gui,
        'pinna_comp': pinna_comp, 'samp_freq_int': samp_freq_int, 'samp_freq_str': samp_freq_str,   
        'bit_depth': bit_depth, 'bit_depth_str': bit_depth_str, 
 
        # Additional variables
        'hrtf_symmetry': hrtf_symmetry, 'er_delay_time': er_delay_time, 'reverb_tail_crop_db': reverb_tail_crop_db,
        'crossover_f_mode': crossover_f_mode, 'crossover_f': crossover_f, 'sub_response': sub_response, 'sub_response_short': sub_response_short, 'hp_rolloff_comp': hp_rolloff_comp,
        'fb_filtering': fb_filtering, 'hrtf_polarity': hrtf_polarity, 'multichan_mapping': multichan_mapping, 'hrtf_low_freq_suppression': hrtf_low_freq_suppression,
        'hrtf_direction_misalign_comp': hrtf_direction_misalign_comp,  'hrtf_df_cal_mode': hrtf_df_cal_mode,  'e_apo_brir_spat_res': e_apo_brir_spat_res,  'fde_brir_spat_res': fde_brir_spat_res
    }



    return brir_meta_dict


############### misc tools and settings
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
        dpg.set_value('output_folder_fde', ash_folder_selected)
        dpg.set_value('output_folder_fde_tooltip', ash_folder_selected)
        #hesuvi path
        if 'EqualizerAPO' in base_folder_selected:
            hesuvi_path_selected = pjoin(base_folder_selected,'HeSuVi')#stored outside of project folder (within hesuvi installation)
        else:
            hesuvi_path_selected = pjoin(base_folder_selected, CN.PROJECT_FOLDER,'HeSuVi')#stored within project folder
        dpg.set_value('output_folder_hesuvi', hesuvi_path_selected)
        dpg.set_value('output_folder_hesuvi_tooltip', hesuvi_path_selected)
        save_settings()
       
    



def remove_brirs(sender, app_data, user_data):
    """ 
    GUI function to delete generated BRIRs
    """
    logz=dpg.get_item_user_data("console_window")#contains logger
    base_folder_selected=dpg.get_value('selected_folder_base')
    brir_export.remove_brirs(base_folder_selected, gui_logger=logz)    
    base_folder_selected=dpg.get_value('e_apo_config_folder')
    brir_export.remove_brirs(base_folder_selected, gui_logger=logz)  
    #disable brir convolution
    dpg.set_value("e_apo_sel_brir_set", 'Deleted')
    #dpg.set_value("e_apo_brir_conv", False)
    hf.set_checkbox_and_sync_button(checkbox_tag="e_apo_brir_conv",button_tag="e_apo_brir_conv_btn",value=False)

    e_apo_toggle_brir_gui(app_data=False)
    
    dpg.configure_item("del_brirs_popup", show=False)
    
def remove_hpcfs(sender, app_data, user_data):
    """ 
    GUI function to remove generated HpCFs
    """
    gui_state = get_hpcf_dict()
    logz=dpg.get_item_user_data("console_window")#contains logger
    base_folder_selected=dpg.get_value('selected_folder_base')
    hpcf_functions.remove_hpcfs(base_folder_selected, gui_logger=logz) 
    base_folder_selected=dpg.get_value('e_apo_config_folder')
    hpcf_functions.remove_hpcfs(base_folder_selected, gui_logger=logz) 
    #disable hpcf convolution
    hf.set_checkbox_and_sync_button(checkbox_tag="e_apo_hpcf_conv",button_tag="e_apo_hpcf_conv_btn",value=False)
    dpg.set_value("e_apo_sel_hpcf", 'Deleted')
    e_apo_toggle_hpcf_gui(app_data=False)
    
    #revert to previous selections
    brand = gui_state["brand"]
    headphone = gui_state["headphone"]
    sample = gui_state["sample"]
    brands_list = gui_state["brands_list"]
    headphones_list = gui_state["headphones_list"]
    samples_list = gui_state["samples_list"]
    # update brand list
    dpg.configure_item('hpcf_brand', items=brands_list)
    dpg.set_value("hpcf_brand", brand)
    # update headphone list
    dpg.configure_item('hpcf_headphone', items=headphones_list)
    dpg.set_value("hpcf_headphone", headphone)
    # update sample list
    dpg.configure_item('hpcf_sample', items=samples_list)
    dpg.set_value("hpcf_sample", sample)
    
    #disable show history
    dpg.set_value("toggle_hpcf_history", False)
    show_hpcf_history(app_data=False)
    
    dpg.configure_item("del_hpcfs_popup", show=False)
    

    


################################### Equalizer APO configuration functions
##

def e_apo_toggle_brir_gui(sender=None, app_data=None, user_data=None):
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
    
    # ---- Non-Windows behaviour (Linux/macOS for now) ----
    if sys.platform != "win32":
        hf.log_with_timestamp("Unable to process binaural simulation in E-APO due to incompatible OS")
        save_settings()
        return  # IMPORTANT: stop here
    
    process_brirs_running=dpg.get_item_user_data("e_apo_brir_tag")
    if activate == False:#toggled off
        dpg.set_value("e_apo_curr_brir_set", '')
        #call main config writer function
        if aquire_config==True:#custom parameter will be none if called by gui
            e_apo_config_acquire(caller='apply_brir')
        
        if process_brirs_running == True:
            #stop processing if already processing brirs
            qc_stop_process_brirs()
        else:
            #reset progress
            dpg.set_value("e_apo_progress_bar_brir", 0)
            dpg.configure_item("e_apo_progress_bar_brir", overlay = CN.PROGRESS_START)
        
    else:#toggled on
        #check if saved brir set name is matching with currently selected params
        brir_name_full = calc_brir_set_name(full_name=True)
        brir_name = calc_brir_set_name(full_name=False)
        sel_brir_set=dpg.get_value('e_apo_sel_brir_set')
        #if matching and not forced to run, enable brir conv in config
        if brir_name_full == sel_brir_set and force_run_process==False:#custom parameter will be none if called by gui
            #dpg.set_value("e_apo_brir_conv", True)
            hf.set_checkbox_and_sync_button(checkbox_tag="e_apo_brir_conv",button_tag="e_apo_brir_conv_btn",value=True)

            dpg.set_value("e_apo_curr_brir_set", brir_name)
            dpg.set_value("e_apo_progress_bar_brir", 1)
            dpg.configure_item("e_apo_progress_bar_brir", overlay = CN.PROGRESS_FIN)
            e_apo_activate_direction(force_reset=True)#run in case direction not found in case of reduced dataset or outdated data
            if aquire_config==True:#custom parameter will be none if called by gui
                e_apo_config_acquire(caller='apply_brir')
        else:#else run brir processing from scratch
            if process_brirs_running == False:#only start if not already running
                qc_start_process_brirs(use_stored_brirs=use_stored_brirs)




def e_apo_auto_apply_hpcf(sender, app_data, user_data):
    """ 
    GUI function to toggle auto apply hpcf convolution
    """
    if app_data == True:
        e_apo_toggle_hpcf_gui(app_data=True)
    

def e_apo_toggle_hpcf_gui(sender=None, app_data=True, user_data=None):
    """ 
    GUI function to toggle hpcf convolution
    """
    aquire_config=True
    e_apo_toggle_hpcf_custom(activate=app_data, aquire_config=aquire_config)
        

            
def e_apo_toggle_hpcf_custom(activate=False, aquire_config=True):
    """ 
    GUI function to toggle hpcf convolution
    """
    force_output = False
    
    # ---- Non-Windows behaviour (Linux/macOS for now) ----
    if sys.platform != "win32":
        hf.log_with_timestamp("Unable to process filters in E-APO due to incompatible OS")
        save_settings()
        return  # IMPORTANT: stop here

    # ---------------------------
    # DISABLE HPCF CONVOLUTION
    # ---------------------------
    if activate == False:
        dpg.set_value("e_apo_curr_hpcf", '')

        if aquire_config == True or aquire_config is None:
            e_apo_config_acquire(caller='apply_hpcf')

        dpg.set_value("e_apo_progress_bar_hpcf", 0)
        dpg.configure_item("e_apo_progress_bar_hpcf", overlay=CN.PROGRESS_START)

    # ---------------------------
    # ENABLE HPCF CONVOLUTION
    # ---------------------------
    else:
        # current expected names
        hpcf_name_full = calc_hpcf_name(full_name=True)
        hpcf_name = calc_hpcf_name(full_name=False)

        # last used (saved) set
        sel_hpcf_set = dpg.get_value('e_apo_sel_hpcf')

        # read GUI headphone & sample
        gui_headphone = dpg.get_value('hpcf_headphone')
        gui_sample = dpg.get_value('hpcf_sample')

        # ---------------------------
        # READ HEADPHONE + SAMPLE FROM CONFIG FILE
        # ---------------------------
        base_folder_selected = dpg.get_value('e_apo_config_folder')
        output_config_path = pjoin(base_folder_selected, CN.PROJECT_FOLDER_CONFIGS)
        custom_file = pjoin(output_config_path, "ASH_Toolset_Config.txt")

        config_headphone = None
        config_sample = None

        if os.path.exists(custom_file):
            try:
                with open(custom_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()

                        if line.startswith("# Headphone:"):
                            config_headphone = line.split(":", 1)[1].strip()

                        elif line.startswith("# Sample:"):
                            config_sample = line.split(":", 1)[1].strip()

                        if config_headphone and config_sample:
                            break

            except Exception as ex:
                hf.log_with_timestamp(
                    f"Config read error (headphone/sample): {ex}", 
                    log_type=2, exception=ex
                )

        else:
            hf.log_with_timestamp("Config file missing → treat as outdated", log_type=0)

        # ---------------------------
        # DETERMINE WHETHER CONFIG IS OUTDATED
        # ---------------------------
        config_outdated = False

        if config_headphone is None or config_sample is None:
            config_outdated = True  # cannot trust old config
        else:
            if config_headphone != gui_headphone:
                config_outdated = True
            if config_sample != gui_sample:
                config_outdated = True
        if config_outdated:
            hf.log_with_timestamp("Config file outdated, processing HpCFs from scratch", log_type=0)

        # ---------------------------
        # FAST PATH: reuse existing HPCF
        # ---------------------------
        if hpcf_name_full == sel_hpcf_set and config_outdated == False:
            #dpg.set_value("e_apo_hpcf_conv", True)
            hf.set_checkbox_and_sync_button(checkbox_tag="e_apo_hpcf_conv",button_tag="e_apo_hpcf_conv_btn",value=True)
            dpg.set_value("e_apo_curr_hpcf", hpcf_name)

            dpg.set_value("e_apo_progress_bar_hpcf", 1)
            dpg.configure_item("e_apo_progress_bar_hpcf", overlay=CN.PROGRESS_FIN)

            if aquire_config == True or aquire_config is None:
                e_apo_config_acquire(caller='apply_hpcf')

        # ---------------------------
        # SLOW PATH: run full HPCF export
        # ---------------------------
        else:
            qc_process_hpcfs(force_output=force_output)    


    


def e_apo_config_acquire(estimate_gain=True, caller='apply_hpcf'):
    """ 
    GUI function to acquire lock on function to write updates to custom E-APO config
    """
    e_apo_conf_lock=dpg.get_item_user_data("qc_e_apo_conf_tab")#contains lock
    e_apo_conf_lock.acquire()
    e_apo_config_write(estimate_gain=estimate_gain, caller=caller)
    e_apo_conf_lock.release()
    
def e_apo_config_acquire_gui(sender=None, app_data=None):
    """ 
    GUI function to acquire lock on function to write updates to custom E-APO config - called from config changes e.g. azimuth change
    """
    e_apo_conf_lock=dpg.get_item_user_data("qc_e_apo_conf_tab")#contains lock
    estimate_gain=True
    e_apo_conf_lock.acquire()
    e_apo_config_write(estimate_gain=estimate_gain, caller='configure_brir')
    e_apo_conf_lock.release()
      

def e_apo_config_write(estimate_gain=True, caller='apply_hpcf'):
    """ 
    GUI function to write updates to custom E-APO config
    """
    

    try:
        
        # ---- Non-Windows behaviour (Linux/macOS for now) ----
        if sys.platform != "win32":
            hf.log_with_timestamp("Unable to write Equalizer APO configuration file due to incompatible OS")
            save_settings()
            return  # IMPORTANT: stop here

        logz=dpg.get_item_user_data("console_window")#contains logger
        base_folder_selected=dpg.get_value('e_apo_config_folder')
        
        #hpcf related selections
        enable_hpcf_selected=dpg.get_value('e_apo_hpcf_conv')

        #will either retrieve current selected headphone or saved headphone depending on who the caller is
        hpcf_dict = get_hpcf_dict(caller=caller)
        
        #brir related selections
        enable_brir_selected=dpg.get_value('e_apo_brir_conv')
        brir_set_folder=CN.FOLDER_BRIRS_LIVE
        gain_oa_selected=dpg.get_value('e_apo_gain_oa')
    
        brir_meta_dict = get_brir_dict()
      
        audio_channels=dpg.get_value('e_apo_audio_channels')
        upmix_method=dpg.get_value('e_apo_upmix_method')
        side_delay=dpg.get_value('e_apo_side_delay')
        rear_delay=dpg.get_value('e_apo_rear_delay')
        
        #get spatial resolution for this brir set
        spatial_res_sel = 0
        
        #run function to write custom config
        gain_conf = e_apo_config_creation.write_ash_e_apo_config(primary_path=base_folder_selected, hpcf_dict=hpcf_dict, brir_meta_dict=brir_meta_dict, audio_channels=audio_channels, gui_logger=logz, spatial_res=spatial_res_sel, upmix_method=upmix_method, side_delay=side_delay, rear_delay=rear_delay)
     
        #run function to load the custom config file in config.txt
        gain_oa_selected=dpg.get_value('e_apo_gain_oa')
        prevent_clipping = dpg.get_value('e_apo_prevent_clip')
        mute_fl=brir_meta_dict.get('mute_fl')
        mute_fr=brir_meta_dict.get('mute_fr')
        
        if enable_hpcf_selected == True or enable_brir_selected == True or gain_oa_selected == CN.EAPO_MUTE_GAIN or mute_fl == True or mute_fr == True:
            load_config = True
        else:
            load_config = False

        #also update estimated peak gain
        if estimate_gain == True:

            #peak gain
            est_pk_gain_2 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '2.0 Stereo', hpcf_dict=hpcf_dict, brir_meta_dict=brir_meta_dict, brir_set=brir_set_folder))
            dpg.set_value("e_apo_gain_peak_2_0", str(est_pk_gain_2))
            est_pk_gain_5 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '5.1 Surround', hpcf_dict=hpcf_dict, brir_meta_dict=brir_meta_dict, brir_set=brir_set_folder))
            dpg.set_value("e_apo_gain_peak_5_1", str(est_pk_gain_5))
            est_pk_gain_7 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '7.1 Surround', hpcf_dict=hpcf_dict, brir_meta_dict=brir_meta_dict, brir_set=brir_set_folder))
            dpg.set_value("e_apo_gain_peak_7_1", str(est_pk_gain_7))
            
            #clipping prevention
            one_chan_mute=False
            if (mute_fl == True or mute_fr == True):#if at least one channel is muted, dont adjust gain
                one_chan_mute=True
            #if clipping prevention enabled, grab 2.0 peak gain, calc new gain and rewrite the custom config with gain override
            if prevent_clipping != CN.AUTO_GAIN_METHODS[0] and load_config == True and one_chan_mute == False:
                if prevent_clipping == CN.AUTO_GAIN_METHODS[2] or prevent_clipping == CN.AUTO_GAIN_METHODS[3]:#low or mid frequencies
                    #peak gain estimation at low or mid frequencies
                    est_pk_gain_reference = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '2.0 Stereo', hpcf_dict=hpcf_dict, brir_meta_dict=brir_meta_dict, brir_set=brir_set_folder,freq_mode = prevent_clipping))
                else:
                    est_pk_gain_reference = est_pk_gain_2
                gain_adjustment = float(est_pk_gain_reference*-1)#change polarity and reduce slightly
                gain_adjustment = min(gain_adjustment, 40.0)#limit to max of 40db
                dpg.set_value("e_apo_gain_oa", gain_oa_selected+gain_adjustment)
                brir_meta_dict = get_brir_dict()
                #run function to write custom config
                gain_conf = e_apo_config_creation.write_ash_e_apo_config(primary_path=base_folder_selected, hpcf_dict=hpcf_dict, brir_meta_dict=brir_meta_dict, audio_channels=audio_channels, gui_logger=logz, spatial_res=spatial_res_sel, upmix_method=upmix_method, side_delay=side_delay, rear_delay=rear_delay)
                #run function to load the custom config file in config.txt
                #if true, edit config.txt to include the custom config
                e_apo_config_creation.include_ash_e_apo_config(primary_path=base_folder_selected, enabled=load_config)
         
                #peak gain
                est_pk_gain_2 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '2.0 Stereo', hpcf_dict=hpcf_dict, brir_meta_dict=brir_meta_dict, brir_set=brir_set_folder))
                dpg.set_value("e_apo_gain_peak_2_0", str(est_pk_gain_2))
                est_pk_gain_5 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '5.1 Surround', hpcf_dict=hpcf_dict, brir_meta_dict=brir_meta_dict, brir_set=brir_set_folder))
                dpg.set_value("e_apo_gain_peak_5_1", str(est_pk_gain_5))
                est_pk_gain_7 = (e_apo_config_creation.est_peak_gain_from_dir(primary_path=base_folder_selected, gain_config=gain_conf, channel_config = '7.1 Surround', hpcf_dict=hpcf_dict, brir_meta_dict=brir_meta_dict, brir_set=brir_set_folder))
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
    e_apo_config_acquire(caller='configure_brir')


def e_apo_update_elev_list():
    """ 
    GUI function to process updates in E-APO config section.
    This updates the elevation lists based on saved spatial resolution.
    """

    # --- Get saved spatial resolution ---
    loaded_values = load_settings()
    spat_res = loaded_values.get("e_apo_brir_spat_res")

    if spat_res not in CN.SPATIAL_RES_LIST:
        return  # safety guard

    spat_res_int = CN.SPATIAL_RES_LIST.index(spat_res)

    # --- Get new elevation list ---
    elev_list_new = CN.ELEV_ANGLES_WAV_ALL[spat_res_int]

    # --- Elevation combo item tags ---
    elev_items = [
        "e_apo_elev_angle_fl",
        "e_apo_elev_angle_fr",
        "e_apo_elev_angle_c",
        "e_apo_elev_angle_sl",
        "e_apo_elev_angle_sr",
        "e_apo_elev_angle_rl",
        "e_apo_elev_angle_rr",
    ]

    # --- Update combos and validate current values ---
    for item in elev_items:
        dpg.configure_item(item, items=elev_list_new)

        try:
            current_val = int(dpg.get_value(item))
        except (TypeError, ValueError):
            current_val = None
        

        # If current value is invalid, default to 0 (or first element)
        if current_val not in elev_list_new:
            fallback = 0 if 0 in elev_list_new else elev_list_new[0]
            dpg.set_value(item, fallback)


def e_apo_update_direction(aquire_config=False, brir_dict_new={}):
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
        
        #update diagram elements
      
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
            e_apo_config_acquire(caller='configure_brir')
            
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
    Used to manage state of elevation and azimuth gui elements, based on selections and currently available wav files
    prevents activating directions that don't exist yet in the outputs
    """
    
    try:    
        logz=dpg.get_item_user_data("console_window")#contains logger
        
        #get current selections
        base_folder_selected=dpg.get_value('e_apo_config_folder')
        channels_selected=dpg.get_value('e_apo_audio_channels')
        brir_set_folder=CN.FOLDER_BRIRS_LIVE
        brir_meta_dict=get_brir_dict()
        #print(brir_meta_dict)
        
        # Explicitly clear any previous BRIRs stored in GUI to avoid leaks
        dpg.configure_item('e_apo_sel_brir_set', user_data=None)
        #Get brir_meta_dict for desired directions, store in a gui element for later
        dpg.configure_item('e_apo_sel_brir_set',user_data=brir_meta_dict)
        
        #run function to check if all brirs currently exist (returns true if brirs are disabled)
        all_brirs_found = e_apo_config_creation.dataset_all_brirs_found(primary_path=base_folder_selected, brir_set=brir_set_folder, brir_meta_dict=brir_meta_dict, channel_config = channels_selected)
        
        #if some files are missing (due to reduced dataset size)
        if all_brirs_found == False:
            #this means we cannot use current selections
            try:
 
                #load previous settings
                loaded_values = load_settings()
                    
                #update directions to previously saved values
                dpg.set_value("e_apo_elev_angle_fl", loaded_values['e_apo_elev_angle_fl'])
                dpg.set_value("e_apo_elev_angle_fr", loaded_values['e_apo_elev_angle_fr'])
                dpg.set_value("e_apo_elev_angle_c", loaded_values['e_apo_elev_angle_c'])
                dpg.set_value("e_apo_elev_angle_sl", loaded_values['e_apo_elev_angle_sl'])
                dpg.set_value("e_apo_elev_angle_sr", loaded_values['e_apo_elev_angle_sr'])
                dpg.set_value("e_apo_elev_angle_rl", loaded_values['e_apo_elev_angle_rl'])
                dpg.set_value("e_apo_elev_angle_rr", loaded_values['e_apo_elev_angle_rr'])
                dpg.set_value("e_apo_az_angle_fl", loaded_values['e_apo_az_angle_fl'])
                dpg.set_value("e_apo_az_angle_fr", loaded_values['e_apo_az_angle_fr'])
                dpg.set_value("e_apo_az_angle_c", loaded_values['e_apo_az_angle_c'])
                dpg.set_value("e_apo_az_angle_sl", loaded_values['e_apo_az_angle_sl'])
                dpg.set_value("e_apo_az_angle_sr", loaded_values['e_apo_az_angle_sr'])
                dpg.set_value("e_apo_az_angle_rl", loaded_values['e_apo_az_angle_rl'])
                dpg.set_value("e_apo_az_angle_rr", loaded_values['e_apo_az_angle_rr'])
                
            except:
                pass
            
            log_string = 'Selected direction was not found'
            hf.log_with_timestamp(log_string, logz)
            
            #is there already a brir dataset in memory?
            brir_dict_list=dpg.get_item_user_data("e_apo_brir_conv")
            
            #reset progress and disable brir conv as not started yet
            if force_reset == True and not brir_dict_list:#only when triggered by apply button or toggle and no stored brir data
                dpg.set_value("e_apo_curr_brir_set", '')
                dpg.set_value("e_apo_progress_bar_brir", 0)
                dpg.configure_item("e_apo_progress_bar_brir", overlay = CN.PROGRESS_START)
                #dpg.set_value("e_apo_brir_conv", False)
                hf.set_checkbox_and_sync_button(checkbox_tag="e_apo_brir_conv",button_tag="e_apo_brir_conv_btn",value=False)

            
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
                dpg.configure_item('e_apo_curr_brir_set',user_data=True)
                log_string = 'Processing and exporting missing direction(s)'
                hf.log_with_timestamp(log_string, logz)
                #If list not populated, trigger new brir processing from scratch (likely restarted app)
                e_apo_toggle_brir_custom(activate=True, use_stored_brirs=False, force_run_process=True)#do not use button due to cancellation logic
                
        #use updated azimuths to update diagram
        e_apo_update_direction()
        

            
        #finally rewrite config file
        if aquire_config == True:#custom parameter will be none if called by gui
            e_apo_config_acquire(caller='configure_brir')
            
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
            dpg.configure_item("e_apo_elev_angle_c_tooltip", show=False)
            dpg.configure_item("e_apo_elev_angle_sl_tooltip", show=False)
            dpg.configure_item("e_apo_elev_angle_sr_tooltip", show=False)
            dpg.configure_item("e_apo_elev_angle_rl_tooltip", show=False)
            dpg.configure_item("e_apo_elev_angle_rr_tooltip", show=False)
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
            dpg.configure_item("e_apo_elev_angle_c_tooltip", show=True)
            dpg.configure_item("e_apo_elev_angle_sl_tooltip", show=True)
            dpg.configure_item("e_apo_elev_angle_sr_tooltip", show=True)
            dpg.configure_item("e_apo_elev_angle_rl_tooltip", show=True)
            dpg.configure_item("e_apo_elev_angle_rr_tooltip", show=True)
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
            dpg.configure_item("e_apo_elev_angle_c_tooltip", show=True)
            dpg.configure_item("e_apo_elev_angle_sl_tooltip", show=True)
            dpg.configure_item("e_apo_elev_angle_sr_tooltip", show=True)
            dpg.configure_item("e_apo_elev_angle_rl_tooltip", show=True)
            dpg.configure_item("e_apo_elev_angle_rr_tooltip", show=True)
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
            dpg.configure_item("e_apo_elev_angle_c_tooltip", show=False)
            dpg.configure_item("e_apo_elev_angle_sl_tooltip", show=False)
            dpg.configure_item("e_apo_elev_angle_sr_tooltip", show=False)
            dpg.configure_item("e_apo_elev_angle_rl_tooltip", show=False)
            dpg.configure_item("e_apo_elev_angle_rr_tooltip", show=False)
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

    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)
        
    #run activate direction in case some directions are missing
    brir_conv_activated=dpg.get_value('e_apo_brir_conv')
    if brir_conv_activated:
        e_apo_activate_direction()
    #finally rewrite config file
    if aquire_config == True:#custom parameter will be none if called by gui
        e_apo_config_acquire(caller='configure_brir')
        

 


###################################### GUI Functions - Additional tools
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
        url = CN.MAIN_APP_META_URL
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
        hf.log_with_timestamp(log_string, gui_logger, log_type=1)
        
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
    hpcf_db_dict = dpg.get_item_user_data("e_apo_sel_hpcf")
    conn = hpcf_db_dict["conn"]
    logz = dpg.get_item_user_data("console_window")

    # check only (no download)
    hpcf_functions.check_for_database_update(
        conn=conn,
        gui_logger=logz,
        download_if_update=False
    )
    
def download_latest_db(sender, app_data, user_data):
    """ 
    GUI function to download latest db
    """
    hpcf_db_dict = dpg.get_item_user_data("e_apo_sel_hpcf")
    conn = hpcf_db_dict["conn"]
    logz = dpg.get_item_user_data("console_window")

    # check + download if required
    hpcf_functions.check_for_database_update(
        conn=conn,
        gui_logger=logz,
        download_if_update=True
    )
    
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
    hpcf_db_dict=dpg.get_item_user_data("e_apo_sel_hpcf")#dict contains db connection object
    conn = hpcf_db_dict['conn']
    logz=dpg.get_item_user_data("console_window")#contains logger
    check_for_app_update(gui_logger=logz)
    hpcf_functions.check_for_database_update(conn=conn, gui_logger=logz)
    air_processing.acoustic_space_updates(download_updates=False, gui_logger=logz)
    hrir_processing.hrir_metadata_updates(download_updates=False, gui_logger=logz)

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
     
def get_app_dir():
    """
    Return the directory of the main script, or the EXE if frozen.
    Works correctly in Spyder, console, and PyInstaller EXE.
    """
    if getattr(sys, 'frozen', False):
        # Running as a bundled EXE
        return os.path.dirname(sys.executable)

    # Running as .py script
    # sys.argv[0] is the main script path (even in Spyder)
    script_path = os.path.abspath(sys.argv[0])
    return os.path.dirname(script_path)


def open_program_folder(sender=None, app_data=None, user_data=None):
    """
    Open the ASH Toolset folder or the log file inside it.
    """
    app_dir = get_app_dir()
    log_path = os.path.join(app_dir, "log.log")  # or ASH-Toolset.log

    # If log exists → open it
    if os.path.isfile(log_path):
        try:
            if sys.platform == "win32":
                os.startfile(log_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", log_path])
            else:
                subprocess.run(["xdg-open", log_path])
            return
        except Exception as e:
            print(f"Failed to open log: {e}")

    # Otherwise → open folder
    open_folder_in_explorer(app_dir)




################################### Acoustic space processing functions
#
#
 
def is_ir_processing_running():
    user_data = dpg.get_item_user_data("start_processing_btn")
    return user_data.get("ir_processing_running", False)

def get_ir_folders():
    """Returns a list of subdirectories in the IR user folder"""
    DATA_DIR_IR_USER = CN.DATA_DIR_IRS_USER

    if not os.path.isdir(DATA_DIR_IR_USER):
        return []

    return [
        f for f in os.listdir(DATA_DIR_IR_USER)
        if os.path.isdir(os.path.join(DATA_DIR_IR_USER, f))
    ]


def update_ir_folder_list():
    folders = get_ir_folders()
    processing = is_ir_processing_running()

    if not folders:
        placeholder = "<No folders found>"

        dpg.configure_item(
            "ir_folder_list",
            items=[placeholder],
            enabled=False
        )
        dpg.set_value("ir_folder_list", placeholder)

        if not processing:
            dpg.set_value("selected_folder_display", "")

    else:
        dpg.configure_item(
            "ir_folder_list",
            items=folders,
            enabled=True
        )

        dpg.set_value("ir_folder_list", folders[0])

        if not processing:
            dpg.set_value("selected_folder_display", folders[0])

def folder_selected_callback(sender=None, app_data=None, user_data=None):
    if is_ir_processing_running():
        return  # Ignore UI changes during processing

    # Get the currently selected folder from the list itself
    hf.update_gui_progress(report_progress=3, progress=0.0)  # reset progress
    current_folder = dpg.get_value("ir_folder_list")
    dpg.set_value("selected_folder_display", current_folder)

def as_launch_processing_thread():
    user_data = dpg.get_item_user_data("start_processing_btn")
    if user_data.get("ir_processing_running", False):
        hf.log_with_timestamp("Processing already in progress. Please wait.")
        return

    user_data["ir_processing_running"] = True  # Set flag to block new launches
    dpg.set_item_user_data("start_processing_btn", user_data)

    def wrapper():
        try:
            as_start_processing_callback()
        finally:
            user_data["ir_processing_running"] = False  # Reset when done
            dpg.set_item_user_data("start_processing_btn", user_data)
            # Safe, single refresh point
            folder_selected_callback()

    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()

def as_start_processing_callback():
    """ 
    GUI function to handle AIR processing and AIR to BRIR conversion.
    """
    process_all_folders = CN.BULK_AS_IMPORT
    logger_obj = dpg.get_item_user_data("import_console_window")
    cancel_event = dpg.get_item_user_data("cancel_processing_button")["cancel_event"]
    cancel_event.clear()  # Reset at start

    try:
        hf.update_gui_progress(report_progress=3, progress=0.0)  # Start

        # --------------------------------------------------
        # Resolve folders to process
        # --------------------------------------------------
        if process_all_folders:
            folder_list = dpg.get_item_configuration("ir_folder_list").get("items", [])
        else:
            selected = dpg.get_value("selected_folder_display")
            folder_list = [selected] if selected else []

        if not folder_list:
            hf.log_with_timestamp("No folder selected.", log_type=2, gui_logger=logger_obj)
            return

        # --------------------------------------------------
        # Read GUI values ONCE
        # --------------------------------------------------
        name = dpg.get_value("space_name")
        description = dpg.get_value("space_description")
        directions = dpg.get_value("unique_directions")
        noise_reduction_mode = dpg.get_value("noise_reduction_mode")
        pitch_high = dpg.get_value("pitch_range_high")
        pitch_low = -abs(pitch_high)
        reverb_tail_mode = dpg.get_value("reverb_tail_mode")
        pitch_shift_comp = dpg.get_value("pitch_shift_comp")
        alignment_freq = dpg.get_value("alignment_freq")
        rise_time = dpg.get_value("as_rise_time")
        as_subwoofer_mode = dpg.get_value("as_subwoofer_mode")
        binaural_meas_inputs = dpg.get_value("binaural_meas_inputs")
        correction_factor = round(dpg.get_value("asi_rm_cor_factor"), 2)
        __version__ = dpg.get_item_user_data("log_text")
        as_listener_type=dpg.get_value("as_listener")  
        brir_meta_dict=get_brir_dict()
        brir_hrtf_gui = brir_meta_dict.get('brir_hrtf')
        if as_subwoofer_mode:#enforce short windowed if low frequency mode
            reverb_tail_mode = 'Short Windowed'

        # --------------------------------------------------
        # MAIN LOOP (single folder or batch)
        # --------------------------------------------------
        for folder_index, selected_folder in enumerate(folder_list, start=1):

            if cancel_event.is_set():
                hf.log_with_timestamp(
                    "Process cancelled by user.",
                    log_type=1,
                    gui_logger=logger_obj
                )
                return

            hf.log_with_timestamp(
                f"Started processing ({folder_index}/{len(folder_list)}): {selected_folder}",
                gui_logger=logger_obj
            )

            as_name = selected_folder if name == '' else name
            name_formatted = as_name.replace(" ", "_")
            file_name = CN.AS_USER_PREFIX + name_formatted
            name_gui = as_name
            name_short = as_name
            name_label = name_formatted
            name_id = name_formatted
    
            # Step 1: Check/download and load HRIR dataset
            hf.log_with_timestamp("Step 1: Loading HRIR dataset...")
            # Map listener type to prebuilt dataset file + URLs
            listener_dataset_map = {
                CN.AS_LISTENER_TYPE_LIST[0]: ("hrir_dataset_comp_max_TU-FABIAN.npy", CN.TU_FAB_MAX_HRIR_URLS),
                CN.AS_LISTENER_TYPE_LIST[1]: ("hrir_dataset_comp_max_THK-KU-100.npy", CN.TU_KU_MAX_HRIR_URLS),
            }
            hrir_selected = None
            hrir_metadata = None  # for user selection
            listener_name = ""
            if as_listener_type in listener_dataset_map:
                listener_name=as_listener_type
                # built-in dataset: check/download and then load
                fname, url_list = listener_dataset_map[as_listener_type]
                npy_fname = pjoin(CN.DATA_DIR_INT, fname)
                status_code = 1  # assume failure until proven otherwise
                for url in url_list:
                    status_code = hf.check_and_download_file(npy_fname, url, download=True, gui_logger=logger_obj)
                    if status_code == 0:
                        break
                if status_code != 0:
                    hf.update_gui_progress(report_progress=3, progress=0.0)  # Reset on failure
                    hf.log_with_timestamp("Error: Failed to download or check HRIR dataset.", log_type=2, gui_logger=logger_obj)
                    return
                # LOAD THE DATASET INTO HRIR_SELECTED
                hrir_selected = hf.load_convert_npy_to_float64(npy_fname)
                hf.log_with_timestamp(f"Loaded HRIR dataset: {npy_fname}", gui_logger=logger_obj)
            elif as_listener_type == CN.AS_LISTENER_TYPE_LIST[2]:
                listener_name=brir_hrtf_gui
                # user-provided dataset
                #set low freq suppression to false temporarily
                hrtf_low_freq_suppression_prev=dpg.get_value('hrtf_low_freq_suppression')
                dpg.set_value('hrtf_low_freq_suppression',False)
                #load HRIR dataset
                hrir_selected, hrir_metadata = _load_hrir_selection(skip_sofa=False)
                #revert parameter
                dpg.set_value('hrtf_low_freq_suppression',hrtf_low_freq_suppression_prev)
                if hrir_selected is None:
                    hf.update_gui_progress(report_progress=3, progress=0.0)
                    hf.log_with_timestamp("Error: Failed to load user HRIR dataset.", log_type=2, gui_logger=logger_obj)
                    return
            else:
                hf.log_with_timestamp(f"Unsupported listener type: {as_listener_type}", log_type=2, gui_logger=logger_obj)
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
            air_dataset, status_code  = air_processing.prepare_air_dataset(
                ir_set=name_id, input_folder=selected_folder, gui_logger=logger_obj,
                desired_measurements=desired_measurements,
                pitch_range=pitch_range, tail_mode=reverb_tail_mode,
                cancel_event=cancel_event, report_progress=3, noise_reduction_mode=noise_reduction_mode, f_alignment = alignment_freq, 
                pitch_shift_comp=pitch_shift_comp,subwoofer_mode=as_subwoofer_mode, binaural_mode=binaural_meas_inputs, correction_factor=correction_factor
            )
    
            if status_code != 0:
                hf.update_gui_progress(report_progress=3, progress=0.0) # Reset on failure
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
            if binaural_meas_inputs:
                distr_mode=2#0=evenly distribute,1=round robin,2=random,3=one measurement per source
            else:
                distr_mode=0#0=evenly distribute,1=round robin,2=random,3=one measurement per source
            hf.log_with_timestamp("Step 3: Converting IRs to BRIRs...", gui_logger=logger_obj)
            brir_reverberation, status_code = air_processing.convert_airs_to_brirs(
                ir_set=name_id, air_dataset=air_dataset, gui_logger=logger_obj,
                 tail_mode=reverb_tail_mode,  correction_factor=correction_factor, as_listener_type=as_listener_type, hrir_dataset=hrir_selected,
                cancel_event=cancel_event, report_progress=3, rise_time=rise_time, subwoofer_mode=as_subwoofer_mode, binaural_mode=binaural_meas_inputs, distr_mode=distr_mode
            )
            if status_code != 0:
                hf.update_gui_progress(report_progress=3, progress=0.0) # Reset on failure
                hf.log_with_timestamp("Error: Failed to convert IRs to BRIRs.", log_type=2, gui_logger=logger_obj)
                return
            elif status_code == 2:
                hf.log_with_timestamp("BRIR conversion cancelled by user.", log_type=1, gui_logger=logger_obj)
                return
            # Insert checks at key points:
            if cancel_event.is_set():
                hf.log_with_timestamp("Process cancelled by user.", log_type=1, gui_logger=logger_obj)
                return
            hf.update_gui_progress(report_progress=3, progress=0.80)
     
            
       
                   
            #
            #save numpy array for later use in BRIR generation functions
            #
            air_processing.save_reverberation_dataset(brir_reverberation=brir_reverberation,ir_set=name_id,gui_logger=logger_obj)
       
    
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
                f"correction_factor={correction_factor}, "
                f"binaural_meas_inputs={binaural_meas_inputs}, "
                f"reverb_tail_mode={reverb_tail_mode}, " 
                f"listener_type={as_listener_type}, "
                f"listener_name={listener_name}"
            )
            description_full = notes if not description.strip() else f"{description}, {notes}"
            low_rt60 = "No" if reverb_tail_mode.lower().startswith("long") else "Yes"
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
                meas_rt60 = int(round(topt_ms))               # Regular rounding
                meas_rt60 = int(np.ceil(meas_rt60 / 50.0) * 50) # Round up to nearest 50
            except Exception as rt60_ex:
                hf.log_with_timestamp(f"RT60 estimation failed: {rt60_ex}", gui_logger=logger_obj, log_type=1)
                meas_rt60 = 600
    
            rows = [{
                "file_name": file_name,
                "name_gui": name_gui,
                "name_short": name_short,
                "label": name_label,
                "id": name_id,
                "meas_rt60": meas_rt60,
                "fade_start": 0,
                "low_rt60": low_rt60,
                "folder": folder,
                "collection_1":"User Imports",
                "collection_2":"",
                "version": "1.0.0",
                "f_crossover": 100,
                "order_crossover": 9,
                "description": description_full,
                "notes": notes,
                "source_dataset": selected_folder
            }]
            air_processing.write_as_metadata_csv(ir_set=name_id, data_rows=rows, sub_mode=as_subwoofer_mode, gui_logger=logger_obj)
            hf.update_gui_progress(report_progress=3, progress=1.0)
    
            hf.log_with_timestamp("Processing complete", gui_logger=logger_obj)
         
            #also save subwoofer metadata file if subwoofer mode is enabled
            if as_subwoofer_mode == True:
                file_name_sub=file_name
                name_gui_sub=name_gui
                name_short_sub=name_id
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
                air_processing.write_sub_metadata_csv(ir_set=name_id, data_rows=rows, gui_logger=logger_obj)
                
            time.sleep(0.1)
            update_as_table_from_csvs()
            
            
        

    except Exception as ex:
        hf.log_with_timestamp(f"Error during processing: {ex}", log_type=2, gui_logger=logger_obj)

    
    
    
    
    
    
    
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

    if not folder_path:
        hf.log_with_timestamp("No folder path provided", gui_logger)
        return

    # Check if the provided path is a valid directory
    if not os.path.isdir(folder_path):
        hf.log_with_timestamp(f"Folder does not exist: {folder_path}", gui_logger)
        return

    try:
        if sys.platform == "win32":
            os.startfile(folder_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", folder_path], check=True)
        else:
            subprocess.run(["xdg-open", folder_path], check=True)

        hf.log_with_timestamp(f"Opened folder: {folder_path}", gui_logger)

    except Exception as e:
        hf.log_with_timestamp(
            f"Failed to open folder: {folder_path} ({e})",
            gui_logger,
            exception=e
        )

def open_output_folder(sender=None, app_data=None, user_data=None):
    """
    GUI function to open the most relevant output folder in the OS file explorer.
    """

    logz = dpg.get_item_user_data("console_window")

    # Priority order
    ash_folder = dpg.get_value('output_folder_fde')        # Main exports
    base_folder = dpg.get_value('selected_folder_base')    # Fallback

    # Try main ASH exports first
    if ash_folder and os.path.isdir(ash_folder):
        open_folder_in_explorer(ash_folder, gui_logger=logz)
        return

    # Fallback to base folder
    if base_folder and os.path.isdir(base_folder):
        open_folder_in_explorer(base_folder, gui_logger=logz)
        return

    # Nothing valid
    hf.log_with_timestamp(
        "No valid output folder available to open",
        logz
    )

def open_eapo_output_folder(sender=None, app_data=None, user_data=None):
    """
    GUI function to open the most relevant output folder in the OS file explorer.
    """

    logz = dpg.get_item_user_data("console_window")

    # Priority order
    ash_folder = dpg.get_value('output_folder_e_apo')        # Main exports
    base_folder = dpg.get_value('e_apo_config_folder')    # Fallback

    # Try main ASH exports first
    if ash_folder and os.path.isdir(ash_folder):
        open_folder_in_explorer(ash_folder, gui_logger=logz)
        return

    # Fallback to base folder
    if base_folder and os.path.isdir(base_folder):
        open_folder_in_explorer(base_folder, gui_logger=logz)
        return

    # Nothing valid
    hf.log_with_timestamp(
        "No valid output folder available to open",
        logz
    )




def delete_selected_callback(): 
    """
    Deletes all selected rows and associated folders based on user_data of 'selected_ir_rows'.
    Safeguards prevent deletion of the root DATA_DIR_AS_USER folder.
    """
    logger_obj = dpg.get_item_user_data("import_console_window")
    selected_indices = dpg.get_item_user_data("selected_ir_rows")

    if not selected_indices:
        hf.log_with_timestamp("No rows selected to delete.", logger_obj)
        dpg.configure_item("del_processed_popup", show=False)
        return

    rows = dpg.get_item_children("processed_irs_table", slot=1)

    root_dir = os.path.abspath(CN.DATA_DIR_AS_USER)

    for selected_idx in sorted(selected_indices, reverse=True):
        if selected_idx < 0 or selected_idx >= len(rows):
            hf.log_with_timestamp(f"Row index {selected_idx} out of range.", logger_obj)
            continue

        row = rows[selected_idx]
        row_children = dpg.get_item_children(row, slot=1)

        if not row_children:
            hf.log_with_timestamp(
                f"Row structure unexpected at index {selected_idx}.", logger_obj
            )
            continue

        selectable = row_children[0]
        dataset = dpg.get_item_user_data(selectable)

        # ---- SAFETY CHECKS ----
        if not dataset:
            hf.log_with_timestamp(
                f"Invalid dataset for row {selected_idx}; skipping deletion.", logger_obj
            )
            continue

        ir_set_folder = os.path.abspath(pjoin(root_dir, dataset))

        # Prevent deleting root directory or anything resolving to it
        if ir_set_folder == root_dir:
            hf.log_with_timestamp(
                f"Refusing to delete root dataset directory: {root_dir}", logger_obj, log_type=2
            )
            continue

        if not ir_set_folder.startswith(root_dir + os.sep):
            hf.log_with_timestamp(
                f"Resolved path outside DATA_DIR_AS_USER: {ir_set_folder}", logger_obj, log_type=2
            )
            continue
        # -----------------------

        if os.path.isdir(ir_set_folder):
            try:
                shutil.rmtree(ir_set_folder)
                hf.log_with_timestamp(f"Deleted folder: {ir_set_folder}", logger_obj)
            except Exception as e:
                hf.log_with_timestamp(
                    f"Failed to delete folder {ir_set_folder}: {e}",
                    logger_obj,
                    log_type=2
                )
        else:
            hf.log_with_timestamp(
                f"Folder does not exist: {ir_set_folder}", logger_obj
            )

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
                                # NEW: skip legacy rows without the new schema
                                if "label" not in row:
                                    continue
                                
                                name = row.get("name_gui", "")
                                rt60 = row.get("meas_rt60", "")
                                description = row.get("description", "")
                                name_id = row.get("id", "")
                                matching_rows.append((name, rt60, description, name_id))
                        hf.log_with_timestamp(f"Loaded: {file_path}", logger_obj)
                    except Exception as e:
                        hf.log_with_timestamp(f"Failed to load {file_path}: {e}", logger_obj, log_type=2)
    
        # Clear existing rows in the table
        existing_rows = dpg.get_item_children("processed_irs_table", 1)
        if existing_rows:
            for row_id in existing_rows:
                dpg.delete_item(row_id)
    
        # Add new rows with selectable in the first column and dataset stored in user_data
        for idx, (name, rt60, description, name_id) in enumerate(matching_rows):
            with dpg.table_row(parent="processed_irs_table"):
                dpg.add_selectable(label=name,callback=on_ir_row_selected,span_columns=True,user_data=name_id)
                dpg.add_text(str(rt60))
                dpg.add_text(description,wrap=1050)
    
        # Reset multi-selection tracking
        dpg.set_item_user_data("selected_ir_rows", [])
    
        hf.log_with_timestamp(f"Table updated with {len(matching_rows)} entries.", logger_obj)
        
        #Update listboxes by refreshing AS constants
        CN.refresh_acoustic_space_metadata()
        #dpg.configure_item("acoustic_space", items=CN.AC_SPACE_LIST_GUI)
        update_ac_space_display()
        
        CN.refresh_sub_responses()#also update sub responses in case sub mode was enabled
        dpg.configure_item("sub_response", items=CN.SUB_RESPONSE_LIST_GUI)
        
    except Exception as e:
        hf.log_with_timestamp(f"Error: {e}", log_type=2, exception=e)






######################################### Room target processing functions
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
        dpg.configure_item("room_target", items=CN.ROOM_TARGET_KEYS)

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
       


############################################ HRTF favourites
#
       


    
def add_hrtf_favourite_qc_callback(sender, app_data):
    """ 
    GUI function to add hrtf to favourites list
    """        
    update_hrtf_favourite( 'add')


    
def remove_hrtf_favourite_qc_callback(sender, app_data):
    """ 
    GUI function to add hrtf to favourites list
    """        
    update_hrtf_favourite( 'remove')
   
  
def update_hrtf_favourite( action): 
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
        brir_meta_dict = get_brir_dict()
        brir_hrtf_type = brir_meta_dict.get('brir_hrtf_type')
        brir_hrtf_dataset = brir_meta_dict.get('brir_hrtf_dataset')
        brir_hrtf = brir_meta_dict.get('brir_hrtf')
        brir_hrtf_short = brir_meta_dict.get('brir_hrtf_short')

        
        if brir_hrtf_type in CN.HRTF_TYPE_LIST and brir_hrtf_short != CN.HRTF_USER_SOFA_DEFAULT:
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
            if brir_meta_dict.get('brir_hrtf_type') == 'Favourites':
                update_hrtf_list(sender=None, app_data=brir_meta_dict.get('brir_hrtf_dataset'))
        
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
            dpg.configure_item("hrtf_average_fav_load_ind", show=True)

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
            # Build list of metadata dictionaries for load_hrirs_list
            hrtf_dict_list = [
                {
                    "hrtf_type": "Favourites",
                    "hrtf_dataset": CN.HRTF_DATASET_FAVOURITES_DEFAULT,
                    "hrtf_gui": fav_name,
                    "hrtf_short": fav_name
                }
                for fav_name in filtered_fav_list
            ]
            # Get currently selected BRIR info
            brir_meta_dict = get_brir_dict()
            hrtf_low_freq_suppression = brir_meta_dict.get('hrtf_low_freq_suppression')
            
            hrir_list_favs_loaded, status, hrir_metadata_list = hrir_processing.load_hrirs_list(hrtf_dict_list=hrtf_dict_list,gui_logger=logz,apply_lf_suppression=hrtf_low_freq_suppression)

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
            output_path = dpg.get_value('selected_folder_base')
            samp_freq_str = dpg.get_value('wav_sample_rate')
            samp_freq_int = CN.SAMPLE_RATE_DICT.get(samp_freq_str)

            brir_export.export_sofa_ir(
                primary_path=output_path,
                ir_arr=hrir_avg,
                ir_set_name=CN.HRTF_AVERAGED_NAME_FILE,
                samp_freq=samp_freq_int,
                gui_logger=logz
            )

            # --- 5. Write metadata file ---
            try:
                metadata = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "included_hrtf_sets": filtered_fav_list,
                    "sample_rate": samp_freq_int,
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
            dpg.set_value("brir_hrtf_type", 'Favourites')
            update_hrtf_dataset_list(sender=None, app_data='Favourites')
            dpg.set_value("brir_hrtf_dataset", CN.HRTF_DATASET_FAVOURITES_DEFAULT)
            update_hrtf_list(sender=None, app_data=CN.HRTF_DATASET_FAVOURITES_DEFAULT)
            hf.log_with_timestamp("HRTF favourites list updated.", logz)
            e_apo_reset_progress()

        except Exception as e:
            hf.log_with_timestamp(f"Error in HRTF averaging callback: {e}", logz)

        finally:
            dpg.configure_item("hrtf_average_fav_load_ind", show=False)
            dpg.configure_item("hrtf_average_fav_load_ind", show=False)
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
  

    

################################################### Presets
# 
    

# --- Helper functions ---
def get_selected_preset():
    """Return currently selected preset from listbox, or None."""
    selection = dpg.get_value("preset_list")
    return selection if selection else None


    
def refresh_preset_list(): 
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
    dpg.configure_item("preset_list", items=presets)

# --- Load callback ---
def load_selected_preset_callback(sender, app_data):
    """Load the selected preset or apply default/current parameters."""
    logz = dpg.get_item_user_data("console_window")
    preset_name = get_selected_preset()
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
            e_apo_curr_brir_set_prev = dpg.get_value('e_apo_curr_brir_set')
            e_apo_sel_brir_set_prev = dpg.get_value('e_apo_sel_brir_set')
            e_apo_curr_hpcf_prev = dpg.get_value('e_apo_curr_hpcf')
            e_apo_sel_hpcf_prev = dpg.get_value('e_apo_sel_hpcf')
            
            #load settings AND apply to gui elements
            load_settings( preset_name=preset_name,  set_gui_values=True, logz=logz)
            hf.log_with_timestamp(f"Loaded: {preset_name}")
            
            #set current e apo selection to previous one since new params havent been applied yet
            dpg.set_value("e_apo_curr_hpcf", e_apo_curr_hpcf_prev)
            dpg.set_value("e_apo_sel_hpcf", e_apo_sel_hpcf_prev) 
            dpg.set_value("e_apo_curr_brir_set", e_apo_curr_brir_set_prev)
            dpg.set_value("e_apo_sel_brir_set", e_apo_sel_brir_set_prev)
            
            enable_hpcf_selected=dpg.get_value('e_apo_hpcf_conv')
            enable_brir_selected=dpg.get_value('e_apo_brir_conv')
    
            #Apply parameters if enabled 
            #trigger activation of e apo config if enabled
            e_apo_toggle_hpcf_custom(activate=enable_hpcf_selected, aquire_config=False)
            e_apo_toggle_brir_custom(activate=enable_brir_selected, aquire_config=False)
            #finally acquire config once
            e_apo_config_acquire()
            
            #disable show history
            dpg.set_value("toggle_hpcf_history", False)
            
            
        except Exception as e:
            hf.log_with_timestamp(f"Failed to load preset '{preset_name}': {e}")


def revert_to_saved_params(sender, app_data):
    """Load primary settings file to revert parameters to saved parameters"""
    
    logz = dpg.get_item_user_data("console_window")
    
    #load settings AND apply to gui elements. This will update all filtered lists and values
    load_settings(set_gui_values=True, logz=logz)
    
    #trigger activation of e apo config if enabled
    hpcf_is_active=dpg.get_value('e_apo_hpcf_conv')
    brir_is_active=dpg.get_value('e_apo_brir_conv')
    e_apo_toggle_hpcf_custom(activate=hpcf_is_active, aquire_config=False)
    e_apo_toggle_brir_custom(activate=brir_is_active, aquire_config=False)
    #finally acquire config once
    e_apo_config_acquire()
        
    hf.log_with_timestamp("Reverted to previously applied parameters")
    

# --- rename callback ---
def rename_selected_preset_callback(sender, app_data):
    """Rename the selected preset file."""
    logz = dpg.get_item_user_data("console_window")
    preset_name = get_selected_preset()
    
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
        new_preset_name = dpg.get_value("rename_selected_preset_text").strip()

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
        refresh_preset_list()

    except Exception as e:
        hf.log_with_timestamp(f"Failed to rename preset '{preset_name}': {e}", logz)


# --- Save callback ---
def save_preset_callback(sender, app_data):
    """Save current parameters as a new preset file."""
    logz = dpg.get_item_user_data("console_window")
    enable_hpcf_selected=dpg.get_value('e_apo_hpcf_conv')
    enable_brir_selected=dpg.get_value('e_apo_brir_conv')
    hpcf_name = calc_hpcf_name(full_name=False)
    brir_name = calc_brir_set_name(full_name=False)
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
        threading.Timer(CN._SAVE_DEBOUNCE_TIME + 0.2, refresh_preset_list).start()
        hf.log_with_timestamp(f"Saved: {preset_name}", gui_logger=logz)
    except Exception as e:
        hf.log_with_timestamp(f"Failed to save preset '{preset_name}': {e}", gui_logger=logz)

# --- Delete callback ---
def delete_selected_preset_callback(sender, app_data):
    """Delete the selected preset file from disk."""
    logz = dpg.get_item_user_data("console_window")
    preset_name = get_selected_preset()

    if not preset_name:
        hf.log_with_timestamp("No preset selected to delete.", gui_logger=logz)
        dpg.configure_item("del_preset_popup", show=False)
        return

    if preset_name in ["Current Parameters", "Default Settings"]:
        hf.log_with_timestamp(f"Cannot delete special preset: {preset_name}", gui_logger=logz)
        dpg.configure_item("del_preset_popup", show=False)
        return

    preset_file = os.path.join(CN.SETTINGS_DIR, f"{preset_name}.ini")

    if os.path.exists(preset_file):
        try:
            os.remove(preset_file)
            refresh_preset_list()
            hf.log_with_timestamp(f"Deleted preset: {preset_name}", gui_logger=logz)
        except Exception as e:
            hf.log_with_timestamp(
                f"Failed to delete preset '{preset_name}': {e}", gui_logger=logz
            )

    # --- Also delete the dataset folder if it exists ---
    ash_folder = dpg.get_value("selected_folder_base")
    brir_name = calc_brir_set_name(full_name=False)
    out_dir_brir_sets = pjoin(ash_folder, CN.PROJECT_FOLDER_BRIRS, brir_name)

    if os.path.isdir(out_dir_brir_sets):
        try:
            shutil.rmtree(out_dir_brir_sets)
            hf.log_with_timestamp(
                f"Deleted BRIR dataset folder: {brir_name}", gui_logger=logz
            )
        except Exception as e:
            hf.log_with_timestamp(
                f"Failed to delete BRIR dataset folder '{brir_name}': {e}",
                gui_logger=logz
            )

    dpg.configure_item("del_preset_popup", show=False)
    

def sort_reference_as_table(sender=None, app_data=None, user_data=None):
    """
    Sort Acoustic Spaces reference table by ID, Name, or RT60.

    app_data should be one of:
        'ID'
        'Name'
        'RT60'
    """

    try:
        table_tag = "reference_as_table"

        # Determine sort order (indices only — do NOT reorder constants)
        if app_data == "Name":
            sorted_indices = sorted(
                range(len(CN.AC_SPACE_LIST_LABEL)),
                key=lambda i: CN.AC_SPACE_LIST_LABEL[i].lower()
            )

        elif app_data == "Reverberation Time":
            sorted_indices = sorted(
                range(len(CN.AC_SPACE_LIST_MEAS_R60)),
                key=lambda i: CN.AC_SPACE_LIST_MEAS_R60[i]
            )

        else:  # Default: ID
            sorted_indices = sorted(
                range(len(CN.AC_SPACE_LIST_ID)),
                key=lambda i: CN.AC_SPACE_LIST_ID[i].lower()
            )

        # Clear existing rows
        existing_rows = dpg.get_item_children(table_tag, 1)
        if existing_rows:
            for row_id in existing_rows:
                dpg.delete_item(row_id)

        # Rebuild table rows in sorted order
        for i in sorted_indices:
            with dpg.table_row(parent=table_tag):
                dpg.add_text(CN.AC_SPACE_LIST_ID[i])
                dpg.add_text(CN.AC_SPACE_LIST_LABEL[i])
                dpg.add_text(str(CN.AC_SPACE_LIST_MEAS_R60[i]))
                dpg.add_text(CN.AC_SPACE_LIST_DESCR[i], wrap=0)
                dpg.add_text(CN.AC_SPACE_LIST_DATASET[i], wrap=0)

    except Exception as e:
        hf.log_with_timestamp(
            f"Failed to sort Acoustic Spaces table: {e}",
            log_type=2,
            exception=e
        )
  
################### Custom azimuth controls        
        
def get_channels_for_azimuth(az): 
    """Return a list of channels assigned to this azimuth."""
    channels = []
    for ch, az_list in CN.AZ_LOOKUP.items():
        if az in az_list:
            channels.append(ch.upper())
    return channels


def update_chan_az_status_text(sender=None, app_data=None, user_data=None):
    """Update the text widget showing current channels for the selected azimuth."""
    az = int(dpg.get_value("custom_az_angle"))
    channels = get_channels_for_azimuth(az)
    text = ", ".join(channels) if channels else "(none)"
    dpg.set_value("custom_az_status", f"Channels assigned to azimuth {az}°:  {text}")


def add_mapping_callback(sender, app_data, user_data):
    """Add the selected azimuth to the selected channel."""
    logz = dpg.get_item_user_data("console_window")
    az = int(dpg.get_value("custom_az_angle"))
    ch = dpg.get_value("custom_az_channel").lower()
    if az not in CN.AZ_LOOKUP[ch]:
        CN.AZ_LOOKUP[ch].append(az)
        CN.AZ_LOOKUP[ch].sort()  # keep azimuths in order
        # --- Log the addition ---
        hf.log_with_timestamp(f"Added azimuth {az} to channel {ch.upper()}", gui_logger=logz)
    else:
        hf.log_with_timestamp(f"Azimuth {az} already assigned to channel {ch.upper()}", gui_logger=logz)

    update_chan_az_status_text()
    save_azimuth_lookup_to_csv()  # <-- save immediately


def remove_mapping_callback(sender, app_data, user_data):
    """Remove the selected azimuth from the selected channel."""
    logz = dpg.get_item_user_data("console_window")
    az = int(dpg.get_value("custom_az_angle"))
    ch = dpg.get_value("custom_az_channel").lower()
    if az in CN.AZ_LOOKUP[ch]:
        CN.AZ_LOOKUP[ch].remove(az)
        # --- Log the removal ---
        hf.log_with_timestamp(f"Removed azimuth {az} from channel {ch.upper()}", gui_logger=logz)
    else:
        hf.log_with_timestamp(f"Azimuth {az} not assigned to channel {ch.upper()}", gui_logger=logz)

    update_chan_az_status_text()
    save_azimuth_lookup_to_csv()  # <-- save immediately
    
def save_azimuth_lookup_to_csv():
    """Save CN.AZ_LOOKUP to the CSV file in the standard format."""
    try:
        # Ensure lowercase header names
        channels_lower = [ch.lower() for ch in CN.CHANNELS_ALL]

        with open(CN.AZIMUTH_LOOKUP_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["azimuth"] + channels_lower)
            writer.writeheader()

            for az in CN.AZIMUTH_ANGLES_ALL:
                row = {"azimuth": az}
                for ch in channels_lower:  # use lowercase for AZ_LOOKUP keys
                    row[ch] = "yes" if az in CN.AZ_LOOKUP[ch] else ""
                writer.writerow(row)

        hf.log_with_timestamp("Azimuth mapping table saved successfully.")

    except Exception as e:
        hf.log_with_timestamp(
            f"Failed to save azimuth mapping table: {e}",
            log_type=2,
            exception=e
        )
        
  
