# -*- coding: utf-8 -*-


"""
Main routine of ASH-Toolset.

Created on Sun Apr 28 11:25:06 2024
@author: Shanon Pearce
License: (see /LICENSE)
"""


import logging.config
from os.path import join as pjoin, normpath
from pathlib import Path
from ash_toolset import constants as CN
from ash_toolset import hpcf_functions
from ash_toolset import helper_functions as hf
from ash_toolset import hrir_processing
from ash_toolset import callbacks as cb
from ash_toolset import e_apo_config_creation
import dearpygui.dearpygui as dpg
import dearpygui_extend as dpge
from dearpygui_ext import logger
import numpy as np
import winreg as wrg
import ctypes
import math
import threading
import sys
import os
import requests
import re



def main():


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
    logging.getLogger('numba').setLevel(logging.WARNING)

    
    #hide splash screen
    try:
        import pyi_splash
        pyi_splash.update_text('UI Loaded ...')
        pyi_splash.close()
        logging.info('Splash screen closed.')
    except:
        pass

    
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    
    
        


    
    #
    ## GUI Related Functions 
    #



    def get_primary_paths(logz=None, create_dirs=True):
        """
        Detects the primary EqualizerAPO path and ASH Toolset project folder.
    
        Returns:
            primary_path (str): path to EqualizerAPO 'config' folder (registry or fallback)
            primary_ash_path (str): path to project folder inside primary_path
            e_apo_path (str): raw EqualizerAPO install path (registry or fallback)
    
        Args:
            logz: log object for hf.log_with_timestamp
            create_dirs (bool): whether to create primary_path and primary_ash_path if missing
        """
        # Fallback install folder
        fallback_root = r"C:\Program Files\EqualizerAPO"
    
        # Attempt registry read
        try:
            key = wrg.OpenKey(wrg.HKEY_LOCAL_MACHINE, r"Software\EqualizerAPO")
            value = wrg.QueryValueEx(key, "InstallPath")[0]
            e_apo_path = value
            if not os.path.isdir(e_apo_path):
                e_apo_path = fallback_root
                hf.log_with_timestamp(f"Registry path does not exist. Using fallback: {fallback_root}", logz)
        except Exception:
            e_apo_path = fallback_root
            hf.log_with_timestamp(f"EqualizerAPO not found in registry. Using fallback: {fallback_root}", logz)
    
        # Normalize path
        e_apo_path = normpath(e_apo_path)
        primary_path = normpath(pjoin(e_apo_path, "config"))
        primary_ash_path = normpath(pjoin(primary_path, getattr(CN, "PROJECT_FOLDER", "")))
    
        # Optionally create folders if missing
        if create_dirs:
            for path in (primary_path, primary_ash_path):
                if not os.path.isdir(path):
                    try:
                        os.makedirs(path, exist_ok=True)
                        hf.log_with_timestamp(f"Created missing folder: {path}", logz)
                    except Exception as e:
                        hf.log_with_timestamp(f"Failed to create folder '{path}': {e}", logz)
    
        return primary_path, primary_ash_path, e_apo_path


    def _hsv_to_rgb(h, s, v):
        """ 
        GUI function to convert hsv to rgb. HSV stands for hue, saturation, and value
        """
        if s == 0.0: return (v, v, v)
        i = int(h*6.) # XXX assume int() truncates!
        f = (h*6.)-i; p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
        if i == 0: return (255*v, 255*t, 255*p)
        if i == 1: return (255*q, 255*v, 255*p)
        if i == 2: return (255*p, 255*v, 255*t)
        if i == 3: return (255*p, 255*q, 255*v)
        if i == 4: return (255*t, 255*p, 255*v)
        if i == 5: return (255*v, 255*p, 255*q)
               
       
        
   
            
    def check_and_update_github_user_guide(local_version, local_path, github_raw_url):
        """
        Asynchronously checks GitHub for a newer version of user_guide.txt.
        If a newer version exists, downloads and replaces the local file.
        """
        try:
            response = requests.get(github_raw_url, timeout=5)
            if response.status_code == 200:
                remote_text = response.text
                version_pattern = re.compile(r"^\s*#version\s*[: ]\s*([\w\.\-]+)", re.IGNORECASE | re.MULTILINE)
                match = version_pattern.search(remote_text)
                if match:
                    remote_version = match.group(1)
                    if remote_version != local_version:
                        hf.log_with_timestamp(
                            f"[User Guide] Newer version detected on GitHub: {remote_version} (local: {local_version})"
                        )
                        # Download and replace
                        hf.download_file(github_raw_url, local_path, overwrite=True)
                        hf.log_with_timestamp("[User Guide] Updated local user_guide.txt to latest version.")
                    else:
                        hf.log_with_timestamp(f"[User Guide] Local version is up-to-date: {local_version}")
                else:
                    hf.log_with_timestamp("[User Guide] Remote user guide has no #VERSION tag")
            else:
                hf.log_with_timestamp(f"[User Guide] Failed to fetch remote user guide: HTTP {response.status_code}")
        except Exception as e:
            hf.log_with_timestamp(f"[User Guide] Error checking/updating GitHub version: {e}")
    
    
    def load_user_guide_into_gui():
        """
        Loads user_guide.txt and populates GUI collapsing headers.
        Also asynchronously checks GitHub for updates and replaces the local file if needed.
        """
        guide_path = pjoin(CN.DOCS_DIR_GUIDE, "user_guide.txt")
        local_version = None
    
        try:
            if not os.path.isfile(guide_path):
                hf.log_with_timestamp(f"[User Guide] File not found: {guide_path}")
                return
    
            # Read file safely
            with open(guide_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
    
            # --- Parse sections and version ---
            version_pattern = re.compile(r"^\s*#version\s*[: ]\s*([\w\.\-]+)", re.IGNORECASE)
            sections = {}
            current_sec = None
            buffer = []
    
            for line in lines:
                stripped = line.strip()
                m = version_pattern.match(stripped)
                if m:
                    local_version = m.group(1)
                    continue
    
                if stripped.startswith("[") and stripped.endswith("]"):
                    if current_sec is not None:
                        sections[current_sec] = buffer[:]
                    current_sec = stripped[1:-1]
                    buffer = []
                    continue
    
                if current_sec is not None:
                    buffer.append(line.rstrip("\n"))
    
            if current_sec is not None:
                sections[current_sec] = buffer[:]
    
            if local_version:
                hf.log_with_timestamp(f"[User Guide] Local version detected: {local_version}")
            else:
                hf.log_with_timestamp("[User Guide] No version tag found in user guide.")
    
            # --- Populate GUI as before ---
            header_map = {
                "Audio Processing Overview": "ug_ap_overview",
                "GUI Overview": "ug_gui_overview",
                "Headphone Correction": "ug_headphone_correction",
                "Binaural Room Simulation over Headphones": "ug_brs_over_hp",
                "Channel Configuration": "ug_chan_config",
                "File Naming and Structure": "ug_file_naming",
                "7.1 Surround Virtualisation": "ug_surround_71",
                "Acoustic Space Import": "ug_acoustic_import",
                "Room Target Generator": "ug_room_target",
                "Additional Settings": "ug_additional",
            }
    
            for sec_name, text_lines in sections.items():
                if sec_name not in header_map:
                    hf.log_with_timestamp(f"[User Guide] Section in TXT does not map to GUI: '{sec_name}'")
                    continue
            
                header_tag = header_map[sec_name]
                if not dpg.does_item_exist(header_tag):
                    hf.log_with_timestamp(f"[User Guide] GUI header missing: '{header_tag}'")
                    continue
            
                # Clear previous text
                try:
                    children = dpg.get_item_children(header_tag, 1)
                    if children:
                        for c in children:
                            dpg.delete_item(c)
                except Exception as e:
                    hf.log_with_timestamp(f"[User Guide] Failed to clear header '{sec_name}': {e}")
                    continue
            
                # --- Build up blocks ---
                current_block = []       # list of normal lines
                block_items = []         # created dpg items
                is_inside_block = False  # when normal text is accumulating
            
                def flush_block():
                    """Flush accumulated normal text into one DPG text item."""
                    nonlocal current_block
                    if not current_block:
                        return
                    combined = "\n".join(current_block)
                    item = dpg.add_text(combined, parent=header_tag, wrap=0)
                    dpg.bind_item_font(item, font_med)
                    current_block.clear()
            
                try:
                    if not text_lines:
                        dpg.add_text("(No content)", parent=header_tag)
                        continue
            
                    for raw in text_lines:
                        stripped = raw.strip()
            
                        # --- Detect bullet ---
                        is_bullet = stripped.startswith("- ") or stripped.startswith("* ")
            
                        # --- Detect heading ---
                        is_heading = (
                            stripped != ""
                            and not is_bullet
                            and not raw.startswith(" ")
                            and (raw.endswith(":") or len(stripped) < 80)
                        )
            
                        if is_heading:
                            # Heading = flush block + create bold title
                            flush_block()
                            item = dpg.add_text(stripped, parent=header_tag, wrap=0)
                            dpg.bind_item_font(item, font_b_med)
                            continue
            
                        # --- Normal line (including bullets) ---
                        if is_bullet:
                            indent = "    " if stripped.startswith("-") else "        "
                            current_block.append(indent + stripped)
                        else:
                            current_block.append(stripped)
            
                    # Final flush
                    flush_block()
            
                except Exception as e:
                    hf.log_with_timestamp(f"[User Guide] Failed to populate section '{sec_name}': {e}")
                
            hf.log_with_timestamp("[User Guide] Successfully loaded documentation.")
    
            # --- Async GitHub check + update ---
            if local_version:
                github_raw_url = CN.USER_GUIDE_URL
                threading.Thread(
                    target=check_and_update_github_user_guide,
                    args=(local_version, guide_path, github_raw_url),
                    daemon=True
                ).start()
    
        except Exception as e:
            hf.log_with_timestamp(f"[User Guide] Unexpected error: {e}")
        
    def intialise_gui():
        """ 
        GUI function to perform further initial configuration of gui elements
        """
        
        # acoustic space manager: Add a logger object inside the child window
        if CN.SHOW_AS_TAB:
            import_console_log = logger.mvLogger(parent="import_console_window")
            dpg.configure_item("import_console_window", user_data=import_console_log)
            cb.update_ir_folder_list()
            cb.update_as_table_from_csvs()
        # #room target manager
        cb.update_room_target_list()
        #qc_presets
        cb.qc_refresh_preset_list()
        
        #check for updates on start
        if loaded_values["check_updates_start_toggle"] == True:
            #start thread
            thread = threading.Thread(target=cb.check_all_updates, args=(), daemon=True)
            thread.start()


        #inital configuration
        #update channel gui elements on load, will also write e-APO config again
        cb.e_apo_select_channels(app_data=dpg.get_value('e_apo_audio_channels'),aquire_config=False)
        #adjust active tab
        try:
            dpg.set_value("tab_bar", loaded_values["tab_bar"])
        except Exception:
            pass

        hpcf_is_active=dpg.get_value('e_apo_hpcf_conv')
        brir_is_active=dpg.get_value('e_apo_brir_conv')
        #show hpcf history but only run function if enabled
        hpcf_hist_toggled = dpg.get_value('qc_toggle_hpcf_history')
        if hpcf_hist_toggled:
            cb.qc_show_hpcf_history(app_data=hpcf_hist_toggled)

        cb.e_apo_toggle_hpcf_custom(activate=hpcf_is_active, aquire_config=False)
        cb.e_apo_toggle_brir_custom(activate=brir_is_active, aquire_config=False)
        #finally acquire config once
        cb.e_apo_config_acquire()
        
        #also update gui labels based on active database
        qc_active_database = dpg.get_value('qc_hpcf_active_database')
        if qc_active_database == CN.HPCF_DATABASE_LIST[0]:#ash filters
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
       
        else:#compilation database
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
        
        #update audio device text
        hf.update_default_output_text(reset_sd=False)
        
        #populate user guide
        load_user_guide_into_gui()
        
    #QC loaded hp and sample lists based on loaded brand and headphone
    def load_db_values(active_db_name):
        """Retrieve brands, headphones, samples, and ensure valid selection."""
        conn = db_map[active_db_name]
        hpcf_db_dict['conn'] = conn
    
        # QC tab
        qc_brands = hpcf_functions.get_brand_list(conn)
        qc_brands, loaded_values["qc_hpcf_brand"] = hf.ensure_valid_selection(
            qc_brands, loaded_values.get("qc_hpcf_brand", "")
        )
        qc_hp_list = hpcf_functions.get_headphone_list(conn, loaded_values["qc_hpcf_brand"])
        qc_hp_list, loaded_values["qc_hpcf_headphone"] = hf.ensure_valid_selection(
            qc_hp_list, loaded_values.get("qc_hpcf_headphone", "")
        )
        qc_sample_list = hpcf_functions.get_samples_list(conn, loaded_values["qc_hpcf_headphone"])
        qc_sample_list, loaded_values["qc_hpcf_sample"] = hf.ensure_valid_selection(
            qc_sample_list, loaded_values.get("qc_hpcf_sample", "")
        )
    
        # FDE tab
        fde_brands = hpcf_functions.get_brand_list(conn)
        fde_brands, loaded_values["fde_hpcf_brand"] = hf.ensure_valid_selection(
            fde_brands, loaded_values.get("fde_hpcf_brand", "")
        )
        fde_hp_list = hpcf_functions.get_headphone_list(conn, loaded_values["fde_hpcf_brand"])
        fde_hp_list, loaded_values["fde_hpcf_headphone"] = hf.ensure_valid_selection(
            fde_hp_list, loaded_values.get("fde_hpcf_headphone", "")
        )
        fde_sample_list = hpcf_functions.get_samples_list(conn, loaded_values["fde_hpcf_headphone"])
        fde_sample_list, loaded_values["fde_hpcf_sample"] = hf.ensure_valid_selection(
            fde_sample_list, loaded_values.get("fde_hpcf_sample", "")
        )
    
        return conn, qc_brands, qc_hp_list, qc_sample_list, fde_brands, fde_hp_list, fde_sample_list
       

    def add_image_checkbox(
        label: str,
        checkbox_tag: str,
        callback,
        user_data,
        on_texture= CN.BUTTON_IMAGE_ON,
        off_texture= CN.BUTTON_IMAGE_OFF,
        width = 50,
        height = 50,
        default_value=False
    ):
        """
        Replace a checkbox with a power-style image button while keeping the
        original checkbox tag and user_data intact.
        
        Parameters:
        - label: Original checkbox label (optional for reference)
        - checkbox_tag: Tag for the hidden checkbox (must match existing code)
        - callback: Original checkbox callback
        - user_data: Original checkbox user_data
        - on_texture / off_texture: texture tags for button images
        - width / height: image button size
        - default_value: initial state of the checkbox
        - parent: optional parent container
        """
      
        #Hidden checkbox with exact same tag and user_data
        dpg.add_checkbox(
            label=label,
            tag=checkbox_tag,
            default_value=default_value,
            callback=callback,
            user_data=user_data,
            show=False
        )
    
        #Image button
        button_tag = f"{checkbox_tag}_btn"
    
        def _button_callback(sender, app_data, user_data_inner):
            # Toggle checkbox value
            new_val = not dpg.get_value(user_data_inner["checkbox"])
            dpg.set_value(user_data_inner["checkbox"], new_val)
    
            # Update button image
            dpg.configure_item(
                sender,
                texture_tag=user_data_inner["on_tex"] if new_val else user_data_inner["off_tex"]
            )
    
            # Call the original checkbox callback with proper signature
            cb_func = user_data_inner.get("callback")
            if cb_func:
                cb_func(app_data=new_val)
    
        dpg.add_image_button(
            texture_tag=on_texture if default_value else off_texture,
            width=width,
            height=height,
            tag=button_tag,
            callback=_button_callback,
            frame_padding=0,
            background_color=[0, 0, 0, 0],
            user_data={
                "checkbox": checkbox_tag,
                "on_tex": on_texture,
                "off_tex": off_texture,
                "callback": callback,
                "user_data": user_data
            }
        )
        dpg.bind_item_theme(button_tag, "transparent_image_button_theme")
        with dpg.tooltip(button_tag):
            dpg.add_text("Activate/Deactivate")
        

        return checkbox_tag, button_tag

 
      
        
    
    #
    #program code
    #
    

    #code to get gui width based on windows resolution
    gui_win_width_default=1722
    gui_win_height_default=717
    gui_win_width_loaded=gui_win_width_default
    gui_win_height_loaded=gui_win_height_default

    try:
        screen_width = ctypes.windll.user32.GetSystemMetrics(0)
        screen_height = ctypes.windll.user32.GetSystemMetrics(1)
        
        if screen_width < gui_win_width_default:
            gui_win_width_loaded=screen_width
        if screen_height < gui_win_height_default:
            gui_win_height_loaded=screen_height
    except Exception as e:
        logging.error(f"Error getting screen size: {e}")
        gui_win_width_loaded=gui_win_width_default
        gui_win_height_loaded=gui_win_height_default
        
    
    # HpCF Database code
    # create database connections to each database
    database_ash = pjoin(CN.DATA_DIR_OUTPUT,'hpcf_database.db')
    database_comp = pjoin(CN.DATA_DIR_OUTPUT,'hpcf_compilation_database.db')
    conn_ash = hpcf_functions.create_connection(database_ash)
    conn_comp = hpcf_functions.create_connection(database_comp)
    # Map database names to connections
    db_map = {
        CN.HPCF_DATABASE_LIST[0]: conn_ash,
        CN.HPCF_DATABASE_LIST[1]: conn_comp
    }
    
    #define hpcf defaults
    brands_list_default = hpcf_functions.get_brand_list(conn_ash)
    brand_default=brands_list_default[0]
    hp_list_default = hpcf_functions.get_headphone_list(conn_ash, brands_list_default[0])#index 0
    headphone_default = hp_list_default[0]
    sample_list_default = hpcf_functions.get_samples_list(conn_ash, headphone_default)
    sample_default = CN.HPCF_SAMPLE_DEFAULT #sample_list_default[0]
    qc_brand_default=brand_default
    qc_headphone_default=headphone_default
    qc_sample_default=sample_default
    #make updates for dynamic defaults in constants dict
    CN.DEFAULTS.update({
        "fde_hpcf_brand": brand_default,
        "fde_hpcf_headphone": headphone_default,
        "fde_hpcf_sample": sample_default,
        "qc_hpcf_brand": qc_brand_default,
        "qc_hpcf_headphone": qc_headphone_default,
        "qc_hpcf_sample": qc_sample_default,
    })
    
    #used for hpcf related callback functions, dict is stored as user data in a gui element  
    # store database info and connections in user data dict for GUI
    hpcf_db_dict = {
        'database_ash': database_ash,
        'conn_ash': conn_ash,
        'database_comp': database_comp,
        'conn_comp': conn_comp,
        'database': database_ash,           # currently active database
        'conn': conn_ash,                   # currently active connection
        'brands_list_default': brands_list_default,#used in reset function
        'hp_list_default': hp_list_default,
        'sample_list_default': sample_list_default,
        'brand_default': brand_default,
        'headphone_default': headphone_default,
        'sample_default': sample_default
    }
    
    
    #generate flat response
    impulse=np.zeros(CN.N_FFT)
    impulse[0]=1
    fr_flat_mag = np.abs(np.fft.fft(impulse))
    
    
    #thread variables
    e_apo_conf_lock = threading.Lock()
    
    #
    # Path handling
    #
    #get equalizer APO path
    primary_path, primary_ash_path, e_apo_path = get_primary_paths()
    qc_primary_path=primary_path
    
    #load settings from settings file, stores in a dict
    loaded_values = cb.load_settings()

    #overwrite paths if user values differ from defaults
    default_path = CN.DEFAULTS["path"]
    loaded_path = loaded_values.get("path", default_path)
    
    # --- Resolve active path ---
    if os.path.normcase(loaded_path) != os.path.normcase(default_path):
        active_path = loaded_path
        hf.log_with_timestamp(f"Using user-saved path for filter and dataset exports: {active_path}")
    else:
        active_path = primary_path
        hf.log_with_timestamp(f"No custom path found. Using detected primary path for filter and dataset exports: {active_path}")

    primary_path = active_path
    primary_ash_path = pjoin(primary_path, CN.PROJECT_FOLDER)
    
    #path handling continued
    #set hesuvi path
    if 'EqualizerAPO' in primary_path:
        primary_hesuvi_path = pjoin(primary_path,'HeSuVi')#stored outside of project folder (within hesuvi installation)
    else:
        primary_hesuvi_path = pjoin(primary_path, CN.PROJECT_FOLDER,'HeSuVi')#stored within project folder
    
    

        
    # Ensure both tabs have the same active database
    qc_active_database = loaded_values['qc_hpcf_active_database']
    loaded_values['fde_hpcf_active_database'] = qc_active_database
    

    # First, try the current active database
    conn = db_map.get(qc_active_database, conn_ash)
    qc_brands_list = hpcf_functions.get_brand_list(conn)
    
    # Check if loaded brand exists in brand list; if not, switch database
    if loaded_values.get("qc_hpcf_brand", "") not in qc_brands_list:
        # Switch to the other database
        new_db_name = CN.HPCF_DATABASE_LIST[1] if qc_active_database == CN.HPCF_DATABASE_LIST[0] else CN.HPCF_DATABASE_LIST[0]
        qc_active_database = new_db_name
        loaded_values['qc_hpcf_active_database'] = new_db_name
        loaded_values['fde_hpcf_active_database'] = new_db_name
        hf.log_with_timestamp("Previously applied HpCF not found in active headphone database. Switching database.")
    
    # Load all values from the active (or switched) database
    conn, qc_brands_list_loaded, qc_hp_list_loaded, qc_sample_list_loaded, fde_brands_list_loaded, fde_hp_list_loaded, fde_sample_list_loaded = load_db_values(qc_active_database)
        
 
    #grab loaded settings for dynamic lists (hrtf lists)
    fde_brir_hrtf_type_loaded=loaded_values["fde_brir_hrtf_type"]
    fde_brir_hrtf_dataset_loaded=loaded_values["fde_brir_hrtf_dataset"]
    fde_hrtf_loaded=loaded_values["fde_brir_hrtf"]
    fde_brir_hrtf_dataset_list_loaded = CN.HRTF_TYPE_DATASET_DICT.get(fde_brir_hrtf_type_loaded) or []
    qc_brir_hrtf_type_loaded=loaded_values["qc_brir_hrtf_type"]
    qc_brir_hrtf_dataset_loaded=loaded_values["qc_brir_hrtf_dataset"]
    qc_hrtf_loaded =loaded_values["qc_brir_hrtf"]
    qc_brir_hrtf_dataset_list_loaded = CN.HRTF_TYPE_DATASET_DICT.get(qc_brir_hrtf_type_loaded) or []
  

    #listener related buttons are dynamic
    add_fav_button_show=True
    remove_fav_button_show=False
    sofa_folder_button_show=False
    qc_add_fav_button_show=True
    qc_remove_fav_button_show=False
    qc_sofa_folder_button_show=False

    
    if fde_brir_hrtf_type_loaded == 'Favourites':#use loaded favourites list if favourites selected
        add_fav_button_show=False
        remove_fav_button_show=True     
    elif fde_brir_hrtf_type_loaded == 'User SOFA Input':
        sofa_folder_button_show=True
        
    if fde_brir_hrtf_type_loaded == 'Favourites':#use loaded favourites list if favourites selected
        hrtf_list_loaded = loaded_values["hrtf_list_favs"]
    else:
        hrtf_list_loaded = hrir_processing.get_listener_list(listener_type=fde_brir_hrtf_type_loaded, dataset_name=fde_brir_hrtf_dataset_loaded)

    if qc_brir_hrtf_type_loaded == 'Favourites':#use loaded favourites list if favourites selected
        qc_add_fav_button_show=False
        qc_remove_fav_button_show=True
    elif qc_brir_hrtf_type_loaded == 'User SOFA Input':
        qc_sofa_folder_button_show=True
        
    if qc_brir_hrtf_type_loaded == 'Favourites':#use loaded favourites list if favourites selected
        qc_hrtf_list_loaded = loaded_values["hrtf_list_favs"]
    else:
        qc_hrtf_list_loaded = hrir_processing.get_listener_list(listener_type=qc_brir_hrtf_type_loaded, dataset_name=qc_brir_hrtf_dataset_loaded)


    #export options are dynamic
    #adjust hrtf list based on loaded spatial resolution
    #also adjust file export selection
    dir_brir_exp_show=True
    ts_brir_exp_show=True
    hesuvi_brir_exp_show=True
    eapo_brir_exp_show=True
    sofa_brir_exp_show=True
    dir_brir_tooltip_show=True
    ts_brir_tooltip_show=True
    hesuvi_brir_tooltip_show=True
    multi_chan_brir_tooltip_show=True
    sofa_brir_tooltip_show=True
    
    if loaded_values["fde_brir_spat_res"] == 'Medium' or loaded_values["fde_brir_spat_res"] == 'Low':
        loaded_values["fde_sofa_brir_toggle"] =False
        sofa_brir_exp_show=False
        sofa_brir_tooltip_show=False
    spatial_res_list_loaded=CN.SPATIAL_RES_LIST_LIM


    # ----- List validation ------
    
    #validate that loaded strings are within associated lists - dynamic lists
    fde_brir_hrtf_dataset_loaded = hf.validate_choice(fde_brir_hrtf_dataset_loaded, fde_brir_hrtf_dataset_list_loaded)
    qc_brir_hrtf_dataset_loaded = hf.validate_choice(qc_brir_hrtf_dataset_loaded, qc_brir_hrtf_dataset_list_loaded)
    fde_hrtf_loaded = hf.validate_choice(fde_hrtf_loaded, hrtf_list_loaded)
    qc_hrtf_loaded = hf.validate_choice(qc_hrtf_loaded, qc_hrtf_list_loaded)

    # Define which additional keys need list validation and their valid option lists - static lists
    VALIDATION_MAP = {
        "fde_acoustic_space": CN.AC_SPACE_LIST_GUI,
        "qc_acoustic_space": CN.AC_SPACE_LIST_GUI,
        "fde_room_target": CN.ROOM_TARGET_LIST,
        "qc_room_target": CN.ROOM_TARGET_LIST,
        "fde_brir_hp_type": CN.HP_COMP_LIST,
        "qc_brir_hp_type": CN.HP_COMP_LIST,
        "fde_sub_response": CN.SUB_RESPONSE_LIST_GUI,
        "qc_sub_response": CN.SUB_RESPONSE_LIST_GUI,
        # "fde_brir_hrtf_dataset": CN.HRTF_TYPE_DATASET_DICT.get(CN.HRTF_TYPE_DEFAULT, []),
        # "qc_brir_hrtf_dataset": CN.HRTF_TYPE_DATASET_DICT.get(CN.HRTF_TYPE_DEFAULT, []),
    }
    # Validate list-based settings after loading 
    for key, valid_list in VALIDATION_MAP.items():
        if key in loaded_values:
            loaded_values[key] = hf.validate_choice(loaded_values[key], valid_list)



    #
    # Equalizer APO related code
    #
    elevation_list_sel= CN.ELEV_ANGLES_WAV_LOW
  
 
    #
    ## GUI CODE
    #
    
    #default plotting data
    default_x = []
    default_y = []
    for i in range(0, 500):
        default_x.append(i)
        default_y.append(i)
 
    dpg.create_context()
    
    image_location_on_blue = pjoin(CN.DATA_DIR_RAW, 'power_on_blue.png')
    image_location_off = pjoin(CN.DATA_DIR_RAW, 'power_off.png')
    image_location_start_blue = pjoin(CN.DATA_DIR_RAW, 'start_blue.png')
    image_location_start_green = pjoin(CN.DATA_DIR_RAW, 'start_green.png')
    image_location_fl = pjoin(CN.DATA_DIR_RAW, 'l_icon.png')
    image_location_fr = pjoin(CN.DATA_DIR_RAW, 'r_icon.png')
    image_location_c = pjoin(CN.DATA_DIR_RAW, 'c_icon.png')
    image_location_sl = pjoin(CN.DATA_DIR_RAW, 'sl_icon.png')
    image_location_sr = pjoin(CN.DATA_DIR_RAW, 'sr_icon.png')
    image_location_rl = pjoin(CN.DATA_DIR_RAW, 'rl_icon.png')
    image_location_rr = pjoin(CN.DATA_DIR_RAW, 'rr_icon.png')
    image_location_listener = pjoin(CN.DATA_DIR_RAW, 'listener_icon.png')
    width_on_blue, height_on_blue, channels_on_blue, data_on_blue = dpg.load_image(image_location_on_blue)
    width_off, height_off, channels_off, data_off = dpg.load_image(image_location_off)
    width_start_blue, height_start_blue, channels_start_blue, data_start_blue = dpg.load_image(image_location_start_blue)
    width_start_green, height_start_green, channels_start_green, data_start_green = dpg.load_image(image_location_start_green)
    width_fl, height_fl, channels_fl, data_fl = dpg.load_image(image_location_fl)
    width_fr, height_fr, channels_fr, data_fr = dpg.load_image(image_location_fr)
    width_c, height_c, channels_c, data_c = dpg.load_image(image_location_c)
    width_sl, height_sl, channels_sl, data_sl = dpg.load_image(image_location_sl)
    width_sr, height_sr, channels_sr, data_sr = dpg.load_image(image_location_sr)
    width_rl, height_rl, channels_rl, data_rl = dpg.load_image(image_location_rl)
    width_rr, height_rr, channels_rr, data_rr = dpg.load_image(image_location_rr)
    width_listener, height_listener, channels_listener, data_listener = dpg.load_image(image_location_listener)

    with dpg.texture_registry(show=False):
        dpg.add_static_texture(width=width_on_blue, height=height_on_blue, default_value=data_on_blue, tag=CN.BUTTON_IMAGE_ON)
        dpg.add_static_texture(width=width_off, height=height_off, default_value=data_off, tag=CN.BUTTON_IMAGE_OFF)
        dpg.add_static_texture(width=width_start_blue, height=height_start_blue, default_value=data_start_blue, tag='start_blue')
        dpg.add_static_texture(width=width_start_green, height=height_start_green, default_value=data_start_green, tag='start_green')
        dpg.add_static_texture(width=width_fl, height=height_fl, default_value=data_fl, tag='fl_image')
        dpg.add_static_texture(width=width_fr, height=height_fr, default_value=data_fr, tag='fr_image')
        dpg.add_static_texture(width=width_c, height=height_c, default_value=data_c, tag='c_image')
        dpg.add_static_texture(width=width_sl, height=height_sl, default_value=data_sl, tag='sl_image')
        dpg.add_static_texture(width=width_sr, height=height_sr, default_value=data_sr, tag='sr_image')
        dpg.add_static_texture(width=width_rl, height=height_rl, default_value=data_rl, tag='rl_image')
        dpg.add_static_texture(width=width_rr, height=height_rr, default_value=data_rr, tag='rr_image')
        dpg.add_static_texture(width=width_listener, height=height_listener, default_value=data_listener, tag='listener_image')

    # add a font registry
    with dpg.font_registry():
 
        in_file_path = pjoin(CN.DATA_DIR_EXT,'font', 'Roboto-Medium.ttf')
        font_def = dpg.add_font(in_file_path, 14)    
        font_small = dpg.add_font(in_file_path, 13)
        font_med = dpg.add_font(in_file_path, 15)
        
        in_file_path = pjoin(CN.DATA_DIR_EXT,'font', 'Roboto-Bold.ttf')
        font_b_def = dpg.add_font(in_file_path, 14)
        font_b_small = dpg.add_font(in_file_path, 13) 
        font_b_large = dpg.add_font(in_file_path, 16)  
        font_b_med = dpg.add_font(in_file_path, 15)

    
        
    #crete viewport
    dpg.create_viewport(title='ASH Toolset', width=gui_win_width_loaded, height=gui_win_height_loaded, small_icon=CN.ICON_LOCATION, large_icon=CN.ICON_LOCATION)#Audio Spatialisation for Headphones
    
    #All elements fall within primary window
    with dpg.window(tag="Primary Window", horizontal_scrollbar=True):
        # set font of app
        dpg.bind_font(font_def)
        # Themes
        with dpg.theme(tag="transparent_image_button_theme"):
            with dpg.theme_component(dpg.mvAll):
                # 1. Make the Button Background Transparent
                dpg.add_theme_color(dpg.mvThemeCol_Button, [0, 0, 0, 0])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [100, 100, 100, 50])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [0, 0, 0, 0])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, [0.0, 0.0, 0.0, 0.0])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, [0.0, 0.0, 0.0, 0.0])
                # 2. Remove surrounding visual frame/padding (Styles)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 0, 0)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
        with dpg.theme(tag="__theme_b"):
            i=3.8#i=2
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(i/7.0, 0.3, 0.5))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(i/7.0, 0.8, 0.8))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(i/7.0, 0.6, 0.7))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, i*5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 7, 7)
        with dpg.theme(tag="__theme_c"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Header, (138, 138, 62), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (135, 163, 78), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (138, 138, 62), category=dpg.mvThemeCat_Core)
        with dpg.theme(tag="__theme_d"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(0.6, 0.15, 0.3))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(0.6, 0.15, 0.6))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(0.6, 0.15, 0.5))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 2, 2)
        with dpg.theme(tag="__theme_f"):
            i=3.9
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, _hsv_to_rgb(i/7.0, 0.05, 0.99))
        with dpg.theme(tag="__theme_g"):
            i=3.5
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, _hsv_to_rgb(i/7.0, 0.2, 0.55)) 
        with dpg.theme(tag="__theme_h"):
            i=4.0
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, _hsv_to_rgb(i/7.0, 0.2, 0.55)) 
        with dpg.theme(tag="__theme_i"):#used for FDE tab
            i=3.5
            j=3.5
            k=3.4
            sat_mult=0.7
            val_mult=0.9
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, _hsv_to_rgb(i/7.0, 0.3*sat_mult, 0.5*val_mult))   
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, _hsv_to_rgb(i/7.0, 0.3*sat_mult, 0.5*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, _hsv_to_rgb(i/7.0, 0.5*sat_mult, 0.8*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, _hsv_to_rgb(i/7.0, 0.5*sat_mult, 0.7*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, _hsv_to_rgb(i/7.0, 0.5*sat_mult, 0.7*val_mult))
                dpg.add_theme_color(dpg.mvPlotCol_TitleText, _hsv_to_rgb(i/7.0, 0.5*sat_mult, 0.7*val_mult))
                dpg.add_theme_color(dpg.mvThemeCol_Header, _hsv_to_rgb(4.4/7.0, 0.1*sat_mult, 0.25*val_mult))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, _hsv_to_rgb(j/7.0, 0.3*sat_mult, 0.5*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, _hsv_to_rgb(j/7.0, 0.3*sat_mult, 0.5*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_TextSelectedBg, _hsv_to_rgb(j/7.0, 0.3*sat_mult, 0.5*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, _hsv_to_rgb(j/7.0, 0.3*sat_mult, 0.5*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(j/7.0, 0.3*sat_mult, 0.5*val_mult)) 
                dpg.add_theme_color(dpg.mvPlotCol_Line, _hsv_to_rgb(k/7.0, 0.25*sat_mult, 0.6*val_mult), category=dpg.mvThemeCat_Plots) 
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, _hsv_to_rgb(j/7.0, 0.7*sat_mult, 0.7*val_mult))
        with dpg.theme(tag="__theme_j"):#used for other tabs
            i=4.5
            j=4.5
            sat_mult=0.3
            val_mult=0.9
            with dpg.theme_component(dpg.mvAll):  
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, _hsv_to_rgb(i/7.0, 0.4*sat_mult, 0.5*val_mult))   
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, _hsv_to_rgb(i/7.0, 0.3*sat_mult, 0.5*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, _hsv_to_rgb(i/7.0, 0.5*sat_mult, 0.8*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, _hsv_to_rgb(i/7.0, 0.5*sat_mult, 0.7*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, _hsv_to_rgb(i/7.0, 0.5*sat_mult, 0.7*val_mult))
                dpg.add_theme_color(dpg.mvPlotCol_TitleText, _hsv_to_rgb(i/7.0, 0.5*sat_mult, 0.7*val_mult))
                dpg.add_theme_color(dpg.mvThemeCol_Header, _hsv_to_rgb(4.4/7.0, 0.1*sat_mult, 0.25*val_mult))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, _hsv_to_rgb(j/7.0, 0.3*sat_mult, 0.5*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, _hsv_to_rgb(j/7.0, 0.3*sat_mult, 0.5*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, _hsv_to_rgb(j/7.0, 0.3*sat_mult, 0.5*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(j/7.0, 0.3*sat_mult, 0.5*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, _hsv_to_rgb(j/7.0, 0.7*sat_mult, 0.7*val_mult))
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, _hsv_to_rgb(i/7.0, 0.2, 0.55)) 
                dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(j/7.0, 0.3*sat_mult, 0.3*val_mult))
        with dpg.theme(tag="__theme_k"):
            i=4.2#i=3
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(i/7.0, 0.3, 0.3))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(i/7.0, 0.8, 0.5))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(i/7.0, 0.6, 0.4))
        with dpg.theme(tag="__theme_l"):
            i=3.8
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(i/7.0, 0.3, 0.3))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(i/7.0, 0.8, 0.5))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(i/7.0, 0.6, 0.4))
        with dpg.theme() as modern_theme:#default theme used for QC tab
            i=4.1
            j=4.1
            k=4.1
            sat_mult=0.7
            val_mult=1.0
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (32, 32, 32), category=dpg.mvThemeCat_Core)  # Neutral dark gray
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (44, 44, 44), category=dpg.mvThemeCat_Core)   # Slightly lighter neutral dark gray
                dpg.add_theme_color(dpg.mvThemeCol_Border, (60, 60, 60), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 60, 60), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 40, 40), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 220, 220), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 4, 3, category=dpg.mvThemeCat_Core)#8, 6,
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 4, 4, category=dpg.mvThemeCat_Core)# 8, 8,
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, _hsv_to_rgb(i/7.0, 0.4*sat_mult, 0.5*val_mult))   
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, _hsv_to_rgb(i/7.0, 0.3*sat_mult, 0.5*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, _hsv_to_rgb(i/7.0, 0.5*sat_mult, 0.8*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, _hsv_to_rgb(i/7.0, 0.5*sat_mult, 0.7*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, _hsv_to_rgb(i/7.0, 0.5*sat_mult, 0.7*val_mult))
                dpg.add_theme_color(dpg.mvPlotCol_TitleText, _hsv_to_rgb(i/7.0, 0.5*sat_mult, 0.7*val_mult))
                dpg.add_theme_color(dpg.mvThemeCol_Header, _hsv_to_rgb(4.4/7.0, 0.1*sat_mult, 0.25*val_mult))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, _hsv_to_rgb(j/7.0, 0.3*sat_mult, 0.5*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, _hsv_to_rgb(j/7.0, 0.3*sat_mult, 0.5*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_TextSelectedBg, _hsv_to_rgb(j/7.0, 0.3*sat_mult, 0.5*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, _hsv_to_rgb(j/7.0, 0.3*sat_mult, 0.5*val_mult)) 
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(j/7.0, 0.3*sat_mult, 0.5*val_mult)) 
                dpg.add_theme_color(dpg.mvPlotCol_Line, _hsv_to_rgb(k/7.0, 0.25*sat_mult, 0.6*val_mult), category=dpg.mvThemeCat_Plots) 
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, _hsv_to_rgb(j/7.0, 0.7*sat_mult, 0.7*val_mult))
        dpg.bind_theme(modern_theme)#global_theme
   
        # ---- main GUI code ------
        with dpg.tab_bar(tag='tab_bar'):

            with dpg.tab(label="Quick Configuration",tag='quick_config', parent="tab_bar"): 
                dpg.add_text("Apply Headphone Correction & Binaural Room Simulation in Equalizer APO")
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    #Section for HpCF Export
                    with dpg.child_window(width=550, height=610):
                        title_1_qc = dpg.add_text("Headphone Correction")
                        dpg.bind_item_font(title_1_qc, font_b_med)
                        with dpg.child_window(autosize_x=True, height=410):
                            

                            
                            with dpg.group(horizontal=True):
                                
                                dpg.add_text("Headphone Database: ")
                                dpg.add_combo(CN.HPCF_DATABASE_LIST,  tag= "qc_hpcf_active_database", default_value=loaded_values["qc_hpcf_active_database"], callback=cb.change_hpcf_database_callback, width=140)
                   
                               
                                dpg.add_text("             ")
                                dpg.add_checkbox(label="Show History", default_value=loaded_values["qc_toggle_hpcf_history"], tag='qc_toggle_hpcf_history', callback=cb.qc_show_hpcf_history)

                                with dpg.tooltip("qc_toggle_hpcf_history"):
                                    dpg.add_text("Shows previously applied headphones")
                                dpg.add_text("     ")
                                
                                # Button to trigger the popup
                                dpg.add_button(label="Clear History", user_data="", tag="qc_clear_history_button",
                                               callback=lambda: dpg.configure_item("qc_clear_history_popup", show=True))
                                # Optional tooltip
                                with dpg.tooltip("qc_clear_history_button"):
                                    dpg.add_text("Clear headphone correction history")
                                # Confirmation popup
                                with dpg.popup("qc_clear_history_button", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="qc_clear_history_popup"):
                                    dpg.add_text("This will clear headphone correction history")
                                    dpg.add_separator()
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="OK", width=75, callback=cb.remove_hpcfs)
                                        dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("qc_clear_history_popup", show=False))
                
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_text("Search Brand:", tag='qc_hpcf_brand_search_title')
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                dpg.add_input_text(label="", callback=cb.filter_brand_list, width=105, tag='qc_hpcf_brand_search')
                                dpg.add_text("Search Headphone:", tag='qc_hpcf_headphone_search_title')
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                dpg.add_input_text(label="", callback=cb.filter_headphone_list, width=209, tag='qc_hpcf_headphone_search')
                            with dpg.group(horizontal=True, width=0):
                                with dpg.group():
                                    dpg.add_text("Brand", tag='qc_hpcf_brand_title')
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    dpg.add_listbox(qc_brands_list_loaded, width=135, num_items=17, tag='qc_hpcf_brand', default_value=loaded_values["qc_hpcf_brand"], callback=cb.qc_update_headphone_list)
                                with dpg.group():
                                    dpg.add_text("Headphone", tag='qc_hpcf_headphone_title')
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    dpg.add_listbox(qc_hp_list_loaded, width=250, num_items=17, tag='qc_hpcf_headphone', default_value=loaded_values["qc_hpcf_headphone"] ,callback=cb.qc_update_sample_list)
                                with dpg.group():
                                    dpg.add_text("Sample", tag='qc_hpcf_sample_title')
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    dpg.add_listbox(qc_sample_list_loaded, width=115, num_items=17, default_value=loaded_values["qc_hpcf_sample"], tag='qc_hpcf_sample', callback=cb.qc_plot_sample) 
                        
                        with dpg.child_window(autosize_x=True, height=153):
                            with dpg.group(horizontal=True):
                                subtitle_3_qc = dpg.add_text("Apply Headphone Correction in Equalizer APO")
                                dpg.bind_item_font(subtitle_3_qc, font_b_def)
                                dpg.add_text("                                    ")
                                dpg.add_checkbox(label="Auto Apply Selection", default_value = loaded_values["qc_auto_apply_hpcf_sel"],  tag='qc_auto_apply_hpcf_sel', callback=cb.e_apo_auto_apply_hpcf)
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
        
                                dpg.add_image_button(texture_tag='start_blue',width=40,height=40,tag="qc_hpcf_tag",callback=cb.qc_apply_hpcf_params,frame_padding=0,background_color=[0, 0, 0, 0])
                                dpg.bind_item_theme(dpg.last_item(), "transparent_image_button_theme")
                                
                                with dpg.tooltip("qc_hpcf_tag"):
                                    dpg.add_text("This will apply the selected filter in Equalizer APO")
                                dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=30, width=412,pos = (60,40), overlay=CN.PROGRESS_START, tag="qc_progress_bar_hpcf")
                                dpg.bind_item_theme(dpg.last_item(), "__theme_h")
                                dpg.add_text(" ")
                                
                                add_image_checkbox(label="Enable Headphone Correction",width=40,height=40,checkbox_tag="e_apo_hpcf_conv",callback=cb.e_apo_toggle_hpcf_gui,user_data=e_apo_conf_lock,default_value=loaded_values["e_apo_hpcf_conv"])
                    
                            dpg.add_separator()
                            with dpg.group():
                                dpg.add_text("Active Filter: ")
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                dpg.add_text(default_value=loaded_values["qc_e_apo_curr_hpcf"],  tag='qc_e_apo_curr_hpcf', wrap=450)
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                dpg.bind_item_theme(dpg.last_item(), "__theme_f")
                                dpg.add_text(default_value=loaded_values["qc_e_apo_sel_hpcf"], tag='qc_e_apo_sel_hpcf',show=False, user_data=hpcf_db_dict)
                            
                    #Section for BRIR generation
                    with dpg.child_window(width=534, height=610):
                        title_2_qc = dpg.add_text("Binaural Room Simulation", tag='qc_brir_title')
                        dpg.bind_item_font(title_2_qc, font_b_med)
                        with dpg.tooltip("qc_brir_title"):
                            dpg.add_text("Customise a new binaural room simulation using below parameters")
                        with dpg.child_window(autosize_x=True, height=410):
                            with dpg.tab_bar(tag='qc_brir_tab_bar'):

                                with dpg.tab(label="Acoustics & EQ Parameters",tag='qc_acoustics_eq_tab', parent="qc_brir_tab_bar"): 

                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Acoustic Space",tag='qc_acoustic_space_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            with dpg.group(horizontal=True):
                                                dpg.add_text("Sort by: ")
                                                dpg.add_combo(CN.AC_SPACE_LIST_SORT_BY, width=140, label="",  tag='qc_sort_by_as',default_value=CN.AC_SPACE_LIST_SORT_BY[0], callback=cb.qc_sort_ac_space)
                                            with dpg.group(horizontal=True):
                                                dpg.add_listbox(CN.AC_SPACE_LIST_GUI, label="",tag='qc_acoustic_space',default_value=loaded_values["qc_acoustic_space"], callback=cb.qc_update_ac_space, num_items=17, width=245)
                                                with dpg.tooltip("qc_acoustic_space_title"):
                                                    dpg.add_text("This will determine the listening environment")
                                                dpg.add_text("  ")
                                        with dpg.group():
                                            dpg.add_text("Direct Sound Gain (dB)", tag='qc_direct_gain_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_input_float(label="Direct Gain (dB)",width=140, format="%.1f", tag='qc_direct_gain', min_value=CN.DIRECT_GAIN_MIN, max_value=CN.DIRECT_GAIN_MAX, default_value=loaded_values["qc_direct_gain"],min_clamped=True, max_clamped=True, callback=cb.qc_update_direct_gain)
                                            with dpg.tooltip("qc_direct_gain_title"):
                                                dpg.add_text("This will control the loudness of the direct signal")
                                                dpg.add_text("Higher values result in lower perceived distance, lower values result in higher perceived distance")
                                            dpg.add_slider_float(label="", min_value=CN.DIRECT_GAIN_MIN, max_value=CN.DIRECT_GAIN_MAX, default_value=loaded_values["qc_direct_gain_slider"], width=140,clamped=True, no_input=True, format="", callback=cb.qc_update_direct_gain_slider, tag='qc_direct_gain_slider')
                                            dpg.add_separator()
                                            dpg.add_text("Room Target", tag='qc_rm_target_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_listbox(CN.ROOM_TARGET_LIST, default_value=loaded_values["qc_room_target"], num_items=8, width=245, tag='qc_room_target', callback=cb.qc_select_room_target)
                                            with dpg.tooltip("qc_rm_target_title"):
                                                dpg.add_text("This will influence the overall balance of low and high frequencies")
                                            dpg.add_separator()
                                            dpg.add_text("Headphone Compensation", tag='qc_brir_hp_type_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_listbox(CN.HP_COMP_LIST, default_value=loaded_values["qc_brir_hp_type"], num_items=4, width=245, callback=cb.qc_select_hp_comp, tag='qc_brir_hp_type')
                                            with dpg.tooltip("qc_brir_hp_type_title"):
                                                dpg.add_text("This will compensate typical interactions between the headphone and the outer ear")
                                                dpg.add_text("Selection should align with the listener's headphone type")
                                                dpg.add_text("Reduce to low strength if sound localisation or timbre is compromised")
                                                
                                with dpg.tab(label="Listener Selection",tag='qc_listener_tab', parent="qc_brir_tab_bar"): 
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Category", tag='qc_brir_hrtf_type_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_radio_button(CN.HRTF_TYPE_LIST, horizontal=False, tag= "qc_brir_hrtf_type", default_value=loaded_values["qc_brir_hrtf_type"], callback=cb.qc_update_hrtf_dataset_list )
                                            with dpg.tooltip("qc_brir_hrtf_type_title"):
                                                dpg.add_text("User SOFA files must be placed in 'ASH Toolset\\_internal\\data\\user\\SOFA' folder")
                                            dpg.add_separator()
                                            dpg.add_text("Dataset", tag='qc_brir_hrtf_dataset_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_listbox(qc_brir_hrtf_dataset_list_loaded, default_value=qc_brir_hrtf_dataset_loaded, num_items=11, width=255, callback=cb.qc_update_hrtf_list, tag='qc_brir_hrtf_dataset')
                                        with dpg.group():
                                            dpg.add_loading_indicator(style=1, pos = (486,355), radius =1.9, color =(120,120,120),show=False, tag='qc_hrtf_average_fav_load_ind')
                                            dpg.add_text("Listener", tag='qc_brir_hrtf_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_listbox(qc_hrtf_list_loaded, default_value=qc_hrtf_loaded, num_items=17, width=240, callback=cb.qc_select_hrtf, tag='qc_brir_hrtf')
                                            with dpg.tooltip("qc_brir_hrtf_title"):
                                                dpg.add_text("This will influence the externalisation and localisation of sounds around the listener")
                                            with dpg.group(horizontal=True):
                                                dpg.add_text("  ")
                                                dpg.add_button(label="Add Favourite", tag="qc_hrtf_add_favourite", callback=cb.add_hrtf_favourite_qc_callback,show=qc_add_fav_button_show)

                                                dpg.add_button(label="Remove Favourite", tag="qc_hrtf_remove_favourite", callback=cb.remove_hrtf_favourite_qc_callback,show=qc_remove_fav_button_show)
                                                dpg.add_text("  ")
                                                dpg.add_button(label="Create Average", tag="qc_hrtf_average_favourite", callback=cb.create_hrtf_favourite_avg,show=qc_remove_fav_button_show)
                                                dpg.bind_item_theme(dpg.last_item(), "__theme_k")
                                                with dpg.tooltip("qc_hrtf_average_favourite"):
                                                    dpg.add_text("Creates an averaged HRTF by interpolating across favourite listeners")
                                                dpg.add_button(label="Open Folder", tag="qc_open_user_sofa_folder", callback=cb.open_user_sofa_folder,show=qc_sofa_folder_button_show)
                                                
                                with dpg.tab(label="Low-frequency Extension",tag='qc_lfe_tab', parent="qc_brir_tab_bar"): 
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Integration Crossover Frequency", tag='qc_crossover_f_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_combo(CN.SUB_FC_SETTING_LIST, default_value=loaded_values["qc_crossover_f_mode"], width=130, callback=cb.qc_update_crossover_f, tag='qc_crossover_f_mode')
                                            with dpg.tooltip("qc_crossover_f_mode"):
                                                dpg.add_text("Auto Select mode will select an optimal frequency for the selected acoustic space")
                                            dpg.add_input_int(label="Crossover Frequency (Hz)",width=140, tag='qc_crossover_f', min_value=CN.SUB_FC_MIN, max_value=CN.SUB_FC_MAX, default_value=loaded_values["qc_crossover_f"],min_clamped=True, max_clamped=True, callback=cb.qc_update_crossover_f)
                                            with dpg.tooltip("qc_crossover_f"):
                                                dpg.add_text("Crossover Frequency can be adjusted to a value up to 150Hz")
                                                dpg.add_text("This will tune the integration of the cleaner LF response and original room response")
                                                dpg.add_text("Higher values may result in a smoother bass response")
                                                dpg.add_text("Set to 0Hz to disable this feature")
                                            dpg.add_separator()
                                            dpg.add_text("Response for Low-frequency Extension", tag='qc_sub_response_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_listbox(CN.SUB_RESPONSE_LIST_GUI, default_value=loaded_values["qc_sub_response"], num_items=11, width=350, callback=cb.qc_select_sub_brir, tag='qc_sub_response')
                                            with dpg.tooltip("qc_sub_response_title"):
                                                dpg.add_text("Refer to reference tables tab and filter preview for comparison")
                                            dpg.add_separator()
                                            dpg.add_text("Additonal EQ", tag='qc_hp_rolloff_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            with dpg.group(horizontal=True):
                                                dpg.add_checkbox(label="Compensate Headphone Roll-off", default_value = loaded_values["qc_hp_rolloff_comp"],  tag='qc_hp_rolloff_comp', callback=cb.qc_update_brir_param)
                                                with dpg.tooltip("qc_hp_rolloff_comp"):
                                                    dpg.add_text("This will compensate the typical reduction in bass response at lower frequencies")
                                                dpg.add_text("      ")
                                                dpg.add_checkbox(label="Forward-Backward Filtering", default_value = loaded_values["qc_fb_filtering"],  tag='qc_fb_filtering', callback=cb.qc_update_brir_param)
                                                with dpg.tooltip("qc_fb_filtering"):
                                                    dpg.add_text("This will eliminate delay introduced by the filters, however can introduce edge artefacts in some cases")
                                            
                                       
                                            
                                                
                        with dpg.child_window(autosize_x=True, height=155):
                            with dpg.group(horizontal=True):
                                subtitle_6_qc = dpg.add_text("Apply Simulation in Equalizer APO")
                                dpg.bind_item_font(subtitle_6_qc, font_b_def)
                                dpg.add_text("                                                      ")
                  
                            dpg.add_separator()
                            
                            with dpg.group(horizontal=True):
                        
                                dpg.add_image_button(texture_tag='start_blue',width=40,height=40,tag="qc_brir_tag",callback=cb.qc_apply_brir_params,frame_padding=0,background_color=[0, 0, 0, 0],user_data=CN.PROCESS_BRIRS_RUNNING)#user data is thread running flag
                                dpg.bind_item_theme(dpg.last_item(), "transparent_image_button_theme")
                                with dpg.tooltip("qc_brir_tag"):
                                    dpg.add_text("This will generate the binaural simulation and apply it in Equalizer APO")  
                                dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=30, width=397,pos = (60,40), overlay=CN.PROGRESS_START, tag="qc_progress_bar_brir",user_data=CN.STOP_THREAD_FLAG)#user data is thread cancel flag
                                dpg.bind_item_theme(dpg.last_item(), "__theme_h")
                                dpg.add_text(" ")
                                add_image_checkbox(label="Enable Binaural Room Simulation",width=40,height=40,checkbox_tag="e_apo_brir_conv",callback=cb.e_apo_toggle_brir_gui,user_data=[],default_value=loaded_values["e_apo_brir_conv"])
                     
                            dpg.add_separator()
       
                            with dpg.group():
                                dpg.add_text("Active Simulation: ")
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                dpg.add_text(default_value=loaded_values["qc_e_apo_curr_brir_set"],  tag='qc_e_apo_curr_brir_set' , wrap=500, user_data=False)#user data used for flagging use of below BRIR dict
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                dpg.bind_item_theme(dpg.last_item(), "__theme_f")
                                dpg.add_text(default_value=loaded_values["qc_e_apo_sel_brir_set"], tag='qc_e_apo_sel_brir_set',show=False, user_data={})#user data used for storing snapshot of BRIR dict 
                                dpg.add_text(default_value=loaded_values["qc_e_apo_sel_brir_set_ts"], tag='qc_e_apo_sel_brir_set_ts',show=False)#timestamp of last brir dataset export
                    
                    #right most section
                    with dpg.group():    
                        #Section for channel config, plotting
                        with dpg.child_window(width=590, height=440):
                            with dpg.group(width=590):
                                with dpg.tab_bar(tag='qc_inner_tab_bar'):
                                    
                                    with dpg.tab(label="Channel Configuration", parent="qc_inner_tab_bar",tag='qc_cc_tab'):
                                        with dpg.group(horizontal=True):
                                            with dpg.group():
                                                with dpg.group(horizontal=True):
                                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                    dpg.add_text("Configure Audio Channels                                                              ")
                                                    dpg.add_text("Reset Configuration:  ")
                                                    dpg.add_button(label="Reset to Default",  callback=cb.reset_channel_config)
                                                dpg.add_separator()
                                                with dpg.group(horizontal=True):
                                                    dpg.add_text("Preamplification (dB) :  ")
                                                    dpg.add_input_float(label=" ",format="%.1f", width=100,min_value=-100, max_value=20,min_clamped=True,max_clamped=True, tag='e_apo_gain_oa',default_value=loaded_values["e_apo_gain_oa"], callback=cb.e_apo_adjust_preamp)
                                                    dpg.add_text("  ")
                                                with dpg.group(horizontal=True):
                                                    dpg.add_text("Auto-Adjust Preamp: ")
                                                    dpg.add_combo(CN.AUTO_GAIN_METHODS, label="", width=160, default_value = loaded_values["e_apo_prevent_clip"],  tag='e_apo_prevent_clip', callback=cb.e_apo_config_acquire_gui)
                                                    with dpg.tooltip("e_apo_prevent_clip"):
                                                        dpg.add_text("This will auto-adjust the preamp to prevent clipping or align low/mid frequency levels for 2.0 inputs")
                                                with dpg.group(horizontal=True):
                                                    dpg.add_text("Audio Channels:  ")
                                                    dpg.add_combo(CN.AUDIO_CHANNELS, width=180, label="",  tag='e_apo_audio_channels',default_value=loaded_values["e_apo_audio_channels"], callback=cb.e_apo_select_channels_gui)
                                                with dpg.group(horizontal=True):
                                                    with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                                        borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                                        borders_outerV=True, borders_innerV=True, delay_search=True):
                                                        dpg.add_table_column(label="Input Channels")
                                                        dpg.add_table_column(label="Max. Peak Gain (dB)")
                                                        for i in range(3):
                                                            with dpg.table_row():
                                                                for j in range(3):
                                                                    if j == 0:#channels
                                                                        if i == 0:
                                                                            dpg.add_text('2.0 Stereo')
                                                                        elif i == 1:
                                                                            dpg.add_text('5.1 Surround')
                                                                        elif i == 2:
                                                                            dpg.add_text('7.1 Surround')
                                                                    elif j == 1:#peak gain
                                                                        if i == 0:
                                                                            dpg.add_text(tag='e_apo_gain_peak_2_0')
                                                                        elif i == 1:
                                                                            dpg.add_text(tag='e_apo_gain_peak_5_1')
                                                                        elif i == 2:
                                                                            dpg.add_text(tag='e_apo_gain_peak_7_1')
                                                    with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                                        borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                                        borders_outerV=True, borders_innerV=True, delay_search=True, tag='upmixing_table'):
                                                        #dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                        dpg.add_table_column(label="Upmixing Parameter")
                                                        dpg.add_table_column(label="Value")
                                                        for i in range(3):
                                                            with dpg.table_row():
                                                                for j in range(2):
                                                                    if j == 0:#channels
                                                                        if i == 0:
                                                                            dpg.add_text('Upmixing Method',tag='e_apo_upmix_method_text')
                                                                            with dpg.tooltip("e_apo_upmix_method_text"):
                                                                                dpg.add_text('Method A performs simple channel duplication')
                                                                                dpg.add_text('Method B includes Mid/Side channel separation')
                                                                        elif i == 1:
                                                                            dpg.add_text('Side Delay (samples)',tag='e_apo_side_delay_text')
                                                                            with dpg.tooltip("e_apo_side_delay_text"):
                                                                                dpg.add_text('Side channels are delayed by specified samples')
                                                                        elif i == 2:
                                                                            dpg.add_text('Rear Delay (samples)',tag='e_apo_rear_delay_text')
                                                                            with dpg.tooltip("e_apo_rear_delay_text"):
                                                                                dpg.add_text('Rear channels are delayed by specified samples')
                                                                    elif j == 1:#peak gain
                                                                        if i == 0:
                                                                            dpg.add_combo(CN.UPMIXING_METHODS, width=100, label="",  tag='e_apo_upmix_method',default_value=loaded_values["e_apo_upmix_method"], callback=cb.e_apo_config_acquire_gui)
                                                                        elif i == 1:
                                                                            dpg.add_input_int(label=" ", width=100,min_value=-100, max_value=100, tag='e_apo_side_delay',min_clamped=True,max_clamped=True, default_value=loaded_values["e_apo_side_delay"], callback=cb.e_apo_config_acquire_gui)
                                                                        elif i == 2:
                                                                            dpg.add_input_int(label=" ", width=100,min_value=-100, max_value=100, tag='e_apo_rear_delay',min_clamped=True,max_clamped=True, default_value=loaded_values["e_apo_rear_delay"], callback=cb.e_apo_config_acquire_gui)  
                                                                        
                                                with dpg.group(horizontal=True):
                                                    with dpg.group():
                                                        with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                                            borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                                            borders_outerV=True, borders_innerV=True, delay_search=True):
                                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                            dpg.add_table_column(label="Channel")
                                                            dpg.add_table_column(label="Mute")
                                                            dpg.add_table_column(label="Gain (dB)")
                                                            dpg.add_table_column(label="Elevation (°)")
                                                            dpg.add_table_column(label="Azimuth (°)")
    
                                                            for i in range(7):
                                                                with dpg.table_row():
                                                                    for j in range(5):
                                                                        if j == 0:#channel
                                                                            if i == 0:
                                                                                dpg.add_text("L")
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                            elif i == 1:
                                                                                dpg.add_text("R")
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                            elif i == 2:
                                                                                dpg.add_text("C + SUB")
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                            elif i == 3:
                                                                                dpg.add_text("SL")
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                            elif i == 4:
                                                                                dpg.add_text("SR")
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                            elif i == 5:
                                                                                dpg.add_text("RL")
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                            elif i == 6:
                                                                                dpg.add_text("RR")
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                        if j == 1:#Mute
                                                                            if i == 0:
                                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_fl',default_value=loaded_values["e_apo_mute_fl"], callback=cb.e_apo_config_acquire_gui)
                                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                            elif i == 1:
                                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_fr',default_value=loaded_values["e_apo_mute_fr"], callback=cb.e_apo_config_acquire_gui)
                                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                            elif i == 2:
                                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_c',default_value=loaded_values["e_apo_mute_c"], callback=cb.e_apo_config_acquire_gui)
                                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                            elif i == 3:
                                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_sl',default_value=loaded_values["e_apo_mute_sl"], callback=cb.e_apo_config_acquire_gui)
                                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                            elif i == 4:
                                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_sr',default_value=loaded_values["e_apo_mute_sr"], callback=cb.e_apo_config_acquire_gui)
                                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                            elif i == 5:
                                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_rl',default_value=loaded_values["e_apo_mute_rl"], callback=cb.e_apo_config_acquire_gui)
                                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                            elif i == 6:
                                                                                dpg.add_selectable(label=" M ", width=30,tag='e_apo_mute_rr',default_value=loaded_values["e_apo_mute_rr"], callback=cb.e_apo_config_acquire_gui)
                                                                                dpg.bind_item_theme(dpg.last_item(), "__theme_c")
                                                                        if j == 2:#gain
                                                                            if i == 0:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_fl',min_clamped=True,max_clamped=True, default_value=loaded_values["e_apo_gain_fl"], callback=cb.e_apo_config_acquire_gui)
                                                                            elif i == 1:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_fr',min_clamped=True,max_clamped=True,default_value=loaded_values["e_apo_gain_fr"], callback=cb.e_apo_config_acquire_gui)
                                                                            elif i == 2:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_c',min_clamped=True,max_clamped=True,default_value=loaded_values["e_apo_gain_c"], callback=cb.e_apo_config_acquire_gui)
                                                                            elif i == 3:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_sl',min_clamped=True,max_clamped=True,default_value=loaded_values["e_apo_gain_sl"], callback=cb.e_apo_config_acquire_gui)
                                                                            elif i == 4:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_sr',min_clamped=True,max_clamped=True,default_value=loaded_values["e_apo_gain_sr"], callback=cb.e_apo_config_acquire_gui)
                                                                            elif i == 5:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_rl',min_clamped=True,max_clamped=True,default_value=loaded_values["e_apo_gain_rl"], callback=cb.e_apo_config_acquire_gui)
                                                                            elif i == 6:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='e_apo_gain_rr',min_clamped=True,max_clamped=True,default_value=loaded_values["e_apo_gain_rr"], callback=cb.e_apo_config_acquire_gui)
                                                                        if j == 3:#elevation
                                                                            if i == 0:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_fl',default_value=loaded_values["e_apo_elev_angle_fl"], callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_fl"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 1:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_fr',default_value=loaded_values["e_apo_elev_angle_fr"], callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_fr"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 2:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_c',default_value=loaded_values["e_apo_elev_angle_c"], callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_c"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 3:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_sl',default_value=loaded_values["e_apo_elev_angle_sl"], callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_sl"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 4:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_sr',default_value=loaded_values["e_apo_elev_angle_sr"], callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_sr"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 5:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_rl',default_value=loaded_values["e_apo_elev_angle_rl"], callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_rl"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 6:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_rr',default_value=loaded_values["e_apo_elev_angle_rr"], callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_rr"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                        if j == 4:#azimuth
                                                                            if i == 0:
                                                                                dpg.add_combo(CN.AZ_ANGLES_FL_WAV, width=70, label="",  tag='e_apo_az_angle_fl',default_value=loaded_values["e_apo_az_angle_fl"], callback=cb.e_apo_activate_direction_gui)
                                                                            elif i == 1:
                                                                                dpg.add_combo(CN.AZ_ANGLES_FR_WAV, width=70, label="",  tag='e_apo_az_angle_fr',default_value=loaded_values["e_apo_az_angle_fr"], callback=cb.e_apo_activate_direction_gui)
                                                                            elif i == 2:
                                                                                dpg.add_combo(CN.AZ_ANGLES_C_WAV, width=70, label="",  tag='e_apo_az_angle_c',default_value=loaded_values["e_apo_az_angle_c"], callback=cb.e_apo_activate_direction_gui)
                                                                            elif i == 3:
                                                                                dpg.add_combo(CN.AZ_ANGLES_SL_WAV, width=70, label="",  tag='e_apo_az_angle_sl',default_value=loaded_values["e_apo_az_angle_sl"], callback=cb.e_apo_activate_direction_gui)
                                                                            elif i == 4:
                                                                                dpg.add_combo(CN.AZ_ANGLES_SR_WAV, width=70, label="",  tag='e_apo_az_angle_sr',default_value=loaded_values["e_apo_az_angle_sr"], callback=cb.e_apo_activate_direction_gui)
                                                                            elif i == 5:
                                                                                dpg.add_combo(CN.AZ_ANGLES_RL_WAV, width=70, label="",  tag='e_apo_az_angle_rl',default_value=loaded_values["e_apo_az_angle_rl"], callback=cb.e_apo_activate_direction_gui)
                                                                            elif i == 6:
                                                                                dpg.add_combo(CN.AZ_ANGLES_RR_WAV, width=70, label="",  tag='e_apo_az_angle_rr',default_value=loaded_values["e_apo_az_angle_rr"], callback=cb.e_apo_activate_direction_gui)
                                       
                                                    with dpg.drawlist(width=250, height=200, tag="channel_drawing"):
                                                        with dpg.draw_layer():
    
                                                            dpg.draw_circle([CN.X_START, CN.Y_START], CN.RADIUS, color=[163, 177, 184])           
                                                            with dpg.draw_node(tag="listener_drawing"):
                                                                dpg.apply_transform(dpg.last_item(), dpg.create_translation_matrix([CN.X_START, CN.Y_START]))
                                                                dpg.draw_image('listener_image',[-23, -23],[23, 23])
                                                                
                                                                with dpg.draw_node(tag="fl_drawing"):
                                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(loaded_values["e_apo_az_angle_fl"]*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
                                                                    with dpg.draw_node(tag="fl_drawing_inner", user_data=45.0):
                                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(loaded_values["e_apo_az_angle_fl"]*-1))/180.0 , [0, 0, -1]))
                                                                        dpg.draw_image('fl_image',[-12, -12],[12, 12])
                                                                        
                                                                with dpg.draw_node(tag="fr_drawing"):
                                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(loaded_values["e_apo_az_angle_fr"]*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
                                                                    with dpg.draw_node(tag="fr_drawing_inner", user_data=45.0):
                                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(loaded_values["e_apo_az_angle_fr"]*-1))/180.0 , [0, 0, -1]))
                                                                        dpg.draw_image('fr_image',[-12, -12],[12, 12])
                                                                
                                                                with dpg.draw_node(tag="c_drawing"):
                                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(loaded_values["e_apo_az_angle_c"]*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
                                                                    with dpg.draw_node(tag="c_drawing_inner", user_data=45.0):
                                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(loaded_values["e_apo_az_angle_c"]*-1))/180.0 , [0, 0, -1]))
                                                                        dpg.draw_image('c_image',[-12, -12],[12, 12])
                                                                        
                                                                with dpg.draw_node(tag="sl_drawing"):
                                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(loaded_values["e_apo_az_angle_sl"]*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
                                                                    with dpg.draw_node(tag="sl_drawing_inner", user_data=45.0):
                                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(loaded_values["e_apo_az_angle_sl"]*-1))/180.0 , [0, 0, -1]))
                                                                        dpg.draw_image('sl_image',[-12, -12],[12, 12])
                                                                        
                                                                with dpg.draw_node(tag="sr_drawing"):
                                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(loaded_values["e_apo_az_angle_sr"]*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
                                                                    with dpg.draw_node(tag="sr_drawing_inner", user_data=45.0):
                                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(loaded_values["e_apo_az_angle_sr"]*-1))/180.0 , [0, 0, -1]))
                                                                        dpg.draw_image('sr_image',[-12, -12],[12, 12])
                                                                        
                                                                with dpg.draw_node(tag="rl_drawing"):
                                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(loaded_values["e_apo_az_angle_rl"]*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
                                                                    with dpg.draw_node(tag="rl_drawing_inner", user_data=45.0):
                                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(loaded_values["e_apo_az_angle_rl"]*-1))/180.0 , [0, 0, -1]))
                                                                        dpg.draw_image('rl_image',[-12, -12],[12, 12])
                                                                        
                                                                with dpg.draw_node(tag="rr_drawing"):
                                                                    dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+(loaded_values["e_apo_az_angle_rr"]*-1))/180.0 , [0, 0, -1])*dpg.create_translation_matrix([CN.RADIUS, 0]))
                                                                    with dpg.draw_node(tag="rr_drawing_inner", user_data=45.0):
                                                                        dpg.apply_transform(dpg.last_item(), dpg.create_rotation_matrix(math.pi*(90.0+180-(loaded_values["e_apo_az_angle_rr"]*-1))/180.0 , [0, 0, -1]))
                                                                        dpg.draw_image('rr_image',[-12, -12],[12, 12])
                                                
                                                  
                      
                                    with dpg.tab(label="Filter Preview", parent="qc_inner_tab_bar",tag='qc_fp_tab'):
                                        #plotting
                                        #dpg.add_separator()
                                        with dpg.group(horizontal=True):
                                            dpg.add_text("Select a filter from list to preview")
                                            dpg.add_text("                                                       ")  
                                            dpg.add_text("Plot Type: ")    
                                            dpg.add_combo(CN.PLOT_TYPE_LIST, default_value=CN.PLOT_TYPE_LIST[0], width=150,callback=cb.plot_type_changed, tag='qc_plot_type',user_data={})# plot_state will live here
                                        
                                        # create plot
                                        with dpg.plot(label="Filter Analysis", height=375, width=575):
                                            # optionally create legend
                                            dpg.add_plot_legend()
                                    
                                            # REQUIRED: create x and y axes
                                            dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)", tag="qc_x_axis", scale=dpg.mvPlotScale_Log10)#, log_scale=True
                                            dpg.set_axis_limits("qc_x_axis", 10, 20000)
                                            dpg.add_plot_axis(dpg.mvYAxis, label="Magnitude (dB)", tag="qc_y_axis")
                                            dpg.set_axis_limits("qc_y_axis", -20, 15)
                                    
                                            # series belong to a y axis
                                            dpg.add_line_series(default_x, default_y, label="Plot", parent="qc_y_axis", tag="qc_series_tag",skip_nan=True)
                                            #initial plot
                                            hf.plot_fir_generic(fir_array=impulse,view=CN.PLOT_TYPE_LIST[0],title_name='',samp_freq=CN.SAMP_FREQ,n_fft=CN.N_FFT,plot_dest=CN.TAB_QC_CODE)
                                        
                                    
                                    with dpg.tab(label="Integration Analysis", parent="qc_inner_tab_bar",tag='qc_ia_tab'):
                                        #plotting
                                        #dpg.add_separator()
                                        with dpg.group(horizontal=True):
                                            dpg.add_text("Refreshes after parameters are applied",tag='qc_ia_tab_text')
                                            with dpg.tooltip("qc_ia_tab_text"):
                                                dpg.add_text("Sample response is taken from 0° elevation, -30° azimuth, left ear")
                                            dpg.add_text("                                             ")
                                            dpg.add_text("Plot Type: ")    
                                            dpg.add_combo(CN.PLOT_TYPE_LIST, default_value=CN.PLOT_TYPE_LIST[0], width=150,callback=cb.ia_plot_type_changed, tag='qc_ia_plot_type',user_data={})# plot_state will live here
                                 
                                        # create plot
                                        with dpg.plot(label="Integrated Binaural Response", height=375, width=575):
                                            # optionally create legend
                                            dpg.add_plot_legend(tag="ia_legend_tag")
                                            
                                            # REQUIRED: create x and y axes
                                            dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)", tag="qc_ia_x_axis", scale=dpg.mvPlotScale_Log10)#, log_scale=True
                                            dpg.set_axis_limits("qc_x_axis", 10, 20000)
                                            dpg.add_plot_axis(dpg.mvYAxis, label="Magnitude (dB)", tag="qc_ia_y_axis")
                                            dpg.set_axis_limits("qc_y_axis", -20, 15)
                                            
                                    
                                            # series belong to a y axis
                                            dpg.add_line_series(default_x, default_y, label="Plot", parent="qc_ia_y_axis", tag="qc_ia_series_tag",skip_nan=True)
                                            #initial plot
                                            hf.plot_fir_generic(fir_array=impulse,view=CN.PLOT_TYPE_LIST[0],title_name='',samp_freq=CN.SAMP_FREQ,n_fft=CN.N_FFT,plot_dest=CN.TAB_QC_IA_CODE)
                                            
                                    with dpg.tab(label="Presets", parent="qc_inner_tab_bar",tag='qc_preset_tab'):
                                        dpg.add_text("Manage Presets")
                                        dpg.add_separator()
                                        with dpg.group(horizontal=True):
          
                                            dpg.add_button(label="Load Preset", tag="qc_load_selected_preset", callback=cb.qc_load_selected_preset_callback)
                                            dpg.add_text("     ")
                                            dpg.add_button(label="Save Preset", tag="qc_save_preset", callback=cb.qc_save_preset_callback)
                                            with dpg.tooltip("qc_save_preset"):
                                                dpg.add_text("Saves current parameters as a preset")
                                            dpg.add_text("     ")
                                            dpg.add_button(label="Delete Preset", tag="qc_delete_selected_preset", callback=lambda: dpg.configure_item("qc_del_preset_popup", show=True))
                                            with dpg.tooltip("qc_delete_selected_preset"):
                                                dpg.add_text("Deletes the selected preset")
                                            dpg.add_text("     ")
                                            dpg.add_button(label="Rename Preset", tag="qc_rename_selected_preset", callback = cb.qc_rename_selected_preset_callback)
                                            with dpg.tooltip("qc_rename_selected_preset"):
                                                dpg.add_text("Enter new name in the text box")
                                            dpg.add_input_text(label="", tag="qc_rename_selected_preset_text", width=190)
                                
                                        with dpg.popup("qc_delete_selected_preset", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="qc_del_preset_popup"):
                                            dpg.add_text("Selected preset will be deleted")
                                            dpg.add_separator()
                                            with dpg.group(horizontal=True):
                                                dpg.add_button(label="OK", width=75, callback=cb.qc_delete_selected_preset_callback)
                                                dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("qc_del_preset_popup", show=False))
                                        dpg.add_listbox(
                                            items=[],
                                            tag="qc_preset_list",
                                            width=574,
                                            num_items=18
                                        )
                                        dpg.bind_item_font(dpg.last_item(), font_b_small)
 
                                    
                     
                        #Section for misc settings
                        with dpg.group():
                            #section to store e apo config path
                            dpg.add_text(tag='qc_selected_folder_base', show=False)
                            dpg.set_value('qc_selected_folder_base', qc_primary_path)
                            #Section for wav settings
                            with dpg.child_window(width=590, height=166):
                                title_4 = dpg.add_text("IR Format", tag='qc_export_title')
                                with dpg.tooltip("qc_export_title"):
                                    dpg.add_text("Configure sample rate and bit depth of exported impulse responses")
                                    dpg.add_text("This should align with the format of the playback audio device")
                                dpg.bind_item_font(title_4, font_b_def)
                                #dpg.add_separator()
                                with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                    borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                    borders_outerV=True, borders_innerV=True, delay_search=True):
                                    #dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    dpg.add_table_column(label="  File Format  ")
                                    dpg.add_table_column(label="   Sample Rate")
                                    dpg.add_table_column(label="   Bit Depth")
                                    for i in range(1):
                                        with dpg.table_row():
                                            for j in range(3):
                                                if j == 0:#File Format
                                                    dpg.add_text("   WAV")
                                                elif j == 1:#Sample Rate
                                                    with dpg.group(horizontal=True):
                                                        dpg.add_text("  ")
                                                        dpg.add_radio_button(CN.SAMPLE_RATE_LIST, horizontal=True, tag= "qc_wav_sample_rate", default_value=loaded_values["qc_wav_sample_rate"], callback=cb.sync_wav_sample_rate )
                                                        dpg.add_text("  ")
                                                elif j == 2:#Bit Depth
                                                    with dpg.group(horizontal=True):
                                                        dpg.add_text("  ")
                                                        dpg.add_radio_button(CN.BIT_DEPTH_LIST, horizontal=True, tag= "qc_wav_bit_depth", default_value=loaded_values["qc_wav_bit_depth"], callback=cb.sync_wav_bit_depth)
                                                        dpg.add_text("  ")
      
                                dpg.add_separator()
                                with dpg.group(horizontal=True):
                                    dpg.add_text("Audio Device Format", tag='qc_device_config_title')
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    with dpg.tooltip("qc_device_config_title"):
                                        dpg.add_text("Ensure the playback audio device sample rate aligns with the IR sample rate above")
                                        #dpg.add_text("Requires restart to refresh below table")
                                    dpg.add_text("                                                                                  ")
                                    dpg.add_button(label="Open Sound Control Panel", callback=cb.open_sound_control_panel)
                
                                with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                    borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                    borders_outerV=True, borders_innerV=True, delay_search=True):
                                    #dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    dpg.add_table_column(label="Default Playback Device Name  ", init_width_or_weight=350)
                                    dpg.add_table_column(label="Default Playback Sample Rate  ", init_width_or_weight=180)
                                    for i in range(1):
                                        with dpg.table_row():
                                            for j in range(2):
                                                if j == 0:#device name
                                                    dpg.add_text(" ", tag= "qc_def_pb_device_name")
                                                elif j == 1:#Sample Rate
                                                    dpg.add_text(" ", tag= "qc_def_pb_device_sr")
                                                    
                            
     

            with dpg.tab(label="Filter & Dataset Export",tag='filter_management', parent="tab_bar"):
                dpg.bind_item_theme(dpg.last_item(), "__theme_i")
                dpg.add_text("Export Headphone Correction Filters & Binaural Room Simulation Datasets")
                #dpg.bind_item_font(dpg.last_item(), font_b_large)
                dpg.add_separator()
            
                with dpg.group(horizontal=True):
                    #Section for HpCF Export
                    with dpg.child_window(width=550, height=610):
                        title_1 = dpg.add_text("Headphone Correction Filters")
                        dpg.bind_item_font(title_1, font_b_med)
                        with dpg.child_window(autosize_x=True, height=390):
                            with dpg.group(horizontal=True):
                                dpg.add_text("Headphone Database: ")
                                dpg.add_combo(CN.HPCF_DATABASE_LIST,  tag= "fde_hpcf_active_database", default_value=loaded_values["fde_hpcf_active_database"], width=140, callback=cb.change_hpcf_database_callback)
                            
                         
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_text("Search Brand:", tag='fde_hpcf_brand_search_title')
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                dpg.add_input_text(label="", callback=cb.filter_brand_list, width=105, tag='fde_hpcf_brand_search')
                                dpg.add_text("Search Headphone:", tag='fde_hpcf_headphone_search_title')
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                dpg.add_input_text(label="", callback=cb.filter_headphone_list, width=209, tag='fde_hpcf_headphone_search')
                            with dpg.group(horizontal=True, width=0):
                                with dpg.group():
                                    dpg.add_text("Brand", tag='fde_hpcf_brand_title')
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    dpg.add_listbox(fde_brands_list_loaded, width=135, num_items=16, tag='fde_hpcf_brand', default_value=loaded_values["fde_hpcf_brand"], callback=cb.update_headphone_list)
                                with dpg.group():
                                    dpg.add_text("Headphone", tag='fde_hpcf_headphone_title')
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    dpg.add_listbox(fde_hp_list_loaded, width=250, num_items=16, tag='fde_hpcf_headphone', default_value=loaded_values["fde_hpcf_headphone"] ,callback=cb.update_sample_list)
                                with dpg.group():
                                    dpg.add_text("Sample", tag='fde_hpcf_sample_title')
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    dpg.add_listbox(fde_sample_list_loaded, width=115, num_items=16, tag='fde_hpcf_sample', default_value=loaded_values["fde_hpcf_sample"], callback=cb.plot_sample)
                                    with dpg.tooltip("fde_hpcf_sample"):
                                        dpg.add_text("Note: all samples will be exported. Select a sample to preview")
                        with dpg.child_window(autosize_x=True, height=88):
                            subtitle_2 = dpg.add_text("Select Files to Include in Export")
                            dpg.bind_item_font(subtitle_2, font_b_def)
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="WAV FIR Filters", default_value = loaded_values["fde_fir_hpcf_toggle"], callback=cb.fde_export_hpcf_file_toggle, tag='fde_fir_hpcf_toggle')
                                with dpg.tooltip("fde_fir_hpcf_toggle"):
                                    dpg.add_text("Min phase FIRs in WAV format for convolution. 1 Channel")
                                    dpg.add_text("Required for Equalizer APO configuration updates")
                                dpg.add_checkbox(label="WAV Stereo FIR Filters", default_value = loaded_values["fde_fir_st_hpcf_toggle"], callback=cb.fde_export_hpcf_file_toggle, tag='fde_fir_st_hpcf_toggle')
                                with dpg.tooltip("fde_fir_st_hpcf_toggle"):
                                    dpg.add_text("Min phase FIRs in WAV format for convolution. 2 Channels")
                                dpg.add_checkbox(label="HeSuVi Filters", default_value = loaded_values["fde_hesuvi_hpcf_toggle"], callback=cb.fde_export_hpcf_file_toggle, tag='fde_hesuvi_hpcf_toggle')
                                with dpg.tooltip("fde_hesuvi_hpcf_toggle"):
                                    dpg.add_text("Graphic EQ configurations with 103 bands. Compatible with HeSuVi. Saved in HeSuVi\\eq folder")
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="Graphic EQ Filters (127 Bands)", default_value = loaded_values["fde_geq_hpcf_toggle"], callback=cb.fde_export_hpcf_file_toggle, tag='fde_geq_hpcf_toggle')
                                with dpg.tooltip("fde_geq_hpcf_toggle"):
                                    dpg.add_text("Graphic EQ configurations with 127 bands. Compatible with Equalizer APO and Wavelet")
                                dpg.add_checkbox(label="Graphic EQ Filters (31 Bands)", default_value = loaded_values["fde_geq_31_hpcf_toggle"], callback=cb.fde_export_hpcf_file_toggle, tag='fde_geq_31_hpcf_toggle')
                                with dpg.tooltip("fde_geq_31_hpcf_toggle"):
                                    dpg.add_text("Graphic EQ configurations with 31 bands. Compatible with 31 band graphic equalizers including Equalizer APO")
                                
                        with dpg.child_window(autosize_x=True, height=82):
                            subtitle_3 = dpg.add_text("Export Correction Filters")
                            dpg.bind_item_font(subtitle_3, font_b_def)
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
               
                                dpg.add_image_button(texture_tag='start_green',width=37,height=37,tag="fde_hpcf_tag",callback=cb.fde_process_hpcfs,frame_padding=0,background_color=[0, 0, 0, 0])
                                dpg.bind_item_theme(dpg.last_item(), "transparent_image_button_theme")
                                
                                with dpg.tooltip("fde_hpcf_tag"):
                                    dpg.add_text("This will export the selected filters to the output directory")
                                    
                                dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=32, width=470, overlay=CN.PROGRESS_START_ALT, tag="progress_bar_hpcf")
                                dpg.bind_item_theme(dpg.last_item(), "__theme_g")
    
                    #Section for BRIR generation
                    with dpg.child_window(width=520, height=610):
                        title_2 = dpg.add_text("Binaural Room Simulation", tag='brir_title')
                        dpg.bind_item_font(title_2, font_b_med)
                        with dpg.tooltip("brir_title"):
                            dpg.add_text("Customise a new binaural room simulation using below parameters")
                        with dpg.child_window(autosize_x=True, height=388):
                            with dpg.tab_bar(tag='brir_tab_bar'):

                                with dpg.tab(label="Acoustics & EQ Parameters",tag='acoustics_eq_tab', parent="brir_tab_bar"): 
  
                                    #dpg.add_separator()
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Acoustic Space",tag='acoustic_space_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            with dpg.group(horizontal=True):
                                                dpg.add_text("Sort by: ")
                                                dpg.add_combo(CN.AC_SPACE_LIST_SORT_BY, width=140, label="",  tag='sort_by_as',default_value=CN.AC_SPACE_LIST_SORT_BY[0], callback=cb.fde_sort_ac_space)
                                            dpg.add_listbox(CN.AC_SPACE_LIST_GUI, label="",tag='fde_acoustic_space',default_value=loaded_values["fde_acoustic_space"], callback=cb.fde_update_ac_space, num_items=16, width=250)
                                            with dpg.tooltip("acoustic_space_title"):
                                                dpg.add_text("This will determine the listening environment")
                                            
                                            
                                        with dpg.group():
                                            dpg.add_text("Direct Sound Gain (dB)", tag='direct_gain_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_input_float(label="Direct Gain (dB)",width=140, format="%.1f", tag='fde_direct_gain', min_value=CN.DIRECT_GAIN_MIN, max_value=CN.DIRECT_GAIN_MAX, default_value=loaded_values["fde_direct_gain"],min_clamped=True, max_clamped=True, callback=cb.fde_update_direct_gain)
                                            with dpg.tooltip("direct_gain_title"):
                                                dpg.add_text("This will control the loudness of the direct signal")
                                                dpg.add_text("Higher values result in lower perceived distance, lower values result in higher perceived distance")
                                            dpg.add_slider_float(label="", min_value=CN.DIRECT_GAIN_MIN, max_value=CN.DIRECT_GAIN_MAX, default_value=loaded_values["fde_direct_gain_slider"], width=140,clamped=True, no_input=True, format="", callback=cb.fde_update_direct_gain_slider, tag='fde_direct_gain_slider')
                                            dpg.add_separator()
                                            dpg.add_text("Room Target", tag='rm_target_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_listbox(CN.ROOM_TARGET_LIST, default_value=loaded_values["fde_room_target"], num_items=6, width=235, tag='fde_room_target', callback=cb.fde_select_room_target)
                                            with dpg.tooltip("rm_target_title"):
                                                dpg.add_text("This will influence the overall balance of low and high frequencies")
                                            dpg.add_separator()
                                            dpg.add_text("Headphone Compensation", tag='brir_hp_type_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_listbox(CN.HP_COMP_LIST, default_value=loaded_values["fde_brir_hp_type"], num_items=4, width=235, callback=cb.fde_select_hp_comp, tag='fde_brir_hp_type')
                                            with dpg.tooltip("brir_hp_type_title"):
                                                dpg.add_text("This will compensate typical interactions between the headphone and the outer ear")
                                                dpg.add_text("Selection should align with the listener's headphone type")
                                                dpg.add_text("Reduce to low strength if sound localisation or timbre is compromised")
                                                
                                with dpg.tab(label="Listener Selection",tag='listener_tab', parent="brir_tab_bar"): 
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Spatial Resolution", tag= "brir_spat_res_title")
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_radio_button(spatial_res_list_loaded, horizontal=True, tag= "fde_brir_spat_res", default_value=loaded_values["fde_brir_spat_res"], callback=cb.fde_select_spatial_resolution )
                                            with dpg.tooltip("brir_spat_res_title"):
                                                dpg.add_text("Increasing resolution will increase number of directions available but will also increase processing time and dataset size")
                                                dpg.add_text("'Low' is recommended unless additional directions or SOFA export is required")
                                                dpg.add_text("Refer to reference tables tab for comparison")
                                            dpg.add_separator()    
                                            dpg.add_text("Category", tag='brir_hrtf_type_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_radio_button(CN.HRTF_TYPE_LIST, horizontal=False, tag= "fde_brir_hrtf_type", default_value=loaded_values["fde_brir_hrtf_type"], callback=cb.fde_update_hrtf_dataset_list )
                                            with dpg.tooltip("brir_hrtf_type_title"):
                                                dpg.add_text("User SOFA files must be placed in 'ASH Toolset\\_internal\\data\\user\\SOFA' folder")
                                            dpg.add_separator()
                                            dpg.add_text("Dataset", tag='brir_hrtf_dataset_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_listbox(fde_brir_hrtf_dataset_list_loaded, default_value=fde_brir_hrtf_dataset_loaded, num_items=7, width=255, callback=cb.fde_update_hrtf_list, tag='fde_brir_hrtf_dataset')
                                        with dpg.group():
                                            dpg.add_loading_indicator(style=1, pos = (480,355), radius =1.8, color =(120,120,120),show=False, tag='hrtf_average_fav_load_ind')
                                            dpg.add_text("Listener", tag='brir_hrtf_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_listbox(hrtf_list_loaded, default_value=fde_hrtf_loaded, num_items=16, width=230, callback=cb.fde_select_hrtf, tag='fde_brir_hrtf')
                                            with dpg.tooltip("brir_hrtf_title"):
                                                dpg.add_text("This will influence the externalisation and localisation of sounds around the listener")
                                            with dpg.group(horizontal=True):
                                                dpg.add_text("")
                                                dpg.add_button(label="Add Favourite", tag="fde_hrtf_add_favourite", callback=cb.add_hrtf_favourite_callback,user_data=loaded_values["hrtf_list_favs"],show=add_fav_button_show)
                                                #dpg.add_text("     ")
                                                dpg.add_button(label="Remove Favourite", tag="fde_hrtf_remove_favourite", callback=cb.remove_hrtf_favourite_callback,show=remove_fav_button_show)
                                                dpg.add_text("  ")
                                                dpg.add_button(label="Create Average", tag="fde_hrtf_average_favourite", callback=cb.create_hrtf_favourite_avg,show=remove_fav_button_show)
                                                dpg.bind_item_theme(dpg.last_item(), "__theme_l")
                                                with dpg.tooltip("fde_hrtf_average_favourite"):
                                                    dpg.add_text("Creates an averaged HRTF by interpolating across favourite listeners")
                                                dpg.add_button(label="Open Folder", tag="fde_open_user_sofa_folder", callback=cb.open_user_sofa_folder,show=sofa_folder_button_show)
                                        
                                with dpg.tab(label="Low-frequency Extension",tag='lfe_tab', parent="brir_tab_bar"): 
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Integration Crossover Frequency", tag='crossover_f_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_combo(CN.SUB_FC_SETTING_LIST, default_value=loaded_values["fde_crossover_f_mode"], width=130, callback=cb.fde_update_crossover_f, tag='fde_crossover_f_mode')
                                            with dpg.tooltip("fde_crossover_f_mode"):
                                                dpg.add_text("Auto Select mode will select an optimal frequency for the selected acoustic space")
                                                
                                            dpg.add_input_int(label="Crossover Frequency (Hz)",width=140, tag='fde_crossover_f', min_value=CN.SUB_FC_MIN, max_value=CN.SUB_FC_MAX, default_value=loaded_values["fde_crossover_f"],min_clamped=True, max_clamped=True, callback=cb.fde_update_crossover_f)
                                            with dpg.tooltip("fde_crossover_f"):
                                                dpg.add_text("Crossover Frequency can be adjusted to a value up to 150Hz")
                                                dpg.add_text("This will tune the integration of the cleaner LF response and original room response")
                                                dpg.add_text("Higher values may result in a smoother bass response")
                                                dpg.add_text("Set to 0Hz to disable this feature")
                                            dpg.add_separator()
                                            dpg.add_text("Response for Low-frequency Extension", tag='sub_response_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_listbox(CN.SUB_RESPONSE_LIST_GUI, default_value=loaded_values["fde_sub_response"], num_items=10, width=350, callback=cb.fde_select_sub_brir, tag='fde_sub_response')
                                            with dpg.tooltip("sub_response_title"):
                                                dpg.add_text("Refer to reference tables tab and filter preview for comparison")
                                            dpg.add_separator()
                                            dpg.add_text("Additonal EQ", tag='hp_rolloff_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            with dpg.group(horizontal=True):
                                                dpg.add_checkbox(label="Compensate Headphone Roll-off", default_value = loaded_values["fde_hp_rolloff_comp"],  tag='fde_hp_rolloff_comp', callback=cb.fde_update_brir_param)
                                                with dpg.tooltip("fde_hp_rolloff_comp"):
                                                    dpg.add_text("This will compensate the typical reduction in bass response at lower frequencies")
                                                dpg.add_text("      ")
                                                dpg.add_checkbox(label="Forward-Backward Filtering", default_value = loaded_values["fde_fb_filtering"],  tag='fde_fb_filtering', callback=cb.fde_update_brir_param)
                                                with dpg.tooltip("fde_fb_filtering"):
                                                    dpg.add_text("This will eliminate delay introduced by the filters, however can introduce edge artefacts in some cases")
                                            
                                            
                                        
                        with dpg.child_window(autosize_x=True, height=88):
                            subtitle_5 = dpg.add_text("Select Files to Include in Export")
                            dpg.bind_item_font(subtitle_5, font_b_def)
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="Direction-specific WAV BRIRs", default_value = loaded_values["fde_dir_brir_toggle"],  tag='fde_dir_brir_toggle', callback=cb.fde_update_brir_param, show=dir_brir_exp_show)
                                with dpg.tooltip("fde_dir_brir_toggle", tag='dir_brir_tooltip', show=dir_brir_tooltip_show):
                                    dpg.add_text("Binaural Room Impulse Responses (BRIRs) in WAV format for convolution")
                                    dpg.add_text("2 channels per file. One file for each direction")
                                    dpg.add_text("Required for Equalizer APO configuration updates")
                                dpg.add_checkbox(label="True Stereo WAV BRIR", default_value = loaded_values["fde_ts_brir_toggle"],  tag='fde_ts_brir_toggle', callback=cb.fde_update_brir_param, show=ts_brir_exp_show)
                                with dpg.tooltip("fde_ts_brir_toggle", tag='fde_ts_brir_tooltip', show=ts_brir_tooltip_show):
                                    dpg.add_text("True Stereo BRIR in WAV format for convolution")
                                    dpg.add_text("4 channels. One file representing L and R speakers")
                                dpg.add_checkbox(label="SOFA File", default_value = loaded_values["fde_sofa_brir_toggle"],  tag='fde_sofa_brir_toggle', callback=cb.fde_update_brir_param, show=sofa_brir_exp_show)
                                with dpg.tooltip("fde_sofa_brir_toggle", tag='fde_sofa_brir_tooltip', show=sofa_brir_tooltip_show):
                                    dpg.add_text("BRIR dataset as a SOFA file")
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="HeSuVi WAV BRIRs", default_value = loaded_values["fde_hesuvi_brir_toggle"],  tag='fde_hesuvi_brir_toggle', callback=cb.fde_update_brir_param, show=hesuvi_brir_exp_show)  
                                with dpg.tooltip("fde_hesuvi_brir_toggle", tag='fde_hesuvi_brir_tooltip', show=hesuvi_brir_tooltip_show):
                                    dpg.add_text("BRIRs in HeSuVi compatible WAV format. 14 channels, 44.1kHz and 48kHz")
                                    
                                dpg.add_checkbox(label="16-Channel WAV BRIRs", default_value = loaded_values["fde_multi_chan_brir_toggle"],  tag='fde_multi_chan_brir_toggle', callback=cb.fde_update_brir_param, show=eapo_brir_exp_show)
                                with dpg.tooltip("fde_multi_chan_brir_toggle", tag='fde_multi_chan_brir_tooltip', show=multi_chan_brir_tooltip_show):
                                    dpg.add_text("BRIRs in FFMPEG compatible WAV format. 16 channels")
                        with dpg.child_window(autosize_x=True, height=83):
                            subtitle_6 = dpg.add_text("Generate and Export Binaural Dataset")
                            dpg.bind_item_font(subtitle_6, font_b_def)
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                       
                                dpg.add_image_button(texture_tag='start_green',width=37,height=37,tag="fde_brir_tag",callback=cb.fde_start_process_brirs,frame_padding=0,background_color=[0, 0, 0, 0],user_data=CN.PROCESS_BRIRS_RUNNING)
                                dpg.bind_item_theme(dpg.last_item(), "transparent_image_button_theme")
                                
                                with dpg.tooltip("fde_brir_tag"):
                                    dpg.add_text("This will generate the binaural dataset and export to the output directory. This may take some time to process")
                                    
                                dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=32, width=440, overlay=CN.PROGRESS_START_ALT, tag="progress_bar_brir",user_data=CN.STOP_THREAD_FLAG)
                                dpg.bind_item_theme(dpg.last_item(), "__theme_g")
                
                    #right most section
                    with dpg.group():    
                        #Section for plotting
                        with dpg.child_window(width=600, height=467):
                            with dpg.group(width=600):
                                with dpg.tab_bar(tag='inner_tab_bar'):
                                    with dpg.tab(label="Filter Preview", parent="inner_tab_bar",tag='fp_tab'):
                     
                                        with dpg.group(horizontal=True):
                                            dpg.add_text("Select a filter from list to preview")
                                            dpg.add_text("                                                       ")  
                                            dpg.add_text("Plot Type: ")    
                                            dpg.add_combo(CN.PLOT_TYPE_LIST, default_value=CN.PLOT_TYPE_LIST[0], width=150,callback=cb.plot_type_changed, tag='fde_plot_type',user_data={})
                                        # create plot
                                        with dpg.plot(label="Magnitude Response Plot", height=390, width=585):
                                            # optionally create legend
                                            dpg.add_plot_legend()
                                    
                                            # REQUIRED: create x and y axes
                                            dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)", tag="fde_x_axis", scale=dpg.mvPlotScale_Log10)#, log_scale=True
                                            dpg.set_axis_limits("fde_x_axis", 10, 20000)
                                            dpg.add_plot_axis(dpg.mvYAxis, label="Magnitude (dB)", tag="fde_y_axis")
                                            dpg.set_axis_limits("fde_y_axis", -20, 15)
                                    
                                            # series belong to a y axis
                                            dpg.add_line_series(default_x, default_y, label="Plot", parent="fde_y_axis", tag="fde_series_tag",skip_nan=True)
                                            #initial plot
                                            hf.plot_fir_generic(fir_array=impulse,view=CN.PLOT_TYPE_LIST[0],title_name='',samp_freq=CN.SAMP_FREQ,n_fft=CN.N_FFT,plot_dest=CN.TAB_FDE_CODE)
                                        
                                                                        
                                    with dpg.tab(label="HeSuVi & Multichannel Configuration", parent="inner_tab_bar",tag='hcc_tab'):
                                        dpg.add_text("Source directions for HeSuVi and 16-Channel exports")
                                        dpg.add_button(label="Reset to Default",  callback=cb.reset_channel_config)
                                        with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                                            borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                                            borders_outerV=True, borders_innerV=True, delay_search=True):
                                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                            dpg.add_table_column(label="Channel")
                                                            dpg.add_table_column(label="Elevation Angle (°)")
                                                            dpg.add_table_column(label="Azimuth Angle (°)")
    
                                                            for i in range(7):
                                                                with dpg.table_row():
                                                                    for j in range(3):
                                                                        if j == 0:#channel
                                                                            if i == 0:
                                                                                dpg.add_text("FL")
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                            elif i == 1:
                                                                                dpg.add_text("FR")
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                            elif i == 2:
                                                                                dpg.add_text("FC/SUB")
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                            elif i == 3:
                                                                                dpg.add_text("SL")
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                            elif i == 4:
                                                                                dpg.add_text("SR")
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                            elif i == 5:
                                                                                dpg.add_text("RL")
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                            elif i == 6:
                                                                                dpg.add_text("RR")
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                        if j == 1:#elevation
                                                                            if i == 0:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='hesuvi_elev_angle_fl',default_value=CN.DEFAULTS["e_apo_elev_angle"])
                                                                                with dpg.tooltip("hesuvi_elev_angle_fl"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 1:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='hesuvi_elev_angle_fr',default_value=CN.DEFAULTS["e_apo_elev_angle"])
                                                                                with dpg.tooltip("hesuvi_elev_angle_fr"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 2:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='hesuvi_elev_angle_c',default_value=CN.DEFAULTS["e_apo_elev_angle"])
                                                                                with dpg.tooltip("hesuvi_elev_angle_c"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 3:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='hesuvi_elev_angle_sl',default_value=CN.DEFAULTS["e_apo_elev_angle"])
                                                                                with dpg.tooltip("hesuvi_elev_angle_sl"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 4:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='hesuvi_elev_angle_sr',default_value=CN.DEFAULTS["e_apo_elev_angle"])
                                                                                with dpg.tooltip("hesuvi_elev_angle_sr"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 5:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='hesuvi_elev_angle_rl',default_value=CN.DEFAULTS["e_apo_elev_angle"])
                                                                                with dpg.tooltip("hesuvi_elev_angle_rl"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 6:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='hesuvi_elev_angle_rr',default_value=CN.DEFAULTS["e_apo_elev_angle"])
                                                                                with dpg.tooltip("hesuvi_elev_angle_rr"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                        if j == 2:#azimuth
                                                                            if i == 0:
                                                                                dpg.add_combo(CN.AZ_ANGLES_FL_WAV, width=80, label="",  tag='hesuvi_az_angle_fl',default_value=CN.DEFAULTS["e_apo_az_angle_fl"])
                                                                                with dpg.tooltip("hesuvi_az_angle_fl"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                            elif i == 1:
                                                                                dpg.add_combo(CN.AZ_ANGLES_FR_WAV, width=80, label="",  tag='hesuvi_az_angle_fr',default_value=CN.DEFAULTS["e_apo_az_angle_fr"])
                                                                                with dpg.tooltip("hesuvi_az_angle_fr"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                            elif i == 2:
                                                                                dpg.add_combo(CN.AZ_ANGLES_C_WAV, width=80, label="",  tag='hesuvi_az_angle_c',default_value=CN.DEFAULTS["e_apo_az_angle_c"])
                                                                                with dpg.tooltip("hesuvi_az_angle_c"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                            elif i == 3:
                                                                                dpg.add_combo(CN.AZ_ANGLES_SL_WAV, width=80, label="",  tag='hesuvi_az_angle_sl',default_value=CN.DEFAULTS["e_apo_az_angle_sl"])
                                                                                with dpg.tooltip("hesuvi_az_angle_sl"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                            elif i == 4:
                                                                                dpg.add_combo(CN.AZ_ANGLES_SR_WAV, width=80, label="",  tag='hesuvi_az_angle_sr',default_value=CN.DEFAULTS["e_apo_az_angle_sr"])
                                                                                with dpg.tooltip("hesuvi_az_angle_sr"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                            elif i == 5:
                                                                                dpg.add_combo(CN.AZ_ANGLES_RL_WAV, width=80, label="",  tag='hesuvi_az_angle_rl',default_value=CN.DEFAULTS["e_apo_az_angle_rl"])
                                                                                with dpg.tooltip("hesuvi_az_angle_rl"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                            elif i == 6:
                                                                                dpg.add_combo(CN.AZ_ANGLES_RR_WAV, width=80, label="",  tag='hesuvi_az_angle_rr',default_value=CN.DEFAULTS["e_apo_az_angle_rr"])
                                                                                with dpg.tooltip("hesuvi_az_angle_rr"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                                    
                                        dpg.add_separator()
                                        dpg.add_text("Channel Order for 16-Channel Exports")
                                        with dpg.group():
                                            dpg.add_listbox(label="", width=200, items=CN.PRESET_16CH_LABELS, num_items=7, default_value=CN.PRESET_16CH_LABELS[0], tag="mapping_16ch_wav")                                            
                                    
                               
                        #Section for Exporting files
                        with dpg.group(horizontal=True):
                            #Section for wav settings
                            with dpg.child_window(width=225, height=139):
                                title_4 = dpg.add_text("IR Format", tag='export_title')
                                with dpg.tooltip("export_title"):
                                    dpg.add_text("Configure sample rate and bit depth of exported impulse responses")
                                    dpg.add_text("This should align with the format of the playback audio device")
                                dpg.bind_item_font(title_4, font_b_def)
                                dpg.add_separator()
                                with dpg.group():
                                    dpg.add_text("Select Sample Rate")
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    dpg.add_radio_button(CN.SAMPLE_RATE_LIST, horizontal=True, tag= "fde_wav_sample_rate", default_value=loaded_values["fde_wav_sample_rate"], callback=cb.sync_wav_sample_rate )
                                #dpg.add_text("          ")
                                with dpg.group():
                                    dpg.add_text("Select Bit Depth")
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    dpg.add_radio_button(CN.BIT_DEPTH_LIST, horizontal=True, tag= "fde_wav_bit_depth", default_value=loaded_values["fde_wav_bit_depth"], callback=cb.sync_wav_bit_depth)
                                    
                            #output locations
                            with dpg.child_window(width=371, height=139):
                                with dpg.group(horizontal=True):
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    dpg.add_text("Output Locations", tag='out_dir_title')
                                    with dpg.tooltip("out_dir_title"):
                                        dpg.add_text("'EqualizerAPO\\config' directory should be selected if using Equalizer APO")
                                        dpg.add_text("Main outputs will be saved under 'ASH-Outputs' sub directory") 
                                        dpg.add_text("HeSuVi outputs will be saved in 'EqualizerAPO\\config\\HeSuVi'")  
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    dpg.add_text("           ")
                                    dpg.add_button(label="Open Folder",user_data="",tag="open_folder_tag", callback=cb.open_output_folder)
                                    dpg.add_text("      ")
                                    dpge.add_file_browser(width=800,height=600,label='Change Folder',show_as_window=True, dirs_only=True,show_ok_cancel=True, allow_multi_selection=False, collapse_sequences=True,callback=cb.show_selected_folder)
                                dpg.add_separator()
                                dpg.add_text("Main Outputs:")
                                dpg.bind_item_font(dpg.last_item(), font_b_small)
                                dpg.add_text(tag='selected_folder_ash')
                                dpg.bind_item_font(dpg.last_item(), font_small)
                                with dpg.tooltip("selected_folder_ash"):
                                    dpg.add_text("Location to save correction filters and binaural datasets",tag="selected_folder_ash_tooltip")
                                dpg.add_text(tag='selected_folder_base',show=False)
                                dpg.add_text(tag='e_apo_program_path',show=False)
                                dpg.add_text("HeSuVi Outputs:")
                                dpg.bind_item_font(dpg.last_item(), font_b_small)
                                dpg.add_text(tag='selected_folder_hesuvi')
                                dpg.bind_item_font(dpg.last_item(), font_small)
                                with dpg.tooltip("selected_folder_hesuvi"):
                                    dpg.add_text("Location to save HeSuVi files",tag="selected_folder_hesuvi_tooltip")
                                
                            dpg.set_value('selected_folder_ash', primary_ash_path)
                            dpg.set_value('selected_folder_ash_tooltip', primary_ash_path)
                            dpg.set_value('selected_folder_base', primary_path)
                            dpg.set_value('selected_folder_hesuvi', primary_hesuvi_path)
                            dpg.set_value('selected_folder_hesuvi_tooltip', primary_hesuvi_path)
                            dpg.set_value('e_apo_program_path', e_apo_path)
    

                        
            if CN.SHOW_AS_TAB:
                with dpg.tab(label="Acoustic Space Import", tag='as_import_tab', parent="tab_bar"):     
                    dpg.bind_item_theme(dpg.last_item(), "__theme_j")
                    dpg.add_text("Import New Acoustic Spaces from Impulse Response (IR) Files")
                    dpg.add_separator()
                
                    with dpg.group(horizontal=True):
                        # Panel 1 - Left
                        with dpg.child_window(width=400, height=610):
                            with dpg.child_window(width=385, height=470):
                                dpg.add_text("Select Inputs")
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                #dpg.add_separator()
                                
                                with dpg.group(horizontal=True):
                                    # dpg.add_text("IR Folder")
                                    # dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    # dpg.add_text("             ")
                                    dpg.add_button(label="Refresh Folder List", callback=cb.update_ir_folder_list, tag="refresh_folders_btn")
                                    with dpg.tooltip("refresh_folders_btn"):
                                        dpg.add_text("Click to reload the list of available IR folders")
                                    dpg.add_text("   ")
                                    dpg.add_button(label="Open Input Folder", tag="open_user_ir_folder_button", callback=cb.open_user_input_as_folder)
                                    with dpg.tooltip("open_user_ir_folder_button"):
                                        dpg.add_text("Click to open in explorer the folder where the user IRs are stored")
                                        dpg.add_text("IRs should be stored toether in a single subfolder, for example: 'Room A'")
                                        dpg.add_text("Supported file types: wav, sofa, mat, npy, hdf5")
                                        dpg.add_text("Files can have any number of channels and samples and any bit depth and sample rate")
                                #dpg.add_listbox(items=[], label="", tag="ir_folder_list", callback=cb.folder_selected_callback, width=360, num_items=4)
                                dpg.add_combo(items=[],label="",tag="ir_folder_list",callback=cb.folder_selected_callback,width=360)
                                with dpg.tooltip("ir_folder_list"):
                                    dpg.add_text("Choose an IR folder from the list")
                                dpg.add_text(" ")
                                dpg.add_separator()
                                dpg.add_text("Set Parameters")
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                #dpg.add_separator()
                           
        
                                with dpg.table(header_row=True,resizable=False,policy=dpg.mvTable_SizingFixedFit,row_background=True,borders_outerH=True,borders_innerH=True,borders_outerV=True,borders_innerV=True,):
                                    dpg.add_table_column(label="Parameter", init_width_or_weight=150)
                                    dpg.add_table_column(label="Value", init_width_or_weight=260)
                            
                                    # ---------- Name ----------
                                    with dpg.table_row():
                                        dpg.add_text("Name (optional)")
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                            
                                        dpg.add_input_text(tag="space_name", width=200)
                                        with dpg.tooltip("space_name"):
                                            dpg.add_text(
                                                "Enter a name for the new acoustic space.\n"
                                                "If left blank, folder name will be used."
                                            )
                            
                                    # ---------- Description ----------
                                    with dpg.table_row():
                                        dpg.add_text("Description (optional)")
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                            
                                        dpg.add_input_text(tag="space_description",width=200,multiline=False)
                                        with dpg.tooltip("space_description"):
                                            dpg.add_text("Enter a brief description of the acoustic space (optional)")
                            
                                    # ---------- Long Reverb Tail ----------
                                    with dpg.table_row():
                                        dpg.add_text("Long Reverb Tail Mode")
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                            
                                        dpg.add_checkbox(label="Enable", tag="long_tail_mode")
                                        with dpg.tooltip("long_tail_mode"):
                                            dpg.add_text("Only enable if IRs have long decay tails (> 1.5 s)")
                                            dpg.add_text("This will increase processing time")
                            
                                    # ---------- Low-frequency Mode ----------
                                    with dpg.table_row():
                                        dpg.add_text("Low-frequency Mode")
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                            
                                        dpg.add_checkbox(label="Enable", tag="as_subwoofer_mode")
                                        with dpg.tooltip("as_subwoofer_mode"):
                                            dpg.add_text("Enable if IRs are low-frequency measurements")
                                            dpg.add_text("Result will appear under Low-frequency responses")
                            
                                    # ---------- Noise Reduction ----------
                                    with dpg.table_row():
                                        dpg.add_text("Noise Reduction")
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                            
                                        dpg.add_checkbox(label="Enable", tag="noise_reduction_mode")
                                        with dpg.tooltip("noise_reduction_mode"):
                                            dpg.add_text("Enable if IRs have a high noise floor")
                            
                                    # ---------- Rise Time ----------
                                    with dpg.table_row():
                                        dpg.add_text("Rise Time (ms)")
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                            
                                        dpg.add_input_float(tag="as_rise_time",width=120,default_value=5.0,min_value=1.0,max_value=20.0,format="%.2f",min_clamped=True,max_clamped=True,)
                                        with dpg.tooltip("as_rise_time"):
                                            dpg.add_text("Applies a fade-in window of specified duration")
                                            dpg.add_text("Min: 1 ms, Max: 20 ms")
                            
                                    # ---------- Desired Directions ----------
                                    with dpg.table_row():
                                        dpg.add_text("Desired Directions")
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                            
                                        dpg.add_input_int(tag="unique_directions",width=120,default_value=3000,min_value=1500,max_value=4000,)
                                        with dpg.tooltip("unique_directions"):
                                            dpg.add_text("Target number of spatial source directions")
                                            dpg.add_text("Lower values reduce processing time")
                            
                                    # ---------- Alignment Frequency ----------
                                    with dpg.table_row():
                                        dpg.add_text("Alignment Frequency (Hz)")
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                            
                                        dpg.add_input_int(tag="alignment_freq",width=120,default_value=110,min_value=50,max_value=150,)
                                        with dpg.tooltip("alignment_freq"):
                                            dpg.add_text("Low-pass cutoff used for time-domain alignment")
                                            dpg.add_text("Min: 50 Hz, Max: 150 Hz")
                            
                                    # ---------- Pitch Shift Range ----------
                                    with dpg.table_row():
                                        dpg.add_text("Pitch Shift Range")
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                            
                                        with dpg.group():
                                            dpg.add_input_float(label="Low",tag="pitch_range_low",width=110,default_value=-12.0,min_value=-48.0,max_value=0.0,format="%.2f",min_clamped=True,max_clamped=True,)
                                            dpg.add_input_float(label="High",tag="pitch_range_high",width=110,default_value=12.0,min_value=0.0,max_value=48.0,format="%.2f",min_clamped=True,max_clamped=True,)
                            
                                        with dpg.tooltip("pitch_range_low"):
                                            dpg.add_text("Minimum pitch shift (semitones)")
                                            dpg.add_text("Used to expand sparse datasets")
                            
                                        with dpg.tooltip("pitch_range_high"):
                                            dpg.add_text("Maximum pitch shift (semitones)")
                                            dpg.add_text("Used to expand sparse datasets")
                            
                                    # ---------- Pitch Shift Compensation ----------
                                    with dpg.table_row():
                                        dpg.add_text("Pitch Shift Compensation")
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                            
                                        dpg.add_checkbox(label="Enable",tag="pitch_shift_comp", default_value=True)
                                        with dpg.tooltip("pitch_shift_comp"):
                                            dpg.add_text("Corrects pitch after dataset expansion")
                            
                                    # ---------- Binaural Measurements ----------
                                    with dpg.table_row():
                                        dpg.add_text("Binaural Measurements")
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                            
                                        dpg.add_checkbox(label="Enable",tag="binaural_meas_inputs", default_value=False)
                                        with dpg.tooltip("binaural_meas_inputs"):
                                            dpg.add_text("Enable if IRs are binaural measurements")
                                            
                                    # ---------- Compensation Factor ----------
                                    with dpg.table_row():
                                        dpg.add_text("Room Correction Factor")
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                            
                                        dpg.add_input_float(label="",tag="asi_rm_cor_factor",width=110,default_value=1.0,min_value=0.0,max_value=1.0,format="%.2f",min_clamped=True,max_clamped=True,)
                                        with dpg.tooltip("asi_rm_cor_factor"):
                                            dpg.add_text("Select a value between 0 and 1 to control the strength of room correction")
                                
                    
                            with dpg.child_window(width=385, height=120):
                                dpg.add_text("Process IRs")
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                dpg.add_separator()
                    
                                with dpg.group(horizontal=True):
                                    dpg.add_text("Selected Folder:", tag="selected_folder_label")
                                    dpg.add_text("", tag="selected_folder_display")
                                    with dpg.tooltip("selected_folder_display"):
                                        dpg.add_text("Displays the selected IR folder")
                    
                                with dpg.group(horizontal=True):
                                    dpg.add_button(
                                        label="Start Processing",
                                        callback=cb.as_launch_processing_thread,
                                        tag="start_processing_btn",
                                        user_data={"ir_processing_running": False}
                                    )
                                    with dpg.tooltip("start_processing_btn"):
                                        dpg.add_text("Begin processing the selected acoustic space")
                                        dpg.add_text("This may take a few minutes to complete")
                                    dpg.add_text("   ")
                                    dpg.add_button(label="Cancel Processing", tag="cancel_processing_button", callback=cb.cancel_processing_callback, user_data={"cancel_event": threading.Event()})
                    
                                with dpg.group(horizontal=True):
                                    dpg.add_text("Progress: ")
                                    dpg.add_progress_bar(tag="progress_bar_as", default_value=0.0, width=290)
                                with dpg.tooltip("progress_bar_as"):
                                    dpg.add_text("Displays the processing progress.")
                
                        with dpg.group(horizontal=False):
                            # Panel 2 - Middle (Imported Acoustic Spaces Table)
                            with dpg.child_window(width=1285, height=310):
                                dpg.add_text("Imported Acoustic Spaces", tag="imported_as_title")
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                with dpg.tooltip("imported_as_title"):
                                    dpg.add_text("Processed IRs are displayed below")
                                    #dpg.add_text("Restart is required for newly processed IRs to be displayed in other tabs")
                                
                                
                                with dpg.group(horizontal=True):
                                    dpg.add_button(label="Refresh Table", tag="refresh_as_table_button", callback=cb.update_as_table_from_csvs)
                                    with dpg.tooltip("refresh_as_table_button"):
                                        dpg.add_text("Searches for imported acoustic spaces.")
                                    dpg.add_text("      ")
                                    dpg.add_button(label="Open Dataset Folder", tag="open_dataset_folder_button", callback=cb.open_user_int_as_folder)
                                    with dpg.tooltip("open_dataset_folder_button"):
                                        dpg.add_text("Click to open in explorer the folder where the processed IRs are stored")
                                    dpg.add_text("      ")
                                    dpg.add_button(label="Delete Selected", tag="delete_selected_button", callback=lambda: dpg.configure_item("del_processed_popup", show=True))
                                    with dpg.tooltip("delete_selected_button"):
                                        dpg.add_text("Delete selected row and associated data")
                    
                                with dpg.popup("delete_selected_button", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="del_processed_popup"):
                                    dpg.add_text("Selected Acoustic Space will be deleted")
                                    dpg.add_separator()
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="OK", width=75, callback=cb.delete_selected_callback)
                                        dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("del_processed_popup", show=False))
                    
                                dpg.add_text("", tag="selected_ir_rows", show=False, user_data=[])
                                with dpg.table(tag="processed_irs_table", header_row=True,
                                       resizable=True,
                                       policy=dpg.mvTable_SizingStretchProp,
                                       borders_innerH=True, borders_outerH=True,
                                       borders_innerV=True, borders_outerV=True,
                                       row_background=True):
                        
                                    #dpg.add_table_column(label="")  # Selectable column (no header)
                                    dpg.add_table_column(label="Name", init_width_or_weight=60)
                                    dpg.add_table_column(label="RT60 (ms)", init_width_or_weight=20)
                                    dpg.add_table_column(label="Description", init_width_or_weight=330)
                    
                            # Panel 3 - Right (Logger)
                            with dpg.child_window(width=1285, height=296, tag="import_console_window"):
                                pass  # Logger will be added here later

                 
            with dpg.tab(label="Room Target Generator",tag='room_target_tab', parent="tab_bar"):    
                dpg.bind_item_theme(dpg.last_item(), "__theme_j")
                dpg.add_text("Create new Room Targets")
                dpg.add_separator()
            
                with dpg.group(horizontal=True):
                    # Panel 1 - Left
                    with dpg.child_window(width=400, height=492):
                        with dpg.child_window(width=385, height=352):
                            dpg.add_text("Set Parameters")
                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                            dpg.add_separator()
   
                            dpg.add_text("Name (optional)")
                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                            dpg.add_input_text(label="", tag="target_name", width=270, callback=cb.update_room_target_name_display)
                            with dpg.tooltip("target_name"):
                                dpg.add_text("Enter a name for the new room target. If left blank, parameters will be used")
                            dpg.add_spacer(height=10)
                            dpg.add_text("Low-shelf Filter")
                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                        
                            dpg.add_input_int(
                                label="Frequency (Hz)", 
                                tag="low_shelf_freq", 
                                width=200, 
                                default_value=100, 
                                min_value=20, 
                                max_value=1000, 
                                min_clamped=True, 
                                max_clamped=True, 
                                callback=cb.generate_room_target_callback, 
                                user_data={"generation_running": False, "save_to_file": False}
                            )
                            with dpg.tooltip("low_shelf_freq"):
                                dpg.add_text("Set the cutoff frequency for the low-shelf filter.\nRecommended range: 20-1000 Hz")
                        
                            dpg.add_input_float(label="Gain (dB)", tag="low_shelf_gain", width=200, default_value=6.0, min_value=-6.0, max_value=18.0, format="%.2f", 
                                                min_clamped=True, max_clamped=True, callback=cb.generate_room_target_callback, user_data={"generation_running": False, "save_to_file": False})
                            with dpg.tooltip("low_shelf_gain"):
                                dpg.add_text("Set the gain for the low-shelf filter.\nNegative values attenuate low frequencies.")
                        
                            dpg.add_input_float(label="Q-Factor", tag="low_shelf_q", width=200, default_value=0.707, min_value=0.1, max_value=5.0, format="%.2f",
                                                min_clamped=True, max_clamped=True, callback=cb.generate_room_target_callback, user_data={"generation_running": False, "save_to_file": False})
                            with dpg.tooltip("low_shelf_q"):
                                dpg.add_text("Set the Q-factor (slope) for the low-shelf filter.\nLower values = broader effect.")
                        
                            dpg.add_spacer(height=10)
                            #dpg.add_separator()
                            dpg.add_text("High-shelf Filter")
                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                        
                            dpg.add_input_int(
                                label="Frequency (Hz)", 
                                tag="high_shelf_freq", 
                                width=200, 
                                default_value=7000, 
                                min_value=1000, 
                                max_value=20000, 
                                min_clamped=True, 
                                max_clamped=True, 
                                callback=cb.generate_room_target_callback, 
                                user_data={"generation_running": False, "save_to_file": False}
                            )
                            with dpg.tooltip("high_shelf_freq"):
                                dpg.add_text("Set the cutoff frequency for the high-shelf filter.\nRecommended range: 1-20 kHz")
                        
                            dpg.add_input_float(label="Gain (dB)", tag="high_shelf_gain", width=200, default_value=-4.0, min_value=-18.0, max_value=6.0, format="%.2f",
                                                min_clamped=True, max_clamped=True, callback=cb.generate_room_target_callback, user_data={"generation_running": False, "save_to_file": False})
                            with dpg.tooltip("high_shelf_gain"):
                                dpg.add_text("Set the gain for the high-shelf filter.\nPositive values boost high frequencies.")
                        
                            dpg.add_input_float(label="Q-Factor", tag="high_shelf_q", width=200, default_value=0.4, min_value=0.1, max_value=5.0, format="%.2f", 
                                                min_clamped=True, max_clamped=True, callback=cb.generate_room_target_callback, user_data={"generation_running": False, "save_to_file": False})
                            with dpg.tooltip("high_shelf_q"):
                                dpg.add_text("Set the Q-factor (slope) for the high-shelf filter.\nLower values = broader effect.")
               
                        with dpg.child_window(width=385, height=120):
                            dpg.add_text("Generate Room Target")
                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                            dpg.add_separator()
                        
                            with dpg.group(horizontal=True):
                                dpg.add_text("Room Target Name:", tag="room_target_name_label")
                                dpg.add_text("", tag="room_target_name_display")
                                with dpg.tooltip("room_target_name_display"):
                                    dpg.add_text("Displays the name entered above for the new Room Target.")
                        
                            with dpg.group(horizontal=True):
                                dpg.add_button(
                                    label="Generate Target",
                                    tag="generate_room_target_btn",
                                    callback=cb.generate_room_target_callback,
                                    user_data={"generation_running": False, "save_to_file": True}
                                )
                                with dpg.tooltip("generate_room_target_btn"):
                                    dpg.add_text("Apply the low/high shelf filter settings and generate a new room target curve.")
                                    dpg.add_text("Newly created targets will be available in the quick config and dataset export tabs")
                        
                            with dpg.group(horizontal=True):
                                dpg.add_text("Progress:")
                                dpg.add_progress_bar(tag="progress_bar_target_gen", default_value=0.0, width=290)
                            with dpg.tooltip("progress_bar_target_gen"):
                                dpg.add_text("Displays progress while generating the new room target.")
            
                    with dpg.group(horizontal=False):
                        with dpg.group(horizontal=True):  # Overall horizontal layout for left plot + right list
                            # Plot Panel - Left
                            with dpg.child_window(width=580, height=492):
                                dpg.add_text("Room Target Preview", tag="target_plot_title")
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                with dpg.tooltip("target_plot_title"):
                                    dpg.add_text("Displays the magnitude response of the selected room target.")
                        
                                with dpg.plot(label="Magnitude Response Plot", height=450, width=560):
                                    dpg.add_plot_legend()
                                    dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)", tag="rt_x_axis", scale=dpg.mvPlotScale_Log10)#, log_scale=True
                                    dpg.set_axis_limits("rt_x_axis", 10, 20000)
                                    dpg.add_plot_axis(dpg.mvYAxis, label="Magnitude (dB)", tag="rt_y_axis")
                                    dpg.set_axis_limits("rt_y_axis", -18, 18)
                                    dpg.add_line_series([], [], label="Target", parent="rt_y_axis", tag="rt_plot_series",skip_nan=True)
                        
                            # Listbox Panel - Right
                            with dpg.child_window(width=690, height=492):  # Adjusted width to fit total ~1285
                                dpg.add_text("Generated Room Targets", tag="generated_targets_title")
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                with dpg.tooltip("generated_targets_title"):
                                    dpg.add_text("Displays the list of user-defined room targets")
                                    dpg.add_text("Targets will also be available in the quick config and dataset export tabs")
                        
                                with dpg.group(horizontal=True):
                                    dpg.add_button(label="Refresh List", tag="refresh_targets_button", callback=cb.update_room_target_list)
                                    with dpg.tooltip("refresh_targets_button"):
                                        dpg.add_text("Scans for newly generated room target filters.")
                        
                                    dpg.add_text("      ")
                                    dpg.add_button(label="Open Targets Folder", tag="open_target_folder_button", callback=cb.open_user_rt_folder)
                                    with dpg.tooltip("open_target_folder_button"):
                                        dpg.add_text("Opens the directory where room targets are stored.")
                        
                                    dpg.add_text("      ")
                                    dpg.add_button(label="Delete Selected", tag="delete_selected_target_button", callback=lambda: dpg.configure_item("del_target_popup", show=True))
                                    with dpg.tooltip("delete_selected_target_button"):
                                        dpg.add_text("Deletes the selected room target and associated data.")
                                        dpg.add_text("Restart is required for room target lists to refresh in other tabs")
                        
                                with dpg.popup("delete_selected_target_button", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="del_target_popup"):
                                    dpg.add_text("Selected Room Target will be deleted")
                                    dpg.add_separator()
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="OK", width=75, callback=cb.delete_selected_target_callback)
                                        dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("del_target_popup", show=False))
                        
                                dpg.add_text("", tag="selected_target_name", show=False, user_data=None)
                        
                                dpg.add_listbox(
                                    items=[],
                                    tag="room_target_listbox",
                                    width=650,
                                    num_items=23,
                                    callback=cb.on_room_target_selected  # Should trigger update to plot
                                )
                
                    
                
            with dpg.tab(label="Additional Tools & Maintenance", tag="additional_tools", parent="tab_bar"):
                dpg.bind_item_theme(dpg.last_item(), "__theme_j")
            
                # Horizontal split: left table, right log
                with dpg.group(horizontal=True):
            
                    # --- Left: Table of Actions ---
                    with dpg.child_window(width=520, height=637):
                        with dpg.collapsing_header(label="Updates", default_open=True):
                   
                            with dpg.table(header_row=True, resizable=False, policy=dpg.mvTable_SizingFixedFit, row_background=True, borders_outerH=True,borders_innerH=True,borders_outerV=True,borders_innerV=True):
                                dpg.add_table_column(label="Section", init_width_or_weight=100)
                                dpg.add_table_column(label="Task", init_width_or_weight=200)
                                dpg.add_table_column(label="Action", init_width_or_weight=200)
                
                                # ---------- App ----------
                                with dpg.table_row():
                                    dpg.add_text("App")
                                    dpg.add_text("Check for Updates on Start")
                                    dpg.add_checkbox(label="Enable", default_value=loaded_values["check_updates_start_toggle"],tag="check_updates_start_toggle", callback=cb.save_settings)
                
                                with dpg.table_row():
                                    dpg.add_text("App")
                                    dpg.add_text("Check for App Updates")
                                    dpg.add_button(label="Check for Updates", tag="app_version_tag", user_data="",
                                                   callback=cb.check_app_version)
                                    with dpg.tooltip("app_version_tag"):
                                        dpg.add_text("This will check for updates to the app and show versions in the log")
                
                                # ---------- Headphone Correction Filters ----------
                                with dpg.table_row():
                                    dpg.add_text("Headphone Filters")
                                    dpg.add_text("Check for Headphone Filter Updates")
                                    dpg.add_button(label="Check for Updates", tag="hpcf_db_version_tag", user_data="",
                                                   callback=cb.check_db_version)
                                    with dpg.tooltip("hpcf_db_version_tag"):
                                        dpg.add_text("This will check for updates to the headphone correction filter datasets and show versions in the log")
                
                                with dpg.table_row():
                                    dpg.add_text("Headphone Filters")
                                    dpg.add_text("Download Latest Headphone Filters")
                                    dpg.add_button(label="Download Latest Datasets", tag="hpcf_db_download_tag", user_data="",
                                                   callback=cb.download_latest_db)
                                    with dpg.tooltip("hpcf_db_download_tag"):
                                        dpg.add_text("This will download the latest version of the datasets and replace local files")
                
                                # ---------- Acoustic Spaces ----------
                                with dpg.table_row():
                                    dpg.add_text("Acoustic Spaces")
                                    dpg.add_text("Check for Acoustic Space Updates")
                                    dpg.add_button(label="Check for Updates", tag="as_version_tag", user_data="",
                                                   callback=cb.check_as_versions)
                                    with dpg.tooltip("as_version_tag"):
                                        dpg.add_text("This will check for updates to acoustic space datasets and show new versions in the log")
                
                                with dpg.table_row():
                                    dpg.add_text("Acoustic Spaces")
                                    dpg.add_text("Download Acoustic Space Updates")
                                    dpg.add_button(label="Download Latest Datasets", tag="as_download_tag", user_data="",
                                                   callback=cb.download_latest_as_sets)
                                    with dpg.tooltip("as_download_tag"):
                                        dpg.add_text("This will download updates to acoustic space datasets and replace local versions")
                
                                # ---------- HRTF Datasets ----------
                                with dpg.table_row():
                                    dpg.add_text("HRTF Datasets")
                                    dpg.add_text("Check for HRTF Dataset Updates")
                                    dpg.add_button(label="Check for Updates", tag="hrtf_version_tag", user_data="",
                                                   callback=cb.check_hrtf_versions)
                                    with dpg.tooltip("hrtf_version_tag"):
                                        dpg.add_text("This will check for updates to HRTF datasets and show new updates in the log")
                
                                with dpg.table_row():
                                    dpg.add_text("HRTF Datasets")
                                    dpg.add_text("Update HRTF Dataset List")
                                    dpg.add_button(label="Download Latest Dataset List", tag="hrtf_download_tag", user_data="",
                                                   callback=cb.download_latest_hrtf_sets)
                                    with dpg.tooltip("hrtf_download_tag"):
                                        dpg.add_text("This will download the latest list of HRTF datasets. Restart required if updates found")
            
                        with dpg.collapsing_header(label="User Data", default_open=True):
                 
                            with dpg.table(header_row=True, resizable=False, policy=dpg.mvTable_SizingFixedFit, row_background=True, borders_outerH=True,borders_innerH=True,borders_outerV=True,borders_innerV=True):
                                dpg.add_table_column(label="Section", init_width_or_weight=100)
                                dpg.add_table_column(label="Task", init_width_or_weight=200)
                                dpg.add_table_column(label="Action", init_width_or_weight=200)
                
                                # ---------- Inputs ----------
                                with dpg.table_row():
                                    dpg.add_text("Inputs")
                                    dpg.add_text("Reset All Settings to Default")
                                    dpg.add_button(label="Reset Settings", tag="reset_settings_tag", callback=cb.reset_settings)
                
                                with dpg.table_row():
                                    dpg.add_text("Inputs")
                                    dpg.add_text("Settings and Presets Folder")
                                    dpg.add_button(label="Open Folder", tag="open_settings_folder_button",
                                                   callback=cb.open_user_settings_folder)
                                    with dpg.tooltip("open_settings_folder_button"):
                                        dpg.add_text("Opens the directory where the settings file is stored")
                
                                # ---------- Outputs ----------
                                with dpg.table_row():
                                    dpg.add_text("Outputs")
                                    dpg.add_text("Delete All Exported Filters")
                                    dpg.add_button(label="Delete Headphone Filters", tag="remove_hpcfs_tag",
                                                   callback=lambda: dpg.configure_item("del_hpcfs_popup", show=True))
                                    with dpg.tooltip("remove_hpcfs_tag"):
                                        dpg.add_text("Warning: this will delete all headphone filters exported to the output directory")
                                    with dpg.popup("remove_hpcfs_tag", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="del_hpcfs_popup"):
                                        dpg.add_text("All exported Headphone Filters will be deleted.")
                                        dpg.add_separator()
                                        with dpg.group(horizontal=True):
                                            dpg.add_button(label="OK", width=75, callback=cb.remove_hpcfs)
                                            dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("del_hpcfs_popup", show=False))
                
                                with dpg.table_row():
                                    dpg.add_text("Outputs")
                                    dpg.add_text("Delete All Exported BRIRs")
                                    dpg.add_button(label="Delete Binaural Datasets", tag="remove_brirs_tag",
                                                   callback=lambda: dpg.configure_item("del_brirs_popup", show=True))
                                    with dpg.tooltip("remove_brirs_tag"):
                                        dpg.add_text("Warning: this will delete all BRIRs exported to the output directory")
                                    with dpg.popup("remove_brirs_tag", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="del_brirs_popup"):
                                        dpg.add_text("All exported Binaural Datasets will be deleted.")
                                        dpg.add_separator()
                                        with dpg.group(horizontal=True):
                                            dpg.add_button(label="OK", width=75, callback=cb.remove_brirs)
                                            dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("del_brirs_popup", show=False))
                
                                with dpg.table_row():
                                    dpg.add_text("Outputs")
                                    dpg.add_text("Exported Files Folder")
                                    dpg.add_button(label="Open Folder", tag="open_output_folder_button", callback=cb.open_output_folder)
            
                        with dpg.collapsing_header(label="Additional Settings", default_open=True):
     
                            with dpg.table(header_row=True, resizable=False, policy=dpg.mvTable_SizingFixedFit, row_background=True, borders_outerH=True,borders_innerH=True,borders_outerV=True,borders_innerV=True):
                                dpg.add_table_column(label="Section", init_width_or_weight=90)
                                dpg.add_table_column(label="Setting", init_width_or_weight=190)
                                dpg.add_table_column(label="Value", init_width_or_weight=210)
                
                                # ---------- Misc Parameters ----------
                                with dpg.table_row():
                                    dpg.add_text("HRTFs")
                                    dpg.add_text("Force Left/Right Symmetry")
                                    dpg.add_combo(CN.HRTF_SYM_LIST, default_value=loaded_values["force_hrtf_symmetry"], width=200,
                                                  callback=cb.qc_update_brir_param, tag='force_hrtf_symmetry')
                                    with dpg.tooltip("force_hrtf_symmetry"):
                                        dpg.add_text("This will mirror the left or right sides of the HATS / dummy head")
                                        dpg.add_text("Applies to the direct sound. Reverberation is not modified")
                                        
                                with dpg.table_row():
                                    dpg.add_text("HRTFs")
                                    dpg.add_text("Direct Sound Polarity")
                                    dpg.add_combo(CN.HRTF_POLARITY_LIST, default_value=loaded_values["hrtf_polarity_rev"], width=200,
                                                  callback=cb.qc_update_brir_param, tag='hrtf_polarity_rev')
                                    with dpg.tooltip("hrtf_polarity_rev"):
                                        dpg.add_text("This can be used to manually reverse the polarity of the direct sound.")
                                        dpg.add_text("Reverberation is not modified")
                                        
                                with dpg.table_row():
                                    dpg.add_text("HRTFs")
                                    dpg.add_text("Diffuse-field Calibration")
                                    dpg.add_combo(CN.HRTF_DF_CAL_MODE_LIST, default_value=loaded_values["hrtf_df_cal_mode"], width=200,callback=cb.qc_update_brir_param, tag='hrtf_df_cal_mode')
                                    with dpg.tooltip("hrtf_df_cal_mode"):
                                        dpg.add_text("Diffuse-field calibration of HRTF datasets is enabled by default but can be disabled")
                                        dpg.add_text("If disabled, HRTF datasets will retain their direction-independent information")
                                        dpg.add_text("Level spectrum ends option will also remove roll-off from low and high frequencies if present")
                                    
                                with dpg.table_row():
                                    dpg.add_text("HRTFs")
                                    dpg.add_text("Direction Misalignment Correction")
                                    dpg.add_combo(CN.HRTF_DIRECTION_FIX_LIST_GUI, default_value=CN.HRTF_DIRECTION_FIX_LIST_GUI[0], width=200,callback=cb.qc_update_brir_param, tag='hrtf_direction_misalign_comp')
                                    with dpg.tooltip("hrtf_direction_misalign_comp"):
                                        dpg.add_text("This can be used to correct common orientation errors in SOFA datasets")
                                        dpg.add_text("Use this if the azimuth or elevation directions appear reversed, rotated, or start from the wrong reference point")
                                        dpg.add_text("Refer to reference tables tab for more details")
                                with dpg.table_row():
                                    dpg.add_text("HRTFs")
                                    dpg.add_text("Direction Misalignment Correction")
                                    dpg.add_checkbox(label="Apply to next simulation", default_value=False,tag="hrtf_direction_misalign_trigger", callback=cb.qc_update_brir_param)
                                    with dpg.tooltip("hrtf_direction_misalign_trigger"):
                                        dpg.add_text("Applies above correction when next binaural simulation is processed")
                                        
                                with dpg.table_row():
                                    dpg.add_text("Reverberation")
                                    dpg.add_text("Early Reflection Delay (ms)")
                                    dpg.add_input_float(label=" ", width=200, format="%.1f", tag='er_delay_time',min_value=CN.ER_RISE_MIN, max_value=CN.ER_RISE_MAX,default_value=loaded_values["er_delay_time"], min_clamped=True,max_clamped=True, callback=cb.qc_update_brir_param)
                                    with dpg.tooltip("er_delay_time"):
                                        dpg.add_text("This will increase the time between the direct sound and early reflections")
                                        dpg.add_text("This can be used to increase perceived distance")
                                        dpg.add_text("Min. 0ms, Max. 10ms")
                
                                with dpg.table_row():
                                    dpg.add_text("SOFA")
                                    dpg.add_text("SOFA Export Convention")
                                    dpg.add_combo(CN.SOFA_OUTPUT_CONV, default_value=loaded_values["sofa_exp_conv"], width=200,
                                                  callback=cb.save_settings, tag='sofa_exp_conv')
            
                            
            
                    # --- Right: Log ---
                    with dpg.child_window(width=1166, height=637, tag="console_window", user_data=None):
                        with dpg.group(horizontal=True):
                            dpg.add_text("Primary Log", tag="log_text", user_data=CN.__version__)
                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                            dpg.add_text("     ")
                            dpg.add_button(label="Open Log File", tag="open_log_folder_button",
                                           callback=cb.open_program_folder)
                            
                    
                    
            with dpg.tab(label="User Guide", tag="user_guide_tab", parent="tab_bar"): 
                dpg.bind_item_theme(dpg.last_item(), "__theme_j")
                with dpg.collapsing_header(label="Audio Processing Overview", default_open=False, tag="ug_ap_overview"):
                    dpg.add_text(" ")
            
                with dpg.collapsing_header(label="GUI Overview", default_open=False, tag="ug_gui_overview"):
                    dpg.add_text(" ")
            
                with dpg.collapsing_header(label="Headphone Correction", default_open=False, tag="ug_headphone_correction"):
                    dpg.add_text(" ")
            
                with dpg.collapsing_header(label="Binaural Room Simulation over Headphones", default_open=False, tag="ug_brs_over_hp"):
                    dpg.add_text(" ")
            
                with dpg.collapsing_header(label="Channel Configuration", default_open=False, tag="ug_chan_config"):
                    dpg.add_text(" ")
            
                with dpg.collapsing_header(label="File Naming and Structure", default_open=False, tag="ug_file_naming"):
                    dpg.add_text(" ")
            
                with dpg.collapsing_header(label="7.1 Surround Virtualisation", default_open=False, tag="ug_surround_71"):
                    dpg.add_text(" ")
            
                with dpg.collapsing_header(label="Acoustic Space Import", default_open=False, tag="ug_acoustic_import"):
                    dpg.add_text(" ")
            
                with dpg.collapsing_header(label="Room Target Generator", default_open=False, tag="ug_room_target"):
                    dpg.add_text(" ")
            
                with dpg.collapsing_header(label="Additional Settings", default_open=False, tag="ug_additional"):
                    dpg.add_text(" ")
                        
            with dpg.tab(label="Reference Tables",tag='reference_tables_tab', parent="tab_bar"):    
                dpg.bind_item_theme(dpg.last_item(), "__theme_j")
                with dpg.collapsing_header(label="Acoustic Spaces", default_open=False):
                    with dpg.child_window(auto_resize_x=True, auto_resize_y=True):
                        data_length = len(CN.AC_SPACE_LIST_GUI)
    
                        with dpg.table(
                            header_row=True,
                            policy=dpg.mvTable_SizingFixedFit,
                            resizable=False,
                            borders_outerH=True,
                            borders_innerH=True,
                            no_host_extendX=True,
                            borders_outerV=True,
                            borders_innerV=True,
                            delay_search=True
                        ):
                            dpg.add_table_column(label="Name")
                            dpg.add_table_column(label="RT60 (ms)")
                            dpg.add_table_column(label="Description", width_fixed=True, init_width_or_weight=600)
                            dpg.add_table_column(label="Source Dataset", width_fixed=True, init_width_or_weight=850)
                        
                            for i in range(data_length):
                                with dpg.table_row():
                                    dpg.add_text(CN.AC_SPACE_LIST_GUI[i])
                                    dpg.add_text(CN.AC_SPACE_MEAS_R60[i])
                                    dpg.add_text(CN.AC_SPACE_DESCR[i], wrap=0)
                                    dpg.add_text(CN.AC_SPACE_DATASET[i], wrap=0)
                                
                with dpg.collapsing_header(label="Spatial Resolutions", default_open=False):
                    with dpg.child_window(auto_resize_x=True, auto_resize_y=True):
                        with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                            borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                            borders_outerV=True, borders_innerV=True, delay_search=True):
                            #dpg.bind_item_font(dpg.last_item(), font_b_def)
                            dpg.add_table_column(label="Resolution")
                            dpg.add_table_column(label="Elevation Range")
                            dpg.add_table_column(label="Elevation Steps")
                            dpg.add_table_column(label="Azimuth Range")
                            dpg.add_table_column(label="Azimuth Steps")
                            for i in range(len(CN.SPATIAL_RES_LIST)):
                                with dpg.table_row():
                                    for j in range(5):
                                        if j == 0:#Resolution
                                            dpg.add_text(CN.SPATIAL_RES_LIST[i])
                                        elif j == 1:#Elevation Range
                                            dpg.add_text(CN.SPATIAL_RES_ELEV_RNG[i])
                                        elif j == 2:#Elevation Steps
                                            dpg.add_text(CN.SPATIAL_RES_ELEV_STP[i])
                                        elif j == 3:#Azimuth Range
                                            dpg.add_text(CN.SPATIAL_RES_AZIM_RNG[i])
                                        elif j == 4:#Azimuth Steps
                                            dpg.add_text(CN.SPATIAL_RES_AZIM_STP[i])
                                            
                with dpg.collapsing_header(label="Low-frequency Responses", default_open=False):  
                    #Section to show sub brir information
                    with dpg.child_window(auto_resize_x=True, auto_resize_y=True):
                        with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                            borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                            borders_outerV=True, borders_innerV=True, delay_search=True):
                            #dpg.bind_item_font(dpg.last_item(), font_b_def)
                            dpg.add_table_column(label="Name")
                            dpg.add_table_column(label="Acoustic Space")
                            dpg.add_table_column(label="RT60 (ms)")
                            dpg.add_table_column(label="Frequency Range")
                            dpg.add_table_column(label="Tolerance")
                            dpg.add_table_column(label="Comments")
                            dpg.add_table_column(label="Source Type")
                            dpg.add_table_column(label="Listener Type")
                            dpg.add_table_column(label="Source Dataset")
                            for i in range(len(CN.SUB_RESPONSE_LIST_GUI)):
                                with dpg.table_row():
                                    dpg.add_text(CN.SUB_RESPONSE_LIST_GUI[i])
                                    dpg.add_text(CN.SUB_RESPONSE_LIST_AS[i],wrap =100)
                                    dpg.add_text(CN.SUB_RESPONSE_LIST_R60[i])
                                    dpg.add_text(CN.SUB_RESPONSE_LIST_RANGE[i])
                                    dpg.add_text(CN.SUB_RESPONSE_LIST_TOL[i])
                                    dpg.add_text(CN.SUB_RESPONSE_LIST_COMM[i])
                                    dpg.add_text(CN.SUB_RESPONSE_LIST_SOURCE[i])
                                    dpg.add_text(CN.SUB_RESPONSE_LIST_LISTENER[i])
                                    dpg.add_text(CN.SUB_RESPONSE_LIST_DATASET[i])
                             
                with dpg.collapsing_header(label="Supported SOFA Conventions", default_open=False):
                    with dpg.child_window(auto_resize_x=True, auto_resize_y=True):               
                        with dpg.group(horizontal=True):
                            with dpg.group():
                                dpg.add_text("Read")
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                    borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                    borders_outerV=True, borders_innerV=True, delay_search=True):
                                    #dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    dpg.add_table_column(label="Convention")
                                    dpg.add_table_column(label="Version")
                                    dpg.add_table_column(label="SOFAConventionsVersion")
                                    for i in range(len(CN.SOFA_COMPAT_CONV)):
                                        with dpg.table_row():
                                            for j in range(3):
                                                if j == 0:#Convention
                                                    dpg.add_text(CN.SOFA_COMPAT_CONV[i])
                                                elif j == 1:#Version
                                                    dpg.add_text(CN.SOFA_COMPAT_VERS[i],wrap =100)
                                                elif j == 2:#SOFAConventionsVersion
                                                    dpg.add_text(CN.SOFA_COMPAT_CONVERS[i])
                            
                            with dpg.group():
                                dpg.add_text("Write")
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                    borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                    borders_outerV=True, borders_innerV=True, delay_search=True):
                                    dpg.add_table_column(label="Convention")
                                    dpg.add_table_column(label="Version")
                                    dpg.add_table_column(label="SOFAConventionsVersion")
                                    for i in range(len(CN.SOFA_OUTPUT_CONV)):
                                        with dpg.table_row():
                                            for j in range(3):
                                                if j == 0:#Convention
                                                    dpg.add_text(CN.SOFA_OUTPUT_CONV[i])
                                                elif j == 1:#Version
                                                    dpg.add_text(CN.SOFA_OUTPUT_VERS[i],wrap =100)
                                                elif j == 2:#SOFAConventionsVersion
                                                    dpg.add_text(CN.SOFA_OUTPUT_CONVERS[i])
                                                    
                with dpg.collapsing_header(label="Direction Misalignment Correction", default_open=False):  
                    #Section to show misalignment options information
                    with dpg.child_window(auto_resize_x=True, auto_resize_y=True):
                        with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                            borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                            borders_outerV=True, borders_innerV=True, delay_search=True):
                            dpg.add_table_column(label="Option")
                            dpg.add_table_column(label="Description")
                            for i in range(len(CN.HRTF_DIRECTION_FIX_LIST)):
                                with dpg.table_row():
                                    dpg.add_text(CN.HRTF_DIRECTION_FIX_LIST_GUI[i])
                                    dpg.add_text(CN.HRTF_DIRECTION_FIX_LIST_DESC[i])
                                        
                dpg.add_text(" ")
            
          
        


    
    
    
    dpg.setup_dearpygui()
    
    logz=logger.mvLogger(parent="console_window")
    dpg.configure_item('console_window',user_data=logz)#store as user data

    #section to log tool version on startup
    #log results
    log_string = 'Started ASH Toolset - Version: ' + CN.__version__
    hf.log_with_timestamp(log_string, logz)
  
    dpg.show_viewport()
    dpg.set_primary_window("Primary Window", True)
    dpg.configure_item("Primary Window", horizontal_scrollbar=True)
    
    #inital configuration
    intialise_gui()
        
    dpg.start_dearpygui()
 
    dpg.destroy_context()
        
    #finally close the connection
    conn_ash.close()
    conn_comp.close()

    logging.info('Finished') 










if __name__ == '__main__':
    
    main()
    
