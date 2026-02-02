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
import dearpygui.dearpygui as dpg
import dearpygui_extend as dpge
from dearpygui_ext import logger
import numpy as np
import math
import threading
import sys
import os
import requests
import re
import platform
from os.path import join as pjoin, expanduser, normpath
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    import winreg as wrg

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



    def get_e_apo_paths(logz=None, create_dirs=True):
        """
        Detects the primary EqualizerAPO path and ASH Toolset project folder.
        On Linux/macOS, returns dummy paths since EqualizerAPO is Windows-only.
    
        Returns:
            e_apo_config_path (str): path to EqualizerAPO 'config' folder
            e_apo_ash_path (str): path to project folder inside primary_path
            e_apo_path (str): raw EqualizerAPO install path
        """
        if IS_WINDOWS:
            # Windows: try registry or fallback
            fallback_root = r"C:\Program Files\EqualizerAPO"
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
        else:
            # Linux / other OS: dummy folder
            e_apo_path = normpath(os.path.expanduser("~/.ash_dummy_eqapo"))
            hf.log_with_timestamp(f"EqualizerAPO is Windows-only. Using dummy folder: {e_apo_path}", logz)
    
        # Normalize paths
        e_apo_path = normpath(e_apo_path)
        e_apo_config_path = normpath(pjoin(e_apo_path, "config"))
        e_apo_ash_path = normpath(pjoin(e_apo_config_path, getattr(CN, "PROJECT_FOLDER", "")))
    
        # Optionally create folders if missing
        if create_dirs:
            for path in (e_apo_config_path, e_apo_ash_path):
                if not os.path.isdir(path):
                    try:
                        os.makedirs(path, exist_ok=True)
                        hf.log_with_timestamp(f"Created missing folder: {path}", logz)
                    except Exception as e:
                        hf.log_with_timestamp(f"Failed to create folder '{path}': {e}", logz)
    
        return e_apo_config_path, e_apo_ash_path, e_apo_path


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
               
       
    def set_initial_output_tab():
        try:
            if platform.system() == "Windows":
                dpg.set_value("hpcf_out_tab_bar", "qc_hpcf_apply_eapo_tab")
                dpg.set_value("brir_out_tab_bar", "brir_apply_eapo_tab")
            else:
                dpg.set_value("hpcf_out_tab_bar", "hpcf_export_tab")
                dpg.set_value("brir_out_tab_bar", "brir_export_tab")
        except Exception:
            pass        

            
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
                        hf.log_with_timestamp("[User Guide] Updated local user_guide.txt to latest version")
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
        guide_path = CN.USER_GUIDE_PATH
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
                hf.log_with_timestamp("[User Guide] No version tag found in user guide")
    
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
                
            hf.log_with_timestamp("[User Guide] Successfully loaded documentation")
    
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
        import_console_log = logger.mvLogger(parent="import_console_window")
        dpg.configure_item("import_console_window", user_data=import_console_log)
        cb.update_ir_folder_list()
        cb.update_as_table_from_csvs()
        
        # #room target manager
        cb.update_room_target_list()
        #presets
        cb.refresh_preset_list()
        
        #check for updates on start
        if loaded_values["check_updates_start_toggle"] == True:
            #start thread
            thread = threading.Thread(target=cb.check_all_updates, args=(), daemon=True)
            thread.start()

        #inital configuration
        
        #update channel gui elements on load, will also write e-APO config again
        cb.e_apo_select_channels(app_data=dpg.get_value('e_apo_audio_channels'),aquire_config=False)
        
        #adjust active tab
        set_initial_output_tab()
        try:
            dpg.set_value("tab_bar", loaded_values["tab_bar"])
        except Exception:
            pass

        #show hpcf history but only run function if enabled
        hpcf_hist_toggled = dpg.get_value('toggle_hpcf_history')
        if hpcf_hist_toggled:
            cb.show_hpcf_history(app_data=hpcf_hist_toggled)
            
        #trigger activation of e apo config if enabled
        hpcf_is_active=dpg.get_value('e_apo_hpcf_conv')
        brir_is_active=dpg.get_value('e_apo_brir_conv')
        cb.e_apo_toggle_hpcf_custom(activate=hpcf_is_active, aquire_config=False)
        cb.e_apo_toggle_brir_custom(activate=brir_is_active, aquire_config=False)
        #finally acquire config once
        cb.e_apo_config_acquire()
        cb.fde_reset_brir_progress()
        cb.fde_reset_hpcf_progress()
        
        #also update gui labels based on active database
        cb.update_hpcf_gui_labels()
        
        cb.update_chan_az_status_text()
       
        #update audio device text
        hf.update_default_output_text(reset_sd=False)
        
        #sort acoustic spaces
        cb.update_ac_space_display()
        cb.update_ac_space_info()
        #refresh direction fix
        cb.refresh_direction_fix_selection()
        
        #update e apo elevation list
        cb.e_apo_update_elev_list()
        cb.fde_update_angle_list()
        
        #populate user guide
        load_user_guide_into_gui()
        
    #QC loaded hp and sample lists based on loaded brand and headphone
    def load_db_values(active_db_name):
        """Retrieve brands, headphones, samples, and ensure valid selection."""
        conn = db_map[active_db_name]
        hpcf_db_dict['conn'] = conn
    
        # QC tab
        brands = hpcf_functions.get_brand_list(conn)
        brands, loaded_values["hpcf_brand"] = hf.ensure_valid_selection(
            brands, loaded_values.get("hpcf_brand", "")
        )
        hp_list = hpcf_functions.get_headphone_list(conn, loaded_values["hpcf_brand"])
        hp_list, loaded_values["hpcf_headphone"] = hf.ensure_valid_selection(
            hp_list, loaded_values.get("hpcf_headphone", "")
        )
        sample_list = hpcf_functions.get_samples_list(conn, loaded_values["hpcf_headphone"])
        sample_list, loaded_values["hpcf_sample"] = hf.ensure_valid_selection(
            sample_list, loaded_values.get("hpcf_sample", "")
        )
   
    
        return conn, brands, hp_list, sample_list
       

    def add_image_checkbox(
        label: str,
        checkbox_tag: str,
        callback,
        user_data=None,
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

    def get_screen_size(default_width=1722, default_height=717):
       """
       Returns the screen width and height in pixels (cross-platform).
       Falls back to default values if detection fails.
   
       Args:
           default_width (int): fallback width if detection fails
           default_height (int): fallback height if detection fails
   
       Returns:
           (width, height) tuple
       """
       IS_WINDOWS = platform.system() == "Windows"
   
       try:
           if IS_WINDOWS:
               import ctypes
               width = ctypes.windll.user32.GetSystemMetrics(0)
               height = ctypes.windll.user32.GetSystemMetrics(1)
           else:
               # Linux/macOS: use Tkinter
               import tkinter as tk
               root = tk.Tk()
               root.withdraw()  # hide root window
               width = root.winfo_screenwidth()
               height = root.winfo_screenheight()
               root.destroy()
   
           # fallback to defaults if for some reason width/height are zero
           if not width or not height:
               width, height = default_width, default_height
   
       except Exception as e:
           logging.error(f"Error getting screen size: {e}")
           width, height = default_width, default_height
   
       return width, height
      
        
    
    #
    ##################  program code
    #
    
    #generate flat response
    impulse=np.zeros(CN.N_FFT)
    impulse[0]=1

    
    #thread variables
    e_apo_conf_lock = threading.Lock()
    
    #code to get gui width based on windows resolution
    gui_win_width_default = 1722
    gui_win_height_default = 717
    
    screen_width, screen_height = get_screen_size(gui_win_width_default, gui_win_height_default)
    
    gui_win_width_loaded = min(gui_win_width_default, screen_width)
    gui_win_height_loaded = min(gui_win_height_default, screen_height)
    

    # Equalizer APO related code
    elevation_list_sel= CN.ELEV_ANGLES_WAV_LOW
    
     
    #
    #################### Dynamic Path handling
    #
    #get equalizer APO path
    e_apo_config_path, e_apo_ash_path, e_apo_path = get_e_apo_paths()
    #load settings from settings file, stores in a dict
    loaded_values = cb.load_settings()
    #overwrite paths if user values differ from defaults
    loaded_path = loaded_values.get("path", "")
    # --- Determine base export path ---
    if loaded_path and os.path.isdir(loaded_path):
        export_base_path = loaded_path
        hf.log_with_timestamp(f"Using user-saved path for main exports: {export_base_path}")
    else:
        # Neutral default: user home directory + ASH-Outputs
        export_base_path = pjoin(expanduser("~"), "ASH-Toolset")
        hf.log_with_timestamp(f"No custom path found. Using default user folder for main exports: {export_base_path}")
    export_ash_path = pjoin(export_base_path, CN.PROJECT_FOLDER) 
    #path handling continued
    #set hesuvi path
    if 'EqualizerAPO' in e_apo_path:
        export_hesuvi_path = pjoin(e_apo_config_path,'HeSuVi')#stored outside of project folder (within hesuvi installation)
    else:
        export_hesuvi_path = pjoin(export_base_path, CN.PROJECT_FOLDER,'HeSuVi')#stored within project folder
    # --- Ensure folders exist ---
    for path in (export_base_path, export_ash_path, export_hesuvi_path):
        os.makedirs(path, exist_ok=True)
    
    
    
    #
    ##################  Dynamic HpCF Database code
    #
    # create database connections to each database
    database_ash = CN.DATABASE_ASH_DIR
    database_comp = CN.DATABASE_COMP_DIR
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
    brand_default=brand_default
    headphone_default=headphone_default
    sample_default=sample_default
    #make updates for dynamic defaults in constants dict
    CN.DEFAULTS.update({
        "hpcf_brand": brand_default,
        "hpcf_headphone": headphone_default,
        "hpcf_sample": sample_default,
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
    
    # Ensure both tabs have the same active database
    active_database = loaded_values['hpcf_active_database']

    # First, try the current active database
    conn = db_map.get(active_database, conn_ash)
    brands_list = hpcf_functions.get_brand_list(conn)
    
    # Check if loaded brand exists in brand list; if not, switch database
    if loaded_values.get("hpcf_brand", "") not in brands_list:
        # Switch to the other database
        new_db_name = CN.HPCF_DATABASE_LIST[1] if active_database == CN.HPCF_DATABASE_LIST[0] else CN.HPCF_DATABASE_LIST[0]
        active_database = new_db_name
        loaded_values['hpcf_active_database'] = new_db_name
        hf.log_with_timestamp("Previously applied HpCF not found in active headphone database. Switching database")
    
    # Load all values from the active (or switched) database
    conn, brands_list_loaded, hp_list_loaded, sample_list_loaded = load_db_values(active_database)

    

    #
    ################### Dynamic HRTF listener lists
    #
  
    #grab loaded settings for dynamic lists (hrtf lists)
    brir_hrtf_type_loaded=loaded_values["brir_hrtf_type"]
    brir_hrtf_dataset_loaded=loaded_values["brir_hrtf_dataset"]
    hrtf_loaded =loaded_values["brir_hrtf"]
    brir_hrtf_dataset_list_loaded = CN.HRTF_TYPE_DATASET_DICT.get(brir_hrtf_type_loaded) or []
  
    #listener related buttons are dynamic
    add_fav_button_show=True
    remove_fav_button_show=False
    sofa_folder_button_show=False

    if brir_hrtf_type_loaded == 'Favourites':#use loaded favourites list if favourites selected
        add_fav_button_show=False
        remove_fav_button_show=True
    elif brir_hrtf_type_loaded == 'User SOFA Input':
        sofa_folder_button_show=True
        
    if brir_hrtf_type_loaded == 'Favourites':#use loaded favourites list if favourites selected
        hrtf_list_loaded = loaded_values["hrtf_list_favs"]
    else:
        hrtf_list_loaded = hrir_processing.get_listener_list(listener_type=brir_hrtf_type_loaded, dataset_name=brir_hrtf_dataset_loaded)

    #
    ###################### Dynamic export options
    #

    #export options are dynamic
    #adjust hrtf list based on loaded spatial resolution
    #also adjust file export selection
    generic_brir_exp_show=True
    sofa_brir_exp_show=True
    sofa_brir_tooltip_show=True
    
    # if loaded_values["fde_brir_spat_res"] == 'Medium' or loaded_values["fde_brir_spat_res"] == 'Low':
    #     loaded_values["fde_sofa_brir_toggle"] =False
    #     sofa_brir_exp_show=False
    #     sofa_brir_tooltip_show=False

    #
    ##################### ----- List validation ------
    #
    
    #validate that loaded strings are within associated lists - dynamic lists
    brir_hrtf_dataset_loaded = hf.validate_choice(brir_hrtf_dataset_loaded, brir_hrtf_dataset_list_loaded)
    hrtf_loaded = hf.validate_choice(hrtf_loaded, hrtf_list_loaded)

    # Define which additional keys need list validation and their valid option lists - static lists
    VALIDATION_MAP = {
        "acoustic_space": CN.AC_SPACE_LIST_GUI,
        "room_target": CN.ROOM_TARGET_LIST,
        "brir_hp_type": CN.HP_COMP_LIST,
        "sub_response": CN.SUB_RESPONSE_LIST_GUI,
        "e_apo_prevent_clip": CN.AUTO_GAIN_METHODS,
        "e_apo_upmix_method": CN.UPMIXING_METHODS
    }
    # Validate list-based settings after loading 
    for key, valid_list in VALIDATION_MAP.items():
        if key in loaded_values:
            loaded_values[key] = hf.validate_choice(loaded_values[key], valid_list)




  
 
    #
    ########################  GUI CODE
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
  
        with dpg.theme(tag="__theme_c"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Header, (138, 138, 62), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (135, 163, 78), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (138, 138, 62), category=dpg.mvThemeCat_Core)
  
        with dpg.theme(tag="__theme_f"):
            i=3.9
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, _hsv_to_rgb(i/7.0, 0.05, 0.99))
        with dpg.theme(tag="__theme_g"):
            i=3.9
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, _hsv_to_rgb(i/7.0, 0.2, 0.55)) 
        with dpg.theme(tag="__theme_h"):
            i=4.0
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, _hsv_to_rgb(i/7.0, 0.2, 0.55)) 
   
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
        with dpg.theme() as modern_theme:#default theme used for main tab
            i=4.1
            j=4.1
            k=4.1
            sat_mult=0.7
            val_mult=1.0
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (32, 32, 32), category=dpg.mvThemeCat_Core)  # Neutral dark gray
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (44, 44, 44), category=dpg.mvThemeCat_Core)   # Slightly lighter neutral dark gray
                dpg.add_theme_color(dpg.mvThemeCol_Border, (60, 60, 60), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 60, 70), category=dpg.mvThemeCat_Core)
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

            with dpg.tab(label="Headphone Correction & Spatial Audio",tag='quick_config', parent="tab_bar"): 
                #dpg.add_text("Apply Headphone Correction & Binaural Room Simulation in Equalizer APO")
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    #Section for HpCF Export
                    with dpg.child_window(width=550, height=634):
                        with dpg.group(horizontal=True):
                            title_1_qc = dpg.add_text("Headphone Correction")
                            dpg.bind_item_font(title_1_qc, font_b_med)
                        
                            # Button to revert selection to previously applied parameters
                            dpg.add_text("                                                                                               ")
                            dpg.add_button(label="Revert Selection", user_data="", tag="hpcf_revert_selection", callback=cb.revert_to_saved_params)
                            with dpg.tooltip("hpcf_revert_selection"):
                                dpg.add_text("Revert selection to previously applied parameters")
                        with dpg.child_window(autosize_x=True, height=410):
                            

                            
                            with dpg.group(horizontal=True):
                                
                                dpg.add_text("Headphone Database: ")
                                dpg.add_combo(CN.HPCF_DATABASE_LIST,  tag= "hpcf_active_database", default_value=loaded_values["hpcf_active_database"], callback=cb.change_hpcf_database_callback, width=140)
                   
                               
                                dpg.add_text("             ")
                                dpg.add_checkbox(label="Show History", default_value=loaded_values["toggle_hpcf_history"], tag='toggle_hpcf_history', callback=cb.show_hpcf_history)

                                with dpg.tooltip("toggle_hpcf_history"):
                                    dpg.add_text("Shows previously applied headphones")
                                dpg.add_text("     ")
                                
                                # Button to trigger the popup
                                dpg.add_button(label="Clear History", user_data="", tag="clear_history_button",
                                               callback=lambda: dpg.configure_item("clear_history_popup", show=True))
                                # Optional tooltip
                                with dpg.tooltip("clear_history_button"):
                                    dpg.add_text("Clear headphone correction history")
                                # Confirmation popup
                                with dpg.popup("clear_history_button", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="clear_history_popup"):
                                    dpg.add_text("This will clear headphone correction history")
                                    dpg.add_separator()
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="OK", width=75, callback=cb.remove_hpcfs)
                                        dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("clear_history_popup", show=False))
                
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_text("Search Brand:", tag='hpcf_brand_search_title')
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                dpg.add_input_text(label="", callback=cb.filter_brand_list, width=105, tag='hpcf_brand_search')
                                dpg.add_text("Search Headphone:", tag='hpcf_headphone_search_title')
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                dpg.add_input_text(label="", callback=cb.filter_headphone_list, width=209, tag='hpcf_headphone_search')
                            with dpg.group(horizontal=True, width=0):
                                with dpg.group():
                                    dpg.add_text("Brand", tag='hpcf_brand_title')
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    dpg.add_listbox(brands_list_loaded, width=135, num_items=17, tag='hpcf_brand', default_value=loaded_values["hpcf_brand"], callback=cb.update_headphone_list)
                                with dpg.group():
                                    dpg.add_text("Headphone", tag='hpcf_headphone_title')
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    dpg.add_listbox(hp_list_loaded, width=250, num_items=17, tag='hpcf_headphone', default_value=loaded_values["hpcf_headphone"] ,callback=cb.update_sample_list)
                                with dpg.group():
                                    dpg.add_text("Sample", tag='hpcf_sample_title')
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    dpg.add_listbox(sample_list_loaded, width=115, num_items=17, default_value=loaded_values["hpcf_sample"], tag='hpcf_sample', callback=cb.plot_sample) 
                        
                        with dpg.child_window(autosize_x=True, height=177):
                            with dpg.tab_bar(tag='hpcf_out_tab_bar'):
                                with dpg.tab(label="Quick Config - Equalizer APO", parent="hpcf_out_tab_bar",tag='qc_hpcf_apply_eapo_tab'):
                                    with dpg.group(horizontal=True):
                                        subtitle_3_qc = dpg.add_text("Apply Headphone Correction in Equalizer APO")
                                        dpg.bind_item_font(subtitle_3_qc, font_b_def)
                                        dpg.add_text("                                    ")
                                        dpg.add_checkbox(label="Auto Apply Selection", default_value = loaded_values["e_apo_auto_apply_hpcf_sel"],  tag='e_apo_auto_apply_hpcf_sel', callback=cb.e_apo_auto_apply_hpcf)
                                    dpg.add_separator()
                                    with dpg.group(horizontal=True):
                
                                        dpg.add_image_button(texture_tag='start_blue',width=40,height=40,tag="e_apo_hpcf_tag",callback=cb.e_apo_apply_hpcf_params,frame_padding=0,background_color=[0, 0, 0, 0])
                                        dpg.bind_item_theme(dpg.last_item(), "transparent_image_button_theme")
                                        
                                        with dpg.tooltip("e_apo_hpcf_tag"):
                                            dpg.add_text("This will apply the selected filter in Equalizer APO")
                                        dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=30, width=412,pos = (60,65), overlay=CN.PROGRESS_START, tag="e_apo_progress_bar_hpcf")
                                        dpg.bind_item_theme(dpg.last_item(), "__theme_h")
                                        dpg.add_text(" ")
                                        
                                        add_image_checkbox(label="Enable Headphone Correction",width=40,height=40,checkbox_tag="e_apo_hpcf_conv",callback=cb.e_apo_toggle_hpcf_gui,default_value=loaded_values["e_apo_hpcf_conv"])#,user_data=e_apo_conf_lock
                            
                                    dpg.add_separator()
                                    with dpg.group():
                                        dpg.add_text("Active Filter: ")
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                                        dpg.add_text(default_value=loaded_values["e_apo_curr_hpcf"],  tag='e_apo_curr_hpcf', wrap=450)
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                                        dpg.bind_item_theme(dpg.last_item(), "__theme_f")
                                        dpg.add_text(default_value=loaded_values["e_apo_sel_hpcf"], tag='e_apo_sel_hpcf',show=False, user_data=hpcf_db_dict)
               
                                with dpg.tab(label="Filter Export", parent="hpcf_out_tab_bar",tag='hpcf_export_tab'):
                                    with dpg.tab_bar(tag='hpcf_export_tab_bar'):
                                       with dpg.tab(label="Export", parent="hpcf_export_tab_bar",tag='hpcf_export_trigger_tab'):
                                            subtitle_3 = dpg.add_text("Export Headphone Correction Filters")
                                            dpg.bind_item_font(subtitle_3, font_b_def)
                                            dpg.add_separator()
                                            with dpg.group(horizontal=True):
                                               
                                                dpg.add_image_button(texture_tag='start_green',width=37,height=37,tag="fde_hpcf_tag",callback=cb.fde_process_hpcfs,frame_padding=0,background_color=[0, 0, 0, 0])
                                                dpg.bind_item_theme(dpg.last_item(), "transparent_image_button_theme")
                                                
                                                with dpg.tooltip("fde_hpcf_tag"):
                                                    dpg.add_text("This will export the selected filters to the output directory")
                                                    
                                                dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=32, width=470, overlay=CN.PROGRESS_START_ALT, tag="fde_progress_bar_hpcf")
                                                dpg.bind_item_theme(dpg.last_item(), "__theme_g")
                                       with dpg.tab(label="Options", parent="hpcf_export_tab_bar",tag='hpcf_export_files_tab'):
                                            subtitle_2 = dpg.add_text("Select Files to Include in Export")
                                            dpg.bind_item_font(subtitle_2, font_b_def)
                                            dpg.add_separator()
                                            with dpg.group(horizontal=True):
                                                dpg.add_checkbox(label="WAV FIR Filters", default_value = loaded_values["fde_fir_hpcf_toggle"], callback=cb.fde_export_hpcf_file_toggle, tag='fde_fir_hpcf_toggle')
                                                with dpg.tooltip("fde_fir_hpcf_toggle"):
                                                    dpg.add_text("Min phase FIRs in WAV format for convolution. 1 Channel")
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
                            
                    #Section for BRIR generation
                    with dpg.child_window(width=534, height=634):
                        with dpg.group(horizontal=True):
                            title_2_qc = dpg.add_text("Binaural Simulation", tag='brir_title')
                            dpg.bind_item_font(title_2_qc, font_b_med)
                            with dpg.tooltip("brir_title"):
                                dpg.add_text("Customise a new binaural simulation using below parameters")
                            # Button to revert selection to previously applied parameters
                            dpg.add_text("                                                                                               ")
                            dpg.add_button(label="Revert Selection", user_data="", tag="brir_revert_selection", callback=cb.revert_to_saved_params)
                            with dpg.tooltip("brir_revert_selection"):
                                dpg.add_text("Revert selection to previously applied parameters")
                                
                        with dpg.child_window(autosize_x=True, height=410):
                            with dpg.tab_bar(tag='brir_tab_bar'):

                                with dpg.tab(label="Acoustics",tag='acoustics_tab', parent="brir_tab_bar"): 

                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Acoustic Space",tag='acoustic_space_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            with dpg.group(horizontal=True):
                                                dpg.add_text("Sort by:      ")
                                                dpg.add_combo(CN.AC_SPACE_LIST_SORT_BY, width=200, label="",  tag='sort_by_as',default_value=CN.AC_SPACE_LIST_SORT_BY[0], callback=cb.update_ac_space_display)
                                            with dpg.group(horizontal=True):
                                                dpg.add_text("Collection: ")
                                                dpg.add_combo(CN.AC_SPACE_LIST_COLLECTIONS, width=200, label="",  tag='as_collection',default_value=CN.AC_SPACE_LIST_COLLECTIONS[0], callback=cb.update_ac_space_display)
                                            with dpg.group(horizontal=True):
                                                dpg.add_listbox(CN.AC_SPACE_LIST_GUI, label="",tag='acoustic_space',default_value=loaded_values["acoustic_space"], callback=cb.update_ac_space, num_items=16, width=345)
                                                with dpg.tooltip("acoustic_space_title"):
                                                    dpg.add_text("This will determine the listening environment")
                                                dpg.add_text("  ")
                                        with dpg.group():
                                            dpg.add_text("Direct Sound Gain (dB)", tag='direct_gain_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_input_float(label="",width=140, format="%.1f", tag='direct_gain', min_value=CN.DIRECT_GAIN_MIN, max_value=CN.DIRECT_GAIN_MAX, default_value=loaded_values["direct_gain"],min_clamped=True, max_clamped=True, callback=cb.update_direct_gain)
                                            with dpg.tooltip("direct_gain_title"):
                                                dpg.add_text("This will control the loudness of the direct signal")
                                                dpg.add_text("Higher values result in lower perceived distance, lower values result in higher perceived distance")
                                            dpg.add_slider_float(label="", min_value=CN.DIRECT_GAIN_MIN, max_value=CN.DIRECT_GAIN_MAX, default_value=loaded_values["direct_gain_slider"], width=140,clamped=True, no_input=True, format="", callback=cb.update_direct_gain_slider, tag='direct_gain_slider')
                                            dpg.add_separator()
                                            
                                            dpg.add_button(label="Add Favourite", tag="as_add_favourite", callback=cb.add_as_favourite_callback,show=True,user_data=loaded_values["as_list_favs"])
                                            dpg.add_button(label="Remove Favourite", tag="as_remove_favourite", callback=cb.remove_as_favourite_callback,show=False)
                                            
                                            dpg.add_text("ID:")
                                            dpg.add_text(" ",tag='acoustic_space_id_text', wrap=145)
                                            dpg.add_text("Name:")
                                            dpg.add_text(" ",tag='acoustic_space_name_text', wrap=145)
                                            dpg.add_text("RT60 (ms):")
                                            dpg.add_text(" ",tag='acoustic_space_rt60_text', wrap=145)
                                            dpg.add_text("Description:")
                                            dpg.add_text(" ",tag='acoustic_space_desc_text', wrap=145)
                                            

                                with dpg.tab(label="Listener Selection",tag='listener_tab', parent="brir_tab_bar"): 
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Category", tag='brir_hrtf_type_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_radio_button(CN.HRTF_TYPE_LIST, horizontal=False, tag= "brir_hrtf_type", default_value=loaded_values["brir_hrtf_type"], callback=cb.update_hrtf_dataset_list )
                                            with dpg.tooltip("brir_hrtf_type_title"):
                                                dpg.add_text("User SOFA files must be placed in '\\data\\user\\SOFA' folder")
                                            dpg.add_separator()
                                            dpg.add_text("Dataset", tag='brir_hrtf_dataset_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_listbox(brir_hrtf_dataset_list_loaded, default_value=brir_hrtf_dataset_loaded, num_items=8, width=255, callback=cb.update_hrtf_list, tag='brir_hrtf_dataset')
                                            
                                            dpg.add_separator()
                                            dpg.add_text("Direction Misalignment Correction")
                                            dpg.add_combo(CN.HRTF_DIRECTION_FIX_LIST_GUI, default_value=loaded_values["hrtf_direction_misalign_comp"], width=250,callback=cb.update_brir_param, tag='hrtf_direction_misalign_comp')
                                            with dpg.tooltip("hrtf_direction_misalign_comp"):
                                                dpg.add_text("This can be used to correct common orientation errors in SOFA datasets")
                                                dpg.add_text("Use this if the azimuth or elevation directions appear reversed, rotated, or start from the wrong reference point")
                                                dpg.add_text("Refer to reference tables tab for more details")
                                        with dpg.group():
                                            dpg.add_loading_indicator(style=1, pos = (488,375), radius =1.9, color =(120,120,120),show=False, tag='hrtf_average_fav_load_ind')
                                            dpg.add_text("Listener", tag='brir_hrtf_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_listbox(hrtf_list_loaded, default_value=hrtf_loaded, num_items=17, width=245, callback=cb.select_hrtf, tag='brir_hrtf')
                                            with dpg.tooltip("brir_hrtf_title"):
                                                dpg.add_text("This will influence the externalisation and localisation of sounds around the listener")
                                            with dpg.group(horizontal=True):
                                                dpg.add_text("  ")
                                                dpg.add_button(label="Add Favourite", tag="hrtf_add_favourite", callback=cb.add_hrtf_favourite_qc_callback,show=add_fav_button_show,user_data=loaded_values["hrtf_list_favs"])

                                                dpg.add_button(label="Remove Favourite", tag="hrtf_remove_favourite", callback=cb.remove_hrtf_favourite_qc_callback,show=remove_fav_button_show)
                                                dpg.add_text("  ")
                                                dpg.add_button(label="Create Average", tag="hrtf_average_favourite", callback=cb.create_hrtf_favourite_avg,show=remove_fav_button_show)
                                                dpg.bind_item_theme(dpg.last_item(), "__theme_k")
                                                with dpg.tooltip("hrtf_average_favourite"):
                                                    dpg.add_text("Creates an averaged HRTF by interpolating across favourite listeners")
                                                dpg.add_button(label="Open Folder", tag="open_user_sofa_folder", callback=cb.open_user_sofa_folder,show=sofa_folder_button_show)
                                                
                                with dpg.tab(label="EQ Parameters",tag='eq_tab', parent="brir_tab_bar"): 

                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Room Target", tag='rm_target_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_listbox(CN.ROOM_TARGET_LIST, default_value=loaded_values["room_target"], num_items=11, width=345, tag='room_target', callback=cb.select_room_target)
                                            with dpg.tooltip("rm_target_title"):
                                                dpg.add_text("This will influence the overall balance of low and high frequencies")
                                            dpg.add_separator()
                                            dpg.add_text("Headphone Compensation", tag='brir_hp_type_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_listbox(CN.HP_COMP_LIST, default_value=loaded_values["brir_hp_type"], num_items=5, width=345, callback=cb.select_hp_comp, tag='brir_hp_type')
                                            with dpg.tooltip("brir_hp_type_title"):
                                                dpg.add_text("This will compensate typical interactions between the headphone and the outer ear")
                                                dpg.add_text("Selection should align with the listener's headphone type")
                                                dpg.add_text("Reduce to low strength if sound localisation or timbre is compromised")
                                                
                                
                                                
                                with dpg.tab(label="Low-frequency Extension",tag='lfe_tab', parent="brir_tab_bar"): 
                                    with dpg.group(horizontal=True):
                                        with dpg.group():
                                            dpg.add_text("Integration Crossover Frequency", tag='crossover_f_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_combo(CN.SUB_FC_SETTING_LIST, default_value=loaded_values["crossover_f_mode"], width=130, callback=cb.update_crossover_f, tag='crossover_f_mode')
                                            with dpg.tooltip("crossover_f_mode"):
                                                dpg.add_text("Auto Select mode will select an optimal frequency for the selected acoustic space")
                                            dpg.add_input_int(label="Crossover Frequency (Hz)",width=140, tag='crossover_f', min_value=CN.SUB_FC_MIN, max_value=CN.SUB_FC_MAX, default_value=loaded_values["crossover_f"],min_clamped=True, max_clamped=True, callback=cb.update_crossover_f)
                                            with dpg.tooltip("crossover_f"):
                                                dpg.add_text("Crossover Frequency can be adjusted to a value up to 150Hz")
                                                dpg.add_text("This will tune the integration of the cleaner LF response and original room response")
                                                dpg.add_text("Higher values may result in a smoother bass response")
                                                dpg.add_text("Set to 0Hz to disable this feature")
                                            dpg.add_separator()
                                            dpg.add_text("Response for Low-frequency Extension", tag='sub_response_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_listbox(CN.SUB_RESPONSE_LIST_GUI, default_value=loaded_values["sub_response"], num_items=11, width=350, callback=cb.select_sub_brir, tag='sub_response')
                                            with dpg.tooltip("sub_response_title"):
                                                dpg.add_text("Refer to reference tables tab and filter preview for comparison")
                                            dpg.add_separator()
                                            dpg.add_text("Additonal EQ", tag='hp_rolloff_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            with dpg.group(horizontal=True):
                                                dpg.add_checkbox(label="Compensate Headphone Roll-off", default_value = loaded_values["hp_rolloff_comp"],  tag='hp_rolloff_comp', callback=cb.update_brir_param)
                                                with dpg.tooltip("hp_rolloff_comp"):
                                                    dpg.add_text("This will compensate the typical reduction in bass response at lower frequencies")
                                                dpg.add_text("      ")
                                     
                                            
                                       
                                            
                                                
                        with dpg.child_window(autosize_x=True, height=175):
                            with dpg.tab_bar(tag='brir_out_tab_bar'):
                                with dpg.tab(label="Quick Config - Equalizer APO", parent="brir_out_tab_bar",tag='brir_apply_eapo_tab'):
                                    with dpg.group(horizontal=True):
                                        subtitle_6_qc = dpg.add_text("Apply Simulation in Equalizer APO")
                                        dpg.bind_item_font(subtitle_6_qc, font_b_def)
                                        dpg.add_text("                                                      ")
                          
                                    dpg.add_separator()
                                    
                                    with dpg.group(horizontal=True):
                                
                                        dpg.add_image_button(texture_tag='start_blue',width=40,height=40,tag="e_apo_brir_tag",callback=cb.e_apo_apply_brir_params,frame_padding=0,background_color=[0, 0, 0, 0],user_data=CN.PROCESS_BRIRS_RUNNING)#user data is thread running flag
                                        dpg.bind_item_theme(dpg.last_item(), "transparent_image_button_theme")
                                        with dpg.tooltip("e_apo_brir_tag"):
                                            dpg.add_text("This will generate the binaural simulation and apply it in Equalizer APO")  
                                        dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=30, width=397,pos = (60,65), overlay=CN.PROGRESS_START, tag="e_apo_progress_bar_brir",user_data=CN.STOP_THREAD_FLAG)#user data is thread cancel flag
                                        dpg.bind_item_theme(dpg.last_item(), "__theme_h")
                                        dpg.add_text(" ")
                                        add_image_checkbox(label="Enable Binaural Room Simulation",width=40,height=40,checkbox_tag="e_apo_brir_conv",callback=cb.e_apo_toggle_brir_gui,user_data=[],default_value=loaded_values["e_apo_brir_conv"])
                             
                                    dpg.add_separator()
               
                                    with dpg.group():
                                        dpg.add_text("Active Simulation: ")
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                                        dpg.add_text(default_value=loaded_values["e_apo_curr_brir_set"],  tag='e_apo_curr_brir_set' , wrap=500, user_data=False)#user data used for flagging use of below BRIR dict
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                                        dpg.bind_item_theme(dpg.last_item(), "__theme_f")
                                        dpg.add_text(default_value=loaded_values["e_apo_sel_brir_set"], tag='e_apo_sel_brir_set',show=False, user_data={})#user data used for storing snapshot of BRIR dict 
                                        dpg.add_text(default_value=loaded_values["e_apo_sel_brir_set_ts"], tag='e_apo_sel_brir_set_ts',show=False)#timestamp of last brir dataset export
                                        
                                with dpg.tab(label="Binaural Simulation Export", parent="brir_out_tab_bar",tag='brir_export_tab'):
                                   with dpg.tab_bar(tag='brir_export_tab_bar'):
                                       with dpg.tab(label="Export", parent="brir_export_tab_bar",tag='brir_export_trigger_tab'):
                                            subtitle_6 = dpg.add_text("Generate and Export Binaural Simulation")
                                            dpg.bind_item_font(subtitle_6, font_b_def)
                                            dpg.add_separator()
                                            with dpg.group(horizontal=True):
                                               
                                                dpg.add_image_button(texture_tag='start_green',width=37,height=37,tag="fde_brir_tag",callback=cb.fde_start_process_brirs,frame_padding=0,background_color=[0, 0, 0, 0],user_data=CN.PROCESS_BRIRS_RUNNING)
                                                dpg.bind_item_theme(dpg.last_item(), "transparent_image_button_theme")
                                                
                                                with dpg.tooltip("fde_brir_tag"):
                                                    dpg.add_text("This will generate the binaural dataset and export to the output directory. This may take some time to process")
                                                    
                                                dpg.add_progress_bar(label="Progress Bar", default_value=0.0, height=32, width=440, overlay=CN.PROGRESS_START_ALT, tag="fde_progress_bar_brir",user_data=CN.STOP_THREAD_FLAG)
                                                dpg.bind_item_theme(dpg.last_item(), "__theme_g")
                                       with dpg.tab(label="Options", parent="brir_export_tab_bar",tag='brir_export_files_tab'):
                                            subtitle_5 = dpg.add_text("Select Files to Include in Export")
                                            dpg.bind_item_font(subtitle_5, font_b_def)
                                            dpg.add_separator()
                                            with dpg.group(horizontal=True):
                                                dpg.add_checkbox(label="Direction-specific WAV BRIRs", default_value = loaded_values["fde_dir_brir_toggle"],  tag='fde_dir_brir_toggle', callback=cb.update_brir_param, show=generic_brir_exp_show)
                                                with dpg.tooltip("fde_dir_brir_toggle", tag='dir_brir_tooltip', show=generic_brir_exp_show):
                                                    dpg.add_text("Binaural Room Impulse Responses (BRIRs) in WAV format for convolution")
                                                    dpg.add_text("2 channels per file. One file for each direction")
                                                dpg.add_checkbox(label="True Stereo WAV BRIR", default_value = loaded_values["fde_ts_brir_toggle"],  tag='fde_ts_brir_toggle', callback=cb.update_brir_param, show=generic_brir_exp_show)
                                                with dpg.tooltip("fde_ts_brir_toggle", tag='fde_ts_brir_tooltip', show=generic_brir_exp_show):
                                                    dpg.add_text("True Stereo BRIR in WAV format for convolution. Compatible with EasyEffects")
                                                    dpg.add_text("4 channels. One file representing L and R speakers")
                                                dpg.add_checkbox(label="SOFA File", default_value = loaded_values["fde_sofa_brir_toggle"],  tag='fde_sofa_brir_toggle', callback=cb.update_brir_param, show=sofa_brir_exp_show)
                                                with dpg.tooltip("fde_sofa_brir_toggle", tag='fde_sofa_brir_tooltip', show=sofa_brir_tooltip_show):
                                                    dpg.add_text("BRIR dataset as a SOFA file")
                                            with dpg.group(horizontal=True):
                                                dpg.add_checkbox(label="HeSuVi WAV BRIRs", default_value = loaded_values["fde_hesuvi_brir_toggle"],  tag='fde_hesuvi_brir_toggle', callback=cb.update_brir_param, show=generic_brir_exp_show)  
                                                with dpg.tooltip("fde_hesuvi_brir_toggle", tag='fde_hesuvi_brir_tooltip', show=generic_brir_exp_show):
                                                    dpg.add_text("BRIRs in HeSuVi compatible WAV format. 14 channels, 44.1kHz and 48kHz")
                                                    
                                                dpg.add_checkbox(label="16-Channel WAV BRIRs", default_value = loaded_values["fde_multi_chan_brir_toggle"],  tag='fde_multi_chan_brir_toggle', callback=cb.update_brir_param, show=generic_brir_exp_show)
                                                with dpg.tooltip("fde_multi_chan_brir_toggle", tag='fde_multi_chan_brir_tooltip', show=generic_brir_exp_show):
                                                    dpg.add_text("BRIRs in FFMPEG compatible WAV format. 16 channels")
                                               
                                            dpg.add_separator()
                                            with dpg.group(horizontal=True):
                                                dpg.add_text("Spatial Resolution:", tag= "brir_spat_res_title")
                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                dpg.add_radio_button(CN.SPATIAL_RES_LIST_LIM, horizontal=True, tag= "fde_brir_spat_res", default_value=loaded_values["fde_brir_spat_res"], callback=cb.fde_select_spatial_resolution )
                                                with dpg.tooltip("brir_spat_res_title"):
                                                    dpg.add_text("Increasing resolution will increase number of directions available but will also increase processing time and dataset size")
                                                    dpg.add_text("'Low' is recommended unless additional directions or SOFA export is required")
                                                    dpg.add_text("Refer to reference tables tab for comparison")
                                           
                    
                    #right most section
                    with dpg.group():    
                        #Section for channel config, plotting
                        with dpg.child_window(width=590, height=440):
                            with dpg.group(width=590):
                                with dpg.tab_bar(tag='inner_tab_bar'):
                                    
                                    with dpg.tab(label="E-APO Configuration", parent="inner_tab_bar",tag='qc_e_apo_conf_tab',user_data=e_apo_conf_lock):
                                        with dpg.group(horizontal=True):
                                            with dpg.group():
                                                with dpg.group(horizontal=True):
                                                    
                                                    dpg.add_text("Configure Audio Channels in Equalizer APO                                      ")
                                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                    dpg.add_text("Reset Configuration:  ")
                                                    dpg.add_button(label="Reset to Default",  callback=cb.reset_channel_config)
                                                
                                                dpg.add_separator()
                                                with dpg.group(horizontal=True):
                                                    dpg.add_text("Audio Channels:           ")
                                                    dpg.add_combo(CN.AUDIO_CHANNELS, width=210, label="",  tag='e_apo_audio_channels',default_value=loaded_values["e_apo_audio_channels"], callback=cb.e_apo_select_channels_gui)
                                                    dpg.add_text("               ")
                              
                                                    dpg.add_text("Spatial Resolution:", tag= "e_apo_spat_res_title")
                                                    #dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                    dpg.add_combo(CN.SPATIAL_RES_LIST_E_APO, label="", width=80, default_value=loaded_values["e_apo_brir_spat_res"],  tag='e_apo_brir_spat_res', callback=cb.update_brir_param)
                                                    with dpg.tooltip("e_apo_spat_res_title"):
                                                        dpg.add_text("Increasing resolution will increase number of directions available but will also increase processing time and dataset size")
                                                        dpg.add_text("'Low' is recommended unless additional elevations are required")
                                                        dpg.add_text("Refer to reference tables tab for comparison") 
                                                with dpg.group(horizontal=True):
                                                    dpg.add_text("Auto-Adjust Preamp:   ")
                                                    dpg.add_combo(CN.AUTO_GAIN_METHODS, label="", width=210, default_value = loaded_values["e_apo_prevent_clip"],  tag='e_apo_prevent_clip', callback=cb.e_apo_config_acquire_gui)
                                                    with dpg.tooltip("e_apo_prevent_clip"):
                                                        dpg.add_text("This will auto-adjust the preamp so that the peak gain is at 0dB for 2.0 inputs")
                                                with dpg.group(horizontal=True):
                                                    dpg.add_text("Preamplification (dB):  ")
                                                    dpg.add_input_float(label=" ",format="%.1f", width=100,min_value=-100, max_value=20,min_clamped=True,max_clamped=True, tag='e_apo_gain_oa',default_value=loaded_values["e_apo_gain_oa"], callback=cb.e_apo_adjust_preamp)
                                                    
                                                        
                                                
                                                
                                                with dpg.group(horizontal=True):
                                                    # --- Input Channels / Peak Gain Table ---
                                                    with dpg.table(
                                                        header_row=True, 
                                                        policy=dpg.mvTable_SizingFixedFit, 
                                                        resizable=False,
                                                        borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                                        borders_outerV=True, borders_innerV=True, delay_search=True
                                                    ):
                                                        dpg.add_table_column(label="Input Channels")
                                                        dpg.add_table_column(label="Peak Gain (dB)")
                                                
                                                        # Row 1: 2.0 Stereo
                                                        with dpg.table_row():
                                                            dpg.add_text('2.0 Stereo')
                                                            dpg.add_text(tag='e_apo_gain_peak_2_0')
                                                
                                                        # Row 2: 5.1 Surround
                                                        with dpg.table_row():
                                                            dpg.add_text('5.1 Surround')
                                                            dpg.add_text(tag='e_apo_gain_peak_5_1')
                                                
                                                        # Row 3: 7.1 Surround
                                                        with dpg.table_row():
                                                            dpg.add_text('7.1 Surround')
                                                            dpg.add_text(tag='e_apo_gain_peak_7_1')
                                                
                                                    dpg.add_text('   ')  # spacing
                                                
                                                    # --- Upmixing Parameters Table ---
                                                    with dpg.table(
                                                        header_row=True, 
                                                        policy=dpg.mvTable_SizingFixedFit, 
                                                        resizable=False,
                                                        borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                                        borders_outerV=True, borders_innerV=True, delay_search=True, 
                                                        tag='upmixing_table'
                                                    ):
                                                        dpg.add_table_column(label="Upmixing Parameter")
                                                        dpg.add_table_column(label="Value")
                                                
                                                        # Row 1: Upmixing Method
                                                        with dpg.table_row():
                                                            dpg.add_text('Upmixing Method', tag='e_apo_upmix_method_text')
                                                            with dpg.tooltip("e_apo_upmix_method_text"):
                                                                dpg.add_text('Method A performs simple channel duplication')
                                                                dpg.add_text('Method B includes Mid/Side channel separation')
                                                            dpg.add_combo(
                                                                CN.UPMIXING_METHODS, width=200, label="",  
                                                                tag='e_apo_upmix_method',
                                                                default_value=loaded_values["e_apo_upmix_method"],
                                                                callback=cb.e_apo_config_acquire_gui
                                                            )
                                                
                                                        # Row 2: Side Delay
                                                        with dpg.table_row():
                                                            dpg.add_text('Side Delay (samples)', tag='e_apo_side_delay_text')
                                                            with dpg.tooltip("e_apo_side_delay_text"):
                                                                dpg.add_text('Side channels are delayed by specified samples')
                                                            dpg.add_input_int(
                                                                label=" ", width=100, min_value=-100, max_value=100, 
                                                                tag='e_apo_side_delay', min_clamped=True, max_clamped=True,
                                                                default_value=loaded_values["e_apo_side_delay"],
                                                                callback=cb.e_apo_config_acquire_gui
                                                            )
                                                
                                                        # Row 3: Rear Delay
                                                        with dpg.table_row():
                                                            dpg.add_text('Rear Delay (samples)', tag='e_apo_rear_delay_text')
                                                            with dpg.tooltip("e_apo_rear_delay_text"):
                                                                dpg.add_text('Rear channels are delayed by specified samples')
                                                            dpg.add_input_int(
                                                                label=" ", width=100, min_value=-100, max_value=100, 
                                                                tag='e_apo_rear_delay', min_clamped=True, max_clamped=True,
                                                                default_value=loaded_values["e_apo_rear_delay"],
                                                                callback=cb.e_apo_config_acquire_gui
                                                            )
                                                                    
                                                       
                                                                    
                                                                    
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
                                                                                dpg.add_text("L",  tag='e_apo_chan_label_fl')
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                                with dpg.tooltip("e_apo_chan_label_fl"):
                                                                                    dpg.add_text("Front Left")
                                                                            elif i == 1:
                                                                                dpg.add_text("R",  tag='e_apo_chan_label_fr')
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                                with dpg.tooltip("e_apo_chan_label_fr"):
                                                                                    dpg.add_text("Front Right")
                                                                            elif i == 2:
                                                                                dpg.add_text("C & LFE",  tag='e_apo_chan_label_c')
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                                with dpg.tooltip("e_apo_chan_label_c"):
                                                                                    dpg.add_text("Front Center and Low Frequency / Subwoofer")
                                                                            elif i == 3:
                                                                                dpg.add_text("SL",  tag='e_apo_chan_label_sl')
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                                with dpg.tooltip("e_apo_chan_label_sl"):
                                                                                    dpg.add_text("Surround Left")
                                                                            elif i == 4:
                                                                                dpg.add_text("SR",  tag='e_apo_chan_label_sr')
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                                with dpg.tooltip("e_apo_chan_label_sr"):
                                                                                    dpg.add_text("Surround Right")
                                                                            elif i == 5:
                                                                                dpg.add_text("RL",  tag='e_apo_chan_label_rl')
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                                with dpg.tooltip("e_apo_chan_label_rl"):
                                                                                    dpg.add_text("Back Left / Rear Left")
                                                                            elif i == 6:
                                                                                dpg.add_text("RR",  tag='e_apo_chan_label_rr')
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                                with dpg.tooltip("e_apo_chan_label_rr"):
                                                                                    dpg.add_text("Back Right / Rear Right")
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
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION, tag='e_apo_elev_angle_fl_tooltip')
                                                                            elif i == 1:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_fr',default_value=loaded_values["e_apo_elev_angle_fr"], callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_fr"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION, tag='e_apo_elev_angle_fr_tooltip')
                                                                            elif i == 2:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_c',default_value=loaded_values["e_apo_elev_angle_c"], callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_c"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION, tag='e_apo_elev_angle_c_tooltip')
                                                                            elif i == 3:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_sl',default_value=loaded_values["e_apo_elev_angle_sl"], callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_sl"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION, tag='e_apo_elev_angle_sl_tooltip')
                                                                            elif i == 4:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_sr',default_value=loaded_values["e_apo_elev_angle_sr"], callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_sr"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION, tag='e_apo_elev_angle_sr_tooltip')
                                                                            elif i == 5:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_rl',default_value=loaded_values["e_apo_elev_angle_rl"], callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_rl"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION, tag='e_apo_elev_angle_rl_tooltip')
                                                                            elif i == 6:
                                                                                dpg.add_combo(elevation_list_sel, width=70, label="",  tag='e_apo_elev_angle_rr',default_value=loaded_values["e_apo_elev_angle_rr"], callback=cb.e_apo_activate_direction_gui)
                                                                                with dpg.tooltip("e_apo_elev_angle_rr"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION, tag='e_apo_elev_angle_rr_tooltip')
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
                                                
                                         
                                                        
                                                        
                                    with dpg.tab(label="Export Configuration", parent="inner_tab_bar",tag='hcc_tab'):
                                        dpg.add_text("Configure Channels for HeSuVi & 16-Channel Exports")
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                                        dpg.add_button(label="Reset to Default",  callback=cb.reset_channel_config)
                                        with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                                            borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                                            borders_outerV=True, borders_innerV=True, delay_search=True):
                                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                            dpg.add_table_column(label="Channel")
                                                            dpg.add_table_column(label="Gain (dB)")
                                                            dpg.add_table_column(label="Elevation Angle (°)")
                                                            dpg.add_table_column(label="Azimuth Angle (°)")
    
                                                            for i in range(7):
                                                                with dpg.table_row():
                                                                    for j in range(4):
                                                                        if j == 0:#channel
                                                                            if i == 0:
                                                                                dpg.add_text("FL",  tag='fde_chan_label_fl')
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                                with dpg.tooltip("fde_chan_label_fl"):
                                                                                    dpg.add_text("Front Left")
                                                                            elif i == 1:
                                                                                dpg.add_text("FR",  tag='fde_chan_label_fr')
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                                with dpg.tooltip("fde_chan_label_fr"):
                                                                                    dpg.add_text("Front Right")
                                                                            elif i == 2:
                                                                                dpg.add_text("FC & LFE",  tag='fde_chan_label_fc')
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                                with dpg.tooltip("fde_chan_label_fc"):
                                                                                    dpg.add_text("Front Center and Low Frequency / Subwoofer")
                                                                            elif i == 3:
                                                                                dpg.add_text("SL",  tag='fde_chan_label_sl')
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                                with dpg.tooltip("fde_chan_label_sl"):
                                                                                    dpg.add_text("Surround Left")
                                                                            elif i == 4:
                                                                                dpg.add_text("SR",  tag='fde_chan_label_sr')
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                                with dpg.tooltip("fde_chan_label_sr"):
                                                                                    dpg.add_text("Surround Right")
                                                                            elif i == 5:
                                                                                dpg.add_text("BL",  tag='fde_chan_label_bl')
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                                with dpg.tooltip("fde_chan_label_bl"):
                                                                                    dpg.add_text("Back Left / Rear Left")
                                                                            elif i == 6:
                                                                                dpg.add_text("BR",  tag='fde_chan_label_br')
                                                                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                                                                with dpg.tooltip("fde_chan_label_br"):
                                                                                    dpg.add_text("Back Right / Rear Right")
                                                                        if j == 1:#gain
                                                                            if i == 0:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='fde_gain_fl',min_clamped=True,max_clamped=True, default_value=loaded_values["fde_gain_fl"])
                                                                            elif i == 1:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='fde_gain_fr',min_clamped=True,max_clamped=True,default_value=loaded_values["fde_gain_fr"])
                                                                            elif i == 2:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='fde_gain_c',min_clamped=True,max_clamped=True,default_value=loaded_values["fde_gain_c"])
                                                                            elif i == 3:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='fde_gain_sl',min_clamped=True,max_clamped=True,default_value=loaded_values["fde_gain_sl"])
                                                                            elif i == 4:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='fde_gain_sr',min_clamped=True,max_clamped=True,default_value=loaded_values["fde_gain_sr"])
                                                                            elif i == 5:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='fde_gain_rl',min_clamped=True,max_clamped=True,default_value=loaded_values["fde_gain_rl"])
                                                                            elif i == 6:
                                                                                dpg.add_input_float(label=" ", format="%.1f", width=90,min_value=-100, max_value=20, tag='fde_gain_rr',min_clamped=True,max_clamped=True,default_value=loaded_values["fde_gain_rr"])
                                                                        if j == 2:#elevation
                                                                            if i == 0:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='fde_elev_angle_fl',default_value=loaded_values["fde_elev_angle_fl"])
                                                                                with dpg.tooltip("fde_elev_angle_fl"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 1:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='fde_elev_angle_fr',default_value=loaded_values["fde_elev_angle_fr"])
                                                                                with dpg.tooltip("fde_elev_angle_fr"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 2:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='fde_elev_angle_c',default_value=loaded_values["fde_elev_angle_c"])
                                                                                with dpg.tooltip("fde_elev_angle_c"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 3:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='fde_elev_angle_sl',default_value=loaded_values["fde_elev_angle_sl"])
                                                                                with dpg.tooltip("fde_elev_angle_sl"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 4:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='fde_elev_angle_sr',default_value=loaded_values["fde_elev_angle_sr"])
                                                                                with dpg.tooltip("fde_elev_angle_sr"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 5:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='fde_elev_angle_rl',default_value=loaded_values["fde_elev_angle_rl"])
                                                                                with dpg.tooltip("fde_elev_angle_rl"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                            elif i == 6:
                                                                                dpg.add_combo(elevation_list_sel, width=80, label="",  tag='fde_elev_angle_rr',default_value=loaded_values["fde_elev_angle_rr"])
                                                                                with dpg.tooltip("fde_elev_angle_rr"):
                                                                                    dpg.add_text(CN.TOOLTIP_ELEVATION)
                                                                        if j == 3:#azimuth
                                                                            if i == 0:
                                                                                dpg.add_combo(CN.AZ_ANGLES_FL_WAV, width=80, label="",  tag='fde_az_angle_fl',default_value=loaded_values["fde_az_angle_fl"])
                                                                                with dpg.tooltip("fde_az_angle_fl"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                            elif i == 1:
                                                                                dpg.add_combo(CN.AZ_ANGLES_FR_WAV, width=80, label="",  tag='fde_az_angle_fr',default_value=loaded_values["fde_az_angle_fr"])
                                                                                with dpg.tooltip("fde_az_angle_fr"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                            elif i == 2:
                                                                                dpg.add_combo(CN.AZ_ANGLES_C_WAV, width=80, label="",  tag='fde_az_angle_c',default_value=loaded_values["fde_az_angle_c"])
                                                                                with dpg.tooltip("fde_az_angle_c"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                            elif i == 3:
                                                                                dpg.add_combo(CN.AZ_ANGLES_SL_WAV, width=80, label="",  tag='fde_az_angle_sl',default_value=loaded_values["fde_az_angle_sl"])
                                                                                with dpg.tooltip("fde_az_angle_sl"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                            elif i == 4:
                                                                                dpg.add_combo(CN.AZ_ANGLES_SR_WAV, width=80, label="",  tag='fde_az_angle_sr',default_value=loaded_values["fde_az_angle_sr"])
                                                                                with dpg.tooltip("fde_az_angle_sr"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                            elif i == 5:
                                                                                dpg.add_combo(CN.AZ_ANGLES_RL_WAV, width=80, label="",  tag='fde_az_angle_rl',default_value=loaded_values["fde_az_angle_rl"])
                                                                                with dpg.tooltip("fde_az_angle_rl"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                            elif i == 6:
                                                                                dpg.add_combo(CN.AZ_ANGLES_RR_WAV, width=80, label="",  tag='fde_az_angle_rr',default_value=loaded_values["fde_az_angle_rr"])
                                                                                with dpg.tooltip("fde_az_angle_rr"):
                                                                                    dpg.add_text(CN.TOOLTIP_AZIMUTH)
                                                                                    
                                        dpg.add_separator()
                                        dpg.add_text("Channel Order for 16-Channel Exports")
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                                        with dpg.group():
                                            dpg.add_listbox(label="", width=400, items=CN.PRESET_16CH_LABELS, num_items=6, default_value=loaded_values["mapping_16ch_wav"], tag="mapping_16ch_wav")        
                                            
                                            
                                    with dpg.tab(label="Presets", parent="inner_tab_bar",tag='preset_tab'):
                                        dpg.add_text("Manage Presets")
                                        dpg.bind_item_font(dpg.last_item(), font_b_def)
                                        dpg.add_separator()
                                        with dpg.group(horizontal=True):
          
                                            
                                            dpg.add_button(label="Save Preset", tag="save_preset", callback=cb.save_preset_callback)
                                            with dpg.tooltip("save_preset"):
                                                dpg.add_text("Saves current parameters as a preset")
                                            dpg.add_text("     ")
                                            dpg.add_button(label="Delete Preset", tag="delete_selected_preset", callback=lambda: dpg.configure_item("del_preset_popup", show=True))
                                            with dpg.tooltip("delete_selected_preset"):
                                                dpg.add_text("Deletes the selected preset")
                                            dpg.add_text("     ")
                                            dpg.add_button(label="Rename Preset", tag="rename_selected_preset", callback = cb.rename_selected_preset_callback)
                                            with dpg.tooltip("rename_selected_preset"):
                                                dpg.add_text("Enter new name in the text box")
                                            dpg.add_input_text(label="", tag="rename_selected_preset_text", width=200)
                                
                                        with dpg.popup("delete_selected_preset", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="del_preset_popup"):
                                            dpg.add_text("Selected preset will be deleted")
                                            dpg.add_separator()
                                            with dpg.group(horizontal=True):
                                                dpg.add_button(label="OK", width=75, callback=cb.delete_selected_preset_callback)
                                                dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("del_preset_popup", show=False))
                                        dpg.add_listbox(
                                            items=[],
                                            tag="preset_list",
                                            width=574,
                                            num_items=18
                                        )
                                        dpg.bind_item_font(dpg.last_item(), font_b_small)
                                        with dpg.group(horizontal=True):
                                            dpg.add_button(label="Load Preset", tag="load_selected_preset", callback=cb.load_selected_preset_callback)
                                            with dpg.tooltip("load_selected_preset"):
                                                dpg.add_text("This will load the selected preset parameters")
                                          
  
                        
                                    with dpg.tab(label="Filter Preview", parent="inner_tab_bar",tag='fp_tab'):
                                        #plotting
                                        #dpg.add_separator()
                                        with dpg.group(horizontal=True):
                                            #dpg.add_text("Select a filter to preview")
                                            #dpg.add_text("                                                       ")  
                                            dpg.add_text("Elevation: ")    
                                            dpg.add_combo(CN.ELEV_ANGLES_WAV_MED, default_value='0', width=50,callback=cb.select_hrtf, tag='filter_prev_elev')
                                            dpg.add_text("Azimuth: ")    
                                            dpg.add_combo(CN.AZ_ANGLES_ALL_WAV_SORTED, default_value='-30', width=60,callback=cb.select_hrtf, tag='filter_prev_azim')
                                            dpg.add_text("Channel: ")    
                                            dpg.add_combo(CN.CHANNELS_HEAD, default_value=CN.CHANNELS_HEAD[0], width=50,callback=cb.select_hrtf, tag='filter_prev_channel')
                                            dpg.add_text("     Plot Type: ")    
                                            dpg.add_combo(CN.PLOT_TYPE_LIST, default_value=CN.PLOT_TYPE_LIST[0], width=150,callback=cb.plot_type_changed, tag='plot_type',user_data={})# plot_state will live here
                                        
                                        # create plot
                                        with dpg.plot(label="Filter Analysis", height=375, width=575):
                                            # optionally create legend
                                            dpg.add_plot_legend()
                                    
                                            # REQUIRED: create x and y axes
                                            dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)", tag="x_axis", scale=dpg.mvPlotScale_Log10)#, log_scale=True
                                            dpg.set_axis_limits("x_axis", 10, 20000)
                                            dpg.add_plot_axis(dpg.mvYAxis, label="Magnitude (dB)", tag="y_axis")
                                            dpg.set_axis_limits("y_axis", -20, 15)
                                    
                                            # series belong to a y axis
                                            dpg.add_line_series(default_x, default_y, label="Plot", parent="y_axis", tag="series_tag",skip_nan=True)
                                            #initial plot
                                            hf.plot_fir_generic(fir_array=impulse,view=CN.PLOT_TYPE_LIST[0],title_name='Select a filter to preview',samp_freq=CN.SAMP_FREQ,n_fft=CN.N_FFT,plot_dest=CN.TAB_QC_CODE)
                                        
                                    
                                    with dpg.tab(label="Integration Analysis", parent="inner_tab_bar",tag='ia_tab'):
                                        #plotting
                                        #dpg.add_separator()
                                        with dpg.group(horizontal=True):
                                            # dpg.add_text("Refreshes after parameters are applied",tag='ia_tab_text')
                                            # with dpg.tooltip("ia_tab_text"):
                                            #     dpg.add_text("Sample response is taken from 0° elevation, -30° azimuth, left ear")
                                                
                                            dpg.add_text("Elevation: ")    
                                            dpg.add_combo(CN.ELEV_ANGLES_WAV_LOW, default_value='0', width=50,callback=cb.ia_new_plot, tag='ia_plot_elev')
                                            dpg.add_text("Azimuth: ")    
                                            dpg.add_combo(CN.AZ_ANGLES_ALL_WAV_SORTED, default_value='-30', width=60,callback=cb.ia_new_plot, tag='ia_plot_azim')
                                            dpg.add_text("Channel: ")    
                                            dpg.add_combo(CN.CHANNELS_HEAD, default_value=CN.CHANNELS_HEAD[0], width=50,callback=cb.ia_new_plot, tag='ia_plot_channel')
                              
                                            dpg.add_text("     Plot Type: ")    
                                            dpg.add_combo(CN.PLOT_TYPE_LIST_IA, default_value=CN.PLOT_TYPE_LIST_IA[0], width=150,callback=cb.ia_new_plot, tag='ia_plot_type',user_data={})# plot_state will live here
                                 
                                        # create plot
                                        with dpg.plot(label="Integrated Binaural Response", height=375, width=575):
                                            # optionally create legend
                                            dpg.add_plot_legend(tag="ia_legend_tag")
                                            
                                            # REQUIRED: create x and y axes
                                            dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)", tag="ia_x_axis", scale=dpg.mvPlotScale_Log10)#, log_scale=True
                                            dpg.set_axis_limits("ia_x_axis", 10, 20000)
                                            dpg.add_plot_axis(dpg.mvYAxis, label="Magnitude (dB)", tag="ia_y_axis")
                                            dpg.set_axis_limits("ia_y_axis", -20, 15)
                                            
                                    
                                            # series belong to a y axis
                                            dpg.add_line_series(default_x, default_y, label="Plot", parent="ia_y_axis", tag="ia_series_tag",skip_nan=True)
                                            #initial plot
                                            hf.plot_fir_generic(fir_array=impulse,view=CN.PLOT_TYPE_LIST[0],title_name='Refreshes after parameters are applied',samp_freq=CN.SAMP_FREQ,n_fft=CN.N_FFT,plot_dest=CN.TAB_QC_IA_CODE)
                                            
                                   
                                  
                                        
                     
                        #Section for misc settings
                        with dpg.group():
                            
                            with dpg.tab_bar(tag='out_config_tab_bar'):
                                with dpg.tab(label="Audio Format", parent="out_config_tab_bar",tag='ir_format_tab'):
                                    #Section for wav settings
                                    with dpg.child_window(width=590, height=166):
                                        title_4 = dpg.add_text("IR Format", tag='export_title')
                                        with dpg.tooltip("export_title"):
                                            dpg.add_text("Configure sample rate and bit depth of exported impulse responses")
                                            dpg.add_text("This should align with the format of the playback audio device")
                                        dpg.bind_item_font(title_4, font_b_def)
                                        #dpg.add_separator()
                                        with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                                            borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                                            borders_outerV=True, borders_innerV=True, delay_search=True):
                                            #dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_table_column(label="   Sample Rate")
                                            dpg.add_table_column(label="   Bit Depth")
                                            for i in range(1):
                                                with dpg.table_row():
                                                    for j in range(2):
                                                        if j == 0:#Sample Rate
                                                            with dpg.group(horizontal=True):
                                                                dpg.add_text("  ")
                                                                dpg.add_radio_button(CN.SAMPLE_RATE_LIST, horizontal=True, tag= "wav_sample_rate", default_value=loaded_values["wav_sample_rate"], callback=cb.sync_wav_sample_rate )
                                                                dpg.add_text("  ")
                                                        elif j == 1:#Bit Depth
                                                            with dpg.group(horizontal=True):
                                                                dpg.add_text("  ")
                                                                dpg.add_radio_button(CN.BIT_DEPTH_LIST, horizontal=True, tag= "wav_bit_depth", default_value=loaded_values["wav_bit_depth"], callback=cb.sync_wav_bit_depth)
                                                                dpg.add_text("  ")
                                        
                                        dpg.add_separator()
                                        with dpg.group(horizontal=True):
                                            dpg.add_text("Audio Device Format", tag='device_config_title')
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            with dpg.tooltip("device_config_title"):
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
                                                            dpg.add_text(" ", tag= "def_pb_device_name")
                                                        elif j == 1:#Sample Rate
                                                            dpg.add_text(" ", tag= "def_pb_device_sr")
                                                    
                                with dpg.tab(label="Output Data", parent="out_config_tab_bar",tag='output_location_tab'):
                                    #output locations
                                    with dpg.child_window(width=590, height=166):
                                        with dpg.group(horizontal=True):
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_text("Output Location", tag='out_dir_title')
                                            with dpg.tooltip("out_dir_title"):
                                                dpg.add_text("'EqualizerAPO\\config' directory should be selected if using HeSuVi")
                                                dpg.add_text("Main outputs will be saved under 'ASH-Outputs' sub directory") 
                                                dpg.add_text("HeSuVi outputs will be saved in 'EqualizerAPO\\config\\HeSuVi'")  
                                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                                            dpg.add_text("           ")
                                            # dpg.add_button(label="Open Export Folder",user_data="",tag="open_export_folder_tag", callback=cb.open_output_folder)
                                            # dpg.add_text("      ")
                                            dpge.add_file_browser(width=800,height=600,label='Change Export Folder',show_as_window=True, dirs_only=True,show_ok_cancel=True, allow_multi_selection=False, collapse_sequences=True,callback=cb.show_selected_folder)
                                        dpg.add_separator()
                                        
                                        # ---------- Output Location Table ----------
                                        with dpg.table(header_row=True,resizable=False,policy=dpg.mvTable_SizingFixedFit,row_background=True,borders_outerH=True,borders_innerH=True,borders_outerV=True,borders_innerV=True):
                                            dpg.add_table_column(label="Output Type", init_width_or_weight=140)
                                            dpg.add_table_column(label="Location", init_width_or_weight=540)
                                    
                                            # ---------- Main Exports ----------
                                            with dpg.table_row():
                                                dpg.add_text("Main Exports")
                                                dpg.bind_item_font(dpg.last_item(), font_b_small)
                                    
                                                dpg.add_text(tag="output_folder_fde")
                                                with dpg.tooltip("output_folder_fde"):
                                                    dpg.add_text("Location to save correction filters and binaural datasets",tag="output_folder_fde_tooltip")
                                    
                                            # ---------- HeSuVi Exports ----------
                                            with dpg.table_row():
                                                dpg.add_text("HeSuVi Exports")
                                                dpg.bind_item_font(dpg.last_item(), font_b_small)
                                    
                                                dpg.add_text(tag="output_folder_hesuvi")
                                                with dpg.tooltip("output_folder_hesuvi"):
                                                    dpg.add_text("Location to save HeSuVi files",tag="output_folder_hesuvi_tooltip")
                                            
                                            # ---------- Quick Config Outputs ----------
                                            with dpg.table_row():
                                                dpg.add_text("Quick Config E-APO WAVs")
                                                dpg.bind_item_font(dpg.last_item(), font_b_small)
                                    
                                                dpg.add_text(tag="output_folder_e_apo")
                                                with dpg.tooltip("output_folder_e_apo"):
                                                    dpg.add_text("Location to save WAV files for quick configuration in E-APO",tag="output_folder_e_apo_tooltip")
                                                    
                                       
                                        # ---------- Hidden State / Paths ----------
                                        dpg.add_text(tag="selected_folder_base", show=False)
                                        dpg.add_text(tag="e_apo_program_path", show=False)
                                        #section to store e apo config path
                                        dpg.add_text(tag='e_apo_config_folder', show=False)
                                        
                                        
                
                                        
                                    dpg.set_value('output_folder_fde', export_ash_path)
                                    dpg.set_value('output_folder_fde_tooltip', export_ash_path)
                                    dpg.set_value('selected_folder_base', export_base_path)
                                    dpg.set_value('output_folder_hesuvi', export_hesuvi_path)
                                    dpg.set_value('output_folder_hesuvi_tooltip', export_hesuvi_path)
                                    dpg.set_value('e_apo_program_path', e_apo_path)
                                    dpg.set_value('e_apo_config_folder', e_apo_config_path)
                                    dpg.set_value('output_folder_e_apo', e_apo_ash_path)
                                    dpg.set_value('output_folder_e_apo_tooltip', e_apo_ash_path)
                                
                        
            
            with dpg.tab(label="Acoustic Space Import Tool", tag='as_import_tab', parent="tab_bar"):     
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
                                            "Enter a name for the new acoustic space\n"
                                            "If left blank, folder name will be used"
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
                             
                                    dpg.add_text("Reverb Tail Mode")
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    dpg.add_combo(items=CN.AS_TAIL_MODE_LIST,default_value=CN.AS_TAIL_MODE_LIST[0],label="",tag="reverb_tail_mode",width=120)
                                    with dpg.tooltip("reverb_tail_mode"):
                                        dpg.add_text("Use short mode for spaces with fast decay (< 1.5s)")
                                        dpg.add_text("Only enable long mode if IRs have long decay tails (>= 1.5 s) as this will increase processing time")
                                        dpg.add_text("Windowed mode can be used to shorten and fade out long decay trails")
                                        
                        
                                # ---------- Low-frequency Mode ----------
                                with dpg.table_row():
                                    dpg.add_text("Low-frequency Mode")
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                        
                                    dpg.add_checkbox(label="Enable", tag="as_subwoofer_mode")
                                    with dpg.tooltip("as_subwoofer_mode"):
                                        dpg.add_text("Enable to treat the IRs as low-frequency responses")
                                        dpg.add_text("The result will appear under low-frequency extension options")
                                        
                                # ---------- Binaural Measurements ----------
                                with dpg.table_row():
                                    dpg.add_text("Binaural Measurements")
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                        
                                    dpg.add_checkbox(label="Enable",tag="binaural_meas_inputs", default_value=False)
                                    with dpg.tooltip("binaural_meas_inputs"):
                                        dpg.add_text("Enable if the IRs are binaural measurements")
                        
                                # ---------- Noise Reduction ----------
                                with dpg.table_row():
                                    dpg.add_text("Noise Reduction")
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                        
                                    dpg.add_checkbox(label="Enable", tag="noise_reduction_mode")
                                    with dpg.tooltip("noise_reduction_mode"):
                                        dpg.add_text("Enable if the IRs have a high noise floor")
                        
                                # ---------- Rise Time ----------
                                with dpg.table_row():
                                    dpg.add_text("Rise Time (ms)")
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                        
                                    dpg.add_input_float(tag="as_rise_time",width=120,default_value=4.0,min_value=1.0,max_value=20.0,format="%.2f",min_clamped=True,max_clamped=True)
                                    with dpg.tooltip("as_rise_time"):
                                        dpg.add_text("Applies a fade-in window of specified duration")
                                        dpg.add_text("Min: 1 ms, Max: 20 ms")
                                        
                                # ---------- Compensation Factor ----------
                                with dpg.table_row():
                                    dpg.add_text("Room Correction Factor")
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                        
                                    dpg.add_input_float(label="",tag="asi_rm_cor_factor",width=120,default_value=1.0,min_value=0.0,max_value=1.0,format="%.2f",min_clamped=True,max_clamped=True,)
                                    with dpg.tooltip("asi_rm_cor_factor"):
                                        dpg.add_text("Select a value between 0 and 1 to control the strength of room correction")# and pitch shift correction
                                        
                                # ---------- User selected listener ----------
                                with dpg.table_row():
                             
                                    dpg.add_text("Listener")
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    dpg.add_combo(items=CN.AS_LISTENER_TYPE_LIST,default_value=CN.AS_LISTENER_TYPE_LIST[0],label="",tag="as_listener",width=170)
                                    with dpg.tooltip("as_listener"):
                                        dpg.add_text("Controls which listener will be used to apply binaural transformations")
                                        dpg.add_text("If set to 'User Selection', the current listener selection in the first tab will be used")
                                        dpg.add_text("Not applicable when binaural measurements option is enabled")
                                        
                                        
                                # ---------- Alignment Frequency ----------
                                with dpg.table_row():
                                    dpg.add_text("Alignment Frequency (Hz)")
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                        
                                    dpg.add_input_int(tag="alignment_freq",width=120,default_value=100,min_value=50,max_value=150)
                                    with dpg.tooltip("alignment_freq"):
                                        dpg.add_text("Low-pass cutoff used for time-domain alignment")
                                        dpg.add_text("Min: 50 Hz, Max: 150 Hz")
                        
                                # ---------- Desired Directions ----------
                                with dpg.table_row():
                                    dpg.add_text("Desired Directions")
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                        
                                    dpg.add_input_int(tag="unique_directions",width=120,default_value=3000,min_value=2000,max_value=5000,min_clamped=True,max_clamped=True)
                                    with dpg.tooltip("unique_directions"):
                                        dpg.add_text("Target number of spatial source directions")
                                        dpg.add_text("Lower values reduce processing time")
                                        dpg.add_text("Min: 2000, Max: 5000")
               
                                # ---------- Pitch Shift Range ----------
                                with dpg.table_row():
                                    dpg.add_text("Pitch Shift Range")
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                                    with dpg.group():
                                        dpg.add_input_float(label="",tag="pitch_range_high",width=120,default_value=12.0,min_value=0.0,max_value=24.0,format="%.2f",min_clamped=True,max_clamped=True,)
                                    with dpg.tooltip("pitch_range_high"):
                                        dpg.add_text("Maximum pitch shift (semitones)")
                                        dpg.add_text("Used to expand sparse datasets")
                        
                                # ---------- Pitch Shift Compensation ----------
                                with dpg.table_row():
                                    dpg.add_text("Pitch Shift Correction")
                                    dpg.bind_item_font(dpg.last_item(), font_b_def)
                        
                                    dpg.add_checkbox(label="Enable",tag="pitch_shift_comp", default_value=True)
                                    with dpg.tooltip("pitch_shift_comp"):
                                        dpg.add_text("Corrects pitch after dataset expansion")
                        
                                
                                        
                          
                                        
                            
                
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
                                dpg.add_text("Displays the processing progress")
            
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
                                    dpg.add_text("Searches for imported acoustic spaces")
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

                 
            with dpg.tab(label="Room Target Tool",tag='room_target_tab', parent="tab_bar"):    
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
                                dpg.add_text("Set the cutoff frequency for the low-shelf filter \nRecommended range: 20-1000 Hz")
                        
                            dpg.add_input_float(label="Gain (dB)", tag="low_shelf_gain", width=200, default_value=6.0, min_value=-6.0, max_value=18.0, format="%.2f", 
                                                min_clamped=True, max_clamped=True, callback=cb.generate_room_target_callback, user_data={"generation_running": False, "save_to_file": False})
                            with dpg.tooltip("low_shelf_gain"):
                                dpg.add_text("Set the gain for the low-shelf filter \nNegative values attenuate low frequencies")
                        
                            dpg.add_input_float(label="Q-Factor", tag="low_shelf_q", width=200, default_value=0.707, min_value=0.1, max_value=5.0, format="%.2f",
                                                min_clamped=True, max_clamped=True, callback=cb.generate_room_target_callback, user_data={"generation_running": False, "save_to_file": False})
                            with dpg.tooltip("low_shelf_q"):
                                dpg.add_text("Set the Q-factor (slope) for the low-shelf filter \nLower values = broader effect")
                        
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
                                dpg.add_text("Set the cutoff frequency for the high-shelf filter \nRecommended range: 1-20 kHz")
                        
                            dpg.add_input_float(label="Gain (dB)", tag="high_shelf_gain", width=200, default_value=-4.0, min_value=-18.0, max_value=6.0, format="%.2f",
                                                min_clamped=True, max_clamped=True, callback=cb.generate_room_target_callback, user_data={"generation_running": False, "save_to_file": False})
                            with dpg.tooltip("high_shelf_gain"):
                                dpg.add_text("Set the gain for the high-shelf filter \nPositive values boost high frequencies")
                        
                            dpg.add_input_float(label="Q-Factor", tag="high_shelf_q", width=200, default_value=0.4, min_value=0.1, max_value=5.0, format="%.2f", 
                                                min_clamped=True, max_clamped=True, callback=cb.generate_room_target_callback, user_data={"generation_running": False, "save_to_file": False})
                            with dpg.tooltip("high_shelf_q"):
                                dpg.add_text("Set the Q-factor (slope) for the high-shelf filter \nLower values = broader effect")
               
                        with dpg.child_window(width=385, height=120):
                            dpg.add_text("Generate Room Target")
                            dpg.bind_item_font(dpg.last_item(), font_b_def)
                            dpg.add_separator()
                        
                            with dpg.group(horizontal=True):
                                dpg.add_text("Room Target Name:", tag="room_target_name_label")
                                dpg.add_text("", tag="room_target_name_display")
                                with dpg.tooltip("room_target_name_display"):
                                    dpg.add_text("Displays the name entered above for the new Room Target")
                        
                            with dpg.group(horizontal=True):
                                dpg.add_button(
                                    label="Generate Target",
                                    tag="generate_room_target_btn",
                                    callback=cb.generate_room_target_callback,
                                    user_data={"generation_running": False, "save_to_file": True}
                                )
                                with dpg.tooltip("generate_room_target_btn"):
                                    dpg.add_text("Apply the low/high shelf filter settings and generate a new room target curve")
                                    dpg.add_text("Newly created targets will be available in the quick config and dataset export tabs")
                        
                            with dpg.group(horizontal=True):
                                dpg.add_text("Progress:")
                                dpg.add_progress_bar(tag="progress_bar_target_gen", default_value=0.0, width=290)
                            with dpg.tooltip("progress_bar_target_gen"):
                                dpg.add_text("Displays progress while generating the new room target")
            
                    with dpg.group(horizontal=False):
                        with dpg.group(horizontal=True):  # Overall horizontal layout for left plot + right list
                            # Plot Panel - Left
                            with dpg.child_window(width=580, height=492):
                                dpg.add_text("Room Target Preview", tag="target_plot_title")
                                dpg.bind_item_font(dpg.last_item(), font_b_def)
                                with dpg.tooltip("target_plot_title"):
                                    dpg.add_text("Displays the magnitude response of the selected room target")
                        
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
                                        dpg.add_text("Scans for newly generated room target filters")
                        
                                    dpg.add_text("      ")
                                    dpg.add_button(label="Open Targets Folder", tag="open_target_folder_button", callback=cb.open_user_rt_folder)
                                    with dpg.tooltip("open_target_folder_button"):
                                        dpg.add_text("Opens the directory where room targets are stored")
                        
                                    dpg.add_text("      ")
                                    dpg.add_button(label="Delete Selected", tag="delete_selected_target_button", callback=lambda: dpg.configure_item("del_target_popup", show=True))
                                    with dpg.tooltip("delete_selected_target_button"):
                                        dpg.add_text("Deletes the selected room target and associated data")
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
                
                    
                
            with dpg.tab(label="Advanced Settings & Maintenance", tag="additional_tools", parent="tab_bar"):
                dpg.bind_item_theme(dpg.last_item(), "__theme_j")
            
                # Horizontal split: left table, right log
                with dpg.group(horizontal=True):
            
                    # --- Left: Table of Actions ---
                    with dpg.child_window(width=520, height=637):
                        
                        with dpg.collapsing_header(label="Advanced Settings", default_open=True):
     
                            with dpg.table(header_row=True, resizable=False, policy=dpg.mvTable_SizingFixedFit, row_background=True, borders_outerH=True,borders_innerH=True,borders_outerV=True,borders_innerV=True):
                                dpg.add_table_column(label="Section", init_width_or_weight=80)
                                dpg.add_table_column(label="Setting", init_width_or_weight=190)
                                dpg.add_table_column(label="Value", init_width_or_weight=200)
                
                                # ---------- Misc Parameters ----------
                                with dpg.table_row():
                                    dpg.add_text("HRTFs")
                                    dpg.add_text("Force Left/Right Symmetry")
                                    dpg.add_combo(CN.HRTF_SYM_LIST, default_value=loaded_values["force_hrtf_symmetry"], width=190,
                                                  callback=cb.update_brir_param, tag='force_hrtf_symmetry')
                                    with dpg.tooltip("force_hrtf_symmetry"):
                                        dpg.add_text("This will mirror the left or right sides of the HATS / dummy head")
                                        dpg.add_text("Applies to the direct sound. Reverberation is not modified")
                                        
                                with dpg.table_row():
                                    dpg.add_text("HRTFs")
                                    dpg.add_text("Direct Sound Polarity")
                                    dpg.add_combo(CN.HRTF_POLARITY_LIST, default_value=loaded_values["hrtf_polarity_rev"], width=190,
                                                  callback=cb.update_brir_param, tag='hrtf_polarity_rev')
                                    with dpg.tooltip("hrtf_polarity_rev"):
                                        dpg.add_text("This can be used to manually reverse the polarity of the direct sound")
                                        dpg.add_text("Reverberation is not modified")
                                        
                                with dpg.table_row():
                                    dpg.add_text("HRTFs")
                                    dpg.add_text("Diffuse-field Calibration")
                                    dpg.add_combo(CN.HRTF_DF_CAL_MODE_LIST, default_value=loaded_values["hrtf_df_cal_mode"], width=190,callback=cb.update_brir_param, tag='hrtf_df_cal_mode')
                                    with dpg.tooltip("hrtf_df_cal_mode"):
                                        dpg.add_text("Diffuse-field calibration of HRTF datasets is enabled by default but can be disabled")
                                        dpg.add_text("If disabled, HRTF datasets will retain their direction-independent information")
                                        dpg.add_text("Level spectrum ends option will also remove any roll-off from low and high frequencies")
                      
                                with dpg.table_row():
                                    dpg.add_text("HRTFs")
                                    dpg.add_text("Low-frequency Noise Suppression")
                                    dpg.add_checkbox(label="Enable", default_value=loaded_values["hrtf_low_freq_suppression"],tag="hrtf_low_freq_suppression", callback=cb.update_brir_param)
                                    with dpg.tooltip("hrtf_low_freq_suppression"):
                                        dpg.add_text("Reduces sub-bass energy to suppress unreliable low-frequency content in some HRIR measurements")
                                        dpg.add_text("Typically results in smoother low frequencies in the integrated binaural simulation")
                                        dpg.add_text("Applies to the direct sound. Reverberation is not modified")
                                        
                                with dpg.table_row():
                                    dpg.add_text("Reverberation")
                                    dpg.add_text("Early Reflection Delay (ms)")
                                    dpg.add_input_float(label=" ", width=190, format="%.1f", tag='er_delay_time',min_value=CN.ER_RISE_MIN, max_value=CN.ER_RISE_MAX,default_value=loaded_values["er_delay_time"], min_clamped=True,max_clamped=True, callback=cb.update_brir_param)
                                    with dpg.tooltip("er_delay_time"):
                                        dpg.add_text("This will increase the time between the direct sound and early reflections")
                                        dpg.add_text("This can be used to increase perceived distance")
                                        dpg.add_text("Min. 0ms, Max. 10ms")
                                        
                                with dpg.table_row():
                                    dpg.add_text("Reverberation")
                                    dpg.add_text("Reverb Tail Crop Threshold (dB)")
                                    dpg.add_input_float(label=" ", width=190, format="%.1f", tag='reverb_tail_crop_db',min_value=CN.REVERB_CROP_DB_MIN, max_value=CN.REVERB_CROP_DB_MAX,default_value=loaded_values["reverb_tail_crop_db"], min_clamped=True,max_clamped=True, callback=cb.update_brir_param)
                                    with dpg.tooltip("reverb_tail_crop_db"):
                                        dpg.add_text("Sets the amplitude threshold below which the reverb tail is cropped")
                                        dpg.add_text("Lower values preserve more tail; higher values crop more aggressively")
                                        dpg.add_text("Typical range: -70 dB to -100 dB")
                                        
                                with dpg.table_row():
                                    dpg.add_text("LF Extension")
                                    dpg.add_text("Forward-Backward Filtering")
                                    dpg.add_checkbox(label="Enable", default_value = loaded_values["fb_filtering"],  tag='fb_filtering', callback=cb.update_brir_param)
                                    with dpg.tooltip("fb_filtering"):
                                        dpg.add_text("This will eliminate the delays introduced by the filters when extending the low frequencies, however can introduce edge artefacts in some cases")
                                        
                
                                with dpg.table_row():
                                    dpg.add_text("SOFA")
                                    dpg.add_text("SOFA Export Convention")
                                    dpg.add_combo(CN.SOFA_OUTPUT_CONV, default_value=loaded_values["sofa_exp_convention"], width=190,
                                                  callback=cb.save_settings, tag='sofa_exp_convention')
                             
                                with dpg.table_row():
                                    dpg.add_text("Hp Correction")
                                    dpg.add_text("FIR Length (samples)")
                                    dpg.add_input_int(label="",width=190, tag='hpcf_fir_length', min_value=256, max_value=4096, default_value=loaded_values["hpcf_fir_length"],min_clamped=True, max_clamped=True, callback=cb.remove_hpcfs)
                                    with dpg.tooltip("hpcf_fir_length"):
                                        dpg.add_text("Defines the length of WAV headphone correction filters in samples (before resampling)")
                                        dpg.add_text("Increasing length improves frequency resolution and accuracy of the minimum-phase filter, especially at low frequencies, but also increases computation time")
            
                                with dpg.table_row():
                                    dpg.add_text("Hp Correction")
                                    dpg.add_text("Headphone Target Curve")
                                    dpg.add_combo(CN.HPCF_TARGET_LIST, default_value=loaded_values["hpcf_target_curve"], width=190, callback=cb.remove_hpcfs, tag='hpcf_target_curve')
                                    with dpg.tooltip("hpcf_target_curve"):
                                        dpg.add_text("The ASH Target will apply diffuse-field calibration however other common target curves can also be selected")
                                        dpg.add_text("Other curves may cause tonal incompatibilities with the room targets due to boosted bass. Select 'Flat' under room target to ensure compatibility")
                                        
                                with dpg.table_row():
                                    dpg.add_text("Hp Correction")
                                    dpg.add_text("Smooth High Frequencies")
                                    dpg.add_checkbox(label="Enable", default_value=loaded_values["hpcf_smooth_hf"],tag="hpcf_smooth_hf", callback=cb.remove_hpcfs)
                                    with dpg.tooltip("hpcf_smooth_hf"):
                                        dpg.add_text("Applies additional smoothing and tapers high frequencies when enabled")
                                        
                                
                                        
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
                                    dpg.add_button(label="Reset Settings", tag="reset_settings_tag", callback=lambda: dpg.configure_item("reset_settings_popup", show=True))
                                    with dpg.popup("reset_settings_tag", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="reset_settings_popup"):
                                        dpg.add_text("All settings will be reverted to default values")
                                        dpg.add_separator()
                                        with dpg.group(horizontal=True):
                                            dpg.add_button(label="OK", width=75, callback=cb.reset_settings)
                                            dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("reset_settings_popup", show=False))
                
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
                                        dpg.add_text("All exported Headphone Filters will be deleted")
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
                                        dpg.add_text("All exported Binaural Datasets will be deleted")
                                        dpg.add_separator()
                                        with dpg.group(horizontal=True):
                                            dpg.add_button(label="OK", width=75, callback=cb.remove_brirs)
                                            dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("del_brirs_popup", show=False))
                
                                with dpg.table_row():
                                    dpg.add_text("Outputs")
                                    dpg.add_text("Exported Files Folder")
                                    dpg.add_button(label="Open Folder", tag="open_output_folder_button", callback=cb.open_output_folder)
                                with dpg.table_row():
                                    dpg.add_text("Outputs")
                                    dpg.add_text("Equalizer APO WAVs Folder")
                                    dpg.add_button(label="Open Folder", tag="open_eapo_wav_folder_button", callback=cb.open_eapo_output_folder)
            
            
                        with dpg.collapsing_header(label="Customise Azimuth Selections", default_open=True):
                            dpg.add_text("Add or remove specific azimuth angles from channel configurations (requires restart)")
                            with dpg.group(horizontal=True):
                                with dpg.group():
                                    with dpg.group(horizontal=True):
                                        dpg.add_text("Azimuth (°): ")
                                        dpg.add_combo(CN.AZIMUTH_ANGLES_ALL, default_value=0, width=80, tag='custom_az_angle', callback=cb.update_chan_az_status_text)
                                    with dpg.group(horizontal=True):
                                        dpg.add_text("Channel:      ")
                                        dpg.add_combo(CN.CHANNELS_ALL, default_value=CN.CHANNELS_ALL[0], width=80, tag='custom_az_channel')
                                dpg.add_text("     ")
                                with dpg.group():
                                    dpg.add_button(label="Add mapping", callback=cb.add_mapping_callback)
                                    dpg.add_button(label="Remove mapping", callback=cb.remove_mapping_callback)
                            # Text showing current channels for the selected azimuth
                            dpg.add_text("", tag="custom_az_status")
                            
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
                        with dpg.group(horizontal=True):
                            dpg.add_text("Sort by: ")
                            dpg.add_combo(items=CN.AC_SPACE_LIST_SORT_BY,default_value=CN.AC_SPACE_LIST_SORT_BY[1],callback=cb.sort_reference_as_table)
                        data_length = len(CN.AC_SPACE_LIST_GUI)
    
                        with dpg.table(tag='reference_as_table',header_row=True,policy=dpg.mvTable_SizingFixedFit,resizable=False,borders_outerH=True,borders_innerH=True,no_host_extendX=True,borders_outerV=True,borders_innerV=True,delay_search=True):
                            dpg.add_table_column(label="ID")
                            dpg.add_table_column(label="Name")
                            dpg.add_table_column(label="RT60 (ms)")
                            dpg.add_table_column(label="Description", width_fixed=True, init_width_or_weight=535)
                            dpg.add_table_column(label="Source Dataset", width_fixed=True, init_width_or_weight=800)
                        
                            for i in range(data_length):
                                with dpg.table_row():
                                    dpg.add_text(CN.AC_SPACE_LIST_ID[i])
                                    dpg.add_text(CN.AC_SPACE_LIST_LABEL[i])
                                    dpg.add_text(CN.AC_SPACE_LIST_MEAS_R60[i])
                                    dpg.add_text(CN.AC_SPACE_LIST_DESCR[i], wrap=0)
                                    dpg.add_text(CN.AC_SPACE_LIST_DATASET[i], wrap=0)
                                    
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
                            dpg.add_table_column(label="Receiver Type")
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
                                    
                with dpg.collapsing_header(label="HRTF Datasets", default_open=False):  
                    #Section to show hrtf attributions
                    with dpg.child_window(auto_resize_x=True, auto_resize_y=True):
                        with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                            borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                            borders_outerV=True, borders_innerV=True, delay_search=True):
                            #dpg.bind_item_font(dpg.last_item(), font_b_def)
                            dpg.add_table_column(label="Type")
                            dpg.add_table_column(label="Dataset")
                            dpg.add_table_column(label="Attribution")
                            for i in range(len(CN.HRTF_UNIQ_TYPES)):
                                with dpg.table_row():
                                    dpg.add_text(CN.HRTF_UNIQ_TYPES[i])
                                    dpg.add_text(CN.HRTF_UNIQ_DATASETS[i],wrap =150)
                                    dpg.add_text(CN.HRTF_UNIQ_ATTR[i],wrap =1200)
                
                with dpg.collapsing_header(label="Data Summary", default_open=False):
                    with dpg.child_window(auto_resize_x=True, auto_resize_y=True):               
                        with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                            borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                            borders_outerV=True, borders_innerV=True, delay_search=True):
                            #dpg.bind_item_font(dpg.last_item(), font_b_def)
                            dpg.add_table_column(label="Type")
                            dpg.add_table_column(label="Stage")
                            dpg.add_table_column(label="File format")
                            dpg.add_table_column(label="Data Structure")
                            dpg.add_table_column(label="Coordinate System")
                            dpg.add_table_column(label="Notes")
                            for i in range(len(CN.DATA_SUMMARY_TYPE)):
                                with dpg.table_row():
                                    dpg.add_text(CN.DATA_SUMMARY_TYPE[i])
                                    dpg.add_text(CN.DATA_SUMMARY_STAGE[i])
                                    dpg.add_text(CN.DATA_SUMMARY_FORMAT[i],wrap =130)
                                    dpg.add_text(CN.DATA_SUMMARY_STRUCT[i],wrap =370)
                                    dpg.add_text(CN.DATA_SUMMARY_COORD[i],wrap =370)
                                    dpg.add_text(CN.DATA_SUMMARY_NOTES[i],wrap =370)
                
                with dpg.collapsing_header(label="Spatial Resolutions", default_open=False):
                    with dpg.child_window(auto_resize_x=True, auto_resize_y=True):
                        with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, resizable=False,
                            borders_outerH=True, borders_innerH=True, no_host_extendX=True,
                            borders_outerV=True, borders_innerV=True, delay_search=True):
                            #dpg.bind_item_font(dpg.last_item(), font_b_def)
                            dpg.add_table_column(label="Resolution")
                            dpg.add_table_column(label="Elevation Range")
                            dpg.add_table_column(label="Elevation Steps")
                            dpg.add_table_column(label="Elevation List (WAV)")
                            dpg.add_table_column(label="Azimuth Range")
                            dpg.add_table_column(label="Azimuth Steps")
                            dpg.add_table_column(label="Azimuth List (WAV)")
                            for i in range(len(CN.SPATIAL_RES_LIST)):
                                with dpg.table_row():
                                    dpg.add_text(CN.SPATIAL_RES_LIST[i])
                                    dpg.add_text(CN.SPATIAL_RES_ELEV_RNG[i])
                                    dpg.add_text(CN.SPATIAL_RES_ELEV_STP[i])
                                    dpg.add_text(CN.ELEV_ANGLES_WAV_GUI[i],wrap =450)
                                    dpg.add_text(CN.SPATIAL_RES_AZIM_RNG[i])
                                    dpg.add_text(CN.SPATIAL_RES_AZIM_STP[i])
                                    dpg.add_text(CN.AZ_ANGLES_ALL_WAV_GUI[i],wrap =450)
                 
                                            

                             
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
                                            dpg.add_text(CN.SOFA_COMPAT_CONV[i])
                                            dpg.add_text(CN.SOFA_COMPAT_VERS[i],wrap =100)
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
                                            dpg.add_text(CN.SOFA_OUTPUT_CONV[i])
                                            dpg.add_text(CN.SOFA_OUTPUT_VERS[i],wrap =100)
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
    
