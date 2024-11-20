import configparser
import logging
import os
from datetime import datetime
from time import sleep
from threading import Thread
from functions import *
import glob
import sys

def get_base_path():
    """
    Get the base path where the executable or script is located.
    
    Returns:
        str: Path to the base directory.
    """
    if getattr(sys, 'frozen', False):
        # If the application is frozen by PyInstaller
        base_path = os.path.dirname(sys.executable)
    else:
        # If the application is running as a script
        base_path = os.path.dirname(os.path.abspath(__file__))
    print(f"[DEBUG] Base path determined as: {base_path}")
    return base_path

# Define BASE_PATH globally
BASE_PATH = get_base_path()

def load_config(config_file):
    """
    Load configuration from an INI file.

    Parameters:
    - config_file (str): Path to the configuration INI file.

    Returns:
    - config (configparser.RawConfigParser): Config parser object containing configuration settings.
    """
    print(f"[DEBUG] Loading configuration from {config_file}")
    config = configparser.RawConfigParser()
    config.read(config_file)
    if not config.sections():
        print(f"[ERROR] Failed to read any sections from {config_file}")
    else:
        print(f"[DEBUG] Successfully loaded configuration from {config_file}")
    return config

def configure_logging(enabled, machine_id):
    """
    Configure logging based on the enabled flag and machine identifier.

    Parameters:
    - enabled (bool): If True, logging is enabled; otherwise, logging is disabled.
    - machine_id (str): Identifier for the machine, used to differentiate log files.
    """
    if enabled:
        log_filename = os.path.join(BASE_PATH, f'TNCRemoCaptureTool_{machine_id}.log')
        logging.basicConfig(
            filename=log_filename,                # Log file path within BASE_PATH
            level=logging.DEBUG,                  # Log level
            format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
            datefmt='%Y-%m-%d %H:%M:%S'           # Date format in log messages
        )
        print(f"[DEBUG] Logging is enabled. Log file: {log_filename}")
    else:
        logging.disable(logging.CRITICAL)  # Disable all logging
        print("[DEBUG] Logging is disabled.")

def load_settings(config):
    """
    Extract settings from the configuration object.

    Parameters:
    - config (configparser.RawConfigParser): Config parser object containing configuration settings.

    Returns:
    - settings (dict): Dictionary containing all required settings.
    """
    print("[DEBUG] Loading settings from configuration.")
    try:
        settings = {
            'machine_ip': config.get('SETTINGS', 'machine_ip'),
            'machine_off_sleep_n_sec': config.getint('SETTINGS', 'machine_off_sleep_n_sec'),
            'capture_interval': config.getint('SETTINGS', 'capture_interval'),
            'tnc_remo_exe_path': config.get('SETTINGS', 'tnc_remo_exe_path'),
            'standard_images_folder': config.get('SETTINGS', 'standard_images_folder'),
            'standard_folder_structure': config.getboolean('SETTINGS', 'standard_folder_structure'),
            'machine_id': config.get('SETTINGS', 'machine_id'),
            'machine_name': config.get('SETTINGS', 'machine_name'),
            'log_enabled': config.getboolean('SETTINGS', 'log_enabled'),
            'notes_enabled': config.getboolean('SETTINGS', 'notes_enabled')
        }
        print(f"[DEBUG] Settings loaded: {settings}")
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        print(f"[ERROR] Error loading settings: {e}")
        settings = {}
    return settings

def initialize_variables(initial_reachable):
    """
    Initialize variables used in the main loop.

    Parameters:
    - initial_reachable (bool): Initial reachability status of the machine.

    Returns:
    - variables (dict): Dictionary containing initialized variables.
    """
    print("[DEBUG] Initializing variables.")
    variables = {
        'failure_screenshot_counter': 0,
        'last_failure_screenshot_ts': None,
        'last_successful_screenshot_ts': None,
        'note_saved_last_successful_screenshot_ts': False,
        'machine_prev_reachable': initial_reachable  # Track previous reachability
    }
    print(f"[DEBUG] Initialized variables: {variables}")
    return variables

def handle_machine_unreachable(settings):
    """
    Handle the scenario when the machine is not reachable.

    Parameters:
    - settings (dict): Dictionary containing configuration settings.

    Returns:
    - machine_reachable (bool): Updated machine reachability status.
    """
    if settings['log_enabled']:
        logging.warning(
            f"Machine {settings['machine_id']} is not reachable, maybe the machine is off. "
            f"Sleeping for {settings['machine_off_sleep_n_sec']} seconds then retrying."
        )
    print(f"[WARNING] Machine {settings['machine_id']} is not reachable. Sleeping for {settings['machine_off_sleep_n_sec']} seconds.")
    sleep(settings['machine_off_sleep_n_sec'])
    machine_reachable = is_device_reachable(settings['machine_ip'])
    print(f"[DEBUG] Rechecking machine reachability: {machine_reachable}")
    return machine_reachable

def handle_screenshot_success(settings, variables, config, timestamp):
    """
    Handle the successful screenshot capture.

    Parameters:
    - settings (dict): Dictionary containing configuration settings.
    - variables (dict): Dictionary containing initialized variables.
    - config (configparser.RawConfigParser): Config parser object containing configuration settings.
    - timestamp (datetime): Timestamp of the screenshot.

    Returns:
    - variables (dict): Updated variables after handling success.
    """
    if settings['log_enabled']:
        logging.info('Screenshot taken successfully.')
    print(f"[INFO] Screenshot taken successfully at {timestamp}.")

    variables['last_successful_screenshot_ts'] = timestamp
    variables['note_saved_last_successful_screenshot_ts'] = False

    if variables['last_failure_screenshot_ts'] is not None:
        if settings['log_enabled']:
            logging.info(
                f"Images are taken again; the machine was reachable, but no images were taken due to an unknown error. "
                f"Last failure timestamp: {variables['last_failure_screenshot_ts']}"
            )
        print(f"[INFO] Images are taken again after failure at {variables['last_failure_screenshot_ts']}.")
        if settings['notes_enabled']:
            create_and_append_to_txt_file(
                os.path.join(BASE_PATH, "Notes.txt"),
                f"Images are taken again; the machine was reachable, but no images were taken due to an unknown error. "
                f"Last failure timestamp: {variables['last_failure_screenshot_ts']}"
            )
            print("[INFO] Note appended to Notes.txt.")
        variables['last_failure_screenshot_ts'] = None
        variables['failure_screenshot_counter'] = 0

    print(f"[DEBUG] Sleeping for {settings['capture_interval']} seconds before next capture.")
    sleep(settings['capture_interval'])
    return variables

def handle_screenshot_failure(settings, variables):
    """
    Handle the failed screenshot capture.

    Parameters:
    - settings (dict): Dictionary containing configuration settings.
    - variables (dict): Dictionary containing initialized variables.

    Returns:
    - variables (dict): Updated variables after handling failure.
    - machine_reachable (bool): Updated machine reachability status.
    """
    if settings['log_enabled']:
        logging.error('Failed to take an image!')
    print("[ERROR] Failed to take a screenshot.")

    variables['last_failure_screenshot_ts'] = datetime.now()
    variables['failure_screenshot_counter'] += 1
    print(f"[DEBUG] Failure counter incremented to {variables['failure_screenshot_counter']}.")

    if variables['failure_screenshot_counter'] > 5:
        if settings['log_enabled']:
            logging.error(f"{variables['failure_screenshot_counter']} consecutive failures to take a screenshot.")
        print(f"[ERROR] {variables['failure_screenshot_counter']} consecutive failures to take a screenshot.")
        if settings['notes_enabled']:
            create_and_append_to_txt_file(
                os.path.join(BASE_PATH, "Notes.txt"),
                f"The machine is reachable, but no images were taken due to an unknown error. "
                f"Last successful screenshot timestamp: {variables['last_successful_screenshot_ts']}"
            )
            print("[INFO] Note appended to Notes.txt due to multiple failures.")
        if variables['failure_screenshot_counter'] > 100 and not variables['note_saved_last_successful_screenshot_ts']:
            if settings['log_enabled']:
                logging.warning("More than 100 failures occurred without a successful screenshot.")
            print("[WARNING] More than 100 failures occurred without a successful screenshot.")
            variables['note_saved_last_successful_screenshot_ts'] = True

    machine_reachable = is_device_reachable(settings['machine_ip'])
    print(f"[DEBUG] Machine reachability after failure: {machine_reachable}")
    return variables, machine_reachable

def process_screenshot(settings, variables, config):
    """
    Process the screenshot capture attempt.

    Parameters:
    - settings (dict): Dictionary containing configuration settings.
    - variables (dict): Dictionary containing initialized variables.
    - config (configparser.RawConfigParser): Config parser object containing configuration settings.

    Returns:
    - variables (dict): Updated variables after processing.
    - machine_reachable (bool): Updated machine reachability status.
    """
    timestamp = datetime.now()
    print(f"[DEBUG] Attempting to take screenshot at {timestamp}.")
    img_name = create_standard_img_name(config, timestamp)
    print(f"[DEBUG] Generated image name: {img_name}")

    if settings['standard_folder_structure']:
        img_dir = get_image_target_folder(timestamp, config)
        print(f"[DEBUG] Using standard folder structure. Image directory: {img_dir}")
    else:
        img_dir = create_folder(os.path.join(settings['standard_images_folder'], f"{settings['machine_name']}_{settings['machine_id']}"))
        print(f"[DEBUG] Using custom folder structure. Image directory: {img_dir}")

    img_path = os.path.join(img_dir, img_name)
    print(f"[DEBUG] Full image path: {img_path}")

    success = take_screenshot_tncremo(settings['tnc_remo_exe_path'], settings['machine_ip'], img_path)
    print(f"[DEBUG] Screenshot taken: {'Success' if success else 'Failure'}")

    if success:
        variables = handle_screenshot_success(settings, variables, config, timestamp)
        return variables, True  # If success, machine remains reachable
    else:
        variables, machine_reachable = handle_screenshot_failure(settings, variables)
        return variables, machine_reachable

def run_machine_capture(config_file):
    """
    Run the screenshot capture loop for a single machine based on its config file.

    Parameters:
    - config_file (str): Path to the machine's configuration INI file.
    """
    print(f"[INFO] Starting capture thread for config file: {config_file}")
    # Load configuration data from the config.ini file
    config = load_config(config_file)

    if not config.sections():
        print(f"[ERROR] Configuration file {config_file} is empty or invalid.")
        return

    # Extract settings from the config
    settings = load_settings(config)

    if not settings:
        print(f"[ERROR] Failed to load settings from {config_file}.")
        return

    # Configure logging based on the config file setting
    configure_logging(settings['log_enabled'], settings['machine_id'])

    # Initial check for machine reachability
    initial_reachable = is_device_reachable(settings['machine_ip'])
    print(f"[DEBUG] Initial machine reachability: {initial_reachable}")

    # Initialize additional variables
    variables = initialize_variables(initial_reachable)

    # Log initial state
    current_time = datetime.now()
    if not initial_reachable:
        if settings['log_enabled']:
            logging.warning(f"Machine {settings['machine_id']} is initially not reachable as of {current_time}.")
        print(f"[WARNING] Machine {settings['machine_id']} is initially not reachable as of {current_time}.")
        if settings['notes_enabled']:
            create_and_append_to_txt_file(
                os.path.join(BASE_PATH, "Notes.txt"),
                f"Machine {settings['machine_id']} is not reachable as of {current_time}."
            )
            print("[INFO] Note appended to Notes.txt for initial unreachable state.")
    else:
        if settings['log_enabled']:
            logging.info(f"Machine {settings['machine_id']} is initially reachable as of {current_time}.")
        print(f"[INFO] Machine {settings['machine_id']} is initially reachable as of {current_time}.")

    # The main loop
    while True:
        if not variables['machine_prev_reachable']:
            # Machine was previously unreachable
            machine_reachable = handle_machine_unreachable(settings)
            # Check for state change
            if machine_reachable != variables['machine_prev_reachable']:
                current_time = datetime.now()
                if not machine_reachable:
                    if settings['log_enabled']:
                        logging.warning(f"Machine {settings['machine_id']} remains unreachable as of {current_time}.")
                    print(f"[WARNING] Machine {settings['machine_id']} remains unreachable as of {current_time}.")
                else:
                    if settings['log_enabled']:
                        logging.info(f"Machine {settings['machine_id']} is reachable again as of {current_time}.")
                    print(f"[INFO] Machine {settings['machine_id']} is reachable again as of {current_time}.")
                    if settings['notes_enabled']:
                        create_and_append_to_txt_file(
                            os.path.join(BASE_PATH, "Notes.txt"),
                            f"Machine {settings['machine_id']} is reachable again as of {current_time}."
                        )
                        print("[INFO] Note appended to Notes.txt for machine becoming reachable again.")
            variables['machine_prev_reachable'] = machine_reachable
        else:
            if settings['log_enabled']:
                logging.info('Machine is reachable.')
            print("[INFO] Machine is reachable.")
            while variables['machine_prev_reachable']:
                variables, machine_reachable = process_screenshot(settings, variables, config)
                # Check for state change
                if machine_reachable != variables['machine_prev_reachable']:
                    current_time = datetime.now()
                    if not machine_reachable:
                        if settings['log_enabled']:
                            logging.warning(f"Machine {settings['machine_id']} became unreachable as of {current_time}.")
                        print(f"[WARNING] Machine {settings['machine_id']} became unreachable as of {current_time}.")
                        if settings['notes_enabled']:
                            create_and_append_to_txt_file(
                                os.path.join(BASE_PATH, "Notes.txt"),
                                f"Machine {settings['machine_id']} became unreachable as of {current_time}."
                            )
                            print("[INFO] Note appended to Notes.txt for machine becoming unreachable.")
                    else:
                        if settings['log_enabled']:
                            logging.info(f"Machine {settings['machine_id']} remains reachable as of {current_time}.")
                        print(f"[INFO] Machine {settings['machine_id']} remains reachable as of {current_time}.")
                    variables['machine_prev_reachable'] = machine_reachable

def main():
    """
    Main function to initiate threads for all machine configurations found in ini_files directory.
    """
    base_path = get_base_path()
    ini_files_path = os.path.join(base_path, 'ini_files', '*.ini')

    print(f"[DEBUG] Searching for configuration files in: {ini_files_path}")
    config_files = glob.glob(ini_files_path)

    if not config_files:
        print("No configuration files found in the 'ini_files' directory.")
        return

    print(f"[INFO] Found {len(config_files)} configuration file(s).")
    threads = []

    for config_file in config_files:
        thread = Thread(target=run_machine_capture, args=(config_file,), daemon=True)
        thread.start()
        threads.append(thread)
        print(f"Started thread for config: {config_file}")

    # Keep the main thread alive to allow daemon threads to run
    try:
        print("[INFO] All threads started. Entering main loop. Press Ctrl+C to exit.")
        while True:
            sleep(1)
    except KeyboardInterrupt:
        print("Shutting down all threads.")


if __name__ == "__main__":
    main()
