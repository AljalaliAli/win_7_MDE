import os
import json
import shutil
import subprocess
from datetime import datetime
import logging
import configparser
#
def create_folder(folder_path):
    """
    Creates a folder if it doesn't exist.

    Parameters:
    - folder_path (str): Path of the folder to be created.

    Returns:
    - str: The path to the created or existing folder.
    """
    logger = logging.getLogger('create_folder')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logger.debug(f"Folder created: {folder_path}")
    else:
        logger.debug(f"Folder already exists: {folder_path}")
    return folder_path

def get_image_target_folder(timestamp, config_data):
    """
    Get the target folder path for saving an image based on the standard folder structure.

    Parameters:
    - timestamp (datetime): The timestamp used for folder determination.
    - config_data (dict): Configuration data containing 'standard_images_folder'.

    Returns:
    - str: The path to the target folder where the new image should be saved.
    """
    logger = logging.getLogger('get_image_target_folder')
    output_dir = config_data.get('standard_images_folder')
    
    # Extract year, month, and day from the datetime object
    year, month, day = timestamp.year, timestamp.month, timestamp.day

    # Create year folder if not exists
    year_path = os.path.join(output_dir, str(year))
    create_folder(year_path)

    # Create month folder inside the year folder if not exists
    month_path = os.path.join(year_path, str(month))
    create_folder(month_path)

    # Create day folder inside the month folder if not exists
    day_path = os.path.join(month_path, str(day))
    target_folder = create_folder(day_path)

    logger.debug(f"Target image folder: {target_folder}")
    return target_folder

def convert_to_datetime(timestamp):
    """
    Converts a timestamp string or datetime object to a datetime object by trying different formats.

    Parameters:
    - timestamp (str or datetime): A string representing a timestamp in various formats,
      or a datetime object to be returned as is.

    Returns:
    - datetime: A datetime object representing the parsed timestamp.

    Raises:
    - ValueError: If the timestamp is a string and does not match any expected format.
    """
    logger = logging.getLogger('convert_to_datetime')
    # Check if the input is already a datetime object
    if isinstance(timestamp, datetime):
        logger.debug("Timestamp is already a datetime object.")
        return timestamp

    # List of date-time formats to try for parsing the timestamp
    formats_to_try = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
        # Add more formats as needed
    ]

    # Try parsing the timestamp using each format
    for format_str in formats_to_try:
        try:
            parsed_timestamp = datetime.strptime(timestamp, format_str)
            logger.debug(f"Parsed timestamp '{timestamp}' using format '{format_str}'.")
            return parsed_timestamp
        except ValueError:
            logger.debug(f"Failed to parse timestamp '{timestamp}' using format '{format_str}'.")
            continue

    # If none of the formats match, raise a ValueError
    error_msg = "Timestamp does not match any expected format."
    logger.error(error_msg)
    raise ValueError(error_msg)

def create_standard_img_name(config, timestamp):
    '''
    Create a standard image name using provided data from an INI file and a timestamp.

    Parameters:
    - config (configparser.ConfigParser): Configuration parser object.
    - timestamp (datetime): The timestamp to be used in the image name.

    Returns:
    - str: The formatted image name with extension.
    '''
    logger = logging.getLogger('create_standard_img_name')
    try:
        formatted_customer_id = f'{int(config.get("SETTINGS", "customer_id")):0{config.getint("SETTINGS", "customer_id_width")}d}'
        # Corrected line:
        formatted_machine_id = f'{config.get("SETTINGS", "machine_id"):0>{config.getint("SETTINGS", "machine_id_width")}}'
        formatted_timestamp = timestamp.strftime(config.get("SETTINGS", "timestamp_format"))
        result_string = config.get("SETTINGS", "img_name_pattern").format(
            formatted_customer_id=formatted_customer_id,
            formatted_machine_id=formatted_machine_id,
            formatted_timestamp=formatted_timestamp
        )
        img_extension = config.get('SETTINGS', 'img_extension')
        img_name = f"{result_string}.{img_extension}"
        logger.debug(f"Generated image name: {img_name}")
        return img_name
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
        logger.error(f"Error creating standard image name: {e}")
        raise


'''def create_standard_img_name(config, timestamp):
    """
    Create a standard image name using provided data from an INI file and a timestamp.

    Parameters:
    - config (configparser.ConfigParser): Configuration parser object.
    - timestamp (datetime): The timestamp to be used in the image name.

    Returns:
    - str: The formatted image name with extension.
    """
    logger = logging.getLogger('create_standard_img_name')
    try:
        formatted_customer_id = f'{int(config.get("SETTINGS", "customer_id")):0{config.getint("SETTINGS", "customer_id_width")}d}'
        formatted_machine_id = f'{config.get("SETTINGS", "machine_id"):0{config.getint("SETTINGS", "machine_id_width")}}'
        formatted_timestamp = timestamp.strftime(config.get("SETTINGS", "timestamp_format"))
        result_string = config.get("SETTINGS", "img_name_pattern").format(
            formatted_customer_id=formatted_customer_id,
            formatted_machine_id=formatted_machine_id,
            formatted_timestamp=formatted_timestamp
        )
        img_extension = config.get('SETTINGS', 'img_extension')
        img_name = f"{result_string}.{img_extension}"
        logger.debug(f"Generated image name: {img_name}")
        return img_name
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
        logger.error(f"Error creating standard image name: {e}")
        raise'''

def is_device_reachable(ip_address, timeout=2):
    """
    Checks if a network device is reachable using the ping command.

    Parameters:
    - ip_address (str): The IP address of the target device.
    - timeout (int): The timeout for the ping command in seconds (default is 2 seconds).

    Returns:
    - bool: True if the device is reachable, False otherwise.
    """
    logger = logging.getLogger('is_device_reachable')
    try:
        # Use the ping command to check device reachability
        # For cross-platform compatibility, adjust the ping parameters
        count_param = '-n' if os.name == 'nt' else '-c'
        timeout_param = '-w' if os.name == 'nt' else '-W'
        timeout_value = str(timeout) if os.name == 'nt' else str(timeout)
        
        logger.debug(f"Pinging {ip_address} with timeout {timeout} seconds.")
        completed_process = subprocess.run(
            ['ping', count_param, '1', timeout_param, timeout_value, ip_address],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.debug(f"Ping to {ip_address} successful.")
        return True
    except subprocess.CalledProcessError:
        logger.debug(f"Ping to {ip_address} failed.")
        return False
    except Exception as e:
        logger.error(f"Error during ping: {e}")
        return False

 

# Define the logger at the module level
logger = logging.getLogger('create_and_append_to_txt_file')

def create_and_append_to_txt_file(file_path, text):
    """
    Create or append text to a text file.

    Parameters:
    - file_path (str): The path to the text file.
    - text (str): The text to be appended to the file.

    Returns:
    - None
    """
    try:
        with open(file_path, "a") as file:
            file.write(f"{text}\n")
        logger.debug(f"Appended text to {file_path}: {text}")
    except Exception as e:
        logger.error(f"Failed to append to {file_path}: {e}")

def take_screenshot_tncremo(tnc_remo_exe_path, machine_ip, img_path):
    """
    Takes a screenshot using TNCRemo on a specified machine.

    Parameters:
    - tnc_remo_exe_path (str): The file path to the TNCRemo executable.
    - machine_ip (str): The IP address of the target machine for taking the screenshot.
    - img_path (str): The file path where the screenshot will be saved.

    Returns:
    - bool: True if the screenshot was taken successfully, False otherwise.
    """
    logger = logging.getLogger('take_screenshot_tncremo')
    # TNCRemo screenshot command parameters
    cmd_args = [tnc_remo_exe_path, "SCREEN", img_path, "-I", machine_ip]

    try:
        logger.debug(f"Executing command: {' '.join(cmd_args)} with timeout 5 seconds.")
        # Run the command with a timeout of 5 seconds
        completed_process = subprocess.run(cmd_args, shell=True, timeout=5)

        # Check if the command was successful
        if completed_process.returncode == 0:
            logger.info("Screenshot taken successfully.")
            return True
        else:
            logger.error(f"Error taking screenshot. Return code: {completed_process.returncode}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Screenshot command timed out after 5 seconds.")
        return False
    except Exception as e:
        logger.error(f"An error occurred while taking screenshot: {e}")
        return False
 


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

 