import configparser
import os
import re
from datetime import datetime
import json
import timeit
import cv2
import time
from imports import *  # Import OCRProcessor
import threading
import sys
import logging

# Set up the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ConfigHandler:
    """
    A class to handle reading and accessing configuration values from an INI file.
    """
    def __init__(self, config_file):
        """
        Initialize ConfigHandler with the given configuration file.

        Args:
            config_file (str): Path to the configuration file.
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def get_config(self, section, key, is_int=False):
        """
        Retrieve a configuration value from a specified section and key.

        Args:
            section (str): The section in the INI file.
            key (str): The key within the section.
            is_int (bool): Whether to return the value as an integer. Defaults to False.

        Returns:
            str or int: The configuration value.
        """
        value = self.config.get(section, key)
        return int(value) if is_int else value

class Utility:
    """
    A utility class providing static methods for timestamp extraction.
    """
    @staticmethod
    def extract_timestamp_from_pattern(pattern, input_string):
        """
        Extract a timestamp from a string using a specified regex pattern.

        Args:
            pattern (str): The regex pattern to match the timestamp.
            input_string (str): The string to extract the timestamp from.

        Returns:
            str or None: The extracted timestamp in '%Y-%m-%d %H:%M:%S' format, or None if extraction fails.
        """
        try:
            match = re.search(pattern, input_string)
            if match:
                timestamp_str = match.group(1)
                extracted_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                timestamp = extracted_time.strftime('%Y-%m-%d %H:%M:%S')
                return timestamp
            else:
                print(f"Failed to extract timestamp. The input string '{input_string}' doesn't match the pattern '{pattern}'.")
                return None
        except ValueError:
            print(f"Failed to parse timestamp. The timestamp part '{timestamp_str}' doesn't match the format '%Y%m%d_%H%M%S'.")
            return None

    @staticmethod
    def extract_ts_from_img_name(filename):
        """
        Extract the timestamp from an image filename.

        Args:
            filename (str): The image filename.

        Returns:
            str or int: The extracted timestamp or -1 if extraction fails.
        """
        file_name_without_extension = os.path.splitext(filename)[0]
        img_name_pattern = r'(\d{8}_\d{6})'  # Define your pattern here
        return Utility.extract_timestamp_from_pattern(img_name_pattern, file_name_without_extension) or -1

class ExtractParametrs:
    """
    A class to handle parameter extraction from images and storage in a database.
    """
    def __init__(self, config, stop_event):
        """
        Initialize ExtractParametrs with the given configuration.

        Args:
            config (ConfigHandler): The configuration handler object.
            stop_event (threading.Event): Event to signal stopping the processing.
        """
        self.config = config
        self.db_dir = self.config.get_config('Paths', 'db_dir')
        self.db = DatabaseManager(db_dir=self.db_dir)
        self.matcher = ImageMatcher(self.config.get_config('Paths', 'configFiles_dir'),
                                    self.config.get_config('Paths', 'mde_config_file_name'),
                                    self.config.get_config('Paths', 'templates_dir_name'))
        self.config_data_dic = self._load_config_data()
        self.add_process_time_col = self.config.get_config('Parametrs', 'add_process_time_col', is_int=True)
        self.setup_directories()
        self.stop_event = stop_event
        
        # Initialize OCRProcessor with configuration parameters
        self.ocr_processor = OCRProcessor(
            tesseract_exe_path=self.config.get_config('Paths', 'tesseract_exe_path'),
            lang=self.config.get_config('Parametrs', 'lang'),
            tessedit_char_blacklist=self.config.get_config('Parametrs', 'tessedit_char_blacklist'),
            add_confidence_cols=self.add_process_time_col
        )

    def _load_config_data(self):
        """
        Load the configuration data from the specified JSON file.

        Returns:
            dict: The loaded configuration data.
        """
        config_path = os.path.join(self.config.get_config('Paths', 'configFiles_dir'),
                                   self.config.get_config('Paths', 'mde_config_file_name'))
        with open(config_path, 'r') as f:
            return json.load(f)

    def setup_directories(self):
        """
        Create necessary directories based on configuration settings.
        """
        # Create the directory for processed images if delete_after_extraction is set to 0
        if self.config.get_config('Parametrs', 'delete_after_extraction', is_int=True) == 0:
            os.makedirs(self.config.get_config('Paths', 'processed_images_dir'), exist_ok=True)
        
        # Create the directory for not recognized images if save_not_recognised_images is set to 1
        if self.config.get_config('Parametrs', 'save_not_recognised_images', is_int=True) == 1:
            os.makedirs(self.config.get_config('Paths', 'not_recognised_images_dir'), exist_ok=True)

    def extract_parameters_from_img(self, image):
        """
        Extract parameters from an input image and measure process time.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            tuple: A tuple containing a dictionary of extracted parameters and the process time (in seconds).
        """
        start_time = timeit.default_timer()
        try:
            if image is not None:
                # Match the image to a template and extract parameters if a match is found
                _, self.id_of_matched_template = self.matcher.match_images(image)
                if int(self.id_of_matched_template) > 0:
                    param_name_position_dic = extract_parameter_coordinates_from_image_template(self.config_data_dic, self.id_of_matched_template)
                    # Use OCRProcessor to extract parameters from image positions
                    extracted_parameters = self.ocr_processor.extract_parameter_values_from_image_positions(param_name_position_dic, image)
                    process_time = timeit.default_timer() - start_time
                    return extracted_parameters, process_time
        except Exception as e:
            print(f"An error occurred: {e}")
        # Return an empty dictionary and process time if extraction fails
        process_time = timeit.default_timer() - start_time
        return {}, process_time

    def generate_and_store_row_data_from_image(self, img_dir):
        """
        Process images in the specified directory, extract parameters, and store them in the database.

        Args:
            img_dir (str): The directory containing the images to process.
        """
        try:
            while not self.stop_event.is_set():
                images_processed = False
                # Traverse the image directory and process each image file
                for root, _, files in os.walk(img_dir):
                    for file in files:
                        if self.stop_event.is_set():
                            print("Stopping image processing...")
                            return
                        if is_image(file):
                            images_processed = True
                            self.process_image(root, file)
                # Wait for the next image if no images were processed and waiting is enabled
                if not images_processed and self.config.get_config('Parametrs', 'waiting_for_next_image', is_int=True) == 1:
                    print(f"No new images found. Waiting for {self.config.get_config('Parametrs', 'waiting_duration', is_int=True)} seconds...")
                    # Use stop_event.wait to check the stop event during the wait
                    self.stop_event.wait(self.config.get_config('Parametrs', 'waiting_duration', is_int=True))
                elif not images_processed:
                    print("No more images to process. Exiting.")
                    break
        except KeyboardInterrupt:
            print("Process interrupted. Exiting cleanly.")
            return

    def process_image(self, root, file):
        """
        Process a single image: extract parameters, store them in the database, and handle the image file.

        Args:
            root (str): The root directory of the image.
            file (str): The image filename.
        """
        img_path = os.path.join(root, file)
        db_name = extract_db_name(file)
        img = cv2.imread(img_path)
        timestamp = Utility.extract_ts_from_img_name(file)

        try:
            # Extract parameters from the image
            extracted_parameters, process_time = self.extract_parameters_from_img(img)
        except Exception as e:
            print(f"An error occurred while extracting parameters: {e}")
            extracted_parameters, process_time = {}, 0

        # Prepare data for database insertion
        parameters_to_insert_to_db = {'matched_template_id': self.id_of_matched_template}
        parameters_to_insert_to_db.update(extracted_parameters)
        if self.add_process_time_col == 1:
            parameters_to_insert_to_db['process_time'] = f'{round(process_time, 3)} Sec.'
        self.db.store_data(timestamp, parameters_to_insert_to_db, 'MDE', db_name)

        # Handle the image file based on configuration
        self.handle_image_file(img_path, file)

    def handle_image_file(self, img_path, file):
        """
        Handle the image file based on whether it was recognized, moved to a different directory, or deleted.

        Args:
            img_path (str): The path of the image file.
            file (str): The image filename.
        """
        try:
            # Move unrecognized images if the setting is enabled
            if self.id_of_matched_template == -1 and self.config.get_config('Parametrs', 'save_not_recognised_images', is_int=True) == 1:
                os.rename(img_path, os.path.join(self.config.get_config('Paths', 'not_recognised_images_dir'), file))
            # Delete images after extraction if the setting is enabled
            elif self.config.get_config('Parametrs', 'delete_after_extraction', is_int=True) == 1:
                os.remove(img_path)
            # Move processed images to the processed images directory
            else:
                os.rename(img_path, os.path.join(self.config.get_config('Paths', 'processed_images_dir'), file))
        except FileNotFoundError as e:
            # Log the error and continue
            print(f"File not found during handling: {img_path}. Error: {e}")
        except Exception as e:
            # Log other errors
            print(f"Error handling image file: {e}")


def process_machine(config_file_path, stop_event):
    """
    Process a single machine's image extraction using the provided configuration file.

    Args:
        config_file_path (str): The path to the machine's config.ini file.
        stop_event (threading.Event): Event to signal stopping the processing.
    """
    # Initialize configuration and start processing images
    config = ConfigHandler(config_file_path)
    extract_parameters_obj = ExtractParametrs(config, stop_event)
    extract_parameters_obj.generate_and_store_row_data_from_image(config.get_config('Paths', 'img_dir'))

def main(ini_files_dir):
    """
    Main function to process multiple machines in parallel.

    Args:
        ini_files_dir (str): The directory containing all config.ini files for the machines.
    """
    # Event to signal stopping threads
    stop_event = threading.Event()

    # List all config.ini files in the directory
    ini_files = [os.path.join(ini_files_dir, f) for f in os.listdir(ini_files_dir) if f.endswith('.ini')]

    # List to hold all threads
    threads = []

    # Create and start a thread for each config file
    for config_file in ini_files:
        thread = threading.Thread(target=process_machine, args=(config_file, stop_event))
        thread.daemon = True  # Set thread as daemon so they terminate when main program exits
        thread.start()
        threads.append(thread)

    try:
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("Interrupted by user. Stopping threads...")
        stop_event.set()
        for thread in threads:
            thread.join()
        print("All threads have been stopped.")

if __name__ == "__main__":
    # Directory containing all config.ini files
    ini_files_dir = 'ini_files'
    
    # Run the main function to start processing all machines in parallel
    main(ini_files_dir)
