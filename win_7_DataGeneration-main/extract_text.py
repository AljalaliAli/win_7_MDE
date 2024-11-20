import cv2
import numpy as np
from Image_functions_v001 import *
from PIL import Image
import pytesseract
from pytesseract import Output

class OCRProcessor:
    """
    A class to handle OCR processing using Tesseract.
    """
    def __init__(self, tesseract_exe_path, lang='eng', tessedit_char_blacklist='', add_confidence_cols=False):
        """
        Initialize OCRProcessor with the given configuration.

        Args:
            tesseract_exe_path (str): Path to the Tesseract executable.
            lang (str): Language for Tesseract OCR. Defaults to 'eng'.
            tessedit_char_blacklist (str): Characters to blacklist during OCR. Defaults to an empty string.
            add_confidence_cols (bool): Whether to add confidence scores to the extracted parameters.
        """
        # Set the path to the Tesseract executable
        pytesseract.pytesseract.tesseract_cmd = tesseract_exe_path
        self.lang = lang
        self.tessedit_char_blacklist = tessedit_char_blacklist
        self.add_confidence_cols = add_confidence_cols

    def img_to_txt(self, image):
        """
        Extract text and confidence score from an image using Tesseract.

        Args:
            image (numpy.ndarray): The image to process.

        Returns:
            tuple: Extracted text and average confidence score.
        """
        img = prepare_img_for_ocr(image)
        if percentage_white_pixels(img) == 100:
            return '', 100

        extracted_text, average_confidence = self.extract_text_with_confidence(img)
        return extracted_text, average_confidence

    def extract_text_using_tessaract_5(self, img):
        """
        Extract text using Tesseract with specified language and configuration.

        Args:
            img (numpy.ndarray): The image to process.

        Returns:
            tuple: Extracted text and None (for compatibility).
        """
        custom_config = r'--oem 3 --psm 10'
        extracted_text = pytesseract.image_to_string(img, config=custom_config, lang=self.lang)
        extracted_text = extracted_text.replace('\n', '').replace('\x0c', '')
        return extracted_text, None

    def extract_text_with_confidence(self, img):
        """
        Extract text from an image with confidence scores using Tesseract.

        Args:
            img (numpy.ndarray): The image to process.

        Returns:
            tuple: Extracted text and average confidence score.
        """
        custom_config = f'--oem 3 --psm 10 -c tessedit_char_blacklist={self.tessedit_char_blacklist}'
        data = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT, lang=self.lang)

        extracted_text = ''
        total_confidence = 0
        count = 0

        for i, text in enumerate(data['text']):
            if text.strip():  # If the text is not empty
                extracted_text += text + ' '
                total_confidence += int(float(data['conf'][i]))  # Sum confidence as integer
                count += 1

        # Calculate average confidence
        average_confidence = total_confidence // count if count > 0 else 0

        # Remove unnecessary characters (like newline and form feed) and strip extra spaces
        extracted_text = extracted_text.replace('\n', '').replace('\x0c', '').strip()

        # Print the average confidence
        print("OCR Average Confidence Score:", average_confidence)

        return extracted_text, average_confidence

    def extract_parameter_values_from_image_positions(self, param_name_position_list, image):
        """
        Extract parameters and their values from an image based on provided positions using OCR.

        Args:
            param_name_position_list (list): A list of dictionaries containing parameter names and their position coordinates.
            image (numpy.ndarray): The input image.

        Returns:
            dict: A dictionary of extracted parameters and their values.
        """
        extracted_parameters = {}
        for param in param_name_position_list:
            param_name = param['name']
            cropped_image = crop_image(image, param['position']['x1'], param['position']['x2'], param['position']['y1'], param['position']['y2'])
            extracted_parameter, average_confidence = self.img_to_txt(cropped_image)
            save_image_cv(cropped_image, 'croped_images', f'{param_name}.tiff')
            print(f"{param_name}: {extracted_parameter}")
            extracted_parameters[param_name] = extracted_parameter

            if self.add_confidence_cols:
                extracted_parameters[f'{param_name}_Confidence'] = f'{average_confidence} %'
                
        return extracted_parameters
