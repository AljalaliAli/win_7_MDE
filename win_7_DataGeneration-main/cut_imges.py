


from imports import *

def cut_images(image,ts, output_folder, param_name_position_list):
    """
    Cut images based on the specified positions and save them to the output folder.

    Args:
    input_image_path (str): Path to the input image file.
    output_folder (str): Path to the folder where the cut images will be saved.
    param_name_position_list (list): List of dictionaries containing parameter names and positions.

    Returns:
    None
    """
     
    if image is None:
        print(f"Error: Unable to read   image  ")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through the parameter name and position list
    for item in param_name_position_list:
        name = item['name']
        position = item['position']
        x1, x2, y1, y2 = int(position['x1']), int(position['x2']), int(position['y1']), int(position['y2'])

        # Cut the image based on the specified positions
        cut_image = image[y1:y2, x1:x2]
       
        img_name=f'{name}.tiff'
        
        save_image_cv(cut_image, output_folder,  img_name)
         
       
        

class ExtractParametrs:
    def __init__(self, configFiles_dir, mde_config_file_name, templates_dir_name ):
        """
        Initializes the ExtractParametrs class by loading configuration settings.
        """
        
        self.db = DatabaseManager()
        config_path= os.path.join(configFiles_dir, mde_config_file_name)
        self.matcher = ImageMatcher(configFiles_dir, mde_config_file_name, templates_dir_name)
        with open(config_path, 'r') as f:
            self.config_data_dic = json.load(f)
            
     
    def cut_and_save(self,ts,  image):
        """
        Extract parameters from an input image and measure process time.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            tuple: A tuple containing a dictionary of extracted parameters and the process time (in seconds).
        """
  
        try:
            # Check if the image is not None
            if image is not None: 
                # Match the image to find the template ID
                _, self.id_of_matched_template = self.matcher.match_images(image)
            
                # Get parameter names and positions based on the matched template ID
                param_name_position_dic= extract_parameter_coordinates_from_image_template(self.config_data_dic, self.id_of_matched_template)
                bw_img, _=convert_to_bw(image)
                cut_images(bw_img,ts, r"D:\future link\AljalaliAli\DataGeneration\DataGeneration\simples", param_name_position_dic)
                
            
            return 1

        except Exception as e:
            # Handle exceptions gracefully (e.g., logging, error reporting)
            print(f"An error occurred: {e}")
            return -1 # Return empty dictionary and 0.0 process time in case of error



    def generate_and_store_row_data_from_image(self, img_dir):
        """
        Generate raw data from images and save them in the database under the timestamp of the image.
            Parameters:
            img_dir (str): The directory containing the images.
        """
        # Walk through all subdirectories
        # sorted_img_files = get_sorted_files(img_to_extract_dir)
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if is_image(file):

                    img_path = os.path.join(root, file)
               
                    img = cv2.imread(img_path)
                    ts = extract_ts_from_img_name(file)
                    self.cut_and_save(ts, img)
                 

   




 
   
import os
from PIL import Image

def change_dpi_in_folder(folder_path, dpi):
       for filename in os.listdir(folder_path):
        # Check if the file is an image
        if filename.endswith('.jpg') or filename.endswith('.png')  or filename.endswith('.tiff'):
            # Create the full file path by joining the folder path and the filename
            file_path = os.path.join(folder_path, filename)

            # Open the image using PIL
            image = Image.open(file_path)

            # Change the DPI
            image.save(file_path, dpi=(dpi, dpi))
 
  
from PIL import Image

def change_dpi(input_image_path, output_image_path, dpi):
    # Open the image using PIL
    image = Image.open(input_image_path)

    # Change the DPI
    image.save(output_image_path, dpi=(dpi, dpi))




import cv2
import numpy as np

def crop_characters(image_path, output_folder):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crop and save each character
    for i, contour in enumerate(contours):
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Crop the character using the bounding box
        cropped_char = image[y:y+h, x:x+w]
        _, bw_cropped_char=convert_to_bw(cropped_char, 0.20)
        img = add_border(bw_cropped_char,1) 
        text = img_to_txt(img)
        #print(text)
        # Save the cropped character
      #  cv2.imwrite(f"{output_folder}/char_{i}.tiff", cropped_char)
        save_image_cv(img, output_folder,  f"{i}.tiff")



import os
from PIL import Image

def merge_images(image_directory, out_dir,  images_per_row):
    # Get a list of all image files in the directory
    image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith('.png') or f.endswith('.tiff')]

    images = [Image.open(x) for x in image_files]

    # Assuming all images are the same size, get dimensions of first image
    w, h = images[0].size

    # Calculate the total width and height of the final image
    total_width = w * images_per_row
    max_height = h * ((len(images) // images_per_row) + (1 if len(images) % images_per_row > 0 else 0))

    # Create a blank canvas with a white background
    merged_image = Image.new('RGB', (total_width, max_height), 'white')

    # Loop over images and paste them onto the canvas
    for i, img in enumerate(images):
        x_offset = (i % images_per_row) * w
        y_offset = (i // images_per_row) * h
        merged_image.paste(img, (x_offset, y_offset))
        if i%200==0:
             
            merged_image.save(f'{out_dir}/{i}.tiff')
    return 1




if __name__ == "__main__":
    # Usage
    change_dpi_in_folder(r"D:\future link\AljalaliAli\DataGeneration\DataGeneration\simples", 300)
    # Example usage
   #input_image_path = r"C:\Users\nizar\OneDrive\Desktop\vv\ID0001_MID0004_20240228_131422.tiff"
   # output_folder_path = "cropped_characters"
   #crop_characters(input_image_path, output_folder_path)
   # Example usage
  
       # Specify the directory where your images are
    #directory = 'cropped_characters'
 

    # Call the function
    #merged_image = merge_images(directory, output_folder_path,15) 
#img_dir = r'C:\Users\nizar\OneDrive\Desktop\tst'
 #    extract_parameters_obj = ExtractParametrs(configFiles_dir, mde_config_file_name, templates_dir_name)
  #   extract_parameters_obj.generate_and_store_row_data_from_image(img_dir)
    
     
