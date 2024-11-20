import json
import os
import shutil

parent_dir = os.path.dirname(os.path.abspath(__file__))
mde_root = os.path.dirname(os.path.dirname(parent_dir))
templates_dir= os.path.join(mde_root, 'ConfigFiles','templates')
def get_last_id(images_dict):
    """
    Finds the last used image ID from the given dictionary.

    Args:
        images_dict (dict): A dictionary containing image IDs as keys.

    Returns:
        str: The last used image ID as a string, or "0" if the dictionary is empty.
    """
    try:
        # Check if the images_dict is not empty
        if images_dict:
            # Convert keys to integers, find the maximum, and return as a string
            return str(max(map(int, images_dict.keys())))
        else:
            return "0"  # No images found, return default ID "0"
    except Exception as e:
        print(f"Error: {e}")
        return None



def add_new_image(images_dict, size, img_path):
    """
    Adds a new image entry to the given dictionary.

    Args:
        images_dict (dict): A dictionary containing existing image entries.
        size (str): The size of the new image (e.g., "1024x768").<<<<-----------------------------------------------------
        img_path (str): The file path or URL of the new image.

    Returns:
        str: The ID of the newly added image.
    """
    try:
        last_id = get_last_id(images_dict)
        new_id = str(int(last_id) + 1)
        print(f"last_id: {last_id} ... new_id: {new_id}")
        new_img_path = copy_and_rename_file(img_path, templates_dir, new_id)
        # Create a new image entry
        images_dict[new_id] = {
            "size": size,
            "path": new_img_path,
            "features": {},  # Placeholder for image features
            "parameters": {}  # Placeholder for image parameters
        }

        print("New image added successfully.")
        return new_id
    except Exception as e:
        print(f"Error adding new image: {e}")
        return None



def add_new_feature(images_dict, temp_img_id, img_path, img_size, position, value='', sypol=''):
    """
    Adds a new feature (feature) entry to the specified image in the given dictionary.

    Args:
        images_dict (dict): A dictionary containing existing image entries.
        temp_img_id (str or int): The ID of the image to which the feature will be added.
        img_path (str): The file path or URL of the image.
        img_size (str): The size of the image (e.g., "1024x768").<<<<-----------------------------------------------------------
        position (str): The position of the feature within the image.
        value (str, optional): Additional value associated with the feature (default is an empty string).
        sypol (str, optional): Symbolic representation of the feature (default is an empty string).

    Returns:
        str: The ID of the newly added feature (feature).
    """
    try:
        # Check if the temp_img_id exists in images_dict
        if temp_img_id == -1:
            print('######### temp_img_id == -1 ###########')
           
            temp_img_id = add_new_image(images_dict, img_size, img_path)
            
        # Get the last feature_id
        last_feature_id = get_last_id(images_dict[temp_img_id]["features"])

        # Generate a new feature_id
        new_feature_id = str(int(last_feature_id) + 1)

        # Add the new feature to the dictionary
        images_dict[temp_img_id]["features"][new_feature_id] = {
            "position": position,
            "value": value,
            "sypol": sypol
        }

        print("New feature (feature) added successfully.")
        return temp_img_id, new_feature_id
    except Exception as e:
        print(f"Error adding new feature (feature): {e}")
        return None



def add_new_parameter(images_dict, temp_img_id, position, name):
    """
    Adds a new parameter entry to the specified image in the given dictionary.

    Args:
        images_dict (dict): A dictionary containing existing image entries.
        temp_img_id (str or int): The ID of the image to which the parameter will be added.
        position (str): The position of the parameter within the image.
        name (str): The name or description of the parameter.

    Returns:
        str: The ID of the newly added parameter.
    """
    try:
        last_parameter_id = get_last_id(images_dict[temp_img_id]["parameters"])
        new_parameter_id = str(int(last_parameter_id) + 1)

        # Add the new parameter to the dictionary
        images_dict[temp_img_id]["parameters"][new_parameter_id] = {
            "name": name,
            "position": position
        }

        print("New parameter added successfully.")
        return new_parameter_id
    except Exception as e:
        print(f"Error adding new parameter: {e}")
        return None



'''def get_parameters_and_features_by_id(json_data, temp_img_id):
    """
    Retrieves parameters and features associated with a specified image ID from the given JSON data.

    Args:
        json_data (dict): A dictionary containing image data.
        temp_img_id (str or int): The ID of the image for which parameters and features are requested.

    Returns:
        dict, dict: A tuple of dictionaries containing parameters and features, respectively.
    """
    try:
        parameters = {}
        features = {}

        # Check if the specified image ID exists in the JSON data
        if temp_img_id in json_data["images"]:
            image_data = json_data["images"][temp_img_id]

            # Check if "parameters" and "features" exist in the image data
            if "parameters" in image_data:
                parameters = image_data["parameters"]

            if "features" in image_data:
                features = image_data["features"]

        return parameters, features
    except Exception as e:
        print(f"Error retrieving parameters and features: {e}")
        return {}, {}'''
    
def extract_parameter_coordinates_from_image_template(config_data_dic, temp_img_id):
    """
    Extracts parameter coordinates from a specified image template in a JSON data structure.

    Args:
    - config_data_dic (dict): JSON data containing image templates and their parameters.
    - image_template_id (str): Identifier for the image template from which to extract parameters.

    Returns:
    - parameter_coordinates_list (list of dict): List of dictionaries containing parameter names and their positions.

    """
    param_name_position_list = []

    try:
        temp_img_dic = config_data_dic["images"][temp_img_id]
        
        parameters = temp_img_dic["parameters"]

        # Iterate over parameters in each image
        for param_id, param_info in parameters.items():
            param_dict = {
                "name": param_info["name"],
                "position": param_info["position"]
            }
            param_name_position_list.append(param_dict)

    except Exception as e:
        print(f"Error extracting parameters from JSON data: {e}")
        return None
    return param_name_position_list



def add_attributes_to_json(json_file_path, id, path, features, parameters):
    """
    Adds new attributes (such as features and parameters) to a JSON file containing image data.

    Args:
        json_file_path (str): The path to the JSON file.
        id (str): The ID of the image.
        path (str): The file path or URL of the image.
        features (dict): A dictionary containing image features .
        parameters (dict): A dictionary containing image parameters.

    Returns:
        None
    """
    try:
        # Check if the JSON file already exists
        if os.path.exists(json_file_path):
            # Load existing JSON content
            with open(json_file_path, 'r') as file:
                data = json.load(file)
        else:
            # If the file does not exist, initialize with an empty structure
            data = {"images": []}

        # Add the new attributes
        new_attributes = {
            "id": id,
            "path": path,
            "features": features,
            "parameters": parameters
        }

        # Append the new attributes to the "images" list
        data["images"].append(new_attributes)

        # Write the updated data back to the JSON file
        with open(json_file_path, 'w') as file:
            json.dump(data, file, indent=2)

        print("Attributes added to JSON file successfully.")
    except Exception as e:
        print(f"Error adding attributes to JSON file: {e}")

 


def copy_and_rename_file(file_path, dst_dir, new_filename):
    """
    Copies a file from the source directory to the destination directory with a new filename.

    Args:
        file_path (str): The path to the original file.
        dst_dir (str): The destination directory where the file will be copied.
        new_filename (str): The desired new filename (without extension).

    Returns:
        str: The new filename with its extension.
    """
    try:
        # Get the directory and filename from the file path
        src_dir, original_filename = os.path.split(file_path)

        # Get the file extension from the original filename
        extension = os.path.splitext(original_filename)[1]

        # Create the new filename with the same extension
        new_filename_with_extension = f"{new_filename}{extension}"

        # Check if the destination directory already has a file with the new filename
        if os.path.exists(os.path.join(dst_dir, new_filename_with_extension)):
            # If so, delete the old file
            os.remove(os.path.join(dst_dir, new_filename_with_extension))

        # Copy the file from the source directory to the destination directory with the new name and extension
        shutil.copy2(os.path.join(src_dir, original_filename), os.path.join(dst_dir, new_filename_with_extension))

        print("File copied and renamed successfully.")
        return new_filename_with_extension
    except Exception as e:
        print(f"Error copying and renaming file: {e}")
        return None
    

def check_and_update_json_config_file(file_path):
    """
    Check if a JSON configuration file exists and is empty.
    If the file exists but is empty, populate it with a predefined structure.
    If the file doesn't exist, create it with the predefined structure.
    
    Parameters:
    - file_path (str): The path to the JSON configuration file.
    
    Returns:
    None
    """
    try:
        # Get the size of the file
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            # If file size is zero, write the specified JSON structure into it
            with open(file_path, 'w') as file:
                json.dump({"images": {}}, file, indent=2)
            print("JSON file was empty. Updated successfully.")
        else:
            print("JSON file is not empty. No updates needed.")
    except FileNotFoundError:
        # If the file doesn't exist, create it with the specified structure
        with open(file_path, 'w') as file:
            json.dump({"images": {}}, file, indent=2)
        print("JSON file didn't exist. Created successfully with initial structure.")


if __name__ == "__main__":
   config_path=r"G:\Meine Ablage\MDE-2024\DataGeneration\DataGeneration\ConfigFiles\mde_config.json"
   with open(config_path, 'r') as f:
            config_data_dic = json.load(f)
   print(config_data_dic)
 