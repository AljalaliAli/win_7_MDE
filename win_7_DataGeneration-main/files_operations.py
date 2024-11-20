import os 

def get_sorted_files(directory):
    """
    Returns a list of files in the directory sorted by modification time.

    Args:
        directory (str): The directory to search for files.

    Returns:
        list: A list of sorted file names.
    """
    return sorted(os.listdir(directory), key=lambda x: os.path.getmtime(os.path.join(directory, x)))



def is_image(file_name):
    """
    Checks if a file is an image.

    Args:
        file_name (str): Name of the file to check.

    Returns:
        bool: True if the file is an image, False otherwise.
    """
    return file_name.endswith(('.jpg', '.png', '.gif', '.tiff'))


def delete_empty_directories(directory):
    """
    Deletes empty directories within the given directory.

    Args:
        directory (str): Directory to search for empty directories.

    Returns:
        None
    """
    for root, dirs, _ in os.walk(directory, topdown=False):
        for dir in dirs:
            full_dir = os.path.join(root, dir)
            if not os.listdir(full_dir):
                print(f"Deleting empty directory: {full_dir}")
                os.rmdir(full_dir)





