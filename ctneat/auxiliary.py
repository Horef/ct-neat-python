"""
This module contains auxiliary functions and classes for the ctneat package.
"""
import os, sys

def clear_console():
    """Clears the console."""
    os.system('cls' if os.name == 'nt' else 'clear')

def ensure_dir_exists(dir_path: str):
    """
    Ensures that a directory exists; if it doesn't, creates it.

    Args:
        dir_path (str): The path of the directory to check or create.
    Args:
        dir_path (str): The path of the directory to check or create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def clear_output_dir(dir_path: str = "ctneat_outputs"):
    """
    Clears all files in the specified directory.
    By default the dir_path is set to the standard output directory "ctneat_outputs".

    Args:
        dir_path (str): The path of the directory to clear.
    """
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(dir_path)