import os

# This file provides functions to cache your computation results so they can be re-used.

# Cache folder is used for retrieving the API key from and to store image query responses to.
config_folder = os.path.expanduser("~/.config/geoalg")
cache_folder = os.path.expanduser("~/.cache/geoalg")
data_folder = "data/"

# Check if cache folder exists, create if it doesn't
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder, exist_ok=True)

if not os.path.exists(config_folder):
    os.makedirs(config_folder, exist_ok=True)

if not os.path.exists(data_folder):
    os.makedirs(data_folder, exist_ok=True)

def get_cache_folder(folder_path):
    location = f"{cache_folder}/{folder_path}"
    if not os.path.exists(location):
        os.makedirs(location, exist_ok=True)
    return location

def get_cache_file(file_path):
    location = f"{cache_folder}/{file_path}"
    return location

def get_data_file(file_path):
    location = f"{data_folder}/{file_path}"
    return location
