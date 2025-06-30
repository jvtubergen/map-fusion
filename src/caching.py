import os

# This file provides functions to cache your computation results so they can be re-used.

# Cache folder is used for retrieving the API key from and to store image query responses to.
config_folder = os.path.expanduser("~/.config/geoalg")
cache_folder = os.path.expanduser("~/.cache/geoalg")
data_folder = "data/"


def get_cache_folder(folder_path):
    """Retrieve cache folder path; create it if it does not exist."""
    location = f"{cache_folder}/{folder_path}"
    if not os.path.exists(location):
        os.makedirs(location, exist_ok=True)
    return location

def get_data_folder(folder_path):
    """Retrieve data folder path; create it if it does not exist."""
    location = f"{data_folder}/{folder_path}"
    if not os.path.exists(location):
        os.makedirs(location, exist_ok=True)
    return location

def get_cache_file(file_path):
    """Retrieve cache file path; create parent directory if it does not exist."""
    location = f"{cache_folder}/{file_path}"
    parent_dir = os.path.dirname(location)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    return location

def get_data_file(file_path):
    """Retrieve data file path; create parent directory if it does not exist."""
    location = f"{data_folder}/{file_path}"
    parent_dir = os.path.dirname(location)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    return location

## Sat2Graph
# Locations of files and folders for a place.
def sat_locations(place):
    return  {
        # Pre-requisites.
        "image": get_data_file(f"sat/{place}.png"),
        "pbmodel": get_cache_file(f"sat/sat2graph/globalv2.pb"),
        # Intermediate data files.
        "partial_gtes": get_cache_file(f"sat/sat2graph/partial-gtes/{place}"),
        "intermediate": get_cache_file(f"sat/sat2graph"),
        # Resulting files.
        "result": get_data_file(f"graphs/sat-{place}.graph"),
        "image-results": get_data_file(f"images/sat/{place}")
    }
def partial_gte_location(place, counter):
    return get_cache_file(f"sat/sat2graph/partial-gtes/{place}/{counter}.pkl")
def image_result(place, phase):
    return get_data_file(f"images/sat/{place}/phase {phase}")
