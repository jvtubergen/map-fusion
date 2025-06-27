# This file provides functions to cache your computation results so they can be re-used.

# Cache folder is used for retrieving the API key from and to store image query responses to.
cache_folder = os.path.expanduser("~/.cache/geoalg")

# Check if cache folder exists, create if it doesn't
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder, exist_ok=True)


