#!/usr/bin/env python3
"""
Script to download and process GPS data from mapconstruction repository.

This script downloads GPS trace data from the mapconstruction GitHub repository,
extracts the data, converts UTM coordinates to WGS84, and saves the processed
traces for Berlin and Chicago.

Usage: python -m map_fusion.pipeline.input.gps <output_folder>
"""

import sys
import os
import tempfile
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import requests

# Import the existing UTM conversion logic from the map_fusion package
from map_fusion.spatial_reference_systems.utm import utm_to_latlon


TARBALL_URL = "https://github.com/pfoser/mapconstruction/tarball/master"

# City configuration
CITY_INFO: Dict[str, Dict[str, str]] = {
    "berlin": {
        "zip_path": "data/tracks/tracks_berlin_large.zip",
        "extracted_folder": "berlin_large",
        "trips_subfolder": "trips"
    },
    "chicago": {
        "zip_path": "data/tracks/tracks_chicago.zip",
        "extracted_folder": "chicago",
        "trips_subfolder": "trips"
    }
}


def download_tarball(url: str, dest_path: str) -> None:
    """Download tarball from URL to destination path."""
    print(f"Downloading GPS data from {url}...")

    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Download complete.")


def extract_tarball(tarball_path: str, extract_dir: str) -> str:
    """Extract tarball and return the path to the extracted directory."""
    print("Extracting tarball...")

    with tarfile.open(tarball_path, 'r:gz') as tar:
        tar.extractall(extract_dir)

    # Find the extracted directory (it will have a git hash in the name)
    extracted_dirs = [
        d for d in Path(extract_dir).iterdir()
        if d.is_dir() and d.name.startswith('pfoser-mapconstruction-')
    ]

    if not extracted_dirs:
        raise RuntimeError("Could not find extracted directory")

    return str(extracted_dirs[0])


def extract_gps_traces(extracted_dir: str, traces_dir: str) -> None:
    """Extract GPS trace ZIP files from the downloaded repository."""
    print("Extracting GPS traces...")

    for city, info in CITY_INFO.items():
        zip_path = Path(extracted_dir) / info["zip_path"]

        if zip_path.exists():
            print(f"Extracting {city} GPS traces...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(traces_dir)
        else:
            print(f"Warning: {city} GPS data not found at {zip_path}")


def convert_gps_file(input_file: Path, output_file: Path, city: str) -> None:
    """Convert a GPS trace file from UTM to WGS84."""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 3:
                continue

            x, y, t = float(parts[0]), float(parts[1]), float(parts[2])

            # Convert UTM to WGS84 using existing spatial reference system module
            # Note: utm_to_latlon expects coordinate as (y, x) tuple
            lat, lon = utm_to_latlon((y, x), city)

            # Write as: lon lat timestamp (standard GPS format)
            f_out.write(f"{lon} {lat} {t}\n")


def process_city(traces_base_dir: str, output_dir: str, city: str) -> None:
    """Process all GPS trace files for a city."""
    info = CITY_INFO[city]

    # Find the correct input directory
    input_dir = Path(traces_base_dir) / info["extracted_folder"] / info["trips_subfolder"]

    if not input_dir.exists():
        print(f"Warning: Input directory not found: {input_dir}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all .txt files in the input directory
    files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix == '.txt']

    print(f"Processing {len(files)} files for {city}...")

    for file in files:
        output_file = output_path / file.name
        convert_gps_file(file, output_file, city)

    print(f"Converted {len(files)} files for {city}")


def main(output_folder: str) -> None:
    """Main function to orchestrate the download and conversion process."""
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()

    try:
        # Download tarball
        tarball_path = os.path.join(temp_dir, "mapconstruction.tar.gz")
        download_tarball(TARBALL_URL, tarball_path)

        # Extract tarball
        extracted_dir = extract_tarball(tarball_path, temp_dir)

        # Create traces directory for extracted GPS data
        traces_dir = os.path.join(temp_dir, "traces")
        os.makedirs(traces_dir, exist_ok=True)

        # Extract GPS traces from ZIP files
        extract_gps_traces(extracted_dir, traces_dir)

        # Convert GPS data to WGS84 for each city
        print("Converting GPS traces to WGS84 format...")
        for city in CITY_INFO.keys():
            city_output_dir = os.path.join(output_folder, city)
            process_city(traces_dir, city_output_dir, city)

        print("\nDone! GPS data saved to:")
        for city in CITY_INFO.keys():
            print(f"  {city.capitalize()}: {output_folder}/{city}/")
        print("\nFormat: Each line contains 'lon lat timestamp' in WGS84 coordinates")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m map_fusion.pipeline.input.gps <output_folder>")
        sys.exit(1)

    output_folder = sys.argv[1]
    main(output_folder)
