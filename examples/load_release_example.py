#!/usr/bin/env python3
"""
Example script demonstrating how to load and work with map_fusion release artifacts.

This script shows how to:
1. Download and extract a release
2. Load release data using typed interfaces
3. Access GPS trace files
4. Parse release metadata
5. Compute statistics

Usage:
    python examples/load_release_example.py /path/to/extracted/release-v0.1.0-alpha.1
"""

import sys
from pathlib import Path

# Import release types and functions
from map_fusion.types import ReleaseArtifact
from map_fusion.pipeline.load import (
    load_release,
    format_release_info,
    parse_version,
)


def main():
    if len(sys.argv) != 2:
        print("Usage: python load_release_example.py <release_directory>")
        print("\nExample:")
        print("  python load_release_example.py release-v0.1.0-alpha.1")
        sys.exit(1)

    release_path = Path(sys.argv[1])

    # Load the release
    print("Loading release artifact...")
    release = load_release(release_path)

    # Print formatted summary
    print("\n" + "=" * 60)
    print(format_release_info(release))
    print("=" * 60 + "\n")

    # Access version information
    version = release['version']
    print(f"Version details:")
    print(f"  Full version: {version['major']}.{version['minor']}.{version['patch']}")
    if version['prerelease']:
        print(f"  Pre-release: {version['prerelease']}")
        print(f"  Type: {version['prerelease_type']}")
        print(f"  Number: {version['prerelease_number']}")
    print()

    # Access metadata
    metadata = release['metadata']
    print(f"Build information:")
    print(f"  Date: {metadata['build_date']}")
    print(f"  Commit: {metadata['git_commit']}")
    print(f"  Branch: {metadata['git_branch']}")
    print()

    # Access GPS data
    gps_data = release['gps']
    print(f"GPS data available for {len(gps_data)} cities:")
    print()

    for city, gps_paths in gps_data.items():
        print(f"  {city.capitalize()}:")
        print(f"    Directory: {gps_paths['directory']}")
        print(f"    Number of trace files: {len(gps_paths['files'])}")

        # Show first few files
        print(f"    Sample files:")
        for file_path in gps_paths['files'][:3]:
            size_kb = file_path.stat().st_size / 1024
            print(f"      - {file_path.name} ({size_kb:.1f} KB)")

        if len(gps_paths['files']) > 3:
            print(f"      ... and {len(gps_paths['files']) - 3} more files")
        print()

    # Example: Read a GPS trace file
    print("=" * 60)
    print("Example: Reading GPS trace data")
    print("=" * 60)
    print()

    # Get first city and first file
    first_city = list(gps_data.keys())[0]
    first_file = gps_data[first_city]['files'][0]

    print(f"Reading: {first_file}")
    print()

    with open(first_file, 'r') as f:
        lines = f.readlines()[:5]  # Read first 5 lines

    print("First 5 GPS points:")
    print("Format: longitude latitude timestamp")
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            lon, lat, timestamp = parts
            print(f"  {lon:>12} {lat:>12} {timestamp:>15}")

    print()
    print(f"Total points in file: {len(open(first_file).readlines())}")

    # Example: Computing custom statistics
    print()
    print("=" * 60)
    print("Custom statistics: Average points per file")
    print("=" * 60)
    print()

    for city, gps_paths in gps_data.items():
        total_points = 0
        for file_path in gps_paths['files']:
            with open(file_path, 'r') as f:
                total_points += sum(1 for line in f if line.strip())

        avg_points = total_points / len(gps_paths['files'])
        print(f"  {city.capitalize()}:")
        print(f"    Total points: {total_points:,}")
        print(f"    Average per file: {avg_points:.1f}")
        print()


if __name__ == '__main__':
    main()
