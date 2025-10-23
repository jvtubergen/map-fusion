"""
Pipeline module for constructing release data artifacts.

This module contains the core logic for programmatically building all data
included in the release.tar distribution:
  - Input: Process raw GPS traces and satellite imagery
  - Unimodal: Generate GPS-based and satellite-based map reconstructions
  - Fused: Merge unimodal maps using fusion algorithms
  - Results: Compute metrics and generate visualizations

Submodules:
  - load: Functions for loading and parsing release artifacts
  - tar: Functions for building and extracting release tar archives
"""

# Release loading functions
from .load import (
    load_release,
    load_release_metadata,
    parse_version,
    scan_gps_directory,
    format_release_info,
)

# Tar archive operations
from .tar import (
    build_release_tar,
    extract_release_tar,
    list_release_contents,
)

__all__ = [
    # Loading
    "load_release",
    "load_release_metadata",
    "parse_version",
    "scan_gps_directory",
    "format_release_info",
    # Tar operations
    "build_release_tar",
    "extract_release_tar",
    "list_release_contents",
]
