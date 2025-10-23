"""Load and parse extracted release artifacts.

This module provides functions for loading release data from extracted
release directories.

Usage:
    from map_fusion.pipeline.load import load_release
    from map_fusion.types.release import ReleaseArtifact

    # Load release data
    release = load_release('/path/to/extracted/release-v0.1.0-alpha.1')

    # Access typed data
    print(f"Version: {release['metadata']['version']}")
    print(f"Build date: {release['metadata']['build_date']}")

    # Iterate over cities with GPS data
    for city, gps in release['gps'].items():
        print(f"{city}: {len(gps['files'])} GPS trace files")
"""

from pathlib import Path
from typing import Dict
import json

from ..types.release import (
    ReleaseArtifact,
    ReleaseMetadata,
    SemanticVersion,
    GPSInputPaths,
)
from ..types.common import Place


def parse_version(version_str: str) -> SemanticVersion:
    """Parse semantic version string into structured components.

    Args:
        version_str: Version string (e.g., '0.1.0-alpha.1')

    Returns:
        Parsed version with major, minor, patch, and prerelease info

    Example:
        >>> v = parse_version('0.1.0-alpha.1')
        >>> print(f"{v['major']}.{v['minor']}.{v['patch']}")
        0.1.0
        >>> print(v['prerelease_type'])
        alpha
    """
    # Remove 'v' prefix if present
    version_str = version_str.lstrip('v')

    # Split on '-' to separate main version from prerelease
    parts = version_str.split('-', 1)
    main_version = parts[0]
    prerelease = parts[1] if len(parts) > 1 else None

    # Parse main version (major.minor.patch)
    major, minor, patch = map(int, main_version.split('.'))

    # Parse prerelease (e.g., 'alpha.1' -> type='alpha', number=1)
    prerelease_type = None
    prerelease_number = None
    if prerelease:
        prerelease_parts = prerelease.split('.')
        prerelease_type = prerelease_parts[0] if len(prerelease_parts) > 0 else None
        if len(prerelease_parts) > 1:
            try:
                prerelease_number = int(prerelease_parts[1])
            except ValueError:
                pass

    return SemanticVersion(
        major=major,
        minor=minor,
        patch=patch,
        prerelease=prerelease,
        prerelease_type=prerelease_type,  # type: ignore
        prerelease_number=prerelease_number,
    )


def load_release_metadata(metadata_path: Path) -> ReleaseMetadata:
    """Load metadata.json from release artifact.

    Args:
        metadata_path: Path to metadata.json file

    Returns:
        Parsed metadata

    Raises:
        FileNotFoundError: If metadata file doesn't exist
        json.JSONDecodeError: If metadata is invalid JSON
    """
    with open(metadata_path, 'r') as f:
        data = json.load(f)

    return ReleaseMetadata(
        version=data['version'],
        build_date=data['build_date'],
        git_commit=data['git_commit'],
        git_branch=data['git_branch'],
    )


def scan_gps_directory(gps_base: Path) -> Dict[Place, GPSInputPaths]:
    """Scan GPS input directory and catalog all GPS trace files by city.

    Args:
        gps_base: Path to input/gps directory

    Returns:
        Dictionary mapping city names to GPS file paths

    Example:
        >>> gps_data = scan_gps_directory(Path('release/input/gps'))
        >>> print(list(gps_data.keys()))
        ['berlin', 'chicago']
    """
    gps_cities: Dict[Place, GPSInputPaths] = {}

    if not gps_base.exists():
        return gps_cities

    for city_dir in gps_base.iterdir():
        if city_dir.is_dir():
            city = city_dir.name
            gps_files = sorted(city_dir.glob('*.txt'))

            gps_cities[city] = GPSInputPaths(
                city=city,
                directory=city_dir,
                files=gps_files,
            )

    return gps_cities


def load_release(release_root: Path | str) -> ReleaseArtifact:
    """Load and parse a complete release artifact.

    This is the main entry point for working with downloaded releases.
    It scans the release directory, loads metadata, and creates a typed
    representation of all available data.

    Args:
        release_root: Path to extracted release directory
                      (e.g., '/path/to/release-v0.1.0-alpha.1')

    Returns:
        Complete release artifact with metadata and file paths

    Raises:
        FileNotFoundError: If release directory or required files don't exist

    Example:
        >>> release = load_release('release-v0.1.0-alpha.1')
        >>> print(f"Version: {release['metadata']['version']}")
        Version: 0.1.0-alpha.1
        >>> for city, gps in release['gps'].items():
        ...     print(f"{city}: {len(gps['files'])} trace files")
        berlin: 1234 trace files
        chicago: 567 trace files
    """
    root = Path(release_root)

    if not root.exists():
        raise FileNotFoundError(f"Release directory not found: {root}")

    # Load core files
    metadata_path = root / 'metadata.json'
    readme_path = root / 'README.md'
    version_path = root / 'VERSION'

    metadata = load_release_metadata(metadata_path)
    version = parse_version(metadata['version'])

    # Load README if it exists
    readme_content = None
    if readme_path.exists():
        with open(readme_path, 'r') as f:
            readme_content = f.read()

    # Scan GPS directory structure
    gps_base = root / 'input' / 'gps'
    gps_cities = scan_gps_directory(gps_base)

    return ReleaseArtifact(
        metadata=metadata,
        version=version,
        gps=gps_cities,
        readme_content=readme_content,
    )


def format_release_info(release: ReleaseArtifact) -> str:
    """Format release information as human-readable string.

    Args:
        release: Loaded release artifact

    Returns:
        Formatted summary string

    Example:
        >>> release = load_release('release-v0.1.0-alpha.1')
        >>> print(format_release_info(release))
        Release v0.1.0-alpha.1
        Built: 2025-10-23T14:30:00Z
        Commit: abc123def
        GPS data: 2 cities, 1801 files (123.4 MB)
    """
    meta = release['metadata']
    version = release['version']
    gps_data = release['gps']

    # Compute statistics from paths
    total_gps_files = sum(len(city['files']) for city in gps_data.values())
    total_gps_cities = len(gps_data)
    total_size = sum(
        f.stat().st_size
        for city in gps_data.values()
        for f in city['files']
    )
    size_mb = total_size / (1024 * 1024)

    prerelease = f"-{version['prerelease']}" if version['prerelease'] else ""
    version_str = f"{version['major']}.{version['minor']}.{version['patch']}{prerelease}"

    return f"""Release v{version_str}
Built: {meta['build_date']}
Commit: {meta['git_commit']}
GPS data: {total_gps_cities} cities, {total_gps_files} files ({size_mb:.1f} MB)"""
