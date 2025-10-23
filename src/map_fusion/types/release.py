"""Type definitions for map_fusion release artifacts.

This module provides pure type definitions for the release.tar artifacts.
Other projects can copy these types to interpret map_fusion releases in their
own language/framework without needing to depend on this codebase.

For loading functions, see map_fusion.pipeline.release.

Usage:
    from map_fusion.types.release import ReleaseArtifact
    from map_fusion.pipeline.release import load_release

    # Load release data
    release = load_release('/path/to/extracted/release-v0.1.0-alpha.1')

    # Access typed data
    print(f"Version: {release['metadata']['version']}")
    print(f"Build date: {release['metadata']['build_date']}")

    # Iterate over cities with GPS data
    for city, gps in release['gps'].items():
        print(f"{city}: {len(gps['files'])} GPS trace files")
"""

from typing import TypedDict, List, Dict, Optional, Literal
from pathlib import Path

from .common import Place


# =============================================================================
# Main Release Type
# =============================================================================

class ReleaseArtifact(TypedDict):
    """Complete representation of a release artifact.

    This is the main type for working with downloaded and extracted
    map_fusion releases. It provides structured access to all release
    data, metadata, and file paths.

    Example:
        >>> release = load_release('/path/to/release-v0.1.0-alpha.1')
        >>> print(release['metadata']['version'])
        '0.1.0-alpha.1'
        >>> for city, gps in release['gps'].items():
        ...     print(f"{city}: {len(gps['files'])} GPS files")
        berlin: 1234 GPS files
        chicago: 567 GPS files
    """
    metadata: "ReleaseMetadata"  # Build metadata from metadata.json
    version: "SemanticVersion"  # Parsed version information
    gps: Dict[Place, "GPSInputPaths"]  # GPS data per city
    readme_content: Optional[str]  # Contents of README.md (if present)


# =============================================================================
# Metadata Types
# =============================================================================

class ReleaseMetadata(TypedDict):
    """Metadata from metadata.json in release artifact.

    This metadata is automatically generated during the build process and
    provides information about the release build environment.
    """
    version: str  # Semantic version string (e.g., '0.1.0-alpha.1')
    build_date: str  # ISO 8601 timestamp (e.g., '2025-10-23T14:30:00Z')
    git_commit: str  # Git commit hash (short or full)
    git_branch: str  # Git branch name


class SemanticVersion(TypedDict):
    """Parsed semantic version number."""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str]  # e.g., 'alpha.1', 'beta.2', 'rc.1'
    prerelease_type: Optional[Literal["alpha", "beta", "rc"]]
    prerelease_number: Optional[int]


# =============================================================================
# GPS Data Types
# =============================================================================

class GPSInputPaths(TypedDict):
    """Paths to GPS input data for a specific city."""
    city: Place
    directory: Path  # Path to city GPS directory (e.g., input/gps/berlin)
    files: List[Path]  # List of GPS trace files (.txt)
