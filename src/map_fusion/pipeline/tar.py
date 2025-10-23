"""Tar archive operations for release artifacts.

This module handles creating, extracting, and inspecting release.tar files.

Usage:
    from map_fusion.pipeline.tar import build_release_tar, extract_release_tar

    # Build a release tar from GPS data
    build_release_tar(
        version='0.1.0-alpha.1',
        gps_data_paths={'berlin': '/data/gps/berlin', 'chicago': '/data/gps/chicago'},
        output_path='release-v0.1.0-alpha.1.tar.gz',
        git_commit='abc123',
        git_branch='main'
    )

    # Extract tar
    release_dir = extract_release_tar('release-v0.1.0-alpha.1.tar.gz')
"""

import tarfile
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional
from tempfile import TemporaryDirectory

from ..types.common import Place


def generate_metadata(
    version: str,
    git_commit: str,
    git_branch: str,
) -> Dict:
    """Generate metadata.json content for release.

    Args:
        version: Semantic version string (e.g., '0.1.0-alpha.1')
        git_commit: Git commit hash
        git_branch: Git branch name

    Returns:
        Metadata dictionary
    """
    return {
        'version': version,
        'build_date': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_commit,
        'git_branch': git_branch,
    }


def build_release_tar(
    version: str,
    gps_data_paths: Dict[Place, Path | str],
    output_path: Path | str,
    git_commit: str,
    git_branch: str = 'main',
    readme_path: Optional[Path | str] = None,
) -> Path:
    """Build a release tar archive from GPS data.

    This creates a complete release.tar.gz file with:
    - GPS trace data organized by city
    - metadata.json with build information
    - VERSION file
    - README.md (optional, if readme_path provided)

    Args:
        version: Semantic version (e.g., '0.1.0-alpha.1')
        gps_data_paths: Dict mapping city names to directories containing GPS .txt files
        output_path: Path where the tar file should be created
        git_commit: Git commit hash for metadata
        git_branch: Git branch name for metadata
        readme_path: Optional path to README.md file to include in release

    Returns:
        Path to created tar file

    Example:
        >>> build_release_tar(
        ...     version='0.1.0-alpha.1',
        ...     gps_data_paths={
        ...         'berlin': '/data/prepared/berlin',
        ...         'chicago': '/data/prepared/chicago'
        ...     },
        ...     output_path='releases/release-v0.1.0-alpha.1.tar.gz',
        ...     git_commit='abc123',
        ...     git_branch='main',
        ...     readme_path='documentation/release-readme.md'
        ... )
        PosixPath('releases/release-v0.1.0-alpha.1.tar.gz')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate GPS data paths exist
    for city, path in gps_data_paths.items():
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"GPS data path for {city} not found: {path}")

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        release_dir = tmpdir / f'release-v{version}'
        release_dir.mkdir()

        # Create directory structure
        input_dir = release_dir / 'input'
        gps_dir = input_dir / 'gps'
        gps_dir.mkdir(parents=True)

        # Copy GPS data for each city
        for city, source_path in gps_data_paths.items():
            source_path = Path(source_path)
            city_dir = gps_dir / city
            city_dir.mkdir()

            # Copy all .txt files
            txt_files = list(source_path.glob('*.txt'))
            if not txt_files:
                raise ValueError(f"No .txt files found in {source_path} for city {city}")

            for txt_file in txt_files:
                shutil.copy2(txt_file, city_dir / txt_file.name)

        # Generate metadata.json
        metadata = generate_metadata(version, git_commit, git_branch)
        with open(release_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Copy README.md if provided
        if readme_path is not None:
            readme_path = Path(readme_path)
            if not readme_path.exists():
                raise FileNotFoundError(f"README file not found: {readme_path}")
            shutil.copy2(readme_path, release_dir / 'README.md')

        # Generate VERSION file
        with open(release_dir / 'VERSION', 'w') as f:
            f.write(version)

        # Create tar archive
        with tarfile.open(output_path, 'w:gz') as tar:
            tar.add(release_dir, arcname=release_dir.name)

    return output_path


def extract_release_tar(
    tar_path: Path | str,
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Extract a release tar archive.

    Args:
        tar_path: Path to release tar file
        output_dir: Directory to extract to (default: current directory)

    Returns:
        Path to extracted release directory

    Example:
        >>> extract_release_tar('release-v0.1.0-alpha.1.tar.gz')
        PosixPath('release-v0.1.0-alpha.1')
    """
    tar_path = Path(tar_path)

    if not tar_path.exists():
        raise FileNotFoundError(f"Tar file not found: {tar_path}")

    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, 'r:gz') as tar:
        # Get the root directory name from tar
        members = tar.getmembers()
        if not members:
            raise ValueError(f"Tar file is empty: {tar_path}")

        root_name = members[0].name.split('/')[0]

        # Extract all files
        tar.extractall(path=output_dir)

        return output_dir / root_name


def list_release_contents(tar_path: Path | str) -> Dict:
    """List contents of a release tar without extracting.

    Args:
        tar_path: Path to release tar file

    Returns:
        Dictionary with release structure information

    Example:
        >>> info = list_release_contents('release-v0.1.0-alpha.1.tar.gz')
        >>> print(info['version'])
        '0.1.0-alpha.1'
        >>> print(info['cities'])
        ['berlin', 'chicago']
    """
    tar_path = Path(tar_path)

    if not tar_path.exists():
        raise FileNotFoundError(f"Tar file not found: {tar_path}")

    with tarfile.open(tar_path, 'r:gz') as tar:
        members = tar.getmembers()

        # Find metadata.json
        metadata = None
        for member in members:
            if member.name.endswith('metadata.json'):
                f = tar.extractfile(member)
                if f:
                    metadata = json.load(f)
                break

        # Find GPS cities
        cities = set()
        gps_file_count = {}
        for member in members:
            if '/input/gps/' in member.name and member.name.endswith('.txt'):
                parts = member.name.split('/input/gps/')
                if len(parts) > 1:
                    city = parts[1].split('/')[0]
                    cities.add(city)
                    gps_file_count[city] = gps_file_count.get(city, 0) + 1

        # Calculate total size
        total_size = sum(m.size for m in members)

        return {
            'version': metadata['version'] if metadata else None,
            'build_date': metadata['build_date'] if metadata else None,
            'git_commit': metadata['git_commit'] if metadata else None,
            'cities': sorted(cities),
            'gps_file_counts': gps_file_count,
            'total_files': len(members),
            'total_size_bytes': total_size,
        }
