# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- None

### Changed
- None

### Fixed
- None

## [0.1.0-alpha.1] - 2025-10-23

### Added

#### Release System
- **Build pipeline** (`build_release_data.sh`) - Complete release artifact generation
  - Automated GPS data download from mapconstruction repository
  - UTM to WGS84 coordinate conversion
  - Release tarball creation with SHA256 checksums
  - Support for test mode with stub data
  - Stage-based build support (input, unimodal, fused, results)

- **Testing infrastructure** (`test_release_e2e.sh`) - End-to-end validation
  - 27 comprehensive tests covering build → extract → load → verify
  - Support for both test mode and full GPS data testing
  - Automated coordinate range validation
  - Checksum verification
  - Directory structure validation

- **GitHub Actions workflow** (`.github/workflows/release.yml`)
  - Automated release builds triggered by version tags
  - VERSION file validation against git tags
  - Checksum verification
  - Automated GitHub Release creation
  - Pre-release marking for alpha versions
  - Manual workflow dispatch for testing

- **Example scripts** (`examples/load_release_example.py`)
  - Demonstrates loading and working with release artifacts
  - Shows metadata parsing and GPS data access
  - Includes statistics computation examples

#### Pipeline Infrastructure
- **Release loading module** (`src/map_fusion/pipeline/load.py`)
  - `load_release()` - Load complete release artifacts
  - `parse_version()` - Semantic version parsing
  - `format_release_info()` - Release summary formatting
  - `scan_gps_directory()` - GPS file cataloging
  - `load_release_metadata()` - Metadata parsing

- **GPS input processing** (`src/map_fusion/pipeline/input/gps.py`)
  - Downloads GPS data from mapconstruction repository
  - Extracts Berlin (27k+ files) and Chicago (800+ files) trace data
  - Converts UTM coordinates to WGS84 (lon/lat) format
  - Validates coordinate ranges for both cities

- **Archive operations** (`src/map_fusion/pipeline/tar.py`)
  - Tarball creation and extraction utilities
  - Release content listing
  - Checksum generation

#### Type System
- **Release types** (`src/map_fusion/types/release.py`)
  - `ReleaseArtifact` - Complete release structure
  - `ReleaseMetadata` - Build metadata (date, commit, branch)
  - `SemanticVersion` - Parsed version components
  - `GPSInputPaths` - GPS file paths per city

#### Documentation
- **Release process** (`documentation/releases.md`)
  - Complete guide to build scripts and workflows
  - Release structure and data format specifications
  - Module reference and API documentation
  - Quick reference for common workflows
  - Troubleshooting guide

- **Pipeline architecture** (`documentation/pipeline.md`)
  - Overview of pipeline stages
  - Data flow documentation

- **Release quickstart** (`documentation/release-quickstart.md`)
  - Getting started guide for releases

#### Data
- **GPS traces** (Berlin: 27,189 files, Chicago: 889 files)
  - WGS84 coordinate format: `<lon> <lat> <timestamp>`
  - Total: 310,583 GPS points
  - Berlin: 192,223 points (avg 7.1 per file)
  - Chicago: 118,360 points (avg 133.1 per file)

### Changed
- **Type imports** - Removed circular import between `types/__init__.py` and `pipeline.load`
- **Build script** - Updated to use `python -m` for module execution
- **Package structure** - Reorganized pipeline modules for better separation

### Fixed
- Circular import issue when running pipeline modules as scripts
- Module resolution in hatch environment
- GPS coordinate conversion accuracy

### Known Limitations
- Only GPS input stage fully implemented
- Satellite imagery stage: Structure created, implementation pending
- Unimodal reconstruction stage: Not yet implemented
- Map fusion stage: Not yet implemented
- Results/metrics stage: Not yet implemented
- Alpha quality - API subject to change

### Technical Details
- **Tarball size**: ~5.3 MB compressed (~114 MB uncompressed)
- **Build time**: ~5 minutes with actual GPS data, ~10 seconds in test mode
- **Python version**: 3.10+
- **Dependencies**: hatch, requests, utm, map_fusion package

---

## Version History

- **0.1.0-alpha.X**: GPS input stage only
- **0.2.0-alpha.X**: GPS + satellite input (planned)
- **0.3.0-alpha.X**: GPS + satellite + unimodal reconstruction (planned)
- **0.4.0-alpha.X**: All stages through fusion (planned)
- **0.5.0-beta.X**: Complete pipeline with results (planned)
- **1.0.0**: First stable release (planned)
