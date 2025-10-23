"""Type definitions for map-fusion data structures.

This module provides comprehensive type definitions for the three main data sources
used in map reconstruction and fusion:
- GPS traces and GPS-derived graphs
- Satellite imagery and satellite-derived graphs
- OpenStreetMap data and fusion graphs
- Release artifacts and metadata

Usage:
    from map_fusion.types import GPSData, SatelliteData, OSMData
    from map_fusion.types import RegionOfInterest, BoundingBox
    from map_fusion.types import MapVariant, Place
    from map_fusion.types import ReleaseArtifact, load_release
"""

# Common types
from .common import (
    Coordinate,
    UTMCoordinate,
    LatLonCoordinate,
    BoundingBox,
    RegionOfInterest,
    Dimensions,
    ImageDimensions,
    Place,
)

# GPS types
from .gps import (
    GPSPoint,
    GPSTrip,
    GPSTraces,
    GPSPaths,
    GPSStatistics,
    GPSData,
    GPSGraphMetadata,
)

# Satellite types
from .satellite import (
    SatelliteImageMetadata,
    SatelliteImage,
    SatellitePaths,
    SatelliteStatistics,
    SatelliteData,
    SatelliteGraphMetadata,
)

# Map types
from .maps import (
    MapVariant,
    BaseVariant,
    FusionVariant,
    OSMPaths,
    MapPaths,
    OSMGraphMetadata,
    OSMData,
    FusionGraphMetadata,
    MapData,
)

# Release types (pure type definitions)
from .release import (
    ReleaseArtifact,
    ReleaseMetadata,
    SemanticVersion,
    GPSInputPaths,
)

# Note: Release utility functions (parse_version, load_release, etc.)
# are available from map_fusion.pipeline.load to avoid circular imports

__all__ = [
    # Common
    "Coordinate",
    "UTMCoordinate",
    "LatLonCoordinate",
    "BoundingBox",
    "RegionOfInterest",
    "Dimensions",
    "ImageDimensions",
    "Place",
    # GPS
    "GPSPoint",
    "GPSTrip",
    "GPSTraces",
    "GPSPaths",
    "GPSStatistics",
    "GPSData",
    "GPSGraphMetadata",
    # Satellite
    "SatelliteImageMetadata",
    "SatelliteImage",
    "SatellitePaths",
    "SatelliteStatistics",
    "SatelliteData",
    "SatelliteGraphMetadata",
    # Maps
    "MapVariant",
    "BaseVariant",
    "FusionVariant",
    "OSMPaths",
    "MapPaths",
    "OSMGraphMetadata",
    "OSMData",
    "FusionGraphMetadata",
    "MapData",
    # Release
    "ReleaseArtifact",
    "ReleaseMetadata",
    "SemanticVersion",
    "GPSInputPaths",
]
