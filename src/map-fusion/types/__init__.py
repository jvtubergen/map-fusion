"""Type definitions for map-fusion data structures.

This module provides comprehensive type definitions for the three main data sources
used in map reconstruction and fusion:
- GPS traces and GPS-derived graphs
- Satellite imagery and satellite-derived graphs
- OpenStreetMap data and fusion graphs

Usage:
    from types import GPSData, SatelliteData, OSMData
    from types import RegionOfInterest, BoundingBox
    from types import MapVariant, Place
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
]
