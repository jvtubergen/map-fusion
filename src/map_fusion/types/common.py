"""Common type definitions shared across GPS, satellite, and map data."""

from typing import TypedDict, Tuple
from numpy.typing import NDArray
import numpy as np


# Coordinate types
Coordinate = Tuple[float, float]  # (latitude, longitude) or (y, x) in various systems
UTMCoordinate = Tuple[float, float]  # (northing, easting) in UTM
LatLonCoordinate = Tuple[float, float]  # (latitude, longitude) in WGS84


class BoundingBox(TypedDict):
    """Geographic bounding box in lat/lon coordinates."""
    north: float
    south: float
    east: float
    west: float


class RegionOfInterest(TypedDict):
    """Region of interest with geographic bounds."""
    north: float
    south: float
    east: float
    west: float


class Dimensions(TypedDict):
    """Physical dimensions in meters."""
    width: float
    height: float


class ImageDimensions(TypedDict):
    """Image dimensions in pixels."""
    width: int
    height: int


# Place identifiers
Place = str  # 'berlin', 'chicago', 'athens'
