"""Type definitions for satellite imagery and satellite-derived graphs."""

from typing import TypedDict, Callable, Any
from numpy.typing import NDArray
from networkx import Graph
from PIL.Image import Image
from .common import Place, BoundingBox, ImageDimensions, Dimensions


class SatelliteImageMetadata(TypedDict):
    """Metadata associated with a satellite image."""
    zoom_level: int  # Google Maps zoom level
    gsd: float  # Ground Sample Distance (meters per pixel)
    bounds: BoundingBox  # Geographic bounds of the image
    upper_left: tuple[float, float]  # (latitude, longitude) of upper-left corner
    lower_right: tuple[float, float]  # (latitude, longitude) of lower-right corner


class SatelliteImage(TypedDict):
    """Satellite image data with metadata."""
    image: NDArray[Any] | Image  # Image array or PIL Image
    metadata: SatelliteImageMetadata
    path: str  # File path to the PNG image


class SatellitePaths(TypedDict):
    """File system paths for satellite data artifacts."""
    image: str  # Path to satellite image PNG
    pbmodel: str  # Path to Sat2Graph TensorFlow model (.pb)
    partial_gtes: str  # Folder for partial GTE (Graph Tensor Encoding) files
    intermediate: str  # Folder for intermediate processing files
    graph_file: str  # Path to final graph file
    image_results: str  # Folder for visualization results
    # Path generation functions
    gte_path: Callable[[int], str]  # Generate path for GTE by phase
    decoded_path: Callable[[int], str]  # Generate path for decoded graph by phase
    refined_path: Callable[[int], str]  # Generate path for refined graph by phase
    partial_gte_path: Callable[[int], str]  # Generate path for partial GTE by counter
    image_result_path: Callable[[int], str]  # Generate path for image results by phase


class SatelliteStatistics(TypedDict):
    """Statistical metrics for satellite imagery dataset."""
    zoom_level: int  # Google Maps zoom level used
    gsd: float  # Ground Sample Distance in meters per pixel
    surface_area_m2: float  # Total surface area in square meters
    surface_area_km2: float  # Total surface area in square kilometers
    surface_width: float  # Width of coverage in meters
    surface_height: float  # Height of coverage in meters
    image_width_pixels: int  # Image width in pixels
    image_height_pixels: int  # Image height in pixels
    total_pixels: int  # Total number of pixels
    bounds: BoundingBox  # Fine-tuned geographic bounds


class SatelliteData(TypedDict):
    """Complete satellite data representation."""
    place: Place  # Geographic location identifier
    image: SatelliteImage  # Satellite image with metadata
    graph: Graph  # NetworkX graph inferred from satellite imagery via Sat2Graph
    paths: SatellitePaths  # File system paths to satellite artifacts
    statistics: SatelliteStatistics  # Dataset statistics


class SatelliteGraphMetadata(TypedDict):
    """Metadata for satellite-derived graph."""
    place: Place
    source: str  # e.g., 'sat2graph'
    num_nodes: int
    num_edges: int
    image_bounds: BoundingBox
    processing_phases: int  # Number of Sat2Graph processing phases
