"""Type definitions for map data (primarily OpenStreetMap)."""

from typing import TypedDict, Literal
from networkx import Graph
from .common import RegionOfInterest, Place, BoundingBox


# Map variant types
MapVariant = Literal["osm", "gps", "sat", "A", "B", "C", "A2", "B2", "C2"]
BaseVariant = Literal["osm", "gps", "sat"]
FusionVariant = Literal["A", "B", "C", "A2", "B2", "C2"]


class OSMPaths(TypedDict):
    """File system paths for OpenStreetMap data artifacts."""
    graph_file: str  # Path to the OSM graph file (.graph format)


class MapPaths(TypedDict):
    """Generic file system paths for any map variant."""
    graph_file: str  # Path to the graph file


class OSMGraphMetadata(TypedDict):
    """Metadata for OpenStreetMap-derived graph."""
    place: Place
    source: str  # e.g., 'openstreetmap'
    network_type: str  # e.g., 'drive', 'walk', 'bike'
    num_nodes: int
    num_edges: int
    roi: RegionOfInterest
    simplified: bool  # Whether the graph was simplified
    retain_all: bool  # Whether all disconnected components were retained


class OSMData(TypedDict):
    """Complete OpenStreetMap data representation (ground truth)."""
    place: Place  # Geographic location identifier
    graph: Graph  # NetworkX graph from OpenStreetMap
    roi: RegionOfInterest  # Bounding box for the map region
    paths: OSMPaths  # File system paths to OSM artifacts
    metadata: OSMGraphMetadata


class FusionGraphMetadata(TypedDict):
    """Metadata for fusion graph (merged from multiple sources)."""
    place: Place
    variant: FusionVariant  # Fusion algorithm variant (A, B, C, A2, B2, C2)
    threshold: float  # Distance threshold used for merging
    inverse: bool  # Whether inverse fusion was applied
    num_nodes: int
    num_edges: int
    source_variants: list[BaseVariant]  # Base variants that were fused


class MapData(TypedDict):
    """Generic map data representation for any variant."""
    place: Place
    variant: MapVariant
    graph: Graph  # NetworkX graph
    roi: RegionOfInterest
    paths: MapPaths
    threshold: float | None  # Only for fusion variants
    inverse: bool  # Only for fusion variants
