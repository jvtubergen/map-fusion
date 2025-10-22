"""Type definitions for GPS trace data and GPS-derived graphs."""

from typing import TypedDict, List
from networkx import Graph
from .common import RegionOfInterest, Place


# GPS trace types
GPSPoint = List[float]  # [x, y, t] - UTM x-coordinate, UTM y-coordinate, timestamp
GPSTrip = List[GPSPoint]  # Sequence of GPS points forming a single trip/trajectory
GPSTraces = List[GPSTrip]  # Collection of all GPS trips


class GPSPaths(TypedDict):
    """File system paths for GPS data artifacts."""
    inferred_folder: str  # Folder containing inferred graph data from Roadster
    graph_file: str  # Path to the processed graph file (.graph format)
    raw_folder: str  # Folder containing raw GPS data
    processed_folder: str  # Folder containing processed GPS data
    traces_folder: str  # Folder containing GPS trace files


class GPSStatistics(TypedDict):
    """Statistical metrics for GPS trace dataset."""
    num_trips: int  # Total number of GPS trips/trajectories
    avg_samples_per_trip: float  # Average number of GPS points per trip
    avg_distance_per_trip: float  # Average trip distance in meters
    total_samples: int  # Total number of GPS points across all trips
    total_distance: float  # Total distance covered in meters
    avg_sample_interval: float  # Average time interval between consecutive samples


class GPSData(TypedDict):
    """Complete GPS data representation."""
    place: Place  # Geographic location identifier
    traces: GPSTraces  # Raw GPS trajectories
    graph: Graph  # NetworkX graph inferred from GPS traces
    roi: RegionOfInterest  # Bounding box derived from GPS traces
    paths: GPSPaths  # File system paths to GPS artifacts
    statistics: GPSStatistics  # Dataset statistics


class GPSGraphMetadata(TypedDict):
    """Metadata for GPS-derived graph."""
    place: Place
    source: str  # e.g., 'roadster', 'custom'
    num_nodes: int
    num_edges: int
    roi: RegionOfInterest
