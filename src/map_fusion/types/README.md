# Types Module

This module provides comprehensive type definitions for the map-fusion project's three main data sources:

## Structure

```
src/types/
├── __init__.py      # Exports all types
├── common.py        # Common types shared across data sources
├── gps.py          # GPS trace and GPS-derived graph types
├── satellite.py    # Satellite imagery and satellite-derived graph types
└── maps.py         # Map/OSM and fusion graph types
```

## Usage

### Importing Types

```python
# Import specific types
from types import GPSData, SatelliteData, OSMData
from types import RegionOfInterest, BoundingBox

# Import all types
from types import *
```

### GPS Data Types

**Key Types:**
- `GPSPoint`: Single GPS measurement `[x, y, t]` (UTM coordinates + timestamp)
- `GPSTrip`: Sequence of GPS points forming a trajectory
- `GPSTraces`: Collection of all GPS trips
- `GPSData`: Complete GPS data structure with traces, graph, ROI, and paths
- `GPSStatistics`: Dataset statistics (number of trips, distances, etc.)

**Example:**
```python
from types import GPSData, GPSTrip

def process_gps(data: GPSData) -> None:
    graph = data['graph']
    roi = data['roi']
    stats = data['statistics']
    print(f"Processing {stats['num_trips']} GPS trips")
```

### Satellite Data Types

**Key Types:**
- `SatelliteImage`: Image array/PIL Image with metadata
- `SatelliteImageMetadata`: Zoom level, GSD, bounds
- `SatelliteData`: Complete satellite data structure
- `SatelliteStatistics`: Coverage area, image dimensions, etc.
- `SatellitePaths`: File paths including path generation functions

**Example:**
```python
from types import SatelliteData, SatelliteImage

def process_satellite(data: SatelliteData) -> None:
    image = data['image']
    metadata = image['metadata']
    print(f"Image GSD: {metadata['gsd']} meters/pixel")
    print(f"Zoom level: {metadata['zoom_level']}")
```

### Map Data Types

**Key Types:**
- `OSMData`: OpenStreetMap ground truth data
- `MapData`: Generic map data for any variant
- `MapVariant`: Literal type for variant identifiers ("osm", "gps", "sat", "A", "B", "C", etc.)
- `FusionGraphMetadata`: Metadata for fusion algorithm results

**Example:**
```python
from types import OSMData, MapVariant

def load_map(place: str, variant: MapVariant) -> MapData:
    # Implementation
    pass

ground_truth: OSMData = load_map("berlin", "osm")
```

### Common Types

**Key Types:**
- `BoundingBox` / `RegionOfInterest`: Geographic bounds (north, south, east, west)
- `Coordinate`: Generic coordinate tuple
- `UTMCoordinate`: UTM coordinate system
- `LatLonCoordinate`: WGS84 lat/lon coordinate system
- `Place`: Location identifier string
- `ImageDimensions`: Pixel dimensions
- `Dimensions`: Physical dimensions in meters

**Example:**
```python
from types import RegionOfInterest, Place

def get_roi(place: Place) -> RegionOfInterest:
    return {
        'north': 52.520,
        'south': 52.515,
        'east': 13.405,
        'west': 13.400
    }
```

## Type Safety

These types are designed for use with:
- **Runtime type checking**: `isinstance()` checks with TypedDict
- **Static type checking**: mypy via `hatch run types:check`
- **IDE autocomplete**: Full IntelliSense support in VS Code, PyCharm, etc.

## Integration with Existing Code

The types are designed to match the existing data structures in:
- `src/data/gps.py` - GPS data handling
- `src/data/sat.py` - Satellite data handling
- `src/data/osm.py` - OSM data handling
- `src/storage/paths.py` - Path generation functions

You can gradually adopt these types in your codebase by adding type annotations to function signatures:

```python
from types import GPSData, Place

def read_gps_data(place: Place) -> GPSData:
    # Existing implementation
    pass
```

## For map-fusion-explorer

These types can be serialized to JSON for the Flask API and used to generate TypeScript types for the React frontend:

```python
# Backend: Return typed data
from types import GPSData

@app.route('/api/gps/<place>')
def get_gps_data(place: str) -> GPSData:
    return read_gps_data(place)
```

Consider using tools like `dataclasses-json` or `pydantic` for automatic JSON serialization if needed.
