# Data related to graphs.

#TODO: Remove this file and the dependency on it.

# Valid graph sets to work with. GPS: Roadster, Sat: Sat2Graph, Truth: OpenStreetMaps. Extend with techniques as you see fit.
graphsets = ["roadster", "sat2graph", "openstreetmaps", "mapconstruction", "intersection", "merge_A", "merge_B", "merge_C"]
places    = ["athens", "berlin", "chicago"]

# Link active graphset to representative of gps, sat, and truth.
links = {
    "gps": "roadster",
    "osm": "openstreetmaps",
    "sat": "sat2graph"
}