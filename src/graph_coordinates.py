from external import *
from utilities import *
from coordinates import *

from srs import *

# Obtain middle latitude coordinate for bounding box that captures all nodes in the graph.
def middle_latitute(G):
    assert G.graph["coordinates"] == "latlon"
    uvk, data = zip(*G.nodes(data=True))
    df = gpd.GeoDataFrame(data, index=uvk)
    alat, alon = df["y"].mean(), df["x"].mean()
    return alat



# Obtain bounding box (the region of interest).
def roi(G):
    assert G.graph["coordinates"] == "latlon"

    G = read_graph(place=place, graphset=graphset)
    coordinates = extract_node_positions_list(G)
    latlon0 = np.min(coordinates, axis=0)
    latlon1 = np.max(coordinates, axis=0)
    # print(np.max(coordinates, axis=0) - np.min(coordinates, axis=0))
    south, west = latlon0
    north, east = latlon1
    roi = {
        "west":  west,
        "south": south,
        "east":  east,
        "north": north,
    }

    return roi



# Compute relative positioning.
# Note: Takes a reference latitude for deciding on the GSD. Make sure to keep this value consistent when applied to different graphs.
# Note: Flip y-axis by subtracting from minimal latitude value (-max_lat) to maintain directionality.
# todo: Rely on UTM conversion instead of your hacky solution.
def transform_geographic_coordinates_into_scaled_pixel_positioning(G, reflat):
    assert G.graph["coordinates"] == "latlon"
    # 0. GSD on average latitude and reference pixel positions.
    zoom = 24 # Sufficiently accurate.
    gsd = compute_gsd(reflat, zoom, 1)
    # 1. Vectorize.
    G = vectorize_graph(G)
    # 2. Map all nodes to relative position.
    uvk, data = zip(*G.nodes(data=True))
    df = gpd.GeoDataFrame(data, index=uvk)
    maxy, _ = latlon_to_pixelcoord(-max_lat, 0, zoom)
    # Map lat,lon to y,x with latlon_to_pixelcoord.
    def latlon_to_relative_pixelcoord(row): 
        lat, lon = row["y"], row["x"]
        y, x = latlon_to_pixelcoord(lat, lon, zoom)
        return {**row, 'y': maxy - gsd * y, 'x': gsd * x }
    # Construct relabel mapping and transform node coordinates to relative scaled pixel position.
    relabel_mapping = {}
    for nid, data in G.nodes(data=True):
        relabel_mapping[nid] = latlon_to_relative_pixelcoord(data)
    nx.set_node_attributes(G, relabel_mapping)
    # 3. Convert back into simplified graph.
    return simplify_graph(G) # If it has curvature it crashes because of non-hashable numpy array in attributes.
    return G
    

# Abstract function with core logic for utm/latlon graph conversion.
def graph_transform_generic(G, coordinate_transformer):

    G = G.copy()

    node_relabel_mapping = {}
    edge_relabel_mapping = {}

    # Adjust coordinates of graph nodes.
    for nid, attrs in G.nodes(data=True):
        y, x = attrs["y"], attrs["x"]
        y, x = coordinate_transformer(y, x)
        node_relabel_mapping[nid] = {**attrs, "y": y, "x": x}
    
    nx.set_node_attributes(G, node_relabel_mapping)
    
    # Update edge curvature.
    for eid, attrs in iterate_edges(G):

        curvature = attrs["curvature"]
        curvature = array([coordinate_transformer(y, x) for y, x in curvature])
        geometry  = to_linestring(curvature)
        edge_length = curve_length(curvature)

        edge_relabel_mapping[eid] = {**attrs, "geometry": geometry, "curvature": curvature, "length": edge_length}

    nx.set_edge_attributes(G, edge_relabel_mapping)

    return G


# Transform graphnodes UTM coordinate system into latitude-longitude coordinates.
@info()
def graph_transform_utm_to_latlon(G, place):

    assert G.graph["coordinates"] == "utm"

    # Convert coordinates.
    coordinate_transformer = lambda y, x: utm_to_latlon((y, x), place)
    G = graph_transform_generic(G, coordinate_transformer)

    G.graph["coordinates"] = "latlon"

    return G


# Transform graphnodes latitude-longitude coordinates into UTM coordinate system.
@info()
def graph_transform_latlon_to_utm(G):

    assert G.graph["coordinates"] == "latlon"

    coordinate_transformer = lambda y, x: latlon_to_utm((y, x))
    G = graph_transform_generic(G, coordinate_transformer)

    G.graph["coordinates"] = "utm"

    return G


# Derive UTM zone number and zone letter from a graph by a latlon coordinate.
def graph_utm_info(G):
    assert G.graph["coordinates"] == "latlon"
    node = G._node[list(G.nodes())[0]]
    lat, lon = node['y'], node['x']
    _, _, zone_number, zone_letter = utm.conversion.from_latlon(lat, lon)
    return {"number": zone_number, "letter": zone_letter}

# Derive place from UTM information
def graph_utm_place(G):
    assert G.graph["coordinates"] == "latlon"
    node = G._node[list(G.nodes())[0]]
    lat, lon = node['y'], node['x']
    _, _, zone_number, zone_letter = utm.conversion.from_latlon(lat, lon)
    for place in places:
        if zone_number == zone_numbers[place] and zone_letter == zone_letters[place]:
            return place
    check(false, expect="Expect to obtain a place from graph UTM coordinate")
    