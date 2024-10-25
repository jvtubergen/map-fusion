from external import *
from internal import *


def workflow_converting_stored_graphs_from_utm_into_latlon_coordinate_system():

    for graphset in graphsets:
        for place in places:
            G = read_graph(graphset=graphset, place=place, use_utm=True)
            G = graph_transform_utm_to_latlon(G, place)
            write_graph(G, graphset=graphset, place=place, overwrite=True, use_utm=False)


def workflow_reading_raw_gps_trajectories(place):

    print("Reading trajectories.")
    trips  = roadster.extract_trips(place)
    paths = [[[x,y] for [x,y,t] in trip] for trip in trips] # Drop t component

    return paths


def workflow_extract_roi_from_graph(place=None, graphset=None):

    G = read_graph(place=place, graphset=graphset)
    coordinates = extract_node_positions(G)
    lonlat0 = np.min(coordinates, axis=0)
    lonlat1 = np.max(coordinates, axis=0)
    # print(np.max(coordinates, axis=0) - np.min(coordinates, axis=0))
    west, south = lonlat0
    east, north = lonlat1
    roi = {
        "west":  west,
        "south": south,
        "east":  east,
        "north": north,
    }

    return roi


# truth = constructing_osm_on_boundaries(place="berlin", roi=roi["berlin"])
# write_graph(truth, place="berlin", graphset="openstreetmaps")
def workflow_constructing_osm_on_boundaries(roi=None):

    west  = roi["west"]
    east  = roi["east"]
    south = roi["south"]
    north = roi["north"]
    G = ox.graph_from_bbox(north=north, south=south, west=west, east=east, network_type="drive", retain_all=False, simplify=False)
    G = nx.Graph(G.to_undirected())

    return G


# workflow_extract_image_on_roi(place='berlin', margin=50, save=True)
def workflow_extract_image_on_roi(place=None, margin=0, save=False):

    roi = roi[place]
    south, west = translate_latlon_by_meters(lat=roi["south"], lon=roi["west"], west=margin, south=margin)
    north, east = translate_latlon_by_meters(lat=roi["north"], lon=roi["east"], east=margin, north=margin)

    lat_reference = 0.5 * south + 0.5 * north  # Reference latitude.
    scale = 2 # Always adhere to scale of 2: results in less requests.
    gsd_goal = 0.5
    deviation = 0.2
    zoom = gmaps.derive_zoom(lat_reference, scale, gsd_goal, deviation=deviation)
    gsd = gmaps.compute_gsd(lat_reference, zoom, scale)

    print(f"zoom level: {zoom}")
    print(f"gsd       : {gsd}")

    api_key = gmaps.read_api_key()
    image, coordinates = gmaps.construct_image(west=west, south=south, north=north, east=east, zoom=zoom, scale=scale, api_key=api_key, square=False, verbose=True)

    if save:
        imagefilename = f"{place}.png"
        coordinatesfilename = f"{place}_coordinates.pkl"
        gmaps.write_image(image, imagefilename)
        pickle.dump(coordinates, open(coordinatesfilename, "wb"))

    return image, coordinates
    

def workflow_render_gps_alongside_truth(place=None):

    gps = read_graph(place=place, graphset="roadster")
    truth = read_graph(place=place, graphset="openstreetmaps")
    plot_without_projection([(gps, {"color": (0,1,0,1)}), (truth, {"color": (0,0,1,1)})], []) # gps is green, truth is blue


def workflow_render_sat_alongside_truth(place=None):

    sat = read_graph(place=place, graphset="sat2graph")
    truth = read_graph(place=place, graphset="openstreetmaps")
    plot_without_projection([(sat, {"color": (0,1,0,1)}), (truth, {"color": (0,0,1,1)})], []) # sat is green, truth is blue


def workflow_render_sat_gps_truth(place=None, gps=True, sat=True, truth=True):

    graphs = []
    
    if sat:
        sat = read_graph(place=place, graphset="sat2graph")
        graphs.append((sat, {"color": (0,1,0,1)})) # sat is green
    
    if gps:
        gps = read_graph(place=place, graphset="roadster") 
        graphs.append((gps, {"color": (0.7,0,0.7,1)})) # gps is purple
    
    if truth:
        truth = read_graph(place=place, graphset="openstreetmaps")
        graphs.append((truth, {"color": (0,0,0,1)})) # truth is black

    plot_without_projection(graphs, []) 


def workflow_inferred_satellite_image_neighborhood_to_graph(place=None): 

    neighborhood = pickle.load(open(f"data/inferred satellite neighborhoods/{place}.pkl", "rb"))
    pixel_locations = pickle.load(open(f"data/satellite images and the pixel coordinates/{place}.pkl", "rb"))

    # locs = np.array(pixel_locations)
    # locs.shape #  (2608, 4216, 2)

    G = nx.Graph()
    nodes = neighborhood.keys()
    nids = {}
    # nodes = np.array([list(v) for v in neighborhood.keys()])
    # np.max(nodes, axis=0) # array([5122, 8316])

    nid = 1
    for element in neighborhood.keys():
        (y, x) = element # Image pixel offsets.
        lat, lon = pixel_locations[y // 2][x // 2] # Half x and y coordinate since we use satellite image scale of 2.
        G.add_node(nid, x=lon, y=lat)
        nids[element] = nid
        nid += 1

    # Add edges (and missing nodes?).
    for element, targets in neighborhood.items():
        snid = nids[element]
        for target in targets:
            if target not in nids.keys():
                print("Injecting missing node.")
                nids[target] = nid
                nid += 1
            tnid = nids[target]
            # Add edge between source and target node identifier.
            G.add_edge(snid, tnid)

    write_graph(G, graphset="sat2graph", place=place, overwrite=True)


# results = workflow_apply_apls(place="chicago", truth_graphset="openstreetmaps", proposed_graphset="roadster")
def workflow_apply_apls(place=None, truth_graphset=None, proposed_graphset=None):

    truth = read_graph(place=place, graphset=truth_graphset)
    proposed = read_graph(place=place, graphset=proposed_graphset)
    results = apls(truth=graph_prepare_apls(truth), proposed=graph_prepare_apls(proposed))
    return results


