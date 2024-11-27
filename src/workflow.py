from external import *
from internal import *

links = {
    "gps": "roadster",
    "osm": "openstreetmaps",
    "sat": "sat2graph"
}

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


# workflow_extract_image_on_roi(region=roi['berlin'], margin=50, savefile="berlin")
def workflow_extract_image_on_roi(region=None, margin=0, savefile=None):

    south, west = translate_latlon_by_meters(lat=region["south"], lon=region["west"], west=margin, south=margin)
    north, east = translate_latlon_by_meters(lat=region["north"], lon=region["east"], east=margin, north=margin)

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

    if savefile != None:
        imagefilename = f"{savefile}.png"
        coordinatesfilename = f"{savefile}_coordinates.pkl"
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
    print("Render sat, gps, and or truth graph.")

    graphs = []
    
    if sat:
        sat = read_graph(place=place, graphset="sat2graph")
        graphs.append((sat, {"color": (0,1,0,1)})) # sat is green
    
    if gps:
        gps = read_graph(place=place, graphset="roadster") 
        graphs.append((gps, {"color": (0.7,0.1,0.9,1), "linewidth": 3})) # gps is purple
    
    if truth:
        truth = read_graph(place=place, graphset="openstreetmaps")
        graphs.append((truth, {"color": (0.3,0.3,0.3,1), "linestyle": ":"})) # truth is black

    plot_without_projection(graphs, []) 


def workflow_inferred_satellite_image_neighborhood_to_graph(place=None): 
    print("Convert pixelwise neighborhood data to latlon-based graph.")

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
        lat, lon = pixel_locations[y][x] # Half x and y coordinate since we use satellite image scale of 2.
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


def workflow_report_apls_and_prime(place=None):

    sat_vs_osm = workflow_apply_apls(place=place, truth_graphset="openstreetmaps", proposed_graphset="sat2graph")
    gps_vs_osm = workflow_apply_apls(place=place, truth_graphset="openstreetmaps", proposed_graphset="roadster")
    sat_vs_gps = workflow_apply_apls(place=place, truth_graphset="sat2graph", proposed_graphset="roadster")

    print("sat vs osm:")
    print("APLS : ", sat_vs_osm[0])
    print("APLS+: ", sat_vs_osm[1])

    print("gps vs osm:")
    print("APLS : ", gps_vs_osm[0])
    print("APLS+: ", gps_vs_osm[1])

    print("sat vs gps:")
    print("APLS : ", sat_vs_gps[0])
    print("APLS+: ", sat_vs_gps[1])


##################################################################################
def workflow_apply_apls_prime(place=None, truth_graphset=None, proposed_graphset=None):

    truth = read_graph(place=place, graphset=truth_graphset)
    proposed = read_graph(place=place, graphset=proposed_graphset)

    return apls_prime(truth=graph_prepare_apls(truth), proposed=graph_prepare_apls(proposed))


# workflow_edge_coverage_by_threshold(place="chicago", left_graphset="gps", right_graphset="sat")
def workflow_edge_coverage_by_threshold(place=None, left_graphset=None, right_graphset=None):

    left  = read_graph(place=place, graphset=links[left_graphset])
    right = read_graph(place=place, graphset=links[right_graphset])

    print("left:", left)
    print("right:", right )

    # Edges in left graph which are not covered by right graph.
    covered_left = edge_wise_coverage_threshold(left, right, max_threshold=100)
    # Edges in right graph which are not covered by left graph.
    covered_right = edge_wise_coverage_threshold(right, left, max_threshold=100)
    
    return covered_left, covered_right


def workflow_construct_coverage_by_threshold(place=None):

    gps_vs_osm, osm_vs_gps = workflow_edge_coverage_by_threshold(place="chicago", left_graphset="gps", right_graphset="osm")
    pickle.dump(gps_vs_osm, open(f"data/coverage/{place}_gps_vs_osm.pkl", "wb"))
    pickle.dump(osm_vs_gps, open(f"data/coverage/{place}_osm_vs_gps.pkl", "wb"))

    sat_vs_osm, osm_vs_sat = workflow_edge_coverage_by_threshold(place="chicago", left_graphset="sat", right_graphset="osm")
    pickle.dump(sat_vs_osm, open(f"data/coverage/{place}_sat_vs_osm.pkl", "wb"))
    pickle.dump(osm_vs_sat, open(f"data/coverage/{place}_osm_vs_sat.pkl", "wb"))

    sat_vs_gps, gps_vs_sat = workflow_edge_coverage_by_threshold(place="chicago", left_graphset="sat", right_graphset="gps")
    pickle.dump(sat_vs_gps, open(f"data/coverage/{place}_sat_vs_gps.pkl", "wb"))
    pickle.dump(gps_vs_sat, open(f"data/coverage/{place}_gps_vs_sat.pkl", "wb"))


def workflow_apls_prime_outcomes_on_different_coverage_thresholds(place="chicago", left="gps", right="osm"):

    left_vs_right = pickle.load(open(f"data/coverage/{place}_{left}_vs_{right}.pkl", "rb"))
    left = read_graph(place="chicago", graphset=links[left])
    right = read_graph(place="chicago", graphset=links[right])

    for curr in range(10, 100, 10):
        subleft = subgraph_by_coverage_thresholds(left, left_vs_right, max_threshold=curr)
        result = apls(truth=graph_prepare_apls(right), proposed=graph_prepare_apls(subleft))
        print(f"threshold up to {curr}: ", result)
        # plot_graphs([subleft])


# workflow_construct_image_and_pixelcoordinates(place="berlin")
# workflow_inferred_satellite_image_neighborhood_to_graph(place="berlin")
# workflow_render_sat_gps_truth(place="berlin", gps=False)
def workflow_construct_image_and_pixelcoordinates(place=None):
    print("Constructing image and pixel coordinates.")
    region = roi[place]
    latlon0 = array((region['north'], region['west']))
    latlon1 = array((region['south'], region['east']))
    lat_reference = (0.5 * (latlon0 + latlon1))[0]  # Reference latitude.

    scale  = 2  # Scale perceptive field of satellite visual cognition part, higher quality/detail.
    gsd_goal  = 0.5
    deviation = 0.25
    zoom = gmaps.derive_zoom(lat_reference, scale, gsd_goal, deviation=deviation)
    gsd  = gmaps.compute_gsd(lat_reference, zoom, scale)

    p0 = array(gmaps.latlon_to_pixelcoord(lat=region["north"], lon=region["west"], zoom=zoom))
    p1 = array(gmaps.latlon_to_pixelcoord(lat=region["south"], lon=region["east"], zoom=zoom))

    p0[0] -= int(100 / scale) 
    p0[1] -= int(100 / scale)
    p1[0] += int(100 / scale)
    p1[1] += int(100 / scale)

    # Ensure multiple of stride.
    stride     = 88        # Step after each inferrence (half the inferrence window size).
    height     = p1[0] - p0[0]
    width      = p1[1] - p0[1]
    cut_height = height % stride
    cut_width  = width  % stride

    p0[0] += cut_height // 2 + cut_height % 2
    p1[0] -= cut_height // 2
    p0[1] += cut_width  // 2 + cut_width % 2
    p1[1] -= cut_width  // 2 

    height     = p1[0] - p0[0]
    width      = p1[1] - p0[1]
    assert height % stride == 0
    assert width % stride == 0

    north, west = gmaps.pixelcoord_to_latlon_secure(p0[0], p0[1], zoom)
    south, east = gmaps.pixelcoord_to_latlon_secure(p1[0], p1[1], zoom)

    # Double check width and height correctness.
    print("p0:", p0)
    p0 = array(gmaps.latlon_to_pixelcoord(lat=north, lon=west, zoom=zoom))
    print("p0:", p0)
    print("p1:", p1)
    p1 = array(gmaps.latlon_to_pixelcoord(lat=south, lon=east, zoom=zoom))
    print("p1:", p1)

    height     = p1[0] - p0[0]
    width      = p1[1] - p0[1]
    assert height % stride == 0
    assert width % stride == 0

    # print("height:", height)
    # print("width:", width)
    # return

    api_key = gmaps.read_api_key()
    image, coordinates = gmaps.construct_image(north=north, south=south, east=east, west=west, scale=2, zoom=zoom, api_key=api_key, verbose=True)
    # print(image.shape[0], image.shape[1])
    # print(coordinates.shape)
    assert(image.shape[0] == coordinates.shape[0])
    assert(image.shape[1] == coordinates.shape[1])

    gmaps.write_image(image, f"data/satellite images and the pixel coordinates/{place}.png")
    pickle.dump(coordinates, open(f"data/satellite images and the pixel coordinates/{place}.pkl", "wb"))