from external import *

from data_handling import *
from coordinates import * 
from utilities import *
from deduplicating import *
from node_extraction import *
from simplifying import *
from graph_curvature import *
from graph_coordinates import *

# Utility function for convenience to extract graph by name.
#   Either construction from raw data in folder or reading from graphml file.
#   Expect folders with raw data to exist at "data/maps", further with properties described by `construct_graph`.
#   Will either read and/or store from "graphs/" data folder.
def extract_graph(name, reconstruct=False):
    graphml_path = "graphs/"+name+".graphml"

    if Path(graphml_path).exists() and not reconstruct:
        G = ox.load_graphml(filepath=graphml_path)
    else:
        G = construct_graph("data/maps/" + name)
        ox.save_graphml(G, filepath=graphml_path)
    
    return G


# Extract stored graphs from a graph set. 
# * Optionally only retrieve graphs which exist, otherwise expect to retrieve gps, sat, and truth.
# * Expect all graphs to be stored as simplified MultiGraphs (thus undirected but potentially parallel edges with unique curvature).
def extract_graphset(name, optional=False):
    graphs = {}
    gps_path = f"graphsets/{name}/gps.graphml"
    sat_path = f"graphsets/{name}/sat.graphml"
    truth_path = f"graphsets/{name}/truth.graphml"
    if optional:
        if Path(gps_path).exists():
            graphs["gps"]   = ox.load_graphml(filepath=gps_path)
        if Path(sat_path).exists():
            graphs["sat"]   = ox.load_graphml(filepath=sat_path)
        if Path(truth_path).exists():
            graphs["truth"]   = ox.load_graphml(filepath=truth_path)
    else:
        graphs["gps"]   = ox.load_graphml(filepath=f"graphsets/{name}/gps.graphml")
        graphs["sat"]   = ox.load_graphml(filepath=f"graphsets/{name}/sat.graphml")
        graphs["truth"] = ox.load_graphml(filepath=f"graphsets/{name}/truth.graphml")
    return graphs


# Retrieve truth graph with a bounding box.
def retrieve_graph_truth(graphset, bbox):
    # First check the graph does not exist.
    graphml_path=f"graphsets/{graphset}/truth.graphml"
    if Path(graphml_path).exists():
        raise BaseException(f"The truth graph already exists at '{graphml_path}'.")
    # Then retrieve and store graph. (Note: G is already simplified at retrieval.)
    G = ox.graph_from_bbox(bbox=bbox, network_type="drive_service") 
    G = G.to_undirected() 
    ox.save_graphml(G, filepath=graphml_path)


# Save a graph to storage as a GraphML format.
# * Optionally overwrite if the target file already exists.
# * Writes the file into the graphs folder.
def save_graph(G, name, overwrite=False):
    graphml_path = "graphs/"+name+".graphml"
    G = G.copy()
    G.graph['crs'] = "EPSG:4326"
    G = nx.MultiDiGraph(G)
    if Path(graphml_path).exists() and not overwrite:
        print("Did not save graph: Not allowed to overwrite existing file.")
    else:
        ox.save_graphml(G, filepath=graphml_path)


# Construct network out of paths (a list of a list of coordinates)
def convert_paths_into_graph(pss, nid=1, gid=1):
    # Provide node_id offset.
    # Provide group_id offset.
    G = nx.Graph()
    for ps in pss:
        i = nid
        # Add nodes to graph.
        for p in ps:
            # Add random noise so its not overlapped in render
            G.add_node(nid, y=p[0] + 0.0001 * random.random(), x=p[1] + 0.0001 * random.random(), gid=gid)
            nid += 1
        # Add edges between nodes to graph.
        while i < nid - 1:
            G.add_edge(i, i+1, gid=gid)
            i += 1
        gid += 1
    return G


# Pick two nodes at random (repeat if in disconnected graphs) and find shortest path.
def gen_random_shortest_path(G):
    nodedict = extract_nodes_dict(G)
    # Pick two nodes from graph at random.
    nodes = random.sample(extract_nodes(G), 2)
    # Extract shortest path.
    path = None
    while path == None:
        nodes = random.sample(extract_nodes(G), 2)
        path = ox.routing.shortest_path(G, nodes[0][0], nodes[1][0])
    # Convert path node ids to coordinates.
    path = np.array([nodedict[nodeid] for nodeid in path])
    return path


#######################################
### Task specific graph preprocessing
#######################################

# Preparing graph for APLS usage (simplified, multi-edge, all edges have geometry property, all edges have an edge length property).
def graph_prepare_apls(G):
    G = simplify_graph(graph_transform_latlon_to_utm(G))
    G = graph_annotate_edge_length(G)
    G = graph_add_geometry_to_straight_edges(G) # Add geometry for straight line segments (edges with no curvature).
    # G.graph['crs'] = "EPSG:4326" # Set EPSG might be necessary for plotting results within APLS logic.
    return G


# Preparing graph for TOPO usage (simplified, multi-edge, all edges have geometry property, all edges have an edge length property).
def graph_prepare_topo(graph):
    # t0 = time()
    graph = simplify_graph(graph)
    graph = graph.to_directed(graph)
    graph = nx.MultiGraph(graph)
    graph = graph_add_geometry_to_straight_edges(graph)
    graph = graph_annotate_edge_length(graph)
    # verbose_print(f"Simplified graph in {int(time() - t0)} seconds.")
    return graph


#######################################
### Sanity check functionality
#######################################

# Perform a few sanity checks on the graph to prevent computation errors down the line.
def graph_sanity_check(G):
    print("Check graph.")

    # Simplification.
    if "simplified" not in G.graph:
        raise Exception("Expect 'simplified' dictionary key in graph.")

    # If simplified, then multigraph.
    if G.graph["simplified"] and type(G) != type(nx.MultiGraph()):
        raise Exception("Expect simplified graph to be an undirected multi-graph.") 
    if not G.graph["simplified"] and type(G) != type(nx.Graph()):
        raise Exception("Expect vectorized graph to be a undirected single-graph.") 

    # Coordinates.    
    if "coordinates" not in G.graph:
        raise Exception("Expect 'coordinates' dictionary key in graph.")
    if G.graph["coordinates"] != "utm" and G.graph["coordinates"] != "latlon":
        raise Exception("Expect 'coordinates' dictionary value to be either 'utm' or 'latlon'.")

    # Nodes.
    nodes = extract_node_positions(G)
    if G.graph["coordinates"] == "utm":
        if np.min(nodes) < 100: 
            print(nodes)
            raise Exception("Expect graph in utm coordinate system. Big chance some node is in latlon coordinate system.")
    if G.graph["coordinates"] == "latlon":
        if np.min(nodes) > 100: 
            print(nodes)
            raise Exception("Expect graph in latlon coordinate system. Big chance some node is in utm coordinate system.")

    # Node (x,y coordinates) flipping.
    coord0 = np.min(nodes, axis=0)
    coord1 = np.max(nodes, axis=0)
    diffa = np.max(nodes[:,0]) - np.min(nodes[:,0])
    diffb = np.max(nodes[:,1]) - np.min(nodes[:,1])
    # print(diffa, diffb)
    diffc = np.max(nodes[:,0]) - np.min(nodes[:,1])
    diffd = np.max(nodes[:,1]) - np.min(nodes[:,0])
    # print(diffc, diffd)

    if G.graph["coordinates"] == "latlon" and (abs(diffa) > 1 or abs(diffb) > 1):
        print(nodes)
        print(abs(diffa), abs(diffb))
        raise Exception("Expect node y, x coordinate consistency.") 
    if G.graph["coordinates"] == "utm" and (abs(diffa) > 100000 or abs(diffb) > 100000):
        print(nodes)
        print(abs(diffa), abs(diffb))
        raise Exception("Expect node y, x coordinate consistency.") 
    
    # Edges.
    nodes = extract_nodes_dict(G)
    if G.graph["simplified"]: 
        for (a, b, k, _) in G.edges(data=True, keys=True):
            ps = edge_curvature(G, a, b, k)
            if G.graph["coordinates"] == "latlon": # Convert to utm for computing in meters.
                ps = array([latlon_to_coord(latlon) for latlon in ps])
            if curve_length(ps) > 1000: # Expect reasonable curvature length.
                raise Exception("Expect edge length less than 1000 meters. Probably some y, x coordinate in edge curvature got flipped.")
            ps = edge_curvature(G, a, b, k) # Expect startpoint matches curvature.
            try:
                if (not np.all(ps[0] == nodes[a])) and (not np.all(ps[-1] != nodes[b])):
                    raise Exception("Expect curvature have same directionality as edge start and end edge.")
            except Exception as e:
                print(traceback.format_exc())
                print(e)
                breakpoint()
        
