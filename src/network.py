from external import *

from data_handling import *
from coordinates import * 
from utilities import *
from graph_deduplicating import *
from graph_node_extraction import *
from graph_simplifying import *
from graph_curvature import *
from graph_coordinates import *


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
    nodedict = extract_node_positions_dictionary(G)
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