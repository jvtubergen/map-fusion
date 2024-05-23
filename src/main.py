import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import numpy as np
from shapely.geometry import LineString, Point

# IO dependencies
from pathlib import Path
from fileinput import input

# Utils
from operator import itemgetter
import random


# Construct graph from data in specified folder. 
# Expect folder to have to files edges.txt and vertices.txt. 
# Expect those files to be CSV with u,v and id,x,y columns respectively .
def construct_graph(folder):

    edges_file_path    = folder + "/edges.txt"
    vertices_file_path = folder + "/vertices.txt"

    # Assuming the text files are CSV formatted
    edges_df = pd.read_csv(edges_file_path)
    vertices_df = pd.read_csv(vertices_file_path)

    # Construct NetworkX graph
    G = nx.Graph()

    # Track node dictionary to simplify node extraction when computing edge lengths.
    nodedict = {}
    for node in vertices_df.iterrows():
        i, x, y = itemgetter('id', 'x', 'y')(node[1]) 
        G.add_node(int(i), x=x*.00001, y=y*.00001)
        nodedict[i] = np.asarray([x, y], dtype=np.float64, order='c')

    for edge in edges_df.iterrows():
        u, v = itemgetter('u', 'v')(edge[1]) 
        # todo: Compute edge lengts (inter-node distance)
        G.add_edge(int(u), int(v), length = np.linalg.norm(nodedict[u] - nodedict[v]))

    G = nx.MultiDiGraph(G)
    G = ox.simplify_graph(G)
    G.graph['crs'] = "EPSG:4326"

    return G


# Utility function for convenience to extract graph by name.
# Either construction from raw data in folder or reading from graphml file.
# Expect folders with raw data to exist at "data/maps", further with properties described by `construct_graph`.
# Will either read and/or store from "graphs/" data folder.
def extract_graph(name, reconstruct=False):
    graphml_path = "graphs/"+name+".graphml"

    if Path(graphml_path).exists() and not reconstruct:
        G = ox.load_graphml(filepath=graphml_path)
    else:
        G = construct_graph("data/maps/" + name)
        ox.save_graphml(G, filepath=graphml_path)
    
    return G


# The names of the graphs we currently support, thus graphs we can work with.
names = [
 'athens_large',
 'athens_small',
 'berlin',
 'chicago',
 'athens_large_kevin',
 'chicago_kevin',
 'roadster_athens'
]

# Extract nodes from a graph into the format `(id, nparray(x,y))`.
def extract_nodes(G):
    return [( node, np.asarray([data['x'], data['y']], dtype=np.float64, order='c') ) for node, data in G.nodes(data = True)]

# Extract nodes from a graph as a dictionary `{nid: nparray([x,y])}`.
def extract_nodes_dict(G):
    d = {}
    for node, data in G.nodes(data = True):
        d[node] = np.asarray([data['x'], data['y']], dtype=np.float64, order='c')
    return d

# Seek nearest vertex in graph of a specific coordinate of interest.
# Expect point to be a 2D numpy array.
def nearest_point(G, p):
    # Example (extracting nearest vertex):
    # nearest_point(extract_graph("chicago"), np.asarray((4.422440 , 46.346080), dtype=np.float64, order='c'))
    points = extract_nodes(G)

    # Seek nearest point
    dmin = 1000000
    ires = None
    qres = None
    for (i, q) in points:
        m = np.linalg.norm(p - q)
        if m < dmin:
            dmin = m
            (ires, qres) = (i, q)

    return (ires, qres)

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


# Example (Render all graphs.):
# for name in names:
#     G = extract_graph(name)
#     ox.plot_graph(G)


# Construct network out of paths (a list of a list of coordinates)
def convert_paths_into_graph(pss):
    nid = 1 # node id
    gid = 1 # group id
    G = nx.Graph()
    for ps in pss:
        i = nid
        # Add nodes to graph.
        for p in ps:
            G.add_node(nid, x=p[0], y=p[1], gid=gid)
            nid += 1
        # Add edges between nodes to graph.
        while i < nid - 1:
            G.add_edge(i, i+1, gid=gid)
            i += 1
        gid += 1
    return G


def gen_random_shortest_path(G):
    nodedict = extract_nodes_dict(G)
    # Pick two nodes from graph at random.
    nodes = random.sample(extract_nodes(G), 2)
    # Extract shortest path.
    path = None
    while path == None:
        path = ox.routing.shortest_path(G, nodes[0][0], nodes[1][0])
    # Convert path node ids to coordinates.
    path = np.array([nodedict[nodeid] for nodeid in path])
    return path


def render_paths(pss):
    G = convert_paths_into_graph(pss)
    G = nx.MultiDiGraph(G)
    G.graph['crs'] = "EPSG:4326"
    nc = ox.plot.get_node_colors_by_attr(G, "gid", cmap="tab20b")
    ec = ox.plot.get_edge_colors_by_attr(G, "gid", cmap="tab20b")
    ox.plot_graph(G, bgcolor="#ffffff", node_color=nc, edge_color=ec)

# Example (rendering multiple paths)
# G = extract_graph("chicago")
# render_paths([gen_random_shortest_path(G), gen_random_shortest_path(G)])

