import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import numpy as np
from shapely.geometry import LineString, Point
import utm

# IO dependencies
from pathlib import Path
from fileinput import input

# Utils
from operator import itemgetter
import random

# Library code
from coverage import *
from network import *


# Return node IDs which are duplicated (exact same coordinate).
# TODO: Group togeter node-IDs that are duplicates of one another as tuples.
def duplicated_nodes(G):
    # NOTE: Keep IDs (integeters) separate from coordinates (floats): Numpy arrays all have same type.
    nodes = G.nodes()
    coordinates = np.array([[info["x"], info["y"]] for node, info in G.nodes(data=True)])
    uniques, inverses, counts = np.unique( coordinates, return_inverse=True, axis=0, return_counts=True )
    duplicated = []
    for node_id, index_to_unique in zip(nodes, inverses):
        if counts[index_to_unique] > 1:
            duplicated.append(node_id)
    return duplicated


# Group duplicated nodes.
def duplicated_nodes_grouped(G):
    # NOTE: Keep IDs (integeters) separate from coordinates (floats): Numpy arrays all have same type.
    node_ids = G.nodes()
    coordinates = np.array([[info["x"], info["y"]] for node, info in G.nodes(data=True)])
    uniques, inverses, counts = np.unique( coordinates, return_inverse=True, axis=0, return_counts=True )
    # Construct dictionary.
    duplicated = {}
    for node_id, index_to_unique in zip(node_ids, inverses):
        if counts[index_to_unique] > 1:
            if index_to_unique in duplicated.keys():
                duplicated[index_to_unique].append(node_id)
            else:
                duplicated[index_to_unique] = [node_id]
    # Convert dictionary into a list.
    result = []
    for v in duplicated:
        result.append(duplicated[v])
    return result


# Return edge IDs which are duplicated (exact same coordinates).
# NOTE: Expects vectorized graph.
# TODO: Group togeter edge-IDs that are duplicates of one another as tuples.
def duplicated_edges(G):

    if type(G) != nx.MultiDiGraph:
        raise BaseException("Expect to call duplicated_edge_grouped on an nx.MultiDiGraph.")

    if G.graph["simplified"]:
        raise BaseException("Duplicated edges function is supposed to be called on a vectorized graph.")

    # Construct edges coordinates in the format of `[x1,y1,x2,y2]`.
    # NOTE: Keep IDs (integeters) separate from coordinates (floats): Numpy arrays all have same type.
    edges = G.edges(keys=True)
    coordinates = []
    for (a, b, k, attrs) in G.edges(keys=True, data=True):
        x1 = G.nodes[a]["x"]
        y1 = G.nodes[a]["y"]
        x2 = G.nodes[b]["x"]
        y2 = G.nodes[b]["y"]
        coordinates.append([x1,y1,x2,y2])
    coordinates = np.array(coordinates)

    # Extract duplications
    uniques, inverses, counts = np.unique( coordinates, return_inverse=True, axis=0, return_counts=True )
    duplicated = []
    for edge_id, index_to_unique in zip(edges, inverses):
        if counts[index_to_unique] > 1:
            duplicated.append(edge_id)
    return duplicated


# Group duplicated edges.
# NOTE: Expects MultiDiGraph
def duplicated_edges_grouped(G):

    if type(G) != nx.MultiDiGraph:
        raise BaseException("Expect to call duplicated_edge_grouped on an nx.MultiDiGraph.")

    if G.graph["simplified"]:
        raise BaseException("Duplicated edges function is supposed to be called on a vectorized graph (thus not simplified).")

    # NOTE: Keep IDs (integeters) separate from coordinates (floats): Numpy arrays all have same type.
    edge_ids = G.edges(keys=True)
    coordinates = []
    for (a, b, k, attrs) in G.edges(keys=True, data=True):
        x1 = G.nodes[a]["x"]
        y1 = G.nodes[a]["y"]
        x2 = G.nodes[b]["x"]
        y2 = G.nodes[b]["y"]
        coordinates.append([x1,y1,x2,y2])
    coordinates = np.array(coordinates)

    # Construct dictionary.
    uniques, inverses, counts = np.unique( coordinates, return_inverse=True, axis=0, return_counts=True )
    duplicated = {}
    for edge_id, index_to_unique in zip(edge_ids, inverses):
        if counts[index_to_unique] > 1:
            if index_to_unique in duplicated.keys():
                duplicated[index_to_unique].append(edge_id)
            else:
                duplicated[index_to_unique] = [edge_id]

    # Convert dictionary into a list.
    result = []
    for v in duplicated:
        result.append(duplicated[v])

    return result


# Uniform add edge curvature attribute to every edge to afterwards allow/simplify comparisons.
# NOTE: Numpy Array cant be used for imhomogeneous dimensionality. 
#       LineString probably too slow because it cant be hashed/indexed.
#       Unsure how to leverage DataFrame or Series for comparison.
def unify_edge_curvature(G):
    for (a, b, k, attrs) in G.edges(keys=True, data=True):
        if "geometry" not in attrs.keys(): 
            p = (G.nodes[a]["x"],G.nodes[a]["y"])
            q = (G.nodes[b]["x"],G.nodes[b]["y"])
            ps = [p, q]
            G.edges[(a,b,k)]["curvature"] = ps
        else:
            linestring = attrs["geometry"]
            ps = list(linestring.coords)
        G.edges[(a,b,k)]["curvature"] = ps


# Debugging function to print node duplicates.
def print_node_duplicates(G):
    nodes = np.array([[node, info["x"], info["y"]]  for node, info in G.nodes(data=True)])
    coordinates = nodes[:,(1,2)]
    # Figure out duplicate coordinates of nodes.
    u, indices, counts = np.unique( coordinates, return_inverse=True, axis=0, return_counts=True )
    return u, indices, counts


# Debugging function to print edge duplicates. Assumes vectorized graph.
def print_duplicated_edges(G):
    edges = G.edges(keys=True, data=True) 
    data = np.array([(a,b,k) for a, b, k, info in G.edges(keys=True, data=True)])
    u, indices, counts = np.unique( data, return_inverse=True, axis=0, return_counts=True )
    return u, indices, counts 


# Extract self-loop graph.
def extract_selfloop_graph(G):
    loops = list(nx.selfloop_edges(G, keys=True))
    H = nx.MultiGraph(G).edge_subgraph(loops)
    return H


# Convert edge geometry (LineString) into a list of coordinates.
# With `as_string` Convert edge geometry (LineString) into its serialized String.
def extract_geometry_from_attributes(attrs, as_string=False):
    if "geometry" in attrs.keys():
        if as_string:
            return attrs["geometry"].wkt
        return list(attrs["geometry"].coords)
    else:
        return [[]]


# Rendering duplicated nodes and edges.
def render_duplicates_highlighted(G):
    G = G.copy()

    # Give everyone GID 2
    nx.set_node_attributes(G, 2, name="gid")
    nx.set_edge_attributes(G, 2, name="gid")

    for key in duplicated_nodes(G):
        G.nodes[key]["gid"] = 1

    for key in duplicated_edges(G):
        G.edges[key]["gid"] = 1

    # Render
    nc = ox.plot.get_node_colors_by_attr(G, "gid", cmap="winter")
    ec = ox.plot.get_edge_colors_by_attr(G, "gid", cmap="winter")
    ox.plot_graph(G, bgcolor="#ffffff", node_color=nc, edge_color=ec)


# TODO: Convert geometry into a list of coordiantes.
#       Either with a function on LineString
#       Or introduce a new attribute like "curvature".
def check_curvature_duplication(G, k1, k2):
    ps = G.edges[k1]["geometry"]
    qs = G.edges[k2]["geometry"]
    return ps == qs or ps == qs[::-1]


# Obtain node key by coordinate.
# TODO: Rather than returning first (thus expecting it exists), return a (potentially empty) list of keys.
def node_by_coord(G, coord):
    nodes = G.nodes()
    coordinates = np.array([[info["x"], info["y"]]  for node, info in G.nodes(data=True)])
    indices = np.where((coordinates == coord).all(axis=1))
    return nodes[indices[0][0]]


def isolated_nodes(G):
    return [node_id for node_id in G.nodes() if len(G[node_id]) == 0]


# Example (removing isolated nodes in G):
# G.remove_nodes_from(isolated_nodes(G))
   


# Remove duplicated nodes and edges from vectorized graph.
# NOTE: Since a vectorized only stores directly
#       No need to 
def deduplicate_vectorized_graph(G):

    G = G.copy()
    G = nx.Graph(G)
    # Assert G is a Graph (unidirectional and single path)

    # Deduplicate nodes: Adjust edges of deleted nodes + Delete nodes.
    for group in duplicated_nodes_grouped(G):
        base = group[0]
        # for n, nbrs in G.adj.items():
        #     # n: from node id
        #     # nbrs: {<neighboring node id>: {<edge-id-connecting n with nbr>: <edge attributes>}}]
        for remove in group[1:]:
            # Replace edges
            nbrs = G.adj[remove]
            to_delete = []
            to_add = []
            for nbr in nbrs:
                edge = G[remove][nbr]
                to_delete.append((remove,nbr))
                to_add.append((base,nbr, *edge))
            # Delete and add in a single go to prevent changing the nbrs loop elements.
            G.remove_edges_from(to_delete)
            G.add_edges_from(to_add)
            
            # Expect remove node to be isolated now.
            assert len(G[remove]) == 0
            G.remove_node(remove)
    
    # Deduplicate edges:
    # Since G is a graph, adding the edges already resolves into edge deduplication.
    G = nx.MultiDiGraph(G) # Convert back into MultiDiGraph.
    
    return G


# Example (vectorize and simplify again):
# Note: It contains a bug: bidirectional self-loops are incorrectly removed.
G = extract_graph("chicago_kevin")
G2 = vectorize_graph(G)
G2 = deduplicate_vectorized_graph(G2)
G3 = ox.simplify_graph(G2)
G5 = ox.simplify_graph(vectorize_graph(G3))
# # Simple check on vectorization validity.
# assert len(G.nodes()) == len(G3.nodes())
# assert len(G.edges()) == len(G3.edges())


# Example (Subgraph of nodes nearby curve):
# G = extract_graph("chicago")
# idx = graphnodes_to_rtree(G)
# ps = gen_random_shortest_path(G)
# bb = bounding_box(ps)
# H = rtree_subgraph_by_bounding_box(G, idx, bb)
