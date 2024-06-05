###################################
###  
###################################


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


# Return node IDs of nodes without any edges.
def isolated_nodes(G):
    return [node_id for node_id in G.nodes() if len(G[node_id]) == 0]
# Example (removing isolated nodes in G):
# G.remove_nodes_from(isolated_nodes(G))


# Apply Minskowski sum to obtain area
# def minskowski_curve(ps, l = 0.05):



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