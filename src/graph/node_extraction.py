from external import *


# Extract nodes from a graph into the format `(id, nparray(x,y))`.
def extract_nodes(G):
    return [( node, np.asarray([data['y'], data['x']], dtype=np.float64, order='c') ) for node, data in G.nodes(data = True)]


# Extract nodes from a graph as a dictionary `{nid: nparray([x,y])}`.
def extract_node_positions_dictionary(G):
    d = {}
    for node, data in G.nodes(data = True):
        d[node] = np.asarray([data['y'], data['x']], dtype=np.float64, order='c')
    return d


# Extract node positions, ignoring node ids.
def extract_node_positions_list(G):
    return np.asarray([[data['y'], data['x']] for node, data in G.nodes(data = True)], dtype=np.float64, order='c')


# Link nid to node position.
def node_positions_nid(G):
    return {(data['y'], data['x']): nid for nid, data in G.nodes(data = True)}


# Seek nearest vertex in graph of a specific coordinate of interest.
# Expect point to be a 2D numpy array.
def nearest_point(G, p):
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


