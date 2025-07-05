from external import *
from utilities import *
from graph.utilities.general import *
import graph.utilities.attributes as attributes

#######################################
### Graph distance
#######################################


# Extract nodes from a graph as a dictionary `{nid: nparray([x,y])}`.
def extract_node_positions_dictionary(G):
    d = {}
    for node, data in G.nodes(data = True):
        d[node] = np.asarray([data['y'], data['x']], dtype=np.float64, order='c')
    return d


# Extract nodes from a graph into the format `(id, nparray(x,y))`.
def extract_nodes_distance_tuple(G):
    return [( node, np.asarray([data['y'], data['x']], dtype=np.float64, order='c') ) for node, data in G.nodes(data = True)]


# Extract node position of nid.
def graphnode_position(G, nid):
    position = y, x = G._node[nid]['y'], G._node[nid]['x'],
    return array(position)


# Extract node positions, ignoring node ids.
def extract_node_positions_list(G):
    return np.asarray([[data['y'], data['x']] for node, data in G.nodes(data = True)], dtype=np.float64, order='c')


# Link nid to node position.
def node_positions_nid(G):
    return {(data['y'], data['x']): nid for nid, data in G.nodes(data = True)}


# Seek nearest vertex in graph of a specific coordinate of interest.
# Expect point to be a 2D numpy array.
def nearest_point(G, p):
    points = extract_nodes_distance_tuple(G)

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


@info()
def graph_distance_node_node(G, u, v):
    p = graphnode_position(G, u)
    q = graphnode_position(G, v)
    return norm(p - q)


# Obtain nearest node for nid in a graph.
@info()
def nearest_node(G, nid, node_tree=None, excluded_nids=set()):

    if node_tree == None:
        node_tree = graphnodes_to_rtree(G)

    bbox = graphnode_to_bbox(G, nid)

    # Exclude target nid from hitting.
    to_exclude = excluded_nids.union([nid])

    # Iterate node tree till we find a nid not excluded.
    for found in nearest_rtree_bbox(node_tree, bbox):
        check(found != None, expect="Expect non-null node identifier found on seeking nearest element in rtree.")
        if found not in to_exclude:
            return found
    
    logger(f"Checked {len(nearest_rtree_bbox(node_tree, bbox))} nearest elements out of {len(node_tree)} node-tree elements.")
    check(False, expect="Expect to find nearest node.")


#######################################
### Edges
#######################################

# Obtain nearest edge for 
@info()
def nearest_edge(G, nid, edge_tree=None, excluded_eids=set()):

    if edge_tree == None:
        edge_tree = graphedges_to_rtree(G)

    # Seek distance to edge of first hit.
    bbox = graphnode_to_bbox(G, nid)

    # Extend excluded eids with those connected to nid.
    to_exclude = excluded_eids.union(set([format_eid(G, eid) for eid in G.edges(nid, keys=True)]))

    eid = None
    for found in nearest_rtree_bbox(edge_tree, bbox):
        if found not in to_exclude:
            eid = found
            break
    
    check(eid != None, expect="Expect to find nearby edge.")

    distance = graph_distance_node_edge(G, nid, eid)

    # Use this distance to find all edge bounding boxes within that distance.
    bbox = graphnode_to_bbox(G, nid, padding=distance)
    eids = intersect_rtree_bbox(edge_tree, bbox)

    # Rerun against all edges and return lowest.
    distances = [(eid, graph_distance_node_edge(G, nid, eid)) for eid in eids if eid not in to_exclude]

    # Obtain lowest
    eid, distance = min(distances, key=lambda x: x[1])

    return eid


# Compute distance between node and edge.
@info()
def graph_distance_node_edge(G, nid, eid):
    point = graphnode_position(G, nid)
    curve = attributes.graphedge_curvature(G, eid)
    # Seek distance between point and curve.
    curvepoint = nearest_position_on_curve_to_point(curve, point)
    return norm(point - curvepoint)


# Compute total graph edge length.
def graph_length(G):
    return sum([curve_length(attrs["curvature"]) for eid, attrs in iterate_edges(G)])



#######################################
### Bounding boxes
#######################################


def graphedge_to_bbox(G, eid, padding=0):
    ps = graphedge_curvature(G, eid)
    miny = min(ps[:,0])
    maxy = max(ps[:,0])
    minx = min(ps[:,1])
    maxx = max(ps[:,1])
    bbox = array([(miny, minx), (maxy, maxx)])
    return pad_bounding_box(bbox, padding)

# Construct bounding box around node position.
def graphnode_to_bbox(G, nid, padding=0):
    position = y, x = G._node[nid]['y'], G._node[nid]['x'],
    bbox = bounding_box(array([position]))
    return pad_bounding_box(bbox, padding)

# Construct dictionary that links node id to a bounding box.
# Note: Padding has to be added manually afterwards if needed.
def graphnodes_to_bboxs(G, padding=0):
    return {nid: graphnode_to_bbox(G, nid, padding=padding) for nid in G.nodes()}

# Construct dictionary that links edge id to a bounding box.
# Note: Padding has to be added manually afterwards if needed.
def graphedges_to_bboxs(G, padding=0):
    return {eid: graphedge_to_bbox(G, eid, padding=padding) for eid, _ in iterate_edges(G)}


#######################################
### Rtree
#######################################

# Construct R-Tree on graph nodes.
def graphnodes_to_rtree(G):

    tree = rtree.index.Index()

    for nid, attrs in iterate_nodes(G):
        y, x = attrs['y'], attrs['x']
        tree.insert(nid, (y, x, y, x))

    return tree


# Construct R-Tree on graph edges.
def graphedges_to_rtree(G):

    tree = rtree.index.RtreeContainer()

    for eid, attrs in iterate_edges(G):
        curvature = attrs["curvature"]
        miny = min(curvature[:,0])
        maxy = max(curvature[:,0])
        minx = min(curvature[:,1])
        maxx = max(curvature[:,1])
        tree.insert(eid, (miny, minx, maxy, maxx))

    return tree