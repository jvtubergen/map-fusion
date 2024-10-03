from dependencies import *
from network import *
from pcm import *
from maps import *
from curve import *
from rendering import *

###################################
###  Curve by curve coverage
###################################

# Curve coverage of ps by qs.
# either return false or provide subcurve with step sequence
# TODO: Optimization to check on bounding boxes before doing the interpolation.
def curve_by_curve_coverage(ps, qs, lam):
    return is_partial_curve_undirected(to_curve(ps), to_curve(qs), lam)


# Check coverage of a curve by a curve-set.
def curve_by_curveset_coverage(ps, qss, lam):
    for qs in qss:
        if is_partial_curve_undirected(to_curve(ps), to_curve(qs), lam):
            return True
    return False


###################################
###  Curve by network coverage
###################################


# Extract path from a network.
# 1. Construct minimal bounding box for area.
# 2. Obtain vectorized graph.
# 3. Extract nodes within area.
# 4. Construct subnetwork.
# 5. Generate all _simple edge_ paths within subnetwork.
# 6. Check for curve by curve set coverage.

# Generate a random curve.
def random_curve(length = 100, a = np.array([-10,-10]), b = np.array([10,10])):
    ps = np.random.random((length, 2))
    return a + (b - a) * ps
        
        
# Length of curve.
def curve_length(ps):
    length = 0
    for p1, p2 in zip(ps, ps[1:]):
        length += np.linalg.norm(p1 - p2)
    return length


# Extract bounding box on a curve. Use padding to lambda pad.
def bounding_box(ps, padding):
    padding = np.array([padding, padding])
    lower = [np.min(ps[:,0]), np.min(ps[:,1])]
    higher = [np.max(ps[:,0]), np.max(ps[:,1])]
    return np.array([lower - padding, higher + padding])


# Pad a bounding box.
def pad_bounding_box(bb, padding):
    padding = np.array([padding, padding])
    return np.array([bb[0] - padding, bb[1] + padding])


# Construct R-Tree on graph nodes.
def graphnodes_to_rtree(G):
    idx = rtree.index.Index()
    for node, data in G.nodes(data = True):
        x, y = data['x'], data['y']
        idx.insert(node, (x, y, x, y))
    return idx


# Construct R-Tree on graph edges.
def graphedges_to_rtree(G):

    edgetree = rtree.index.RtreeContainer()

    if type(G) == nx.MultiGraph:
        elements = [((u, v, k), edge_curvature(G, u, v, k=k)) for (u, v, k) in G.edges(keys=True)]
    if type(G) == nx.Graph:
        elements = [((u, v), edge_curvature(G, u, v)) for (u, v) in G.edges()]

    for (eid, curvature) in elements:
            minx = min(curvature[:,0])
            maxx = max(curvature[:,0])
            miny = min(curvature[:,1])
            maxy = max(curvature[:,1])
            edgetree.insert(eid, (minx, miny, maxx, maxy))
    
    return edgetree


# Construct dictionary that links edge id to a bounding box.
def graphedges_to_bboxs(G):

    bboxs = {}

    if type(G) == nx.MultiGraph:
        elements = [((u, v, k), edge_curvature(G, u, v, k=k)) for (u, v, k) in G.edges(keys=True)]
    if type(G) == nx.Graph:
        elements = [((u, v), edge_curvature(G, u, v)) for (u, v) in G.edges()]
    
    for (eid, curvature) in elements:
        minx = min(curvature[:,0])
        maxx = max(curvature[:,0])
        miny = min(curvature[:,1])
        maxy = max(curvature[:,1])
        bbox = array([(minx, miny), (maxx, maxy)])
        bboxs[eid] = bbox

    return bboxs


# Compute per (simplified) edge of S (Source graph) the coverage threshold in order to be matched by T (Target graph).
# Assume S and T are a MultiGraph, simplified, and in appropriate coordinate system to measure distance differences in meter.
# todo optimize: Have a minimal distance per edge to start evaluation (e.g. some thresholds start at 700 meters).
def edge_wise_coverage_threshold(S, T, max_threshold=None):
    
    graph      = graph_to_rust_graph(T)
    edgebboxs  = graphedges_to_bboxs(S) # Have a bounding box per edge so we can quickly pad for intersection test against T.
    edgetree   = graphedges_to_rtree(T) # Place graph edges by coordinates in accelerated data structure (R-Tree).
    edges_todo = S.edges()

    nodedict = extract_nodes_dict(S)

    edge_results = {}

    # Increment threshold and seek nearby path till all edges have found a threshold (or max threshold is reached).
    lam = 1 # Start with a threshold of 1 meter.
    while len(edges_todo) > 0 and (max_threshold == None or lam <= max_threshold):

        print(f"Lambda: {lam}. Edges: {len(edges_todo)}")

        # Iterate every edge left to do.
        for uv in edges_todo:

            u, v = uv
            ps = edge_curvature(S, u, v)
            curve = to_curve(ps)
            result = partial_curve_graph(graph, curve, lam)
            if result != None:
                path = result
                edges_todo = edges_todo - set([uv]) # Remove edge from edge set.
                edge_results[uv] = {
                    "threshold": lam,
                    "path": path,
                }
        lam += 1 # Increment lambda

    return edge_results


###################################
###  Tests: Curve by curve coverage
###################################

# Test:
# * Generate a curve randomly
# * Per point generate three in range
# * Pick one of those and represent as curve
# * Pool unused nodes with some more randomly generated curves
# * Add some arbitrary other nodes of these and add to curve
# Verify:
# * Expect to find some subcurve
# * Generated subcurve within distance lambda
    
