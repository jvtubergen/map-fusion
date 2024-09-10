from dependencies import *
from network import *
from pcm import *

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
    assert type(G) == nx.MultiGraph
    edgetree = rtree.index.RtreeContainer()
    for uvk in G.edges(keys=True):
        curvature = edge_curvature(G, uvk)
        minx = min(curvature[:,0])
        maxx = max(curvature[:,0])
        miny = min(curvature[:,1])
        maxy = max(curvature[:,1])
        edgetree.insert(uvk, (minx, miny, maxx, maxy))
    return edgetree



# Optimization to check coverage of a curve by a network. 
# We only have to consider the edges in the subnetwork which are nearby the curve.
# The curves have no loops in it, so we only have to check for simple paths.
def coverage_curve_by_network(G, ps, lam):
    
    G = vectorize_graph(G) # Vectorize graph.
    G = deduplicate_vectorized_graph(G)
    idx = graphnodes_to_rtree(G) # Place graph nodes coordinates in accelerated data structure (R-Tree).
    bb = bounding_box(ps, lam) # Construct lambda-padded bounding box.
    nodes = list(idx.intersection((bb[0][0], bb[0][1], bb[1][0], bb[1][1]))) # Extract nodes within bounding box.
    H = G.subgraph(nodes) # Extract subgraph with nodes.

    # Compute relevant paths.
    # Convert paths into curves.
    # Check coverage of curve by curve set.
    # On coverage, return True with path that covers the curve.
    # Otherwise return False without further data.
    breakpoint()


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
    
