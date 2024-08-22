from dependencies import *
from network import *
from pcm import *

###################################
###  Curve by curve coverage
###################################

# Curve coverage of ps by qs.
# either return false or provide subcurve with step sequence
# TODO: Optimization to check on bounding boxes before doing the interpolation.
def curve_by_curve_coverage(ps, qs, lam=1):
    return is_partial_curve_undirected(to_curve(ps), to_curve(qs), lam)

# Check coverage of a curve by a curve-set.
def curve_by_curveset_coverage(ps, qss, lam=1):
    for qs in qss:
        if is_partial_curve_undirected(to_curve(ps), to_curve(qs), lam):
            return True, data
    return False, {}


###################################
###  Curve by network coverage
###################################

# Extract path from a network.
# 1. Construct minimal bounding box for area
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
def bounding_box(ps, padding=0):
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


def graphedges_to_rtree(G):
    idx = rtree.index.Index()
    maxv = max(G.nodes())+1
    if type(G) == nx.Graph:
        for (a, b) in G.edges():
            x1 = G.nodes[a]["x"]
            y1 = G.nodes[a]["y"]
            x2 = G.nodes[b]["x"]
            y2 = G.nodes[b]["y"]
            # Unique id 
            idx.insert(maxv*a+b, (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)), obj=(a,b))
    else:
        for (a, b, k) in G.edges(keys=True):
            x1 = G.nodes[a]["x"]
            y1 = G.nodes[a]["y"]
            x2 = G.nodes[b]["x"]
            y2 = G.nodes[b]["y"]
            idx.insert((a,b), (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)))
    return idx


# Compute coverage of curve by a network.
def coverage_curve_by_network(G, ps, lam=1):
    
    G = vectorize_graph(G) # Vectorize graph.
    G = deduplicate_vectorized_graph(G)
    idx = graphnodes_to_rtree(G) # Place graph nodes coordinates in accelerated data structure (R-Tree).
    bb = bounding_box(ps, padding=lam) # Construct lambda-padded bounding box.
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
    

# Check data valid
def check_curve_curve_data_validity(data):
    try:
        if not data["found"]:
            return False 
        ps = data["ps"]
        qs = data["qs"]
        steps = data["steps"]
        lam = data["lam"]

        # Steps are increasing (both ps and qs)
        stemp = np.array(data["steps"])[:,0]
        for i, j in zip(stemp, stemp[1:]):
            if not (i == j or i == j - 1):
                raise BaseException("Steps in ps are expected to be increasing.")
        stemp = np.array(data["steps"])[:,1]
        for i, j in zip(stemp, stemp[1:]):
            if not (i == j or i == j - 1):
                raise BaseException("Steps in qs are expected to be increasing.")

        # Within distance
        if not np.all(np.array( [np.linalg.norm(ps[ip] - qs[iq]) for (ip, iq) in steps] ) < lam):
            raise BaseException("Sequence of ps and qs are not within distance.")
        return True
    except Exception as e:
        for line in traceback.format_stack():
            print(line)
        breakpoint()
        return False



# Coverage of curve by another curve
def test_curve_curve_coverage_subcurve():
    # Create a set of points all 
    ps = np.array([[x,0] for x in range(10,20)])
    qs = np.array([[x,0.02] for x in range(0, 30)])

    found, data = curve_by_curve_coverage(ps, qs, lam=0.05)
    assert found == True and check_curve_curve_data_validity(data)

# Leave out one 
# BUG: This test should succeed: Discrete interpolation is converted into interval agnostic.
def test_curve_curve_coverage_leave_one_out():
    ps = np.array([[x,0] for x in range(10,20)])
    for i in range(10, 20): # Leave out index and test
        qslist = list(range(0,30))
        qslist = qslist[:i] + qslist[i+1:]
        qs = np.array([[x,0] for x in qslist])

        found, data = curve_by_curve_coverage(ps, qs, lam=0.05)
        assert found == True and check_curve_curve_data_validity(data)

# One to three points per point, thus subsequence
def test_curve_curve_coverage_three_per_point():
    ps = np.array([[x,0] for x in range(10,20)])
    qslist = list(range(0,30))
    qs = []
    for x in range(0,30):
        for i in range(1,random.randrange(2,5)):
            qs.append([x, 0.5 - random.random()])
    qs = np.array(qs)
    found, data = curve_by_curve_coverage(ps, qs, lam=0.51)
    assert found == True and check_curve_curve_data_validity(data)

# Generating ps on unit circle, while qs scattered with one point at center point.
def test_curve_all_points_within_range():
    # ps is all within unit distance circle
    ps = []
    for i in range(20):
        tau = 2 * math.pi * random.random()
        ps.append([math.cos(tau), math.sin(tau)])
    ps = 0.5 * np.array(ps)

    # just generate 20 points at random in space and make some value at 0,0 (so must have coverage).
    qs = []
    for i in range(30):
        qs.append([10*random.random(), 10*random.random()])
    qs[random.randrange(0,30)] = [0,0]
    qs = np.array(qs)

    found, data = curve_by_curve_coverage(ps, qs, lam=0.51)
    assert found == True and check_curve_curve_data_validity(data)

# Nodes occur at different intervals, causing mismatch even though practically identical.
# Requires line interpolation.
def test_curves_with_different_interpolation_frequencies():
    ps = 10 * np.array([[v,0] for v in range(30)])
    qs = np.array([[v,0] for v in range(300)])
    found, data = curve_by_curve_coverage(ps, qs, lam=1, eps=0.5)
    assert found and check_curve_curve_data_validity(data)

# Proof deviation measure succeeds.
def test_curve_deviation_by_epsilon():
    lam = 5 + random.random() * 5
    eps = 0.5
    dev = lambda_deviation(lam, eps)
    # Add one percentage, so should succeed.
    ps = random_curve(40) # Pick points at random.
    # qs = reinterpolate_curve(ps) # Reinterpolate curve at ar

# Reinterpolating curve with lambda and epsilon value should have max distance per node pair.
def test_curve_resampling_interval():
    lam = 1
    eps = 0.05
    ps = random_curve()
    qs = interpolate_curve(ps, lam=lam, eps=eps)
    ds = curve_point_intervals(qs)
    assert max(ds) <= lam*eps


testfuncts = [
    test_curve_curve_coverage_subcurve,
    test_curve_curve_coverage_leave_one_out,
    test_curve_curve_coverage_three_per_point,
    test_curve_all_points_within_range,
    test_curves_with_different_interpolation_frequencies
]



# Run tests
def run_tests():
    for func in testfuncts:
        print(func.__name__)
        func()

    

###################################
###  Tests: Curve by network coverage
###################################