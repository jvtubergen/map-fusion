from external import * 

### R-Tree

# Construct R-Tree on graph nodes.
def graphnodes_to_rtree(G):

    tree = rtree.index.Index()

    for nid, attrs in G.nodes(data = True):
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


### Bounding boxes

# Construct dictionary that links edge id to a bounding box.
# Note: Padding has to be added manually afterwards if needed.
def graphedges_to_bboxs(G):

    bboxs = {}
    
    for eid, attrs in iterate_edges(G):
        ps = attrs["curvature"]
        miny = min(ps[:,0])
        maxy = max(ps[:,0])
        minx = min(ps[:,1])
        maxx = max(ps[:,1])
        bbox = array([(miny, minx), (maxy, maxx)])
        bboxs[eid] = bbox

    return bboxs

# Construct dictionary that links node id to a bounding box.
# Note: Padding has to be added manually afterwards if needed.
def graphnodes_to_bboxs(G):

    bboxs = {}
    
    for nid, attrs in G.nodes(data=True):
        position = [attrs["y"], attrs["x"]]
        bboxs[nid] = bounding_box(array([position]))

    return bboxs


# Extract bounding box on a curve. Use padding to lambda pad.
def bounding_box(ps, padding=0):
    padding = array([padding, padding])
    lower = [np.min(ps[:,0]), np.min(ps[:,1])]
    higher = [np.max(ps[:,0]), np.max(ps[:,1])]
    return array([lower - padding, higher + padding])

# Pad a bounding box.
def pad_bounding_box(bb, padding):
    padding = array([padding, padding])
    return array([bb[0] - padding, bb[1] + padding])

intersect_rtree_bbox = lambda tree, bbox: list(tree.intersection((bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])))

## Curves

# Generate a random curve.
def random_curve(length = 100, a = np.array([-10,-10]), b = np.array([10,10])):
    ps = np.random.random((length, 2))
    return a + (b - a) * ps


# Length of curve, return accumulated length of piecewise linear segments.
def curve_length(ps):
    return sum([norm(p1 - p2) for p1, p2 in zip(ps, ps[1:])])


# Cut curve at specified intervals.
def curve_cut_intervals(ps, intervals):

    assert intervals[0]  > 0.0001
    assert intervals[-1] < 1 - 0.0001

    n = len(ps)
    m = len(intervals)

    lengths    = np.linalg.norm(ps[1:] - ps[:-1], axis=1)
    weights    = lengths / np.sum(lengths)
    cumulative = np.hstack(([0], np.cumsum(weights)))

    # Iterate points and inject curve when necessary.
    i = 1 # Current curve point we are at.
    j = 0 # Current interval we are cutting for.
    qss = [] # Resulting collection of subcurves.
    qs = [ps[0]] # Current subcurve we are constructing.

    # Invariants:
    while i < n: # Current point (and end of current line segment).

        u, v = i - 1, i     # Curve point indices current line segment.
        p, q = ps[u], ps[v] # Curve points of current line segment.

        # If we have reached the final interval.
        if j >= m:
            # Then append curve points until we are done.
            qs.append(q)
            i += 1 # Move to next line segment.

        # If interval falls exactly on end of current line segment.
        elif abs(cumulative[i] - intervals[j]) < 0.0001:
            # Then construct subcurve with current line segment included.
            qs.append(q)
            qss.append(qs)
            qs = [q]
            i += 1 # Move to next line segment.
            j += 1 # Move to next interval.
        
        # If this interval stops somewhere within this line segment.
        elif cumulative[i] > intervals[j]:
            # Then construct subcurve to cutpoint on current line segment.
            interval = intervals[j]
            weight = weights[i - 1]
            percentage = (interval - cumulative[i - 1]) / weight
            cutpoint = p * (1 - percentage) + q * percentage
            qs.append(cutpoint)
            qss.append(qs)
            qs = [cutpoint]
            j += 1 # Move to next interval.

        # If the interval contains this entire line segment.
        else:
            qs.append(q) 
            i += 1 # Move to next line segment.
    
    qss.append(qs)

    # Convert each subcurve into a numpy array.
    qss = [array(qs) for qs in qss]

    return qss


# Cut a curve at a percentage (interval of [0, 1]).
curve_cut = lambda ps, percentage: curve_cut_intervals(ps, [percentage])

# Cut curve in half.
curve_cut_in_half = lambda ps: curve_cut(ps, 0.5)

# Find cutpoints, cut curve into pieces, store curvature for each cut curve segment.
def curve_cut_pieces(ps, amount=10):

    # Compute (uniform) intervals to cut at given the number of pieces to cut.
    step_size = 1 / amount
    intervals = [i * step_size for i in range(1, amount)]
    return curve_cut_intervals(ps, intervals)


# Cut curve into subcurves each with less than max distance length, thus returning muliple subcurves.
def curve_cut_max_distance(ps, max_distance=10):
    assert len(ps) >= 2
    step_lengths = array([norm(p1 - p2) for p1, p2 in zip(ps, ps[1:])])
    total_length = sum(step_lengths)
    assert total_length > 0 # Expect non-zero length.
    pieces = ceil(total_length / max_distance)
    return curve_cut_pieces(ps, amount=pieces)


# Insert vertices in curve in such that that maximal distance between vertices is lower than `max_distance`.
def curve_insert_vertices_max_distance(ps, max_distance=10):

    # Compute number of (uniform length) intervals we need to make the curve respect the maximal distance threshold.
    assert len(ps) >= 2
    steps = array([norm(p1 - p2) for p1, p2 in zip(ps, ps[1:])])
    length = sum(steps)
    assert length > 0 # Expect non-zero length.

    qs = []
    for i in range(len(steps)):
        qs.append(ps[i])
        step_length = norm(ps[i+1] - ps[i])
        if step_length > max_distance + 0.0001: # Inject nodes.
            amount, _ = divmod(step_length, max_distance + 0.0001)
            amount = int(amount)
            for j in range(amount):
                percentage = ((j + 1) / (amount + 1))
                p = percentage * ps[i + 1] + (1 - percentage) * ps[i]
                qs.append(p)

    qs.append(ps[-1])

    return array(qs)

#### Curve tests
def test_curve_cut_intervals():
    ps = random_curve()
    qss = curve_cut_intervals(ps, [0.2, 0.53, 0.99])
    assert abs(curve_length(ps) - sum([curve_length(qs) for qs in qss])) < 0.0001


# Test curve cutting into pieces (basic test).
def test_curve_cut_pieces():
    curve = array([(0., i) for i in range(11)])
    qss = curve_cut_pieces(curve, amount=10)
    assert len(qss) == 10
    assert all([abs(curve_length(qs) - 1) < 0.0001 for qs in qss])

    qss = curve_cut_pieces(curve, amount=5)
    assert len(qss) == 5
    assert all([abs(curve_length(qs) - 2) < 0.0001 for qs in qss])

    qss = curve_cut_pieces(curve, amount=2)
    assert len(qss) == 2
    assert all([abs(curve_length(qs) - 5) < 0.0001 for qs in qss])

    qss = curve_cut_pieces(curve, amount=3)
    assert len(qss) == 3
    assert all([abs(curve_length(qs) - 3.333333) < 0.0001 for qs in qss])


# Test (basic test).
def test_curve_insert_vertices_max_distance():

    # Within range.
    curve = array([(0., i) for i in range(11)])
    qs = curve_insert_vertices_max_distance(curve, max_distance=10)
    assert len(qs) == len(curve)
    assert (curve == qs).all()
    assert abs(curve_length(curve) - curve_length(qs)) < 0.0001

    # Double range.
    curve = array([(0., 2*i) for i in range(11)])
    qs = curve_insert_vertices_max_distance(curve, max_distance=1)
    assert len(qs) - 1 == 2 * (len(curve) - 1)
    assert abs(curve_length(curve) - curve_length(qs)) < 0.0001
    assert np.max(qs[1:] - qs[:-1]) < 1 + 0.0001

    # Incomplete.
    curve = array([(0., 10*i+j) for i in range(1,4) for j in range(1,4)])
    qs = curve_insert_vertices_max_distance(curve, max_distance=1)
    assert abs(curve_length(curve) - curve_length(qs)) < 0.0001
    assert np.max(qs[1:] - qs[:-1]) < 1 + 0.0001


def test_curve_cut_max_distance():
    # Within range.
    curve = array([(0., i) for i in range(11)])
    qs = curve_cut_max_distance(curve, max_distance=10)
    assert len(qs) == 1 # The entire curve is within 10 meter, so no need to cut.
    assert (qs[0] == curve).all() # Expect the curve to be the first entry.

    # Double range.
    curve = array([(0., 2*i) for i in range(11)])
    qs = curve_cut_max_distance(curve, max_distance=1)
    assert len(qs) == 20 # Expect 20 subcurves.
    assert abs(sum([curve_length(subcurve) for subcurve in qs]) - curve_length(curve)) < 0.0001 # Expect total length of subcurves is equal to original curve.
    qs = array(qs) # We can do this here since we expect every subcurve to consist of exactly two coordinates.
    assert np.max(array([curve_length(qs[i]) for i in range(len(qs))])) < 1 + 0.0001 # Expect each subcurve to be a length of 1.

    # Incomplete.
    curve = array([(0., 10*i+j) for i in range(1,4) for j in range(1,4)])
    qs = curve_cut_max_distance(curve, max_distance=1.4)
    assert abs(sum([curve_length(subcurve) for subcurve in qs]) - curve_length(curve)) < 0.0001 # Expect total length of subcurves is equal to original curve.
    assert np.max(array([curve_length(qs[i]) for i in range(len(qs))])) < 1.4 + 0.0001 # Expect each subcurve to be a length of 1.4.


## Linestring and curves.

# Convert an array into a LineString consisting of Points.
to_linestring   = lambda curvature: LineString([Point(x, y) for y, x in curvature]) # Coordinates are flipped.

# Convert a LineString into an array.
from_linestring = lambda geometry : array([(y, x) for x, y in geometry.coords]) # Coordinates are flipped.


### Partial curve matching logic

# Convert 2d numpy array into a list of Vectors used by the partial curve matching algorithm.
def curve_to_vector_list(ps):
    result = []
    for [y, x] in ps:
        result.append(Vector(y, x))
    return result


# Convert a nx.T2 into a graph structure used by the partial curve matching algorithm.
def graph_to_rust_graph(G):

    assert type(G) == nx.Graph

    # Extract node data as Vectors from the graph.
    def extract_nodes_list(G):
        l = []
        for nid, data in G.nodes(data = True):
            l.append((nid, Vector(data['y'], data['x'])))
        return l

    # Extract vertices as Vec<(NID, Vector)>.
    vertices = extract_nodes_list(G)
    # Extract edges as Vec<(NID, NID)>.
    edges = G.edges()
    return make_graph(vertices, edges)

# Compute partial curve matching between curve ps and some subcurve of qs within eps distance threshold.
# If convert is true automatically convert input curves into vector lists.
def is_partial_curve_undirected(ps, qs, eps, convert=False):
    if convert:
        ps = curve_to_vector_list(ps)
        qs = curve_to_vector_list(qs)
    assert type(ps[0]) == Vector
    assert type(qs[0]) == Vector
    try:
        return partial_curve(ps, qs, eps) != None or partial_curve(ps[::-1], qs, eps)
    except Exception as e:
        print("Failed partial curve: ", e)
        print("Parameters:")
        print("  ps : ", ps)
        print("  qs : ", qs)
        print("  eps: ", eps)


### Graph related

# Abstract function to iterate edge attributes (works for both simplified and vectorized graphs).
# Example:
# ``` 
#   for attrs in iterate_edge_attributes(G):
#       attrs["color"] = (random.random(), random.random(), random.random(), 1)
# ```
def iterate_edge_attributes(G):
    if G.graph["simplified"]:
        for u, v, k, attrs in G.edges(data=True, keys=True):
            yield attrs
    else:
        for u, v, attrs in G.edges(data=True):
            yield attrs

# Iterate all edge identifiers alongside their attributes. Iterated element attributes are overwritable.
def iterate_edges(G):
    if G.graph["simplified"]:
        for u, v, k, attrs in G.edges(data=True, keys=True):
            yield (u, v, k), attrs
    else:
        for u, v, attrs in G.edges(data=True):
            yield (u, v), attrs

def iterate_nodes(G):
    for nid, attrs in G.nodes(data=True):
        yield nid, attrs

# Get specific edge from graph. Using built-in `eid` filter hopefully improves performance (in comparison to list filtering).
def get_edge(G, eid):
    if G.graph["simplified"]:
        u, v, k = eid
        return G.get_edge_data(u, v, k)
    else:
        u, v = eid
        return G.get_edge_data(u, v)

# Iterate graph edges as `(eid, attrs)` pair. Helps generalizing simplified/vectorized graph logic.
def graph_edges(G):
    if not G.graph["simplified"]:
        return G.edges(data=True)
    else:
        return G.edges(data=True, keys=True)


## Curve-point related logic.

# Rotating (x, y)
def rotate(a):
    return array([
        [cos(a), sin(-a)],
        [sin(a), cos( a)],
    ])


# Compute the distance (and the interval along the line-segment) between a line-segment `p-q` and a point `a`.
# * Assume as arrays.
# * Linesegment p, q
# * Point a
# * All input points are `(y, x)`, but we compute on `(x, y)`.
def distance_point_to_linesegment(p, q, a):

    # Flip coordinates around from `(y, x)` to `(x, y)`.
    flip = lambda _x: array([_x[1], _x[0]])

    # Act on x, y
    p = flip(p)
    q = flip(q)
    a = flip(a)

    # Translate p to the origin.
    q = q - p
    a = a - p

    # Rotate a and q around p (and thereby the origin) in such that q lies horizontally to the right.
    rotation = -atan2(q[1], q[0])  # Find rotation (take orientation of q).
    q = rotate(rotation).dot(q)
    a = rotate(rotation).dot(a)
    assert abs(q[1]) < 0.0001

    # Decide on orthogonality.
    if a[0] < 0: # Point is on the left side of the origin.
        return norm(a), 0 # Distance from point to startpoint linesegment, start of interval.
    elif a[0] > q[0]: # Point is to the right of the endpoint linesegment.
        return norm(q - a), 1  # Distance from point to end linesegment, end of interval.
    else: # Distance from point y coordinate to x axis (where y equals 0), relative x coordinate to end of line segment.
        return a[1], a[0] / q[0]


# Compute position on curve which lies nearest to a point.
def nearest_position_and_interval_on_curve_to_point(ps, point): 
    
    items = []
    for i, linesegment in enumerate(zip(ps,ps[1:])):
        p, q = linesegment
        distance, interval = distance_point_to_linesegment(p, q, point)
        items.append((i, distance, interval))

    # Seek lowest distance.
    i, distance, interval = min(items, key=lambda x: x[1])
    position = ps[i] * (1 - interval) + ps[i + 1] * interval

    # Compute interval along curve.
    lengths    = norm(ps[1:] - ps[:-1], axis=1)
    weights    = lengths / np.sum(lengths)
    cumulative = np.hstack(([0], np.cumsum(weights)))
    actual_interval = cumulative[i] + interval * weights[i]

    return position, actual_interval

# Wrapper function to only obtain nearest position on curve to point.
nearest_position_on_curve_to_point = lambda curve, point: nearest_position_and_interval_on_curve_to_point(curve, point)[0]

# Wrapper function to only obtain nearest curve interval on curve to point.
nearest_interval_on_curve_to_point = lambda curve, point: nearest_position_and_interval_on_curve_to_point(curve, point)[1]


## Arbitrary

# Unzip a list of pairs into a pair of lists.
def unzip(data):
    left, right = [], []
    for l, r in data:
        left.append(l)
        right.append(r)
    return left, right