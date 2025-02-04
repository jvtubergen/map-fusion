from external import * 
from graph_node_extraction import *

#######################################
### Printing stuff with decorators.
#######################################

# Tracks current context (function stack).
current_context = []

# Track times.
times = []

# Decorator function to set context for printing debugging information.
# Optionally print context on function launch.
def info(print_context=True, timer=False):

    # The decorator to return.
    def decorator(func):

        def wrapper(*args, **kwargs):

            current_context.append(func.__name__)

            context = " - ".join(current_context)

            if print_context:
                print(context)

            if timer:
                start_time = time()

            result = func(*args, **kwargs)

            if timer:
                end_time = time()
                execution_time = end_time - start_time
                times.append([context, execution_time])

            current_context.pop()

            return result

        return wrapper

    return decorator


# `log` is the same as `print`  with function context prepended.
def logger(*args):
    print(f"{" - ".join(current_context)}:", *args)



### R-Tree

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


### Bounding boxes

def graphedge_curvature(G, eid):
    attrs = get_edge_attributes(G, eid)
    ps = attrs["curvature"]
    return ps

def graphedge_to_bbox(G, eid, padding=0):
    ps = graphedge_curvature(G, eid)
    miny = min(ps[:,0])
    maxy = max(ps[:,0])
    minx = min(ps[:,1])
    maxx = max(ps[:,1])
    bbox = array([(miny, minx), (maxy, maxx)])
    return pad_bounding_box(bbox, padding)

# Construct dictionary that links edge id to a bounding box.
# Note: Padding has to be added manually afterwards if needed.
def graphedges_to_bboxs(G, padding=0):
    return {eid: graphedge_to_bbox(G, eid, padding=padding) for eid, _ in iterate_edges(G)}

def graphnode_position(G, nid):
    position = y, x = G._node[nid]['y'], G._node[nid]['x'],
    return array(position)

# Construct bounding box around node position.
def graphnode_to_bbox(G, nid, padding=0):
    position = y, x = G._node[nid]['y'], G._node[nid]['x'],
    bbox = bounding_box(array([position]))
    return pad_bounding_box(bbox, padding)

# Construct dictionary that links node id to a bounding box.
# Note: Padding has to be added manually afterwards if needed.
def graphnodes_to_bboxs(G, padding=0):
    return {nid: graphnode_to_bbox(G, nid, padding=padding) for nid in G.nodes()}

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

flatten_bbox         = lambda bbox: (bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])
del_rtree_bbox       = lambda tree, value: tree.delete(value, tree.bounds)
add_rtree_bbox       = lambda tree, bbox, value: tree.insert(value, flatten_bbox(bbox))
intersect_rtree_bbox = lambda tree, bbox: list(tree.intersection(flatten_bbox(bbox)))
nearest_rtree_bbox   = lambda tree, bbox: list(tree.nearest(flatten_bbox(bbox), num_results=len(tree)))

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

    # No need to cut if no intervals provided.
    if len(intervals) == 0:
        return [ps]

    # Ensure minimal interval difference of at least 0.0001.
    if len(intervals) > 1:
        arr = array(intervals)
        interval_steps = arr[1:] - arr[:-1]
        intervals_to_drop = [i for i, x in enumerate(interval_steps) if x < 0.00001]

        # Pick elements except those of intervals to drop.
        intervals = [x for i, x in enumerate(intervals) if i not in intervals_to_drop]

    # Perform check on interval length, because we might have decreased the interval set in the previous step.
    if len(intervals) > 1:
        r = array(intervals)
        check(min(r[1:] - r[:-1]) > 0.00001, expect="Expect minimal interval difference of 0.0001.")
    
    # Optionally drop injecting nodes if they are too close at the edge endpoints.
    if intervals[0] < 0.00001:
        intervals = intervals[1:]

    if intervals[-1] < 1 - 0.00001:
        intervals = intervals[:1]

    check(intervals[0] > 0.00001     , expect="Expect first interval to be greater than 0.0001.")
    check(intervals[-1] < 1 - 0.00001, expect="Expect final interval to be less than 0.9999.")

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

    if amount == 1:
        return [ps]

    # Compute (uniform) intervals to cut at given the number of pieces to cut.
    step_size = 1 / amount
    intervals = [i * step_size for i in range(1, amount)]
    return curve_cut_intervals(ps, intervals)


# Cut curve into subcurves each with less than max distance length, thus returning muliple subcurves.
def curve_cut_max_distance(ps, max_distance=10):
    assert len(ps) >= 2

    length = curve_length(ps)

    if max_distance >= length:
        return [ps]

    pieces = ceil(length / max_distance)

    return curve_cut_pieces(ps, amount=pieces)


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
def test_curve_cut_max_distance():

    # Within range.
    curve = array([(0., i) for i in range(11)])
    qs = curve_cut_max_distance(curve, max_distance=10)
    assert len(qs) == len(curve)
    assert (curve == qs).all()
    assert abs(curve_length(curve) - curve_length(qs)) < 0.0001

    # Double range.
    curve = array([(0., 2*i) for i in range(11)])
    qs = curve_cut_max_distance(curve, max_distance=1)
    assert len(qs) - 1 == 2 * (len(curve) - 1)
    assert abs(curve_length(curve) - curve_length(qs)) < 0.0001
    assert np.max(qs[1:] - qs[:-1]) < 1 + 0.0001

    # Incomplete.
    curve = array([(0., 10*i+j) for i in range(1,4) for j in range(1,4)])
    qs = curve_cut_max_distance(curve, max_distance=1)
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
    eids = [eid for eid, _ in iterate_edges(G)]
    return make_graph(vertices, eids)

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

# Enum to differentiate between node or edge-related task (used by abstract functions).
class GraphEntity(Enum):
    Edges = 0
    Nodes = 1

get_eids = lambda G: [eid for eid, _ in iterate_edges(G)]
get_nids = lambda G: list(G.nodes())

# Iterate all edge identifiers alongside their attributes. Iterated element attributes are overwritable.
def iterate_edges(G):
    if G.graph["simplified"]:
        for u, v, k, attrs in G.edges(data=True, keys=True):
            if u <= v: 
                yield (u, v, k), attrs
            else:
                yield (v, u, k), attrs
    else:
        for u, v, attrs in G.edges(data=True):
            if u <= v: 
                yield (u, v), attrs
            else:
                yield (v, u), attrs

def iterate_nodes(G):
    for nid, attrs in G.nodes(data=True):
        yield nid, attrs


# Format edge identifier to graph type.
def format_eid(G, eid):

    u, v = sorted(eid[:2])

    # Obtain `k`.
    k = 0
    if len(eid) == 3:
        k = eid[2]

    # Pair or triplet.
    eid = u, v
    if G.graph["simplified"]:
        eid = u, v, k 
    
    return eid


# Get specific edge from graph. Using built-in `eid` filter hopefully improves performance (in comparison to list filtering).
def get_edge_attributes(G, eid):
    # eid = format_eid(G, eid)
    return G.get_edge_data(*eid)

# Set edge attribute.
def set_edge_attributes(G, eid, attrs):
    # eid = format_eid(G, eid)
    nx.set_edge_attributes(G, {eid: attrs})


def get_node_attributes(G, nid):
    return G.get_node_data(nid)

def set_node_attributes(G, nid, attrs):
    nx.set_node_attributes(G, {nid: attrs})


# Obtain connected edge identifiers to the provided node identifier.
# Note: `(u, v)` or `(u, v, k)` always have lowest nid first (thus `u <= v`).
def get_connected_eids(G, nid):
    if G.graph["simplified"]:
        return [(u, v, k) if u <= v else (v, u, k) for (u, v, k) in list(G.edges(nid, keys=True))]
    else:
        return [(u, v) if u <= v else (v, u) for (u, v) in list(G.edges(nid))]

# Annotate nodes by appending new attributes (optionally to a subselection of node identifiers.
def annotate_nodes(G, new_attrs, nids=None):
    if nids != None:
        check(type(nids[0]) == type(1) , expect="Expect to receive node identifiers (it probably has received edge identifiers).")
        nx.set_node_attributes(G, {nid: {**attrs, **new_attrs} for nid, attrs in iterate_nodes(G) if nid in nids}) 
    else:
        nx.set_node_attributes(G, {nid: {**attrs, **new_attrs} for nid, attrs in iterate_nodes(G)}) 

# Annotate edges by appending new attributes (optionally to a subselection of edge identifiers.
def annotate_edges(G, new_attrs, eids=None):
    if eids != None:
        check(len(eids[0]) == 2 or len(eids[0]) == 3, expect="Expect to receive edge identifiers (it probably has received node identifiers).")
        nx.set_edge_attributes(G, {eid: {**attrs, **new_attrs} for eid, attrs in iterate_edges(G) if eid in eids}) 
    else:
        nx.set_edge_attributes(G, {eid: {**attrs, **new_attrs} for eid, attrs in iterate_edges(G)}) 


# Abstract function to filter node/edge identifiers by their attributes (can be both a subset of attributes or a filter function on attributes).
def abstract_filter_by_attribute(entity, G, filter_attributes=None, filter_func=None):

    check(type(entity) == GraphEntity)
    check(filter_func != None or filter_attributes != None, expect="Expect to filter either by attributes dictionary or a filter function.")

    logger("entity")
    match entity:
        case GraphEntity.Edges:
            logger("edges")
            iterator = iterate_edges
        case GraphEntity.Nodes:
            logger("nodes")
            iterator = iterate_nodes

    if filter_func != None:

        return [identifier for identifier, attrs in iterator(G) if filter_func(attrs)]
    
    if filter_attributes != None:

        filtered_identifiers = []
        for identifier, attrs in iterator(G):

            found = True
            for filter_attr in filter_attributes.keys():
                if filter_attr not in attrs or attrs[filter_attr] != filter_attributes[filter_attr]:
                    found = False

            if found:
                filtered_identifiers.append(identifier)
    
        return filtered_identifiers

# Filter out nodes either by a collection of attributes or a specific filtering function.
filter_nids_by_attribute = lambda *params, **named: abstract_filter_by_attribute(GraphEntity.Nodes, *params, **named)

# Filter out edges either by a collection of attributes or a specific filtering function.
filter_eids_by_attribute = lambda *params, **named: abstract_filter_by_attribute(GraphEntity.Edges, *params, **named)

# Compute total graph edge length.
def graph_length(G):
    return sum([curve_length(attrs["curvature"]) for eid, attrs in iterate_edges(G)])


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

def test_nearest_position_and_interval_on_curve_to_point():

    curve = array([[i, 0.0] for i in range(11)])
    point = array([5, 5])
    check(abs(nearest_interval_on_curve_to_point(curve, point) - 0.5) < 0.001)
    check(abs(norm(nearest_position_on_curve_to_point(curve, point) - point) - 5) < 0.001)


### Graph distance

# Compute distance between node and edge.
@info()
def graph_distance_node_edge(G, nid, eid):
    point = graphnode_position(G, nid)
    curve = graphedge_curvature(G, eid)
    # Seek distance between point and curve.
    curvepoint = nearest_position_on_curve_to_point(curve, point)
    return norm(point - curvepoint)

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


## Arbitrary

# Unzip a list of pairs into a pair of lists.
def unzip(data):
    left, right = [], []
    for l, r in data:
        left.append(l)
        right.append(r)
    return left, right


#######################################
### Sanity check functionality
#######################################

# Perform a few sanity checks on the graph to prevent computation errors down the line.
def graph_sanity_check(G):
    print("Check graph.")

    # Simplification.
    if "simplified" not in G.graph:
        raise Exception("Expect 'simplified' dictionary key in graph.")

    # If simplified, then multigraph.
    if G.graph["simplified"] and type(G) != type(nx.MultiGraph()):
        raise Exception("Expect simplified graph to be an undirected multi-graph.") 
    if not G.graph["simplified"] and type(G) != type(nx.Graph()):
        raise Exception("Expect vectorized graph to be a undirected single-graph.") 

    # Coordinates.    
    if "coordinates" not in G.graph:
        raise Exception("Expect 'coordinates' dictionary key in graph.")
    if G.graph["coordinates"] != "utm" and G.graph["coordinates"] != "latlon":
        raise Exception("Expect 'coordinates' dictionary value to be either 'utm' or 'latlon'.")

    # Nodes.
    sanity_check_node_positions(G)
    nodes = extract_node_positions_dictionary(G)
    if G.graph["coordinates"] == "utm":
        if np.min(nodes) < 100: 
            print(nodes)
            raise Exception("Expect graph in utm coordinate system. Big chance some node is in latlon coordinate system.")
    if G.graph["coordinates"] == "latlon":
        if np.min(nodes) > 100: 
            print(nodes)
            raise Exception("Expect graph in latlon coordinate system. Big chance some node is in utm coordinate system.")

    # Node (x,y coordinates) flipping.
    coord0 = np.min(nodes, axis=0)
    coord1 = np.max(nodes, axis=0)
    diffa = np.max(nodes[:,0]) - np.min(nodes[:,0])
    diffb = np.max(nodes[:,1]) - np.min(nodes[:,1])
    # print(diffa, diffb)
    diffc = np.max(nodes[:,0]) - np.min(nodes[:,1])
    diffd = np.max(nodes[:,1]) - np.min(nodes[:,0])
    # print(diffc, diffd)

    if G.graph["coordinates"] == "latlon" and (abs(diffa) > 1 or abs(diffb) > 1):
        print(nodes)
        print(abs(diffa), abs(diffb))
        raise Exception("Expect node y, x coordinate consistency.") 
    if G.graph["coordinates"] == "utm" and (abs(diffa) > 100000 or abs(diffb) > 100000):
        print(nodes)
        print(abs(diffa), abs(diffb))
        raise Exception("Expect node y, x coordinate consistency.") 
    
    # Edges.
    nodes = extract_node_positions_dictionary(G)
    if G.graph["simplified"]: 
        for eid, attrs in iterate_edges(G):
            a, b, k = eid
            ps = edge_curvature(G, a, b, k)
            if G.graph["coordinates"] == "latlon": # Convert to utm for computing in meters.
                ps = array([latlon_to_coord(latlon) for latlon in ps])
            if curve_length(ps) > 1000: # Expect reasonable curvature length.
                raise Exception("Expect edge length less than 1000 meters. Probably some y, x coordinate in edge curvature got flipped.")
            # Expect start and endpoint of edge curvature match the node position.
            ps = edge_curvature(G, a, b, k) # Expect startpoint matches curvature.
            try:
                if (not np.all(ps[0] == nodes[a])) and (not np.all(ps[-1] != nodes[b])):
                    raise Exception("Expect curvature have same directionality as edge start and end edge.")
            except Exception as e:
                print(traceback.format_exc())
                print(e)
                breakpoint()
            
            assert "geometry" in attrs
            assert "curvature" in attrs


# Sanity check that all curvature annotations are numpy array.
def sanity_check_curvature_type(G):
    for eid, attrs in iterate_edges(G):
        check(type(attrs["curvature"]) == type(array([])))


# Sanity check all edges have non-zero edge length.
def sanity_check_edge_length(G):
    for eid, attrs in iterate_edges(G):
        check(attrs["length"] > 0)


# Sanity check nodes have unique position.
def sanity_check_node_positions(G, eps=0.0001):

    assert G.graph["coordinates"] == "utm" # Act only on UTM for epsilon to make sense.

    positions = extract_node_positions_dictionary(G)
    tree = graphnodes_to_rtree(G)
    bboxs = graphnodes_to_bboxs(G)

    for nid in G.nodes():

        # Find nearby nodes.
        bbox = pad_bounding_box(bboxs[nid], eps)
        nids = intersect_rtree_bbox(tree, bbox)
        assert len(nids) == 1 # Expect to only intersect with itself.


# Sanity check graph curvature starts/end at node positions _and_ in the correct direction (starting at `u` and ending at `v` with `u <= v`).
def sanity_check_graph_curvature(G):

    nid_positions = extract_node_positions_dictionary(G)
    for eid, attrs in iterate_edges(G):
        check("curvature" in attrs, expect="Expect every edge to have 'curvature' attribute annotation.")
        u, v = eid[:2]
        ps = attrs["curvature"]
        check(u <= v, expect="Expect `u < v` for all edges.")
        p, q = nid_positions[u], nid_positions[v]
        check(np.all(p == ps[0]), expect="Expect curvature of all connected edges starts/end at node position.")
        check(np.all(q == ps[-1]), expect="Expect curvature of all connected edges starts/end at node position.")

# Assert with a breakpoint, so we can debug if an exception occurs.
def check(statement, expect=None):
    try:
        assert statement
    except:
        if expect != None:
            print(f"Assertion failed: {expect}")
        breakpoint()

# Raises exception at unimplemented code.
def todo(task):
    raise Exception(f"TODO: {task}")

