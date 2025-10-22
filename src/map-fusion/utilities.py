from external import * 

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


#######################################
### Bounding boxes
#######################################

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

rtree_identifiers    = lambda tree: list(tree.intersection(tree.bounds))
rtree_ids_bboxs      = lambda tree: list(tree.intersection(tree.bounds, objects=True))

def rtree_id_bbox(tree, id):
    for item in rtree_ids_bboxs(tree):
        if item.id == id:
            bounds = item.bounds  # Non-interleaved: [xmin, xmax, ymin, ymax, ...]
            bbox = item.bbox      # Interleaved: [xmin, ymin, xmax, ymax, ...]
            return bbox




#######################################
## Curves
#######################################

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


# Obtain position at interval (in [0,1]) along curve ps.
def position_at_curve_interval(ps, interval):
    lengths    = norm(ps[1:] - ps[:-1], axis=1)
    weights    = lengths / sum(lengths)

    cumulative = 0
    for i, weight in enumerate(weights):
        cumulative += weight
        if interval < cumulative: # interval > cumulative - weight
            percentage = (interval - (cumulative - weight)) / weight
            return ps[i] * (1 - percentage) + ps[i+1] * percentage
    
    return ps[-1]


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


#######################################
### Linestring
#######################################

# Convert an array into a LineString consisting of Points.
to_linestring   = lambda curvature: LineString([Point(x, y) for y, x in curvature]) # Coordinates are flipped.

# Convert a LineString into an array.
from_linestring = lambda geometry : array([(y, x) for x, y in geometry.coords]) # Coordinates are flipped.


#######################################
### Partial curve matching logic
#######################################

# Convert 2d numpy array into a list of Vectors used by the partial curve matching algorithm.
def curve_to_vector_list(ps):
    result = []
    for [y, x] in ps:
        result.append(Vector(y, x))
    return result


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


# Curve coverage of ps by qs.
# either return false or provide subcurve with step sequence
# TODO: Optimization to check on bounding boxes before doing the interpolation.
def curve_by_curve_coverage(ps, qs, lam):
    return is_partial_curve_undirected(curve_to_vector_list(ps), curve_to_vector_list(qs), lam)


# Check coverage of a curve by a curve-set.
def curve_by_curveset_coverage(ps, qss, lam):
    for qs in qss:
        if is_partial_curve_undirected(curve_to_vector_list(ps), curve_to_vector_list(qs), lam):
            return True
    return False


#######################################
## Curve-point related logic.
#######################################

# Rotating (x, y)
def rotate(a):
    return array([
        [cos(a), sin(-a)],
        [sin(a), cos( a)],
    ])



def point_to_line_segment_distance_with_closest_point(u, v, p):
    """Compute distance, position, and interval of the nearest point of the line segment to the point."""
    px, py = p
    x1, y1 = u
    x2, y2 = v
    
    dx = x2 - x1
    dy = y2 - y1
    
    if abs(dx) < 0.0001 and abs(dy) < 0.0001:
        return math.sqrt((px - x1)**2 + (py - y1)**2), (x1, y1)
    
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    closest_point = array([closest_x, closest_y])
    
    distance = math.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    return distance, closest_point, t


def compute_curve_to_point(ps, point): 
    """Compute distance, position, and weighted interval (length offset from startpoint curve) of nearest point of curve to point."""
    
    items = []
    for i, linesegment in enumerate(zip(ps,ps[1:])):
        u, v = linesegment
        distance, position, interval = point_to_line_segment_distance_with_closest_point(u, v, point)
        items.append((i, distance, position, interval))

    # Seek lowest distance.
    i, distance, position, interval = min(items, key=lambda x: x[1])

    # Compute interval along curve.
    lengths    = norm(ps[1:] - ps[:-1], axis=1)
    weights    = lengths / np.sum(lengths)
    cumulative = np.hstack(([0], np.cumsum(weights)))
    actual_interval = cumulative[i] + interval * weights[i]

    return distance, position, actual_interval


# Wrapper function to only obtain nearest position on curve to point.
distance_curve_to_point         = lambda curve, point: compute_curve_to_point(curve, point)[0]
nearest_position_on_curve_to_point = lambda curve, point: compute_curve_to_point(curve, point)[1]
nearest_interval_on_curve_to_point = lambda curve, point: compute_curve_to_point(curve, point)[2]

nearest_position_and_interval_on_curve_to_point = lambda curve, point: compute_curve_to_point(curve, point)[1:3]


def test_nearest_position_and_interval_on_curve_to_point():

    # At arbitrary position to arbitrary location.
    for i in range(1000):
        curve = array([[i, 0.0] for i in range(100)])
        y = 50 * random.random()
        point = array([100*random.random(), y])
        check(abs(distance_curve_to_point(curve, point) - y) < 0.1)
        check(abs(norm(nearest_position_on_curve_to_point(curve, point) - point) - y) < 0.1)

    # At arbitrary position on aligned line.
    curve = array([[i, 0.0] for i in range(11)])
    point = array([5, 5])
    check(abs(norm(nearest_position_on_curve_to_point(curve, point) - point) - 5) < 0.01)

    # In extension of line segment
    ps = array([[0, 0], [1,0]])
    p = array([-1,0])
    q = nearest_position_on_curve_to_point(ps, p)
    print(norm(q - p))
    assert abs(norm(q - p) - 1) < 0.0001

    # At vertex position.
    for i in range(1000):
        ps = random_curve()
        p = ps[floor(random.random()*99)]
        q = nearest_position_on_curve_to_point(ps, p)
        print(norm(q - p))
        check(norm(q - p) < 0.1)


def test_compute_curve_to_point():
    """Test consistency of nearest_position_on_curve_to_point with 1000 random curves and points."""
    
    for i in range(1000):
        # Generate random curve and random point
        curve = random_curve()
        point = array([20 * random.random() - 10, 20 * random.random() - 10])

        # Generate info of nearest point on curve.
        distance, position, interval = compute_curve_to_point(curve, point)

        # Check that interval is in valid range [0, 1]
        check(0 - 1e-10 <= interval <= 1 + 1e-10)
    
        # TODO: Check position is at the curve interval.
        # TODO: Check distance matches norm between point and curve position.


#######################################
### Arbitrary
#######################################

# Unzip a list of pairs into a pair of lists.
def unzip(data):
    left, right = [], []
    for l, r in data:
        left.append(l)
        right.append(r)
    return left, right


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

