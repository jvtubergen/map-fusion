from external import *
from utilities import *
from network import *
from graph_coordinates import *

###  Curve by curve coverage

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


###  Curve by network coverage

# Obtain threshold per simplified edge of S in comparison to T.
@info()
def edge_graph_coverage(S, T, max_threshold=None): 

    S = S.copy()

    # Sanity check the graph is simplified.
    check(S.graph["simplified"], expect="Expect the source graph is simplified" \
                                        ", we do not compute edge coverage for line segments" \
                                        ", nor do we want to restore (vectorized graph with proper edge threshold attribute annotation) in this function.")

    # Sanity checks each edge has a threshold set..
    check("threshold" not in S.graph, expect="Expect the graph to not have a 'max_threshold' attribute set.")
    for (u, v, attrs) in S.edges(data=True):
        check("threshold" not in attrs, expect="Expect edge in source to not have the 'threshold' attribute set" \
                                               ", because such existence suggests we are overwriting a previous coverage check" \
                                               ", suggesting some coverage computation is accidentally out of place.")

    # Make sure both source and target are in UTM coordinates (for threshold to make sense).
    convert_to_utm = S.graph["coordinates"] != "utm"

    if S.graph["coordinates"] != "utm":
        utm_info = graph_utm_info(S)
        S = graph_transform_latlon_to_utm(S)
    if T.graph["coordinates"] != "utm":
        T = graph_transform_latlon_to_utm(T)

    # We allow target to be vectorized, it causes no loss of information (since target is not being adjusted).
    if T.graph["simplified"]:
        T = vectorize_graph(T)

    T = graph_to_rust_graph(T)

    # Threshold computation iteration variables.
    leftS  = set(S.edges(keys=True)) # Edges we seek a threshold value for.
    lam    = 1 # Start with a threshold of 1 meter.
    thresholds = {} # Currently found thresholds.
    
    # Link a curve to every simplified edge.
    curves = {}
    for eid in leftS:
        ps = get_edge(S, eid)["curvature"]
        curve = curve_to_vector_list(ps)
        curves[eid] = curve
    
    # Increment threshold and seek nearby path till all edges have found a threshold (or max threshold is reached).
    while len(leftS) > 0 and (max_threshold == None or lam <= max_threshold):
        logger(f"Lambda: {lam}. Edges: {len(leftS)}")

        for eid in leftS:
            curve = curves[eid]
            path = partial_curve_graph(T, curve, lam)

            # Annotate threshold to edge if applicable.
            if path != None:

                # Remove edge from edge set.
                leftS = leftS - set([eid]) 
                # Save threshold to apply later.
                thresholds[eid] = lam

        lam += 1 # Increment lambda.

    # Set unprocessed edges to have infinite threshold.
    for eid in leftS:
        thresholds[eid] = inf

    # Set thresholds for each edge.
    nx.set_edge_attributes(S, {eid: {**attrs, "threshold": thresholds[eid]} for eid, attrs in iterate_edges(S)}) 
    
    for eid, attrs in iterate_edges(S):
        check("threshold" in attrs)

    # Restore graph to input state.
    if convert_to_utm:
        S = graph_transform_utm_to_latlon(S, "", **utm_info) # Convert back into latlon.

    # Apply threshold annotation.
    S.graph['max_threshold'] = max_threshold # Mention till what threshold we have searched.

    return S

