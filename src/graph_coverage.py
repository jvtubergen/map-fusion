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
# * `vectorized`: Whether input graph is vectorized (and thereby whether we have to annotate vectorized or simplified edges).
def edge_graph_coverage(S, T, max_threshold=None, vectorized=True, convert_to_utm=True): 

    S = S.copy()

    # Sanity checks.
    assert (convert_to_utm and vectorized) or (not convert_to_utm) # We can only convert a vectorized graph to UTM coordinates.
    assert vectorized == (not S.graph["simplified"]) # Expect vectorized graph if we are supposed to simplify the graphs, otherwise not.
    assert convert_to_utm == (S.graph["coordinates"] == "latlon") # Expect to convert to utm iff graph is in latlon.
    for (u, v, attrs) in S.edges(data=True): # Check each edge has a threshold set.
        assert "threshold" not in attrs

    # Transform to local coordinate system.
    if convert_to_utm:
        utm_info = graph_utm_info(S)
        S = graph_transform_latlon_to_utm(S)

    # Prepare graphs.
    if vectorized:
        O = S # Store source under O, we apply changes to O after finding thresholds on the simplified edges.
        S = simplify_graph(S)
    
    if T.graph["simplified"]:
        T = vectorize_graph(T)
    if T.graph["coordinates"] != "utm":
        T = graph_transform_latlon_to_utm(T)

    # Sanity check no duplicated nodes.
    # Bug: Somehow duplicated nodes occur if S and T are simplified..?
    if not S.graph["simplified"]:
        assert len(duplicated_nodes(S)) == 0
        assert len(duplicated_nodes(T)) == 0

    T = graph_to_rust_graph(T)

    # Iteration variables.
    leftS  = S.edges(keys=True) # Edges we seek a threshold value for.
    lam  = 1 # Start with a threshold of 1 meter.
    thresholds = {} # Currently found thresholds.
    
    # Link a curve to every simplified edge.
    curves = {}
    for uvk in leftS:
        u, v, k = uvk # Edges always have a key, because S is always simplified at this point.
        ps = edge_curvature(S, u, v, k=k)
        curve = curve_to_vector_list(ps)
        curves[uvk] = curve
    
    # Increment threshold and seek nearby path till all edges have found a threshold (or max threshold is reached).
    while len(leftS) > 0 and (max_threshold == None or lam <= max_threshold):
        print(f"Lambda: {lam}. Edges: {len(leftS)}")

        for uvk in leftS:
            curve = curves[uvk]
            path = partial_curve_graph(T, curve, lam)

            # Annotate threshold to edge if applicable.
            if path != None:

                # Remove edge from edge set.
                leftS = leftS - set([uvk]) 
                # Save threshold to apply later.
                thresholds[uvk] = lam

        lam += 1 # Increment lambda.

    # Set processed edges to found threshold (vectorized edges from annotated simplified edges).
    if vectorized:
        # Convert threshold annotations of simplified edges (u, v, k) to vectorized edges (u, v).
        uvk_thresholds = thresholds
        thresholds = {}
        for (u, v, k) in uvk_thresholds:

            # Obtain vectorized edges concerning this simplified edge.
            edge_info = S.get_edge_data(u, v, k)

            # Depending on whether a edge contains curvature.
            annotate = [(u, v)]
            if "merged_edges" in edge_info:
                annotate += edge_info["merged_edges"]
            else:
                assert len(edge_curvature(S, u, v, k=k)) == 2
                # print((u,v))
                annotate = [(u, v)]

            # Set same threshold for each line segment of the simplified edge.
            for uv in annotate:
                thresholds[uv] = uvk_thresholds[(u, v, k)]
    
    # Set unprocessed edges to have infinite threshold.
    for (u, v, k) in leftS:
        if vectorized:
            # Obtain vectorized edges concerning this simplified edge.
            edge_info = S.get_edge_data(u, v, k)

            # Depending on whether a edge contains curvature.
            if "merged_edges" in edge_info:
                to_drop = edge_info["merged_edges"]
            else:
                # print((u,v))
                to_drop = [(u, v)]

            # Set same threshold for each line segment of the simplified edge.
            for uv in to_drop:
                thresholds[uv] = inf
        else:
            thresholds[(u, v, k)] = inf

    # Convert thresholds into dictionary elements processable by the `set_edge_attributes` function.
    attributes = {}
    for uv_k in thresholds:
        attributes[uv_k] = {"threshold": thresholds[uv_k]}

    # Restore graph to input state.
    if vectorized:
        S = O
    if convert_to_utm:
        S = graph_transform_utm_to_latlon(S, "", **utm_info) # Convert back into latlon.

    # Apply threshold annotation.
    nx.set_edge_attributes(S, attributes) # Set thresholds for each edge.
    S.graph['max_threshold'] = max_threshold # Mention till what threshold we have searched.

    # Account for missing edges (this is a bug, somehow edges got skipped. Possibly an error in the OSMnx library simplify function.?)
    thresholds = {}
    if vectorized:
        for (u, v, attrs) in S.edges(data=True): # Check each edge has a threshold set.
            if not "threshold" in attrs:
                thresholds[(u, v)] = inf
    else:
        for (u, v, k, attrs) in S.edges(data=True, keys=True): # Check each edge has a threshold set.
            if not "threshold" in attrs:
                thresholds[(u, v, k)] = inf
    attributes = {}
    for uv_k in thresholds:
        attributes[uv_k] = {"threshold": thresholds[uv_k]}
    nx.set_edge_attributes(S, attributes) # Set thresholds for each edge.

    # Sanity checks.
    assert vectorized == (not S.graph["simplified"]) # Expect vectorized graph if we are supposed to simplify the graphs, otherwise not.
    assert convert_to_utm == (S.graph["coordinates"] == "latlon") # Expect to convert to utm iff graph is in latlon.
    for (u, v, attrs) in S.edges(data=True): # Check each edge has a threshold set.
        assert "threshold" in attrs

    return S

