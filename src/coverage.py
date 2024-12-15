from external import *
from utilities import *
from network import *

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

# Compute per (simplified) edge of S (Source graph) the coverage threshold in order to be matched by T (Target graph).
# Assume S and T are a MultiGraph, simplified, and in appropriate coordinate system to measure distance differences in meter.
# todo optimize: Have a minimal distance per edge to start evaluation (e.g. some thresholds start at 700 meters).
def edge_wise_coverage_threshold(S, T, max_threshold=None):
    
    assert type(S) == nx.Graph
    assert type(T) == nx.Graph

    assert not S.graph.get("simplified") # Vectorized.
    assert not T.graph.get("simplified") # Vectorized.

    # Transform to local coordinate system.
    S = graph_transform_latlon_to_utm(S)
    T = graph_transform_latlon_to_utm(T)

    # Source graph should be simplified.
    S = simplify_graph(S)

    # Construct rust graph for target.
    graph        = graph_to_rust_graph(T)
    # edgebboxs  = graphedges_to_bboxs(S) # Have a bounding box per edge so we can quickly pad for intersection test against T.
    # edgetree   = graphedges_to_rtree(T) # Place graph edges by coordinates in accelerated data structure (R-Tree).
    edges_todo   = S.edges()
    nodedict     = extract_nodes_dict(S)
    edge_results = {}

    # Increment threshold and seek nearby path till all edges have found a threshold (or max threshold is reached).
    lam = 1 # Start with a threshold of 1 meter.
    while len(edges_todo) > 0 and (max_threshold == None or lam <= max_threshold):

        print(f"Lambda: {lam}. Edges: {len(edges_todo)}")

        # Iterate every edge left to do.
        for uv in edges_todo:

            u, v = uv
            ps = edge_curvature(S, u, v)
            curve = curve_to_vector_list(ps)
            result = partial_curve_graph(graph, curve, lam)
            if result != None:
                path = result
                edges_todo = edges_todo - set([uv]) # Remove edge from edge set.
                edge_results[uv] = {
                    "threshold": lam,
                    "path": path,
                }
        lam += 1 # Increment lambda

    return edge_results, edges_todo


# Extract subgraph covered below given threshold (feed in coverage data).
def subgraph_by_coverage_thresholds(graph, coverage_data, max_threshold=10):

    edges = []
    thresholds = []
    for edge in coverage_data[0].keys():
        threshold = coverage_data[0][edge]["threshold"]
        edges.append(edge)
        thresholds.append(threshold)

    # Set threshold above 100 to inf.
    for edge in coverage_data[1]:
        threshold = inf
        edges.append(edge)
        thresholds.append(threshold)

    # Transform to array masking to easily filter out thresholds below or above certain value.
    edges = array(edges)
    thresholds = array(thresholds)
    valids = edges[np.where(thresholds <= max_threshold)]
    invalids = edges[np.where(thresholds > max_threshold)]

    # Extract subgraph on valid edges.
    valid_edges = set()
    [valid_edges.add((edge[0], edge[1])) for edge in valids.tolist()]
    subgraph = graph.edge_subgraph(valid_edges)

    # Extract largest connected component.
    subgraph = ox.utils_graph.get_largest_component(subgraph.to_directed()).to_undirected()

    return subgraph


# Obtain threshold per simplified edge of S in comparison to T.
def edge_graph_coverage(S, T, max_threshold=None): # We should always act on simplified graph S.

    S = S.copy()

    # Sanity checks.
    assert type(S) == nx.Graph 
    assert not S.graph["simplified"]
    assert S.graph["coordinates"] == "latlon"
    assert type(T) == nx.Graph 
    assert not T.graph["simplified"] 
    assert T.graph["coordinates"] == "latlon"
    for (u, v, attrs) in S.edges(data=True): # Check each edge has a threshold set.
        assert "threshold" not in attrs

    # Transform to local coordinate system.
    utm_info = get_utm_info_from_graph(S)
    S = graph_transform_latlon_to_utm(S)
    T = graph_transform_latlon_to_utm(T)

    # Prepare graphs.
    O = S # Store source under O, we apply changes to O after finding thresholds on the simplified edges.
    S = simplify_graph(S)
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

        # Obtain vectorized edges concerning this simplified edge.
        edge_info = S.get_edge_data(u, v, k)

        # Depending on whether a edge contains curvature.
        if "merged_edges" in edge_info:
            to_drop = edge_info["merged_edges"]
        else:
            print((u,v))
            to_drop = [(u, v)]

        # Set same threshold for each line segment of the simplified edge.
        for uv in to_drop:
            thresholds[uv] = inf

    # Convert thresholds into dictionary elements processable by the `set_edge_attributes` function.
    attributes = {}
    for uv_k in thresholds:
        attributes[uv_k] = {"threshold": thresholds[uv_k]}

    # Restore graph to input state.
    S = O
    S = graph_transform_utm_to_latlon(S, "", **utm_info) # Convert back into latlon.

    # Apply threshold annotation.
    nx.set_edge_attributes(S, attributes) # Set thresholds for each edge.
    S.graph['max_threshold'] = max_threshold # Mention till what threshold we have searched.

    # Account for missing edges (this is a bug, somehow edges got skipped. Possibly an error in the OSMnx library simplify function.?)
    thresholds = {}
    for (u, v, attrs) in S.edges(data=True): # Check each edge has a threshold set.
        if not "threshold" in attrs:
            thresholds[(u, v)] = inf
    attributes = {}
    for uv_k in thresholds:
        attributes[uv_k] = {"threshold": thresholds[uv_k]}
    nx.set_edge_attributes(S, attributes) # Set thresholds for each edge.

    # Sanity checks.
    assert type(S) == nx.Graph 
    assert not S.graph["simplified"]
    assert S.graph["coordinates"] == "latlon"
    for (u, v, attrs) in S.edges(data=True): # Check each edge has a threshold set.
        assert "threshold" in attrs

    return S


# Prune graph with threshold-annotated edges.
def prune_coverage_graph(G, prune_threshold=10, invert=False):
    assert G.graph['max_threshold'] > 0 # Make sure thresholds are set.
    assert prune_threshold <= G.graph['max_threshold'] # Should not try to prune above max threshold used by annotation.

    retain = []
    for (a, b, attrs) in G.edges(data=True):
        # Iterate each edge and drop it if its threshold exceeds prune_threshold.
        if not invert and attrs["threshold"] <= prune_threshold:
            # Retain edge
            retain.append((a, b))
        elif invert and attrs["threshold"] > prune_threshold:
            retain.append((a, b))
    G = G.edge_subgraph(retain)
    return G
