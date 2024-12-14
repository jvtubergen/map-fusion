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
    assert type(S) == nx.Graph and S.graph["vectorized"] and S.graph["coordinates"] == "latlon"
    assert type(T) == nx.Graph and T.graph["vectorized"] and S.graph["coordinates"] == "latlon"
    for (u, v, attrs) in S.edges(data=True): # Check each edge has a threshold set.
        assert "threshold" not in attrs

    # Transform to local coordinate system.
    S = graph_transform_latlon_to_utm(S)
    T = graph_transform_latlon_to_utm(T)

    # Prepare graphs.
    S2 = simplify_graph(S)
    T2 = graph_to_rust_graph(T)
    
    # Increment threshold and seek nearby path till all edges have found a threshold (or max threshold is reached).
    leftS  = S.edges()
    leftS2 = S2.edges(keys=True)
    lam  = 1 # Start with a threshold of 1 meter.
    thresholds = {}
    while len(leftS2) > 0 and (max_threshold == None or lam <= max_threshold):

        print(f"Lambda: {lam}. Edges: {len(leftS2)}")
        for uvk in leftS2:
            u, v, k = uvk
            ps = edge_curvature(S2, u, v, k=k)
            curve = curve_to_vector_list(ps)

            path = partial_curve_graph(T2, curve, lam)
            if path != None:

                leftS2 = leftS2 - set([uvk]) # Remove edge from edge set.

                # Obtain vectorized edges concerning this simplified edge.
                edge_info = S2.get_edge_data(u, v, k)
                annotate = [(u, v)]
                if "merged_edges" in edge_info:
                    annotate = annotate + edge_info["merged_edges"]

                # No need to add reverse, we consider undirected graphs only. Library fixes the edge node identifiers order.
                for i in range(len(annotate)): # Do have uprunning sequence for our own administration.
                    (u, v) = annotate[i]
                    if v < u:
                        annotate[i] = (v, u)

                # Annotate S with result information.
                for uv in annotate:
                    # print(f"Annotating {uv}.")
                    leftS = leftS - set([uv])
                    thresholds[uv] = {"threshold": lam}

        lam += 1 # Increment lambda
    
    for uv in leftS:
        thresholds[uv] = {"threshold": inf}
    
    S = vectorize_graph(S) # Vectorize again.
    nx.set_edge_attributes(S, thresholds) # Set thresholds for each edge.
    S.graph['max_threshold'] = max_threshold # Mention till what threshold we have searched.
    S = graph_transform_utm_to_latlon(S) # Convert back into latlon.

    # Sanity checks.
    assert type(S) == nx.Graph and S.graph["vectorized"] and S.graph["coordinates"] == "latlon"
    for (u, v, attrs) in S.edges(data=True): # Check each edge has a threshold set.
        assert "threshold" in attrs

    return G


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
