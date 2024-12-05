from external import *
from utilities import *
from network import *

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


# Obtain threshold per simplified edge of S.
def edge_graph_coverage(S, T, max_threshold=None):
    assert type(S) == nx.Graph
    assert type(T) == nx.Graph
    assert not S.graph.get("simplified") # Vectorized.
    assert not T.graph.get("simplified") # Vectorized.

    G = S.copy()

    # Transform to local coordinate system.
    S = graph_transform_latlon_to_utm(S)
    T = graph_transform_latlon_to_utm(T)

    # We need a multi-graph because there can be multiple roads 
    S2 = ox.simplify_graph(nx.MultiGraph(S).to_directed(), track_merged=True).to_undirected()
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

                # Add reverse.
                to_add = []
                for uv in annotate:
                    u, v = uv
                    uv = (v, u)
                    to_add += [uv]
                annotate += to_add

                # Annotate S with result information.
                for uv in annotate:
                    # print(f"Annotating {uv}.")
                    leftS = leftS - set([uv])
                    thresholds[uv] = {"threshold": lam}

        lam += 1 # Increment lambda
    
    for uv in leftS:
        thresholds[uv] = {"threshold": inf}
    
    nx.set_edge_attributes(G, thresholds)
    G.graph['max_threshold'] = max_threshold
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
