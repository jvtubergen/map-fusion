# Rewrite of https://github.com:CosmiQ/apls
from utilities import *
from graph_curvature import *

def todo():
    raise Exception("todo")


# Relate nodes of G to H and inject control points if necessary.
# If no nearby point on H can be found (in relation to a node `nid` of G), then `H_to_G[nid]` is `None`.
# Example:
# ```python
# H_to_G, H_to_G_relations = inject_and_relate_control_points(H, G)
# assert len(H_to_G.nodes()) >= len(H.nodes())
# H_to_G.nodes()[H_to_G_relations[G.nodes()[0]]] # Link nodes of G to H.
# ```
def inject_and_relate_control_points(G, H, max_distance=4):

    G_to_H = {} # All nodes of G related to a node of H.

    # Construct bounding boxes on nodes and edges of H to compute neighboring nodes/edges more quickly.
    H_node_rtree = graphnodes_to_rtree(H)
    H_edge_rtree = graphedges_to_rtree(H)

    G_node_bboxs = graphnodes_to_bboxs(G)
    G_node_positions = extract_nodes_dict(G)

    H_with_control_points = H.copy() # The resulting graph of H after injecting control nodes.

    to_inject = {}

    # Find (and optionally inject) node within H nearby every node of G.
    for G_nid, attrs in iterate_nodes(G):

        # Initiate an empty relation.
        G_to_H[G_nid] = None

        # Pick the node bounding box and pad it with the max distance.
        G_node_bbox = G_node_bboxs[G_nid]
        G_node_bbox = pad_bounding_box(G_node_bbox, max_distance + 0.0001)

        G_node_position = G_node_positions[G_nid]

        # Try to find nearby node (in H) nearby control point (of G).
        intersections = intersect_rtree_bbox(H_node_rtree, G_node_bbox)
        if len(intersections) > 0: 
            # Extract nid from intersection result.
            H_nid = intersections[0]
            G_to_H[G_nid] = H_nid
            continue
    
        # Try to find edge (in H) nearby control point (of G).
        # Crude subselection/filtering by bounding boxes.
        intersections = intersect_rtree_bbox(H_edge_rtree, G_node_bbox)
        if len(intersections) == 0:
            continue # No nearby edge curvatures per bounding box.
    
        # Closely check curvatures.
        H_eids = intersections
        curves = [attrs["curvature"] for eid, attrs in iterate_edges(H) if eid in set(H_eids)]
        curve_points, curve_intervals = unzip([nearest_position_and_interval_on_curve_to_point(curve, G_node_position) for curve in curves])
        distances = [norm(curve_point - G_node_position) for curve_point in curve_points]

        # Seek lowest value.
        i, distance = min(enumerate(distances), key=lambda x: x[1])

        # Check distance is below threshold.
        if distance > max_distance - 0.0001: # Otherwise this node of G has no relation to H.
            continue # No nearby edge curvatures checked precisely by curvature.
            
        # Values of selected curve (which lies sufficiently close to the point of interest).
        H_eid = H_eids[i]
        curve = curves[i]
        curve_point = curve_points[i]
        curve_interval = curve_intervals[i]

        # Store interval (at which to cut the edge) to perform later.
        item = (curve_interval, G_nid)
        if H_eid in to_inject:
            to_inject[H_eid].append(item)
        else:
            to_inject[H_eid] = [item]

    # Perform injection and linking.
    for H_eid in to_inject.keys():

        # H_eid is the H edge to cut.
        items = to_inject[H_eid]

        # Take out curve intervals from low to high.
        curve_intervals, G_nids = unzip(sorted(set(items))) # (By building a set we filter out unique elements)

        # Cut intervals.
        H_with_control_points, data = graph_cut_edge_intervals(H_with_control_points, H_eid, curve_intervals)

        # Link resulting node identifiers.
        H_nids = data["nids"]
        for G_nid, H_nid in zip(G_nids, H_nids):
            G_to_H[G_nid] = H_nid

    return H_with_control_points, G_to_H


# Compute shortest path data (for a number of randomly picked node identifiers from the graph).
def precompute_shortest_path_data(G, n=500, nids=None):

    # If we did not provide a specific set of nodes, start with the entire collection of nodes of the graph.
    if control_nids == None:
        control_nids = list(G.nodes())

    # Sample n control points (to find all shortest paths between).
    control_nids = set(random.sample(control_nids, min(n, len(G.nodes()))))

    # Compute distance matrix between these points.
    distance_matrix = {}
    for u in control_nids:

        # Compute all reachable points from this node.
        distances = nx.single_source_dijkstra_path_length(G, u, weight="length")

        # Filter out dictionary to only include end nodes which are:
        # * Contained in the control nodes list.
        # * Have a node identifier higher than the control nid coming from (prevents duplicated checks in subsequent logic).
        filtered = {}
        for v in distances.keys():
            if v > u and v in control_nids:
                filtered[v] = distances[v]
        
        distance_matrix[u] = filtered

    return distance_matrix


# Perform all samples and categorize them into the three categories:
# * A. Proposed graph does not have a control point.
# * B. Proposed graph does not have a path between control points.
# * C. Both graphs have control points and a path between them.
def perform_sampling(G, Hc, G_to_Hc, G_shortest_paths, Hc_shortest_paths):

    samples = {} # A sample itself is a value between 0 and 1. Higher value implies worse. (So perfect sample has a value of 0).
    sample_paths = [] # The start and end node pair related to a sample. This is taken from the G graph (so to reconstruct for Hc you require to apply G_to_Hc).

    nids = set(list(G_to_Hc.keys())) # The control points we sample (and thus seek paths between).
    nids_not_covered = set([nid for nid in nids if G_to_Hc[nid] == None]) 
    nids_covered     = nid - nids_not_covered
    
    ## Category A: No control point in the proposed graph.
    sample_paths["A"] = [(start, end) for start in nids_not_covered for end in G_shortest_paths[start]]

    ## Category B: Control nodes exist and a path exists in the ground truth, but not in the proposed graph. 
    sample_paths["B"] = [(start, end) for start in nids_covered for end in G_shortest_paths[start] if G_to_Hc[end] not in Hc_shortest_paths[G_to_Hc[start]]]

    ## Category C: Both graphs have control points and a path between them.
    sample_paths["C"] = [(start, end) for start in nids_covered for end in G_shortest_paths[start] if G_to_Hc[end] in Hc_shortest_paths[G_to_Hc[start]]]

    return sample_paths


# Asymmetric APLS computes by only considering the control nodes into the proposed graph.
def apls_asymmetric(G, H):

    assert G.graph["simplified"]
    assert H.graph["simplified"]

    assert G.graph["coordinates"] == "utm"
    assert H.graph["coordinates"] == "utm"

    # Ensure lengths within 50m.
    G = ensure_max_edge_length(G)

    # Find and relate control points of G to H.
    # Note: All nodes (of the simplified graph) are control points.
    Hc, G_to_Hc = inject_and_relate_control_points(G, H)

    # Edge length is necessary for computing shortest distance.
    G  = graph_annotate_edge_length(G)
    Hc = graph_annotate_edge_length(Hc)

    # Control nodes which have coverage (subselection of control points for APLS prime metric).
    G_prime_nodes = [nid for nid in G_to_Hc.keys() if G_to_Hc[nid] != None]
    Hc_prime_nodes = [G_to_Hc[nid] for nid in G_prime_nodes]

    # Graphs are prepared, we can apply APLS or APLS prime specific logic.
    if prime:

        # Limit the number of nodes we are computing from.
        G_shortest_paths = precompute_shortest_path_data(G, nids=G_prime_nodes)
        Hc_shortest_paths = precompute_shortest_path_data(Hc, nids=Hc_prime_nodes)

    else:

        # For normal APLS we require to include the dangling nodes within the distance matrix, to know the number of reachable paths to penalize for.
        G_shortest_paths = precompute_shortest_path_data(G)
        Hc_shortest_paths = precompute_shortest_path_data(Hc)

    # Perform sampling.
    samples = perform_sampling(H, G_to_Hc, G, G_to_Hc, H_to_G_shortest_paths)

    # Compute path score.
    def score(start, end):
        a = G_shortest_paths[start][end]
        b = Hc_shortest_paths[G_to_Hc[start]][G_to_Hc[end]]
        value = 1 - min(abs(a - b) / a, 1)
        return value

    # Compute metric.
    n = len(samples["B"]) + len(samples["C"])

    if not prime:
        n += samples["A"]

    # Add sample points
    result = sum([score(start, end) for (start, end) in samples["C"]]) / n

    return result


# Compute the APLS metric (a similarity value between two graphs in the range [0, 1]).
def apls(G, H):

    result = 0.5 * (apls_asymmetric(G, H) + apls_asymmetric(H, G))

    return result