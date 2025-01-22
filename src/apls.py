# Rewrite of https://github.com:CosmiQ/apls
from utilities import *
from graph_simplifying import *
from graph_coordinates import *
from graph_curvature import *
from network import *

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
@info(timer=True)
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
        H_nids = [el[0] for el in data["nids"]]
        for G_nid, H_nid in zip(G_nids, H_nids):
            G_to_H[G_nid] = H_nid

    return H_with_control_points, G_to_H


# Prepare graph data to be sampled for APLS.
# * G has edges of max 50m length.
# * H has nearby control points injected.
# * Relation between control nodes of G to H.
# * All edges have length annotated.
@info(timer=True)
def prepare_graph_data(G, H):

    G = G.copy()
    H = H.copy()

    graph_annotate_edge_curvature(G)
    graph_annotate_edge_curvature(H)

    if not G.graph["coordinates"] == "utm":
        G = graph_transform_latlon_to_utm(G)
    if not H.graph["coordinates"] == "utm":
        H = graph_transform_latlon_to_utm(H)

    if not G.graph["simplified"]:
        G = simplify_graph(G)
    if not H.graph["simplified"]:
        H = simplify_graph(H)

    assert G.graph["simplified"]
    assert H.graph["simplified"]

    assert G.graph["coordinates"] == "utm"
    assert H.graph["coordinates"] == "utm"

    sanity_check_edge_length(G)
    sanity_check_edge_length(H)
    sanity_check_node_positions(G)
    sanity_check_node_positions(H)

    logger("Before: ", duplicated_nodes(G))

    # Ensure lengths within 50m.
    G = graph_ensure_max_edge_length(G, max_length=50)

    logger("After:  ", duplicated_nodes(G))

    sanity_check_edge_length(G)
    sanity_check_edge_length(H)
    sanity_check_node_positions(G)
    sanity_check_node_positions(H)

    # Find and relate control points of G to H.
    # Note: All nodes (of the simplified graph) are control points.
    Hc, G_to_Hc = inject_and_relate_control_points(G, H)

    return {
        "G": G,
        "Hc": Hc,
        "G_to_Hc": G_to_Hc
    }


# Compute shortest path data (between all pairs of start-end nodes within a selection of node identifiers).
@info(timer=True)
def precompute_shortest_path_data(G, control_nids):

    # Sanity check control nids exist in graph.
    for nid in control_nids:
        if nid not in G.nodes():
            raise Exception(f"Control nid {nid} does not exist in the graph.")

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
@info(timer=True)
def perform_sampling(G, Hc, G_to_Hc, G_shortest_paths, Hc_shortest_paths):

    sample_paths = {} # The start and end node pair related to a sample. This is taken from the G graph (so to reconstruct for Hc you require to apply G_to_Hc).

    nids = set(list(G_to_Hc.keys())) # The control points we sample (and thus seek paths between).
    nids_not_covered = set([nid for nid in nids if G_to_Hc[nid] == None]) 
    nids_covered     = nids - nids_not_covered
    
    ## Category A: No control point in the proposed graph.
    sample_paths["A"] = [(start, end) for start in nids_not_covered for end in G_shortest_paths[start]]

    ## Category B: Control nodes exist and a path exists in the ground truth, but not in the proposed graph. 
    sample_paths["B"] = [(start, end) for start in nids_covered for end in G_shortest_paths[start] if G_to_Hc[end] not in Hc_shortest_paths[G_to_Hc[start]]]

    ## Category C: Both graphs have control points and a path between them.
    sample_paths["C"] = [(start, end) for start in nids_covered for end in G_shortest_paths[start] if G_to_Hc[end] in Hc_shortest_paths[G_to_Hc[start]]]

    return sample_paths


# Asymmetric APLS computes by only considering the control nodes into the proposed graph.
# * Optionally provide a predetermined set of control nodes.
# * Optionally extract control nodes specifically viable for computing prime (thus control point is related to proposed graph).
@info(timer=True)
def apls_asymmetric_sampling(prepared_graph_data, n=500, prime=False):

    # Prepared graph data for sampling.
    G = prepared_graph_data["G"]
    Hc = prepared_graph_data["Hc"]
    G_to_Hc = prepared_graph_data["G_to_Hc"]

    # Select control nodes to perform sampling on.
    if prime:
        nids_to_sample_from = [nid for nid in G_to_Hc.keys() if G_to_Hc[nid] != None]

    else:
        # Sample randomly from the original graph until we have the number of control nodes.
        nids_to_sample_from = list(G.nodes())
    
    G_control_nids = set(random.sample(nids_to_sample_from, min(n, len(nids_to_sample_from))))
    
    # Take subset of `G_to_Hc` to the control nodes (these are the only nodes we have to relate with another).
    for nid in [nid for nid in G_to_Hc.keys()]:
        if nid not in G_control_nids:
            G_to_Hc.pop(nid)
    
    # Sanity check that all control nids of G are contained in the G-to-Hc mapping.
    for nid in G_control_nids:
        if nid not in G_to_Hc:
            raise Exception(f"Expect all nids of G_control_nids to be present in G_to_Hc.")
    
    # Obtain control nids to use in `H`.
    Hc_control_nids = set([G_to_Hc[nid] for nid in G_control_nids if G_to_Hc[nid] != None])

    # Compute shortest paths between this set of control nodes.
    G_shortest_paths = precompute_shortest_path_data(G, G_control_nids)
    Hc_shortest_paths = precompute_shortest_path_data(Hc, Hc_control_nids)

    # Perform sampling.
    samples = perform_sampling(G, Hc, G_to_Hc, G_shortest_paths, Hc_shortest_paths)

    # Compute path score.
    # Note: This function is bound to scope (it relies on variables in scope) to keep things simple.
    def score(start, end):
        a = G_shortest_paths[start][end]
        b = Hc_shortest_paths[G_to_Hc[start]][G_to_Hc[end]]
        value = 1 - min(abs(a - b) / a, 1)
        return value

    # Compute metric.
    path_scores      = [score(start, end) for (start, end) in samples["C"]]

    # Arbitrary data for debugging/visualization purposes.
    data = {
        "samples": samples,
        "path_scores": path_scores,
        "G_control_nids": G_control_nids,
        "prepared_graph_data": prepared_graph_data
    }
    
    return data


# Compute score on the data object.
def compute_score(data, prime=False):

    path_scores = data["path_scores"]
    samples     = data["samples"]
    sample_sum  = sum(path_scores)

    if prime:
        n = len(samples["B"]) + len(samples["C"])
    else:
        n = len(samples["B"]) + len(samples["C"]) + len(samples["A"])

    score = sample_sum / (len(samples["B"]) + len(samples["C"]) + len(samples["A"]))
    
    return score


# Compute the APLS metric (a similarity value between two graphs in the range [0, 1]).
@info(timer=True)
def apls(G, H, n=500, prime=False, prepared_graph_data=None):

    if prepared_graph_data == None:
        prepared_graph_data = {
            "left" : prepare_graph_data(G, H),
            "right": prepare_graph_data(H, G),
        }
    else:
        # Deep copy to prevent mangling (`apls_asymmetric_sampling` pops unneeded nids from `G_to_Hc`).
        prepared_graph_data = deepcopy(prepared_graph_data)

    left  = apls_asymmetric_sampling(prepared_graph_data["left"] , n=n, prime=prime)
    right = apls_asymmetric_sampling(prepared_graph_data["right"], n=n, prime=prime)

    score = 0.5 * (compute_score(left, prime=prime) + compute_score(right, prime=prime))

    data = {
        "left" : left,
        "right": right,
    }

    return score, data