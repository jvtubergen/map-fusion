# Rewrite of https://github.com:CosmiQ/apls with additional new functionality.
from data_handling import *
from utilities import *
from graph import *

@info(timer=True)
def inject_and_relate_control_points(G, H, max_distance=25):
    """
    Relate nodes of target graph G to source graph H and inject control points if necessary.
    Return the updated source graph H alongside a dictionary that links nodes of G to H (keys are nids of G, values are nids of H they connect to).
    """

    G_to_H = {} # All nodes of G related to a node of H.

    # Construct bounding boxes on nodes and edges of H to compute neighboring nodes/edges more quickly.
    H_node_rtree = graphnodes_to_rtree(H)
    H_edge_rtree = graphedges_to_rtree(H)

    G_node_bboxs = graphnodes_to_bboxs(G)
    G_node_positions = extract_node_positions_dictionary(G)

    H_with_control_points = H.copy() # The resulting graph of H after injecting control nodes.

    to_inject = {}

    # Find (and optionally inject) node within H nearby every node of G.
    for G_nid, attrs in iterate_nodes(G):

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
        H_nids = [el for el in data["nids"]]
        for G_nid, H_nid in zip(G_nids, H_nids):
            G_to_H[G_nid] = H_nid

    return H_with_control_points, G_to_H



@info(timer=True)
def perform_sampling(G, Hc, G_to_Hc, G_shortest_paths, Hc_shortest_paths):
    """
    Perform all samples and categorize them into the three categories:
    * A. Proposed graph does not have a control point.
    * B. Proposed graph does not have a path between control points.
    * C. Both graphs have control points and a path between them.
    """

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
def apls_sampling(G, Hc, G_to_Hc, G_paths, Hc_paths, n=10000):
    """Obtain samples.""" 

    all_nids       = set(G.nodes())
    all_nids_prime = set(G_to_Hc.keys())

    # Normal: Generate `n` samples with (start, end).
    normal_samples = set()
    while len(normal_samples) < min(n, len(all_nids)):
        start = random.select(all_nids, 1)
        end   = random.select(all_nids - start, 1)
        if end in G_paths[start]:
            normal_samples.add(sorted([start, end]))
    
    # Primal: Generate `n` samples with (start, end).
    primal_samples = set()
    while len(primal_samples) < min(n, len(all_nids_prime)):
        start = random.select(all_nids_prime, 1)
        end   = random.select(all_nids_prime - start, 1)
        if end in G_paths[start]:
            primal_samples.append(sorted([start, end]))

    ## Category A: No control point in the proposed graph.
    A_normal = set([(start, end) for (start, end) in normal_samples if start not in all_nids_prime or end not in all_nids_prime])
    A_primal = set()

    ## Category B: Control nodes exist and a path exists in the ground truth, but not in the proposed graph. 
    B_normal = set([(start, end) for (start, end) in normal_samples - A_normal if G_to_Hc[end] not in Hc_paths[G_to_Hc[start]]])
    B_primal = set([(start, end) for (start, end) in primal_samples - A_primal if G_to_Hc[end] not in Hc_paths[G_to_Hc[start]]])

    ## Category C: Both graphs have control points and a path between them.
    C_normal = normal_samples - A_normal.union(B_normal)
    C_primal = primal_samples - A_primal.union(B_primal)

    # Collect.
    samples_normal = {
        "A": A_normal,
        "B": B_normal,
        "C": C_normal,
    }

    samples_primal = {
        "A": A_primal,
        "B": B_primal,
        "C": C_primal,
    }

    return samples_normal, samples_primal


# Compute score on the data object.
def compute_score(samples, path_scores):
    sample_sum  = sum(path_scores)
    n = len(samples["B"]) + len(samples["C"]) + len(samples["A"])
    score = sample_sum / n
    return score


@info(timer=True)
def asymmetric_apls(G, H, n=10000):
    """Compute the APLS and APLS* metric (a similarity value between two graphs in the range [0, 1])."""

    Hc, G_to_Hc = inject_and_relate_control_points(G, H)
    Hc = graph_annotate_edges(Hc)

    logger("Compute shortest paths.")
    G_paths  = dict(nx.shortest_path(G, weight="length"))
    Hc_paths = dict(nx.shortest_path(Hc, weight="length"))

    # Obtain samples.
    samples_normal, samples_primal = apls_sampling(G, Hc, G_to_Hc, G_paths, Hc_paths, n=n)

    # Obtain path scores.
    path_scores_normal = {(start, end): {"source": Hc_paths[G_to_Hc[start]][G_to_Hc[end]], "target":G_paths[start][end]} for (start, end) in samples_normal["C"]}
    path_scores_primal = {(start, end): {"source": Hc_paths[G_to_Hc[start]][G_to_Hc[end]], "target":G_paths[start][end]} for (start, end) in samples_primal["C"]}

    # Compute sample values.
    sample_values_normal = [0 for _ in samples_normal["A"]] + [0 for _ in samples_normal["B"]] + [0 for _ in samples_normal["A"]]
    
    # At this moment the APLS is computed symmetrically only.
    apls_score       = compute_score(samples_normal, path_scores_normal)
    apls_prime_score = compute_score(samples_primal, path_scores_primal)

    # Compute metadata: 
    metadata = {
        "normal": {
            "samples": samples_normal,
            "path_scores": path_scores_normal,
        },
        "primal": {
            "samples": samples_primal,
            "path_scores": path_scores_primal,
        }
    }

    return apls_score, apls_prime_score, metadata


@info(timer=True)
def symmetric_apls(G, H, n=10000):
    left = asymmetric_apls(G, H, n=n)
    right= asymmetric_apls(H, G, n=n)
    gc.collect()

    apls_score = 0.5 * (left[0] + right[0])
    apls_prime_score = 0.5 * (left[1] + right[1])
    metadata = {
        "left": left[2],
        "right": right[2]
    }

    return apls_score, apls_prime_score, metadata