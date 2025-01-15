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

    relationship = {} # All nodes of G 

    # Construct bounding boxes on nodes and edges of H to compute neighboring nodes/edges more quickly.
    H_node_rtree = graphnodes_to_rtree(H)
    H_edge_rtree = graphedges_to_rtree(H)

    G_node_bboxs = graphnodes_to_bboxs(G)
    G_node_positions = extract_nodes_dict(G)

    H_to_G = H.copy() # The resulting graph of H after injecting control nodes.

    to_inject = {}

    # Find (and optionally inject) node within H nearby every node of G.
    for G_nid, attrs in iterate_nodes(G):

        # Initiate an empty relation.
        relationship[G_nid] = None

        # Pick the node bounding box and pad it with the max distance.
        G_node_bbox = G_node_bboxs[G_nid]
        G_node_bbox = pad_bounding_box(G_node_bbox, max_distance + 0.0001)

        G_node_position = G_node_positions[G_nid]

        # Try to find nearby node (in H) nearby control point (of G).
        intersections = intersect_rtree_bbox(H_node_rtree, G_node_bbox)
        if len(intersections) > 0: 
            # Extract nid from intersection result.
            H_nid = intersections[0]
            relationship[G_nid] = H_nid
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
        H_to_G, data = graph_cut_edge_intervals(H_to_G, H_eid, curve_intervals)

        # Link resulting node identifiers.
        H_nids = data["nids"]
        for G_nid, H_nid in zip(G_nids, H_nids):
            relationship[G_nid] = H_nid

    return H_to_G, relationship


# Compute shortest path data (for a number of randomly picked node identifiers from the graph).
def precompute_shortest_path_data(G, n=500, nids=None):

    # If we did not provide a specific set of nodes, start with the entire collection of nodes of the graph.
    if nids == None:
        nids = list(G.nodes())

    # Sample n nids (to find all shortest paths between).
    nids = set(random.sample(nids, min(n, len(G.nodes()))))

    # Compute distance matrix between these points.
    distance_matrix = {}
    for nid in nids:
        distance_matrix[nid] = nx.single_source_dijkstra_path_length(G, nid, weight="length")

    return distance_matrix


# Perform all samples and categorize them into the three categories:
# * Proposed graph does not have a control point.
# * Proposed graph does not have a path between control points.
# * The difference in path length.
def perform_sampling(G, H_to_G, H_to_G_relations, H_to_G_shortest_paths):
    todo()


# Compute the APLS metric (a similarity value between two graphs in the range [0, 1]).
def apls(G, H):

    assert G.graph["coordinates"] == "utm"
    assert H.graph["coordinates"] == "utm"

    # Ensure lengths within 50m.
    G = ensure_max_edge_length(G)
    H = ensure_max_edge_length(H)

    # Find and relate control points of G to H.
    # Note: All nodes (of the simplified graph) are control points.
    G_to_H, G_to_H_relations = inject_and_relate_control_points(G, H)
    H_to_G, H_to_G_relations = inject_and_relate_control_points(H, G)

    # Pre-compute shortest path data.
    G_to_H_shortest_paths = precompute_shortest_path_data(G_to_H)
    H_to_G_shortest_paths = precompute_shortest_path_data(H_to_G)

    # Perform sampling.
    no_point, no_path, valid = perform_sampling(G, H_to_G, G, H_to_G_relations, H_to_G_shortest_paths)
    no_point, no_path, valid = perform_sampling(G, H_to_G, G, H_to_G_relations, H_to_G_shortest_paths)

    # Compute APLS and APLS* from samples.
    apls_value = todo()
    apls_prime_value = todo()

    return apls_value, apls_prime_value