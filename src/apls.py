# Rewrite of https://github.com:CosmiQ/apls
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
def inject_and_relate_control_points(G, H):
    todo()


# Compute shortest path data.
def precompute_shortest_path_data(G):
    # `nx.all_pairs_shortest_path_length`
    todo()


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