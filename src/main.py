# Library code
from dependencies import *
from coverage import *
from network import *
from rendering import *


# Find nodes in region of interest.
def nodes_in_ROI(idx, coord, lam=1):
    bb = bounding_box(np.array([coord]), padding=lam)
    nodes = list(idx.intersection((bb[0][0], bb[0][1], bb[1][0], bb[1][1])))
    return nodes


# Compute coverage of curve by a network.
def coverage_curve_by_network(G, ps, lam=1, measure=frechet):

    G = G.copy()
    if G.graph["simplified"]:
        G = vectorize_graph(G) # Vectorize graph.
        G = deduplicate_vectorized_graph(G)
    
    if type(G) != nx.Graph:
        G = nx.Graph(G)

    assert len(ps) >= 2

    idx = graphnodes_to_rtree(G) # Place graph nodes coordinates in accelerated data structure (R-Tree).
    bb = bounding_box(ps, padding=lam) # Construct lambda-padded bounding box.
    nodes = list(idx.intersection((bb[0][0], bb[0][1], bb[1][0], bb[1][1]))) # Extract nodes within bounding box.
    G = G.subgraph(nodes) # Extract subgraph with nodes.
    idx = graphnodes_to_rtree(G) # Replay idx to lower bounding box.

    start_nodes = nodes_in_ROI(idx, ps[0], lam=lam)
    end_nodes = nodes_in_ROI(idx, ps[-1], lam=lam)

    if len(start_nodes) == 0:
        return False, {}
    if len(end_nodes) == 0:
        return False, {}
    # for all combinations
    for (a,b) in itertools.product(start_nodes, end_nodes):
        paths = nx.all_simple_edge_paths(G, a, b)
        is_covered, data = curve_by_curveset_coverage(ps, paths, lam=lam, measure=measure)
        if is_covered:
            breakpoint()


# Example (Check arbitrary shortest path (with some noise) is covered by network):
G = extract_graph("chicago")
ps = gen_random_shortest_path(G)
# Add some noise to path.
lam = 0.0015
noise = np.random.random((len(ps),2)) * lam - np.ones((len(ps),2)) * 0.5 * lam
ps = ps + noise
result = coverage_curve_by_network(G, ps, lam=lam)
