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


    edge_idx = graphedges_to_rtree(G)
    node_idx = graphnodes_to_rtree(G)


    edge_set = set() # Collected edges to consider.

    # Construct bounding boxes over path.
    for (p, q) in nx.utils.pairwise(ps):
        bb = bounding_box(np.array([p,q]), padding=lam)
        edge_ids = edge_idx.intersection((bb[0][0], bb[0][1], bb[1][0], bb[1][1]), objects=True)
        for i in edge_ids:
            edge_set.add(i.object)

    node_set = set()
    for (a,b) in edge_set:
        node_set.add(a)
        node_set.add(b)

    nodes = [v for v in node_set]
    G = G.subgraph(nodes) # Extract subgraph with nodes.
    idx = graphnodes_to_rtree(G) # Replay idx to lower bounding box.

    start_nodes = nodes_in_ROI(idx, ps[0], lam=lam)
    end_nodes = nodes_in_ROI(idx, ps[-1], lam=lam)

    if len(start_nodes) == 0:
        return False, {}
    if len(end_nodes) == 0:
        return False, {}
    # for all combinations
    node_dict = extract_nodes_dict(G)
    for (a,b) in itertools.product(start_nodes, end_nodes):
        # for path in nx.shortest_simple_paths(nx.Graph(G), a, b):
        for path in nx.shortest_simple_paths(G, a, b):
            # edge to nodes
            # nodes = [a for (a,b) in path]
            # nodes.append(path[-1][1])
            # nodes to coordinates
            qs = np.array([node_dict[n] for n in path])
            plot_graph_and_curves(nx.MultiDiGraph(G), ps, qs)
        # is_covered, data = curve_by_curveset_coverage(ps, paths, lam=lam, measure=measure)
        # if is_covered:
        #     breakpoint()


# Example (Check arbitrary shortest path (with some noise) is covered by network):
G = extract_graph("chicago")
ps = gen_random_shortest_path(G)
# Add some noise to path.
lam = 0.0010
noise = np.random.random((len(ps),2)) * lam - np.ones((len(ps),2)) * 0.5 * lam
ps = ps + noise
result = coverage_curve_by_network(G, ps, lam=lam)
