from external import *

from utilities import *


# Prune graph with threshold-annotated edges.
# * TODO: Only add edge connections with sat if gps edge as adjacent covered edges (thus concatenate with `merge_graph` logic).
# * Rather than filtering out below threshold, we can as well seek edges above threshold (thus inverting the result).
def prune_coverage_graph(G, prune_threshold=10, invert=False):

    assert not G.graph["simplified"]

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


# Merges graph A into graph C.
# * Injects uncovered edges of graph A into graph C. 
# * Graph A has got its edges annotated with coverage threshold in relation to graph C.
def merge_graphs(C=None, A=None, prune_threshold=20):

    is_vectorized = not A.graph["simplified"]

    assert A.graph["simplified"] == C.graph["simplified"]
    assert A.graph['max_threshold'] > 0 # Make sure thresholds are set.
    assert prune_threshold <= A.graph['max_threshold'] # Should not try to prune above max threshold used by annotation.

    A = A.copy()
    C = C.copy()

    # Relabel additional to prevent node id overlap. / # Adjust nids of A to ensure uniqueness once added to C.
    nid = max(C.nodes()) + 1
    relabel_mapping = {}
    for nidH in A.nodes():
        relabel_mapping[nidH] = nid
        nid += 1
    A = nx.relabel_nodes(A, relabel_mapping)

    # Edges above and below the prune threshold. We retain edges below the prune threshold.
    if is_vectorized:
        drop   = above = above_threshold = [(u, v) for u, v, attrs in A.edges(data=True) if attrs["threshold"] <= prune_threshold]
        retain = below = below_threshold = [(u, v) for u, v, attrs in A.edges(data=True) if attrs["threshold"] >  prune_threshold]
    else:
        drop   = above = above_threshold = [(u, v, k) for u, v, k, attrs in A.edges(data=True, keys=True) if attrs["threshold"] <= prune_threshold]
        retain = below = below_threshold = [(u, v, k) for u, v, k, attrs in A.edges(data=True, keys=True) if attrs["threshold"] >  prune_threshold]

    # Sanity check that retain and drop are disjoint.
    assert len(set(drop) & set(retain)) == 0
    assert len(set(drop) ^ set(retain)) == len(A.edges())

    B = A.edge_subgraph(retain)

    # Extract nids which are connected to an edge above and below threshold.
    if is_vectorized:
        nodes_above = set([nid for el in above for nid in el]) 
        nodes_below = set([nid for el in below for nid in el]) 
    else:
        nodes_above = set([nid for el in above for nid in [el[0], el[1]]]) 
        nodes_below = set([nid for el in below for nid in [el[0], el[1]]]) 

    # (Assume `nearest_node` strategy: Only have to list what nodes of B should get connected to C.)
    # List nodes of B to connect with C.
    connect_nodes = []
    for nid in B.nodes():
        # This logic checks every node whether it is connected to both a covered (below threshold) and uncovered (above threshold) edge.
        # With the `nearest_node` strategy, exactly these nodes (of B) have to be connected with C.
        if nid in nodes_below and nid in nodes_above:
            connect_nodes.append(nid)
            B.nodes[nid]["render"] = "connection" # Annotate as connection point.
        else:
            B.nodes[nid]["render"] = "injected"

    for nid, attributes in C.nodes(data=True):
        attributes["render"] = "original"
    
    if is_vectorized:
        for u, v, attributes in B.edges(data=True):
            attributes["render"] = "injected"
        for u, v, attributes in C.edges(data=True):
            attributes["render"] = "original"
    else:
        for u, v, k, attributes in B.edges(data=True, keys=True):
            attributes["render"] = "injected"
        for u, v, k, attributes in C.edges(data=True, keys=True):
            attributes["render"] = "original"

    # Construct rtree on nodes in C.
    nodetree = graphnodes_to_rtree(C)

    # Register edge connections for A to C.
    connections = [] 
    for nid in connect_nodes: # In case of `nearest_node` strategy we only care for node, ignore edge.

        # Draw edge between nearest node in C and edge endpoint at w in B.
        y, x = B._node[nid]['y'], A._node[nid]['x'],
        hit = list(nodetree.nearest((y, x, y, x)))[0] # Seek nearest node.

        # Add straight line curvature and geometry 
        if is_vectorized:
            connections.append((hit, nid, {"render": "connection"}))
        else:
            y2, x2 = C._node[hit]['y'], C._node[hit]['x'],
            curvature = array([(y, x), (y2, x2)])
            geometry = to_linestring(curvature)
            connections.append((hit, nid, {"render": "connection", "geometry": geometry, "curvature": curvature}))
    
    # Inject B into C.
    C.add_nodes_from(B.nodes(data=True))

    if is_vectorized:
        C.add_edges_from(B.edges(data=True))
    else:
        C.add_edges_from(B.edges(data=True, keys=True))
    
    # Add edge connections between B and C.
    C.add_edges_from(connections)

    return C
