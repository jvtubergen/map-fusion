from external import *

from graph_simplifying import *
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

    assert A.graph["coordinates"] == C.graph["coordinates"]
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
    above = [eid for eid, attrs in iterate_edges(A) if attrs["threshold"] >  prune_threshold]
    below = [eid for eid, attrs in iterate_edges(A) if attrs["threshold"] <= prune_threshold]

    # Sanity check that retain and drop are disjoint.
    # NOTE: Set overlap here is about _edges_, not nodes. Thus therefore we can demand this uniqueness (non-overlapping) constraint.
    assert len(set(above) & set(below)) == 0
    assert len(set(above) ^ set(below)) == len(A.edges())

    # Retain edges above the coverage threshold (thus those edges of GPS not being covered by Sat).
    B = A.edge_subgraph(above)

    # Extract nids which are connected to an edge above and below threshold.
    nodes_above = set([nid for el in above for nid in el[0:2]]) 
    nodes_below = set([nid for el in below for nid in el[0:2]]) 

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

    for attrs in iterate_edge_attributes(B):
        attrs["render"] = "injected"

    for nid, attrs in C.nodes(data=True):
        attrs["render"] = "original"
    for attrs in iterate_edge_attributes(C):
        attrs["render"] = "original"

    # Construct rtree on nodes in C.
    nodetree = graphnodes_to_rtree(C)

    # Register edge connections for A to C. (Node-based inserted.)
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
            connections.append((nid, hit, {"render": "connection", "geometry": geometry, "curvature": curvature}))
    # Inject B into C.
    C.add_nodes_from(B.nodes(data=True))
    C.add_edges_from(graph_edges(B))
    
    # Add edge connections between B and C.
    C.add_edges_from(connections)

    return C
