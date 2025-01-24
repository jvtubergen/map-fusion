from external import *

from graph_simplifying import *
from graph_coverage import * # Necessary for coverage computing of duplicated sat edges on injected gps edges.
from utilities import *


# Prune graph with threshold-annotated edges.
# * TODO: Only add edge connections with sat if gps edge as adjacent covered edges (thus concatenate with `merge_graph` logic).
# * Rather than filtering out below threshold, we can as well seek edges above threshold (thus inverting the result).
@info()
def prune_coverage_graph(G, prune_threshold=10, invert=False):

    check(G.graph["simplified"], expect="Expect the graph is simplified when edges have a threshold annotated.")
    check(prune_threshold <= G.graph['max_threshold'], expect="Expect we are pruning (on a threshold) below the maximum computed.")

    if invert:
        attribute_filter = lambda attrs: attrs["threshold"] > prune_threshold
    else:
        attribute_filter = lambda attrs: attrs["threshold"] <= prune_threshold

    retain = filter_eids_by_attribute(G, filter_func=attribute_filter)
    G = G.edge_subgraph(retain)

    return G


# Merges graph A into graph C.
# * Injects uncovered edges of graph A into graph C. 
# * Graph A has got its edges annotated with coverage threshold in relation to graph C.
# * Extension 1: Removal of duplicates.
# * Extension 2: Reconnecting C edges to injected A edges.
@info()
def merge_graphs(C=None, A=None, prune_threshold=20, remove_duplicates=False, reconnect_after=False):

    # Sanity checks.
    check(A.graph["simplified"], expect="Expect the graph is simplified for merging.")
    check(C.graph["simplified"], expect="Expect the graph is simplified for merging.")
    check(prune_threshold <= A.graph['max_threshold'], expect="Expect we are pruning (on a threshold) below the maximum computed.")
    check(remove_duplicates or (remove_duplicates == reconnect_after), expect="Expect to only reconnect if duplicates are to be removed.")

    _A = A
    A = _A.copy()

    _C = C
    C = _C.copy()

    if A.graph["coordinates"] == "latlon":
        A = graph_transform_latlon_to_utm(A)

    if C.graph["coordinates"] == "latlon":
        C = graph_transform_latlon_to_utm(C)

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
    assert len(set(above) ^ set(below)) == len(list(iterate_edges(A)))

    # Retain edges above the coverage threshold (thus those edges of A not being covered by C).
    B = A.edge_subgraph(above)

    # Extract nids which are connected to an edge above and below threshold.
    nodes_above = set([nid for el in above for nid in el[0:2]]) 
    nodes_below = set([nid for el in below for nid in el[0:2]]) 

    ## Annotating render attribute on B and C.
    # Obtain what nodes of B to connect with C (those nodes of A which are connected to both a covered and uncovered edge).
    # + Render nodes of B as either injected or connection points.
    connect_nodes = []
    for nid in B.nodes():
        # This logic checks every node whether it is connected to both a covered (below threshold) and uncovered (above threshold) edge.
        # With the `nearest_node` strategy, exactly these nodes (of B) have to be connected with C.
        if nid in nodes_below and nid in nodes_above:
            connect_nodes.append(nid)
            B.nodes[nid]["render"] = "connection" # Annotate as connection point.
        else:
            B.nodes[nid]["render"] = "injected"

    # Render edges of B as injected.
    annotate_edges(B, {"render": "injected"})
    
    # Render nodes and edges of C as original.
    annotate_nodes(C, {"render": "original"})
    annotate_edges(C, {"render": "original"})

    # Construct rtree on nodes in C.
    nodetree = graphnodes_to_rtree(C)

    # Register edge connections for A to C. (Node-based inserted.)
    connections = [] 
    for nid in connect_nodes: # In case of `nearest_node` strategy we only care for node, ignore edge.

        # Draw edge between nearest node in C and edge endpoint at w in B.
        y, x = B._node[nid]['y'], A._node[nid]['x'],
        hit = list(nodetree.nearest((y, x, y, x)))[0] # Seek nearest node.

        # Add straight line curvature and geometry 
        y2, x2 = C._node[hit]['y'], C._node[hit]['x'],
        curvature = array([(y, x), (y2, x2)])
        geometry = to_linestring(curvature)
        connections.append((nid, hit, {"render": "connection", "geometry": geometry, "curvature": curvature, "origin": "B"}))
    
    ## Annotate origin attribute on B and C (Necessary in case we want to apply extensions).
    # Annotate "origin" of nodes and edges of C.
    annotate_edges(C, {"origin": "C"})
    annotate_nodes(C, {"origin": "C"})

    # Annotate "origin" of injected nodes and edges of B.
    annotate_edges(B, {"origin": "B"})
    annotate_nodes(B, {"origin": "B"})

    ## Connecting B to C.
    # Inject B into C.
    C.add_nodes_from(list(iterate_nodes(B)))
    C.add_edges_from([(*eid[:2], attrs) for eid, attrs in iterate_edges(B)])
    
    # Add edge connections between B and C.
    C.add_edges_from(connections)

    # TODO: Extension a: Remove duplicated edges of C.
    if remove_duplicates: 

        # Extract A_prime (injected subgraph of A with connection edges to C).
        A_prime_eids = filter_eids_by_attribute(C, filter_attributes={"origin": "B"})
        A_prime_nids = filter_nids_by_attribute(C, filter_attributes={"origin": "B"})
        A_prime = C.edge_subgraph(A_prime_eids)
        
        C_prime = C.edge_subgraph(set([eid for eid, _ in iterate_edges(C)]) - set(A_prime_eids))

        # Sanity check that A_prime contains exactly those nodes in A_prime_nids.
        hits = [connection[1] for connection in connections] # Those nodes endpoints of C connected to injected A edges.
        check(set(A_prime_nids).union(hits) == set(A_prime.nodes()), expect="Expect nids subgraph from C to match those nodes attributed with origin of B.")

        ## Find edges of C which are covered by the injected B edges (which are all of them in A_prime) and then remove them.

        # Reset coverage threshold information on a graph.
        def clear_coverage_threshold_information(G):
            G.graph.pop("max_threshold") 
            for eid, attrs in iterate_edges(G):
                if "threshold" in attrs:
                    attrs.pop("threshold")

        # Reset coverage threshold information on C.
        clear_coverage_threshold_information(C_prime)
        C_prime_covered_by_A_prime = edge_graph_coverage(C_prime, A_prime, max_threshold=prune_threshold)

        # Obtain edges of C covered (or uncovered) by edges of A_prime
        above = [eid for eid, attrs in iterate_edges(C_prime_covered_by_A_prime) if attrs["threshold"] >  prune_threshold]
        below = [eid for eid, attrs in iterate_edges(C_prime_covered_by_A_prime) if attrs["threshold"] <= prune_threshold]
        edges_to_be_deleted = below

        # Obtain nodes to be deleted from C (those nodes which are part of covered edges but of no uncovered edge).
        nodes_above = set([nid for el in above for nid in el[0:2]]) 
        nodes_below = set([nid for el in below for nid in el[0:2]]) 
        nodes_to_be_deleted = nodes_below - nodes_above

        # Mark edges that are deleted.
        render_update = {}
        for eid in edges_to_be_deleted:
            render_update[eid] = {**C_covered_by_B.edges[eid], "render": "deleted"}
        nx.set_edge_attributes(C, render_update)
        annotate_edges(C, {"render": "deleted"}, eids=edges_to_be_deleted)
        # C.remove_edges_from(edges_to_be_deleted)

        # Delete nodes (Mark nodes for deletion).
        annotate_nodes(C, {"render": "deleted"}, nids=nodes_to_be_deleted)
        # C.remove_nodes_from(nodes_to_be_deleted)

    # TODO: Extension b: Reconnect edges of C to injected edges of A into B.
    if reconnect_after:

        # We require a list of removed sat edges (or what sat edges have a sat edge removed) _and_ a list of injected gps edges (B_prime).
        #   Then we can find what sat nodes have a missing sat edge, _and_ what collection of (gps) edges it can be connected to.

        # Rules for sat edges that have to be reconnected:
        # * C edge is above threshold.
        # * C edge is connected an A edge which is below threshold.
        # * C edge is not contained in the set of injected A edges.

        # Obtain sat nodes which are to be connected to injected GPS edges.
        nodes_above = set([nid for el in above for nid in el[0:2]]) 
        nodes_below = set([nid for el in below for nid in el[0:2]]) 
        nodes_to_connect = set(nodes_above) & set(nodes_below)
        
        # # Find edges of sat to reconnect to injected gps edge.
        # connect_edges = [] # (eid and endpoint)
        # nodes_to_connect = 
        # for u, v in C_covered_by_B.edges(data=True):
        #     if nid in nodes_below and nid in nodes_above:
        #         connect_node()
        #         # We find a node at which a covered edge is connected to an uncovered edge.
        #         # Pick edges which are above threshold.
        #         candidates = [eid for eid in above if eid[0] == nid or eid[1] == nid]
        #         # Intersect edge with injected gps edge.

        #         merge_nearest_edge()
        #         connect_nodes.append(nid)

        # Apply nearest_edge injection.
        for nid in nodes_to_connect:
            connect_nearest_edge(C, nid)

    # Convert back graph to latlon coordinates if necessary.
    if _C.graph["coordinates"] == "latlon":
        utm_info = graph_utm_info(_C)
        C = graph_transform_utm_to_latlon(C, "", **utm_info) 

    return C


def connect_nearest_edge(G, nid):
    position = G.nodes(data=True)[nid]
    y, x = position["y"], position["x"]

    # Find nearest node.
    hit, distance = osmnx.distance.nearest_nodes(G, x, y, return_dist=True)
    if distance < 10: # simply connect to node.
        if not G.graph["simplified"]:
            G.add_edges_from([(nid, hit, {"render": "connection"})])
        else:
            y2, x2 = G._node[hit]['y'], G._node[hit]['x'],
            curvature = array([(y, x), (y2, x2)])
            geometry = to_linestring(curvature)
            G.add_edges_from([(nid, hit, {"render": "connection", "geometry": geometry, "curvature": curvature})])
        return G

    # TODO: Optimization: Prefilter out edges within `distance` bounding box.
    # relevant_edges = [] # [(eid, attrs)]
    # Convert edges into curve_points, eid
    curves = G
        # TODO: Link eids to identifiers.
    # TODO: Find nearest edge and point of intersection.
    # TODO: Cut nearest edge at point of intersection, connect new node, resolve.
    nid = max(G.nodes()) + 1 
    return G


# Functionality for merging an edge with the `nearest_edge` strategy.
# Inject edge at endpoint with G.
# Find edge in G that is most suitable for injection.
# Apply 
def merge_nearest_edge(G, nidstart, edge, endpoint):
    nid = max(G.nodes()) + 1
    # Naive approach: 
    # Take curvature 