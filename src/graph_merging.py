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

    # We have to simplify this graph again as some crossroads may have disappeared.
    G = simplify_graph(G, retain_attributes=True)

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

    ## Add edge connections between B and C.

    # We want to connect edge endpoints (of B) to arbitrary node/edge of C.
    # Therefore the node and edge tree we seek to search for are defined by C only.
    node_tree = graphnodes_to_rtree(C)
    edge_tree = graphedges_to_rtree(C)

    excluded_eids = set(get_eids(B))
    excluded_nids = set(get_nids(B))

    # Iterate all nids which have to be connected (from B to C).
    connections = []
    logger("Connecting nodes.")
    for nid in connect_nodes:

        # Add new edge connection from nid (potentially cuts an edge in half).
        new_eid, injection_data = reconnect_node(C, nid, node_tree=node_tree, edge_tree=edge_tree, excluded_nids=excluded_nids, excluded_eids=excluded_eids.union(connections)) 
        set_edge_attributes(C, new_eid, {"render": "connection", "origin": "B"})
        connections.append(new_eid)

        # If we had to inject a node (to C) for a connection.
        if injection_data != None:

            # Then update the node-tree and edge-tree.
            new_nid, new_eids, old_eid = injection_data["new_nid"], injection_data["new_eids"], injection_data["old_eid"]

            # Update attributes of injected node as well.
            set_node_attributes(C, new_nid, {"render": "connection", "origin": "C"})
            # (Note: No need to set render attribute of injected edges, those already have been copied over from the deleted edge in the `reconnect_node` function.)

            # (Updating node tree.)
            bbox = graphnode_to_bbox(C, new_nid)
            add_rtree_bbox(node_tree, bbox, new_nid)

            # (Insert new subedges to edge tree.)
            add_rtree_bbox(edge_tree, graphedge_to_bbox(C, new_eids[0]), new_eids[0])
            add_rtree_bbox(edge_tree, graphedge_to_bbox(C, new_eids[1]), new_eids[1])

            # Exclude removed eid from intersection set.
            # (Don't remove old edge from edge tree: Bug in rtree software causing error on subsequent hits.)
            excluded_eids.add(old_eid)

    # Correctify edge curvature.
    graph_correctify_edge_curvature(C)

    graphs = {
        "a": C.copy()
    }
    
    # Extension a: Remove duplicated edges of C.
    if remove_duplicates: 

        # Update B: It now includes connection edges (the injected subgraph of A with connection edges to C).
        B_eids = set(filter_eids_by_attribute(C, filter_attributes={"origin": "B"}))
        B_eids = B_eids.union(set(filter_eids_by_attribute(C, filter_attributes={"render": "connection"})))
        B_eids = list(B_eids)

        B_nids = set(filter_nids_by_attribute(C, filter_attributes={"origin": "B"}))
        B_nids = B_nids.union(set(filter_nids_by_attribute(C, filter_attributes={"render": "connection"})))
        B_nids = list(B_nids)

        B = C.edge_subgraph(B_eids)

        # Sanity check that B contains exactly those nodes in B_nids.
        hits = [nid for connection in connections for nid in connection[:2]] # Those nodes endpoints of C connected to injected A edges.
        check(set(B_nids).union(hits) == set(get_nids(B)), expect="Expect nids subgraph from C to match those nodes attributed with origin of B.")
        
        # Obtain C graph before we injected edges.
        C_original = C.edge_subgraph(set([eid for eid, _ in iterate_edges(C)]) - set(B_eids))

        ## Find edges of C which are covered by the injected B edges (which are all of them in B) and then remove them.

        # Reset coverage threshold information on a graph.
        def clear_coverage_threshold_information(G):
            G.graph.pop("max_threshold") 
            for eid, attrs in iterate_edges(G):
                if "threshold" in attrs:
                    attrs.pop("threshold")

        # Reset coverage threshold information on C.
        clear_coverage_threshold_information(C_original)
        C_original_covered_by_B = edge_graph_coverage(C_original, B, max_threshold=prune_threshold)

        # Obtain edges of C covered (or uncovered) by edges of B
        above = [eid for eid, attrs in iterate_edges(C_original_covered_by_B) if attrs["threshold"] >  prune_threshold]
        below = [eid for eid, attrs in iterate_edges(C_original_covered_by_B) if attrs["threshold"] <= prune_threshold]
        edges_to_be_deleted = below

        # Obtain nodes to be deleted from C (those nodes which are part of covered edges but of no uncovered edge).
        nodes_above = set([nid for eid in above for nid in eid[0:2]]) 
        nodes_below = set([nid for eid in below for nid in eid[0:2]]) 
        nodes_to_be_deleted = nodes_below - nodes_above

        # Exclude edges covered by connection nodes: These edges are unrelated to curvature of injected node.
        B_eids = filter_eids_by_attribute(C, filter_attributes={"origin": "B"})
        subgraph = C.edge_subgraph(B_eids)
        connect_nodes = [nid for nid, _ in iterate_nodes(subgraph) if subgraph.degree[nid] == 1] # Nodes of B with a degree of 1.
        edges_to_ignore = [eid for nid in connect_nodes for eid in edges_covered_by_nid(C, nid, prune_threshold)] # Edges of C covered by these connect nodes.

        # Mark edges for deletion.
        annotate_edges(C, {"render": "deleted"}, eids=list(set(edges_to_be_deleted) - set(edges_to_ignore)))

        # Delete nodes (Mark nodes for deletion).
        annotate_nodes(C, {"render": "deleted"}, nids=list(nodes_to_be_deleted))
    
    graphs["b"] = C.copy()
    
    # Extension b: Reconnect edges of C to injected edges of A into B.
    if reconnect_after:

        # Obtain the graph of C with B edges injected and duplicated C edges removed.
        # This means dropping edges with attribute `{"render": "deleted"}`.
        C_original_eids = filter_eids_by_attribute(C, filter_attributes={"origin": "C"})
        C_current_eids = filter_eids_by_attribute(C, filter_func=lambda attrs: attrs["render"] != "deleted")
        C_deleted_eids = list(set(get_eids(C)) - set(C_current_eids))

        check(set(C_deleted_eids) == set(filter_eids_by_attribute(C, filter_func=lambda attrs: attrs["render"] == "deleted")), expect="Expect all edges consist of union of disjoint deleted/not deleted edges.")

        C_current = C.edge_subgraph(C_current_eids)

        # Node properties for reconnection:
        # * Connected to at least one edge of C (so it has not been removed yet).
        # * Connected to two deleted edges that had coverage by the same eid (so a continuous edge of A replaced it, thus a reconnection necessary).
        
        # Find nodes connected to both original edges of C of which one is deleted.
        nids_connected_to_C_deleted_eids  = set([nid for eid in C_deleted_eids for nid in eid[0:2]]) 
        nids_connected_to_C_original_eids = set([nid for eid in C_original_eids for nid in eid[0:2]]) 
        relevant_nids = nids_connected_to_C_deleted_eids & nids_connected_to_C_original_eids

        # Connected to two deleted edges covered by the same eid.
        nids_to_reconnect = []
        for nid in relevant_nids:
            
            # Count number of original edges of C which have been deleted.
            deleted_eids = [eid for eid in C_deleted_eids if nid in set(eid[:2])]
            if len(deleted_eids) < 2:
                continue

            # Check whether same "covered_by" occurs.
            #   We check all combinations of the removed edges, to see if any continuous road got removed due to the same injected edge.
            found_something = False
            for eid1, eid2 in combinations(deleted_eids, 2):
                #   Check whether the deleted edges of C were _both_ covered by the same injected edge.
                intersection = get_edge_attributes(C_original_covered_by_B, eid1)["covered_by"] & get_edge_attributes(C_original_covered_by_B, eid2)["covered_by"]
                if len(intersection) > 0:
                    found_something = True
            
            if not found_something:
                continue
            
            # If so, reconnect nid
            nids_to_reconnect.append(nid)

        # Filter out nodes for reconnection which have not been deleted themselves.
        C_deleted_nids = filter_nids_by_attribute(C, filter_attributes={"render": "deleted"})
        nids_to_reconnect = list(set(nids_to_reconnect) - set(C_deleted_nids))

        logger("Points to reconnect: ", nids_to_reconnect)

        # Reconnect these (original nodes of C) to the injected nodes of B.
        edge_tree = graphedges_to_rtree(C)
        node_tree = graphnodes_to_rtree(C)

        excluded_eids = set(get_eids(C)) - set(filter_eids_by_attribute(C, filter_attributes={"origin": "B"}))
        excluded_nids = set(get_nids(C)) - set(filter_nids_by_attribute(C, filter_attributes={"origin": "B"}))

        for nid in nids_to_reconnect:
            sanity_check_graph_curvature(C)
            new_eid, injection_data = reconnect_node(C, nid, edge_tree=edge_tree, node_tree=node_tree, excluded_eids=excluded_eids, excluded_nids=excluded_nids, update_trees=True)

            # Updat new eid with rendering.
            set_edge_attributes(C, new_eid, {"render": "connection", "origin": "C"})

            if injection_data != None:

                # Update render attributes on injected elements.
                new_nid, new_eids, old_eid = injection_data["new_nid"], injection_data["new_eids"], injection_data["old_eid"]
                set_node_attributes(C, new_nid    , {"render": "connection", "origin": "B"})
                # (Note: No need to set render attribute of injected edges, those already have been copied over from the deleted edge in the `reconnect_node` function.)

                # Performance: Update reconnection data.
                edge_tree = injection_data["edge_tree"]
                node_tree = injection_data["node_tree"]
                excluded_eids = injection_data["excluded_eids"]
    
    graphs["c"] = C.copy()
    
    # Convert back graph to latlon coordinates if necessary.
    if _C.graph["coordinates"] == "latlon":
        utm_info = graph_utm_info(_C)
        C = graph_transform_utm_to_latlon(C, "", **utm_info) 

    return graphs


# Reconnect node to graph.
# * Returns inject eid.
# * Optionally allow only to reconnect to a subselection of nodes and/or edges.
# * Pass forward node_tree and edge_tree to significantly improve performance.
@info()
def reconnect_node(G, nid, node_tree=None, edge_tree=None, nid_distance=10, excluded_nids=set(), excluded_eids=set(), update_trees=False):

    # Injection data (Set if we cut edge).
    injection_data = None

    # Find nearest node.
    hit = nearest_node(G, nid, node_tree=node_tree, excluded_nids=excluded_nids)
    eid = nearest_edge(G, nid, edge_tree=edge_tree, excluded_eids=excluded_eids)

    # Check distance below `nid_distance`.
    hit_distance = graph_distance_node_node(G, nid, hit)
    eid_distance = graph_distance_node_edge(G, nid, eid)

    # If eid is significantly more nearby than hid.
    if hit_distance - eid_distance > nid_distance:

        ## Then add a cutpoint to the edge and use that cutpoint as the connection point.

        # Find curve interval nearest to nid.
        point = graphnode_position(G, nid)
        attrs = get_edge_attributes(G, eid)
        curve = attrs["curvature"]
        interval = nearest_interval_on_curve_to_point(curve, point)
        # check(interval > 0.001 and interval < 0.999, expect="Expect to have interval not nearby edge endpoints, ")

        # Cut at interval.
        G, data = graph_cut_edge_intervals(G, eid, [interval])
        new_nid  = data["nids"][0]
        new_eids = [format_eid(G, eid) for eid in data["eids"]]

        # Annotate both subedges with original edge attributes (`graph_cut_edge_intervals` has already annotated curvature, geometry, length).
        nx.set_edge_attributes(G, {new_eid: {**attrs, **get_edge_attributes(G, new_eid)} for new_eid in new_eids})
        graph_correctify_edge_curvature_single(G, new_eids[0])
        graph_correctify_edge_curvature_single(G, new_eids[1])

        hit = new_nid

        # Set injection data for post-processing in caller.
        injection_data = {
            "new_nid": new_nid,
            "new_eids": new_eids,
            "old_eid": eid
        }
    
    # Inject new edge and annotate it.
    eid = format_eid(G, (nid, hit)) # Format eid to with/without key (thus double or triplet).

    # Ensure injected edge has curvature etcetera set.
    G.add_edge(*eid)
    graph_annotate_edge(G, eid)
    graph_correctify_edge_curvature_single(G, eid)

    # Optionally update tree and node exclusion information if a node is injected to nearest edge.
    if update_trees and injection_data != None:

        # Then update the node-tree and edge-tree.
        new_nid, new_eids, old_eid = injection_data["new_nid"], injection_data["new_eids"], injection_data["old_eid"]

        # (Updating node tree.)
        bbox = graphnode_to_bbox(G, new_nid)
        add_rtree_bbox(node_tree, bbox, new_nid)

        # (Insert new subedges to edge tree.)
        add_rtree_bbox(edge_tree, graphedge_to_bbox(G, new_eids[0]), new_eids[0])
        add_rtree_bbox(edge_tree, graphedge_to_bbox(G, new_eids[1]), new_eids[1])

        # Exclude removed eid from intersection set.
        # (Don't remove old edge from edge tree: Bug in rtree software causing error on subsequent hits.)
        excluded_eids.add(old_eid)
    
        injection_data["edge_tree"] = edge_tree
        injection_data["node_tree"] = node_tree
        injection_data["excluded_eids"] = excluded_eids

    return eid, injection_data
