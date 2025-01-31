from external import *
from utilities import *
from network import *
from graph_coordinates import *

###  Curve by curve coverage

# Curve coverage of ps by qs.
# either return false or provide subcurve with step sequence
# TODO: Optimization to check on bounding boxes before doing the interpolation.
def curve_by_curve_coverage(ps, qs, lam):
    return is_partial_curve_undirected(curve_to_vector_list(ps), curve_to_vector_list(qs), lam)


# Check coverage of a curve by a curve-set.
def curve_by_curveset_coverage(ps, qss, lam):
    for qs in qss:
        if is_partial_curve_undirected(curve_to_vector_list(ps), curve_to_vector_list(qs), lam):
            return True
    return False


# Obtain edges covered by specific node.
def edges_covered_by_nid(G, nid, threshold):

    # Find nearby edges.
    edge_tree = graphedges_to_rtree(G)
    node_bbox = graphnode_to_bbox(G, nid, padding=threshold)
    nearby_eids = intersect_rtree_bbox(edge_tree, node_bbox)

    # Obtain max distance.
    point = graphnode_position(G, nid)

    # Max distance per edge to node.
    distances = [max([norm(vec) for vec in graphedge_curvature(G, eid) - point]) for eid in nearby_eids]

    # Return those below threshold.
    eids_below_threshold = [eid for eid, distance in zip(nearby_eids, distances) if distance <= threshold]

    return eids_below_threshold


###  Curve by network coverage

# Obtain threshold per simplified edge of S in comparison to T.
@info()
def edge_graph_coverage(S, T, max_threshold=None): 

    S = S.copy()

    # Sanity check the graph is simplified.
    check(S.graph["simplified"], expect="Expect the source graph is simplified" \
                                        ", we do not compute edge coverage for line segments" \
                                        ", nor do we want to restore (vectorized graph with proper edge threshold attribute annotation) in this function.")

    # Sanity checks each edge has a threshold set..
    check("threshold" not in S.graph, expect="Expect the graph to not have a 'max_threshold' attribute set.")
    for _, attrs in iterate_edges(S):
        check("threshold" not in attrs, expect="Expect edge in source to not have the 'threshold' attribute set" \
                                               ", because such existence suggests we are overwriting a previous coverage check" \
                                               ", suggesting some coverage computation is accidentally out of place.")

    # Make sure both source and target are in UTM coordinates (for threshold to make sense).
    convert_to_utm = S.graph["coordinates"] != "utm"

    if S.graph["coordinates"] != "utm":
        utm_info = graph_utm_info(S)
        S = graph_transform_latlon_to_utm(S)
    if T.graph["coordinates"] != "utm":
        T = graph_transform_latlon_to_utm(T)

    # We allow target to be vectorized, it causes no loss of information (since target is not being adjusted).
    was_simplified = T.graph["simplified"]
    if T.graph["simplified"]:
        T = vectorize_graph(T)

    # Threshold computation iteration variables.
    leftS  = set([eid for eid, _ in iterate_edges(S)]) # Edges we seek a threshold value for.
    lam    = 1 # Start with a threshold of 1 meter.
    thresholds = {} # Currently found thresholds.
    covered_by = {} # Track (collection of) edges of T which covers the edge of S.
    
    # Link a curve to every simplified edge.
    curves = {}
    for eid in leftS:
        ps = get_edge_attributes(S, eid)["curvature"]
        curve = curve_to_vector_list(ps)
        curves[eid] = curve
    
    ## Performance: Construct graph per edge (subgraph with nodes in `threshold` meter radius to edge curvature).
    graph_annotate_edge_curvature(T)
    graph_annotate_edge_curvature(S)
    edge_tree = graphedges_to_rtree(T)
    edge_bboxs = graphedges_to_bboxs(S, padding=max_threshold)
    subgraphs = {}
    # Per simplified edge of S, construct a subgraph of nearby edges of T.
    for eid in leftS:
        
        # Obtain nearby node identifiers.
        nearby_eids = intersect_rtree_bbox(edge_tree, edge_bboxs[eid])

        # Extract subgraph.
        subgraph = T.edge_subgraph(nearby_eids)

        # Convert the subgraph a rust graph.
        subgraph = graph_to_rust_graph(subgraph)
        
        # Store.
        subgraphs[eid] = subgraph
    
    # Sanity check subgraphs make sense.
    node_tree = graphnodes_to_rtree(S)
    edge_bboxs = graphedges_to_bboxs(S, padding=1)
    for eid in leftS:
        nearby_nids = intersect_rtree_bbox(node_tree, edge_bboxs[eid])
        eids = set(flatten([get_connected_eids(S, nid) for nid in nearby_nids]))
        check(eid in eids, expect="Expect the eid to be present within the set: The edges adjacent to nodes captured by the edgebbox of eid.")

    # Increment threshold and seek nearby path till all edges have found a threshold (or max threshold is reached).
    while len(leftS) > 0 and (max_threshold == None or lam <= max_threshold):
        logger(f"Lambda: {lam}. Edges: {len(leftS)}")

        for eid in leftS:
            curve = curves[eid]
            subgraph = subgraphs[eid]
            path = partial_curve_graph(subgraph, curve, lam)

            # Annotate threshold to edge if applicable.
            if path != None:

                # Remove edge from edge set.
                leftS = leftS - set([eid]) 
                # Save threshold to apply later.
                thresholds[eid] = lam
                # Store the path (edge identifiers) which curvature is used to cover this edge of S.
                # NOTE: Path is a sequence of node identifiers traversed. This represents thereby as well the traversed edges (and thus the curvature).
                #       These eids are always `(u, v)` because T is vectorized at this point.
                #       If T was originally a simplified graph, we will reconstruct the simplified edges involved at the end of this function
                covered_by[eid] = list(zip(path[:-1], path[1:]))

        lam += 1 # Increment lambda.

    # Set unprocessed edges to have infinite threshold and no coverage edge identifiers.
    for eid in leftS:
        thresholds[eid] = inf
        covered_by[eid] = []
    
    # If T was simplified at input.
    if was_simplified:
        # Then transform the "covered_by" of vectorized edges into its simplified edges origin.

        new_covered_by = {}
        for S_eid, _ in iterate_edges(S): # We want to annotate every edge of S.

            new_covered_by[S_eid] = set()

            for T_eid in covered_by[S_eid]: # We extract simplified edges from all vectorized edges that participated.

                related_simplified_edge = get_edge_attributes(T, T_eid)["vectorized_from"]
                new_covered_by[S_eid] = new_covered_by[S_eid].union(set([related_simplified_edge]))
        
        covered_by = new_covered_by

    # Set threshold and covered_by for each edge.
    nx.set_edge_attributes(S, {eid: {**attrs, "threshold": thresholds[eid], "covered_by": covered_by[eid]} for eid, attrs in iterate_edges(S)}) 

    # Restore graph to input state.
    if convert_to_utm:
        S = graph_transform_utm_to_latlon(S, "", **utm_info) # Convert back into latlon.

    # Apply threshold annotation.
    S.graph['max_threshold'] = max_threshold # Mention till what threshold we have searched.

    return S

