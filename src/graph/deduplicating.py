from external import *
from graph.node_extraction import *
from graph.curvature import *
from graph.coordinates import *
from utilities import *

# Group duplicated nodes.
def duplicated_nodes(G, eps=0.001):

    # Make sure to act on UTM coordinated graph (to make sense of epsilon).
    is_transformed = False
    if G.graph["coordinates"] != "utm":
        is_transformed = True
        place = graph_utm_place(G)
        G = graph_transform_latlon_to_utm(G)

    positions = extract_node_positions_dictionary(G)
    tree = graphnodes_to_rtree(G)
    bboxs = graphnodes_to_bboxs(G)

    regions = [] # Collection of nodes which are adjacent.

    # Nodes are iterated incrementally.
    for nid in G.nodes():

        # Find nearby nodes.
        bbox = pad_bounding_box(bboxs[nid], eps)
        nids = set(intersect_rtree_bbox(tree, bbox))

        # Clusters require at least two elements.
        if len(nids) == 1:
            continue

        # Check some node already belongs to an existing cluster. 
        has_found = False
        for i, region in enumerate(regions):
            # If such a cluster consist.
            if len(nids & region) > 0:
                # Then add all elements to this cluster.
                regions[i] = region.union(nids)
                break
            
        # Otherwise if we haven't found a cluster it partially belongs to.
        if not has_found:
            # Then we found a new isolated cluster.
            regions.append(nids)
    
    if is_transformed:
        G = graph_transform_utm_to_latlon(G, place)

    # Convert regions into lists.
    return [list(region) for region in regions]


# Deduplicates a graph. Reconnects edges of removed nodes (if any). 
@info()
def graph_deduplicate(G, eps=0.001):

    check(not G.graph["simplified"])
    check(G.graph["coordinates"] == "utm")

    G = G.copy()

    length = graph_length(G)

    duplicated_groups = duplicated_nodes(G, eps=eps)

    # Initiate all nodes to link to themselves (used for rebinding edges).
    nid_relink = {}
    for nid in G.nodes():
        nid_relink[nid] = nid

    # Link all nids to their unique target nid.
    for nids_group in duplicated_groups:
        source = nids_group[0]
        for nid in nids_group[1:]:
            nid_relink[nid] = source

    # Obtain duplicated nodes to delete and edges to reconnect.
    new_edges = []
    for eid, _ in iterate_edges(G):
        u, v = eid[:2]
        if nid_relink[u] != u or nid_relink[v] != v:
            new_edges.append((nid_relink[u], nid_relink[v]))
    nids_to_delete = set(flatten([nids[1:] for nids in duplicated_groups]))

    logger(f"Dropping {len(nids_to_delete)} duplicated node positions.")
    G.remove_nodes_from(nids_to_delete)

    # Link (partially) dangling edges to connect between the remaining nodes.
    logger(f"Reconnecting {len(new_edges)} edges.")
    G.add_edges_from(new_edges)

    # Sanity check that total graph edge length remains the same after removing duplicated nodes.
    graph_annotate_edge_curvature(G)
    new_length = graph_length(G)
    check(abs(length - new_length) <= len(nids_to_delete) * eps, expect="Expect total graph edge length remains the same after removing duplicated nodes.")

    return G
