from external import *
from graph_node_extraction import *
from utilities import *

# Group duplicated nodes.
def duplicated_nodes(G, eps=0.001):

    positions = extract_nodes_dict(G)
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

    # Convert regions into lists.
    return [list(region) for region in regions]


# Deduplicates a graph. Reconnects edges of removed nodes (if any). 
@info()
def deduplicate(G):

    G = G.copy()

    duplicated_groups = duplicated_nodes(G)

    # Initiate all nodes to link to themselves (used for rebinding edges).
    nid_relink = {}
    for nid in G.nodes():
        nid_relink[nid] = nid

    # Link all nids to their unique target nid.
    duplicated_nids = {nids[0]: nids[1:] for nids in duplicated_groups}
    for target in duplicated_nids.keys():
        nid_relink[target] = target
        for source in duplicated_nids[target]:
            nid_relink[source] = target

    # Delete duplicated nodes.
    nids_to_delete = set(flatten([nids[1:] for nids in duplicated_groups]))
    logger(f"Dropping {len(nids_to_delete)} duplicated node positions.")
    G.remove_nodes_from(nids_to_delete)

    # Link (partially) dangling edges to connect between the remaining nodes.
    new_edges = [(nid_relink[u], nid_relink[v]) for u, v in G.edges() if nid_relink[u] != u or nid_relink[v] != v]
    logger(f"Reconnecting {len(new_edges)} edges.")
    G.add_edges_from(new_edges)

    return G
