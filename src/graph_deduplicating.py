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


# Deduplicates a vectorized graph. Reconnects edges of removed nodes (if any). 
def deduplicate(G):

    assert not G.graph["simplified"]

    G = G.copy()

    # Trash edges of duplicated nodes (we can do so since it is vectorized).
    nodes_to_delete = []
    edges_to_delete = []
    edges_to_insert = []
    for node_group in duplicated_nodes(G):
        assert len(node_group) > 1
        first = node_group[0]
        for nid in node_group[1:]:

            nodes_to_delete.append(nid)

            # Obtain edges to delete.
            old_edges = G.edges(nid) # Edges connected to node that is about to be removed.
            edges_to_delete.extend(old_edges) 

            # Convert edge endpoint to `first` node identifier.
            new_edges = [(first, edge[1]) for edge in list(old_edges)] # Reconnect edge to the remaining node.
            edges_to_insert.extend(new_edges)
        
    # print("Edges to delete: ", edges_to_delete)
    G.remove_edges_from(edges_to_delete)
    # print("Number of edges after deletion of edges: ", len(G.edges))
    G.add_edges_from(edges_to_insert)
    # print("Number of edges after insertion of edges: ", len(G.edges))
    G.remove_nodes_from(nodes_to_delete)
    # print("Number of edges after deletion of nodes: ", len(G.edges))
    # print("Number of nodes after deletion of nodes: ", len(G.nodes))
    # print(duplicated_nodes(G))
    # print(duplicated_edges(G))

    return G

