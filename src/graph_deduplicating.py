from external import *
from graph_node_extraction import *
from utilities import *

# Group duplicated nodes.
def duplicated_nodes(G, eps=0.0001):

    positions = extract_nodes_dict(G)
    tree = graphnodes_to_rtree(G)
    bboxs = graphnodes_to_bboxs(G)

    duplicated = []

    # Nodes are iterated incrementally.
    for nid in G.nodes():

        # Find nearby nodes.
        bbox = pad_bounding_box(bboxs[nid], eps)
        nids = sorted(intersect_rtree_bbox(tree, bbox))

        # All nodes intersect at least once (namely with itself).
        # If more intersections occur, we group the duplicates under the lowest node identifier.
        if len(nids) > 1 and nids[0] == nid:
            duplicated.append(nids)
    
    return duplicated


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

