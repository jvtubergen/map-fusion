from external import *


# Group duplicated nodes.
def duplicated_nodes(G):

    # NOTE: Keep IDs (integeters) separate from coordinates (floats): Numpy arrays all have same type.
    node_ids = G.nodes()
    coordinates = np.array([[info["y"], info["x"]] for node, info in G.nodes(data=True)])
    uniques, inverses, counts = np.unique( coordinates, return_inverse=True, axis=0, return_counts=True )

    # Construct dictionary.
    duplicated = {}
    for node_id, index_to_unique in zip(node_ids, inverses):
        if counts[index_to_unique] > 1:
            if index_to_unique in duplicated.keys():
                duplicated[index_to_unique].append(node_id)
            else:
                duplicated[index_to_unique] = [node_id]

    # Convert dictionary into a list.
    result = []
    for v in duplicated:
        result.append(duplicated[v])

    return result


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
            old_edges = G.edges(nid)
            edges_to_delete.extend(old_edges)

            # Convert edge endpoint to `first` node identifier.
            new_edges = [(first, edge[1]) for edge in list(old_edges)]
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

