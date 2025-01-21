from external import *

from graph_node_extraction import *
from graph_curvature import *

# Drop self-loops and multi-edges from graph.
def graph_sanitize_simplified_edges(G):

    if not G.graph["simplified"]:
        return G
    
    G = G.copy()

    self_loops = [(u, v, k) for (u, v, k), _ in iterate_edges(G) if u == v]
    multi_edges = [(u, v, k) for (u, v, k), _ in iterate_edges(G) if u != v and k > 0]

    # Drop self-loops and multi-edges.
    log(f"Dropping {len(self_loops)} self-loops.")
    log(f"Dropping {len(multi_edges)} self-loops.")
    G.remove_edges_from(self_loops)
    G.remove_edges_from(multi_edges)

    # Sanity check that we have multi-edge at `k == 0`.
    for (u, v, k) in multi_edges:
        edge = get_edge(G, (u, v, 0))

    return G


# Wrapping the OSMnx graph simplification logic, and being consistent in MultiGraph convention by converting between directed and undirected.
def simplify_graph(G):
    assert not G.graph["simplified"] 
    G = ox.simplify_graph(nx.MultiGraph(G).to_directed(), track_merged=True).to_undirected()
    G.graph["simplified"] = True
    G = graph_sanitize_simplified_edges(G)
    graph_annotate_edge_curvature(G)
    graph_correctify_edge_curvature(G)
    graph_annotate_edge_length(G)
    graph_annotate_edge_geometry(G)
    return G


# Vectorize a network.
def vectorize_graph(G):
    assert G.graph["simplified"] 

    G = G.copy()

    if not G.graph.get("simplified"):
        msg = "Graph has to be simplified in order to vectorize it."
        raise BaseException(msg)
    
    if not type(G) == nx.MultiGraph:
        msg = "Graph has to be MultiGraph (undirected but potentially multiple connections) for it to work here."
        raise BaseException(msg)

    # Extract nodes and edges.
    nodes = extract_nodes_dict(G)
    edges = np.array(list(G.edges(data=True, keys=True)))

    # Obtain unique (incremental) node ID to use.
    newnodeid = max(G.nodes()) + 1 # Move beyond highest node ID to ensure uniqueness.

    for (a, b, k, attrs) in edges:

        # We only have to perform work if an edge contains curvature. (If there is no geometry component, there is no curvature to take care of. Thus already vectorized format.)
        if len(attrs["curvature"]) > 2:

            # Delete this edge from network.
            G.remove_edge(a,b,k)
            # print("Removing edge ", a, b, k)

            # Add curvature as separate nodes/edges.
            linestring = attrs["geometry"]
            ps = array([(y, x) for (x, y) in list(linestring.coords)])

            # Sanity checks. 
            # try:
            assert np.all(array(ps[0]) == array(nodes[a])) or np.all(array(ps[-1]) == array(nodes[a])) # Geometry starts at first node coordinate.
            assert np.all(array(ps[0]) == array(nodes[b])) or np.all(array(ps[-1]) == array(nodes[b])) # Geometry ends at last node coordinate.
            assert len(ps) >= 1 # We expect at least one point in between start and end node.
            # except Exception as e:
            #     print(traceback.format_exc())
            #     print(e)
            #     breakpoint()

            # Drop first and last point because these are start and end node..
            ps = ps[1:-1]

            # Ensured we are adding new curvature.  Add new node ID to each coordinate.
            pathcoords = list(ps)
            sequential = np.all(array(ps[0]) == array(nodes[a]))
            if not sequential:
                pathcoords.reverse()
                
            pathids = list(range(newnodeid, newnodeid + len(ps)))
            newnodeid += len(ps) # Increment id appropriately.

            for node, coord in zip(pathids, pathcoords):
                G.add_node(node, y=coord[0], x=coord[1])

            pathids = [a] + pathids + [b]
            for a,b in zip(pathids, pathids[1:]):
                G.add_edge(a, b, 0) # Key can be zero because nodes in curvature implies a single path between nodes.

    G.graph["simplified"] = False # Mark the graph as no longer being simplified.
    G = nx.Graph(G)

    return G
