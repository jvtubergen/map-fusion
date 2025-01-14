from external import *

from graph_node_extraction import *
from graph_curvature import *




# Ensures each simplified edge has "geometry" and "curvature" attribute set.
def consolidate_edge_geometry_and_curvature(G):

    assert G.graph["simplified"]

    G = G.copy()

    # Edges contain curvature information.
    edge_attrs = {}
    for (a, b, k, attrs) in G.edges(data=True, keys=True):
        # print(a, b, attrs)

        # Obtain "geometry" and "curvature" attribute.
        if "geometry" in attrs.keys():
            geometry = attrs["geometry"]
            curvature = from_linestring(geometry)
        else:
            # No curvature in edge, thus a straight line segment.
            p1 = G.nodes()[a]
            p2 = G.nodes()[b]
            latlon1 = p1["y"], p1["x"]
            latlon2 = p2["y"], p2["x"]
            curvature = array([latlon1, latlon2])
            geometry = to_linestring(curvature)
        
        # Sanity check to always have sensible curvature.
        assert len(curvature) >= 2
        
        # Add both attributes to the edge.
        edge_attrs[(a, b, k)] = {**attrs, "curvature": curvature, "geometry": geometry}
    
    nx.set_edge_attributes(G, edge_attrs)

    return G


# Wrapping the OSMnx graph simplification logic, and being consistent in MultiGraph convention by converting between directed and undirected.
def simplify_graph(G):
    assert not G.graph["simplified"] 
    G = ox.simplify_graph(nx.MultiGraph(G).to_directed(), track_merged=True).to_undirected()
    G = graph_correctify_edge_curvature(G)
    G.graph["simplified"] = True
    G = consolidate_edge_geometry_and_curvature(G)
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
