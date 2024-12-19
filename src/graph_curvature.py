from external import *
from graph_simplifying import *
from graph_coordinates import *
from graph_deduplicating import *
from graph_node_extraction import *
from utilities import *


# Add path length data element.
def graph_annotate_edge_length(G):

    assert G.graph["simplified"]

    G = G.copy()

    edge_attrs = {}
    for (a, b, k, attrs) in G.edges(data=True, keys=True):

        ps = attrs["curvature"]
        length = curve_length(ps)
        assert length > 0 # Assert non-zero length.
        edge_attrs[(a, b, k)] = {**attrs, "length": length}
    
    nx.set_edge_attributes(G, edge_attrs)
    return G


# Extract curvature (stored potentially as a LineString under geometry) from an edge as an array.
def edge_curvature(G, u, v, k = None):

    # We expect a key in case of a simplified graph, and not otherwise.
    assert G.graph["simplified"] == (k != None)
    
    # Obtain edge data.
    if k == None:
        data = G.get_edge_data(u, v)
    else:
        data = G.get_edge_data(u, v, k)

    # Either extract 
    if not "geometry" in data:
        p1 = G.nodes()[u]
        p2 = G.nodes()[v]
        ps = array([(p1["y"], p1["x"]), (p2["y"], p2["x"])])
        return ps
    else:
        assert G.graph["simplified"] # We do not accept geometry on a vectorized graph, because the curvature is implicit.
        linestring = data["geometry"]
        ps = array([(y, x) for (x, y) in list(linestring.coords)])
        assert len(ps) >= 2 # Expect start and end position.
        return ps


# Transform the path in a graph to a curve (polygonal chain). Assumes path is correct and exists. Input is a list of graph nodes.
def path_to_curve(G, path=[], start_node=None, end_node=None):

    assert len(path) >= 1 # We traverse at least one edge.

    # Collect subcurves.
    pss = [] 
    current = start_node # Node we are currently at as we are walking along the path.

    def _get_curvature(G, path):
        if G.graph["simplified"]:
            for a, b, k in path: # Expect key on each edge.
                ps = edge_curvature(G, a, b, k=k)
                yield a, b, ps
        else: # Graph is vectorized.
            for a, b in path:
                ps = edge_curvature(G, a, b)
                yield a, b, ps
    
    for (a, b, ps) in _get_curvature(G, path):
        # Reverse curvature in case we move from b to a.
        if current == b: 
            ps = ps[::-1]
        # Move current pointer to next node.
        if current == a:
            current = b
        else:
            current = a
        pss.append(ps)
    
    qs = array([(G.nodes()[start_node]["y"], G.nodes()[start_node]["x"])])
    assert np.all(pss[0][0] == qs[0]) # Expect curvature to begin at coordinates of the startnode.
    assert len(pss) >= 1
    for ps in pss:
        assert np.all(ps[0] == qs[-1]) # Expect to have curvature of adjacent edge to match endpoint (but it might be in opposite direction).
        assert len(ps) >= 2
        qs = np.append(qs, ps[1:], axis=0) # Drop first element of `ps`, because the curvature contains the node (endpoint locations) as well.

    return qs


# Split each _simplified_ edge into line segments with at most `max_distance` line segments lengths.
def graph_split_edges(G, max_distance=10):

    assert not G.graph["simplified"]

    G = G.copy() # Transform to local coordinate system.

    # Convert coordinates into UTM.
    utm_info = graph_utm_info(G)
    G = graph_transform_latlon_to_utm(G)

    # Simplify since we want to cut in curvature.
    G = simplify_graph(G)

    nid   = max(G.nodes()) + 1 # NID value for injected nodes at cut subcurves.
    nodes = extract_nodes_dict(G)
    edges_to_add = [] # Store edges to insert afterwards (otherwise we have edge iteration change issue).
    nodes_to_add = [] # Same for nodes.
    edges_to_delete = []

    # Cut each edge into smaller pieces if necessary.
    for edge in G.edges(data=True, keys=True):

        (a, b, k, attrs) = edge

        # "Bugfix": Somehow self-looped simplified edge causes node duplication even though zero impact on topology.
        # It is quite trivial (barely occurs), so lets just assume max 1 self-loop curve. This solves the duplication issue.
        if a == b and k > 0:
            print(f"dropping one, {(a, b, k)}")
            edges_to_delete.append((a, b, k))
            continue
        # print(edge)

        # Convert edge into a curve.
        try:
            curve = path_to_curve(G, start_node=a, end_node=b, path=[(a, b, k)]) 
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            breakpoint()

        # Cut line curvature into multiple edges in such that maximal distance between edges.
        subcurves = curve_cut_max_distance(curve, max_distance=max_distance) 

        # Various sanity checks.
        try: 
            assert np.all(subcurves[0][0] == nodes[a]) # Expect starting point of first subcurve to match with the starting point of the path.
            assert np.all(subcurves[-1][-1] == nodes[b]) # Expect endpoint of the last subcurve to match with the endpoint of the path.
            assert abs(curve_length(curve) - sum(map(lambda c: curve_length(c), subcurves))) < 0.001 # Expect the summation of subcurve lengths to be approximately the same as the entire path length.
            assert len(subcurves) < 1000 # Expect less than a thousand subcurves (thus thereby expect each simplified edge to be max 1 kilometer long).
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            breakpoint()

        # Schedule original edge for deletion.
        edges_to_delete.append((a, b, k)) 

        # Generate new edges (one per subcurve) which adhere to max distance.
        v = a
        for i, subcurve in enumerate(subcurves):
            # Start node.
            u = v

            if i == len(subcurves) - 1:
                v = b
            else:
                nid += 1
                v = nid
                position = subcurve[-1]
                y, x = position[0], position[1]
                nodes_to_add.append((nid, {"y": y, "x": x}))

            geometry = to_linestring(subcurve)
            edges_to_add.append((u, v, {**attrs, "geometry": geometry, "curvature": subcurve}))

    assert len(duplicated_nodes(G)) == 0
    # print("before edges:", len(G.edges()))
    # print("before nodes:", len(G.nodes()))
    G.remove_edges_from(edges_to_delete)
    G.add_nodes_from(nodes_to_add)
    G.add_edges_from(edges_to_add)
    # print("after edges:", len(G.edges()))
    # print("after nodes:", len(G.nodes()))
    assert len(duplicated_nodes(G)) == 0

    # Sanity check each edge adheres to `max_distance`.
    for u, v, k, attrs in G.edges(data=True, keys=True):
        ps = attrs["curvature"]
        assert curve_length(ps) <= max_distance + 0.0001

    assert G.graph["simplified"]
    assert G.graph["coordinates"] == "utm"

    return G    


# Conserving multi-edge curvature when converting from a MultiGraph into a Graph.
# def multi_edge_conserving(G):

#     assert type(G) == nx.MultiGraph
#     assert G.graph["simplified"]
#     G = G.copy()

#     # Extract multi-edges from graph.
#     multiedge_groups = set()
#     for u, v, k in G.edges(keys=True):
#         if k > 0:
#             multiedge_groups.add((u, v)) # u <= v by the G.edges() function.

#     # Per multi-edge set, check the curvature differs (PCM threshold is larger than zero).
#     edges = [(u, v, k) for u, v, k in G.edges(keys=True)]
#     nodes = extract_nodes_dict(G)
#     for u,v in multiedge_groups:
#         multiedges = list(filter(lambda x: x[0] == u and x[1] == v, edges))
#         assert multiedges[0] == (u, v, 0)
#         unique_curves = [] # Store uvk alongside curvature (which is sufficiently unique).
#         unique_curves.append((u, v, 0, edge_curvature(G, u, v, k=0))) # Start with unique in first.
#         edges_to_delete = []

#         # Extract multi-edge ids with unique curvature.
#         for k in range(1, len(multiedges)): # Check every subsequent element.
#             is_unique = True # Consider true unless proven otherwise.
#             ps = edge_curvature(G, u, v, k=k) # Curvature of this element to check.
#             for qs in map(lambda x: x[3], unique_curves): # Curvature of currently unique multi-edges.
#                 if is_partial_curve_undirected(ps, qs, 1, convert=True): # Check for being a partial curve.
#                     is_unique = False # Its to similar to existing curvature.
#             if is_unique: # Add to list.
#                 unique_curves.append((u, v, k, ps))
#             else:
#                 edges_to_delete.append((u, v, k))

#         # For all unique curves, filter out those with a curvature of at least three elements (otherwise we cannot introduce nodes).
#         # And then add those as new nodes to the graph and cut the initial edge into two pieces.
#         nidmax = max(G.nodes()) + 1 # Maximal node ID to prevent overwriting existing node IDs in the graph.
#         for (u, v, k, ps) in unique_curves:
#             if u == v: # In case of self-loop we have to add two edges in between
#                 if len(ps) > 3: # At least 2 vertices for curvature (Besides start and end node).
#                     i = floor(len(ps)/3) # Index to cut curve at.
#                     j = floor(2*len(ps)/3) # Index to cut curve at.
#                     x0, y0 = ps[i]
#                     G.add_node(nidmax, x=x0, y=y0)
#                     x1, y1 = ps[j]
#                     G.add_node(nidmax+1, x=x1, y=y1)
#                     G.add_edge(u, nidmax, 0, geometry=to_linestring(ps[0:i+1]))
#                     G.add_edge(nidmax, nidmax+1, 0, geometry=to_linestring(ps[i:j+1]))
#                     G.add_edge(nidmax+1, v, 0, geometry=to_linestring(ps[j:]))
#                     edges_to_delete.append((u, v, k))
#                     nidmax += 2

#             else:
#                 if len(ps) > 2: # Add node in between.
#                     # a. Add node with nidmax and x,y position ps[floor(len(ps)/2)]
#                     i = floor(len(ps)/2) # Index to cut curve at.
#                     x, y = ps[i]
#                     G.add_node(nidmax, x=x, y=y)
#                     nodes[nidmax] = ps[i]
#                     # b. Add two edges to the graph with u-nidmax and nidmax-v.
#                     #    Make sure to extract geometry and ad 
#                     # print("total edge curvature:\n", ps)
#                     # print(f"Adding edge {u, nidmax} with geometry: \n", ps[0:i+1])
#                     curvature = ps[0:i+1]
#                     G.add_edge(u, nidmax, 0, geometry=to_linestring(curvature))
#                     # Sanity check: Start and end node of curvature match with node position.
#                     if not (np.all(array(curvature[0]) == array(nodes[u])) or np.all(array(curvature[-1]) == array(nodes[u]))):
#                         breakpoint()
#                     assert np.all(array(curvature[0]) == array(nodes[u])) or np.all(array(curvature[-1]) == array(nodes[u])) # Geometry starts at first node coordinate.
#                     if not (np.all(array(curvature[0]) == array(nodes[nidmax])) or np.all(array(curvature[-1]) == array(nodes[nidmax]))):
#                         breakpoint()
#                     assert np.all(array(curvature[0]) == array(nodes[nidmax])) or np.all(array(curvature[-1]) == array(nodes[nidmax])) # Geometry ends at last node coordinate.
                    
#                     # print(f"Adding edge {nidmax, v} with geometry: \n", ps[i:])
#                     curvature = ps[i:]
#                     G.add_edge(nidmax, v, 0, geometry=to_linestring(curvature))
#                     # Sanity check: Start and end node of curvature match with node position.
#                     assert np.all(array(curvature[0]) == array(nodes[nidmax])) or np.all(array(curvature[-1]) == array(nodes[nidmax])) # Geometry starts at first node coordinate.
#                     assert np.all(array(curvature[0]) == array(nodes[v])) or np.all(array(curvature[-1]) == array(nodes[v])) # Geometry ends at last node coordinate.
#                     # c. Mark the edge for deletion.
#                     edges_to_delete.append((u, v, k))
#                     # d. Increment nidmax for subsequent element.
#                     nidmax += 1
        
#         print("Deleting edges ", edges_to_delete)
#         G.remove_edges_from(edges_to_delete)
    
#     return G