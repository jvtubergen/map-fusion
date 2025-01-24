from external import *
from graph_coordinates import *
from graph_node_extraction import *
from utilities import *


## Graph edge attribute annotation.

# Add path length data element.
# Optionally delete glitchy nodes (that somehow ended up in the input graphs).
@info()
def graph_annotate_edge_length(G):

    edge_attrs = {}
    edges_to_remove = []
    for eid, attrs in iterate_edges(G):
        ps = attrs["curvature"]
        length = curve_length(ps)
        if eid[0] == eid[1] and length == 0:
            edges_to_remove.append(eid)
        else:
            assert length > 0 # Assert non-zero length.
            edge_attrs[eid] = {**attrs, "length": length}
    
    nx.set_edge_attributes(G, edge_attrs)
    G.remove_edges_from(edges_to_remove)


# Add geometry attribute to every edge.
@info()
def graph_annotate_edge_geometry(G):

    edge_attrs = {}
    for eid, attrs in iterate_edges(G):

        ps = attrs["curvature"]
        geometry = to_linestring(ps)
        edge_attrs[eid] = {**attrs, "geometry": geometry}
    
    nx.set_edge_attributes(G, edge_attrs)


# Annotate each edge with curvature (if not already the case).
# * Try to derive from the geometry.
# * Otherwise extract from 
@info()
def graph_annotate_edge_curvature(G):

    for eid, attrs in iterate_edges(G):

        if "curvature" in attrs:
            if type(attrs["curvature"]) != type(array([])):
                attrs["curvature"] = array(attrs["curvature"])

        elif "geometry" in attrs:

            ps = from_linestring(attrs["geometry"])

            attrs["curvature"] = ps

        else:

            u, v = eid[0:2]

            p1 = G.nodes()[u]
            p2 = G.nodes()[v]
            ps = array([[p1["y"], p1["x"]], [p2["y"], p2["x"]]])

            attrs["curvature"] = ps

    # Sanity check all curves got annotated.
    for eid, attrs in iterate_edges(G):
        check("curvature" in attrs)
        check(type(attrs["curvature"]) == type(array([])))


# Correct potentially incorect node curvature (may be moving in opposing direction in comparison to start-node and end-node of edge).
@info()
def graph_correctify_edge_curvature(G):

    node_positions = extract_node_positions_dictionary(G)
    for eid, attrs in iterate_edges(G):

        # We expect to have curvature in the edge, obtain it..
        u, v = eid[0:2]
        ps = attrs["curvature"]

        # Check whether the curvature is aligned with the node positions.
        is_correct_direction = np.all(array(ps[0]) == array(node_positions[u])) and np.all(array(ps[-1]) == array(node_positions[v]))
        is_inverted_direction = np.all(array(ps[0]) == array(node_positions[v])) and np.all(array(ps[-1]) == array(node_positions[u]))

        check(is_correct_direction or is_inverted_direction, expect="Expect curvature of all connected edges starts/end at node position.")

        # In case the direction of the curvature is inverted.
        if is_inverted_direction: 
            # Then invert the direction back.
            # print("flip around geometry", (u, v, k))
            ps = ps[::-1]
            geometry = to_linestring(ps)
            nx.set_edge_attributes(G, {eid: {**attrs, "geometry": geometry, "curvature": ps}}) # Update geometry.


## Graph edge curvature cutting.

# Replace edge with subedges with provided subcurves.
# * Allow to provide either curve intervals to cut at or the actual subcurves to replace the edge with.
@info()
def graph_cut_edge_subcurves(G, eid, qss):

    # Store graph length for sanity checking graph length consistency.
    length = graph_length(G)

    # Sanity check: First point of first subcurve and last point of final subcurve match eid curvature.
    ps = get_edge_attributes(G, eid)["curvature"]
    assert (ps[0]  == qss[0][0]).all()
    assert (ps[-1] == qss[-1][-1]).all()
    assert abs(curve_length(ps) - sum([curve_length(qs) for qs in qss])) < 0.0001
    if len(ps) > 1:
        for diff in ps[1:] - ps[:-1]:
            assert norm(diff) > 0.0001

    # Remove the original edge.
    G.remove_edges_from([eid])

    nid = max(G.nodes()) + 1 # Nid for injected edge.

    nodes_to_add = [] # New nodes to inject.
    edges_to_add = [] # New edges to inject.

    # Number of edges to inject.
    n = len(qss)
    assert n > 1 # Expect at least two subcurves here.

    # Obtain node positions.
    new_points = [qs[-1] for qs in qss[:-1]]

    # Obtain nids for new nodes.
    new_nids = [nid + i for i in range(n - 1)] # We have `n - 1` new nodes.
    nid += n - 1 # Update new nid.

    # Schedule new nodes for injection.
    for nid, position in zip(new_nids, new_points):
        y, x = position
        nodes_to_add.append((nid, {"y": y, "x": x}))

    # Schedule new edges for injection.
    u, v = eid[0:2]
    edge_links = [(u, new_nids[0])] + list(zip(new_nids, new_nids[1:])) + [(new_nids[-1], v)]
    for (u, v), curvature in zip(edge_links, qss):
        edges_to_add.append((u, v, {"curvature": curvature, "length": curve_length(curvature), "geometry": to_linestring(curvature)}))

    G.add_nodes_from(nodes_to_add)
    G.add_edges_from(edges_to_add)

    logger("Injecting nodes: ", new_nids)
    check(abs(graph_length(G) - length) < 0.001, expect="Expect graph length remains consistent after cutting edges into subcurves." )

    return G, {"nids": nodes_to_add, "eids": edges_to_add}


# Replace edge with subedges at provided intervals.
def graph_cut_edge_intervals(G, eid, intervals):

    # Obtain the curvature to cut in.
    curve = get_edge_attributes(G, eid)["curvature"]

    # Compute the subcurves.
    qss = curve_cut_intervals(curve, intervals)

    return graph_cut_edge_subcurves(G, eid, qss)


# Ensure all edges have a maximal curve length. Cut curves if necessary.
@info()
def graph_ensure_max_edge_length(G, max_length=50):

    scheduled = [] # eid with number of necessary intervals.

    # Compute subcurves to replace an edge with.
    for eid, attrs in iterate_edges(G):

        ps = attrs["curvature"]

        qss = curve_cut_max_distance(ps, max_distance=max_length)

        # Sanity check on subcurves.
        if len(qss) > 1:
            for qs in qss:
                check(curve_length(qs) > 0, expect="Expect non-zero curvature on subcurve.")
            for a, b in zip(qss, qss[1:]):
                check((a[-1] == b[0]).all(), expect="Expect endpoint of previous subcurve matches startpoint of current subcurve.")

        # If the edge is being cut (thus more than 1 subcurve), then schedule it for injection.
        if len(qss) > 1:
            scheduled.append((eid, qss))

    # Inject edges.
    for eid, qss in scheduled:

        # Graph is updated iteratively by replacing edge (by subcurves) one at a time.
        G, _ = graph_cut_edge_subcurves(G, eid, qss)
    
    return G


# Cut graph edge at an interval (in range `[0, 1]`).
graph_cut_edge = lambda G, eid, interval: graph_cut_edge_intervals(G, eid, [interval])


## Graph path-related curvature.

# Transform the path in a graph to a curve (polygonal chain). Assumes path is correct and exists. Input is a list of graph nodes.
def path_to_curve(G, path=[], start_node=None, end_node=None):

    assert len(path) >= 1 # We traverse at least one edge.

    # Collect subcurves.
    pss = [] 
    current = start_node # Node we are currently at as we are walking along the path.

    def path_get_curvature(G, path):
        if G.graph["simplified"]:
            for a, b, k in path: # Expect key on each edge.
                eid = (a, b, k)
                ps = get_edge_attributes(G, eid)["curvature"]
                yield a, b, ps
        else: # Graph is vectorized.
            for a, b in path:
                eid = (a, b)
                ps = get_edge_attributes(G, eid)["curvature"]
                yield a, b, ps
    
    for (a, b, ps) in path_get_curvature(G, path):
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