from external import *

from graph_node_extraction import *
from graph_curvature import *


#####
### Simplification logic from OSMnx adapted for our use-case of simplifying undirected graphs.
##### 

# Walks from node along first edge till it reaches an endpoint.
# Returns a pair of nodes and edges walked.
def walk_path(G, startpoint, first_edge_to_walk, endpoints):

    next_nid = first_edge_to_walk[1] if startpoint != first_edge_to_walk[1] else first_edge_to_walk[0]

    # start building path from endpoint node through its successor
    nid_walk = [startpoint, next_nid]
    eid_walk = [first_edge_to_walk]

    if next_nid in endpoints:
        return (nid_walk, eid_walk)

    previous_eid = first_edge_to_walk
    while next_nid not in endpoints:

        # Keep on walking.
        current_nid = next_nid

        # Find next eid (exists on this nid, otherwise we wouldnt get here in the code).
        connected_eids = get_connected_eids(G, current_nid)
        check(len(connected_eids) == 2, expect="Expect exactly two edge connections on traversing to node which is not an endpoint.")
        next_eid = connected_eids[1] if connected_eids[0] == previous_eid else connected_eids[0]
        next_nid = next_eid[1] if current_nid != next_eid[1] else next_eid[0]

        eid_walk.append(next_eid)
        nid_walk.append(next_nid)
        
        previous_eid = next_eid

    return (nid_walk, eid_walk)


# Determine if a node is a true endpoint of an edge.
# * It is its own neighbor (ie, it self-loops).
# * It does not have exactly two neighbors and degree of 2 or 4.
def is_endpoint(G, nid):

    # Obtain neighbors for node.
    neighbors = list(G.adj[nid].keys()) # list(G.neighbors(nid))
    n = len(neighbors)

    return (nid in neighbors) or (n != 2)
    

# Generate all the paths to be simplified between endpoint nodes.
# * Yields a list of `(startnid, endnid, traversed_nids, traversed_eids)`.
def graph_paths_to_simplify(G):

    endpoints = {nid for nid in G.nodes if is_endpoint(G, nid)}

    # Obtain paths to simplify (fold) into a single edge.
    walked_eids = set()
    for startpoint in endpoints:

        # Adjacent edges to the startpoint.
        connected_eids = get_connected_eids(G, startpoint)

        # Try to construct path for connected edge.
        for connected_eid in connected_eids:

            # Don't rediscover routes.
            if connected_eid in walked_eids:
                continue

            # Obtain path.
            path = nids, eids = walk_path(G, startpoint, connected_eid, endpoints)

            # Mark path edges as discovered.
            walked_eids = walked_eids.union(eids)

            # Only yield path if it is traversing at one nid in the middle or if its a self-loop.
            if len(nids) > 2 or nids[0] == nids[1]:
                yield path


# Simplify graph (fuse edge curvature).
# Optionally retain attributes on edges.
@info()
def simplify_graph(G, retain_attributes=False):

    graph_annotate_edge_curvature(G)
    graph_correctify_edge_curvature(G)

    G = G.copy()

    if type(G) == nx.Graph:
        
        # We require it to be a multigraph.
        G = nx.MultiGraph(G)

        # Set simplified to True here, since most of your functions decide on Graph/Multigraph logic based on this attribute.
        G.graph["simplified"] = True

    # Sanity check that curvature attribute is present on every edge in the graph.
    [check("curvature" in attrs, expect="Expect curvature in all edges to concatenate with simplification (if necessary).") for eid, attrs in iterate_edges(G)]

    nid_positions = extract_node_positions_dictionary(G)

    # Sanity check node position starts/ends at all edge curves.
    sanity_check_graph_curvature(G)

    # We regenerate length and geometry afterwards. Curvature is dealt with separately.
    attributes_to_ignore = ["length", "curvature", "geometry"]

    nids_to_drop = []
    eids_to_drop = []
    
    new_edges = []

    # Iterate all paths we have to simplify.
    for path in graph_paths_to_simplify(G):

        visited_nids, visited_eids = path

        ## Accumulate curvature.

        # Start curvature with first nid position.
        curvature = [nid_positions[visited_nids[0]]]

        for eid in visited_eids: 

            subcurve = get_edge(G, eid=eid)["curvature"]

            # Sanity check on subcurve.    
            check((subcurve[0] == curvature[-1]).all() or (subcurve[-1] == curvature[-1]).all(), expect="Expect subcurve to start (or end) at current curvature endpoint.")

            # Optionally reverse the extension.
            if np.all(subcurve[-1] == curvature[-1]):
                subcurve = list(reversed(subcurve))

            # (Since curvature contains endpoint to endpoint, we take entire curvature except the first point.)
            curvature.extend(subcurve[1:])

        # Flatten and array curvature.
        curvature = array(curvature) 

        # Sanity checks on curvature (array shape and length consistency).
        check(curvature.shape[1] == 2, expect="Expect concatenated curvature to be flattened into a sequence of two-dimensional points.")
        check(curve_length(curvature) == sum([curve_length(get_edge(G, eid=eid)["curvature"]) for eid in visited_eids]), expect="Expect curvature length to be consistent after concatenation.")

        # Reverse curvature if necessary.
        u, v = min(visited_nids[0], visited_nids[-1]), max(visited_nids[0], visited_nids[-1])
        if np.all(nid_positions[u] == curvature[-1]):
            curvature = array(list(reversed(curvature)))

        p, q = nid_positions[u], nid_positions[v]
        ps = curvature
        check(np.all(p == ps[0]), expect="Expect curvature of all connected edges starts/end at node position.")
        check(np.all(q == ps[-1]), expect="Expect curvature of all connected edges starts/end at node position.")

        attributes = {"curvature": curvature}

        ## Collecting edge attributes.
        if retain_attributes:

            path_attributes = {}
            for eid in visited_eids:
                attrs = get_edge(G, eid=eid)
                for attr in [attr for attr in attrs if attr not in attributes_to_ignore]:
                    if attr not in attributes_to_ignore:
                        # If this attribute is already seen in a previous edge.
                        if attr in path_attributes:
                            # Then append this attribute value.
                            path_attributes[attr].append(edge_data[attr])
                        else:
                            # Otherwise initiate this attribute with this attritube value.
                            path_attributes[attr] = [edge_data[attr]]
        
            # Expect attribute consistency (The edges do not all need to have the attribute, but the total collection of those is exactly one unique attribute value).
            for attr in path_attributes:
                check(len(set(path_attributes[attr])) == 1, expect="Expect for any edge attribute to be identical among all subedges.")
            
            # Set the path attribute to this single unique value.
            for attr in path_attributes:
                path_attributes[attr] = path_attributes[attr][0]

            # Add these attributes alongside the curvature.
            attributes = {"curvature": curvature, **path_attributes}

        # Prepare data for insertion/deletion to/from graph.
        new_edges.append((visited_nids[0], visited_nids[-1], attributes))
        eids_to_drop.extend(visited_eids)
        nids_to_drop.extend(visited_nids[1:-1])

    # Add new edges to the graph.
    for u, v, attrs in new_edges:
        G.add_edge(u, v, **attrs)

    # Drop intermediate nids in walk.
    G.remove_edges_from(eids_to_drop)
    # Drop all walked eids.
    G.remove_nodes_from(nids_to_drop)

    # Mark the graph as having been simplified.
    G.graph["simplified"] = True

    # Annotate geometry and length on the simplified edges.
    graph_annotate_edge_geometry(G)
    graph_annotate_edge_length(G)

    return G


#####
### End of simplification code rewrite.
#####

# Drop self-loops and multi-edges from graph.
@info()
def graph_sanitize_simplified_edges(G):

    if not G.graph["simplified"]:
        return G
    
    G = G.copy()

    self_loops = [(u, v, k) for (u, v, k), _ in iterate_edges(G) if u == v]
    multi_edges = [(u, v, k) for (u, v, k), _ in iterate_edges(G) if u != v and k > 0]

    # Drop self-loops and multi-edges.
    logger(f"Dropping {len(self_loops)} self-loops.")
    logger(f"Dropping {len(multi_edges)} self-loops.")
    G.remove_edges_from(self_loops)
    G.remove_edges_from(multi_edges)

    # Sanity check that we have multi-edge at `k == 0`.
    for (u, v, k) in multi_edges:
        edge = get_edge(G, (u, v, 0))

    # Sanity check that no more self-loops exist.
    self_loops = [(u, v, k) for (u, v, k), _ in iterate_edges(G) if u == v]
    check(len(self_loops) == 0, expect="Expect that no more self-loops exist after dropping self-loops.")

    # Sanity check that no more multi-edges exist.
    multi_edges = [(u, v, k) for (u, v, k), _ in iterate_edges(G) if u != v and k > 0]
    check(len(multi_edges) == 0, expect="Expect that no more multi-edges exist after dropping multi-edges.")

    return G


# Group together multi-edges in a list.
def group_multi_edges(G):

    # Vectorized graphs do not have multi-edges.
    if not G.graph["simplified"]:
        return []

    # Collect `(u, v)` pairs where `k > 0`.
    groups = {}
    for (u, v, k), attrs in iterate_edges(G):
        if (u, v) not in groups:
            groups[(u, v)] = [(u, v, k)]
        else: 
            groups[(u, v)].append((u, v, k))
    
    # Filter out groups with one element.
    groups = [group for group in groups if len(group) > 1]
    
    return groups


# Vectorize a network.
# Note: Edge attributes are propogated to all new subcurve edges.
# Note: Node attributes are propogated from starting node (index `u`).
@info()
def vectorize_graph(G):
    
    check(G.graph["simplified"], expect="Expect graph to be simplified in order to vectorize it.")

    G = G.copy()

    # Extract nodes and edges.
    node_positions = extract_node_positions_dictionary(G)
    node_attributes = {nid: attrs for nid, attrs in G.nodes(data = True)}

    # Obtain unique (incremental) node ID to use.
    new_nid = max(G.nodes()) + 1 # Move beyond highest node ID to ensure uniqueness.

    old_edges = []
    new_nodes = []
    new_edges = []

    for eid, old_edge_attrs in iterate_edges(G):

        ps = old_edge_attrs["curvature"]

        # We only have to perform work if an edge contains curvature. (If there is no geometry component, there is no curvature to take care of. Thus already vectorized format.)
        if len(ps) > 2:

            logger("Vectorizing edge ", eid)

            # Delete this edge from the network.
            old_edges.append(eid)

            u, v, k = eid

            # Sanity check curvature goes from `u` to `v`.
            check(np.all(ps[0] == node_positions[u]), expect="Expect curvature starts at first node position.")
            check(np.all(ps[-1] == node_positions[v]), expect="Expect curvature ends at final node position.")

            # Don't place a node at the first and last curve point, because these already exist.
            ps = ps[1:-1]
                
            # Construct node identifiers.
            nids = list(range(new_nid, new_nid + len(ps)))

            # Increment node identifier for next iteration.
            new_nid += len(ps) 

            # Inject nodes.
            old_nid_attrs = node_attributes[u]
            for nid, p in zip(nids, ps):

                # Inject node to graph.
                new_nodes.append((nid, {**old_nid_attrs, "y":p[0], "x":p[1]}))

                # Add new node positions (used when retrieving position when injecting edges).
                node_positions[nid] = p

            # Inject edges.
            nids = [u] + nids + [v]
            for u, v in zip(nids, nids[1:]):

                # New curvature-related attributes.
                curvature = [node_positions[u], node_positions[v]]
                geometry = to_linestring(curvature)
                length = curve_length(curvature)

                # Note: Set key at zero, because nodes in curvature implies a single path between nodes.
                new_edges.append((u, v, {**old_edge_attrs, "curvature": curvature, "geometry": geometry, "length": length}))

    G.remove_edges_from(old_edges)
    G.add_nodes_from(new_nodes)
    G.add_edges_from(new_edges)

    G.graph["simplified"] = False # Mark the graph as no longer being simplified.
    G = nx.Graph(G)

    return G


# Unit test that simplification and vectorization leaves graph length intact.
@info()
def test_simplify_vectorize():
    G = read_graph(place="chicago", graphset=links["sat"])
    G = graph_transform_latlon_to_utm(G)
    # a. Simplify graph.
    G = simplify_graph(G)
    sanity_check_graph_curvature(G)
    # b. Compute total graph length.
    length = graph_length(G)
    # c. Vectorize graph.
    G = vectorize_graph(G)
    check(graph_length(G) == length, expect="Expect no changes in graph length.")
    # d. Simplify graph.
    G = simplify_graph(G)
    check(graph_length(G) == length, expect="Expect no changes in graph length.")