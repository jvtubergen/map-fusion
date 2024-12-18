from external import *

from data_handling import *
from coordinates import * 
from utilities import *

# Utility function for convenience to extract graph by name.
#   Either construction from raw data in folder or reading from graphml file.
#   Expect folders with raw data to exist at "data/maps", further with properties described by `construct_graph`.
#   Will either read and/or store from "graphs/" data folder.
def extract_graph(name, reconstruct=False):
    graphml_path = "graphs/"+name+".graphml"

    if Path(graphml_path).exists() and not reconstruct:
        G = ox.load_graphml(filepath=graphml_path)
    else:
        G = construct_graph("data/maps/" + name)
        ox.save_graphml(G, filepath=graphml_path)
    
    return G


# Extract stored graphs from a graph set. 
# * Optionally only retrieve graphs which exist, otherwise expect to retrieve gps, sat, and truth.
# * Expect all graphs to be stored as simplified MultiGraphs (thus undirected but potentially parallel edges with unique curvature).
def extract_graphset(name, optional=False):
    graphs = {}
    gps_path = f"graphsets/{name}/gps.graphml"
    sat_path = f"graphsets/{name}/sat.graphml"
    truth_path = f"graphsets/{name}/truth.graphml"
    if optional:
        if Path(gps_path).exists():
            graphs["gps"]   = ox.load_graphml(filepath=gps_path)
        if Path(sat_path).exists():
            graphs["sat"]   = ox.load_graphml(filepath=sat_path)
        if Path(truth_path).exists():
            graphs["truth"]   = ox.load_graphml(filepath=truth_path)
    else:
        graphs["gps"]   = ox.load_graphml(filepath=f"graphsets/{name}/gps.graphml")
        graphs["sat"]   = ox.load_graphml(filepath=f"graphsets/{name}/sat.graphml")
        graphs["truth"] = ox.load_graphml(filepath=f"graphsets/{name}/truth.graphml")
    return graphs


# Retrieve truth graph with a bounding box.
def retrieve_graph_truth(graphset, bbox):
    # First check the graph does not exist.
    graphml_path=f"graphsets/{graphset}/truth.graphml"
    if Path(graphml_path).exists():
        raise BaseException(f"The truth graph already exists at '{graphml_path}'.")
    # Then retrieve and store graph. (Note: G is already simplified at retrieval.)
    G = ox.graph_from_bbox(bbox=bbox, network_type="drive_service") 
    G = G.to_undirected() 
    ox.save_graphml(G, filepath=graphml_path)


# Save a graph to storage as a GraphML format.
# * Optionally overwrite if the target file already exists.
# * Writes the file into the graphs folder.
def save_graph(G, name, overwrite=False):
    graphml_path = "graphs/"+name+".graphml"
    G = G.copy()
    G.graph['crs'] = "EPSG:4326"
    G = nx.MultiDiGraph(G)
    if Path(graphml_path).exists() and not overwrite:
        print("Did not save graph: Not allowed to overwrite existing file.")
    else:
        ox.save_graphml(G, filepath=graphml_path)


# Construct network out of paths (a list of a list of coordinates)
def convert_paths_into_graph(pss, nid=1, gid=1):
    # Provide node_id offset.
    # Provide group_id offset.
    G = nx.Graph()
    for ps in pss:
        i = nid
        # Add nodes to graph.
        for p in ps:
            # Add random noise so its not overlapped in render
            G.add_node(nid, y=p[0] + 0.0001 * random.random(), x=p[1] + 0.0001 * random.random(), gid=gid)
            nid += 1
        # Add edges between nodes to graph.
        while i < nid - 1:
            G.add_edge(i, i+1, gid=gid)
            i += 1
        gid += 1
    return G


###################################
###  Node and path construction/extraction
###################################

# Extract nodes from a graph into the format `(id, nparray(x,y))`.
def extract_nodes(G):
    return [( node, np.asarray([data['y'], data['x']], dtype=np.float64, order='c') ) for node, data in G.nodes(data = True)]


# Extract nodes from a graph as a dictionary `{nid: nparray([x,y])}`.
def extract_nodes_dict(G):
    d = {}
    for node, data in G.nodes(data = True):
        d[node] = np.asarray([data['y'], data['x']], dtype=np.float64, order='c')
    return d


# Extract node positions, ignoring node ids.
def extract_node_positions(G):
    return np.asarray([[data['y'], data['x']] for node, data in G.nodes(data = True)], dtype=np.float64, order='c')


# Seek nearest vertex in graph of a specific coordinate of interest.
# Expect point to be a 2D numpy array.
def nearest_point(G, p):
    points = extract_nodes(G)

    # Seek nearest point
    dmin = 1000000
    ires = None
    qres = None
    for (i, q) in points:
        m = np.linalg.norm(p - q)
        if m < dmin:
            dmin = m
            (ires, qres) = (i, q)

    return (ires, qres)


# Pick two nodes at random (repeat if in disconnected graphs) and find shortest path.
def gen_random_shortest_path(G):
    nodedict = extract_nodes_dict(G)
    # Pick two nodes from graph at random.
    nodes = random.sample(extract_nodes(G), 2)
    # Extract shortest path.
    path = None
    while path == None:
        nodes = random.sample(extract_nodes(G), 2)
        path = ox.routing.shortest_path(G, nodes[0][0], nodes[1][0])
    # Convert path node ids to coordinates.
    path = np.array([nodedict[nodeid] for nodeid in path])
    return path


###############################################
###  Edge annotation utilities
###############################################

# Ensure geometry element part of graph.
def graph_add_geometry_to_straight_edges(G):

    _G = annotate_edge_curvature_as_array(G)
    edges = np.array(list(_G.edges(data=True, keys=True)))

    edge_attrs = {}
    for (a, b, k, attrs) in edges:
        if "geometry" not in attrs:
            curvature = attrs["curvature"]
            geometry = to_linestring(curvature)
            edge_attrs[(a, b, k)] = {"geometry": geometry}

    nx.set_edge_attributes(G, edge_attrs)
    return G


# Add path length data element.
def graph_annotate_edge_length(G):

    _G = annotate_edge_curvature_as_array(G)
    edges = np.array(list(_G.edges(data=True, keys=True)))

    edge_attrs = {}
    for (a, b, k, attrs) in edges:

        ps = attrs["curvature"]
        length = curve_length(ps)
        assert length > 0 # Assert non-zero length.
        edge_attrs[(a, b, k)] = {"length": length}
    
    nx.set_edge_attributes(G, edge_attrs)
    return G


###############################################
###  Graph vectorization and simplification ###
###############################################

# Wrapping the OSMnx graph simplification logic, and being consistent in MultiGraph convention by converting between directed and undirected.
def simplify_graph(G):
    assert not G.graph["simplified"] 
    G = ox.simplify_graph(nx.MultiGraph(G).to_directed(), track_merged=True).to_undirected()
    G = correctify_edge_curvature(G)
    G.graph["simplified"] = True
    G = consolidate_edge_geometry_and_curvature(G)
    return G

# Vectorize a network
# BUG: Somehow duplicated edge with G.edges(data=True)
# SOLUTION: Reduce MultiDiGraph to MultiGraph.
# BUG: Connecting first node with final node for no reason.
# SOLUTION: node_id got overwritten after initializing on unique value.
# BUG: Selfloops are incorrectly reduced when undirectionalized.
# SOLUTION: Build some function yourself to detect and remove duplicates
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
            assert np.all(array(ps[0]) == array(nodes[a])) or np.all(array(ps[-1]) == array(nodes[a])) # Geometry starts at first node coordinate.
            assert np.all(array(ps[0]) == array(nodes[b])) or np.all(array(ps[-1]) == array(nodes[b])) # Geometry ends at last node coordinate.
            assert len(ps) >= 1 # We expect at least one point in between start and end node.

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


# Correct potentially incorect node curvature (may be moving in opposing direction in comparison to start-node and end-node of edge).
def correctify_edge_curvature(G):
    assert type(G) == nx.MultiGraph
    assert G.graph["simplified"]
    G = G.copy()
    nodes = extract_nodes_dict(G)
    for (u, v, k, attrs) in G.edges(keys=True, data=True):
        if "geometry" in attrs.keys(): # If no geometry there is nothing to check.
            linestring = attrs["geometry"]
            ps = array([(y, x) for (x, y) in list(linestring.coords)])
            a = np.all(array(ps[0]) == array(nodes[u])) and np.all(array(ps[-1]) == array(nodes[v]))
            b = np.all(array(ps[0]) == array(nodes[v])) and np.all(array(ps[-1]) == array(nodes[u]))
            if b: # Convert around
                # print("flip around geometry", (u, v, k))
                ps = array(ps)
                ps = ps[::-1]
                geometry = to_linestring(ps)
                nx.set_edge_attributes(G, {(u, v, k): {"geometry": geometry, "curvature": ps}}) # Update geometry.
    return G


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


# Convert an array into a LineString consisting of Points.
to_linestring   = lambda curvature: LineString([Point(x, y) for y, x in curvature]) # Coordinates are flipped.
from_linestring = lambda geometry : array([(y, x) for x, y in geometry.coords]) # Coordinates are flipped.

# Extract subgraph by a point and a radius (using a square rather than circle for distance measure though).
def extract_subgraph(G, ps, lam):
    edgetree = graphedges_to_rtree(G) # Place graph edges by coordinates in accelerated data structure (R-Tree).
    bbox = bounding_box(ps, lam)
    edges = list(edgetree.intersection((bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]))) # Extract edges within bounding box.
    subG = G.edge_subgraph(edges)
    return subG


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


# Split each _simplified_ edge into line segments with at most `max_distance` line segments lengths.
def graph_split_edges(G, max_distance=10):

    assert type(G) == nx.Graph
    assert not G.graph.get("simplified") # Vectorized.
    G = G.copy() # Transform to local coordinate system.

    utm_info = get_utm_info_from_graph(G)
    G = graph_transform_latlon_to_utm(G)
    G = simplify_graph(G)

    nid = max(G.nodes()) + 1
    nodes = extract_nodes_dict(G)

    edges_to_add = [] # Store edges to insert afterwards (otherwise we have edge iteration change issue).
    nodes_to_add = [] # Same for nodes.
    edges_to_delete = []
    
    # Cut each edge into smaller pieces if necessary.
    for edge in G.edges(data=True, keys=True):

        (a, b, k, attrs) = edge
        # print(edge)

        # Convert edge into a curve.
        try:
            ps = path_to_curve(G, start_node=a, end_node=b, path=[(a, b, k)]) 
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            breakpoint()

        # Cut line curvature into multiple edges in such that maximal distance between edges.
        subcurves = curve_cut_max_distance(ps, max_distance=max_distance) 

        # Various sanity checks.
        try: 
            assert np.all(subcurves[0][0] == nodes[a]) # Expect starting point of first subcurve to match with the starting point of the path.
            assert np.all(subcurves[-1][-1] == nodes[b]) # Expect endpoint of the last subcurve to match with the endpoint of the path.
            assert abs(curve_length(ps) - sum(map(lambda c: curve_length(c), subcurves))) < 0.001 # Expect the summation of subcurve lengths to be approximately the same as the entire path length.
            assert len(subcurves) < 1000 # Expect less than a thousand subcurves (thus thereby expect each simplified edge to be max 1 kilometer long).
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            breakpoint()

        # Schedule original edge for deletion.
        edges_to_delete.append((a, b, k)) 

        # Generate new edges (one per subcurve) which adhere to max distance.
        for i, subcurve in enumerate(subcurves):
            # Start node.
            if i == 0: # Pick start node identifier.
                u = a
            else:
                u = nid
            
            # End node.
            if i == len(subcurves) - 1: # Pick end node identifier.
                v = b
            else:
                nid += 1
                v = nid 
            
            geometry = to_linestring(subcurve)
            edges_to_add.append((u, v, {"geometry": geometry, "curvature": subcurve}))

    G.remove_edges_from(edges_to_delete)
    G.add_nodes_from(nodes_to_add)
    G.add_edges_from(edges_to_add)

    assert G.graph["simplified"]
    assert G.graph["coordinates"] == "utm"

    return G    


###################################
###  Deduplication functionality
###################################

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
        
    G.remove_edges_from(edges_to_delete)
    G.add_edges_from(edges_to_insert)
    G.remove_nodes_from(nodes_to_delete)

    return G


# Remove duplicated nodes and edges from vectorized graph.
# NOTE: Since a vectorized only stores directly
#       No need to 
def deduplicate_vectorized_graph(G):

    G = G.copy()

    if not type(G) == nx.MultiGraph:
        msg = "Graph has to be a MultiGraph for it to work here."
        raise BaseException(msg)

    # Deduplicate nodes: Adjust edges of deleted nodes + Delete nodes.
    removed_nodes = 0 
    for group in duplicated_nodes_grouped(G):
        base = group[0]
        # for n, nbrs in G.adj.items():
        #     # n: from node id
        #     # nbrs: {<neighboring node id>: {<edge-id-connecting n with nbr>: <edge attributes>}}]
        for remove in group[1:]:
            removed_nodes += 1
            # Replace edges
            nbrs = G.adj[remove]
            to_delete = []
            to_add = []
            for nbr in nbrs:
                edge = G[remove][nbr]
                to_delete.append((remove,nbr))
                to_add.append((base,nbr, *edge))
            # Delete and add in a single go to prevent changing the nbrs loop elements.
            G.remove_edges_from(to_delete)
            G.add_edges_from(to_add)
            
            # Expect remove node to be isolated now.
            assert len(G[remove]) == 0
            G.remove_node(remove)

    print(f"Deduplicated, removed {removed_nodes} nodes.")
    
    # Deduplicate edges:
    # Since G is a graph, adding the edges already resolves into edge deduplication.
    # G = nx.MultiDiGraph(G) # Convert back into MultiDiGraph.
    
    return G


###################################
###  Subgraph functionality
###################################

# Cut out ROI subgraph.
# * Drop any edge with an endpoint beyond the ROI.
def cut_out_ROI(G, p1, p2):
    G = G.copy()

    # bb = {a: {y: p1[0], x: p1[1]}, b: {y: p2[0], x: p2[1]}}

    bb = [p1,p2]
    bb = [
        [ min(p1[0], p2[0]), min(p1[1], p2[1]) ],
        [ max(p1[0], p2[0]), max(p1[1], p2[1]) ]
    ]
    assert bb[0][0] <= bb[1][0]
    assert bb[0][1] <= bb[1][1]
    
    def contains(bb, p):
        return p[0] >= bb[0][0] and p[1] >= bb[0][1] and p[0] <= bb[1][0] and p[1] <= bb[1][1]

    to_drop = []
    to_keep = []
    # We have to iterate graph nodes only once to check bounding box.
    for nid, data in G.nodes(data = True):
        y, x = data['y'], data['x']
        if contains(bb, [y, x]):
            to_keep.append(nid)
        else:
            to_drop.append(nid)
    
    # Filtering out nodes in ROI.
    return G.subgraph(to_keep)

#######################################
###  Graph coordinate transformations
#######################################

# Obtain middle latitude coordinate for bounding box that captures all nodes in the graph.
def middle_latitute(G):
    assert G.graph["coordinates"] == "latlon"
    uvk, data = zip(*G.nodes(data=True))
    df = gpd.GeoDataFrame(data, index=uvk)
    alat, alon = df["y"].mean(), df["x"].mean()
    return alat


# Compute relative positioning.
# Note: Takes a reference latitude for deciding on the GSD. Make sure to keep this value consistent when applied to different graphs.
# Note: Flip y-axis by subtracting from minimal latitude value (-max_lat) to maintain directionality.
# todo: Rely on UTM conversion instead of your hacky solution.
def transform_geographic_coordinates_into_scaled_pixel_positioning(G, reflat):
    assert G.graph["coordinates"] == "latlon"
    # 0. GSD on average latitude and reference pixel positions.
    zoom = 24 # Sufficiently accurate.
    gsd = compute_gsd(reflat, zoom, 1)
    # 1. Vectorize.
    G = vectorize_graph(G)
    # 2. Map all nodes to relative position.
    uvk, data = zip(*G.nodes(data=True))
    df = gpd.GeoDataFrame(data, index=uvk)
    maxy, _ = latlon_to_pixelcoord(-max_lat, 0, zoom)
    # Map lat,lon to y,x with latlon_to_pixelcoord.
    def latlon_to_relative_pixelcoord(row): 
        lat, lon = row["y"], row["x"]
        y, x = latlon_to_pixelcoord(lat, lon, zoom)
        return {**row, 'y': maxy - gsd * y, 'x': gsd * x }
    # Construct relabel mapping and transform node coordinates to relative scaled pixel position.
    relabel_mapping = {}
    for nid, data in G.nodes(data=True):
        relabel_mapping[nid] = latlon_to_relative_pixelcoord(data)
    nx.set_node_attributes(G, relabel_mapping)
    # 3. Convert back into simplified graph.
    return simplify_graph(G) # If it has curvature it crashes because of non-hashable numpy array in attributes.
    

# Abstract function with core logic for utm/latlon graph conversion.
def graph_transform_generic(G, coordinate_transformer):

    G = G.copy()

    node_relabel_mapping = {}
    edge_relabel_mapping = {}

    # Adjust coordinates of graph nodes.
    for nid, attrs in G.nodes(data=True):
        y, x = attrs["y"], attrs["x"]
        y, x = coordinate_transformer(y, x)
        node_relabel_mapping[nid] = {**attrs, "y": y, "x": x}
    
    nx.set_node_attributes(G, node_relabel_mapping)
    
    # Update edge geometry and curvature properties in case the graph is simplified.
    if G.graph["simplified"]: 

        for (u, v, k, attrs) in G.edges(data=True, keys=True):

            curvature = attrs["curvature"]
            curvature = [coordinate_transformer(y, x) for y, x in curvature]
            geometry  = to_linestring(curvature)

            edge_relabel_mapping[(u, v, k)] = {**attrs, "geometry": geometry, "curvature": curvature}
        
        nx.set_edge_attributes(G, edge_relabel_mapping)

    return G


# Transform graphnodes UTM coordinate system into latitude-longitude coordinates.
def graph_transform_utm_to_latlon(G, place, letter=None, number=None):

    assert G.graph["coordinates"] == "utm"

    # Obtain utm information.
    if letter == None or number == None:
        letter, number = zone_letters[place], zone_numbers[place]
    utm_info = {"number": number, "letter": letter}

    # Convert coordinates.
    coordinate_transformer = lambda y, x: utm_to_latlon_by_utm_info((y, x), **utm_info)
    G = graph_transform_generic(G, coordinate_transformer)

    G.graph["coordinates"] = "latlon"

    return G


# Transform graphnodes latitude-longitude coordinates into UTM coordinate system.
def graph_transform_latlon_to_utm(G):

    assert G.graph["coordinates"] == "latlon"

    coordinate_transformer = lambda y, x: latlon_to_utm((y, x))
    G = graph_transform_generic(G, coordinate_transformer)

    G.graph["coordinates"] = "utm"

    return G


# Derive UTM zone number and zone letter of a graph by taking arbitrary latlon coordinate from graph.
def graph_utm_info(G):
    assert G.graph["coordinates"] == "latlon"
    node = G._node[list(G.nodes())[0]]
    lat, lon = node['y'], node['x']
    _, _, zone_number, zone_letter = utm.conversion.from_latlon(lat, lon)
    return {"zone_number": zone_number, "zone_letter": zone_letter}


#######################################
### Task specific graph preprocessing
#######################################

# Preparing graph for APLS usage (simplified, multi-edge, all edges have geometry property, all edges have an edge length property).
def graph_prepare_apls(G):
    G = simplify_graph(graph_transform_latlon_to_utm(G))
    G = graph_annotate_edge_length(G)
    G = graph_add_geometry_to_straight_edges(G) # Add geometry for straight line segments (edges with no curvature).
    # G.graph['crs'] = "EPSG:4326" # Set EPSG might be necessary for plotting results within APLS logic.
    return G


# Preparing graph for TOPO usage (simplified, multi-edge, all edges have geometry property, all edges have an edge length property).
def graph_prepare_topo(graph):
    # t0 = time()
    graph = simplify_graph(graph)
    graph = graph.to_directed(graph)
    graph = nx.MultiGraph(graph)
    graph = graph_add_geometry_to_straight_edges(graph)
    graph = graph_annotate_edge_length(graph)
    # verbose_print(f"Simplified graph in {int(time() - t0)} seconds.")
    return graph


#######################################
### Sanity check functionality
#######################################

# Perform a few sanity checks on the graph to prevent computation errors down the line.
def graph_sanity_check(G):
    print("Check graph.")

    # Simplification.
    if "simplified" not in G.graph:
        raise Exception("Expect 'simplified' dictionary key in graph.")

    # If simplified, then multigraph.
    if G.graph["simplified"] and type(G) != type(nx.MultiGraph()):
        raise Exception("Expect simplified graph to be an undirected multi-graph.") 
    if not G.graph["simplified"] and type(G) != type(nx.Graph()):
        raise Exception("Expect vectorized graph to be a undirected single-graph.") 

    # Coordinates.    
    if "coordinates" not in G.graph:
        raise Exception("Expect 'coordinates' dictionary key in graph.")
    if G.graph["coordinates"] != "utm" and G.graph["coordinates"] != "latlon":
        raise Exception("Expect 'coordinates' dictionary value to be either 'utm' or 'latlon'.")

    # Nodes.
    nodes = extract_node_positions(G)
    if G.graph["coordinates"] == "utm":
        if np.min(nodes) < 100: 
            print(nodes)
            raise Exception("Expect graph in utm coordinate system. Big chance some node is in latlon coordinate system.")
    if G.graph["coordinates"] == "latlon":
        if np.min(nodes) > 100: 
            print(nodes)
            raise Exception("Expect graph in latlon coordinate system. Big chance some node is in utm coordinate system.")

    # Node (x,y coordinates) flipping.
    coord0 = np.min(nodes, axis=0)
    coord1 = np.max(nodes, axis=0)
    diffa = np.max(nodes[:,0]) - np.min(nodes[:,0])
    diffb = np.max(nodes[:,1]) - np.min(nodes[:,1])
    # print(diffa, diffb)
    diffc = np.max(nodes[:,0]) - np.min(nodes[:,1])
    diffd = np.max(nodes[:,1]) - np.min(nodes[:,0])
    # print(diffc, diffd)

    if G.graph["coordinates"] == "latlon" and (abs(diffa) > 1 or abs(diffb) > 1):
        print(nodes)
        print(abs(diffa), abs(diffb))
        raise Exception("Expect node y, x coordinate consistency.") 
    if G.graph["coordinates"] == "utm" and (abs(diffa) > 100000 or abs(diffb) > 100000):
        print(nodes)
        print(abs(diffa), abs(diffb))
        raise Exception("Expect node y, x coordinate consistency.") 
    
    # Edges.
    nodes = extract_nodes_dict(G)
    if G.graph["simplified"]: 
        for (a, b, k, _) in G.edges(data=True, keys=True):
            ps = edge_curvature(G, a, b, k)
            if G.graph["coordinates"] == "latlon": # Convert to utm for computing in meters.
                ps = array([latlon_to_coord(lat, lon) for [lat, lon] in ps])
            if curve_length(ps) > 1000: # Expect reasonable curvature length.
                raise Exception("Expect edge length less than 1000 meters. Probably some y, x coordinate in edge curvature got flipped.")
            ps = edge_curvature(G, a, b, k) # Expect startpoint matches curvature.
            try:
                if (not np.all(ps[0] == nodes[a])) and (not np.all(ps[-1] != nodes[b])):
                    raise Exception("Expect curvature have same directionality as edge start and end edge.")
            except Exception as e:
                print(traceback.format_exc())
                print(e)
                breakpoint()
        
