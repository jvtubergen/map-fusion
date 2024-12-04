from external import *

from data_handling import *
from coordinates import * 
from utilities import *

# Log debug actions of OSMnx to stdout.
ox.settings.log_console = True

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
            G.add_node(nid, x=p[0] + 0.0001 * random.random(), y=p[1] + 0.0001 * random.random(), gid=gid)
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
    return [( node, np.asarray([data['x'], data['y']], dtype=np.float64, order='c') ) for node, data in G.nodes(data = True)]


# Extract nodes from a graph as a dictionary `{nid: nparray([x,y])}`.
def extract_nodes_dict(G):
    d = {}
    for node, data in G.nodes(data = True):
        d[node] = [data['x'], data['y']]
    return d


# Extract node positions, ignoring node ids.
def extract_node_positions(G):
    return np.asarray([[data['x'], data['y']] for node, data in G.nodes(data = True)], dtype=np.float64, order='c')


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
    return ox.simplify_graph(G.to_directed()).to_undirected()

# Vectorize a network
# BUG: Somehow duplicated edge with G.edges(data=True)
# SOLUTION: Reduce MultiDiGraph to MultiGraph.
# BUG: Connecting first node with final node for no reason.
# SOLUTION: node_id got overwritten after initializing on unique value.
# BUG: Selfloops are incorrectly reduced when undirectionalized.
# SOLUTION: Build some function yourself to detect and remove duplicates
def vectorize_graph(G):

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
    # newnodeid = np.max(np.array(nodes)[:,0]) + 1 # Move beyond highest node ID to ensure uniqueness.
    newnodeid = max(G.nodes()) + 1 # Move beyond highest node ID to ensure uniqueness.

    # Edges contain curvature information, extract.
    for (a, b, k, attrs) in edges:

        # If there is no geometry component, there is no curvature to take care of. Thus already vectorized format.
        if "geometry" in attrs.keys():

            # Delete this edge from network.
            G.remove_edge(a,b,k)
            print("Removing edge ", a, b, k)

            # Add curvature as separate nodes/edges.
            linestring = attrs["geometry"]
            ps = list(linestring.coords)

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
                G.add_node(node, x=coord[0], y=coord[1])

            pathids = [a] + pathids + [b]
            for a,b in zip(pathids, pathids[1:]):
                G.add_edge(a, b, k) # Key can be zero because nodes in curvature implies a single path between nodes.

    G.graph["simplified"] = False # Mark the graph as no longer being simplified.

    return G


# Extract curvature (stored potentially as a LineString under geometry) from an edge as an array.
def edge_curvature(G, u, v, k = None):
    
    # Obtain edge data.
    if k == None:
        data = G.get_edge_data(u, v)
    else:
        data = G.get_edge_data(u, v, k)

    # Either extract 
    if not "geometry" in data:
        p1 = G.nodes()[u]
        p2 = G.nodes()[v]
        ps = array([(p1["x"], p1["y"]), (p2["x"], p2["y"])])
        return ps
    else:
        linestring = data["geometry"]
        ps = array(list(linestring.coords))
        assert len(ps) >= 2 # Expect start and end position.
        return ps


# Transform the path in a graph to a curve (polygonal chain). Assumes path is correct and exists. Input is a list of graph nodes.
def path_nodes_to_curve(G, path):
    assert G.graph["simplified"]
    assert type(G) == nx.Graph # We reconstruct edges out of node pairs, if G would be a MultiGraph then information is lost.
    assert len(path) >= 2
    qs = array([(G.nodes()[path[0]]["x"], G.nodes()[path[0]]["y"])])
    for a, b in zip(path, path[1:]): # BUG: Conversion from nodes to path results in loss of information at possible multipaths.
        qs = np.append(qs, edge_curvature(G, a, b, k=0)[1:], axis=0) # Ignore first point when adding.
    return qs


# Convert an array into a LineString consisting of Points.
to_linestring = lambda ps: LineString([Point(x, y) for x, y in ps])


# Extract subgraph by a point and a radius (using a square rather than circle for distance measure though).
def extract_subgraph(G, ps, lam):
    edgetree = graphedges_to_rtree(G) # Place graph edges by coordinates in accelerated data structure (R-Tree).
    bbox = bounding_box(ps, lam)
    edges = list(edgetree.intersection((bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]))) # Extract edges within bounding box.
    subG = G.edge_subgraph(edges)
    return subG


# For a simplified graph, annotate edges with its curvature as a numpy array rather than the encoded shapely string.
def annotate_edge_curvature_as_array(G):

    G = G.copy()

    if not G.graph.get("simplified"):
        msg = "Graph has to be simplified in order to annotate curvature as an array."
        raise BaseException(msg)
    
    if not type(G) == nx.MultiGraph:
        msg = "Graph has to be MultiGraph (undirected but potentially multiple connections) for data extraction and annotation to work."
        raise BaseException(msg)

    edges = np.array(list(G.edges(data=True, keys=True)))

    edge_attrs = {}

    # Edges contain curvature information, extract.
    for (a, b, k, attrs) in edges:
        # print(a, b, attr)

        if not "geometry" in attrs.keys():
            p1 = G.nodes()[a]
            p2 = G.nodes()[b]
            latlon1 = p1["y"], p1["x"]
            latlon2 = p2["y"], p2["x"]
            edge_attrs[(a, b, k)] = {"curvature": np.array([latlon1, latlon2])}
        else:
            linestring = attrs["geometry"]
            # print(list(linestring.coords))
            # Flip lonlat to latlon.
            ps = np.array([(lat,lon) for (lon, lat) in list(linestring.coords)])
            # assert len(ps) >= 3 # We expect at least one point in between start and end node.
            edge_attrs[(a, b, k)] = {"curvature": ps}
    
    nx.set_edge_attributes(G, edge_attrs)
    return G


# Conserving multi-edge curvature when converting from a MultiGraph into a Graph.
def multi_edge_conserving(G):

    assert type(G) == nx.MultiGraph
    G = G.copy()

    # Extract multi-edges from graph.
    multiedge_groups = set()
    for u, v, k in G.edges(keys=True):
        if k > 0:
            multiedge_groups.add((u, v)) # u <= v by the G.edges() function.

    # Per multi-edge set, check the curvature differs (PCM threshold is larger than zero).
    edges = [(u, v, k) for u, v, k in G.edges(keys=True)]
    nodes = extract_nodes_dict(G)
    for u,v in multiedge_groups:
        multiedges = list(filter(lambda x: x[0] == u and x[1] == v, edges))
        assert multiedges[0] == (u, v, 0)
        unique_curves = [] # Store uvk alongside curvature (which is sufficiently unique).
        unique_curves.append((u, v, 0, edge_curvature(G, u, v, k=0))) # Start with unique in first.
        edges_to_delete = []

        # Extract multi-edge ids with unique curvature.
        for k in range(1, len(multiedges)): # Check every subsequent element.
            is_unique = True # Consider true unless proven otherwise.
            ps = edge_curvature(G, u, v, k=k) # Curvature of this element to check.
            for qs in map(lambda x: x[3], unique_curves): # Curvature of currently unique multi-edges.
                if is_partial_curve_undirected(ps, qs, 1, convert=True): # Check for being a partial curve.
                    is_unique = False # Its to similar to existing curvature.
            if is_unique: # Add to list.
                unique_curves.append((u, v, k, ps))
            else:
                edges_to_delete.append((u, v, k))

        # For all unique curves, filter out those with a curvature of at least three elements (otherwise we cannot introduce nodes).
        # And then add those as new nodes to the graph and cut the initial edge into two pieces.
        nidmax = max(G.nodes()) + 1 # Maximal node ID to prevent overwriting existing node IDs in the graph.
        for (u, v, k, ps) in unique_curves:
            if u == v: # In case of self-loop we have to add two edges in between
                if len(ps) > 3: # At least 2 vertices for curvature (Besides start and end node).
                    i = floor(len(ps)/3) # Index to cut curve at.
                    j = floor(2*len(ps)/3) # Index to cut curve at.
                    x0, y0 = ps[i]
                    G.add_node(nidmax, x=x0, y=y0)
                    x1, y1 = ps[j]
                    G.add_node(nidmax+1, x=x1, y=y1)
                    G.add_edge(u, nidmax, 0, geometry=to_linestring(ps[0:i+1]))
                    G.add_edge(nidmax, nidmax+1, 0, geometry=to_linestring(ps[i:j+1]))
                    G.add_edge(nidmax+1, v, 0, geometry=to_linestring(ps[j:]))
                    edges_to_delete.append((u, v, k))
                    nidmax += 2

            else:
                if len(ps) > 2: # Add node in between.
                    # a. Add node with nidmax and x,y position ps[floor(len(ps)/2)]
                    i = floor(len(ps)/2) # Index to cut curve at.
                    x, y = ps[i]
                    G.add_node(nidmax, x=x, y=y)
                    nodes[nidmax] = ps[i]
                    # b. Add two edges to the graph with u-nidmax and nidmax-v.
                    #    Make sure to extract geometry and ad 
                    # print("total edge curvature:\n", ps)
                    # print(f"Adding edge {u, nidmax} with geometry: \n", ps[0:i+1])
                    curvature = ps[0:i+1]
                    G.add_edge(u, nidmax, 0, geometry=to_linestring(curvature))
                    # Sanity check: Start and end node of curvature match with node position.
                    if not (np.all(array(curvature[0]) == array(nodes[u])) or np.all(array(curvature[-1]) == array(nodes[u]))):
                        breakpoint()
                    assert np.all(array(curvature[0]) == array(nodes[u])) or np.all(array(curvature[-1]) == array(nodes[u])) # Geometry starts at first node coordinate.
                    if not (np.all(array(curvature[0]) == array(nodes[nidmax])) or np.all(array(curvature[-1]) == array(nodes[nidmax]))):
                        breakpoint()
                    assert np.all(array(curvature[0]) == array(nodes[nidmax])) or np.all(array(curvature[-1]) == array(nodes[nidmax])) # Geometry ends at last node coordinate.
                    
                    # print(f"Adding edge {nidmax, v} with geometry: \n", ps[i:])
                    curvature = ps[i:]
                    G.add_edge(nidmax, v, 0, geometry=to_linestring(curvature))
                    # Sanity check: Start and end node of curvature match with node position.
                    assert np.all(array(curvature[0]) == array(nodes[nidmax])) or np.all(array(curvature[-1]) == array(nodes[nidmax])) # Geometry starts at first node coordinate.
                    assert np.all(array(curvature[0]) == array(nodes[v])) or np.all(array(curvature[-1]) == array(nodes[v])) # Geometry ends at last node coordinate.
                    # c. Mark the edge for deletion.
                    edges_to_delete.append((u, v, k))
                    # d. Increment nidmax for subsequent element.
                    nidmax += 1
        
        print("Deleting edges ", edges_to_delete)
        G.remove_edges_from(edges_to_delete)
    
    return G


###################################
###  Deduplication functionality
###################################

# Return node IDs which are duplicated (exact same coordinate).
# TODO: Group togeter node-IDs that are duplicates of one another as tuples.
def duplicated_nodes(G):
    # NOTE: Keep IDs (integeters) separate from coordinates (floats): Numpy arrays all have same type.
    nodes = G.nodes()
    coordinates = np.array([[info["x"], info["y"]] for node, info in G.nodes(data=True)])
    uniques, inverses, counts = np.unique( coordinates, return_inverse=True, axis=0, return_counts=True )
    duplicated = []
    for node_id, index_to_unique in zip(nodes, inverses):
        if counts[index_to_unique] > 1:
            duplicated.append(node_id)
    return duplicated


# Group duplicated nodes.
def duplicated_nodes_grouped(G):
    # NOTE: Keep IDs (integeters) separate from coordinates (floats): Numpy arrays all have same type.
    node_ids = G.nodes()
    coordinates = np.array([[info["x"], info["y"]] for node, info in G.nodes(data=True)])
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


# Return edge IDs which are duplicated (exact same coordinates).
# NOTE: Expects vectorized graph.
def duplicated_edges(G):

    G = G.copy()

    if type(G) != nx.MultiGraph:
        raise BaseException("Expect to call duplicated_edge_grouped on an MultiGraph.")

    if G.graph["simplified"]:
        raise BaseException("Duplicated edges function is supposed to be called on a vectorized graph.")

    # Construct edges coordinates in the format of `[x1,y1,x2,y2]`.
    # NOTE: Keep IDs (integeters) separate from coordinates (floats): Numpy arrays all have same type.
    edges = G.edges(keys=True)
    coordinates = []
    for (a, b, k, attrs) in G.edges(keys=True, data=True):
        x1 = G.nodes[a]["x"]
        y1 = G.nodes[a]["y"]
        x2 = G.nodes[b]["x"]
        y2 = G.nodes[b]["y"]
        coordinates.append([x1,y1,x2,y2])
    coordinates = np.array(coordinates)

    # Extract duplications
    uniques, inverses, counts = np.unique(coordinates, return_inverse=True, axis=0, return_counts=True)
    duplicated = []
    for edge_id, index_to_unique in zip(edges, inverses):
        if counts[index_to_unique] > 1:
            duplicated.append(edge_id)
    return duplicated


# Group together edge-IDs that are duplicates of one another as tuples.
# NOTE: Expects vectorized graph.
def duplicated_edges_grouped(G):

    G = G.copy()

    if type(G) != nx.MultiGraph:
        raise BaseException("Expect to call duplicated_edge_grouped on an nx.MultiGraph.")

    if G.graph["simplified"]:
        raise BaseException("Duplicated edges function is supposed to be called on a vectorized graph (thus not simplified).")

    # NOTE: Keep IDs (integeters) separate from coordinates (floats): Numpy arrays all have same type.
    edge_ids = G.edges(keys=True)
    coordinates = []
    for (a, b, k, attrs) in G.edges(keys=True, data=True):
        x1 = G.nodes[a]["x"]
        y1 = G.nodes[a]["y"]
        x2 = G.nodes[b]["x"]
        y2 = G.nodes[b]["y"]
        coordinates.append([x1,y1,x2,y2])
    coordinates = np.array(coordinates)

    # Construct dictionary.
    uniques, inverses, counts = np.unique(coordinates, return_inverse=True, axis=0, return_counts=True)
    duplicated = {}
    for edge_id, index_to_unique in zip(edge_ids, inverses):
        if counts[index_to_unique] > 1:
            if index_to_unique in duplicated.keys():
                duplicated[index_to_unique].append(edge_id)
            else:
                duplicated[index_to_unique] = [edge_id]

    # Convert dictionary into a list.
    result = []
    for v in duplicated:
        result.append(duplicated[v])

    return result


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
        if contains(bb, [x, y]):
            to_keep.append(nid)
        else:
            to_drop.append(nid)
    
    # Filtering out nodes in ROI.
    return G.subgraph(to_keep)


# Merge two networks by adding edges from additional into current. 
# (Strategy is a dummy parameter at this moment.)
def merge_graphs(current, additional, strategy="nearest_node"):

    # strategy nearest_node is the default.
    current = current.copy()

    strategies = [
        "nearest_node",
        "nearest_edge"
    ]

    if strategy == "nearest_node":

        # * Construct rtree on nodes in current.
        nodetree = graphnodes_to_rtree(current)

        # Relabel additional to prevent node id overlap.
        nid=max(current.nodes())+1
        relabel_mapping = {}
        for nidH in additional.nodes():
            relabel_mapping[nidH] = nid
            nid += 1
        additional = nx.relabel_nodes(additional, relabel_mapping)

        # Iterate edges of additional.
        # Place each edge into current.
        for edge in additional.edges(data=True):
            (a, b, attrs) = edge

            x, y = additional._node[a]['x'], additional._node[a]['y']
            node_a = (x, y, x, y)

            x, y = additional._node[b]['x'], additional._node[b]['y']
            node_b = (x, y, x, y)

            # Add node_a and node_b to graph.
            current.add_node(a, **additional._node[a])
            current.add_node(b, **additional._node[b])
            current.add_edge(a, b) # And draw edge between them.

            # Draw edge to nearest node.
            hit = list(nodetree.nearest(node_a))[0] # Seek nearest node.
            current.add_edge(hit, a)
            hit = list(nodetree.nearest(node_b))[0] # Seek nearest node.
            current.add_edge(hit, b)

            # Insert just added edge endpoints for finding nearest neighbor (for subsequent iterations).
            nodetree.insert(a, node_a)
            nodetree.insert(b, node_b)

    elif strategy == "nearest_edge":
        raise Exception("todo.")
    
    else:
        raise Exception(f"Invalid merging strategy '{strategy}'.")
    
    return current


#######################################
###  Graph coordinate transformations
#######################################

# Obtain middle latitude coordinate for bounding box that captures all nodes in the graph.
def middle_latitute(G):
    uvk, data = zip(*G.nodes(data=True))
    df = gpd.GeoDataFrame(data, index=uvk)
    alat, alon = df["y"].mean(), df["x"].mean()
    return alat


# Compute relative positioning.
# Note: Takes a reference latitude for deciding on the GSD. Make sure to keep this value consistent when applied to different graphs.
# Note: Flip y-axis by subtracting from minimal latitude value (-max_lat) to maintain directionality.
# todo: Rely on UTM conversion instead of your hacky solution.
def transform_geographic_coordinates_into_scaled_pixel_positioning(G, reflat):
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
        return {'x': gsd * x, 'y': maxy - gsd * y }
    # Construct relabel mapping and transform node coordinates to relative scaled pixel position.
    relabel_mapping = {}
    for nid, data in G.nodes(data=True):
        relabel_mapping[nid] = latlon_to_relative_pixelcoord(data)
    nx.set_node_attributes(G, relabel_mapping)
    # 3. Convert back into simplified graph.
    return simplify_graph(G) # If it has curvature it crashes because of non-hashable numpy array in attributes.
    return G
    

# Transform graphnodes UTM coordinate system into latitude-longitude coordinates.
def graph_transform_utm_to_latlon(G, place):

    G = G.copy()
    letter, number = zone_letters[place], zone_numbers[place]

    def transformer(row): 
        x, y = row["x"], row["y"]
        lat, lon = coord_to_latlon_by_place((x, y), place)
        return {'x': lon, 'y': lat }

    relabel_mapping = {}
    for nid, data in G.nodes(data=True):
        relabel_mapping[nid] = transformer(data)

    nx.set_node_attributes(G, relabel_mapping)
    return G


# Transform graphnodes latitude-longitude coordinates into UTM coordinate system.
def graph_transform_latlon_to_utm(G):

    G = G.copy()

    def transformer(row): 
        lat, lon = row["y"], row["x"]
        x, y = latlon_to_coord((lat, lon))
        return {'x': x, 'y': y }
        
    relabel_mapping = {}
    for nid, data in G.nodes(data=True):
        relabel_mapping[nid] = transformer(data)

    nx.set_node_attributes(G, relabel_mapping)
    return G


# Transform graphnodes x,y coordinates by custom function.
def graph_transform_coordinates(G, transformer):

    G = G.copy()

    relabel_mapping = {}
    for nid, data in G.nodes(data=True):
        x, y = data["x"], data["y"]
        x, y = transformer(x, y)
        relabel_mapping[nid] = {"x": x, "y": y}

    nx.set_node_attributes(G, relabel_mapping)
    return G


# Derive UTM zone number and zone letter of a graph by taking arbitrary latlon coordinate from graph.
def graph_utm_info(G):
    node = G._node[list(G.nodes())[0]]
    lat, lon = node['y'], node['x']
    _, _, zone_number, zone_letter = utm.conversion.from_latlon(lat, lon)
    return {"zone_number": zone_number, "zone_letter": zone_letter}


#######################################
### Task specific graph preprocessing
#######################################

# Preparing graph for APLS usage (simplified, multi-edge, all edges have geometry property, all edges have an edge length property).
def graph_prepare_apls(G):
    G = nx.MultiGraph(simplify_graph(graph_transform_latlon_to_utm(G)))
    G = graph_annotate_edge_length(G)
    G = graph_add_geometry_to_straight_edges(G) # Add geometry for straight line segments (edges with no curvature).
    # G.graph['crs'] = "EPSG:4326" # Set EPSG might be necessary for plotting results within APLS logic.
    return G


# Preparing graph for TOPO usage (simplified, multi-edge, all edges have geometry property, all edges have an edge length property).
def graph_prepare_topo(graph):
    t0 = time()
    graph = simplify_graph(graph)
    graph = graph.to_directed(graph)
    graph = nx.MultiGraph(graph)
    graph = graph_add_geometry_to_straight_edges(graph)
    graph = graph_annotate_edge_length(graph)
    verbose_print(f"Simplified graph in {int(time() - t0)} seconds.")
    return graph
