from dependencies import *
from network import *
from pcm import *
from maps import *
from curve import *
from rendering import *

###################################
###  Curve by curve coverage
###################################

# Curve coverage of ps by qs.
# either return false or provide subcurve with step sequence
# TODO: Optimization to check on bounding boxes before doing the interpolation.
def curve_by_curve_coverage(ps, qs, lam):
    return is_partial_curve_undirected(to_curve(ps), to_curve(qs), lam)


# Check coverage of a curve by a curve-set.
def curve_by_curveset_coverage(ps, qss, lam):
    for qs in qss:
        if is_partial_curve_undirected(to_curve(ps), to_curve(qs), lam):
            return True
    return False


###################################
###  Curve by network coverage
###################################


# Extract path from a network.
# 1. Construct minimal bounding box for area.
# 2. Obtain vectorized graph.
# 3. Extract nodes within area.
# 4. Construct subnetwork.
# 5. Generate all _simple edge_ paths within subnetwork.
# 6. Check for curve by curve set coverage.

# Generate a random curve.
def random_curve(length = 100, a = np.array([-10,-10]), b = np.array([10,10])):
    ps = np.random.random((length, 2))
    return a + (b - a) * ps
        
        
# Length of curve.
def curve_length(ps):
    length = 0
    for p1, p2 in zip(ps, ps[1:]):
        length += np.linalg.norm(p1 - p2)
    return length


# Extract bounding box on a curve. Use padding to lambda pad.
def bounding_box(ps, padding):
    padding = np.array([padding, padding])
    lower = [np.min(ps[:,0]), np.min(ps[:,1])]
    higher = [np.max(ps[:,0]), np.max(ps[:,1])]
    return np.array([lower - padding, higher + padding])


# Pad a bounding box.
def pad_bounding_box(bb, padding):
    padding = np.array([padding, padding])
    return np.array([bb[0] - padding, bb[1] + padding])


# Construct R-Tree on graph nodes.
def graphnodes_to_rtree(G):
    idx = rtree.index.Index()
    for node, data in G.nodes(data = True):
        x, y = data['x'], data['y']
        idx.insert(node, (x, y, x, y))
    return idx


# Construct R-Tree on graph edges.
def graphedges_to_rtree(G):
    assert type(G) == nx.MultiGraph
    edgetree = rtree.index.RtreeContainer()
    for uvk in G.edges(keys=True):
        u, v, k = uvk
        curvature = edge_curvature(G, u, v, k=k)
        minx = min(curvature[:,0])
        maxx = max(curvature[:,0])
        miny = min(curvature[:,1])
        maxy = max(curvature[:,1])
        edgetree.insert(uvk, (minx, miny, maxx, maxy))
    return edgetree


# Construct dictionary that links edge id to a bounding box.
def graphedges_to_bboxs(G):
    assert type(G) == nx.MultiGraph
    bboxs = {}
    for uvk in G.edges(keys=True):
        u, v, k = uvk
        curvature = edge_curvature(G, u, v, k=k)
        minx = min(curvature[:,0])
        maxx = max(curvature[:,0])
        miny = min(curvature[:,1])
        maxy = max(curvature[:,1])
        bbox = array([(minx, miny), (maxx, maxy)])
        bboxs[uvk] = bbox
    return bboxs



# Compute per (simplified) edge of S (Source graph) the coverage threshold in order to be matched by T (Target graph).
def edge_wise_coverage_threshold(S, T):

    # Consistency.
    assert S.graph["simplified"]
    assert type(S) == nx.MultiGraph
    assert T.graph["simplified"]
    assert type(T) == nx.MultiGraph

    # Assume S and T are a MultiGraph, simplified, and in appropriate coordinate system to measure distance differences in meter.
    lam = 1 # Start with a threshold of 1 meter.

    edgebboxs= graphedges_to_bboxs(S) # Have a bounding box per edge so we can quickly pad for intersection test against T.
    edgetree = graphedges_to_rtree(T) # Place graph edges by coordinates in accelerated data structure (R-Tree).
    edges_todo = S.edges(keys=True)

    nodedict = extract_nodes_dict(S)

    edge_results = {}

    while len(edges_todo) > 0:
        print(f"Lambda: {lam}. Edges: {len(edges_todo)}")
        for uvk in edges_todo:
            # print(uvk)
            u, v, k = uvk
            bbox = pad_bounding_box(edgebboxs[uvk], lam) # Pad bounding box.
            edges = array(list(edgetree.intersection((bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])))) # Extract edges within bounding box.
            # edges = array(list(edgetree.intersection(bbox))) # Extract edges within bounding box.
            if len(edges) == 0: # No nearby edges found so this curve is not covered.
                print(f"Lambda: {lam}. Insufficient nearby edges for {uvk}.")
                continue
            nodes = list(set(np.append(edges[:,0], edges[:,1])))
            if len(nodes) == 0: # No nearby nodes found so this curve is not covered.
                print(f"Lambda: {lam}. Insufficient nearby nodes for {uvk}.")
                continue
            subT = T.subgraph(list(nodes))
            if len(subT.edges()) == 0: # Empty generated subgraph.
                print(f"Lambda: {lam}. Generated empty subT for {uvk}.")
                continue

            # Extract edges which are near start and end point.
            pu, pv = nodedict[u], nodedict[v]
            # Pad pu and pv.
            pu = pad_bounding_box(array([[pu[0], pu[1]], [pu[0], pu[1]]]), lam)
            pv = pad_bounding_box(array([[pv[0], pv[1]], [pv[0], pv[1]]]), lam)
            start_edges = list(edgetree.intersection((pu[0][0], pu[0][1], pu[1][0], pu[1][1])))
            end_edges   = list(edgetree.intersection((pv[0][0], pv[0][1], pv[1][0], pv[1][1])))
            if len(start_edges) == 0 or len(end_edges) == 0: # No nearby edge so this curve is not covered.
                print(f"Lambda: {lam}. Insufficient nearby edges for {uvk}.")
                continue
            # Obtain start nodes and end nodes, then figure out what the ~maximal~ paths are. (TODO: Reduce set to maximal paths only.)
            start_nodes = set()
            for abl in start_edges:
                a, b, _ = abl
                start_nodes.add(a)
                start_nodes.add(b)
            end_nodes = set()
            for abl in end_edges:
                a, b, _ = abl
                end_nodes.add(a)
                end_nodes.add(b)
            # Find all valid simple paths from start-node to end node
            #   New step: check whether partial matches the source curve and add start/end edges afterwards.
            valid_edges = []
            ps = edge_curvature(S, u, v, k=k) # == array(list(S.get_edge_data(uvk[0],uvk[1],uvk[2])["geometry"].coords)
            for (a, b, l) in subT.edges(keys=True):
                # Check for partial curve matching.
                edgeT = subT.get_edge_data(a, b, l)
                qs = edge_curvature(subT, a, b, k=l)
                if is_partial_curve_undirected2(qs, ps, lam):
                    valid_edges.append((a, b, l))
            for abl in start_edges:
                valid_edges.append(abl)
            for abl in end_edges:
                valid_edges.append(abl)
            #   Remove all edges from set which are not part of 

            # edges2 = subT.edges(keys=True)
            # nodes  = set(nodes)
            # nodes2 = set(list(set(np.append(edges[:,0], edges[:,1]))))
            # breakpoint()

            # edges = set([(u, v, k) for u, v, k in edges.tolist()])
            # all_edges = set(subT.edges(keys=True))
            # nodes = set(nodes)
            # pruned_edges = set(pruned_edges)
            # print(len(pruned_edges), len(edges & pruned_edges))
            # print(len(pruned_edges), len(edges & pruned_edges))
            # print(len(edges ^ all_edges)) # How can edges differ...?
            # print(len(edges & all_edges)) # How can edges differ...?
            
            # # Ensure at least pu and pv are within bbox.
            # print(bbox[1] - pu)
            # print(bbox[1] - pv)
            # print(pu - bbox[0])
            # print(pv - bbox[0])
            # breakpoint()

            # Act on the edges of the subT graph.
            # edges2 = set(subT.edges(keys=True))
            #   Remove edges which are not valid (Cannot do so because subT is frozen, instead create new graph).
            valid_edges = array(valid_edges)
            nodes = list(set(np.append(valid_edges[:,0], valid_edges[:,1])))
            if len(nodes) == 0: # No nearby nodes found so this curve is not covered.
                print(f"Lambda: {lam}. Insufficient nearby nodes for pruned {uvk}.")
                continue
            subT = T.subgraph(list(nodes))
            if len(subT.edges()) == 0: # Empty generated subgraph.
                print(f"Lambda: {lam}. Generated empty subT for pruned {uvk}.")
                continue

            paths = [] # TODO: Optimize to filter out maximal paths.
            # Lazily evaluate paths: Often times the first path one succeeds, because of the aggressive pruning strategy applied.
            found_curve = False
            for a in start_nodes:
                if found_curve: # Pre-emptively cancel
                    break
                for b in end_nodes:
                    if found_curve:
                        break
                    # For each edge in the graph, check whether it partial matches the source curve.
                    #   If this is not the case, we can remove it from the graph and try again
                    count = 0
                    for path in nx.shortest_simple_paths(subT, a, b): 
                        if count > 10: # Expect to find a result in less paths taken.
                            breakpoint()
                            # plot_graph_and_curve(subT, array([pu, pv]))
                            # plot_graph_and_curve(subT, path_to_curve(subT, path))
                            # plot_graph_and_curves(subT, array([pu, pv]), path_to_curve(subT, path))
                        count += 1
                        assert len(path) > 0
                        if len(path) == 1:
                            nid = path[0]
                            if subT.get_edge_data(nid, nid) != None:
                                path = [nid, nid]
                        else:
                            found = True
                            for nida, nidb in zip(path, path[1:]):
                                if subT.get_edge_data(nida, nidb) == None:
                                    found = False
                            if not found:
                                breakpoint()

                        # Act on path.
                        qs = array([(subT.nodes()[path[0]]["x"], subT.nodes()[path[0]]["y"])])
                        for a, b in zip(path, path[1:]): # BUG: Conversion from nodes to path results in loss of information at possible multipaths.
                            edgepoints = edge_curvature(subT, a, b)
                            qs = np.append(qs, edgepoints[1:], axis=0) # Ignore first point when adding.
                        # Use points for partial curve matching.
                        if is_partial_curve_undirected(to_curve(ps), to_curve(qs), lam):
                            print("Found partial curve within threshold.")
                            found_curve = True
                            break
            if not found_curve:
                print(f"Lambda: {lam}. Insufficient nearby curve for {uvk}.")
            else:
                print(f"Lambda: {lam}. Edge {uvk} has lambda distance threshold met.")
                # breakpoint()
                # Removing egde uvk from edges_todo.
                edges_todo = edges_todo - set([uvk])
                edge_results[uvk] = lam

        lam += 1 # Increment lambda
            
    return edge_results


###################################
###  Tests: Curve by curve coverage
###################################

# Test:
# * Generate a curve randomly
# * Per point generate three in range
# * Pick one of those and represent as curve
# * Pool unused nodes with some more randomly generated curves
# * Add some arbitrary other nodes of these and add to curve
# Verify:
# * Expect to find some subcurve
# * Generated subcurve within distance lambda
    
