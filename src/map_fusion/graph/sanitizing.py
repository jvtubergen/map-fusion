from ..utilities import *
from ..spatial_reference_systems import *
from .utilities import *
from networkx import Graph

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
    sanity_check_node_positions(G)
    nodes = extract_node_positions_dictionary(G)
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
    nodes = extract_node_positions_dictionary(G)
    if G.graph["simplified"]: 
        for eid, attrs in iterate_edges(G):
            a, b, k = eid
            ps = edge_curvature(G, a, b, k)
            if G.graph["coordinates"] == "latlon": # Convert to utm for computing in meters.
                ps = array([latlon_to_utm(latlon) for latlon in ps])
            if curve_length(ps) > 1000: # Expect reasonable curvature length.
                raise Exception("Expect edge length less than 1000 meters. Probably some y, x coordinate in edge curvature got flipped.")
            # Expect start and endpoint of edge curvature match the node position.
            ps = edge_curvature(G, a, b, k) # Expect startpoint matches curvature.
            try:
                if (not np.all(ps[0] == nodes[a])) and (not np.all(ps[-1] != nodes[b])):
                    raise Exception("Expect curvature have same directionality as edge start and end edge.")
            except Exception as e:
                print(traceback.format_exc())
                print(e)
                breakpoint()
            
            assert "geometry" in attrs
            assert "curvature" in attrs


# Sanity check that all curvature annotations are numpy array.
def sanity_check_curvature_type(G):
    for eid, attrs in iterate_edges(G):
        check(type(attrs["curvature"]) == type(array([])))


# Sanity check all edges have non-zero edge length.
def sanity_check_edge_length(G):
    for eid, attrs in iterate_edges(G):
        check(attrs["length"] > 0)


# Sanity check nodes have unique position.
def sanity_check_node_positions(G, eps=0.0001):

    assert G.graph["coordinates"] == "utm" # Act only on UTM for epsilon to make sense.

    positions = extract_node_positions_dictionary(G)
    tree = graphnodes_to_rtree(G)
    bboxs = graphnodes_to_bboxs(G)

    for nid in G.nodes():

        # Find nearby nodes.
        bbox = pad_bounding_box(bboxs[nid], eps)
        nids = intersect_rtree_bbox(tree, bbox)
        assert len(nids) == 1 # Expect to only intersect with itself.


# Sanity check graph curvature starts/end at node positions _and_ in the correct direction (starting at `u` and ending at `v` with `u <= v`).
def sanity_check_graph_curvature(G):

    nid_positions = extract_node_positions_dictionary(G)
    for eid, attrs in iterate_edges(G):
        check("curvature" in attrs, expect="Expect every edge to have 'curvature' attribute annotation.")
        u, v = eid[:2]
        ps = attrs["curvature"]
        check(u <= v, expect="Expect `u < v` for all edges.")
        p, q = nid_positions[u], nid_positions[v]
        check(np.all(p == ps[0]), expect="Expect curvature of all connected edges starts/end at node position.")
        check(np.all(q == ps[-1]), expect="Expect curvature of all connected edges starts/end at node position.")



#######################################
### New sanitization logic.
#######################################

EPS=0.0001

def sanitize_graph(G: Graph) -> Graph:
    """Sanitize Berlin|Chicago OSM|SAT|GPS graph for map fusion and experiments.
    Expects as input a graph that is simplified in WSG coordinates, then constructs and returns a graph with the following properties:
    * Simplified.
    * UTM coordinates.
    * Duplicated nodes deleted.
    * Edge length annotated.
    * Edge curvature annotated.
    * Zero-length edges deleted.
    """
    from .deduplicating import deduplicate_graph
    from .simplifying import simplify_graph
    G.graph["simplified"]  = False
    G.graph["coordinates"] = "latlon"
    G = graph_annotate_edges(G)
    G = graph_transform_latlon_to_utm(G)
    G = graph_annotate_edges(G)
    G = deduplicate_graph(G)
    G = simplify_graph(G)
    return G


def sanity_check_graph(G: Graph) -> None:
    """
    Check sanity of graph:
    * simplified
    * in utm coordinates
    * every node has a unique coordinate
    * every edge has curvature
    * every edge has a length
    """

    check(G.graph["simplified"] == True,    expect="Expect graph to be simplified.")
    check(type(G) == type(nx.MultiGraph()), expect="Expect graph to be an undirected multi-graph.")
    check(G.graph["coordinates"] == "utm",  expect="Expect graph to be in UTM coordinates.")

    sanity_check_edges(G)
    sanity_check_nodes(G)
    

def sanity_check_nodes(G):
    """
    Sanity check nodes 
    * coordinate consistency: All coordinates larger than 100 (we're in UTM)
    * to have unique position.
    """

    # Coordinate node coordinates in UTM (and not in WSG).
    positions = extract_node_positions_list(G)
    check(np.min(positions) > 100, expect="Expect graph in utm coordinate system. Big chance some node coordinates are in latlon coordinate system.")

    # Check unique position.
    tree = graphnodes_to_rtree(G)
    bboxs = graphnodes_to_bboxs(G)
    for nid in G.nodes():
        # Find nearby nodes.
        bbox = pad_bounding_box(bboxs[nid], EPS)
        nids = intersect_rtree_bbox(tree, bbox)
        check(len(nids) == 1, expect="Expect to only intersect with itself.")


def sanity_check_edges(G):
    """
    Sanity check edges
    * has curvature attribute
    * curvature is a numpy array
    * curvature starts/end at node positions
    * curvature moves in the correct direction (starting at `u` and ending at `v` with `u <= v`).
    * has length attribute
    * length is non-zero yet smaller than 1000
    """

    nid_positions = extract_node_positions_dictionary(G)
    for eid, attrs in iterate_edges(G):
        # Edge curvature.
        check("curvature" in attrs, expect="Expect every edge to have 'curvature' attribute annotation.")
        check(type(attrs["curvature"]) == type(array([])), expect="Expect that all curvature annotations are numpy array.")
        # * Edge curvature direction.
        u, v = eid[:2]
        ps = attrs["curvature"]
        check(u <= v, expect="Expect `u < v` for all edges.")
        p, q = nid_positions[u], nid_positions[v]
        check(np.all(p == ps[0]), expect="Expect curvature of all connected edges starts/end at node position.")
        check(np.all(q == ps[-1]), expect="Expect curvature of all connected edges starts/end at node position.")
        # Edge length.
        check(attrs["length"] > EPS, expect="Expect non-zero edge length.")
        check(attrs["length"] < 10000, expect="Expect edge length less than 10000 meters. Probably some y, x coordinate in edge curvature got flipped.")
    
