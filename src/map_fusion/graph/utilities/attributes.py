from external import *
from utilities import *
from graph.utilities.general import *

#######################################
### Attribute retrieval general
#######################################

def get_node_attributes(G, nid):
    return G.get_node_data(nid)


# Get specific edge from graph. Using built-in `eid` filter hopefully improves performance (in comparison to list filtering).
def get_edge_attributes(G, eid):
    # eid = format_eid(G, eid)
    return G.get_edge_data(*eid)


def graphedge_curvature(G, eid):
    attrs = get_edge_attributes(G, eid)
    ps = attrs["curvature"]
    return ps


def graphedge_length(G, eid):
    attrs = get_edge_attributes(G, eid)
    ps = attrs["length"]
    return ps


# Enum to differentiate between node or edge-related task (used by abstract functions).
class GraphEntity(Enum):
    Edges = 0
    Nodes = 1


# Abstract function to filter node/edge identifiers by their attributes (can be both a subset of attributes or a filter function on attributes).
def abstract_filter_by_attribute(entity, G, filter_attributes=None, filter_func=None):

    check(type(entity) == GraphEntity)
    check(filter_func != None or filter_attributes != None, expect="Expect to filter either by attributes dictionary or a filter function.")

    match entity:
        case GraphEntity.Edges:
            iterator = iterate_edges
        case GraphEntity.Nodes:
            iterator = iterate_nodes

    if filter_func != None:

        return [identifier for identifier, attrs in iterator(G) if filter_func(attrs)]
    
    if filter_attributes != None:

        filtered_identifiers = []
        for identifier, attrs in iterator(G):

            found = True
            for filter_attr in filter_attributes.keys():
                if filter_attr not in attrs or attrs[filter_attr] != filter_attributes[filter_attr]:
                    found = False

            if found:
                filtered_identifiers.append(identifier)
    
        return filtered_identifiers


# Filter out nodes either by a collection of attributes or a specific filtering function.
filter_nids_by_attribute = lambda *params, **named: abstract_filter_by_attribute(GraphEntity.Nodes, *params, **named)


# Filter out edges either by a collection of attributes or a specific filtering function.
filter_eids_by_attribute = lambda *params, **named: abstract_filter_by_attribute(GraphEntity.Edges, *params, **named)


#######################################
### Attribute setting general
#######################################

def set_node_attributes(G, nid, attrs):
    nx.set_node_attributes(G, {nid: attrs})


# Set edge attribute.
def set_edge_attributes(G, eid, attrs):
    # eid = format_eid(G, eid)
    nx.set_edge_attributes(G, {eid: attrs})


# Annotate nodes by appending new attributes (optionally to a subselection of node identifiers.
def annotate_nodes(G, new_attrs, nids=None):
    if nids != None:
        if len(nids) == 0: # Don't act on an empty list.
            return
        check(type(nids[0]) == type(1) , expect="Expect to receive node identifiers (it probably has received edge identifiers).")
        nx.set_node_attributes(G, {nid: {**attrs, **new_attrs} for nid, attrs in iterate_nodes(G) if nid in nids}) 
    else:
        nx.set_node_attributes(G, {nid: {**attrs, **new_attrs} for nid, attrs in iterate_nodes(G)}) 


# Annotate edges by appending new attributes (optionally to a subselection of edge identifiers.
def annotate_edges(G, new_attrs, eids=None):
    if eids != None:
        if len(eids) == 0: # Don't act on an empty list.
            return
        check(len(eids[0]) == 2 or len(eids[0]) == 3, expect="Expect to receive edge identifiers (it probably has received node identifiers).")
        nx.set_edge_attributes(G, {eid: {**attrs, **new_attrs} for eid, attrs in iterate_edges(G) if eid in eids}) 
    else:
        nx.set_edge_attributes(G, {eid: {**attrs, **new_attrs} for eid, attrs in iterate_edges(G)}) 


#######################################
### Graph edge attribute annotation.
#######################################

from graph.utilities.distance import graphnode_position

def graph_annotate_edges(G):
    """ Annotate each edge with (corrected) curvature, geometry, length and delete if zero-length edge length."""

    eids_to_delete = []
    for eid, attrs in iterate_edges(G):
        graph_annotate_edge(G, eid, attrs=attrs)
        if eid[0] == eid[1] and attrs["length"] <= 0.0001:
            eids_to_delete.append(eid)

    G.remove_edges_from(eids_to_delete)
    
    return G


def graph_annotate_edge(G, eid, attrs = None, from_geometry = False):
    if attrs == None:
        attrs = get_edge_attributes(G, eid)
    # Retrieve curvature from (curvature or geometry) attribute or construct straight line segment.
    if "curvature" in attrs:
        if type(attrs["curvature"]) != type(array([])):
            ps = array(attrs["curvature"])
        else:
            ps = attrs["curvature"]
    elif "geometry" in attrs and from_geometry:
        ps = from_linestring(attrs["geometry"])
    else: # Construct straight line segment.
        u, v = eid[0:2]
        p1 = G.nodes()[u]
        p2 = G.nodes()[v]
        ps = array([[p1["y"], p1["x"]], [p2["y"], p2["x"]]])

    # Check start/end position of curvature.
    u, v = eid[0:2]
    p1 = graphnode_position(G, u)
    p2 = graphnode_position(G, v)
    is_correct_direction = np.all(array(ps[0]) == array(p1)) and np.all(array(ps[-1]) == array(p2))
    is_inverted_direction = np.all(array(ps[0]) == array(p2)) and np.all(array(ps[-1]) == array(p1))
    check(is_correct_direction or is_inverted_direction, expect="Expect curvature of all connected edges starts/end at node position.")

    # In case the direction of the curvature is inverted.
    if u != v and is_inverted_direction: 
        k = 0 if len(eid) == 2 else eid[2]
        # Then invert the direction back.
        # logger("Invert curvature of edge ", (u, v, k))
        ps = ps[::-1]

    geometry = to_linestring(ps)
    length = curve_length(ps)

    attrs["curvature"] = ps
    attrs["geometry"] = geometry
    attrs["length"] = length
    