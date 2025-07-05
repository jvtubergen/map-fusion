from external import *
from utilities import *
    
get_eids = lambda G: [eid for eid, _ in iterate_edges(G)]
get_nids = lambda G: list(G.nodes())

# Iterate all edge identifiers alongside their attributes. Iterated element attributes are overwritable.
def iterate_edges(G):
    if G.graph["simplified"]:
        for u, v, k, attrs in G.edges(data=True, keys=True):
            if u <= v: 
                yield (u, v, k), attrs
            else:
                yield (v, u, k), attrs
    else:
        for u, v, attrs in G.edges(data=True):
            if u <= v: 
                yield (u, v), attrs
            else:
                yield (v, u), attrs


# Iterate all edge identifiers alongside their attributes. Iterated element attributes are overwritable.
def iterate_nodes(G):
    for nid, attrs in G.nodes(data=True):
        yield nid, attrs


#######################################
### Graph related
#######################################

# Format edge identifier to graph type.
def format_eid(G, eid):

    u, v = sorted(eid[:2])

    # Obtain `k`.
    k = 0
    if len(eid) == 3:
        k = eid[2]

    # Pair or triplet.
    eid = u, v
    if G.graph["simplified"]:
        eid = u, v, k 
    
    return eid

# Obtain connected edge identifiers to the provided node identifier.
# Note: `(u, v)` or `(u, v, k)` always have lowest nid first (thus `u <= v`).
def get_connected_eids(G, nid):
    if G.graph["simplified"]:
        return [(u, v, k) if u <= v else (v, u, k) for (u, v, k) in list(G.edges(nid, keys=True))]
    else:
        return [(u, v) if u <= v else (v, u) for (u, v) in list(G.edges(nid))]
