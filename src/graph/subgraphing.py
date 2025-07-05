from external import *
from utilities import *
from graph.utilities import *

# Extract subgraph by a point and a radius (using a square rather than circle for distance measure though).
def extract_subgraph(G, ps, lam):
    edgetree = graphedges_to_rtree(G) # Place graph edges by coordinates in accelerated data structure (R-Tree).
    bbox = bounding_box(ps, lam)
    edges = list(edgetree.intersection((bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]))) # Extract edges within bounding box.
    subG = G.edge_subgraph(edges)
    return subG


# Cut out ROI subgraph. (Drop any edge with an endpoint beyond the ROI.)
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