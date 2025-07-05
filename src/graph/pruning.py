from external import *
from utilities import *
from graph.utilities import *
from graph.simplifying import *

# Prune graph with threshold-annotated edges.
# * TODO: Only add edge connections with sat if gps edge as adjacent covered edges (thus concatenate with `merge_graph` logic).
# * Rather than filtering out below threshold, we can as well seek edges above threshold (thus inverting the result).
@info()
def prune_coverage_graph(G, prune_threshold=10, invert=False):

    check(G.graph["simplified"], expect="Expect the graph is simplified when edges have a threshold annotated.")
    check(prune_threshold <= G.graph['max_threshold'], expect="Expect we are pruning (on a threshold) below the maximum computed.")

    if invert:
        attribute_filter = lambda attrs: attrs["threshold"] > prune_threshold
    else:
        attribute_filter = lambda attrs: attrs["threshold"] <= prune_threshold

    retain = filter_eids_by_attribute(G, filter_func=attribute_filter)
    G = G.edge_subgraph(retain)

    # We have to simplify this graph again as some crossroads may have disappeared.
    G = simplify_graph(G, retain_attributes=True)

    return G

