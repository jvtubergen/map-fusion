# Library code
from dependencies import *
from coverage import *
from network import *
from rendering import *
from maps import *

# Convert Sat2Graph elements into a graph.
elements = json.load(open("chicago_zoomed.json", "r"))
edges    = elements["graph"]["edges"]
vertices = elements["graph"]["vertices"]
coordinates = pickle.load(open("chicago_zoomed.pkl", "rb"))

# Add ID to nodes.
nodeid = 1
nodes = set()
for v in vertices: 
    nodes.add(tuple(v))

# Add nodes to graph
G = nx.Graph()
D = {}
for i,v in enumerate(nodes):
    D[v] = i
    w = coordinates[v[0]][v[1]]   
    G.add_node(i, y=w[0], x=w[1])

# Add edges to graph:
for e in edges:
    a = tuple(e[0])
    b = tuple(e[1])
    if a in D and b in D:
        G.add_edge(D[a], D[b])
    # (Don't expect for all edge endpoints to have a valid node: Some are outside of range?)
    # assert a in nodes
    # assert b in nodes


plot_graph_presentation(G)
