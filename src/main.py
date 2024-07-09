# Library code
from dependencies import *
from coverage import *
from network import *
from rendering import *
from maps import *

# Convert Sat2Graph elements into a graph.
# * json_file: Sat2Graph inferred road network data.
# * pickle_file: Maps pixel coordinates.
def sat2graph_json_to_graph(json_file, pickle_file):

    elements = json.load(open(json_file, "r"))
    edges    = elements["graph"]["edges"]
    vertices = elements["graph"]["vertices"]
    coordinates = pickle.load(open(pickle_file, "rb"))

    # Sat2Graph image has a padding of 176.
    def padding_offset(v):
        return (v[0] + 176, v[1] + 176)

    # Add ID to nodes.
    nodeid = 1
    nodes = set()
    for v in vertices: 
        nodes.add(padding_offset(tuple(v)))

    # Add edge endpoints to nodes.
    for e in edges:
        v1 = e[0]
        v2 = e[1]
        nodes.add(padding_offset(tuple(v1)))
        nodes.add(padding_offset(tuple(v2)))

    # Add nodes to graph
    G = nx.Graph()
    D = {}
    for i,v in enumerate(nodes):
        D[v] = i
        w = coordinates[v[0]][v[1]]   
        G.add_node(i, y=w[0], x=w[1])

    # Add edges to graph:
    for e in edges:
        a = padding_offset(tuple(e[0]))
        b = padding_offset(tuple(e[1]))
        if a in D and b in D:
            G.add_edge(D[a], D[b])

plot_graph_presentation(G)
