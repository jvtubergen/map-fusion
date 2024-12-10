from external import *
from coordinates import *

# Valid graph sets to work with. GPS: Roadster, Sat: Sat2Graph, Truth: OpenStreetMaps. Extend with techniques as you see fit.
graphsets = ["roadster", "sat2graph", "openstreetmaps", "mapconstruction", "intersection", "merging_A", "merging_B", "merging_C"]
places    = ["athens", "berlin", "chicago"]


### Reading graph information.

def get_graph_path(graphset=None, place=None):
    assert graphset in graphsets
    assert place in places
    return f"data/graphs/{graphset}/{place}"


# Obtain (inferred/ground truth) graph from disk.
# Construct graph from edges.txt and vertex.txt text file in specified folder. 
# Expect those files to be CSV with u,v and id,x,y columns respectively.
# We act only on undirected vectorized graphs.
def read_graph(graphset=None, place=None, use_utm=False):

    folder = get_graph_path(graphset=graphset, place=place)
    edges_file_path    = folder + "/edges.txt"
    vertices_file_path = folder + "/vertices.txt"

    # Expect the text files are CSV formatted
    edges_df = pd.read_csv(edges_file_path)
    vertices_df = pd.read_csv(vertices_file_path)

    # Construct NetworkX graph.
    G = nx.Graph()

    # Track node dictionary to simplify node extraction when computing edge lengths.
    for node in vertices_df.iterrows():
        if use_utm:
            i, x, y = itemgetter('id', 'x', 'y')(node[1]) 
            G.add_node(int(i), y=y, x=x)
        else:
            i, lat, lon = itemgetter('id', 'lat', 'lon')(node[1]) 
            G.add_node(int(i), y=lat, x=lon)

    for edge in edges_df.iterrows():
        u, v = itemgetter('u', 'v')(edge[1]) 
        G.add_edge(int(u), int(v))

    # G = nx.MultiDiGraph(G)
    # G = ox.simplify_graph(G)
    # G.graph['crs'] = "EPSG:4326"
    # Generate undirected, vectorized graph.
    G.graph["coordinates"] = "latlon"
    G.graph["simplified"] = False
    return G


### Writing graph information.

# Save a graph to storage as a GraphML format.
# * Optionally overwrite if the target file already exists.
# * Writes the file into the graphs folder.
def write_graph(G, graphset=None, place=None, overwrite=False, use_utm=False):
    assert graphset in graphsets
    assert place in places
    graph_path = get_graph_path(graphset=graphset, place=place)
    if Path(graph_path).exists() and not overwrite:
        raise Exception(f"Did not save graph: Not allowed to overwrite existing file at {graph_path}.")
    if not Path(graph_path).exists():
        path = Path(graph_path)
        path.mkdir(parents=True)
    # Expect a vectorized undirected graph.
    assert type(G) == nx.Graph
    assert not G.graph.get("simplified")
    # Vertices.
    vertices = G.nodes(data=True)
    vertices = pd.DataFrame.from_records(list(vertices), columns=['id', 'attrs'])
    vertices = pd.concat([vertices[['id']], vertices['attrs'].apply(pd.Series)], axis=1)
    if use_utm:
        raise Exception("Should only store latlon graphs to prevent data issues.")
        vertices.to_csv(f"{graph_path}/vertices.txt", index=False, columns=["id", "x", "y"])    
    else:
        vertices = vertices.rename(columns={"x": "lon", "y": "lat"})
        vertices.to_csv(f"{graph_path}/vertices.txt", index=False, columns=["id", "lat", "lon"])    
    # Edges
    edges = G.edges(data=True)
    edges = pd.DataFrame.from_records(list(edges), columns=['u', 'v', ""])
    edges.to_csv(f"{graph_path}/edges.txt", index_label="id", columns=["u", "v"])

