from external import *
from graph.edge_cutting import *
from graph.deduplicating import *
from utilities import *
from networkx import Graph

### Write and reading graphs.

def write_graph(location: str, G: Graph) -> None:
    print(f"Writing graph to {location}")

    parent_dir = os.path.dirname(location)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    write_pickle(location, graph_to_pickle(G))


def read_graph(location: str) -> Graph:
    return pickle_to_graph(read_pickle(location))

# Convert graph into dictionary so it can be pickled.
def graph_to_pickle(G):

    return {
        "graph": G.graph,
        "nodes": list(iterate_nodes(G)),
        "edges": list(iterate_edges(G))
    }

# Convert dictionary (retrieved from pickled data) into a graph.
def pickle_to_graph(data):

    if data["graph"]["simplified"]:
        G = nx.MultiGraph()
    else:
        G = nx.Graph()

    G.graph = data["graph"]
    G.add_nodes_from(data["nodes"])
    G.add_edges_from([(*eid, attrs) for eid, attrs in data["edges"]])

    return G

# Obtain `folderpath/vertices.csv` and `folderpath/edges.csv` from disk and construct vectorized graph from it.
def read_graph_csv(folderpath: str) -> Graph:

    edges_file_path    = f"{folderpath}/edges.csv"
    vertices_file_path = f"{folderpath}/vertices.csv"

    edges_df = pd.read_csv(edges_file_path)
    vertices_df = pd.read_csv(vertices_file_path)

    # Construct NetworkX graph.
    G = nx.Graph()

    # Look at column identifiers whether its stored as UTM or WSG.
    use_utm = 'x' in vertices_df.keys()

    # Insert vertices.
    for node in vertices_df.iterrows():
        if use_utm:
            i, x, y = itemgetter('id', 'x', 'y')(node[1]) 
            G.add_node(int(i), y=y, x=x)
        else:
            i, lat, lon = itemgetter('id', 'lat', 'lon')(node[1]) 
            G.add_node(int(i), y=lat, x=lon)

    # Insert edges.
    for edge in edges_df.iterrows():
        u, v = itemgetter('u', 'v')(edge[1]) 
        G.add_edge(int(u), int(v))

    # Annotate graph information.
    G.graph["coordinates"] = "utm" if use_utm else "latlon"
    G.graph["simplified"] = False

    # Annotate graph edges.
    graph_annotate_edges(G)

    return G

    

# Obtain file age (since last write).
def file_age(filename):
    if not os.path.exists(filename):
        return time()
    
    return time() - os.path.getmtime(filename)



# Read and/or write pickle or graph with a specific action to perform in case we failed to read.
@info()
def read(filename, is_graph=True):
    
    filename = f"{filename}.pkl"
    if is_graph:
        result = pickle_to_graph(pickle.load(open(filename, "rb")))
    else:
        result = pickle.load(open(filename, "rb"))

    return result


# Pickle-related.
def read_pickle(filename):
    return pickle.load(open(filename, "rb"))

def write_pickle(location, data):
    print(f"Writing pickle to {location}")

    parent_dir = os.path.dirname(location)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    with open(location, "wb") as f:
        pickle.dump(data, f)


# Read image from disk.

Image.MAX_IMAGE_PIXELS = None # Disable decompression bomb warning globally

def read_png(filepath):

    # Obtain metadata from PNG as a dictionary.
    def get_png_metadata(png):
        if hasattr(png, 'text'):
            return dict(png.text.items())
        else:
            return {}

    img = Image.open(filepath)
    metadata = get_png_metadata(img)
    img = img.convert("RGB")
    img = array(img)
    img = img.astype(float)
    return img, metadata

# Extend PNG metadata.
def extend_png_metadata(metadata, to_add):
    return metadata | to_add

# Write image to disk.
def write_png(location, png, metadata=None):
    print(f"Writing image to {location}")

    parent_dir = os.path.dirname(location)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    pnginfo = PngInfo()
    if metadata != None:
        for key, value in metadata.items():
            pnginfo.add_text(key, str(value))

    img = Image.fromarray(png, mode="RGB")
    img.save(location, pnginfo=pnginfo)


def workflow_update_image_with_pixelcoord_metadata():
    zooms = {"berlin": 16, "chicago": 17}
    for place in ["berlin", "chicago"]:
        png, metadata = read_png(f"data/sat/{place}.png")
        pixelcoords = read_pickle(f"data/sat/{place}.pkl")
        latlon = pixelcoords[0][0]
        pixelcoord = latlon_to_pixelcoord(latlon[0], latlon[1], zoom)
        metadata = metadata | {"y": pixelcoord[0], "x": pixelcoord[1], "zoom": zooms[place]}
        write_png(f"{place}.png", png, metadata=metadata)

def path_exists(path):
    return Path(path).exists()