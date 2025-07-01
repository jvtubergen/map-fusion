from external import *
from graph_curvature import *
from graph_deduplicating import *
from utilities import *

# Valid graph sets to work with. GPS: Roadster, Sat: Sat2Graph, Truth: OpenStreetMaps. Extend with techniques as you see fit.
graphsets = ["roadster", "sat2graph", "openstreetmaps", "mapconstruction", "intersection", "merge_A", "merge_B", "merge_C"]
places    = ["athens", "berlin", "chicago"]

# Link active graphset to representative of gps, sat, and truth.
links = {
    "gps": "roadster",
    "osm": "openstreetmaps",
    "sat": "sat2graph"
}

### Write and reading graphs.

def write_graph(location, G):
    print(f"Writing graph to {location}")

    parent_dir = os.path.dirname(location)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    write_pickle(location, graph_to_pickle(G))


def read_graph(location):
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
def read_graph_csv(folderpath):

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
    graph_annotate_edge_curvature(G)
    graph_annotate_edge_length(G)
    graph_annotate_edge_geometry(G)

    return G

    

# Obtain file age (since last write).
def file_age(filename):
    if not os.path.exists(filename):
        return time()
    
    return time() - os.path.getmtime(filename)



# Read and/or write pickle or graph with a specific action to perform in case we failed to read.
@info()
def read_and_or_write(filename, action, use_storage=True, is_graph=True, overwrite=False, rerun=False, reset_time=None, overwrite_if_old=False):
    
    filename = f"{filename}.pkl"

    result = None
    file_exists = os.path.exists(filename)
    
    is_old = file_age(filename) > reset_time if reset_time != None else True

    # If we provide a reset time, it can set the overwrite and rerun variable.
    if reset_time != None and is_old:
        hours = file_age(filename) / 3600
        logger(f"Age of file ({hours} hours) exceeds reset time. Rerunning.")
        rerun = True

    # Reading previous result from disk.
    if file_exists and use_storage and not rerun: # No need to read if we are going to rerun.
        logger("Try reading file from disk.")
        try:
            if is_graph:
                result = pickle_to_graph(pickle.load(open(filename, "rb")))
            else:
                result = pickle.load(open(filename, "rb"))
        except Exception as e:
            logger(traceback.format_exc())
            logger(e)
            logger(f"Failed to read {filename}. Running instead.")
        
    # Rerunning result.
    if type(result) == type(None):
        logger("Performing action.")
        result = action()

    # Store (overwrite) data.
    # * We save if the file does not exist or we mention to overwrite (which can be so only if outdated file).
    if (not file_exists) or overwrite or (is_old and overwrite_if_old):
        logger(f"(Over)writing {filename}")
        if is_graph:
            pickle.dump(graph_to_pickle(result), open(filename, "wb"))
        else:
            pickle.dump(result, open(filename, "wb"))

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