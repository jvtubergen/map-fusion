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

### Reading graph information.

def get_graph_path(graphset=None, place=None):
    assert graphset in graphsets
    assert place in places
    return f"data/graphs/{graphset}/{place}"


# Obtain (inferred/ground truth) graph from disk.
# Construct graph from edges.txt and vertex.txt text file in specified folder. 
# Expect those files to be CSV with u,v and id,x,y columns respectively.
# We act only on undirected vectorized graphs.
@info(timer=True)
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

    # Annotate edges.
    graph_annotate_edge_curvature(G)
    graph_annotate_edge_length(G)
    graph_annotate_edge_geometry(G)

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
    edges = iterate_edges(G)
    edges = pd.DataFrame.from_records(list(edges), columns=['u', 'v', ""])
    edges.to_csv(f"{graph_path}/edges.txt", index_label="id", columns=["u", "v"])


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

def write_pickle(filename, data):
    return pickle.dump(data, open(filename, "wb"))


# Image related

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
def write_png(filepath, png, metadata=None):

    pnginfo = PngInfo()
    if metadata != None:
        for key, value in metadata.items():
            pnginfo.add_text(key, str(value))

    img = Image.fromarray(png, mode="RGB")
    img.save(filepath, pnginfo=pnginfo)


def workflow_update_image_with_pixelcoord_metadata():
    zooms = {"berlin": 16, "chicago": 17}
    for place in ["berlin", "chicago"]:
        png, metadata = read_png(f"data/sat/{place}.png")
        pixelcoords = read_pickle(f"data/sat/{place}.pkl")
        latlon = pixelcoords[0][0]
        pixelcoord = latlon_to_pixelcoord(latlon[0], latlon[1], zoom)
        metadata = metadata | {"y": pixelcoord[0], "x": pixelcoord[1], "zoom": zooms[place]}
        write_png(f"{place}.png", png, metadata=metadata)
