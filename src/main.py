# Library code
from dependencies import *
from coverage import *
from network import *
from rendering import *

cos = math.cos
pi  = math.pi


# Read image file as numpy array.
def read_image(filename):
    image = pil.Image.open(filename)
    image = image.convert("RGB")
    image = np.array(image)
    return image


# Write numpy array with size (width, height, RGB) as an image file.
def write_image(image, filename):
    pil.Image.fromarray(image).save(filename) 


# Cut out left and bottom 44 pixels.
def cut_logo(image):
    return image[44:-44,44:-44,:]


# Compute the distance covered by the surface visible in the image.
# NOTE: Assume image is taken orthogonally.
def earth_surface_distance(zoom):
    # Earth radius is approximated to 6378137 meters.
    # At zero zoom, an image of the entire globe is taken.
    return 2 * math.pi * 6378137 / math.pow(2, zoom)

earth_circumference = earth_surface_distance(0)

# What zoom we should adhere to for obtaining a certain gsd.
# GSD (Ground Sampling Distance), or meter per pixel
# NOTE: We assume satellite images are taken from above.
#       Therefore, latitudal and longitudal GSD is equal.
#       Thereby, latitude per pixel and longitude per pixel differs.
def compute_gsd(resolution, zoom):
    d = earth_surface_distance(zoom)
    meter_per_pixel = d/resolution
    return meter_per_pixel

# compute_gsd(1280, 17) == 0.4777314267823516
# Stick to a zoom level of 17 to obtain a GSD of 0.5.

# We want to compute longitudal and latitudal changes on image corners.

def latlon_to_meters(lat, lon):
    y = lat * earth_circumference / 360.0
    x = lon * earth_circumference * cos(pi*lat/180) / 360.0
    return y, x


# NOTE: When latitude changes, the value of x represents different longitude.
def meters_to_latlon(y, x):
    lat = y * 360.0 / earth_circumference
    lon = x * 360.0 / (earth_circumference * cos(pi*lat/180))
    return lat, lon


# Compute new latlon after walking x,y meters.
def walk(lat, lon, dy, dx):
    # It should be irrelevant if we walk first x or y.

    # Compute current y.
    (y,_) = latlon_to_meters(lat, 0)
    # Compute new y.
    y2 = y + dy
    # Compute new latitude.
    (lat2,_) = meters_to_latlon(y2, 0)
    # Compute x at new longitude.
    (_,x) = latlon_to_meters(lat2, lon)
    # Compute new x.
    x2 = x + dx
    # Compute new latitude.
    (_,lon2) = meters_to_latlon(y2, x2)
    return (lat2,lon2)


def four_neighbors(lat, lon, zoom=16, margin=44, resolution=1280):

    meter_per_pixel = compute_gsd(resolution, zoom)
    pixels = resolution - 2 * margin
    meters = pixels * meter_per_pixel
    pixel_per_meter = pixels / meters
    
    step = meters

    return [
        (lat, lon),
        walk(lat, lon, step,    0),
        walk(lat, lon, 0   , step),
        walk(lat, lon, step, step),
    ]


with open("api_key.txt") as f: 
    api_key = f.read()


# Build filename and url for image.
def build_url(lat, lon, zoom=18, scale=2, size=640, api_key=api_key):
    filename = "image_lat=%.6f_lon=%.6f_zoom=%d_scale=%d_size=%d.png" % (lat, lon, zoom, scale, size)
    url = "https://maps.googleapis.com/maps/api/staticmap?center="+("%.6f" % lat)+","+("%.6f" % lon)+"&zoom="+str(int(zoom))+"&scale="+str(int(scale))+"&size="+str(int(size))+"x"+str(int(size))+"&maptype=satellite&style=element:labels|visibility:off&key=" + api_key
    return filename, url


# Fetch url and store under fname.
def fetch_url(fname, url):
    if 0 == subprocess.Popen("timeout 5s wget -O tmp.png \"" + url + "\"", shell = True).wait():
        return 0 == subprocess.Popen("mv tmp.png " + fname, shell=True).wait()
    return False


# Example (Compute four images):
lat = 41.857029
lon = -87.687379

latlons = four_neighbors(lat, lon)
# for latlon in latlons:
    # print(latlon)
for (fname, url) in [build_url(lat, lon) for (lat, lon) in latlons]:
    if not os.path.isfile(fname):
        assert fetch_url(fname, url)


# Link four images.
images = [read_image(fname) for (fname, _) in [build_url(lat, lon) for (lat, lon) in latlons]]
images = [cut_logo(im) for im in images]

# Construct new image
superimage = np.ndarray((2*1192, 2*1192, 3), dtype="uint8")
superimage[:1192,:1192,:] = images[0]
superimage[:1192,1192:,:] = images[1]
superimage[1192:,:1192,:] = images[2]
superimage[1192:,1192:,:] = images[3]
write_image(superimage, "superimage.png")




# Find nodes in region of interest.
def nodes_in_ROI(idx, coord, lam=1):
    bb = bounding_box(np.array([coord]), padding=lam)
    nodes = list(idx.intersection((bb[0][0], bb[0][1], bb[1][0], bb[1][1])))
    return nodes


# Compute coverage of curve by a network.
def coverage_curve_by_network(G, ps, lam=1, measure=frechet):

    G = G.copy()
    if G.graph["simplified"]:
        G = vectorize_graph(G) # Vectorize graph.
        G = deduplicate_vectorized_graph(G)
    
    if type(G) != nx.Graph:
        G = nx.Graph(G)

    assert len(ps) >= 2


    edge_idx = graphedges_to_rtree(G)
    node_idx = graphnodes_to_rtree(G)


    edge_set = set() # Collected edges to consider.

    # Construct bounding boxes over path.
    for (p, q) in nx.utils.pairwise(ps):
        bb = bounding_box(np.array([p,q]), padding=lam)
        edge_ids = edge_idx.intersection((bb[0][0], bb[0][1], bb[1][0], bb[1][1]), objects=True)
        for i in edge_ids:
            edge_set.add(i.object)

    node_set = set()
    for (a,b) in edge_set:
        node_set.add(a)
        node_set.add(b)

    nodes = [v for v in node_set]
    G = G.subgraph(nodes) # Extract subgraph with nodes.
    idx = graphnodes_to_rtree(G) # Replay idx to lower bounding box.

    start_nodes = nodes_in_ROI(idx, ps[0], lam=lam)
    end_nodes = nodes_in_ROI(idx, ps[-1], lam=lam)

    if len(start_nodes) == 0:
        return False, {}
    if len(end_nodes) == 0:
        return False, {}
    # for all combinations
    node_dict = extract_nodes_dict(G)
    for (a,b) in itertools.product(start_nodes, end_nodes):
        # for path in nx.shortest_simple_paths(nx.Graph(G), a, b):
        for path in nx.shortest_simple_paths(G, a, b):
            # edge to nodes
            # nodes = [a for (a,b) in path]
            # nodes.append(path[-1][1])
            # nodes to coordinates
            qs = np.array([node_dict[n] for n in path])
            plot_graph_and_curves(nx.MultiDiGraph(G), ps, qs)
        # is_covered, data = curve_by_curveset_coverage(ps, paths, lam=lam, measure=measure)
        # if is_covered:
        #     breakpoint()


# Example (Check arbitrary shortest path (with some noise) is covered by network):
G = extract_graph("chicago")
ps = gen_random_shortest_path(G)
# Add some noise to path.
lam = 0.0010
noise = np.random.random((len(ps),2)) * lam - np.ones((len(ps),2)) * 0.5 * lam
ps = ps + noise
result = coverage_curve_by_network(G, ps, lam=lam)
