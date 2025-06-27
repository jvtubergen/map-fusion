from external import *
from caching import *
from data_handling import *


# Read API key stored locally.
def read_api_key():
    with open(cache_folder + "/api_key.txt") as f: 
        api_key = f.read()
    return api_key


# Write API key and store it locally.
def write_api_key(api_key):
    with open(cache_folder + "/api_key.txt", "w") as f:
        f.write(api_key)


def cached_sat_image(fname):
    file_path = cache_folder + "/retrieved_satellite_images" + fname


# Cut out pixels of image at borders.
def cut_logo(image, scale, margin):
    off = scale * margin
    return image[off:-off,off:-off,:]


# Build filename and url for image.
def build_filename_and_url(lat, lon, zoom, resolution, scale, api_key):
    filename = "image_lat=%.6f_lon=%.6f_zoom=%d_scale=%d_size=%d.png" % (lat, lon, zoom, scale, resolution)
    url = "https://maps.googleapis.com/maps/api/staticmap?center="+("%.6f" % lat)+","+("%.6f" % lon)+"&zoom="+str(int(zoom))+"&scale="+str(int(scale))+"&size="+str(int(resolution))+"x"+str(int(resolution))+"&maptype=satellite&style=element:labels|visibility:off&key=" + api_key
    return filename, url


def fetch_image(lat, lon, zoom, resolution, scale, api_key):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    else:
        print("Success!")
    
    png = response.encoding.content

    return png
    

# Fetch url and store under fname.
def fetch_url(fname, url, *params): # params is a list of [("query", "value")] pairs
    try:
        response = requests.get(url, params, timeout=5)
        response.raise_for_status()

        with open(fname, 'wb') as file:
            file.write(response.content)

        return True

    except Exception as e:
        print(e)

        return False


# Adapt coordinates into a square.
# * Assumes uniform horizonal and vertical distance (thus don't apply to latlon coordinates).
# * Assume to be inclusive (thus increasing area size rather than lowering it).
def squarify_utm_coordinates(p1, p2):
    y1, x1 = p1
    y2, x2 = p2
    w = x2 - x1
    h = y2 - y1
    if h > w:
        diff = h - w
        left = floor(0.5 * diff)
        right= ceil(0.5 * diff)
        x1 = x1 - left
        x2 = x2 + right
    if w > h:
        diff = w - h
        above= floor(0.5 * diff)
        below= ceil(0.5 * diff)
        y1 = y1 - above
        y2 = y2 + below
    return [y1, x1], [y2, x2]


# Retrieve images, stitch them together.
# * Adjust to have tile consistency, this should reduce the number of requests we are making.
# * Prefer higher scale, lowers image retrieval count as well.
def construct_image(north=None, west=None, east=None, south=None, zoom=None, api_key=None, full_tiles=False, square=False, verbose=False):
    
    assert north   != None
    assert west    != None
    assert east    != None
    assert south   != None
    assert zoom    != None
    assert api_key != None

    scale = 2 # Force scale to 2 to reduce API calls.
    zoom_for_requests = zoom - 1 # Reduce the zoom level for requests, because the scale is increased.

    # Deconstruct latlons.
    lat1, lon1 = north, west # Upper-left  corner (thus higher latitude and lower longitude).
    lat2, lon2 = south, east # Lower-right corner (thus lower latitude and higher longitude).

    # Obtain pixel range in google maps at given zoom for requests.
    y1, x1 = latlon_to_pixelcoord(lat1, lon1, zoom_for_requests)
    y2, x2 = latlon_to_pixelcoord(lat2, lon2, zoom_for_requests)

    # Optionally make the requested image square.
    if square:
        p1, p2 = [y1, x1], [y2, x2]
        p1, p2 = squarify_utm_coordinates(p1, p2)
        [y1, x1], [y2, x2] = p1, p2

    max_resolution = 640 # Google Maps images up to 640x640.
    margin = 22 # Necessary to cut out logo.

    # Construct tiles (to fetch).
    step = max_resolution - 2*margin # Tile step size.
    t1 = (y1 // step, x1 // step) # Tile in which upper-left  pixel lives.
    t2 = (y2 // step, x2 // step) # Tile in which lower-right pixel lives.
    tiles  = [(j, i) for j in range(t1[0], t2[0] + 1) for i in range(t1[1], t2[1] + 1)]
    width  = len(range(t1[1],t2[1] + 1)) # Tile width.
    height = len(range(t1[0],t2[0] + 1)) # Tile height.

    # Convert tiles into pixel coordinates (at their center).
    tile_to_pixel = lambda t: int((t + 0.5) * step)
    pixelcoords  = [(tile_to_pixel(j), tile_to_pixel(i)) for (j,i) in tiles]
    latloncoords = [pixelcoord_to_latlon(y, x, zoom_for_requests) for y,x in pixelcoords]

    # Make sure the image cache folder exists.
    image_cache_folder = cache_folder + "/retrieved_satellite_images"
    if not os.path.exists(image_cache_folder):
        os.makedirs(image_cache_folder, exist_ok=True)

    # Construct and fetch urls.
    if verbose:
        print("Retrieving images.")
    count = 0
    fnames_and_urls = [build_filename_and_url(lat, lon, zoom_for_requests, max_resolution, scale, api_key) for (lat, lon) in latloncoords]
    for (fname, url) in fnames_and_urls:

        count += 1
        if verbose:
            print(f"Retrieving satellite image: {count}/{len(fnames_and_urls)}.")

        if not os.path.isfile(image_cache_folder + fname):
            if verbose:
                print("Fetching image: " + url)
            assert fetch_url(image_cache_folder + fname, url)
        else:
            if verbose:
                print("Image already exists: " + url)

    # Stitch image tiles together.
    if verbose:
        print("Constructing image.")
    images = [read_image(cache_folder + fname) for (fname, _) in fnames_and_urls]
    images = [cut_logo(image, scale, margin) for image in images]

    size = scale * step # Image size.
    superimage = np.ndarray((height*size, width*size, 3), dtype="uint8")
    m = size
    for y in range(height):
        for x in range(width):
            superimage[y*m:(y+1)*m,x*m:(x+1)*m,:] = images[y*width+x]

    # Cut out part of interest (of latlon).
    if not full_tiles:

        def subtest():
            # Sanity check offset applied to lower-right tile is correct.
            def formula(y2, step):
                return step - (y2 % step) - 1

            step = 88

            y2 = 4 * step 
            off2 = formula(y2, step)
            assert off2 == step - 1

            y2 = 4 * step - 1
            off2 = formula(y2, step)
            assert off2 == 0 

        off1 = scale * np.array((y1 % step, x1 % step)) # Pixel offset in upper-left tile.
        off2 = scale * np.array((step - (y2 % step), step - (x2 % step))) # Pixel offset in lower-right tile.
        superimage = superimage[off1[0]:-off2[0],off1[1]:-off2[1],:] # Note how we cut off in y with first axis (namely rows) and x in second axis (columns).

    pixelcoord = (y1, x1) # upper-left corner pixel coordinate.

    return superimage, pixelcoord