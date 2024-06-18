# Google Maps retrieval functionality
from dependencies import *

earth_radius = 6378137 # meters
earth_circumference = 2*pi*earth_radius

# Gudermann function. Real argument abs(rho) < 0.5*pi 
# https://en.wikipedia.org/wiki/Gudermannian_function
def gd(tau):
    return 2 * atan(exp(tau)) - 0.5 * pi
def gd_inv(rho):
    return log(sec(rho) + tan(rho))


# Conversion between radians and degrees (for readability).
def rad_to_deg(phi):
    return phi * 180 / pi
def deg_to_rad(rho):
    return rho * pi / 180


# Maximal latitude (of web mercator projection).
# https://en.wikipedia.org/wiki/Mercator_projection
max_lat = rad_to_deg(gd(pi)) # = atan(sinh(pi)) / pi * 180


# Read image file as numpy array.
def read_image(filename):
    image = pil.Image.open(filename)
    image = image.convert("RGB")
    image = np.array(image)
    return image


# Write numpy array with size (width, height, RGB) as an image file.
def write_image(image, filename):
    pil.Image.fromarray(image).save(filename) 


# Cut out pixels of image at borders.
def cut_logo(image, scale, margin):
    off = scale * margin
    return image[off:-off,off:-off,:]


# Build filename and url for image.
def build_url(lat, lon, zoom, resolution, scale, margin, api_key):
    filename = "image_lat=%.6f_lon=%.6f_zoom=%d_scale=%d_size=%d.png" % (lat, lon, zoom, scale, resolution)
    url = "https://maps.googleapis.com/maps/api/staticmap?center="+("%.6f" % lat)+","+("%.6f" % lon)+"&zoom="+str(int(zoom))+"&scale="+str(int(scale))+"&size="+str(int(resolution))+"x"+str(int(resolution))+"&maptype=satellite&style=element:labels|visibility:off&key=" + api_key
    return filename, url


# Fetch url and store under fname.
def fetch_url(fname, url):
    if 0 == subprocess.Popen("timeout 5s wget -O tmp.png \"" + url + "\"", shell = True).wait():
        return 0 == subprocess.Popen("mv tmp.png " + fname, shell=True).wait()
    return False


# Uniform web mercator projection
# y,x in [0,1], starting in upper-left corner (thus lat 85 and lon -180)
# lat in [-85, 85], lon in [-180, 180]
def latlon_to_webmercator_uniform(lat, lon):
    x = 0.5 + deg_to_rad(lon) / (2*pi)
    y = 0.5 - gd_inv(deg_to_rad(lat)) / (2*pi)
    return y, x
def webmercator_uniform_to_latlon(y, x):
    lon = rad_to_deg(2*pi * (x - 0.5))
    lat = rad_to_deg(gd(2*pi * (0.5 - y)))
    return lat, lon


# Conversion between latlon and world/tile/pixelcoordinates.
# Tile and pixel coordinates depends on zoom level.
# In general, the uniform webmercator y,x values are scaled.
def latlon_to_pixelcoord(lat, lon, zoom):
    y, x = latlon_to_webmercator_uniform(lat, lon)
    pixelcount = int(256 * pow(2, zoom))
    return int(y * pixelcount), int(x * pixelcount)

def latlon_to_tilecoord(lat, lon, zoom):
    y, x = latlon_to_webmercator_uniform(lat, lon)
    tilecount = int(pow(2, zoom))
    return int(y * tilecount), int(x * tilecount)

def latlon_to_worldcoord(lat, lon):
    y, x = latlon_to_webmercator_uniform(lat, lon)
    return y * 256, x * 256

def pixelcoord_to_latlon(y, x, zoom):
    pixelcount = int(256 * pow(2, zoom))
    return webmercator_uniform_to_latlon(y / pixelcount, x / pixelcount)


# GSD (Ground Sampling Distance): spatial resolution (in meters) of the image.
def compute_gsd(lat, zoom, resolution, scale):
    k = sec(deg_to_rad(lat)) # Scale factor by mercator projection.
    w = earth_circumference  # Total image distance on 256x256 world image
    return w / (256 * pow(2, zoom) * k * scale)


# # Example (Retrieve four neighboring images, retrieve, and glue together):
# lat = 41.857029
# lon = -87.687379
# zoom = 16 # For given latitude and scale results in gsd of ~ 0.88
# scale = 2
# resolution = 640
# margin = 22

# def image_neighbors(lat, lon, zoom, resolution, scale, margin):
#     fetchto = [(lat, lon)]
#     # Obtain base pixels.
#     y, x = latlon_to_pixelcoord(lat, lon, zoom)
#     # Add pixel offset.
#     step = resolution - 2 * margin
#     fetchto.append(pixelcoord_to_latlon(y       , x + step, zoom)) # Moving right
#     fetchto.append(pixelcoord_to_latlon(y + step, x       , zoom)) # Moving down
#     fetchto.append(pixelcoord_to_latlon(y + step, x + step, zoom)) # Moving right down
#     return fetchto

# # Read API key stored locally.
# with open("api_key.txt") as f: 
#     api_key = f.read()

# # Retrieve latlons.
# print("gsd: ", compute_gsd(lat, zoom, resolution, scale))
# latlons = four_neighbors(lat, lon, zoom, resolution, scale, margin)
# for latlon in latlons:
#     print(latlon)

# # Construct and fetch urls.
# urls = [build_url(lat, lon, zoom, resolution, scale, margin, api_key) for (lat, lon) in latlons]
# for (fname, url) in urls:
#     if not os.path.isfile(fname):
#         assert fetch_url(fname, url)

# # Glue images together.
# size = scale * (resolution - 2 * margin)

# images = [read_image(fname) for (fname, _) in urls]
# images = [cut_logo(image, scale, margin) for image in images]

# superimage = np.ndarray((2*size, 2*size, 3), dtype="uint8")
# superimage[:size,:size,:] = images[0] # Upper left
# superimage[:size,size:,:] = images[1] # Upper right
# superimage[size:,:size,:] = images[2] # Lower left
# superimage[size:,size:,:] = images[3]
# write_image(superimage, "superimage.png")
