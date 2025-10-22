# Web Mercator
from math import *

earth_radius   = 6371007 # meters
earth_circumference = 2*pi*earth_radius
sec  = lambda phi: 1/cos(phi) # Secant

# Gudermannian function. Real argument abs(rho) < 0.5*pi 
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

# Number imprecision by conversion math can cause minor deviations.
# Solve this by iteratively tweaking till inverse is identical.
# Thus adjust as necessary to prevent overlap with adjacent pixels.
def pixelcoord_to_latlon(y, x, zoom):

    def _pixelcoord_to_latlon(y, x, zoom):
        pixelcount = int(256 * pow(2, zoom))
        return webmercator_uniform_to_latlon(y / pixelcount, x / pixelcount)

    # Check above and below, and adjust as necessary to prevent overlap with adjacent pixels.
    lat, lon = _pixelcoord_to_latlon(y, x, zoom)
    y2 , x2  = latlon_to_pixelcoord(lat, lon, zoom)

    while y2 != y:
        # print("Fixing latitude")
        # print("Goal:", y)
        # print("Curr:", y2)
        assert(abs(y2 - y) == 1)
        if y2 < y:
            lat -= 0.000001
        else:
            lat += 0.000001
        y2 , x2  = latlon_to_pixelcoord(lat, lon, zoom)

    while x2 != x:
        # print("Fixing longitude")
        # print("Goal:", x)
        # print("Curr:", x2)
        assert(abs(x2 - x) == 1)
        if x2 < x:
            lon += 0.000001
        else:
            lon -= 0.000001
        y2 , x2  = latlon_to_pixelcoord(lat, lon, zoom)
    
    return lat, lon


# GSD (Ground Sampling Distance): spatial resolution (in meters) of the image.
# Note: Ignore scale because it is confusing to compute with; its implicit in image retrieval queries.
def compute_gsd(lat, zoom):
    k = sec(deg_to_rad(lat)) # Scale factor by mercator projection.
    w = earth_circumference  # Total image distance on 256x256 world image
    return w / (256 * pow(2, zoom) * k)


# Compute zoom in such that related GSD is smaller or equal to the goal GSD.
# Note: Ignore scale because it is confusing to compute with; its implicit in image retrieval queries.
def derive_zoom(lat, goal_gsd, deviation=0.0):

    k = sec(deg_to_rad(lat)) # Scale factor by mercator projection.
    w = earth_circumference  # Total image distance on 256x256 world image
    # Derive equation to compute zoom from gsd:
    #       gsd                              = w / (256 * pow(2, zoom) * k)
    #       w / gsd                          = 256 * pow(2, zoom) * k
    #       w / (256 * gsd)                  = pow(2, zoom) * k
    #       w / (256 * gsd * k)              = pow(2, zoom)
    #       log(w / (256 * goal_gsd * k), 2) = zoom
    zoom = log(w / (256 * goal_gsd * k), 2)

    zoom1 = floor(zoom)
    zoom2 = ceil(zoom)
    zoom0 = zoom1 - 1

    gsd0 = compute_gsd(lat, zoom0) 
    gsd1 = compute_gsd(lat, zoom1) 
    gsd2 = compute_gsd(lat, zoom2) 

    if gsd0 <= goal_gsd + deviation:
        return zoom0
    elif gsd1 <= goal_gsd + deviation:
        return zoom1
    else:
        assert gsd2 <= goal_gsd + deviation
        return zoom2