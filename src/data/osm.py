from external import *
from caching import *
from data_handling import * 

from data.gps import derive_roi

def obtain_osm_graph(place):
    """Download OpenStreetMaps graph on the ROI of the place and store as graph."""

    roi = derive_roi(place)

    west  = roi["west"]
    east  = roi["east"]
    south = roi["south"]
    north = roi["north"]

    G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive", retain_all=False, simplify=False)
    G = nx.Graph(G.to_undirected())
    G.graph["simplified"] = False

    write_graph(osm_locations(place)["graph_file"], G)
