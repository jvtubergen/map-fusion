import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import numpy as np
from shapely.geometry import LineString, Point
import utm

# IO dependencies
from pathlib import Path
from fileinput import input

# Utils
from operator import itemgetter
import random

# Library code
from coverage import *
from network import *



# Example (vectorize and simplify again):
G = extract_graph("chicago_kevin")
G2 = vectorize_graph(G)
G3 = ox.simplify_graph(G2)
# assert len(G.nodes()) == len(G3.nodes())












# idx = graphnodes_to_rtree(G)
# ps = gen_random_shortest_path(G)
# bb = bounding_box(ps)
# H = subgraph_by_bounding_box(bb)
# # 
