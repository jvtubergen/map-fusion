# Remote libraries
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import numpy as np
# Rendering
import matplotlib
import matplotlib.pyplot as plt
import PIL as pil
# Geometry
from shapely.geometry import LineString, Point
import utm
import rtree
from partial_curve_matching import Vector, partial_curve_graph, make_graph # (Shared library)
# IO dependencies
from pathlib import Path
from fileinput import input
import json
import pickle
import os
# Standard library
import math
import itertools
import random
import subprocess
# Utils
from operator import itemgetter
import traceback
from time import time

# Shortcuts
cos  = math.cos
sec  = lambda phi: 1/cos(phi) # Secant
sin  = math.sin
tan  = math.tan
asin = math.asin
atan = math.atan
atan2= math.atan2
acos = math.acos
sinh = math.sinh
exp  = math.exp
log  = math.log
pow  = math.pow
pi   = math.pi
sqrt = math.sqrt
inf  = math.inf

ceil = math.ceil
floor = math.floor

norm = np.linalg.norm
array = np.array
dataframe = gpd.GeoDataFrame


# Configurations.
ox.settings.log_console = False # Whether to log debug actions of OSMnx to stdout.