import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL as pil

import os
import subprocess

# Standard library
import math
import itertools
import random

# Geometry utilities
from shapely.geometry import LineString, Point
import utm
import rtree
from partial_curve_matching import Vector, partial_curve_graph, make_graph

# IO dependencies
from pathlib import Path
from fileinput import input
import json
import pickle

# Utils
from operator import itemgetter
import traceback

# Handlers
import handlers.sat2graph as sat2graph
import handlers.roadster as roadster
import handlers.gmaps.lib as gmaps


# Math functions
cos  = math.cos
sec  = lambda phi: 1/cos(phi) # Secant
sin  = math.sin
tan  = math.tan
asin = math.asin
atan = math.atan
acos = math.acos
sinh = math.sinh
exp  = math.exp
log  = math.log
pow  = math.pow
pi   = math.pi
sqrt = math.sqrt

ceil = math.ceil
floor = math.floor

norm = np.linalg.norm
array = np.array
dataframe = gpd.GeoDataFrame
