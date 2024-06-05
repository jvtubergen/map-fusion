import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import numpy as np
import math

# Geometry utilities
from shapely.geometry import LineString, Point
import utm
import rtree

# IO dependencies
from pathlib import Path
from fileinput import input

# Utils
from operator import itemgetter
import random
import traceback