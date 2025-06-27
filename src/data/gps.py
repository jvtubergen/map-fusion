import os
from spatial_reference_systems.utm import utm_to_latlon

# Extracting raw GPS traces from text.
# By providing the place we know what UTM variables (for coordinate transformation) to use.
def extract_trips(place, folder=None):
    if folder == None:
        folder = f"data/tracks/{place}/"
    files = os.listdir(folder)
    trips = []
    for file in files:
        with open(folder + file) as f: 
            trip = []
            for step in f.read().strip().split('\n'):
                [x, y, t] = [float(value) for value in step.split(' ')]
                y, x = utm_to_latlon((y, x), place)
                trip.append([x, y, t])
        trips.append(trip)
    return trips

#TODO: List gps information.
# extract_trips("athens")
# number of trips:
# average samples per trip:
# number of samples:
# total distance travelled:
# average sample interval
def read_raw_gps_trajectories(place):
    print("Reading trajectories.")
    trips  = roadster.extract_trips(place)
    paths = [[[x,y] for [x,y,t] in trip] for trip in trips] # Drop t component
    return paths