from external import *
from storage import *
from spatial_reference_systems.utm import utm_to_latlon
from graph import *

def read_gps_graph(place):
    return read_graph(gps_locations(place)["graph_file"])

def obtain_gps_graph(place):
    """End-to-end logic to infer GPS graph of a place.
    
    This function reads in the inferred GPS graph of Roadster and writes it as a graph file to disk.
    """
    paths = gps_locations(place)
    G = read_graph_csv(paths["inferred_folder"])
    G = sanitize_graph(G)
    sanity_check_graph(G)
    write_graph(paths["graph_file"], G)


# Extracting raw GPS traces from text.
# By providing the place we know what UTM variables (for coordinate transformation) to use.
def extract_trips(place, folder=None):
    if folder == None:
        folder = gps_locations(place)["traces_folder"] + "/"
    files = os.listdir(folder)
    trips = []
    for file in files:
        with open(folder + file) as f: 
            trip = []
            for step in f.read().strip().split('\n'):
                [x, y, t] = [float(value) for value in step.split(' ')]
                trip.append([x, y, t])
        trips.append(trip)
    return trips


def derive_roi(place):
    """Derive the region of interest (ROI) at a place by computing the bounding box on the GPS traces."""
    trips = extract_trips(place)
    
    # Flatten all trips into a single numpy array of coordinates.
    all_coords = []
    for trip in trips:
        for point in trip:
            all_coords.append([point[1], point[0]])  # [y, x] from [x, y, t]
    
    coords_array = np.array(all_coords)
    
    # Compute bounding box on coordinates.
    west  = float(np.min(coords_array[:, 1]))  # min longitude
    east  = float(np.max(coords_array[:, 1]))  # max longitude
    south = float(np.min(coords_array[:, 0]))  # min latitude
    north = float(np.max(coords_array[:, 0]))  # max latitude

    # Convert UTM to WSG.
    north, west = utm_to_latlon((north, west), place)
    south, east = utm_to_latlon((south, east), place)
    
    return {
        'west' : float(west),
        'south': float(south),
        'east' : float(east),
        'north': float(north),
    }


def derive_rois():
    """Derive ROIs from all cities (athens, berlin, chicago)."""
    cities = ['athens', 'berlin', 'chicago']
    rois = {}
    
    for city in cities:
        rois[city] = derive_roi(city)
    
    return rois


def extract_gps_statistics():
    """Extract GPS statistics for each city dataset"""
    cities = ['athens', 'berlin', 'chicago']
    stats = {}
    
    for city in cities:
        try:
            folder = gps_locations(city)["traces_folder"] + "/"
            files = os.listdir(folder)
            
            total_trips = len(files)
            total_samples = 0
            total_distance = 0
            total_time_intervals = []
            
            for file in files:
                with open(folder + file) as f:
                    trip_data = []
                    for step in f.read().strip().split('\n'):
                        [x, y, t] = [float(value) for value in step.split(' ')]
                        trip_data.append([x, y, t])
                    
                    total_samples += len(trip_data)
                    
                    # Calculate trip distance using UTM coordinates (Euclidean distance)
                    trip_distance = 0
                    for i in range(1, len(trip_data)):
                        x1, y1, t1 = trip_data[i-1]
                        x2, y2, t2 = trip_data[i]
                        trip_distance += math.sqrt((x2-x1)**2 + (y2-y1)**2)
                        total_time_intervals.append(t2 - t1)
                    
                    total_distance += trip_distance
            
            avg_samples_per_trip = total_samples / total_trips if total_trips > 0 else 0
            avg_distance_per_trip = total_distance / total_trips if total_trips > 0 else 0
            avg_sample_interval = sum(total_time_intervals) / len(total_time_intervals) if total_time_intervals else 0
            
            stats[city] = {
                'num_trips': total_trips,
                'avg_samples_per_trip': avg_samples_per_trip,
                'avg_distance_per_trip': avg_distance_per_trip,
                'total_samples': total_samples,
                'total_distance': total_distance,
                'avg_sample_interval': avg_sample_interval
            }
            
        except Exception as e:
            print(f"Error processing {city}: {e}")
            stats[city] = None
    
    return stats
