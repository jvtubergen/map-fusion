import os
import math
from spatial_reference_systems.utm import utm_to_latlon

# Extracting raw GPS traces from text.
# By providing the place we know what UTM variables (for coordinate transformation) to use.
def extract_trips(place, folder=None):
    if folder == None:
        folder = f"data/gps/traces/{place}/"
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

def extract_city_statistics():
    """Extract GPS statistics for each city dataset"""
    cities = ['athens', 'berlin', 'chicago']
    stats = {}
    
    for city in cities:
        try:
            folder = f"data/gps/traces/{city}/"
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
