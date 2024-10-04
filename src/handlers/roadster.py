import os
import utm

# Latlons for obtaining zone-numbers and zone-letters.
centrums = {
    "athens" : (37.97, 23.72),
    "berlin" : (52.51, 13.40),
    "chicago": (41.88, -87.68),
}

zone_numbers = {
    "athens" : utm.latlon_to_zone_number(centrums["athens" ][0], centrums["athens" ][1]),
    "berlin" : utm.latlon_to_zone_number(centrums["berlin" ][0], centrums["berlin" ][1]),
    "chicago": utm.latlon_to_zone_number(centrums["chicago"][0], centrums["chicago"][1]),
}

zone_letters = {
    "athens" : utm.latitude_to_zone_letter(centrums["athens" ][0] ),
    "berlin" : utm.latitude_to_zone_letter(centrums["berlin" ][0] ),
    "chicago": utm.latitude_to_zone_letter(centrums["chicago"][0]),
}

# Extracting raw GPS traces from text.
# By providing the place we know what UTM variables (for coordinate transformation) to use.
def extract_trips(place, folder=None):
    if folder == None:
        folder = f"data/tracks/{place}/"
    files = os.listdir(folder)
    zone_number = zone_numbers[place]
    zone_letter = zone_letters[place]
    trips = []
    for file in files:
        with open(folder + file) as f: 
            trip = []
            for step in f.read().strip().split('\n'):
                [x, y, t] = [float(value) for value in step.split(' ')]
                y, x = utm.conversion.to_latlon(x, y, zone_number, zone_letter=zone_letter)
                trip.append([x, y, t])
        trips.append(trip)
    return trips
