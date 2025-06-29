import utm

places = ["athens", "berlin", "chicago"]

# Latlons for obtaining zone-numbers and zone-letters.
centrums = {
    "athens" : (37.97, 23.72),
    "berlin" : (52.51, 13.40),
    "chicago": (41.88, -87.68),
}

# Regions of interest.
roi = {
    'athens' : {'west': 19.33130554465339,  'south': 37.83053085246328,  'east': 21.0185797594304,   'north': 38.73821983601968}, 
    'berlin' : {'west': 13.377779268155852, 'south': 52.4923742180644,   'east': 13.457342807683187, 'north': 52.54701973676834}, 
    'chicago': {'west': -87.68723676727973, 'south': 41.861979512439795, 'east': -87.63973695049994, 'north': 41.883300448220425}
}

# Zone numbers necessary when converting coordinates into latlons.
zone_numbers = {
    "athens" : utm.latlon_to_zone_number(centrums["athens" ][0], centrums["athens" ][1]),
    "berlin" : utm.latlon_to_zone_number(centrums["berlin" ][0], centrums["berlin" ][1]),
    "chicago": utm.latlon_to_zone_number(centrums["chicago"][0], centrums["chicago"][1]),
}

# Zone letters necessary when converting coordinates into latlons.
zone_letters = {
    "athens" : utm.latitude_to_zone_letter(centrums["athens" ][0]),
    "berlin" : utm.latitude_to_zone_letter(centrums["berlin" ][0]),
    "chicago": utm.latitude_to_zone_letter(centrums["chicago"][0]),
}

# Convert a utm coordinate into a latlon.
def utm_to_latlon(coordinate, place):
    y, x = coordinate
    zone_number = zone_numbers[place]
    zone_letter = zone_letters[place]
    latlon = utm.conversion.to_latlon(x, y, zone_number, zone_letter=zone_letter)
    return latlon

# Convert latlon into a utm coordinate.
def latlon_to_utm(latlon):
    lat, lon = latlon
    x, y, _, _ = utm.conversion.from_latlon(lat, lon)
    return (y, x)

# Update latlon by translating it in meters (uses UTM projection for this).
def translate_latlon_by_meters(lat=None, lon=None, west=None, north=None, east=None, south=None):

    latlon = lat, lon
    x, y, zone_number, zone_letter = utm.conversion.from_latlon(lat, lon)
    # print(x,y)

    if type(west) == type(0):
        x -= west
    if type(east) == type(0):
        x += east

    if type(south) == type(0):
        y -= south
    if type(north) == type(0):
        y += north

    # print(x,y)
    latlon = utm.conversion.to_latlon(x, y, zone_number, zone_letter=zone_letter)
    return latlon

# Sanity check:
# place = "athens"
# lat, lon = centrums[place]
# latlon = lat, lon
# x, y = latlon_to_utm(latlon)
# latlon2 = utm_to_latlon((x, y), place)
# lat2, lon2 = latlon2 
# print(latlon, latlon2)
# assert abs(lat - lat2) < 0.00001
# assert abs(lon - lon2) < 0.00001