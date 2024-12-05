import utm

roi = {
    "athens" : {'west': 23.77222618197142, 'south': 37.99243374688059, 'east': 23.840460380116067, 'north': 38.037610439228885},
    "berlin" : {'west': 13.378073999957254, 'south': 52.49230840079104, 'east': 13.456155699977558, 'north': 52.54682070045891},
    "chicago": {'west': -87.68616669406222, 'south': 41.86235129291769, 'east': -87.64093705730774, 'north': 41.88318300100296},
}

roi_full = {
    "athens" : {'west': 23.565760443554986, 'south': 37.83047726403514, 'east': 24.01668661561085, 'north': 38.20114927381213},
    "berlin" : {'west': 13.378073999957254, 'south': 52.49230840079104, 'east': 13.456155699977558, 'north': 52.54682070045891},
    "chicago": {'west': -87.68616669406222, 'south': 41.86235129291769, 'east': -87.64093705730774, 'north': 41.88318300100296},
}

# Latlons for obtaining zone-numbers and zone-letters.
centrums = {
    "athens" : (37.97, 23.72),
    "berlin" : (52.51, 13.40),
    "chicago": (41.88, -87.68),
}

# Zone numbers necessary when converting coordinates into latlons.
zone_numbers = {
    "athens" : utm.latlon_to_zone_number(centrums["athens" ][0], centrums["athens" ][1]),
    "berlin" : utm.latlon_to_zone_number(centrums["berlin" ][0], centrums["berlin" ][1]),
    "chicago": utm.latlon_to_zone_number(centrums["chicago"][0], centrums["chicago"][1]),
}

# Zone letters necessary when converting coordinates into latlons.
zone_letters = {
    "athens" : utm.latitude_to_zone_letter(centrums["athens" ][0] ),
    "berlin" : utm.latitude_to_zone_letter(centrums["berlin" ][0] ),
    "chicago": utm.latitude_to_zone_letter(centrums["chicago"][0]),
}

# Convert a local coordinate into a latlon.
def coord_to_latlon_by_place(coordinate, place):
    y, x = coordinate
    zone_number = zone_numbers[place]
    zone_letter = zone_letters[place]
    
    _, _, zone_number, zone_letter = utm.conversion.from_latlon(*centrums[place])
    latlon = utm.conversion.to_latlon(x, y, zone_number, zone_letter=zone_letter)
    return latlon

# Convert latlon into a local coordinate.
def latlon_to_coord(latlon):
    lat, lon = latlon
    x, y, _, _ = utm.conversion.from_latlon(lat, lon)
    return (y, x)

# Convert utm to latlon by zone and number information.
def coord_to_latlon_by_utm_info(coordinate, number=None, letter=None):
    y, x = coordinate
    latlon = utm.conversion.to_latlon(x, y, number, zone_letter=letter)
    return latlon

# Obtain UTM information from a graph by looking at a random latlon coordinate of a node in that graph.
def get_utm_info_from_graph(G):
    randomnid = list(G.nodes())[0]
    lat = G.nodes(data=True)[randomnid]['y']
    lon = G.nodes(data=True)[randomnid]['x']
    _, _, zone_number, zone_letter = utm.conversion.from_latlon(lat, lon)
    info = {"number": zone_number, "letter": zone_letter}
    return info

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
# place = places[0]
# lat, lon = centrums[place]
# latlon = lat, lon
# x, y = latlon_to_coord(latlon)
# latlon2 = coord_to_latlon_by_place((x, y), place)
# lat2, lon2 = latlon2 
# print(latlon, latlon2)
# assert abs(lat - lat2) < 0.00001
# assert abs(lon - lon2) < 0.00001

