from spatial_reference_systems import *
from data_handling import *
from data.gmaps import fine_tune_image_coordinates, download_and_construct_image
from caching import *



def get_satellite_image(place, gsd_goal=0.5, deviation=0.25):

    region = roi[place]
    upper_left = array((region['north'], region['west']))
    lower_right = array((region['south'], region['east']))

    # Derive zoom.
    lat_reference = (0.5 * (upper_left + lower_right))[0]  # Reference latitude.
    zoom = derive_zoom(lat_reference, gsd_goal, deviation=deviation)

    upper_left, lower_right = fine_tune_image_coordinates(upper_left, lower_right, zoom)
    image, metadata = download_and_construct_image(upper_left, lower_right, zoom)

    write_png(get_cache_file(f"sat/{place}.png"), image, metadata=metadata)



# Print the zoom levels that will be applied for retrieval of satellite images.
# Note: Scaling is applied implicitly, so ignore this value.
def workflow_print_zoom_levels():
    print("zoom levels:")
    for place in ["athens", "berlin", "chicago"]:
        region = roi[place]
        latlon0 = array((region['north'], region['west']))
        latlon1 = array((region['south'], region['east']))
        lat_reference = (0.5 * (latlon0 + latlon1))[0]  # Reference latitude.
        gsd_goal  = 0.5
        deviation = 0.25
        zoom = derive_zoom(lat_reference, gsd_goal, deviation=deviation)
        gsd  = compute_gsd(lat_reference, zoom)
        print(f"{place}: {zoom} (gsd: {gsd})")
