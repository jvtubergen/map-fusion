from spatial_reference_systems import *
from data_handling import *
from data.gmaps import fine_tune_image_coordinates, download_and_construct_image
from caching import *


def get_satellite_image(place, gsd_goal=0.5, deviation=0.25):
    """Generate satellite image on the place of interest and write to disk."""

    region = roi[place]
    upper_left = array((region['north'], region['west']))
    lower_right = array((region['south'], region['east']))

    # Derive zoom.
    lat_reference = (0.5 * (upper_left + lower_right))[0]  # Reference latitude.
    zoom = derive_zoom(lat_reference, gsd_goal, deviation=deviation)

    upper_left, lower_right = fine_tune_image_coordinates(upper_left, lower_right, zoom)
    image, metadata = download_and_construct_image(upper_left, lower_right, zoom)

    write_png(get_cache_file(f"sat/{place}.png"), image, metadata=metadata)


def extract_image_statistics():
    """Obtain statistics on the satellite image per city."""
    cities = ['athens', 'berlin', 'chicago']
    stats = {}
    
    for city in cities:
        try:
            # Default parameters (matching get_satellite_image)
            gsd_goal = 0.5
            deviation = 0.25
            
            region = roi[city]
            ul = array((region['north'], region['west'])) # ul: upper-left
            lr = array((region['south'], region['east'])) # lr: lower-right

            # Derive zoom and gsd.
            lat_reference = (0.5 * (ul + lr))[0]  # Reference latitude.
            zoom = derive_zoom(lat_reference, gsd_goal, deviation=deviation)
            gsd = compute_gsd(lat_reference, zoom)
            
            # Get fine-tuned coordinates (padding and stride multiple).
            ul_tuned, lr_tuned = fine_tune_image_coordinates(ul, lr, zoom)
            
            # Compute surface area (use UTM).
            ul_utm = latlon_to_utm(ul_tuned)
            lr_utm = latlon_to_utm(lr_tuned)
            
            width = abs(lr_utm[1] - ul_utm[1]) 
            height = abs(ul_utm[0] - lr_utm[0])  

            surface_area = width * height
            
            # Compute image dimensions (use web mercator pixel coordinates).
            ul_pixels = latlon_to_pixelcoord(ul_tuned[0], ul_tuned[1], zoom)
            lr_pixels = latlon_to_pixelcoord(lr_tuned[0], lr_tuned[1], zoom)
            
            image_width = abs(lr_pixels[1] - ul_pixels[1]) 
            image_height = abs(lr_pixels[0] - ul_pixels[0]) 
            
            stats[city] = {
                'zoom_level': zoom,
                'gsd': gsd,
                'surface_area_m2': surface_area,
                'surface_area_km2': surface_area / 1000000,
                'surface_width': width,
                'surface_height': height,
                'image_width_pixels': image_width,
                'image_height_pixels': image_height,
                'total_pixels': image_width * image_height,
                'bounds': {
                    'north': ul_tuned[0],
                    'west': ul_tuned[1], 
                    'south': lr_tuned[0],
                    'east': lr_tuned[1]
                }
            }
            
        except Exception as e:
            print(f"Error processing {city}: {e}")
            stats[city] = None
    
    return stats
