import os

places = ["berlin", "chicago"]
base_variants = ["osm", "sat", "gps"]
fusion_variants = ["A", "B", "C"]
fusion_variants_covered = ["A2", "B2", "C2"]
variants = base_variants + fusion_variants + fusion_variants_covered
metrics = ["apls", "topo"]

def get_fusion_variants(covered_injection_only=False):
    """Return the appropriate fusion variants based on covered_injection_only parameter."""
    return fusion_variants_covered if covered_injection_only else fusion_variants

# This file provides functions to cache your computation results so they can be re-used.

# Cache folder is used for retrieving the API key from and to store image query responses to.
config_folder = os.path.expanduser("~/.config/map_fusion")
cache_folder = os.path.expanduser("~/.cache/map_fusion")
data_folder = "data"


def get_cache_folder(folder_path):
    """Retrieve cache folder path; create it if it does not exist."""
    location = f"{cache_folder}/{folder_path}"
    if not os.path.exists(location):
        os.makedirs(location, exist_ok=True)
    return location

def get_data_folder(folder_path):
    """Retrieve data folder path; create it if it does not exist."""
    location = f"{data_folder}/{folder_path}"
    if not os.path.exists(location):
        os.makedirs(location, exist_ok=True)
    return location

def get_cache_file(file_path):
    """Retrieve cache file path; create parent directory if it does not exist."""
    location = f"{cache_folder}/{file_path}"
    parent_dir = os.path.dirname(location)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    return location

def get_data_file(file_path):
    """Retrieve data file path; create parent directory if it does not exist."""
    location = f"{data_folder}/{file_path}"
    parent_dir = os.path.dirname(location)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    return location

def data_location(place, variant, threshold = None, inverse = False):
    """Unified location logic for different data variants and places."""
    if variant == "osm":
        return {
            "graph_file": get_data_file(f"graphs/osm-{place}.graph")
        }
    elif variant == "gps":
        return {
            "inferred_folder": get_data_folder(f"gps/inferred/{place}"),
            "graph_file": get_data_file(f"graphs/gps-{place}.graph"),
            "raw_folder": get_data_folder(f"gps/raw/{place}"),
            "processed_folder": get_data_folder(f"gps/processed/{place}"),
            "traces_folder": get_data_folder(f"gps/traces/{place}")
        }
    elif variant == "sat":
        base = {
            # Pre-requisites.
            "image": get_data_file(f"sat/{place}.png"),
            "pbmodel": get_cache_file(f"sat/sat2graph/globalv2.pb"),
            # Intermediate data files.
            "partial_gtes": get_cache_file(f"sat/sat2graph/{place}/partial-gtes"),
            "intermediate": get_cache_file(f"sat/sat2graph/{place}/intermediate"),
            # Resulting files.
            "graph_file": get_data_file(f"graphs/sat-{place}.graph"),
            "image-results": get_data_file(f"images/sat/{place}")
        }
        
        # Add phase-specific path functions
        intermediate_dir = base["intermediate"]
        def gte_path(phase): 
            return f"{intermediate_dir}/gte{phase}.pkl"
        def decoded_path(phase): 
            return f"{intermediate_dir}/graph-raw{phase}.pkl"
        def refined_path(phase): 
            return f"{intermediate_dir}/graph-refined{phase}.pkl"
        
        # Add other location functions
        def partial_gte_path(counter):
            return f"{base["partial_gtes"]}/{counter}.pkl"
        def image_result_path(phase):
            return f"{base["image-results"]}/phase {phase}"
        
        base.update({
            "gte_path": gte_path,
            "decoded_path": decoded_path,
            "refined_path": refined_path,
            "partial_gte_path": partial_gte_path,
            "image_result_path": image_result_path
        })
        
        return base
    elif variant in ["A", "B", "C"] or variant in ["A2", "B2", "C2"]:
        assert threshold != None
        if inverse:
            return {
                "graph_file": get_data_file(f"graphs/{variant}-{place}-{threshold}-inverse.graph")
            }
        else:
            return {
                "graph_file": get_data_file(f"graphs/{variant}-{place}-{threshold}.graph")
            }
    else:
        raise ValueError(f"Unknown variant: {variant}")

def experiment_location(place, variant, threshold = None, inverse = False, metric = None, metric_threshold = None, metric_interval = None, plot_name = None, prime_samples = False):

    filename = f"{variant}-{place}"
    if (variant in fusion_variants or variant in fusion_variants_covered) and threshold != None:
        filename = f"{filename}-{threshold}"
    if (variant in fusion_variants or variant in fusion_variants_covered) and inverse:
        filename = f"{filename}-inverse"
    if metric != None:
        filename = f"{filename}-{metric}"
    if metric_threshold != None:
        filename = f"{filename}-{metric_threshold}"
    if metric_interval != None:
        filename = f"{filename}-{metric_interval}"
    if plot_name != None:
        filename = f"{filename}-{plot_name}"

    assert variant in variants
    assert place in places
    if variant in fusion_variants or variant in fusion_variants_covered:
        assert threshold != None # Expect a threshold is given if its a fuse variant.

    metrics_samples_dir = "metrics_samples_prime" if prime_samples else "metrics_samples"
    
    return  {
        "prepared_graph"     : get_data_file(f"experiments/prepared_graph/{filename}.graph"),
        "apls_shortest_paths": get_data_file(f"experiments/apls_shortest_paths/{filename}.pkl"),
        "metrics"            : get_data_file(f"experiments/metrics/{filename}.pkl"),
        "metrics_samples"    : get_data_file(f"experiments/{metrics_samples_dir}/{filename}.pkl"),
        # "plots"              : get_data_file(f"experiments/plots/{filename}.svg"),
        # "distance_samples"   : get_data_file(f"experiments/distance_samples/{filename}.pkl"),
        # "threshold_maps"     : get_data_file(f"experiments/threshold_maps/{filename}.pkl"),
    }


## Legacy functions for backward compatibility
def sat_locations(place):
    return data_location(place, "sat")

def gps_locations(place):
    return data_location(place, "gps")

def osm_locations(place):
    return data_location(place, "osm")