"""
Shared utility functions for experiments module.

This module contains shared helper functions that can be used across different experiment series.
Currently, most utility functions are embedded within their respective modules, but this provides
a central location for shared utilities as the codebase evolves.
"""

from storage import *
from utilities import *
from data import *
from map_similarity import *
from graph import *


# Common imports that all experiment modules need
__all__ = [
    # Re-export commonly used items
    'logger',
    'places', 
    'base_variants',
    'metrics',
    'get_fusion_variants',
    'read_pickle',
    'write_pickle',
    'experiment_location',
    'data_location',
    'read_graph',
    'write_graph',
    'iterate_edges',
    'iterate_nodes',
    'filter_eids_by_attribute',
    'filter_nids_by_attribute',
    'sanity_check_graph',
    'edge_graph_coverage',
    'map_fusion',
    'vectorize_graph',
    'extract_subgraph_by_graph',
    'precompute_shortest_path_data',
    'apls_sampling',
    'topo_sampling',
    'asymmetric_apls_from_samples',
    'asymmetric_topo_from_samples',
    'prime_apls_samples',
    'prime_topo_samples',
    'compute_topo_sample',
    'generate_sample_pairs',
    'sample_pair_distance',
    'read_osm_graph',
    'read_gps_graph',
    'read_sat_graph',
]


def get_experiment_metadata():
    """Return metadata about the experiment setup."""
    return {
        "places": places,
        "base_variants": base_variants,
        "metrics": metrics,
        "fusion_variants": {
            "standard": ["A", "B", "C"],
            "covered_injection": ["A2", "B2", "C2"]
        }
    }


def validate_experiment_parameters(threshold=None, place=None, variant=None, metric=None):
    """Validate common experiment parameters."""
    if place is not None and place not in places:
        raise ValueError(f"Invalid place '{place}'. Valid places: {places}")
    
    if variant is not None and variant not in (base_variants + ["A", "B", "C", "A2", "B2", "C2"]):
        raise ValueError(f"Invalid variant '{variant}'. Valid variants: {base_variants + ['A', 'B', 'C', 'A2', 'B2', 'C2']}")
    
    if metric is not None and metric not in metrics:
        raise ValueError(f"Invalid metric '{metric}'. Valid metrics: {metrics}")
    
    if threshold is not None and (threshold < 1 or threshold > 500):
        raise ValueError(f"Invalid threshold '{threshold}'. Must be between 1 and 500.")
    
    return True


def format_variant_name(variant, inverse=False, covered_injection_only=False):
    """Format variant names for display in tables and plots."""
    if variant in base_variants:
        return variant.upper()
    
    mapping = {
        "A": "I",
        "B": "ID", 
        "C": "IDR",
        "A2": "I*",
        "B2": "ID*",
        "C2": "IDR*"
    }
    
    base_name = mapping.get(variant, variant)
    
    if inverse:
        suffix = "_GPS"
    else:
        suffix = "_SAT"
    
    return f"${base_name}{suffix}$"