from data_handling import *
from utilities import *
from data import *
from map_similarity import *
from graph import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .data_preparation import (
    obtain_fusion_maps,
    obtain_prepared_metric_maps,
    remove_deleted
)
from .visualization import (
    plot_comprehensive_covariance_matrices
)


def experiment_unimodal_fusion_analysis(threshold=30):
    """
    Perform map fusion with unimodal maps as base and ground truth as patch, and vice versa.
    Count injected edges and report total edge count and edge length for each fusion scenario.
    Returns dictionary with data results and generates typst table string.
    """
    logger("Starting unimodal fusion analysis experiment.")
    
    results = {}
    
    for place in places:
        logger(f"Analyzing {place}...")
        results[place] = {}
        
        # Load base maps
        osm = read_osm_graph(place)
        gps = read_gps_graph(place)
        sat = read_sat_graph(place)
        
        print(f"\n=== {place.upper()} ===")
        
        # Scenario 1: GPS as base, OSM as patch
        logger("GPS as base, OSM as patch")
        osm_vs_gps = edge_graph_coverage(osm, gps, max_threshold=threshold)
        sanity_check_graph(gps)
        sanity_check_graph(osm_vs_gps)
        graphs_gps_base = map_fusion(C=gps, A=osm_vs_gps, prune_threshold=threshold, 
                                   remove_duplicates=True, reconnect_after=True, 
                                   covered_injection_only=True)
        fusion_gps_base = graphs_gps_base["c"]
        
        # Count metrics for GPS base scenario
        injected_count_gps = len(filter_eids_by_attribute(fusion_gps_base, filter_attributes={"render": "injected"}))
        total_edges_gps = len(fusion_gps_base.edges)
        total_length_gps = sum([attrs["length"] for _, attrs in iterate_edges(fusion_gps_base)]) / 1000
        
        results[place]["gps_base_osm_patch"] = {
            "injected_edges": injected_count_gps,
            "total_edges": total_edges_gps,
            "total_length_km": total_length_gps
        }
        
        print(f"GPS base + OSM patch:")
        print(f"  Injected edges: {injected_count_gps}")
        print(f"  Total edges: {total_edges_gps}")
        print(f"  Total length: {total_length_gps:.2f} km")
        
        # Scenario 2: OSM as base, GPS as patch  
        logger("OSM as base, GPS as patch")
        gps_vs_osm = edge_graph_coverage(gps, osm, max_threshold=threshold)
        sanity_check_graph(osm)
        sanity_check_graph(gps_vs_osm)
        graphs_osm_base = map_fusion(C=osm, A=gps_vs_osm, prune_threshold=threshold,
                                   remove_duplicates=True, reconnect_after=True,
                                   covered_injection_only=True)
        fusion_osm_base = graphs_osm_base["c"]
        
        # Count metrics for OSM base scenario
        injected_count_osm = len(filter_eids_by_attribute(fusion_osm_base, filter_attributes={"render": "injected"}))
        total_edges_osm = len(fusion_osm_base.edges)
        total_length_osm = sum([attrs["length"] for _, attrs in iterate_edges(fusion_osm_base)]) / 1000
        
        results[place]["osm_base_gps_patch"] = {
            "injected_edges": injected_count_osm,
            "total_edges": total_edges_osm,
            "total_length_km": total_length_osm
        }
        
        print(f"OSM base + GPS patch:")
        print(f"  Injected edges: {injected_count_osm}")
        print(f"  Total edges: {total_edges_osm}")
        print(f"  Total length: {total_length_osm:.2f} km")
        
        # Scenario 3: SAT as base, OSM as patch
        logger("SAT as base, OSM as patch")
        osm_vs_sat = edge_graph_coverage(osm, sat, max_threshold=threshold)
        sanity_check_graph(sat)
        sanity_check_graph(osm_vs_sat)
        graphs_sat_base = map_fusion(C=sat, A=osm_vs_sat, prune_threshold=threshold,
                                   remove_duplicates=True, reconnect_after=True,
                                   covered_injection_only=True)
        fusion_sat_base = graphs_sat_base["c"]
        
        # Count metrics for SAT base scenario
        injected_count_sat = len(filter_eids_by_attribute(fusion_sat_base, filter_attributes={"render": "injected"}))
        total_edges_sat = len(fusion_sat_base.edges)
        total_length_sat = sum([attrs["length"] for _, attrs in iterate_edges(fusion_sat_base)]) / 1000
        
        results[place]["sat_base_osm_patch"] = {
            "injected_edges": injected_count_sat,
            "total_edges": total_edges_sat,
            "total_length_km": total_length_sat
        }
        
        print(f"SAT base + OSM patch:")
        print(f"  Injected edges: {injected_count_sat}")
        print(f"  Total edges: {total_edges_sat}")
        print(f"  Total length: {total_length_sat:.2f} km")
        
        # Scenario 4: OSM as base, SAT as patch
        logger("OSM as base, SAT as patch")
        sat_vs_osm = edge_graph_coverage(sat, osm, max_threshold=threshold)
        sanity_check_graph(osm)
        sanity_check_graph(sat_vs_osm)
        graphs_osm_sat_base = map_fusion(C=osm, A=sat_vs_osm, prune_threshold=threshold,
                                       remove_duplicates=True, reconnect_after=True,
                                       covered_injection_only=True)
        fusion_osm_sat_base = graphs_osm_sat_base["c"]
        
        # Count metrics for OSM base + SAT patch scenario
        injected_count_osm_sat = len(filter_eids_by_attribute(fusion_osm_sat_base, filter_attributes={"render": "injected"}))
        total_edges_osm_sat = len(fusion_osm_sat_base.edges)
        total_length_osm_sat = sum([attrs["length"] for _, attrs in iterate_edges(fusion_osm_sat_base)]) / 1000
        
        results[place]["osm_base_sat_patch"] = {
            "injected_edges": injected_count_osm_sat,
            "total_edges": total_edges_osm_sat,
            "total_length_km": total_length_osm_sat
        }
        
        print(f"OSM base + SAT patch:")
        print(f"  Injected edges: {injected_count_osm_sat}")
        print(f"  Total edges: {total_edges_osm_sat}")
        print(f"  Total length: {total_length_osm_sat:.2f} km")
    
    # Generate typst table string
    typst_table = generate_fusion_typst_table(results, threshold, table_type="unimodal")
    print("\n" + "="*50)
    print("TYPST TABLE:")
    print("="*50)
    print(typst_table)
    
    logger("Unimodal fusion analysis experiment completed.")
    return results, typst_table


def generate_fusion_typst_table(results, threshold, table_type="unimodal"):
    """Generate typst table string from fusion analysis results.
    
    Args:
        results: Dictionary containing fusion analysis results
        threshold: Distance threshold used in fusion
        table_type: Either "unimodal" or "selective_injection"
    """
    
    # Set scenario labels and caption based on table type
    if table_type == "unimodal":
        scenario_labels = {
            "gps_base_osm_patch": ("GPS", "OSM"),
            "osm_base_gps_patch": ("OSM", "GPS"),
            "sat_base_osm_patch": ("SAT", "OSM"),
            "osm_base_sat_patch": ("OSM", "SAT")
        }
        caption = f"Unimodal fusion analysis results (threshold = {threshold}m)."
        table_label = "table:unimodal-fusion-analysis"
    else:  # selective_injection
        scenario_labels = {
            "idr_sat_base_osm_patch": ("I*DR_{SAT}", "OSM"),
            "osm_base_idr_sat_patch": ("OSM", "I*DR_{SAT}"),
            "idr_gps_base_osm_patch": ("I*DR_{GPS}", "OSM"),
            "osm_base_idr_gps_patch": ("OSM", "I*DR_{GPS}")
        }
        caption = f"Selective injection (I*DR) fusion analysis results (threshold = {threshold}m)."
        table_label = "table:selective-injection-fusion-analysis"
    
    typst_header = f"""#show table.cell.where(y: 0): strong
#set table(
  stroke: (x, y) => 
    if y == 0 {{
      ( bottom: 0.7pt + black)
    }},
  align: (x, y) => (
    if x == 0 {{ left }}
    else {{ center }}
  ),
  column-gutter: auto
)

#figure(
  table(
    columns: 9,
    table.header(
      [Scenario],
      [Place],
      [Base],
      [Patch],
      [Injected Edges],
      [Total Edges],
      [Total Length (km)],
      [Inj./Tot. Edges],
      [Inj./Tot. Length],
    ),"""

    typst_rows = []
    
    scenario_counter = 1
    for scenario, (base, patch) in scenario_labels.items():
        for place in sorted(results.keys()):
            data = results[place][scenario]
            place_display = place.title()
            
            # Calculate ratios
            inj_edges_ratio = data['injected_edges'] / data['total_edges'] if data['total_edges'] > 0 else 0
            inj_length_ratio = data['injected_edges'] / data['total_length_km'] if data['total_length_km'] > 0 else 0
            
            typst_rows.append(f"    [Scenario {scenario_counter}], [{place_display}], [*{base}*], [*{patch}*], [{data['injected_edges']}], [{data['total_edges']}], [{data['total_length_km']:.2f}], [{inj_edges_ratio:.4f}], [{inj_length_ratio:.2f}],")
        scenario_counter += 1
    
    typst_footer = f"""  ),
  caption: [{caption}],
) <{table_label}>"""

    return typst_header + "\n" + "\n".join(typst_rows) + "\n" + typst_footer


def experiment_selective_injection_fusion_analysis(threshold=30):
    """
    Analyze selective injection (I*DR) fused maps against ground truth (OSM).
    Uses existing fused maps (A2, B2, C2) and applies them as base with OSM as patch, and vice versa.
    Count injected edges and report total edge count and edge length for each fusion scenario.
    Returns dictionary with data results and generates typst table string.
    """
    logger("Starting selective injection fusion analysis experiment.")
    
    # First ensure we have the I*DR fused maps available
    obtain_fusion_maps(threshold=threshold, covered_injection_only=True)
    obtain_fusion_maps(threshold=threshold, inverse=True, covered_injection_only=True)
    
    results = {}
    
    for place in places:
        logger(f"Analyzing {place}...")
        results[place] = {}
        
        # Load ground truth and fused maps
        osm = read_osm_graph(place)
        
        # Load I*DR fused maps (A2, B2, C2) and clean them up
        idr_sat = read_graph(data_location(place, "C2", threshold=threshold)["graph_file"])  # I*DR_SAT
        idr_gps = read_graph(data_location(place, "C2", threshold=threshold, inverse=True)["graph_file"])  # I*DR_GPS
        
        # Remove deleted edges and clear coverage annotations from fused maps
        idr_sat = remove_deleted(idr_sat)
        idr_gps = remove_deleted(idr_gps)
        
        # Clear edge coverage annotations (threshold attributes) from previous fusion operations
        for eid, attrs in iterate_edges(idr_sat):
            if "threshold" in attrs:
                del attrs["threshold"]
        for eid, attrs in iterate_edges(idr_gps):
            if "threshold" in attrs:
                del attrs["threshold"]
        
        print(f"\n=== {place.upper()} ===")
        
        # Scenario 1: I*DR_SAT as base, OSM as patch
        logger("I*DR_SAT as base, OSM as patch")
        osm_vs_idr_sat = edge_graph_coverage(osm, idr_sat, max_threshold=threshold)
        sanity_check_graph(idr_sat)
        sanity_check_graph(osm_vs_idr_sat)
        graphs_idr_sat_base = map_fusion(C=idr_sat, A=osm_vs_idr_sat, prune_threshold=threshold,
                                       remove_duplicates=True, reconnect_after=True,
                                       covered_injection_only=True)
        fusion_idr_sat_base = graphs_idr_sat_base["c"]
        
        # Count metrics for I*DR_SAT base scenario
        injected_count_idr_sat = len(filter_eids_by_attribute(fusion_idr_sat_base, filter_attributes={"render": "injected"}))
        total_edges_idr_sat = len(fusion_idr_sat_base.edges)
        total_length_idr_sat = sum([attrs["length"] for _, attrs in iterate_edges(fusion_idr_sat_base)]) / 1000
        
        results[place]["idr_sat_base_osm_patch"] = {
            "injected_edges": injected_count_idr_sat,
            "total_edges": total_edges_idr_sat,
            "total_length_km": total_length_idr_sat
        }
        
        print(f"I*DR_SAT base + OSM patch:")
        print(f"  Injected edges: {injected_count_idr_sat}")
        print(f"  Total edges: {total_edges_idr_sat}")
        print(f"  Total length: {total_length_idr_sat:.2f} km")
        
        # Scenario 2: OSM as base, I*DR_SAT as patch
        logger("OSM as base, I*DR_SAT as patch")
        idr_sat_vs_osm = edge_graph_coverage(idr_sat, osm, max_threshold=threshold)
        sanity_check_graph(osm)
        sanity_check_graph(idr_sat_vs_osm)
        graphs_osm_idr_sat_base = map_fusion(C=osm, A=idr_sat_vs_osm, prune_threshold=threshold,
                                           remove_duplicates=True, reconnect_after=True,
                                           covered_injection_only=True)
        fusion_osm_idr_sat_base = graphs_osm_idr_sat_base["c"]
        
        # Count metrics for OSM base + I*DR_SAT patch scenario
        injected_count_osm_idr_sat = len(filter_eids_by_attribute(fusion_osm_idr_sat_base, filter_attributes={"render": "injected"}))
        total_edges_osm_idr_sat = len(fusion_osm_idr_sat_base.edges)
        total_length_osm_idr_sat = sum([attrs["length"] for _, attrs in iterate_edges(fusion_osm_idr_sat_base)]) / 1000
        
        results[place]["osm_base_idr_sat_patch"] = {
            "injected_edges": injected_count_osm_idr_sat,
            "total_edges": total_edges_osm_idr_sat,
            "total_length_km": total_length_osm_idr_sat
        }
        
        print(f"OSM base + I*DR_SAT patch:")
        print(f"  Injected edges: {injected_count_osm_idr_sat}")
        print(f"  Total edges: {total_edges_osm_idr_sat}")
        print(f"  Total length: {total_length_osm_idr_sat:.2f} km")
        
        # Scenario 3: I*DR_GPS as base, OSM as patch
        logger("I*DR_GPS as base, OSM as patch")
        osm_vs_idr_gps = edge_graph_coverage(osm, idr_gps, max_threshold=threshold)
        sanity_check_graph(idr_gps)
        sanity_check_graph(osm_vs_idr_gps)
        graphs_idr_gps_base = map_fusion(C=idr_gps, A=osm_vs_idr_gps, prune_threshold=threshold,
                                       remove_duplicates=True, reconnect_after=True,
                                       covered_injection_only=True)
        fusion_idr_gps_base = graphs_idr_gps_base["c"]
        
        # Count metrics for I*DR_GPS base scenario
        injected_count_idr_gps = len(filter_eids_by_attribute(fusion_idr_gps_base, filter_attributes={"render": "injected"}))
        total_edges_idr_gps = len(fusion_idr_gps_base.edges)
        total_length_idr_gps = sum([attrs["length"] for _, attrs in iterate_edges(fusion_idr_gps_base)]) / 1000
        
        results[place]["idr_gps_base_osm_patch"] = {
            "injected_edges": injected_count_idr_gps,
            "total_edges": total_edges_idr_gps,
            "total_length_km": total_length_idr_gps
        }
        
        print(f"I*DR_GPS base + OSM patch:")
        print(f"  Injected edges: {injected_count_idr_gps}")
        print(f"  Total edges: {total_edges_idr_gps}")
        print(f"  Total length: {total_length_idr_gps:.2f} km")
        
        # Scenario 4: OSM as base, I*DR_GPS as patch
        logger("OSM as base, I*DR_GPS as patch")
        idr_gps_vs_osm = edge_graph_coverage(idr_gps, osm, max_threshold=threshold)
        sanity_check_graph(osm)
        sanity_check_graph(idr_gps_vs_osm)
        graphs_osm_idr_gps_base = map_fusion(C=osm, A=idr_gps_vs_osm, prune_threshold=threshold,
                                           remove_duplicates=True, reconnect_after=True,
                                           covered_injection_only=True)
        fusion_osm_idr_gps_base = graphs_osm_idr_gps_base["c"]
        
        # Count metrics for OSM base + I*DR_GPS patch scenario
        injected_count_osm_idr_gps = len(filter_eids_by_attribute(fusion_osm_idr_gps_base, filter_attributes={"render": "injected"}))
        total_edges_osm_idr_gps = len(fusion_osm_idr_gps_base.edges)
        total_length_osm_idr_gps = sum([attrs["length"] for _, attrs in iterate_edges(fusion_osm_idr_gps_base)]) / 1000
        
        results[place]["osm_base_idr_gps_patch"] = {
            "injected_edges": injected_count_osm_idr_gps,
            "total_edges": total_edges_osm_idr_gps,
            "total_length_km": total_length_osm_idr_gps
        }
        
        print(f"OSM base + I*DR_GPS patch:")
        print(f"  Injected edges: {injected_count_osm_idr_gps}")
        print(f"  Total edges: {total_edges_osm_idr_gps}")
        print(f"  Total length: {total_length_osm_idr_gps:.2f} km")
    
    # Generate typst table string
    typst_table = generate_fusion_typst_table(results, threshold, table_type="selective_injection")
    print("\n" + "="*50)
    print("TYPST TABLE:")
    print("="*50)
    print(typst_table)
    
    logger("Selective injection fusion analysis experiment completed.")
    return results, typst_table


def get_performance_data_for_place(place, threshold=30, covered_injection_only=True, metric_threshold=None, metric_interval=None, sample_count=10000, prime_sample_count=2000):
    """Extract performance data for a single place, similar to experiments_one_base_table but returning data."""
    
    if metric_threshold == None:
        metric_threshold = 5
    if metric_interval == None:
        metric_interval = metric_threshold

    # Read in TOPO and APLS samples and compute metric scores.
    table_results = {}
    selected_variants = set(base_variants + get_fusion_variants(covered_injection_only)) - set(["osm"])
    for variant in selected_variants:
        table_results[variant] = {}
        for inverse in [False,True]:
            if inverse and variant in base_variants:
                continue
            table_results[variant][inverse] = {}
            print(f"{place}-{variant}-{inverse}")
            
            # Asymmetric APLS results.
            location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric="apls", metric_threshold=metric_threshold, metric_interval=metric_interval)["metrics_samples"]
            samples = read_pickle(location)
            assert len(samples) >= sample_count
            
            # Load prime samples from the correct location
            prime_location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric="apls", metric_threshold=metric_threshold, metric_interval=metric_interval, prime_samples=True)["metrics_samples"]
            prime_samples = read_pickle(prime_location)
            assert len(prime_samples) >= prime_sample_count
            
            apls       = asymmetric_apls_from_samples(samples[:sample_count], prime=False)
            apls_prime = asymmetric_apls_from_samples(prime_samples[:prime_sample_count], prime=True)

            # Asymmetric TOPO results.
            location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric="topo", metric_threshold=metric_threshold, metric_interval=metric_interval)["metrics_samples"]
            samples = read_pickle(location)
            assert len(samples) >= sample_count
            
            # Load prime samples from the correct location
            prime_location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric="topo", metric_threshold=metric_threshold, metric_interval=metric_interval, prime_samples=True)["metrics_samples"]
            prime_samples = read_pickle(prime_location)
            assert len(prime_samples) >= prime_sample_count
            
            topo_results       = asymmetric_topo_from_samples(samples[:sample_count], False)
            topo_prime_results = asymmetric_topo_from_samples(prime_samples[:prime_sample_count], True)

            table_results[variant][inverse] = {
                "apls": apls,
                "apls_prime": apls_prime,
                "topo": {
                    "recall": topo_results["recall"],
                    "precision": topo_results["precision"],
                    "f1": topo_results["f1"],
                },
                "topo_prime": {
                    "recall": topo_prime_results["recall"],
                    "precision": topo_prime_results["precision"],
                    "f1": topo_prime_results["f1"],
                },
            }
    
    return table_results


def experiment_continuation_performance_correlation_heatmap(threshold=30):
    """
    Analyze correlations between road continuation quality and map performance metrics.
    
    Generates a 2×4 correlation heatmap that shows the statistical relationship between:
    - Y-axis (2 metrics): Road continuation quality changes (correct continuation change, correct discontinuation change)
    - X-axis (4 metrics): Map performance differences (TOPO F1, TOPO* F1, APLS, APLS*)
    
    Data Sources:
    - Locations: Berlin and Chicago datasets
    - Base maps: GPS and SAT-derived maps (2 per location = 4 total data points)
    - Comparison: I*DR fused maps vs their corresponding base unimodal maps
    
    Road Continuation Quality Change Metrics:
    1. Correct Continuation Change: Improvement in continuation quality from base unimodal map to I*DR fused map.
       - Base quality: 1.0 - (injected_edges / total_edges) from fusion(base=OSM, patch=unimodal)
       - Fused quality: 1.0 - (injected_edges / total_edges) from fusion(base=OSM, patch=I*DR_fused)
       - Change: Fused quality - Base quality
       - Positive values indicate fewer false continuations in fused map
       
    2. Correct Discontinuation Change: Improvement in discontinuation quality from base unimodal map to I*DR fused map.
       - Base quality: 1.0 - (injected_edges / total_edges) from fusion(base=unimodal, patch=OSM)
       - Fused quality: 1.0 - (injected_edges / total_edges) from fusion(base=I*DR_fused, patch=OSM)
       - Change: Fused quality - Base quality
       - Positive values indicate fewer missing roads in fused map
    
    Map Performance Differences:
    - TOPO F1 Change: Difference in topological similarity F1-score (fused - base)
    - TOPO* F1 Change: Difference in prime topological similarity F1-score (fused - base)
    - APLS Change: Difference in Average Path Length Similarity score (fused - base)
    - APLS* Change: Difference in prime APLS score (fused - base)
    
    Implementation Details:
    - Uses cached results when available (threshold-specific caching)
    - Calls experiment_unimodal_fusion_analysis() for base continuation quality data
    - Calls experiment_selective_injection_fusion_analysis() for fused continuation quality data
    - Calls get_performance_data_for_place() for similarity metric performance data
    - For GPS: compares I*DR_GPS (C2 inverse) vs base GPS for both continuation and performance
    - For SAT: compares I*DR_SAT (C2 normal) vs base SAT for both continuation and performance
    - Visualizes results using plot_continuation_performance_heatmap()
    
    Args:
        threshold (int): Distance threshold in meters for map fusion operations (default: 30)
        
    Returns:
        tuple: (combined_df, heatmap_matrix)
            - combined_df: DataFrame with all data points and computed change metrics
            - heatmap_matrix: 2×4 correlation matrix showing correlations between road continuation quality changes and performance changes
    """
    logger("Starting continuation performance correlation heatmap experiment.")
    
    # Check for cached results first
    cache_file = f"data/experiments/continuation_performance_correlation_cache_t{threshold}.pkl"
    try:
        cached_data = read_pickle(cache_file)
        if cached_data.get("threshold") == threshold:
            logger(f"Loading cached correlation data from {cache_file}")
            combined_df = cached_data["combined_df"]
            heatmap_matrix = cached_data["heatmap_matrix"]
            
            # Create visualization with cached data
            plot_continuation_performance_heatmap(heatmap_matrix, threshold, combined_df)
            
            logger("Continuation performance correlation heatmap experiment completed (from cache).")
            return combined_df, heatmap_matrix
        else:
            logger(f"Cache threshold mismatch, recomputing...")
    except (FileNotFoundError, KeyError, Exception) as e:
        logger(f"Cache not found or corrupted ({e}), computing from scratch...")
    
    # Ensure we have the necessary data prepared
    obtain_prepared_metric_maps(threshold=threshold, covered_injection_only=True)
    obtain_prepared_metric_maps(threshold=threshold, inverse=True, covered_injection_only=True)
    
    # Get fusion analysis data for road continuation metrics
    unimodal_results, _ = experiment_unimodal_fusion_analysis(threshold=threshold)
    selective_results, _ = experiment_selective_injection_fusion_analysis(threshold=threshold)
    
    # Collect all data points from all combinations
    all_data = []
    
    places = ["berlin", "chicago"]
    base_maps = ["gps", "sat"]
    
    for place in places:
        logger(f"Processing correlation data for {place}...")
        
        # Get performance data for this place
        performance_data = get_performance_data_for_place(place, threshold=threshold, covered_injection_only=True)
        
        for base_map in base_maps:
            logger(f"Processing {place} × {base_map} combination...")
            
            # Get the correct fusion variant and performance data
            if base_map == "gps":
                # For GPS base: use unimodal GPS vs OSM data and I*DR_GPS (C2 inverse) vs base GPS performance
                base_continuation_data = unimodal_results[place]["osm_base_gps_patch"]
                base_discontinuation_data = unimodal_results[place]["gps_base_osm_patch"]
                
                # For I*DR_GPS: use selective injection results
                fused_continuation_data = selective_results[place]["osm_base_idr_gps_patch"]
                fused_discontinuation_data = selective_results[place]["idr_gps_base_osm_patch"]
                
                # Performance comparison: I*DR_GPS (C2 inverse) vs base GPS
                fused_perf = performance_data["C2"][True]  # I*DR_GPS
                base_perf = performance_data["gps"][False]
                
            else:  # base_map == "sat"
                # For SAT base: use unimodal SAT vs OSM data and I*DR_SAT (C2 not inverse) vs base SAT performance  
                base_continuation_data = unimodal_results[place]["osm_base_sat_patch"]
                base_discontinuation_data = unimodal_results[place]["sat_base_osm_patch"]
                
                # For I*DR_SAT: use selective injection results
                fused_continuation_data = selective_results[place]["osm_base_idr_sat_patch"]
                fused_discontinuation_data = selective_results[place]["idr_sat_base_osm_patch"]
                
                # Performance comparison: I*DR_SAT (C2 not inverse) vs base SAT
                fused_perf = performance_data["C2"][False]  # I*DR_SAT
                base_perf = performance_data["sat"][False]
            
            # Calculate road continuation quality changes (fused - base)
            # Base continuation quality
            base_incorrect_continuation_ratio = base_continuation_data["injected_edges"] / base_continuation_data["total_edges"]
            base_correct_continuation = 1.0 - base_incorrect_continuation_ratio
            
            # Fused continuation quality
            fused_incorrect_continuation_ratio = fused_continuation_data["injected_edges"] / fused_continuation_data["total_edges"]
            fused_correct_continuation = 1.0 - fused_incorrect_continuation_ratio
            
            # Continuation quality change
            correct_continuation_change = fused_correct_continuation - base_correct_continuation
            
            # Base discontinuation quality
            base_incorrect_discontinuation_ratio = base_discontinuation_data["injected_edges"] / base_discontinuation_data["total_edges"]
            base_correct_discontinuation = 1.0 - base_incorrect_discontinuation_ratio
            
            # Fused discontinuation quality
            fused_incorrect_discontinuation_ratio = fused_discontinuation_data["injected_edges"] / fused_discontinuation_data["total_edges"]
            fused_correct_discontinuation = 1.0 - fused_incorrect_discontinuation_ratio
            
            # Discontinuation quality change
            correct_discontinuation_change = fused_correct_discontinuation - base_correct_discontinuation
            
            # Calculate performance changes (relative changes)
            topo_change = fused_perf["topo"]["f1"] - base_perf["topo"]["f1"]
            topo_prime_change = fused_perf["topo_prime"]["f1"] - base_perf["topo_prime"]["f1"]
            apls_change = fused_perf["apls"] - base_perf["apls"]
            apls_prime_change = fused_perf["apls_prime"] - base_perf["apls_prime"]
            
            # Collect all metrics for this data point
            all_data.append({
                "correct_continuation_change": correct_continuation_change,
                "correct_discontinuation_change": correct_discontinuation_change,
                "topo_change": topo_change,
                "topo_prime_change": topo_prime_change,
                "apls_change": apls_change,
                "apls_prime_change": apls_prime_change,
                "place": place,
                "base_map": base_map
            })
    
    # Convert to DataFrame
    combined_df = pd.DataFrame(all_data)
    
    # Select correlation columns: 2 road continuation changes + 4 performance changes
    corr_columns = [
        "correct_continuation_change", "correct_discontinuation_change",
        "topo_change", "topo_prime_change", "apls_change", "apls_prime_change"
    ]
    
    # Create correlation matrix
    correlation_matrix = combined_df[corr_columns].corr()
    
    # Extract the 2×4 submatrix we want (road continuation changes vs performance changes)
    road_continuation_cols = ["correct_continuation_change", "correct_discontinuation_change"]  
    performance_cols = ["topo_change", "topo_prime_change", "apls_change", "apls_prime_change"]
    
    # Create 2×4 correlation matrix (road continuation quality vs map performance)
    heatmap_matrix = correlation_matrix.loc[road_continuation_cols, performance_cols]
    
    # Store computed data to cache before visualization
    cache_data = {
        "combined_df": combined_df,
        "heatmap_matrix": heatmap_matrix,
        "threshold": threshold
    }
    try:
        write_pickle(cache_file, cache_data)
        logger(f"Cached correlation data to {cache_file}")
    except Exception as e:
        logger(f"Warning: Failed to cache data ({e}), continuing with visualization...")
    
    # Create visualization
    plot_continuation_performance_heatmap(heatmap_matrix, threshold, combined_df)
    
    logger("Continuation performance correlation heatmap experiment completed.")
    return combined_df, heatmap_matrix


def plot_continuation_performance_heatmap(correlation_matrix, threshold, data_df):
    """
    Create a single 2×4 heatmap visualization for the correlation matrix.
    Shows relationships between road continuation quality (rows) and map performance (columns).
    Each cell displays both correlation coefficient and average change value.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Set up the plot: single heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig.suptitle(f'Road Continuation Quality Changes vs Performance Changes Correlations (threshold={threshold}m)', fontsize=14)
    
    # Better labels for the variables
    row_labels = [
        "Δ Correct\nContinuation",
        "Δ Correct\nDiscontinuation"
    ]
    
    col_labels = [
        "Δ TOPO\n(Normal)",
        "Δ TOPO\n(Prime)", 
        "Δ APLS\n(Normal)",
        "Δ APLS\n(Prime)"
    ]
    
    # Calculate average values for each metric
    road_continuation_cols = ["correct_continuation_change", "correct_discontinuation_change"]  
    performance_cols = ["topo_change", "topo_prime_change", "apls_change", "apls_prime_change"]
    
    # Create matrix of average values
    avg_matrix = data_df[road_continuation_cols + performance_cols].mean()
    continuation_avg_matrix = np.zeros((2, 4))
    performance_avg_matrix = np.zeros((2, 4))
    
    for i, road_col in enumerate(road_continuation_cols):
        for j, perf_col in enumerate(performance_cols):
            continuation_avg_matrix[i, j] = avg_matrix[road_col]
            performance_avg_matrix[i, j] = avg_matrix[perf_col]
    
    # Create custom annotation matrix combining correlation and average values
    annot_matrix = []
    for i in range(correlation_matrix.shape[0]):
        row = []
        for j in range(correlation_matrix.shape[1]):
            corr_val = correlation_matrix.iloc[i, j]
            cont_avg = continuation_avg_matrix[i, j]
            perf_avg = performance_avg_matrix[i, j]
            # Format: correlation on top line, (continuation avg, performance avg) on bottom line
            annotation = f"{corr_val:.3f}\n({cont_avg:.3f}, {perf_avg:.3f})"
            row.append(annotation)
        annot_matrix.append(row)
    
    # For correlation matrices, values are always between -1 and 1
    vmin, vmax = -1, 1
    
    # Create heatmap with custom annotations
    sns.heatmap(
        correlation_matrix, 
        ax=ax,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=annot_matrix,
        fmt='',  # Empty format since we're providing custom strings
        cmap='RdBu_r',
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=False,
        cbar_kws={'label': 'Correlation Coefficient'},
        annot_kws={'fontsize': 10}
    )
    
    # Add subplot details
    n_data_points = len(data_df)
    places = sorted(data_df['place'].unique())
    base_maps = sorted(data_df['base_map'].unique())
    
    ax.set_xlabel("Map Performance Changes", fontsize=12)
    ax.set_ylabel("Road Continuation Quality Changes", fontsize=12)
    
    # Add caption with data details and explanation
    caption = f"n={n_data_points} data points from {places} × {base_maps}\nEach cell shows: correlation coefficient (continuation change avg, performance change avg)"
    plt.figtext(0.5, 0.02, caption, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.18)
    plt.show()
