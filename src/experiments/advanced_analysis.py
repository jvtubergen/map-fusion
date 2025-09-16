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


def experiment_unimodal_fusion_analysis(threshold=30, covered_injection_only=False):
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
                                   covered_injection_only=covered_injection_only)
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
                                   covered_injection_only=covered_injection_only)
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
                                   covered_injection_only=covered_injection_only)
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
                                       covered_injection_only=covered_injection_only)
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
    typst_table = generate_unimodal_fusion_typst_table(results, threshold)
    print("\n" + "="*50)
    print("TYPST TABLE:")
    print("="*50)
    print(typst_table)
    
    logger("Unimodal fusion analysis experiment completed.")
    return results, typst_table


def generate_unimodal_fusion_typst_table(results, threshold):
    """Generate typst table string from unimodal fusion analysis results."""
    
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
    
    scenario_labels = {
        "gps_base_osm_patch": ("GPS", "OSM"),
        "osm_base_gps_patch": ("OSM", "GPS"),
        "sat_base_osm_patch": ("SAT", "OSM"),
        "osm_base_sat_patch": ("OSM", "SAT")
    }
    
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
  caption: [Unimodal fusion analysis results (threshold = {threshold}m).],
) <table:unimodal-fusion-analysis>"""

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
    typst_table = generate_selective_injection_fusion_typst_table(results, threshold)
    print("\n" + "="*50)
    print("TYPST TABLE:")
    print("="*50)
    print(typst_table)
    
    logger("Selective injection fusion analysis experiment completed.")
    return results, typst_table


def generate_selective_injection_fusion_typst_table(results, threshold):
    """Generate typst table string from selective injection fusion analysis results."""
    
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
    
    scenario_labels = {
        "idr_sat_base_osm_patch": ("I*DR_{SAT}", "OSM"),
        "osm_base_idr_sat_patch": ("OSM", "I*DR_{SAT}"),
        "idr_gps_base_osm_patch": ("I*DR_{GPS}", "OSM"),
        "osm_base_idr_gps_patch": ("OSM", "I*DR_{GPS}")
    }
    
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
  caption: [Selective injection (I*DR) fusion analysis results (threshold = {threshold}m).],
) <table:selective-injection-fusion-analysis>"""

    return typst_header + "\n" + "\n".join(typst_rows) + "\n" + typst_footer


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


def experiment_road_continuation_correlation_analysis(threshold=30):
    """
    Build comprehensive covariance matrices connecting road continuation quality metrics with map performance changes.
    Creates single 6×6 covariance matrices per place showing all pairwise relationships between:
    - road_cont_discontinuation: edges injected when I*DR as base, OSM as patch vs ground truth
    - road_cont_continuation: edges injected when OSM as base, I*DR as patch vs ground truth  
    - topo_change, topo_prime_change, apls_change, apls_prime_change
    
    Road continuation quality measurements:
    - Discontinuation: I*DR map as base, OSM (ground truth) as patch - captures missing roads
    - Continuation: OSM (ground truth) as base, I*DR map as patch - captures extra roads
    """
    logger("Starting road continuation correlation analysis experiment.")
    
    # First ensure we have all the necessary data prepared
    obtain_prepared_metric_maps(threshold=threshold, covered_injection_only=True)
    obtain_prepared_metric_maps(threshold=threshold, inverse=True, covered_injection_only=True)
    
    # Get road continuation data from both selective injection and unimodal fusion experiments
    selective_results, _ = experiment_selective_injection_fusion_analysis(threshold=threshold)
    unimodal_results, _ = experiment_unimodal_fusion_analysis(threshold=threshold)
    
    # Collect comprehensive correlation data for each place
    comprehensive_matrices = {}
    comprehensive_data = {}
    
    for place in places:
        logger(f"Processing comprehensive correlation data for {place}...")
        
        # Get performance table data for this place
        performance_data = get_performance_data_for_place(place, threshold=threshold, covered_injection_only=True)
        
        # Initialize data collection for this place
        place_data = []
        
        # Process I*DR vs unimodal base comparisons
        fused_variants = {
            "I*DR_SAT": ("C2", False, "sat"),  # I*DR_SAT (fusion variant C2, not inverse, base is sat)
            "I*DR_GPS": ("C2", True, "gps")    # I*DR_GPS (fusion variant C2, inverse, base is gps)
        }
        
        for fused_name, (variant, inverse, base_variant) in fused_variants.items():
            
            # Get road continuation quality metrics
            # Type 1: Incorrectly road discontinuation (I*DR as base, OSM as patch vs ground truth)
            if fused_name == "I*DR_SAT":
                discontinuation_data = selective_results[place]["idr_sat_base_osm_patch"]
            else:  # I*DR_GPS
                discontinuation_data = selective_results[place]["idr_gps_base_osm_patch"]
            
            discontinuation_ratio = discontinuation_data["injected_edges"] / discontinuation_data["total_edges"]
            
            # Type 2: Incorrectly road continuation conflicts (OSM as base, I*DR as patch vs ground truth)
            if fused_name == "I*DR_SAT":
                continuation_data = selective_results[place]["osm_base_idr_sat_patch"]
            else:  # I*DR_GPS  
                continuation_data = selective_results[place]["osm_base_idr_gps_patch"]
            
            continuation_ratio = continuation_data["injected_edges"] / continuation_data["total_edges"]
            
            # Get performance metrics for fused and base maps
            fused_perf = performance_data[variant][inverse]
            base_perf = performance_data[base_variant][False]
            
            # Calculate performance changes for both regular and prime metrics
            topo_change = fused_perf["topo"]["f1"] - base_perf["topo"]["f1"]
            apls_change = fused_perf["apls"] - base_perf["apls"]
            topo_prime_change = fused_perf["topo_prime"]["f1"] - base_perf["topo_prime"]["f1"]
            apls_prime_change = fused_perf["apls_prime"] - base_perf["apls_prime"]
            
            # Collect all metrics for this data point
            place_data.append({
                "road_cont_discontinuation": discontinuation_ratio,
                "road_cont_continuation": continuation_ratio,
                "topo_change": topo_change,
                "topo_prime_change": topo_prime_change,
                "apls_change": apls_change,
                "apls_prime_change": apls_prime_change,
                "fused_variant": fused_name,
                "base_variant": base_variant
            })
        
        # Convert to DataFrame and create comprehensive covariance matrix
        place_df = pd.DataFrame(place_data)
        
        # Select the 6 correlation columns
        corr_columns = [
            "road_cont_discontinuation", "road_cont_continuation",
            "topo_change", "topo_prime_change", "apls_change", "apls_prime_change"
        ]
        
        # Create comprehensive 6×6 covariance matrix
        comprehensive_cov_matrix = place_df[corr_columns].cov()
        
        # Store results
        comprehensive_matrices[place] = comprehensive_cov_matrix
        comprehensive_data[place] = place_df
    
    # Create visualization
    plot_comprehensive_covariance_matrices(comprehensive_matrices, threshold)
    
    logger("Road continuation correlation analysis experiment completed.")
    return comprehensive_data, comprehensive_matrices


def experiment_continuation_performance_correlation_analysis(threshold=30):
    """
    Create four 6×6 correlation matrices representing Berlin/Chicago × GPS/SAT combinations.
    Each matrix shows relationships between road continuation quality and map performance.
    It uses the difference in map performance between the base map (a unimodal map) and the fused map (with that unimodal map as base map), similarly for road continuation.
    This provides for every combination (in choice of map performance metric and road continuation quality) a data point to correlate (namely whether performance/quality improved/decreased) and is applicable to four combinations (in choice of place and base map).
    
    Each 6×6 correlation matrix includes:
    * Road continuation quality (2 measures):
        * Correct continuation: OSM base + base map patch injected ratio
        * Correct discontinuation: base map base + OSM patch injected ratio  
    * Map performance (4 measures):
        * TOPO (normal), TOPO (prime), APLS (normal), APLS (prime)
    """
    logger("Starting continuation performance correlation analysis experiment.")
    
    # Ensure we have the necessary data prepared
    obtain_prepared_metric_maps(threshold=threshold, covered_injection_only=True)
    obtain_prepared_metric_maps(threshold=threshold, inverse=True, covered_injection_only=True)
    
    # Get fusion analysis data for road continuation metrics
    selective_results, _ = experiment_selective_injection_fusion_analysis(threshold=threshold)
    unimodal_results, _ = experiment_unimodal_fusion_analysis(threshold=threshold)
    
    # Collect correlation matrices for each place × base map combination
    correlation_matrices = {}
    combined_data = {}
    
    places = ["berlin", "chicago"]
    base_maps = ["gps", "sat"]
    
    for place in places:
        logger(f"Processing correlation data for {place}...")
        
        # Get performance data for this place
        performance_data = get_performance_data_for_place(place, threshold=threshold, covered_injection_only=True)
        
        for base_map in base_maps:
            matrix_key = f"{place}_{base_map}"
            logger(f"Creating matrix for {matrix_key}...")
            
            # Initialize data collection for this place × base_map combination
            place_base_data = []
            
            # Get the correct fusion variant and performance data
            if base_map == "gps":
                # For GPS base: use unimodal GPS vs OSM data and I*DR_GPS (C2 inverse) vs base GPS performance
                continuation_data = unimodal_results[place]["osm_base_gps_patch"]
                discontinuation_data = unimodal_results[place]["gps_base_osm_patch"]
                
                # Performance comparison: I*DR_GPS (C2 inverse) vs base GPS
                fused_perf = performance_data["C2"][True]  # I*DR_GPS
                base_perf = performance_data["gps"][False]
                
            else:  # base_map == "sat"
                # For SAT base: use unimodal SAT vs OSM data and I*DR_SAT (C2 not inverse) vs base SAT performance  
                continuation_data = unimodal_results[place]["osm_base_sat_patch"]
                discontinuation_data = unimodal_results[place]["sat_base_osm_patch"]
                
                # Performance comparison: I*DR_SAT (C2 not inverse) vs base SAT
                fused_perf = performance_data["C2"][False]  # I*DR_SAT
                base_perf = performance_data["sat"][False]
            
            # Calculate road continuation quality metrics
            continuation_ratio = continuation_data["injected_edges"] / continuation_data["total_edges"]
            discontinuation_ratio = discontinuation_data["injected_edges"] / discontinuation_data["total_edges"]
            
            # Calculate performance changes
            topo_change = fused_perf["topo"]["f1"] - base_perf["topo"]["f1"]
            topo_prime_change = fused_perf["topo_prime"]["f1"] - base_perf["topo_prime"]["f1"]
            apls_change = fused_perf["apls"] - base_perf["apls"]
            apls_prime_change = fused_perf["apls_prime"] - base_perf["apls_prime"]
            
            # Collect all metrics for this data point
            place_base_data.append({
                "correct_continuation": continuation_ratio,
                "correct_discontinuation": discontinuation_ratio,
                "topo_change": topo_change,
                "topo_prime_change": topo_prime_change,
                "apls_change": apls_change,
                "apls_prime_change": apls_prime_change
            })
            
            # Convert to DataFrame and create 6×6 correlation matrix
            place_base_df = pd.DataFrame(place_base_data)
            
            # Create 6×6 correlation matrix (normalized covariance)
            corr_matrix = place_base_df.corr()
            
            # Store results
            correlation_matrices[matrix_key] = corr_matrix
            combined_data[matrix_key] = place_base_df
    
    # Create visualization
    plot_continuation_performance_correlation_heatmaps(correlation_matrices, threshold)
    
    logger("Continuation performance correlation analysis experiment completed.")
    return combined_data, correlation_matrices


def plot_continuation_performance_correlation_heatmaps(correlation_matrices, threshold):
    """
    Create 4 heatmap visualizations for the correlation matrices.
    Each heatmap shows relationships between road continuation quality and map performance.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Set up the plot grid: 2 rows × 2 columns = 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Road Continuation Quality vs Performance Correlation Matrices (threshold={threshold}m)', fontsize=16)
    
    # Better labels for the 6 variables
    labels = [
        "Correct\nContinuation",
        "Correct\nDiscontinuation", 
        "TOPO\n(Normal)",
        "TOPO\n(Prime)",
        "APLS\n(Normal)", 
        "APLS\n(Prime)"
    ]
    
    # For correlation matrices, values are always between -1 and 1
    vmin, vmax = -1, 1
    
    # Plot configuration
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    keys = ["berlin_gps", "berlin_sat", "chicago_gps", "chicago_sat"]
    titles = ["Berlin × GPS", "Berlin × SAT", "Chicago × GPS", "Chicago × SAT"]
    
    for idx, (key, title) in enumerate(zip(keys, titles)):
        row, col = positions[idx]
        ax = axes[row, col]
        
        if key in correlation_matrices:
            matrix = correlation_matrices[key]
            
            # Create heatmap
            sns.heatmap(
                matrix, 
                ax=ax,
                xticklabels=labels,
                yticklabels=labels,
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                vmin=vmin,
                vmax=vmax,
                square=True,
                cbar_kws={'label': 'Correlation'}
            )
            
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ax.text(0.5, 0.5, f'No data\nfor {title}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()


def experiment_continuation_performance_correlation_heatmap(threshold=30):
    """
    Create a single 2×4 correlation heatmap showing relationships between:
    - Rows (2): Road continuation quality measures (correct continuation, correct discontinuation)  
    - Columns (4): Map performance measures (TOPO, TOPO prime, APLS, APLS prime)
    
    Uses all data points from Berlin+Chicago × GPS+SAT combinations to compute correlations.
    Road continuation quality is measured as relative change between fused and unimodal maps.
    Map performance is measured as relative change between fused and unimodal maps.
    
    Road Continuation Quality Metrics Computation:
    - Correct Continuation: Measures how well roads are correctly continued by computing
      the complement of incorrect continuation. Incorrect continuation occurs when OSM 
      (ground truth) is used as base and unimodal map (GPS/SAT) is used as patch, measuring
      roads the unimodal map adds that weren't in ground truth (continuation conflicts).
      Formula: 100% - (injected_edges / total_edges) from fusion(base=OSM, patch=unimodal)
      
    - Correct Discontinuation: Measures how well roads are correctly discontinued by computing
      the complement of incorrect discontinuation. Incorrect discontinuation occurs when 
      unimodal map (GPS/SAT) is used as base and OSM (ground truth) is used as patch, measuring
      roads the ground truth adds that weren't in the unimodal map (discontinuation issues).
      Formula: 100% - (injected_edges / total_edges) from fusion(base=unimodal, patch=OSM)
    
    Both metrics use unimodal fusion analysis results where maps are fused with threshold
    distance matching and selective injection to identify structural differences. Higher
    percentages indicate better road continuation/discontinuation quality.
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
                continuation_data = unimodal_results[place]["osm_base_gps_patch"]
                discontinuation_data = unimodal_results[place]["gps_base_osm_patch"]
                
                # Performance comparison: I*DR_GPS (C2 inverse) vs base GPS
                fused_perf = performance_data["C2"][True]  # I*DR_GPS
                base_perf = performance_data["gps"][False]
                
            else:  # base_map == "sat"
                # For SAT base: use unimodal SAT vs OSM data and I*DR_SAT (C2 not inverse) vs base SAT performance  
                continuation_data = unimodal_results[place]["osm_base_sat_patch"]
                discontinuation_data = unimodal_results[place]["sat_base_osm_patch"]
                
                # Performance comparison: I*DR_SAT (C2 not inverse) vs base SAT
                fused_perf = performance_data["C2"][False]  # I*DR_SAT
                base_perf = performance_data["sat"][False]
            
            # Calculate road continuation quality metrics (correct percentages)
            # Correct continuation = 100% - incorrect continuation percentage
            incorrect_continuation_ratio = continuation_data["injected_edges"] / continuation_data["total_edges"]
            correct_continuation = 1.0 - incorrect_continuation_ratio  # As proportion (0-1)
            
            # Correct discontinuation = 100% - incorrect discontinuation percentage  
            incorrect_discontinuation_ratio = discontinuation_data["injected_edges"] / discontinuation_data["total_edges"]
            correct_discontinuation = 1.0 - incorrect_discontinuation_ratio  # As proportion (0-1)
            
            # Calculate performance changes (relative changes)
            topo_change = fused_perf["topo"]["f1"] - base_perf["topo"]["f1"]
            topo_prime_change = fused_perf["topo_prime"]["f1"] - base_perf["topo_prime"]["f1"]
            apls_change = fused_perf["apls"] - base_perf["apls"]
            apls_prime_change = fused_perf["apls_prime"] - base_perf["apls_prime"]
            
            # Collect all metrics for this data point
            all_data.append({
                "correct_continuation": correct_continuation,
                "correct_discontinuation": correct_discontinuation,
                "topo_change": topo_change,
                "topo_prime_change": topo_prime_change,
                "apls_change": apls_change,
                "apls_prime_change": apls_prime_change,
                "place": place,
                "base_map": base_map
            })
    
    # Convert to DataFrame
    combined_df = pd.DataFrame(all_data)
    
    # Select correlation columns: 2 road continuation + 4 performance measures
    corr_columns = [
        "correct_continuation", "correct_discontinuation",
        "topo_change", "topo_prime_change", "apls_change", "apls_prime_change"
    ]
    
    # Create correlation matrix
    correlation_matrix = combined_df[corr_columns].corr()
    
    # Extract the 2×4 submatrix we want (road continuation vs performance)
    road_continuation_cols = ["correct_continuation", "correct_discontinuation"]  
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
    fig.suptitle(f'Road Continuation Quality vs Performance Correlations (threshold={threshold}m)', fontsize=14)
    
    # Better labels for the variables
    row_labels = [
        "Correct\nContinuation",
        "Correct\nDiscontinuation"
    ]
    
    col_labels = [
        "TOPO\n(Normal)",
        "TOPO\n(Prime)", 
        "APLS\n(Normal)",
        "APLS\n(Prime)"
    ]
    
    # Calculate average values for each metric
    road_continuation_cols = ["correct_continuation", "correct_discontinuation"]  
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
    ax.set_ylabel("Road Continuation Quality", fontsize=12)
    
    # Add caption with data details and explanation
    caption = f"n={n_data_points} data points from {places} × {base_maps}\nEach cell shows: correlation coefficient (continuation avg, performance avg)"
    plt.figtext(0.5, 0.02, caption, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.18)
    plt.show()


def generate_correlation_typst_table(correlations, data_df, threshold):
    """Generate typst table for road continuation correlation analysis results."""
    
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
    columns: 3,
    table.header(
      [Performance Metric],
      [Correlation with Road Cont. Quality],
      [Interpretation],
    ),"""
    
    metric_labels = {
        "rec_change": "Δ Recall",
        "prec_change": "Δ Precision", 
        "topo_change": "Δ TOPO",
        "apls_change": "Δ APLS",
        "rec_prime_change": "Δ Recall*",
        "prec_prime_change": "Δ Precision*",
        "topo_prime_change": "Δ TOPO*", 
        "apls_prime_change": "Δ APLS*"    }
    
    typst_rows = []
    for metric, corr in correlations.items():
        if metric in metric_labels:
            label = metric_labels[metric]
            
            # Determine interpretation
            if abs(corr) >= 0.7:
                interpretation = "Strong"
            elif abs(corr) >= 0.3:
                interpretation = "Moderate" 
            else:
                interpretation = "Weak"
                
            if corr > 0:
                interpretation += " positive"
            else:
                interpretation += " negative"
            
            typst_rows.append(f"    [*{label}*], [{corr:.4f}], [{interpretation}],")
    
    # Add summary statistics
    n_comparisons = len(data_df)
    places = data_df['place'].unique()
    
    typst_footer = f"""  ),
  caption: [Road continuation quality vs. performance correlation analysis (n={n_comparisons} comparisons, {len(places)} places, threshold={threshold}m). Road continuation quality measured as injected edges / total edges ratio from selective injection fusion analysis.],
) <table:road-continuation-correlation>"""

    return typst_header + "\n" + "\n".join(typst_rows) + "\n" + typst_footer