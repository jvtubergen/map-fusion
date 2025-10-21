from storage import *
from utilities import *
from data import *
from map_similarity import *
from graph import *


def read_base_maps():
    maps = {}
    for place in places:
        maps[place] = {}
        maps[place]["osm"] = read_osm_graph(place)
        maps[place]["gps"] = read_gps_graph(place)
        maps[place]["sat"] = read_sat_graph(place)
    return maps


def read_fusion_maps(threshold = 30, covered_injection_only = False):
    """Read fusion maps of a specific threshold from disk."""
    maps = {}
    fusion_vars = get_fusion_variants(covered_injection_only)
    for place in places:
        maps[place] = {}
        for variant in fusion_vars:
            maps[place][variant] = read_graph(data_location(place, variant, threshold = threshold)["graph_file"])
    return maps


def read_all_maps(threshold = 30, covered_injection_only = False):
    """Read fusion maps of a specific threshold alongside the ground truth and two base maps from disk."""
    maps = {}
    fusion_vars = get_fusion_variants(covered_injection_only)
    for place in places:
        maps[place] = {}
        maps[place]["osm"] = read_osm_graph(place)
        maps[place]["gps"] = read_gps_graph(place)
        maps[place]["sat"] = read_sat_graph(place)
        for variant in fusion_vars:
            maps[place][variant] = read_graph(data_location(place, variant, threshold = threshold)["graph_file"])
    return maps


@info()
def obtain_fusion_maps(threshold = 30, debugging = False, inverse = False, re_use = True, covered_injection_only = False):
    """Obtain fusion maps and write them to disk."""

    fusion_vars = get_fusion_variants(covered_injection_only)
    
    for place in places: 
        if re_use and all(path_exists(data_location(place, variant, threshold=threshold, inverse=inverse)["graph_file"]) 
                         for variant in fusion_vars):
            continue

        logger(f"Apply map fusion to {place}.")
        osm = read_osm_graph(place)
        gps = read_gps_graph(place)
        sat = read_sat_graph(place)

        if debugging:
            # Prune Sat graph from edges too far away from GPS edge.s
            # In some cases it is convenient to only act around sat edges nearby gps edges where the action happens.
            logger("DEBUGGING: Pruning Sat graph for relevant edges concerning merging.")
            sat_vs_gps   = edge_graph_coverage(sat, gps, max_threshold=threshold)
            intersection = prune_coverage_graph(sat_vs_gps, prune_threshold=threshold)
            sat = intersection # (Update sat so we can continue further logic.)
        
        if inverse:
            logger(f"Apply map fusion inverse{' with covered injection only' if covered_injection_only else ''}.")
            sat_vs_gps = edge_graph_coverage(sat, gps, max_threshold=threshold)
            sanity_check_graph(gps)
            sanity_check_graph(sat_vs_gps)
            graphs = map_fusion(C=gps, A=sat_vs_gps, prune_threshold=threshold, remove_duplicates=True, reconnect_after=True, covered_injection_only=covered_injection_only)
        else:
            logger(f"Apply map fusion{' with covered injection only' if covered_injection_only else ''}.")
            gps_vs_sat = edge_graph_coverage(gps, sat, max_threshold=threshold)
            sanity_check_graph(sat)
            sanity_check_graph(gps_vs_sat)
            graphs = map_fusion(C=sat, A=gps_vs_sat, prune_threshold=threshold, remove_duplicates=True, reconnect_after=True, covered_injection_only=covered_injection_only)

        A = graphs["a"]
        B = graphs["b"]
        C = graphs["c"]

        # Write using the appropriate variant names (A/B/C or A2/B2/C2)
        for i, variant in enumerate(fusion_vars):
            graph_data = [A, B, C][i]
            write_graph(data_location(place, variant, threshold=threshold, inverse=inverse)["graph_file"], graph_data)


def remove_deleted(G):
    """Remove nodes and edges with `{"render": "deleted"}` attribute."""
    G = G.copy()
    edges_to_be_deleted = filter_eids_by_attribute(G, filter_attributes={"render": "deleted"})
    nodes_to_be_deleted = filter_nids_by_attribute(G, filter_attributes={"render": "deleted"})
    G.remove_edges_from(edges_to_be_deleted)
    G.remove_nodes_from(nodes_to_be_deleted)
    return G


@info()
def obtain_prepared_metric_maps(threshold = 30, fusion_only = False, inverse = False, re_use = True, covered_injection_only = False): # APLS + TOPO
    """
    Prepare graphs (identical for APLS and TOPO):  Remove edges and nodes with the {"render": "delete"} attribute.
    Basically only necessary for "B" and "C" (because other graphs nowhere annotate this specific "render" attribute value),
    yet for consistency in pipeline I just apply it to all graphs (effectively copying them over from data to experiments folder).
    """
    if inverse:
        fusion_only = True
    if fusion_only:
        _variants = get_fusion_variants(covered_injection_only)
    else:
        _variants = base_variants + get_fusion_variants(covered_injection_only)

    for place in places: 
        for variant in _variants:
            location = experiment_location(place, variant, threshold=threshold, inverse = inverse)["prepared_graph"]
            if re_use and path_exists(location):
                continue
            G = read_graph(data_location(place, variant, threshold = threshold, inverse = inverse)["graph_file"])
            G = remove_deleted(G)
            sanity_check_graph(G)
            write_graph(location, G)


@info()
def obtain_shortest_distance_dictionaries(threshold = 30, fusion_only = False, inverse = False, re_use = True, covered_injection_only = False): # APLS
    """
    Obtain shortest distance on a graph and write to disk.
    """
    if inverse:
        fusion_only = True
    if fusion_only:
        _variants = get_fusion_variants(covered_injection_only)
    else:
        _variants = base_variants + get_fusion_variants(covered_injection_only)

    for place in places: 
        for variant in _variants:
            print(f"{place}-{variant}")
            if re_use and path_exists(experiment_location(place, variant, threshold=threshold, inverse=inverse)["apls_shortest_paths"]):
                continue
            G = read_graph(experiment_location(place, variant, threshold=threshold, inverse=inverse)["prepared_graph"])
            shortest_paths = precompute_shortest_path_data(G)
            write_pickle(experiment_location(place, variant, threshold=threshold, inverse=inverse)["apls_shortest_paths"], shortest_paths)


def obtain_apls_samples(**props):
    obtain_metric_samples("apls", **props)


def obtain_topo_samples(**props):
    obtain_metric_samples("topo", **props)


def obtain_metric_samples(metric, threshold = 30, fusion_only = False, _variants = None, inverse = False, sample_count = 500, metric_threshold = None, prime = False, extend = True, covered_injection_only = False): # APLS
    """Pregenerate samples, so its easier to experiment with taking different sample countes etcetera."""

    assert metric in metrics

    if metric_threshold == None:
        metric_threshold = 5

    metric_interval = metric_threshold

    if inverse:
        fusion_only = True
        
    if _variants == None:
        if fusion_only:
            _variants = set(get_fusion_variants(covered_injection_only)) - set(["osm"])
        else:
            _variants = set(base_variants + get_fusion_variants(covered_injection_only))
    else:
        _variants = set(_variants)

    prime_metric_samples = prime_apls_samples if metric == "apls" else prime_topo_samples

    # First check what we even have to compute.
    do_we_have_to_compute = False
    for place in places: 
        for variant in set(_variants):
            if do_we_have_to_compute:
                continue

            location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric=metric, metric_threshold=metric_threshold, metric_interval=metric_interval, prime_samples=prime)["metrics_samples"]
            if not extend or not path_exists(location):
                do_we_have_to_compute = True
                continue
            
            # Compute remaining samples.
            samples = read_pickle(location)
            remaining = sample_count - len(prime_metric_samples(samples)) if prime else sample_count - len(samples)
                
            if remaining > 0:
                # what_to_compute_for[variant] = remaining
                do_we_have_to_compute = True

    # if len(what_to_compute_for) == 0:
    if not do_we_have_to_compute:
        return 

    logger("Reading prepared maps.")
    maps = {}
    for place in places: 
        maps[place] = {}
        for variant in _variants.union(["osm"]):
            print(f"{place}-{variant}")
            location = experiment_location(place, variant, threshold=threshold, inverse=inverse)["prepared_graph"]
            maps[place][variant] = read_graph(location)
    
    if metric == "apls":
        logger("Reading shortest paths maps.")
        shortest_paths = {}
        for place in places: 
            shortest_paths[place] = {}
            for variant in _variants.union(["osm"]):
                print(f"{place}-{variant}")
                location = experiment_location(place, variant, threshold=threshold, inverse=inverse)["apls_shortest_paths"]
                shortest_paths[place][variant] = read_pickle(location)

    logger("Computing samples.")
    for place in places: 
        for variant in set(_variants):
            logger(f"{place} - {variant}.")
            target_graph = maps[place]["osm"]
            source_graph = maps[place][variant]

            # Apply subgraphing for prime samples to focus on areas near target graph
            if prime and metric == "apls":
                logger(f"Applying subgraphing with distance {threshold}m for prime samples.")
                source_graph = extract_subgraph_by_graph(source_graph, target_graph, threshold)

            if metric == "apls":
                target_paths = shortest_paths[place]["osm"]
                source_paths = shortest_paths[place][variant]

            location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric=metric, metric_threshold=metric_threshold, metric_interval=metric_interval, prime_samples=prime)["metrics_samples"]

            print(f"* {location}")
            if extend and path_exists(location):
                samples = read_pickle(location)
                remaining = sample_count - len(samples)
                if prime:
                    remaining = sample_count - len(prime_metric_samples(samples))
                print(f"* {remaining}")
                if remaining <= 0:
                    continue

                if metric == "apls":
                    new_samples = apls_sampling(target_graph, source_graph, target_paths, source_paths, max_distance=metric_threshold, n=remaining, prime=prime)
                else:
                    new_samples = topo_sampling(target_graph, source_graph,  interval=metric_interval, hole_size=metric_threshold, n_measurement_nodes=remaining, prime=prime)

                samples += new_samples
            else:
                if metric == "apls":
                    samples = apls_sampling(target_graph, source_graph, target_paths, source_paths, max_distance=metric_threshold, n=sample_count, prime=prime)
                else:
                    samples = topo_sampling(target_graph, source_graph,  interval=metric_interval, hole_size=metric_threshold, n_measurement_nodes=sample_count, prime=prime)

            write_pickle(location, samples)