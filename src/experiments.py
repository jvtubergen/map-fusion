from data_handling import *
from utilities import *
from data import *
from map_similarity import *
from graph import *
from rendering import * 


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


##################
### APLS + TOPO
##################


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


##################
### APLS
##################


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

def obtain_metric_samples(metric, threshold = 30, fusion_only = False, _variants = None, inverse = False, sample_count = 500, metric_threshold = None, metric_interval = None, prime = False, extend = True, covered_injection_only = False): # APLS
    """Pregenerate samples, so its easier to experiment with taking different sample countes etcetera."""

    assert metric in metrics

    if metric_threshold == None:
        if metric == "apls":
            metric_threshold = 5
        else:
            metric_threshold = 5.5

    if metric_interval == None:
        if metric == "topo":
            metric_interval = 5

    if inverse:
        fusion_only = True
        
    if _variants == None:
        if fusion_only:
            _variants = set(get_fusion_variants(covered_injection_only))
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

            location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric=metric, metric_threshold=metric_threshold, metric_interval=metric_interval)["metrics_samples"]
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
            if prime:
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
    

##################
### Experiments 0
##################


# Experiment TOPO and APLS score stabilization by sample count.
def experiment_zero_score_stabilization():
    """
    Experiment TOPO and APLS score stabilization by sample count.

    Starting at 1, going to 1000, generate resulting TOPO and APLS score on place - base variant.
    """

    start = 1
    end = 1000

    # Generate the data (as a DataFrame).
    rows = []
    for place in places:
        for variant in set(base_variants) - set(["osm"]):
            for metric in metrics:
                if metric == "apls":
                    metric_threshold = 5
                    metric_interval = None
                if metric == "topo":
                    metric_threshold = 5.5
                    metric_interval = 5
                location = experiment_location(place, variant, threshold=None, metric=metric, metric_threshold=metric_threshold, metric_interval=metric_interval)["metrics_samples"]
                samples = read_pickle(location)
                # for sample_count in range(100, 1001, 100):
                for sample_count in range(10, 1001, 10):
                # for sample_count in range(1, 1001):
                    if metric == "apls":
                        score = asymmetric_apls_from_samples(samples, prime=False, sample_count=sample_count)
                    else:
                        score = asymmetric_topo_from_samples(samples, prime=False, sample_count=sample_count)["f1"]
                    rows.append({
                        "place": place,
                        "variant": variant,
                        "metric": metric,
                        "score": score,
                        "sample_count": sample_count
                    })

    # Convert to dataframe.
    data = pd.DataFrame(rows)
    # * Render single graph with apls and topo in different colors.
    # subset = data[(data["place"] == "berlin") & (data["variant"] == "gps")]
    # subset = subset.pivot(index="sample_count", columns="metric", values="score")
    # sns.lineplot(data=subset)
    # plt.show()

    # Generate plot.
    plt.figure(figsize=(30, 30))
    # sns.set_style("whitegrid")
    sns.set_theme(style="ticks")
    g = sns.FacetGrid(data, col = "place", row = "variant", hue="metric", margin_titles=True)
    g.map(sns.lineplot, "sample_count", "score")
    g.set_axis_labels("Sample size", "Score")
    g.add_legend()
    plt.show()


def experiment_zero_base_info_graph():
    """Obtain basic information on base graphs:
    - Number of nodes.
    - Average node degree.
    - Number of edges.
    - Average edge length.
    """
    logger("Reading prepared maps.")
    maps = {}
    for place in places: 
        maps[place] = {}
        for variant in base_variants:
            print(f"* {place}-{variant}")
            location = experiment_location(place, variant)["prepared_graph"]
            maps[place][variant] = read_graph(location)
    
    logger("Generate base information.")
    data = {}
    for place in places: 
        data[place] = {}
        for variant in base_variants:
            print(f"* {place}-{variant}")
            G = maps[place][variant]
            obj = {}
            obj["nr of nodes"] = len(G.nodes)
            obj["nr of edges"] = len(G.edges)
            obj["avg node deg"] = sum([G.degree[nid] for nid, _ in iterate_nodes(G)]) / len(G.nodes)
            obj["total length"] = sum([attrs["length"] for _, attrs in iterate_edges(G)])
            obj["avg edge len"] = obj["total length"] / obj["nr of edges"]
            data[place][variant] = obj
    
    print(data)


def experiment_zero_edge_coverage_base_graphs():
    """Compute edge coverage of base graphs."""
    logger("Reading prepared maps.")
    maps = {}
    for place in places: 
        maps[place] = {}
        for variant in base_variants:
            print(f"* {place}-{variant}")
            location = experiment_location(place, variant)["prepared_graph"]
            maps[place][variant] = read_graph(location)
    
    max_thresh = 500
    logger("Generate edge coverage information.")
    data = {}
    for place in places: 
        data[place] = {}
        for target in base_variants:
            data[place][target] = {}
            T = vectorize_graph(maps[place][target])
            for source in base_variants:
                print(f"* {place}-{target}-{source}")
                S = maps[place][source]
                S = edge_graph_coverage(S, T, max_threshold=max_thresh)
                thresholds = {i: 0 for i in range(1, max_thresh + 1)}
                thresholds[inf] = 0
                # Group number of edges per threshold.
                for eid, attrs in iterate_edges(S):
                    thresholds[attrs["threshold"]] += 1

                data[place][target][source] = thresholds
    
    # Convert into dataframe.
    rows = []
    for place in data:
        for target in base_variants:
            for source in base_variants:
                for i in range(1, max_thresh + 1):
                    rows.append({
                        "place": place,
                        "target": target,
                        "source": source,
                        "threshold": i,
                        "amount": data[place][target][source][i]
                    })
                rows.append({
                    "place": place,
                    "target": target,
                    "source": source,
                    "threshold": inf,
                    "amount": data[place][target][source][inf]
                })
        
    df = pd.DataFrame(rows)

    for place in places:

        subset = df[(df["place"] == place) & (df["amount"] != inf)] 

        # Single histplot.
        # subset = df[(df["place"] == place) & (df["target"] == "osm") & (df["source"] == "sat")]
        # sns.histplot(data=subset, x="threshold", weights="amount")

        # Generate Facetgrid.
        plt.figure(figsize=(30, 30))
        sns.set_theme(style="whitegrid")
        # ylim_dict = {(source, target): sum(subset[(subset["source"] == source) & (subset["target"] == target)]["amount"]) for source in base_variants for target in base_variants}
        ylim_dict = {source: sum(subset[(subset["source"] == source) & (subset["target"] == target)]["amount"]) for source in base_variants for target in base_variants}
        g = sns.displot(data=subset, x="threshold", weights="amount", col="target", kind="ecdf", row="source", log_scale=(10, None), stat="count", facet_kws={"margin_titles": True, "sharey":False})
        g.set(xticks=[1, 5, 10, 50, 100, 500])
        g.set_xticklabels([1, 5, 10, 50, 100, 500])
        
        # Add dashed horizontal line at ylim for each subplot
        for (row_var, col_var), ax in g.axes_dict.items():
            current_ylim = ax.get_ylim()
            limit_value = ylim_dict[row_var]
            # Set ylim first to accommodate the line
            ax.set_ylim(current_ylim[0], max(current_ylim[1], limit_value * 1.1))
            ax.axhline(y=limit_value, color='red', linestyle='--', alpha=0.7)
        
        plt.show()
    
    return df


def experiment_zero_graph_distances(sample_size = 10000):
    """
    For every pair of base graph generate kernel density on sample distance.
    It acts as additional information on top of APLS/TOPO.
    """
    logger("Reading prepared maps.")
    maps = {}
    for place in places: 
        maps[place] = {}
        for variant in base_variants:
            print(f"* {place}-{variant}")
            location = experiment_location(place, variant)["prepared_graph"]
            maps[place][variant] = read_graph(location)
    
    logger("Generate distance samples.")
    data = {}
    for place in places: 
        data[place] = {}
        for target in base_variants:
            data[place][target] = {}
            T = maps[place][target]
            for source in base_variants:
                print(f"* {place}-{target}-{source}")
                S = maps[place][source]
                distances = []
                while len(distances) < sample_size:
                    print(f"* {len(distances)}")
                    samples    = generate_sample_pairs(T, S, 100)
                    distances += [sample_pair_distance(sample) for sample in samples]
                data[place][target][source] = distances
    
    # Construct dataframe.
    rows = []
    for place in places:
        for target in base_variants:
            for source in base_variants:
                for distance in data[place][target][source]:
                    rows.append({
                        "place" : place,
                        "target": target,
                        "source": source,
                        "distance": distance,
                    })
    df = pd.DataFrame(rows)
        
    # Plot data.
    for place in places:

        subset = df[(df["place"] == place)] 

        # Generate Facetgrid.
        plt.figure(figsize=(30, 30))
        sns.set_theme(style="whitegrid")
        g = sns.displot(data=subset, x="distance", col="target", kind="kde", row="source", facet_kws={"margin_titles": True, "sharey": False, "sharex": False})
        g.set(xscale='log')
        
        plt.show()
    
    return df
    

##################
### Experiments 1
##################


def experiments_one_base_table(place, threshold = 30, sample_count = 10000, prime_sample_count = 2000, covered_injection_only = False, metric_threshold = None):
    """Table list SAT, GPS,  A, B, C, A^-1, B^-1, C^-1."""

    # Precompute shortest path dictionaries first
    obtain_shortest_distance_dictionaries(threshold=threshold, covered_injection_only=covered_injection_only)
    obtain_shortest_distance_dictionaries(threshold=threshold, inverse=True, covered_injection_only=covered_injection_only)

    obtain_apls_samples(metric_threshold=metric_threshold, threshold=threshold, sample_count=sample_count      , extend=True, prime=False, covered_injection_only=covered_injection_only)
    obtain_apls_samples(metric_threshold=metric_threshold, threshold=threshold, sample_count=prime_sample_count, extend=True, prime=True , covered_injection_only=covered_injection_only)
    obtain_apls_samples(metric_threshold=metric_threshold, threshold=threshold, sample_count=sample_count      , extend=True, prime=False, inverse=True, covered_injection_only=covered_injection_only)
    obtain_apls_samples(metric_threshold=metric_threshold, threshold=threshold, sample_count=prime_sample_count, extend=True, prime=True , inverse=True, covered_injection_only=covered_injection_only)
                                                        
    obtain_topo_samples(metric_threshold=metric_threshold, threshold=threshold, sample_count=sample_count      , extend=True, prime=False, covered_injection_only=covered_injection_only)
    obtain_topo_samples(metric_threshold=metric_threshold, threshold=threshold, sample_count=prime_sample_count, extend=True, prime=True , covered_injection_only=covered_injection_only)
    obtain_topo_samples(metric_threshold=metric_threshold, threshold=threshold, sample_count=sample_count      , extend=True, prime=False, inverse=True, covered_injection_only=covered_injection_only)
    obtain_topo_samples(metric_threshold=metric_threshold, threshold=threshold, sample_count=prime_sample_count, extend=True, prime=True , inverse=True, covered_injection_only=covered_injection_only)

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
            metric_threshold = 5
            metric_interval = None
            location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric="apls", metric_threshold=metric_threshold, metric_interval=metric_interval)["metrics_samples"]
            samples = read_pickle(location)
            assert len(samples) >= sample_count
            assert len(prime_apls_samples(samples)) >= prime_sample_count
            apls       = asymmetric_apls_from_samples(samples[:sample_count], prime=False)
            apls_prime = asymmetric_apls_from_samples(prime_apls_samples(samples)[:prime_sample_count], prime=True)

            # Asymmetric TOPO results.
            metric_threshold = 5.5
            metric_interval = 5
            location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric="topo", metric_threshold=metric_threshold, metric_interval=metric_interval)["metrics_samples"]
            samples = read_pickle(location)
            assert len(samples) >= sample_count
            assert len(prime_topo_samples(samples)) >= prime_sample_count
            topo_results       = asymmetric_topo_from_samples(samples[:sample_count], False)
            topo_prime_results = asymmetric_topo_from_samples(prime_topo_samples(samples)[:prime_sample_count], True)

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

    @info()
    def measurements_to_table(measurements):
        """Construct typst table out of measurements data."""
        
        before = """
        #show table.cell.where(y: 0): strong
        #set table(
        stroke: (x, y) => 
            if y == 0 {
            if x == 5 { ( bottom: 0.7pt + black, right: 0.7pt + black) }
            else if x == 6 { ( bottom: 0.7pt + black, left: 0.7pt + black) }
            else { ( bottom: 0.7pt + black)}
            } else if x == 5 {
            ( right: 0.7pt + black)
            } else if x == 6 {
            ( left: 0.7pt + black)
            },
        align: (x, y) => (
            if x > 0 { center }
            else { left }
        ),
        column-gutter: (auto, auto, auto, auto, auto, 2.2pt, auto)
        )

        #let pat = pattern(size: (30pt, 30pt))[
        #place(line(start: (0%, 0%), end: (100%, 100%)))
        #place(line(start: (0%, 100%), end: (100%, 0%)))
        ]

        #figure(
        rect(
        width: 375pt,
        table(
        columns: 10,
        table.header(
            [],
            [],
            [Rec],
            [Prec],
            [TOPO],
            [APLS],
            [Rec#super[$star$]],
            [Prec#super[$star$]],
            [TOPO#super[$star$]],
            [APLS#super[$star$]],
        ),
        table.cell(
            rowspan: 8,
            align: horizon,
            [*"""+place+"""*]
        ),
        """

        between = """
        table.hline(
            stroke: (
            paint: luma(100),
            dash: "dotted"
            ),
            start: 1,
            end: 6
        ),
        table.hline(
            stroke: (
            paint: luma(100),
            dash: "dotted"
            ),
            start: 6,
            end:10 
        ),
        """

        after = """
        )),
        caption: [Experiment 1 - Base table.],
        ) <table:experiment-1-base-table-"""+place+""">
        """

        data = []
        fusion_vars = get_fusion_variants(covered_injection_only)
        variant_list = [
            ("sat", False),
            ("gps", False),
        ]
        # Add fusion variants with both inverse states
        for variant in fusion_vars:
            variant_list.append((variant, False))
        for variant in fusion_vars:
            variant_list.append((variant, True))
        
        for variant, inverse in variant_list:

            if inverse and variant in base_variants:
                continue

            row = []
            row.append(measurements[variant][inverse]["topo"]["recall"])
            row.append(measurements[variant][inverse]["topo"]["precision"])
            row.append(measurements[variant][inverse]["topo"]["f1"])
            row.append(measurements[variant][inverse]["apls"])

            row.append(measurements[variant][inverse]["topo_prime"]["recall"])
            row.append(measurements[variant][inverse]["topo_prime"]["precision"])
            row.append(measurements[variant][inverse]["topo_prime"]["f1"])
            row.append(measurements[variant][inverse]["apls_prime"])
        
            variant_name = f'$bold("{variant}")$'
            if inverse:
                variant_name = f'$bold("{variant}")^(-1)$'

            data.append((variant_name, row))
    
        print(before)

        # Print results.
        for i, rows in enumerate(data):
            print(f"[{rows[0].upper()}], ", end="")
            for row in rows[1]:
                print(f"[{row:.3f}], ", end="")
            if i == 1 or i == 4:
                print(between)
            print()

        print(after)

    measurements_to_table(table_results)


##################
### Experiments 2
##################

def obtain_fusion_maps_range(lowest = 1, highest = 50, step = 1, values = None, include_inverse = True, covered_injection_only = False):
    """Construct fusion maps on different thresholds."""
    elements = values if values != None else range(lowest, highest + step, step)
    for i in elements:
        print("Threshold: ", i)
        for metric in metrics:
            for inverse in [False, True] if include_inverse else [False]:
                obtain_fusion_maps(threshold=i, inverse=inverse, covered_injection_only=covered_injection_only)

def obtain_threshold_data(lowest = 1, highest = 50, step = 1, values = None, sample_count = 5000, prime_sample_count = 1000, include_inverse = True, covered_injection_only = False):
    """Construct fusion maps on different thresholds and compute samples."""
    elements = values if values != None else range(lowest, highest + step, step)
    for i in elements:
        print("Threshold", i)
        for metric in metrics:
            for inverse in [False, True] if include_inverse else [False]:
                obtain_fusion_maps(threshold=i, inverse=inverse, covered_injection_only=covered_injection_only)
                obtain_prepared_metric_maps(threshold=i, fusion_only=True, inverse=inverse, covered_injection_only=covered_injection_only)
                obtain_shortest_distance_dictionaries(threshold=i, fusion_only=True, inverse=inverse, covered_injection_only=covered_injection_only)
                for prime in [False, True]:
                    count = prime_sample_count if prime else sample_count
                    obtain_metric_samples(metric, threshold=i, sample_count=count, extend=True, fusion_only=True, prime=prime, inverse=inverse, covered_injection_only=covered_injection_only)

        
def experiment_two_threshold_performance(lowest = 1, highest = 50, step = 1, values = None, sample_count = 5000, prime_sample_count = 1000, inverse = False, covered_injection_only = False):
    """
    Measure TOPO/TOPO* and APLS/APLS* on Berlin and Chicago for different map fusion threshold values.

    Renders plot for default (SAT base graph) or inverse (GPS base graph), but not both.
    """
    elements = values if values != None else range(lowest, highest + step, step)

    # Read in TOPO and APLS samples and compute metric scores.
    rows = []
    for place in places:
        selected_fusion_variants = set(get_fusion_variants(covered_injection_only))
        for variant in selected_fusion_variants:
            for prime in [False, True]:
                for threshold in elements:
                    print(f"{place}-{variant}-{inverse}-{prime}-{threshold}")
        
                    # Asymmetric APLS results.
                    metric_threshold = 5
                    metric_interval = None
                    location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric="apls", metric_threshold=metric_threshold, metric_interval=metric_interval)["metrics_samples"]
                    samples = read_pickle(location)
                    assert len(samples) >= sample_count
                    assert len(prime_apls_samples(samples)) >= prime_sample_count
                    apls       = asymmetric_apls_from_samples(samples[:sample_count], prime=False)
                    apls_prime = asymmetric_apls_from_samples(prime_apls_samples(samples)[:prime_sample_count], prime=True)

                    # Asymmetric TOPO results.
                    metric_threshold = 5.5
                    metric_interval = 5
                    location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric="topo", metric_threshold=metric_threshold, metric_interval=metric_interval)["metrics_samples"]
                    samples = read_pickle(location)
                    assert len(samples) >= sample_count
                    assert len(prime_topo_samples(samples)) >= prime_sample_count
                    topo       = asymmetric_topo_from_samples(samples[:sample_count], False)
                    topo_prime = asymmetric_topo_from_samples(prime_topo_samples(samples)[:prime_sample_count], True)

                    data = [
                        { "metric": "apls", "prime" : False, "score" : apls },
                        { "metric": "apls", "prime" : True , "score" : apls_prime },
                        { "metric": "topo", "prime" : False, "score" : topo["f1"] },
                        { "metric": "topo", "prime" : True , "score" : topo_prime["f1"] },
                    ]

                    for element in data:
                        rows.append({
                            "place"    : place,
                            "variant"  : variant,
                            "inverse"  : inverse,
                            "threshold": threshold,
                            "metric"   : element["metric"],
                            "prime"    : element["prime"],
                            "score"    : element["score"],
                        })

    # Convert into dataframe.
    if covered_injection_only:
        if inverse:
            variant_mapping = {"A2": "$I_{GPS}^{cov}$", "B2": "$ID_{GPS}^{cov}$", "C2": "$IDR_{GPS}^{cov}$"}
            hue_order = ["$I_{GPS}^{cov}$", "$ID_{GPS}^{cov}$", "$IDR_{GPS}^{cov}$"]
        else:
            variant_mapping = {"A2": "$I_{SAT}^{cov}$", "B2": "$ID_{SAT}^{cov}$", "C2": "$IDR_{SAT}^{cov}$"}
            hue_order = ["$I_{SAT}^{cov}$", "$ID_{SAT}^{cov}$", "$IDR_{SAT}^{cov}$"]
    else:
        variant_mapping = {"A": "$I_{GPS}$", "B": "$ID_{GPS}$", "C": "$IDR_{GPS}$"} if inverse else {"A": "$I_{SAT}$", "B": "$ID_{SAT}$", "C": "$IDR_{SAT}$"}
        hue_order       = ["$I_{GPS}$", "$ID_{GPS}$", "$IDR_{GPS}$"] if inverse else ["$I_{SAT}$", "$ID_{SAT}$", "$IDR_{SAT}$"]

    df = pd.DataFrame(rows)
    df['column'] = df[['metric', 'prime']].apply(lambda row: f"{f"{row.metric}*".upper()}" if row.prime else f"{row.metric}".upper(), axis=1)
    df['variant'] = df['variant'].map(variant_mapping)
    df['place'] = df['place'].map({"berlin": "Berlin", "chicago": "Chicago"})
    df = df[(df["inverse"] == inverse)]

    # for place in places:
    # subset = df[(df["place"] == place)]
    subset = df

    # Construct a FacetGrid lineplot on every (place, variant) combination
    plt.figure(figsize=(30, 30))
    sns.set_theme(style="ticks")
    sns.set_style("whitegrid")
    g = sns.FacetGrid(subset, col="column", row="place", hue="variant", hue_order=hue_order, margin_titles=True, palette="tab10")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.map(sns.lineplot, "threshold", "score")
    g.set_axis_labels("Threshold (m)", "Score")
    g.add_legend(title="")

    plt.show()


def experiment_two_threshold_impact_on_metadata(lowest = 1, highest = 50, step = 1, include_inverse = True, covered_injection_only = False):
    """Show injection, deletion, reconnection behavior on Berlin and Chicago for different map fusion threshold values."""

    # Get appropriate fusion variants based on covered_injection_only parameter
    fusion_vars = get_fusion_variants(covered_injection_only)
    a_variant = fusion_vars[0]  # A or A2
    c_variant = fusion_vars[2]  # C or C2

    # Read metadata from graphs.
    rows = []
    for place in places:
        for threshold in range(lowest, highest + step, step):
            logger(f"Computing fusion metadata on {place}-{threshold}.")
            for inverse in [False, True] if include_inverse else [False]:
                a = read_graph(data_location(place, a_variant, threshold = threshold, inverse = inverse)["graph_file"])
                c = read_graph(data_location(place, c_variant, threshold = threshold, inverse = inverse)["graph_file"])

                injection     = len(filter_eids_by_attribute(c, filter_attributes={"render": "injected"}))
                deletion      = len(filter_eids_by_attribute(c, filter_attributes={"render": "deleted"}))
                reconnection = len(filter_eids_by_attribute(c, filter_attributes={"render": "connection"})) - len(filter_eids_by_attribute(a, filter_attributes={"render": "connection"})) # Note: Injection of step 1 also creates connection edges, ignore those from this metadata.

                if covered_injection_only:
                    name = "$I^*DR_{SAT}$" if not inverse else "$I^*DR_{GPS}$"
                else:
                    name = "$IDR_{SAT}$" if not inverse else "$IDR_{GPS}$"
                rows.append({ "place": place, "threshold": threshold, "inverse": inverse, "name": name, "action": "injection", "value": injection })
                rows.append({ "place": place, "threshold": threshold, "inverse": inverse, "name": name, "action": "deletion", "value": deletion })
                rows.append({ "place": place, "threshold": threshold, "inverse": inverse, "name": name, "action": "reconnection", "value": reconnection })

    # Convert into dataframe.
    df = pd.DataFrame(rows)

    if not include_inverse:
        # Exclude inverse from plot.
        df = df[(df["inverse"] == False)]

    df['place'] = df['place'].apply(lambda row: row.title())
    df['action'] = df['action'].apply(lambda row: row.title())

    subset = df

    # Construct a FacetGrid lineplot on every (place, variant) combination
    plt.figure(figsize=(30, 30))
    sns.set_theme(style="ticks")
    sns.set_style("whitegrid")
    if include_inverse:
        g = sns.FacetGrid(subset, row = "place", hue = "action", hue_order = ["Injection", "Deletion", "Reconnection"], margin_titles=True, col = "name")
    else:
        g = sns.FacetGrid(subset, col = "place", hue = "action", hue_order = ["Injection", "Deletion", "Reconnection"], margin_titles=True)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.map(sns.lineplot, "threshold", "value")
    g.set_axis_labels("Threshold (m)", "Amount")
    g.add_legend(title="")
    # g.set(xticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    g.set(xticks=[0, 10, 20, 30, 40, 50])

    plt.show()


def experiment_two_basic_information(lowest = 1, highest = 50, step = 1, include_inverse = True, covered_injection_only = False):
    """Number of nodes, edges, total length."""

    # Get appropriate fusion variants based on covered_injection_only parameter
    fusion_vars = get_fusion_variants(covered_injection_only)
    c_variant = fusion_vars[2]  # C or C2

    # Obtain base
    rows = []
    for place in places:
        for threshold in range(lowest, highest + step, step):
            for inverse in [False, True] if include_inverse else [False]:
                logger(f"Computing basic information on {place}-{threshold}-{inverse}.")
                G = read_graph(experiment_location(place, c_variant, threshold=threshold, inverse=inverse)["prepared_graph"])
                nodes = len(G.nodes)
                edges = len(G.edges)
                length = sum([attrs["length"] for _, attrs in iterate_edges(G)]) / 1000
                if covered_injection_only:
                    name = "$I^*DR_{SAT}$" if not inverse else "$I^*DR_{GPS}$"
                else:
                    name = "$IDR_{SAT}$" if not inverse else "$IDR_{GPS}$"
                rows.append({ "place": place, "threshold": threshold, "name": name, "item": "nodes" , "value": nodes })
                rows.append({ "place": place, "threshold": threshold, "name": name, "item": "edges" , "value": edges })
                rows.append({ "place": place, "threshold": threshold, "name": name, "item": "length", "value": length})
    
    # Add SAT and GPS graph complexity values as constant data points
    for place in places:
        for variant in base_variants:
            G = read_graph(experiment_location(place, variant)["prepared_graph"])
            nodes = len(G.nodes)
            edges = len(G.edges)
            length = sum([attrs["length"] for _, attrs in iterate_edges(G)]) / 1000
            name = variant.upper()
            for threshold in range(lowest, highest + step, step):
                rows.append({ "place": place, "threshold": threshold, "name": name, "item": "nodes" , "value": nodes  })
                rows.append({ "place": place, "threshold": threshold, "name": name, "item": "edges" , "value": edges  })
                rows.append({ "place": place, "threshold": threshold, "name": name, "item": "length", "value": length })

    # Convert into dataframe.
    df = pd.DataFrame(rows)
    df['place'] = df['place'].apply(lambda row: row.title())
    df['item'] = df['item'].map({"nodes": "Nodes", "edges": "Edges", "length": "Graph Length (km)"})
    subset = df

    # Construct a FacetGrid lineplot on every (place, variant) combination
    plt.figure(figsize=(30, 30))

    sns.set_theme(style="ticks")
    sns.set_style("whitegrid")
    g = sns.FacetGrid(subset, row = "place", hue = "item", hue_order = ["Nodes", "Edges", "Graph Length (km)"], margin_titles=True, col = "name")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.map(sns.lineplot, "threshold", "value")
    g.set_axis_labels("Threshold (m)", "Amount")
    g.add_legend(title="")
    g.set(xticks=[0, 10, 20, 30, 40, 50])

    plt.show()


##################
### Experiments 3
##################


def experiment_three_TOPO_hole_size(sample_count = 10000, prime_sample_count = 2000):
    """
    Plot TOPO score based on different hole sizes. 
    
    Show against IDR_SAT-10 IDR_SAT-30 IDR_SAT-50.
    Use 10000 TOPO samples, 2000 TOPO* samples.

    Experiment aims to show impact of different TOPO hole sizes on TOPO score.
    It therefore suffices to only compute this on the default.
    """
    metric_interval = 5
    rows = []
    for place in places:
        for threshold in [10, 30, 50]:
            for metric_threshold in [1, 3, 5.5, 10, 15, 20, 30, 40, 50, 65, 80, 100]:
                logger(f"Computing TOPO with hole size {metric_threshold} on {place}-{threshold}.")
                G = read_graph(experiment_location(place, "C", threshold=threshold, inverse=False)["prepared_graph"])

                # Asymmetric TOPO results.
                obtain_metric_samples("topo", threshold=threshold, _variants=["C"], sample_count=sample_count      , metric_threshold=metric_threshold, metric_interval=metric_interval, prime=False, extend=True)
                obtain_metric_samples("topo", threshold=threshold, _variants=["C"], sample_count=prime_sample_count, metric_threshold=metric_threshold, metric_interval=metric_interval, prime=True, extend=True)
                location = experiment_location(place, "C", threshold=threshold, inverse=False, metric="topo", metric_threshold=metric_threshold, metric_interval=metric_interval)["metrics_samples"]
                samples = read_pickle(location)
                check(len(samples) >= sample_count)
                check(len(prime_topo_samples(samples)) >= prime_sample_count)
                topo       = asymmetric_topo_from_samples(samples[:sample_count], False)["f1"]
                topo_prime = asymmetric_topo_from_samples(prime_topo_samples(samples)[:prime_sample_count], True)["f1"]

                rows.append({
                    "place": place,
                    "threshold": threshold,
                    "hole size": metric_threshold,
                    "prime": False,
                    "score": topo
                })

                rows.append({
                    "place": place,
                    "threshold": threshold,
                    "hole size": metric_threshold,
                    "prime": True,
                    "score": topo_prime
                })
    
    # Convert into dataframe.
    df = pd.DataFrame(rows)
    df['place'] = df['place'].apply(lambda row: row.title())
    df['prime']  = df['prime'].map({False: "TOPO", True: "TOPO*"})
    df['threshold']  = df['threshold'].map({10: "$IDR_{SAT}-10$", 30: "$IDR_{SAT}-30$", 50: "$IDR_{SAT}-50$"})
    subset = df

    # Construct a FacetGrid lineplot.
    plt.figure(figsize=(30, 30))
    sns.set_theme(style="ticks")
    sns.set_style("whitegrid")
    
    g = sns.FacetGrid(subset, col="prime", row="place", hue="threshold", hue_order=["$IDR_{SAT}-10$", "$IDR_{SAT}-30$", "$IDR_{SAT}-50$"], margin_titles=True)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.map(sns.lineplot, "hole size", "score")
    g.set_axis_labels("Hole Size (m)", "Score")
    g.add_legend(title="")
    # use logarithmic on hole size with x-ticks at 1, 5, 10, 20, 50 similarly to edge coverage empirical cumulative distribution matrix.
    g.set(xscale='log')
    g.set(xticks=[1, 5, 10, 20, 50, 100])
    g.set_xticklabels([1, 5, 10, 20, 50, 100])

    plt.show()


def experiment_three_sample_distribution(sample_count = 10000):
    """
    Sample distributions of TOPO and APLS of the variants SAT, GPS, IDR_SAT and IDR_GPS at the places Berlin and Chicago
    """
    #TODO: Inverse?
    threshold = 30
    inverse = False
    rows = []
    for place in places:
        for (variant, inverse) in [("gps", False), ("sat", False), ("C", False), ("C", True)]:
            print(f"{place}-{variant}-{inverse}-{threshold}")

            # Asymmetric APLS samples.
            metric_threshold = 5
            metric_interval = None
            location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric="apls", metric_threshold=metric_threshold, metric_interval=metric_interval)["metrics_samples"]
            samples = read_pickle(location)[:sample_count]
            for sample in samples:
                rows.append({
                    "place": place,
                    "metric": "apls",
                    "variant": variant if not inverse else f"{variant}-1",
                    "inverse": inverse,
                    "score": sample["score"]
                })

            # Asymmetric TOPO results.
            metric_threshold = 5.5
            metric_interval = 5
            location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric="topo", metric_threshold=metric_threshold, metric_interval=metric_interval)["metrics_samples"]
            samples = read_pickle(location)[:sample_count]
            for sample in samples:
                sample = compute_topo_sample(sample)
                rows.append({
                    "place": place,
                    "metric": "topo",
                    "variant": variant if not inverse else f"{variant}-1",
                    "inverse": inverse,
                    "score": sample["f1"]
                })
    
    df = pd.DataFrame(rows)
    df['place'] = df['place'].apply(lambda row: row.title())
    df['metric'] = df['metric'].apply(lambda row: row.upper())
    df['variant'] = df['variant'].map({"gps": "GPS", "sat": "SAT", "C": "$IDR_{SAT}$", "C-1": "$IDR_{GPS}$"})
    subset = df

    # Construct a FacetGrid density plot.
    plt.figure(figsize=(30, 30))
    sns.set_theme(style="ticks")
    sns.set_style("whitegrid")
    
    g = sns.displot(data=subset, x="score",  col="metric", kind="kde", common_norm=True, row="place", hue="variant", hue_order=["GPS", "SAT", "$IDR_{SAT}$", "$IDR_{GPS}$"], facet_kws={"margin_titles": True, "sharey":False})
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels("Score", "Density")
    g.add_legend(title="")

    plt.show()

def experiment_three_prime_sample_distribution(prime_sample_count = 2000):
    """
    Sample distributions of TOPO and APLS of the variants SAT, GPS, IDR_SAT and IDR_GPS at the places Berlin and Chicago
    """
    #TODO: Inverse?
    threshold = 30
    inverse = False
    rows = []
    for place in places:
        for (variant, inverse) in [("gps", False), ("sat", False), ("C", False), ("C", True)]:
            print(f"{place}-{variant}-{inverse}-{threshold}")

            # Asymmetric APLS samples.
            metric_threshold = 5
            metric_interval = None
            location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric="apls", metric_threshold=metric_threshold, metric_interval=metric_interval)["metrics_samples"]
            samples = read_pickle(location)
            prime_samples = prime_apls_samples(samples)[:prime_sample_count]
            for sample in prime_samples:
                rows.append({
                    "place": place,
                    "metric": "apls",
                    "variant": variant if not inverse else f"{variant}-1",
                    "inverse": inverse,
                    "score": sample["score"]
                })

            # Asymmetric TOPO results.
            metric_threshold = 5.5
            metric_interval = 5
            location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric="topo", metric_threshold=metric_threshold, metric_interval=metric_interval)["metrics_samples"]
            samples = read_pickle(location)
            prime_samples = prime_topo_samples(samples)[:prime_sample_count]
            for sample in prime_samples:
                sample = compute_topo_sample(sample)
                rows.append({
                    "place": place,
                    "metric": "topo",
                    "variant": variant if not inverse else f"{variant}-1",
                    "inverse": inverse,
                    "score": sample["f1"]
                })
    
    df = pd.DataFrame(rows)
    df['place'] = df['place'].apply(lambda row: row.title())
    df['metric'] = df['metric'].apply(lambda row: f"{row.upper()}*")
    df['variant'] = df['variant'].map({"gps": "GPS", "sat": "SAT", "C": "$IDR_{SAT}$", "C-1": "$IDR_{GPS}$"})
    subset = df

    # Construct a FacetGrid density plot.
    plt.figure(figsize=(30, 30))
    sns.set_theme(style="ticks")
    sns.set_style("whitegrid")
    
    g = sns.displot(data=subset, x="score",  col="metric", kind="kde", common_norm=True, row="place", hue="variant", hue_order=["GPS", "SAT", "$IDR_{SAT}$", "$IDR_{GPS}$"], facet_kws={"margin_titles": True, "sharey":False})
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels("Score", "Density")
    g.add_legend(title="")

    plt.show()

##################
### Qualitative
##################
    
@info()
def render_maps_to_images():
    """Render each map as a high-quality image, so we can zoom in to sufficient detailed graph curvature."""

    for quality in ["low", "high"]:
        for place in maps.keys():
            for map_variant in maps[place]:
                logger(f"{quality} - {place} - {map_variant}.")
                graph = maps[place][map_variant]
                graph = apply_coloring(graph)
                render_graph(graph, f"results/{place}-{map_variant}-{quality}.png", quality=quality, title=f"{quality}-{place}-{map_variant}")

def plot_base_maps():
    """Plot base maps in presentation style."""
    for place in places:
        for variant in base_variants:
            filename = f"Experiments Qualitative - {place.title()} {variant.upper()}.png"
            location = f"images/{filename}"
            print(location)
            G = read_graph(data_location(place, variant)["graph_file"])
            plot_graph_presentation(G, location)


def plot_IDR_maps(threshold = 30):
    """
    Plot IDR maps in presentation style.
    
    Note: Has annotated edges of deleted removed from graph.
    """
    for place in places:
        for inverse in [False, True]:
            variant_name = "IDR_SAT" if not inverse else "IDR_GPS"
            filename = f"Experiments Qualitative - {place.title()} {variant_name}.png"
            location = f"images/{filename}"
            print(location)
            G = read_graph(experiment_location(place, "C", threshold=threshold, inverse=inverse)["prepared_graph"])
            plot_graph_presentation(G, location=location)

def plot_IDR_maps_with_actions(threshold = 30, for_zoomed = False, covered_injection_only = False, save = False):
    """
    Plot IDR maps in presentation style alongside actions (insertion, deletion, reconnection).
    """
    for place in places:
        for inverse in [False, True]:
            variant_name = "IDR" if not covered_injection_only else "I*DR"
            variant_name += "_SAT" if not inverse else "_GPS"
            variant_name += f"-{threshold}"
            filename = f"Experiments Qualitative - {place.title()} {variant_name} with actions.png"
            location = f"images/{filename}"
            print(location)
            fusion_name = "C" if not covered_injection_only else "C2"
            G = read_graph(data_location(place, fusion_name, threshold=threshold, inverse=inverse)["graph_file"])
            plot_graph_presentation(G, location=location, with_actions=True, for_zoomed=for_zoomed, save=save)

def plot_IDR_maps_with_actions_at_extremes(place = "berlin", low_threshold = 5, high_threshold = 50):
    """
    Plot IDR maps in presentation style alongside actions (insertion, deletion, reconnection).
    """
    for threshold in [low_threshold, high_threshold]:
        for inverse in [False, True]:
            variant_name = "IDR_SAT" if not inverse else "IDR_GPS"
            filename = f"Experiments Qualitative - extremes {place.title()} {variant_name} with actions at threshold {threshold}.png"
            location = f"images/{filename}"
            print(location)
            G = read_graph(data_location(place, "C", threshold=threshold, inverse=inverse)["graph_file"])
            plot_graph_presentation(G, location=location, with_actions=True)


def render_map(place = "berlin", variant = "gps", threshold = 30):
    """Render specific map. Use it to zoom in and export region of interest as an SVG."""
    return