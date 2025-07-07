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


def read_fusion_maps(threshold = 30):
    """Read fusion maps of a specific threshold from disk."""
    maps = {}
    for place in places:
        maps[place] = {}
        maps[place]["A"]   = read_graph(data_location(place, "A", threshold = threshold)["graph_file"])
        maps[place]["B"]   = read_graph(data_location(place, "B", threshold = threshold)["graph_file"])
        maps[place]["C"]   = read_graph(data_location(place, "C", threshold = threshold)["graph_file"])
    return maps


def read_all_maps(threshold = 30):
    """Read fusion maps of a specific threshold alongside the ground truth and two base maps from disk."""
    maps = {}
    for place in places:
        maps[place] = {}
        maps[place]["osm"] = read_osm_graph(place)
        maps[place]["gps"] = read_gps_graph(place)
        maps[place]["sat"] = read_sat_graph(place)
        maps[place]["A"]   = read_graph(data_location(place, "A", threshold = threshold)["graph_file"])
        maps[place]["B"]   = read_graph(data_location(place, "B", threshold = threshold)["graph_file"])
        maps[place]["C"]   = read_graph(data_location(place, "C", threshold = threshold)["graph_file"])
    return maps


@info()
def obtain_fusion_maps(threshold = 30, debugging=False, **reading_props):
    """Obtain fusion maps and write them to disk."""

    for place in places: 

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

        logger(f"Apply map fusion.")
        gps_vs_sat = edge_graph_coverage(gps, sat, max_threshold=threshold)
        graphs     = map_fusion(C=sat, A=gps_vs_sat, prune_threshold=threshold, remove_duplicates=True, reconnect_after=True)

        A = graphs["a"]
        B = graphs["b"]
        C = graphs["c"]

        write_graph(data_location(place, "A", threshold = threshold)["graph_file"], A)
        write_graph(data_location(place, "B", threshold = threshold)["graph_file"], B)
        write_graph(data_location(place, "C", threshold = threshold)["graph_file"], C)


def obtain_prepared_graphs():
    """
    DEPRECATED: APLS computes directly on stored graph, the max edge length not necessary with updated APLS logic.
    Obtain prepared graphs (inject node every 50 meters).
    """
    for place in places: 
        for variant in variants:
            G = read_graph(data_location(place, variant, threshold = threshold)["graph_file"])
            G = remove_deleted(G)
            logger(f"Nodes before: {len(G.nodes())}")
            G = graph_ensure_max_edge_length(G, max_length=100)
            G = graph_annotate_edges(G)
            sanity_check_graph(G)
            logger(f"Nodes after: {len(G.nodes())}")
            write_graph(experiment_location(place, variant, threshold=threshold)["prepared_graph"], G)


def obtain_shortest_distance_dictionaries(threshold = 30):
    """
    Obtain shortest distance on a graph and write to disk.
    """
    for place in places: 
        for variant in variants:
            print(f"{place}-{variant}")
            shortest_paths = None
            G = read_graph(data_location(place, variant, threshold = threshold)["graph_file"])
            if variant in ["B","C"]:
                G = remove_deleted(G)
            shortest_paths = precompute_shortest_path_data(G)
            write_pickle(experiment_location(place, variant, threshold=threshold)["shortest_paths"], shortest_paths)
    

def remove_deleted(G):
    """Remove nodes and edges with `{"render": "deleted"}` attribute."""
    G = G.copy()
    edges_to_be_deleted = filter_eids_by_attribute(G, filter_attributes={"render": "deleted"})
    nodes_to_be_deleted = filter_nids_by_attribute(G, filter_attributes={"render": "deleted"})
    G.remove_edges_from(edges_to_be_deleted)
    G.remove_nodes_from(nodes_to_be_deleted)
    return G


def obtain_apls_score(threshold = 30):
    logger("Reading prepared maps.")
    maps = {}
    for place in places: 
        maps[place] = {}
        for variant in variants:
            print(f"{place}-{variant}")
            location = experiment_location(place, variant, threshold=threshold)["prepared_graph"]
            maps[place][variant] = read_graph(location)
            if variant in ["B","C"]:
                maps[place][variant] = remove_deleted(maps[place][variant])
    
    logger("Reading shortest paths maps.")
    shortest_paths = {}
    for place in places: 
        shortest_paths[place] = {}
        for variant in variants:
            print(f"{place}-{variant}")
            location = experiment_location(place, variant, threshold=threshold)["shortest_paths"]
            shortest_paths = read_pickle(location)

    logger("Computing APLS metric.")
    result = {}
    for place in places:

        result[place] = {}
        for variant in set(variants) - set(["osm"]):

            logger(f"{place} - {variant}.")
            target_graph = maps[place]["osm"]
            source_graph = maps[place][variant]

            apls, apls_prime = symmetric_apls(target_graph, source_graph)

            obj = {
                "apls": apls,
                "apls_prime": apls_prime,
            }

            location = experiment_location(place, variant, threshold=threshold, metric="apls")
            write_pickle(location, obj)


def obtain_topo_score(threshold = 30):
    # TODO
    return


# Experiment one.
# Measure TOPO, TOPO*, APLS, APLS* on Berlin and Chicago for all maps (OSM, SAT, GPS, A, B, C).
@info()
def experiment_one_base_table(threshold = 30):
    # TODO: Read in TOPO and APLS.


    # Construct typst table out of measurements data.
    @info()
    def measurements_to_table(measurements):
        
        # Construct a list of elements to print.
        data = {}
        for place in ["berlin", "chicago"]:

            rows = []

            for map_variant in set(measurements[place].keys()) - set(["osm"]):

                row = []
                row.append(measurements[place][map_variant]["topo"][1]["recall"])
                row.append(measurements[place][map_variant]["topo"][1]["precision"])
                row.append(measurements[place][map_variant]["topo"][1]["f1"])
                row.append(measurements[place][map_variant]["apls"])

                row.append(measurements[place][map_variant]["topo_prime"][1]["recall"])
                row.append(measurements[place][map_variant]["topo_prime"][1]["precision"])
                row.append(measurements[place][map_variant]["topo_prime"][1]["f1"])
                row.append(measurements[place][map_variant]["apls_prime"])
            
                rows.append((map_variant, row))
        
            data[place] = rows


        print(before)

        # TODO: Upper-case and correct order.
        # Print berlin results.
        for rows in data["berlin"]:
            print(f"[{rows[0]}], ", end="")
            for row in rows[1]:
                print(f"[{row:.3f}], ", end="")
            print()

        print(between)
        
        # Print chicago results.
        for rows in data["chicago"]:
            print(f"[{rows[0]}], ", end="")
            for row in rows[1]:
                print(f"[{row:.3f}], ", end="")
            print()

        print(after)


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

    #table(
    columns: 10,
    table.header(
        [],
        [],
        [Acc],
        [Prec],
        [$F_1$],
        [APLS],
        [Acc#super[$star$]],
        [Prec#super[$star$]],
        [$F_1$#super[$star$]],
        [APLS#super[$star$]],
    ),
    table.cell(
        rowspan: 2,
        align: horizon,
        [Berlin]
    ),
    """

    between = """
    table.hline(
        stroke: (
        paint: luma(100),
        dash: "dashed"
        ),
        start: 1,
        end: 6
    ),
    table.hline(
        stroke: (
        paint: luma(100),
        dash: "dashed"
        ),
        start: 6,
        end:10 
    ),
    table.cell(
        rowspan: 2,
        align: horizon,
        [Chicago]
    ),
    """

    after = """
    )
    """

    # maps         = read_and_or_write("results/Experiment 1 - maps 30m", lambda: generate_maps(threshold=threshold)  , **reading_props)
    # precomputed  = read_and_or_write("results/Experiment 1 - precomputed maps for metrics", lambda: precompute_measurements_maps(maps)  , **reading_props)
    # measurements = read_and_or_write("results/Experiment 1 - apply measurements to maps"  , lambda: apply_measurements_maps(precomputed), **reading_props)
    print(measurements_to_table(result))


# Experiment two - A.
# Measure TOPO, TOPO*, APLS, APLS* on Berlin and Chicago for different threshold values.
def experiment_two_measure_threshold_values(lowest = 1, highest = 51, step = 1):

    reading_props = {
        "is_graph": False,
        "overwrite_if_old": True,
        "reset_time": 365*24*60*60, # Keep it for a year.
    }

    # Generate threshold_maps for thresholds.
    def compute_threshold_maps():
        threshold_maps = {}
        for threshold in range(lowest, highest, step):
            print(f"Generating map with threshold {threshold}.")
            maps = read_and_or_write(f"data/pickled/threshold_maps-{threshold}", lambda: generate_maps(threshold = threshold, **reading_props), **reading_props)
            threshold_maps[threshold] = {}
            threshold_maps[threshold]["berlin"]  = maps["berlin"]["c"]
            threshold_maps[threshold]["chicago"] = maps["chicago"]["c"]
        return threshold_maps
    

    # Prepare graphs for TOPO and APLS.
    def precompute_graphs_for_metrics(threshold_maps):

        # Prepare map for TOPO and APLS computation.
        def precompute_measurement_map(graph):
            # Drop deleted edges before continuing.
            def remove_deleted(G):

                G = G.copy()

                edges_to_be_deleted = filter_eids_by_attribute(G, filter_attributes={"render": "deleted"})
                nodes_to_be_deleted = filter_nids_by_attribute(G, filter_attributes={"render": "deleted"})

                G.remove_edges_from(edges_to_be_deleted)
                G.remove_nodes_from(nodes_to_be_deleted)

                return G
        
            graph = remove_deleted(graph)
            return {
                "topo": prepare_graph_for_topo(graph),
                "apls": prepare_graph_for_apls(graph),
            }

        precomputed_graphs = {}
        for threshold in range(lowest, highest, step):
            print(f"Preparing graph for topo and apls ({threshold}).")
            precomputed_graphs[threshold] = {}
            precomputed_graphs[threshold]["berlin"]  = precompute_measurement_map(threshold_maps[threshold]["berlin"])
            precomputed_graphs[threshold]["chicago"] = precompute_measurement_map(threshold_maps[threshold]["chicago"])
        return precomputed_graphs


    # Compute TOPO and APLS on graphs.
    def compute_metrics(precomputed_graphs):

        # We need prepared APLS and TOPO on Berlin and Chicago.
        # (Read out truth on Berlin and Chicago and preprocess.)
        simp = simplify_graph
        dedup = graph_deduplicate
        to_utm = graph_transform_latlon_to_utm
        osm_berlin = simp(dedup(to_utm(read_graph(get_graph_path(graphset=links["osm"], place="berlin")))))
        osm_chicago = simp(dedup(to_utm(read_graph(get_graph_path(graphset=links["osm"], place="chicago")))))
        truth = {}
        truth["berlin"] = {}
        truth["berlin"]["apls"]  = prepare_graph_for_apls(osm_berlin)
        truth["berlin"]["topo"]  = prepare_graph_for_topo(osm_berlin)
        truth["chicago"] = {}
        truth["chicago"]["apls"] = prepare_graph_for_apls(osm_chicago)
        truth["chicago"]["topo"] = prepare_graph_for_topo(osm_chicago)

        result = {}
        result["berlin"] = {}
        result["chicago"] = {}
        # Compute similarity for every map.
        for threshold in range(lowest, highest, step):
            print(f"Computing metric for threshold {threshold}.")

            def compute_metric(threshold):
                metric_result = {}
                for place in ["berlin", "chicago"]:
            
                    # Compute apls 
                    truth_apls = truth[place]["apls"]
                    truth_topo = truth[place]["topo"]

                    proposed_apls = precomputed_graphs[threshold][place]["apls"]
                    proposed_topo = precomputed_graphs[threshold][place]["topo"]

                    apls, apls_prime = compute_apls(truth_apls, proposed_apls)
                    topo, topo_prime = compute_topo(truth_topo, proposed_topo)

                    metric_result[place] = {
                        "apls": apls,
                        "apls_prime": apls_prime,
                        "topo": topo,
                        "topo_prime": topo_prime,
                    }
                return metric_result
            
            result[threshold] = read_and_or_write(f"data/pickled/metric_result-{threshold}", lambda: compute_metric(threshold), **reading_props)
            
        
        return result
    

    # Plot threshold values on Berlin and Chicago.
    def render_thresholds(measure_results):

        # Data format: `data[threshold][place][apls/topo]`
        # 
        # We want `threshold` on the x-axis, place and metric as different coloring/line style, value on the y-axis.

        plt.figure(figsize=(14, 8))
        sns.set_style("whitegrid")
        
        data = {}
        for i in range(1, 50):
            data[i] = {}
            for place in ["berlin", "chicago"]:
                data[i][place] = {
                    "apls"      : float(measure_results[i][place]["apls"]),
                    "apls_prime": float(measure_results[i][place]["apls_prime"]),
                    "topo"      : float(measure_results[i][place]["topo"][0]),
                    "topo_prime": float(measure_results[i][place]["topo_prime"][0]),
                }
        
        # Convert to DataFrame
        rows = []
        for i in data:
            for place in data[i]:
                for metric_type, value in data[i][place].items():
                    rows.append({
                        "threshold": i,
                        "place": place,
                        "metric_type": metric_type,
                        "value": value
                    })
        
        df = pd.DataFrame(rows)


        # Define colors for better differentiation
        base_colors = {
            "berlin": "#1f77b4", 
            "chicago": "#d62728", 
        }

        # Define color variations for each metric type
        metric_variations = {
            "apls": 0,        # Base color (no adjustment)
            "topo": 0.15,     # Slightly darker
            "apls_prime": -0.15,  # Slightly lighter
            "topo_prime": -0.3    # Even lighter
        }

        # Create combined category for legend
        df['place_metric'] = df['place'] + "_" + df['metric_type']

        # Function to adjust color (make it lighter or darker)
        def adjust_color(hex_color, amount):
            """
            Adjust hex color by making it lighter or darker
            - amount < 0: lighter
            - amount > 0: darker
            """
            import colorsys
            import matplotlib.colors as mc
            
            # Convert hex to RGB
            rgb = mc.to_rgb(hex_color)
            # Convert RGB to HSV
            h, s, v = colorsys.rgb_to_hsv(*rgb)
            
            # Adjust brightness (value)
            if amount < 0:
                v = min(1, v * (1 - amount))  # Lighter
            else:
                v = max(0, v * (1 - amount))  # Darker
                
            # Convert back to RGB
            rgb = colorsys.hsv_to_rgb(h, s, v)
            # Convert back to hex
            return mc.rgb2hex(rgb)

        # Plot each place-metric combination
        for place in ["berlin", "chicago"]:
            for metric in ["apls", "apls_prime", "topo", "topo_prime"]:

                # Set line style based on metric
                if metric == 'apls':
                    linestyle = '-'  # solid
                elif metric == 'topo':
                    linestyle = '--'  # dashed
                elif metric == 'apls_prime':  # reconnection
                    linestyle = ':'  # dotted
                else:  # perhaps for another metric
                    linestyle = '-.'  # dash-dot

                if place == "berlin":
                    marker = "o"
                else:
                    marker = "x"
                
                # Adjust color based on metric type
                base_color = base_colors[place]
                variation = metric_variations[metric]
                adjusted_color = adjust_color(base_color, variation)

                subset = df[(df["place"] == place) & (df["metric_type"] == metric)]
                place_metric = f"{place}_{metric}"
                
                # Sort by threshold to ensure correct line drawing
                subset = subset.sort_values("threshold")
                
                plt.plot(
                    subset["threshold"], 
                    subset["value"], 
                    marker=marker, 
                    linestyle=linestyle,
                    color=adjusted_color,
                    label=f"{place.capitalize()} - {metric}", 
                    alpha=0.9, 
                    markersize=5
                )

        # plt.title("Performance Metrics Across Thresholds", fontsize=16)
        plt.xlabel("Threshold", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc="best", frameon=True, fancybox=True, shadow=True)

        # Add a horizontal line at y=0.5 for reference
        plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

        plt.tight_layout()
        # plt.show()
        plt.savefig("results/Experiment 2 - Thresholds metric results.svg")


    threshold_maps     = compute_threshold_maps()
    precomputed_graphs = read_and_or_write(f"data/pickled/precomputed_graphs", lambda: precompute_graphs_for_metrics(threshold_maps), **reading_props)
    measure_results    = read_and_or_write(f"data/pickled/measure_results", lambda: compute_metrics(precomputed_graphs), **reading_props)
    render_thresholds(measure_results)


# Experiment two - B.
# Render three thresholds to see impact of different threshold values.
def experiment_two_render_minimal_and_maximal_thresholds():

    reading_props = {
        "is_graph": False,
        "overwrite_if_old": True,
        "reset_time": 365*24*60*60, # Keep it for a year.
        # "reset_time": 60, # Keep it for a minute.
    }

    # Render chicago with threshold of 1 and 50 as svg.
    for threshold in [1, 25, 50]:
        maps = read_and_or_write(f"data/pickled/threshold_maps-{threshold}", lambda: generate_maps(threshold = threshold, **reading_props), **reading_props)
        fusion_map = maps["chicago"]["c"]
        render_graph_as_svg(fusion_map, f"results/Experiment 2 - fusion map threshold {threshold}m.svg")


# Experiment two - C.
# Compute metadata (injected, deleted, reconnected) on thresholds.
@info()
def experiment_two_fusion_metadata():

    reading_props = {
        "is_graph": False,
        "overwrite_if_old": True,
        "reset_time": 365*24*60*60, # Keep it for a year.
    }

    # Obtain metadata.
    data = {}
    for threshold in range(1, 51):
        data[threshold] = {}
        maps = pickle.load(open(f"data/pickled/threshold_maps-{threshold}.pkl", "rb"))
        for place in ["berlin", "chicago"]:
            logger(f"Computing fusion metadata on {place}-{threshold}.")
            # Compute metadata on map differences.        
            start = maps[place]["sat"]
            a = maps[place]["a"]
            b = maps[place]["b"]
            c = maps[place]["c"]

            injection     = len(filter_eids_by_attribute(c, filter_attributes={"render": "injected"}))
            deletion      = len(filter_eids_by_attribute(c, filter_attributes={"render": "deleted"}))
            reconnection = len(filter_eids_by_attribute(c, filter_attributes={"render": "connection"})) - len(filter_eids_by_attribute(a, filter_attributes={"render": "connection"})) # Note: Injection of step 1 also creates connection edges, ignore those from this metadata.

            metadata = {
                "injection": injection,
                "deletion": deletion,
                "reconnection": reconnection,
            }
            data[threshold][place] = metadata
    

    # Plot metadata.
    def render_metadata(data):

        # Convert the nested dictionary to a pandas DataFrame
        rows = []
        for threshold, places in data.items():
            for place, metrics in places.items():
                for metric, value in metrics.items():
                    rows.append({
                        "threshold": threshold,
                        "place": place,
                        "metric": metric,
                        "value": value
                    })

        df = pd.DataFrame(rows)

        # Set up the plot
        plt.figure(figsize=(14, 8))
        sns.set_style("whitegrid")

        # Define custom colors for better distinction between metrics
        custom_palette = {"injection": "#1f77b4", "deletion": "#d62728", "reconnection": "#2ca02c"}

        # Loop through each metric and place combination to plot with correct styling
        for metric in df['metric'].unique():
            for place in df['place'].unique():
                # Filter data for this metric and place
                subset = df[(df['metric'] == metric) & (df['place'] == place)]
                
                # Set line style based on metric
                if metric == 'injection':
                    linestyle = '-'  # solid
                elif metric == 'deletion':
                    linestyle = '--'  # dashed
                else:  # reconnection
                    linestyle = ':'  # dotted
                
                # Set marker based on place
                marker = 'o' if place == 'berlin' else 'x'
                
                # Plot this subset
                plt.plot(
                    subset['threshold'], 
                    subset['value'],
                    linestyle=linestyle,
                    marker=marker,
                    color=custom_palette[metric],
                    label=f"{place} - {metric}"
                )

        # Customize the plot
        # plt.title("Fusion metadata by threshold", fontsize=16)
        plt.xlabel("Threshold", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend(title="", loc="best", frameon=True)
        plt.grid(True, linestyle="--", alpha=0.7)

        # Add annotations for clarity
        # plt.annotate("Berlin: circles (o)\nChicago: crosses (x)\n\nInjection: Solid line\nDeletion: Dashed line\nReconnection: Dotted line", 
        #             xy=(0.02, 0.02), 
        #             xycoords="figure fraction",
        #             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

        # Set x-axis to show more tick marks
        plt.xticks(np.arange(0, 51, 5))

        # Show the plot
        plt.tight_layout()
        # plt.show()
        plt.savefig("results/Experiment 2 - Thresholds metadata.svg")
    
    render_metadata(data)


# Experiment three.
# Render TOPO and APLS samples on Berlin/Chicago on GPS/SAT/fused.
def experiment_three_sample_histogram():

    reading_props = {
        "is_graph": False,
        "overwrite_if_old": True,
        "reset_time": 365*24*60*60, # Keep it for a year.
        # "reset_time": 60, # Keep it for a minute.
    }

    # Compute APLS and TOPO samples on Berlin and Chicago for sat, gps, fused. 
    def compute_data_apls_topo():

        logger("Compute APLS and TOPO samples on Berlin and Chicago for sat, gps, fused.")
        threshold = 30

        simp = simplify_graph
        dedup = graph_deduplicate
        to_utm = graph_transform_latlon_to_utm
        osm_berlin = simp(dedup(to_utm(read_graph(get_graph_path(graphset=links["osm"], place="berlin")))))
        osm_chicago = simp(dedup(to_utm(read_graph(get_graph_path(graphset=links["osm"], place="chicago")))))
        truth = {}
        truth["berlin"] = {}
        truth["berlin"]["apls"]  = prepare_graph_for_apls(osm_berlin)
        truth["berlin"]["topo"]  = prepare_graph_for_topo(osm_berlin)
        truth["chicago"] = {}
        truth["chicago"]["apls"] = prepare_graph_for_apls(osm_chicago)
        truth["chicago"]["topo"] = prepare_graph_for_topo(osm_chicago)

        # Load Sat, GPS, merged.
        maps = read_and_or_write(f"data/pickled/threshold_maps-{threshold}", lambda: generate_maps(threshold = threshold, **reading_props), **reading_props)

        # Compute TOPO and APLS.
        result = {}
        for place in ["berlin", "chicago"]:
            result[place] = {}
            for maptype in ["sat", "gps", "a", "b", "c"]:
                logger(f"Computing TOPO and APLS on {place}-{maptype}.")
                result[place][maptype] = {}

                # Samples APLS (bins 0 to 100).
                apls_samples = apls(maps[place][maptype], maps[place]["osm"])[1]
                result[place][maptype]["apls"] = {i: 0 for i in range(101)}
                #. samples["A"] # No control point in the proposed graph.
                #. samples["B"] # Control nodes exist and a path exists in the ground truth, but not in the proposed graph.
                #. samples["C"] # Both graphs have control points and a path between them.
                #. A and B both move to zero.
                #. C moves to path_scores `[float(v) for v in data["left"]["path_scores"]]`
                result[place][maptype]["apls"][0] = len(apls_samples["left"]["samples"]["A"]) + len(apls_samples["left"]["samples"]["B"]) \
                                                    + len(apls_samples["right"]["samples"]["A"]) + len(apls_samples["right"]["samples"]["B"])
                for v in [floor(float(v) * 100) for v in apls_samples["left"]["path_scores"]]:
                    result[place][maptype]["apls"][v] += 1
                for v in [floor(float(v) * 100) for v in apls_samples["right"]["path_scores"]]:
                    result[place][maptype]["apls"][v] += 1

                # Samples TOPO (bins 0 to 100).
                truth = maps[place]["osm"]
                proposed = maps[place][maptype]
                truth, proposal = prepare_graph_for_topo(truth), prepare_graph_for_topo(proposed)
                topo_samples = compute_topo_on_prepared_graph(truth, proposed)[1]["samples"]
                result[place][maptype]["topo"] = {i: 0 for i in range(101)}
                for v in topo_samples:
                    result[place][maptype]["topo"][floor(100* v)] += 1
        
        return result
    
    result = read_and_or_write(f"data/pickled/experiment 3 - topo and apls bins", lambda: compute_data_apls_topo(), **reading_props)
    
    # Convert the nested dictionaries to a DataFrame for easier plotting
    def convert_to_dataframe():
        logger("Convert the nested dictionaries to a DataFrame for easier plotting")
        data_rows = []
        for place in result:
            for maptype in result[place]:
                for metric in result[place][maptype]:
                    for score, count in result[place][maptype][metric].items():
                        if count > 0:  # Only include non-zero counts
                            data_rows.append({
                                'place': place,
                                'maptype': maptype,
                                'metric': metric,
                                'score': score,
                                'count': count
                            })
        df = pd.DataFrame(data_rows)
        return df

    df = read_and_or_write(f"data/pickled/experiment 3 - topo and apls bins dataframe", lambda: convert_to_dataframe(), **reading_props)

    # Render dataframe as a KDE.
    def render_dataframe_KDE(df):
        # Create a single figure for the histogram
        plt.figure(figsize=(14, 10))
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)

        # Create a new column combining place and metric for better visualization
        df['place_metric'] = df['place'] + '_' + df['metric']
    
        # Create a mapping for display names (change 'c' to 'fused' for display)
        maptype_display = {
            'sat': 'sat',
            'gps': 'gps',
            'c': 'fused'
        }

        # Create a display column with the renamed values
        df['maptype_display'] = df['maptype'].map(maptype_display)

        # Create a color palette that distinguishes between different combinations
        # We'll use different color families for different maptypes
        maptype_colors = {
            'sat': 'Blues',
            'gps': 'Greens',
            'c': 'Reds'
        }

        # Define distinct styles for each place-metric combination
        place_metric_styles = {
            'berlin_apls': {'linestyle': '-', 'alpha': 0.9, 'linewidth': 3.5},
            'berlin_topo': {'linestyle': '-.', 'alpha': 0.9, 'linewidth': 3.0},
            'chicago_apls': {'linestyle': '--', 'alpha': 0.9, 'linewidth': 2.5},
            'chicago_topo': {'linestyle': ':', 'alpha': 0.9, 'linewidth': 3.0}
        }

        # Flatten the data - we need scores repeated by their count
        flat_data = []
        for _, row in df.iterrows():
            for _ in range(int(row['count'])):
                flat_data.append({
                    'place': row['place'],
                    'metric': row['metric'],
                    'maptype': row['maptype'],
                    'maptype_display': row['maptype_display'],  # Display maptype (for labels)
                    'place_metric': row['place_metric'],
                    'score': row['score'] / 100.0 # Normalize scores to 0-1 range.
                })

        flat_df = pd.DataFrame(flat_data)

        # Calculate sample counts for each category
        sample_counts = {}
        for maptype in flat_df['maptype'].unique():
            sample_counts[maptype] = {}
            for place in flat_df['place'].unique():
                sample_counts[maptype][place] = {}
                for metric in flat_df['metric'].unique():
                    count = len(flat_df[(flat_df['maptype'] == maptype) & 
                                    (flat_df['place'] == place) & 
                                    (flat_df['metric'] == metric)])
                    sample_counts[maptype][place][metric] = count

        # Get unique combinations
        unique_place_metrics = sorted(flat_df['place_metric'].unique())
        unique_maptypes = sorted(["gps", "sat", "c"])

        # Create the figure
        plt.figure(figsize=(16, 10))
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)

        # Pre-calculate all KDE curves to find the maximum density value for normalization
        max_density = 0
        kde_curves = {}

        for maptype in unique_maptypes:
            kde_curves[maptype] = {}
            for place_metric in unique_place_metrics:
                place, metric = place_metric.split('_')
                
                # Filter data for this combination
                combo_data = flat_df[(flat_df['maptype'] == maptype) & 
                                (flat_df['place'] == place) & 
                                (flat_df['metric'] == metric)]
                
                if len(combo_data) > 0:
                    # Calculate KDE manually
                    x = np.linspace(0, 1, 1000)
                    kde = stats.gaussian_kde(combo_data['score'])
                    y = kde(x)
                    
                    # Store the curve data
                    kde_curves[maptype][place_metric] = {'x': x, 'y': y}
                    
                    # Update the maximum density value
                    max_density = max(max_density, np.max(y))

        # Now plot the normalized KDE curves
        for maptype in unique_maptypes:
            # Get display name for this maptype
            maptype_disp = maptype_display[maptype]
            
            # Get base color map for this maptype
            cmap_name = maptype_colors.get(maptype, 'Greys')
            cmap = plt.cm.get_cmap(cmap_name)
            
            # For each maptype, use a specific base color from its colormap
            base_color_position = 0.7  # Position in the colormap (0-1)
            base_color = cmap(base_color_position)
            
            for place_metric in unique_place_metrics:
                # Extract place and metric
                place, metric = place_metric.split('_')
                
                if place_metric in kde_curves[maptype]:
                    # Get the pre-calculated curve data
                    x = kde_curves[maptype][place_metric]['x']
                    y = kde_curves[maptype][place_metric]['y'] / max_density  # Normalize to 0-1
                    
                    # Get style parameters for this place-metric combination
                    style = place_metric_styles[place_metric]
                    
                    # Plot the normalized KDE curve
                    plt.plot(
                        x, y,
                        color=base_color,
                        alpha=style['alpha'],
                        linestyle=style['linestyle'],
                        linewidth=style['linewidth'],
                        label=f"{maptype_disp} - {place} ({metric})"
                    )

        # Set titles and labels
        # plt.title('Score Distribution for Map Types: sat, gps, fused', fontsize=18)
        plt.xlabel('Metric Score (0-1)', fontsize=16)
        plt.ylabel('Normalized Density (0-1)', fontsize=16)  # Updated to indicate normalized density

        # Adjust axes to show full ranges
        plt.xlim(0, 1)
        plt.ylim(0, 1.05)  # Add a little space at the top

        # Set ticks to display in proper 0-1 format
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))

        # Add grid for better readability
        plt.grid(axis='both', alpha=0.3)

        # Create a custom legend with better organization
        handles, labels = plt.gca().get_legend_handles_labels()

        # Group the legend items by maptype
        by_maptype = {}
        for handle, label in zip(handles, labels):
            maptype = label.split(' - ')[0]
            if maptype not in by_maptype:
                by_maptype[maptype] = []
            by_maptype[maptype].append((handle, label))

        # Create a custom ordered legend
        legend_handles = []
        legend_labels = []
        for maptype in sorted(by_maptype.keys()):
            for handle, label in by_maptype[maptype]:
                legend_handles.append(handle)
                legend_labels.append(label)

        # Add an additional legend to explain place-metric line styles
        # First create custom line objects for the styles legend
        style_handles = []
        style_labels = []

        for place_metric, style in place_metric_styles.items():
            place, metric = place_metric.split('_')
            line = Line2D([0], [0], color='black', 
                        linestyle=style['linestyle'], 
                        linewidth=style['linewidth'],
                        alpha=0.8)
            style_handles.append(line)
            style_labels.append(f"{place} ({metric})")

        # Main legend for maptype combinations
        plt.legend(
            legend_handles,
            legend_labels,
            title="Map Type - Place (Metric)",
            loc="upper right",
            fontsize=10,
            ncol=1,
            framealpha=0.9
        )

        # Add a second legend for line styles (outside the plot area)
        # plt.figlegend(
        #     style_handles,
        #     style_labels,
        #     title="Line Styles",
        #     loc="lower center",
        #     ncol=4,
        #     fontsize=10,
        #     framealpha=0.9,
        #     bbox_to_anchor=(0.5, 0.02)
        # )

        # Add statistics table as text with normalized values (0-1 range)
        stats_text = "Statistics (Mean ± Std):\n"
        stats_rows = []

        for maptype in unique_maptypes:
            # Get display name for this maptype
            maptype_disp = maptype_display[maptype]
            
            for place in ["berlin", "chicago"]:
                for metric in ["apls", "topo"]:
                    combo_data = flat_df[(flat_df['maptype'] == maptype) & 
                                        (flat_df['place'] == place) & 
                                        (flat_df['metric'] == metric)]
                    
                    if len(combo_data) > 0:
                        mean = combo_data['score'].mean()
                        std = combo_data['score'].std()
                        stats_rows.append(f"{maptype_disp} - {place} ({metric}): {mean:.2f} ± {std:.2f}")

        # Sort stats for better readability
        stats_rows.sort()
        stats_text += "\n".join(stats_rows)

        # Add statistics text box for means and standard deviations
        plt.annotate(
            stats_text,
            xy=(0.01, 0.99),
            xycoords='axes fraction',
            fontsize=10,
            ha='left',
            va='top',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
        )

        # Create a second statistics box for sample counts
        sample_counts_text = "Sample Counts:\n"
        sample_rows = []

        for maptype in unique_maptypes:
            # Get display name for this maptype
            maptype_disp = maptype_display[maptype]
            
            for place in ["berlin", "chicago"]:
                for metric in ["apls", "topo"]:
                    count = sample_counts[maptype][place][metric]
                    sample_rows.append(f"{maptype_disp} - {place} ({metric}): {count:,d} samples")

        # Sort sample count rows for better readability
        sample_rows.sort()
        sample_counts_text += "\n".join(sample_rows)

        # Add sample counts text box
        plt.annotate(
            sample_counts_text,
            xy=(0.01, 0.60),  # Position below the first statistics box
            xycoords='axes fraction',
            fontsize=10,
            ha='left',
            va='top',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
        )

        # Add markers for mean values to make them stand out
        for maptype in unique_maptypes:
            # Get display name for stats label (not used in this loop but useful for consistency)
            maptype_disp = maptype_display[maptype]
            
            # Get base color for this maptype
            cmap_name = maptype_colors.get(maptype, 'Greys')
            cmap = plt.cm.get_cmap(cmap_name)
            base_color = cmap(0.7)
            
            for place in ["berlin", "chicago"]:
                for metric in ["apls", "topo"]:
                    combo_data = flat_df[(flat_df['maptype'] == maptype) & 
                                    (flat_df['place'] == place) & 
                                    (flat_df['metric'] == metric)]
                    
                    if len(combo_data) > 0:
                        mean = combo_data['score'].mean()
                        place_metric = f"{place}_{metric}"
                        style = place_metric_styles[place_metric]
                        
                        # Plot a vertical line at the mean
                        plt.axvline(
                            x=mean,
                            color=base_color,
                            linestyle=style['linestyle'],
                            alpha=0.5,
                            linewidth=style['linewidth'] * 0.8
                        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for the line style legend at the bottom
        # plt.show()
        plt.savefig("results/Experiment 3 - TOPO and APLS samples.svg")


    render_dataframe_KDE(df)


# Render each map as a high-quality image, so we can zoom in to sufficient detailed graph curvature.
@info()
def render_maps_to_images():

    for quality in ["low", "high"]:
        for place in maps.keys():
            for map_variant in maps[place]:
                logger(f"{quality} - {place} - {map_variant}.")
                graph = maps[place][map_variant]
                graph = apply_coloring(graph)
                render_graph(graph, f"results/{place}-{map_variant}-{quality}.png", quality=quality, title=f"{quality}-{place}-{map_variant}")
