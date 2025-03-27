from data_handling import *
from graph_merging import *
from graph_simplifying import *
from graph_deduplicating import *
from graph_curvature import * 
from apls import *
from topo.topo_metric import compute_topo as compute_topo_on_prepared_graph

# Generate all maps related to thesis.
# TODO: Add split-point merging graph.
# * Allow to customize the coverage threshold.
# * Allow to instead of gps against sat to act on subselection of sat which is nearby gps edges (easier to look at merging effect).
@info()
def generate_maps(threshold = 30, debugging=False, **reading_props):

    maps = {}

    simp = simplify_graph
    dedup = graph_deduplicate
    to_utm = graph_transform_latlon_to_utm

    for place in ["chicago", "berlin"]: # First run chicago (that one goes faster, so earlier error detection).

        logger(f"{place}.")
        logger("Preparing input graphs (osm, sat, gps).")

        _read_and_or_write = lambda filename, action, **props: read_and_or_write(f"data/pickled/{place}-{filename}", action, **props)

        # Source graph.
        osm = _read_and_or_write("osm", lambda:simp(dedup(to_utm(read_graph(place=place, graphset=links["osm"])))), **reading_props)

        # Starting graphs.
        sat = _read_and_or_write("sat", lambda:simp(dedup(to_utm(read_graph(place=place, graphset=links["sat"])))), **reading_props)
        gps = _read_and_or_write("gps", lambda:simp(dedup(to_utm(read_graph(place=place, graphset=links["gps"])))), **reading_props)

        # If we are debugging on the merging logic.
        if debugging:
            # Then (it is convenient) to only act around sat edges nearby gps edges (where the action happens).

            logger("DEBUGGING: Pruning Sat graph for relevant edges concerning merging.")

            sat_vs_gps   = edge_graph_coverage(sat, gps, max_threshold=threshold)
            intersection = prune_coverage_graph(sat_vs_gps, prune_threshold=threshold)
            sat = intersection # (Update sat so we can continue further logic.)

        # Three merging graphs.
        logger(f"Generating merging graphs.")
        gps_vs_sat = edge_graph_coverage(gps, sat, max_threshold=threshold)
        graphs     = merge_graphs(C=sat, A=gps_vs_sat, prune_threshold=threshold, remove_duplicates=True, reconnect_after=True)

        maps[place] = {
            "osm": osm,
            "sat": sat,
            "gps": gps,
            "a": graphs["a"],
            "b": graphs["b"],
            "c": graphs["c"]
        }

    return maps    


# Copmute TOPO metric between two graphs.
@info()
def compute_apls(truth, proposed):

    prepared_graph_data = {
        "left" : prepare_graph_data(truth, proposed),
        "right": prepare_graph_data(proposed, truth),
    }

    apls_score      , _ = apls(truth, proposed, prepared_graph_data=prepared_graph_data)
    apls_prime_score, _ = apls(truth, proposed, prepared_graph_data=prepared_graph_data, prime=True)

    return apls_score, apls_prime_score


# Prepare graph for TOPO computations.
@info()
def prepare_graph_for_topo(G):

    if "prepared" in G.graph and G.graph["prepared"] == "topo":
        return G

    G = G.copy()

    G = simplify_graph(G)
    G = G.to_directed(G)
    G = nx.MultiGraph(G)

    graph_annotate_edge_curvature(G)
    graph_annotate_edge_geometry(G)
    graph_annotate_edge_length(G)

    G.graph["prepared"] = "topo"

    return G

# Copmute TOPO metric between two graphs.
@info()
def compute_topo(truth, proposed):

    truth, proposal = prepare_graph_for_topo(truth), prepare_graph_for_topo(proposed)
    topo_score = compute_topo_on_prepared_graph(truth, proposed)
    topo_prime_score = compute_topo_on_prepared_graph(truth, proposed, prime=True)

    return topo_score, topo_prime_score


# Precompute maps for measurements.
def precompute_measurements_maps(maps):

    result = {}

    for place in ["chicago", "berlin"]:

        result[place] = {}

        for map_variant in set(maps[place].keys()):

            logger(f"{place} - {map_variant}.")
            
            # Drop deleted edges before continuing.
            def remove_deleted(G):

                G = G.copy()

                edges_to_be_deleted = filter_eids_by_attribute(G, filter_attributes={"render": "deleted"})
                nodes_to_be_deleted = filter_nids_by_attribute(G, filter_attributes={"render": "deleted"})

                G.remove_edges_from(edges_to_be_deleted)
                G.remove_nodes_from(nodes_to_be_deleted)

                return G

            graph = maps[place][map_variant]
            graph = remove_deleted(graph)

            result[place][map_variant] = {
                "topo": prepare_graph_for_topo(graph),
                "apls": prepare_graph_for_apls(graph),
            }
    
    return result


# Compute TOPO/APLS results on maps.
@info(timer=True)
def apply_measurements_maps(prepared_maps, threshold=30):

    result = {}

    for place in ["chicago", "berlin"]:

        result[place] = {}

        truth_apls = prepared_maps[place]["osm"]["apls"]
        truth_topo = prepared_maps[place]["osm"]["topo"]

        check("prepared" in truth_apls.graph and truth_apls.graph["prepared"] == "apls", expect="Expect prepared truth graph when computing apls metric.")
        check("prepared" in truth_topo.graph and truth_topo.graph["prepared"] == "topo", expect="Expect prepared truth graph when computing topo metric.")

        for map_variant in set(prepared_maps[place].keys()) - set(["osm"]):

            logger(f"{place} - {map_variant}.")

            proposed_apls = prepared_maps[place][map_variant]["apls"]
            proposed_topo = prepared_maps[place][map_variant]["topo"]

            check("prepared" in proposed_apls.graph and proposed_apls.graph["prepared"] == "apls", expect="Expect prepared proposed graph when computing apls metric.")
            check("prepared" in proposed_topo.graph and proposed_topo.graph["prepared"] == "topo", expect="Expect prepared proposed graph when computing topo metric.")

            apls, apls_prime = compute_apls(truth_apls, proposed_apls)
            topo, topo_prime = compute_topo(truth_topo, proposed_topo)

            result[place][map_variant] = {
                "apls": apls,
                "apls_prime": apls_prime,
                "topo": topo,
                "topo_prime": topo_prime,
            }

    return result


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


# Measure TOPO, TOPO*, APLS, APLS* on Berlin and Chicago.
def measure_threshold_values(lowest = 1, highest = 50, step = 1):

    reading_props = {
        "is_graph": False,
        "overwrite_if_old": True,
        "reset_time": 7*24*60*60, # Keep it for a week.
    }

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

    # Generate threshold_maps for thresholds.
    # def compute_threshold_maps():
    threshold_maps = {}
    for threshold in range(lowest, highest, step):
        print(f"Generating map with threshold {threshold}.")
        maps = read_and_or_write(f"data/pickled/threshold_maps-{threshold}", lambda: generate_maps(threshold = threshold, **reading_props), **reading_props)
        threshold_maps[threshold] = {}
        threshold_maps[threshold]["berlin"]  = maps["berlin"]["c"]
        threshold_maps[threshold]["chicago"] = maps["chicago"]["c"]
    
    
    # Prepare graphs for TOPO and APLS.
    def precompute_graphs(threshold_maps):
        precomputed_graphs = {}
        for threshold in range(lowest, highest, step):
            print(f"Preparing graph for topo and apls ({threshold}).")
            precomputed_graphs[threshold] = {}
            precomputed_graphs[threshold]["berlin"]  = precompute_measurement_map(threshold_maps[threshold]["berlin"])
            precomputed_graphs[threshold]["chicago"] = precompute_measurement_map(threshold_maps[threshold]["chicago"])
        return precomputed_graphs

    precomputed_graphs = read_and_or_write(f"data/pickled/precomputed_graphs", lambda: precompute_graphs(threshold_maps), **reading_props)
    
    # Compute TOPO and APLS.
    def compute_metrics(precomputed_graphs):

        # We need prepared APLS and TOPO on Berlin and Chicago.
        # (Read out truth on Berlin and Chicago and preprocess.)
        simp = simplify_graph
        dedup = graph_deduplicate
        to_utm = graph_transform_latlon_to_utm
        osm_berlin = simp(dedup(to_utm(read_graph(place="berlin", graphset=links["osm"]))))
        osm_chicago = simp(dedup(to_utm(read_graph(place="chicago", graphset=links["osm"]))))
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
    
    measure_results = read_and_or_write(f"data/pickled/measure_results", lambda: compute_metrics(precomputed_graphs), **reading_props)

    return measure_results


# Plot threshold values on Berlin and Chicago.
def render_thresholds():

    reading_props = {
        "is_graph": False,
        "overwrite_if_old": True,
        "reset_time": 365*24*60*60, # Keep it for a year.
    }

    # Data format: `data[threshold][place][apls/topo]`
    # 
    # We want `threshold` on the x-axis, place and metric as different coloring/line style, value on the y-axis.
    measure_results = read_and_or_write(f"data/pickled/measure_results", lambda: compute_metrics(precomputed_graphs), **reading_props)

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
    colors = {
        "berlin_apls": "#1f77b4",       # blue
        "berlin_apls_prime": "#9467bd",  # purple
        "berlin_topo": "#2ca02c",        # green
        "berlin_topo_prime": "#d62728",  # red
        "chicago_apls": "#ff7f0e",       # orange
        "chicago_apls_prime": "#8c564b", # brown
        "chicago_topo": "#e377c2",       # pink
        "chicago_topo_prime": "#7f7f7f"  # gray
    }

    # Create combined category for legend
    df['place_metric'] = df['place'] + "_" + df['metric_type']

    # Line styles to differentiate further
    line_styles = {
        "berlin": "-",    # solid line
        "chicago": "--"   # dashed line
    }

    # Plot each place-metric combination
    for place in ["berlin", "chicago"]:
        for metric in ["apls", "apls_prime", "topo", "topo_prime"]:
            subset = df[(df["place"] == place) & (df["metric_type"] == metric)]
            place_metric = f"{place}_{metric}"
            
            # Sort by threshold to ensure correct line drawing
            subset = subset.sort_values("threshold")
            
            plt.plot(
                subset["threshold"], 
                subset["value"], 
                marker="o", 
                linestyle=line_styles[place], 
                color=colors[place_metric],
                label=f"{place.capitalize()} - {metric}", 
                alpha=0.9, 
                markersize=5
            )

    plt.title("Performance Metrics Across Thresholds", fontsize=16)
    plt.xlabel("Threshold", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="best", frameon=True, fancybox=True, shadow=True)

    # Add a horizontal line at y=0.5 for reference
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()


