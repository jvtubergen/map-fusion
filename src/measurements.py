from data_handling import *
from graph_merging import *
from graph_simplifying import *
from graph_deduplicating import *
from graph_curvature import * 
from apls import *
from topo.topo_metric import compute_topo as compute_topo_on_prepared_graph

# Generate all maps related to thesis.
# TODO: Add split-point merging graph.
@info()
def generate_maps(threshold = 30):

    maps = {}

    simp = simplify_graph
    dedup = graph_deduplicate
    to_utm = graph_transform_latlon_to_utm

    for place in ["chicago", "berlin"]: # First run chicago (that one goes faster, so earlier error detection).

        _read_and_or_write = lambda filename, action, **props: read_and_or_write(f"data/pickled/{threshold}-{place}-{filename}", action, **props)

        # Source graph.
        osm = _read_and_or_write("osm", lambda: simp(dedup(to_utm(read_graph(place=place, graphset=links["osm"])))))

        # Starting graphs.
        sat = _read_and_or_write("sat", lambda: simp(dedup(to_utm(read_graph(place=place, graphset=links["sat"])))))
        gps = _read_and_or_write("gps", lambda: simp(dedup(to_utm(read_graph(place=place, graphset=links["gps"])))))

        # Intermediate graph representation.
        logger("Constructing Sat-vs-GPS coverage graph.") # Start with satellite graph and per edge check coverage by GPS.
        sat_vs_gps   = _read_and_or_write("sat_vs_graph", lambda: edge_graph_coverage(sat, gps, max_threshold=threshold))
        logger("Pruning Sat-vs-GPS graph.") # Extract edges of sat which are covered by gps.
        intersection = _read_and_or_write("intersection", lambda: prune_coverage_graph(sat_vs_gps, prune_threshold=threshold))

        # Three merging graphs.
        gps_vs_intersection = _read_and_or_write("gps_vs_intersection", lambda: edge_graph_coverage(gps, intersection, max_threshold=threshold))
        graphs              = _read_and_or_write("graphs", lambda: merge_graphs(C=intersection, A=gps_vs_intersection, prune_threshold=threshold), is_graph=False)

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
            
            graph = maps[place][map_variant]

            result[place][map_variant] = {
                "original": graph,
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