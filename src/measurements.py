from data_handling import *
from graph_merging import *
from graph_simplifying import *
from graph_deduplicating import *
from graph_curvature import * 

# Generate all maps related to thesis.
# TODO: Add split-point merging graph.
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
        graphs              = _read_and_or_write("graphs", lambda: merge_graphs(C=intersection, A=gps_vs_intersection, prune_threshold=threshold))

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
def compute_apls(truth, proposed):

    prepared_graph_data = {
        "left" : prepare_graph_data(truth, proposed),
        "right": prepare_graph_data(proposed, truth),
    }

    apls_score      , _ = apls(truth, proposed, prepared_graph_data=prepared_graph_data)
    apls_prime_score, _ = apls(truth, proposed, prepared_graph_data=prepared_graph_data, prime=True)

    return apls_score, apls_prime_score


# Copmute TOPO metric between two graphs.
def compute_topo(truth, proposed):

    def prepare_graph_for_topo(G):

        G = G.copy()

        G = simplify_graph(G)
        G = graph.to_directed(G)
        G = nx.MultiGraph(G)

        graph_annotate_edge_curvature(G)
        graph_annotate_edge_geometry(G)
        graph_annotate_edge_length(G)

        return G

    truth, proposal = prepare_graph_for_topo(truth), prepare_graph_for_topo(propsed)
    topo_score = topo(truth, proposed)
    topo_prime_score = topo(truth, proposed, prime=True)

    return topo_score, topo_prime_score


# Compute TOPO/APLS results on maps.
def measurements_maps(maps):

    measurements_results = {}

    for place in ["chicago", "berlin"]:

        measurements_results[place] = {}

        osm = maps[place]["osm"]
        truth = osm

        for map_variant in set(maps[place].keys()) - set(["osm"]):

            proposed = maps[place][map_variant]

            apls, apls_prime = compute_apls(truth, proposed)
            topo, topo_prime = compute_topo(truth, proposed)

            measurements_results[place][map_variant] = {
                "apls": apls,
                "apls_prime": apls_prime,
                "topo": topo,
                "topo_prime": topo_prime,
            }

    return measurements_results


# Construct typst table out of measurements data.
def measurements_to_table(measurements):
    todo("Implement measurements to table.")

