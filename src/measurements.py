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


# Copmute TOPO metric between two graphs.
@info()
def compute_topo(truth, proposed):

    def prepare_graph_for_topo(G):

        G = G.copy()

        G = simplify_graph(G)
        G = G.to_directed(G)
        G = nx.MultiGraph(G)

        graph_annotate_edge_curvature(G)
        graph_annotate_edge_geometry(G)
        graph_annotate_edge_length(G)

        return G

    truth, proposal = prepare_graph_for_topo(truth), prepare_graph_for_topo(proposed)
    topo_score = compute_topo_on_prepared_graph(truth, proposed)
    topo_prime_score = compute_topo_on_prepared_graph(truth, proposed, prime=True)

    return topo_score, topo_prime_score


# Compute TOPO/APLS results on maps.
@info()
def measurements_maps(maps, threshold=30):

    measurements_results = {}

    for place in ["chicago", "berlin"]:

        measurements_results[place] = {}

        osm = maps[place]["osm"]
        truth = osm

        for map_variant in set(maps[place].keys()) - set(["osm"]):

            logger(f"{place} - {map_variant}.")

            _read_and_or_write = lambda filename, action, **props: read_and_or_write(f"data/pickled/{threshold}-{place}-{map_variant}-{filename}", action, **props)

            proposed = maps[place][map_variant]

            apls, apls_prime = _read_and_or_write("apls", lambda: compute_apls(truth, proposed), is_graph=False)
            topo, topo_prime = _read_and_or_write("topo", lambda: compute_topo(truth, proposed), is_graph=False)

            measurements_results[place][map_variant] = {
                "apls": apls,
                "apls_prime": apls_prime,
                "topo": topo,
                "topo_prime": topo_prime,
            }

    return measurements_results


# Construct typst table out of measurements data.
@info()
def measurements_to_table(measurements):
    todo("Implement measurements to table.")

