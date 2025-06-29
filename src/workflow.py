from external import *
from internal import *

# truth = constructing_osm_on_boundaries(place="berlin", roi=roi["berlin"])
# write_graph(truth, place="berlin", graphset="openstreetmaps")
def workflow_constructing_osm_on_boundaries(roi=None):

    west  = roi["west"]
    east  = roi["east"]
    south = roi["south"]
    north = roi["north"]
    G = ox.graph_from_bbox(north=north, south=south, west=west, east=east, network_type="drive", retain_all=False, simplify=False)
    G = nx.Graph(G.to_undirected())

    return G

    

def workflow_render_gps_alongside_truth(place=None):

    gps = read_graph(get_graph_path(graphset="roadster", place=place))
    truth = read_graph(get_graph_path(graphset="openstreetmaps", place=place))
    plot_without_projection([(gps, {"color": (0,1,0,1)}), (truth, {"color": (0,0,1,1)})], []) # gps is green, truth is blue


def workflow_render_sat_alongside_truth(place=None):

    sat = read_graph(get_graph_path(graphset="sat2graph", place=place))
    truth = read_graph(get_graph_path(graphset="openstreetmaps", place=place))
    plot_without_projection([(sat, {"color": (0,1,0,1)}), (truth, {"color": (0,0,1,1)})], []) # sat is green, truth is blue


def workflow_render_sat_gps_truth(place=None, gps=True, sat=True, truth=True):
    print("Render sat, gps, and or truth graph.")

    graphs = []
    
    if sat:
        sat = read_graph(get_graph_path(graphset="sat2graph", place=place))
        graphs.append((sat, {"color": (0,1,0,1)})) # sat is green
    
    if gps:
        gps = read_graph(get_graph_path(graphset="roadster", place=place)) 
        graphs.append((gps, {"color": (0.7,0.1,0.9,1), "linewidth": 3})) # gps is purple
    
    if truth:
        truth = read_graph(get_graph_path(graphset="openstreetmaps", place=place))
        graphs.append((truth, {"color": (0.3,0.3,0.3,1), "linestyle": ":"})) # truth is black

    plot_without_projection(graphs, []) 




# results = workflow_apply_apls(place="chicago", truth_graphset="openstreetmaps", proposed_graphset="roadster")
def workflow_apply_apls(place=None, truth_graphset=None, proposed_graphset=None):

    truth = read_graph(get_graph_path(graphset=truth_graphset, place=place))
    proposed = read_graph(get_graph_path(graphset=proposed_graphset, place=place))
    results = apls(truth=graph_prepare_apls(truth), proposed=graph_prepare_apls(proposed))

    return results


# results = workflow_apply_topo(place="chicago", truth_graphset="openstreetmaps", proposed_graphset="roadster")
def workflow_apply_topo(place=None, truth_graphset=None, proposed_graphset=None):

    truth = read_graph(get_graph_path(graphset=truth_graphset, place=place))
    proposed = read_graph(get_graph_path(graphset=proposed_graphset, place=place))
    # result = topo(graph_prepare_apls(truth), graph_prepare_apls(proposed))

    subgraph_radius=150
    interval=30
    hole_size=5
    n_measurement_nodes=20
    x_coord='x'
    y_coord='y'
    allow_multi_hole=False,
    make_plots=False
    verbose=False
    topo(truth, proposed,
                    subgraph_radius=subgraph_radius,
                    interval=interval, hole_size=hole_size,
                    n_measurement_nodes=n_measurement_nodes,
                    x_coord=x_coord, y_coord=y_coord,
                    allow_multi_hole=allow_multi_hole,
                    make_plots=False, verbose=False)

    # tp, fp, fn, precision, recall, f1 = result
    return result
    

def workflow_apply_topo_prime():
    truth = read_graph(get_graph_path(graphset=truth_graphset, place=place))
    proposed = read_graph(get_graph_path(graphset=proposed_graphset, place=place))
    result = topo_prime(graph_prepare_apls(truth), graph_prepare_apls(proposed))
    # tp, fp, fn, precision, recall, f1 = result
    return result


def workflow_report_apls_and_prime(place=None):

    sat_vs_osm = workflow_apply_apls(place=place, truth_graphset="openstreetmaps", proposed_graphset="sat2graph")
    gps_vs_osm = workflow_apply_apls(place=place, truth_graphset="openstreetmaps", proposed_graphset="roadster")
    sat_vs_gps = workflow_apply_apls(place=place, truth_graphset="sat2graph", proposed_graphset="roadster")

    print("sat vs osm:")
    print("APLS : ", sat_vs_osm[0])
    print("APLS+: ", sat_vs_osm[1])

    print("gps vs osm:")
    print("APLS : ", gps_vs_osm[0])
    print("APLS+: ", gps_vs_osm[1])

    print("sat vs gps:")
    print("APLS : ", sat_vs_gps[0])
    print("APLS+: ", sat_vs_gps[1])


def workflow_table_basic():
    t0 = time()

    result = []

    for place in ["berlin", "chicago"]:
    
        # Place.
        gps = read_graph(get_graph_path(graphset=links["gps"], place="berlin"))
        osm = read_graph(get_graph_path(graphset=links["osm"], place="berlin"))
        sat = read_graph(get_graph_path(graphset=links["sat"], place="berlin"))

        osm = graph_prepare_apls(osm)
        sat = graph_prepare_apls(sat)
        gps = graph_prepare_apls(gps)


        # Place sat.
        _, _, _, precision, recall, f1 = topo(osm, sat)
        result.extend([precision, recall, f1])
        result.append(apls(osm, sat))

        _, _, _, precision, recall, f1 = topo_prime(osm, sat)
        result.extend([precision, recall, f1])
        result.append(apls_prime(osm, sat))

        # Place gps.
        _, _, _, precision, recall, f1 = topo(osm, gps)
        result.extend([precision, recall, f1])
        result.append(apls(osm, gps))

        _, _, _, precision, recall, f1 = topo_prime(osm, gps)
        result.extend([precision, recall, f1])
        result.append(apls_prime(osm, gps))

    # Print results.
    string = ""
    for i, value in enumerate(result):
        if i % 8 == 0 and i > 0:
            string += "\n"
        string += f"[{value:.3f}],"

    print("Time run:", time() - t0)
    print(string)


def workflow_apply_apls_prime(place=None, truth_graphset=None, proposed_graphset=None):

    truth = read_graph(get_graph_path(graphset=truth_graphset, place=place))
    proposed = read_graph(get_graph_path(graphset=proposed_graphset, place=place))

    return apls_prime(truth=graph_prepare_apls(truth), proposed=graph_prepare_apls(proposed))


# workflow_edge_coverage_by_threshold(place="chicago", left_graphset="gps", right_graphset="sat")
def workflow_edge_coverage_by_threshold(place=None, left_graphset=None, right_graphset=None):

    left  = read_graph(get_graph_path(graphset=links[left_graphset], place=place))
    right = read_graph(get_graph_path(graphset=links[right_graphset], place=place))

    print("left:", left)
    print("right:", right )

    # Edges in left graph which are not covered by right graph.
    covered_left = edge_wise_coverage_threshold(left, right, max_threshold=100)
    # Edges in right graph which are not covered by left graph.
    covered_right = edge_wise_coverage_threshold(right, left, max_threshold=100)
    
    return covered_left, covered_right


def workflow_construct_coverage_by_threshold(place=None):

    gps_vs_osm, osm_vs_gps = workflow_edge_coverage_by_threshold(place="chicago", left_graphset="gps", right_graphset="osm")
    pickle.dump(gps_vs_osm, open(f"data/coverage/{place}_gps_vs_osm.pkl", "wb"))
    pickle.dump(osm_vs_gps, open(f"data/coverage/{place}_osm_vs_gps.pkl", "wb"))

    sat_vs_osm, osm_vs_sat = workflow_edge_coverage_by_threshold(place="chicago", left_graphset="sat", right_graphset="osm")
    pickle.dump(sat_vs_osm, open(f"data/coverage/{place}_sat_vs_osm.pkl", "wb"))
    pickle.dump(osm_vs_sat, open(f"data/coverage/{place}_osm_vs_sat.pkl", "wb"))

    sat_vs_gps, gps_vs_sat = workflow_edge_coverage_by_threshold(place="chicago", left_graphset="sat", right_graphset="gps")
    pickle.dump(sat_vs_gps, open(f"data/coverage/{place}_sat_vs_gps.pkl", "wb"))
    pickle.dump(gps_vs_sat, open(f"data/coverage/{place}_gps_vs_sat.pkl", "wb"))


def workflow_apls_prime_outcomes_on_different_coverage_thresholds(place="chicago", left="gps", right="osm"):

    left_vs_right = pickle.load(open(f"data/coverage/{place}_{left}_vs_{right}.pkl", "rb"))
    left = read_graph(get_graph_path(graphset=links[left], place="chicago"))
    right = read_graph(get_graph_path(graphset=links[right], place="chicago"))

    for curr in range(10, 100, 10):
        subleft = subgraph_by_coverage_thresholds(left, left_vs_right, max_threshold=curr)
        result = apls(truth=graph_prepare_apls(right), proposed=graph_prepare_apls(subleft))
        print(f"threshold up to {curr}: ", result)
        # plot_graphs([subleft])



# Remove nodes and edges with `{"render": "deleted"}` attribute.
def remove_deleted(G):
    G = G.copy()
    edges_to_be_deleted = filter_eids_by_attribute(G, filter_attributes={"render": "deleted"})
    nodes_to_be_deleted = filter_nids_by_attribute(G, filter_attributes={"render": "deleted"})
    G.remove_edges_from(edges_to_be_deleted)
    G.remove_nodes_from(nodes_to_be_deleted)
    return G


# Sanity check various graph conversion functions.
def workflow_sanity_check_graph_conversion_functions():
    osm = read_graph(graphset=links['osm'], place='chicago')
    graph_sanity_check(osm)

    # Coordinate changes on vectorized graph.
    utm_info = graph_utm_info(osm) # UTM information necessary to revert back to latlon later on.
    osm = graph_transform_latlon_to_utm(osm)
    graph_sanity_check(osm)
    osm = graph_transform_utm_to_latlon(osm, "", **utm_info)
    graph_sanity_check(osm)

    # Coordinate changes on simplified graph.
    osm = simplify_graph(osm)
    graph_sanity_check(osm)
    osm = graph_transform_latlon_to_utm(osm)
    graph_sanity_check(osm)
    osm = graph_transform_utm_to_latlon(osm, "", **utm_info)
    graph_sanity_check(osm)