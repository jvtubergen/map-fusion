from storage import *
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
    obtain_shortest_distance_dictionaries,
    obtain_metric_samples
)


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