from data_handling import *
from utilities import *
from data import *
from map_similarity import *
from graph import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import inf


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