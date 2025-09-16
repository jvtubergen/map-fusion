from data_handling import *
from utilities import *
from data import *
from map_similarity import *
from graph import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .data_preparation import obtain_metric_samples


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