from data_handling import *
from utilities import *
from data import *
from map_similarity import *
from graph import *
from .data_preparation import (
    obtain_shortest_distance_dictionaries,
    obtain_apls_samples,
    obtain_topo_samples
)


def experiments_one_base_table(place, threshold = 30, sample_count = 10000, prime_sample_count = 2000, covered_injection_only = False, metric_threshold = None, metric_interval = None):
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

    if metric_interval == None:
        metric_interval = metric_threshold

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
            location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric="apls", metric_threshold=metric_threshold, metric_interval=metric_interval)["metrics_samples"]
            samples = read_pickle(location)
            assert len(samples) >= sample_count
            
            # Load prime samples from the correct location
            prime_location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric="apls", metric_threshold=metric_threshold, metric_interval=metric_interval, prime_samples=True)["metrics_samples"]
            prime_samples = read_pickle(prime_location)
            assert len(prime_samples) >= prime_sample_count
            
            apls       = asymmetric_apls_from_samples(samples[:sample_count], prime=False)
            apls_prime = asymmetric_apls_from_samples(prime_samples[:prime_sample_count], prime=True)

            # Asymmetric TOPO results.
            location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric="topo", metric_threshold=metric_threshold, metric_interval=metric_interval)["metrics_samples"]
            samples = read_pickle(location)
            assert len(samples) >= sample_count
            
            # Load prime samples from the correct location
            prime_location = experiment_location(place, variant, threshold=threshold, inverse=inverse, metric="topo", metric_threshold=metric_threshold, metric_interval=metric_interval, prime_samples=True)["metrics_samples"]
            prime_samples = read_pickle(prime_location)
            assert len(prime_samples) >= prime_sample_count
            
            topo_results       = asymmetric_topo_from_samples(samples[:sample_count], False)
            topo_prime_results = asymmetric_topo_from_samples(prime_samples[:prime_sample_count], True)

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