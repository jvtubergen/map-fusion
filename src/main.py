from workflow import *

# Experiment 0: General information, edge coverage, and graph distance on base graphs.
# experiment_zero_edge_coverage_base_graphs()
# experiment_zero_graph_distances(sample_size=5000)


# Experiment 1: Prepare map information for future experiments + Obtain map similarity table.
# for inverse in [False, True]:
#     obtain_fusion_maps(inverse=inverse)
#     obtain_prepared_metric_maps(inverse=True)
#     obtain_shortest_distance_dictionaries(inverse=inverse)
# experiments_one_base_table("berlin",  threshold=30, sample_count=10000, prime_sample_count=2000)
# experiments_one_base_table("chicago", threshold=30, sample_count=10000, prime_sample_count=2000)


# Experiment 2: Obtain threshold-related graphs.
# obtain_threshold_data(sample_count=50, prime_sample_count=10)
# obtain_threshold_data(sample_count=500, prime_sample_count=100)
# obtain_threshold_data(sample_count=2000, prime_sample_count=400)
# experiment_two_threshold_performance(sample_count=50, prime_sample_count=10, inverse=False)
# experiment_two_threshold_performance(sample_count=50, prime_sample_count=10, inverse=True)
# experiment_two_threshold_performance(sample_count=500, prime_sample_count=100, inverse=False)
# experiment_two_threshold_performance(sample_count=500, prime_sample_count=100, inverse=True)
# experiment_two_threshold_performance(sample_count=2000, prime_sample_count=400, inverse=False)
# experiment_two_threshold_performance(sample_count=2000, prime_sample_count=400, inverse=True)
# experiment_two_basic_information()
# experiment_two_threshold_impact_on_metadata()


# Experiment 3: Obtain graphs related to TOPO and APLS sampling options.
# experiment_three_TOPO_hole_size(sample_count=2000, prime_sample_count=1000)
# experiment_three_sample_distribution(sample_count=2000)
# experiment_three_prime_sample_distribution(prime_sample_count=2000)


# Qualitative: Obtain qualitative data for the map reconstruction effects of the proposed map fusion algorithm.
# plot_base_maps()
# plot_IDR_maps()
# plot_IDR_maps_with_actions()
# plot_IDR_maps_with_actions(for_zoomed=True)
# plot_IDR_maps_with_actions_at_extremes(place="berlin", low_threshold=1)
# plot_IDR_maps_with_actions_at_extremes(place="chicago", low_threshold=1)


# Re-run for experiments 1 t/m 3 for map fusion with the selective edge injection approach.
# obtain_fusion_maps_range(covered_injection_only=True)
# plot_IDR_maps_with_actions(threshold=1 , for_zoomed=False, covered_injection_only=True, save=True)
# plot_IDR_maps_with_actions(threshold=30, for_zoomed=False, covered_injection_only=True, save=True)
# plot_IDR_maps_with_actions(threshold=50, for_zoomed=False, covered_injection_only=True, save=True)
# plot_IDR_maps_with_actions(threshold=1 , for_zoomed=True, covered_injection_only=True)
# plot_IDR_maps_with_actions(threshold=30, for_zoomed=True, covered_injection_only=True)
# plot_IDR_maps_with_actions(threshold=50, for_zoomed=True, covered_injection_only=True)

# experiment_two_threshold_impact_on_metadata(covered_injection_only=True)

# for i in range(1, 51):
#     for inverse in [False, True]:
#         obtain_prepared_metric_maps(threshold = i, fusion_only = True, inverse = inverse, re_use = True, covered_injection_only = True)
# experiment_two_basic_information(covered_injection_only=True)

# experiments_one_base_table("berlin", threshold = 30, sample_count = 10000, prime_sample_count = 2000, covered_injection_only = True, metric_threshold = 5)
# experiments_one_base_table("chicago", threshold = 30, sample_count = 10000, prime_sample_count = 2000, covered_injection_only = True, metric_threshold = 5)
# experiments_one_base_table("berlin", threshold = 30, sample_count = 10000, prime_sample_count = 2000, covered_injection_only = True, metric_threshold = 10)
# experiments_one_base_table("chicago", threshold = 30, sample_count = 10000, prime_sample_count = 2000, covered_injection_only = True, metric_threshold = 10)

# experiment_fusion_analysis(threshold=30, selective_injection=False) 
# experiment_fusion_analysis(threshold=30, selective_injection=True)  

# experiment_continuation_performance_correlation_heatmap()
