"""
Geoalg Experiments Module

This module serves as the main entry point for all experiments in the geoalg project.
It imports all experiment functions from the organized submodules for convenient access.

The experiments are organized by series:
- Series 0: Basic analysis and graph statistics
- Series 1: Performance analysis and comparison tables  
- Series 2: Threshold analysis and impact studies
- Series 3: Sampling analysis and distribution studies

Additional modules:
- Data preparation: Loading and preprocessing functions
- Visualization: Plotting and rendering functions
- Advanced analysis: Complex fusion and correlation experiments
- Utilities: Shared helper functions
"""

# Import all experiment functions from organized modules
from .experiments.data_preparation import *
from .experiments.series_0 import *
from .experiments.series_1 import *
from .experiments.series_2 import *
from .experiments.series_3 import *
from .experiments.visualization import *
from .experiments.advanced_analysis import *
from .experiments.utilities import *

# Re-export all functions for backward compatibility
__all__ = [
    # Data preparation functions
    'read_base_maps', 'read_fusion_maps', 'read_all_maps',
    'obtain_fusion_maps', 'obtain_prepared_metric_maps',
    'obtain_shortest_distance_dictionaries', 'obtain_metric_samples',
    'obtain_apls_samples', 'obtain_topo_samples', 'remove_deleted',
    
    # Series 0 experiments
    'experiment_zero_score_stabilization', 'experiment_zero_base_info_graph',
    'experiment_zero_edge_coverage_base_graphs', 'experiment_zero_graph_distances',
    
    # Series 1 experiments  
    'experiments_one_base_table',
    
    # Series 2 experiments
    'obtain_fusion_maps_range', 'obtain_threshold_data',
    'experiment_two_threshold_performance', 'experiment_two_threshold_impact_on_metadata',
    'experiment_two_basic_information',
    
    # Series 3 experiments
    'experiment_three_TOPO_hole_size', 'experiment_three_sample_distribution',
    'experiment_three_prime_sample_distribution',
    
    # Visualization functions
    'render_maps_to_images', 'plot_base_maps', 'plot_IDR_maps',
    'plot_IDR_maps_with_actions', 'plot_IDR_maps_with_actions_at_extremes',
    'render_map', 'plot_correlation_matrices', 'plot_comprehensive_covariance_matrices',
    
    # Advanced analysis functions
    'experiment_fusion_analysis',
    'experiment_continuation_performance_correlation_heatmap', 'get_performance_data_for_place', 
    'generate_fusion_typst_table', 'plot_continuation_performance_heatmap',
    
    # Utilities
    'get_experiment_metadata', 'validate_experiment_parameters', 'format_variant_name',
]