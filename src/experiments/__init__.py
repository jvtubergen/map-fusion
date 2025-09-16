"""
Experiments package for geometric algorithm research.

This package contains organized experiment modules for analyzing map fusion performance,
similarity metrics, and various experimental analyses.

Modules:
- data_preparation: Data loading and preparation functions
- series_0: Basic stability and information experiments  
- series_1: Base performance table experiments
- series_2: Threshold impact analysis experiments
- series_3: TOPO hole size and distribution experiments
- visualization: Plotting and rendering functions
- advanced_analysis: Advanced fusion analysis and correlation studies
- utilities: Shared helper functions and utilities
"""

# Import key functions from each module for convenient access
from .data_preparation import (
    read_base_maps,
    read_fusion_maps, 
    read_all_maps,
    obtain_fusion_maps,
    obtain_prepared_metric_maps,
    obtain_shortest_distance_dictionaries,
    obtain_metric_samples,
    obtain_apls_samples,
    obtain_topo_samples,
)

from .series_0 import (
    experiment_zero_score_stabilization,
    experiment_zero_base_info_graph,
    experiment_zero_edge_coverage_base_graphs,
    experiment_zero_graph_distances,
)

from .series_1 import (
    experiments_one_base_table,
)

from .series_2 import (
    obtain_fusion_maps_range,
    obtain_threshold_data,
    experiment_two_threshold_performance,
    experiment_two_threshold_impact_on_metadata,
    experiment_two_basic_information,
)

from .series_3 import (
    experiment_three_TOPO_hole_size,
    experiment_three_sample_distribution,
    experiment_three_prime_sample_distribution,
)

from .visualization import (
    render_maps_to_images,
    plot_base_maps,
    plot_IDR_maps,
    plot_IDR_maps_with_actions,
    plot_IDR_maps_with_actions_at_extremes,
    render_map,
    plot_correlation_matrices,
    plot_comprehensive_covariance_matrices,
)

from .advanced_analysis import (
    experiment_unimodal_fusion_analysis,
    experiment_selective_injection_fusion_analysis,
    experiment_continuation_performance_correlation_heatmap,
    get_performance_data_for_place,
    generate_unimodal_fusion_typst_table,
    generate_selective_injection_fusion_typst_table,
    plot_continuation_performance_heatmap,
)

from .utilities import (
    get_experiment_metadata,
    validate_experiment_parameters,
    format_variant_name,
)

__version__ = "1.0.0"

__all__ = [
    # Data preparation
    "read_base_maps", "read_fusion_maps", "read_all_maps", 
    "obtain_fusion_maps", "obtain_prepared_metric_maps",
    "obtain_shortest_distance_dictionaries", "obtain_metric_samples",
    "obtain_apls_samples", "obtain_topo_samples",
    
    # Series 0 experiments
    "experiment_zero_score_stabilization", "experiment_zero_base_info_graph",
    "experiment_zero_edge_coverage_base_graphs", "experiment_zero_graph_distances",
    
    # Series 1 experiments  
    "experiments_one_base_table",
    
    # Series 2 experiments
    "obtain_fusion_maps_range", "obtain_threshold_data",
    "experiment_two_threshold_performance", "experiment_two_threshold_impact_on_metadata",
    "experiment_two_basic_information",
    
    # Series 3 experiments
    "experiment_three_TOPO_hole_size", "experiment_three_sample_distribution", 
    "experiment_three_prime_sample_distribution",
    
    # Visualization
    "render_maps_to_images", "plot_base_maps", "plot_IDR_maps",
    "plot_IDR_maps_with_actions", "plot_IDR_maps_with_actions_at_extremes", 
    "render_map", "plot_correlation_matrices", "plot_comprehensive_covariance_matrices",
    
    # Advanced analysis
    "experiment_unimodal_fusion_analysis", "experiment_selective_injection_fusion_analysis",
    "experiment_continuation_performance_correlation_heatmap", "get_performance_data_for_place", 
    "generate_unimodal_fusion_typst_table", "generate_selective_injection_fusion_typst_table",
    "plot_continuation_performance_heatmap",
    
    # Utilities
    "get_experiment_metadata", "validate_experiment_parameters", "format_variant_name",
]