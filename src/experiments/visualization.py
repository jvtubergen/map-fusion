from data_handling import *
from utilities import *
from data import *
from map_similarity import *
from graph import *
from rendering import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


@info()
def render_maps_to_images():
    """Render each map as a high-quality image, so we can zoom in to sufficient detailed graph curvature."""

    for quality in ["low", "high"]:
        for place in maps.keys():
            for map_variant in maps[place]:
                logger(f"{quality} - {place} - {map_variant}.")
                graph = maps[place][map_variant]
                graph = apply_coloring(graph)
                render_graph(graph, f"results/{place}-{map_variant}-{quality}.png", quality=quality, title=f"{quality}-{place}-{map_variant}")


def plot_base_maps():
    """Plot base maps in presentation style."""
    for place in places:
        for variant in base_variants:
            filename = f"Experiments Qualitative - {place.title()} {variant.upper()}.png"
            location = f"images/{filename}"
            print(location)
            G = read_graph(data_location(place, variant)["graph_file"])
            plot_graph_presentation(G, location)


def plot_IDR_maps(threshold = 30):
    """
    Plot IDR maps in presentation style.
    
    Note: Has annotated edges of deleted removed from graph.
    """
    for place in places:
        for inverse in [False, True]:
            variant_name = "IDR_SAT" if not inverse else "IDR_GPS"
            filename = f"Experiments Qualitative - {place.title()} {variant_name}.png"
            location = f"images/{filename}"
            print(location)
            G = read_graph(experiment_location(place, "C", threshold=threshold, inverse=inverse)["prepared_graph"])
            plot_graph_presentation(G, location=location)


def plot_IDR_maps_with_actions(threshold = 30, for_zoomed = False, covered_injection_only = False, save = False):
    """
    Plot IDR maps in presentation style alongside actions (insertion, deletion, reconnection).
    """
    for place in places:
        for inverse in [False, True]:
            variant_name = "IDR" if not covered_injection_only else "I*DR"
            variant_name += "_SAT" if not inverse else "_GPS"
            variant_name += f"-{threshold}"
            filename = f"Experiments Qualitative - {place.title()} {variant_name} with actions.png"
            location = f"images/{filename}"
            print(location)
            fusion_name = "C" if not covered_injection_only else "C2"
            G = read_graph(data_location(place, fusion_name, threshold=threshold, inverse=inverse)["graph_file"])
            plot_graph_presentation(G, location=location, with_actions=True, for_zoomed=for_zoomed, save=save)


def plot_IDR_maps_with_actions_at_extremes(place = "berlin", low_threshold = 5, high_threshold = 50):
    """
    Plot IDR maps in presentation style alongside actions (insertion, deletion, reconnection).
    """
    for threshold in [low_threshold, high_threshold]:
        for inverse in [False, True]:
            variant_name = "IDR_SAT" if not inverse else "IDR_GPS"
            filename = f"Experiments Qualitative - extremes {place.title()} {variant_name} with actions at threshold {threshold}.png"
            location = f"images/{filename}"
            print(location)
            G = read_graph(data_location(place, "C", threshold=threshold, inverse=inverse)["graph_file"])
            plot_graph_presentation(G, location=location, with_actions=True)


def render_map(place = "berlin", variant = "gps", threshold = 30):
    """Render specific map. Use it to zoom in and export region of interest as an SVG."""
    return


def plot_correlation_matrices(all_correlation_matrices, threshold):
    """
    Plot correlation matrices using seaborn heatmaps.
    Creates a grid showing correlation matrices for each combination of:
    (place, continuation_type, metric_type, prime)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Set up the plot grid: 4 rows × 4 columns = 16 subplots
    # Rows: (Berlin_discontinuation, Berlin_continuation, Chicago_discontinuation, Chicago_continuation)
    # Cols: (TOPO, TOPO*, APLS, APLS*)
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle(f'Road Continuation Quality vs Performance Correlation Matrices (threshold={threshold}m)', fontsize=16)
    
    places = ["berlin", "chicago"] 
    cont_types = ["discontinuation", "continuation"]
    metrics = ["topo", "apls"]
    primes = [False, True]
    
    # Create row and column labels
    row_labels = []
    for place in places:
        for cont_type in cont_types:
            place_name = place.title()
            cont_name = "Discontinuation" if cont_type == "discontinuation" else "Continuation"
            row_labels.append(f"{place_name}\n{cont_name}")
    
    col_labels = []
    for metric in metrics:
        for prime in primes:
            metric_name = metric.upper()
            if prime:
                metric_name += "*"
            col_labels.append(metric_name)
    
    # Find global min/max for consistent color scaling
    all_corr_values = []
    for place in places:
        for cont_type in cont_types:
            for metric in metrics:
                for prime in primes:
                    if (place in all_correlation_matrices and 
                        cont_type in all_correlation_matrices[place] and
                        metric in all_correlation_matrices[place][cont_type] and
                        prime in all_correlation_matrices[place][cont_type][metric]):
                        
                        corr_matrix = all_correlation_matrices[place][cont_type][metric][prime]
                        # Get the off-diagonal correlation value
                        corr_val = corr_matrix.iloc[0, 1]  # road_cont_quality vs performance_change
                        if not np.isnan(corr_val):
                            all_corr_values.append(corr_val)
    
    if all_corr_values:
        vmin = min(all_corr_values)
        vmax = max(all_corr_values)
        # Make symmetric around 0 for better color interpretation
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = -1, 1
    
    # Plot each correlation matrix
    row_idx = 0
    for place in places:
        for cont_type in cont_types:
            col_idx = 0
            for metric in metrics:
                for prime in primes:
                    ax = axes[row_idx, col_idx]
                    
                    if (place in all_correlation_matrices and 
                        cont_type in all_correlation_matrices[place] and
                        metric in all_correlation_matrices[place][cont_type] and
                        prime in all_correlation_matrices[place][cont_type][metric]):
                        
                        corr_matrix = all_correlation_matrices[place][cont_type][metric][prime]
                        
                        # Create heatmap
                        sns.heatmap(corr_matrix, 
                                  annot=True, 
                                  fmt='.3f', 
                                  cmap='RdBu_r',
                                  center=0,
                                  vmin=vmin, 
                                  vmax=vmax,
                                  ax=ax,
                                  cbar=col_idx == 3,  # Only show colorbar on rightmost column
                                  square=True)
                        
                        # Customize labels
                        ax.set_xticklabels(['Road Cont.\nQuality', f'Δ {col_labels[col_idx]}'], rotation=0)
                        ax.set_yticklabels(['Road Cont.\nQuality', f'Δ {col_labels[col_idx]}'], rotation=0)
                        
                    else:
                        # No data available
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                        ax.set_xticks([])
                        ax.set_yticks([])
                    
                    # Set column titles only on top row
                    if row_idx == 0:
                        ax.set_title(col_labels[col_idx], pad=10)
                    
                    col_idx += 1
            
            # Set row titles only on leftmost column
            if col_idx == 4:  # After processing all columns
                axes[row_idx, 0].set_ylabel(row_labels[row_idx], rotation=90, labelpad=20)
            
            row_idx += 1
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle
    plt.show()


def plot_comprehensive_covariance_matrices(comprehensive_matrices, threshold):
    """
    Plot comprehensive 6×6 covariance matrices using seaborn heatmaps.
    Creates one heatmap per place showing all pairwise relationships between:
    - road_cont_discontinuation, road_cont_continuation
    - topo_change, topo_prime_change, apls_change, apls_prime_change
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Set up the plot grid: 1 row × 2 columns = 2 subplots (Berlin, Chicago)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Comprehensive Road Continuation Quality vs Performance Covariance Matrices (threshold={threshold}m)', fontsize=16)
    
    places = ["berlin", "chicago"]
    
    # Custom labels for better readability
    metric_labels = [
        "Road Cont.\nDiscontinuation",
        "Road Cont.\nContinuation", 
        "Δ TOPO",
        "Δ TOPO*",
        "Δ APLS", 
        "Δ APLS*"
    ]
    
    # Find global min/max for consistent color scaling across both places
    all_cov_values = []
    for place in places:
        if place in comprehensive_matrices:
            cov_matrix = comprehensive_matrices[place]
            # Get all values except diagonal (since diagonal is always high)
            mask = ~np.eye(cov_matrix.shape[0], dtype=bool)
            all_cov_values.extend(cov_matrix.values[mask])
    
    if all_cov_values:
        vmin = np.min(all_cov_values)
        vmax = np.max(all_cov_values)
        # Make symmetric around 0 for better color interpretation
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = -1, 1
    
    # Plot each comprehensive covariance matrix
    for idx, place in enumerate(places):
        ax = axes[idx]
        
        if place in comprehensive_matrices:
            cov_matrix = comprehensive_matrices[place]
            
            # Create heatmap
            sns.heatmap(cov_matrix, 
                      annot=True, 
                      fmt='.4f', 
                      cmap='RdBu_r',
                      center=0,
                      vmin=vmin, 
                      vmax=vmax,
                      ax=ax,
                      cbar=idx == 1,  # Only show colorbar on rightmost plot
                      square=True,
                      xticklabels=metric_labels,
                      yticklabels=metric_labels)
            
            # Customize labels and title
            ax.set_title(place.title(), pad=20, fontsize=14)
            ax.set_xlabel('Metrics', fontsize=12)
            ax.set_ylabel('Metrics', fontsize=12)
            
            # Rotate labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            
        else:
            # No data available
            ax.text(0.5, 0.5, f'No Data Available\nfor {place.title()}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(place.title(), pad=20, fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Make room for suptitle
    plt.show()