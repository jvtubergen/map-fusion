# Rewrite of https://github.com:CosmiQ/apls with additional new functionality.
from data_handling import *
from utilities import *
from graph import *


@info(timer=True)
def relate_control_points(G, H, max_distance=25):
    """
    Relate nodes of target graph G to edges of source graph H and mention offset in H edge (so we can compute distance of shortest path).
    Return a dictionary that links nodes of G to H (keys are nids of G, values are nid with distance offset of H it connects to).
    """

    G_to_H = {} # All nodes of G related to a node alongside offset of H.

    H_edge_rtree = graphedges_to_rtree(H)

    # Find (and optionally inject) node within H nearby every node of G.
    for G_nid, attrs in iterate_nodes(G):

        G_position = graphnode_position(G, G_nid)
        H_eid = nearest_edge_for_position(H, G_position, edge_tree=H_edge_rtree)
        H_nid = H_eid[0] # Take start value.
        H_curve = graphedge_curvature(H_eid)
        G_H_distance, interval = nearest_interval_on_curve_to_point(H_curve, G_position)
        H_length = curve_length(H_curve) * interval

        if G_H_distance <= max_distance:
            G_to_H[G_nid] = (H_nid, H_length)

    return G_to_H


@info(timer=True)
def apls_sampling(G, H, G_to_H, G_paths, H_paths, n=10000):
    """
    Obtain samples. 
    * G: Target graph
    * H: Source graph
    * G_to_H: Link graph node of G to nearest edge of H
    * G_paths: Shortest paths dictionary on graph G
    * H_paths: Shortest paths dictionary on graph H
    """ 

    all_nids       = set(G.nodes())
    all_nids_prime = set(G_to_H.keys())

    # Normal: Generate `n` samples with (start, end).
    normal_samples = set()
    while len(normal_samples) < min(n, len(all_nids)):
        start = random.select(all_nids, 1)
        end   = random.select(all_nids - start, 1)
        if end in G_paths[start]:
            normal_samples.add(sorted([start, end]))
    
    # Primal: Generate `n` samples with (start, end).
    primal_samples = set()
    while len(primal_samples) < min(n, len(all_nids_prime)):
        start = random.select(all_nids_prime, 1)
        end   = random.select(all_nids_prime - start, 1)
        if end in G_paths[start]:
            primal_samples.append(sorted([start, end]))

    ## Category A: No control point in the proposed graph.
    A_normal = set([(start, end) for (start, end) in normal_samples if start not in all_nids_prime or end not in all_nids_prime])
    A_primal = set()

    ## Category B: Control nodes exist and a path exists in the ground truth, but not in the proposed graph. 
    B_normal = set([(start, end) for (start, end) in normal_samples - A_normal if G_to_H[end] not in H_paths[G_to_H[start]]])
    B_primal = set([(start, end) for (start, end) in primal_samples - A_primal if G_to_H[end] not in H_paths[G_to_H[start]]])

    ## Category C: Both graphs have control points and a path between them.
    C_normal = normal_samples - A_normal.union(B_normal)
    C_primal = primal_samples - A_primal.union(B_primal)

    # Collect.
    samples_normal = {
        "A": A_normal,
        "B": B_normal,
        "C": C_normal,
    }

    samples_primal = {
        "A": A_primal,
        "B": B_primal,
        "C": C_primal,
    }

    return samples_normal, samples_primal


# Compute score on the data object.
def compute_score(samples, path_scores):
    sample_sum  = sum(path_scores)
    n = len(samples["B"]) + len(samples["C"]) + len(samples["A"])
    score = sample_sum / n
    return score


def asymmetric_apls(G, H, G_paths, H_paths, n=10000, threshold=25):
    """Compute asymmetric APLS that leverages the following precomputed data:
    * G has edges to 50m pregenerated.
    * H has edges to 50m pregenerated.
    * G_paths has shortest path distances between all nodes of G precomputed.
    * H_paths has shortest path distances between all nodes of H precomputed.
    * n provides the total number of samples to take (in case of prime this are 10000 primal samples).
    * threshold is control point distance of node of G to edge curvature of H.

    Uses an alternative G_to_H structure, which stores nearest edge instead of nearest node G
    """

    # Relate control points.
    G_to_H = relate_control_points(G, H, max_distance=threshold)

    # Obtain samples.
    samples_normal, samples_primal = apls_sampling(G, H, G_to_H, G_paths, H_paths, n=n)

    def sample_score(a, b):
        """
        * a: target path length
        * b: source path length
        """ 
        return 1 - min(abs(a - b) / a, 1)
    
    def total_score(samples, path_scores):
        n = len(samples["A"]) + len(samples["B"]) + len(samples["C"])
        sample_sum = sum(path_scores)
        return sample_sum / n

    # Obtain path scores.
    path_scores_normal = {}
    for (start, end) in samples_normal["C"]:
        target_path_distance = G_paths[start][end]
        source_path_distance = H_paths[G_to_H[start][0]][G_to_H[end][0]] + G_to_H[start][1] + G_to_H[end][1], # Include distance offset from H node.
        score = sample_score(target_path_distance, source_path_distance)
        path_scores_normal[(start, end)] = {
            "target": target_path_distance,
            "source": source_path_distance,
            "score" : score,
        }

    path_scores_primal = {}
    for (start, end) in samples_primal["C"]:
        target_path_distance = G_paths[start][end]
        source_path_distance = H_paths[G_to_H[start][0]][G_to_H[end][0]] + G_to_H[start][1] + G_to_H[end][1], # Include distance offset from H node.
        score = sample_score(target_path_distance, source_path_distance)
        path_scores_primal[(start, end)] = {
            "target": target_path_distance,
            "source": source_path_distance,
            "score" : score,
        }
    
    # Compute APLS score (accumulate all sample scores).
    apls_score       = sum([element["score"] for element in path_scores_normal]) / (len(samples["A"]) + len(samples["B"]) + len(samples["C"]))
    apls_prime_score = sum([element["score"] for element in path_scores_primal]) / (len(samples["A"]) + len(samples["B"]) + len(samples["C"]))

    # Compute metadata: 
    metadata = {
        "normal": {
            "samples": samples_normal,
            "path_scores": path_scores_normal,
        },
        "primal": {
            "samples": samples_primal,
            "path_scores": path_scores_primal,
        }
    }

    return apls_score, apls_prime_score, metadata


@info(timer=True)
def symmetric_apls(G, H, G_paths, H_paths, n=10000, threshold=25):
    left = asymmetric_apls(G, H, G_paths, H_paths, n=n, threshold=threshold)
    right= asymmetric_apls(H, G, G_paths, H_paths, n=n, threshold=threshold)

    apls_score = 0.5 * (left[0] + right[0])
    apls_prime_score = 0.5 * (left[1] + right[1])
    metadata = {
        "left": left[2],
        "right": right[2]
    }

    return apls_score, apls_prime_score, metadata