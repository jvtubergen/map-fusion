# Rewrite of https://github.com:CosmiQ/apls with additional new functionality.
from data_handling import *
from utilities import *
from graph import *

@info(timer=True)
def precompute_shortest_path_data(G):
    """
    Compute shortest path data (between all pairs of start-end nodes within a selection of node identifiers).
    
    Storing distances as floats and constructing the distance matrix with v > u results in significantly better data
    sizes compared to running nx.shortest_path_lengths:
    ```txt
    55M	 vs 832M on shortest_paths/A-berlin-30.pkl
    57M	 vs 607M on shortest_paths/A-chicago-30.pkl
    55M	 vs 728M on shortest_paths/B-berlin-30.pkl
    57M	 vs 544M on shortest_paths/B-chicago-30.pkl
    56M	 vs 752M on shortest_paths/C-berlin-30.pkl
    62M	 vs 612M on shortest_paths/C-chicago-30.pkl
    20M	 vs 322M on shortest_paths/gps-berlin.pkl
    8.0K vs 140K on shortest_paths/gps-chicago.pkl
    28M	 vs 424M on shortest_paths/osm-berlin.pkl
    4.2M vs 73M  on shortest_paths/osm-chicago.pkl
    38M	 vs 638M on shortest_paths/sat-berlin.pkl
    57M	 vs 600M on shortest_paths/sat-chicago.pkl
    ```
    """

    nids = G.nodes()

    # Expect all edges to have edge length annotated.
    for eid, attrs in iterate_edges(G):
        assert attrs["length"] > 0

    # Compute distance matrix between these points.
    distance_matrix = {}
    for u in nids:

        # Compute all reachable points from this node.
        distances = nx.single_source_dijkstra_path_length(G, u, weight="length")

        # Filter out dictionary to only include end nodes which are:
        # * Have a node identifier higher than the control nid coming from (prevents duplicated checks in subsequent logic).
        filtered = {}
        for v in distances.keys():
            if v >= u:
                filtered[v] = float(distances[v])
        
        distance_matrix[u] = filtered

    return distance_matrix


def pick_random_edge_weighted(G):
    """
    Pick a random edge on G.
    Scale chance to pick an edge by its edge length.
    """
    eids    = get_eids(G)
    lengths = array([attrs["length"] for eid, attrs in iterate_edges(G)])
    total   = sum(lengths)
    weights = lengths / total

    _eid_indices = [i for i in range(len(eids))]
    _eid_index   = np.random.choice(_eid_indices, 1, list(weights))[0]
    return  eids[_eid_index]


def gen_position_by_nearest_point(H, p, H_edge_rtree):
    """Obtain nearest point on H for a point p on G."""
    H_eid   = nearest_edge_for_position(H, p, edge_tree=H_edge_rtree)
    H_curve = graphedge_curvature(H, H_eid)
    H_position, H_interval = nearest_position_and_interval_on_curve_to_point(H_curve, p)
    H_distance = graphedge_length(H, H_eid) * H_interval
    return {
        "eid": H_eid,
        "position": H_position,
        "distance": H_distance
    }


def random_position_on_graph(G, random_edge_picker=None):
    """Obtain a random position on graph G alongside related eid and (curvature) distance from eid[0] endpoint."""
    if random_edge_picker != None:
        return random_edge_picker()

    G_eid = pick_random_edge_weighted(G)
    attrs  = get_edge_attributes(G, G_eid)
    length = attrs["length"]
    interval = random.random()
    G_distance = interval * length
    G_position = position_at_curve_interval(attrs["curvature"], interval)
    return {
        "eid": G_eid,
        "position": G_position,
        "distance": G_distance
    }


def get_sample(G, H, G_paths, H_paths, H_edge_rtree, max_distance, random_edge_picker=None):
    """Obtain a random position on G and its nearest position on H."""
    # Start and end position in G.
    G_start = random_position_on_graph(G)
    G_end   = random_position_on_graph(G)

    a, b = sorted([G_start["eid"][0], G_end["eid"][0]])
    if b not in G_paths[a]:
        return None

    # Related positions in H.
    H_start = gen_position_by_nearest_point(H, G_start["position"], H_edge_rtree)
    H_end   = gen_position_by_nearest_point(H, G_end["position"], H_edge_rtree)

    # Compute categorization properties.
    is_primal = float(norm(H_start["position"] - G_start["position"])) < max_distance and float(norm(H_end["position"] - G_end["position"]) < max_distance)
    a, b = sorted([H_start["eid"][0], H_end["eid"][0]])
    path_exists = b in H_paths[a]

    return {
        "A": not is_primal,
        "B": is_primal and not path_exists,
        "C": is_primal and path_exists,
        "G": {
            "start": G_start,
            "end"  : G_end,
        }, 
        "H": {
            "start": H_start,
            "end"  : H_end,
        }, 
    }


@info(timer=True)
def apls_sampling(G, H, G_paths, H_paths, n=10000, max_distance=5):
    """
    Obtain samples. 
    * G: Target graph
    * H: Source graph
    * G_to_H: Link graph node of G to nearest edge of H
    * G_paths: Shortest paths dictionary on graph G
    * H_paths: Shortest paths dictionary on graph H
    """ 

    # Pick n samples at random in G.
    eids    = get_eids(G)
    lengths = array([attrs["length"] for eid, attrs in iterate_edges(G)])
    total   = sum(lengths)
    weights = lengths / total
    _eid_indices = [i for i in range(len(eids))]

    def _random_edge_picker():
        """Define a local random edge picker to save significant computation costs (in computing weights)."""
        _eid_index = np.random.choice(_eid_indices, 1, list(weights))[0]
        return _eid_indices[_eid_index]

    H_edge_rtree = graphedges_to_rtree(H)
    
    normal_samples = []
    while len(normal_samples) < n:
        if random.random() < 0.001:
            print(f"Number of normal samples generated: {len(normal_samples)}/{n}.")
        sample = get_sample(G, H, G_paths, H_paths, H_edge_rtree, max_distance, random_edge_picker=_random_edge_picker)
        if sample != None:
            normal_samples.append(sample)

    primal_samples = [sample for sample in normal_samples if not sample["A"]]
    i = 0
    while len(primal_samples) < n:
        i += 1
        if random.random() < 0.001:
            print(f"Number of primal samples generated: {len(primal_samples)}/{n}. (Total attempts: {i})")
        sample = get_sample(G, H, G_paths, H_paths, H_edge_rtree, max_distance, random_edge_picker=_random_edge_picker)
        if sample != None and not sample["A"]:
            primal_samples.append(sample)

    # Collect.
    samples_normal = {
        "A": [sample for sample in normal_samples if sample["A"]],
        "B": [sample for sample in normal_samples if sample["B"]],
        "C": [sample for sample in normal_samples if sample["C"]],
    }

    samples_primal = {
        "A": [],
        "B": [sample for sample in primal_samples if sample["B"]],
        "C": [sample for sample in primal_samples if sample["C"]],
    }

    return samples_normal, samples_primal


# Compute score on the data object.
def compute_score(samples, path_scores):
    sample_sum  = sum(path_scores)
    n = len(samples["B"]) + len(samples["C"]) + len(samples["A"])
    score = sample_sum / n
    return score


def asymmetric_apls(G, H, G_paths, H_paths, n=1000, threshold=5):
    """
    Obtain _asymmetric_ APLS aand APLS* score alongside metadata to reconstruct APLS and APLS* score from.
    Input variables are:
    * G: Target graph (ground truth).
    * H: Source graph (inferred graph).
    * H_paths: Shortest paths matrix (generated with `precompute_shortest_path_data`).
    * n: Number of samples (start to endpoint pairs)
    * threshold: APLS distance threshold seeking nearest curvature to target point.
    """

    # Obtain samples.
    samples_normal, samples_primal = apls_sampling(G, H, G_paths, H_paths, n=n, max_distance=threshold)

    def sample_score(a, b):
        """
        Compute score of a sample.
        * a: target path length
        * b: source path length
        """ 
        return 1 - min(abs(a - b) / a, 1)
    
    def total_score(samples, path_scores):
        """
        Compute score of all samples (category A, B, C) combined.
        """
        n = len(samples["A"]) + len(samples["B"]) + len(samples["C"])
        sample_sum = sum(path_scores)
        return sample_sum / n
    
    def reconstruct_shortest_path(sample, graph, paths):
        """
        Reconstruct the shortest path from a sample.
        Expect the shortest path for this sample exists.

        The reason its this complicated in comparison to just picking nodes on the graph:
        * This logic allows to sample _all_ locations on source graph.
        * It uses the most minimal distance matric (see `precompute_shortest_path_data`) which improves drastically computation and storage costs by not having to deal with thousands of additional nodes.
        """

        u, v = sample["start"]["eid"][:2]
        x, y = sample["end"]["eid"][:2]
        d1   = sample["start"]["distance"]
        d2   = sample["end"]["distance"]

        a = paths[u][x] if u <= x else paths[x][u] 
        b = paths[u][y] if u <= y else paths[y][u] 
        c = paths[v][x] if v <= x else paths[x][v]
        d = paths[v][y] if v <= y else paths[y][v]
        
        l1 = get_edge_attributes(graph, sample["start"]["eid"])["length"]
        l2 = get_edge_attributes(graph, sample["end"]["eid"])["length"]

        check(float(d1 - l1) < 0.0001)
        check(float(d2 - l2) < 0.0001)

        check(float(abs(c - a) - l1) < 0.0001)
        check(float(abs(d - b) - l1) < 0.0001)
        check(float(abs(b - a) - l2) < 0.0001)
        check(float(abs(d - c) - l2) < 0.0001)
        
        lowest = min(a,b,c,d)

        if a == lowest:
            return a + d1 + d2
        elif b == lowest:
            return a + d1 + (l2 - d2)
        elif c == lowest:
            return a + (l1 - d1) + d2
        elif d == lowest:
            return a + (l1 - d1) + (l2 - d2)

    # Obtain path scores.
    path_scores_normal = []
    for sample in samples_normal["C"]:
        
        target_path_distance = reconstruct_shortest_path(sample["G"], G, G_paths)
        source_path_distance = reconstruct_shortest_path(sample["H"], H, H_paths)
        score = sample_score(target_path_distance, source_path_distance)
        path_scores_normal.append({
            "target": target_path_distance,
            "source": source_path_distance,
            "score" : score,
        })

    path_scores_primal = []
    # TODO: Re-use overlap with normal samples.
    for sample in samples_primal["C"]:

        target_path_distance = reconstruct_shortest_path(sample["G"], G, G_paths)
        source_path_distance = reconstruct_shortest_path(sample["H"], H, H_paths)
        score = sample_score(target_path_distance, source_path_distance)
        path_scores_primal.append({
            "target": target_path_distance,
            "source": source_path_distance,
            "score" : score,
        })
    
    # Compute APLS score (accumulate all sample scores).
    apls_score       = sum([element["score"] for element in path_scores_normal]) / (len(samples_normal["A"]) + len(samples_normal["B"]) + len(samples_normal["C"]))
    apls_prime_score = sum([element["score"] for element in path_scores_primal]) / (len(samples_primal["A"]) + len(samples_primal["B"]) + len(samples_primal["C"]))

    # Compute metadata: 
    metadata = {
        "n": n,
        "apls_threshold": threshold, 
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
def symmetric_apls(G, H, G_paths, H_paths, n=1000, threshold=5):
    """
    Obtain _symmetric_ APLS aand APLS* score alongside metadata to reconstruct APLS and APLS* score from.
    Input variables are:
    * G: Target graph (ground truth).
    * H: Source graph (inferred graph).
    * H_paths: Shortest paths matrix (generated with `precompute_shortest_path_data`).
    * n: Number of samples (start to endpoint pairs)
    * threshold: APLS distance threshold seeking nearest curvature to target point.
    """
    left = asymmetric_apls(G, H, G_paths, H_paths, n=n, threshold=threshold)
    right= asymmetric_apls(H, G, H_paths, G_paths, n=n, threshold=threshold)

    apls_score       = 0.5 * (left[0] + right[0])
    apls_prime_score = 0.5 * (left[1] + right[1])

    metadata = {
        "n": n,
        "apls_threshold": threshold, 
        "left": left[2],
        "right": right[2]
    }

    return apls_score, apls_prime_score, metadata


def symmetric_apls_from_metadata(metadata):
    """
    Compute/Derive symmetric APLS and APLS* value from metadata.
    Ideal for experimentation with large graphs/data and many samples (saves a lot of recomputing sample data).
    """

    # Left
    samples_normal     = metadata["left"]["normal"]["samples"]
    samples_primal     = metadata["left"]["primal"]["samples"]

    path_scores_normal = metadata["left"]["normal"]["path_scores"]
    path_scores_primal = metadata["left"]["primal"]["path_scores"]

    l_apls_score       = sum([element["score"] for element in path_scores_normal]) / (len(samples_normal["A"]) + len(samples_normal["B"]) + len(samples_normal["C"]))
    l_apls_prime_score = sum([element["score"] for element in path_scores_primal]) / (len(samples_primal["A"]) + len(samples_primal["B"]) + len(samples_primal["C"]))

    # Right
    samples_normal     = metadata["right"]["normal"]["samples"]
    samples_primal     = metadata["right"]["primal"]["samples"]

    path_scores_normal = metadata["right"]["normal"]["path_scores"]
    path_scores_primal = metadata["right"]["primal"]["path_scores"]

    r_apls_score       = sum([element["score"] for element in path_scores_normal]) / (len(samples_normal["A"]) + len(samples_normal["B"]) + len(samples_normal["C"]))
    r_apls_prime_score = sum([element["score"] for element in path_scores_primal]) / (len(samples_primal["A"]) + len(samples_primal["B"]) + len(samples_primal["C"]))

    apls_score       = 0.5 * (l_apls_score + r_apls_score)
    apls_prime_score = 0.5 * (l_apls_prime_score + r_apls_prime_score)

    breakpoint()

    return apls_score, apls_prime_score