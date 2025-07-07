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

    # Compute distance matrix between these points.
    distance_matrix = {}
    for u in nids:

        # Compute all reachable points from this node.
        distances = nx.single_source_dijkstra_path_length(G, u, weight="length")

        # Filter out dictionary to only include end nodes which are:
        # * Contained in the control nodes list.
        # * Have a node identifier higher than the control nid coming from (prevents duplicated checks in subsequent logic).
        filtered = {}
        for v in distances.keys():
            if v >= u and v in nids:
                filtered[v] = float(distances[v])
        
        distance_matrix[u] = filtered

    return distance_matrix


@info(timer=True)
def relate_control_points(G, H, max_distance=25):
    """
    Relate nodes of target graph G to edges of source graph H and mention offset in H edge (so we can compute distance of shortest path).
    Return a dictionary that links nodes of G to H (keys are nids of G, values are eid with distance offset of H it connects to).
    Note: eid always has lowest nid first. In combination with offset from lowest nid this provides therefore all information necessary.
    """

    G_to_H = {} # All nodes of G related to a node alongside offset of H.

    H_edge_rtree = graphedges_to_rtree(H)

    # Find (and optionally inject) node within H nearby every node of G.
    for G_nid, attrs in iterate_nodes(G):

        G_position = graphnode_position(G, G_nid)
        H_eid = nearest_edge_for_position(H, G_position, edge_tree=H_edge_rtree)
        H_nid = H_eid[0] # Take start value.
        H_curve = graphedge_curvature(H, H_eid)
        H_position, H_edge_interval = nearest_position_and_interval_on_curve_to_point(H_curve, G_position)
        G_H_distance = norm(H_position - G_position)

        if G_H_distance <= max_distance:
            G_to_H[G_nid] = (H_nid, H_edge_interval)

    return G_to_H


def pick_random_edge_weighted(G):
    """
    Pick a random edge on G.
    Scale chance to pick an edge by its edge length.
    """
    eids    = get_eids(G)
    lengths = array([attrs["length"] for eid, attrs in iterate_edges(G)])
    total   = sum(lengths)
    weights = lengths / total
    return np.random.choice(eids, 1, weights)


@info(timer=True)
def apls_sampling(G, H, G_paths, H_paths, n=10000, max_distance=25):
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

    H_edge_rtree = graphedges_to_rtree(H)

    def gen_random_position(G):
        _eid_indices = [i for i in range(len(eids))]
        _eid_index   = np.random.choice(_eid_indices, 1, list(weights))[0]
        G_eid  = eids[_eid_index]
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
    
    def gen_position_by_nearest_point(H, p):
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

    def get_sample():
        """Obtain a random position on G and its nearest position on H."""
        # Start and end position in G.
        G_start = gen_random_position(G)
        G_end   = gen_random_position(G)
        a, b = sorted([G_start["eid"][0], G_end["eid"][0]])
        if b not in G_paths[a]:
            return None
        # Expect shortest path for other three edge endpoints exist as well.
        u, v = G_start["eid"][:2]
        x, y = G_end["eid"][:2]
        # Related positions in H.
        H_start = gen_position_by_nearest_point(H, G_start["position"])
        H_end   = gen_position_by_nearest_point(H, G_end["position"])

        is_primal = norm(H_start["position"] - G_start["position"]) < max_distance and norm(H_end["position"] - G_end["position"]) < max_distance,
        a, b = sorted([H_start["eid"][0], H_end["eid"][0]])
        path_exists = b in H_paths[a]

        if path_exists:
            u, v = H_start["eid"][:2]
            x, y = H_end["eid"][:2]
            H_paths[u][x] if u <= x else H_paths[x][u] 
            H_paths[u][y] if u <= y else H_paths[y][u] 
            H_paths[v][x] if v <= x else H_paths[x][v]
            H_paths[v][y] if v <= y else H_paths[y][v]

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

    normal_samples = []
    while len(normal_samples) < n:
        if random.random() < 0.001:
            print(len(normal_samples))
        sample = get_sample()
        if sample != None:
            normal_samples.append(sample)

    primal_samples = [sample for sample in normal_samples if not sample["A"]]
    while len(primal_samples) < n:
        sample = get_sample()
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

    # Obtain samples.
    samples_normal, samples_primal = apls_sampling(G, H, G_paths, H_paths, n=n)

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
    
    def reconstruct_shortest_path(sample, graph, paths):
        # Obtain distance target path.
        # Expect shortest path exists.
        # Pick one of the four edge endpoint combinations is lowest distance?
        # Account for distance offset from edge endpoint.

        u, v = sample["start"]["eid"][:2]
        x, y = sample["end"]["eid"][:2]
        d1   = sample["start"]["distance"]
        d2   = sample["end"]["distance"]

        l1 = get_edge_attributes(graph, sample["start"]["eid"])["length"]
        l2 = get_edge_attributes(graph, sample["end"]["eid"])["length"]

        check(d1 <= l1)
        check(d2 <= l2)
    
        check(abs(c - a) - l1 < 0.0001)
        check(abs(d - b) - l1 < 0.0001)
        check(abs(b - a) - l2 < 0.0001)
        check(abs(d - c) - l2 < 0.0001)
        
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
    right= asymmetric_apls(H, G, H_paths, G_paths, n=n, threshold=threshold)

    apls_score = 0.5 * (left[0] + right[0])
    apls_prime_score = 0.5 * (left[1] + right[1])
    metadata = {
        "left": left[2],
        "right": right[2]
    }

    return apls_score, apls_prime_score, metadata