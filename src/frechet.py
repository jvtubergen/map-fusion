# https://github.com/spiros/discrete_frechet
# License: Apache-2.0
from frechetdist import frdist

def discrete_frechet_distance(ps,qs):
    return frdist(ps,qs)

from frechet_distance_module import compute_single_threshold as continuous_frechet_distance_threshold

def continuous_frechet_distance_threshold_undirected(ps, qs, dif):
    return continuous_frechet_distance_threshold(ps, qs, dif) or continuous_frechet_distance_threshold(ps, qs[::-1], dif)

def continuous_frechet_distance(ps, qs, dif = 0.0001):
    a = min(np.min(qs),np.min(ps))
    b = max(np.max(qs),np.max(ps))
    d = sqrt(2 * pow(b - a, 2)) # Maximal distance
    cur = eps = d / 2
    while eps > dif:
        print(cur)
        eps = eps / 2
        if continuous_frechet_distance_threshold(ps, qs, cur):
            cur -= eps
        else:
            cur += eps
    return cur

def continuous_frechet_distance_undirected(ps, qs):
    return min(frechet_distance(ps, qs), frechet_distance(ps, qs[::-1]))
