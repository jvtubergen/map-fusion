from scipy.spatial.distance import directed_hausdorff

def hausdorff(ps,qs):
    return max(directed_hausdorff(ps, qs)[0], directed_hausdorff(qs, ps)[0])

# todo: Cheap Hausdorff function within lambda distance
