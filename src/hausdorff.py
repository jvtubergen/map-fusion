from scipy.spatial.distance import directed_hausdorff

def hausdorff(p,q):
    return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
