from partial_curve_matching import Vector, partial_curve

# Convert 2d numpy array into a Curve (list of Vectors).
def to_curve(ps):
    result = []
    for [x,y] in ps:
        result.append(Vector(x,y))
    return result

# Compute partial curve matching between curve ps and some subcurve of qs within eps distance threshold.
def pcm(ps, qs, eps):
    if type(ps[0]) != Vector:
        ps = to_curve(ps)
        qs = to_curve(qs)
    return partial_curve(ps, qs, eps)
    

