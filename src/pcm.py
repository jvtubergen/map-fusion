from partial_curve_matching import Vector, partial_curve

# Convert 2d numpy array into a Curve (list of Vectors).
def to_curve(ps):
    result = []
    for [x,y] in ps:
        result.append(Vector(x,y))
    return result

# Compute partial curve matching between curve ps and some subcurve of qs within eps distance threshold.
def is_partial_curve_undirected(ps, qs, eps):
    assert type(ps[0]) == Vector
    assert type(qs[0]) == Vector
    return partial_curve(ps, qs, eps) != None or partial_curve(ps[::-1], qs, eps)