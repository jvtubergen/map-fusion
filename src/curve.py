from dependencies import *

def curve_length(ps):
    return sum([norm(p1 - p2) for p1, p2 in zip(ps, ps[1:])])

# Return a pair of the first half and second half of the curve.
def curve_cut(ps, percentage):
    assert len(ps) >= 2
    steps = [norm(p1 - p2) for p1, p2 in zip(ps, ps[1:])]
    length = sum(steps)
    target = percentage * length
    assert length > 0
    current = 0
    i = 0
    while True:
        step = steps[i]
        upcoming = current + step
        if upcoming > target: # Cut current edge.
            line_percentage = (target - current) / step
            p = line_percentage * ps[i+1] + (1 - line_percentage) * ps[i]
            left  = np.append(ps[:i+1], [p], axis=0)
            right = np.append([p], ps[i+1:], axis=0)
            return (left, right)
        current = upcoming

curve_cut_in_half = lambda ps: curve_cut(ps, 0.5)

# ps  = array([[0,0], [1,1]])
# (ls, rs) = curve_cut_in_half(ps)
# print(ps, curve_length(ps))
# print(ls, curve_length(ls))
# print(rs, curve_length(rs))
# assert curve_length(ls) == 0.5 * curve_length(ps) 
# assert curve_length(rs) == 0.5 * curve_length(ps) 

