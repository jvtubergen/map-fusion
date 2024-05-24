import numpy as np
from frechet import *
from hausdorff import *

# Note: A point curve is a sequence of points, linearly interpolated.
# Note: Algorithms consider point curves only.

# Discrete curve coverage of ps by qs.
def curve_by_curve_coverage(ps, qs, lam=1, measure=frechet):

    rev = False
    found, hist = _curve_by_curve_coverage(ps, qs, lam=lam, measure=frechet)
    if not found:
        rev = True
        found, hist = _curve_by_curve_coverage(ps, qs[::-1], lam=lam, measure=frechet)
    return found, hist, rev

# Check curve ps is covered by curve qs.
def _curve_by_curve_coverage(ps, qs, lam=1, measure=frechet):

    # Method 1:
    # * Per point p find ids of qs that are within range.
    # * Walk all possible routes, see if any is a viable subcurve within qs.

    # Optimizations:
    # 1. Walk and push paths that are viable 
    # 2. Only check p against q at specific index when checking possible route (minimizing distance queries)
    # 3. Use r-tree when seeking points in q nearby (improving distance query speed)

    # Method 2:
    # 1. Seek ranges per point.
    # 2. Per step seek valid ranges, take those subsets.
    # 3. If hausdorff, full subset is valid
    # 4. If Frechet, only subsets incremented.

    def convert_into_intervals(rs):
        # Assume ranges to be incremental.
        ranges = []

        i = rs[0]
        t = i
        k = i
        for k in rs[1:]:
            if k == t + 1:
                t += 1
            else:
                ranges.append((i,t))
                i = k
                t = i
        
        if i == k:
            ranges.append((i,i))
        else:
            ranges.append((i,t))
        
        return ranges
    

    
    # Hausdorff specific:

    # Pad an interval by value.
    def pad(interval, value):
        return (interval[0] - value, interval[1] + value)
    
    # Check overlap of two intervals.
    def overlap(i1, i2):
        return (i1[0] <= i2[0] and i1[-1] >= i2[0]) or (i2[0] <= i1[0] and i2[-1] >= i1[0])


    # Frechet specific:

    # Pad an interval by value.
    def pad_larger_than(i1, i2):
        if i2[0] < i1[0]:
            i2 = (i1[0], i2[1])
        
        if i2[1] < i2[0]:
            return [-1,-1] # empty interval
        
        return i2
    
    # Pad left by value
    def pad_left(interval, value):
        return (interval[0] - value, interval[1])

    # Check overlap of two intervals, but second interval must be larger than.
    def overlap_larger_than(i1, i2):
        return (i1[0] <= i2[0] and i1[-1] >= i2[0]) or (i2[0] <= i1[0] and i2[-1] >= i1[0])

    
    # Per point in ps, see what nodes of qs are within range.
    in_range = []
    for p in ps:
        # Extract subcollection within range of point p.        
        ids = [i for (i,q) in enumerate(qs) if np.linalg.norm(p - q) <= lam ]
        # Cancel early if no node at all within range.
        if len(ids) == 0:
            return False, []
        in_range.append(ids)

    # Convert into bounding boxes.
    intervals = [convert_into_intervals(ids) for ids in in_range]


    # Hausdorff:
    # * Seek valid intervals for subsequent element.
    # * Start with intervals of initial.
    # if measure == hausdorff:
    #     valids = ranged[0]
    #     history = [valids] # Chaining valid intervals to 
    #     for rs in ranged[1:]:
    #         # Any subset that has overlap (margin of 1) is valid and represents next step
    #         valids2 = []
    #         history2 = []
    #         for i, valid in enumerate(valids):
    #             for r in rs:
    #                 if overlap(valid, pad(r, 1)):
    #                     history2.append([history[i] ++ r])
    #                     valids2.append(r)
    #         valids = valids2[:]
    #         history = history2[:]


    # Frechet:
    # Valids should not just overlap, overlap should be larger than.
    # elif measure == frechet:
    histories = [intervals[0]] # Initially, all valid intervals at first point entail valid history
    for inters in (intervals[1:]):
        # print("histories:",histories)
        histories2 = []
        for i, hs in enumerate(histories): # Iterate each interval sequence of current history which is valid
            for (ida, idb) in inters: # Check per range whether it can add to historic.
                valid = hs[-1] # Final range within current historic range sequence.e
                padded = pad_larger_than(valid, (ida, idb))
                if overlap(valid, pad_left(padded, 1)):
                    histories2.append( hs + [padded] )
        histories = histories2[:]
    
    # else:
    #     raise ValueError("invalid measure must be 'hausdorff', 'frechet'.")
    
    return len(histories) > 0, histories


# # Convert history object into a sequence.
# def history_to_sequence(history):

#     def intersection(i1, i2):
#         return (max(i1[0], i2[0]), min(i1[1], i2[1]))

#     # Construct sequence for ps and qs.
#     # With coverage, we have to find _some_ subcurve of qs that covers _all_ of ps.
#     # Simple method approach: Keep value of qs as low as possible.
#     # sequence = []

#     # Assume Frechet: Only incremental allowed.
#     # Simply stay at lowest value as possible.

#     # cp: Current p, simply start at 0.
#     # cq: Current q, simply start at lowest possible curve value.
#     steps = []

#     for cp, (h1,h2) in enumerate(zip(history,history[1:])):

#         cq = max(cq, h1[0])
#         steps.append((cp,cq))

#         # Walk all of h1 until lowest value of h2 is reached.
#         while cq < h2[0] + 1:
#             cq += 1
#             steps.append((cp, cq))
    
#     # Final value
#     steps.append((len(cp) - 1, max(history[-1][0],cq)))

#     return steps
#     # return np.array(steps)





def history_to_sequence(history):

    steps = []
    cq = -1
    for cp, (h1,h2) in enumerate(zip(history,history[1:])):

        cq = max(cq, h1[0])
        steps.append((cp,cq))

        # Walk all of h1 until lowest value of h2 is reached.
        while cq < h2[0]:
            cq += 1
            steps.append((cp, cq))
    
    # Final value

    return steps
    # return np.array(steps)





# Test:
# * Generate a curve randomly
# * Per point generate three in range
# * Pick one of those and represent as curve
# * Pool unused nodes with some more randomly generated curves
# * Add some arbitrary other nodes of these and add to curve
# Verify:
# * Expect to find some subcurve
# * Generated subcurve within distance lambda
    
