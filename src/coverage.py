import numpy as np
from frechet import *
from hausdorff import *

# Note: A point curve is a sequence of points, linearly interpolated.
# Note: Algorithms consider point curves only.

# Discrete coverage of ps by qs.
# def curve_coverage(ps, qs, l = 0.05, measure=frechet):

#     return 0

# Check curve ps is covered by curve qs.
def curve_by_curve_coverage(ps, qs, lam=1, measure=frechet):

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

    # Per point in ps, see what nodes of qs are within range.
    ranged = []
    for p in ps:
        # Extract subcollection within range of point p.        
        filter
        r = [i for (i,q) in enumerate(qs) if np.linalg.norm(p - q) <= lam ]
        # Cancel early if no node at all within range.
        if len(r) == 0:
            return False
        ranged.append(r)

    def convert_into_ranges(rs):
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
    
    # Convert into bounding boxes.
    ranged = [convert_into_ranges(r) for r in ranged]


    # Hausdorff specifc:

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

    


    # Hausdorff:
    # * Seek valid intervals for subsequent element.
    # * Start with intervals of initial.
    if measure == hausdorff:
        valids = ranged[0]
        for rs in ranged[1:]:
            # Any subset that has overlap (margin of 1) is valid and represents next step
            valids2 = []
            for valid in valids:
                for r in rs:
                    if overlap(valid, pad(r, 1)):
                        valids2.append(r)
            valids = valids2[:]
    


    # Frechet:
    # Valids should not just overlap, overlap should be larger than.
    elif measure == frechet:
        valids = ranged[0]
        for rs in ranged[1:]:
            # Any subset that has overlap (margin of 1) is valid and represents next step.
            valids2 = []
            for valid in valids:
                for r in rs:
                    # pad differently: min value of r is >= to lowest value of valid.
                    padded = pad_larger_than(valid, r[:])
                    if overlap(valid, pad_left(padded, 1)):
                        valids2.append(padded)
            valids = valids2[:]
    
    else:
        raise ValueError("invalid measure must be 'hausdorff', 'frechet'.")

    return len(valids) > 0


    # Frechet:

    # Start at first node as viable solutions
    # solutions = []
    # for ranged ``

    # rs = []


    # # If Hausdorff, seek for subcurve
    # # 1. Convert into set
    # # 2. Seek lowest value 


    # Hausdorff approach (will not work): Per node something within range should exist.
    # 1. Construct Minskowski surface.
    # 2. Check for subcurve within Minskowski sum:


# Test:
# * Generate a curve randomly
# * Per point generate three in range
# * Pick one of those and represent as curve
# * Pool unused nodes with some more randomly generated curves
# * Add some arbitrary other nodes of these and add to curve
# Verify:
# * Expect to find some subcurve
# * Generated subcurve within distance lambda
    
