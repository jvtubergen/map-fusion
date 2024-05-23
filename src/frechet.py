# https://github.com/spiros/discrete_frechet
# License: Apache-2.0
from frechetdist import frdist

def frechet(ps,qs):
    return frdist(ps,qs)

# todo: Cheap Frechet function within lambda distance