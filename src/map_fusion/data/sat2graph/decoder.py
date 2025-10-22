import scipy 
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from rtree import index 
from data.sat2graph.common import * 


vector_norm = 25.0 

def vNorm(v1):
    l = distance(v1,(0,0))+0.0000001
    return (v1[0]/l, v1[1]/l)

def anglediff(v1, v2):
    v1 = vNorm(v1)
    v2 = vNorm(v2)

    return v1[0]*v2[0] + v1[1] * v2[1]

def graph_refine(graph, isolated_thr = 150, spurs_thr = 30, three_edge_loop_thr = 70):
    neighbors = graph

    gid = 0 
    grouping = {}

    for k, v in neighbors.items():
        if k not in grouping:
            # start a search 

            queue = [k]

            while len(queue) > 0:
                n = queue.pop(0)

                if n not in grouping:
                    grouping[n] = gid 
                    for nei in neighbors[n]:
                        queue.append(nei)

            gid += 1 

    group_count = {}

    for k, v in grouping.items():
        if v not in group_count:
            group_count[v] = (1,0)
        else:
            group_count[v] = (group_count[v][0] + 1, group_count[v][1])


        for nei in neighbors[k]:
            a = k[0] - nei[0]
            b = k[1] - nei[1]

            d = np.sqrt(a*a + b*b)

            group_count[v] = (group_count[v][0], group_count[v][1] + d/2)

    # short spurs
    remove_list = []
    for k, v in neighbors.items():
        if len(v) == 1:
            if len(neighbors[v[0]]) >= 3:
                a = k[0] - v[0][0]
                b = k[1] - v[0][1]

                d = np.sqrt(a*a + b*b)	

                if d < spurs_thr:
                    remove_list.append(k)


    remove_list2 = []
    remove_counter = 0
    new_neighbors = {}

    def isRemoved(k):
        gid = grouping[k]
        if group_count[gid][0] <= 1:
            return True 
        elif group_count[gid][1] <= isolated_thr:
            return True 
        elif k in remove_list:
            return True 
        elif k in remove_list2:
            return True
        else:
            return False

    for k, v in neighbors.items():
        if isRemoved(k): 
            remove_counter += 1
            pass
        else:
            new_nei = []
            for nei in v:
                if isRemoved(nei):
                    pass 
                else:
                    new_nei.append(nei)

            new_neighbors[k] = list(new_nei)

    #print(len(new_neighbors), "remove", remove_counter, "nodes")

    return new_neighbors

def graph_shave(graph, spurs_thr = 50):
    neighbors = graph

    # short spurs
    remove_list = []
    for k, v in neighbors.items():
        if len(v) == 1:
            d = distance(k,v[0])
            cur = v[0]
            l = [k]
            while True:
                if len(neighbors[cur]) >= 3:
                    break
                elif len(neighbors[cur]) == 1:
                    l.append(cur)
                    break

                else:

                    if neighbors[cur][0] == l[-1]:
                        next_node = neighbors[cur][1]
                    else:
                        next_node = neighbors[cur][0]

                    d += distance(cur, next_node)
                    l.append(cur)

                    cur = next_node 

            if d < spurs_thr:
                for n in l:
                    if n not in remove_list:
                        remove_list.append(n)

    
    def isRemoved(k):
        if k in remove_list:
            return True 
        else:
            return False

    new_neighbors = {}
    remove_counter = 0

    for k, v in neighbors.items():
        if isRemoved(k): 
            remove_counter += 1
            pass
        else:
            new_nei = []
            for nei in v:
                if isRemoved(nei):
                    pass 
                else:
                    new_nei.append(nei)

            new_neighbors[k] = list(new_nei)

    #print("shave", len(new_neighbors), "remove", remove_counter, "nodes")

    return new_neighbors

def graph_refine_deloop(neighbors, max_step = 10, max_length = 200, max_diff = 5):

    removed = []
    impact = []

    remove_edge = []
    new_edge = []

    for k, v in neighbors.items():
        if k in removed:
            continue

        if k in impact:
            continue


        if len(v) < 2:
            continue


        for nei1 in v:
            if nei1 in impact:
                continue

            if k in impact:
                continue
            
            for nei2 in v:
                if nei2 in impact:
                    continue
                if nei1 == nei2 :
                    continue



                if neighbors_cos(neighbors, k, nei1, nei2) > 0.984:
                    l1 = neighbors_dist(neighbors, k, nei1)
                    l2 = neighbors_dist(neighbors, k, nei2)

                    #print("candidate!", l1,l2,neighbors_cos(neighbors, k, nei1, nei2))

                    if l2 < l1:
                        nei1, nei2 = nei2, nei1 

                    remove_edge.append((k,nei2))
                    remove_edge.append((nei2,k))

                    new_edge.append((nei1, nei2))

                    impact.append(k)
                    impact.append(nei1)
                    impact.append(nei2)

                    break

    new_neighbors = {}

    def isRemoved(k):
        if k in removed:
            return True 
        else:
            return False

    for k, v in neighbors.items():
        if isRemoved(k): 
            pass
        else:
            new_nei = []
            for nei in v:
                if isRemoved(nei):
                    pass 
                elif (nei, k) in remove_edge:
                    pass
                else:
                    new_nei.append(nei)

            new_neighbors[k] = list(new_nei)

    for new_e in new_edge:
        nk1 = new_e[0]
        nk2 = new_e[1]

        if nk2 not in new_neighbors[nk1]:
            new_neighbors[nk1].append(nk2)
        if nk1 not in new_neighbors[nk2]:
            new_neighbors[nk2].append(nk1)



    #print("remove %d edges" % len(remove_edge))

    return new_neighbors, len(remove_edge)

def locate_stacking_road(graph):

    idx = index.Index()

    edges = []
    
    for n1, v in graph.items():
        for n2 in v:
            if (n1,n2) in edges or (n2,n1) in edges:
                continue

            x1 = min(n1[0], n2[0])
            x2 = max(n1[0], n2[0])

            y1 = min(n1[1], n2[1])
            y2 = max(n1[1], n2[1])

            idx.insert(len(edges), (x1,y1,x2,y2))

            edges.append((n1,n2))

    adjustment = {}

    crossing_point = {}


    for edge in edges:
        n1 = edge[0]
        n2 = edge[1]



        x1 = min(n1[0], n2[0])
        x2 = max(n1[0], n2[0])

        y1 = min(n1[1], n2[1])
        y2 = max(n1[1], n2[1])

        candidates = list(idx.intersection((x1,y1,x2,y2)))

        for _candidate in candidates:
            # todo mark the overlap point 
            candidate = edges[_candidate]


            if n1 == candidate[0] or n1 == candidate[1] or n2 == candidate[0] or n2 == candidate[1]:
                continue

            if intersect(n1,n2,candidate[0], candidate[1]):

                ip = intersectPoint(n1,n2,candidate[0], candidate[1])



                if (candidate, edge) not in crossing_point:
                    crossing_point[(edge, candidate)] = ip

                #release points 

                d = distance(ip, n1)
                thr = 5.0
                if d < thr:
                    vec = neighbors_norm(graph, n1, n2)
                    #vec = (vec[0] * (thr-d), vec[1] * (thr-d))
                    
                    
                    if n1 not in adjustment:
                        adjustment[n1] = [vec] 
                    else:
                        adjustment[n1].append(vec)

                d = distance(ip, n2)
                if d < thr:
                    vec = neighbors_norm(graph, n2, n1)
                    #vec = (vec[0] * (thr-d), vec[1] * (thr-d))
                    
                    
                    if n2 not in adjustment:
                        adjustment[n2] = [vec] 
                    else:
                        adjustment[n2].append(vec)


                c1 = candidate[0]
                c2 = candidate[1]


                d = distance(ip, c1)
                if d < thr:
                    vec = neighbors_norm(graph, c1, c2)
                    #vec = (vec[0] * (thr-d), vec[1] * (thr-d))
                    
                    
                    if c1 not in adjustment:
                        adjustment[c1] = [vec] 
                    else:
                        adjustment[c1].append(vec)

                d = distance(ip, c2)
                if d < thr:
                    vec = neighbors_norm(graph, c2, c1)
                    #vec = (vec[0] * (thr-d), vec[1] * (thr-d))
                    
                    
                    if c2 not in adjustment:
                        adjustment[c2] = [vec] 
                    else:
                        adjustment[c2].append(vec)


    return crossing_point, adjustment

def detect_local_minima(arr, mask, threshold = 0.5):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value 
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # 
    # we create the mask of the background
    background = (arr==0)
    # 
    # a little technicality: we must erode the background in order to 
    # successfully subtract it from local_min, otherwise a line will 
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    # 
    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background
    return np.where((detected_minima & (mask > threshold)))  
    #return np.where(detected_minima)



def findClearKeypoints(vertexness, thr = 0.5):
    kp = np.copy(vertexness)
    smooth_kp = scipy.ndimage.filters.gaussian_filter(np.copy(kp), 1)
    smooth_kp = smooth_kp / max(np.amax(smooth_kp),0.001)
    keypoints = detect_local_minima(-smooth_kp, smooth_kp, thr)

    newkp = np.zeros_like(vertexness)
    dim = np.shape(vertexness)

    idx = index.Index()
    for i in range(len(keypoints[0])):
        x,y = keypoints[0][i], keypoints[1][i]
        idx.insert(i,(x-1,y-1,x+1,y+1))

    for i in range(len(keypoints[0])):
        x,y = keypoints[0][i], keypoints[1][i]
        candidates = list(idx.intersection((x-8,y-8,x+8,y+8)))
        if len(candidates) > 1:
            continue

        if x > 3 and x < dim[0]-3 and y>3 and y<dim[1]-3:
            v1 = vertexness[x,y]
            v2 = np.amin(vertexness[x-2:x+3, y-2:y+3])
            if v1 > 2.0 * v2:
                for _x in range(x-1,x+2):
                    for _y in range(y-1,y+2):
                        newkp[_x,_y] = 1.0 

    return newkp 


# Decode GTE into a graph.
def decode_gte(gte, properties=None):

    assert properties != None

    # Derive image size from gte shape.
    height, width, _ = gte.shape

    thr = properties["vertex_threshold"]
    edge_thr= properties["edge_threshold"]
    max_degree = properties["max_edge_degree"]
    snap = properties["snap"]
    drop = properties["drop"]
    use_graph_refine = properties["refine"]
    angledistance_weight = properties["snap_weight"]
    snap_dist = properties["snap_distance"]

    # Step-1: Find vertices
    # Step-1 (a): Find vertices through local minima detection. 
    vertexness = gte[:,:,0]
    assert vertexness.shape == (height, width)

    kp = np.copy(vertexness)
    smooth_kp = scipy.ndimage.filters.gaussian_filter(np.copy(kp), 1)
    smooth_kp = smooth_kp / max(np.amax(smooth_kp),0.001)

    keypoints = detect_local_minima(-smooth_kp, smooth_kp, thr)

    # Step-1 (b): There could be a case where the local minima detection algorithm fails
    # to detect some of the vertices. 
    # For example, we have links a<-->b and b<-->c but b is missing. 
    # In this case, we use the edges a-->b and b<--c to recover b.
    # 
    # To do so, we locate the endpoint of each edge (from the detected vertices so far.),
    # draw all the endpoints on a numpy array (a 2D image), blur it, and use the same minima
    # detection algorithm to find vertices. 
    #
    edgeEndpointMap = np.zeros((height, width))

    for i in range(len(keypoints[0])):

        x,y = keypoints[0][i], keypoints[1][i]

        for j in range(max_degree):

            if gte[x,y,2+4*j] * gte[x,y,0] > thr * thr: # or thr < 0.2:
                
                x1 = int(x + vector_norm * gte[x,y,2+4*j+2])
                y1 = int(y + vector_norm * gte[x,y,2+4*j+3])

                if x1 >= 0 and x1 < height and y1 >= 0 and y1 < width:
                    edgeEndpointMap[x1,y1] = gte[x,y,2+4*j] * gte[x,y,0]

    edgeEndpointMap = scipy.ndimage.filters.gaussian_filter(edgeEndpointMap, 3)
    edgeEndpoints = detect_local_minima(-edgeEndpointMap, edgeEndpointMap, thr*thr*thr)

    # Step-1 (c): Create rtree index to speed up the queries.
    # We need to insert the vertices detected in Step-1(a) and Step-1(b) to the rtree.
    # For the vertices detected in Step-1(b), to avoid duplicated vertices, we only 
    # insert them when there are no nearby vertices around them. 
    # 
    idx = index.Index()

    if snap == True:
        # Insert keypoints to the rtree
        for i in range(len(keypoints[0])):
            x,y = keypoints[0][i], keypoints[1][i]
            idx.insert(i,(x-1,y-1,x+1,y+1))

        # Insert edge endpoints (the other vertex of the edge) to the rtree
        # To avoid duplicated vertices, we only insert the vertex when there is no
        # other vertex nearby.
        for i in range(len(edgeEndpoints[0])):

            x,y = edgeEndpoints[0][i], edgeEndpoints[1][i]

            candidates = list(idx.intersection((x-5,y-5,x+5,y+5)))

            if len(candidates) == 0:
                idx.insert(i + len(keypoints[0]),(x-1,y-1,x+1,y+1))


    # Step-2 Connect the vertices to build a graph. 

    # endpoint lookup 
    neighbors = {}

    for i in range(len(keypoints[0])):


        x,y = keypoints[0][i], keypoints[1][i]

        for j in range(max_degree):
            # gte[x,y,2+4*j] --> edgeness
            # gte[x,y,0] --> vertexness
            if gte[x,y,2+4*j] * gte[x,y,0] > thr*edge_thr and gte[x,y,2+4*j] > edge_thr: 
                
                x1 = int(x + vector_norm * gte[x,y,2+4*j+2])
                y1 = int(y + vector_norm * gte[x,y,2+4*j+3])

                skip = False

                l = vector_norm * np.sqrt(gte[x,y,2+4*j+2]*gte[x,y,2+4*j+2] + gte[x,y,2+4*j+3]*gte[x,y,2+4*j+3])

                if snap==True:
                    

                    # We look for a candidate vertex to connect through three passes
                    # Here, we use d(a-->b) to represent the distance metric for edge a-->b .
                    # Pass-1 For a link a<-->b, we connect them only if d(a-->b) + d(a<--b) <= snap_dist.
                    # Pass-2 (relaxed) For a link a<-->b, we connect them only if 2*d(a-->b) <= snap_dist or 2*d(a<--b) <= snap_dist.
                    # Pass-3 (more relaxed) For a link a<-->b, we connect them only if d(a-->b) <= snap_dist or d(a<--b) <= snap_dist.
                    # 
                    # In Pass-1 and Pass-2, we only consider the keypoints detected directly by the minima detection algorithm (Step-1(a)).
                    # In Pass-3, we only consider the edge end points detected in Step-1(b)
                    # 
                    best_candidate = -1 
                    min_distance = snap_dist #15.0 

                    candidates = list(idx.intersection((x1-20,y1-20,x1+20,y1+20)))
                    
                    # Pass-1 (restrict distance metric)
                    for candidate in candidates:
                        # only snap to keypoints 
                        if candidate >= len(keypoints[0]):
                            continue

                        if candidate < len(keypoints[0]):
                            x_c = keypoints[0][candidate]
                            y_c = keypoints[1][candidate]
                        else:
                            x_c = edgeEndpoints[0][candidate-len(keypoints[0])]
                            y_c = edgeEndpoints[1][candidate-len(keypoints[0])]

                        d = distance((x_c,y_c), (x1,y1))
                        if d > l :
                            continue 

                        # vector from the edge endpoint (the other side of the edge) to the current vertex. 
                        v0 = (x - x_c, y - y_c)

                        min_sd = angledistance_weight

                        for jj in range(max_degree):
                            if gte[x_c,y_c,2+4*jj] * gte[x_c,y_c,0] > thr*edge_thr and gte[x,y,2+4*jj] > edge_thr:
                                vc = (vector_norm * gte[x_c,y_c,2+4*jj+2], vector_norm * gte[x_c,y_c,2+4*jj+3])

                                # cosine distance 
                                ad = 1.0 - anglediff(v0,vc)
                                ad = ad * angledistance_weight 

                                if ad < min_sd:
                                    min_sd = ad 

                        d = d + min_sd


                        # cosine distance between the original output edge direction and the edge direction after snapping.  
                        v1 = (x_c - x, y_c - y)
                        v2 = (x1 - x, y1 - y)
                        # cosine distance 
                        ad = 1.0 - anglediff(v1,v2) # -1 to 1

                        d = d + ad * angledistance_weight # 0.15 --> 15 degrees 

                        if d < min_distance:
                            min_distance = d 
                            best_candidate = candidate

                    # Pass-2 (relax the distance metric)
                    min_distance = snap_dist #15.0 
                    # only need the second pass when there is no good candidate found in the first pass. 
                    if best_candidate == -1:
                        for candidate in candidates:
                            # only snap to keypoints 
                            if candidate >= len(keypoints[0]):
                                continue

                            if candidate < len(keypoints[0]):
                                x_c = keypoints[0][candidate]
                                y_c = keypoints[1][candidate]
                            else:
                                x_c = edgeEndpoints[0][candidate-len(keypoints[0])]
                                y_c = edgeEndpoints[1][candidate-len(keypoints[0])]

                            d = distance((x_c,y_c), (x1,y1))
                            if d > l*0.5 :
                                continue 

                            # cosine distance between the original output edge direction and the edge direction after snapping.  
                            v1 = (x_c - x, y_c - y)
                            v2 = (x1 - x, y1 - y)

                            ad = 1.0 - anglediff(v1,v2) # -1 to 1
                            d = d + ad * angledistance_weight * 2 # 0.15 --> 30

                            if d < min_distance:
                                min_distance = d 
                                best_candidate = candidate

                    # Pass-3 (relax the distance metric even more)
                    if best_candidate == -1:
                        for candidate in candidates:
                            # only snap to edge endpoints 
                            if candidate < len(keypoints[0]):
                                continue

                            if candidate < len(keypoints[0]):
                                x_c = keypoints[0][candidate]
                                y_c = keypoints[1][candidate]
                            else:
                                x_c = edgeEndpoints[0][candidate-len(keypoints[0])]
                                y_c = edgeEndpoints[1][candidate-len(keypoints[0])]

                            d = distance((x_c,y_c), (x1,y1))
                            if d > l :
                                continue 


                            v1 = (x_c - x, y_c - y)
                            v2 = (x1 - x, y1 - y)

                            ad = 1.0 - anglediff(v1,v2) # -1 to 1
                            d = d + ad * angledistance_weight # 0.15 --> 15

                            if d < min_distance:
                                min_distance = d 
                                best_candidate = candidate

                    if best_candidate != -1 :
                        if best_candidate < len(keypoints[0]):
                            x1 = keypoints[0][best_candidate]
                            y1 = keypoints[1][best_candidate]
                        else:
                            x1 = edgeEndpoints[0][best_candidate-len(keypoints[0])]
                            y1 = edgeEndpoints[1][best_candidate-len(keypoints[0])]
                    else:
                        skip = True
                            

                # visualization 
                c = int(gte[x,y,2+4*j] * 200.0) + 55
                color = (c,c,c)

                if snap == False: 
                    color = (255-c, 255-c, 255-c)

                w = 2

                # draw the edges and add them in 'neighbors'
                if skip == False or drop==False:

                    nk1 = (x1,y1)
                    nk2 = (x,y)

                    if nk1 != nk2:
                        if nk1 in neighbors:
                            if nk2 in neighbors[nk1]:
                                pass
                            else:
                                neighbors[nk1].append(nk2)
                        else:
                            neighbors[nk1] = [nk2]

                        if  nk2 in neighbors:
                            if nk1 in neighbors[nk2]:
                                pass 
                            else:
                                neighbors[nk2].append(nk1)
                        else:
                            neighbors[nk2] = [nk1]


    graph = neighbors 

    return graph


# Refine and shave graph.
def optimize_graph(graph):
    spurs_thr = 50 
    isolated_thr = 200

    graph = graph_refine(graph, isolated_thr=isolated_thr, spurs_thr=spurs_thr)

    rc = 100
    while rc > 0:
        graph, rc = graph_refine_deloop(graph_refine(graph, isolated_thr=isolated_thr, spurs_thr=spurs_thr))

    graph = graph_shave(graph, spurs_thr = spurs_thr)

    return graph
