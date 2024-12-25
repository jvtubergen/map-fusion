#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:32:19 2017

@author: avanetten

"""

# from __future__ import print_function
# from . import apls_utils
# from . import apls_plots
# from . import osmnx_funcs
# import sp_metric
# import topo_metric
# import apls_tools
import networkx as nx
import scipy.spatial
import scipy.stats
import numpy as np
import random
import utm           # pip install utm
import copy
import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import time
import math
import os
import sys

import argparse
import pandas as pd
import shapely.wkt
# import osmnx as ox   # https://github.com/gboeing/osmnx
# import pickle
# import shutil

path_apls_src = os.path.dirname(os.path.realpath(__file__))
path_apls = os.path.dirname(path_apls_src)
# print("path_apls:", path_apls)
sys.path.append(path_apls_src)
import apls_utils
import apls_plots
import osmnx_funcs
import topo_metric
import sp_metric

# if in docker, the line below may be necessary
# matplotlib.use('agg')

_verbose = False
_plot    = False

def verbose_print(*args):
    global _verbose
    if _verbose:
        print(*args)
    

###############################################################################
def add_travel_time(G_, speed_key='inferred_speed_mps', length_key='length',
                    travel_time_key='travel_time_s', default_speed=13.41,
                    verbose=False):
    """
    Compute and add travel time estimaes to each graph edge.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes speed.
    speed_key : str
        Key in the edge properties dictionary to use for the edge speed.
        Defaults to ``'inferred_speed_mps'``.
    length_key : str
        Key in the edge properties dictionary to use for the edge length.
        Defaults to ``'length'`` (asumed to be in meters).
    travel_time_key : str
        Name to assign travel time in the edge properties dictionary.
        Defaults to ``'travel_time_s'``.
    default_speed : float
        Default speed to use if speed_key is not found in edge properties
        Defaults to ``13.41`` (this is in m/s, and corresponds to 30 mph).
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    G_ : networkx graph
        Updated graph with travel time attached to each edge.
    """

    for i, (u, v, data) in enumerate(G_.edges(data=True)):
        if speed_key in data:
            speed = data[speed_key]
            if type(speed) == list:
                speed = np.mean(speed)
        else:
            speed = default_speed

        verbose_print("data[length_key]:", data[length_key])
        verbose_print("speed:", speed)

        travel_time_seconds = data[length_key] / speed
        data[travel_time_key] = travel_time_seconds

    return G_


###############################################################################
def create_edge_linestrings(G_, remove_redundant=True, verbose=False):
    """
    Ensure all edges have the 'geometry' tag, use shapely linestrings.

    Notes
    -----
    If identical edges exist, remove extras.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that may or may not include 'geometry'.
    remove_redundant : boolean
        Switch to remove identical edges, if they exist.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    G_ : networkx graph
        Updated graph with every edge containing the 'geometry' tag.
    """

    # clean out redundant edges with identical geometry
    edge_seen_set = set([])
    geom_seen = []
    bad_edges = []

    # G_ = G_.copy()
    # for i,(u, v, key, data) in enumerate(G_.edges(keys=True, data=True)):
    for i, (u, v, data) in enumerate(G_.edges(data=True)):
        # create linestring if no geometry reported
        if 'geometry' not in data:
            sourcex, sourcey = G_.nodes[u]['x'],  G_.nodes[u]['y']
            targetx, targety = G_.nodes[v]['x'],  G_.nodes[v]['y']
            line_geom = LineString([Point(sourcex, sourcey),
                                    Point(targetx, targety)])
            data['geometry'] = line_geom

            # get reversed line
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)
            # G_.edges[u][v]['geometry'] = lstring
        else:
            # check which direction linestring is travelling (it may be going
            #   from v -> u, which means we need to reverse the linestring)
            #   otherwise new edge is tangled
            line_geom = data['geometry']
            # print (u,v,key,"create_edge_linestrings() line_geom:", line_geom)
            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            # print "dist_to_u, dist_to_v:", dist_to_u, dist_to_v
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)
            if dist_to_u > dist_to_v:
                # data['geometry'].coords = list(line_geom.coords)[::-1]
                data['geometry'] = line_geom_rev
            # else:
            #    continue

        # flag redundant edges
        if remove_redundant:
            if i == 0:
                edge_seen_set = set([(u, v)])
                edge_seen_set.add((v, u))
                geom_seen.append(line_geom)

            else:
                if ((u, v) in edge_seen_set) or ((v, u) in edge_seen_set):
                    # test if geoms have already been seen
                    for geom_seen_tmp in geom_seen:
                        if (line_geom == geom_seen_tmp) \
                                or (line_geom_rev == geom_seen_tmp):
                            bad_edges.append((u, v))  # , key))
                            verbose_print("\nRedundant edge:", u, v)  # , key)
                else:
                    edge_seen_set.add((u, v))
                    geom_seen.append(line_geom)
                    geom_seen.append(line_geom_rev)

    if remove_redundant:
        verbose_print("\nedge_seen_set:", edge_seen_set)
        verbose_print("redundant edges:", bad_edges)
        for (u, v) in bad_edges:
            if G_.has_edge(u, v):
                G_.remove_edge(u, v)  # , key)

    return G_


###############################################################################
def cut_linestring(line, distance, verbose=False):
    """
    Cuts a shapely linestring at a specified distance from its starting point.

    Notes
    ----
    Return orignal linestring if distance <= 0 or greater than the length of
    the line.
    Reference:
        http://toblerity.org/shapely/manual.html#linear-referencing-methods

    Arguments
    ---------
    line : shapely linestring
        Input shapely linestring to cut.
    distanct : float
        Distance from start of line to cut it in two.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    [line1, line2] : list
        Cut linestrings.  If distance <= 0 or greater than the length of
        the line, return input line.
    """

    verbose_print("Cutting linestring at distance", distance, "...")
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]

    # iterate through coorda and check if interpolated point has been passed
    # already or not
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pdl = line.project(Point(p))
        verbose_print(i, p, "line.project point:", pdl)
        if pdl == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pdl > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]

    # if we've reached here then that means we've encountered a self-loop and
    # the interpolated point is between the final midpoint and the the original
    # node
    i = len(coords) - 1
    cp = line.interpolate(distance)
    return [
        LineString(coords[:i] + [(cp.x, cp.y)]),
        LineString([(cp.x, cp.y)] + coords[i:])]


###############################################################################
def get_closest_edge_from_G(G_, point, nearby_nodes_set=set([]),
                            verbose=False):
    """
    Return closest edge to point, and distance to said edge.

    Notes
    -----
    Just discovered a similar function:
        https://github.com/gboeing/osmnx/blob/master/osmnx/utils.py#L501

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    point : shapely Point
        Shapely point containing (x, y) coordinates.
    nearby_nodes_set : set
        Set of possible edge endpoints to search.  If nearby_nodes_set is not
        empty, only edges with a node in this set will be checked (this can
        greatly speed compuation on large graphs).  If nearby_nodes_set is
        empty, check all possible edges in the graph.
        Defaults to ``set([])``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    best_edge, min_dist, best_geom : tuple
        best_edge is the closest edge to the point
        min_dist is the distance to that edge
        best_geom is the geometry of the ege
    """

    # get distances from point to lines
    dist_list = []
    edge_list = []
    geom_list = []
    p = point  # Point(point_coords)
    for i, (u, v, key, data) in enumerate(G_.edges(keys=True, data=True)):

        # skip if u,v not in nearby nodes
        if len(nearby_nodes_set) > 0:
            if (u not in nearby_nodes_set) and (v not in nearby_nodes_set):
                continue

        verbose_print(("u,v,key,data:", u, v, key, data))
        verbose_print(("  type data['geometry']:", type(data['geometry'])))

        try:
            line = data['geometry']
        except KeyError:
            line = data['attr_dict']['geometry']
        geom_list.append(line)
        dist_list.append(p.distance(line))
        edge_list.append([u, v, key])

    # get closest edge
    min_idx = np.argmin(dist_list)
    min_dist = dist_list[min_idx]
    best_edge = edge_list[min_idx]
    best_geom = geom_list[min_idx]

    return best_edge, min_dist, best_geom


###############################################################################
def insert_point_into_G(G_, point, node_id=100000, max_distance_meters=5,
                        nearby_nodes_set=set([]), allow_renaming=True,
                        verbose=False, super_verbose=False):
    """
    Insert a new node in the graph closest to the given point.

    Notes
    -----
    If the point is too far from the graph, don't insert a node.
    Assume all edges have a linestring geometry
    http://toblerity.org/shapely/manual.html#object.simplify
    Sometimes the point to insert will have the same coordinates as an
    existing point.  If allow_renaming == True, relabel the existing node.
    convert linestring to multipoint?
     https://github.com/Toblerity/Shapely/issues/190

    TODO : Implement a version without renaming that tracks which node is
        closest to the desired point.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    point : shapely Point
        Shapely point containing (x, y) coordinates
    node_id : int
        Unique identifier of node to insert. Defaults to ``100000``.
    max_distance_meters : float
        Maximum distance in meters between point and graph. Defaults to ``5``.
    nearby_nodes_set : set
        Set of possible edge endpoints to search.  If nearby_nodes_set is not
        empty, only edges with a node in this set will be checked (this can
        greatly speed compuation on large graphs).  If nearby_nodes_set is
        empty, check all possible edges in the graph.
        Defaults to ``set([])``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    G_, node_props, min_dist : tuple
        G_ is the updated graph
        node_props gives the properties of the inserted node
        min_dist is the distance from the point to the graph
    """

    # check if node_id already exists in G
    # if node_id in set(G_.nodes()):
    #    print ("node_id:", node_id, "already in G, cannot insert node!")
    #    return

    best_edge, min_dist, best_geom = get_closest_edge_from_G(
            G_, point, nearby_nodes_set=nearby_nodes_set,
            verbose=super_verbose)
    [u, v, key] = best_edge
    G_node_set = set(G_.nodes())

    verbose_print("Inserting point:", node_id)
    verbose_print("best edge:", best_edge)
    verbose_print("  best edge dist:", min_dist)
    u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
    v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
    verbose_print("ploc:", (point.x, point.y))
    verbose_print("uloc:", u_loc)
    verbose_print("vloc:", v_loc)

    if min_dist > max_distance_meters:
        verbose_print("min_dist > max_distance_meters, skipping...")
        return G_, {}, -1, -1

    else:
        # updated graph

        # skip if node exists already
        if node_id in G_node_set:
            verbose_print("Node ID:", node_id, "already exists, skipping...")
            return G_, {}, -1, -1

        # G_.edges[best_edge[0]][best_edge[1]][0]['geometry']
        line_geom = best_geom

        # Length along line that is closest to the point
        line_proj = line_geom.project(point)

        # Now combine with interpolated point on line
        new_point = line_geom.interpolate(line_geom.project(point))
        x, y = new_point.x, new_point.y

        #################
        # create new node
        
        try:
            # first get zone, then convert to latlon
            _, _, zone_num, zone_letter = utm.from_latlon(G_.nodes[u]['lat'],
                                                          G_.nodes[u]['lon'])
            # convert utm to latlon
            lat, lon = utm.to_latlon(x, y, zone_num, zone_letter)
        except:
            lat, lon = y, x

        # set properties
        # props = G_.nodes[u]
        node_props = {'highway': 'insertQ',
                      'lat':     lat,
                      'lon':     lon,
                      'osmid':   node_id,
                      'x':       x,
                      'y':       y}
        # add node
        G_.add_node(node_id, **node_props)

        # assign, then update edge props for new edge
        _, _, edge_props_new = copy.deepcopy(
            list(G_.edges([u, v], data=True))[0])
        # remove extraneous 0 key

        # print ("edge_props_new.keys():", edge_props_new)
        # if list(edge_props_new.keys()) == [0]:
        #    edge_props_new = edge_props_new[0]

        # cut line
        split_line = cut_linestring(line_geom, line_proj)
        # line1, line2, cp = cut_linestring(line_geom, line_proj)
        if split_line is None:
            verbose_print("Failure in cut_linestring()...")
            verbose_print("type(split_line):", type(split_line))
            verbose_print("split_line:", split_line)
            verbose_print("line_geom:", line_geom)
            verbose_print("line_geom.length:", line_geom.length)
            verbose_print("line_proj:", line_proj)
            verbose_print("min_dist:", min_dist)
            return G_, {}, 0, 0

        if _verbose:
            verbose_print("split_line:", split_line)

        # if cp.is_empty:
        if len(split_line) == 1:
            verbose_print("split line empty, min_dist:", min_dist)
            # get coincident node
            outnode = ''
            outnode_x, outnode_y = -1, -1
            x_p, y_p = new_point.x, new_point.y
            x_u, y_u = G_.nodes[u]['x'], G_.nodes[u]['y']
            x_v, y_v = G_.nodes[v]['x'], G_.nodes[v]['y']
            # if _verbose:
            #    print "x_p, y_p:", x_p, y_p
            #    print "x_u, y_u:", x_u, y_u
            #    print "x_v, y_v:", x_v, y_v

            # sometimes it seems that the nodes aren't perfectly coincident,
            # so see if it's within a buffer
            buff = 0.05  # meters
            if (abs(x_p - x_u) <= buff) and (abs(y_p - y_u) <= buff):
                outnode = u
                outnode_x, outnode_y = x_u, y_u
            elif (abs(x_p - x_v) <= buff) and (abs(y_p - y_v) <= buff):
                outnode = v
                outnode_x, outnode_y = x_v, y_v
            # original method with exact matching
            # if (x_p == x_u) and (y_p == y_u):
            #    outnode = u
            #    outnode_x, outnode_y = x_u, y_u
            # elif (x_p == x_v) and (y_p == y_v):
            #    outnode = v
            #    outnode_x, outnode_y = x_v, y_v
            else:
                print("Error in determining node coincident with node: "
                      + str(node_id) + " along edge: " + str(best_edge))
                verbose_print("x_p, y_p:", x_p, y_p)
                verbose_print("x_u, y_u:", x_u, y_u)
                verbose_print("x_v, y_v:", x_v, y_v)
                # return
                return G_, {}, 0, 0

            # if the line cannot be split, that means that the new node
            # is coincident with an existing node.  Relabel, if desired
            if allow_renaming:
                node_props = G_.nodes[outnode]
                # A dictionary with the old labels as keys and new labels
                #  as values. A partial mapping is allowed.
                mapping = {outnode: node_id}
                Gout = nx.relabel_nodes(G_, mapping)
                verbose_print("Swapping out node ids:", mapping)
                return Gout, node_props, x_p, y_p

            else:
                # new node is already added, presumably at the exact location
                # of an existing node.  So just remove the best edge and make
                # an edge from new node to existing node, length should be 0.0

                line1 = LineString([new_point, Point(outnode_x, outnode_y)])
                edge_props_line1 = edge_props_new.copy()
                edge_props_line1['length'] = line1.length
                edge_props_line1['geometry'] = line1
                # make sure length is zero
                if line1.length > buff:
                    verbose_print("Nodes should be coincident and length 0!")
                    verbose_print("  line1.length:", line1.length)
                    verbose_print("  x_u, y_u :", x_u, y_u)
                    verbose_print("  x_v, y_v :", x_v, y_v)
                    verbose_print("  x_p, y_p :", x_p, y_p)
                    verbose_print("  new_point:", new_point)
                    verbose_print("  Point(outnode_x, outnode_y):",
                          Point(outnode_x, outnode_y))
                    return

                # add edge of length 0 from new node to neareest existing node
                G_.add_edge(node_id, outnode, **edge_props_line1)
                return G_, node_props, x, y

                # originally, if not renaming nodes,
                # just ignore this complication and return the orignal
                # return G_, node_props, 0, 0

        else:
            # else, create new edges
            line1, line2 = split_line

            # get distances
            # print ("insert_point(), G_.nodes[v]:", G_.nodes[v])
            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
            # compare to first point in linestring
            geom_p0 = list(line_geom.coords)[0]
            # or compare to inserted point? [this might fail if line is very
            #    curved!]
            # geom_p0 = (x,y)
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            # reverse edge order if v closer than u
            if dist_to_v < dist_to_u:
                line2, line1 = split_line

            verbose_print("Creating two edges from split...")
            verbose_print("   original_length:", line_geom.length)
            verbose_print("   line1_length:", line1.length)
            verbose_print("   line2_length:", line2.length)
            verbose_print("   u, dist_u_to_point:", u, dist_to_u)
            verbose_print("   v, dist_v_to_point:", v, dist_to_v)
            verbose_print("   min_dist:", min_dist)

            # add new edges
            edge_props_line1 = edge_props_new.copy()
            edge_props_line1['length'] = line1.length
            edge_props_line1['geometry'] = line1
            # remove geometry?
            # edge_props_line1.pop('geometry', None)
            # line2
            edge_props_line2 = edge_props_new.copy()
            edge_props_line2['length'] = line2.length
            edge_props_line2['geometry'] = line2
            # remove geometry?
            # edge_props_line1.pop('geometry', None)

            # insert edge regardless of direction
            # G_.add_edge(u, node_id, **edge_props_line1)
            # G_.add_edge(node_id, v, **edge_props_line2)

            # check which direction linestring is travelling (it may be going
            # from v -> u, which means we need to reverse the linestring)
            # otherwise new edge is tangled
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            # if _verbose:
            #    print "dist_to_u, dist_to_v:", dist_to_u, dist_to_v
            if dist_to_u < dist_to_v:
                G_.add_edge(u, node_id, **edge_props_line1)
                G_.add_edge(node_id, v, **edge_props_line2)
            else:
                G_.add_edge(node_id, u, **edge_props_line1)
                G_.add_edge(v, node_id, **edge_props_line2)

            verbose_print("insert edges:", u, '-', node_id, 'and', node_id, '-', v)

            # remove initial edge
            G_.remove_edge(u, v, key)

            return G_, node_props, x, y


###############################################################################
def insert_control_points(G_, control_points, max_distance_meters=10,
                          allow_renaming=True,
                          n_nodes_for_kd=1000, n_neighbors=20,
                          x_coord='x', y_coord='y',
                          verbose=True, super_verbose=False):
    """
    Wrapper around insert_point_into_G() for all control_points.

    Notes
    -----
    control_points are assumed to be of the format:
        [[node_id, x, y], ... ]

    TODO : Implement a version without renaming that tracks which node is
        closest to the desired point.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    control_points : array
        Points to insert in the graph, assumed to the of the format:
            [[node_id, x, y], ... ]
    max_distance_meters : float
        Maximum distance in meters between point and graph. Defaults to ``5``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    n_nodes_for_kd : int
        Minumu size of graph to render to kdtree to speed node placement.
        Defaults to ``1000``.
    n_neighbors : int
        Number of neigbors to return if building a kdtree. Defaults to ``20``.
    x_coord : str
        Name of x_coordinate, can be 'x' or 'lon'. Defaults to ``'x'``.
    y_coord : str
        Name of y_coordinate, can be 'y' or 'lat'. Defaults to ``'y'``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    Gout, new_xs, new_ys : tuple
        Gout is the updated graph
        new_xs, new_ys are coordinates of the inserted points
    """

    t0 = time.time()

    # insertion can be super slow so construct kdtree if a large graph
    if len(G_.nodes()) > n_nodes_for_kd:
        # construct kdtree of ground truth
        kd_idx_dic, kdtree, pos_arr = apls_utils.G_to_kdtree(G_)
    # print("kd_idx_dic:", kd_idx_dic)
    # print("kdtree:", kdtree)
    # print("pos_arr:", pos_arr)

    Gout = G_.copy()
    new_xs, new_ys = [], []
    if len(G_.nodes()) == 0:
        return Gout, new_xs, new_ys

    for i, [node_id, x, y] in enumerate(control_points):
        
        if math.isinf(x) or math.isinf(y):
            verbose_print("Infinity in coords!:", x, y)
            return
        
        # if _verbose:
        if (i % 20) == 0:
            verbose_print(i, "/", len(control_points),
                  "Insert control point:", node_id, "x =", x, "y =", y)
        point = Point(x, y)

        # if large graph, determine nearby nodes
        if len(G_.nodes()) > n_nodes_for_kd:
            # get closest nodes
            node_names, dists_m_refine = apls_utils.nodes_near_point(
                    x, y, kdtree, kd_idx_dic, x_coord=x_coord, y_coord=y_coord,
                    # radius_m=radius_m,
                    n_neighbors=n_neighbors,
                    verbose=False)
            nearby_nodes_set = set(node_names)
        else:
            nearby_nodes_set = set([])

        # insert point
        Gout, node_props, xnew, ynew = insert_point_into_G(
            Gout, point, node_id=node_id,
            max_distance_meters=max_distance_meters,
            nearby_nodes_set=nearby_nodes_set,
            allow_renaming=allow_renaming,
            verbose=super_verbose)
        # xnew = node_props['x']
        # ynew = node_props['y']
        if (x != 0) and (y != 0):
            new_xs.append(xnew)
            new_ys.append(ynew)

    t1 = time.time()
    verbose_print("Time to run insert_control_points():", t1-t0, "seconds")
    return Gout, new_xs, new_ys


###############################################################################
def create_graph_midpoints(G_, linestring_delta=50, is_curved_eps=0.03,
                           n_id_add_val=1, allow_renaming=True,
                           figsize=(0, 0),
                           verbose=False, super_verbose=False):
    """
    Insert midpoint nodes into long edges on the graph.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    linestring_delta : float
        Distance in meters between linestring midpoints. Defaults to ``50``.
    is_curved_eps : float
        Minumum curvature for injecting nodes (if curvature is less than this
        value, no midpoints will be injected). If < 0, always inject points
        on line, regardless of curvature.  Defaults to ``0.3``.
    n_id_add_val : int
        Sets min midpoint id above existing nodes
        e.g.: G.nodes() = [1,2,4], if n_id_add_val = 5, midpoints will
        be [9,10,11,...]
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    figsize : tuple
        Figure size for optional plot. Defaults to ``(0,0)`` (no plot).
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    Gout, xms, yms : tuple
        Gout is the updated graph
        xms, yms are coordinates of the inserted points
    """

    # midpoint_loc = 0.5        # take the central midpoint for straight lines
    if len(G_.nodes()) == 0:
        return G_, [], []

    # midpoints
    xms, yms = [], []
    Gout = G_.copy()
    # midpoint_name_val, midpoint_name_inc = 0.01, 0.01
    midpoint_name_val, midpoint_name_inc = np.max(G_.nodes())+n_id_add_val, 1
    # for u, v, key, data in G_.edges(keys=True, data=True):
    for u, v, data in G_.edges(data=True):

        # curved line
        if 'geometry' in data:

            # first edge props and  get utm zone and letter
            edge_props_init = G_.edges([u, v])
            # _, _, zone_num, zone_letter = utm.from_latlon(G_.nodes[u]['lat'],
            #                                              G_.nodes[u]['lon'])

            linelen = data['length']
            line = data['geometry']

            xs, ys = line.xy  # for plotting

            #################
            # check if curved or not
            minx, miny, maxx, maxy = line.bounds
            # get euclidean distance
            dst = scipy.spatial.distance.euclidean([minx, miny], [maxx, maxy])
            # ignore if almost straight
            if np.abs(dst - linelen) / linelen < is_curved_eps:
                # print "Line straight, skipping..."
                continue
            #################

            #################
            # also ignore super short lines
            if linelen < 0.75*linestring_delta:
                # print "Line too short, skipping..."
                continue
            #################

            verbose_print("create_graph_midpoints()...")
            verbose_print("  u,v:", u, v)
            verbose_print("  data:", data)
            verbose_print("  edge_props_init:", edge_props_init)

            # interpolate midpoints
            # if edge is short, use midpoint, else get evenly spaced points
            if linelen <= linestring_delta:
                interp_dists = [0.5 * line.length]
            else:
                # get evenly spaced points
                npoints = len(np.arange(0, linelen, linestring_delta)) + 1
                interp_dists = np.linspace(0, linelen, npoints)[1:-1]
                verbose_print("  interp_dists:", interp_dists)

            # create nodes
            node_id_new_list = []
            xms_tmp, yms_tmp = [], []
            for j, d in enumerate(interp_dists):
                verbose_print("    ", j, "interp_dist:", d)

                midPoint = line.interpolate(d)
                xm0, ym0 = midPoint.xy
                xm = xm0[-1]
                ym = ym0[-1]
                point = Point(xm, ym)
                xms.append(xm)
                yms.append(ym)
                xms_tmp.append(xm)
                yms_tmp.append(ym)
                verbose_print("    midpoint:", xm, ym)

                # add node to graph, with properties of u
                node_id = midpoint_name_val
                # node_id = np.round(u + midpoint_name_val,2)
                midpoint_name_val += midpoint_name_inc
                node_id_new_list.append(node_id)
                verbose_print("    node_id:", node_id)

                # if j > 3:
                #    continue

                # add to graph
                Gout, node_props, _, _ = insert_point_into_G(
                    Gout, point, node_id=node_id,
                    allow_renaming=allow_renaming,
                    verbose=super_verbose)

        # plot, if desired
        if figsize != (0, 0):
            fig, (ax) = plt.subplots(1, 1, figsize=(1*figsize[0], figsize[1]))
            ax.plot(xs, ys, color='#6699cc', alpha=0.7,
                    linewidth=3, solid_capstyle='round', zorder=2)
            ax.scatter(xm, ym, color='red')
            ax.set_title('Line Midpoint')
            plt.axis('equal')

    return Gout, xms, yms


###############################################################################
def _clean_sub_graphs(G_, min_length=80, max_nodes_to_skip=100,
                      weight='length', verbose=True,
                      super_verbose=False):
    """
    Remove subgraphs with a max path length less than min_length,
    if the subgraph has more than max_noxes_to_skip, don't check length
       (this step great reduces processing time)
    """

    if len(G_.nodes()) == 0:
        return G_

    verbose_print("Running clean_sub_graphs...")
    sub_graphs = list(nx.connected_component_subgraphs(G_))
    bad_nodes = []
    verbose_print(" len(G_.nodes()):", len(G_.nodes()))
    verbose_print(" len(G_.edges()):", len(G_.edges()))
    verbose_print("G_.nodes:", G_.nodes())
    edge_tmp = G_.edges()[np.random.randint(len(G_.edges()))]
    verbose_print(edge_tmp, "G.edge props:", G_.edges[edge_tmp[0]][edge_tmp[1]])

    for G_sub in sub_graphs:
        # don't check length if too many nodes in subgraph
        if len(G_sub.nodes()) > max_nodes_to_skip:
            continue

        else:
            all_lengths = dict(
                nx.all_pairs_dijkstra_path_length(G_sub, weight=weight))
            verbose_print("  \nGs.nodes:", G_sub.nodes())
            verbose_print("  all_lengths:", all_lengths)
            # get all lenghts
            lens = []

            # for u,v in all_lengths.iteritems():
            for u in all_lengths.keys():
                v = all_lengths[u]
                # for uprime, vprime in v.iteritems():
                for uprime in v.keys():
                    vprime = v[uprime]
                    lens.append(vprime)
                    verbose_print("  u, v", u, v)
                    verbose_print("    uprime, vprime:", uprime, vprime)
            max_len = np.max(lens)
            verbose_print("  Max length of path:", max_len)
            if max_len < min_length:
                bad_nodes.extend(G_sub.nodes())
                verbose_print(" appending to bad_nodes:", G_sub.nodes())

    # remove bad_nodes
    G_.remove_nodes_from(bad_nodes)
    verbose_print(" num bad_nodes:", len(bad_nodes))
    verbose_print(" len(G'.nodes()):", len(G_.nodes()))
    verbose_print(" len(G'.edges()):", len(G_.edges()))
    verbose_print("  G_.nodes:", G_.nodes())

    return G_


###############################################################################
def make_graphs(G_gt_, G_p_,
                weight='length',
                speed_key='inferred_speed_mps',
                travel_time_key='travel_time_s',
                max_nodes_for_midpoints=500,
                linestring_delta=50,
                is_curved_eps=0.012,
                max_snap_dist=4,
                allow_renaming=True,
                verbose=False,
                super_verbose=False):
    """
    Match nodes in ground truth and propsal graphs, and get paths.

    Notes
    -----
    The path length dictionaries returned by this function will be fed into
    compute_metric().

    Arguments
    ---------
    G_gt_ : networkx graph
        Ground truth graph.
    G_p_ : networkd graph
        Proposal graph over the same region.
    weight : str
        Key in the edge properties dictionary to use for the path length
        weight.  Defaults to ``'length'``.
    speed_key : str
        Key in the edge properties dictionary to use for the edge speed.
        Defaults to ``'inferred_speed_mps'``.
    travel_time_key : str
        Name to assign travel time in the edge properties dictionary.
        Defaults to ``'travel_time_s'``.
    max_nodes_for_midpoints : int
        Maximum number of gt nodes to inject midpoints.  If there are more
        gt nodes than this, skip midpoints and use this number of points
        to comput APLS.
    linestring_delta : float
        Distance in meters between linestring midpoints.
        If len gt nodes > max_nodes_for_midppoints this argument is ignored.
        Defaults to ``50``.
    is_curved_eps : float
        Minumum curvature for injecting nodes (if curvature is less than this
        value, no midpoints will be injected). If < 0, always inject points
        on line, regardless of curvature.
        If len gt nodes > max_nodes_for_midppoints this argument is ignored.
        Defaults to ``0.012``.
    max_snap_dist : float
        Maximum distance a node can be snapped onto a graph.
        Defaults to ``4``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Return
    ------
    G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
            control_points_gt, control_points_prop, \
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime : tuple
        G_gt_cp  is ground truth with control points inserted
        G_p_cp is proposal with control points inserted
        G_gt_cp_prime is ground truth with control points from prop inserted
        G_p_cp_prime is proposal with control points from gt inserted
        all_pairs_lengths_gt_native is path length dict corresponding to G_gt_cp
        all_pairs_lengths_prop_native is path length dict corresponding to G_p_cp
        all_pairs_lengths_gt_prime is path length dict corresponding to G_gt_cp_prime
        all_pairs_lenfgths_prop_prime is path length dict corresponding to G_p_cp_prime 
    """

    t0 = time.time()
    verbose_print("Executing make_graphs()...")

    verbose_print("Ensure", weight, "in gt graph prpperties")
    verbose_print("type(G_gt_)", type(G_gt_))
    for i, (u, v, data) in enumerate(G_gt_.edges(keys=False, data=True)):
        verbose_print("G_gt_.edges[u, v]:", G_gt_.edges[u, v])
        verbose_print("G_gt_.edges[u, v][weight]:", G_gt_.edges[u, v][weight])
        if weight not in data.keys():
            print("Error!", weight, "not in G_gt_ edge u, v, data:", u, v, data)
            return

    verbose_print("Ensure G_gt 'geometry' is a shapely geometry, not a linestring...")
    for i, (u, v, key, data) in enumerate(G_gt_.edges(keys=True, data=True)):
        if i == 0:
            verbose_print(("u,v,key,data:", u, v, key, data))
            verbose_print(("  type data['geometry']:", type(data['geometry'])))
        try:
            line = data['geometry']
        except KeyError:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    # create graph with midpoints
    G_gt0 = create_edge_linestrings(G_gt_.to_undirected())

    verbose_print("len G_gt.nodes():", len(list(G_gt0.nodes())))
    verbose_print("len G_gt.edges():", len(list(G_gt0.edges())))

    verbose_print("Creating gt midpoints")
    G_gt_cp0, xms, yms = create_graph_midpoints(
        G_gt0.copy(),
        linestring_delta=linestring_delta,
        figsize=(0, 0),
        is_curved_eps=is_curved_eps,
        verbose=False)
    # add travel time
    G_gt_cp = add_travel_time(G_gt_cp0.copy(),
                              speed_key=speed_key,
                              travel_time_key=travel_time_key)

    # get ground truth control points
    control_points_gt = []
    for n in G_gt_cp.nodes():
        u_x, u_y = G_gt_cp.nodes[n]['x'], G_gt_cp.nodes[n]['y']
        control_points_gt.append([n, u_x, u_y])
    
    verbose_print("len control_points_gt:", len(control_points_gt))

    # get ground truth paths
    verbose_print("Get ground truth paths...")
    all_pairs_lengths_gt_native = dict(
        nx.shortest_path_length(G_gt_cp, weight=weight))
    # all_pairs_lengths_gt_native = dict(
    #    nx.all_pairs_dijkstra_path_length(G_gt_cp, weight=weight))
    ###############

    ###############
    # Proposal

    verbose_print("Ensure", weight, "in prop graph prpperties")
    verbose_print("type(G_p_)", type(G_p_))
    for i, (u, v, data) in enumerate(G_p_.edges(keys=False, data=True)):
        if weight not in data.keys():
            print("Error!", weight, "not in G_p_ edge u, v, data:", u, v, data)
            return

    # get proposal graph with native midpoints
    verbose_print("Ensure G_p 'geometry' is a shapely geometry, not a linestring...")
    for i, (u, v, key, data) in enumerate(G_p_.edges(keys=True, data=True)):
        if i == 0:
            verbose_print(("u,v,key,data:", u, v, key, data))
            verbose_print(("  type data['geometry']:", type(data['geometry'])))
        try:
            line = data['geometry']
        except:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    G_p0 = create_edge_linestrings(G_p_.to_undirected())
    # add travel time
    G_p = add_travel_time(G_p0.copy(),
                          speed_key=speed_key,
                          travel_time_key=travel_time_key)

    verbose_print("len G_p.nodes():", len(G_p.nodes()))
    verbose_print("len G_p.edges():", len(G_p.edges()))

    verbose_print("Creating proposal midpoints")
    G_p_cp0, xms_p, yms_p = create_graph_midpoints(
        G_p.copy(),
        linestring_delta=linestring_delta,
        figsize=(0, 0),
        is_curved_eps=is_curved_eps,
        verbose=False)
    # add travel time
    G_p_cp = add_travel_time(G_p_cp0.copy(),
                             speed_key=speed_key,
                             travel_time_key=travel_time_key)

    verbose_print("len G_p_cp.nodes():", len(G_p_cp.nodes()))
    verbose_print("len G_p_cp.edges():", len(G_p_cp.edges()))

    # set proposal control nodes, originally just all nodes in G_p_cp
    # original method sets proposal control points as all nodes in G_p_cp
    # get proposal control points
    control_points_prop = []
    for n in G_p_cp.nodes():
        u_x, u_y = G_p_cp.nodes[n]['x'], G_p_cp.nodes[n]['y']
        control_points_prop.append([n, u_x, u_y])

    # get paths
    all_pairs_lengths_prop_native = dict(
        nx.shortest_path_length(G_p_cp, weight=weight))
        # nx.all_pairs_dijkstra_path_length(G_p_cp, weight=weight))

    ###############
    # insert gt control points into proposal
    verbose_print("Inserting", len(control_points_gt),
            "control points into G_p...")
    verbose_print("G_p.nodes():", G_p.nodes())
    G_p_cp_prime0, xn_p, yn_p = insert_control_points(
        G_p.copy(), control_points_gt,
        max_distance_meters=max_snap_dist,
        allow_renaming=allow_renaming,
        verbose=super_verbose)
    # add travel time
    G_p_cp_prime = add_travel_time(G_p_cp_prime0.copy(),
                                   speed_key=speed_key,
                                   travel_time_key=travel_time_key)
    # G_p_cp, xn_p, yn_p = insert_control_points(G_p_cp, control_points_gt,
    #                                     max_distance_meters=max_snap_dist,
    #                                     allow_renaming=allow_renaming,
    #                                     verbose=verbose)

    ###############
    # now insert control points into ground truth
    verbose_print("\nInserting", len(control_points_prop),
              "control points into G_gt...")
    # permit renaming of inserted nodes if coincident with existing node
    G_gt_cp_prime0, xn_gt, yn_gt = insert_control_points(
        G_gt_,
        control_points_prop,
        max_distance_meters=max_snap_dist,
        allow_renaming=allow_renaming,
        verbose=super_verbose)
    # add travel time
    G_gt_cp_prime = add_travel_time(G_gt_cp_prime0.copy(),
                                    speed_key=speed_key,
                                    travel_time_key=travel_time_key)

    ###############
    # get paths
    all_pairs_lengths_gt_prime = dict(
        nx.shortest_path_length(G_gt_cp_prime, weight=weight))
        # nx.all_pairs_dijkstra_path_length(G_gt_cp_prime, weight=weight))
    all_pairs_lengths_prop_prime = dict(
        nx.shortest_path_length(G_p_cp_prime, weight=weight))
        # nx.all_pairs_dijkstra_path_length(G_p_cp_prime, weight=weight))

    tf = time.time()
    verbose_print("Time to run make_graphs in apls.py:", tf - t0, "seconds")

    return G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
        control_points_gt, control_points_prop, \
        all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
        all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime


###############################################################################
def make_graphs_huge(G_gt, G_p,
                     weight='length',
                     speed_key='inferred_speed_mps',
                     travel_time_key='travel_time_s',
                     max_nodes=500,
                     max_snap_dist=4,
                     allow_renaming=True,
                     verbose=True, super_verbose=False):
    """
    Match nodes in large ground truth and propsal graphs, and get paths.

    Notes
    -----
    Skip midpoint injection and only select a subset of routes to compare.
    The path length dictionaries returned by this function will be fed into
    compute_metric().

    Arguments
    ---------
    G_gt : networkx graph
        Ground truth graph.
    G_p : networkd graph
        Proposal graph over the same region.
    weight : str
        Key in the edge properties dictionary to use for the path length
        weight.  Defaults to ``'length'``.
    speed_key : str
        Key in the edge properties dictionary to use for the edge speed.
        Defaults to ``'inferred_speed_mps'``.
    travel_time_key : str
        Name to assign travel time in the edge properties dictionary.
        Defaults to ``'travel_time_s'``.
    max_nodess : int
        Maximum number of gt nodes to inject midpoints.  If there are more
        gt nodes than this, skip midpoints and use this number of points
        to comput APLS.
    max_snap_dist : float
        Maximum distance a node can be snapped onto a graph.
        Defaults to ``4``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Return
    ------
    G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
            control_points_gt, control_points_prop, \
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime : tuple
        G_gt_cp  is ground truth with control points inserted
        G_p_cp is proposal with control points inserted
        G_gt_cp_prime is ground truth with control points from prop inserted
        G_p_cp_prime is proposal with control points from gt inserted
        all_pairs_lengths_gt_native is path length dict corresponding to G_gt_cp
        all_pairs_lengths_prop_native is path length dict corresponding to G_p_cp
        all_pairs_lengths_gt_prime is path length dict corresponding to G_gt_cp_prime
        all_pairs_lenfgths_prop_prime is path length dict corresponding to G_p_cp_prime 
    """
    
    t0 = time.time()
    verbose_print("Executing make_graphs_huge()...")

    verbose_print("Ensure G_gt 'geometry' is a shapely geometry, not a linestring...")
    for i, (u, v, key, data) in enumerate(G_gt.edges(keys=True, data=True)):
        if i == 0:
            try:
                verbose_print(("u,v,key,data:", u, v, key, data))
            except:
                pass
            verbose_print(("  type data['geometry']:", type(data['geometry'])))
        try:
            line = data['geometry']
        except:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    verbose_print("Ensure G_p 'geometry' is a shapely geometry, not a linestring...")
    for i, (u, v, key, data) in enumerate(G_p.edges(keys=True, data=True)):
        if i == 0:
            verbose_print(("u,v,key,data:", u, v, key, data))
            verbose_print(("  type data['geometry']:", type(data['geometry'])))
        try:
            line = data['geometry']
        except:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    # create graph with linestrings?
    G_gt_cp = G_gt.to_undirected()
    # G_gt_cp = create_edge_linestrings(G_gt.to_undirected())

    if _verbose:
        verbose_print("len(G_gt.nodes()):", len(G_gt_cp.nodes()))
        verbose_print("len(G_gt.edges()):", len(G_gt_cp.edges()))
        # verbose_print("G_gt.nodes():", G_gt_cp.nodes())
        # verbose_print("G_gt.edges()):", G_gt_cp.edges())
        # gt node and edge props
        node = random.choice(list(G_gt.nodes()))
        verbose_print("node:", node, "G_gt random node props:", G_gt.nodes[node])
        edge_tmp = random.choice(list(G_gt.edges()))
        verbose_print("G_gt edge_tmp:", edge_tmp)
        try:
            verbose_print("edge:", edge_tmp, "G_gt random edge props:",
                  G_gt.edges[edge_tmp[0]][edge_tmp[1]])
        except:
            try:
                verbose_print("edge:", edge_tmp, "G_gt random edge props:",
                  G_gt.edges[edge_tmp[0], edge_tmp[1], 0])
            except:
                pass
        # prop node and edge props
        node = random.choice(list(G_p.nodes()))
        verbose_print("node:", node, "G_p random node props:", G_p.nodes[node])
        edge_tmp = random.choice(list(G_p.edges()))
        verbose_print("G_p edge_tmp:", edge_tmp)
        try:
            verbose_print("edge:", edge_tmp, "G_p random edge props:",
                  G_p.edges[edge_tmp[0]][edge_tmp[1]])
        except:
            try:
                verbose_print("edge:", edge_tmp, "G_p random edge props:",
                  G_p.edges[edge_tmp[0], edge_tmp[1], 0])
            except:
                pass

    # get ground truth control points, which will be a subset of nodes
    sample_size = min(max_nodes, len(G_gt_cp.nodes()))
    rand_nodes_gt = random.sample(list(G_gt_cp.nodes()), sample_size)
    rand_nodes_gt_set = set(rand_nodes_gt)
    control_points_gt = []
    for itmp,n in enumerate(rand_nodes_gt):
        if _verbose and (i % 20) == 0:
            print ("control_point", itmp, ":", n, ":", G_gt_cp.nodes[n])
        u_x, u_y = G_gt_cp.nodes[n]['x'], G_gt_cp.nodes[n]['y']
        control_points_gt.append([n, u_x, u_y])
    if _verbose:
        verbose_print("len control_points_gt:", len(control_points_gt))
    # add travel time
    G_gt_cp = add_travel_time(G_gt_cp,
                              speed_key=speed_key,
                              travel_time_key=travel_time_key)

    # get route lengths between all control points
    # gather all paths from nodes of interest, keep only routes to control nodes
    tt = time.time()
    if _verbose:
        verbose_print("Computing all_pairs_lengths_gt_native...")
    all_pairs_lengths_gt_native = {}
    for itmp, source in enumerate(rand_nodes_gt):
        if _verbose and ((itmp % 50) == 0):
            verbose_print((itmp, "source:", source))
        paths_tmp = nx.single_source_dijkstra_path_length(
            G_gt_cp, source, weight=weight)
        # delete items
        for k in list(paths_tmp.keys()):
            if k not in rand_nodes_gt_set:
                del paths_tmp[k]
        all_pairs_lengths_gt_native[source] = paths_tmp
    if _verbose:
        verbose_print(("Time to compute all source routes for",
               sample_size, "nodes:", time.time() - tt, "seconds"))

    # get individual routes (super slow!)
    #t0 = time.time()
    #all_pairs_lengths_gt_native = {}
    # for source in rand_nodes_gt:
    #    print ("source:", source)
    #    source_dic = {}
    #    for target in rand_nodes_gt:
    #        print ("target:", target)
    #        p = nx.dijkstra_path_length(G_gt_init, source, target, weight=weight)
    #        source_dic[target] = p
    #    all_pairs_lengths_gt_native[source] = source_dic
    #print ("Time to compute all source routes:", time.time() - t0, "seconds")
    ## ('Time to compute all source routes:', 9.418055057525635, 'seconds')

    #all_pairs_lengths_gt_native = nx.all_pairs_dijkstra_path_length(G_gt_cp, weight=weight)
    ###############

    ###############
    # get proposal graph with native midpoints
    G_p_cp = G_p.to_undirected()
    #G_p_cp = create_edge_linestrings(G_p.to_undirected())
    if _verbose:
        verbose_print("len G_p_cp.nodes():", len(G_p_cp.nodes()))
        verbose_print("G_p_cp.edges():", len(G_p_cp.edges()))

    # get control points, which will be a subset of nodes
    # (original method sets proposal control points as all nodes in G_p_cp)
    sample_size = min(max_nodes, len(G_p_cp.nodes()))
    rand_nodes_p = random.sample(list(G_p_cp.nodes()), sample_size)
    rand_nodes_p_set = set(rand_nodes_p)
    control_points_prop = []
    for n in rand_nodes_p:
        u_x, u_y = G_p_cp.nodes[n]['x'], G_p_cp.nodes[n]['y']
        control_points_prop.append([n, u_x, u_y])
    # add travel time
    G_p_cp = add_travel_time(G_p_cp,
                             speed_key=speed_key,
                             travel_time_key=travel_time_key)

    # get paths
    # gather all paths from nodes of interest, keep only routes to control nodes
    tt = time.time()
    if _verbose:
        verbose_print("Computing all_pairs_lengths_prop_native...")
    all_pairs_lengths_prop_native = {}
    for itmp, source in enumerate(rand_nodes_p):
        if _verbose and ((itmp % 50) == 0):
            verbose_print((itmp, "source:", source))
        paths_tmp = nx.single_source_dijkstra_path_length(
            G_p_cp, source, weight=weight)
        # delete items
        for k in list(paths_tmp.keys()):
            if k not in rand_nodes_p_set:
                del paths_tmp[k]
        all_pairs_lengths_prop_native[source] = paths_tmp
    if _verbose:
        verbose_print(("Time to compute all source routes for",
               max_nodes, "nodes:", time.time() - tt, "seconds"))

    ###############
    # insert gt control points into proposal
    if _verbose:
        verbose_print("Inserting", len(control_points_gt),
              "control points into G_p...")
        verbose_print("len G_p.nodes():", len(G_p.nodes()))
    G_p_cp_prime, xn_p, yn_p = insert_control_points(
        G_p.copy(), control_points_gt, max_distance_meters=max_snap_dist,
        allow_renaming=allow_renaming, verbose=super_verbose)
    # add travel time
    G_p_cp_prime = add_travel_time(G_p_cp_prime,
                                   speed_key=speed_key,
                                   travel_time_key=travel_time_key)

    ###############
    # now insert control points into ground truth
    if _verbose:
        verbose_print("\nInserting", len(control_points_prop),
              "control points into G_gt...")
    # permit renaming of inserted nodes if coincident with existing node
    G_gt_cp_prime, xn_gt, yn_gt = insert_control_points(
        G_gt, control_points_prop, max_distance_meters=max_snap_dist,
        allow_renaming=allow_renaming, verbose=super_verbose)
    G_gt_cp_prime = add_travel_time(G_gt_cp_prime,
                                    speed_key=speed_key,
                                    travel_time_key=travel_time_key)

    ###############
    # get paths for graphs_prime
    # gather all paths from nodes of interest, keep only routes to control nodes
    # gt_prime
    tt = time.time()
    all_pairs_lengths_gt_prime = {}
    if _verbose:
        verbose_print("Computing all_pairs_lengths_gt_prime...")
    G_gt_cp_prime_nodes_set = set(G_gt_cp_prime.nodes())
    # for source in G_gt_cp_prime_nodes_set:
    #     if source in G_gt_cp_prime_nodes_set:
    #         paths_tmp = nx.single_source_dijkstra_path_length(G_gt_cp_prime, source, weight=weight)
    for itmp, source in enumerate(rand_nodes_p_set):
        if _verbose and ((itmp % 50) == 0):
            verbose_print((itmp, "source:", source))
        if source in G_gt_cp_prime_nodes_set:
            paths_tmp = nx.single_source_dijkstra_path_length(
                G_gt_cp_prime, source, weight=weight)
            # delete items
            for k in list(paths_tmp.keys()):
                if k not in rand_nodes_p_set:
                    del paths_tmp[k]
            all_pairs_lengths_gt_prime[source] = paths_tmp
    if _verbose:
        verbose_print(("Time to compute all source routes for",
               max_nodes, "nodes:", time.time() - tt, "seconds"))

    # prop_prime
    tt = time.time()
    all_pairs_lengths_prop_prime = {}
    if _verbose:
        verbose_print("Computing all_pairs_lengths_prop_prime...")
    G_p_cp_prime_nodes_set = set(G_p_cp_prime.nodes())
    # for source in G_p_cp_prime_nodes_set:
    #     if source in G_p_cp_prime_nodes_set:
    #         paths_tmp = nx.single_source_dijkstra_path_length(G_p_cp_prime, source, weight=weight)
    for itmp, source in enumerate(rand_nodes_gt_set):
        if _verbose and ((itmp % 50) == 0):
            verbose_print((itmp, "source:", source))
        if source in G_p_cp_prime_nodes_set:
            paths_tmp = nx.single_source_dijkstra_path_length(
                G_p_cp_prime, source, weight=weight)
            # delete items
            for k in list(paths_tmp.keys()):
                if k not in rand_nodes_gt_set:
                    del paths_tmp[k]
            all_pairs_lengths_prop_prime[source] = paths_tmp
    if _verbose:
        verbose_print(("Time to compute all source routes for",
               max_nodes, "nodes:", time.time() - tt, "seconds"))

    #all_pairs_lengths_gt_prime = nx.all_pairs_dijkstra_path_length(G_gt_cp_prime, weight=weight)
    #all_pairs_lengths_prop_prime = nx.all_pairs_dijkstra_path_length(G_p_cp_prime, weight=weight)

    ###############
    tf = time.time()
    verbose_print("Time to run make_graphs_huge in apls.py:", tf - t0, "seconds")

    return G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
        control_points_gt, control_points_prop, \
        all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
        all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime


###############################################################################
def single_path_metric(len_gt, len_prop, diff_max=1):
    """
    Compute APLS metric for single path.

    Notes
    -----
    Compute normalize path difference metric, if len_prop < 0, return diff_max

    Arguments
    ---------
    len_gt : float
        Length of ground truth edge.
    len_prop : float
        Length of proposal edge.
    diff_max : float
        Maximum value to return. Defaults to ``1``.

    Returns
    -------
    metric : float
        Normalized path difference.
    """

    if len_gt <= 0:
        return 0
    elif len_prop < 0 and len_gt > 0:
        return diff_max
    else:
        diff_raw = np.abs(len_gt - len_prop) / len_gt
        return np.min([diff_max, diff_raw])


###############################################################################
def path_sim_metric(all_pairs_lengths_gt, all_pairs_lengths_prop,
                    control_nodes=[], min_path_length=10,
                    diff_max=1, missing_path_len=-1, normalize=True,
                    verbose=False):
    """
    Compute metric for multiple paths.

    Notes
    -----
    Assume nodes in ground truth and proposed graph have the same names.
    Assume graph is undirected so don't evaluate routes in both directions
    control_nodes is the list of nodes to actually evaluate; if empty do all
        in all_pairs_lenghts_gt
    min_path_length is the minimum path length to evaluate
    https://networkx.github.io/documentation/networkx-2.2/reference/algorithms/shortest_paths.html

    Parameters
    ----------
    all_pairs_lengths_gt : dict
        Dictionary of path lengths for ground truth graph.
    all_pairs_lengths_prop : dict
        Dictionary of path lengths for proposal graph.
    control_nodes : list
        List of control nodes to evaluate.
    min_path_length : float
        Minimum path length to evaluate.
    diff_max : float
        Maximum value to return. Defaults to ``1``.
    missing_path_len : float
        Value to assign a missing path.  Defaults to ``-1``.
    normalize : boolean
        Switch to normalize outputs. Defaults to ``True``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    C, diffs, routes, diff_dic
        C is the APLS score
        diffs is a list of the the route differences
        routes is a list of routes
        diff_dic is a dictionary of path differences
    """

    diffs = []
    routes = []
    diff_dic = {}
    gt_start_nodes_set = set(all_pairs_lengths_gt.keys())
    prop_start_nodes_set = set(all_pairs_lengths_prop.keys())
    t0 = time.time()

    verbose_print()
    if len(gt_start_nodes_set) == 0:
        return 0, [], [], {}

    # set nodes to inspect
    if len(control_nodes) == 0:
        good_nodes = list(all_pairs_lengths_gt.keys())
    else:
        good_nodes = control_nodes

    if _verbose:
        verbose_print("\nComputing path_sim_metric()...")
        verbose_print("good_nodes:", good_nodes)

    # iterate over all start nodes
    # for start_node, paths in all_pairs_lengths.iteritems():
    for start_node in good_nodes:
        if _verbose:
            verbose_print("start node:", start_node)
        node_dic_tmp = {}

        # if we are not careful with control nodes, it's possible that the
        # start node will not be in all_pairs_lengths_gt, in this case use max
        # diff for all routes to that node
        # if the start node is missing from proposal, use maximum diff for
        # all possible routes to that node
        if start_node not in gt_start_nodes_set:
            verbose_print("for ss, node", start_node, "not in set")
            verbose_print("   skipping N paths:", len(
                list(all_pairs_lengths_prop[start_node].keys())))
            for end_node, len_prop in all_pairs_lengths_prop[start_node].items():
                diffs.append(diff_max)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff_max
            return

        paths = all_pairs_lengths_gt[start_node]

        # CASE 1
        # if the start node is missing from proposal, use maximum diff for
        # all possible routes to the start node
        if start_node not in prop_start_nodes_set:
            for end_node, len_gt in paths.items():
                if (end_node != start_node) and (end_node in good_nodes):
                    diffs.append(diff_max)
                    routes.append([start_node, end_node])
                    node_dic_tmp[end_node] = diff_max
            diff_dic[start_node] = node_dic_tmp
            # print ("start_node missing:", start_node)
            continue

        # else get proposed paths
        else:
            paths_prop = all_pairs_lengths_prop[start_node]

            # get set of all nodes in paths_prop, and missing_nodes
            end_nodes_gt_set = set(paths.keys()).intersection(good_nodes)
            # end_nodes_gt_set = set(paths.keys()) # old version with all nodes

            end_nodes_prop_set = set(paths_prop.keys())
            missing_nodes = end_nodes_gt_set - end_nodes_prop_set
            if _verbose:
                verbose_print("missing nodes:", missing_nodes)

            # iterate over all paths from node
            for end_node in end_nodes_gt_set:
                # for end_node, len_gt in paths.iteritems():

                len_gt = paths[end_node]
                # skip if too short
                if len_gt < min_path_length:
                    continue

                # get proposed path
                if end_node in end_nodes_prop_set:
                    # CASE 2, end_node in both paths and paths_prop, so
                    # valid path exists
                    len_prop = paths_prop[end_node]
                else:
                    # CASE 3: end_node in paths but not paths_prop, so assign
                    # length as diff_max
                    len_prop = missing_path_len

                if _verbose:
                    verbose_print("end_node:", end_node)
                    verbose_print("   len_gt:", len_gt)
                    verbose_print("   len_prop:", len_prop)

                # compute path difference metric
                diff = single_path_metric(len_gt, len_prop, diff_max=diff_max)
                diffs.append(diff)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff

            diff_dic[start_node] = node_dic_tmp

    if len(diffs) == 0:
        return 0, [], [], {}

    # compute Cost
    diff_tot = np.sum(diffs)
    if normalize:
        norm = len(diffs)
        diff_norm = diff_tot / norm
        C = 1. - diff_norm # Here the inversion occurs.
    else:
        C = diff_tot

    verbose_print("Time to compute metric (score = ", C, ") for ", len(diffs),
          "routes:", time.time() - t0, "seconds")

    return C, diffs, routes, diff_dic


###############################################################################
def compute_apls_metric(all_pairs_lengths_gt_native,
                        all_pairs_lengths_prop_native,
                        all_pairs_lengths_gt_prime,
                        all_pairs_lengths_prop_prime,
                        control_points_gt, control_points_prop,
                        res_dir='', min_path_length=10,
                        verbose=False, super_verbose=False):
    """
    Compute APLS metric and plot results (optional)

    Notes
    -----
    Computes APLS and creates plots in res_dir (if it is not empty)

    Arguments
    ---------
    all_pairs_lengths_gt_native : dict
        Dict of paths for gt graph.
    all_pairs_lengths_prop_native : dict
        Dict of paths for prop graph.
    all_pairs_lengths_gt_prime : dict
        Dict of paths for gt graph with control points from prop.
    all_pairs_lengths_prop_prime : dict
        Dict of paths for prop graph with control points from gt.
    control_points_gt : list
        Array of control points.
    control_points_prop : list
        Array of control points.
    res_dir : str
        Output dir for plots.  Defaults to ``''`` (no plotting).
    min_path_length : float
        Minimum path length to evaluate.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    C_tot, C_gt_onto_prop, C_prop_onto_gt : tuple
        C_tot is the total APLS score
        C_gt_onto_prop is the score when inserting gt control nodes onto prop
        C_prop_onto_gt is the score when inserting prop control nodes onto gt
    """

    t0 = time.time()

    # return 0 if no paths
    if (len(list(all_pairs_lengths_gt_native.keys())) == 0) \
            or (len(list(all_pairs_lengths_prop_native.keys())) == 0):
        verbose_print("len(all_pairs_lengths_gt_native.keys()) == 0)")
        return 0, 0, 0

    ####################
    # compute metric (gt to prop)
    # if _verbose:
    verbose_print("Compute metric (gt snapped onto prop)")
    # control_nodes = all_pairs_lengths_gt_native.keys()
    control_nodes = [z[0] for z in control_points_gt]
    if _verbose:
        verbose_print(("control_nodes_gt:", control_nodes))
    C_gt_onto_prop, diffs, routes, diff_dic = path_sim_metric(
        all_pairs_lengths_gt_native,
        all_pairs_lengths_prop_prime,
        control_nodes=control_nodes,
        min_path_length=min_path_length,
        diff_max=1, missing_path_len=-1, normalize=True,
        verbose=super_verbose)
    dt1 = time.time() - t0

    verbose_print("len(diffs):", len(diffs))
    if len(diffs) > 0:
        verbose_print("  max(diffs):", np.max(diffs))
        verbose_print("  min(diffs)", np.min(diffs))

    if _plot:
        scatter_png = os.path.join(
            res_dir, 'all_pairs_paths_diffs_gt_to_prop.png')
        hist_png = os.path.join(
            res_dir, 'all_pairs_paths_diffs_hist_gt_to_prop.png')
        # can't plot route names if there are too many...
        if len(routes) > 100:
            routes_str = []
        else:
            routes_str = [str(z[0]) + '-' + str(z[1]) for z in routes]
        apls_plots.plot_metric(
            C_gt_onto_prop, diffs, routes_str=routes_str,
            figsize=(10, 5), scatter_alpha=0.8, scatter_size=8,
            scatter_png=scatter_png,
            hist_png=hist_png)
    diffs_gt = diffs
    ######################

    ####################
    # compute metric (prop to gt)
    # if _verbose:
    verbose_print("Compute metric (prop snapped onto gt)")
    t1 = time.time()
    # control_nodes = all_pairs_lengths_prop_native.keys()
    control_nodes = [z[0] for z in control_points_prop]
    verbose_print("control_nodes:", control_nodes)
    C_prop_onto_gt, diffs, routes, diff_dic = path_sim_metric(
        all_pairs_lengths_prop_native,
        all_pairs_lengths_gt_prime,
        control_nodes=control_nodes,
        min_path_length=min_path_length,
        diff_max=1, missing_path_len=-1, normalize=True,
        verbose=super_verbose)
    dt2 = time.time() - t1

    verbose_print("len(diffs):", len(diffs))
    if len(diffs) > 0:
        verbose_print("  max(diffs):", np.max(diffs))
        verbose_print("  min(diffs)", np.min(diffs))

    if _plot:
        scatter_png = os.path.join(
            res_dir, 'all_pairs_paths_diffs_prop_to_gt.png')
        hist_png = os.path.join(
            res_dir, 'all_pairs_paths_diffs_hist_prop_to_gt.png')
        if len(routes) > 100:
            routes_str = []
        else:
            routes_str = [str(z[0]) + '-' + str(z[1]) for z in routes]
        apls_plots.plot_metric(
            C_prop_onto_gt, diffs, routes_str=routes_str,
            figsize=(10, 5), scatter_alpha=0.8, scatter_size=8,
            scatter_png=scatter_png,
            hist_png=hist_png)
    diffs_pr = diffs
    ####################

    ####################
    # Total

    verbose_print("C_gt_onto_prop, C_prop_onto_gt:", C_gt_onto_prop, C_prop_onto_gt)
    if (C_gt_onto_prop <= 0) or (C_prop_onto_gt <= 0) \
            or (np.isnan(C_gt_onto_prop)) or (np.isnan(C_prop_onto_gt)):
        C_tot = 0
    else:
        C_tot = scipy.stats.hmean([C_gt_onto_prop, C_prop_onto_gt])
        if np.isnan(C_tot):
            C_tot = 0
    verbose_print("Total APLS Metric = Mean(", np.round(C_gt_onto_prop, 2), "+",
          np.round(C_prop_onto_gt, 2),
          ") =", np.round(C_tot, 2))
    verbose_print("Total time to compute metric:", str(dt1 + dt2), "seconds")

    return C_tot, C_gt_onto_prop, C_prop_onto_gt, diffs_gt, diffs_pr


###############################################################################
def run(truth=None, proposed=None):
    output_name = "apls_test"
    weight = "length"
    im_dir = "apls_images"
    return execute(output_name, [truth], [proposed], ["randomstring"], )


# APLS metric performed on two simplified nx.MultiGraphs.
def apls_detailed(truth=None, proposed=None):

    weight = "length"
    speed_key='inferred_speed_mps'
    travel_time_key='travel_time_s'
    test_method='gt_json_prop_json'
    max_files=1000
    linestring_delta=50
    is_curved_eps=10**3
    max_snap_dist=4
    max_nodes=500
    n_plots=10
    min_path_length=10
    topo_hole_size=4
    topo_subgraph_radius=150
    topo_interval=30
    sp_length_buffer=0.05
    use_pix_coords=False
    allow_renaming=True
    verbose=False
    super_verbose=False

    if _plot:
        res_dir = "apls_results"
    else:
        res_dir = ""

    if len(truth.nodes()) < 500:  # 2000:
        G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
            control_points_gt, control_points_prop, \
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime  \
            = make_graphs(truth, proposed,
                            weight=weight,
                            speed_key=speed_key,
                            travel_time_key=travel_time_key,
                            linestring_delta=linestring_delta,
                            is_curved_eps=is_curved_eps,
                            max_snap_dist=max_snap_dist,
                            allow_renaming=allow_renaming,
                            verbose=verbose)

    # get large graphs and paths
    else:
        G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
            control_points_gt, control_points_prop, \
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime  \
            = make_graphs_huge(truth, proposed,
                                weight=weight,
                                speed_key=speed_key,
                                travel_time_key=travel_time_key,
                                max_nodes=max_nodes,
                                max_snap_dist=max_snap_dist,
                                allow_renaming=allow_renaming,
                                verbose=verbose,
                                super_verbose=super_verbose)

    C, C_gt_onto_prop, C_prop_onto_gt, diffs_gt, diffs_pr = compute_apls_metric(
        all_pairs_lengths_gt_native, all_pairs_lengths_prop_native,
        all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime,
        control_points_gt, control_points_prop,
        min_path_length=min_path_length,
        verbose=verbose, res_dir=res_dir)
    
    return C, C_gt_onto_prop, C_prop_onto_gt, diffs_gt, diffs_pr


def apls_both(truth=None, proposed=None, verbose=False, plot=False):

    global _verbose
    global _plot
    _verbose = verbose
    _plot = plot

    results = apls_detailed(truth=truth, proposed=proposed)

    gt = np.array(results[3])
    pr = np.array(results[4])

    apls_result = scipy.stats.hmean([np.average(1 - gt), np.average(1 - pr)])

    # Filter out failed results.
    gt = gt[np.where(gt < 1)]
    pr = pr[np.where(pr < 1)]

    apls_prime_result = scipy.stats.hmean([np.average(1 - gt), np.average(1 - pr)])

    return apls_result, apls_prime_result


def apls(truth, proposed, verbose=False, plot=False):
    global _verbose
    global _plot
    _verbose = verbose
    _plot = plot

    results = apls_detailed(truth=truth, proposed=proposed)

    gt = np.array(results[3])
    pr = np.array(results[4])

    apls_result = scipy.stats.hmean([np.average(1 - gt), np.average(1 - pr)])

    return apls_result
    


def apls_prime(truth=None, proposed=None):

    results = apls_detailed(truth=truth, proposed=proposed)

    gt = np.array(results[3])
    pr = np.array(results[4])

    # Filter out failed results.
    gt = gt[np.where(gt < 1)]
    pr = pr[np.where(pr < 1)]

    apls_result = scipy.stats.hmean([np.average(1 - gt), np.average(1 - pr)])

    return apls_result