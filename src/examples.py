

# Example (re-construct all graphs from mapconstruction dataset):
# for name in graphnames["mapconstruction"]:
#     extract_graph(name, True)


# Example (extracting nearest vertex):
# nearest_point(extract_graph("chicago"), np.asarray((4.422440 , 46.346080), dtype=np.float64, order='c'))


# Example (Render chicago):
# G = extract_graph("maps_chicago")
# G = extract_graph("berlin")
# ox.plot_graph(G)


# Example (Extract specific position and render):
# ox.settings.use_cache = True
# G = ox.graph.graph_from_place("berlin", network_type='drive', simplify=False, retain_all=True)


# Example (Render all mapconstruction graphs):
# for name in graphnames["mapconstruction"]:
#     G = extract_graph(name)
#     ox.plot_graph(G)


# Example (Extract historical OSM of Chicago dataset) FAILS:
#   Place: South Campus Parkway, Chicago, IL, USA
#   Date: 2011-09-05
#   Position: 41.8625,-87.6453
# coordinate = (41.8625,-87.6453)
# dist = 500 # meters
# ox.settings.overpass_settings = f'[out:json][timeout:90][date:"2011-09-05T00:00:00Z"]'
# ox.graph_from_point(coordinate, dist=dist, retain_all=True, simplify=False)


# Example (vectorize and simplify again):
# Note: It contains a bug: bidirectional self-loops are incorrectly removed.
# G  = extract_graph("chicago_kevin")
# G2 = vectorize_graph(G)
# G2 = deduplicate_vectorized_graph(G2)
# G3 = ox.simplify_graph(G2)
# G5 = ox.simplify_graph(vectorize_graph(G3))
# # Simple check on vectorization validity (Wont pass: Intersection nodes are add).
# assert len(G.nodes()) == len(G3.nodes())
# assert len(G.edges()) == len(G3.edges())


# Example (Subgraph of nodes nearby curve):
# G = extract_graph("chicago")
# idx = graphnodes_to_rtree(G)
# ps = gen_random_shortest_path(G)
# bb = bounding_box(ps)
# H = rtree_subgraph_by_bounding_box(G, idx, bb)


###################################
###  Examples: Graph and Curve Rendering
###################################

# Example (rendering multiple paths)
# G = extract_graph("chicago")
# render_paths([gen_random_shortest_path(G), gen_random_shortest_path(G)])

# Example (obtain distance to obtain full chicago, use all_public just like mapconstruction graphs):
# coord_center = (41.87168, -87.65985)  # coordinate at approximately center 
# coord_edge   = (41.88318, -87.64129)  # coordinate at approximately furthest edge
# dist = ox.distance.great_circle(41.87168, -87.65985, 41.88318, -87.64129) # == 1999 
# G = ox.graph_from_point(coord_from, network_type="all_public", dist=dist) # padding included automatically  

# Example (difference all_public to drive_service network filter):
# coord_center = (41.87168, -87.65985)  # coordinate at approximately center 
# G = ox.graph_from_point(coord_center, network_type="all_public", dist=2000)
# H = ox.graph_from_point(coord_center, network_type="drive_service", dist=2000)
# plot_two_graphs(G, H)

# Example (mapconstruction.org chicago vs up-to-date OSM chicago):
# coord_center = (41.87168, -87.65985)  # coordinate at approximately center 
# G = ox.graph_from_point(coord_center, network_type="all_public", dist=2000)
# H = extract_graph("chicago")
# plot_two_graphs(G, H)


# Example (subgraph around random shortest path and rendering both)
# G = extract_graph("chicago")
# G = vectorize_graph(G) # Vectorize graph.
# G = deduplicate_vectorized_graph(G)
# ps = gen_random_shortest_path(G)
# lam = 0.0015
# idx = graphnodes_to_rtree(G) # Place graph nodes coordinates in accelerated data structure (R-Tree).
# bb = bounding_box(ps, padding=lam) # Construct lambda-padded bounding box.
# nodes = list(idx.intersection((bb[0][0], bb[0][1], bb[1][0], bb[1][1]))) # Extract nodes within bounding box.
# H = G.subgraph(nodes) # Extract subgraph with nodes.
# # plot_graph_and_curve(G,ps) # Full graph with curve of interest
# plot_graph_and_curve(H,ps) # Subgraph with curve of interest


# Example (Retrieve/Construct image with GSD ~0.88 between two coordinates):
# upperleft  = (41.799575, -87.606117)
# lowerright = (41.787669, -87.585498)
# scale = 1
# zoom = 17 # For given latitude and scale results in gsd of ~ 0.88
# api_key = read_api_key()
# # superimage = construct_image(upperleft, lowerright, zoom, scale, api_key)   # Same result as below.
# superimage = construct_image(upperleft, lowerright, zoom-1, scale+1, api_key) # Same result as above.
# write_image(superimage, "superimage.png")


# Example (Extracting chicago ROI including pixel coordinates):
# upperleft = (41.880126, -87.659200)
# lowerright = (41.863563, -87.634062)
# scale = 2
# zoom = 17 # For given latitude and scale results in gsd of ~ 0.88
# image, coordinates = construct_image(upperleft, lowerright, zoom, scale, read_api_key())
# write_image(image, "chicago_zoomed.png")
# pickle.dump(coordinates, open("chicago_zoomed.pkl", "wb"))


# Example (Converting sat2graph inferred data into a graph):
# G = sat2graph_json_to_graph("chicago_zoomed.json", "chicago_zoomed.pkl")
# plot_graph_presentation(G)


# Example (Load networks sat, gps, truth and render gps-truth and sat-truth):
# ground_truth = extract_graph("maps_chicago")
# ground_truth2 = extract_graph("chicago")
# inferred_gps = extract_graph("inferredgps_chicago")
# inferred_sat = sat2graph_json_to_graph("chicago_zoomed.json", "chicago_zoomed.pkl")
# plot_two_graphs(ground_truth2, inferred_gps)
# plot_two_graphs(ground_truth2, inferred_sat)


# Example (rendering three graphs from graph set):
# graphs = extract_graphset("chicago")
# truth = graphs["truth"]
# gps = graphs["gps"]
# sat = graphs["sat"]
# plot_three_graphs(truth, sat, gps)


# Example (Load networks sat, gps, truth and render gps-truth and sat-truth):
# web_coordinates = {
#     "chicago_zoomed"      : ((41.880126, -87.659200), (41.863563, -87.634062)),
#     "chicago_super_zoomed": ((41.878978, -87.651714), (41.871960, -87.640646)),
#     "chicago_gps"         : ((41.87600 , -87.68800 ), (41.86100 , -87.63900 ))
# }
# scale = 2
# zoom = 17
# roi    = "chicago_zoomed"
# roi    = "chicago_gps"
# action = "extract_image"
# action = "infer_graph"
# action = "plot_graphs"
# action = "plot_truth_gps"
# p1, p2 = web_coordinates[roi]
# p1, p2 = squarify_web_mercator_coordinates(p1, p2, zoom)
# match action:
#     case "extract_image":
#         image, coordinates = construct_image(p1, p2, zoom, scale, read_api_key())
#         write_image(image, roi+".png")
#     case "infer_graph":
#         json_file = roi + ".json"
#         G = sat2graph_json_to_graph(json_file, p1, p2)
#         plot_graph_presentation(G)
#         save_graph(G, "chicago_inferred_sat", overwrite=True)
#     case "plot_graphs":
#         truth = extract_graph("chicago_truth")
#         truth = cut_out_ROI(truth, p1, p2)
#         sat   = extract_graph("chicago_inferred_sat")
#         gps   = extract_graph("chicago_inferred_gps")
#         plot_three_graphs(truth, sat, gps)
#     case "plot_truth_gps":
#         truth = extract_graph("chicago_truth")
#         truth = cut_out_ROI(truth, p1, p2)
#         gps   = extract_graph("chicago_inferred_gps")
#         plot_two_graphs(truth, gps)


# Example (partial curve matching):
# ps = random_curve(length = 100, a = np.array([-10,-10]), b = np.array([10,10]))
# qs = random_curve(length = 100, a = np.array([-10,-10]), b = np.array([10,10]))
# assert is_partial_curve_undirecteto_curve(ps), to_curve(qs), sqrt(2*100))


# Example (compute distance between graph nodes):
# def example_graphnodes_distances():
    # graphs = extract_graphset("chicago")
    # G = graphs["truth"]
    # max_deviation = 0

    # uvk, data = zip(*G.nodes(data=True))
    # df = gpd.GeoDataFrame(data, index=uvk)
    # zoom = 20
    # alat = df["y"].mean()
    # gsd = compute_gsd(alat, zoom, 1)

    # for k, nbrs in S.adj.items():
    #     for nbr, data in nbrs.items():

    #         # print(nbr, data)
    #         p1 = S._node[k]
    #         p2 = S._node[nbr]

    #         # print(p1, p2)
    #         latlon1 = [p1["y"], p1["x"]]
    #         latlon2 = [p2["y"], p2["x"]]
    #         lat1, lon1 = latlon1
    #         lat2, lon2 = latlon2

    #         ## Haversine distance.
    #         d_haver = haversine(latlon1, latlon2)

    #         ## Equirectangular distance.
    #         d_rect = equirectangular(latlon1, latlon2)

    #         ## Webmercator-pixelcoordinate distance.
    #         q1 = gsd * np.array(latlon_to_pixelcoord(lat1, lon1, zoom))
    #         q2 = gsd * np.array(latlon_to_pixelcoord(lat2, lon2, zoom))
    #         d_merc = np.linalg.norm(q2 - q1)

    #         ## Webmercator-pixelcoordinate distance, but each position is fixated with its own gsd. (result is off by 30%..)
    #         # print(np.linalg.norm(compute_gsd(lat2, zoom, 1) * q2 - compute_gsd(lat1, zoom, 1) * q1))

    #         print(d_haver)
    #         print(d_rect)
    #         print(d_merc)
    #         max_deviation = max(max_deviation, abs(d_haver - d_merc))

    # print("max deviation haversine versus webmercator:", max_deviation)


# Example (Rendering graph with distances rather than geographic coordinates):
#   Note: Incomplete approach, it takes a graph with geographic coordinates. 
#         Ideally we first transform all nodes (and edge curvature) and render that instead.
# def example_render_graph_distances():

#     truth = graphs["truth"]
#     truth = nx.Graph(truth)
#     G = annotate_edge_curvature_as_array(truth)

#     ## Nodes (GeoPandas Dataframe).
#     uvk, data = zip(*G.nodes(data=True))
#     gdf_nodes = gpd.GeoDataFrame(data, index=uvk)
#     alat = gdf_nodes["y"].mean()
#     zoom = 20
#     gsd = compute_gsd(alat, zoom, 1)

#     # Reference point to compute relative distances from.
#     refy, refx = gdf_nodes["y"].max(), gdf_nodes["x"].min()
#     refy, refx = latlon_to_pixelcoord(refy, refx, zoom)
#     refy, refx = gsd * refy, gsd * refx

#     # Map lat,lon to y,x with latlon_to_pixelcoord.
#     def latlon_to_relative_pixelcoord(row):
#         lat, lon = row["y"], row["x"]
#         y, x = latlon_to_pixelcoord(lat, lon, zoom)
#         return pd.Series({'x': gsd * x - refx, 'y': gsd * y - refy})

#     gdf_nodes[["y","x"]] = gdf_nodes.apply(latlon_to_relative_pixelcoord, axis=1)[["y","x"]]

#     ## Edges (GeoPandas Dataframe).
#     u, v, data = zip(*G.edges(data=True))

#     # Construct geometry out of edge curvature.
#     def edge_latlon_curvature_to_relative_pixelcoord(u, v, data):
#         curvature = [gsd * np.array(latlon_to_pixelcoord(lat, lon, zoom)) for lat, lon in data["curvature"]]
#         # Convert into a LineString and Points.
#         return LineString([Point(x - refx, y - refy) for y,x in curvature])

#     edge_geoms = map(edge_latlon_curvature_to_relative_pixelcoord, u, v, data)
#     gdf_edges = gpd.GeoDataFrame(data, geometry=list(edge_geoms))
#     gdf_edges["u"] = u
#     gdf_edges["v"] = v
#     gdf_edges = gdf_edges.set_index(["u", "v"])

#     ## Plot construction.
#     fig, ax = plt.subplots()
#     ax = gdf_edges['geometry'].plot(ax=ax)
#     ax.scatter( x=gdf_nodes["x"], y=gdf_nodes["y"],)
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     plt.show()


# # Example (Transforming points to relative distance and render graph):
# def example_relative_distance_plotting():
#     # 0. Obtain graph.
#     graphs = extract_graphset("chicago")
#     G = graphs["truth"]
#     G = nx.Graph(G)
#     # G = annotate_edge_curvature_as_array(truth)
#     # 1. First vectorize.
#     G = vectorize_graph(G)
#     G = deduplicate_vectorized_graph(G)
#     G = nx.Graph(G)
#     # 2. Map all nodes to relative position.
#     uvk, data = zip(*G.nodes(data=True))
#     df = gpd.GeoDataFrame(data, index=uvk)
#     # GSD on average latitude.
#     zoom = 20
#     alat = df["y"].mean()
#     gsd = compute_gsd(alat, zoom, 1)
#     # Reference point to compute relative distances from.
#     refy, refx = df["y"].min(), df["x"].min()
#     refy, refx = latlon_to_pixelcoord(refy, refx, zoom)
#     refy, refx = gsd * refy, gsd * refx
#     # Map lat,lon to y,x with latlon_to_pixelcoord.
#     def latlon_to_relative_pixelcoord(row): 
#         lat, lon = row["y"], row["x"]
#         y, x = latlon_to_pixelcoord(lat, lon, zoom)
#         return {'x': gsd * x - refx, 'y': refy - gsd * y}
#     # Construct relabel mapping.
#     relabel_mapping = {}
#     for nid, data in G.nodes(data=True):
#         relabel_mapping[nid] = latlon_to_relative_pixelcoord(data)
#     nx.set_node_attributes(G, relabel_mapping)
#     # 3. Convert back into simplified graph.
#     G = ox.simplify_graph(nx.MultiDiGraph(G)) # If it has curvature it crashes because of non-hashable numpy array in attributes.
#     # 4. Render graph.
#     # Nodes.
#     uvk, data = zip(*G.nodes(data=True))
#     gdf_nodes = gpd.GeoDataFrame(data, index=uvk)
#     # Edges.
#     u, v, data = zip(*G.edges(data=True))
#     x_lookup = nx.get_node_attributes(G, "x")
#     y_lookup = nx.get_node_attributes(G, "y")
#     def edge_latlon_curvature_to_relative_pixelcoord(u, v, data):
#         if "geometry" in data:
#             return data["geometry"]
#         else:
#             return LineString((Point((x_lookup[u], y_lookup[u])), Point((x_lookup[v], y_lookup[v]))))
#     edge_geoms = map(edge_latlon_curvature_to_relative_pixelcoord, u, v, data)
#     gdf_edges = gpd.GeoDataFrame(data, geometry=list(edge_geoms))
#     gdf_edges["u"] = u
#     gdf_edges["v"] = v
#     gdf_edges = gdf_edges.set_index(["u", "v"])
#     # Plot.
#     fig, ax = plt.subplots()
#     ax = gdf_edges['geometry'].plot(ax=ax)
#     ax.scatter(x=gdf_nodes["x"], y=gdf_nodes["y"],)
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     plt.show()
#     # crs = G.graph["crs"]


# Example (Decomposed transform geographic coordinates to relative pixelpositions and )
# graphs = extract_graphset("chicago")
# truth = graphs["truth"]
# truth = nx.Graph(truth)
# G = truth
# # Compute average latitude.
# uvk, data = zip(*G.nodes(data=True))
# df = gpd.GeoDataFrame(data, index=uvk)
# alat, alon = df["y"].mean(), df["x"].mean()
# # Apply
# G = relative_positioning(G, alat, alon)
# plot_without_projection(G)