from dependencies import *
from coverage import *
from network import *


# Convert a collection of paths into gid-annotated nodes and edges to thereby render with different colors.
def render_paths(pss):
    G = convert_paths_into_graph(pss)
    G = nx.MultiDiGraph(G)
    G.graph['crs'] = "EPSG:4326"
    nc = ox.plot.get_node_colors_by_attr(G, "gid", cmap="Paired")
    ec = ox.plot.get_edge_colors_by_attr(G, "gid", cmap="Paired")
    ox.plot_graph(G, bgcolor="#ffffff", node_color=nc, edge_color=ec, save=True)


# Pick random shortest paths until coverage, then render.
def render_random_nearby_shortest_paths():
    G = extract_graph("athens_small")
    found = False
    attempt = 0
    while True:
        while not found:
            ps = gen_random_shortest_path(G)
            qs = gen_random_shortest_path(G)
            found, histories, rev = curve_by_curve_coverage(ps,qs, lam=0.003)
            attempt += 1

            if random.random() < 0.01:
                print(attempt)
            
            if rev:
                qs = qs[::-1]

        print(found, histories, rev)

        # Render
        for history in histories:
            print("history:", history)
            steps = history_to_sequence(history)
            print("steps:", steps)

            maxdist = -1

            if not np.all(np.array( [np.linalg.norm(ps[ip] - qs[iq]) for (ip, iq) in steps] ) < 0.003):
                print( np.array([np.linalg.norm(ps[ip] - qs[iq]) for (ip, iq) in steps]) )
                breakpoint()

        ids = np.array(steps)[:,1]
        subqs = qs[ids]

        render_paths([ps, subqs])
        found = False


def plot_two_graphs(G,H):
    G = G.copy()
    H = H.copy()
    G.graph['crs'] = "EPSG:4326"
    G = nx.MultiDiGraph(G)
    H.graph['crs'] = "EPSG:4326"
    H = nx.MultiDiGraph(H)

    # To prevent node interference, update node IDs of H to start at highest index of G.
    nid=max(G.nodes())+1
    relabel_mapping = {}
    for nidH in H.nodes():
        relabel_mapping[nidH] = nid
        nid += 1
    H = nx.relabel_nodes(H, relabel_mapping)

    # Add gid 1 to all nodes and edges of G, 2 for H.
    # G = Blue
    # H = Green
    nx.set_node_attributes(G, 1, name="gid")
    nx.set_edge_attributes(G, 1, name="gid")
    nx.set_node_attributes(H, 2, name="gid")
    nx.set_edge_attributes(H, 2, name="gid")

    # Add two graphs together
    F = nx.compose(G,H)

    # Coloring of edges and nodes per gid.
    nc = ox.plot.get_node_colors_by_attr(F, "gid", cmap="winter")
    ec = ox.plot.get_edge_colors_by_attr(F, "gid", cmap="winter")
    ox.plot_graph(F, bgcolor="#ffffff", node_color=nc, edge_color=ec, save=True)


# Relabel all nodes in graph starting at a given node ID.
def relabel_graph_from_nid(G, nid):
    relabel_mapping = {}
    for nidG in G.nodes():
        relabel_mapping[nidG] = nid
        nid += 1
    G = nx.relabel_nodes(G, relabel_mapping)
    return G


def plot_three_graphs(G,H,I):
    G = G.copy()
    G.graph['crs'] = "EPSG:4326"
    G = nx.MultiDiGraph(G)

    H = H.copy()
    H = nx.MultiDiGraph(H)
    H.graph['crs'] = "EPSG:4326"

    I = I.copy()
    I = nx.MultiDiGraph(I)
    I.graph['crs'] = "EPSG:4326"

    # To prevent node interference, update node IDs of H and I.
    H = relabel_graph_from_nid(H, max(G.nodes())+1)
    I = relabel_graph_from_nid(I, max(H.nodes())+1)

    # Add gid 1 to all nodes and edges of G, 2 for H.
    # G = Blue
    # H = ?
    # I = ?
    nx.set_node_attributes(G, 1, name="gid")
    nx.set_edge_attributes(G, 1, name="gid")
    nx.set_node_attributes(H, 2, name="gid")
    nx.set_edge_attributes(H, 2, name="gid")
    nx.set_node_attributes(I, 3, name="gid")
    nx.set_edge_attributes(I, 3, name="gid")

    # Add two graphs together
    F = nx.compose(G,H)
    F = nx.compose(F,I)

    # Coloring of edges and nodes per gid.
    nc = ox.plot.get_node_colors_by_attr(F, "gid", cmap="tab20b")
    ec = ox.plot.get_edge_colors_by_attr(F, "gid", cmap="tab20b")
    ox.plot_graph(F, bgcolor="#ffffff", node_color=nc, edge_color=ec, save=True)


# Rendering duplicated nodes and edges.
def render_duplicates_highlighted(G):
    G = G.copy()

    # Give everyone GID 2
    nx.set_node_attributes(G, 2, name="gid")
    nx.set_edge_attributes(G, 2, name="gid")

    for key in duplicated_nodes(G):
        G.nodes[key]["gid"] = 1

    for key in duplicated_edges(G):
        G.edges[key]["gid"] = 1

    # Render
    nc = ox.plot.get_node_colors_by_attr(G, "gid", cmap="winter")
    ec = ox.plot.get_edge_colors_by_attr(G, "gid", cmap="winter")
    ox.plot_graph(G, bgcolor="#ffffff", node_color=nc, edge_color=ec, save=True)


# Render a graph that meets the styling for presentation.
def plot_graph_presentation(G):
    # Coloring of edges and nodes per gid.
    G = G.copy()
    G.graph['crs'] = "EPSG:4326"
    G = nx.MultiDiGraph(G)
    white = "#fafafa"
    black = "#040404"
    ox.plot_graph(
        G, 
        bgcolor=white, 
        edge_color=black,
        edge_linewidth=1,
        node_color=white,
        node_edgecolor=black,
        node_size=10,
        save=True,
        # dpi=500,
        # figsize=(1024,1024)
    )


# Render curve and graph
def plot_graph_and_curve(G, ps):

    G = G.copy()
    nx.set_node_attributes(G, 2, name="gid")
    nx.set_edge_attributes(G, 2, name="gid")
    G.graph['crs'] = "EPSG:4326"
    G = G.to_directed()

    # Construct subgraph from ps.
    H = convert_paths_into_graph([ps], nid=max(G.nodes())+1)
    nx.set_node_attributes(H, 1, name="gid")
    nx.set_edge_attributes(H, 1, name="gid")
    H.graph['crs'] = "EPSG:4326"
    H = nx.MultiGraph(H)
    H = H.to_directed()

    F = nx.compose(G,H)
    nc = ox.plot.get_node_colors_by_attr(F, "gid", cmap="winter")
    ec = ox.plot.get_edge_colors_by_attr(F, "gid", cmap="winter")

    ox.plot_graph(F, bgcolor="#ffffff", node_color=nc, edge_color=ec, save=True)


def plot_graph_and_curves(G, ps, qs):

    G = G.copy()
    nx.set_node_attributes(G, 2, name="gid")
    nx.set_edge_attributes(G, 2, name="gid")
    G.graph['crs'] = "EPSG:4326"
    G = G.to_directed()

    # Construct subgraph for ps.
    H = convert_paths_into_graph([ps], nid=max(G.nodes())+1)
    nx.set_node_attributes(H, 1, name="gid")
    nx.set_edge_attributes(H, 1, name="gid")
    H.graph['crs'] = "EPSG:4326"
    H = nx.MultiGraph(H)
    H = H.to_directed()

    F = nx.compose(G,H)

    # Construct subgraph for qs.
    H = convert_paths_into_graph([qs], nid=max(F.nodes())+1)
    nx.set_node_attributes(H, 3, name="gid")
    nx.set_edge_attributes(H, 3, name="gid")
    H.graph['crs'] = "EPSG:4326"
    H = nx.MultiGraph(H)
    H = H.to_directed()

    F = nx.compose(F,H)

    nc = ox.plot.get_node_colors_by_attr(F, "gid", cmap="Paired")
    ec = ox.plot.get_edge_colors_by_attr(F, "gid", cmap="Paired")
    ox.plot_graph(F, bgcolor="#ffffff", node_color=nc, edge_color=ec, save=True)


def preplot_graph(G, ax, **properties): 
    
    assert type(G) == nx.Graph
    
    print("Extracting node attributes.")
    # Nodes.
    uv, data = zip(*G.nodes(data=True))
    gdf_nodes = gpd.GeoDataFrame(data, index=uv)
    # Edges.
    u, v, data = zip(*G.edges(data=True))
    x_lookup = nx.get_node_attributes(G, "x")
    y_lookup = nx.get_node_attributes(G, "y")

    def extract_edge_geometry(u, v, data):
        if "geometry" in data:
            return data["geometry"]
        else:
            return LineString((Point((x_lookup[u], y_lookup[u])), Point((x_lookup[v], y_lookup[v]))))

    print("Mapping edge geometry.")
    edge_geoms = map(extract_edge_geometry, u, v, data)
    gdf_edges = gpd.GeoDataFrame(data, geometry=list(edge_geoms))
    gdf_edges["u"] = u
    gdf_edges["v"] = v
    gdf_edges = gdf_edges.set_index(["u", "v"])

    # Plot.
    print("Plotting data.")
    ax.scatter(x=gdf_nodes["x"], y=gdf_nodes["y"], **properties)
    n = len(gdf_edges)
    gdf_edges.plot(ax=ax, **properties)


def preplot_curve(ps, ax, **properties):
    # Construct GeoDataFrame .
    edge = dataframe({"geometry": to_linestring(ps)}, index=[0])
    # edge.plot(ax=ax, color=color, linewidth=linewidth, linestyle=linestyle)
    edge.plot(ax=ax, **properties)


# Render target graph (dotted gray) + curve (green) + path (blue).
def plot_without_projection(Gs, pss):

    fig, ax = plt.subplots()

    for i, obj in enumerate(Gs):
        print(f"Plotting graph {i}.")
        if type(obj) == tuple:
            G, properties = obj
            preplot_graph(G,  ax, **properties) 
        else:
            G = obj
            preplot_graph(G,  ax) 

    for i, obj in enumerate(pss):
        print(f"Plotting paths {i}.")
        if type(obj) == tuple:
            ps, properties = obj
            preplot_curve(ps, ax, **properties) 
        else:
            ps = obj
            preplot_curve(ps, ax) 

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show()

# Example usage of plot_without_projection:
#   plot_without_projection2([S], [])
#   plot_without_projection2([], [ (random_curve(),{"color":(0,0,0,1)}) ])
#   plot_without_projection2([], [ (random_curve(),{"color":(0,0,0,1), "linewidth":1, "linestyle": ":"}), (random_curve(), {"color":(0.1,0.5,0.1,1), "linewidth":3}) ])


# Plot graph without projecting nodes (used for e.g. our relative pixel position coordinates).
def plot_with_weight_without_projection(G, H, data_dict):

    print("Plotting graphs.")
    
    def extract_edge_geometry(u, v, k, data):
        if "geometry" in data:
            return data["geometry"]
        else:
            return LineString((Point((x_lookup[u], y_lookup[u])), Point((x_lookup[v], y_lookup[v]))))

    def extract_edge_weight(u, v, k, data):
        return data['weight']

    def extract_edge_color(u, v, k, data):
        weight = data['weight']
        color = cmap(normalize_weight(weight))
        return color

    def normalize_weight(w):
        return (w - minweight) / (maxweight - minweight)

    fig, ax = plt.subplots()

    # Render H.
    # Nodes.
    uvk, data = zip(*H.nodes(data=True))
    gdf_nodes = gpd.GeoDataFrame(data, index=uvk)
    # Edges.
    u, v, k, data = zip(*H.edges(data=True, keys=True))
    x_lookup = nx.get_node_attributes(H, "x")
    y_lookup = nx.get_node_attributes(H, "y")

    edge_geoms = map(extract_edge_geometry, u, v, k, data)
    edge_weights = map(extract_edge_weight, u, v, k, data)
    edge_colors = map(extract_edge_color, u, v, k, data)
    gdf_edges = gpd.GeoDataFrame(data, geometry=list(edge_geoms))

    gdf_edges["u"] = u
    gdf_edges["v"] = v
    gdf_edges = gdf_edges.set_index(["u", "v"])

    # Plot.
    gdf_edges.plot(ax=ax, color=(0.4, 0.4, 0.4, 1), linewidth=1, linestyle=":")

    # Render G.
    assert type(G) == nx.MultiGraph
    G = G.copy()

    # Assign weight to edges.
    for uv in data_dict.keys():
        (u, v) = uv
        threshold = data_dict[uv]["threshold"]
        G[u][v][0]['weight'] = threshold
    
    weights = array([G[u][v][key]['weight'] for u, v, key in G.edges(keys=True)])
    minweight = weights.min()
    maxweight = weights.max()

    # Obtain normalized weights.
    weights = array([G[u][v][key]['weight'] for u, v, key in G.edges(keys=True)])
    norm_weights = (weights - weights.min()) / (weights.max() - weights.min())  # Normalize to [0, 1]

    # Map weight to color map.
    cmap = plt.cm.Greens
    edge_colors = [cmap(weight) for weight in norm_weights]

    # Nodes.
    uvk, data = zip(*G.nodes(data=True))
    gdf_nodes = gpd.GeoDataFrame(data, index=uvk)

    # Edges.
    u, v, k, data = zip(*G.edges(data=True, keys=True))
    x_lookup = nx.get_node_attributes(G, "x")
    y_lookup = nx.get_node_attributes(G, "y")
    edge_geoms = map(extract_edge_geometry, u, v, k, data)
    edge_weights = map(extract_edge_weight, u, v, k, data)
    edge_colors = map(extract_edge_color, u, v, k, data)
    gdf_edges = gpd.GeoDataFrame(data, geometry=list(edge_geoms))
    
    gdf_edges["u"] = u
    gdf_edges["v"] = v
    gdf_edges["color"] = [cmap(weight) for weight in norm_weights]
    gdf_edges = gdf_edges.set_index(["u", "v"])

    # Plot G.
    gdf_edges.plot(ax=ax, color=row['color'], linewidth=2)
    
    print("Show plot.")
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show()