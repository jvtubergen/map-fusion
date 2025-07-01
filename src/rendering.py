from external import *
from graph_deduplicating import *
from utilities import *

import matplotlib as mpl
from matplotlib import cm
from matplotlib import colormaps
from matplotlib import colors

color_name_index = {
    "blue": 0,
    "orange": 2,
    "green": 4,
    "red": 6,
    "purple": 8,
    "brown": 10,
    "pink": 12,
    "gray": 14,
    "ugly": 16,
    "cyan": 18
}
color_names = list(color_name_index.keys())

def my_colors(name, dark = True): # :)
    colors = mpl.color_sequences["tab20"]
    index = color_name_index[name]
    actual_index = index + 1 if not dark else index
    return colors[actual_index]


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


# Preplot a graph. Can be performed multiple times to render graphs together.
# * Optional to have general node and/or edge rendering properties (thus rendering all nodes/edges the same).
# * Otherwise each edge and node is checked for rendering properties in its attributes (thus each node and edge is considered uniquely).
def preplot_graph(G, ax, node_properties=None, edge_properties=None): 

    print("Plotting nodes.")
    # Nodes.
    uv, data = zip(*G.nodes(data=True))
    gdf_nodes = gpd.GeoDataFrame(data, index=uv)

    if node_properties != None:
        # Render all nodes with same render properties.
        render_attributes = node_properties
    else:
        # Render nodes with their specific render properties (stored under its attributes).
        render_attributes = {}
        for prop in ["color"]: 
            if prop in gdf_nodes.keys():
                render_attributes["color"] = gdf_nodes["color"]

    plotted_nodes = ax.scatter(**render_attributes, x=gdf_nodes["x"], y=gdf_nodes["y"])
    
    print("Plotting edges.")
    # Edges.
    x_lookup = nx.get_node_attributes(G, "x")
    y_lookup = nx.get_node_attributes(G, "y")

    def extract_edge_geometry(u, v, data):
        if not G.graph["simplified"]:
            return LineString((Point((x_lookup[u], y_lookup[u])), Point((x_lookup[v], y_lookup[v]))))
        else:
            return data["geometry"] # Always exists on simplified graph.

    if not G.graph["simplified"]:
        u, v, data = zip(*[(u, v, attrs) for (u, v), attrs in iterate_edges(G)])
        edge_geoms = map(extract_edge_geometry, u, v, data)
        gdf_edges  = gpd.GeoDataFrame(data, geometry=list(edge_geoms))
        gdf_edges["u"] = u
        gdf_edges["v"] = v
        gdf_edges = gdf_edges.set_index(["u", "v"])
    else:
        u, v, k, data = zip(*[(u, v, k, attrs) for (u, v, k), attrs in iterate_edges(G)])
        gdf_edges  = gpd.GeoDataFrame(data) # Simplified edges already have geometry attribute.
        gdf_edges["u"] = u
        gdf_edges["v"] = v
        gdf_edges["k"] = k
        gdf_edges = gdf_edges.set_index(["u", "v", "k"])

    if edge_properties != None: 
        # Render all edges with same render properties.
        render_attributes = edge_properties
    else:
        # Render edges with their specific render properties (stored under its attributes).
        render_attributes = {}
        for prop in ["color", "linestyle", "linewidth"]: 
            if prop in gdf_edges.keys():
                render_attributes[prop] = gdf_edges[prop]
        
        # # If edges don't have color but nodes do, inherit node colors for edges
        # if "color" not in render_attributes and "color" in gdf_nodes.keys():
        #     # Map edge colors to their source node colors
        #     edge_colors = []
        #     for idx in gdf_edges.index:
        #         if isinstance(idx, tuple) and len(idx) >= 2:
        #             source_node = idx[0]  # u node
        #             if source_node in gdf_nodes.index:
        #                 edge_colors.append(gdf_nodes.loc[source_node, "color"])
        #             else:
        #                 edge_colors.append("blue")  # fallback
        #         else:
        #             edge_colors.append("blue")  # fallback
        #     render_attributes["color"] = edge_colors

    plotted_edges = gdf_edges.plot(ax=ax, **render_attributes).collections[-1]

    return plotted_nodes, plotted_edges


def preplot_curve(ps, ax, **properties):
    # Construct GeoDataFrame .
    edge = dataframe({"geometry": to_linestring(ps)}, index=[0])
    # edge.plot(ax=ax, color=color, linewidth=linewidth, linestyle=linestyle)
    edge.plot(ax=ax, **properties)


# Render target graph (dotted gray) + curve (green) + path (blue).
#   Example usage:
#   plot_without_projection2([S], [])
#   plot_without_projection2([], [ (random_curve(),{"color":(0,0,0,1)}) ])
#   plot_without_projection2([], [ (random_curve(),{"color":(0,0,0,1), "linewidth":1, "linestyle": ":"}), (random_curve(), {"color":(0.1,0.5,0.1,1), "linewidth":3}) ])
def plot_without_projection(Gs, pss):

    fig, ax = plt.subplots()

    for i, obj in enumerate(Gs):
        print(f"Plotting graph {i}.")
        if type(obj) == tuple:
            G, properties = obj
            preplot_graph(G,  ax, **properties) 
        else:
            G = obj
            properties = {"color": my_colors(color_names[i])}
            preplot_graph(G,  ax, node_properties=properties, edge_properties=properties)

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

    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        # print (f'x = {ix}, y = {iy}')
        coord = (float(iy), float(ix))
        print(coord)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

def render_as_svg(filename, Gs, pss):
    fig, ax = plt.subplots()

    for i, obj in enumerate(Gs):
        print(f"Plotting graph {i}.")
        if type(obj) == tuple:
            G, properties = obj
            preplot_graph(G,  ax, **properties) 
        else:
            G = obj
            node_properties = {"color": my_colors(color_names[i]), "s": 0.3}
            edge_properties = {"color": my_colors(color_names[i]), "linewidth": 0.3}
            preplot_graph(G,  ax, node_properties=node_properties, edge_properties=edge_properties)

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

    plt.axis("off")
    plt.savefig(filename)


# Plot a list of graphs.
def plot_graphs(graphs):
    plot_without_projection(graphs, [])


# Example: `render_graphs("test.svg", [osm, gps])`
def render_graphs(filename, graphs):
    render_as_svg(filename, graphs, [])


# Annotate duplicated nodes as red.
def annotate_duplicated_nodes(G):
    duplicated = set([nid for group in duplicated_nodes(G) for nid in group])
    # print("duplicated:", duplicated)
    for nid, attrs in G.nodes(data=True):
        if nid in duplicated:
            attrs["color"] = (1., 0, 0, 1.) # Make duplicated node red.
            # print("Found duplicate", nid)
        else:
            attrs["color"] = (0, 0, 0, 1)


# Map each "render" attribute to a "color" and 
def color_mapper(render):
    match render:
        case "injected":
            return (0.3, 1, 0.3, 1) # green
        case "deleted":
            return (1, 0.3, 0.3, 1) # red
        case "connection":
            return (0.3, 0.3, 1, 1) # blue
        case "original":
            return (0, 0, 0, 1) # black
def linestyle_mapper(render):
    match render:
        case "injected":
            return "-" 
        case "deleted":
            return "-" 
        case "connection":
            return ":"
        case "original":
            return "-"
def linewidth_mapper(render):
    match render:
        case "injected":
            return 2 
        case "deleted":
            return 2
        case "connection":
            return 2 
        case "original":
            return 1

# Apply coloring and styling to nodes and edges by their "render" attribute.
def apply_coloring(G):

    # Fill in "render" attribute for those nodes/edges missing it.
    for _, attributes in iterate_nodes(G):
        if "render" not in attributes:
            attributes["render"] = "original"
    for _, attributes in iterate_edges(G):
        if "render" not in attributes:
            attributes["render"] = "original"

    # Map render type to render styling.
    for _, attributes in iterate_nodes(G):
        attributes["color"] = color_mapper(attributes["render"])
    for _, attributes in iterate_edges(G):
        attributes["color"] = color_mapper(attributes["render"])
        attributes["linestyle"] = linestyle_mapper(attributes["render"])
        attributes["linewidth"] = linewidth_mapper(attributes["render"])
    
    return G


# Simple function to generate an image on a graph with a custom title and quality level.
def render_graph(graph, filename, quality="low", title=None):

    if quality == "low":    
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)  # 20 inches * 100 dpi = 2000 pixels
    else:
        fig, ax = plt.subplots(figsize=(100, 100), dpi=100)  # 20 inches * 100 dpi = 2000 pixels

    preplot_graph(graph,  ax) 

    if title != None:
        plt.title(title)

    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.savefig(filename, dpi=100, bbox_inches="tight")


# Simple function to plot a graph with a custom title.
def plot_graph(graph, title=None):
    fig, ax = plt.subplots(figsize=(100, 100))  # 20 inches * 100 dpi = 2000 pixels

    preplot_graph(graph,  ax) 

    if title != None:
        plt.title(title)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.tight_layout()

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()  # Full-screen mode

    plt.show()


def plot_graph_interactively(graph):

    fig, ax = plt.subplots(figsize=(100, 100))
    fig.subplots_adjust(left=0.3)  # Leave space for check buttons

    nodes, edges = preplot_graph(graph, ax)

    lines = {
        "nodes": nodes,
        "edges": edges
    }

    ax.legend()

    # Create CheckButtons
    rax = plt.axes([0.05, 0.4, 0.2, 0.2])  # Position of buttons
    labels = list(lines.keys())
    visibility = [line.get_visible() for line in lines.values()]
    check = CheckButtons(rax, labels, visibility)

    # Toggle function based on label
    def toggle_visibility(label):
        line = lines[label]  # Get the corresponding line
        line.set_visible(not line.get_visible())  # Toggle visibility
        ax.legend()  # Update legend
        plt.draw()

    # Connect check buttons to toggle function
    check.on_clicked(toggle_visibility)

    # fig.canvas.draw()
    # fig.canvas.flush_events()

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()  # Full-screen mode

    plt.tight_layout()
    plt.show()


# Render multiple graphs which you can enable/disable with interface buttons.
# Provide graphs as dictionary by label, e.g. `{"osm": read_graph(...), "gps": read_graph(...)}`
def plot_graphs_interactively(graphs):

    fig, ax = plt.subplots(figsize=(100, 100))
    fig.subplots_adjust(left=0.3)  # Leave space for check buttons

    plotted_graphs = {label: preplot_graph(apply_coloring(graph), ax) for label, graph in graphs.items()}

    ax.legend()

    labels = list(plotted_graphs.keys())

    # Only render first element at startup.
    for label in labels:
        plotted_graphs[label][0].set_visible(False)
        plotted_graphs[label][1].set_visible(False)

    plotted_graphs[labels[0]][0].set_visible(True)
    plotted_graphs[labels[0]][1].set_visible(True)

    # Create CheckButtons
    rax = plt.axes([0.05, 0.4, 0.2, 0.2])  # Position of buttons
    labels = list(plotted_graphs.keys())
    visibility = [plotted_graph[0].get_visible() for plotted_graph in plotted_graphs.values()]
    check = CheckButtons(rax, labels, visibility)

    # Toggle function based on label
    def toggle_visibility(label):

        for lbl in labels:
            plotted_graphs[lbl][0].set_visible(False)
            plotted_graphs[lbl][1].set_visible(False)

        plotted_graphs[label][0].set_visible(True)
        plotted_graphs[label][1].set_visible(True)

        ax.legend()  # Update legend
        plt.draw()
    
    key_label_map = {str(i+1): label for i, label in enumerate(labels)}
        
    def on_key_generator(label_map):

        def on_key(event):

            if event.key in label_map.keys():
                label = label_map[event.key]
                toggle_visibility(label)
                ax.legend()
                plt.draw()
        
        return on_key

    # Connect check buttons to toggle function
    check.on_clicked(toggle_visibility)

    fig.canvas.draw()
    fig.canvas.mpl_connect("key_press_event", on_key_generator(key_label_map))
    fig.canvas.flush_events()

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()  # Full-screen mode

    plt.tight_layout()
    plt.show()


# Render each map as a high and low-quality image.
@info()
def render_maps_to_images(maps):

    for quality in ["low", "high"]:
        for place in maps.keys():
            for map_variant in maps[place]:
                logger(f"{quality} - {place} - {map_variant}.")
                graph = maps[place][map_variant]
                graph = apply_coloring(graph)
                render_graph(graph, f"data/graph_images/{quality}-{place}-{map_variant}.png", quality=quality, title=f"{quality}-{place}-{map_variant}")


#TODO: Render graph as an SVG.
def render_graph_as_svg(graph, filename):
    graph = apply_coloring(graph)

    fig, ax = plt.subplots(figsize=(100, 62))  # 20 inches * 100 dpi = 2000 pixels

    # Update preplot for rendering to PDF.
    preplot_graph(graph,  ax) 

    fig.canvas.draw()
    fig.canvas.flush_events()

    # Turn off axes
    ax.set_axis_off()
    # Remove padding around the plot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # Remove margins
    plt.margins(0)
    # Remove frame
    ax.set_frame_on(False)

    plt.savefig(filename)