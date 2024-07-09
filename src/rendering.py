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
    ox.plot_graph(G, bgcolor="#ffffff", node_color=nc, edge_color=ec)


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
    ox.plot_graph(F, bgcolor="#ffffff", node_color=nc, edge_color=ec)


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
    ox.plot_graph(G, bgcolor="#ffffff", node_color=nc, edge_color=ec)



# Render curve and graph
def plot_graph_and_curve(G, ps):

    G = G.copy()
    nx.set_node_attributes(G, 2, name="gid")
    nx.set_edge_attributes(G, 2, name="gid")

    # Construct subgraph from ps.
    H = convert_paths_into_graph([ps], nid=max(G.nodes())+1)
    nx.set_node_attributes(H, 1, name="gid")
    nx.set_edge_attributes(H, 1, name="gid")
    H.graph['crs'] = "EPSG:4326"
    H = nx.MultiDiGraph(H)

    F = nx.compose(G,H)
    nc = ox.plot.get_node_colors_by_attr(F, "gid", cmap="winter")
    ec = ox.plot.get_edge_colors_by_attr(F, "gid", cmap="winter")

    ox.plot_graph(F, bgcolor="#ffffff", node_color=nc, edge_color=ec)


def plot_graph_and_curves(G, ps, qs):

    G = G.copy()
    nx.set_node_attributes(G, 2, name="gid")
    nx.set_edge_attributes(G, 2, name="gid")

    # Construct subgraph for ps.
    H = convert_paths_into_graph([ps], nid=max(G.nodes())+1)
    nx.set_node_attributes(H, 1, name="gid")
    nx.set_edge_attributes(H, 1, name="gid")
    H.graph['crs'] = "EPSG:4326"
    H = nx.MultiDiGraph(H)

    F = nx.compose(G,H)

    # Construct subgraph for qs.
    H = convert_paths_into_graph([qs], nid=max(F.nodes())+1)
    nx.set_node_attributes(H, 3, name="gid")
    nx.set_edge_attributes(H, 3, name="gid")
    H.graph['crs'] = "EPSG:4326"
    H = nx.MultiDiGraph(H)

    F = nx.compose(F,H)

    nc = ox.plot.get_node_colors_by_attr(F, "gid", cmap="Paired")
    ec = ox.plot.get_edge_colors_by_attr(F, "gid", cmap="Paired")

    ox.plot_graph(F, bgcolor="#ffffff", node_color=nc, edge_color=ec)