
import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================================================
# CONSTANTS
# =========================================================

GEAR_RANK = {
    "Urban": 1,
    "Trail": 2,
    "Mountain": 3,
    "Snow": 4,
}

GROUP_RULES = {
    "beginner": {
        "max_gain": 1000,         
        "max_time": 8*60,         
        "max_gear": 3,           
    },
    "intermediate": {
        "max_gain": 2000,
        "max_time": 12*60,
        "max_gear": 4,          
    },
    "advanced": {
        "max_gain": 2500,
        "max_time": 12*60,
        "max_gear": 4,         
    },
}

# =========================================================
# DATA BUILDERS
# =========================================================

def build_places_df() -> pd.DataFrame:
    places_data = [
        (1, "Pico Aneto", "Peak", 3404, "Snow", "Snow"),
        (2, "Tuca Alba", "Peak", 3122, "Snow", "Mountain"),
        (3, "Benasque", "Town", 1135, "Urban", "Urban"),
        (4, "Cerler", "Town", 1530, "Urban", "Urban"),
        (5, "Ancils", "Town", 1140, "Urban", "Urban"),
        (6, "Portillón de Benás", "Landmark", 2444, "Mountain", "Trail"),
        (7, "Ski resort", "Snow", 1530, "Snow", "Trail"),
        (8, "Baños de Benasque", "Resting area", 1550, "Urban", "Urban"),
        (9, "Hospital de Benasque", "Resting area", 1747, "Urban", "Urban"),
        (10, "Forau d'aiguallut", "Landmark", 2020, "Mountain", "Trail"),
        (11, "Tres Cascadas", "Landmark", 1900, "Mountain", "Trail"),
        (12, "Salvaguardia", "Peak", 2736, "Snow", "Mountain"),
        (13, "Tuca Maladeta", "Peak", 3312, "Snow", "Mountain"),
        (14, "Cap Llauset", "Refugio", 2425, "Snow", "Mountain"),
        (15, "Ibón Cregüeña", "Lake", 2632, "Snow", "Mountain"),
        (16, "Batisielles", "Lake", 2216, "Mountain", "Trail"),
        (17, "Eriste", "Town", 1089, "Urban", "Urban"),
        (18, "Ibón Eriste", "Lake", 2407, "Snow", "Mountain"),
        (19, "Tempestades", "Peak", 3289, "Snow", "Snow"),
        (20, "La Besurta", "Resting area", 1860, "Trail", "Trail"),
        (21, "La Renclusa", "Refugio", 2160, "Snow", "Trail"),
        (22, "Escaleta", "Lake", 2630, "Snow", "Mountain"),
        (23, "Mulleres", "Peak", 3013, "Snow", "Snow"),
        (24, "Salterillo", "Lake", 2460, "Snow", "Mountain"),
        (25, "Tres Barrancos", "Landmark", 1460, "Trail", "Trail"),
    ]
    return pd.DataFrame(
        places_data,
        columns=["id", "place", "type", "elevation", "winter_gear", "summer_gear"]
    )


def build_edges_raw():
    return [
    (1, 21, "4'50"),
    (1, 25, "4'35"),

    (2, 8, "4'20"),

    (3, 4, "1'25"),
    (3, 5, "0'45"),
    (3, 8, "0"),
    (3, 9, "0"),
    (3, 16, "3'10"),
    (3, 17, "0"),
    (3, 20, "0"),
    (3, 25, "1'00"),

    (4, 5, "1'30"),
    (4, 7, "0"),
    (4, 8, "0"),
    (4, 9, "0"),
    (4, 11, "1'10"),
    (4, 17, "0"),
    (4, 20, "0"),

    (5, 8, "0"),
    (5, 9, "0"),
    (5, 17, "0"),
    (5, 20, "0"),

    (6, 9, "2'25"),
    (6, 12, "1'00"),

    (8, 15, "3'35"),
    (8, 20, "0"),

    (9, 20, "0"),

    (10, 21, "1'00"),
    (10, 22, "1'50"),

    (13, 20, "4'20"),

    (14, 19, "3'40"),
    (14, 21, "5'50"),

    (16, 18, "2'30"),

    (20, 21, "4'20"),
    (20, 22, "3'05"),

    (21, 24, "1'35"),

    (22, 23, "1'15"),

    (23, 24, "0'50"),
    ]

def time_to_minutes(t: str, zero_replacement: int = 5) -> int:
    t = str(t).strip()
    if t == "0":
        return zero_replacement
    if "'" in t:
        h, m = t.split("'")
        return int(h) * 60 + int(m)
    raise ValueError(f"Unrecognized time format: {t}")


def build_graph(df_places: pd.DataFrame, edges_raw, zero_replacement: int = 5) -> nx.Graph:
    G = nx.Graph()

    for _, row in df_places.iterrows():
        G.add_node(
            row["id"],
            place=row["place"],
            type=row["type"],
            elevation=row["elevation"],
            winter_gear=row["winter_gear"],
            summer_gear=row["summer_gear"],
        )

    for i, j, t in edges_raw:
        G.add_edge(
            i,
            j,
            weight=time_to_minutes(t, zero_replacement=zero_replacement),
            time_str=t
        )

    return G


# =========================================================
# PRESELECTION
# =========================================================

def preselect_places(
    df_places: pd.DataFrame,
    G: nx.Graph,
    base_node: int,
    season: str,
    expertise: str,
    preferred_types=None,
    min_degree: int = 2,
    use_time_filter: bool = True
):
    season = season.lower().strip()
    expertise = expertise.lower().strip()
    preferred_types = preferred_types or []

    if season not in ["winter", "summer"]:
        raise ValueError("season must be 'winter' or 'summer'")
    if expertise not in GROUP_RULES:
        raise ValueError("expertise must be one of: beginner, intermediate, advanced")
    if base_node not in df_places["id"].values:
        raise ValueError(f"base_node {base_node} not found in df_places")

    gear_col = "winter_gear" if season == "winter" else "summer_gear"
    rules = GROUP_RULES[expertise]

    base_elevation = df_places.loc[df_places["id"] == base_node, "elevation"].iloc[0]

    df = df_places.copy()
    df["required_gear"] = df[gear_col]
    df["gear_rank"] = df["required_gear"].map(GEAR_RANK)

    shortest_times = nx.single_source_dijkstra_path_length(
        G, source=base_node, weight="weight"
    )
    df["travel_time_from_base"] = df["id"].map(shortest_times)
    df["reachable"] = df["travel_time_from_base"].notna()

    df["allowed_by_gear"] = df["gear_rank"] <= rules["max_gear"]
    df["elevation_gain"] = (df["elevation"] - base_elevation).clip(lower=0)

    if use_time_filter:
        df["allowed_by_time"] = df["travel_time_from_base"] <= rules["max_time"]
    else:
        df["allowed_by_time"] = True

    df["feasible_place"] = (
        df["reachable"] &
        df["allowed_by_gear"] &
        df["allowed_by_time"]
    )

    feasible_nodes = df.loc[df["feasible_place"], "id"].tolist()
    if base_node not in feasible_nodes:
        feasible_nodes.append(base_node)

    G_feasible = G.subgraph(feasible_nodes).copy()

    changed = True
    while changed:
        changed = False
        low_degree_nodes = [
            n for n in G_feasible.nodes()
            if n != base_node and G_feasible.degree(n) < min_degree
        ]
        if low_degree_nodes:
            G_feasible.remove_nodes_from(low_degree_nodes)
            changed = True

    if base_node in G_feasible.nodes():
        component_with_base = nx.node_connected_component(G_feasible, base_node)
        G_final = G_feasible.subgraph(component_with_base).copy()
    else:
        G_final = nx.Graph()

    final_nodes = list(G_final.nodes())
    df["in_final_graph"] = df["id"].isin(final_nodes)

    degree_map = dict(G_final.degree()) if len(G_final.nodes()) > 0 else {}
    df["final_degree"] = df["id"].map(degree_map).fillna(0).astype(int)

    df["is_preferred_type"] = df["type"].isin(preferred_types)

    final_df = df[df["in_final_graph"]].copy()
    has_preferred_type = final_df["is_preferred_type"].any() if not final_df.empty else False

    def exclusion_reason(row):
        reasons = []
        if not row["reachable"]:
            reasons.append("unreachable")
        else:
            if not row["allowed_by_gear"]:
                reasons.append("gear")
            if not row["allowed_by_time"]:
                reasons.append("travel_time")
            if row["feasible_place"] and not row["in_final_graph"]:
                reasons.append("graph_degree_or_connectivity")
        return ", ".join(reasons) if reasons else "selected"

    df["reason"] = df.apply(exclusion_reason, axis=1)

    summary = {
        "base_node": base_node,
        "base_place": df_places.loc[df_places["id"] == base_node, "place"].iloc[0],
        "base_elevation": int(base_elevation),
        "season": season,
        "expertise": expertise,
        "max_time": rules["max_time"],
        "max_gear": rules["max_gear"],
        "time_filter_enabled": use_time_filter,
        "preferred_types": preferred_types,
        "has_preferred_type_in_final_graph": has_preferred_type,
        "final_node_count": G_final.number_of_nodes(),
        "final_edge_count": G_final.number_of_edges(),
    }

    return df, final_df, G_final, summary


# =========================================================
# NODE SELECTION
# =========================================================

def choose_n_places_for_tsp(
    G_selected: nx.Graph,
    selected_df: pd.DataFrame,
    n_sites: int,
    base_node: int = 3,
    preferred_types=None,
    require_preferred: bool = False,
    preference_bonus: float = 0,
    max_town_like=None
):
    if n_sites < 2:
        raise ValueError("n_sites must be at least 2")

    preferred_types = preferred_types or []
    candidate_nodes = [n for n in G_selected.nodes() if n != base_node]

    if len(candidate_nodes) < n_sites - 1:
        raise ValueError(
            f"Not enough candidate nodes: need {n_sites - 1}, have {len(candidate_nodes)}"
        )

    best_combo = None
    best_score = float("inf")
    results = []

    rejected_no_path = 0
    rejected_no_preference = 0
    rejected_town_limit = 0
    total_combinations = 0

    for combo in combinations(candidate_nodes, n_sites - 1):
        total_combinations += 1
        nodes = [base_node] + list(combo)
        sub_df = selected_df[selected_df["id"].isin(nodes)].copy()

        if max_town_like is not None:
            town_like_count = sub_df["type"].isin(["Town", "Resting area"]).sum()
            if town_like_count > max_town_like:
                rejected_town_limit += 1
                continue

        has_preferred = sub_df["type"].isin(preferred_types).any() if preferred_types else False
        if require_preferred and preferred_types and not has_preferred:
            rejected_no_preference += 1
            continue

        total_time = 0
        valid = True

        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                try:
                    d = nx.shortest_path_length(G_selected, source=u, target=v, weight="weight")
                    total_time += d
                except nx.NetworkXNoPath:
                    rejected_no_path += 1
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            continue

        if preferred_types and has_preferred:
            total_time -= preference_bonus

        results.append({
            "nodes": nodes,
            "score": total_time,
            "has_preferred": has_preferred,
            "types": sub_df[["id", "place", "type"]].sort_values("id").to_dict("records")
        })

        if total_time < best_score:
            best_score = total_time
            best_combo = nodes

    debug_info = {
        "n_sites": n_sites,
        "total_candidates": len(candidate_nodes),
        "total_combinations": total_combinations,
        "valid_combinations": len(results),
        "rejected_no_path": rejected_no_path,
        "rejected_no_preference": rejected_no_preference,
        "rejected_town_limit": rejected_town_limit,
    }

    return best_combo, best_score, results, debug_info


# =========================================================
# WEIGHT MATRIX
# =========================================================

def build_weight_matrix_uphill(
    G_selected: nx.Graph,
    selected_df: pd.DataFrame,
    best_nodes,
    altitude_factor: float = 0.05
):
    n = len(best_nodes)

    place_lookup = selected_df.set_index("id")["place"].to_dict()
    elev_lookup = selected_df.set_index("id")["elevation"].to_dict()

    time_matrix = np.zeros((n, n), dtype=float)
    uphill_matrix = np.zeros((n, n), dtype=float)
    weight_matrix = np.zeros((n, n), dtype=float)

    for i, u in enumerate(best_nodes):
        for j, v in enumerate(best_nodes):
            if i == j:
                continue

            t = nx.shortest_path_length(G_selected, source=u, target=v, weight="weight")
            uphill = max(0, elev_lookup[v] - elev_lookup[u])
            w = t + altitude_factor * uphill

            time_matrix[i, j] = t
            uphill_matrix[i, j] = uphill
            weight_matrix[i, j] = w

    matrix_labels = [f"{node} - {place_lookup[node]}" for node in best_nodes]

    return weight_matrix, time_matrix, uphill_matrix, matrix_labels


# =========================================================
# ORCHESTRATION
# =========================================================
def plan_tsp_instance(
    season: str,
    expertise: str,
    n_sites: int,
    base_node: int = 3,
    preferred_types=None,
    require_preferred: bool = False,
    preference_bonus: float = 0,
    max_town_like=None,
    min_degree: int = 2,
    use_time_filter: bool = True,
    altitude_factor: float = 0.05,
    zero_replacement: int = 5
):
    preferred_types = preferred_types or []

    df_places = build_places_df()
    edges_raw = build_edges_raw()
    G = build_graph(df_places, edges_raw, zero_replacement=zero_replacement)

    full_eval_df, selected_df, G_selected, summary = preselect_places(
        df_places=df_places,
        G=G,
        base_node=base_node,
        season=season,
        expertise=expertise,
        preferred_types=preferred_types,
        min_degree=min_degree,
        use_time_filter=use_time_filter
    )

    best_nodes, best_score, all_results, selection_debug = choose_n_places_for_tsp(
        G_selected=G_selected,
        selected_df=selected_df,
        n_sites=n_sites,
        base_node=base_node,
        preferred_types=preferred_types,
        require_preferred=require_preferred,
        preference_bonus=preference_bonus,
        max_town_like=max_town_like
    )

    if best_nodes is None:
        return {
            "best_nodes": None,
            "best_score": None,
            "selected_places_df": None,
            "full_eval_df": full_eval_df,
            "G_selected": G_selected,
            "weight_matrix": None,
            "time_matrix": None,
            "uphill_matrix": None,
            "matrix_labels": None,
            "summary": summary,
            "selection_debug": selection_debug,
            "all_results": all_results,
        }

    chosen_places_df = (
        selected_df[selected_df["id"].isin(best_nodes)]
        .sort_values("id")
        .copy()
    )

    weight_matrix, time_matrix, uphill_matrix, matrix_labels = build_weight_matrix_uphill(
        G_selected=G_selected,
        selected_df=selected_df,
        best_nodes=best_nodes,
        altitude_factor=altitude_factor
    )

    return {
        "best_nodes": best_nodes,
        "best_score": best_score,
        "selected_places_df": chosen_places_df,
        "full_eval_df": full_eval_df,
        "G_selected": G_selected,
        "weight_matrix": weight_matrix,
        "time_matrix": time_matrix,
        "uphill_matrix": uphill_matrix,
        "matrix_labels": matrix_labels,
        "summary": summary,
        "selection_debug": selection_debug,
        "all_results": all_results,
    }

def plot_best_tsp_subset(G_selected, selected_df, best_nodes, base_node=None):
    if best_nodes is None:
        print("No best_nodes to plot.")
        return

    H = nx.Graph()

    for n in best_nodes:
        row = selected_df[selected_df["id"] == n].iloc[0]
        H.add_node(
            n,
            place=row["place"],
            type=row["type"],
            elevation=row["elevation"]
        )

    for i, u in enumerate(best_nodes):
        for v in best_nodes[i + 1:]:
            d = nx.shortest_path_length(G_selected, source=u, target=v, weight="weight")
            H.add_edge(u, v, weight=d)

    type_color_map = {
        "Peak": "red",
        "Lake": "blue",
        "Landmark": "orange",
        "Refugio": "green",
        "Resting area": "purple",
        "Town": "gray",
        "Snow": "cyan",
    }

    node_colors = [
        type_color_map.get(H.nodes[n]["type"], "black")
        for n in H.nodes()
    ]

    elevations = [H.nodes[n]["elevation"] for n in H.nodes()]
    min_elev = min(elevations)
    max_elev = max(elevations)

    if max_elev == min_elev:
        node_sizes = [900 for _ in H.nodes()]
    else:
        node_sizes = [
            500 + 1200 * ((H.nodes[n]["elevation"] - min_elev) / (max_elev - min_elev))
            for n in H.nodes()
        ]

    labels = {
        n: f"{n}\n{H.nodes[n]['place']}"
        for n in H.nodes()
    }

    edge_labels = {}
    for u, v in H.edges():
        w = H.edges[u, v]["weight"]
        h = int(w) // 60
        m = int(w) % 60
        edge_labels[(u, v)] = f"{h}h {m}m" if h > 0 else f"{m}m"

    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(H, seed=42, k=1.8)

    nx.draw_networkx_nodes(
        H,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.95
    )

    if base_node is not None and base_node in H.nodes():
        nx.draw_networkx_nodes(
            H,
            pos,
            nodelist=[base_node],
            node_color="gold",
            edgecolors="black",
            linewidths=2,
            node_size=[node_sizes[list(H.nodes()).index(base_node)] * 1.2]
        )

    nx.draw_networkx_edges(H, pos, width=2, alpha=0.8)
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=9)
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=8)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=place_type,
               markerfacecolor=color, markersize=10)
        for place_type, color in type_color_map.items()
    ]
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', label='Base node',
               markerfacecolor='gold', markeredgecolor='black', markersize=12)
    )

    plt.legend(handles=legend_elements, title="Legend", loc="best")
    plt.title("Chosen TSP Subset", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_feasible_graph(G_selected, base_node=None):
    if G_selected is None or G_selected.number_of_nodes() == 0:
        print("No feasible graph to plot.")
        return

    type_color_map = {
        "Peak": "red",
        "Lake": "blue",
        "Landmark": "orange",
        "Refugio": "green",
        "Resting area": "purple",
        "Town": "gray",
        "Snow": "cyan",
    }

    node_colors = [
        type_color_map.get(G_selected.nodes[n]["type"], "black")
        for n in G_selected.nodes()
    ]

    elevations = [G_selected.nodes[n]["elevation"] for n in G_selected.nodes()]
    min_elev = min(elevations)
    max_elev = max(elevations)

    if max_elev == min_elev:
        node_sizes = [700 for _ in G_selected.nodes()]
    else:
        node_sizes = [
            400 + 1200 * ((G_selected.nodes[n]["elevation"] - min_elev) / (max_elev - min_elev))
            for n in G_selected.nodes()
        ]

    node_labels = {
        n: f"{n}\n{G_selected.nodes[n]['place']}"
        for n in G_selected.nodes()
    }

    edge_labels = {}
    for u, v in G_selected.edges():
        w = G_selected.edges[u, v]["weight"]
        h = int(w) // 60
        m = int(w) % 60
        edge_labels[(u, v)] = f"{h}h {m}m" if h > 0 else f"{m}m"

    plt.figure(figsize=(16, 10))
    pos = nx.spring_layout(G_selected, seed=42, k=1.5)

    nx.draw_networkx_nodes(
        G_selected,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9
    )

    if base_node is not None and base_node in G_selected.nodes():
        nx.draw_networkx_nodes(
            G_selected,
            pos,
            nodelist=[base_node],
            node_color="gold",
            node_size=[node_sizes[list(G_selected.nodes()).index(base_node)] * 1.3],
            edgecolors="black",
            linewidths=2
        )

    nx.draw_networkx_edges(G_selected, pos, width=2, alpha=0.7)
    nx.draw_networkx_labels(G_selected, pos, labels=node_labels, font_size=8)
    nx.draw_networkx_edge_labels(G_selected, pos, edge_labels=edge_labels, font_size=8)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=place_type,
               markerfacecolor=color, markersize=10)
        for place_type, color in type_color_map.items()
    ]
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', label='Base node',
               markerfacecolor='gold', markeredgecolor='black', markersize=12)
    )

    plt.legend(handles=legend_elements, title="Legend", loc="best")
    plt.title("Feasible Graph After Preselection", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

#%%