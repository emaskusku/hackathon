#%%

from tsp_planner import (
    plan_tsp_instance,
    plot_feasible_graph,
    plot_best_tsp_subset,
)

result = plan_tsp_instance(
    season="summer",
    expertise="intermediate",
    n_sites=6,
    base_node=3,
    preferred_types=["Peak"],
    require_preferred=False,
    preference_bonus=60,
    max_town_like=2,
    min_degree=2,
    use_time_filter=True,
    altitude_factor=0.05,
    zero_replacement=5
)

print("Best nodes:", result["best_nodes"])
print("Weight matrix shape:", None if result["weight_matrix"] is None else result["weight_matrix"].shape)
print("Weight matrix:")
print(result["weight_matrix"])

print("Labels:")
print(result["matrix_labels"])

plot_feasible_graph(result["G_selected"], base_node=3)

if result["best_nodes"] is not None:
    plot_best_tsp_subset(
        G_selected=result["G_selected"],
        selected_df=result["selected_places_df"],
        best_nodes=result["best_nodes"],
        base_node=3
    )

#%%