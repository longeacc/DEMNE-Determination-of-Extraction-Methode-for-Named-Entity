import json

import matplotlib.pyplot as plt
import networkx as nx


def visualize_decision_tree(
    decision_config_path: str = "data/decision_config.json",
    output_path: str = "Results/figures/decision_tree_visualization.png",
    output_format: str = "png",
) -> None:
    """
    Generate a decision tree figure from the config file.
    """
    # Load config
    try:
        with open(decision_config_path, encoding="utf-8") as f:
            config = json.load(f)

        if "entities" in config:
            entities_config = config["entities"]
        else:
            # Fallback for flat (legacy) format
            entities_config = {k: v for k, v in config.items() if isinstance(v, dict)}

    except FileNotFoundError:
        print(f"Error: {decision_config_path} not found.")
        return

    g_graph = nx.DiGraph()

    # Root node
    root = "DuraXELL\nDecision Node"
    g_graph.add_node(root, color="#2196F3", shape="box")

    # Colors by method
    colors = {
        "RULES": "#4CAF50",  # Green
        "RULES DEFAULT": "#4CAF50",
        "TBM": "#FFC107",  # Yellow/Amber
        "TBM DEFAULT": "#FFC107",
        "TRANSFORMER": "#FF9800",  # Orange
        "LLM": "#F44336",  # Red
    }

    # One-level graph (Entity -> Method): visualises the final decision per entity.
    # Visualising the full logical tree (conditions) would require a deeper pass.

    for entity, details in entities_config.items():
        # Skip non-dict keys in flat format
        if not isinstance(details, dict):
            continue

        method = details.get("method", "Unknown")

        # Entity node
        entity_node = entity
        g_graph.add_node(entity_node, color="lightgrey", shape="ellipse")
        g_graph.add_edge(root, entity_node)

        # Method node (leaf)
        # Metrics may be nested under "metrics" or at the root level
        metrics = details.get("metrics", details)
        te_val = metrics.get("Te", "N/A")

        leaf_label = f"{method}\n(Te={te_val})"
        color = colors.get(method, "#9E9E9E")

        # Unique ID to avoid duplicate method nodes
        leaf_id = f"{entity}_{method}"
        g_graph.add_node(leaf_id, label=leaf_label, color=color, shape="box", style="filled")
        g_graph.add_edge(entity_node, leaf_id)

    # Draw
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(g_graph, seed=42)

    # A hierarchical layout could be used for a cleaner view if needed
    # (graphviz often required for hierarchical layout; using spring for portability)

    node_colors = [
        nx.get_node_attributes(g_graph, "color").get(n, "#FFFFFF") for n in g_graph.nodes()
    ]
    labels = {n: g_graph.nodes[n].get("label", n) for n in g_graph.nodes()}

    nx.draw(
        g_graph,
        pos,
        with_labels=True,
        labels=labels,
        node_color=node_colors,
        node_size=3000,
        font_size=8,
        font_weight="bold",
        arrows=True,
    )

    plt.title("DEMNE Decision Tree by Entity")
    plt.axis("off")

    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format=output_format, dpi=300)
    plt.close()
    print(f"Visualisation saved: {output_path}")


if __name__ == "__main__":
    visualize_decision_tree()
