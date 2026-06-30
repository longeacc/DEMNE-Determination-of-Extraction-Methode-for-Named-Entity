import json

import matplotlib.pyplot as plt
import networkx as nx


def visualize_decision_tree(
    decision_config_path: str = "data/decision_config.json",
    output_path: str = "Results/figures/decision_tree_visualization.png",
    format: str = "png",
) -> None:
    """
    Génère une figure de l'arbre de décision à partir du fichier de config.
    """
    # Chargement de la config
    try:
        with open(decision_config_path, encoding="utf-8") as f:
            config = json.load(f)

        # Adaptation au format JSON
        if "entities" in config:
            entities_config = config["entities"]
        else:
            # Fallback pour format plat (ancien)
            entities_config = {k: v for k, v in config.items() if isinstance(v, dict)}

    except FileNotFoundError:
        print(f"Erreur: {decision_config_path} introuvable.")
        return

    # Création du graphe
    g_graph = nx.DiGraph()

    # Nœud racine
    root = "DuraXELL\nDecision Node"
    g_graph.add_node(root, color="#2196F3", shape="box")

    # Couleurs par méthode
    colors = {
        "RÈGLES": "#4CAF50",  # Vert
        "RÈGLES PAR DÉFAUT": "#4CAF50",
        "ML LÉGER": "#FFC107",  # Jaune/Ambre
        "ML LÉGER PAR DÉFAUT": "#FFC107",
        "TRANSFORMER BIDIRECTIONNEL": "#FF9800",  # Orange
        "LLM": "#F44336",  # Rouge
    }

    # Construction du graphe (un niveau de profondeur : Entité -> Méthode)
    # L'arbre réel est logique (if/else), mais ici on visualise le résultat final :
    # Quel chemin a été pris pour chaque entité.

    # Pour visualiser l'arbre logique complet (les conditions), ce serait plus complexe.
    # Ici, on représente la décision finale pour chaque entité comme demandé.

    for entity, details in entities_config.items():
        # Ignorer les clés non-dictionnaires si format plat
        if not isinstance(details, dict):
            continue

        method = details.get("method", "Unknown")

        # Nœud entité
        entity_node = entity
        g_graph.add_node(entity_node, color="lightgrey", shape="ellipse")
        g_graph.add_edge(root, entity_node)

        # Nœud méthode (Feuille)
        # Gestion des métriques pouvant être dans "metrics" ou à la racine
        metrics = details.get("metrics", details)
        te_val = metrics.get("Te", "N/A")

        leaf_label = f"{method}\n(Te={te_val})"
        color = colors.get(method, "#9E9E9E")

        # Pour éviter les doublons de nœuds méthodes identiques, on utilise un ID unique
        leaf_id = f"{entity}_{method}"
        g_graph.add_node(leaf_id, label=leaf_label, color=color, shape="box", style="filled")
        g_graph.add_edge(entity_node, leaf_id)

    # Dessin
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(g_graph, seed=42)

    # On peut utiliser une disposition hiérarchique plus propre si besoin
    # (nécessite graphviz souvent, donc on reste sur spring ou shell pour compatibilité)

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

    plt.title("Arbre de Décision DuraXELL par Entité")
    plt.axis("off")

    # Sauvegarde
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format=format, dpi=300)
    plt.close()
    print(f"Visualisation sauvegardée : {output_path}")


if __name__ == "__main__":
    visualize_decision_tree()
