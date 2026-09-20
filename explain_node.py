import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from models.GNNs.sage import SAGE


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "cora": {
        "root": "dataset/cora",
        "name": "Cora",
        "model": "sage_cora.pt",
        "classes": [
            "Case_Based",
            "Genetic_Algorithms",
            "Neural_Networks",
            "Probabilistic_Methods",
            "Reinforcement_Learning",
            "Rule_Learning",
            "Theory",
        ],
    },

    "citeseer": {
        "root": "dataset/citeseer",
        "name": "CiteSeer",
        "model": "sage_citeseer.pt",
        "classes": [
            "Agents",
            "AI",
            "DB",
            "IR",
            "ML",
            "HCI",
        ],
    },
}


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(dataset_name):

    config = DATASETS[dataset_name]

    dataset = Planetoid(
        root=config["root"],
        name=config["name"]
    )

    data = dataset[0].to(device)

    return dataset, data, config


# ============================================================
# LOAD GRAPH SAGE MODEL
# ============================================================

def load_model(data, dataset, model_file):

    print(f"Loading checkpoint: {model_file}")

    checkpoint = torch.load(
        model_file,
        map_location=device,
        weights_only=True
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Cora was saved as a checkpoint containing:
    #
    # {
    #     "model_state_dict": ...,
    #     "in_channels": ...,
    #     "hidden_channels": ...,
    #     "out_channels": ...,
    #     "num_layers": ...,
    #     "dropout": ...,
    #     "test_accuracy": ...
    # }
    #
    # CiteSeer was saved directly as a state_dict.
    #
    # This handles BOTH formats.
    # --------------------------------------------------------

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        print("Checkpoint format: full training checkpoint")

        state_dict = checkpoint["model_state_dict"]

        # Use the architecture stored in the checkpoint
        hidden_channels = checkpoint.get(
            "hidden_channels",
            128
        )

        num_layers = checkpoint.get(
            "num_layers",
            2
        )

        dropout = checkpoint.get(
            "dropout",
            0.5
        )

    else:

        print("Checkpoint format: state_dict")

        state_dict = checkpoint

        hidden_channels = 128
        num_layers = 2
        dropout = 0.5

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = SAGE(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    model.load_state_dict(state_dict)

    model.eval()

    return model


# ============================================================
# GET GRAPH NEIGHBORS
# ============================================================

def get_neighbors(data, node_id):

    edge_index = data.edge_index

    neighbors = set()

    for source, target in zip(
        edge_index[0].tolist(),
        edge_index[1].tolist()
    ):

        if source == node_id:
            neighbors.add(target)

        if target == node_id:
            neighbors.add(source)

    neighbors.discard(node_id)

    return sorted(neighbors)


# ============================================================
# EXPLAIN NODE
# ============================================================

def explain_node(
    node_id,
    dataset_name,
    dataset,
    data,
    model,
    class_names
):

    # --------------------------------------------------------
    # Validate node
    # --------------------------------------------------------

    if node_id < 0 or node_id >= data.num_nodes:

        raise ValueError(
            f"Invalid node ID {node_id}. "
            f"Valid range: 0-{data.num_nodes - 1}"
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            data.x,
            data.edge_index
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )

        predictions = logits.argmax(
            dim=1
        )

    predicted_class = int(
        predictions[node_id].item()
    )

    actual_class = int(
        data.y[node_id].item()
    )

    confidence = (
        probabilities[
            node_id,
            predicted_class
        ].item()
        * 100
    )

    correct = (
        predicted_class == actual_class
    )

    # --------------------------------------------------------
    # GRADIENT-BASED XAI
    # --------------------------------------------------------

    x = data.x.detach().clone()

    x.requires_grad_(True)

    model.zero_grad()

    output = model(
        x,
        data.edge_index
    )

    target_score = output[
        node_id,
        predicted_class
    ]

    target_score.backward()

    importance = x.grad[
        node_id
    ].abs()

    # Top 20 features

    top_k = min(
        20,
        data.num_features
    )

    values, indices = torch.topk(
        importance,
        k=top_k
    )

    important_features = []

    for rank, (
        feature_index,
        importance_value
    ) in enumerate(
        zip(
            indices.tolist(),
            values.tolist()
        ),
        start=1
    ):

        feature_value = x[
            node_id,
            feature_index
        ].item()

        important_features.append(
            {
                "rank": rank,

                "feature_index": feature_index,

                "importance": float(
                    importance_value
                ),

                "feature_value": float(
                    feature_value
                ),

                "active": bool(
                    feature_value != 0
                ),
            }
        )

    # --------------------------------------------------------
    # GRAPH NEIGHBORS
    # --------------------------------------------------------

    neighbor_ids = get_neighbors(
        data,
        node_id
    )

    neighbors = []

    for neighbor_id in neighbor_ids:

        neighbor_prediction = int(
            predictions[
                neighbor_id
            ].item()
        )

        neighbor_actual = int(
            data.y[
                neighbor_id
            ].item()
        )

        neighbor_confidence = (
            probabilities[
                neighbor_id,
                neighbor_prediction
            ].item()
            * 100
        )

        neighbors.append(
            {
                "node_id": neighbor_id,

                "predicted_class":
                    class_names[
                        neighbor_prediction
                    ],

                "actual_class":
                    class_names[
                        neighbor_actual
                    ],

                "confidence":
                    neighbor_confidence,
            }
        )

    # --------------------------------------------------------
    # SUPPORTING NEIGHBORS
    # --------------------------------------------------------

    predicted_label = class_names[
        predicted_class
    ]

    supporting_neighbors = sum(
        1
        for neighbor in neighbors
        if neighbor["predicted_class"]
        == predicted_label
    )

    if len(neighbors) > 0:

        support_percentage = (
            supporting_neighbors
            / len(neighbors)
            * 100
        )

    else:

        support_percentage = 0.0

    # --------------------------------------------------------
    # NEIGHBOR CLASS DISTRIBUTION
    # --------------------------------------------------------

    neighbor_distribution = {}

    for neighbor in neighbors:

        label = neighbor[
            "predicted_class"
        ]

        neighbor_distribution[label] = (
            neighbor_distribution.get(
                label,
                0
            ) + 1
        )

    # --------------------------------------------------------
    # NATURAL LANGUAGE EXPLANATION
    # --------------------------------------------------------

    top_features_text = ", ".join(
        f'feature {feature["feature_index"]} '
        f'(importance '
        f'{feature["importance"]:.3f})'
        for feature in important_features[:5]
    )

    if len(neighbors) > 0:

        graph_text = (
            f"The node has "
            f"{len(neighbors)} citation "
            f"neighbors. "
            f"{supporting_neighbors} "
            f"({support_percentage:.1f}%) "
            f"have the same predicted class."
        )

    else:

        graph_text = (
            "The node has no citation "
            "neighbors in the loaded graph."
        )

    if correct:

        correctness_text = (
            "The prediction matches the "
            "ground-truth class."
        )

    else:

        correctness_text = (
            "The prediction does not match "
            "the ground-truth class."
        )

    explanation = (
        f"GraphSAGE classified "
        f"{dataset_name} node {node_id} "
        f"as {predicted_label} with "
        f"{confidence:.2f}% confidence. "
        f"{correctness_text} "
        f"Gradient-based feature attribution "
        f"identified {top_features_text} "
        f"among the most influential input "
        f"dimensions. "
        f"{graph_text}"
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 75)

    print(
        f"{dataset_name.upper()} "
        f"GRAPHSAGE + XAI EXPLANATION"
    )

    print("=" * 75)

    print(
        f"Node ID        : {node_id}"
    )

    print(
        f"Predicted class: "
        f"{predicted_label}"
    )

    print(
        f"Actual class   : "
        f"{class_names[actual_class]}"
    )

    print(
        f"Confidence     : "
        f"{confidence:.2f}%"
    )

    print(
        f"Correct        : "
        f"{'YES' if correct else 'NO'}"
    )

    print()
    print("-" * 75)
    print("TOP IMPORTANT FEATURES")
    print("-" * 75)

    for feature in important_features:

        print(
            f'{feature["rank"]:2d}. '
            f'Feature '
            f'{feature["feature_index"]:4d} | '
            f'Importance: '
            f'{feature["importance"]:.6f} | '
            f'Value: '
            f'{feature["feature_value"]} | '
            f'Active: '
            f'{feature["active"]}'
        )

    print()
    print("-" * 75)
    print("CITATION NEIGHBORS")
    print("-" * 75)

    if neighbors:

        for neighbor in neighbors:

            print(
                f'Node '
                f'{neighbor["node_id"]}: '
                f'{neighbor["predicted_class"]} '
                f'('
                f'{neighbor["confidence"]:.2f}%'
                f')'
            )

    else:

        print("No neighbors found.")

    print()
    print(
        f"Supporting neighbors: "
        f"{supporting_neighbors}/"
        f"{len(neighbors)} "
        f"({support_percentage:.1f}%)"
    )

    print()
    print(
        "Neighbor prediction distribution:"
    )

    if neighbor_distribution:

        for label, count in sorted(
            neighbor_distribution.items()
        ):

            print(
                f"  {label}: {count}"
            )

    else:

        print("  None")

    print()
    print("-" * 75)
    print("EXPLANATION")
    print("-" * 75)

    print(explanation)

    print()
    print("=" * 75)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Universal GraphSAGE node "
            "explanation tool"
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "cora",
            "citeseer"
        ],
        help="Dataset to explain"
    )

    parser.add_argument(
        "--id",
        required=True,
        type=int,
        help="Node ID to explain"
    )

    args = parser.parse_args()

    config = DATASETS[
        args.dataset
    ]

    print(
        f"Device: {device}"
    )

    print(
        f"Loading dataset: "
        f"{config['name']}"
    )

    dataset, data, config = load_dataset(
        args.dataset
    )

    print(
        f"Nodes: {data.num_nodes}"
    )

    print(
        f"Features: {data.num_features}"
    )

    print(
        f"Classes: {dataset.num_classes}"
    )

    model = load_model(
        data,
        dataset,
        config["model"]
    )

    print(
        f"Loaded model: "
        f"{config['model']}"
    )

    explain_node(
        args.id,
        config["name"],
        dataset,
        data,
        model,
        config["classes"]
    )


if __name__ == "__main__":
    main()
