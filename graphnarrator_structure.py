import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from models.GNNs.sage import SAGE


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


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

def load_dataset(dataset_name):

    config = DATASETS[dataset_name]

    dataset = Planetoid(
        root=config["root"],
        name=config["name"]
    )

    data = dataset[0].to(device)

    return dataset, data, config


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

def load_model(data, dataset, model_file):

    checkpoint = torch.load(
        model_file,
        map_location=device,
        weights_only=True
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

        hidden_channels = checkpoint.get(
            "hidden_channels", 128
        )

        num_layers = checkpoint.get(
            "num_layers", 2
        )

        dropout = checkpoint.get(
            "dropout", 0.5
        )

    else:
        state_dict = checkpoint
        hidden_channels = 128
        num_layers = 2
        dropout = 0.5

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


# ------------------------------------------------------------
# Build adjacency list
# ------------------------------------------------------------

def build_adjacency(data):

    adjacency = {
        i: set()
        for i in range(data.num_nodes)
    }

    sources = data.edge_index[0].tolist()
    targets = data.edge_index[1].tolist()

    for source, target in zip(sources, targets):

        adjacency[source].add(target)
        adjacency[target].add(source)

    return adjacency


# ------------------------------------------------------------
# Get 1-hop and 2-hop neighborhood
# ------------------------------------------------------------

def get_two_hop_graph(
    adjacency,
    root
):

    one_hop = sorted(
        adjacency[root]
    )

    two_hop = {}

    one_hop_set = set(one_hop)

    for neighbor in one_hop:

        candidates = (
            adjacency[neighbor]
            - one_hop_set
            - {root}
        )

        two_hop[neighbor] = sorted(
            candidates
        )

    return one_hop, two_hop


# ------------------------------------------------------------
# Generate predictions
# ------------------------------------------------------------

def get_predictions(
    model,
    data
):

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

    return (
        predictions,
        probabilities
    )


# ------------------------------------------------------------
# Build verbalized graph
# ------------------------------------------------------------

def build_verbalized_graph(
    root,
    one_hop,
    two_hop,
    predictions,
    probabilities,
    actual_labels,
    class_names
):

    root_prediction = int(
        predictions[root].item()
    )

    root_actual = int(
        actual_labels[root].item()
    )

    root_confidence = (
        probabilities[
            root,
            root_prediction
        ].item()
        * 100
    )

    lines = []

    lines.append(
        f"ROOT: Node {root}"
    )

    lines.append(
        f"Prediction: "
        f"{class_names[root_prediction]}"
    )

    lines.append(
        f"Actual: "
        f"{class_names[root_actual]}"
    )

    lines.append(
        f"Confidence: "
        f"{root_confidence:.2f}%"
    )

    lines.append("")

    lines.append(
        "DIRECT NEIGHBORS:"
    )

    if not one_hop:

        lines.append(
            "  None"
        )

    for index, neighbor in enumerate(
        one_hop,
        start=1
    ):

        prediction = int(
            predictions[neighbor].item()
        )

        actual = int(
            actual_labels[neighbor].item()
        )

        confidence = (
            probabilities[
                neighbor,
                prediction
            ].item()
            * 100
        )

        lines.append(
            f"  Node-{index}: "
            f"Node {neighbor} | "
            f"Prediction={class_names[prediction]} | "
            f"Actual={class_names[actual]} | "
            f"Confidence={confidence:.2f}%"
        )

        children = two_hop.get(
            neighbor,
            []
        )

        for child_index, child in enumerate(
            children,
            start=1
        ):

            child_prediction = int(
                predictions[child].item()
            )

            child_actual = int(
                actual_labels[child].item()
            )

            child_confidence = (
                probabilities[
                    child,
                    child_prediction
                ].item()
                * 100
            )

            lines.append(
                f"    Node-{index}.{child_index}: "
                f"Node {child} | "
                f"Prediction={class_names[child_prediction]} | "
                f"Actual={class_names[child_actual]} | "
                f"Confidence={child_confidence:.2f}%"
            )

    return "\n".join(lines)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cora", "citeseer"]
    )

    parser.add_argument(
        "--id",
        required=True,
        type=int
    )

    args = parser.parse_args()

    config = DATASETS[
        args.dataset
    ]

    print(
        f"Device: {device}"
    )

    dataset, data, config = load_dataset(
        args.dataset
    )

    print(
        f"Dataset: {config['name']}"
    )

    print(
        f"Nodes: {data.num_nodes}"
    )

    print(
        f"Features: {data.num_features}"
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

    adjacency = build_adjacency(
        data
    )

    root = args.id

    if root < 0 or root >= data.num_nodes:

        raise ValueError(
            f"Invalid node ID {root}. "
            f"Valid range: "
            f"0-{data.num_nodes - 1}"
        )

    predictions, probabilities = (
        get_predictions(
            model,
            data
        )
    )

    one_hop, two_hop = get_two_hop_graph(
        adjacency,
        root
    )

    verbalized_graph = (
        build_verbalized_graph(
            root,
            one_hop,
            two_hop,
            predictions,
            probabilities,
            data.y,
            config["classes"]
        )
    )

    print()
    print("=" * 80)
    print("GRAPHNARRATOR-STYLE VERBALIZED GRAPH")
    print("=" * 80)
    print()
    print(verbalized_graph)
    print()
    print("=" * 80)

    output_file = (
        f"{args.dataset}_node_{root}_graph.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(verbalized_graph)

    print()
    print(
        f"Saved to: {output_file}"
    )


if __name__ == "__main__":
    main()
