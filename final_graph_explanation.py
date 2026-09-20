import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from dataset import CORA
from models.GNNs.sage import SAGE


DATASETS = {
    "cora": {
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


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

def load_dataset(dataset_name):

    config = DATASETS[dataset_name]

    if dataset_name == "cora":

        data, citeid = CORA.get_cora_casestudy(0)

        data = data.to(DEVICE)

        num_classes = int(
            data.y.max().item()
        ) + 1

        return data, num_classes, config

    dataset = Planetoid(
        root=config["root"],
        name=config["name"]
    )

    data = dataset[0].to(DEVICE)

    return data, dataset.num_classes, config


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def load_model(
    data,
    num_classes,
    model_path
):

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint["model_state_dict"]

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

        state_dict = checkpoint
        hidden_channels = 128
        num_layers = 2
        dropout = 0.5

    model = SAGE(
        in_channels=data.x.shape[1],
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        num_layers=num_layers,
        dropout=dropout
    ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    return model


# ---------------------------------------------------------
# Adjacency
# ---------------------------------------------------------

def build_adjacency(data):

    adjacency = [
        set()
        for _ in range(data.num_nodes)
    ]

    for src, dst in zip(
        data.edge_index[0].tolist(),
        data.edge_index[1].tolist()
    ):

        adjacency[src].add(dst)
        adjacency[dst].add(src)

    return adjacency


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict(model, data):

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

    return logits, probabilities, predictions


# ---------------------------------------------------------
# Gradient feature attribution
# ---------------------------------------------------------

def feature_attribution(
    model,
    data,
    node,
    predicted_class
):

    model.zero_grad()

    x = data.x.detach().clone()
    x.requires_grad_(True)

    logits = model(
        x,
        data.edge_index
    )

    target_score = logits[
        node,
        predicted_class
    ]

    target_score.backward()

    gradient = x.grad[node].detach()

    importance = gradient.abs()

    top_k = min(
        20,
        importance.numel()
    )

    values, indices = torch.topk(
        importance,
        k=top_k
    )

    features = []

    for value, index in zip(
        values.tolist(),
        indices.tolist()
    ):

        features.append({
            "feature_index": int(index),
            "importance": float(value),
            "feature_value": float(
                data.x[node, index].item()
            ),
            "active": bool(
                data.x[node, index].item() != 0
            )
        })

    return features


# ---------------------------------------------------------
# Direct neighbor analysis
# ---------------------------------------------------------

def analyze_neighbors(
    adjacency,
    node,
    predictions,
    probabilities,
    actual_labels,
    class_names
):

    neighbors = sorted(
        adjacency[node]
    )

    root_prediction = int(
        predictions[node].item()
    )

    records = []
    supporting = 0

    for neighbor in neighbors:

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

        supports = (
            prediction == root_prediction
        )

        if supports:
            supporting += 1

        records.append({
            "node_id": int(neighbor),
            "predicted_class": class_names[
                prediction
            ],
            "actual_class": class_names[
                actual
            ],
            "confidence": round(
                confidence,
                2
            ),
            "supports_root_prediction": supports
        })

    support_percentage = (
        supporting / len(neighbors) * 100
        if neighbors
        else 0
    )

    return (
        records,
        supporting,
        support_percentage
    )


# ---------------------------------------------------------
# Select relevant 2-hop nodes
# ---------------------------------------------------------

def select_relevant_children(
    candidates,
    root_prediction,
    predictions,
    probabilities,
    max_children=3
):

    records = []

    for node in candidates:

        prediction = int(
            predictions[node].item()
        )

        confidence = (
            probabilities[
                node,
                prediction
            ].item()
            * 100
        )

        supports = (
            prediction == root_prediction
        )

        records.append({
            "node_id": int(node),
            "prediction_id": prediction,
            "confidence": confidence,
            "supports_root_prediction": supports
        })

    # Supporting nodes first, then confidence.
    records.sort(
        key=lambda x: (
            x["supports_root_prediction"],
            x["confidence"]
        ),
        reverse=True
    )

    selected = records[
        :max_children
    ]

    omitted = max(
        0,
        len(records) - len(selected)
    )

    return selected, omitted


# ---------------------------------------------------------
# GraphNarrator-style structure
# ---------------------------------------------------------

def build_graph_structure(
    adjacency,
    root,
    predictions,
    probabilities,
    actual_labels,
    class_names,
    max_children=3
):

    root_prediction = int(
        predictions[root].item()
    )

    direct_neighbors = sorted(
        adjacency[root]
    )

    direct_set = set(
        direct_neighbors
    )

    structure = []

    for direct_index, neighbor in enumerate(
        direct_neighbors,
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

        candidates = (
            adjacency[neighbor]
            - direct_set
            - {root}
        )

        selected, omitted = (
            select_relevant_children(
                candidates,
                root_prediction,
                predictions,
                probabilities,
                max_children
            )
        )

        children = []

        for child in selected:

            child_id = child["node_id"]

            child_prediction = int(
                predictions[child_id].item()
            )

            child_actual = int(
                actual_labels[child_id].item()
            )

            children.append({
                "node_id": child_id,
                "predicted_class": class_names[
                    child_prediction
                ],
                "actual_class": class_names[
                    child_actual
                ],
                "confidence": round(
                    child["confidence"],
                    2
                ),
                "supports_root_prediction":
                    child[
                        "supports_root_prediction"
                    ]
            })

        structure.append({
            "label": f"Node-{direct_index}",
            "node_id": int(neighbor),
            "predicted_class": class_names[
                prediction
            ],
            "actual_class": class_names[
                actual
            ],
            "confidence": round(
                confidence,
                2
            ),
            "children": children,
            "omitted_2hop_nodes": omitted
        })

    return structure


# ---------------------------------------------------------
# Human-readable explanation
# ---------------------------------------------------------

def generate_explanation(
    dataset_name,
    node,
    predicted_class,
    actual_class,
    confidence,
    correct,
    top_features,
    neighbors,
    supporting_neighbors,
    support_percentage
):

    active_features = [
        feature
        for feature in top_features
        if feature["active"]
    ]

    if correct:

        correctness = (
            "The prediction matches the ground-truth label."
        )

    else:

        correctness = (
            "The prediction does not match the "
            "ground-truth label."
        )

    if neighbors:

        graph_evidence = (
            f"{supporting_neighbors} of "
            f"{len(neighbors)} direct neighbors "
            f"({support_percentage:.1f}%) support "
            f"the predicted class."
        )

    else:

        graph_evidence = (
            "The target node has no direct neighbors."
        )

    return (
        f"GraphSAGE classified node {node} as "
        f"{predicted_class} with {confidence:.2f}% "
        f"confidence. {correctness} "
        f"Gradient-based attribution identified "
        f"{len(active_features)} active feature dimensions "
        f"among the top {len(top_features)} attributed "
        f"dimensions. {graph_evidence} "
        f"The neighborhood information is treated as "
        f"local graph evidence rather than causal proof."
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

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

    data, num_classes, config = load_dataset(
        args.dataset
    )

    node = args.id

    if node < 0 or node >= data.num_nodes:

        raise ValueError(
            f"Invalid node ID {node}. "
            f"Valid range: 0-{data.num_nodes - 1}"
        )

    model = load_model(
        data,
        num_classes,
        config["model"]
    )

    adjacency = build_adjacency(data)

    logits, probabilities, predictions = predict(
        model,
        data
    )

    predicted_id = int(
        predictions[node].item()
    )

    actual_id = int(
        data.y[node].item()
    )

    predicted_class = config["classes"][
        predicted_id
    ]

    actual_class = config["classes"][
        actual_id
    ]

    confidence = (
        probabilities[
            node,
            predicted_id
        ].item()
        * 100
    )

    correct = (
        predicted_id == actual_id
    )

    top_features = feature_attribution(
        model,
        data,
        node,
        predicted_id
    )

    neighbors, supporting, support_percentage = (
        analyze_neighbors(
            adjacency,
            node,
            predictions,
            probabilities,
            data.y,
            config["classes"]
        )
    )

    graph_structure = build_graph_structure(
        adjacency,
        node,
        predictions,
        probabilities,
        data.y,
        config["classes"],
        max_children=3
    )

    explanation = generate_explanation(
        config["name"],
        node,
        predicted_class,
        actual_class,
        confidence,
        correct,
        top_features,
        neighbors,
        supporting,
        support_percentage
    )

    result = {
        "dataset": config["name"],
        "node_id": node,

        "prediction": {
            "predicted_class": predicted_class,
            "actual_class": actual_class,
            "confidence": round(
                confidence,
                2
            ),
            "correct": correct
        },

        "feature_attribution": top_features,

        "neighborhood": {
            "num_direct_neighbors": len(
                neighbors
            ),
            "supporting_neighbors": supporting,
            "support_percentage": round(
                support_percentage,
                2
            ),
            "neighbors": neighbors
        },

        "graph_structure": graph_structure,

        "explanation": explanation
    }

    # -----------------------------------------------------
    # Console output
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("GRAPHNARRATOR-STYLE EXPLANATION")
    print("=" * 80)

    print()
    print(
        f"ROOT: Node {node}"
    )

    print(
        f"Prediction: {predicted_class}"
    )

    print(
        f"Actual: {actual_class}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print(
        f"Correct: {'YES' if correct else 'NO'}"
    )

    print()
    print("TOP FEATURE ATTRIBUTIONS")
    print("-" * 40)

    for feature in top_features[:10]:

        print(
            f"Feature {feature['feature_index']:4d} | "
            f"Importance {feature['importance']:.6f} | "
            f"Value {feature['feature_value']:.0f} | "
            f"Active {feature['active']}"
        )

    print()
    print("DIRECT NEIGHBORS")
    print("-" * 40)

    for record in neighbors:

        print(
            f"Node {record['node_id']} | "
            f"Prediction={record['predicted_class']} | "
            f"Confidence={record['confidence']:.2f}%"
        )

    print()
    print("NEIGHBOR SUPPORT")
    print("-" * 40)

    print(
        f"{supporting}/{len(neighbors)} direct neighbors "
        f"support the prediction "
        f"({support_percentage:.2f}%)"
    )

    print()
    print("GRAPH STRUCTURE")
    print("-" * 40)

    for branch in graph_structure:

        print(
            f"{branch['label']}: "
            f"Node {branch['node_id']} | "
            f"Prediction={branch['predicted_class']} | "
            f"Confidence={branch['confidence']:.2f}%"
        )

        for child_index, child in enumerate(
            branch["children"],
            start=1
        ):

            print(
                f"  "
                f"{branch['label']}.{child_index}: "
                f"Node {child['node_id']} | "
                f"Prediction={child['predicted_class']} | "
                f"Confidence={child['confidence']:.2f}%"
            )

        if branch["omitted_2hop_nodes"] > 0:

            print(
                f"  ... "
                f"{branch['omitted_2hop_nodes']} "
                f"additional 2-hop nodes omitted"
            )

    print()
    print("EXPLANATION")
    print("-" * 40)

    print(explanation)

    print()
    print("=" * 80)

    # -----------------------------------------------------
    # Save JSON
    # -----------------------------------------------------

    output_dir = Path("results")

    output_dir.mkdir(
        exist_ok=True
    )

    output_file = (
        output_dir
        / f"{args.dataset}_node_{node}_final_explanation.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print(
        f"Saved to: {output_file}"
    )


if __name__ == "__main__":
    main()
