import json
import sys
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from models.GNNs.sage import SAGE


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device:", device)


# ============================================================
# Load CiteSeer
# ============================================================

dataset = Planetoid(
    root="dataset/citeseer",
    name="CiteSeer"
)

data = dataset[0].to(device)

print("Dataset: CiteSeer")
print("Nodes:", data.num_nodes)
print("Features:", data.num_features)
print("Classes:", dataset.num_classes)


# ============================================================
# Load trained GraphSAGE
# ============================================================

model = SAGE(
    in_channels=data.num_features,
    hidden_channels=128,
    out_channels=dataset.num_classes,
    num_layers=2,
    dropout=0.5
).to(device)

model.load_state_dict(
    torch.load(
    "sage_citeseer.pt",
    map_location=device,
    weights_only=True
    )
)
model.eval()

print("Loaded model: sage_citeseer.pt")


# ============================================================
# Class names
# ============================================================

class_names = [
    "Agents",
    "AI",
    "DB",
    "IR",
    "ML",
    "HCI"
]


# ============================================================
# Generate prediction
# ============================================================

with torch.no_grad():
    logits = model(data.x, data.edge_index)
    probabilities = F.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1)


# ============================================================
# Explain one node
# ============================================================

def explain_node(node_id):

    node_id = int(node_id)

    if node_id < 0 or node_id >= data.num_nodes:
        print(f"Invalid node ID: {node_id}")
        print(f"Valid node IDs: 0 - {data.num_nodes - 1}")
        return

    # --------------------------------------------------------
    # Prediction information
    # --------------------------------------------------------

    predicted_class = int(predictions[node_id].item())
    actual_class = int(data.y[node_id].item())

    confidence = (
        probabilities[node_id, predicted_class].item() * 100
    )

    correct = predicted_class == actual_class

    # --------------------------------------------------------
    # Gradient-based feature importance
    # --------------------------------------------------------

    x = data.x.detach().clone()
    x.requires_grad_(True)

    model.zero_grad()

    output = model(x, data.edge_index)

    target_score = output[node_id, predicted_class]

    target_score.backward()

    importance = x.grad[node_id].abs()

    top_k = min(20, data.num_features)

    values, indices = torch.topk(
        importance,
        k=top_k
    )

    # --------------------------------------------------------
    # Active feature information
    # --------------------------------------------------------

    important_features = []

    for rank, (feature_index, importance_value) in enumerate(
        zip(indices.tolist(), values.tolist()),
        start=1
    ):

        feature_value = x[node_id, feature_index].item()

        important_features.append(
            {
                "rank": rank,
                "feature_index": feature_index,
                "importance": round(
                    float(importance_value), 6
                ),
                "feature_value": round(
                    float(feature_value), 6
                ),
                "active": bool(feature_value != 0)
            }
        )

    # --------------------------------------------------------
    # Find graph neighbors
    # --------------------------------------------------------

    edge_index = data.edge_index

    neighbor_ids = set()

    for source, target in zip(
        edge_index[0].tolist(),
        edge_index[1].tolist()
    ):

        if source == node_id:
            neighbor_ids.add(target)

        if target == node_id:
            neighbor_ids.add(source)

    neighbor_ids.discard(node_id)

    # --------------------------------------------------------
    # Neighbor information
    # --------------------------------------------------------

    neighbors = []

    for neighbor_id in sorted(neighbor_ids):

        neighbor_prediction = int(
            predictions[neighbor_id].item()
        )

        neighbor_actual = int(
            data.y[neighbor_id].item()
        )

        neighbor_confidence = (
            probabilities[
                neighbor_id,
                neighbor_prediction
            ].item() * 100
        )

        neighbors.append(
            {
                "node_id": neighbor_id,
                "predicted_class": class_names[
                    neighbor_prediction
                ],
                "actual_class": class_names[
                    neighbor_actual
                ],
                "confidence": round(
                    neighbor_confidence,
                    2
                )
            }
        )

    # --------------------------------------------------------
    # Supporting neighbors
    # --------------------------------------------------------

    supporting_neighbors = sum(
        1
        for n in neighbors
        if n["predicted_class"] ==
        class_names[predicted_class]
    )

    # --------------------------------------------------------
    # Explanation text
    # --------------------------------------------------------

    feature_text = ", ".join(
        f'feature {f["feature_index"]} '
        f'(importance {f["importance"]:.3f})'
        for f in important_features[:5]
    )

    if not feature_text:
        feature_text = "no important feature dimensions were found"

    if neighbors:

        neighbor_text = (
            f"The node has {len(neighbors)} citation "
            f"neighbor(s), of which "
            f"{supporting_neighbors} have the same "
            f"predicted class."
        )

    else:

        neighbor_text = (
            "The node has no recorded citation neighbors."
        )

    correctness_text = (
        "The prediction matches the actual class."
        if correct
        else "The prediction does not match the actual class."
    )

    explanation = (
        f"GraphSAGE classified CiteSeer node {node_id} "
        f"as {class_names[predicted_class]} with "
        f"{confidence:.2f}% confidence. "
        f"{correctness_text} "
        f"Gradient-based XAI identified {feature_text} "
        f"as influential input dimensions. "
        f"{neighbor_text}"
    )

    # ========================================================
    # Display
    # ========================================================

    print()
    print("=" * 70)
    print("CITESEER GRAPHSA GE + XAI EXPLANATION")
    print("=" * 70)

    print(f"Node ID        : {node_id}")
    print(
        f"Predicted class: "
        f"{class_names[predicted_class]}"
    )
    print(
        f"Actual class   : "
        f"{class_names[actual_class]}"
    )
    print(f"Confidence     : {confidence:.2f}%")
    print(
        f"Correct        : "
        f"{'YES' if correct else 'NO'}"
    )

    print()
    print("-" * 70)
    print("TOP IMPORTANT FEATURES")
    print("-" * 70)

    for feature in important_features:
        print(
            f'{feature["rank"]:2d}. '
            f'Feature {feature["feature_index"]:4d} | '
            f'Importance: {feature["importance"]:.6f} | '
            f'Value: {feature["feature_value"]} | '
            f'Active: {feature["active"]}'
        )

    print()
    print("-" * 70)
    print("CITATION NEIGHBORS")
    print("-" * 70)

    if neighbors:

        for neighbor in neighbors:

            print(
                f'Node {neighbor["node_id"]}: '
                f'{neighbor["predicted_class"]} '
                f'({neighbor["confidence"]:.2f}% confidence)'
            )

    else:

        print("No neighbors found.")

    print()
    print(
        f"Supporting neighbors: "
        f"{supporting_neighbors}/{len(neighbors)}"
    )

    print()
    print("-" * 70)
    print("EXPLANATION")
    print("-" * 70)

    print(explanation)

    print()
    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print()
        print("Usage:")
        print("  python explain_citeseer.py <node_id>")
        print()
        print("Example:")
        print("  python explain_citeseer.py 2578")
        print()
        print(
            f"Valid node IDs: 0 - {data.num_nodes - 1}"
        )

        sys.exit(1)

    explain_node(sys.argv[1])
