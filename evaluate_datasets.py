import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from dataset import CORA
from models.GNNs.sage import SAGE


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def load_model(data, checkpoint_path, out_channels):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

        hidden_channels = checkpoint["hidden_channels"]
        num_layers = checkpoint["num_layers"]
        dropout = checkpoint["dropout"]

    else:

        state_dict = checkpoint
        hidden_channels = 128
        num_layers = 2
        dropout = 0.5

    model = SAGE(
        in_channels=data.x.shape[1],
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=dropout
    ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint


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


def evaluate_cora():

    print()
    print("=" * 75)
    print("CORA")
    print("=" * 75)

    # IMPORTANT:
    # Use the exact same loader used during training.
    data, citeid = CORA.get_cora_casestudy(0)

    data = data.to(DEVICE)

    print(
        f"Nodes:                  {data.num_nodes}"
    )

    print(
        f"Features:               {data.x.shape[1]}"
    )

    print(
        f"Classes:                {int(data.y.max()) + 1}"
    )

    print(
        f"Edges:                  {data.edge_index.shape[1]}"
    )

    model, checkpoint = load_model(
        data,
        "sage_cora.pt",
        7
    )

    adjacency = build_adjacency(data)

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

    test_nodes = (
        data.test_mask.nonzero(
            as_tuple=False
        ).view(-1)
    )

    correct = 0

    confidence_values = []
    support_values = []

    high_confidence_errors = []

    for node_tensor in test_nodes:

        node = int(
            node_tensor.item()
        )

        prediction = int(
            predictions[node].item()
        )

        actual = int(
            data.y[node].item()
        )

        confidence = probabilities[
            node,
            prediction
        ].item()

        confidence_values.append(
            confidence
        )

        if prediction == actual:

            correct += 1

        else:

            high_confidence_errors.append(
                (
                    confidence,
                    node,
                    prediction,
                    actual
                )
            )

        neighbors = adjacency[node]

        if neighbors:

            supporting = sum(
                1
                for neighbor in neighbors
                if int(
                    predictions[neighbor].item()
                ) == prediction
            )

            support_values.append(
                supporting / len(neighbors)
            )

    accuracy = (
        correct / len(test_nodes)
    )

    average_confidence = (
        sum(confidence_values)
        / len(confidence_values)
    )

    average_support = (
        sum(support_values)
        / len(support_values)
        if support_values
        else 0
    )

    high_confidence_errors.sort(
        reverse=True
    )

    print(
        f"Test nodes:             {len(test_nodes)}"
    )

    print(
        f"Checkpoint test acc:    "
        f"{checkpoint['test_accuracy'] * 100:.2f}%"
    )

    print(
        f"Recomputed test acc:    "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Correct predictions:    {correct}"
    )

    print(
        f"Incorrect predictions:  "
        f"{len(test_nodes) - correct}"
    )

    print(
        f"Average confidence:     "
        f"{average_confidence * 100:.2f}%"
    )

    print(
        f"Average neighbor support: "
        f"{average_support * 100:.2f}%"
    )

    print()
    print("HIGH-CONFIDENCE ERRORS")
    print("-" * 75)

    for (
        confidence,
        node,
        prediction,
        actual
    ) in high_confidence_errors[:5]:

        print(
            f"Node {node:4d} | "
            f"Confidence={confidence * 100:6.2f}% | "
            f"Predicted={prediction} | "
            f"Actual={actual}"
        )


def evaluate_citeseer():

    print()
    print("=" * 75)
    print("CITESEER")
    print("=" * 75)

    dataset = Planetoid(
        root="dataset/citeseer",
        name="CiteSeer"
    )

    data = dataset[0].to(DEVICE)

    print(
        f"Nodes:                  {data.num_nodes}"
    )

    print(
        f"Features:               {data.num_features}"
    )

    print(
        f"Classes:                {dataset.num_classes}"
    )

    print(
        f"Edges:                  {data.edge_index.shape[1]}"
    )

    model, checkpoint = load_model(
        data,
        "sage_citeseer.pt",
        dataset.num_classes
    )

    adjacency = build_adjacency(data)

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

    test_nodes = (
        data.test_mask.nonzero(
            as_tuple=False
        ).view(-1)
    )

    correct = 0

    confidence_values = []
    support_values = []

    high_confidence_errors = []

    for node_tensor in test_nodes:

        node = int(
            node_tensor.item()
        )

        prediction = int(
            predictions[node].item()
        )

        actual = int(
            data.y[node].item()
        )

        confidence = probabilities[
            node,
            prediction
        ].item()

        confidence_values.append(
            confidence
        )

        if prediction == actual:

            correct += 1

        else:

            high_confidence_errors.append(
                (
                    confidence,
                    node,
                    prediction,
                    actual
                )
            )

        neighbors = adjacency[node]

        if neighbors:

            supporting = sum(
                1
                for neighbor in neighbors
                if int(
                    predictions[neighbor].item()
                ) == prediction
            )

            support_values.append(
                supporting / len(neighbors)
            )

    accuracy = (
        correct / len(test_nodes)
    )

    average_confidence = (
        sum(confidence_values)
        / len(confidence_values)
    )

    average_support = (
        sum(support_values)
        / len(support_values)
        if support_values
        else 0
    )

    high_confidence_errors.sort(
        reverse=True
    )

    print(
        f"Test nodes:             {len(test_nodes)}"
    )

    print(
        f"Recomputed test acc:    "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Correct predictions:    {correct}"
    )

    print(
        f"Incorrect predictions:  "
        f"{len(test_nodes) - correct}"
    )

    print(
        f"Average confidence:     "
        f"{average_confidence * 100:.2f}%"
    )

    print(
        f"Average neighbor support: "
        f"{average_support * 100:.2f}%"
    )

    print()
    print("HIGH-CONFIDENCE ERRORS")
    print("-" * 75)

    for (
        confidence,
        node,
        prediction,
        actual
    ) in high_confidence_errors[:5]:

        print(
            f"Node {node:4d} | "
            f"Confidence={confidence * 100:6.2f}% | "
            f"Predicted={prediction} | "
            f"Actual={actual}"
        )


def main():

    print(
        f"Device: {DEVICE}"
    )

    evaluate_cora()
    evaluate_citeseer()


if __name__ == "__main__":
    main()
