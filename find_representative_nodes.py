import argparse
import torch
from torch_geometric.datasets import Planetoid

from models.GNNs.sage import SAGE


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name == "cora":
        from dataset import CORA

        data, citeid = CORA.get_cora_casestudy(0)

        class_names = [
            "Case_Based",
            "Genetic_Algorithms",
            "Neural_Networks",
            "Probabilistic_Methods",
            "Reinforcement_Learning",
            "Rule_Learning",
            "Theory",
        ]

        checkpoint_path = "sage_cora.pt"

    elif dataset_name == "citeseer":
        dataset = Planetoid(
            root="dataset/citeseer",
            name="CiteSeer"
        )

        data = dataset[0]

        class_names = [
            "Agents",
            "AI",
            "DB",
            "IR",
            "ML",
            "HCI",
        ]

        checkpoint_path = "sage_citeseer.pt"

    else:
        raise ValueError(
            "Unsupported dataset. Use 'cora' or 'citeseer'."
        )

    return data, class_names, checkpoint_path


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(data, checkpoint_path, dataset_name, device):

    if dataset_name == "cora":

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

        model = SAGE(
            in_channels=checkpoint["in_channels"],
            hidden_channels=checkpoint["hidden_channels"],
            out_channels=checkpoint["out_channels"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model = SAGE(
            in_channels=data.x.shape[1],
            hidden_channels=128,
            out_channels=len(torch.unique(data.y)),
            num_layers=2,
            dropout=0.5,
        )

        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# NEIGHBOR SUPPORT
# ============================================================

def get_neighbors(data, node_id):

    edge_index = data.edge_index

    src = edge_index[0]
    dst = edge_index[1]

    neighbors = set()

    for i in range(edge_index.shape[1]):
        s = int(src[i])
        d = int(dst[i])

        if s == node_id:
            neighbors.add(d)

        if d == node_id:
            neighbors.add(s)

    neighbors.discard(node_id)

    return sorted(neighbors)


# ============================================================
# REPRESENTATIVE NODE SEARCH
# ============================================================

def analyze_nodes(data, model, class_names, device):

    data = data.to(device)

    with torch.no_grad():

        logits = model(
            data.x,
            data.edge_index
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = probabilities.argmax(
            dim=1
        )

        confidence = probabilities.max(
            dim=1
        ).values

    # --------------------------------------------------------
    # Select test nodes
    # --------------------------------------------------------

    if hasattr(data, "test_mask"):

        test_nodes = torch.where(
            data.test_mask
        )[0].tolist()

    else:

        raise RuntimeError(
            "Dataset does not contain test_mask."
        )

    print(f"Device: {device}")
    print(
        f"Dataset: "
        f"{'Cora' if len(class_names) == 7 else 'CiteSeer'}"
    )
    print(f"Test nodes: {len(test_nodes)}")

    # --------------------------------------------------------
    # Build records
    # --------------------------------------------------------

    records = []

    for node_id in test_nodes:

        node_id = int(node_id)

        pred = int(predictions[node_id])
        actual = int(data.y[node_id])

        conf = float(
            confidence[node_id].item()
        ) * 100

        neighbors = get_neighbors(
            data,
            node_id
        )

        if len(neighbors) > 0:

            neighbor_predictions = [
                int(predictions[n].item())
                for n in neighbors
            ]

            support_count = sum(
                p == pred
                for p in neighbor_predictions
            )

            support = (
                support_count /
                len(neighbors)
            ) * 100

        else:

            support = 0.0

        records.append({
            "node_id": node_id,
            "pred": pred,
            "actual": actual,
            "confidence": conf,
            "neighbors": len(neighbors),
            "support": support,
            "correct": pred == actual,
        })

    # ========================================================
    # CATEGORY 1: HIGH-CONFIDENCE CORRECT
    # ========================================================

    high_correct = [
        r for r in records
        if r["correct"]
    ]

    high_correct.sort(
        key=lambda r: r["confidence"],
        reverse=True
    )

    # ========================================================
    # CATEGORY 2: LOW-CONFIDENCE CORRECT
    # ========================================================

    low_correct = [
        r for r in records
        if r["correct"]
    ]

    low_correct.sort(
        key=lambda r: r["confidence"]
    )

    # ========================================================
    # CATEGORY 3: HIGH-CONFIDENCE INCORRECT
    # ========================================================

    high_incorrect = [
        r for r in records
        if not r["correct"]
    ]

    high_incorrect.sort(
        key=lambda r: r["confidence"],
        reverse=True
    )

    # ========================================================
    # CATEGORY 4: MIXED NEIGHBORHOOD
    # ========================================================

    mixed = [
        r for r in records
        if 0 < r["support"] < 100
        and r["neighbors"] >= 2
    ]

    mixed.sort(
        key=lambda r: (
            abs(r["support"] - 50),
            -r["confidence"]
        )
    )

    # ========================================================
    # CATEGORY 5: STRONG NEIGHBORHOOD AGREEMENT
    # ========================================================

    strong_support = [
        r for r in records
        if r["support"] == 100
        and r["neighbors"] > 0
    ]

    strong_support.sort(
        key=lambda r: r["confidence"],
        reverse=True
    )

    # ========================================================
    # PRINT FUNCTION
    # ========================================================

    def print_category(title, node_records):

        print("\n" + "=" * 90)
        print(title)
        print("=" * 90)

        for r in node_records[:5]:

            pred_name = class_names[r["pred"]]
            actual_name = class_names[r["actual"]]

            print(
                f"Node {r['node_id']:4d} | "
                f"Pred={pred_name:<25} "
                f"| Actual={actual_name:<25} "
                f"| Conf={r['confidence']:6.2f}% "
                f"| Neighbors={r['neighbors']:3d} "
                f"| Support={r['support']:6.2f}%"
            )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print_category(
        "1. HIGH-CONFIDENCE CORRECT",
        high_correct
    )

    print_category(
        "2. LOW-CONFIDENCE CORRECT",
        low_correct
    )

    print_category(
        "3. HIGH-CONFIDENCE INCORRECT",
        high_incorrect
    )

    print_category(
        "4. MIXED NEIGHBORHOOD",
        mixed
    )

    print_category(
        "5. STRONG NEIGHBORHOOD AGREEMENT",
        strong_support
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Find representative nodes for "
            "GraphSAGE explainability analysis."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cora", "citeseer"],
        help="Dataset to analyze."
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    data, class_names, checkpoint_path = load_dataset(
        args.dataset
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        data,
        checkpoint_path,
        args.dataset,
        device
    )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    analyze_nodes(
        data,
        model,
        class_names,
        device
    )


if __name__ == "__main__":
    main()