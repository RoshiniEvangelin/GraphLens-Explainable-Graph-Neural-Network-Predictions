
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import CORA
from models.GNNs.sage import SAGE


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------
DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_PATH = "sage_cora.pt"
OUTPUT_PATH = "graphsage_explanations.json"


# ---------------------------------------------------------
# Cora class names
# ---------------------------------------------------------
CLASS_NAMES = [
    "Case-Based",
    "Genetic Algorithms",
    "Neural Networks",
    "Probabilistic Methods",
    "Reinforcement Learning",
    "Rule Learning",
    "Theory",
]


# ---------------------------------------------------------
# Get graph neighbors
# ---------------------------------------------------------
def get_neighbors(edge_index, node_id):
    source = edge_index[0]
    target = edge_index[1]

    neighbors = set()

    for src, dst in zip(source.tolist(), target.tolist()):
        if src == node_id:
            neighbors.add(dst)

        if dst == node_id:
            neighbors.add(src)

    neighbors.discard(node_id)

    return sorted(neighbors)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    print("Using device:", DEVICE)

    # -----------------------------------------------------
    # Load Cora
    # -----------------------------------------------------
    data, data_citeid = CORA.get_cora_casestudy(0)

    data = data.to(DEVICE)

    print("Nodes:", data.num_nodes)
    print("Features:", data.x.shape)
    print("Classes:", len(CLASS_NAMES))

    # -----------------------------------------------------
    # Create the SAME GraphSAGE architecture
    # -----------------------------------------------------
    model = SAGE(
        in_channels=data.x.shape[1],
        hidden_channels=128,
        out_channels=7,
        num_layers=2,
        dropout=0.5,
    ).to(DEVICE)

    # -----------------------------------------------------
    # Load trained model
    # -----------------------------------------------------
    print("\nLoading model:", MODEL_PATH)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    # Handle either a raw state_dict or a checkpoint dictionary
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print("Model loaded successfully.")

    model.eval()

    # -----------------------------------------------------
    # Run inference
    # -----------------------------------------------------
    print("\nRunning GraphSAGE inference...")

    with torch.no_grad():

        logits = model(
            data.x,
            data.edge_index,
        )

        probabilities = F.softmax(
            logits,
            dim=1,
        )

    predictions = probabilities.argmax(dim=1)

    # -----------------------------------------------------
    # Generate explanations
    # -----------------------------------------------------
    print("Generating explanations...")

    results = []

    for node_id in range(data.num_nodes):

        predicted_class = int(
            predictions[node_id].item()
        )

        actual_class = int(
            data.y[node_id].item()
        )

        confidence = float(
            probabilities[
                node_id,
                predicted_class
            ].item()
        )

        # Get graph neighbors
        neighbors = get_neighbors(
            data.edge_index,
            node_id,
        )

        # Get predicted classes of neighbors
        neighbor_classes = []

        for neighbor_id in neighbors:

            neighbor_prediction = int(
                predictions[neighbor_id].item()
            )

            neighbor_classes.append(
                {
                    "node_id": int(neighbor_id),
                    "predicted_class": CLASS_NAMES[
                        neighbor_prediction
                    ],
                }
            )

        # -------------------------------------------------
        # Explanation
        # -------------------------------------------------
        explanation = (
            f"This paper was classified as "
            f"{CLASS_NAMES[predicted_class]} with "
            f"{confidence * 100:.2f}% confidence. "
            f"The GraphSAGE model used the paper's "
            f"1,433-dimensional feature representation "
            f"together with its graph connections to "
            f"{len(neighbors)} neighboring papers. "
            f"The prediction was "
            f"{'correct' if predicted_class == actual_class else 'incorrect'}."
        )

        result = {
            "node_id": int(node_id),

            "paper_id": str(
                data_citeid[node_id]
            ),

            "predicted_class": CLASS_NAMES[
                predicted_class
            ],

            "actual_class": CLASS_NAMES[
                actual_class
            ],

            "correct": (
                predicted_class == actual_class
            ),

            "confidence": round(
                confidence * 100,
                2,
            ),

            "num_neighbors": len(
                neighbors
            ),

            "neighbors": neighbor_classes,

            "explanation": explanation,
        }

        results.append(result)

    # -----------------------------------------------------
    # Overall accuracy
    # -----------------------------------------------------
    accuracy = (
        sum(
            item["correct"]
            for item in results
        )
        / len(results)
        * 100
    )

    # -----------------------------------------------------
    # Save output
    # -----------------------------------------------------
    output = {

        "model": "GraphSAGE",

        "dataset": "Cora",

        "num_nodes": int(
            data.num_nodes
        ),

        "num_features": int(
            data.x.shape[1]
        ),

        "num_classes": len(
            CLASS_NAMES
        ),

        "accuracy": round(
            accuracy,
            2,
        ),

        "explanations": results,
    }

    Path(OUTPUT_PATH).write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Finished
    # -----------------------------------------------------
    print("\n===================================")
    print("DONE")
    print("===================================")

    print(
        "Saved explanations to:",
        OUTPUT_PATH,
    )

    print(
        "Total nodes:",
        output["num_nodes"],
    )

    print(
        "Accuracy:",
        output["accuracy"],
        "%",
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
