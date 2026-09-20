
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import CORA
from models.GNNs.sage import SAGE


# =========================================================
# DEVICE
# =========================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device:", device)


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "sage_cora.pt"
OUTPUT_PATH = "graphsage_xai_explanations.json"

HIDDEN_CHANNELS = 128
NUM_LAYERS = 2
DROPOUT = 0.5

TOP_K_FEATURES = 20


# =========================================================
# CORA CLASS NAMES
# =========================================================

class_names = [
    "Case-Based",
    "Genetic Algorithms",
    "Neural Networks",
    "Probabilistic Methods",
    "Reinforcement Learning",
    "Rule Learning",
    "Theory",
]


# =========================================================
# LOAD CORA
# =========================================================

data, citeid = CORA.get_cora_casestudy(0)

data = data.to(device)

print("Nodes:", data.num_nodes)
print("Features:", data.x.shape)
print("Classes:", int(data.y.max()) + 1)


# =========================================================
# RECREATE GRAPHSAGE MODEL
# =========================================================

model = SAGE(
    in_channels=data.x.shape[1],
    hidden_channels=HIDDEN_CHANNELS,
    out_channels=len(class_names),
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(device)


# =========================================================
# LOAD TRAINED CHECKPOINT
# =========================================================

print("\nLoading trained model:", MODEL_PATH)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Loaded trained GraphSAGE model")


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def calculate_accuracy(mask):
    """
    Calculate accuracy for a specific dataset split.
    """

    with torch.no_grad():

        logits = model(
            data.x,
            data.edge_index,
        )

        predictions = logits.argmax(
            dim=1
        )

        correct = (
            predictions[mask]
            == data.y[mask]
        ).sum().item()

        total = mask.sum().item()

    if total == 0:
        return 0.0

    return correct / total


# =========================================================
# CALCULATE TRAIN / VALIDATION / TEST ACCURACY
# =========================================================

print("\nCalculating dataset accuracies...")

train_accuracy = calculate_accuracy(
    data.train_mask
)

validation_accuracy = calculate_accuracy(
    data.val_mask
)

test_accuracy = calculate_accuracy(
    data.test_mask
)

print(
    f"Train Accuracy: "
    f"{train_accuracy * 100:.2f}%"
)

print(
    f"Validation Accuracy: "
    f"{validation_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# =========================================================
# GET CITATION NEIGHBORS
# =========================================================

def get_neighbors(node_id):
    """
    Return unique citation neighbors for a node.
    """

    edge_index = (
        data.edge_index
        .detach()
        .cpu()
    )

    neighbors = []

    for source, target in edge_index.t().tolist():

        if source == node_id:
            neighbors.append(target)

        elif target == node_id:
            neighbors.append(source)

    # Remove duplicates while preserving order
    neighbors = list(
        dict.fromkeys(neighbors)
    )

    # Do not include the node itself
    neighbors = [
        n
        for n in neighbors
        if n != node_id
    ]

    return neighbors


# =========================================================
# EXPLAIN ONE NODE
# =========================================================

def explain_node(node_id):

    node_id = int(node_id)

    paper_id = str(
        citeid[node_id]
    )

    true_class = int(
        data.y[node_id].item()
    )


    # -----------------------------------------------------
    # Create independent feature tensor
    # -----------------------------------------------------

    x = (
        data.x
        .clone()
        .detach()
        .requires_grad_(True)
    )


    # -----------------------------------------------------
    # Forward pass
    # -----------------------------------------------------

    model.zero_grad()

    output = model(
        x,
        data.edge_index,
    )


    probabilities = F.softmax(
        output[node_id],
        dim=0,
    )


    predicted_class = int(
        output[node_id]
        .argmax()
        .item()
    )


    prediction_confidence = float(
        probabilities[
            predicted_class
        ].item()
    )


    # -----------------------------------------------------
    # Gradient-based feature importance
    # -----------------------------------------------------

    prediction_score = output[
        node_id,
        predicted_class,
    ]

    model.zero_grad()

    if x.grad is not None:
        x.grad.zero_()

    prediction_score.backward()


    # Absolute gradient for this node's features
    importance = (
        x.grad[node_id]
        .abs()
    )


    # Get top K features
    k = min(
        TOP_K_FEATURES,
        importance.shape[0],
    )

    top_values, top_indices = (
        torch.topk(
            importance,
            k=k,
        )
    )


    # -----------------------------------------------------
    # Store important features
    # -----------------------------------------------------

    important_features = []

    for rank, (
        feature_id,
        score,
    ) in enumerate(
        zip(
            top_indices,
            top_values,
        ),
        start=1,
    ):

        feature_index = int(
            feature_id.item()
        )

        importance_score = float(
            score.item()
        )

        # Check whether this feature
        # is actually active for this paper.
        feature_value = float(
            data.x[
                node_id,
                feature_index,
            ].item()
        )

        important_features.append(
            {
                "rank": rank,
                "feature_index": feature_index,
                "importance": round(
                    importance_score,
                    6,
                ),
                "feature_value": feature_value,
                "active": feature_value != 0,
            }
        )


    # -----------------------------------------------------
    # Citation neighbors
    # -----------------------------------------------------

    neighbors = get_neighbors(
        node_id
    )

    citation_neighbors = []

    for neighbor_id in neighbors:

        neighbor_prediction = int(
            output[
                neighbor_id
            ].argmax().item()
        )

        neighbor_actual = int(
            data.y[
                neighbor_id
            ].item()
        )

        neighbor_confidence = float(
            F.softmax(
                output[neighbor_id],
                dim=0,
            )[neighbor_prediction]
            .item()
        )

        citation_neighbors.append(
            {
                "node_id": int(
                    neighbor_id
                ),

                "paper_id": str(
                    citeid[neighbor_id]
                ),

                "predicted_class": (
                    class_names[
                        neighbor_prediction
                    ]
                ),

                "actual_class": (
                    class_names[
                        neighbor_actual
                    ]
                ),

                "confidence": round(
                    neighbor_confidence * 100,
                    2,
                ),
            }
        )


    # -----------------------------------------------------
    # Count neighbor class support
    # -----------------------------------------------------

    neighbor_class_counts = {}

    for neighbor in citation_neighbors:

        neighbor_class = (
            neighbor[
                "predicted_class"
            ]
        )

        neighbor_class_counts[
            neighbor_class
        ] = (
            neighbor_class_counts.get(
                neighbor_class,
                0,
            )
            + 1
        )


    # -----------------------------------------------------
    # Number of neighbors supporting
    # predicted class
    # -----------------------------------------------------

    predicted_class_name = (
        class_names[
            predicted_class
        ]
    )

    supporting_neighbors = sum(
        1
        for neighbor in citation_neighbors
        if neighbor[
            "predicted_class"
        ]
        == predicted_class_name
    )


    # -----------------------------------------------------
    # Active important features
    # -----------------------------------------------------

    active_important_features = [
        feature
        for feature in important_features
        if feature["active"]
    ]


    # -----------------------------------------------------
    # Generate human-readable explanation
    # -----------------------------------------------------

    explanation = (
        f"GraphSAGE classified paper "
        f"{paper_id} as "
        f"{predicted_class_name} "
        f"with "
        f"{prediction_confidence * 100:.2f}% "
        f"confidence. "
    )


    if len(active_important_features) > 0:

        active_feature_ids = [
            str(
                feature[
                    "feature_index"
                ]
            )
            for feature
            in active_important_features[:5]
        ]

        explanation += (
            f"The prediction was influenced "
            f"by active input feature dimensions "
            f"{', '.join(active_feature_ids)} "
            f"among the top gradient-attributed "
            f"features. "
        )

    else:

        explanation += (
            "The top gradient-attributed "
            "features were not active in the "
            "paper's input representation. "
        )


    if len(citation_neighbors) > 0:

        explanation += (
            f"The paper has "
            f"{len(citation_neighbors)} "
            f"citation neighbors, of which "
            f"{supporting_neighbors} "
            f"were also predicted as "
            f"{predicted_class_name}. "
        )

    else:

        explanation += (
            "The paper has no citation "
            "neighbors in the graph. "
        )


    if (
        predicted_class
        == true_class
    ):

        explanation += (
            "The prediction matches the "
            "ground-truth class."
        )

    else:

        explanation += (
            "The prediction does not match "
            "the ground-truth class."
        )


    # -----------------------------------------------------
    # Return complete explanation
    # -----------------------------------------------------

    return {

        "node_id": node_id,

        "paper_id": paper_id,

        "predicted_class": (
            predicted_class_name
        ),

        "actual_class": (
            class_names[
                true_class
            ]
        ),

        "correct": (
            predicted_class
            == true_class
        ),

        "confidence": round(
            prediction_confidence * 100,
            2,
        ),

        "num_neighbors": len(
            citation_neighbors
        ),

        "supporting_neighbors": (
            supporting_neighbors
        ),

        "neighbor_class_counts": (
            neighbor_class_counts
        ),

        "important_features": (
            important_features
        ),

        "citation_neighbors": (
            citation_neighbors
        ),

        "explanation": explanation,
    }


# =========================================================
# EXPLAIN ALL TEST NODES
# =========================================================

test_nodes = torch.where(
    data.test_mask
)[0]

print(
    "\nNumber of test nodes:",
    len(test_nodes),
)

print(
    "Generating XAI explanations "
    "for all test nodes..."
)


explanations = []

for index, node_id in enumerate(
    test_nodes.tolist(),
    start=1,
):

    result = explain_node(
        node_id
    )

    explanations.append(
        result
    )

    if index % 100 == 0:

        print(
            f"Processed "
            f"{index} / "
            f"{len(test_nodes)} "
            f"test nodes"
        )


# =========================================================
# CREATE FINAL OUTPUT
# =========================================================

output = {

    "model": "GraphSAGE",

    "dataset": "Cora",

    "device": str(
        device
    ),

    "model_checkpoint": (
        MODEL_PATH
    ),

    "num_nodes": int(
        data.num_nodes
    ),

    "num_features": int(
        data.x.shape[1]
    ),

    "num_classes": len(
        class_names
    ),

    "train_accuracy": round(
        train_accuracy * 100,
        2,
    ),

    "validation_accuracy": round(
        validation_accuracy * 100,
        2,
    ),

    "test_accuracy": round(
        test_accuracy * 100,
        2,
    ),

    "num_test_nodes": len(
        test_nodes
    ),

    "top_k_features": (
        TOP_K_FEATURES
    ),

    "explanations": explanations,
}


# =========================================================
# SAVE JSON
# =========================================================

Path(
    OUTPUT_PATH
).write_text(
    json.dumps(
        output,
        indent=2,
    ),
    encoding="utf-8",
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("=" * 70)
print("GRAPHNARRATOR XAI COMPLETE")
print("=" * 70)

print(
    f"Train Accuracy: "
    f"{output['train_accuracy']:.2f}%"
)

print(
    f"Validation Accuracy: "
    f"{output['validation_accuracy']:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{output['test_accuracy']:.2f}%"
)

print(
    "Test nodes explained:",
    output["num_test_nodes"],
)

print(
    "Output file:",
    OUTPUT_PATH,
)

print("=" * 70)
