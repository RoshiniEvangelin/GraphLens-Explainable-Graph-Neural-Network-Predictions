import json

INPUT_FILE = "graphsage_xai_explanations.json"
OUTPUT_FILE = "graphsage_local_explanations.json"


def generate_explanation(item):
    prediction = item["predicted_class"]
    actual = item["actual_class"]
    confidence = item["confidence"]
    neighbors = item["num_neighbors"]
    supporting = item["supporting_neighbors"]

    features = item.get("important_features", [])

    top_features = features[:5]

    feature_text = ", ".join(
        f'feature {f["feature_index"]} '
        f'(importance {f["importance"]:.3f})'
        for f in top_features
    )

    if not feature_text:
        feature_text = "no feature-attribution dimensions were recorded"

    if neighbors > 0:
        neighbor_text = (
            f"The paper has {neighbors} citation neighbor(s), "
            f"of which {supporting} support the predicted class."
        )
    else:
        neighbor_text = "No citation neighbors were recorded."

    correctness = (
        "The prediction matches the actual class."
        if item["correct"]
        else "The prediction does not match the actual class."
    )

    return (
        f"GraphSAGE classified paper {item['paper_id']} as "
        f"{prediction} with {confidence:.2f}% confidence. "
        f"{correctness} "
        f"The gradient-based XAI analysis identified {feature_text} "
        f"as influential input dimensions. "
        f"{neighbor_text} "
        f"These feature-attribution and citation-neighborhood signals "
        f"provide the structural evidence used by GraphSAGE for the "
        f"node classification."
    )


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for item in data["explanations"]:
        results.append({
            "node_id": item["node_id"],
            "paper_id": item["paper_id"],
            "predicted_class": item["predicted_class"],
            "actual_class": item["actual_class"],
            "confidence": item["confidence"],
            "correct": item["correct"],
            "explanation": generate_explanation(item)
        })

    output = {
        "model": "GraphSAGE",
        "dataset": "Cora",
        "method": "Gradient-based XAI + rule-based natural-language generation",
        "num_explanations": len(results),
        "explanations": results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("DONE")
    print(f"Generated {len(results)} explanations.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
