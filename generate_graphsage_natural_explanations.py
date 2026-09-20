import json
import os
from pathlib import Path

from openai import OpenAI


INPUT_FILE = "graphsage_xai_explanations.json"
OUTPUT_FILE = "graphsage_natural_explanations.json"

# Change this if you want a different OpenAI model.
MODEL = "gpt-5.6-mini"


CLASS_NAMES = [
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
]


def build_prompt(item):
    label = item["predicted_class"]
    actual = item["actual_class"]

    important_features = item.get("important_features", [])
    neighbors = item.get("citation_neighbors", [])

    feature_lines = []
    for f in important_features[:20]:
        feature_lines.append(
            f'- Feature {f["feature_index"]}: '
            f'importance={f["importance"]:.4f}, '
            f'value={f["feature_value"]}, '
            f'active={f["active"]}'
        )

    if not feature_lines:
        feature_text = "No feature-attribution information was recorded."
    else:
        feature_text = "\n".join(feature_lines)

    neighbor_lines = []

    for n in neighbors:
        neighbor_lines.append(
            f'- Neighbor node {n.get("node_id")}, '
            f'paper ID {n.get("paper_id")}: '
            f'predicted={n.get("predicted_class")}, '
            f'actual={n.get("actual_class")}, '
            f'confidence={n.get("confidence", 0):.2f}%'
        )

    if not neighbor_lines:
        neighbor_text = "No citation neighbors were recorded."
    else:
        neighbor_text = "\n".join(neighbor_lines)

    prompt = f"""
You are generating a GraphNarrator-style explanation for a GraphSAGE
graph neural network classification on the Cora citation network.

The model classified the target paper into one of these seven categories:

{CLASS_NAMES}

TARGET PAPER
------------
Paper ID: {item["paper_id"]}
Node ID: {item["node_id"]}
Predicted class: {label}
Actual class: {actual}
Prediction confidence: {item["confidence"]:.2f}%
Prediction correct: {item["correct"]}

GRAPH INFORMATION
-----------------
Number of citation neighbors: {item["num_neighbors"]}
Supporting neighbors: {item["supporting_neighbors"]}

Neighbor predictions:
{neighbor_text}

FEATURE-LEVEL XAI
-----------------
The GraphSAGE explanation used gradient-based feature importance.
The Cora representation contains 1,433 numeric feature dimensions.

Important input dimensions:
{feature_text}

IMPORTANT INTERPRETATION RULE
-----------------------------
The numeric feature indices are NOT word names or human-readable
keywords. Do not invent words or topics for a feature index.

Instead, explain them as influential input dimensions/features.

Use the graph structure and the neighboring papers' predicted classes
to provide contextual evidence.

Generate TWO sections.

REASONING
---------
Explain:
1. The target prediction and confidence.
2. The most influential feature dimensions.
3. Whether those influential dimensions are active in the target.
4. The citation-neighborhood evidence.
5. How the neighboring predictions support or differ from the target.
6. The distinction between target-level evidence and graph-level evidence.

FREE-TEXT EXPLANATION
---------------------
Write a concise human-readable explanation suitable for a research
paper. Explain why GraphSAGE classified the target node as the predicted
class using the feature-level XAI evidence and graph neighborhood.

Do not claim that the numeric feature indices correspond to specific
words.
Do not invent paper titles, keywords, abstracts, or semantic meanings
that are not present in the supplied evidence.

Return exactly this structure:

### Reasoning

[reasoning]

### Free-Text Explanation

[concise explanation]
"""

    return prompt


def main():
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. "
            "Run tag_graphsage_cora.py first."
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it with: export OPENAI_API_KEY='your-key'"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    explanations = data.get("explanations", [])

    print(f"Loaded {len(explanations)} GraphSAGE explanations.")
    print(f"Using model: {MODEL}")

    client = OpenAI()

    results = []

    for i, item in enumerate(explanations, start=1):
        print(
            f"[{i}/{len(explanations)}] "
            f"Paper {item['paper_id']} -> "
            f"{item['predicted_class']}"
        )

        prompt = build_prompt(item)

        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            max_tokens=1200,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful scientific explanation generator. "
                        "Only make claims supported by the supplied GraphSAGE "
                        "XAI evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        explanation = response.choices[0].message.content

        results.append(
            {
                "node_id": item["node_id"],
                "paper_id": item["paper_id"],
                "predicted_class": item["predicted_class"],
                "actual_class": item["actual_class"],
                "confidence": item["confidence"],
                "correct": item["correct"],
                "supporting_neighbors": item["supporting_neighbors"],
                "num_neighbors": item["num_neighbors"],
                "generated_explanation": explanation,
            }
        )

    output = {
        "model": "GraphSAGE + gradient XAI + natural-language explanation",
        "dataset": data.get("dataset", "Cora"),
        "source_file": INPUT_FILE,
        "llm_model": MODEL,
        "num_explanations": len(results),
        "explanations": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print("DONE")
    print(f"Saved explanations to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
