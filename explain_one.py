import json
import sys

INPUT_FILE = "graphsage_local_explanations.json"


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python explain_one.py <paper_id>")
        print()
        print("Example:")
        print("  python explain_one.py 28491")
        return

    paper_id = sys.argv[1]

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data["explanations"]:
        if str(item["paper_id"]) == paper_id:
            print()
            print("=" * 70)
            print("GraphSAGE PAPER EXPLANATION")
            print("=" * 70)

            print(f"Paper ID       : {item['paper_id']}")
            print(f"Node ID        : {item['node_id']}")
            print(f"Predicted class: {item['predicted_class']}")
            print(f"Actual class   : {item['actual_class']}")
            print(f"Confidence     : {item['confidence']:.2f}%")
            print(
                f"Correct        : "
                f"{'YES' if item['correct'] else 'NO'}"
            )

            print()
            print("-" * 70)
            print("NATURAL-LANGUAGE EXPLANATION")
            print("-" * 70)

            print(item["explanation"])

            print()
            print("=" * 70)

            return

    print()
    print(f"Paper ID {paper_id} was not found.")
    print()
    print("Make sure you are using a paper ID from")
    print("graphsage_local_explanations.json")


if __name__ == "__main__":
    main()
