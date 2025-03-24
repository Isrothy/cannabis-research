import os
import argparse
import matplotlib.pyplot as plt
from pymongo import MongoClient


def get_nested_value(doc, dotted_field):
    """Retrieve the value from a document using a dotted field name."""
    keys = dotted_field.split(".")
    value = doc
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value


def draw_histogram(collection_name, model_name):
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("Environment variable MONGO_URI is not set.")

    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    collection = db[collection_name]

    normalized_model_name = model_name.replace("/", "_")
    field_name = f"classifierScores.{normalized_model_name }"

    query = {field_name: {"$exists": True}}
    cursor = collection.find(query, {field_name: 1, "_id": 0})

    scores = []
    for doc in cursor:
        score = get_nested_value(doc, field_name)
        if score is not None:
            scores.append(score)

    if not scores:
        print("No scores found for the given field.")
        exit(0)

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=30, edgecolor="black")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {field_name} from {collection_name}")

    output_dir = "img"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"histogram-{collection_name}-{field_name}.png"
    )
    plt.savefig(output_path)
    print(f"Histogram saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a histogram of classifier scores from a MongoDB collection."
    )
    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Name of the collection containing app records.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (used in the field classifierScores.{model_name}).",
    )
    args = parser.parse_args()
    draw_histogram(args.collection, args.model)
