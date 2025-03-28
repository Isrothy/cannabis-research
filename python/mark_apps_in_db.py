import os
import sys
import numpy as np
import configargparse
from pymongo import MongoClient
from model_utils import load_saved_model


def print_progress(current, total, bar_length=50):
    """
    Prints a progress bar to the terminal.
    """
    progress = current / total
    block = int(round(bar_length * progress))
    text = "\rProgress: [{0}] {1:.1f}%".format(
        "#" * block + "-" * (bar_length - block), progress * 100
    )
    sys.stdout.write(text)
    sys.stdout.flush()


def parse_args():
    parser = configargparse.ArgumentParser(
        description="Mark all apps in the MongoDB database using a saved model.",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument(
        "--config", is_config_file=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model (directory for NN or file for RF)",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        required=True,
        help="Name of the MongoDB collection to process",
    )
    return parser.parse_args()


def extract_feature_value(doc, feature):
    """
    Try to extract a feature value from a document.
    Looks in "classifierScores" first, then in "keywordCounts".
    Returns 0.0 if not found.
    """
    if "classifierScores" in doc and feature in doc["classifierScores"]:
        return doc["classifierScores"][feature]
    elif "keywordCounts" in doc and feature in doc["keywordCounts"]:
        return doc["keywordCounts"][feature]
    else:
        return 0.0


def main():
    args = parse_args()

    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError(
            "MongoDB URI must be provided via --mongo_uri or the MONGO_URI environment variable."
        )

    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    collection = db[args.collection_name]
    print(f"Connected to MongoDB collection: {args.collection_name}")

    model = load_saved_model(args.model_path)
    feature_cols = model["feature_cols"]
    scaler = model["scaler"]

    total_docs = collection.count_documents({})
    print(f"Processing {total_docs} documents...")

    cursor = collection.find({})
    update_count = 0
    for doc in cursor:
        features = np.array(
            [extract_feature_value(doc, feature) for feature in feature_cols]
        ).reshape(1, -1)
        features_scaled = scaler.transform(features)
        label = int(model["predict"](features_scaled)[0])
        probability = float(model["predict_proba"](features_scaled)[0])
        update = {
            "predictedLabel": label,
            "predictedProbability": probability,
        }
        collection.update_one({"_id": doc["_id"]}, {"$set": update})
        update_count += 1
        print_progress(update_count, total_docs)
    print(f"\nFinished updating {update_count} documents.")


if __name__ == "__main__":
    main()
