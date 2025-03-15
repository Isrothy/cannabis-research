import os
import argparse
import yaml
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_data_from_db(collection_name, classifier_names):
    """
    Connect to MongoDB and fetch documents that contain the nested fields.
    For each classifier name provided (e.g. "facebook/bart-large-mnli"),
    we normalize it (replace "/" with "_") and then build the field name as:
      "classifier-score.{normalized_name}"

    Returns:
      data: a list of documents (each document is a dict)
      field_mapping: a dict mapping the nested field name to the normalized classifier name.
                     e.g., { "classifier-score.facebook_bart-large-mnli": "facebook_bart-large-mnli", ... }
    """
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise Exception("Error: MONGO_URI environment variable is not set.")

    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    collection = db[collection_name]

    field_mapping = {}
    projection = {}
    for classifier in classifier_names:
        normalized = classifier.replace("/", "_")
        nested_field = f"classifier-score.{normalized}"
        field_mapping[nested_field] = normalized
        projection[nested_field] = 1
    projection["_id"] = 0

    query = {"$and": [{field: {"$exists": True}} for field in field_mapping.keys()]}

    cursor = collection.find(query, projection)
    data = list(cursor)

    flattened_data = []
    for doc in data:
        flat_doc = {}
        nested = doc.get("classifier-score", {})
        for nested_field, normalized in field_mapping.items():
            key = nested_field.split(".", 1)[1]
            if key in nested:
                flat_doc[normalized] = nested[key]
        flattened_data.append(flat_doc)
    return flattened_data, field_mapping


def normalize_fields(df, field_names):
    """
    Normalize specified fields using z-score normalization.
    The normalized values are stored in new columns with suffix '_zscore'.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[field_names])
    normalized_fields = []
    for i, field in enumerate(field_names):
        new_field = field + "_zscore"
        df[new_field] = X[:, i]
        normalized_fields.append(new_field)
    return df, normalized_fields


def evaluate_kmeans(X, k_values, n_runs):
    """
    For each k in k_values, run KMeans clustering n_runs times,
    record the inertia (loss), and return a dictionary mapping k to the mean loss.
    """
    k_results = {}
    for k in k_values:
        losses = []
        for run in range(n_runs):
            kmeans = KMeans(n_clusters=k, random_state=run)
            kmeans.fit(X)
            losses.append(kmeans.inertia_)
        mean_loss = np.mean(losses)
        k_results[k] = mean_loss
        print(f"k = {k}, mean loss = {mean_loss}")
    return k_results


def plot_results(k_results, output_file):
    """Plot the k vs. mean loss graph and save the image."""
    ks = sorted(k_results.keys())
    losses = [k_results[k] for k in ks]

    plt.figure(figsize=(10, 6))
    plt.plot(ks, losses, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Mean Loss (Inertia)")
    plt.title("KMeans Clustering Loss vs. k")
    plt.xticks(ks)
    plt.savefig(output_file)
    print(f"Graph saved to: {output_file}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Cluster normalized MongoDB data using KMeans and plot loss vs. k."
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--collection", type=str, help="MongoDB collection name")
    parser.add_argument(
        "--classifiers", type=str, nargs="+", help="List of classifier names"
    )
    parser.add_argument(
        "--k_min", type=int, default=None, help="Minimum number of clusters (k)"
    )
    parser.add_argument(
        "--k_max", type=int, default=None, help="Maximum number of clusters (k)"
    )
    parser.add_argument(
        "--n_runs", type=int, default=None, help="Number of KMeans runs per k"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for the graph",
    )
    args = parser.parse_args()

    config = {}
    if args.config:
        config = load_config(args.config)

    collection_name = args.collection or config.get("collection")
    classifier_names = args.classifiers or config.get("classifiers")
    k_min = args.k_min if args.k_min is not None else config.get("k_min", 2)
    k_max = args.k_max if args.k_max is not None else config.get("k_max", 10)
    n_runs = args.n_runs if args.n_runs is not None else config.get("n_runs", 5)
    output_file = (
        args.output
        if args.output is not None
        else config.get("output", "clustering_loss.png")
    )

    if not collection_name or not classifier_names:
        raise Exception(
            "Error: Both collection name and classifier names must be provided."
        )

    print("Connecting to DB and fetching data...")
    data, field_mapping = get_data_from_db(collection_name, classifier_names)
    if not data:
        raise Exception("No data found matching the criteria.")
    df = pd.DataFrame(data)

    df.rename(columns=field_mapping, inplace=True)
    normalized_field_list = list(field_mapping.values())

    df, zscore_fields = normalize_fields(df, normalized_field_list)
    X = df[zscore_fields].values
    print("Data normalized using z-score. Feature matrix shape:", X.shape)

    k_values = list(range(k_min, k_max + 1))
    k_results = evaluate_kmeans(X, k_values, n_runs)

    plot_results(k_results, output_file)


if __name__ == "__main__":
    main()
