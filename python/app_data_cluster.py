import os
import argparse
import yaml
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_data_from_db(collection_name, classifier_names):
    """
    Connect to MongoDB and fetch documents that contain the nested fields.
    For each classifier name (e.g. "facebook/bart-large-mnli"),
    we normalize it (replace "/" with "_") and then build the field name as:
      "classifier-score.{normalized_name}"

    Also projects additional fields like "title" and "appId".

    Returns:
      flattened_data: list of dicts with keys "title", "appId", and each normalized classifier.
      field_mapping: dict mapping the nested field to the normalized classifier name.
    """
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise Exception("Error: MONGO_URI environment variable is not set.")

    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    collection = db[collection_name]

    field_mapping = {}
    # Also project app title and id:
    projection = {"title": 1, "appId": 1}
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
        flat_doc["title"] = doc.get("title")
        flat_doc["appId"] = doc.get("appId")
        nested = doc.get("classifier-score", {})
        for nested_field, normalized in field_mapping.items():
            key = nested_field.split(".", 1)[1]
            if key in nested:
                flat_doc[normalized] = nested[key]
        flattened_data.append(flat_doc)
    return flattened_data, field_mapping


def evaluate_kmeans(X, k_values, n_runs):
    """
    For each k, run KMeans n_runs times, record the inertia (loss),
    and return a dict mapping k to mean loss.
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
    plt.xticks(ks)  # Ensure x-axis ticks are integer values.
    plt.savefig(output_file)
    print(f"Graph saved to: {output_file}")
    plt.show()


def cluster_data(df, feature_cols, k):
    """
    Cluster the data using KMeans with k clusters.
    Returns a DataFrame with columns: title, appId, original classifier scores,
    cluster label, and cluster size.
    """
    X = df[feature_cols].values
    kmeans = KMeans(n_clusters=k, random_state=42)
    df["cluster"] = kmeans.fit_predict(X)

    # Record cluster sizes.
    cluster_sizes = df.groupby("cluster").size().rename("cluster_size")
    df = df.merge(cluster_sizes, left_on="cluster", right_index=True)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Cluster MongoDB data and either test clustering parameters or perform clustering."
    )
    # General arguments
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--collection", type=str, help="MongoDB collection name")
    parser.add_argument(
        "--classifiers", type=str, nargs="+", help="List of classifier names"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "cluster"],
        default=None,
        help="Mode: 'test' for parameter testing, 'cluster' for clustering data",
    )
    # Arguments for test mode
    parser.add_argument(
        "--k_min", type=int, default=None, help="Minimum number of clusters (k)"
    )
    parser.add_argument(
        "--k_max", type=int, default=None, help="Maximum number of clusters (k)"
    )
    parser.add_argument(
        "--n_runs", type=int, default=None, help="Number of KMeans runs per k"
    )
    # Arguments for cluster mode
    parser.add_argument(
        "--k", type=int, default=None, help="Number of clusters (k) for cluster mode"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for graph (test mode) or CSV (cluster mode)",
    )
    args = parser.parse_args()

    # Load YAML config if provided.
    config = {}
    if args.config:
        config = load_config(args.config)

    # Command-line arguments override YAML config.
    collection_name = args.collection or config.get("collection")
    classifier_names = args.classifiers or config.get("classifiers")
    mode = args.mode or config.get("mode", "cluster")
    output_filename = args.output or config.get("output")

    if not collection_name or not classifier_names:
        raise Exception(
            "Error: Both collection name and classifier names must be provided."
        )

    print("Connecting to DB and fetching data...")
    data, field_mapping = get_data_from_db(collection_name, classifier_names)
    if not data:
        raise Exception("No data found matching the criteria.")
    df = pd.DataFrame(data)
    print(f"Fetched {len(df)} records.")

    # Use the normalized classifier names (from the field mapping) as our field list.
    original_field_list = list(field_mapping.values())

    # Instead of normalization, we use the original scores.
    print(
        "Using original classifier scores. Feature matrix shape:",
        df[original_field_list].shape,
    )

    if mode == "test":
        # For test mode, use k_min, k_max, and n_runs.
        k_min = args.k_min if args.k_min is not None else config.get("k_min", 2)
        k_max = args.k_max if args.k_max is not None else config.get("k_max", 10)
        n_runs = args.n_runs if args.n_runs is not None else config.get("n_runs", 5)
        k_values = list(range(k_min, k_max + 1))
        k_results = evaluate_kmeans(df[original_field_list].values, k_values, n_runs)
        if not output_filename:
            output_filename = "clustering_loss.png"
        plot_results(k_results, output_filename)
    elif mode == "cluster":
        # For cluster mode, use k to cluster the data and write to DB
        k_value = args.k if args.k is not None else config.get("k", 6)

        # Cluster the data
        clustered_df = cluster_data(df, original_field_list, k=k_value)

        # Connect to MongoDB to update records with cluster information
        mongo_uri = os.getenv("MONGO_URI")
        client = MongoClient(mongo_uri)
        db = client.get_default_database()
        collection = db[collection_name]

        # Update each record in the database with its cluster assignment
        update_count = 0
        for _, row in clustered_df.iterrows():
            app_id = row["appId"]
            cluster_label = int(row["cluster"])
            cluster_size = int(row["cluster_size"])

            # Update the document in MongoDB
            result = collection.update_one(
                {"appId": app_id},
                {
                    "$set": {
                        "cluster.label": cluster_label,
                        "cluster.size": cluster_size,
                        "cluster.k_value": k_value,
                    }
                },
            )

            if result.modified_count > 0:
                update_count += 1

        print(f"Updated {update_count} records with cluster information (k={k_value}).")

        # Save cluster information to CSV if output filename is provided
        if output_filename:
            output_columns = [
                "title",
                "appId",
                "cluster",
                "cluster_size",
            ] + original_field_list
            result = clustered_df[output_columns].reset_index(drop=True)
            result.to_csv(output_filename, index=False)
            print(f"Cluster information saved to: {output_filename}")


if __name__ == "__main__":
    main()
