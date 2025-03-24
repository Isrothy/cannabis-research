import os
import configargparse
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def parse_args():
    parser = configargparse.ArgumentParser(
        description="Cluster MongoDB data and either test clustering parameters or perform clustering.",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    # General arguments
    parser.add_argument(
        "--config", is_config_file=True, help="Path to YAML configuration file"
    )
    parser.add_argument("--collection", type=str, help="MongoDB collection name")
    parser.add_argument(
        "--classifiers", type=str, nargs="+", help="List of classifier names"
    )
    parser.add_argument("--keywords", type=str, nargs="+", help="List of keywords")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "cluster"],
        default=None,
        help="Mode: 'test' for parameter testing, 'cluster' for clustering data",
    )
    # Arguments for test mode
    parser.add_argument(
        "--k_min", type=int, default=2, help="Minimum number of clusters (k)"
    )
    parser.add_argument(
        "--k_max", type=int, default=10, help="Maximum number of clusters (k)"
    )
    parser.add_argument(
        "--n_runs", type=int, default=10, help="Number of KMeans runs per k"
    )
    # Arguments for cluster mode
    parser.add_argument(
        "--k", type=int, default=10, help="Number of clusters (k) for cluster mode"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.csv",
        help="Output file for graph (test mode) or CSV (cluster mode)",
    )
    return parser.parse_args()


def get_data_from_db(collection_name, classifier_names, keywords):
    """
    Connect to MongoDB and fetch documents that contain classifier scores and keyword counts.

    For each classifier name (e.g. "facebook/bart-large-mnli"),
    we normalize it (replace "/" with "_") and then build the field name as:
      "classifier-score.{normalized_name}"

    For keywords, we fetch the counts from the "keywords" field.

    Also projects additional fields like "title" and "appId".

    Args:
        collection_name: Name of the MongoDB collection
        classifier_names: List of classifier names to use as features
        keywords: List of keywords to use as features

    Returns:
        flattened_data: list of dicts with keys "title", "appId", normalized classifiers, and keyword counts
        feature_columns: list of all feature column names (classifiers and keywords)
    """
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise Exception("Error: MONGO_URI environment variable is not set.")

    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    collection = db[collection_name]

    projection = {"_id": 0, "title": 1, "appId": 1}
    query_conditions = []
    feature_columns = []

    for classifier_name in classifier_names:
        regularized_classifier_name = classifier_name.replace("/", "_")
        field_name = f"classifierScores.{regularized_classifier_name}"
        projection[field_name] = 1
        query_conditions.append({field_name: {"$exists": True}})
        feature_columns.append(regularized_classifier_name)

    for keyword in keywords:
        field_name = f"keywordCounts.{keyword}"
        projection[field_name] = 1
        query_conditions.append({field_name: {"$exists": True}})
        feature_columns.append(f"ky_{keyword}_normalized")

    query = {"$and": query_conditions} if query_conditions else {}
    cursor = collection.find(query, projection)
    count = collection.count_documents(query)
    print(f"Found {count} documents with required fields")

    data = []
    for doc in cursor:
        flat_doc = {}
        flat_doc["title"] = doc.get("title", "")
        flat_doc["appId"] = doc.get("appId", "")

        classifier_scores = doc.get("classifierScores", {})
        for classifier_name in classifier_names:
            regularized_classifier_name = classifier_name.replace("/", "_")
            score = classifier_scores.get(regularized_classifier_name)
            flat_doc[regularized_classifier_name] = score

        keyword_counts = doc.get("keywordCounts", {})
        for keyword in keywords:
            count = keyword_counts.get(keyword)
            flat_doc[f"ky_{keyword}_count"] = count
            flat_doc[f"ky_{keyword}_normalized"] = 1 - np.exp(-count)

        data.append(flat_doc)

    return data, feature_columns


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
    args = parse_args()

    if not args.collection:
        raise Exception(
            "Error: Both collection name and classifier names must be provided."
        )

    print("Connecting to DB and fetching data...")
    data, feature_columns = get_data_from_db(
        args.collection, args.classifiers, args.keywords
    )
    if not data:
        raise Exception("No data found matching the criteria.")
    df = pd.DataFrame(data)
    print(f"Fetched {len(df)} records.")

    print(
        "Using classifier scores and keyword counts. Feature matrix shape:",
        df[feature_columns].shape,
    )

    if args.mode == "test":
        k_values = list(range(args.k_min, args.k_max + 1))
        k_results = evaluate_kmeans(df[feature_columns].values, k_values, args.n_runs)
        output_filename = args.output if args.output else "clustering_loss.png"
        plot_results(k_results, output_filename)
    elif args.mode == "cluster":
        clustered_df = cluster_data(df, feature_columns, k=args.k)

        mongo_uri = os.getenv("MONGO_URI")
        client = MongoClient(mongo_uri)
        db = client.get_default_database()
        collection = db[args.collection]

        update_count = 0
        for _, row in clustered_df.iterrows():
            app_id = row["appId"]
            cluster_label = int(row["cluster"])

            result = collection.update_one(
                {"appId": app_id},
                {
                    "$set": {
                        "clusterLabel": cluster_label,
                    }
                },
            )

            if result.modified_count > 0:
                update_count += 1

        print(f"Updated {update_count} records with cluster information (k={args.k}).")


if __name__ == "__main__":
    main()
