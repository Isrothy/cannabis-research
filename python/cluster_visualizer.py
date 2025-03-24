import os
import configargparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from pandas.plotting import parallel_coordinates
import seaborn as sns


def get_data_from_db(collection_name, classifier_names, keywords):
    """
    Connect to MongoDB and fetch documents with cluster labels, classifier scores,
    and keyword counts.

    Args:
        collection_name: MongoDB collection name
        classifier_names: List of classifier names to include
        keywords: List of keywords to include
        k_value: The k value used for clustering

    Returns:
        DataFrame containing the clustered data and feature_columns
    """
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise Exception("Error: MONGO_URI environment variable is not set.")

    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    collection = db[collection_name]

    projection = {"title": 1, "appId": 1, "clusterLabel": 1, "_id": 0}
    feature_columns = []

    for classifier in classifier_names:
        regularized = classifier.replace("/", "_")
        field_name = f"classifierScores.{regularized}"
        projection[field_name] = 1
        feature_columns.append(regularized)

    for keyword in keywords:
        field_name = f"keywordCounts.{keyword}"
        projection[field_name] = 1
        feature_columns.append(f"ky_{keyword}_normalized")

    query = {"clusterLabel": {"$exists": True}}

    cursor = collection.find(query, projection)
    data = list(cursor)

    if not data:
        raise Exception(
            f"No data found with cluster labels in collection {collection_name}"
        )

    flattened_data = []
    for doc in data:
        flat_doc = {}
        flat_doc["title"] = doc.get("title", "")
        flat_doc["appId"] = doc.get("appId", "")
        flat_doc["cluster_label"] = doc.get("clusterLabel")

        classifier_scores = doc.get("classifierScores", {})
        for classifier in classifier_names:
            regularized = classifier.replace("/", "_")
            score = classifier_scores.get(regularized, 0.0)
            flat_doc[regularized] = score

        keyword_counts = doc.get("keywordCounts", {})
        for keyword in keywords:
            count = keyword_counts.get(keyword, 0)
            flat_doc[f"ky_{keyword}_count"] = count
            flat_doc[f"ky_{keyword}_normalized"] = 1 - np.exp(-count)

        flattened_data.append(flat_doc)

    df = pd.DataFrame(flattened_data)
    return df, feature_columns


def print_cluster_statistics(df, feature_cols):
    """
    Print statistics for each feature in each cluster.

    Args:
        df: DataFrame with cluster labels and feature values
        feature_cols: List of feature column names
    """
    print("\n" + "=" * 80)
    print("CLUSTER STATISTICS")
    print("=" * 80)

    clusters = df["cluster_label"].unique()
    clusters.sort()

    for cluster in clusters:
        cluster_df = df[df["cluster_label"] == cluster]
        print(f"\nCluster {cluster} (n={len(cluster_df)})")
        print("-" * 40)

        for col in feature_cols:
            # Get friendly name for display
            if col.startswith("ky_") and col.endswith("_normalized"):
                friendly_name = f"Keyword: {col[3:-12]}"
            else:
                friendly_name = f"Classifier: {col}"

            if col in df.columns:
                stats = cluster_df[col].describe(percentiles=[0.25, 0.75])
                print(f"  {friendly_name}:")
                print(f"    Mean: {stats['mean']:.4f}")
                print(f"    Std Dev: {stats['std']:.4f}")
                print(f"    25% Quartile: {stats['25%']:.4f}")
                print(f"    75% Quartile: {stats['75%']:.4f}")

    print("\n" + "=" * 80)


def create_boxplots(df, feature_cols, output_prefix=None):
    """
    Create box plots for each cluster showing feature values.
    Creates separate plots for classifier scores and keyword counts.

    Args:
        df: DataFrame with cluster labels and feature values
        feature_cols: List of feature column names
        output_prefix: Prefix for output files
    """
    classifier_cols = [col for col in feature_cols if not col.startswith("ky_")]
    keyword_cols = [col for col in feature_cols if col.startswith("ky_")]

    flierprops = dict(
        marker="o",
        markerfacecolor="gray",
        markersize=3,
        linestyle="none",
        markeredgecolor="gray",
        alpha=0.5,
    )

    if classifier_cols:
        classifier_df = df.melt(
            id_vars=["cluster_label", "appId", "title"],
            value_vars=classifier_cols,
            var_name="Feature",
            value_name="Score",
        )

        plt.figure(figsize=(14, 10))
        sns.boxplot(
            x="cluster_label",
            y="Score",
            hue="Feature",
            data=classifier_df,
            flierprops=flierprops,
            fliersize=2,
            showfliers=True,
        )

        plt.title("Classifier Scores by Cluster", fontsize=16)
        plt.xlabel("Cluster", fontsize=14)
        plt.ylabel("Score", fontsize=14)
        plt.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        if output_prefix:
            output_file = f"{output_prefix}_classifier_boxplots.png"
            plt.savefig(output_file)
            print(f"Classifier box plots saved to: {output_file}")

        plt.show()

    if keyword_cols:
        keyword_mapping = {col: col[3:-11] for col in keyword_cols}

        keyword_df = df.melt(
            id_vars=["cluster_label", "appId", "title"],
            value_vars=keyword_cols,
            var_name="Feature",
            value_name="Score",
        )

        keyword_df["Keyword"] = keyword_df["Feature"].map(
            lambda x: keyword_mapping.get(x, x)
        )

        plt.figure(figsize=(14, 10))
        sns.boxplot(
            x="cluster_label",
            y="Score",
            hue="Keyword",
            data=keyword_df,
            flierprops=flierprops,
            fliersize=2,
            showfliers=True,
        )

        plt.title("Normalized Keyword Counts by Cluster", fontsize=16)
        plt.xlabel("Cluster", fontsize=14)
        plt.ylabel("Normalized Count", fontsize=14)
        plt.legend(title="Keyword", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        if output_prefix:
            output_file = f"{output_prefix}_keyword_boxplots.png"
            plt.savefig(output_file)
            print(f"Keyword box plots saved to: {output_file}")

        plt.show()


def parse_args():
    parser = configargparse.ArgumentParser(
        description="Visualize and analyze clusters from MongoDB data.",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument(
        "--config", is_config_file=True, help="Path to YAML configuration file"
    )
    parser.add_argument("--collection", type=str, help="MongoDB collection name")
    parser.add_argument("--k", type=int, help="The k value used for clustering")
    parser.add_argument(
        "--classifiers",
        type=str,
        nargs="+",
        help="List of classifier names to use as features",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        help="List of keywords to use as features",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        help="Prefix for output image files",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.collection or args.k is None:
        raise Exception("Error: Collection name and k value must be provided.")

    if not args.classifiers and not args.keywords:
        raise Exception("Error: At least one classifier or keyword must be provided.")

    classifier_names = args.classifiers or []
    keywords = args.keywords or []

    print(f"Retrieving data from collection {args.collection} with k={args.k}...")
    print(f"Using {len(classifier_names)} classifiers and {len(keywords)} keywords")

    df, feature_columns = get_data_from_db(args.collection, classifier_names, keywords)

    print(f"Retrieved {len(df)} records with cluster labels.")

    if not feature_columns:
        print("No feature columns found. Please specify valid classifiers or keywords.")
        return

    print(f"Using {len(feature_columns)} features for analysis and visualization")

    print_cluster_statistics(df, feature_columns)

    create_boxplots(df, feature_columns, args.output_prefix)


if __name__ == "__main__":
    main()
