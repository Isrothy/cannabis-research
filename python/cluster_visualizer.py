import os
import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from pandas.plotting import parallel_coordinates
import seaborn as sns


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_data_from_db(collection_name, classifier_names, k_value):
    """
    Connect to MongoDB and fetch documents with cluster labels.

    Args:
        collection_name: MongoDB collection name
        classifier_names: List of classifier names to include
        k_value: The k value used for clustering

    Returns:
        DataFrame containing the clustered data and field_mapping
    """
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise Exception("Error: MONGO_URI environment variable is not set.")

    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    collection = db[collection_name]

    field_mapping = {}
    # Project app title, id, and cluster label
    projection = {"title": 1, "appId": 1, "cluster_label": 1, "_id": 0}

    # Add classifier scores to projection
    for classifier in classifier_names:
        normalized = classifier.replace("/", "_")
        nested_field = f"classifier-score.{normalized}"
        field_mapping[nested_field] = normalized
        projection[nested_field] = 1

    # Build query to find records with cluster_label
    query = {"cluster_label": {"$exists": True}}

    # Retrieve the data
    cursor = collection.find(query, projection)
    data = list(cursor)

    if not data:
        raise Exception(
            f"No data found with cluster labels in collection {collection_name}"
        )

    # Flatten the data
    flattened_data = []
    for doc in data:
        flat_doc = {}
        flat_doc["title"] = doc.get("title")
        flat_doc["appId"] = doc.get("appId")
        flat_doc["cluster_label"] = doc.get("cluster_label")

        nested = doc.get("classifier-score", {})
        for nested_field, normalized in field_mapping.items():
            key = nested_field.split(".", 1)[1]
            if key in nested:
                flat_doc[normalized] = nested[key]
        flattened_data.append(flat_doc)

    df = pd.DataFrame(flattened_data)
    return df, field_mapping


def print_cluster_statistics(df, feature_cols):
    """
    Print statistics for each classifier in each cluster.

    Args:
        df: DataFrame with cluster labels and classifier scores
        feature_cols: List of classifier column names
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
            friendly_name = col
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
    Create box plots for each cluster showing classifier scores.

    Args:
        df: DataFrame with cluster labels and classifier scores
        feature_cols: List of classifier column names
        output_prefix: Prefix for output files
    """
    # Melt the DataFrame for easier plotting
    plot_df = df.melt(
        id_vars=["cluster_label", "appId", "title"],
        value_vars=feature_cols,
        var_name="Classifier",
        value_name="Score",
    )

    # Create a figure with subplots
    plt.figure(figsize=(14, 10))

    # Create box plot with clusters on x-axis and reduced outliers
    # Set flierprops to reduce the number of outliers displayed
    flierprops = dict(
        marker="o",
        markerfacecolor="gray",
        markersize=3,
        linestyle="none",
        markeredgecolor="gray",
        alpha=0.5,
    )

    # Create box plot with clusters on x-axis
    sns.boxplot(
        x="cluster_label",
        y="Score",
        hue="Classifier",
        data=plot_df,
        flierprops=flierprops,  # Reduce outlier visibility
        fliersize=2,  # Make outlier points smaller
        showfliers=True,  # Still show outliers but with reduced visibility
    )

    plt.title("Classifier Scores by Cluster", fontsize=16)
    plt.xlabel("Cluster", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the plot if output prefix is specified
    if output_prefix:
        output_file = f"{output_prefix}_boxplots.png"
        plt.savefig(output_file)
        print(f"Box plots saved to: {output_file}")

    plt.show()


def create_parallel_coordinates_plot(df, feature_cols, output_prefix=None):
    """
    Create a parallel coordinates plot to visualize clusters.

    Args:
        df: DataFrame with cluster labels and classifier scores
        feature_cols: List of classifier column names
        output_prefix: Prefix for output files
    """
    # Create a copy of the DataFrame with only the needed columns
    plot_df = df[["cluster_label"] + feature_cols].copy()

    # Normalize the data for better visualization
    for col in feature_cols:
        if col in plot_df.columns:
            min_val = plot_df[col].min()
            max_val = plot_df[col].max()
            if max_val > min_val:  # Avoid division by zero
                plot_df[col] = (plot_df[col] - min_val) / (max_val - min_val)

    # Create the parallel coordinates plot
    plt.figure(figsize=(14, 8))

    # Use cluster_label as the class column
    parallel_coordinates(plot_df, "cluster_label")

    plt.title("Parallel Coordinates Plot of Clusters", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save the plot if output prefix is specified
    if output_prefix:
        output_file = f"{output_prefix}_parallel_coordinates.png"
        plt.savefig(output_file)
        print(f"Parallel coordinates plot saved to: {output_file}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and analyze clusters from MongoDB data."
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--collection", type=str, help="MongoDB collection name")
    parser.add_argument("--k", type=int, help="The k value used for clustering")
    parser.add_argument(
        "--classifiers",
        type=str,
        nargs="+",
        help="List of classifier names to use as features",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        help="Prefix for output image files",
    )

    args = parser.parse_args()

    # Load YAML config if provided
    config = {}
    if args.config:
        config = load_config(args.config)

    # Command-line arguments override YAML config
    collection_name = args.collection or config.get("collection")
    classifier_names = args.classifiers or config.get("classifiers")
    k_value = args.k if args.k is not None else config.get("k")
    output_prefix = args.output_prefix or config.get("output_prefix")

    if not collection_name or not classifier_names or k_value is None:
        raise Exception(
            "Error: Collection name, classifier names, and k value must be provided."
        )

    print(f"Retrieving data from collection {collection_name} with k={k_value}...")
    df, field_mapping = get_data_from_db(collection_name, classifier_names, k_value)

    print(f"Retrieved {len(df)} records with cluster labels.")

    # Use the normalized classifier names as our feature columns
    feature_cols = list(field_mapping.values())

    if not feature_cols:
        print("No feature columns found. Please specify valid classifiers.")
        return

    print(f"Using {len(feature_cols)} features for analysis and visualization")

    # Print statistics for each classifier in each cluster
    print_cluster_statistics(df, feature_cols)

    # Create box plots
    create_boxplots(df, feature_cols, output_prefix)

    # Create parallel coordinates plot
    create_parallel_coordinates_plot(df, feature_cols, output_prefix)


if __name__ == "__main__":
    main()
