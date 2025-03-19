import os
import argparse
import pandas as pd
import yaml
from pymongo import MongoClient
from typing import Dict, Any, Optional


def get_cluster_samples(
    collection_name: str,
    samples_per_cluster: int,
    classifiers: Optional[list] = None,
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Sample records from each cluster in the database.

    Args:
        collection_name: MongoDB collection name
        samples_per_cluster: Number of samples to take from each cluster
        classifiers: Optional list of classifier names to include their scores
        output_file: Optional file path to save the samples as CSV

    Returns:
        DataFrame containing the sampled records with appId, title, cluster_label,
        cluster_weight, and classifier scores
    """
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise Exception("Error: MONGO_URI environment variable is not set.")

    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    collection = db[collection_name]

    query = {"cluster_label": {"$exists": True}}

    distinct_clusters = collection.distinct("cluster_label", query)
    print(f"Found {len(distinct_clusters)} clusters")

    # Get total number of records with cluster labels
    total_records = collection.count_documents(query)

    all_samples = []
    cluster_sizes = {}

    for cluster_label in distinct_clusters:
        cluster_query = {"cluster_label": cluster_label}

        cluster_size = collection.count_documents(cluster_query)
        cluster_sizes[cluster_label] = cluster_size
        print(f"Cluster {cluster_label}: {cluster_size} records")

        n_samples = min(samples_per_cluster, cluster_size)

        # Build projection to include required fields
        projection = {"appId": 1, "title": 1, "cluster_label": 1, "_id": 0}

        # Add classifier scores to projection if specified
        if classifiers:
            for classifier in classifiers:
                normalized = classifier.replace("/", "_")
                projection[f"classifier-score.{normalized}"] = 1

        if n_samples == cluster_size:
            cursor = collection.find(cluster_query, projection)
        else:
            pipeline = [
                {"$match": cluster_query},
                {"$project": projection},
                {"$sample": {"size": n_samples}},
            ]
            cursor = collection.aggregate(pipeline)

        cluster_samples = list(cursor)

        # Add cluster weight to each sample
        for sample in cluster_samples:
            # Calculate weight as proportion of total records
            sample["cluster_weight"] = cluster_size / total_records

            # Extract classifier scores if they exist
            if classifiers:
                classifier_scores = sample.get("classifier-score", {})
                for classifier in classifiers:
                    normalized = classifier.replace("/", "_")
                    if normalized in classifier_scores:
                        sample[normalized] = classifier_scores[normalized]

                # Remove the nested classifier-score object
                if "classifier-score" in sample:
                    del sample["classifier-score"]

        all_samples.extend(cluster_samples)
        print(f"Sampled {len(cluster_samples)} records from cluster {cluster_label}")

    df = pd.DataFrame(all_samples)

    if df.empty:
        print("No samples found.")
        return df

    required_columns = ["appId", "title", "cluster_label", "cluster_weight"]

    # Add classifier columns to required columns if specified
    if classifiers:
        for classifier in classifiers:
            normalized = classifier.replace("/", "_")
            required_columns.append(normalized)

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        for col in missing_columns:
            df[col] = None

    # Select only columns that exist in the DataFrame
    existing_columns = [col for col in required_columns if col in df.columns]
    result_df = df[existing_columns].copy()

    if output_file and not result_df.empty:
        result_df.to_csv(output_file, index=False)
        print(f"Saved {len(result_df)} samples to {output_file}")

    return result_df


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_file: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Sample records from clusters in MongoDB."
    )
    parser.add_argument("--collection", type=str, help="MongoDB collection name")
    parser.add_argument("--k", type=int, help="The k value used for clustering")
    parser.add_argument(
        "--samples",
        type=int,
        default=15,
        help="Number of samples to take from each cluster",
    )
    parser.add_argument("--output", type=str, help="Output CSV file path")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "--classifiers",
        type=str,
        nargs="+",
        help="List of classifier names to include their scores",
    )

    args = parser.parse_args()

    config = {}
    if args.config:
        config = load_config(args.config)

    collection_name = args.collection or config.get("collection")
    k = args.k if args.k is not None else config.get("k")
    samples = args.samples if args.samples is not None else config.get("samples", 15)
    output = args.output or config.get("output")
    classifiers = args.classifiers or config.get("classifiers")

    if not collection_name:
        parser.error("Collection name is required (--collection or in config file)")
    if k is None:
        parser.error("K value is required (--k or in config file)")

    # Get samples
    df = get_cluster_samples(collection_name, samples, classifiers, output)

    print(f"Total samples: {len(df)}")

    # Print a preview of the samples
    if not df.empty:
        print("\nSample preview:")
        print(df.head())


if __name__ == "__main__":
    main()
