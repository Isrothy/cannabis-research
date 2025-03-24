import os
import configargparse
import pandas as pd
from pymongo import MongoClient
from typing import Dict, Any, Optional


def parse_args():
    parser = configargparse.ArgumentParser(
        description="Sample records from clusters in MongoDB.",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument(
        "--config", is_config_file=True, help="Path to YAML configuration file"
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
    parser.add_argument(
        "--classifiers",
        type=str,
        nargs="+",
        help="List of classifier names to include their scores",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        help="List of keywords to use as features",
    )
    return parser.parse_args()


def get_cluster_samples(
    collection_name: str,
    samples_per_cluster: int,
    classifiers: Optional[list[str]] = None,
    keywords: Optional[list[str]] = None,
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
        DataFrame containing the sampled records with appId, title, clusterLabel,
        clusterWeight, and classifier scores
    """
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise Exception("Error: MONGO_URI environment variable is not set.")

    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    collection = db[collection_name]

    query = {"clusterLabel": {"$exists": True}}

    distinct_clusters = collection.distinct("clusterLabel", query)
    print(f"Found {len(distinct_clusters)} clusters")

    total_records = collection.count_documents(query)

    all_samples = []
    cluster_sizes = {}

    projection: dict[str, int | str] = {
        "appId": 1,
        "title": 1,
        "clusterLabel": 1,
        "_id": 0,
    }
    if classifiers:
        for classifier in classifiers:
            regularized: str = classifier.replace("/", "_")
            projection[regularized] = f"$classifierScores.{regularized}"
    if keywords:
        for keyword in keywords:
            projection[keyword] = f"$keywordCounts.{keyword}"

    for clusterLabel in distinct_clusters:
        cluster_query = {"clusterLabel": clusterLabel}

        cluster_size = collection.count_documents(cluster_query)
        cluster_sizes[clusterLabel] = cluster_size
        print(f"Cluster {clusterLabel}: {cluster_size} records")

        n_samples = min(samples_per_cluster, cluster_size)

        if n_samples >= cluster_size:
            cursor = collection.find(cluster_query, projection)
        else:
            pipeline = [
                {"$match": cluster_query},
                {"$project": projection},
                {"$sample": {"size": n_samples}},
            ]
            cursor = collection.aggregate(pipeline)

        cluster_samples = list(cursor)

        for sample in cluster_samples:
            sample["clusterWeight"] = cluster_size / total_records

        all_samples.extend(cluster_samples)
        print(f"Sampled {len(cluster_samples)} records from cluster {clusterLabel}")

    df = pd.DataFrame(all_samples)

    if df.empty:
        print("No samples found.")
        return df

    required_columns = ["appId", "title", "clusterLabel", "clusterWeight"]

    if classifiers:
        for classifier in classifiers:
            regularized = classifier.replace("/", "_")
            required_columns.append(regularized)
    if keywords:
        for keyword in keywords:
            required_columns.append(keyword)

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


def main():
    args = parse_args()

    # Get samples
    df = get_cluster_samples(
        args.collection,
        args.samples,
        args.classifiers,
        args.keywords,
        args.output,
    )

    print(f"Total samples: {len(df)}")

    # Print a preview of the samples
    if not df.empty:
        print("\nSample preview:")
        print(df.head())


if __name__ == "__main__":
    main()
