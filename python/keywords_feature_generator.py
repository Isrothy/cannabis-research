import configargparse
import sys
import os
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer


def parse_args():
    parser = configargparse.ArgumentParser(
        description="Classify app descriptions from MongoDB and update records with the occurance of keywords.",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument(
        "--config", is_config_file=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="Name of the collection containing app records.",
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help="Overwrite the exisiting feature vectors",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        help="List of keywords to search for in app titles and descriptions.",
    )
    return parser.parse_args()


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


def generate_keywords_features(collection_name, overwrite, keywords):
    """
    Count occurrences of keywords in app titles and descriptions using CountVectorizer,
    then update the database with these counts.

    Args:
        collection_name: Name of the MongoDB collection
        keywords: List of keywords to search for
    """
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("Environment variable MONGO_URI is not set.")

    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    collection = db[collection_name]

    field_name = "keywordCounts"
    if overwrite:
        collection.update_many({}, {"$unset": {"keywordCounts": ""}})
    query = {
        field_name: {"$exists": False},
    }

    total = collection.count_documents(query)
    print(f"Found {total} records to process.")

    vectorizer = CountVectorizer(
        vocabulary=keywords, lowercase=True, token_pattern=r"\b\w+\b"
    )

    batch_size = 100
    processed = 0

    while processed < total:
        batch_records = list(collection.find(query).limit(batch_size))
        if not batch_records:
            break

        batch_texts = []
        batch_ids = []

        for record in batch_records:
            title = record.get("title", "")
            desc = record.get("description", "")
            if not title:
                title = ""
            if not desc:
                desc = ""

            text = title + " " + desc
            batch_texts.append(text)
            batch_ids.append(record["_id"])

        X = vectorizer.fit_transform(batch_texts)

        counts = X.toarray()

        for i, record_id in enumerate(batch_ids):
            keyword_counts = {}
            for j, keyword in enumerate(keywords):
                keyword_counts[keyword] = int(counts[i, j])

            collection.update_one(
                {"_id": record_id}, {"$set": {field_name: keyword_counts}}
            )

            processed += 1
            print_progress(processed, total)

    print("\nFinished updating records.")


if __name__ == "__main__":
    args = parse_args()

    generate_keywords_features(args.collection, args.overwrite, args.keywords)
