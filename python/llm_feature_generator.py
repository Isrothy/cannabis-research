import argparse
import sys
import os
from pymongo import MongoClient
from transformers import pipeline

candidate_labels = ["cannabis-related app", "not cannabis related app"]


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


def classify_and_update_records(collection_name, model_name):
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("Environment variable MONGO_URI is not set.")

    classifier = pipeline("zero-shot-classification", model=model_name)

    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    collection = db[collection_name]

    normalized_model_name = model_name.replace("/", "_")
    field_name = f"classifierScores.{normalized_model_name}"
    query = {
        field_name: {"$exists": False},
    }

    total = collection.count_documents(query)
    print(f"Found {total} records to process.")

    records = collection.find(query)

    count = 0
    for record in records:
        count += 1
        title = record.get("title")
        desc = record.get("description")
        if not title:
            title = ""
        if not desc:
            desc = ""

        result = classifier(
            f"This app is {title}. The description is: {desc}",
            candidate_labels,
            hypothesis_template="This app is {}",
        )
        cannabis_score = result["scores"][
            result["labels"].index("cannabis-related app")
        ]

        collection.update_one(
            {"_id": record["_id"]}, {"$set": {field_name: cannabis_score}}
        )

        print_progress(count, total)

    print("\nFinished updating records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify app descriptions from MongoDB and update records with classifier scores."
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
        help="Name of the Hugging Face model to use in the pipeline.",
    )
    args = parser.parse_args()

    classify_and_update_records(args.collection, args.model)
