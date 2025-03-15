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


def classify_and_update_records(collection_name, model_name, task):
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("Environment variable MONGO_URI is not set.")

    if task.lower() == "sentiment":
        classifier = pipeline("text-classification", model=model_name)
    elif task.lower() == "classification":
        classifier = pipeline("zero-shot-classification", model=model_name)
    else:
        raise ValueError("Task must be either 'sentiment' or 'classification'.")

    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    collection = db[collection_name]

    field_name = f"classifier-score.{model_name}"
    query = {
        "description": {"$exists": True},
        field_name: {"$exists": False},
    }

    total = collection.count_documents(query)
    print(f"Found {total} records to process.")

    records = collection.find(query)

    count = 0
    for record in records:
        count += 1
        desc = record.get("description")
        if not desc:
            print(f"\nNo description found in record: {record}")
            continue

        if task.lower() == "sentiment":
            # Run sentiment analysis using the text-classification pipeline.
            result = classifier(desc, truncation=True)[0]
            label = result["label"]
            score = result["score"]
            # Map sentiment to a cannabis-related score.
            cannabis_score = score if label.upper() == "POSITIVE" else 1 - score
        else:
            # Run zero-shot classification.
            result = classifier(
                desc, candidate_labels, hypothesis_template="This app is {}"
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
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["sentiment", "classification"],
        help="Type of analysis to perform: 'sentiment' or 'classification'.",
    )
    args = parser.parse_args()

    classify_and_update_records(args.collection, args.model, args.task)
