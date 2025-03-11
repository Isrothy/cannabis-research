import argparse
import os
from pymongo import MongoClient
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["cannabis-related app", "non-cannabis related app"]


def classify_and_update_records(collection_name):
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("Environment variable MONGO_URI is not set.")

    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    collection = db[collection_name]

    records = collection.find({"description": {"$exists": True}})

    for record in records:
        desc = record.get("description")
        title = record.get("title", "Unknown")
        if not desc:
            print("No description found in record:", record)
            continue

        result = classifier(desc, candidate_labels)
        cannabis_score = result["scores"][
            result["labels"].index("cannabis-related app")
        ]

        print("-" * 50)
        print(f"Title: {title}")
        print(f"Description: {desc}")
        print(f"Cannabis App Score: {cannabis_score:.4f}")

        collection.update_one(
            {"_id": record["_id"]}, {"$set": {"cannabisScore": cannabis_score}}
        )

    print("Finished updating records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify app descriptions from MongoDB and update the records with cannabis app scores."
    )
    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Name of the collection containing app records.",
    )
    args = parser.parse_args()

    classify_and_update_records(args.collection)
