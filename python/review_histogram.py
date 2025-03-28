import os
import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient


def main():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("Error: MONGO_URI environment variable not set.")
        return

    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    collection = db["appleApps"]

    cursor = collection.find({"reviews": {"$exists": True}}, {"reviews": 1, "_id": 0})

    reviews = [doc["reviews"] for doc in cursor if "reviews" in doc]

    if not reviews:
        print("No reviews found in the database.")
        return

    reviews = np.array(reviews)
    log_reviews = np.log10(reviews + 1)

    plt.figure(figsize=(10, 6))
    plt.hist(log_reviews, bins=30, edgecolor="black")
    plt.xlabel("Log10(Number of Reviews)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Log-Transformed Number of Reviews for All Apps")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
