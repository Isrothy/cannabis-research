import csv
import os
from pymongo import MongoClient

mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client.get_default_database()
collection = db["apple"]

query = {
    "$or": [
        {"predictedLabel": {"$eq": 1}},
        {"keywordCounts.cannabis": {"$gt": 0}},
        {"keywordCounts.marijuana": {"$gt": 0}},
    ],
    "reviews": {"$gt": 100},
}

projection = {"_id": 0, "title": 1, "appId": 1}

cursor = collection.find(query, projection)

output_file = "apple_cannabis_apps.csv"

with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["appId", "title"])
    writer.writeheader()
    for document in cursor:
        writer.writerow(document)

print(f"Data saved to {output_file}")
