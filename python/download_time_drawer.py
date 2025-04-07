import os
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt

# Define the database name
db_name = "appleApps"
# db_name = "googleApps"
# format = "%b %d, %Y"
format = None


def fetch_data():
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    collection = db[db_name]

    query = {
        "$or": [
            {"predictedLabel": {"$eq": 1}},
            {"keywordCounts.cannabis": {"$gt": 0}},
            {"keywordCounts.marijuana": {"$gt": 0}},
        ],
    }
    projection = {"_id": 0, "title": 1, "appId": 1, "released": 1}

    cursor = collection.find(query, projection).sort("released", -1)
    return list(cursor)


def plot_release_years(data):
    df = pd.DataFrame(data)

    df["released"] = pd.to_datetime(df["released"], format=format, errors="coerce")
    df["year"] = df["released"].dt.year

    release_counts = df.groupby("year").size().reset_index(name="count")

    plt.figure(figsize=(12, 6))
    plt.bar(release_counts["year"], release_counts["count"], color="skyblue")
    plt.xlabel("Year")
    plt.ylabel("Number of Released Apps")
    plt.title("Number of Released Apps per Year")
    plt.xticks(release_counts["year"], rotation=45)
    plt.grid(axis="y")
    plt.tight_layout()

    filename = f"img/{db_name}_App_Released_per_year"
    plt.savefig(filename)
    print(f"Graph saved as {filename}")
    plt.show()


def main():
    data = fetch_data()
    if not data:
        print("No data found matching the query criteria.")
    else:
        plot_release_years(data)


if __name__ == "__main__":
    main()
