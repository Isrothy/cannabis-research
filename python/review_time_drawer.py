import os
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt

db_name = "googleApps"


def fetch_review_data():
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    collection = db[db_name]

    pipeline = [
        {
            "$match": {
                "$or": [
                    {"predictedLabel": {"$eq": 1}},
                    {"keywordCounts.cannabis": {"$gt": 0}},
                    {"keywordCounts.marijuana": {"$gt": 0}},
                ]
            }
        },
        # Unwind the reviewList array to process each review separately
        {"$unwind": "$reviewList"},
        # Convert the review date string to a Date and extract the year
        {"$project": {"year": {"$year": {"$toDate": "$reviewList.date"}}}},
        # Group by year and count the reviews
        {"$group": {"_id": "$year", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]

    result = list(collection.aggregate(pipeline))
    return result


def plot_reviews_by_year(data):
    # Convert the aggregated data to a DataFrame
    df = pd.DataFrame(data)
    # Rename _id to 'year'
    df = df.rename(columns={"_id": "year"})

    plt.figure(figsize=(12, 6))
    plt.bar(df["year"].astype(str), df["count"], color="skyblue")
    plt.xlabel("Year")
    plt.ylabel("Number of Reviews")
    plt.title("Number of Reviews by Year")
    plt.grid(axis="y")
    plt.tight_layout()

    filename = f"img/{db_name}_Reviews_by_Year.png"
    plt.savefig(filename)
    print(f"Graph saved as {filename}")
    plt.show()


def main():
    data = fetch_review_data()
    if not data:
        print("No review data found matching the query criteria.")
    else:
        plot_reviews_by_year(data)


if __name__ == "__main__":
    main()
