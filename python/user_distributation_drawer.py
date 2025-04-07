import os
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt

# db = "googleApps"
# field = "maxInstalls"

db_name = "googleApps"
# db_name = "appleApps"
field = "ratings"
metric = "ratings"
# field = "maxInstalls"
# metric = "intalls"
# field = "reviews"
# metric = "reviews"


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
    projection = {"_id": 0, "title": 1, "appId": 1, field: 1}

    cursor = collection.find(query, projection).sort(field, -1)
    return list(cursor)


def analyze_data(data):
    df = pd.DataFrame(data)
    total_num = df[field].sum()
    top10_num = df.head(10)[field].sum()
    top20_num = df.head(20)[field].sum()
    ratio10 = top10_num / total_num if total_num != 0 else 0
    ratio20 = top20_num / total_num if total_num != 0 else 0

    print(f"Total {metric} for all apps: {total_num}")
    print(f"Total {metric} for the top 10 apps: {top10_num}")
    print(f"Total {metric} for the top 20 apps: {top20_num}")
    print(f"The top 10 apps account for {ratio10:.2%} of the total {metric}.")
    print(f"The top 20 apps account for {ratio20:.2%} of the total {metric}.")


def plot_installs(data):
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(df) + 1), df[field], marker="o", linestyle="-")
    plt.xlabel("Rank")
    plt.ylabel(f"Number of {metric.capitalize()}")
    plt.title(f"App {metric.capitalize()} vs. Rank")
    plt.grid(True)

    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()

    filename = f"{db_name}_{metric.capitalize()}_vs_Rank.png"
    plt.savefig(filename)
    print(f"Graph saved as {filename}")
    plt.show()


def main():
    data = fetch_data()
    if not data:
        print("No data found matching the query criteria.")
    else:
        analyze_data(data)
        plot_installs(data)


if __name__ == "__main__":
    main()
