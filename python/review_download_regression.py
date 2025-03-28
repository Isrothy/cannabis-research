import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform linear regression on maxInstalls and ratings from MongoDB data."
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="your_collection_name",
        help="Name of the MongoDB collection to query.",
    )
    return parser.parse_args()


def load_data_from_mongodb(collection_name):
    """
    Connect to MongoDB using the MONGO_URI environment variable,
    and retrieve documents with fields 'maxInstalls' and 'ratings'.

    Returns a DataFrame with these fields.
    """
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise Exception("MONGO_URI environment variable is not set.")

    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    collection = db[collection_name]

    # Query for documents that contain both 'maxInstalls' and 'ratings'
    cursor = collection.find(
        {"maxInstalls": {"$exists": True}, "ratings": {"$gt": 0}},
        {"maxInstalls": 1, "ratings": 1, "_id": 0},
    )

    data = list(cursor)
    if not data:
        raise Exception("No documents with required fields found.")

    df = pd.DataFrame(data)
    return df


def main():
    args = parse_args()

    print(f"Loading data from MongoDB collection: {args.collection} ...")
    df = load_data_from_mongodb(args.collection)
    print(f"Retrieved {len(df)} documents.")

    # Ensure the fields are numeric
    df["maxInstalls"] = pd.to_numeric(df["maxInstalls"], errors="coerce")
    df["ratings"] = pd.to_numeric(df["ratings"], errors="coerce")
    df.dropna(subset=["maxInstalls", "ratings"], inplace=True)

    # Apply logarithm transformation to both fields.
    # We add 1 to avoid log(0)
    df["log_maxInstalls"] = np.log10(df["maxInstalls"] + 1)
    df["log_ratings"] = np.log10(df["ratings"] + 1)

    # Prepare the data for regression
    X = df[["log_maxInstalls"]].values  # Predictor
    y = df["log_ratings"].values  # Target

    # Fit a linear regression model
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)

    print(f"Regression Coefficient: {reg.coef_[0]:.4f}")
    print(f"Regression Intercept: {reg.intercept_:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # Plot the scatter plot and regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["log_maxInstalls"], df["log_ratings"], alpha=0.5, label="Data points"
    )
    plt.plot(df["log_maxInstalls"], y_pred, color="red", label="Regression line")
    plt.xlabel("Log10(maxInstalls)")
    plt.ylabel("Log10(ratings)")
    plt.title("Linear Regression: Log10(maxInstalls) vs Log10(ratings)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
