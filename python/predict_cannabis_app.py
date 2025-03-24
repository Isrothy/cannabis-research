import argparse
import pandas as pd
import numpy as np
import json
from cannabis_app_classifier import load_saved_model, predict_cannabis_app


def predict_from_csv(model, input_file, output_file=None):
    """
    Predict cannabis-related apps from a CSV file.

    Args:
        model: Trained model object
        input_file: Path to input CSV file
        output_file: Optional path to save predictions

    Returns:
        DataFrame with predictions
    """
    # Load the data
    df = pd.read_csv(input_file)

    # Check if required feature columns exist
    missing_cols = [col for col in model["feature_cols"] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input file: {missing_cols}")

    # Make predictions
    features = df[model["feature_cols"]].values
    scaled_features = model["scaler"].transform(features)

    # Get predictions and probabilities
    predictions = []
    probabilities = []

    for i in range(len(df)):
        pred = model["predict"](scaled_features[i : i + 1])[0]
        prob = model["predict_proba"](scaled_features[i : i + 1])[0]
        predictions.append(pred)
        probabilities.append(prob)

    # Add predictions to the DataFrame
    df["predicted_cannabis"] = predictions
    df["cannabis_probability"] = probabilities

    # Save predictions if output file is specified
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")

    return df


def predict_from_json(model, input_file, output_file=None):
    """
    Predict cannabis-related apps from a JSON file.

    Args:
        model: Trained model object
        input_file: Path to input JSON file
        output_file: Optional path to save predictions

    Returns:
        List of dictionaries with predictions
    """
    # Load the data
    with open(input_file, "r") as f:
        data = json.load(f)

    # Make predictions
    results = []

    for app in data:
        # Check if required feature columns exist
        missing_cols = [col for col in model["feature_cols"] if col not in app]
        if missing_cols:
            print(
                f"Warning: App {app.get('appId', 'unknown')} missing columns: {missing_cols}"
            )
            continue

        # Extract features
        features = np.array([[app.get(col, 0) for col in model["feature_cols"]]])
        scaled_features = model["scaler"].transform(features)

        # Get prediction and probability
        pred = model["predict"](scaled_features)[0]
        prob = model["predict_proba"](scaled_features)[0]

        # Add prediction to the app data
        app_result = app.copy()
        app_result["predicted_cannabis"] = int(pred)
        app_result["cannabis_probability"] = float(prob)
        results.append(app_result)

    # Save predictions if output file is specified
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Predictions saved to: {output_file}")

    return results


def predict_single_app(model, app_data):
    """
    Predict whether a single app is cannabis-related.

    Args:
        model: Trained model object
        app_data: Dictionary containing app features

    Returns:
        Prediction and probability
    """
    # Check if required feature columns exist
    missing_cols = [col for col in model["feature_cols"] if col not in app_data]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Extract features
    features = np.array([[app_data.get(col, 0) for col in model["feature_cols"]]])
    scaled_features = model["scaler"].transform(features)

    # Get prediction and probability
    pred = model["predict"](scaled_features)[0]
    prob = model["predict_proba"](scaled_features)[0]

    return pred, prob


def main():
    parser = argparse.ArgumentParser(
        description="Predict cannabis-related apps using the trained classifier."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/cannabis_app_classifier.joblib",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input file (CSV or JSON)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save predictions",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json"],
        help="Input file format (auto-detected if not specified)",
    )

    args = parser.parse_args()

    # Load the model
    print(f"Loading model from {args.model}...")
    model = load_saved_model(args.model)

    # Determine input format if not specified
    if not args.format:
        if args.input.lower().endswith(".csv"):
            args.format = "csv"
        elif args.input.lower().endswith(".json"):
            args.format = "json"
        else:
            raise ValueError(
                "Could not determine input format. Please specify --format."
            )

    # Make predictions
    print(f"Making predictions on {args.input}...")
    if args.format == "csv":
        results = predict_from_csv(model, args.input, args.output)
        print(f"Processed {len(results)} apps")

        # Print summary
        cannabis_count = results["predicted_cannabis"].sum()
        print(
            f"Predicted cannabis-related apps: {cannabis_count} ({cannabis_count/len(results)*100:.2f}%)"
        )
        print(
            f"Predicted non-cannabis apps: {len(results)-cannabis_count} ({(len(results)-cannabis_count)/len(results)*100:.2f}%)"
        )

    else:  # json
        results = predict_from_json(model, args.input, args.output)
        print(f"Processed {len(results)} apps")

        # Print summary
        cannabis_count = sum(1 for app in results if app["predicted_cannabis"] == 1)
        print(
            f"Predicted cannabis-related apps: {cannabis_count} ({cannabis_count/len(results)*100:.2f}%)"
        )
        print(
            f"Predicted non-cannabis apps: {len(results)-cannabis_count} ({(len(results)-cannabis_count)/len(results)*100:.2f}%)"
        )


if __name__ == "__main__":
    main()
