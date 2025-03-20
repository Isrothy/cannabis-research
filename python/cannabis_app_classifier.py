import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
import json


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a random forest classifier to identify cannabis-related apps."
    )

    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "--apple_data", type=str, help="Path to the Apple apps CSV file"
    )
    parser.add_argument(
        "--google_data", type=str, help="Path to the Google apps CSV file"
    )
    parser.add_argument(
        "--output_model", type=str, help="Path to save the trained model"
    )
    parser.add_argument(
        "--output_metrics", type=str, help="Path to save the evaluation metrics"
    )
    parser.add_argument("--plot_dir", type=str, help="Directory to save plots")

    parser.add_argument(
        "--n_estimators", type=int, help="Number of trees in the random forest"
    )
    parser.add_argument("--max_depth", type=int, help="Maximum depth of the trees")
    parser.add_argument(
        "--min_samples_split", type=int, help="Minimum samples to split"
    )
    parser.add_argument(
        "--min_samples_leaf", type=int, help="Minimum samples in a leaf"
    )
    parser.add_argument(
        "--random_state", type=int, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use_class_weight",
        type=bool,
        help="Use class weighting to handle imbalanced classes",
    )
    parser.add_argument(
        "--use_sample_weight",
        type=bool,
        help="Use sample weighting based on cluster weights",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        help="Number of folds for StratifiedKFold cross-validation",
    )

    args = parser.parse_args()

    default_values = {
        "apple_data": "data/appleAppsSamples-marked.csv",
        "google_data": "data/googleAppsSamples-marked.csv",
        "output_model": "models/cannabis_app_classifier.joblib",
        "output_metrics": "models/cannabis_app_metrics.json",
        "plot_dir": "img",
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_leaf": 8,
        "min_samples_split": 10,
        "random_state": 42,
        "use_class_weight": True,
        "use_sample_weight": True,
        "n_splits": 5,
    }

    if args.config:
        config = load_config(args.config)
        default_values.update({k: v for k, v in config.items() if k in default_values})

    for key, value in default_values.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    return args


def load_data(apple_path, google_path):
    """
    Load and combine the Apple and Google app datasets.

    Args:
        apple_path: Path to the Apple apps CSV file
        google_path: Path to the Google apps CSV file

    Returns:
        Combined DataFrame with all app data
    """
    apple_df = pd.read_csv(apple_path)
    google_df = pd.read_csv(google_path)

    apple_df["source"] = "apple"
    google_df["source"] = "google"

    combined_df = pd.concat([apple_df, google_df], ignore_index=True)

    return combined_df


def analyze_data(df):
    """
    Analyze the dataset and print statistics.

    Args:
        df: DataFrame containing app data

    Returns:
        None
    """
    print("\n=== Dataset Analysis ===")
    print(f"Total samples: {len(df)}")

    # Class distribution
    class_counts = df["cannabis_related"].value_counts()
    print("\nClass distribution:")
    for label, count in class_counts.items():
        percentage = count / len(df) * 100
        print(f"  Class {label}: {count} samples ({percentage:.2f}%)")

    # Distribution by source
    print("\nDistribution by source:")
    source_counts = df["source"].value_counts()
    for source, count in source_counts.items():
        percentage = count / len(df) * 100
        print(f"  {source.capitalize()}: {count} samples ({percentage:.2f}%)")

    # Distribution by cluster
    print("\nDistribution by cluster:")
    cluster_counts = df["cluster_label"].value_counts()
    for cluster, count in cluster_counts.items():
        percentage = count / len(df) * 100
        print(f"  Cluster {cluster}: {count} samples ({percentage:.2f}%)")

    # Cannabis-related apps by cluster
    print("\nCannabis-related apps by cluster:")
    for cluster in df["cluster_label"].unique():
        cluster_df = df[df["cluster_label"] == cluster]
        cannabis_count = cluster_df["cannabis_related"].sum()
        percentage = cannabis_count / len(cluster_df) * 100
        print(
            f"  Cluster {cluster}: {cannabis_count}/{len(cluster_df)} ({percentage:.2f}%)"
        )


def plot_feature_importance(model, feature_names, output_file=None):
    """
    Plot feature importance from the trained model.

    Args:
        model: Trained RandomForestClassifier
        feature_names: List of feature names
        output_file: Optional path to save the plot

    Returns:
        None
    """
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(
        range(len(importances)), [feature_names[i] for i in indices], rotation=90
    )
    plt.xlim([-1, len(importances)])
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Feature importance plot saved to: {output_file}")

    plt.show()


def plot_confusion_matrix(y_true, y_pred, output_file=None):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_file: Optional path to save the plot

    Returns:
        None
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Cannabis", "Cannabis"],
        yticklabels=["Non-Cannabis", "Cannabis"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if output_file:
        plt.savefig(output_file)
        print(f"Confusion matrix plot saved to: {output_file}")

    plt.show()


def plot_precision_recall_curve(y_true, y_score, output_file=None):
    """
    Plot precision-recall curve.

    Args:
        y_true: True labels
        y_score: Predicted probabilities
        output_file: Optional path to save the plot

    Returns:
        None
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker=".", label=f"Random Forest (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)

    if output_file:
        plt.savefig(output_file)
        print(f"Precision-recall curve saved to: {output_file}")

    plt.show()


def train_model(
    df,
    feature_cols,
    n_estimators=100,
    max_depth=None,
    min_samples_split=None,
    min_samples_leaf=None,
    random_state=42,
    use_class_weight=True,
    use_sample_weight=True,
    n_splits=5,
):
    """
    Train a random forest classifier with class weighting and sample weighting
    using StratifiedKFold cross-validation.

    Args:
        df: DataFrame containing app data
        feature_cols: List of feature column names
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        min_samples_split: Minimum samples required to split a node
        random_state: Random seed for reproducibility
        use_class_weight: Whether to use class weighting
        use_sample_weight: Whether to use sample weighting based on cluster_weight
        n_splits: Number of folds for cross-validation

    Returns:
        Trained model, test data, test predictions, and sample weights for test data
    """
    # Prepare the data
    X = df[feature_cols].values
    y = df["cannabis_related"].values

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine class weight parameter
    class_weight_param = "balanced" if use_class_weight else None
    if use_class_weight:
        print("Using balanced class weights")
    else:
        print("Class weighting disabled")

    # Initialize the classifier
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight_param,
        bootstrap=True,
        criterion="gini",
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize arrays to store cross-validation results
    cv_scores = []
    cv_predictions = []
    cv_probabilities = []
    cv_true_labels = []
    cv_test_indices = []
    cv_weighted_scores = []

    print(f"\nPerforming {n_splits}-fold cross-validation...")

    # Check if sample weights are available
    has_sample_weights = use_sample_weight and "cluster_weight" in df.columns
    if has_sample_weights:
        print(
            "Using cluster weights as sample weights for both training and evaluation"
        )
    elif use_sample_weight:
        print("Warning: cluster_weight column not found, using uniform sample weights")
        has_sample_weights = False
    else:
        print("Sample weighting disabled")
        has_sample_weights = False

    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Get sample weights for training
        if has_sample_weights:
            train_weights = df.iloc[train_idx]["cluster_weight"].values
            test_weights = df.iloc[test_idx]["cluster_weight"].values
        else:
            train_weights = np.ones(len(train_idx))
            test_weights = np.ones(len(test_idx))

        # Fit the model with or without sample weights
        if use_sample_weight:
            rf.fit(X_train, y_train, sample_weight=train_weights)
        else:
            rf.fit(X_train, y_train)

        # Make predictions
        fold_pred = rf.predict(X_test)
        fold_proba = rf.predict_proba(X_test)[:, 1]

        # Calculate standard accuracy for this fold
        fold_accuracy = np.mean(fold_pred == y_test)
        cv_scores.append(fold_accuracy)

        # Calculate weighted accuracy if using sample weights
        if has_sample_weights:
            weighted_correct = np.sum((fold_pred == y_test) * test_weights)
            weighted_total = np.sum(test_weights)
            weighted_accuracy = weighted_correct / weighted_total
            cv_weighted_scores.append(weighted_accuracy)
            print(
                f"Fold {fold+1}/{n_splits} - Accuracy: {fold_accuracy:.4f}, Weighted Accuracy: {weighted_accuracy:.4f}"
            )
        else:
            print(f"Fold {fold+1}/{n_splits} - Accuracy: {fold_accuracy:.4f}")

        # Store predictions, true labels, and test indices for later evaluation
        cv_predictions.extend(fold_pred)
        cv_probabilities.extend(fold_proba)
        cv_true_labels.extend(y_test)
        cv_test_indices.extend(test_idx)

    # Print average cross-validation scores
    print(f"\nAverage cross-validation accuracy: {np.mean(cv_scores):.4f}")
    if has_sample_weights:
        print(
            f"Average weighted cross-validation accuracy: {np.mean(cv_weighted_scores):.4f}"
        )

    # Train the final model on the entire dataset
    print("\nTraining final model on entire dataset...")
    if has_sample_weights:
        rf.fit(X_scaled, y, sample_weight=df["cluster_weight"].values)
    else:
        rf.fit(X_scaled, y)

    # Create a model object
    model = {
        "model": rf,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "predict": lambda X: rf.predict(X),
        "predict_proba": lambda X: rf.predict_proba(X)[:, 1],
    }

    # Get sample weights for test data
    test_sample_weights = None
    if has_sample_weights:
        # Reorder the weights to match the order of cv_true_labels
        test_sample_weights = np.array(
            [df.iloc[idx]["cluster_weight"] for idx in cv_test_indices]
        )
    else:
        test_sample_weights = np.ones(len(cv_true_labels))

    # For compatibility with the rest of the code, return the collected predictions from all folds
    return (
        model,
        X_scaled,
        np.array(cv_true_labels),
        np.array(cv_predictions),
        np.array(cv_probabilities),
        test_sample_weights,
    )


def evaluate_model(y_true, y_pred, y_proba, sample_weights=None):
    """
    Evaluate the model and print performance metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        sample_weights: Optional sample weights for weighted evaluation

    Returns:
        Dictionary of evaluation metrics
    """
    print("\n=== Model Evaluation ===")

    # Check if we're using sample weights
    using_weights = sample_weights is not None and not np.all(sample_weights == 1.0)

    # Standard classification report
    print("\nStandard Classification Report:")
    print(
        classification_report(y_true, y_pred, target_names=["Non-Cannabis", "Cannabis"])
    )

    # Weighted classification report if using weights
    if using_weights:
        print("\nWeighted Classification Report:")
        print(
            classification_report(
                y_true,
                y_pred,
                target_names=["Non-Cannabis", "Cannabis"],
                sample_weight=sample_weights,
            )
        )

    # Standard confusion matrix
    print("\nStandard Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Weighted confusion matrix if using weights
    if using_weights:
        print("\nWeighted Confusion Matrix:")
        cm_weighted = confusion_matrix(y_true, y_pred, sample_weight=sample_weights)
        print(cm_weighted)

        # Use the weighted confusion matrix for metrics
        cm_for_metrics = cm_weighted
    else:
        cm_for_metrics = cm

    # ROC AUC score (standard and weighted)
    roc_auc = roc_auc_score(y_true, y_proba)
    print(f"\nStandard ROC AUC Score: {roc_auc:.4f}")

    if using_weights:
        weighted_roc_auc = roc_auc_score(y_true, y_proba, sample_weight=sample_weights)
        print(f"Weighted ROC AUC Score: {weighted_roc_auc:.4f}")
    else:
        weighted_roc_auc = roc_auc

    # Calculate precision, recall, and F1 score from confusion matrix
    tn, fp, fn, tp = cm_for_metrics.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Calculate weighted accuracy directly if using weights
    if using_weights:
        weighted_accuracy = np.sum((y_pred == y_true) * sample_weights) / np.sum(
            sample_weights
        )
        print(f"\nWeighted Accuracy: {weighted_accuracy:.4f}")
    else:
        weighted_accuracy = accuracy

    # Return evaluation metrics
    metrics = {
        "accuracy": accuracy,
        "weighted_accuracy": weighted_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "weighted_roc_auc": weighted_roc_auc,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    return metrics


def save_model(model, output_path):
    """
    Save the trained model to a file.

    Args:
        model: Trained model object
        output_path: Path to save the model

    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the model components
    model_data = {
        "model": model["model"],
        "scaler": model["scaler"],
        "feature_cols": model["feature_cols"],
    }

    joblib.dump(model_data, output_path)
    print(f"Model saved to: {output_path}")


def load_saved_model(model_path):
    """
    Load a saved model from a file.

    Args:
        model_path: Path to the saved model

    Returns:
        Loaded model object
    """
    model_data = joblib.load(model_path)

    # Create a model object with prediction functions
    loaded_model = {
        "model": model_data["model"],
        "scaler": model_data["scaler"],
        "feature_cols": model_data["feature_cols"],
        "predict": lambda X: model_data["model"].predict(X),
        "predict_proba": lambda X: model_data["model"].predict_proba(X)[:, 1],
    }

    return loaded_model


def predict_cannabis_app(model, app_data):
    """
    Predict whether an app is cannabis-related.

    Args:
        model: Trained model object
        app_data: Dictionary or DataFrame containing app features

    Returns:
        Prediction (1 for cannabis-related, 0 for not) and probability
    """
    # Extract features
    if isinstance(app_data, dict):
        features = np.array([[app_data.get(col, 0) for col in model["feature_cols"]]])
    else:
        features = app_data[model["feature_cols"]].values

    # Scale features
    scaled_features = model["scaler"].transform(features)

    # Make prediction
    prediction = model["predict"](scaled_features)
    probability = model["predict_proba"](scaled_features)

    return prediction, probability


def main():
    args = parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)

    print(f"Loading data from {args.apple_data} and {args.google_data}...")
    df = load_data(args.apple_data, args.google_data)

    analyze_data(df)

    feature_cols = [
        "facebook_bart-large-mnli",
        "roberta-large-mnli",
        "distilbert-base-uncased-finetuned-sst-2-english",
        "valhalla_distilbart-mnli-12-9",
    ]

    # Train the model
    print("\n=== Training Model ===")
    print(f"Using features: {feature_cols}")
    print(
        f"Random Forest parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}"
    )
    print(f"Class weighting: {'enabled' if args.use_class_weight else 'disabled'}")
    print(f"Sample weighting: {'enabled' if args.use_sample_weight else 'disabled'}")
    print(f"Using StratifiedKFold cross-validation with {args.n_splits} folds")

    model, X_test, y_test, y_pred, y_proba, sample_weights = train_model(
        df,
        feature_cols,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        use_class_weight=args.use_class_weight,
        use_sample_weight=args.use_sample_weight,
        n_splits=args.n_splits,
    )

    # Evaluate the model
    metrics = evaluate_model(y_test, y_pred, y_proba, sample_weights)

    # Plot feature importance
    plot_feature_importance(
        model["model"],
        feature_cols,
        output_file=os.path.join(args.plot_dir, "feature_importance.png"),
    )

    # Plot confusion matrix
    plot_confusion_matrix(
        y_test, y_pred, output_file=os.path.join(args.plot_dir, "confusion_matrix.png")
    )

    # Plot precision-recall curve
    plot_precision_recall_curve(
        y_test,
        y_proba,
        output_file=os.path.join(args.plot_dir, "precision_recall_curve.png"),
    )

    # Save the model
    save_model(model, args.output_model)

    # Save the metrics
    os.makedirs(os.path.dirname(args.output_metrics), exist_ok=True)
    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation metrics saved to: {args.output_metrics}")


if __name__ == "__main__":
    main()
