import os
import configargparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from pytorch_optimizer import SoftF1Loss
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from sklearn.preprocessing import StandardScaler
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model_utils import (
    AppDataset,
    SimpleNN,
    save_model,
)


def parse_args():
    parser = configargparse.ArgumentParser(
        description="Train a model to identify cannabis-related apps.",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument(
        "--config", is_config_file=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--apple_data",
        type=str,
        default="data/appleAppsSamples-marked.csv",
        help="Path to the Apple apps CSV file",
    )
    parser.add_argument(
        "--google_data",
        type=str,
        default="data/googleAppsSamples-marked.csv",
        help="Path to the Google apps CSV file",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="models/cannabis_app_model",
        help="Path to save the trained model (for NN, a directory will be created)",
    )
    parser.add_argument(
        "--output_metrics",
        type=str,
        default="models/cannabis_app_metrics.json",
        help="Path to save the evaluation metrics",
    )
    parser.add_argument(
        "--plot_dir", type=str, default="img", help="Directory to save plots"
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of trees in the random forest",
    )
    parser.add_argument(
        "--max_depth", type=int, default=4, help="Maximum depth of the trees"
    )
    parser.add_argument(
        "--min_samples_split", type=int, default=10, help="Minimum samples to split"
    )
    parser.add_argument(
        "--min_samples_leaf", type=int, default=5, help="Minimum samples in a leaf"
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use_class_weight",
        action="store_true",
        help="Use class weighting for imbalanced classes",
    )
    parser.add_argument(
        "--no_use_class_weight",
        dest="use_class_weight",
        action="store_false",
        help="Disable class weighting",
    )
    parser.add_argument(
        "--use_sample_weight",
        action="store_true",
        help="Use sample weighting based on cluster weights",
    )
    parser.add_argument(
        "--no_use_sample_weight",
        dest="use_sample_weight",
        action="store_false",
        help="Disable sample weighting",
    )
    parser.add_argument(
        "--feature_columns",
        type=str,
        nargs="+",
        help="List of features to train the model",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of folds for StratifiedKFold cross-validation",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="nn",
        choices=["rf", "nn"],
        help="Type of model to train: 'rf' for Random Forest, 'nn' for Neural Network (PyTorch)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of epochs for neural network training (if using NN)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for neural network training (if using NN)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for NN training (if using NN)",
    )
    return parser.parse_args()


def load_data(apple_path, google_path):
    apple_df = pd.read_csv(apple_path)
    google_df = pd.read_csv(google_path)
    apple_df["source"] = "apple"
    google_df["source"] = "google"
    combined_df = pd.concat([apple_df, google_df], ignore_index=True)
    return combined_df


def analyze_data(df):
    print("\n=== Dataset Analysis ===")
    print(f"Total samples: {len(df)}")
    class_counts = df["cannabis_related"].value_counts()
    print("\nClass distribution:")
    for label, count in class_counts.items():
        percentage = count / len(df) * 100
        print(f"  Class {label}: {count} samples ({percentage:.2f}%)")
    print("\nDistribution by source:")
    source_counts = df["source"].value_counts()
    for source, count in source_counts.items():
        percentage = count / len(df) * 100
        print(f"  {source.capitalize()}: {count} samples ({percentage:.2f}%)")
    print("\nDistribution by cluster:")
    cluster_counts = df["clusterLabel"].value_counts()
    for cluster, count in cluster_counts.items():
        percentage = count / len(df) * 100
        print(f"  Cluster {cluster}: {count} samples ({percentage:.2f}%)")
    print("\nCannabis-related apps by cluster:")
    for cluster in df["clusterLabel"].unique():
        cluster_df = df[df["clusterLabel"] == cluster]
        cannabis_count = cluster_df["cannabis_related"].sum()
        percentage = cannabis_count / len(cluster_df) * 100
        print(
            f"  Cluster {cluster}: {cannabis_count}/{len(cluster_df)} ({percentage:.2f}%)"
        )


def plot_feature_importance(
    model, feature_names, output_file=None, sample_weights=None
):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    if sample_weights is not None and not np.all(sample_weights == 1.0):
        plt.title("Weighted Feature Importances")
    else:
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


def plot_confusion_matrix(y_true, y_pred, output_file=None, sample_weights=None):
    if sample_weights is not None and not np.all(sample_weights == 1.0):
        cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weights)
        title = "Weighted Confusion Matrix"
        fmt = ".1f"
    else:
        cm = confusion_matrix(y_true, y_pred)
        title = "Confusion Matrix"
        fmt = "d"
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=["Non-Cannabis", "Cannabis"],
        yticklabels=["Non-Cannabis", "Cannabis"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    if output_file:
        plt.savefig(output_file)
        print(f"{title} saved to: {output_file}")
    plt.show()


def plot_precision_recall_curve(y_true, y_score, output_file=None, sample_weights=None):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker=".", label=f"Standard (AUC = {pr_auc:.3f})")
    if sample_weights is not None and not np.all(sample_weights == 1.0):
        precision_w, recall_w, _ = precision_recall_curve(
            y_true, y_score, sample_weight=sample_weights
        )
        pr_auc_w = auc(recall_w, precision_w)
        plt.plot(
            recall_w,
            precision_w,
            marker=".",
            linestyle="--",
            label=f"Weighted (AUC = {pr_auc_w:.3f})",
        )
        title = "Weighted Precision-Recall Curve"
    else:
        title = "Precision-Recall Curve"
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if output_file:
        plt.savefig(output_file)
        print(f"{title} saved to: {output_file}")
    plt.show()


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    losses = []
    for X, y, weights in loader:
        X, y, weights = X.to(device), y.to(device), weights.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = (loss_fn(logits, y) * weights).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def validate(loader, model, device):
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for X_batch, _, _ in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            preds = (probs >= 0.5).astype(int)
            all_preds.extend(preds)
            all_probs.extend(probs)
    return all_preds, all_probs


def train_model(model, loader, optimizer, loss_fn, device, epochs):
    model.train()
    for epoch in range(epochs):
        losses = []
        for X, y, weights in loader:
            X, y, weights = X.to(device), y.to(device), weights.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss = (loss * weights).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")
    return model


def train_nn_model(
    df,
    feature_cols,
    random_state=42,
    use_sample_weight=True,
    n_splits=5,
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    patience=5,
    l2_reg=0.001,
):
    X = df[feature_cols].values
    y = df["cannabis_related"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_predictions, cv_probabilities, cv_true_labels, cv_test_indices = [], [], [], []
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device = {device}")
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    # loss_fn = SoftF1Loss()

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if use_sample_weight and "clusterWeight" in df.columns:
            train_weights = df.iloc[train_idx]["clusterWeight"].values
            test_weights = df.iloc[test_idx]["clusterWeight"].values
        else:
            train_weights = np.ones(len(train_idx))
            test_weights = np.ones(len(test_idx))

        train_ds = AppDataset(X_train, y_train, train_weights)
        test_ds = AppDataset(X_test, y_test, test_weights)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = SimpleNN(input_dim=X_train.shape[1]).to(device)
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=l2_reg
        )
        model = train_model(model, train_loader, optimizer, loss_fn, device, epochs)

        preds, probs = validate(test_loader, model, device)
        cv_predictions.extend(preds)
        cv_probabilities.extend(probs)
        cv_true_labels.extend(y_test)
        cv_test_indices.extend(test_idx)
        fold_acc = np.mean(np.array(preds) == y_test)
        print(f"Fold {fold+1} - Accuracy: {fold_acc:.4f}")

    full_ds = AppDataset(
        X_scaled,
        y,
        (
            df["clusterWeight"].values
            if use_sample_weight and "clusterWeight" in df.columns
            else None
        ),
    )
    val_size = max(1, int(0.1 * len(full_ds)))
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [len(full_ds) - val_size, val_size]
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    final_model = SimpleNN(input_dim=X_scaled.shape[1]).to(device)
    optimizer = optim.Adam(
        final_model.parameters(), lr=learning_rate, weight_decay=l2_reg
    )
    final_model = train_model(
        final_model, val_loader, optimizer, loss_fn, device, epochs
    )

    def predict_fn(X_input):
        final_model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
            return (
                torch.sigmoid(final_model(X_tensor)).cpu().numpy().ravel() >= 0.5
            ).astype(int)

    def predict_proba_fn(X_input):
        final_model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
            return torch.sigmoid(final_model(X_tensor)).cpu().numpy().ravel()

    model_obj = {
        "model": final_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "predict": predict_fn,
        "predict_proba": predict_proba_fn,
    }

    if use_sample_weight and "clusterWeight" in df.columns:
        test_sample_weights = np.array(
            [df.iloc[idx]["clusterWeight"] for idx in cv_test_indices]
        )
    else:
        test_sample_weights = np.ones(len(cv_true_labels))

    return (
        model_obj,
        X_scaled,
        np.array(cv_true_labels),
        np.array(cv_predictions),
        np.array(cv_probabilities),
        test_sample_weights,
    )


def train_rf_model(
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
    X = df[feature_cols].values
    y = df["cannabis_related"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    class_weight_param = "balanced" if use_class_weight else None
    if use_class_weight:
        print("Using balanced class weights")
    else:
        print("Class weighting disabled")
    from sklearn.ensemble import RandomForestClassifier

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
        verbose=0,
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = []
    cv_predictions = []
    cv_probabilities = []
    cv_true_labels = []
    cv_test_indices = []
    print(f"\nPerforming {n_splits}-fold cross-validation for RF...")
    has_sample_weights = use_sample_weight and "clusterWeight" in df.columns
    if has_sample_weights:
        print(
            "Using cluster weights as sample weights for both training and evaluation"
        )
    elif use_sample_weight:
        print("Warning: clusterWeight column not found, using uniform sample weights")
        has_sample_weights = False
    else:
        print("Sample weighting disabled")
        has_sample_weights = False

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if has_sample_weights:
            train_weights = df.iloc[train_idx]["clusterWeight"].values
            test_weights = df.iloc[test_idx]["clusterWeight"].values
        else:
            train_weights = np.ones(len(train_idx))
            test_weights = np.ones(len(test_idx))
        if use_sample_weight:
            rf.fit(X_train, y_train, sample_weight=train_weights)
        else:
            rf.fit(X_train, y_train)
        fold_pred = rf.predict(X_test)
        fold_proba = rf.predict_proba(X_test)[:, 1]
        fold_accuracy = np.mean(fold_pred == y_test)
        cv_scores.append(fold_accuracy)
        if has_sample_weights:
            weighted_correct = np.sum((fold_pred == y_test) * test_weights)
            weighted_total = np.sum(test_weights)
            weighted_accuracy = weighted_correct / weighted_total
            print(
                f"Fold {fold+1}/{n_splits} - Accuracy: {fold_accuracy:.4f}, Weighted Accuracy: {weighted_accuracy:.4f}"
            )
        else:
            print(f"Fold {fold+1}/{n_splits} - Accuracy: {fold_accuracy:.4f}")
        cv_predictions.extend(fold_pred)
        cv_probabilities.extend(fold_proba)
        cv_true_labels.extend(y_test)
        cv_test_indices.extend(test_idx)
    print(f"\nAverage cross-validation accuracy: {np.mean(cv_scores):.4f}")
    print("\nTraining final Random Forest model on entire dataset...")
    if has_sample_weights:
        rf.fit(X_scaled, y, sample_weight=df["clusterWeight"].values)
    else:
        rf.fit(X_scaled, y)
    model_obj = {
        "model": rf,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "predict": lambda X: rf.predict(X),
        "predict_proba": lambda X: rf.predict_proba(X)[:, 1],
    }
    if has_sample_weights:
        test_sample_weights = np.array(
            [df.iloc[idx]["clusterWeight"] for idx in cv_test_indices]
        )
    else:
        test_sample_weights = None
    return (
        model_obj,
        X_scaled,
        np.array(cv_true_labels),
        np.array(cv_predictions),
        np.array(cv_probabilities),
        test_sample_weights,
    )


def evaluate_model(y_true, y_pred, y_proba, sample_weights=None):
    print("\n=== Model Evaluation ===")
    using_weights = sample_weights is not None and not np.all(sample_weights == 1.0)
    print("\nStandard Classification Report:")
    print(
        classification_report(y_true, y_pred, target_names=["Non-Cannabis", "Cannabis"])
    )
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
    print("\nStandard Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    if using_weights:
        print("\nWeighted Confusion Matrix:")
        cm_weighted = confusion_matrix(y_true, y_pred, sample_weight=sample_weights)
        print(cm_weighted)
        cm_for_metrics = cm_weighted
    else:
        cm_for_metrics = cm
    roc_auc = roc_auc_score(y_true, y_proba)
    print(f"\nStandard ROC AUC Score: {roc_auc:.4f}")
    if using_weights:
        weighted_roc_auc = roc_auc_score(y_true, y_proba, sample_weight=sample_weights)
        print(f"Weighted ROC AUC Score: {weighted_roc_auc:.4f}")
    else:
        weighted_roc_auc = roc_auc
    tn, fp, fn, tp = cm_for_metrics.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if using_weights:
        weighted_accuracy = np.sum((y_pred == y_true) * sample_weights) / np.sum(
            sample_weights
        )
        print(f"\nWeighted Accuracy: {weighted_accuracy:.4f}")
    else:
        weighted_accuracy = accuracy
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


def main():
    args = parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)

    print(f"Loading data from {args.apple_data} and {args.google_data}...")
    df = load_data(args.apple_data, args.google_data)
    analyze_data(df)

    feature_cols = args.feature_columns
    print("\n=== Training Model ===")
    print(f"Using features: {feature_cols}")
    print(f"Model type: {args.model_type}")

    if args.model_type == "rf":
        print(
            f"Random Forest parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}"
        )
        model, X_test, y_test, y_pred, y_proba, sample_weights = train_rf_model(
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
        plot_feature_importance(
            model["model"],
            feature_cols,
            output_file=os.path.join(args.plot_dir, "feature_importance.png"),
            sample_weights=sample_weights if args.use_sample_weight else None,
        )
    elif args.model_type == "nn":
        print(
            f"Neural Network parameters: epochs={args.epochs}, batch_size={args.batch_size}, learning_rate={args.learning_rate}"
        )
        model, X_test, y_test, y_pred, y_proba, sample_weights = train_nn_model(
            df,
            feature_cols,
            random_state=args.random_state,
            use_sample_weight=args.use_sample_weight,
            n_splits=args.n_splits,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    metrics = evaluate_model(y_test, y_pred, y_proba, sample_weights)

    plot_confusion_matrix(
        y_test,
        y_pred,
        output_file=os.path.join(args.plot_dir, "confusion_matrix.png"),
        sample_weights=sample_weights,
    )

    plot_precision_recall_curve(
        y_test,
        y_proba,
        output_file=os.path.join(args.plot_dir, "precision_recall_curve.png"),
        sample_weights=sample_weights,
    )

    save_model(model, args.output_model)

    os.makedirs(os.path.dirname(args.output_metrics), exist_ok=True)
    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation metrics saved to: {args.output_metrics}")


if __name__ == "__main__":
    main()
