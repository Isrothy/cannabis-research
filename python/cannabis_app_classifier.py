import os
import configargparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from sklearn.preprocessing import StandardScaler
import joblib
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


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
        "--plot_dir",
        type=str,
        default="img",
        help="Directory to save plots",
    )
    # Random Forest parameters (used if model_type == 'rf')
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
    # Common parameters
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
        help="Use class weighting for imbalanced classes",
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
        help="Disable sample weighting based on cluster weights",
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
    # New arguments for choosing model type and NN training
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
        default=25,
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


# -----------------------------
# Data Loading & Analysis
# -----------------------------
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


# -----------------------------
# Plotting Functions
# -----------------------------
def plot_feature_importance(
    model, feature_names, output_file=None, sample_weights=None
):
    # Only applicable for Random Forest
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
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
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


# -----------------------------
# Dataset and Model Definitions
# -----------------------------
class AppDataset(Dataset):
    def __init__(self, X, y, weights=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.weights = (
            torch.tensor(weights, dtype=torch.float32).view(-1, 1)
            if weights is not None
            else torch.ones_like(self.y)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx]


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Helper Functions for Training
# -----------------------------
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


def validate_model(model, loader, loss_fn, device):
    model.eval()
    losses, all_preds, all_true = [], [], []
    with torch.no_grad():
        for X, y, weights in loader:
            X, y, weights = X.to(device), y.to(device), weights.to(device)
            logits = model(X)
            loss = (loss_fn(logits, y) * weights).mean()
            losses.append(loss.item())
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            preds = (probs >= 0.5).astype(int)
            all_preds.extend(preds)
            all_true.extend(y.cpu().numpy().ravel())
    return np.mean(losses), all_preds, all_true


def train_model_with_early_stopping(
    model, train_loader, val_loader, optimizer, loss_fn, device, epochs, patience
):
    best_val_loss = np.inf
    epochs_no_improve = 0
    best_state = None
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, _, _ = validate_model(model, val_loader, loss_fn, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break
    model.load_state_dict(best_state)
    return model


# -----------------------------
# PyTorch Neural Network Training (Simplified with L2 Regularization)
# -----------------------------
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
    l2_reg=0.001,  # L2 regularization parameter (weight_decay)
):
    # Prepare data
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
        model = train_model_with_early_stopping(
            model,
            train_loader,
            test_loader,
            optimizer,
            loss_fn,
            device,
            epochs,
            patience,
        )

        # Evaluate fold
        _, preds, _ = validate_model(model, test_loader, loss_fn, device)
        probs = []
        model.eval()
        with torch.no_grad():
            for X_batch, _, _ in test_loader:
                X_batch = X_batch.to(device)
                batch_probs = torch.sigmoid(model(X_batch)).cpu().numpy().ravel()
                probs.extend(batch_probs)
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
    final_model = train_model_with_early_stopping(
        final_model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        epochs,
        patience,
    )

    # Define prediction functions for the final model
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

    # Return actual sample weights if available, otherwise uniform weights.
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


# -----------------------------
# Random Forest Training (unchanged)
# -----------------------------
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
        verbose=1,
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
    # print(f"use_sample_weights: {use_sample_weight}")
    # print(f"has_sample_weights: {has_sample_weights}")

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


# -----------------------------
# Evaluation, Saving, and Prediction
# -----------------------------
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


def save_model(model, output_path):
    """
    Save the trained model.
    For neural networks, the PyTorch model is saved along with metadata.
    For Random Forest, the model is saved via joblib.
    """
    if isinstance(model["model"], torch.nn.Module):
        if os.path.exists(output_path) and os.path.isfile(output_path):
            os.remove(output_path)
        os.makedirs(output_path, exist_ok=True)
        torch.save(
            model["model"].state_dict(), os.path.join(output_path, "pytorch_model.pt")
        )
        meta_data = {"scaler": model["scaler"], "feature_cols": model["feature_cols"]}
        joblib.dump(meta_data, os.path.join(output_path, "model_meta.joblib"))
        print(f"Neural network model saved to directory: {output_path}")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model_data = {
            "model": model["model"],
            "scaler": model["scaler"],
            "feature_cols": model["feature_cols"],
        }
        joblib.dump(model_data, output_path)
        print(f"Model saved to: {output_path}")


def load_saved_model(model_path):
    """
    Load a saved model.
    """
    if os.path.exists(os.path.join(model_path, "pytorch_model.pt")):
        meta_data = joblib.load(os.path.join(model_path, "model_meta.joblib"))
        input_dim = len(meta_data["feature_cols"])
        model_nn = SimpleNN(input_dim=input_dim)
        model_nn.load_state_dict(
            torch.load(
                os.path.join(model_path, "pytorch_model.pt"),
                map_location=torch.device("cpu"),
            )
        )
        loaded_model = {
            "model": model_nn,
            "scaler": meta_data["scaler"],
            "feature_cols": meta_data["feature_cols"],
            "predict": lambda X: (
                torch.sigmoid(model_nn(torch.tensor(X, dtype=torch.float32)))
                .detach()
                .numpy()
                .ravel()
                >= 0.5
            ).astype(int),
            "predict_proba": lambda X: torch.sigmoid(
                model_nn(torch.tensor(X, dtype=torch.float32))
            )
            .detach()
            .numpy()
            .ravel(),
        }
        return loaded_model
    else:
        model_data = joblib.load(model_path)
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
    """
    if isinstance(app_data, dict):
        features = np.array([[app_data.get(col, 0) for col in model["feature_cols"]]])
    else:
        features = app_data[model["feature_cols"]].values
    scaled_features = model["scaler"].transform(features)
    prediction = model["predict"](scaled_features)
    probability = model["predict_proba"](scaled_features)
    return prediction, probability


# -----------------------------
# Main Function
# -----------------------------
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

    # Plot confusion matrix
    plot_confusion_matrix(
        y_test,
        y_pred,
        output_file=os.path.join(args.plot_dir, "confusion_matrix.png"),
        sample_weights=sample_weights,
    )

    # Plot precision-recall curve
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
