import os
import joblib
import torch
import torch.nn as nn


class AppDataset(torch.utils.data.Dataset):
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


def save_model(model, output_path):
    """
    Save the trained model.
    For PyTorch models, save the state_dict along with metadata.
    For Random Forest, save via joblib.
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
        if os.path.exists(output_path) and os.path.isdir(output_path):
            output_path = os.path.join(output_path, "rf_model.joblib")
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
    For a PyTorch neural network, model_path is expected to be a directory containing "pytorch_model.pt" and "model_meta.joblib".
    For Random Forest models, model_path is assumed to be a joblib file.
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
        model_nn.eval()  # Set model to evaluation mode so BatchNorm uses running stats
        loaded_model = {
            "model": model_nn,
            "scaler": meta_data["scaler"],
            "feature_cols": meta_data["feature_cols"],
            "predict": lambda X: _predict_with_eval(model_nn, X),
            "predict_proba": lambda X: _predict_proba_with_eval(model_nn, X),
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


def _predict_with_eval(model, X):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    preds = torch.sigmoid(model(X_tensor)).detach().numpy().ravel() >= 0.5
    return preds.astype(int)


def _predict_proba_with_eval(model, X):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    return torch.sigmoid(model(X_tensor)).detach().numpy().ravel()
