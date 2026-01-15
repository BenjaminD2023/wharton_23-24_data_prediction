import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset


# Simple CNN for tabular data treated as a 1D image
class TabularCnn(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        return self.classifier(self.features(x))


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


def load_data(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    target_cols = ["HomeScore", "AwayScore"]
    feature_cols = [
        col
        for col in train_df.columns
        if col not in target_cols and col not in {"game_id", "n"}
    ]

    # Encode teams as integers
    team_cols = ["HomeTeam", "AwayTeam"]
    encoders = {
        col: LabelEncoder().fit(pd.concat([train_df[col], test_df[col]]))
        for col in team_cols
    }
    for col in team_cols:
        train_df[col] = encoders[col].transform(train_df[col])
        test_df[col] = encoders[col].transform(test_df[col])

    # Fill missing numeric values
    train_df[feature_cols] = train_df[feature_cols].fillna(
        train_df[feature_cols].median()
    )
    test_df[feature_cols] = test_df[feature_cols].fillna(
        train_df[feature_cols].median()
    )

    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df[target_cols].values.astype(np.float32)

    X_test = test_df[feature_cols].values.astype(np.float32)
    return X, y, X_test, feature_cols, target_cols


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_size: float = 0.2,
    seed: int = 42,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    y_encoded = encode_targets(y)
    num_classes = int(y_encoded.max() + 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=val_size, random_state=seed, stratify=y_encoded
    )

    train_loader = DataLoader(
        TabularDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=batch_size)

    model = TabularCnn(in_channels=1, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
                correct += (preds.argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total if total else 0
        print(
            f"Epoch {epoch:03d} - train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.3f}"
        )

    return model


def encode_targets(y: np.ndarray) -> np.ndarray:
    # Example: classify match outcome (home win/draw/away win)
    outcomes = np.sign(y[:, 0] - y[:, 1])  # -1 away win, 0 draw, 1 home win
    mapping = {-1: 0, 0: 1, 1: 2}
    return np.vectorize(mapping.get)(outcomes)


def predict(model: nn.Module, X_test: np.ndarray, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(TabularDataset(X_test), batch_size=64)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def save_submission(
    test_df: pd.DataFrame, preds: np.ndarray, target_cols: list[str], output: Path
):
    outcome_map = {0: "AwayWin", 1: "Draw", 2: "HomeWin"}
    labels = np.vectorize(outcome_map.get)(preds)
    submission = pd.DataFrame({"game_id": test_df["game_id"], "Outcome": labels})
    submission.to_csv(output, index=False)
    print(f"Saved predictions to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a simple CNN for tabular match prediction"
    )
    parser.add_argument(
        "--train", type=Path, default=Path("wharton-data-science-2023-2024/train.csv")
    )
    parser.add_argument(
        "--test", type=Path, default=Path("wharton-data-science-2023-2024/test.csv")
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    train_df, test_df = load_data(args.train, args.test)
    X, y, X_test, feature_cols, target_cols = preprocess(train_df, test_df)
    model = train_model(
        X,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_size=args.val_size,
        seed=args.seed,
    )
    preds = predict(model, X_test)
    save_submission(test_df, preds, target_cols, args.output)


if __name__ == "__main__":
    main()
