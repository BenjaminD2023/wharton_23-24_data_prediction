import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def compute_team_stats(train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute historical statistics for each team from training data."""
    team_stats = defaultdict(lambda: {
        'games_home': 0, 'games_away': 0,
        'wins_home': 0, 'wins_away': 0,
        'draws_home': 0, 'draws_away': 0,
        'losses_home': 0, 'losses_away': 0,
        'goals_scored_home': 0, 'goals_scored_away': 0,
        'goals_conceded_home': 0, 'goals_conceded_away': 0,
        'xG_home': 0, 'xG_away': 0,
        'xG_against_home': 0, 'xG_against_away': 0,
        'shots_home': 0, 'shots_away': 0,
        'shots_against_home': 0, 'shots_against_away': 0,
        'corners_home': 0, 'corners_away': 0,
        'ToP_home': 0, 'ToP_away': 0,
    })

    for _, row in train_df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_score = row['HomeScore']
        away_score = row['AwayScore']

        # Home team stats
        team_stats[home_team]['games_home'] += 1
        team_stats[home_team]['goals_scored_home'] += home_score
        team_stats[home_team]['goals_conceded_home'] += away_score

        if pd.notna(row.get('Home_xG', np.nan)):
            team_stats[home_team]['xG_home'] += row['Home_xG']
            team_stats[home_team]['xG_against_home'] += row['Away_xG']
        if pd.notna(row.get('Home_shots', np.nan)):
            team_stats[home_team]['shots_home'] += row['Home_shots']
            team_stats[home_team]['shots_against_home'] += row['Away_shots']
        if pd.notna(row.get('Home_corner', np.nan)):
            team_stats[home_team]['corners_home'] += row['Home_corner']
        if pd.notna(row.get('Home_ToP', np.nan)):
            team_stats[home_team]['ToP_home'] += row['Home_ToP']

        # Away team stats
        team_stats[away_team]['games_away'] += 1
        team_stats[away_team]['goals_scored_away'] += away_score
        team_stats[away_team]['goals_conceded_away'] += home_score

        if pd.notna(row.get('Away_xG', np.nan)):
            team_stats[away_team]['xG_away'] += row['Away_xG']
            team_stats[away_team]['xG_against_away'] += row['Home_xG']
        if pd.notna(row.get('Away_shots', np.nan)):
            team_stats[away_team]['shots_away'] += row['Away_shots']
            team_stats[away_team]['shots_against_away'] += row['Home_shots']
        if pd.notna(row.get('Away_corner', np.nan)):
            team_stats[away_team]['corners_away'] += row['Away_corner']
        if pd.notna(row.get('Away_ToP', np.nan)):
            team_stats[away_team]['ToP_away'] += row['Away_ToP']

        # Win/Draw/Loss
        if home_score > away_score:
            team_stats[home_team]['wins_home'] += 1
            team_stats[away_team]['losses_away'] += 1
        elif home_score < away_score:
            team_stats[home_team]['losses_home'] += 1
            team_stats[away_team]['wins_away'] += 1
        else:
            team_stats[home_team]['draws_home'] += 1
            team_stats[away_team]['draws_away'] += 1

    return dict(team_stats)


def compute_elo_ratings(train_df: pd.DataFrame, k: float = 32, home_advantage: float = 100) -> Dict[str, float]:
    """Compute Elo ratings for each team based on match results."""
    elo = defaultdict(lambda: 1500)

    for _, row in train_df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_score = row['HomeScore']
        away_score = row['AwayScore']

        # Expected scores
        home_elo = elo[home_team] + home_advantage
        away_elo = elo[away_team]

        exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        exp_away = 1 - exp_home

        # Actual scores
        if home_score > away_score:
            actual_home, actual_away = 1, 0
        elif home_score < away_score:
            actual_home, actual_away = 0, 1
        else:
            actual_home, actual_away = 0.5, 0.5

        # Update Elo
        elo[home_team] += k * (actual_home - exp_home)
        elo[away_team] += k * (actual_away - exp_away)

    return dict(elo)


def compute_head_to_head(train_df: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Compute head-to-head statistics for each pair of teams."""
    h2h = defaultdict(lambda: {'home_wins': 0, 'away_wins': 0, 'draws': 0, 'games': 0,
                                'home_goals': 0, 'away_goals': 0})

    for _, row in train_df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_score = row['HomeScore']
        away_score = row['AwayScore']

        key = (home_team, away_team)
        h2h[key]['games'] += 1
        h2h[key]['home_goals'] += home_score
        h2h[key]['away_goals'] += away_score

        if home_score > away_score:
            h2h[key]['home_wins'] += 1
        elif home_score < away_score:
            h2h[key]['away_wins'] += 1
        else:
            h2h[key]['draws'] += 1

        # Also track reverse (away team hosting)
        rev_key = (away_team, home_team)
        h2h[rev_key]['games'] += 1
        h2h[rev_key]['away_goals'] += home_score
        h2h[rev_key]['home_goals'] += away_score

        if away_score > home_score:
            h2h[rev_key]['home_wins'] += 1
        elif away_score < home_score:
            h2h[rev_key]['away_wins'] += 1
        else:
            h2h[rev_key]['draws'] += 1

    return dict(h2h)


def compute_recent_form(train_df: pd.DataFrame, n_games: int = 5) -> Dict[str, List[int]]:
    """Compute recent form (last n results) for each team."""
    form = defaultdict(list)

    for _, row in train_df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_score = row['HomeScore']
        away_score = row['AwayScore']

        # 3 = win, 1 = draw, 0 = loss
        if home_score > away_score:
            form[home_team].append(3)
            form[away_team].append(0)
        elif home_score < away_score:
            form[home_team].append(0)
            form[away_team].append(3)
        else:
            form[home_team].append(1)
            form[away_team].append(1)

    # Keep only last n_games
    for team in form:
        form[team] = form[team][-n_games:]

    return dict(form)


def build_engineered_features(
    df: pd.DataFrame,
    team_stats: Dict,
    elo_ratings: Dict,
    h2h: Dict,
    recent_form: Dict,
    is_train: bool = True
) -> np.ndarray:
    """Build feature matrix from engineered features."""
    features = []

    for _, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        feat = []

        # Elo ratings
        home_elo = elo_ratings.get(home_team, 1500)
        away_elo = elo_ratings.get(away_team, 1500)
        feat.extend([home_elo, away_elo, home_elo - away_elo])

        # Team stats - Home team
        hs = team_stats.get(home_team, {})
        home_games = hs.get('games_home', 0) + hs.get('games_away', 0)
        if home_games > 0:
            feat.append((hs.get('wins_home', 0) + hs.get('wins_away', 0)) / home_games)
            feat.append((hs.get('draws_home', 0) + hs.get('draws_away', 0)) / home_games)
            feat.append((hs.get('goals_scored_home', 0) + hs.get('goals_scored_away', 0)) / home_games)
            feat.append((hs.get('goals_conceded_home', 0) + hs.get('goals_conceded_away', 0)) / home_games)
        else:
            feat.extend([0.33, 0.33, 1.5, 1.5])

        # Home-specific performance
        if hs.get('games_home', 0) > 0:
            feat.append(hs.get('wins_home', 0) / hs.get('games_home', 1))
            feat.append((hs.get('xG_home', 0) / hs.get('games_home', 1)))
            feat.append((hs.get('shots_home', 0) / hs.get('games_home', 1)))
        else:
            feat.extend([0.5, 1.5, 12])

        # Team stats - Away team
        aws = team_stats.get(away_team, {})
        away_games = aws.get('games_home', 0) + aws.get('games_away', 0)
        if away_games > 0:
            feat.append((aws.get('wins_home', 0) + aws.get('wins_away', 0)) / away_games)
            feat.append((aws.get('draws_home', 0) + aws.get('draws_away', 0)) / away_games)
            feat.append((aws.get('goals_scored_home', 0) + aws.get('goals_scored_away', 0)) / away_games)
            feat.append((aws.get('goals_conceded_home', 0) + aws.get('goals_conceded_away', 0)) / away_games)
        else:
            feat.extend([0.33, 0.33, 1.5, 1.5])

        # Away-specific performance
        if aws.get('games_away', 0) > 0:
            feat.append(aws.get('wins_away', 0) / aws.get('games_away', 1))
            feat.append((aws.get('xG_away', 0) / aws.get('games_away', 1)))
            feat.append((aws.get('shots_away', 0) / aws.get('games_away', 1)))
        else:
            feat.extend([0.33, 1.2, 10])

        # Head-to-head
        h2h_key = (home_team, away_team)
        h2h_stats = h2h.get(h2h_key, {'home_wins': 0, 'away_wins': 0, 'draws': 0, 'games': 0})
        if h2h_stats['games'] > 0:
            feat.append(h2h_stats['home_wins'] / h2h_stats['games'])
            feat.append(h2h_stats['draws'] / h2h_stats['games'])
            feat.append(h2h_stats['away_wins'] / h2h_stats['games'])
            feat.append(h2h_stats['games'])
        else:
            feat.extend([0.45, 0.25, 0.30, 0])  # Default slight home advantage

        # Recent form
        home_form = recent_form.get(home_team, [1, 1, 1, 1, 1])
        away_form = recent_form.get(away_team, [1, 1, 1, 1, 1])
        feat.append(sum(home_form) / max(len(home_form), 1))
        feat.append(sum(away_form) / max(len(away_form), 1))
        feat.append(sum(home_form) - sum(away_form))

        # Goal difference features
        home_gd = (hs.get('goals_scored_home', 0) + hs.get('goals_scored_away', 0) -
                   hs.get('goals_conceded_home', 0) - hs.get('goals_conceded_away', 0))
        away_gd = (aws.get('goals_scored_home', 0) + aws.get('goals_scored_away', 0) -
                   aws.get('goals_conceded_home', 0) - aws.get('goals_conceded_away', 0))
        feat.append(home_gd / max(home_games, 1))
        feat.append(away_gd / max(away_games, 1))

        features.append(feat)

    return np.array(features, dtype=np.float32)


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.dropout(x + self.block(x)))


class DeepResidualMLP(nn.Module):
    """Deep MLP with residual connections and batch normalization."""
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_head(x)


class TabularMLP(nn.Module):
    """Simple MLP for comparison."""
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WideAndDeepMLP(nn.Module):
    """Wide & Deep architecture combining linear and deep pathways."""
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        deep_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()

        # Wide (linear) component
        self.wide = nn.Linear(input_dim, num_classes)

        # Deep component
        layers = []
        prev_dim = input_dim
        for dim in deep_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.deep = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wide(x) + self.deep(x)


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Original preprocessing - for backward compatibility."""
    target_cols = ["HomeScore", "AwayScore"]

    # Drop rows with missing targets in training data
    train_df = train_df.dropna(subset=target_cols).copy()

    team_cols = ["HomeTeam", "AwayTeam"]

    # Encode teams as integers
    encoders = {
        col: LabelEncoder().fit(pd.concat([train_df[col], test_df[col]]))
        for col in team_cols
    }
    for col in team_cols:
        train_df[col] = encoders[col].transform(train_df[col])
        test_df[col] = encoders[col].transform(test_df[col])

    # Build feature list (numeric + encoded teams), drop ids/targets
    feature_cols = [
        col
        for col in train_df.columns
        if col not in target_cols
        and col not in {"game_id", "n"}
        and (np.issubdtype(train_df[col].dtype, np.number) or col in team_cols)
    ]

    # Ensure test has all feature columns
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = np.nan

    # Median fill (fallback to 0 for all-NaN columns)
    medians = train_df[feature_cols].median().fillna(0)
    train_df[feature_cols] = train_df[feature_cols].fillna(medians)
    test_df[feature_cols] = test_df[feature_cols].fillna(medians)

    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df[target_cols].values.astype(np.float32)

    X_test = test_df[feature_cols].values.astype(np.float32)
    return X, y, X_test, feature_cols, target_cols


def preprocess_with_engineering(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Advanced preprocessing with feature engineering."""
    target_cols = ["HomeScore", "AwayScore"]

    # Drop rows with missing targets
    train_df = train_df.dropna(subset=target_cols).copy()

    # Compute engineered features from training data
    team_stats = compute_team_stats(train_df)
    elo_ratings = compute_elo_ratings(train_df)
    h2h = compute_head_to_head(train_df)
    recent_form = compute_recent_form(train_df)

    # Build engineered features
    X_eng_train = build_engineered_features(train_df, team_stats, elo_ratings, h2h, recent_form, is_train=True)
    X_eng_test = build_engineered_features(test_df, team_stats, elo_ratings, h2h, recent_form, is_train=False)

    # Also include original numeric features for training data
    numeric_cols = [col for col in train_df.columns
                    if col not in target_cols + ['game_id', 'n', 'HomeTeam', 'AwayTeam']
                    and np.issubdtype(train_df[col].dtype, np.number)]

    if numeric_cols:
        X_orig = train_df[numeric_cols].fillna(0).values.astype(np.float32)
        X_train = np.hstack([X_eng_train, X_orig])
        # Test data doesn't have these features, so pad with zeros
        X_test = np.hstack([X_eng_test, np.zeros((len(test_df), len(numeric_cols)), dtype=np.float32)])
    else:
        X_train = X_eng_train
        X_test = X_eng_test

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y = train_df[target_cols].values.astype(np.float32)

    return X_train, y, X_test, None, target_cols, (team_stats, elo_ratings, h2h, recent_form, scaler)


# ============================================================================
# TRAINING
# ============================================================================

def encode_targets(y: np.ndarray) -> np.ndarray:
    """Encode match outcomes: -1 (away win) -> 0, 0 (draw) -> 1, 1 (home win) -> 2"""
    outcomes = np.sign(y[:, 0] - y[:, 1])
    mapping = {-1: 0, 0: 1, 1: 2}
    return np.array([mapping.get(x, 1) for x in outcomes], dtype=int)


def train_single_model(
    X: np.ndarray,
    y: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    model_type: str = 'deep_residual',
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_size: float = 0.2,
    seed: int = 42,
    device: Optional[str] = None,
    hidden_dim: int = 256,
    num_blocks: int = 3,
    dropout: float = 0.3,
    weight_decay: float = 1e-4,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    early_stop_patience: int = 30,
    early_stop_min_delta: float = 1e-5,
    label_smoothing: float = 0.1,
    use_class_weights: bool = True,
    noise_std: float = 0.02,
    mixup_alpha: float = 0.2,
    verbose: bool = True,
):
    """Train a single model with advanced techniques."""
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    y_encoded = encode_targets(y)
    num_classes = int(y_encoded.max() + 1)

    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=val_size, random_state=seed, stratify=y_encoded
        )
    else:
        X_train, y_train = X, y_encoded
        y_val = encode_targets(y_val)

    train_loader = DataLoader(
        TabularDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=batch_size)

    # Create model based on type
    if model_type == 'deep_residual':
        model = DeepResidualMLP(
            input_dim=X_train.shape[1],
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
        ).to(device)
    elif model_type == 'wide_deep':
        model = WideAndDeepMLP(
            input_dim=X_train.shape[1],
            num_classes=num_classes,
            deep_dims=(hidden_dim, hidden_dim // 2, hidden_dim // 4),
            dropout=dropout,
        ).to(device)
    else:  # simple MLP
        model = TabularMLP(
            input_dim=X_train.shape[1],
            num_classes=num_classes,
            hidden_dims=(hidden_dim, hidden_dim // 2),
            dropout=dropout,
        ).to(device)

    # Class weights for imbalanced data
    class_weights = None
    if use_class_weights:
        counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (num_classes * counts)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=scheduler_factor, patience=scheduler_patience
    )

    best_val_f1 = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # Gaussian noise augmentation
            if noise_std > 0:
                xb = xb + torch.randn_like(xb) * noise_std

            # Mixup augmentation
            if mixup_alpha > 0 and np.random.random() < 0.5:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                idx = torch.randperm(xb.size(0))
                xb = lam * xb + (1 - lam) * xb[idx]
                # For mixup, use soft labels approach
                yb_mixed = yb[idx]

                optimizer.zero_grad()
                preds = model(xb)
                loss = lam * criterion(preds, yb) + (1 - lam) * criterion(preds, yb_mixed)
            else:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                all_val_preds.append(preds.argmax(dim=1).cpu().numpy())
                all_val_targets.append(yb.cpu().numpy())

        val_preds = np.concatenate(all_val_preds)
        val_targets = np.concatenate(all_val_targets)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_acc = accuracy_score(val_targets, val_preds)

        scheduler.step(val_f1)

        if val_f1 > best_val_f1 + early_stop_min_delta:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:03d} - loss: {train_loss:.4f} val_acc: {val_acc:.3f} val_f1: {val_f1:.3f}")

        if epochs_no_improve >= early_stop_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_f1


def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_seeds: int = 3,
    model_types: List[str] = ['deep_residual', 'wide_deep', 'mlp'],
    epochs: int = 200,
    batch_size: int = 32,
    device: Optional[str] = None,
    verbose: bool = True,
    **kwargs
):
    """Train an ensemble of models using cross-validation and multiple seeds."""
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    y_encoded = encode_targets(y)
    models = []
    val_scores = []

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y_encoded)):
        if verbose:
            print(f"\n=== Fold {fold + 1}/{n_folds} ===")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        for seed_offset, model_type in enumerate(model_types):
            for seed in range(n_seeds):
                actual_seed = 42 + fold * 100 + seed_offset * 10 + seed
                torch.manual_seed(actual_seed)
                np.random.seed(actual_seed)

                if verbose:
                    print(f"  Training {model_type} (seed {actual_seed})...")

                model, val_f1 = train_single_model(
                    X_train_fold, y_train_fold,
                    X_val=X_val_fold, y_val=y_val_fold,
                    model_type=model_type,
                    epochs=epochs,
                    batch_size=batch_size,
                    seed=actual_seed,
                    device=device,
                    verbose=False,
                    **kwargs
                )

                models.append(model)
                val_scores.append(val_f1)

                if verbose:
                    print(f"    Val F1: {val_f1:.4f}")

    if verbose:
        print(f"\nEnsemble trained with {len(models)} models")
        print(f"Mean Val F1: {np.mean(val_scores):.4f} (+/- {np.std(val_scores):.4f})")

    return models, val_scores


# ============================================================================
# LEGACY FUNCTION FOR BACKWARD COMPATIBILITY
# ============================================================================

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_size: float = 0.2,
    seed: int = 42,
    device: Optional[str] = None,
    hidden_channels: int = 64,
    deep_channels: int = 128,
    dropout_conv: float = 0.1,
    dropout_fc: float = 0.3,
    fc_dim: int = 128,
    weight_decay: float = 1e-4,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 1e-4,
    label_smoothing: float = 0.0,
    use_class_weights: bool = True,
    noise_std: float = 0.0,
):
    """Legacy training function for backward compatibility."""
    return train_single_model(
        X, y, X_val, y_val,
        model_type='mlp',
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        val_size=val_size,
        seed=seed,
        device=device,
        hidden_dim=fc_dim,
        dropout=dropout_fc,
        weight_decay=weight_decay,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        label_smoothing=label_smoothing,
        use_class_weights=use_class_weights,
        noise_std=noise_std,
        mixup_alpha=0.0,
        verbose=True,
    )[0]


# ============================================================================
# PREDICTION
# ============================================================================

def predict(model: nn.Module, X_test: np.ndarray, device: Optional[str] = None):
    """Predict with a single model."""
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    loader = DataLoader(TabularDataset(X_test), batch_size=64)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def predict_ensemble(
    models: List[nn.Module],
    X_test: np.ndarray,
    device: Optional[str] = None,
    method: str = 'soft_voting'
) -> np.ndarray:
    """Predict using ensemble of models with voting."""
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    all_probs = []

    for model in models:
        model.eval()
        model.to(device)
        loader = DataLoader(TabularDataset(X_test), batch_size=64)

        model_probs = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)
                model_probs.append(probs.cpu().numpy())

        all_probs.append(np.concatenate(model_probs, axis=0))

    # Stack and average probabilities
    all_probs = np.stack(all_probs, axis=0)  # (n_models, n_samples, n_classes)

    if method == 'soft_voting':
        avg_probs = np.mean(all_probs, axis=0)  # (n_samples, n_classes)
        return np.argmax(avg_probs, axis=1)
    else:  # hard voting
        votes = np.argmax(all_probs, axis=2)  # (n_models, n_samples)
        # Mode along models axis
        from scipy import stats
        return stats.mode(votes, axis=0, keepdims=False)[0]


def predict_with_probabilities(
    models: List[nn.Module],
    X_test: np.ndarray,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict with ensemble and return both predictions and probabilities."""
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    all_probs = []

    for model in models:
        model.eval()
        model.to(device)
        loader = DataLoader(TabularDataset(X_test), batch_size=64)

        model_probs = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)
                model_probs.append(probs.cpu().numpy())

        all_probs.append(np.concatenate(model_probs, axis=0))

    all_probs = np.stack(all_probs, axis=0)
    avg_probs = np.mean(all_probs, axis=0)
    predictions = np.argmax(avg_probs, axis=1)

    return predictions, avg_probs


# ============================================================================
# SUBMISSION
# ============================================================================

def save_submission(
    test_df: pd.DataFrame, preds: np.ndarray, target_cols: List[str], output: Path
):
    outcome_map = {0: "AwayWin", 1: "Draw", 2: "HomeWin"}
    labels = np.vectorize(outcome_map.get)(preds)
    submission = pd.DataFrame({"game_id": test_df["game_id"], "Outcome": labels})
    submission.to_csv(output, index=False)
    print(f"Saved predictions to {output}")
