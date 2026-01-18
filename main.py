import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from model import (
    load_data,
    predict,
    predict_ensemble,
    preprocess,
    preprocess_with_engineering,
    save_submission,
    train_model,
    train_ensemble,
    train_single_model,
    encode_targets,
)


def plot_results(y_train_raw, train_preds, test_preds):
    outcome_map = {0: "AwayWin", 1: "Draw", 2: "HomeWin"}

    # y_train_raw is (N, 2) -> [HomeScore, AwayScore]
    outcomes = np.sign(y_train_raw[:, 0] - y_train_raw[:, 1])
    actual_indices = [{-1: 0, 0: 1, 1: 2}.get(x, 1) for x in outcomes]
    train_actual_labels = [outcome_map[i] for i in actual_indices]

    train_pred_labels = [outcome_map[i] for i in train_preds]
    test_pred_labels = [outcome_map[i] for i in test_preds]

    plot_data = []
    for l in train_actual_labels:
        plot_data.append({"Set": "Train (Actual)", "Outcome": l})
    for l in train_pred_labels:
        plot_data.append({"Set": "Train (Predicted)", "Outcome": l})
    for l in test_pred_labels:
        plot_data.append({"Set": "Test (Predicted)", "Outcome": l})

    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=df_plot,
        x="Outcome",
        hue="Set",
        order=["HomeWin", "Draw", "AwayWin"],
        palette="viridis",
    )
    plt.title("Distribution of Outcomes: Actual vs Predicted")
    plt.xlabel("Match Outcome")
    plt.ylabel("Count")
    plt.tight_layout()


def plot_validation_comparison(y_val, val_preds):
    outcome_map = {0: "AwayWin", 1: "Draw", 2: "HomeWin"}

    outcomes = np.sign(y_val[:, 0] - y_val[:, 1])
    y_true = [{-1: 0, 0: 1, 1: 2}.get(x, 1) for x in outcomes]
    val_actual_labels = [outcome_map[i] for i in y_true]
    val_pred_labels = [outcome_map[i] for i in val_preds]

    plot_data = []
    for l in val_actual_labels:
        plot_data.append({"Type": "Actual (Validation)", "Outcome": l})
    for l in val_pred_labels:
        plot_data.append({"Type": "Predicted (Validation)", "Outcome": l})

    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=df,
        x="Outcome",
        hue="Type",
        order=["HomeWin", "Draw", "AwayWin"],
        palette="Set2",
    )
    plt.title("Comparison: Actual vs Predicted (Validation Data)")
    plt.ylabel("Count")
    plt.xlabel("Match Outcome")
    plt.legend(title="Data Source")
    plt.tight_layout()


def plot_confusion_matrix(y_val, val_preds):
    outcome_map = {0: "AwayWin", 1: "Draw", 2: "HomeWin"}

    outcomes = np.sign(y_val[:, 0] - y_val[:, 1])
    y_true = [{-1: 0, 0: 1, 1: 2}.get(x, 1) for x in outcomes]

    cm = confusion_matrix(y_true, val_preds, labels=[0, 1, 2])

    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["AwayWin", "Draw", "HomeWin"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Validation Set)")
    plt.tight_layout()


def plot_accuracy_pie_chart(y_val, val_preds):
    """Plot pie chart showing correct vs incorrect predictions."""
    outcomes = np.sign(y_val[:, 0] - y_val[:, 1])
    y_true = np.array([{-1: 0, 0: 1, 1: 2}.get(x, 1) for x in outcomes])

    correct = np.sum(y_true == val_preds)
    incorrect = len(y_true) - correct
    accuracy = correct / len(y_true) * 100

    plt.figure(figsize=(8, 8))
    sizes = [correct, incorrect]
    labels = [f'Correct\n{correct} ({accuracy:.1f}%)', f'Incorrect\n{incorrect} ({100-accuracy:.1f}%)']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)

    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='', startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})
    plt.title(f'Overall Prediction Accuracy: {accuracy:.1f}%', fontsize=16, fontweight='bold')
    plt.tight_layout()


def plot_per_class_accuracy(y_val, val_preds):
    """Plot bar chart showing accuracy percentage per class."""
    outcome_map = {0: "AwayWin", 1: "Draw", 2: "HomeWin"}

    outcomes = np.sign(y_val[:, 0] - y_val[:, 1])
    y_true = np.array([{-1: 0, 0: 1, 1: 2}.get(x, 1) for x in outcomes])

    # Calculate per-class accuracy
    classes = [0, 1, 2]
    class_names = ["AwayWin", "Draw", "HomeWin"]
    accuracies = []
    counts = []

    for c in classes:
        mask = y_true == c
        if mask.sum() > 0:
            acc = (val_preds[mask] == c).sum() / mask.sum() * 100
        else:
            acc = 0
        accuracies.append(acc)
        counts.append(mask.sum())

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(class_names, accuracies, color=['#3498db', '#9b59b6', '#e67e22'], edgecolor='black', linewidth=1.5)

    # Add percentage labels on bars
    for bar, acc, count in zip(bars, accuracies, counts):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%\n(n={count})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Match Outcome', fontsize=12)
    ax.set_title('Prediction Accuracy by Outcome Class', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.axhline(y=sum(accuracies)/3, color='red', linestyle='--', label=f'Mean: {sum(accuracies)/3:.1f}%')
    ax.legend()
    plt.tight_layout()


def plot_prediction_distribution_pie(y_val, val_preds):
    """Plot pie charts comparing actual vs predicted outcome distributions."""
    outcome_map = {0: "AwayWin", 1: "Draw", 2: "HomeWin"}
    colors = ['#3498db', '#9b59b6', '#e67e22']

    outcomes = np.sign(y_val[:, 0] - y_val[:, 1])
    y_true = np.array([{-1: 0, 0: 1, 1: 2}.get(x, 1) for x in outcomes])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Actual distribution
    actual_counts = [np.sum(y_true == i) for i in range(3)]
    actual_pcts = [c / len(y_true) * 100 for c in actual_counts]
    labels_actual = [f'{outcome_map[i]}\n{actual_counts[i]} ({actual_pcts[i]:.1f}%)' for i in range(3)]

    axes[0].pie(actual_counts, labels=labels_actual, colors=colors,
                autopct='', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[0].set_title('Actual Outcomes', fontsize=14, fontweight='bold')

    # Predicted distribution
    pred_counts = [np.sum(val_preds == i) for i in range(3)]
    pred_pcts = [c / len(val_preds) * 100 for c in pred_counts]
    labels_pred = [f'{outcome_map[i]}\n{pred_counts[i]} ({pred_pcts[i]:.1f}%)' for i in range(3)]

    axes[1].pie(pred_counts, labels=labels_pred, colors=colors,
                autopct='', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[1].set_title('Predicted Outcomes', fontsize=14, fontweight='bold')

    plt.suptitle('Outcome Distribution: Actual vs Predicted', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()


def print_metrics(y_true_raw, y_pred, title="Metrics"):
    """Print detailed metrics."""
    outcomes = np.sign(y_true_raw[:, 0] - y_true_raw[:, 1])
    y_true = np.array([{-1: 0, 0: 1, 1: 2}.get(x, 1) for x in outcomes])

    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["AwayWin", "Draw", "HomeWin"]))


def main():
    parser = argparse.ArgumentParser(description="Train CNN, Predict, and Plot Results")
    parser.add_argument(
        "--train", type=Path, default=Path("wharton-data-science-2023-2024/train.csv")
    )
    parser.add_argument(
        "--test", type=Path, default=Path("wharton-data-science-2023-2024/test.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)

    # Model Architecture Params
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension for models"
    )
    parser.add_argument(
        "--num_blocks", type=int, default=3, help="Number of residual blocks"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout rate"
    )

    # Optimizer Params
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=10,
        help="Epochs to wait before reducing LR",
    )
    parser.add_argument(
        "--scheduler_factor", type=float, default=0.5, help="Factor to reduce LR by"
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=30,
        help="Stop if val loss doesn't improve for N epochs",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing for classification loss",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.02,
        help="Gaussian noise std added to inputs during training",
    )
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.2,
        help="Mixup alpha parameter (0 to disable)",
    )

    # Ensemble params
    parser.add_argument(
        "--n_folds", type=int, default=3, help="Number of CV folds for ensemble"
    )
    parser.add_argument(
        "--n_seeds", type=int, default=1, help="Number of seeds per model type"
    )
    parser.add_argument(
        "--use_ensemble", action="store_true", default=True,
        help="Use ensemble training"
    )
    parser.add_argument(
        "--use_feature_engineering", action="store_true", default=True,
        help="Use advanced feature engineering"
    )

    # Legacy params (for backward compatibility)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--deep_channels", type=int, default=128)
    parser.add_argument("--fc_dim", type=int, default=128)
    parser.add_argument("--dropout_conv", type=float, default=0.1)
    parser.add_argument("--dropout_fc", type=float, default=0.3)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-5)
    parser.add_argument("--use_class_weights", action="store_true", default=True)

    args = parser.parse_args()

    print(f"Loading data from {args.train} and {args.test}...")
    train_df, test_df = load_data(args.train, args.test)

    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    if args.use_feature_engineering:
        print("\nUsing advanced feature engineering...")
        X, y, X_test, feature_cols, target_cols, metadata = preprocess_with_engineering(
            train_df, test_df
        )
        print(f"Engineered features: {X.shape[1]}")
    else:
        print("\nUsing standard preprocessing...")
        X, y, X_test, feature_cols, target_cols = preprocess(train_df, test_df)

    # Split for validation visualization
    print("\nSplitting data for validation...")
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=encode_targets(y)
    )

    if args.use_ensemble:
        print(f"\n{'='*60}")
        print("TRAINING ENSEMBLE MODEL")
        print(f"{'='*60}")
        print(f"Folds: {args.n_folds}, Seeds per model: {args.n_seeds}")
        print(f"Model types: deep_residual, wide_deep, mlp")

        # Train ensemble on training subset for validation
        models_val, val_scores = train_ensemble(
            X_train_sub, y_train_sub,
            n_folds=args.n_folds,
            n_seeds=args.n_seeds,
            model_types=['deep_residual', 'wide_deep', 'mlp'],
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            scheduler_patience=args.scheduler_patience,
            scheduler_factor=args.scheduler_factor,
            early_stop_patience=args.early_stop_patience,
            label_smoothing=args.label_smoothing,
            noise_std=args.noise_std,
            mixup_alpha=args.mixup_alpha,
            verbose=True,
        )

        # Validate on held-out data
        val_preds = predict_ensemble(models_val, X_val)
        print_metrics(y_val, val_preds, "VALIDATION SET METRICS (Ensemble)")

        # Train final ensemble on ALL data
        print(f"\n{'='*60}")
        print("TRAINING FINAL ENSEMBLE ON FULL DATA")
        print(f"{'='*60}")

        models_full, full_scores = train_ensemble(
            X, y,
            n_folds=args.n_folds,
            n_seeds=args.n_seeds,
            model_types=['deep_residual', 'wide_deep', 'mlp'],
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            scheduler_patience=args.scheduler_patience,
            scheduler_factor=args.scheduler_factor,
            early_stop_patience=args.early_stop_patience,
            label_smoothing=args.label_smoothing,
            noise_std=args.noise_std,
            mixup_alpha=args.mixup_alpha,
            verbose=True,
        )

        # Final predictions
        test_preds = predict_ensemble(models_full, X_test)

    else:
        # Single model training (legacy mode)
        print("\nTraining single model on split data...")
        model_val, val_f1 = train_single_model(
            X_train_sub,
            y_train_sub,
            X_val=X_val,
            y_val=y_val,
            model_type='deep_residual',
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            scheduler_patience=args.scheduler_patience,
            scheduler_factor=args.scheduler_factor,
            early_stop_patience=args.early_stop_patience,
            label_smoothing=args.label_smoothing,
            noise_std=args.noise_std,
            mixup_alpha=args.mixup_alpha,
        )

        val_preds = predict(model_val, X_val)
        print_metrics(y_val, val_preds, "VALIDATION SET METRICS")

        print("\nRetraining on full dataset for final submission...")
        model_full, _ = train_single_model(
            X, y,
            model_type='deep_residual',
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            scheduler_patience=args.scheduler_patience,
            scheduler_factor=args.scheduler_factor,
            early_stop_patience=args.early_stop_patience,
            label_smoothing=args.label_smoothing,
            noise_std=args.noise_std,
            mixup_alpha=args.mixup_alpha,
        )
        test_preds = predict(model_full, X_test)

    # Save submission
    save_submission(test_df, test_preds, target_cols, args.output)

    # Plotting
    print("\nPlotting validation comparison...")
    plot_validation_comparison(y_val, val_preds)

    print("Plotting Confusion Matrix...")
    plot_confusion_matrix(y_val, val_preds)

    print("Plotting Accuracy Pie Chart...")
    plot_accuracy_pie_chart(y_val, val_preds)

    print("Plotting Per-Class Accuracy...")
    plot_per_class_accuracy(y_val, val_preds)

    print("Plotting Distribution Comparison...")
    plot_prediction_distribution_pie(y_val, val_preds)

    plt.show()


if __name__ == "__main__":
    main()
