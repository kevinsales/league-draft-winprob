"""Train, evaluate, calibrate, and save the win-probability model.

Run (from anywhere):  python src/model.py

M4 baseline: regularized logistic regression on Tier-1 (multi-hot picks + bans),
evaluated calibration-forward (log-loss, ROC-AUC, Brier, accuracy, calibration
curve) against two baselines it must beat -- a 50/50 coin flip and a naive
"sum of champion win-rates" rule.

Evaluation is out-of-fold cross-validation (StratifiedKFold): with only ~50
matches a single held-out split is too noisy, so every match gets a prediction
from a model that never saw it. Baselines are refit on the same folds for a fair
comparison. There is no outcome leakage -- champ win-rates for the naive baseline
are computed on training rows only, inside each fold.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: save figures, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import features as F  # noqa: E402

N_SPLITS = 5
RANDOM_STATE = 42
# Regularization strength is TUNED, not guessed: with ~500 sparse features a fixed
# C overfits badly (overconfident probabilities -> log-loss worse than a coin flip).
# An inner CV picks C on training rows only, inside each outer fold -> nested CV,
# so the reported OOF metrics stay honest.
C_GRID = np.logspace(-4, 1, 10)


def make_model() -> GridSearchCV:
    return GridSearchCV(
        LogisticRegression(solver="liblinear", max_iter=1000),  # liblinear defaults to L2
        param_grid={"C": C_GRID},
        scoring="neg_log_loss",
        cv=5,
    )


def champ_winrates(df: pd.DataFrame) -> dict[int, float]:
    """Per-champion win rate from the *team that picked it*, over train rows only.

    Blue champs are credited the match label y; red champs are credited (1 - y).
    """
    wins: dict[int, float] = {}
    games: dict[int, int] = {}
    for _, r in df.iterrows():
        for c in (r[col] for col in F.BLUE_COLS):
            wins[c] = wins.get(c, 0) + r["win"]
            games[c] = games.get(c, 0) + 1
        for c in (r[col] for col in F.RED_COLS):
            wins[c] = wins.get(c, 0) + (1 - r["win"])
            games[c] = games.get(c, 0) + 1
    return {c: wins[c] / games[c] for c in games}


def naive_scores(df: pd.DataFrame, wr: dict[int, float], base: float) -> np.ndarray:
    """blue mean champ win-rate minus red mean champ win-rate (unseen -> base)."""
    out = []
    for _, r in df.iterrows():
        b = np.mean([wr.get(r[col], base) for col in F.BLUE_COLS])
        rd = np.mean([wr.get(r[col], base) for col in F.RED_COLS])
        out.append(b - rd)
    return np.array(out).reshape(-1, 1)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "accuracy": accuracy_score(y, p > 0.5),
    }


def out_of_fold(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
    """Return OOF probabilities for the model and the naive baseline."""
    Xv, yv = X.values, y.values
    oof_model = np.zeros(len(y))
    oof_naive = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in skf.split(Xv, yv):
        # main model: inner CV tunes C on this fold's training rows only
        m = make_model()
        m.fit(Xv[tr], yv[tr])
        oof_model[te] = m.predict_proba(Xv[te])[:, 1]

        # naive baseline: champ win-rates from this fold's train rows, then a
        # 1-feature logistic to turn the score into a probability.
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        wr = champ_winrates(tr_df)
        base = tr_df["win"].mean()
        s_tr = naive_scores(tr_df, wr, base)
        s_te = naive_scores(te_df, wr, base)
        lr = LogisticRegression(max_iter=1000).fit(s_tr, yv[tr])
        oof_naive[te] = lr.predict_proba(s_te)[:, 1]
    return oof_model, oof_naive


def plot_calibration(y, p, path: Path) -> None:
    prob_true, prob_pred = calibration_curve(y, np.clip(p, 1e-6, 1 - 1e-6), n_bins=5)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(prob_pred, prob_true, "o-", label="baseline LR")
    ax.set_xlabel("predicted win probability")
    ax.set_ylabel("observed win rate")
    ax.set_title("Baseline calibration (out-of-fold, Tier-1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def top_coefficients(X: pd.DataFrame, y: pd.Series, labels: pd.DataFrame, k: int = 10):
    """Fit on all data and surface the champions that most move win probability."""
    m = make_model()
    m.fit(X.values, y.values)
    print(f"\n(final model chose C={m.best_params_['C']:.4g} from the grid)")
    coef = pd.Series(m.best_estimator_.coef_[0], index=X.columns)

    def label(col: str) -> str:
        cid = int(col.split("_")[-1])
        name = labels.loc[cid, "name"] if cid in labels.index else f"id{cid}"
        return col.replace(str(cid), name)

    coef.index = [label(c) for c in coef.index]
    ranked = coef.sort_values()
    return ranked.tail(k)[::-1], ranked.head(k)  # most positive, most negative


def main() -> None:
    df = F.load_matches()
    labels = F.load_labels()
    X, y = F.tier1_features(df)
    print(f"Baseline on Tier-1: X={X.shape}, blue win rate={y.mean():.1%}, {N_SPLITS}-fold OOF\n")

    oof_model, oof_naive = out_of_fold(df, X, y)
    yv = y.values
    # Base rate is an extra, tougher-than-required bar: a constant predictor at the
    # observed blue win rate. Shown for honesty; the M4 gate is coin flip + naive.
    results = {
        "logreg (Tier-1)": metrics(yv, oof_model),
        "naive champ-WR": metrics(yv, oof_naive),
        "coin flip (0.5)": metrics(yv, np.full(len(yv), 0.5)),
        "base rate (const)": metrics(yv, np.full(len(yv), yv.mean())),
    }
    table = pd.DataFrame(results).T[["log_loss", "roc_auc", "brier", "accuracy"]]
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    m_ll = results["logreg (Tier-1)"]["log_loss"]
    beats = m_ll < results["naive champ-WR"]["log_loss"] and m_ll < results["coin flip (0.5)"]["log_loss"]
    print(f"\nModel beats BOTH baselines on log-loss: {beats}")

    fig_path = config.FIGURES / "calibration_baseline.png"
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    plot_calibration(yv, oof_model, fig_path)
    print(f"Saved calibration curve -> {fig_path.relative_to(config.ROOT)}")

    pos, neg = top_coefficients(X, y, labels)
    print("\nTop win-contributing features (+ = raises blue win prob):")
    print(pos.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("\nTop loss-contributing features (-):")
    print(neg.to_string(float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
