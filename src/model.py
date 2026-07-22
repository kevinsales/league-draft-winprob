"""Train, evaluate, calibrate, and save the win-probability model.

Run (from anywhere):  python src/model.py

  M4 baseline  regularized logistic regression on Tier-1 (multi-hot picks + bans)
  M5 stronger  LightGBM on Tier-2 (archetype aggregates + blue-minus-red diffs)

Everything is scored the same way so the comparison is fair: out-of-fold
StratifiedKFold predictions, with every model's hyperparameters tuned by an
INNER cross-validation on training rows only (nested CV). No model ever sees the
rows it is scored on, and the naive baseline's champion win-rates are recomputed
inside each fold. All features are known at champ-select lock, so there is no
outcome leakage by construction.

Outputs: a model-comparison table, calibration + feature-importance + win-prob
spread figures in reports/figures/, and the fitted model saved for predict.py.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: save figures, never open a window
import joblib
import lightgbm as lgb
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
import viz  # noqa: E402

N_SPLITS = 5
RANDOM_STATE = 42

# Plain-English names for anything a reader sees on a chart.
FRIENDLY = {
    "logreg  Tier-1 (M4)": "our model (champions)",
    "LightGBM Tier-2 (M5)": "fancier model (did worse)",
}

# Regularization is TUNED, not guessed. With ~500 sparse Tier-1 features a fixed C
# overfits badly (overconfident probabilities -> log-loss worse than a coin flip).
C_GRID = np.logspace(-4, 1, 10)


def make_logreg() -> GridSearchCV:
    return GridSearchCV(
        LogisticRegression(solver="liblinear", max_iter=1000),  # liblinear defaults to L2
        param_grid={"C": C_GRID},
        scoring="neg_log_loss",
        cv=5,
    )


def make_lgbm() -> GridSearchCV:
    """Deliberately small/shallow trees: the draft signal is weak, so an
    unconstrained booster would just memorize noise."""
    return GridSearchCV(
        lgb.LGBMClassifier(
            objective="binary", importance_type="gain",
            random_state=RANDOM_STATE, verbose=-1,
        ),
        param_grid={
            "num_leaves": [4, 8],
            "learning_rate": [0.02, 0.05],
            "n_estimators": [200, 400],
            "min_child_samples": [20, 50],
            "reg_lambda": [1.0],
        },
        scoring="neg_log_loss",
        cv=5,
    )


# ----------------------------------------------------------------- baselines --
def champ_winrates(df: pd.DataFrame) -> dict[int, float]:
    """Per-champion win rate credited to the team that picked it (train rows only)."""
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


# ---------------------------------------------------------------- evaluation --
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "accuracy": accuracy_score(y, p > 0.5),
    }


def oof_probs(factory, X: pd.DataFrame, y: pd.Series, skf) -> np.ndarray:
    """Out-of-fold probabilities; hyperparameters tuned inside each fold.

    Keep DataFrames (not .values) end to end so feature names survive into the
    estimators -- LightGBM warns if it is fitted with names and scored without.
    """
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        m = factory()
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


def oof_naive(df: pd.DataFrame, y: pd.Series, skf) -> np.ndarray:
    """Naive 'sum of champion win-rates' baseline, refit on each fold's train rows."""
    yv = y.values
    oof = np.zeros(len(y))
    for tr, te in skf.split(np.zeros(len(y)), yv):
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        wr = champ_winrates(tr_df)
        base = tr_df["win"].mean()
        lr = LogisticRegression(max_iter=1000).fit(naive_scores(tr_df, wr, base), yv[tr])
        oof[te] = lr.predict_proba(naive_scores(te_df, wr, base))[:, 1]
    return oof


# ------------------------------------------------------------------- figures --
def plot_calibration(y, curves: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = viz.figure(5.6, 5.4)
    ax.plot([0, 1], [0, 1], color=viz.INK_MUTED, lw=1, zorder=1, label="perfect")
    for name, colour in zip(curves, (viz.BLUE_SIDE, viz.LOST)):
        prob_true, prob_pred = calibration_curve(y, np.clip(curves[name], 1e-6, 1 - 1e-6),
                                                 n_bins=10)
        ax.plot(prob_pred, prob_true, "o-", color=colour, lw=2, ms=6,
                mec=viz.SURFACE, mew=1.5, label=FRIENDLY.get(name.strip(), name.strip()), zorder=3)
    ax.set_xlabel("win chance the model predicted")
    ax.set_ylabel("how often that actually happened")
    viz.title_block(ax, "Are these predictions honest?",
                    "Dots on the grey line mean the model's stated chance matches reality.")
    ax.legend(loc="upper left")
    viz.caption(ax, "They sit near the line, so the model is honest -- but look how tiny a "
                    "slice of the chart it uses. It only ever says 'roughly even'.")
    return viz.save(fig, path.name)


def plot_champion_coefficients(X: pd.DataFrame, y: pd.Series, labels: pd.DataFrame,
                               path: Path, k: int = 12):
    """D4: what the model that ACTUALLY works learned, replacing the LightGBM
    importance plot (which ranked a below-chance model, i.e. noise).

    Diverging: blue = shifts the game toward blue side, red = toward red side.
    """
    m = make_logreg()
    m.fit(X, y)
    coef = pd.Series(m.best_estimator_.coef_[0], index=X.columns)

    def pretty(col: str) -> str:
        cid = int(col.split("_")[-1])
        name = labels.loc[cid, "name"] if cid in labels.index else f"id{cid}"
        kind = ("banned" if col.startswith("ban")
                else "on the blue team" if "blue" in col else "on the red team")
        return f"{name}  ({kind})"

    ranked = pd.concat([coef.sort_values().head(k), coef.sort_values().tail(k)])
    ranked.index = [pretty(c) for c in ranked.index]
    ranked = viz.to_points(ranked)  # log-odds -> points of win chance
    colours = [viz.RED_SIDE if v < 0 else viz.BLUE_SIDE for v in ranked.values]

    fig, ax = viz.figure(7.6, 7)
    ax.barh(range(len(ranked)), ranked.values, color=colours, height=0.72)
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(ranked.index, fontsize=9)
    ax.axvline(0, color=viz.BASELINE, lw=1)
    viz.despine_x(ax)
    ax.set_xlabel("shifts the game toward the blue team (percentage points)")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.1f}")
    viz.title_block(ax, "Which champions actually change your odds?",
                    f"The {k} biggest movers each way, across 1,679 Master games.")
    viz.caption(ax, "A bar to the right means the blue team wins more often when that "
                    "happens; a bar to the left means the red team does. Either way, read "
                    "the scale: the biggest mover on this chart is worth under 2 games in "
                    "100. Champion select is not what wins the game.")
    return viz.save(fig, path.name)


def plot_winprob_spread(p: np.ndarray, path: Path):
    """The headline picture: if draft decided games, these predictions would fan
    out toward 0 and 1. They don't -- they huddle around the base rate."""
    fig, ax = viz.figure(8, 4.2)
    ax.hist(p, bins=32, color=viz.BLUE_SIDE, linewidth=0)
    viz.reference_line(ax, 0.5, "coin flip", axis="x")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("win chance the draft gave the blue team")
    ax.set_ylabel("number of games")
    lo, hi = p.min(), p.max()
    ax.annotate(f"every prediction lives in here\n{lo:.0%} - {hi:.0%}",
                xy=((lo + hi) / 2, ax.get_ylim()[1] * 0.72),
                xytext=(0.78, ax.get_ylim()[1] * 0.8),
                fontsize=9.5, color=viz.INK_SECONDARY, ha="center",
                arrowprops=dict(arrowstyle="->", color=viz.INK_MUTED, lw=1))
    viz.title_block(ax, "The draft almost never picks a winner",
                    "Each bar counts games. If champion select decided matches, these bars "
                    "would spread out toward 0% and 100%.")
    viz.caption(ax, "Instead they pile up on 50/50. After seeing both full drafts, the best "
                    "guess is still almost a coin flip.")
    return viz.save(fig, path.name)


# ---------------------------------------------------------------------- main --
def main() -> None:
    df = F.load_matches()
    labels = F.load_labels()
    X1, y = F.tier1_features(df)
    X2, _ = F.tier2_features(df, labels)
    yv = y.values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print(f"Matches: {len(df)} | patches {config.TARGET_PATCHES} | blue win rate {y.mean():.1%}")
    print(f"Tier-1: {X1.shape}  Tier-2: {X2.shape}  ({N_SPLITS}-fold nested OOF)\n")

    oof = {
        "logreg  Tier-1 (M4)": oof_probs(make_logreg, X1, y, skf),
        "logreg  Tier-2": oof_probs(make_logreg, X2, y, skf),
        "LightGBM Tier-2 (M5)": oof_probs(make_lgbm, X2, y, skf),
        "naive champ-WR": oof_naive(df, y, skf),
        "coin flip (0.5)": np.full(len(yv), 0.5),
        "base rate (const)": np.full(len(yv), yv.mean()),
    }
    table = pd.DataFrame({k: metrics(yv, p) for k, p in oof.items()}).T
    table = table[["log_loss", "roc_auc", "brier", "accuracy"]].sort_values("log_loss")
    print("=== MODEL COMPARISON (out-of-fold, identical folds) ===")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    # ---- figures
    viz.apply_style()
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    plot_calibration(yv, {k: oof[k] for k in ["logreg  Tier-1 (M4)", "LightGBM Tier-2 (M5)"]},
                     config.FIGURES / "calibration_baseline.png")
    # LightGBM is still fitted (it may win and get saved) but we no longer plot its
    # importances -- it scored below chance, so that ranking was noise.
    lgbm_full = make_lgbm()
    lgbm_full.fit(X2, y)
    plot_champion_coefficients(X1, y, labels, config.FIGURES / "champion_effects.png")

    # Pick the best *trained* model (baselines aren't models we can ship).
    TRAINED = ["logreg  Tier-1 (M4)", "logreg  Tier-2", "LightGBM Tier-2 (M5)"]
    best_name = min(TRAINED, key=lambda n: metrics(yv, oof[n])["log_loss"])
    best_probs = oof[best_name]

    plot_winprob_spread(best_probs, config.FIGURES / "winprob_spread.png")
    print(f"\nFigures -> reports/figures/: calibration_baseline.png, "
          f"champion_effects.png, winprob_spread.png")

    # ---- save the best model for predict.py (M7)
    if best_name == "LightGBM Tier-2 (M5)":
        final, cols, tier = lgbm_full, list(X2.columns), "tier2"
    elif best_name == "logreg  Tier-2":
        final, cols, tier = make_logreg().fit(X2, y), list(X2.columns), "tier2"
    else:
        final, cols, tier = make_logreg().fit(X1, y), list(X1.columns), "tier1"
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "columns": cols, "tier": tier},
                config.DATA_PROCESSED / "model.joblib")
    print(f"Saved best model ('{best_name.strip()}', {tier}) -> data/processed/model.joblib")

    # ---- headline analysis
    m = metrics(yv, best_probs)
    majority = max(yv.mean(), 1 - yv.mean())  # accuracy of always guessing the common side
    print("\n=== HEADLINE: how much does the draft actually decide? ===")
    print(f"Best model: {best_name}")
    print(f"  ROC-AUC   {m['roc_auc']:.3f}  ->  {(m['roc_auc'] - 0.5) * 100:+.1f} pp above chance (0.500)")
    print(f"  Accuracy  {m['accuracy']:.1%}  ->  {(m['accuracy'] - 0.5) * 100:+.1f} pp above a coin flip")
    print(f"            {'':13}{(m['accuracy'] - majority) * 100:+.1f} pp above always-guess-the-common-side ({majority:.1%})")
    print(f"  Log-loss  {m['log_loss']:.4f} vs {metrics(yv, oof['base rate (const)'])['log_loss']:.4f} for a constant predictor")
    print(f"  Predicted win prob spans {best_probs.min():.1%}-{best_probs.max():.1%} "
          f"(std {best_probs.std():.3f}) -- draft rarely moves us far from even odds.")
    print("\n  Read: champion select alone gets us only a few points above chance.")
    print("  The draft is not what decides a solo-queue game -- execution is.")


if __name__ == "__main__":
    main()
