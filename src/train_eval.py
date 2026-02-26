import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def walk_forward_backtest(df, feature_cols, target_col="regime", start_train=60):
    X = df[feature_cols]
    y = df[target_col]

    preds_lr = []
    preds_rf = []
    preds_base = []
    y_true = []
    dates = []

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    rf = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        random_state=42
    )

    for i in range(start_train, len(df)-1):
        X_train, y_train = X.iloc[:i], y.iloc[:i]
        X_test, y_test = X.iloc[i:i+1], y.iloc[i:i+1]

        # Baseline: most frequent class in training window
        base_class = y_train.value_counts().idxmax()

        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        preds_base.append(base_class)
        preds_lr.append(int(lr.predict(X_test)[0]))
        preds_rf.append(int(rf.predict(X_test)[0]))

        y_true.append(int(y_test.iloc[0]))
        dates.append(df.index[i])

    if len(dates) == 0:
        raise ValueError(
            f"Walk-forward produced 0 predictions. start_train={start_train}, len(df)={len(df)}. "
            "Lower start_train or check why features.csv is short."
        )

    results = pd.DataFrame({
        "date": dates,
        "y_true": y_true,
        "pred_base": preds_base,
        "pred_lr": preds_lr,
        "pred_rf": preds_rf
    }).set_index("date")

    return results

def score_model(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return {"model": name, "accuracy": acc, "macro_f1": f1m}

def main():
    df = pd.read_csv("data/features.csv", index_col=0, parse_dates=True)

    n = len(df)
    print("Features rows:", n)
    
    # If too few rows, fail early with a clear message
    if n < 30:
        raise ValueError(f"Not enough rows after feature engineering: {n}. Check feature creation/dropna.")
    
    # Auto-adjust start_train so we always have a test set
    start_train = min(84, max(24, n // 3))  # at least 24 months, otherwise 1/3 of data
    print("Using start_train:", start_train)

    feature_cols = [c for c in df.columns if c not in ["regime", "spy_ret_fwd1"]]
    res = walk_forward_backtest(df, feature_cols, start_train=start_train)
    
    scores = []
    scores.append(score_model(res["y_true"], res["pred_base"], "baseline_majority"))
    scores.append(score_model(res["y_true"], res["pred_lr"], "log_reg"))
    scores.append(score_model(res["y_true"], res["pred_rf"], "random_forest"))

    score_df = pd.DataFrame(scores).sort_values(["macro_f1","accuracy"], ascending=False)
    print(score_df)

    print("\nClassification report (best of LR/RF shown separately):")
    print("\nLR report:\n", classification_report(res["y_true"], res["pred_lr"]))
    print("\nRF report:\n", classification_report(res["y_true"], res["pred_rf"]))

    res.to_csv("data/predictions.csv")
    score_df.to_csv("data/scores.csv", index=False)
    print("Saved data/predictions.csv and data/scores.csv")

if __name__ == "__main__":
    main()
