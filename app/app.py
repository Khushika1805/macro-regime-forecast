import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import subprocess
from pathlib import Path

st.set_page_config(page_title="Macro → Equity Regime Backtest", layout="wide")
st.title("Macro → Equity Regime Prediction (Walk-Forward Backtest)")

def run(cmd):
    # Capture stdout/stderr so we can display it in Streamlit
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
        env=os.environ.copy()
    )

def ensure_outputs_exist():
    DATA_DIR.mkdir(exist_ok=True)

    if not (SCORES_PATH.exists() and PREDS_PATH.exists()):
        with st.spinner("Generating data for the dashboard (first run)…"):
            try:
                out1 = run(["python3", "src/fetch_data.py"])
                out2 = run(["python3", "src/build_features.py"])
                out3 = run(["python3", "src/train_eval.py"])

                st.success("Data generated successfully.")
                with st.expander("Build logs"):
                    st.code(out1.stdout + "\n" + out1.stderr)
                    st.code(out2.stdout + "\n" + out2.stderr)
                    st.code(out3.stdout + "\n" + out3.stderr)

            except subprocess.CalledProcessError as e:
                st.error("Failed to generate data files on the server.")
                st.code(" ".join(e.cmd))
                st.code(e.stdout or "")
                st.code(e.stderr or "")
                st.stop()

ensure_outputs_exist()

scores = pd.read_csv(SCORES_PATH)
preds = pd.read_csv(PREDS_PATH, index_col=0, parse_dates=True)
# ----------------------------
# Helpers
# ----------------------------
LABEL_MAP = {0: "Down", 1: "Flat", 2: "Up"}

def decode(series: pd.Series) -> pd.Series:
    return series.map(LABEL_MAP)


# Sidebar controls
st.sidebar.header("Controls")
model_choice = st.sidebar.selectbox(
    "Choose model for evaluation",
    options=["pred_lr", "pred_rf", "pred_base"],
    format_func=lambda x: {
        "pred_lr": "Logistic Regression",
        "pred_rf": "Random Forest",
        "pred_base": "Baseline (Majority Class)",
    }[x],
)

show_raw_labels = st.sidebar.checkbox("Show numeric labels (0/1/2)", value=False)

# ----------------------------
# Row 1: Scores + class distribution
# ----------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Model Scores")
    st.dataframe(scores)

with col2:
    st.subheader("Class Distribution (True Regimes)")
    # counts of true labels
    counts = preds["y_true"].value_counts().sort_index()
    counts_named = counts.rename(index=LABEL_MAP)

    fig = plt.figure()
    plt.bar(counts_named.index, counts_named.values)
    plt.ylabel("Count")
    plt.xlabel("Regime")
    st.pyplot(fig)

    st.caption(
        "If one class is much more common (often 'Up'), a baseline that always predicts it can look good on accuracy."
    )

# ----------------------------
# Row 2: Time-series predictions (line chart)
# ----------------------------
st.subheader("Predictions Over Time")

plot_df = preds[["y_true", "pred_lr", "pred_rf", "pred_base"]].copy()

if not show_raw_labels:
    for c in plot_df.columns:
        plot_df[c] = decode(plot_df[c])

st.line_chart(plot_df)

st.caption("Walk-forward: train on all data up to month t, predict regime for month t+1.")

# ----------------------------
# Row 3: Confusion Matrix for selected model
# ----------------------------
st.subheader("Confusion Matrix")

# Drop NaNs just in case + align
tmp = preds[["y_true", model_choice]].dropna()
y_true = tmp["y_true"].astype(int).to_numpy()
y_pred = tmp[model_choice].astype(int).to_numpy()

fig2, ax2 = plt.subplots()
ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    display_labels=["Down", "Flat", "Up"],
    values_format="d",
    ax=ax2,
    colorbar=False,
)
ax2.set_title("Confusion Matrix (Selected Model)")
st.pyplot(fig2, clear_figure=True)
# ----------------------------
# Optional: quick metrics for selected model
# ----------------------------
st.subheader("Quick Summary (Selected Model)")

acc = (y_true == y_pred).mean()

# macro f1 without importing extra: compute per-class f1 manually (simple)
f1s = []
for cls in [0, 1, 2]:
    tp = np.sum((y_true == cls) & (y_pred == cls))
    fp = np.sum((y_true != cls) & (y_pred == cls))
    fn = np.sum((y_true == cls) & (y_pred != cls))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f1s.append(f1)

macro_f1 = float(np.mean(f1s))

st.write(
    {
        "Selected model": model_choice,
        "Accuracy": round(float(acc), 4),
        "Macro F1": round(float(macro_f1), 4),
        "Note": "Macro F1 weights Down/Flat/Up equally, so it’s more informative than accuracy when classes are imbalanced.",
    }
)

st.caption("Label meaning: 0=Down, 1=Flat, 2=Up.")
