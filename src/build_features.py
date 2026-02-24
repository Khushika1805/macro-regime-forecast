import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # SPY monthly returns
    df["spy_ret"] = df["spy_close"].pct_change()

    # CPI YoY inflation (%)
    df["infl_yoy"] = (df["cpi"] / df["cpi"].shift(12) - 1.0) * 100

    # Industrial production YoY (%)
    df["indpro_yoy"] = (df["indpro"] / df["indpro"].shift(12) - 1.0) * 100

    # Yield curve slope (10y - 2y)
    df["yc_slope"] = df["gs10"] - df["gs2"]

    # Changes (momentum) in rates / unemployment
    df["ffr_chg_3m"] = df["ffr"] - df["ffr"].shift(3)
    df["unrate_chg_3m"] = df["unrate"] - df["unrate"].shift(3)

    # Lag all macro features by 1 month so they represent info available at end of month t
    macro_cols = ["infl_yoy", "unrate", "ffr", "yc_slope", "indpro_yoy", "ffr_chg_3m", "unrate_chg_3m"]
    for c in macro_cols:
        df[c] = df[c].shift(1)

    # Target is next-month return
    df["spy_ret_fwd1"] = df["spy_ret"].shift(-1)

    # Regime labels (3-class) based on forward return
    down_thr = -0.01
    up_thr = 0.01
    df["regime"] = np.select(
        [df["spy_ret_fwd1"] < down_thr, df["spy_ret_fwd1"] > up_thr],
        [0, 2],
        default=1
    )
    # 0=down, 1=flat, 2=up

    # Keep only usable rows
    df = df.dropna()

    feature_cols = macro_cols + ["spy_ret"]  # include last month return as a baseline-ish feature (lagged already)
    # Ensure spy_ret is lagged too (it is at t, and target is t+1, so that's OK)
    df = df[feature_cols + ["regime", "spy_ret_fwd1"]]

    return df

def main():
    df = pd.read_csv("data/raw_monthly.csv")

    # If the date got saved as a column, set it as index
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.set_index("DATE")
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    elif "Unnamed: 0" in df.columns:
        df["Unnamed: 0"] = pd.to_datetime(df["Unnamed: 0"])
        df = df.set_index("Unnamed: 0")

    # Normalize SPY close column name
    if "spy_close" not in df.columns:
        if "Close" in df.columns:
            df = df.rename(columns={"Close": "spy_close"})
        elif "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "spy_close"})
        else:
            raise KeyError(f"Could not find SPY close column. Columns: {df.columns.tolist()}")

    feat = add_features(df)
    feat.to_csv("data/features.csv")
    print("Saved data/features.csv", feat.shape)

if __name__ == "__main__":
    main()