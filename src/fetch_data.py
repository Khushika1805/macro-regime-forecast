import os
import pandas as pd
import yfinance as yf
from fredapi import Fred

def fetch_spy_monthly(start="1993-01-01"):
    spy = yf.download("SPY", start=start, auto_adjust=True, progress=False)

    # yfinance sometimes returns MultiIndex columns depending on version
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] for c in spy.columns]  # flatten

    close_col = "Close" if "Close" in spy.columns else "Adj Close"
    spy = spy[[close_col]].rename(columns={close_col: "spy_close"})

    spy_m = spy.resample("ME").last()
    return spy_m

def fetch_fred_series(fred, series_id, name):
    s = fred.get_series(series_id)
    s = pd.Series(s, name=name)
    s.index = pd.to_datetime(s.index)
    # convert to month-end to align with SPY
    s_m = s.resample("ME").last()
    return s_m

def main():
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("Missing FRED_API_KEY env var.")
    fred = Fred(api_key=key)

    spy_m = fetch_spy_monthly()

    # Common, strong macro series (monthly or daily -> month-end)
    cpi = fetch_fred_series(fred, "CPIAUCSL", "cpi")              # CPI index
    unrate = fetch_fred_series(fred, "UNRATE", "unrate")          # unemployment rate (%)
    ffr = fetch_fred_series(fred, "FEDFUNDS", "ffr")              # fed funds rate (%)
    gs10 = fetch_fred_series(fred, "GS10", "gs10")                # 10y treasury (daily)
    gs2 = fetch_fred_series(fred, "GS2", "gs2")                   # 2y treasury (daily)
    indpro = fetch_fred_series(fred, "INDPRO", "indpro")          # industrial production index

    df = pd.concat([spy_m, cpi, unrate, ffr, gs10, gs2, indpro], axis=1).dropna()

    df.index.name = "DATE"
    df.to_csv("data/raw_monthly.csv")

if __name__ == "__main__":
    main()