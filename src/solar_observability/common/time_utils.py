from __future__ import annotations
import pandas as pd


def make_time_index(start_date: str, horizon_days: int, sampling_minutes: int, tz: str = "UTC") -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date, tz=tz)
    end = start + pd.Timedelta(days=horizon_days)
    freq = f"{sampling_minutes}min"
    return pd.date_range(start=start, end=end, freq=freq, inclusive="left")
