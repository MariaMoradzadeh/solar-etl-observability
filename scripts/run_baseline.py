from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

def load_cfg(p: str) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def zscore_flags(x: np.ndarray, thr: float = 3.0) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x) + 1e-9
    z = (x - mu) / sd
    return np.abs(z) > thr

def expected_count(t_start: pd.Timestamp, t_end: pd.Timestamp, sampling_minutes: int) -> int:
    dt = pd.Timedelta(minutes=sampling_minutes)
    if t_end < t_start:
        return 0
    return int(((t_end - t_start) / dt)) + 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--windows", nargs="+", type=int, default=[6,12,24,48])
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    inj_root = Path(cfg["paths"]["injected_dir"])
    out_root = Path(cfg["paths"]["results_dir"]) / "baseline"
    ensure_dir(out_root)

    sampling_minutes = int(cfg["experiment"]["sampling_minutes"])
    th = cfg["edge"]["thresholds"]
    min_c = float(th.get("min_completeness", 0.98))
    late_m = float(th.get("late_minutes", 30))
    late_p = float(th.get("late_ratio", 0.05))

    for scenario in args.scenarios:
        for seed in range(1, args.seeds + 1):
            df = pd.read_parquet(inj_root / scenario / f"seed_{seed}.parquet")
            df["event_time"] = pd.to_datetime(df["event_time"])
            df["arrival_time"] = pd.to_datetime(df["arrival_time"])
            df = df.sort_values("event_time").reset_index(drop=True)

            for w in args.windows:
                rows = []
                if len(df) < w:
                    continue

                for end in range(w - 1, len(df)):
                    df_w = df.iloc[end - w + 1 : end + 1]
                    t_end = df_w["event_time"].max()
                    t_start = t_end - pd.Timedelta(minutes=(w - 1) * sampling_minutes)

                    exp = expected_count(t_start, t_end, sampling_minutes)
                    obs = int(df_w["event_time"].nunique())
                    completeness = float(obs / exp) if exp > 0 else 0.0

                    late = ((df_w["arrival_time"] - df_w["event_time"]).dt.total_seconds() / 60.0) > late_m
                    late_ratio = float(np.mean(late)) if len(df_w) else 0.0

                    # anomaly baseline using z-score on power_kw if exists
                    col = "power_kw" if "power_kw" in df_w.columns else None
                    anomaly = False
                    if col is not None:
                        anomaly = bool(np.any(zscore_flags(df_w[col].to_numpy(dtype=float), thr=3.0)))

                    alert_type = "none"
                    reason = ""

                    if completeness < min_c:
                        alert_type = "fault"
                        reason = f"completeness<{min_c}"
                    elif late_ratio > late_p:
                        alert_type = "fault"
                        reason = f"late_ratio>{late_p}"
                    elif anomaly:
                        alert_type = "anomaly"
                        reason = "zscore>3"

                    rows.append({
                        "window_end_time": t_end,
                        "alert_type": alert_type,
                        "reason": reason,
                        "completeness": completeness,
                        "late_ratio": late_ratio,
                    })

                out_dir = out_root / scenario / f"seed_{seed}"
                ensure_dir(out_dir)
                out_path = out_dir / f"w_{w}.parquet"
                pd.DataFrame(rows).to_parquet(out_path, index=False)
                print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
