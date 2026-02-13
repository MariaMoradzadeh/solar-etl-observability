
import argparse
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import yaml

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_baseline(cfg: dict) -> dict:
    p = Path(cfg["baseline_stats"]["output_file"])
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--windows", nargs="+", type=int, default=[6, 12, 24, 48])
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    baseline = load_baseline(cfg)
    mu0 = float(baseline["P"]["mean"])
    sigma0 = float(baseline["P"]["std"])

    dt = int(cfg["experiment"]["sampling_minutes"])
    inj_root = Path(cfg["paths"]["injected_dir"])
    out_root = Path(cfg["paths"]["results_dir"]) / "edge_timegrid"
    ensure_dir(out_root)

    th = cfg["edge"]["thresholds"]
    min_c = float(th.get("min_completeness", 0.9995))
    late_m = float(th.get("late_minutes", 30))
    late_p = float(th.get("late_ratio", 0.05))

    baseline = load_baseline(cfg)

    dt = int(cfg["experiment"]["sampling_minutes"])
    inj_root = Path(cfg["paths"]["injected_dir"])
    out_root = Path(cfg["paths"]["results_dir"]) / "edge_timegrid"
    ensure_dir(out_root)

    th = cfg["edge"]["thresholds"]
    min_c = float(th.get("min_completeness", 0.9995))
    late_m = float(th.get("late_minutes", 30))
    late_p = float(th.get("late_ratio", 0.05))

    # اگر این‌ها رو داری در config، به همان‌ها وصل می‌شیم
    kpi_cfg = cfg.get("edge", {}).get("kpis", {})

    for scenario in args.scenarios:
        for seed in range(1, args.seeds + 1):
            df = pd.read_parquet(inj_root / scenario / f"seed_{seed}.parquet")
            df["event_time"] = pd.to_datetime(df["event_time"])
            df["arrival_time"] = pd.to_datetime(df["arrival_time"])
            df = df.sort_values("event_time").reset_index(drop=True)

            # Time grid
            t_min = df["event_time"].min()
            t_max = df["event_time"].max()
            grid = pd.date_range(t_min, t_max, freq=f"{dt}min")

            # Index by event_time for fast slicing
            df = df.set_index("event_time", drop=False)

            for w in args.windows:
                rows = []
                win_minutes = (w - 1) * dt

                # CUSUM state (reset per scenario/seed/window)
                s_pos = 0.0
                s_neg = 0.0
               
                for end_time in grid:
                    start_time = end_time - pd.Timedelta(minutes=win_minutes)
                    df_w = df.loc[start_time:end_time]

                    expected = w  # دقیقاً w نقطه زمانی باید باشد
                    observed = int(df_w.index.nunique())
                    completeness = float(observed / expected) if expected > 0 else 0.0

                    # Late arrivals
                    late = ((df_w["arrival_time"] - df_w["event_time"]).dt.total_seconds() / 60.0) > late_m
                    late_ratio = float(np.mean(late)) if len(df_w) else 0.0
                    window_mean_P = float(df_w["P"].mean()) if len(df_w) else float("nan")
                    # CUSUM drift (gradual change) on window_mean_P
                    drift = False
                    if np.isfinite(window_mean_P):
                        x0 = mu0  # baseline mean for P
                        k = 0.5 * sigma0
                        h = 5.0 * sigma0

                        s_pos = max(0.0, s_pos + ((window_mean_P - x0) - k))
                        s_neg = min(0.0, s_neg + ((window_mean_P - x0) + k))

                        if s_pos > h or abs(s_neg) > h:
                            drift = True
                    # Simple anomaly baseline (keep compatible)
                    # (اگر ستون power_kw داری، روی آن نگاه می‌کنیم)
                    anomaly = False
                    if "P" in df_w.columns and len(df_w):
                        x = df_w["P"].to_numpy(dtype=float)
                        
                        mu = np.nanmean(x)
                        sd = np.nanstd(x) + 1e-9
                        z = np.abs((x - mu) / sd)
                        anomaly = bool(np.any(z > 3.0))

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

                    rows.append(
                        {
                            "window_end_time": end_time,
                            "window_mean_P": window_mean_P,
                            "alert_type": alert_type,
                            "reason": reason,
                            "completeness": completeness,
                            "late_ratio": late_ratio,
                        }
                    )

                out_dir = out_root / scenario / f"seed_{seed}"
                ensure_dir(out_dir)
                out_path = out_dir / f"w_{w}.parquet"
                pd.DataFrame(rows).to_parquet(out_path, index=False)
                print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
