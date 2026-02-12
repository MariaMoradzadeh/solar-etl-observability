from __future__ import annotations
import argparse
from pathlib import Path
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


def ewma_cusum_down(y: np.ndarray, lam: float, k: float, h: float) -> np.ndarray:
    """Returns boolean array alarms for downward drift."""
    z = np.empty_like(y, dtype=float)
    s = np.empty_like(y, dtype=float)
    alarms = np.zeros_like(y, dtype=bool)

    z0 = y[0]
    z[0] = z0
    s[0] = 0.0

    for i in range(1, len(y)):
        z[i] = lam * y[i] + (1 - lam) * z[i - 1]
        s[i] = max(0.0, s[i - 1] + (k - z[i]))
        alarms[i] = s[i] > h

    return alarms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenario", default="S8_efficiency_drop")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--windows", nargs="+", type=int, default=[6, 12, 24, 48])
    ap.add_argument("--signal", default="power_kw")
    ap.add_argument("--lambda_", type=float, default=0.1)
    ap.add_argument("--k", type=float, default=0.5)
    ap.add_argument("--h", type=float, default=8.0)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    baseline = load_baseline(cfg)

    mu0 = float(baseline.get(args.signal, {}).get("mean", np.nan))
    sd0 = float(baseline.get(args.signal, {}).get("std", np.nan))
    if not np.isfinite(mu0) or not np.isfinite(sd0) or sd0 <= 0:
        raise ValueError(f"Baseline stats for '{args.signal}' missing/invalid. Check baseline_stats.json")

    edge_root = Path(cfg["paths"]["results_dir"]) / "edge_timegrid"
    out_root = Path(cfg["paths"]["results_dir"]) / "edge_timegrid_drift"
    ensure_dir(out_root)

    for seed in range(1, args.seeds + 1):
        for w in args.windows:
            p = edge_root / args.scenario / f"seed_{seed}" / f"w_{w}.parquet"
            df = pd.read_parquet(p)

            # برای drift نیاز داریم window-level mean از سیگنال
            # اگر run_edge_timegrid سیگنال را داخل خروجی ذخیره نکرده، باید آن را اضافه کنیم.
            # فعلاً فرض می‌کنیم ستون 'window_mean' از قبل هست؛ اگر نیست، خطا می‌گیری و می‌گیم چطور اضافه کنی.
            if "window_mean" not in df.columns:
                raise KeyError("Missing column 'window_mean' in edge_timegrid output. We must add it in run_edge_timegrid.py.")

            y = (df["window_mean"].to_numpy(dtype=float) - mu0) / sd0
            alarms = ewma_cusum_down(y, lam=args.lambda_, k=args.k, h=args.h)

            df = df.copy()
            df["drift_alarm"] = alarms
            # اگر قبلاً anomaly داشته‌ای، اینجا ترکیب می‌کنیم:
            df.loc[df["drift_alarm"], "alert_type"] = "anomaly"
            df.loc[df["drift_alarm"], "reason"] = df.get("reason", "").astype(str) + "|ewma_cusum_down"

            out_dir = out_root / args.scenario / f"seed_{seed}"
            ensure_dir(out_dir)
            out_path = out_dir / f"w_{w}.parquet"
            df.to_parquet(out_path, index=False)
            print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
