from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_clean(cfg: dict) -> pd.DataFrame:
    clean_path = Path(cfg["paths"]["generated_dir"]) / "clean.parquet"
    df = pd.read_parquet(clean_path)
    df["event_time"] = pd.to_datetime(df["event_time"])
    df["arrival_time"] = pd.to_datetime(df["arrival_time"])
    return df


def pick_random_start(rng: np.random.Generator, times: pd.Series, duration_samples: int) -> pd.Timestamp:
    max_i = len(times) - duration_samples - 1
    if max_i <= 1:
        return times.iloc[0]
    i = int(rng.integers(0, max_i))
    return times.iloc[i]


def make_labels(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["scenario", "label", "t0", "t1", "severity"])
    return pd.DataFrame(rows, columns=["scenario", "label", "t0", "t1", "severity"])


def inject_missing_burst(cfg: dict, df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    sc = next(s for s in cfg["fault_injection"]["scenarios"] if s["id"] == "S1_missing_burst")
    durations = [int(x) for x in sc["params"]["duration_samples"]]
    occ = int(sc["params"]["occurrences_per_seed"])
    times = df["event_time"]

    labels = []
    out = df.copy()

    for _ in range(occ):
        d = int(rng.choice(durations))
        t0 = pick_random_start(rng, times, d)
        t1 = t0 + pd.Timedelta(minutes=int(cfg["experiment"]["sampling_minutes"]) * d)

        mask = (out["event_time"] >= t0) & (out["event_time"] < t1)
        out = out.loc[~mask].copy()

        labels.append(
            {
                "scenario": "S1_missing_burst",
                "label": "fault",
                "t0": t0,
                "t1": t1,
                "severity": f"duration_samples={d}",
            }
        )

    return out, make_labels(labels)


def inject_late_arrivals(cfg: dict, df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    sc = next(s for s in cfg["fault_injection"]["scenarios"] if s["id"] == "S3_late_arrivals")
    late_pct = float(sc["params"]["late_percentage"])
    delays = [int(x) for x in sc["params"]["delay_minutes"]]

    out = df.copy()
    n = len(out)
    k = int(np.floor(late_pct * n))
    if k <= 0:
        return out, make_labels([])

    idx = rng.choice(np.arange(n), size=k, replace=False)
    chosen_delay = rng.choice(delays, size=k, replace=True)

    out.loc[idx, "arrival_time"] = out.loc[idx, "event_time"] + pd.to_timedelta(chosen_delay, unit="m")

    # MVP label: mark broad interval covering late arrivals
    t0 = out.loc[idx, "event_time"].min()
    t1 = out.loc[idx, "event_time"].max() + pd.Timedelta(minutes=int(cfg["experiment"]["sampling_minutes"]))

    labels = [
        {
            "scenario": "S3_late_arrivals",
            "label": "fault",
            "t0": t0,
            "t1": t1,
            "severity": f"late_pct={late_pct},delays={sorted(set(delays))}",
        }
    ]
    return out, make_labels(labels)


def inject_efficiency_drop(cfg: dict, df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    sc = next(s for s in cfg["fault_injection"]["scenarios"] if s["id"] == "S8_efficiency_drop")
    drop = float(sc["params"]["drop_percent"]) / 100.0
    durations = [int(x) for x in sc["params"]["duration_samples"]]
    occ = int(sc["params"]["occurrences_per_seed"])

    day_thr = float(sc["params"]["day_condition"]["value"])
    out = df.copy()
    labels = []

    candidates = out.loc[out["G"] > day_thr, "event_time"].reset_index(drop=True)
    if len(candidates) < max(durations) + 10:
        return out, make_labels([])

    for _ in range(occ):
        d = int(rng.choice(durations))
        t0 = pick_random_start(rng, candidates, d)
        t1 = t0 + pd.Timedelta(minutes=int(cfg["experiment"]["sampling_minutes"]) * d)

        mask = (out["event_time"] >= t0) & (out["event_time"] < t1) & (out["G"] > day_thr)
        out.loc[mask, "P"] = out.loc[mask, "P"] * (1.0 - drop)

        labels.append(
            {
                "scenario": "S8_efficiency_drop",
                "label": "anomaly",
                "t0": t0,
                "t1": t1,
                "severity": f"drop={int(drop*100)}%,duration_samples={d}",
            }
        )

    return out, make_labels(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenarios", nargs="+", default=["S1", "S3", "S8"])
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    injected_root = Path(cfg["paths"]["injected_dir"])
    ensure_dir(injected_root)

    clean = load_clean(cfg)
    scenarios = set(args.scenarios)

    for seed in range(1, args.seeds + 1):
        rng = np.random.default_rng(seed)

        if "S1" in scenarios:
            df1, lab1 = inject_missing_burst(cfg, clean, rng)
            out_dir = injected_root / "S1_missing_burst"
            ensure_dir(out_dir)
            df1.to_parquet(out_dir / f"seed_{seed}.parquet", index=False)
            lab1.to_parquet(out_dir / f"seed_{seed}_labels.parquet", index=False)

        if "S3" in scenarios:
            df3, lab3 = inject_late_arrivals(cfg, clean, rng)
            out_dir = injected_root / "S3_late_arrivals"
            ensure_dir(out_dir)
            df3.to_parquet(out_dir / f"seed_{seed}.parquet", index=False)
            lab3.to_parquet(out_dir / f"seed_{seed}_labels.parquet", index=False)

        if "S8" in scenarios:
            df8, lab8 = inject_efficiency_drop(cfg, clean, rng)
            out_dir = injected_root / "S8_efficiency_drop"
            ensure_dir(out_dir)
            df8.to_parquet(out_dir / f"seed_{seed}.parquet", index=False)
            lab8.to_parquet(out_dir / f"seed_{seed}_labels.parquet", index=False)

    print("✅ Injection done.")


if __name__ == "__main__":
    main()
