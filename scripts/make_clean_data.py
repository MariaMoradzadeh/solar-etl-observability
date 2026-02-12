from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def make_time_index(start_date: str, horizon_days: int, sampling_minutes: int, tz: str = "UTC") -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date, tz=tz)
    end = start + pd.Timedelta(days=horizon_days)
    freq = f"{sampling_minutes}min"
    return pd.date_range(start=start, end=end, freq=freq, inclusive="left")


def synthetic_solar(cfg: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    p = cfg["generator"]["params"]
    rng = np.random.default_rng(123)

    seconds_in_day = 24 * 3600
    tsec = (index.view("int64") // 10**9)
    phase = float(p.get("daily_phase", 0.0))
    omega = 2 * np.pi / seconds_in_day
    s = np.maximum(0.0, np.sin(omega * tsec + phase))

    # Cloud AR(1)
    rho = float(p.get("ghi_cloud_ar1_rho", 0.8))
    u = np.zeros(len(index))
    eta = rng.normal(0, 0.15, size=len(index))
    for i in range(1, len(index)):
        u[i] = rho * u[i - 1] + eta[i]
    u = np.clip(u, -0.6, 0.6)

    A_g = float(p.get("ghi_scale", 800.0))
    ghi_noise = rng.normal(0, float(p.get("ghi_noise_std", 20.0)), size=len(index))
    G = A_g * s * (1.0 + u) + ghi_noise
    G = np.clip(G, 0.0, None)

    T0 = float(p.get("temp_base", 10.0))
    AT = float(p.get("temp_amp", 15.0))
    T_noise = rng.normal(0, float(p.get("temp_noise_std", 1.5)), size=len(index))
    T = T0 + AT * s + T_noise

    alpha = float(p.get("alpha", 0.0012))
    beta = float(p.get("beta", 0.004))
    Tref = float(p.get("tref", 25.0))
    P_noise = rng.normal(0, float(p.get("power_noise_std", 0.05)), size=len(index))
    P = alpha * G * (1.0 - beta * (T - Tref)) + P_noise
    P = np.clip(P, 0.0, None)

    df = pd.DataFrame({"event_time": index, "P": P, "G": G, "T": T})
    df["arrival_time"] = df["event_time"]  # clean: on-time
    return df


def compute_baseline_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for col in ["P", "G", "T"]:
        mu = float(df[col].mean())
        sd = float(df[col].std(ddof=1) if df[col].std(ddof=1) > 0 else 1e-9)
        var = float(df[col].var(ddof=1))
        stats[col] = {"mean": mu, "std": sd, "var": var}
    return stats


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    # Read from YAML structure you already have
    gen_dir = Path(cfg["paths"]["generated_dir"])
    ensure_dir(gen_dir)

    tz = cfg["time"].get("timezone", "UTC")
    start_date = cfg["time"]["start_date"]
    horizon_days = int(cfg["experiment"]["horizon_days"])
    sampling_minutes = int(cfg["experiment"]["sampling_minutes"])

    index = make_time_index(start_date, horizon_days, sampling_minutes, tz)
    df = synthetic_solar(cfg, index)

    clean_path = gen_dir / "clean.parquet"
    df.to_parquet(clean_path, index=False)

    baseline = compute_baseline_stats(df)
    baseline_path = Path(cfg["baseline_stats"]["output_file"])
    ensure_dir(baseline_path.parent)
    baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    print(f"✅ Wrote: {clean_path}")
    print(f"✅ Wrote: {baseline_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
