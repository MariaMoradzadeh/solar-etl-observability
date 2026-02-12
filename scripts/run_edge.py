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


def load_baseline(cfg: dict) -> dict:
    p = Path(cfg["baseline_stats"]["output_file"])
    return json.loads(p.read_text(encoding="utf-8"))


def freshness_score(df_w: pd.DataFrame, delta_max_minutes: float) -> float:
    delta = (df_w["arrival_time"] - df_w["event_time"]).dt.total_seconds() / 60.0
    delta = np.clip(delta, 0.0, delta_max_minutes)
    return float(1.0 - np.mean(delta / delta_max_minutes)) if len(delta) > 0 else 0.0


def completeness_score(n_obs: int, w: int) -> float:
    return float(n_obs / w) if w > 0 else 0.0


def duplicate_rate(df_w: pd.DataFrame) -> float:
    n = len(df_w)
    if n == 0:
        return 0.0
    u = df_w["event_time"].nunique()
    return float(1.0 - (u / n))


def variance_score(df_w: pd.DataFrame, baseline: dict) -> float:
    scores = []
    for col in ["P", "G", "T"]:
        v_ref = baseline[col]["var"]
        v_ref = v_ref if v_ref > 0 else 1e-9
        v = float(df_w[col].var(ddof=1)) if len(df_w) >= 2 else 0.0
        scores.append(abs((v - v_ref) / v_ref))
    return float(np.mean(scores))


def drift_score(df_w: pd.DataFrame, baseline: dict) -> float:
    scores = []
    for col in ["P", "G", "T"]:
        mu = float(df_w[col].mean()) if len(df_w) > 0 else 0.0
        mu_ref = baseline[col]["mean"]
        sd_ref = baseline[col]["std"] if baseline[col]["std"] > 0 else 1e-9
        scores.append(abs((mu - mu_ref) / sd_ref))
    return float(np.mean(scores))


def decide_alert(kpis: dict, cfg: dict) -> tuple[str, str]:
    th = cfg["edge"]["thresholds"]
    tau_v = float(th["tau_v"])
    tau_r = float(th["tau_r"])

    reasons = []
    alert = "none"

    # Fault rules
    if kpis["completeness"] < 1.0:
        alert = "fault"
        reasons.append("completeness<1.0")
    if kpis["duplicate_rate"] > 0.0:
        alert = "fault"
        reasons.append("duplicates>0")

    # Anomaly rules (only if not already fault)
    if alert == "none":
        if kpis["variance_score"] > tau_v:
            alert = "anomaly"
            reasons.append(f"variance_score>{tau_v}")
        if kpis["drift_score"] > tau_r:
            alert = "anomaly"
            reasons.append(f"drift_score>{tau_r}")

    return alert, ";".join(reasons) if reasons else ""


def run_edge_on_file(cfg: dict, scenario: str, seed: int, w: int, baseline: dict) -> Path:
    inj_dir = Path(cfg["paths"]["injected_dir"]) / scenario
    df = pd.read_parquet(inj_dir / f"seed_{seed}.parquet")
    df["event_time"] = pd.to_datetime(df["event_time"])
    df["arrival_time"] = pd.to_datetime(df["arrival_time"])
    df = df.sort_values("event_time").reset_index(drop=True)

    delta_max = float(cfg["edge"]["kpis"]["freshness"]["delta_max_minutes"])
    out_rows = []

    for end_idx in range(w - 1, len(df)):
        df_w = df.iloc[end_idx - w + 1 : end_idx + 1]
        window_end_time = df_w["event_time"].iloc[-1]

        kpis = {
            "freshness": freshness_score(df_w, delta_max),
            "completeness": completeness_score(len(df_w), w),
            "duplicate_rate": duplicate_rate(df_w),
            "variance_score": variance_score(df_w, baseline),
            "drift_score": drift_score(df_w, baseline),
        }

        alert_type, reason = decide_alert(kpis, cfg)

        out_rows.append(
            {
                "window_end_time": window_end_time,
                "w": w,
                **kpis,
                "alert_type": alert_type,
                "reason": reason,
            }
        )

    out = pd.DataFrame(out_rows)
    out_dir = Path(cfg["paths"]["results_dir"]) / "edge" / scenario / f"seed_{seed}"
    ensure_dir(out_dir)
    out_path = out_dir / f"w_{w}.parquet"
    out.to_parquet(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--windows", nargs="+", type=int, default=[6, 12, 24, 48])
    args = ap.parse_args()

    cfg = load_config(args.config)
    baseline = load_baseline(cfg)

    for scenario in args.scenarios:
        for seed in range(1, args.seeds + 1):
            for w in args.windows:
                out_path = run_edge_on_file(cfg, scenario, seed, w, baseline)
                print(f"Wrote: {out_path}")

    print("✅ Edge done.")


if __name__ == "__main__":
    main()
