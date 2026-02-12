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


def load_labels(cfg: dict, scenario: str, seed: int) -> pd.DataFrame:
    inj_dir = Path(cfg["paths"]["injected_dir"]) / scenario
    lab_path = inj_dir / f"seed_{seed}_labels.parquet"
    lab = pd.read_parquet(lab_path)
    if len(lab) == 0:
        return lab
    lab["t0"] = pd.to_datetime(lab["t0"])
    lab["t1"] = pd.to_datetime(lab["t1"])
    return lab


def load_edge(cfg: dict, scenario: str, seed: int, w: int) -> pd.DataFrame:
    p = Path(cfg["paths"]["results_dir"]) / "edge_timegrid" / scenario / f"seed_{seed}" / f"w_{w}.parquet"
    df = pd.read_parquet(p)
    if "window_end_time" in df.columns:
        df["window_end_time"] = pd.to_datetime(df["window_end_time"])
    return df


def first_detection_time(edge: pd.DataFrame, t0: pd.Timestamp, target: str) -> pd.Timestamp | None:
    m = (edge["window_end_time"] >= t0) & (edge["alert_type"] == target)
    if not m.any():
        return None
    return edge.loc[m, "window_end_time"].min()


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def fp_per_day(edge: pd.DataFrame, labels: pd.DataFrame, target: str) -> float:
    if len(edge) == 0:
        return 0.0
    pos = edge[edge["alert_type"] == target].copy()
    if len(pos) == 0:
        return 0.0

    intervals = []
    if len(labels) > 0:
        intervals = labels[labels["label"] == target][["t0", "t1"]].to_records(index=False)

    def is_true(t):
        return any((t >= i[0]) and (t < i[1]) for i in intervals)

    fp_count = sum(1 for t in pos["window_end_time"] if not is_true(t))

    span_days = (edge["window_end_time"].max() - edge["window_end_time"].min()).total_seconds() / 86400.0
    span_days = span_days if span_days > 0 else 1.0
    return float(fp_count / span_days)


def make_delay_cdf(delays: list[float]) -> pd.DataFrame:
    if not delays:
        return pd.DataFrame({"delay_minutes": [], "cdf": []})
    x = np.sort(np.array(delays, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    return pd.DataFrame({"delay_minutes": x, "cdf": y})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--windows", nargs="+", type=int, default=[6, 12, 24, 48])
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = Path(cfg["paths"]["results_dir"]) / "metrics_timegrid"
    ensure_dir(out_root)

    expected = cfg["evaluation"]["expected_label"]

    for scenario in args.scenarios:
        target = expected[scenario]  # "fault" or "anomaly"
        scenario_rows = []
        all_delays_by_w = {w: [] for w in args.windows}

        for w in args.windows:
            tp_sum = fp_sum = fn_sum = 0
            fpday_vals = []
            delays_all: list[float] = []

            for seed in range(1, args.seeds + 1):
                labels = load_labels(cfg, scenario, seed)
                edge = load_edge(cfg, scenario, seed, w)

                # event-level FN + delays
                events = labels[labels["label"] == target] if len(labels) else labels
                for _, ev in events.iterrows():
                    t_det = first_detection_time(edge, ev["t0"], target)
                    if t_det is None:
                        fn_sum += 1
                    else:
                        delays_all.append(float((t_det - ev["t0"]).total_seconds() / 60.0))

                # window-level TP/FP (rough but OK for MVP)
                pos = edge[edge["alert_type"] == target]
                if len(pos) > 0 and len(labels) > 0:
                    intervals = labels[labels["label"] == target][["t0", "t1"]].to_records(index=False)
                    for t in pos["window_end_time"]:
                        inside = any((t >= i[0]) and (t < i[1]) for i in intervals)
                        if inside:
                            tp_sum += 1
                        else:
                            fp_sum += 1
                else:
                    fp_sum += len(pos)

                fpday_vals.append(fp_per_day(edge, labels, target))

            prec, rec, f1 = precision_recall_f1(tp_sum, fp_sum, fn_sum)
            fpday = float(np.mean(fpday_vals)) if fpday_vals else 0.0

            if delays_all:
                d = np.array(delays_all, dtype=float)
                mean_d = float(np.mean(d))
                med_d = float(np.median(d))
                p90 = float(np.percentile(d, 90))
                p95 = float(np.percentile(d, 95))
            else:
                mean_d = med_d = p90 = p95 = float("nan")

            scenario_rows.append(
                {
                    "scenario": scenario,
                    "w": w,
                    "target": target,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "false_positives_per_day": fpday,
                    "mean_delay_minutes": mean_d,
                    "median_delay_minutes": med_d,
                    "p90_delay_minutes": p90,
                    "p95_delay_minutes": p95,
                    "tp_windows": tp_sum,
                    "fp_windows": fp_sum,
                    "fn_events": fn_sum,
                }
            )
            all_delays_by_w[w] = delays_all

        out_dir = out_root / scenario
        ensure_dir(out_dir)

        summary = pd.DataFrame(scenario_rows).sort_values("w")
        summary.to_csv(out_dir / "summary.csv", index=False)

        for w, delays in all_delays_by_w.items():
            cdf = make_delay_cdf(delays)
            cdf.to_csv(out_dir / f"delay_cdf_w{w}.csv", index=False)

        print(f"✅ Wrote metrics for {scenario} -> {out_dir}")


if __name__ == "__main__":
    main()
