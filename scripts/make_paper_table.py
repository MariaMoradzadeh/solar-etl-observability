from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path("paper/results")
INPUTS = {
    "S1_missing_burst": ROOT / "S1_summary.csv",
    "S3_late_arrivals": ROOT / "S3_summary.csv",
    "S8_efficiency_drop": ROOT / "S8_summary.csv",
}

OUT_DIR = Path("paper/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure numeric columns are numeric (safe coercion)
    for c in [
        "w",
        "precision",
        "recall",
        "f1",
        "false_positives_per_day",
        "mean_delay_minutes",
        "median_delay_minutes",
        "p90_delay_minutes",
        "p95_delay_minutes",
        "tp_windows",
        "fp_windows",
        "fn_events",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def pick_best(df: pd.DataFrame) -> pd.Series:
    """
    Heuristic for paper:
    1) Prefer recall==1 (if exists)
    2) Among those: max F1
    3) Tie-breaker: min false_positives_per_day
    4) Tie-breaker: min median_delay_minutes (if present)
    """
    df2 = df.copy()

    if "recall" in df2.columns:
        df_r1 = df2[df2["recall"] >= 0.999999]
        if len(df_r1) > 0:
            df2 = df_r1

    sort_cols = []
    asc = []

    if "f1" in df2.columns:
        sort_cols.append("f1"); asc.append(False)
    if "false_positives_per_day" in df2.columns:
        sort_cols.append("false_positives_per_day"); asc.append(True)
    if "median_delay_minutes" in df2.columns:
        sort_cols.append("median_delay_minutes"); asc.append(True)

    if sort_cols:
        df2 = df2.sort_values(sort_cols, ascending=asc)

    return df2.iloc[0]


def main() -> None:
    rows = []
    long = []

    for scenario, path in INPUTS.items():
        df = load_one(path)
        df["scenario"] = scenario
        long.append(df)

        best = pick_best(df)
        rows.append(best)

    df_best = pd.DataFrame(rows).reset_index(drop=True)

    # Keep only the columns we want in the paper table (edit freely later)
    cols = [
        "scenario",
        "w",
        "target",
        "precision",
        "recall",
        "f1",
        "false_positives_per_day",
        "median_delay_minutes",
        "p90_delay_minutes",
        "p95_delay_minutes",
        "tp_windows",
        "fp_windows",
        "fn_events",
    ]
    cols = [c for c in cols if c in df_best.columns]
    df_best = df_best[cols]

    # Write CSV
    out_csv = OUT_DIR / "table_best_windows.csv"
    df_best.to_csv(out_csv, index=False)

    # Write Markdown table (nice for paper draft / README)
    out_md = OUT_DIR / "table_best_windows.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(df_best.to_markdown(index=False))

    # Also write a combined “all results” CSV for backup
    df_all = pd.concat(long, ignore_index=True)
    out_all = OUT_DIR / "all_results_long.csv"
    df_all.to_csv(out_all, index=False)

    print(f"✅ Wrote: {out_csv}")
    print(f"✅ Wrote: {out_md}")
    print(f"✅ Wrote: {out_all}")


if __name__ == "__main__":
    main()
