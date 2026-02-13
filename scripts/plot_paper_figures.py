from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def main():
    in_path = Path("paper/tables/all_results_long.csv")
    out_dir = Path("paper/figures")
    ensure_dir(out_dir)

    df = pd.read_csv(in_path)

    # --- Figure 1: F1 vs window size for S1 & S3 (fault detection) ---
    df_fault = df[df["scenario"].isin(["S1_missing_burst", "S3_late_arrivals"])].copy()
    df_fault = df_fault.sort_values(["scenario", "w"])

    plt.figure()
    for sc in ["S1_missing_burst", "S3_late_arrivals"]:
        d = df_fault[df_fault["scenario"] == sc]
        plt.plot(d["w"], d["f1"], marker="o", label=sc)
    plt.xlabel("Window size (w)")
    plt.ylabel("F1 score")
    plt.title("Fault detection: F1 vs window size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_f1_vs_w_fault.png", dpi=200)
    plt.close()

    # --- Figure 2: FP/day vs window size for S1 & S3 ---
    plt.figure()
    for sc in ["S1_missing_burst", "S3_late_arrivals"]:
        d = df_fault[df_fault["scenario"] == sc]
        plt.plot(d["w"], d["false_positives_per_day"], marker="o", label=sc)
    plt.xlabel("Window size (w)")
    plt.ylabel("False positives per day")
    plt.title("Fault detection: FP/day vs window size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_fpday_vs_w_fault.png", dpi=200)
    plt.close()

    # --- Figure 3: S8 (anomaly) median delay & FP/day vs window size ---
    df_s8 = df[df["scenario"] == "S8_efficiency_drop"].copy().sort_values("w")

    # Median delay
    plt.figure()
    plt.plot(df_s8["w"], df_s8["median_delay_minutes"], marker="o")
    plt.xlabel("Window size (w)")
    plt.ylabel("Median detection delay (minutes)")
    plt.title("Anomaly detection (S8): median delay vs window size")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_s8_delay_vs_w.png", dpi=200)
    plt.close()

    # FP/day
    plt.figure()
    plt.plot(df_s8["w"], df_s8["false_positives_per_day"], marker="o")
    plt.xlabel("Window size (w)")
    plt.ylabel("False positives per day")
    plt.title("Anomaly detection (S8): FP/day vs window size")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_s8_fpday_vs_w.png", dpi=200)
    plt.close()

    print("✅ Wrote figures to:", out_dir)

if __name__ == "__main__":
    main()
