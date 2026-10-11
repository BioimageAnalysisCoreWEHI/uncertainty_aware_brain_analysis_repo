# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:26:42 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

from pathlib import Path
import re
import pandas as pd
from datetime import timedelta

LOG_DIR = Path(P("<IMAGING>/Ishrat/JoM/hpc_processing_logs"))  # <-- change this

records = []

def clean_wrapped_path(text):
    """
    Reconstruct wrapped brainreg output paths.
    """
    lines = text.splitlines()

    path_chunks = []
    capture = False

    for line in lines:
        if "Results can be found" in line:
            capture = True
            continue

        if capture:
            if "Finished. Total time taken" in line or "✅ Done" in line:
                break

            cleaned = line.strip()

            if "/vast/" in cleaned:
                cleaned = cleaned[cleaned.find("/vast/"):]
                path_chunks.append(cleaned)
            elif path_chunks:
                cleaned = cleaned.strip()
                if cleaned:
                    path_chunks.append(cleaned)

    if path_chunks:
        return "".join(path_chunks).replace(" ", "")

    return None


for log_file in LOG_DIR.rglob("*inner*"):

    if not log_file.is_file():
        continue

    text = log_file.read_text(errors="ignore")

    # -----------------------------
    # START TIME
    # -----------------------------
    start_match = re.search(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+PM.*Starting logging",
        text,
        re.DOTALL
    )

    start_time = (
        pd.to_datetime(start_match.group(1))
        if start_match else pd.NaT
    )

    # -----------------------------
    # END TIME
    # -----------------------------
    end_match = re.search(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+PM.*Finished\.\s+Total time taken",
        text,
        re.DOTALL
    )

    end_time = (
        pd.to_datetime(end_match.group(1))
        if end_match else pd.NaT
    )

    # -----------------------------
    # TOTAL RUNTIME STRING
    # -----------------------------
    runtime_match = re.search(
        r"Finished\. Total time taken:\s*([0-9]+:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?)",
        text
    )

    runtime = (
        pd.to_timedelta(runtime_match.group(1))
        if runtime_match else pd.NaT
    )

    output_path = clean_wrapped_path(text)

    records.append({
        "log_file": str(log_file),
        "output_path": output_path,
        "start_time": start_time,
        "end_time": end_time,
        "runtime": runtime,
    })

# -----------------------------
# Create dataframe
# -----------------------------
df = pd.DataFrame(records)
df = df.dropna()
df = df.sort_values("start_time").reset_index(drop=True)

# -----------------------------
# Summary stats
# -----------------------------
total_runtime = df["runtime"].sum()

overall_elapsed = (
    df["end_time"].max() -
    df["start_time"].min()
)

print(df)

print("\nSummary")
print("-------")
print(f"Total summed runtime: {total_runtime}")
print(f"Overall elapsed wall time: {overall_elapsed}")

# Optional save
df.to_csv(LOG_DIR / "inner_brainreg_runtime_summary.csv", index=False)




#### PLOTTING ####


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_hpc_advantage_publication(
    df,
    output_path=P(r'<IMAGING>\Ishrat\JoM\final_figures\Figure1 - Problem\hpc_efficiency_actual_walltime.png')
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # -----------------------------
    # Clean dataframe
    # -----------------------------
    df = df.dropna(subset=["start_time", "end_time", "runtime"]).copy()

    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    df["runtime"] = pd.to_timedelta(df["runtime"])

    total_registrations = len(df)

    # -----------------------------
    # Actual timing values
    # -----------------------------
    sequential_time_hours = df["runtime"].sum().total_seconds() / 3600

    wall_time = df["end_time"].max() - df["start_time"].min()
    wall_time_hours = wall_time.total_seconds() / 3600

    speedup = sequential_time_hours / wall_time_hours

    # Effective concurrency
    effective_parallelism = speedup

    # -----------------------------
    # Scalability curve
    # -----------------------------
    avg_runtime_hours = df["runtime"].mean().total_seconds() / 3600

    slots_range = np.array([1, 2, 4, 8, 16, 20, 30, 40, 60, 80, 100])

    times_at_slots = [
        (np.ceil(total_registrations / s) * avg_runtime_hours)
        for s in slots_range
    ]

    # -----------------------------
    # Plotting setup
    # -----------------------------
    plt.rcParams.update({
        "font.family": "Arial",

        # General text
        "font.size": 14,

        # Titles
        "axes.titlesize": 22,
        "axes.titleweight": "bold",

        # Axis labels
        "axes.labelsize": 20,
        "axes.labelweight": "bold",

        # Tick labels
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,

        # Legend
        "legend.fontsize": 17,

        # Axis lines
        "axes.linewidth": 1.3,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,

        # Editable text in Illustrator
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(14, 6.5),
        gridspec_kw={"width_ratios": [1, 1.4]}
    )

    # -----------------------------
    # PANEL A: WALL-TIME ADVANTAGE
    # -----------------------------
    labels = [
        "Sequential execution\n(sum of runtimes)",
        "HPC execution\n(actual elapsed wall time)"
    ]

    times = [sequential_time_hours, wall_time_hours]

    colors = ["#ED9413", "#21918c"]

    bars = ax1.bar(
        labels,
        times,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        width=0.62,
        alpha=0.92
    )

    ax1.set_ylabel("Processing Time (hours)")

    ax1.set_title(
        f"HPC parallelisation reduces wall-clock time\n{total_registrations} registrations",
        pad=18
    )

    # Runtime labels
    ax1.text(
        0,
        sequential_time_hours * 1.02,
        f"{sequential_time_hours:.1f} h",
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        color="#D48004"
    )

    ax1.text(
        1,
        wall_time_hours * 1.2,
        f"{wall_time_hours:.1f} h",
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        color="#21918c"
    )

    # Speedup box
# =============================================================================
#     ax1.text(
#         0.5,
#         sequential_time_hours * 0.55,
#         f"{speedup:.1f}× faster\nwall-clock turnaround",
#         ha="center",
#         va="center",
#         fontsize=14,
#         fontweight="bold",
#         bbox=dict(
#             boxstyle="round,pad=0.45",
#             fc="white",
#             ec="black",
#             lw=1.2
#         )
#     )
# =============================================================================

    # -----------------------------
    # PANEL B: SCALABILITY
    # -----------------------------
    ax2.plot(
        slots_range,
        times_at_slots,
        "o-",
        color="#21918c",
        linewidth=3,
        markersize=10
    )

    ax2.fill_between(
        slots_range,
        times_at_slots,
        alpha=0.12,
        color="#21918c"
    )

    ax2.axhline(
        wall_time_hours,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Observed elapsed time: {wall_time_hours:.1f} h"
    )

    ax2.axvline(
        effective_parallelism,
        color="#ED9413",
        linestyle=":",
        linewidth=2.5,
        label=f"Effective parallelism: {effective_parallelism:.1f}×"
    )

    ax2.scatter(
        effective_parallelism,
        wall_time_hours,
        s=140,
        color="#ED9413",
        edgecolor="black",
        linewidth=1,
        zorder=5
    )

    ax2.annotate(
        f"Observed\n{wall_time_hours:.1f} h",
        xy=(effective_parallelism, wall_time_hours),
        xytext=(effective_parallelism + 10, wall_time_hours + 10),
        arrowprops=dict(
            facecolor="black",
            shrink=0.08,
            width=1.8,
            headwidth=8
        ),
        fontsize=13,
        fontweight="bold"
    )

    ax2.set_xlabel("Parallel compute slots / effective concurrency")
    ax2.set_ylabel("Estimated time to solution (hours)")

    ax2.set_title(
        "Scalability of registration sweep",
        pad=18
    )

    ax2.grid(True, linestyle=":", alpha=0.6)

    ax2.legend(
        loc="upper right",
        frameon=True
    )

    # -----------------------------
    # Styling
    # -----------------------------
    sns.despine()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        output_path.replace(".png", ".pdf"),
        bbox_inches="tight"
    )

    plt.show()

    # -----------------------------
    # Console output
    # -----------------------------
    print(f"Publication-ready figure saved to: {output_path}")
    print(f"Total registrations: {total_registrations}")
    print(f"Sequential runtime: {sequential_time_hours:.2f} h")
    print(f"Observed wall time: {wall_time_hours:.2f} h")
    print(f"Speedup: {speedup:.1f}×")

plot_hpc_advantage_publication(df)
