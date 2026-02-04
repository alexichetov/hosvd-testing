import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time

BINARY_PATH = "./build/bench_qr_memory"

# Square: M == N
# Tall-Skinny: Fixed M, varying N
EXPERIMENTS = {
    "Square": {
        "params": [(s, s) for s in [500, 1000, 2000]],
        "x_label": "Matrix Size (N x N)",
        "x_axis": "N"
    },
    "Tall-Skinny": {
        "params": [(8000, n) for n in [50, 100, 250, 500]],
        "x_label": "Number of Columns (N) [Fixed M=8000]",
        "x_axis": "N"
    }
}

ALGOS = [
    "CGS", "MGS", "MGS_Inplace",
    "Householder_Explicit", "Householder_Implicit",
    "Givens_Explicit", "Givens_Inplace"
]
PRECISIONS = ["float", "double"]

def run_benchmark():
    results = []
    if not os.path.exists(BINARY_PATH):
        print(f"Error : Could not find binary at {BINARY_PATH}")
        sys.exit(1)

    for exp_name, cfg in EXPERIMENTS.items():
        print(f"\n>>> Running Experiment: {exp_name}")

        for m, n in cfg["params"]:
            for prec in PRECISIONS:
                for algo in ALGOS:
                    try:
                        cmd = [BINARY_PATH, algo, str(m), str(n), prec]
                        start_time = time.time()
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        elapsed = time.time() - start_time

                        line = result.stdout.strip()
                        parts = line.split(',')

                        peak_mb = float(parts[4])
                        baseline_mb = float(parts[5])
                        aux_mb = max(0.0, peak_mb - baseline_mb)

                        results.append({
                            "Exp": exp_name,
                            "M": m, "N": n,
                            "Algorithm": algo,
                            "Precision": prec,
                            "PeakMB": peak_mb,
                            "AuxMB": aux_mb
                        })

                    except Exception as e:
                        print(f"Error skipping {algo} {m}x{n}: {e}")

    return pd.DataFrame(results)

def plot_exp(df, exp_name):
    algo_styles = {
        "CGS":                  {"color": "#ff7f0e", "marker": "x"},
        "MGS":                  {"color": "#d62728", "marker": "+"},
        "MGS_Inplace":          {"color": "#2ca02c", "marker": "^"},
        "Householder_Explicit": {"color": "#9467bd", "marker": "d"},
        "Householder_Implicit": {"color": "#1f77b4", "marker": "s"},
        "Givens_Explicit":      {"color": "#8c564b", "marker": "*"},
        "Givens_Inplace":       {"color": "#e377c2", "marker": "v"}
    }

    subset_exp = df[df["Exp"] == exp_name]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(top=0.92)
    fig.suptitle(f"Memory Benchmark: {exp_name} Matrices", fontsize=18)

    for i, prec in enumerate(PRECISIONS):
        data_prec = subset_exp[subset_exp["Precision"] == prec]

        # Plot Total Peak
        ax_total = axes[i, 0]
        for algo in ALGOS:
            d = data_prec[data_prec["Algorithm"] == algo]
            if d.empty: continue
            ax_total.plot(d["N"], d["PeakMB"], label=algo, **algo_styles[algo])
        ax_total.set_title(f"{prec.capitalize()} Precision: Total Peak Memory")
        ax_total.set_ylabel("Memory (MB)")
        ax_total.grid(True, alpha=0.3)
        ax_total.legend()

        # Plot Aux Overhead
        ax_aux = axes[i, 1]
        for algo in ALGOS:
            d = data_prec[data_prec["Algorithm"] == algo]
            if d.empty: continue
            ax_aux.plot(d["N"], d["AuxMB"], label=algo, **algo_styles[algo])
        ax_aux.set_title(f"{prec.capitalize()} Precision: Auxiliary Overhead (Peak - Input)")
        ax_aux.set_ylabel("Memory (MB)")
        ax_aux.grid(True, alpha=0.3)
        ax_aux.legend()

    for ax in axes.flatten():
        ax.set_xlabel(EXPERIMENTS[exp_name]["x_label"])

    output_file = f"qr_mem_{exp_name.lower()}.png"
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot: {output_file}")

if __name__ == "__main__":
    df = run_benchmark()
    if not df.empty:
        plot_exp(df, "Square")
        plot_exp(df, "Tall-Skinny")
    plt.show()
