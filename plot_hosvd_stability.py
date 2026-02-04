import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
import numpy as np

# Configuration
BINARY_PATH = "./build/bench_hosvd_stability"
NUM_RUNS = 25
EPS_FLOAT = 1.19209e-07
EPS_DOUBLE = 2.22044e-16
TENSOR_SIZE = 32
TARGET_RANKS = [3, 6, 9, 10]

def run_batch(precision_flag, size, rank, runs=25):
    print(f"Starting batch: {precision_flag} | Size: {size} | Rank: {rank} ({runs} runs)...")

    if not os.path.exists(BINARY_PATH):
        print(f"Error: Binary {BINARY_PATH} not found. Please compile first.")
        sys.exit(1)

    dfs = []

    for i in range(runs):
        seed = 42 + i # deterministic
        cmd = [
            BINARY_PATH,
            "--seed", str(seed),
            "--size", str(size),
            "--rank", str(rank),
            precision_flag
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            df = pd.read_csv(io.StringIO(result.stdout))
            dfs.append(df)

        except subprocess.CalledProcessError as e:
            print(f"Error running binary: {e}")
            sys.exit(1)

    print(f"Aggregating results for {precision_flag}...")
    combined = pd.concat(dfs)
    averaged = combined.groupby("ConditionNumber").mean().reset_index()
    return averaged

def plot_subplot(ax, df, title, epsilon):
    styles = {
        'Gram':           {'color': 'black',   'marker': 'o', 'ls': '-',  'label': 'Gram (Baseline)'},
        'CGS':            {'color': '#ff7f0e', 'marker': 'x', 'ls': '--', 'label': 'Classical Gram-Schmidt'},
        'MGS':            {'color': '#d62728', 'marker': '+', 'ls': '-',  'label': 'Modified Gram-Schmidt'},
        'MGS_Inplace':    {'color': '#2ca02c', 'marker': '^', 'ls': '--', 'label': 'MGS In-place'},
        'H_Explicit':     {'color': '#9467bd', 'marker': 'd', 'ls': '-',  'label': 'Householder Explicit'},
        'H_Implicit':     {'color': '#1f77b4', 'marker': 's', 'ls': '-',  'label': 'Householder Implicit'},
        'Givens':         {'color': '#8c564b', 'marker': '*', 'ls': ':',  'label': 'Givens Rotations'},
        'Givens_Inplace': {'color': '#e377c2', 'marker': 'h', 'ls': '--', 'label': 'Givens In-place'},
        'LibTorch_QR':    {'color': '#d4ac0d', 'marker': 'p', 'ls': '-',  'label': 'LibTorch Built-in'}
    }

    for col in df.columns:
        if col == 'ConditionNumber': continue
        style = styles.get(col, {'label': col, 'ls': '-'})
        ax.loglog(df['ConditionNumber'], df[col], **style, alpha=0.8, markersize=6)

    cond = df['ConditionNumber']

    # Noise floor for the specific data type
    ax.axhline(y=epsilon, color='gray', ls='-.', alpha=0.6, lw=1.5, label=r'Machine Precision ($\epsilon$)')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(r'Condition Number $\kappa(\mathcal{X})$', fontsize=12)
    ax.set_ylabel(r'Relative Reconstruction Error', fontsize=12)
    ax.grid(True, which="major", ls="-", alpha=0.4)
    ax.grid(True, which="minor", ls="--", alpha=0.1)

    # Adjust Y-limit based on expected noise floor vs signal
    ax.set_ylim(epsilon * 1e-2, 10)

def main():

    for rank in TARGET_RANKS:
        print(f"\n=== Processing Target Rank: {rank} ===")

        df_float = run_batch("float", TENSOR_SIZE, rank, runs=NUM_RUNS)
        df_double = run_batch("double", TENSOR_SIZE, rank, runs=NUM_RUNS)

        print(f"Plotting comparison for Rank {rank}...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        fig.suptitle(f"ST-HOSVD Stability Analysis (Tensor Size={TENSOR_SIZE}, Rank={rank})", fontsize=16)

        plot_subplot(ax1, df_float, f"Float32 - Rank {rank}", EPS_FLOAT)
        plot_subplot(ax2, df_double, f"Float64 - Rank {rank}", EPS_DOUBLE)

        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05), fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.9)

        output_file = f"hosvd_stability_rank_{rank}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        plt.close(fig)

if __name__ == "__main__":
    main()
