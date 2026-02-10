import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys
import tqdm

BINARY_PATH = "./build/bench_compression"
OUTPUT_CSV = "compression_results.csv"

ALGOS = ["Gram", "LibTorch_QR", "MGS_InPlace", "Householder_Imp"]
CONDS = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16]
EPSILONS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12 , 1e-13, 1e-14, 1e-15, 1e-16]
PRECISIONS = ["float", "double"]

def run_single_config(algo, cond, eps, prec):
    if not os.path.exists(BINARY_PATH):
        raise FileNotFoundError(f"Binary {BINARY_PATH} not found.")

    cmd = [
        BINARY_PATH,
        "--algo", str(algo),
        "--cond", str(cond),
        "--eps", str(eps),
        "--prec", str(prec)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()

def run_full_benchmark():
    results = []
    total = len(ALGOS) * len(CONDS) * len(EPSILONS) * len(PRECISIONS)
    print(f"Starting benchmark sweep ({total} runs)...")

    pbar = tqdm.tqdm(total=total)

    header = "Algorithm,Precision,ConditionNumber,TargetEpsilon,RuntimeSec,MemoryKB,ReconstructionError,CompressionRatio"
    results.append(header)

    for cond in CONDS:
        for eps in EPSILONS:
            for algo in ALGOS:
                for prec in PRECISIONS:
                    try:
                        line = run_single_config(algo, cond, eps, prec)
                        results.append(line)
                    except Exception as e:
                        print(f"\nFailed {algo} {prec} C={cond} E={eps}: {e}")
                    pbar.update(1)

    pbar.close()
    return "\n".join(results)

def plot_data(df):
    sns.set_theme(style="whitegrid")

    algos = df['Algorithm'].unique()
    palette = sns.color_palette("bright", n_colors=len(algos))

    def plot_stability(prec, target_eps_val=1e-12):
        subset = df[(df['Precision'] == prec) & (df['TargetEpsilon'] == target_eps_val)]
        if subset.empty: return

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=subset, x='ConditionNumber', y='ReconstructionError',
                     hue='Algorithm', style='Algorithm', markers=True, dashes=False, palette=palette)
        plt.xscale('log')
        plt.yscale('log')
        plt.axhline(target_eps_val, color='gray', linestyle='--', label=f'Target Epsilon ({target_eps_val})')

        mach_eps = 1.19e-7 if prec == 'float' else 2.22e-16
        plt.axhline(mach_eps, color='red', linestyle=':', label=f'Machine Epsilon ({prec})')

        plt.title(f'Stability ({prec}): Condition Number vs Error (Target Eps={target_eps_val})')
        plt.ylabel('Relative Reconstruction Error')
        plt.xlabel('Condition Number')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'plot_stability_cond_{prec}.png', dpi=300)
        plt.close()

    def plot_compression(prec, stable_cond_val=1e4):
        subset = df[(df['Precision'] == prec) & (df['ConditionNumber'] == stable_cond_val)]
        if subset.empty: return

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=subset, x='TargetEpsilon', y='CompressionRatio',
                     hue='Algorithm', style='Algorithm', markers=True, dashes=False, palette=palette)
        plt.xscale('log')
        plt.yscale('log')
        plt.gca().invert_xaxis()
        plt.title(f'Compression Efficiency ({prec}) (Cond={stable_cond_val})')
        plt.ylabel('Compression Ratio')
        plt.xlabel('Target Epsilon (log scale)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'plot_compression_ratio_{prec}.png', dpi=300)
        plt.close()

    plot_stability("double")
    plot_stability("float", target_eps_val=1e-4)

    plot_compression("double")
    plot_compression("float")

    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df, x='Algorithm', y='RuntimeSec', hue='Precision', palette="Set2")
    plt.yscale('log')
    plt.title('Runtime Distribution: Float vs Double')
    plt.ylabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig('plot_runtime.png', dpi=300)
    plt.close()

    plt.figure(figsize=(14, 7))
    sns.barplot(data=df, x='Algorithm', y='MemoryKB', hue='Precision', palette="Set2", errorbar='sd')
    plt.title('Peak Algorithm Overhead: Float vs Double')
    plt.ylabel('Memory Overhead (KB)')
    plt.tight_layout()
    plt.savefig('plot_memory.png', dpi=300)
    plt.close()

    print("Plots saved: Stability (Float/Double), Compression (Float/Double), Runtime, Memory.")

if __name__ == "__main__":
    csv_data = run_full_benchmark()

    with open(OUTPUT_CSV, "w") as f:
        f.write(csv_data)

    df = pd.read_csv(io.StringIO(csv_data))
    plot_data(df)
