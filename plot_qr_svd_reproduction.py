import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import io
import sys

BINARY_PATH = "./build/bench_qr_svd_reproduction"

def main():
    if not os.path.exists(BINARY_PATH):
        print(f"Error: Binary {BINARY_PATH} not found. Please compile first.")
        sys.exit(1)

    print(f"Running {BINARY_PATH}...")
    try:
        result = subprocess.run([BINARY_PATH], capture_output=True, text=True, check=True)
        csv_data = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running binary: {e}")
        sys.exit(1)

    df = pd.read_csv(io.StringIO(csv_data))

    plt.figure(figsize=(14, 10))
    ax = plt.gca()

    truth = df[df['Algorithm'] == 'True']
    ax.semilogy(truth['Index'], truth['Value'], 'k-', linewidth=3.0, label='True Singular Values', zorder=100, alpha=0.3)

    algo_colors = {
        'CGS':             '#d62728',
        'MGS':             '#ff7f0e',
        'MGS_Inplace':     '#bcbd22',
        'Householder_Exp': '#1f77b4',
        'Householder_Imp': '#17becf',
        'Givens_Exp':      '#2ca02c',
        'Givens_Imp':      '#98df8a',
        'LibTorch':        '#7f7f7f'
    }

    prec_styles = {
        'Double': {'ls': '-',  'lw': 1.5, 'alpha': 0.9},
        'Float':  {'ls': ':',  'lw': 2.0, 'alpha': 0.8}
    }

    unique_algos = [alg for alg in df['Algorithm'].unique() if alg != 'True']

    for algo in unique_algos:
        if algo not in algo_colors: continue

        for prec in ['Double', 'Float']:
            subset = df[(df['Algorithm'] == algo) & (df['Precision'] == prec)]
            if subset.empty: continue

            style = prec_styles[prec]
            ax.semilogy(subset['Index'], subset['Value'],
                        color=algo_colors[algo],
                        linestyle=style['ls'],
                        linewidth=style['lw'],
                        alpha=style['alpha'])

    algo_handles = []
    for alg, color in algo_colors.items():
        label_name = alg.replace('_', ' ')
        handle = mlines.Line2D([], [], color=color, marker='o', markersize=5, label=label_name, ls='-')
        algo_handles.append(handle)

    legend1 = ax.legend(handles=algo_handles, title="Algorithms", loc='lower left', bbox_to_anchor=(0, 0), ncol=2)
    ax.add_artist(legend1)

    prec_handles = [
        mlines.Line2D([], [], color='black', ls='-', label='Double Precision (Float64)'),
        mlines.Line2D([], [], color='black', ls=':', lw=2, label='Single Precision (Float32)'),
        mlines.Line2D([], [], color='black', ls='-', lw=3, alpha=0.3, label='Ground Truth')
    ]
    ax.legend(handles=prec_handles, title="Precision", loc='lower left', bbox_to_anchor=(0.4, 0))

    ax.set_title("Stability Comparison: All QR Implementations (Single vs Double)", fontsize=16)
    ax.set_xlabel("Singular Value Index", fontsize=12)
    ax.set_ylabel("Computed Singular Value", fontsize=12)

    ax.grid(True, which="major", ls="-", alpha=0.4)
    ax.grid(True, which="minor", ls="--", alpha=0.1)

    ax.set_ylim(1e-19, 10)
    ax.set_xlim(0, 80)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    outfile = "qr_svd_stability_reproduction_plot.png"
    plt.savefig(outfile, dpi=300)
    print(f"Plot saved to {outfile}")
    plt.show()

if __name__ == "__main__":
    main()
