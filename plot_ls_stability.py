import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
import numpy as np

def run_experiment():
    print("Running Least Squares Stability Experiment...")
    result = subprocess.run(
        ["./build/ls_stability"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout

def plot_results(csv_data):
    print("Plotting results...")
    df = pd.read_csv(io.StringIO(csv_data))

    plt.figure(figsize=(10, 7))

    # Gram
    plt.loglog(df['ConditionNumber'], df['GramAvg'], 'o-', label='Normal Equations (Gram)', color='#d62728', markersize=4, linewidth=1.5)
    plt.fill_between(df['ConditionNumber'], df['GramMin'], df['GramMax'], color='#d62728', alpha=0.2)

    # QR
    plt.loglog(df['ConditionNumber'], df['QRAvg'], 's-', label='Householder QR', color='#1f77b4', markersize=4, linewidth=1.5)
    plt.fill_between(df['ConditionNumber'], df['QRMin'], df['QRMax'], color='#1f77b4', alpha=0.2)

    # breakdown lines
    plt.axvline(x=1e8, color='#d62728', linestyle=':', alpha=0.6, label='Expected Gram Breakdown')
    plt.axvline(x=1e16, color='#1f77b4', linestyle=':', alpha=0.6, label='Expected QR Breakdown)')

    # Formatting
    plt.title('Least Squares Stability: Normal Equations vs. QR (N=10 Trials)')
    plt.xlabel('Condition Number')
    plt.ylabel('Relative Error')

    plt.grid(True, which="major", ls="-", alpha=0.4)
    plt.grid(True, which="minor", ls="--", alpha=0.1)

    plt.legend(loc='upper left')

    plt.ylim(bottom=1e-17, top=10)

    output_file = "ls_stability_plot.png"
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    try:
        if not os.path.exists("build/ls_stability"):
            print("Error: Run from project root. Ensure binaries are compiled.")
            sys.exit(1)

        output = run_experiment()
        plot_results(output)

    except Exception as e:
        print(f"Error: {e}")
