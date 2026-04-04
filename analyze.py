import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys

def generate_telemetry_report(filepath='verification_telemetry.pkl'):
    try:
        with open(filepath, 'rb') as f:
            telemetry_payload = pickle.load(f)
    except FileNotFoundError:
        print("Telemetry payload not found. Execute model.py prior to analysis.")
        sys.exit(1)

    df_results = telemetry_payload['metrics_dataframe']
    trajectory_samples = telemetry_payload['trajectory_samples']

    df_summary = df_results.groupby('Scenario').agg(
        ITAE_Mean=('ITAE', 'mean'),
        ITAE_Std=('ITAE', 'std'),
        TV_Mean=('TV', 'mean'),
        TV_Std=('TV', 'std'),
        Settling_Mean=('SettlingTime', 'mean'),
        Settling_Std=('SettlingTime', 'std')
    ).round(4)

    print("\n================ ROBUSTNESS VERIFICATION MATRIX ================\n")
    print(df_summary.to_markdown())
    print("\n==============================================================\n")

    sns.set_theme(style="whitegrid")
    fig_dist, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.violinplot(data=df_results, x='Scenario', y='ITAE', ax=axes[0], inner='quartile', palette='mako')
    axes[0].set_title('ITAE Distribution Matrix')
    axes[0].set_ylabel('Integral Time-weighted Absolute Error')

    sns.violinplot(data=df_results, x='Scenario', y='TV', ax=axes[1], inner='quartile', palette='flare')
    axes[1].set_title('Control TV Distribution Matrix')
    axes[1].set_ylabel('Total Variation')

    sns.boxplot(data=df_results, x='Scenario', y='SettlingTime', ax=axes[2], palette='crest')
    axes[2].set_title('Asymptotic Settling Time')
    axes[2].set_ylabel('Seconds')

    plt.tight_layout()
    plt.savefig('monte_carlo_distributions.png', dpi=300)

    fig_phase, ax_phase = plt.subplots(figsize=(10, 8))
    
    colors = {'Nominal': '#2ecc71', 'Parametric Mismatch': '#e74c3c', 'Sensor Degradation': '#3498db'}
    
    for scenario, data in trajectory_samples.items():
        X = data['X']
        ax_phase.plot(X[:, 0], X[:, 1], label=scenario, color=colors[scenario], linewidth=2.0, alpha=0.8)
        ax_phase.scatter(X[0, 0], X[0, 1], color='black', marker='o', zorder=5)
        ax_phase.scatter(X[-1, 0], X[-1, 1], color=colors[scenario], marker='X', s=100, zorder=5)

    ax_phase.set_title('Phase Portrait Trajectory Analysis (Terminal Convergence)')
    ax_phase.set_xlabel('Displacement (q)')
    ax_phase.set_ylabel('Velocity (q_dot)')
    ax_phase.grid(True, linestyle='--', alpha=0.6)
    ax_phase.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('comparative_phase_portrait.png', dpi=300)
    print("Analytics render complete. Artifacts saved locally.")

if __name__ == "__main__":
    generate_telemetry_report()