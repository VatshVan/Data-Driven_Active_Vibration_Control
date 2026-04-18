import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import os


def load_csv(filepath):
    data = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            data.append([float(x) for x in row])
    return header, np.array(data)


def compute_metrics(positions, eq_pos=-1.0):
    deviations = positions - eq_pos
    rms = np.sqrt(np.mean(deviations * deviations, axis=0))
    peak = np.max(np.abs(deviations), axis=0)
    mean_abs = np.mean(np.abs(deviations), axis=0)
    return rms, peak, mean_abs


def plot_3way(unc_data, pid_data, ctrl_data, save_dir):
    joint_labels = ["Front-Left", "Front-Right", "Rear-Left", "Rear-Right"]
    eq_pos = -1.0

    t_unc = unc_data[:, 1]
    t_pid = pid_data[:, 1]
    t_ctrl = ctrl_data[:, 1]

    pos_cols = [2, 4, 6, 8]
    vel_cols = [3, 5, 7, 9]

    # ---- Position comparison ----
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Position Comparison: Uncontrolled vs PID vs Neural MPC", fontsize=14)
    for i in range(4):
        axes[i].plot(t_unc, unc_data[:, pos_cols[i]], "r-", alpha=0.6, label="Uncontrolled")
        axes[i].plot(t_pid, pid_data[:, pos_cols[i]], "g-", alpha=0.6, label="PID")
        axes[i].plot(t_ctrl, ctrl_data[:, pos_cols[i]], "b-", alpha=0.6, label="Neural MPC")
        axes[i].axhline(y=eq_pos, color="k", linestyle="--", alpha=0.4, label="Equilibrium")
        axes[i].set_ylabel(f"{joint_labels[i]} (m)")
        axes[i].legend(loc="upper right", fontsize=7)
        axes[i].grid(True, alpha=0.3)
    axes[3].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "3way_position.png"), dpi=150)
    plt.close()
    print("Saved 3way_position.png")

    # ---- Velocity comparison ----
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Velocity Comparison: Uncontrolled vs PID vs Neural MPC", fontsize=14)
    for i in range(4):
        axes[i].plot(t_unc, unc_data[:, vel_cols[i]], "r-", alpha=0.6, label="Uncontrolled")
        axes[i].plot(t_pid, pid_data[:, vel_cols[i]], "g-", alpha=0.6, label="PID")
        axes[i].plot(t_ctrl, ctrl_data[:, vel_cols[i]], "b-", alpha=0.6, label="Neural MPC")
        axes[i].axhline(y=0.0, color="k", linestyle="--", alpha=0.4)
        axes[i].set_ylabel(f"{joint_labels[i]} (m/s)")
        axes[i].legend(loc="upper right", fontsize=7)
        axes[i].grid(True, alpha=0.3)
    axes[3].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "3way_velocity.png"), dpi=150)
    plt.close()
    print("Saved 3way_velocity.png")

    # ---- Deviation comparison ----
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Deviation from Equilibrium: Uncontrolled vs PID vs Neural MPC", fontsize=14)
    for i in range(4):
        axes[i].plot(t_unc, unc_data[:, pos_cols[i]] - eq_pos, "r-", alpha=0.6, label="Uncontrolled")
        axes[i].plot(t_pid, pid_data[:, pos_cols[i]] - eq_pos, "g-", alpha=0.6, label="PID")
        axes[i].plot(t_ctrl, ctrl_data[:, pos_cols[i]] - eq_pos, "b-", alpha=0.6, label="Neural MPC")
        axes[i].axhline(y=0.0, color="k", linestyle="--", alpha=0.4)
        axes[i].set_ylabel(f"{joint_labels[i]} Dev (m)")
        axes[i].legend(loc="upper right", fontsize=7)
        axes[i].grid(True, alpha=0.3)
    axes[3].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "3way_deviation.png"), dpi=150)
    plt.close()
    print("Saved 3way_deviation.png")

    # ---- FFT comparison ----
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("FFT Analysis: Uncontrolled vs PID vs Neural MPC", fontsize=14)
    for i in range(4):
        for label, t_arr, d_arr, color in [("Uncontrolled", t_unc, unc_data, "r"),
                                             ("PID", t_pid, pid_data, "g"),
                                             ("Neural MPC", t_ctrl, ctrl_data, "b")]:
            sig = d_arr[:, pos_cols[i]] - eq_pos
            n = len(sig)
            dt = np.mean(np.diff(t_arr)) if len(t_arr) > 1 else 0.01
            fft_mag = np.abs(np.fft.rfft(sig)) / n
            freq = np.fft.rfftfreq(n, d=dt)
            axes[i].plot(freq, fft_mag, color=color, alpha=0.6, label=label)
        axes[i].set_ylabel(f"{joint_labels[i]} FFT")
        axes[i].legend(loc="upper right", fontsize=7)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim([0, 15])
    axes[3].set_xlabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "3way_fft.png"), dpi=150)
    plt.close()
    print("Saved 3way_fft.png")

    # ---- Control forces comparison ----
    ctrl_force_cols = [18, 19, 20, 21]
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Control Forces: PID vs Neural MPC", fontsize=14)
    for i in range(4):
        if ctrl_force_cols[i] < pid_data.shape[1]:
            axes[i].plot(t_pid, pid_data[:, ctrl_force_cols[i]], "g-", alpha=0.6, label="PID")
        if ctrl_force_cols[i] < ctrl_data.shape[1]:
            axes[i].plot(t_ctrl, ctrl_data[:, ctrl_force_cols[i]], "b-", alpha=0.6, label="Neural MPC")
        axes[i].set_ylabel(f"{joint_labels[i]} Force (N)")
        axes[i].legend(loc="upper right", fontsize=7)
        axes[i].grid(True, alpha=0.3)
    axes[3].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "3way_control_forces.png"), dpi=150)
    plt.close()
    print("Saved 3way_control_forces.png")

    # ---- Metrics bar chart ----
    unc_pos = unc_data[:, pos_cols]
    pid_pos = pid_data[:, pos_cols]
    ctrl_pos = ctrl_data[:, pos_cols]

    rms_unc, peak_unc, _ = compute_metrics(unc_pos, eq_pos)
    rms_pid, peak_pid, _ = compute_metrics(pid_pos, eq_pos)
    rms_ctrl, peak_ctrl, _ = compute_metrics(ctrl_pos, eq_pos)

    x = np.arange(4)
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Performance Metrics: 3-Way Comparison", fontsize=14)

    axes[0].bar(x - width, rms_unc, width, label="Uncontrolled", color="red", alpha=0.7)
    axes[0].bar(x, rms_pid, width, label="PID", color="green", alpha=0.7)
    axes[0].bar(x + width, rms_ctrl, width, label="Neural MPC", color="blue", alpha=0.7)
    axes[0].set_ylabel("RMS Deviation (m)")
    axes[0].set_title("RMS Position Error")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["FL", "FR", "RL", "RR"])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(x - width, peak_unc, width, label="Uncontrolled", color="red", alpha=0.7)
    axes[1].bar(x, peak_pid, width, label="PID", color="green", alpha=0.7)
    axes[1].bar(x + width, peak_ctrl, width, label="Neural MPC", color="blue", alpha=0.7)
    axes[1].set_ylabel("Peak Deviation (m)")
    axes[1].set_title("Peak Position Error")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["FL", "FR", "RL", "RR"])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "3way_metrics.png"), dpi=150)
    plt.close()
    print("Saved 3way_metrics.png")

    return rms_unc, peak_unc, rms_pid, peak_pid, rms_ctrl, peak_ctrl


def main():
    save_dir = os.path.expanduser("~/ros2_ws/results")
    os.makedirs(save_dir, exist_ok=True)

    unc_path = os.path.expanduser("~/ros2_ws/eval_uncontrolled.csv")
    pid_path = os.path.expanduser("~/ros2_ws/eval_pid.csv")
    ctrl_path = os.path.expanduser("~/ros2_ws/eval_controlled.csv")

    if not os.path.exists(unc_path):
        print(f"ERROR: {unc_path} not found")
        return
    if not os.path.exists(pid_path):
        print(f"ERROR: {pid_path} not found")
        return
    if not os.path.exists(ctrl_path):
        print(f"ERROR: {ctrl_path} not found")
        return

    print("Loading data...")
    _, unc_data = load_csv(unc_path)
    _, pid_data = load_csv(pid_path)
    _, ctrl_data = load_csv(ctrl_path)

    print(f"Uncontrolled: {unc_data.shape[0]} samples")
    print(f"PID: {pid_data.shape[0]} samples")
    print(f"Neural MPC: {ctrl_data.shape[0]} samples")

    print("\nGenerating plots...")
    rms_unc, peak_unc, rms_pid, peak_pid, rms_ctrl, peak_ctrl = plot_3way(
        unc_data, pid_data, ctrl_data, save_dir
    )

    joint_labels = ["FL", "FR", "RL", "RR"]

    print("\n" + "=" * 70)
    print("3-WAY PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"{'Joint':<6} {'RMS Unc':>10} {'RMS PID':>10} {'PID Red':>9} {'RMS MPC':>10} {'MPC Red':>9}")
    print("-" * 56)
    for i in range(4):
        pid_red = (1.0 - rms_pid[i] / rms_unc[i]) * 100 if rms_unc[i] > 0 else 0
        mpc_red = (1.0 - rms_ctrl[i] / rms_unc[i]) * 100 if rms_unc[i] > 0 else 0
        print(f"{joint_labels[i]:<6} {rms_unc[i]:>10.4f} {rms_pid[i]:>10.4f} {pid_red:>8.1f}% {rms_ctrl[i]:>10.4f} {mpc_red:>8.1f}%")

    print(f"\n{'Joint':<6} {'Peak Unc':>10} {'Peak PID':>10} {'PID Red':>9} {'Peak MPC':>10} {'MPC Red':>9}")
    print("-" * 56)
    for i in range(4):
        pid_red = (1.0 - peak_pid[i] / peak_unc[i]) * 100 if peak_unc[i] > 0 else 0
        mpc_red = (1.0 - peak_ctrl[i] / peak_unc[i]) * 100 if peak_unc[i] > 0 else 0
        print(f"{joint_labels[i]:<6} {peak_unc[i]:>10.4f} {peak_pid[i]:>10.4f} {pid_red:>8.1f}% {peak_ctrl[i]:>10.4f} {mpc_red:>8.1f}%")

    avg_rms_unc = np.mean(rms_unc)
    avg_rms_pid = np.mean(rms_pid)
    avg_rms_ctrl = np.mean(rms_ctrl)
    pid_overall = (1.0 - avg_rms_pid / avg_rms_unc) * 100 if avg_rms_unc > 0 else 0
    mpc_overall = (1.0 - avg_rms_ctrl / avg_rms_unc) * 100 if avg_rms_unc > 0 else 0

    print(f"\nOverall Average RMS:")
    print(f"  Uncontrolled: {avg_rms_unc:.4f}")
    print(f"  PID:          {avg_rms_pid:.4f}  ({pid_overall:+.1f}%)")
    print(f"  Neural MPC:   {avg_rms_ctrl:.4f}  ({mpc_overall:+.1f}%)")
    print(f"\nAll plots saved to: {save_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
