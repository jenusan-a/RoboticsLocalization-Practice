# Wheel odometry (dead reckoning) with error tracking
import csv
import math

import matplotlib.pyplot as plt
import numpy as np

from sim import ENCODER_COUNTS_PER_REV, WHEEL_RADIUS, WHEEL_SEPARATION

NUM_RUNS = 100               # noisy odometry runs for error tracking
SCALE_ERROR = 0.02           # std of each wheel's scale error per run (2%)
SEED = 0


def load_log(path="data/log.csv"):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def wheel_distances(left_counts, right_counts):
    """Distance travelled by each wheel per step"""
    ds_l = 2 * math.pi * WHEEL_RADIUS * np.diff(left_counts) / ENCODER_COUNTS_PER_REV
    ds_r = 2 * math.pi * WHEEL_RADIUS * np.diff(right_counts) / ENCODER_COUNTS_PER_REV
    return ds_l, ds_r


def odometry(ds_l, ds_r, start_pose):
    """Integrate the pose with the odometry update equations """
    x, y, theta = start_pose
    poses = [(x, y, theta)]
    for dl, dr in zip(ds_l, ds_r):
        ds = (dr + dl) / 2
        dtheta = (dr - dl) / WHEEL_SEPARATION
        x += ds * math.cos(theta + dtheta / 2)
        y += ds * math.sin(theta + dtheta / 2)
        theta += dtheta
        poses.append((x, y, theta))
    return np.array(poses)


def noisy_runs(ds_l, ds_r, start_pose):
    """Repeat the odometry with a systematic wheel distance error"""
    rng = np.random.default_rng(SEED)
    runs = []
    for _ in range(NUM_RUNS):
        scale_l = 1 + rng.normal(0, SCALE_ERROR)   # one number for the whole run
        scale_r = 1 + rng.normal(0, SCALE_ERROR)
        noisy_l = ds_l * scale_l
        noisy_r = ds_r * scale_r
        runs.append(odometry(noisy_l, noisy_r, start_pose))
    return np.array(runs)


def main():
    log = load_log()
    true_x, true_y, true_yaw = log["true_x"], log["true_y"], log["true_yaw"]
    start_pose = (true_x[0], true_y[0], true_yaw[0])  # known start pose 

    ds_l, ds_r = wheel_distances(log["left_counts"], log["right_counts"])


    odom = odometry(ds_l, ds_r, start_pose)
    runs = noisy_runs(ds_l, ds_r, start_pose)

    odom_error = np.hypot(odom[-1, 0] - true_x[-1], odom[-1, 1] - true_y[-1])
    run_errors = np.hypot(runs[:, -1, 0] - true_x[-1], runs[:, -1, 1] - true_y[-1])
    print(f"Odometry final position error:   {odom_error:.2f} m")
    print(f"Noisy runs final position error: {run_errors.mean():.2f} ± {run_errors.std():.2f} m")

    fig, ax_path = plt.subplots(figsize=(7, 6))

    for run in runs:
        ax_path.plot(run[:, 0], run[:, 1], color="tab:orange", alpha=0.08, linewidth=0.8)
    ax_path.plot(true_x, true_y, color="black", linewidth=2, label="Ground truth")
    ax_path.plot(odom[:, 0], odom[:, 1], color="tab:blue", linewidth=2, label="Odometry")
    ax_path.plot([], [], color="tab:orange", label=f"Odometry + wheel scale error ({NUM_RUNS} runs)")
    start_x, start_y, _ = start_pose
    ax_path.plot(start_x, start_y, "ko", label="Start")
    ax_path.set_xlabel("x (m)")
    ax_path.set_ylabel("y (m)")
    ax_path.set_title("Trajectory")
    ax_path.axis("equal")
    ax_path.legend()

    fig.tight_layout()
    fig.savefig("data/odometry.png", dpi=150)


if __name__ == "__main__":
    main()
