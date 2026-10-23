# Wheel odometry (dead reckoning) with error tracking
import csv
import math

import matplotlib.pyplot as plt
import numpy as np

from sim import ENCODER_COUNTS_PER_REV, WHEEL_RADIUS, WHEEL_SEPARATION

NUM_RUNS = 100               # noisy odometry runs for error tracking
SLIP_NOISE = 0.02            # wheel slip std
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


def wrap_angle(a):
    """Keep angles within ±π, so e.g. 359° and -1° are facing the same way."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def position_error(poses, true_x, true_y):
    """Distance between estimated and true position at every step."""
    return np.hypot(poses[:, 0] - true_x, poses[:, 1] - true_y)


def noisy_runs(ds_l, ds_r, start_pose):
    """Repeat the odometry with random wheel slip on every step"""
    rng = np.random.default_rng(SEED)
    runs = []
    for _ in range(NUM_RUNS):
        noisy_l = ds_l + rng.normal(0, SLIP_NOISE * np.sqrt(np.abs(ds_l)))
        noisy_r = ds_r + rng.normal(0, SLIP_NOISE * np.sqrt(np.abs(ds_r)))
        runs.append(odometry(noisy_l, noisy_r, start_pose))
    return np.array(runs)


def main():
    log = load_log()
    true_x, true_y, true_yaw = log["true_x"], log["true_y"], log["true_yaw"]
    start_pose = (0.0, 0.0, 0.0)

    ds_l, ds_r = wheel_distances(log["left_counts"], log["right_counts"])


    odom = odometry(ds_l, ds_r, start_pose)
    runs = noisy_runs(ds_l, ds_r, start_pose)

    odom_error = position_error(odom, true_x, true_y)
    heading_error = np.degrees(wrap_angle(odom[:, 2] - true_yaw))
    noisy_run_errors = np.array([position_error(run, true_x, true_y) for run in runs])
    noisy_run_final = noisy_run_errors[:, -1]
    noisy_run_rms = np.sqrt(np.mean(noisy_run_errors ** 2, axis=1))
    noisy_run_max = noisy_run_errors.max(axis=1)
    noisy_run_heading = np.abs(np.degrees(wrap_angle(runs[:, -1, 2] - true_yaw[-1])))
    print(f"Odometry final position error:   {odom_error[-1]:.3f} m")
    print(f"Odometry RMS position error:     {np.sqrt(np.mean(odom_error ** 2)):.3f} m")
    print(f"Odometry max position error:     {odom_error.max():.3f} m")
    print(f"Odometry final heading error:    {heading_error[-1]:.1f} deg")
    print(f"Noisy runs final position error: {noisy_run_final.mean():.2f} ± {noisy_run_final.std():.2f} m")
    print(f"Noisy runs RMS position error:   {noisy_run_rms.mean():.2f} ± {noisy_run_rms.std():.2f} m")
    print(f"Noisy runs max position error:   {noisy_run_max.mean():.2f} ± {noisy_run_max.std():.2f} m")
    print(f"Noisy runs final heading error:  {noisy_run_heading.mean():.1f} ± {noisy_run_heading.std():.1f} deg")

    fig, ax_path = plt.subplots(figsize=(7, 7))

    for run in runs:
        ax_path.plot(run[:, 0], run[:, 1], color="tab:orange", alpha=0.08, linewidth=0.8)
    ax_path.plot(true_x, true_y, color="black", linewidth=2, label="Ground truth")
    ax_path.plot(odom[:, 0], odom[:, 1], color="tab:blue", linewidth=2, label="Odometry")
    ax_path.plot([], [], color="tab:orange", label=f"Odometry + wheel slip ({NUM_RUNS} runs)")
    start_x, start_y, _ = start_pose
    ax_path.plot(start_x, start_y, "ko", label="Start")
    ax_path.set_xlabel("x (m)")
    ax_path.set_ylabel("y (m)")
    ax_path.set_title("Trajectory")
    ax_path.axis("equal")
    ax_path.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2) 
    fig.tight_layout()
    fig.savefig("data/odometry.png", dpi=150)


if __name__ == "__main__":
    main()
