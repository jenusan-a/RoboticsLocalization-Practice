import argparse
import csv
import math
import time
from pathlib import Path

import pybullet as p

SIM_HZ = 240
WHEEL_RADIUS = 0.05
WHEEL_SEPARATION = 0.22

LEFT_WHEEL = 0             # left_wheel_joint
RIGHT_WHEEL = 1            # right_wheel_joint
CASTER = 2                 # caster_wheel link
ENCODER_COUNTS_PER_REV = 1000

# Each segment drives until its goal is reached.
STRAIGHT = {"distance": 1.0, "left_speed": 8.0, "right_speed": 8.0}  # m
TURN_LEFT = {"angle": 90.0, "left_speed": 3.2, "right_speed": 8.0}  # deg

# Rounded square loop that ends near the start
TRAJECTORY = [STRAIGHT, TURN_LEFT] * 4


def setup(gui):
    p.connect(p.GUI if gui else p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / SIM_HZ)
    p.loadURDF("worlds/plane/plane.urdf")
    robot = p.loadURDF("robots/diff_drive_robot/diff_drive_robot.urdf", [0, 0, 0.1])
    p.changeDynamics(robot, CASTER, lateralFriction=0, rollingFriction=0, spinningFriction=0)
    return robot


def set_wheel_speeds(robot, left, right):
    p.setJointMotorControl2(robot, LEFT_WHEEL, p.VELOCITY_CONTROL, targetVelocity=left)
    p.setJointMotorControl2(robot, RIGHT_WHEEL, p.VELOCITY_CONTROL, targetVelocity=right)


def read_encoder_counts(robot):
    """Encoder counts per side - Sensor Readings"""
    left_angle = p.getJointState(robot, LEFT_WHEEL)[0]
    right_angle = p.getJointState(robot, RIGHT_WHEEL)[0]
    left = int(left_angle / (2 * math.pi) * ENCODER_COUNTS_PER_REV)
    right = int(right_angle / (2 * math.pi) * ENCODER_COUNTS_PER_REV)
    return left, right


def read_true_pose(robot):
    """Ground truth pose from the simulator"""
    pos, orn = p.getBasePositionAndOrientation(robot)
    yaw = p.getEulerFromQuaternion(orn)[2]
    return pos[0], pos[1], yaw


def segment_done(segment, start_pose, pose):
    """True once the segment's distance or turn angle has been reached."""
    start_x, start_y, start_yaw = start_pose
    x, y, yaw = pose

    if "distance" in segment:
        dx = x - start_x
        dy = y - start_y
        distance_travelled = math.hypot(dx, dy)
        return distance_travelled >= segment["distance"]

    dyaw = yaw - start_yaw
    angle_turned = math.atan2(math.sin(dyaw), math.cos(dyaw))
    return abs(math.degrees(angle_turned)) >= segment["angle"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="show the 3D window")
    args = parser.parse_args()

    robot = setup(args.gui)
    for _ in range(SIM_HZ):  # let the robot settle onto the ground
        p.stepSimulation()

    Path("data").mkdir(exist_ok=True)

    with open("data/log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "true_x", "true_y", "true_yaw", "left_counts", "right_counts"])

        t = 0.0
        for segment in TRAJECTORY:
            set_wheel_speeds(robot, segment["left_speed"], segment["right_speed"])
            start_pose = pose = read_true_pose(robot)
            while not segment_done(segment, start_pose, pose):
                p.stepSimulation()
                t += 1.0 / SIM_HZ
                pose = read_true_pose(robot)
                writer.writerow([round(t, 5), *pose, *read_encoder_counts(robot)])
                if args.gui:
                    time.sleep(1.0 / SIM_HZ)

    x, y, yaw = read_true_pose(robot)
    print(f"Done: {t:.1f} s simulated, final true pose x={x:.2f} m, y={y:.2f} m, yaw={math.degrees(yaw):.1f} deg")
    print("Log written to data/log.csv")
    p.disconnect()


if __name__ == "__main__":
    main()
