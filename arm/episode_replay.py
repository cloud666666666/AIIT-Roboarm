#!/usr/bin/env python3
"""Replay a recorded episode on the Piper arm using PiperBySDK.

Loads actions from a LeRobot dataset and replays them frame-by-frame with
direct joint control at the original (or overridden) FPS.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = Path(os.environ.get("LEROBOT_SRC", PROJECT_ROOT / "lerobot" / "src"))
LEROBOT_CODES = LEROBOT_SRC.parent / "codes"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LEROBOT_SRC))
sys.path.insert(0, str(LEROBOT_CODES))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from arm.piper_ctrl_by_sdk import PiperBySDK

DEFAULT_DATASET_ROOT = "/home/czn/dataset/piper_yolopick"
DEFAULT_REPO_ID = "a/b"
MAX_GRIPPER_ANGLE_DEG = 100.0
FACTOR = 1000.0


def replay_episode(
    episode_idx: int = 0,
    dataset_root: str = DEFAULT_DATASET_ROOT,
    repo_id: str = DEFAULT_REPO_ID,
    fps: int | None = None,
    speed: float = 1.0,
    loop: bool = False,
) -> None:
    """Replay a recorded episode on the Piper arm.

    Args:
        episode_idx: Index of the episode to replay.
        dataset_root: Root directory of the LeRobot dataset.
        repo_id: LeRobot dataset repo ID.
        fps: Frames per second override. If None, use the dataset's original fps.
        speed: Playback speed multiplier. 1.0 = original speed, 2.0 = double speed.
        loop: If True, loop the episode indefinitely.
    """
    # ---- load dataset and extract episode actions ----
    dataset = LeRobotDataset(repo_id, episodes=[episode_idx], root=dataset_root)

    episode_frames = dataset.hf_dataset.filter(
        lambda x: x["episode_index"] == episode_idx
    )
    num_frames = len(episode_frames)
    if num_frames == 0:
        raise ValueError(
            f"Episode {episode_idx} not found in dataset: {dataset_root}"
        )

    target_fps = (fps or dataset.fps) * speed
    print(
        f"Episode {episode_idx}: {num_frames} frames, "
        f"original_fps={dataset.fps}, replay_fps={target_fps:.1f}"
    )

    # Pre-extract all actions to avoid per-frame dataset overhead.
    actions = []
    for idx in range(num_frames):
        action_values = episode_frames[idx]["action"]
        angles_deg = [float(action_values[i]) for i in range(6)]
        gripper_deg = float(action_values[6])
        gripper_0to1 = float(np.clip(gripper_deg / MAX_GRIPPER_ANGLE_DEG, 0.0, 1.0))
        actions.append((angles_deg, gripper_0to1))

    # ---- connect arm ----
    arm = PiperBySDK()
    arm.move_to_home(gripper_open_0to1=0.8)
    print("Arm initialized and at home position.")

    def _send_joint_action(angles_deg: list[float], gripper_0to1: float) -> None:
        """Send a single joint-angle + gripper command directly (no interpolation)."""
        ctrl = np.array(angles_deg) * FACTOR
        arm.piper.JointCtrl(
            joint_1=int(ctrl[0]),
            joint_2=int(ctrl[1]),
            joint_3=int(ctrl[2]),
            joint_4=int(ctrl[3]),
            joint_5=int(ctrl[4]),
            joint_6=int(ctrl[5]),
        )
        arm.piper.GripperCtrl(
            int(gripper_0to1 * MAX_GRIPPER_ANGLE_DEG * FACTOR),
            gripper_effort=5000,
            gripper_code=0x03,
            set_zero=0,
        )

    try:
        print(f"Replaying episode {episode_idx} ...  (Ctrl+C to stop)")
        frame_interval = 1.0 / target_fps

        while True:
            t_start = time.perf_counter()
            for i, (angles_deg, gripper_0to1) in enumerate(actions):
                t0 = time.perf_counter()
                _send_joint_action(angles_deg, gripper_0to1)

                if i % 60 == 0:
                    print(
                        f"  frame {i + 1}/{num_frames}  "
                        f"j1={angles_deg[0]:.1f} … j6={angles_deg[5]:.1f}  "
                        f"gripper={gripper_0to1:.2f}"
                    )

                elapsed = time.perf_counter() - t0
                time.sleep(max(frame_interval - elapsed, 0.0))

            if not loop:
                break
            print(f"--- loop restart ---")
            # Re-sync: the episode may have drifted, go to home first then restart
            arm.move_to_home(gripper_open_0to1=0.8)

        print(f"Episode {episode_idx} replay finished.")

    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")
    finally:
        arm.move_to_home(gripper_open_0to1=0.8)
        arm.disconnect_arm()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a recorded episode on the Piper arm."
    )
    parser.add_argument(
        "--episode", type=int, default=0,
        help="Episode index to replay (default: 0)",
    )
    parser.add_argument(
        "--dataset-root", default=DEFAULT_DATASET_ROOT,
        help="Root directory of the LeRobot dataset",
    )
    parser.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID,
        help="LeRobot dataset repo ID",
    )
    parser.add_argument(
        "--fps", type=int, default=None,
        help="Replay FPS override (default: use dataset original FPS)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier (default: 1.0, 2.0 = double speed)",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Loop the episode indefinitely",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    replay_episode(
        episode_idx=args.episode,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        fps=args.fps,
        speed=args.speed,
        loop=args.loop,
    )
