#!/usr/bin/env python3
"""Replay a LeRobot episode on a Piper follower arm."""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = Path(os.environ.get("LEROBOT_SRC", PROJECT_ROOT / "lerobot" / "src"))
LEROBOT_CODES = Path(os.environ.get("LEROBOT_CODES", LEROBOT_SRC.parent / "codes"))

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LEROBOT_SRC))
sys.path.insert(0, str(LEROBOT_CODES))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfig
from lerobot.robots.piper_follower.piper_follower import PiperFollower
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

DEFAULT_REPO_ID = "piper_yolopick"
DEFAULT_DATASET_ROOT = "/home/czn/dataset/piper_yolopick"
DEFAULT_PORT = "can4"
DEFAULT_ROBOT_ID = "piper"


def set_joint_move_mode(robot: PiperFollower, timeout_s: float = 5.0) -> None:
    """Switch Piper to joint-control mode before sending JointCtrl actions."""
    start_t = time.time()
    robot.piper.MotionCtrl_2(
        ctrl_mode=0x01,
        move_mode=0x01,
        move_spd_rate_ctrl=100,
        is_mit_mode=0x00,
    )
    while True:
        status = robot.piper.GetArmStatus()
        if status.arm_status.mode_feed == 0x01:
            return
        if time.time() - start_t > timeout_s:
            raise TimeoutError("Failed to switch Piper to joint-control mode.")
        time.sleep(0.01)


def replay(
    episode_idx: int = 0,
    dataset_root: str = DEFAULT_DATASET_ROOT,
    repo_id: str = DEFAULT_REPO_ID,
    port: str = DEFAULT_PORT,
    robot_id: str = DEFAULT_ROBOT_ID,
    fps: int | None = None,
    play_sounds: bool = False,
) -> None:
    follower_cfg = PiperFollowerConfig(
        port=port,
        id=robot_id,
        calibration_dir=None,
    )

    robot = PiperFollower(follower_cfg)
    dataset = LeRobotDataset(repo_id, episodes=[episode_idx], root=dataset_root)
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_idx)
    if len(episode_frames) == 0:
        raise ValueError(f"Episode {episode_idx} not found in dataset: {dataset_root}")

    actions = episode_frames.select_columns("action")
    action_names = dataset.features["action"]["names"]
    target_fps = fps or dataset.fps

    robot.connect()
    try:
        set_joint_move_mode(robot)
        log_say(f"Replaying episode {episode_idx}", play_sounds)
        for idx in range(len(episode_frames)):
            t0 = time.perf_counter()

            action_values = actions[idx]["action"]
            action = {
                name: float(action_values[i])
                for i, name in enumerate(action_names)
            }
            robot.send_action(action)

            precise_sleep(max(1.0 / target_fps - (time.perf_counter() - t0), 0.0))
    finally:
        robot.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a LeRobot episode on a Piper follower arm.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--play-sounds", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    replay(
        episode_idx=args.episode,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        port=args.port,
        robot_id=args.robot_id,
        fps=args.fps,
        play_sounds=args.play_sounds,
    )
