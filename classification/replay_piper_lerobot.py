#!/usr/bin/env python3
"""Replay a LeRobot episode on a Piper follower arm."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

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
DEFAULT_PORT = "can0"
DEFAULT_ROBOT_ID = "piper"



def _copy_episode_video(dataset_root: str, episode_idx: int) -> None:
    """Trim and copy only *episode_idx*'s video segment (not the whole chunk)."""
    meta_episodes_dir = Path(dataset_root) / "meta" / "episodes"
    if not meta_episodes_dir.is_dir():
        log_say(f"Warning: {meta_episodes_dir} not found")
        return

    # Locate the row for this episode across all meta parquet files
    row = None
    for chunk_dir in sorted(meta_episodes_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue
        for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
            try:
                table = pq.read_table(str(parquet_file))
            except Exception:
                continue
            for i in range(len(table)):
                if table.column("episode_index")[i].as_py() == episode_idx:
                    row = {col: table.column(col)[i].as_py() for col in table.column_names}
                    break
            if row is not None:
                break
        if row is not None:
            break

    if row is None:
        log_say(f"Warning: Could not find episode {episode_idx} in meta/episodes")
        return

    video_dir = Path(dataset_root) / "videos"
    dest_dir = Path.cwd()

    for camera in ("observation.images.above", "observation.images.wrist"):
        chunk_idx = row.get(f"videos/{camera}/chunk_index")
        file_idx = row.get(f"videos/{camera}/file_index")
        from_ts = row.get(f"videos/{camera}/from_timestamp")
        to_ts = row.get(f"videos/{camera}/to_timestamp")

        if None in (chunk_idx, file_idx, from_ts, to_ts):
            log_say(f"Warning: Missing video metadata for {camera}, episode {episode_idx}")
            continue

        src = video_dir / camera / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
        if not src.exists():
            log_say(f"Warning: {src} not found, skipping")
            continue

        dest = dest_dir / f"ep{episode_idx}_{camera.rsplit('.', 1)[-1]}.mp4"
        # ffmpeg trim: copy codec, no re-encode
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(from_ts), "-to", str(to_ts),
             "-i", str(src), "-c", "copy", str(dest)],
            check=True, capture_output=True,
        )
        log_say(f"Trimmed episode {episode_idx} from {src.name} -> {dest.name}")


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

    _copy_episode_video(dataset_root, episode_idx)

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
    parser.add_argument("--episode", type=int, default=700)
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
