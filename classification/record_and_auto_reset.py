"""
Integrated recording + auto-reset data-collection pipeline (Piper 版).

Workflow per episode:
1. Recording phase — YOLO detect target → catch → place into box → save MP4 (LeRobot)
2. Auto-reset phase — YOLO detect all objects → move each to random position
3. Repeat until NUM_EPISODES reached

Combines catch_with_arm_record_piper.py (recording) and auto_reset_record.py (reset).
Keyboard: n/→ = end episode, r/← = rerecord, q/Esc = stop, s = skip reset.
"""

import os
import random
import select
import shutil
import sys
import threading
import time
import termios
import tty
from math import inf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = Path(os.environ.get("LEROBOT_SRC", "/home/czn/xxx/lerobot/src"))
LEROBOT_ROOT = LEROBOT_SRC.parent
LEROBOT_CODES = LEROBOT_ROOT / "codes"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LEROBOT_SRC))
sys.path.insert(0, str(LEROBOT_CODES))

import cv2
import numpy as np
import yaml

from arm.piper_ctrl_by_sdk import PiperBySDK
from camera import orb_camera  # pyorbbecsdk — required for Orbbec Gemini 336L
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import precise_sleep
from object_detect.detect import detect_objects_in_frame, draw_box, load_model
from utils.config_getter import get_config_value
from utils.cv2_display import show_image, show_img_by_web, destroy_all_windows

HAS_DISPLAY = os.environ.get("DISPLAY") is not None or show_img_by_web()


# ===== 配置 =====
FPS = 30
DATASET_REPO_ID = "a/b"
DATASET_ROOT = "/home/czn/dataset/piper_yolopick"
NUM_EPISODES = 20
EPISODE_TIME_S = 6000
VIDEO_CODEC = "h264"
ROBOT_TYPE = "piper_follower"
ROBOT_ID = "piper"
RESUME = True
TARGET_CLASS = "carrot"  # potato,carrot,tomato
TASK = f"pick the {TARGET_CLASS} toy and place into box"
MOTOR_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
MOTOR_FEATURE_NAMES = [f"{motor}.pos" for motor in MOTOR_NAMES]
MAX_GRIPPER_ANGLE_DEG = 100.0

# --- auto-reset settings (loaded from config.yaml in main()) ---
WORKSPACE_X_RANGE: tuple[float, float] | None = None
WORKSPACE_Y_RANGE: tuple[float, float] | None = None
RESET_MIN_PLACE_DIST_M: float | None = None
RESET_MAX_OBJECTS_PER_CYCLE: int | None = None

# --- autonomous mode ---

# After a successful catch & place, wait this many seconds then auto-advance
# to the next episode. Set to 0 to advance immediately after grasp.
AUTO_ADVANCE_DELAY_S = 3.0

camera_config: dict[str, OpenCVCameraConfig] = {
    "orbbec": OpenCVCameraConfig(
        index_or_path="/dev/video12",
        width=640,
        height=480,
        fps=FPS,
    ),
    "realsense": OpenCVCameraConfig(
        index_or_path="/dev/video4",
        width=1280,
        height=720,
        fps=FPS,
    ),
}

# Target recording resolution for dataset storage.
# Cameras may output native resolutions that differ from the stored size;
# frames are resized to the target before being added to the dataset.
RECORD_RESOLUTION: dict[str, tuple[int, int]] = {
    "orbbec": (640, 480),
    "realsense": (640, 480),
}


# ===== 辅助函数 (from catch_with_arm_record_piper.py) =====

def get_place_pos(class_pos: dict, class_name: str) -> list[float]:
    place_config = class_pos.get(class_name)
    if place_config is None:
        # Use explicit "default" entry if present, otherwise fall back to hardcoded values.
        default_cfg = class_pos.get("default")
        if isinstance(default_cfg, dict):
            place_config = default_cfg
        else:
            # Legacy: if only one class is configured, use it.
            configured_places = list(class_pos.values())
            if len(configured_places) == 1:
                place_config = configured_places[0]
            else:
                place_config = {"pos": [0.2, 0.4]}
    if isinstance(place_config, dict):
        return place_config.get("pos", [0.2, 0.4])
    return place_config


def resolve_place_pos(place_pos: list, target_x: float, target_y: float) -> list[float]:
    """Resolve symbolic coordinates (x/-x/y/-y) to actual float values."""
    resolved = []
    for val in place_pos:
        if isinstance(val, str):
            match val:
                case "x":   resolved.append(target_x)
                case "-x":  resolved.append(-target_x)
                case "y":   resolved.append(target_y)
                case "-y":  resolved.append(-target_y)
                case _:
                    try:
                        resolved.append(float(val))
                    except ValueError:
                        raise ValueError(f"Unknown place coordinate symbol: {val!r}")
        else:
            resolved.append(float(val))
    return resolved


def get_random_pos_ranges(
    class_pos: dict, class_name: str
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Extract per-class random placement ranges from class_pos config.

    random_pos format: [[x_min, x_max], [y_min, y_max]] — same two-element
    list structure as pos, but each element is a [min, max] range.

    Returns ((x_min, x_max), (y_min, y_max)). Falls back to the "default"
    entry in class_pos, then to hardcoded values.
    """
    HARD_DEFAULT_X = (0.0, 0.5)
    HARD_DEFAULT_Y = (-0.2, 0.3)
    place_config = class_pos.get(class_name)
    if not isinstance(place_config, dict):
        place_config = class_pos.get("default")
    if isinstance(place_config, dict):
        random_cfg = place_config.get("random_pos")
        if isinstance(random_cfg, list) and len(random_cfg) == 2:
            x_range = tuple(random_cfg[0]) if len(random_cfg[0]) == 2 else HARD_DEFAULT_X
            y_range = tuple(random_cfg[1]) if len(random_cfg[1]) == 2 else HARD_DEFAULT_Y
            return x_range, y_range
        elif isinstance(random_cfg, dict):
            # Legacy dict format: {x_range: [...], y_range: [...]}
            x_range = tuple(random_cfg.get("x_range", HARD_DEFAULT_X))
            y_range = tuple(random_cfg.get("y_range", HARD_DEFAULT_Y))
            return x_range, y_range
    return HARD_DEFAULT_X, HARD_DEFAULT_Y


def get_features() -> dict:
    features = {
        "action": {
            "dtype": "float32",
            "shape": (len(MOTOR_FEATURE_NAMES),),
            "names": MOTOR_FEATURE_NAMES,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTOR_FEATURE_NAMES),),
            "names": MOTOR_FEATURE_NAMES,
        },
    }
    for cam_name in camera_config:
        rec_w, rec_h = RECORD_RESOLUTION.get(
            cam_name, (camera_config[cam_name].width, camera_config[cam_name].height)
        )
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": (rec_h, rec_w, 3),
            "names": ["height", "width", "channels"],
        }
    return features


# ===== 相机管理 (from catch_with_arm_record_piper.py) =====

class RecordCameras:
    """Manages detection camera (orbbec/above) + LeRobot recording cameras (realsense/wrist).

    The orbbec detection camera uses pyorbbecsdk (via orb_camera.py) because
    OpenCV's V4L2 backend cannot properly initialise the Orbbec Gemini 336L
    hardware — without the SDK the camera outputs uninitialised/corrupt frames.
    """

    def __init__(self, detection_camera_name: str = "orbbec"):
        self.detection_camera_name = (
            detection_camera_name if detection_camera_name in camera_config else None
        )
        # Orbbec requires the proprietary SDK; OpenCV V4L2 won't work.
        self._orb_pipeline = None
        self._detection_is_orbbec = (self.detection_camera_name == "orbbec")
        # Use OpenCVCamera for detection only if it's NOT an orbbec camera.
        self.detection_camera: OpenCVCamera | None = (
            None if self._detection_is_orbbec
            else OpenCVCamera(camera_config[self.detection_camera_name])
            if self.detection_camera_name
            else None
        )
        self.last_detection_frame_bgr: np.ndarray | None = None
        self.last_record_frames: dict[str, np.ndarray | None] = {}
        # Recording cameras: everything EXCEPT the detection camera.
        self.cameras: dict[str, OpenCVCamera] = {
            name: OpenCVCamera(config)
            for name, config in camera_config.items()
            if name != self.detection_camera_name
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_interval = 1.0 / FPS
        if self.detection_camera_name is not None:
            self._frame_interval = 1.0 / camera_config[self.detection_camera_name].fps
        self._last_capture_t = 0.0
        self._frame_seq: int = 0
        self._last_frame_fingerprint: bytes | None = None

    def connect(self):
        for camera in self.cameras.values():
            camera.connect()
        if self._detection_is_orbbec:
            # Orbbec Gemini 336L must be initialised via pyorbbecsdk.
            self._orb_pipeline = orb_camera.open_camera(color=True, depth=False)
        elif self.detection_camera is not None:
            self.detection_camera.connect()
        if self.detection_camera_name is not None and self._thread is None:
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._capture_detection_camera,
                name="record-camera-detection",
                daemon=True,
            )
            self._thread.start()

    def _capture_detection_camera(self):
        while not self._stop_event.is_set():
            loop_start_t = time.perf_counter()
            capture_interval_s = loop_start_t - self._last_capture_t if self._last_capture_t else 0.0
            self._last_capture_t = loop_start_t
            camera_frame = None
            try:
                if self._detection_is_orbbec and self._orb_pipeline is not None:
                    # orb_camera returns {"color": bgr, "depth": ...} — already BGR.
                    camera_frame = orb_camera.get_frames(self._orb_pipeline).get("color")
                elif self.detection_camera is not None:
                    # OpenCVCamera has a background read thread; read_latest()
                    # returns the most recent frame without blocking.
                    camera_frame = self.detection_camera.read_latest(max_age_ms=500)
            except (TimeoutError, Exception) as exc:
                print(f"读取检测相机失败: {exc}")
                camera_frame = None

            frame_bgr = None
            frame_rgb = None
            if camera_frame is not None and self.detection_camera_name is not None:
                # Content dedup: skip if the camera returned the same image.
                fingerprint = camera_frame[::8, ::8, :].tobytes()
                if fingerprint == self._last_frame_fingerprint:
                    dt_s = time.perf_counter() - loop_start_t
                    precise_sleep(max(self._frame_interval - dt_s, 0.0))
                    continue
                self._last_frame_fingerprint = fingerprint

                if self._detection_is_orbbec:
                    # orb_camera.get_frames() already returns BGR (via frame_to_bgr_image).
                    frame_bgr = camera_frame
                    # Convert BGR → RGB for recording.
                    record_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                else:
                    # OpenCVCamera outputs RGB by default; convert to BGR for YOLO.
                    frame_bgr = cv2.cvtColor(camera_frame, cv2.COLOR_RGB2BGR)
                    record_frame = camera_frame  # already RGB

                rec_w, rec_h = RECORD_RESOLUTION[self.detection_camera_name]
                if record_frame.shape[1] != rec_w or record_frame.shape[0] != rec_h:
                    record_frame = cv2.resize(record_frame, (rec_w, rec_h))
                frame_rgb = record_frame

            with self._lock:
                self.last_detection_frame_bgr = None if frame_bgr is None else frame_bgr.copy()
                if self.detection_camera_name is not None:
                    self.last_record_frames[self.detection_camera_name] = (
                        None if frame_rgb is None else frame_rgb.copy()
                    )
                self._frame_seq += 1

            dt_s = time.perf_counter() - loop_start_t
            if capture_interval_s > self._frame_interval * 1.5:
                print(
                    f"相机采集间隔偏大: interval={capture_interval_s * 1000:.1f}ms, "
                    f"read={dt_s * 1000:.1f}ms"
                )
            precise_sleep(max(self._frame_interval - dt_s, 0.0))

    def get_frames(
        self, last_seq: int = -1
    ) -> tuple[dict[str, np.ndarray | None], int]:
        """Return (frames, seq).  If *last_seq* is given and no new frame has
        arrived since, returns ({}, last_seq) — the caller should skip/retry."""
        with self._lock:
            current_seq = self._frame_seq
            if last_seq >= 0 and current_seq <= last_seq:
                return {}, last_seq
            frames: dict[str, np.ndarray | None] = {}
            if self.detection_camera_name is not None:
                frame = self.last_record_frames.get(self.detection_camera_name)
                frames[self.detection_camera_name] = None if frame is None else frame.copy()
        for name, camera in self.cameras.items():
            try:
                frame = camera.async_read()
                if frame is not None and name in RECORD_RESOLUTION:
                    rec_w, rec_h = RECORD_RESOLUTION[name]
                    if frame.shape[1] != rec_w or frame.shape[0] != rec_h:
                        frame = cv2.resize(frame, (rec_w, rec_h))
                frames[name] = frame
            except TimeoutError:
                frames[name] = None
        return frames, current_seq

    def get_detection_frame(self) -> np.ndarray | None:
        with self._lock:
            if self.last_detection_frame_bgr is not None:
                return self.last_detection_frame_bgr.copy()
        return None

    def close(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._detection_is_orbbec and self._orb_pipeline is not None:
            orb_camera.close_camera(self._orb_pipeline)
            self._orb_pipeline = None
        elif self.detection_camera is not None and self.detection_camera.is_connected:
            self.detection_camera.disconnect()
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()


# ===== 录制机械臂 (from catch_with_arm_record_piper.py) =====

class RecordingArm(PiperBySDK):
    """Piper arm with LeRobot recording capability."""

    def __init__(self, dataset: LeRobotDataset, cameras: RecordCameras, task: str, fps: int, **kwargs):
        self.recording = False
        super().__init__(**kwargs)
        self.dataset = dataset
        self.cameras = cameras
        self.task = task
        self.record_interval = 1.0 / fps
        self.last_action_values: list[float] | None = None
        self._action_lock = threading.Lock()
        self._record_stop_event = threading.Event()
        self._record_thread: threading.Thread | None = None
        self._save_thread: threading.Thread | None = None

    def start_recording(self):
        state = self._get_record_state()
        with self._action_lock:
            self.last_action_values = None if state is None else state.tolist()
        self.recording = True
        self._record_stop_event.clear()
        if self._record_thread is None:
            self._record_thread = threading.Thread(
                target=self._record_loop,
                name="lerobot-record-piper",
                daemon=True,
            )
            self._record_thread.start()

    def stop_recording(self):
        self.recording = False
        self._record_stop_event.set()
        if self._record_thread is not None:
            self._record_thread.join(timeout=2.0)
            self._record_thread = None

    def save_episode_async(self) -> bool:
        """Swap buffer and start background save of the just-recorded episode.

        Must be called after stop_recording(). Extracts the current episode
        buffer, immediately creates a fresh empty buffer for the next episode,
        then kicks off a background thread to save the old one.

        Returns True if a save was started (buffer had frames), False if empty.
        """
        self._wait_save()

        self.dataset._wait_image_writer()

        old_buffer = self.dataset.episode_buffer
        if not isinstance(old_buffer, dict) or old_buffer.get("size", 0) == 0:
            self.dataset.clear_episode_buffer()
            return False

        # Capture the PNG directory index *before* save_episode overwrites it.
        # save_episode() internally sets episode_index from self.meta.total_episodes,
        # which matches the value create_episode_buffer() used when this episode
        # started recording — so the original int is the correct PNG directory.
        episode_index: int = old_buffer.get("episode_index")  # type: ignore[assignment]

        # Create a fresh buffer for the next recording *now*.
        # CRITICAL: override episode_index to a collision-free pending value.
        # The background save is still reading PNGs from episode_{episode_index}/,
        # so the new recording must write PNGs to a different directory.
        # The real episode_index will be set by save_episode() later.
        self.dataset.episode_buffer = self.dataset.create_episode_buffer()
        self.dataset.episode_buffer["episode_index"] = (
            self.dataset.meta.total_episodes + 1
        )

        self._save_thread = threading.Thread(
            target=self._save_episode_bg,
            args=(old_buffer, episode_index),
            name="lerobot-save-episode",
            daemon=True,
        )
        self._save_thread.start()
        return True

    def _save_episode_bg(self, episode_data: dict, episode_index: int):
        """Background episode save — encode video, write parquet, clean up PNGs."""
        try:
            self.dataset.save_episode(episode_data=episode_data)
            # clear_episode_buffer normally deletes temp PNGs, but it's skipped
            # when episode_data is passed — clean up manually here.
            for cam_key in self.dataset.meta.camera_keys:
                img_dir = self.dataset._get_image_file_dir(episode_index, cam_key)
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)
        except Exception as exc:
            print(f"Background save failed for episode {episode_index}: {exc}")

    def _wait_save(self):
        """Wait for any in-progress background save to complete."""
        if self._save_thread is not None:
            self._save_thread.join()
            self._save_thread = None

    def wait_save_complete(self):
        """Public: block until the current background save finishes."""
        self._wait_save()

    def _get_record_state(self) -> np.ndarray | None:
        angles, gripper = self.get_arm_angles(retry_times=0)
        if angles is None or gripper is None:
            return None
        values = list(angles) + [gripper * MAX_GRIPPER_ANGLE_DEG]
        return np.array(values, dtype=np.float32)

    def _set_last_action(self, action_values: list[float]):
        with self._action_lock:
            self.last_action_values = [float(value) for value in action_values]

    def _get_last_action(self) -> list[float] | None:
        with self._action_lock:
            if self.last_action_values is None:
                return None
            return list(self.last_action_values)

    def _record_loop(self):
        next_t = time.perf_counter()
        last_frame_seq = -1
        while not self._record_stop_event.is_set():
            loop_start_t = time.perf_counter()
            action_values = self._get_last_action()
            state = self._get_record_state()
            if state is not None:
                if action_values is None:
                    action_values = state.tolist()
                    self._set_last_action(action_values)
                camera_frames, last_frame_seq = self.cameras.get_frames(
                    last_seq=last_frame_seq
                )
                if camera_frames and all(
                    image is not None for image in camera_frames.values()
                ):
                    frame = {
                        "observation.state": state,
                        "action": np.array(action_values, dtype=np.float32),
                        "task": self.task,
                    }
                    for cam_name, image in camera_frames.items():
                        frame[f"observation.images.{cam_name}"] = image
                    self.dataset.add_frame(frame)

            next_t += self.record_interval
            sleep_s = next_t - time.perf_counter()
            if sleep_s < -self.record_interval:
                print(
                    f"录制循环滞后: lag={-sleep_s * 1000:.1f}ms, "
                    f"loop={((time.perf_counter() - loop_start_t) * 1000):.1f}ms"
                )
                next_t = time.perf_counter()
                sleep_s = self.record_interval
            precise_sleep(max(sleep_s, 0.0))

    def record_hold_frame(self):
        """Recording is driven by the background record loop."""
        return

    def _sleep_and_record(self, duration_s: float):
        precise_sleep(max(duration_s, 0.0))

    def catch(
        self,
        target_x: float,
        target_y: float,
        rot_rad: float,
        height: float = inf,
        step_callback=None,
    ) -> bool:
        target_z = self.desktop_height if height == inf else height

        res = self.move_to(
            [target_x, target_y, target_z + self.catch_raise_height],
            gripper_open_0to1=1,
            rot_rad=rot_rad,
            step_callback=step_callback,
        )
        if not res:
            print("移动到目标位置上方失败，取消抓取")
            self.move_to_home(gripper_open_0to1=1)
            return False
        self._sleep_and_record(self.catch_time_interval_s * 2)

        res = self.move_to(
            [target_x, target_y, target_z],
            gripper_open_0to1=1,
            rot_rad=rot_rad,
            step_callback=step_callback,
        )
        if not res:
            print("移动到目标位置失败，取消抓取")
            self.move_to_home(gripper_open_0to1=1)
            return False
        self._sleep_and_record(self.catch_time_interval_s)

        self.set_gripper(gripper_open_0to1=0, step_callback=step_callback)
        self._sleep_and_record(self.catch_time_interval_s)

        res = self.move_to(
            [target_x, target_y, target_z + self.catch_raise_height],
            rot_rad=rot_rad,
            step_callback=step_callback,
        )
        if not res:
            print("抬起失败，取消抓取")
            self.move_to_home(gripper_open_0to1=1)
            return False
        self._sleep_and_record(self.catch_time_interval_s)

        _, current_gripper_open_0to1 = self.get_arm_angles()
        if (
            current_gripper_open_0to1 is None
            or current_gripper_open_0to1 < self.default_gripper_close_threshold
        ):
            print("夹取失败")
            self.move_to_home(gripper_open_0to1=1)
            return False
        return True

    def place(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        rot_rad: float = 0,
        step_callback=None,
    ) -> bool:
        res = self.move_to(
            [target_x, target_y, target_z + self.place_raise_height],
            rot_rad=rot_rad,
            step_callback=step_callback,
        )
        if not res:
            print("移动到放置位置上方失败，取消放置")
            self.move_to_home(gripper_open_0to1=1)
            return False
        self._sleep_and_record(self.catch_time_interval_s * 2)

        down = get_config_value(
            "go_down_before_open_gripper_in_place", False, raise_if_missing=False
        )
        if down:
            res = self.move_to(
                [target_x, target_y, target_z],
                rot_rad=rot_rad,
                step_callback=step_callback,
            )
            if not res:
                print("移动到放置位置失败，取消放置")
                self.move_to_home(gripper_open_0to1=1)
                return False
            self._sleep_and_record(self.catch_time_interval_s)

        self.set_gripper(gripper_open_0to1=1, step_callback=step_callback)

        if down:
            res = self.move_to(
                [target_x, target_y, target_z + self.place_raise_height],
                rot_rad=rot_rad,
                step_callback=step_callback,
            )
            if not res:
                print("移动到放置位置上方失败，取消放置")
                self.move_to_home(gripper_open_0to1=1)
                return False
            self._sleep_and_record(self.catch_time_interval_s)

        return True

    def _record_step(self, step_info: dict):
        """Callback for set_arm_angles to record each interpolation step."""
        sent_action = step_info.get("sent_action")
        if sent_action is not None:
            action_values = [float(sent_action[name]) for name in MOTOR_FEATURE_NAMES]
        else:
            target_angles = step_info.get("target_joint_angles_deg")
            target_gripper = step_info.get("target_gripper_open_0to1")
            if target_angles is None:
                return
            action_values = [float(angle) for angle in target_angles]
            gripper_deg = (target_gripper or 0.0) * MAX_GRIPPER_ANGLE_DEG
            action_values.append(gripper_deg)
        self._set_last_action(action_values)

    def set_arm_angles(self, angles_deg=None, gripper_open_0to1=None, step_callback=None):
        callbacks = []
        if step_callback is not None:
            callbacks.append(step_callback)
        if self.recording:
            callbacks.append(self._record_step)

        def combined_step_callback(step_info: dict):
            for cb in callbacks:
                cb(step_info)

        return super().set_arm_angles(
            angles_deg,
            gripper_open_0to1=gripper_open_0to1,
            step_callback=combined_step_callback if callbacks else None,
        )


# ===== 键盘输入（headless SSH 兼容）=====

class TerminalKeyPoller:
    """Non-blocking key reader for headless SSH sessions."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def poll(self, events: dict):
        """Key mapping (matches lerobot-record's pynput behavior):

           右箭头 / n  → exit_early (immediately ends current loop)
           左箭头 / r  → rerecord_episode + exit_early (ends loop, discards episode)
           Esc / q     → stop_recording + exit_early (ends loop, stops entire recording)
           s           → skip_reset (skip auto-reset this cycle)
        """
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                # Wait briefly for escape sequence (SSH latency)
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    seq = sys.stdin.read(1)
                    if seq == "[" and select.select([sys.stdin], [], [], 0.05)[0]:
                        code = sys.stdin.read(1)
                        if code == "C":  # right arrow → next episode
                            events["exit_early"] = True
                        elif code == "D":  # left arrow → rerecord + exit loop
                            events["rerecord_episode"] = True
                            events["exit_early"] = True
                    else:
                        events["stop_recording"] = True
                        events["exit_early"] = True
                else:  # bare Esc → stop + exit loop
                    events["stop_recording"] = True
                    events["exit_early"] = True
            elif ch == "n":  # next episode (same as right arrow)
                events["exit_early"] = True
            elif ch == "r":  # rerecord (same as left arrow)
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif ch == "q":  # stop (same as Esc)
                events["stop_recording"] = True
                events["exit_early"] = True
            elif ch == "s":  # skip reset
                events["skip_reset"] = True

    def stop(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


# ===== 数据集管理 =====

def create_or_resume_dataset() -> LeRobotDataset:
    features = get_features()
    root = Path(DATASET_ROOT)
    if RESUME:
        if not root.exists():
            print(f"RESUME=True but dataset root does not exist — creating new dataset at {root}")
            return LeRobotDataset.create(
                repo_id=DATASET_REPO_ID,
                fps=FPS,
                root=DATASET_ROOT,
                features=features,
                robot_type=ROBOT_TYPE,
                use_videos=True,
                image_writer_threads=4 * len(camera_config),
                batch_encoding_size=1,
                vcodec=VIDEO_CODEC,
            )

        # Verify the existing dataset has valid metadata before loading.
        # If meta/episodes is missing or empty, LeRobotDataset.__init__ falls
        # through to HF hub code (get_safe_version → list_repo_refs) which
        # fails for fake repo_ids like "a/b".  Recreate instead.
        meta_episodes = root / "meta" / "episodes"
        if not meta_episodes.is_dir() or not list(meta_episodes.rglob("*.parquet")):
            print(f"Incomplete dataset at {root} (no episode metadata), recreating...")
            shutil.rmtree(root)
            return LeRobotDataset.create(
                repo_id=DATASET_REPO_ID,
                fps=FPS,
                root=DATASET_ROOT,
                features=features,
                robot_type=ROBOT_TYPE,
                use_videos=True,
                image_writer_threads=4 * len(camera_config),
                batch_encoding_size=1,
                vcodec=VIDEO_CODEC,
            )

        # revision="local" skips the HF Hub get_safe_version() check
        # (is_valid_version("local") returns False — it's not PEP 440).
        # Required because DATASET_REPO_ID is fake ("a/b") and the Jetson
        # may not have internet access to the HF Hub.
        try:
            dataset = LeRobotDataset(
                DATASET_REPO_ID,
                root=DATASET_ROOT,
                revision="local",
                batch_encoding_size=1,
                vcodec=VIDEO_CODEC,
            )
        except Exception as exc:
            # Corrupted parquet files (e.g. from interrupted writes or power loss)
            # prevent LeRobotDataset from loading metadata OR episode data.
            # LeRobotDataset.__init__ first loads meta/episodes/*.parquet, then
            # loads data/*/*.parquet — either step can fail.  Scan both trees,
            # remove every corrupted/empty parquet, and retry.
            print(f"Failed to load dataset: {exc}")
            import pyarrow.parquet as pq
            corrupted_files = []
            for scan_dir in (root / "meta", root / "data"):
                if not scan_dir.is_dir():
                    continue
                for pf in sorted(scan_dir.rglob("*.parquet")):
                    try:
                        table = pq.read_table(str(pf))
                        del table
                    except Exception:
                        corrupted_files.append(pf)
                    else:
                        if pf.stat().st_size == 0:
                            corrupted_files.append(pf)
            if corrupted_files:
                print(f"Found {len(corrupted_files)} corrupted/empty parquet file(s):")
                for cf in corrupted_files:
                    print(f"  Removing: {cf}")
                    cf.unlink(missing_ok=True)
            else:
                print("No corrupted parquet files found — re-raising original error")
                raise
            print("Reinitializing dataset with remaining valid episodes...")
            dataset = LeRobotDataset(
                DATASET_REPO_ID,
                root=DATASET_ROOT,
                revision="local",
                batch_encoding_size=1,
                vcodec=VIDEO_CODEC,
            )
        if dataset.fps != FPS:
            raise ValueError(f"Dataset fps mismatch: expected {FPS}, got {dataset.fps}")
        if dataset.meta.robot_type != ROBOT_TYPE:
            raise ValueError(
                f"Dataset robot_type mismatch: expected {ROBOT_TYPE}, got {dataset.meta.robot_type}"
            )
        for key in dataset.meta.video_keys:
            codec = dataset.features[key].get("info", {}).get("video.codec")
            if codec is not None and codec != VIDEO_CODEC:
                raise ValueError(
                    f"Dataset video codec mismatch for {key}: "
                    f"expected {VIDEO_CODEC}, got {codec}"
                )
        missing_features = sorted(set(features) - set(dataset.features))
        if missing_features:
            raise ValueError(f"Dataset missing LeRobot record features: {missing_features}")
        for key, expected in features.items():
            actual = dataset.features[key]
            if actual["dtype"] != expected["dtype"] or tuple(actual["shape"]) != tuple(expected["shape"]):
                raise ValueError(
                    f"Dataset feature mismatch for {key}: expected {expected}, got {actual}"
                )
            if expected.get("names") is not None and actual.get("names") != expected.get("names"):
                raise ValueError(
                    f"Dataset feature names mismatch for {key}: "
                    f"expected {expected['names']}, got {actual.get('names')}"
                )
        dataset.start_image_writer(
            num_processes=0,
            num_threads=4 * len(camera_config),
        )
        return dataset

    if root.exists():
        raise FileExistsError(f"RESUME=False, but dataset root already exists: {root}")

    return LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        fps=FPS,
        root=DATASET_ROOT,
        features=features,
        robot_type=ROBOT_TYPE,
        use_videos=True,
        image_writer_threads=4 * len(camera_config),
        batch_encoding_size=1,
        vcodec=VIDEO_CODEC,
    )


# ===== Auto-Reset（集成自 auto_reset_record.py）=====

def _pos_in_workspace(x: float, y: float) -> bool:
    """Check whether (x, y) is within the arm's reachable workspace."""
    wx_min, wx_max = WORKSPACE_X_RANGE
    wy_min, wy_max = WORKSPACE_Y_RANGE
    return (wx_min <= x <= wx_max) and (wy_min <= y <= wy_max)


def auto_reset_phase(
    arm: RecordingArm,
    cameras: RecordCameras,
    models: list,
    default_conf_thres: float,
    target_class: str,
    key_poller: TerminalKeyPoller,
    events: dict,
    class_pos: dict,
) -> int:
    """Move objects from the collection box back to random table positions.

    Only resets up to RESET_MAX_OBJECTS_PER_CYCLE objects (default 1) since
    each recording episode collects one object.  The arm is NOT recording
    during this phase (arm.recording == False).

    Returns:
        Number of objects successfully reset.
    """
    print("\n=== Auto-Reset Phase ===")

    # Move arm to home so the camera has a clear view of the table.
    arm.move_to_home(gripper_open_0to1=0.8)
    time.sleep(1.0)  # let camera stabilize & capture clean frames

    objects_reset = 0

    # Each recording episode only grabs one object, so the reset phase
    # only needs to handle what it finds. Loop until no targets remain.
    _safety_limit = 50  # unreachable safety cap to prevent infinite loops
    for attempt in range(_safety_limit):
        # — check for user interrupt —
        key_poller.poll(events)
        if events["stop_recording"]:
            print("  Auto-reset stopped by user")
            break
        if events.get("skip_reset"):
            events["skip_reset"] = False
            print("  Auto-reset skipped by user")
            break

        # — get fresh camera frame —
        display_frame = cameras.get_detection_frame()
        if display_frame is None:
            print(f"  No camera frame (attempt {attempt + 1})")
            time.sleep(0.5)
            continue

        # — YOLO detection (same confidence threshold as recording) —
        detections = []
        for model in models:
            detections.extend(
                detect_objects_in_frame(model, display_frame, conf_thres=default_conf_thres)
            )

        # — filter targets by class and workspace —
        targets = []
        for (u, v, w, h, r), score, _class_id, class_name in detections:
            if target_class and class_name != target_class:
                continue
            target_x, target_y = arm.pixel2pos(u, v)
            # Reject detections outside the arm's reachable workspace
            # (e.g. YOLO false positives on the arm itself).
            if not _pos_in_workspace(target_x, target_y):
                print(f"  Skipping {class_name} at ({target_x:.3f},{target_y:.3f}) — outside workspace")
                continue
            angle_deg = np.rad2deg(r)
            gripper_angle_rad = arm.gripper_angle_by_longer(u, v, w, h, angle_deg)
            targets.append((target_x, target_y, gripper_angle_rad, class_name, u, v, w, h, r, score))

        if not targets:
            print(f"  No targets found — reset complete ({objects_reset} objects moved)")
            break

        # — pick the first target and generate a random place position —
        tx, ty, grad, class_name, u, v, w, h, r, score = targets[0]

        # Each class has its own random_pos region in class_pos config,
        # so we don't need per-object distance checks — the regions
        # themselves prevent objects from being placed on top of each other.
        (rx_min, rx_max), (ry_min, ry_max) = get_random_pos_ranges(class_pos, class_name)

        place_x = random.uniform(rx_min, rx_max)
        place_y = random.uniform(ry_min, ry_max)
        if not _pos_in_workspace(place_x, place_y):
            # Clamp to workspace if the class region exceeds it.
            place_x = max(WORKSPACE_X_RANGE[0], min(WORKSPACE_X_RANGE[1], place_x))
            place_y = max(WORKSPACE_Y_RANGE[0], min(WORKSPACE_Y_RANGE[1], place_y))

        print(
            f"  Reset [{objects_reset + 1}]: {class_name} "
            f"({tx:.3f},{ty:.3f}) -> random({place_x:.3f},{place_y:.3f}) "
            f"score={score:.2f}"
        )

        # — draw detection overlay for visual feedback —
        for (du, dv, dw, dh, dr), dscore, _did, dname in detections:
            draw_box(display_frame, du, dv, dw, dh, np.rad2deg(dr), f"{dname}: {dscore:.2f}")
        cv2.putText(
            display_frame,
            f"RESET [{objects_reset + 1}/{len(targets)}]",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        if HAS_DISPLAY:
            show_image("Recording", display_frame)

        # — execute pick & random-place (recording=False, so no data written) —
        success = arm.catch_and_place(tx, ty, grad, [place_x, place_y])
        if success:
            objects_reset += 1
            if objects_reset >= RESET_MAX_OBJECTS_PER_CYCLE:
                print(f"  Reset limit reached ({RESET_MAX_OBJECTS_PER_CYCLE}) — moving to next episode")
                break
        else:
            print(f"  Warning: catch_and_place failed for {class_name}, continuing...")

        # — Stabilization delay: catch_and_place already moved the arm home,
        #    but the camera thread may still have frames showing the arm in
        #    motion.  Wait for clean frames before the next detection.
        time.sleep(1.0)

    print(f"  Auto-reset done: {objects_reset} objects scattered")
    return objects_reset


# ===== 主循环 =====

def main():
    config_path = PROJECT_ROOT / "config.yaml"
    config_yaml = yaml.safe_load(open(config_path, encoding="utf-8"))

    model_paths = [
        str(PROJECT_ROOT / path)
        for path in config_yaml.get("classification_YOLO_model_path", [])
    ]
    default_conf_thres = config_yaml.get("default_conf_thres", 0.8)
    class_pos = config_yaml.get("class_pos", {})
    place_distance_threshold = config_yaml.get("place_distance_threshold", 0.03)

    # Load auto-reset settings from config (fall back to defaults if keys are missing).
    global WORKSPACE_X_RANGE, WORKSPACE_Y_RANGE
    global RESET_MIN_PLACE_DIST_M, RESET_MAX_OBJECTS_PER_CYCLE
    WORKSPACE_X_RANGE = tuple(config_yaml.get("workspace_x_range", (-0.1, 0.55)))
    WORKSPACE_Y_RANGE = tuple(config_yaml.get("workspace_y_range", (-0.3, 0.55)))
    RESET_MIN_PLACE_DIST_M = config_yaml.get("reset_min_place_dist_m", 0.20)
    RESET_MAX_OBJECTS_PER_CYCLE = config_yaml.get("reset_max_objects_per_cycle", 1)

    cameras = RecordCameras()
    cameras.connect()

    # — warm up the headless display server early so the browser can connect
    #    before the first recording episode starts (Flask takes ~2-5 s on Jetson).
    if HAS_DISPLAY:
        _warmup = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(_warmup, "Starting...", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        show_image("Recording", _warmup)
        print("Headless display server warming up...")

    dataset = create_or_resume_dataset()

    arm = RecordingArm(
        dataset=dataset,
        cameras=cameras,
        task=TASK,
        fps=FPS,
    )
    arm.timeout = 15
    arm.move_to_home(gripper_open_0to1=0.8)

    models = [load_model(path) for path in model_paths]
    episode_count = dataset.num_episodes

    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "skip_reset": False,
    }
    key_poller = TerminalKeyPoller()

    print(f"Integrated record + auto-reset pipeline")
    print(f"  Data saved to: {DATASET_ROOT}")
    print(f"  Target class:  {TARGET_CLASS}")
    print(f"  Episodes:      {NUM_EPISODES}")
    print(f"  Resume:        {RESUME} (starting from episode {episode_count})")
    print("Controls:")
    print("  n/→  = end current episode")
    print("  r/←  = discard episode & rerecord")
    print("  s    = skip auto-reset this cycle")
    print("  q/Esc = stop entirely")

    target_episodes = dataset.num_episodes + NUM_EPISODES
    print(f"  Target:         {target_episodes} episodes total")
    try:
        while episode_count < target_episodes and not events["stop_recording"]:
            # ================================================================
            #  Phase 1 — Recording
            # ================================================================
            print(f"\n=== Recording episode {episode_count} ===")
            arm.start_recording()
            episode_start_t = time.perf_counter()
            episode_grasped = False
            grasp_time = 0.0
            detections = []

            while time.perf_counter() - episode_start_t < EPISODE_TIME_S:
                loop_start_t = time.perf_counter()
                key_poller.poll(events)
                if events["exit_early"]:
                    events["exit_early"] = False
                    break

                display_frame = cameras.get_detection_frame()
                if display_frame is None:
                    precise_sleep(1 / FPS)
                    continue

                if not episode_grasped:
                    detections = []
                    for model in models:
                        detections.extend(
                            detect_objects_in_frame(model, display_frame, conf_thres=default_conf_thres)
                        )

                    target = None
                    for (u, v, w, h, r), _score, _class_id, class_name in detections:
                        if TARGET_CLASS and class_name != TARGET_CLASS:
                            continue

                        target_x, target_y = arm.pixel2pos(u, v)
                        place_pos = resolve_place_pos(
                            get_place_pos(class_pos, class_name), target_x, target_y
                        )
                        if (
                            np.linalg.norm(np.array(place_pos) - np.array([target_x, target_y]))
                            < place_distance_threshold
                        ):
                            continue

                        angle_deg = np.rad2deg(r)
                        gripper_angle_rad = arm.gripper_angle_by_longer(u, v, w, h, angle_deg)

                        target = (
                            target_x,
                            target_y,
                            gripper_angle_rad,
                            class_name,
                        )
                        break

                    if target is not None:
                        tx, ty, grad, class_name = target
                        print(f"目标: pixel=({u:.0f},{v:.0f}) -> pos=({tx:.4f},{ty:.4f}) rot={grad:.3f}rad class={class_name}")
                        arm.catch_and_place(
                            tx, ty, grad,
                            resolve_place_pos(get_place_pos(class_pos, class_name), tx, ty),
                        )
                        episode_grasped = True
                        grasp_time = time.perf_counter()
                        print(f"抓取完成，{AUTO_ADVANCE_DELAY_S}s 后自动进入下一轮...")

                # — auto-advance after grasp: no keyboard needed for unattended runs —
                if episode_grasped and (time.perf_counter() - grasp_time >= AUTO_ADVANCE_DELAY_S):
                    print("Auto-advancing to next episode")
                    break

                for (u, v, w, h, r), score, _class_id, class_name in detections:
                    draw_box(display_frame, u, v, w, h, np.rad2deg(r), f"{class_name}: {score:.2f}")
                if episode_grasped:
                    remaining = max(0, AUTO_ADVANCE_DELAY_S - (time.perf_counter() - grasp_time))
                    status_text = (
                        f"Episode: {episode_count} | Grasped | Auto-advance in {remaining:.1f}s"
                    )
                else:
                    status_text = f"Episode: {episode_count} | Searching..."
                cv2.putText(
                    display_frame,
                    f"{status_text} | Frames: {dataset.episode_buffer['size']}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                if HAS_DISPLAY:
                    show_image("Recording", display_frame)

                arm.record_hold_frame()

                dt_s = time.perf_counter() - loop_start_t
                precise_sleep(max(1 / FPS - dt_s, 0.0))

            arm.stop_recording()

            # — rerecord check —
            if events["rerecord_episode"]:
                print("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                arm.wait_save_complete()
                dataset.clear_episode_buffer()
                continue

            # — save episode (async — MP4 encoding runs in background) —
            episode_saved = arm.save_episode_async()
            if episode_saved:
                episode_count += 1
                print(f"Episode {episode_count} saving in background")
            else:
                print("Episode buffer empty, skipping save")

            # — stop check after save —
            if events["stop_recording"]:
                break

            # ================================================================
            #  Phase 2 — Auto-Reset (scatter objects for next episode)
            # ================================================================
            if episode_count < NUM_EPISODES:
                auto_reset_phase(
                    arm=arm,
                    cameras=cameras,
                    models=models,
                    default_conf_thres=default_conf_thres,
                    target_class=TARGET_CLASS,
                    key_poller=key_poller,
                    events=events,
                    class_pos=class_pos,
                )

                # — rerecord check from reset phase —
                if events["rerecord_episode"]:
                    print(f"Re-record episode (discarding episode {episode_count})")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    arm.wait_save_complete()
                    dataset.clear_episode_buffer()
                    if episode_saved:
                        episode_count -= 1
                    continue

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        key_poller.stop()
        arm.stop_recording()
        arm.wait_save_complete()
        if dataset.episode_buffer is not None and dataset.episode_buffer["size"] > 0:
            dataset.save_episode()
            episode_count += 1
            print(f"Episode {episode_count} saved (interrupted)")
        dataset.finalize()
        arm.move_to_home(gripper_open_0to1=0.8)
        arm.disconnect_arm()
        cameras.close()
        if HAS_DISPLAY:
            destroy_all_windows()
        print(f"Pipeline done — {episode_count} episodes saved to {DATASET_ROOT}")


if __name__ == "__main__":
    main()
