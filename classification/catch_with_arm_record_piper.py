"""
YOLO 自动抓取 + LeRobot record 数据采集（Piper 版）。

基于 catch_with_arm_record.py (Koch版) 适配松灵 Piper 机械臂。
键盘控制逻辑不变：右箭头=结束当前阶段，左箭头=重录，Esc=停止。
"""

import os
import sys
import time
import select
import tty
import termios
import threading
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
from camera.camera_api import Camera
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
NUM_EPISODES = 1000
EPISODE_TIME_S = 6000
RESET_TIME_S = 6000
VIDEO_CODEC = "h264"
ROBOT_TYPE = "piper_follower"
ROBOT_ID = "piper"
RESUME = True
TARGET_CLASS = "carrot"  # potato,carrot,tomato
TASK = f"pick the {TARGET_CLASS} and place into box"
MOTOR_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
MOTOR_FEATURE_NAMES = [f"{motor}.pos" for motor in MOTOR_NAMES]
MAX_GRIPPER_ANGLE_DEG = 100.0

camera_config: dict[str, OpenCVCameraConfig] = {
    "above": OpenCVCameraConfig(
        index_or_path=int(os.environ.get("LEROBOT_ABOVE_CAMERA", 4)),
        width=640,
        height=480,
        fps=FPS,
    ),
}


def get_place_pos(class_pos: dict, class_name: str) -> list[float]:
    place_config = class_pos.get(class_name)
    if place_config is None:
        configured_places = list(class_pos.values())
        if len(configured_places) == 1:
            place_config = configured_places[0]
        else:
            place_config = {"pos": [0.0, 0.5]}
    if isinstance(place_config, dict):
        return place_config.get("pos", [0.0, 0.5])
    return place_config


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
    for cam_name, cam_cfg in camera_config.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": (cam_cfg.height, cam_cfg.width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


class RecordCameras:
    """Manages detection camera (orb) + LeRobot recording cameras."""

    def __init__(self, detection_camera_name: str = "above"):
        self.detection_camera_name = (
            detection_camera_name if detection_camera_name in camera_config else None
        )
        self.detection_camera = (
            Camera(color=True, depth=False) if self.detection_camera_name else None
        )
        self.last_detection_frame_bgr: np.ndarray | None = None
        self.last_record_frames: dict[str, np.ndarray | None] = {}
        self.cameras = {
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

    def connect(self):
        for camera in self.cameras.values():
            camera.connect()
        if self.detection_camera is not None and self._thread is None:
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._capture_detection_camera,
                name="record-camera-above",
                daemon=True,
            )
            self._thread.start()

    def _capture_detection_camera(self):
        while not self._stop_event.is_set():
            loop_start_t = time.perf_counter()
            capture_interval_s = loop_start_t - self._last_capture_t if self._last_capture_t else 0.0
            self._last_capture_t = loop_start_t
            try:
                camera_frame = self.detection_camera.get_frames().get("color")
            except Exception as exc:
                print(f"读取检测相机失败: {exc}")
                camera_frame = None

            frame_bgr = None
            frame_rgb = None
            if camera_frame is not None and self.detection_camera_name is not None:
                frame_bgr = camera_frame
                config = camera_config[self.detection_camera_name]
                record_frame = camera_frame
                if record_frame.shape[1] != config.width or record_frame.shape[0] != config.height:
                    record_frame = cv2.resize(record_frame, (config.width, config.height))
                frame_rgb = cv2.cvtColor(record_frame, cv2.COLOR_BGR2RGB)

            with self._lock:
                self.last_detection_frame_bgr = None if frame_bgr is None else frame_bgr.copy()
                if self.detection_camera_name is not None:
                    self.last_record_frames[self.detection_camera_name] = (
                        None if frame_rgb is None else frame_rgb.copy()
                    )

            dt_s = time.perf_counter() - loop_start_t
            if capture_interval_s > self._frame_interval * 1.5:
                print(
                    f"相机采集间隔偏大: interval={capture_interval_s * 1000:.1f}ms, "
                    f"read={dt_s * 1000:.1f}ms"
                )
            precise_sleep(max(self._frame_interval - dt_s, 0.0))

    def get_frames(self) -> dict[str, np.ndarray | None]:
        frames = {}
        if self.detection_camera_name is not None:
            with self._lock:
                frame = self.last_record_frames.get(self.detection_camera_name)
                frames[self.detection_camera_name] = None if frame is None else frame.copy()
        for name, camera in self.cameras.items():
            try:
                frames[name] = camera.async_read()
            except TimeoutError:
                frames[name] = None
        return frames

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
        if self.detection_camera is not None:
            self.detection_camera.close()
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()


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
        while not self._record_stop_event.is_set():
            loop_start_t = time.perf_counter()
            action_values = self._get_last_action()
            state = self._get_record_state()
            if state is not None:
                if action_values is None:
                    action_values = state.tolist()
                    self._set_last_action(action_values)
                camera_frames = self.cameras.get_frames()
                if all(image is not None for image in camera_frames.values()):
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


def create_or_resume_dataset() -> LeRobotDataset:
    features = get_features()
    root = Path(DATASET_ROOT)
    if RESUME:
        if not root.exists():
            raise FileNotFoundError(f"RESUME=True, but dataset root does not exist: {root}")
        dataset = LeRobotDataset(
            DATASET_REPO_ID,
            root=DATASET_ROOT,
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


class TerminalKeyPoller:
    """Non-blocking key reader for headless SSH sessions."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def poll(self, events: dict):
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                # Wait briefly for escape sequence (SSH latency)
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    seq = sys.stdin.read(1)
                    if seq == "[" and select.select([sys.stdin], [], [], 0.05)[0]:
                        code = sys.stdin.read(1)
                        if code == "C":  # right arrow
                            events["exit_early"] = True
                        elif code == "D":  # left arrow
                            events["rerecord_episode"] = True
                    else:
                        events["stop_recording"] = True
                else:  # bare Esc
                    events["stop_recording"] = True
            elif ch == "n":  # next episode (same as right arrow)
                events["exit_early"] = True
            elif ch == "r":  # rerecord (same as left arrow)
                events["rerecord_episode"] = True
            elif ch == "q":
                events["stop_recording"] = True

    def stop(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


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

    cameras = RecordCameras()
    cameras.connect()
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

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    key_poller = TerminalKeyPoller()

    print(f"开始录制，数据保存到 {DATASET_ROOT}")
    print("操作方式：")
    print("  n/右箭头 -> 结束当前 episode")
    print("  r/左箭头 -> 丢弃当前 episode 并重录")
    print("  q/Esc    -> 停止录制")

    try:
        while episode_count < NUM_EPISODES and not events["stop_recording"]:
            print(f"\n=== Recording episode {episode_count} ===")
            arm.start_recording()
            episode_start_t = time.perf_counter()
            episode_grasped = False
            detections = []

            while time.perf_counter() - episode_start_t < EPISODE_TIME_S:
                loop_start_t = time.perf_counter()
                key_poller.poll(events)
                if events["exit_early"]:
                    events["exit_early"] = False
                    break

                camera_frames = cameras.get_frames()
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
                        place_pos = get_place_pos(class_pos, class_name)
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
                        arm.catch_and_place(tx, ty, grad, get_place_pos(class_pos, class_name))
                        episode_grasped = True

                for (u, v, w, h, r), score, _class_id, class_name in detections:
                    draw_box(display_frame, u, v, w, h, np.rad2deg(r), f"{class_name}: {score:.2f}")
                cv2.putText(
                    display_frame,
                    f"Episode: {episode_count} | Frames: {dataset.episode_buffer['size']}",
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

            if not events["stop_recording"] and (
                (episode_count < NUM_EPISODES - 1) or events["rerecord_episode"]
            ):
                print("Reset the environment")
                reset_start_t = time.perf_counter()
                while time.perf_counter() - reset_start_t < RESET_TIME_S:
                    loop_start_t = time.perf_counter()
                    key_poller.poll(events)
                    if events["exit_early"] or events["stop_recording"]:
                        events["exit_early"] = False
                        break

                    cameras.get_frames()
                    display_frame = cameras.get_detection_frame()
                    if display_frame is not None:
                        cv2.putText(
                            display_frame,
                            "RESET - Press Right arrow when ready",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                        if HAS_DISPLAY:
                            show_image("Recording", display_frame)

                    dt_s = time.perf_counter() - loop_start_t
                    precise_sleep(max(1 / FPS - dt_s, 0.0))

            if events["rerecord_episode"]:
                print("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            if dataset.episode_buffer["size"] > 0:
                dataset.save_episode()
                episode_count += 1
                print(f"Episode {episode_count} saved")
            else:
                dataset.clear_episode_buffer()

    except KeyboardInterrupt:
        pass
    finally:
        key_poller.stop()
        arm.stop_recording()
        if dataset.episode_buffer["size"] > 0:
            dataset.save_episode()
            print(f"Episode saved (interrupted)")
        dataset.finalize()
        arm.move_to_home(gripper_open_0to1=0.8)
        arm.disconnect_arm()
        cameras.close()
        if HAS_DISPLAY:
            destroy_all_windows()
        print(f"录制完成，共 {episode_count} 个episode，保存在 {DATASET_ROOT}")


if __name__ == "__main__":
    main()
