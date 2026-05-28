"""
YOLO 自动抓取 + LeRobot record 兼容数据采集。

如果本项目旧录制逻辑和 LeRobot record 冲突，以
/home/czn/xxx/lerobot 的 record 配置/流程为准。
"""

import os
import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = Path(os.environ.get("LEROBOT_SRC", "/home/czn/xxx/lerobot/src"))
if not LEROBOT_SRC.exists():
    LEROBOT_SRC = PROJECT_ROOT / "lerobot" / "src"
LEROBOT_ROOT = LEROBOT_SRC.parent
LEROBOT_CODES = LEROBOT_ROOT / "codes"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LEROBOT_SRC))
sys.path.insert(0, str(LEROBOT_CODES))

import cv2
import numpy as np
import yaml

from arm.lerobo_arm_control import LeroboArm
from camera.camera_api import Camera
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from object_detect.detect import detect_objects_in_frame, draw_box, load_model
from x_config import FPS as LEROBOT_RECORD_FPS
from x_config import camera_config as LEROBOT_CAMERA_CONFIG
from x_config import follower_cfg as LEROBOT_FOLLOWER_CFG


# ===== LeRobot record 对齐配置 =====
DATASET_REPO_ID = "a/b"
DATASET_ROOT = "/home/czn/dataset/yolopick"
TASK = "pick the orange toy and place into box"
FPS = LEROBOT_RECORD_FPS
NUM_EPISODES = 1000
EPISODE_TIME_S = 6000
RESET_TIME_S = 6000
VIDEO_CODEC = "libsvtav1"
ROBOT_TYPE = "koch_follower"
ROBOT_ID = LEROBOT_FOLLOWER_CFG.id
RESUME = False
TARGET_CLASS = "orange toy"  # 设为 None 则抓取所有类别


def resolve_record_calibration_dir(calibration_dir: str | Path) -> Path:
    calibration_dir = Path(calibration_dir)
    if not calibration_dir.is_absolute():
        calibration_dir = LEROBOT_ROOT / calibration_dir
    return calibration_dir.resolve()


CALIBRATION_DIR = resolve_record_calibration_dir(LEROBOT_FOLLOWER_CFG.calibration_dir)
LEROBOT_FOLLOWER_CFG.calibration_dir = CALIBRATION_DIR
RECORD_CALIBRATION_FILE = CALIBRATION_DIR / f"{ROBOT_ID}.json"

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
MOTOR_FEATURE_NAMES = [f"{motor}.pos" for motor in MOTOR_NAMES]


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_place_pos(class_pos: dict, class_name: str) -> list[float]:
    place_config = class_pos.get(class_name)
    if place_config is None:
        configured_places = list(class_pos.values())
        if len(configured_places) == 1:
            place_config = configured_places[0]
            print(f"类别 {class_name} 未配置放置点，使用唯一配置的放置点: {place_config}")
        else:
            place_config = {"pos": [0.0, 0.5]}
            print(f"类别 {class_name} 未配置放置点，使用默认右侧放置点: {place_config}")
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
    for camera_name, camera_cfg in LEROBOT_CAMERA_CONFIG.items():
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": (camera_cfg.height, camera_cfg.width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


class RecordCameras:
    """Record LeRobot images while keeping the calibrated detection camera."""

    def __init__(self, detection_camera_name: str = "above"):
        self.detection_camera_name = (
            detection_camera_name
            if detection_camera_name in LEROBOT_CAMERA_CONFIG
            else None
        )
        self.detection_camera = (
            Camera(color=True, depth=False)
            if self.detection_camera_name is not None
            else None
        )
        self.last_detection_frame_bgr: np.ndarray | None = None
        self.cameras = {
            name: OpenCVCamera(config)
            for name, config in LEROBOT_CAMERA_CONFIG.items()
            if name != self.detection_camera_name
        }

    def connect(self):
        for camera in self.cameras.values():
            camera.connect()

    def get_frames(self) -> dict[str, np.ndarray | None]:
        frames = {}
        if self.detection_camera is not None and self.detection_camera_name is not None:
            camera_frame = self.detection_camera.get_frames().get("color")
            if camera_frame is None:
                frames[self.detection_camera_name] = None
                self.last_detection_frame_bgr = None
            else:
                self.last_detection_frame_bgr = camera_frame
                config = LEROBOT_CAMERA_CONFIG[self.detection_camera_name]
                record_frame = camera_frame
                if (
                    record_frame.shape[1] != config.width
                    or record_frame.shape[0] != config.height
                ):
                    record_frame = cv2.resize(record_frame, (config.width, config.height))
                frames[self.detection_camera_name] = cv2.cvtColor(
                    record_frame,
                    cv2.COLOR_BGR2RGB,
                )
        for name, camera in self.cameras.items():
            try:
                frames[name] = camera.async_read()
            except TimeoutError:
                frames[name] = None
        return frames

    def get_detection_frame(self) -> np.ndarray | None:
        if self.last_detection_frame_bgr is not None:
            return self.last_detection_frame_bgr.copy()
        return None

    def close(self):
        if self.detection_camera is not None:
            self.detection_camera.close()
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()


class RecordingArm(LeroboArm):
    """Use the normal arm controller and record its action stream."""

    def __init__(self, dataset: LeRobotDataset, cameras: RecordCameras, task: str, fps: int, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.cameras = cameras
        self.task = task
        self.record_interval = 1.0 / fps
        self.last_record_time = 0.0
        self.recording = False
        self.record_calibration = load_json(RECORD_CALIBRATION_FILE)

    def start_recording(self):
        self.recording = True
        self.last_record_time = 0.0

    def stop_recording(self):
        self.recording = False

    def _action_dict_to_values(self, action: dict[str, float]) -> list[float]:
        values = []
        for motor_name in MOTOR_NAMES:
            key = f"{motor_name}.pos"
            values.append(float(action[key] if key in action else action[motor_name]))
        return values

    def _raw_to_record_values(self, raw_values: dict[str, float]) -> list[float]:
        values = []
        for motor_name in MOTOR_NAMES:
            calib = self.record_calibration[motor_name]
            min_value = calib["range_min"]
            max_value = calib["range_max"]
            drive_mode = calib["drive_mode"]
            raw_value = min(max_value, max(min_value, raw_values[motor_name]))
            if motor_name == "gripper":
                norm = ((raw_value - min_value) / (max_value - min_value)) * 100
                values.append(100 - norm if drive_mode else norm)
            else:
                norm = (((raw_value - min_value) / (max_value - min_value)) * 200) - 100
                values.append(-norm if drive_mode else norm)
        return values

    def _control_action_to_record_values(self, action: dict[str, float]) -> list[float]:
        if self.arm_backend == "sim":
            return self._action_dict_to_values(action)

        motor_targets = {
            key.removesuffix(".pos"): value
            for key, value in action.items()
            if key.endswith(".pos")
        }
        raw_values = self.arm.bus._unnormalize(
            {
                self.arm.bus.motors[motor_name].id: motor_targets[motor_name]
                for motor_name in motor_targets
            }
        )
        raw_by_name = {
            self.arm.bus._id_to_name(motor_id): raw_value
            for motor_id, raw_value in raw_values.items()
        }
        return self._raw_to_record_values(raw_by_name)

    def _get_record_state(self) -> np.ndarray | None:
        if self.arm_backend == "sim":
            angles, gripper = self.get_arm_angles()
            if angles is None or gripper is None:
                return None
            values = [
                angle + offset for angle, offset in zip(angles, self.offset, strict=True)
            ]
            values.append(gripper * self.MAX_GRIPPER_ANGLE_DEG)
            return np.array(values, dtype=np.float32)

        state = self.arm.bus.sync_read("Present_Position", normalize=False)
        return np.array(self._raw_to_record_values(state), dtype=np.float32)

    def _record_frame(self, action_values: list[float]):
        if not self.recording:
            return
        now = time.time()
        if now - self.last_record_time < self.record_interval:
            return
        self.last_record_time = now

        state = self._get_record_state()
        if state is None:
            return

        camera_frames = self.cameras.get_frames()
        if any(image is None for image in camera_frames.values()):
            return

        frame = {
            "observation.state": state,
            "action": np.array(action_values, dtype=np.float32),
            "task": self.task,
        }
        for camera_name, image in camera_frames.items():
            frame[f"observation.images.{camera_name}"] = image
        self.dataset.add_frame(frame)

    def record_hold_frame(self):
        """Record a frame where arm is idle: action = current state (not moving)."""
        state = self._get_record_state()
        if state is not None:
            self._record_frame(state.tolist())

    def _record_step(self, step_info: dict):
        action = step_info.get("sent_action")
        if action is None:
            joint_names = step_info["joint_names"]
            action = {
                motor_name + ".pos": angle + self.offset[index]
                for index, (motor_name, angle) in enumerate(
                    zip(
                        joint_names[:-1],
                        step_info["target_joint_angles_deg"],
                        strict=True,
                    )
                )
            }
            action["gripper.pos"] = (
                step_info["target_gripper_open_0to1"]
                * self.MAX_GRIPPER_ANGLE_DEG
            )
        if action is not None:
            self._record_frame(self._control_action_to_record_values(action))

    def set_arm_angles(
        self,
        angles_deg=None,
        gripper_open_0to1=None,
        step_callback=None,
    ):
        callbacks = []
        if step_callback is not None:
            callbacks.append(step_callback)
        if self.recording:
            callbacks.append(self._record_step)

        def combined_step_callback(step_info: dict):
            for callback in callbacks:
                callback(step_info)

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
            raise ValueError(f"Dataset robot_type mismatch: expected {ROBOT_TYPE}, got {dataset.meta.robot_type}")
        missing_features = sorted(set(features) - set(dataset.features))
        if missing_features:
            raise ValueError(f"Dataset missing LeRobot record features: {missing_features}")
        for key, expected in features.items():
            actual = dataset.features[key]
            if actual["dtype"] != expected["dtype"] or tuple(actual["shape"]) != tuple(expected["shape"]):
                raise ValueError(f"Dataset feature mismatch for {key}: expected {expected}, got {actual}")
            if expected.get("names") is not None and actual.get("names") != expected.get("names"):
                raise ValueError(f"Dataset feature names mismatch for {key}: expected {expected['names']}, got {actual.get('names')}")
        dataset.start_image_writer(
            num_processes=0,
            num_threads=4 * len(LEROBOT_CAMERA_CONFIG),
        )
        return dataset

    if root.exists():
        raise FileExistsError(f"RESUME=False, but dataset root already exists: {root}")

    return LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        fps=FPS,
        features=features,
        root=DATASET_ROOT,
        robot_type=ROBOT_TYPE,
        use_videos=True,
        image_writer_threads=4 * len(LEROBOT_CAMERA_CONFIG),
        vcodec=VIDEO_CODEC,
    )


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
    offset = config_yaml.get("catch_offset", 0.00)

    cameras = RecordCameras()
    cameras.connect()
    dataset = create_or_resume_dataset()

    arm = RecordingArm(
        dataset=dataset,
        cameras=cameras,
        task=TASK,
        fps=FPS,
        calibration_dir=CALIBRATION_DIR,
        robot_id=ROBOT_ID,
    )
    arm.move_to_home(gripper_open_0to1=0.8)

    models = [load_model(path) for path in model_paths]
    episode_count = dataset.num_episodes

    listener, events = init_keyboard_listener()

    print(f"开始录制，数据保存到 {DATASET_ROOT}")
    print("操作方式（LeRobot record）：")
    print("  右箭头 -> 结束当前阶段")
    print("  左箭头 -> 丢弃当前 episode 并重录")
    print("  Esc    -> 停止录制")

    try:
        while episode_count < NUM_EPISODES and not events["stop_recording"]:
            print(f"\n=== Recording episode {episode_count} ===")
            arm.start_recording()
            episode_start_t = time.perf_counter()
            episode_grasped = False
            detections = []

            while time.perf_counter() - episode_start_t < EPISODE_TIME_S:
                loop_start_t = time.perf_counter()
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
                            detect_objects_in_frame(
                                model,
                                display_frame,
                                conf_thres=default_conf_thres,
                            )
                        )

                    target = None
                    for (u, v, w, h, r), _score, _class_id, class_name in detections:
                        if TARGET_CLASS and class_name != TARGET_CLASS:
                            continue

                        target_x, target_y = arm.pixel2pos(u, v)
                        place_pos = get_place_pos(class_pos, class_name)
                        if (
                            np.linalg.norm(
                                np.array(place_pos) - np.array([target_x, target_y])
                            )
                            < place_distance_threshold
                        ):
                            continue

                        angle_deg = np.rad2deg(r)
                        gripper_angle_rad = arm.gripper_angle_by_longer(u, v, w, h, angle_deg)

                        target = (
                            target_x + offset * np.cos(gripper_angle_rad),
                            target_y + offset * np.sin(-gripper_angle_rad),
                            gripper_angle_rad,
                            class_name,
                        )
                        break

                    if target is not None:
                        tx, ty, grad, class_name = target
                        arm.catch_and_place(tx, ty, grad, get_place_pos(class_pos, class_name))
                        episode_grasped = True

                for (u, v, w, h, r), score, _class_id, class_name in detections:
                    draw_box(
                        display_frame,
                        u,
                        v,
                        w,
                        h,
                        np.rad2deg(r),
                        f"{class_name}: {score:.2f}",
                    )
                cv2.putText(
                    display_frame,
                    f"Episode: {episode_count} | Frames: {dataset.episode_buffer['size']}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Recording", display_frame)
                cv2.waitKey(1)

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
                        cv2.imshow("Recording", display_frame)
                        cv2.waitKey(1)

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
        if listener is not None:
            listener.stop()
        arm.stop_recording()
        dataset.finalize()
        arm.move_to_home(gripper_open_0to1=0.8)
        arm.disconnect_arm()
        cameras.close()
        cv2.destroyAllWindows()
        print(f"录制完成，共 {episode_count} 个episode，保存在 {DATASET_ROOT}")


if __name__ == "__main__":
    main()
