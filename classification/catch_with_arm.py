# Description: 调用目标识别和机械臂控制，实现抓取功能。
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from arm.arm_base import Arm
from utils.config_getter import get_config_value
import argparse
import numpy as np
from object_detect.detect import (
    detect_objects_in_frame,
    load_model,
    draw_box,
)
from camera.camera_api import Camera
import cv2
import time
import concurrent.futures
import copy
from utils.cv2_display import show_image, poll_key, destroy_all_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="调用目标识别和机械臂控制，实现抓取功能。"
    )
    parser.add_argument(
        "--target",
        default='tomato',
        help=(
            "指定要抓取的物体类别名（class_name）。"
            "指定后只抓该类别中置信度最高的物体；"
            "不指定则抓全部检测中置信度最高的物体。"
        ),
    )
    return parser.parse_args()


def main(target_class: str = None):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    model_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
        for path in get_config_value("classification_YOLO_model_path", [])
    ]
    default_gripper_aside_pos = get_config_value(
        "default_gripper_aside_pos", raise_if_missing=False
    )
    default_conf_thres = get_config_value("default_conf_thres")
    place_pos = get_config_value("place_pos", default={}, raise_if_missing=False)
    place_distance_threshold = get_config_value(
        "place_distance_threshold", default=0, raise_if_missing=False
    )
    offset = get_config_value("catch_offset")

    arm = Arm()
    arm.move_to_home(gripper_open_0to1=1)
    cam = Camera(color=True, depth=False, undistort=True)
    models = [load_model(model_path) for model_path in model_paths]
    detections = []
    future = None

    while True:
        start_time = time.time()
        try:
            frames = cam.get_frames()
            if frames is None:
                continue
            frame = frames.get("color")
            if frame is None:
                continue

            frame_w, frame_h = frame.shape[:2]
            if future is None or future.done():
                detections = []
                for model in models:
                    detections.extend(
                        detect_objects_in_frame(
                            model, frame, conf_thres=default_conf_thres
                        )
                    )

            # 先画出所有检测框，便于观察
            for (u, v, w, h, r), score, class_id, class_name in detections:
                draw_box(
                    frame, u, v, w, h, np.rad2deg(r), f"{class_name}: {score:.2f}"
                )

            # 选出本轮要抓取的目标：
            # - 指定了 target_class 时，只在该类别里挑；
            # - 未指定时，从所有检测里挑；
            # 两种情况都取置信度（score）最高的那个。
            if future is None or future.done():
                candidates = detections
                if target_class is not None:
                    candidates = [
                        d for d in detections if d[3] == target_class
                    ]
                if candidates:
                    (u, v, w, h, r), score, class_id, class_name = max(
                        candidates, key=lambda d: d[1]
                    )
                    angle_deg = np.rad2deg(r)
                    # 将图像坐标转换为机械臂坐标系
                    target_x, target_y = arm.pixel2pos(u, v)
                    gripper_angle_rad = arm.gripper_angle_by_longer(
                        u, v, w, h, angle_deg
                    )
                    class_place_pos = copy.deepcopy(place_pos.get(class_name, None))
                    if class_place_pos is None or "pos" not in class_place_pos:
                        print("No placement location specified, place in origin.")
                        class_place_pos = {"pos": [target_x, target_y]}
                    else:
                        match class_place_pos["pos"][0]:
                            case "x":
                                class_place_pos["pos"][0] = target_x
                            case "-x":
                                class_place_pos["pos"][0] = -target_x
                            case "y":
                                class_place_pos["pos"][0] = target_y
                            case "-y":
                                class_place_pos["pos"][0] = -target_y
                        match class_place_pos["pos"][1]:
                            case "x":
                                class_place_pos["pos"][1] = target_x
                            case "-x":
                                class_place_pos["pos"][1] = -target_x
                            case "y":
                                class_place_pos["pos"][1] = target_y
                            case "-y":
                                class_place_pos["pos"][1] = -target_y
                    if (
                        np.linalg.norm(
                            np.array(class_place_pos.get("pos"))
                            - np.array([target_x, target_y])
                        )
                        < place_distance_threshold
                    ):
                        print(
                            f"Object {class_name} is too close to place position, skipping catch."
                        )
                    else:
                        future = executor.submit(
                            arm.catch_and_place,
                            # 夹爪向外偏移一些，避免刚好顶到物体
                            target_x + offset * np.cos(gripper_angle_rad),
                            target_y + offset * np.sin(-gripper_angle_rad),
                            gripper_angle_rad,
                            class_place_pos["pos"],
                        )

            if default_gripper_aside_pos and (future is None or future.done()):
                # 移到旁边以免挡住视野
                future = executor.submit(
                    arm.move_to,
                    default_gripper_aside_pos,
                    1,
                )
            end_time = time.time()
            if end_time - start_time == 0:
                fps = 0.0
            else:
                fps = 1 / (end_time - start_time)
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            show_image("Detections", frame)
            if poll_key(1) & 0xFF == 27:  # 按Esc键退出
                break
        except KeyboardInterrupt:
            print("Exiting...")
            break

    arm.move_to_home(gripper_open_0to1=1)
    time.sleep(1)
    arm.disconnect_arm()
    cam.close()
    destroy_all_windows()


if __name__ == "__main__":
    args = parse_args()
    main(target_class=args.target)
