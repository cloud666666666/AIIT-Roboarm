"""
Auto reset — YOLO 检测物体 + 随机摆放，用于 VLA 数据采集后的环境复原。

在 catch_with_arm_record_piper.py 执行后运行：
检测桌面上的物体，逐个抓取并放置到随机位置，完成后机械臂回原位。
只执行动作，不收集数据（无 LeRobot 录制）。
"""

import os
import random
import sys
from math import inf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import yaml

from arm.piper_ctrl_by_sdk import PiperBySDK
from camera.camera_api import Camera
from object_detect.detect import detect_objects_in_frame, draw_box, load_model
from utils.cv2_display import show_image, show_img_by_web, destroy_all_windows

HAS_DISPLAY = os.environ.get("DISPLAY") is not None or show_img_by_web()

TARGET_CLASS = "potato"  # 要重置的目标类别，空字符串表示所有类别


def _pos_in_workspace(x: float, y: float, wx_range, wy_range) -> bool:
    return (wx_range[0] <= x <= wx_range[1]) and (wy_range[0] <= y <= wy_range[1])


def main():
    config_path = PROJECT_ROOT / "config.yaml"
    config_yaml = yaml.safe_load(open(config_path, encoding="utf-8"))

    model_paths = [
        str(PROJECT_ROOT / path)
        for path in config_yaml.get("classification_YOLO_model_path", [])
    ]
    default_conf_thres = config_yaml.get("default_conf_thres", 0.8)

    # — auto-reset config —
    class_pos = config_yaml.get("class_pos", {})
    wx_range = tuple(config_yaml.get("workspace_x_range", (-0.1, 0.55)))
    wy_range = tuple(config_yaml.get("workspace_y_range", (-0.3, 0.55)))
    min_place_dist = config_yaml.get("reset_min_place_dist_m", 0.20)

    arm = PiperBySDK()
    arm.timeout = 15
    arm.move_to_home(gripper_open_0to1=0.8)

    camera = Camera(color=True, depth=False)
    models = [load_model(path) for path in model_paths]

    print("Auto reset — YOLO detection + random placement")
    print(f"  Target class: {TARGET_CLASS or 'all'}")

    try:
        # 获取相机帧
        camera_frame = camera.get_frames().get("color")
        if camera_frame is None:
            print("无法获取相机帧")
            return

        # YOLO 检测
        detections = []
        for model in models:
            detections.extend(
                detect_objects_in_frame(model, camera_frame, conf_thres=default_conf_thres)
            )

        # 筛选目标类别
        targets = []
        for (u, v, w, h, r), _score, _class_id, class_name in detections:
            if TARGET_CLASS and class_name != TARGET_CLASS:
                continue

            target_x, target_y = arm.pixel2pos(u, v)
            if not _pos_in_workspace(target_x, target_y, wx_range, wy_range):
                print(f"  跳过 {class_name} ({target_x:.3f},{target_y:.3f}) — 超出 workspace")
                continue
            angle_deg = np.rad2deg(r)
            gripper_angle_rad = arm.gripper_angle_by_longer(u, v, w, h, angle_deg)
            targets.append((target_x, target_y, gripper_angle_rad, class_name, u, v))

        if not targets:
            print("未检测到目标物体")
            return

        # 所有目标位置 → 碰撞避免
        all_target_positions = [(t[0], t[1]) for t in targets]

        print(f"检测到 {len(targets)} 个目标物体")

        for tx, ty, grad, class_name, u, v in targets:
            # 读取该类别的随机范围
            place_cfg = class_pos.get(class_name) if isinstance(class_pos.get(class_name), dict) else class_pos.get("default")
            if isinstance(place_cfg, dict) and isinstance(place_cfg.get("random_pos"), list) and len(place_cfg["random_pos"]) == 2:
                rx_min, rx_max = place_cfg["random_pos"][0]
                ry_min, ry_max = place_cfg["random_pos"][1]
            else:
                rx_min, rx_max = 0.0, 0.5
                ry_min, ry_max = -0.2, 0.3

            # 随机放置位置 — 避开所有桌面物体
            place_x, place_y = tx, ty
            for _ in range(50):
                place_x = random.uniform(rx_min, rx_max)
                place_y = random.uniform(ry_min, ry_max)
                if not _pos_in_workspace(place_x, place_y, wx_range, wy_range):
                    continue
                if any(
                    np.hypot(place_x - ox, place_y - oy) < min_place_dist
                    for ox, oy in all_target_positions
                ):
                    continue
                break
            else:
                # Fallback: 四个方向尝试
                for angle in [0.0, np.pi / 2, np.pi, -np.pi / 2]:
                    place_x = tx + min_place_dist * np.cos(angle)
                    place_y = ty + min_place_dist * np.sin(angle)
                    if not _pos_in_workspace(place_x, place_y, wx_range, wy_range):
                        continue
                    if any(
                        np.hypot(place_x - ox, place_y - oy) < min_place_dist
                        for ox, oy in all_target_positions
                    ):
                        continue
                    break
                else:
                    place_x = tx + min_place_dist
                    place_y = ty

            print(f"  {class_name}: pos=({tx:.3f},{ty:.3f}) -> random place=({place_x:.3f},{place_y:.3f})")
            arm.catch_and_place(tx, ty, grad, [place_x, place_y])

        # 显示检测结果
        for (u, v, w, h, r), score, _class_id, class_name in detections:
            draw_box(camera_frame, u, v, w, h, np.rad2deg(r), f"{class_name}: {score:.2f}")
        if HAS_DISPLAY:
            cv2.putText(
                camera_frame,
                f"RESET | Targets: {len(targets)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            show_image("Reset", camera_frame)

        print("Reset done.")

    finally:
        arm.disconnect_arm()
        camera.close()
        if HAS_DISPLAY:
            destroy_all_windows()


if __name__ == "__main__":
    main()
