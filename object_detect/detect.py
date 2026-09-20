from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.config_getter import get_config_value
from camera.camera_api import Camera
from utils.cv2_display import show_image, poll_key, destroy_all_windows

def detect_objects_in_frame(model, frame, conf_thres=0.8, iou_thres=0.45):
    # verbose=False 关闭 ultralytics 每帧打印的检测摘要与耗时，避免刷屏
    results = model(frame, conf=conf_thres, iou=iou_thres, verbose=False)[0]
    detections = results.obb.xywhr.cpu().numpy()  # xywhr format
    scores = results.obb.conf.cpu().numpy()
    class_ids = results.obb.cls.cpu().numpy().astype(int)
    class_names = [model.names[i] for i in class_ids]
    return [
        ((u, v, w, h, r), score, class_id, class_name)
        for (u, v, w, h, r), score, class_id, class_name in zip(
            detections, scores, class_ids, class_names
        )
    ]


def draw_box(frame, u, v, w, h, angle_deg, label):
    box_points = cv2.boxPoints(((u, v), (w, h), angle_deg))
    box_points = np.int64(box_points)
    cv2.drawContours(frame, [box_points], 0, (0, 255, 0), 2)
    cv2.putText(
        frame,
        label,
        (int(u - w / 2), int(v - h / 2) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )


def load_model(model_path, device=""):
    model = YOLO(model_path)
    if device:
        model.to(device)
    return model


def main():
    model_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
        for path in get_config_value("classification_YOLO_model_path", [])
    ]
    default_conf_thres = get_config_value("default_conf_thres")

    # Load model
    model = load_model(model_paths[0])

    # Open video capture
    camera = Camera(color=True, depth=False)

    while True:
        frames = camera.get_frames()
        if frames.get("color") is None:
            print("Failed to grab frame")
            continue

        start_time = time.time()

        # Perform inference
        results = detect_objects_in_frame(
            model,
            frames["color"],
            conf_thres=default_conf_thres,
        )
        annotated_frame = frames["color"].copy()
        for (x, y, w, h, r), score, class_id, class_name in results:
            draw_box(
                annotated_frame,
                x,
                y,
                w,
                h,
                np.rad2deg(r),
                f"{class_name}: {score:.2f}",
            )
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        show_image("YOLOv11-obb Object Detection", annotated_frame)
        if poll_key(1) & 0xFF == 27:  # Press 'ESC' to exit
            break
    camera.close()
    destroy_all_windows()


if __name__ == "__main__":
    main()
