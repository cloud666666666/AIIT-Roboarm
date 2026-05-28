# Description: 奥比中光的深度相机控制
# 安装pyorbbecsdk库见https://orbbec.github.io/pyorbbecsdk/source/2_installation/install_the_package.html
# 此外还需注册元数据，见https://orbbec.github.io/pyorbbecsdk/source/2_installation/registration_script.html
# ******************************************************************************

from pyorbbecsdk import (
    Config,
    Pipeline,
    OBSensorType,
    FrameSet,
)
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import camera_utils
import cv2
import numpy as np
from utils.cv2_display import show_image, poll_key, destroy_all_windows

def open_camera(color: bool, depth: bool) -> Pipeline:
    if not color and not depth:
        raise ValueError("At least one of color or depth must be True")
    config = Config()
    pipeline = Pipeline()
    device = pipeline.get_device()
    video_sensors = []
    if color:
        video_sensors.append(OBSensorType.COLOR_SENSOR)
    if depth:
        video_sensors.append(OBSensorType.DEPTH_SENSOR)
    sensor_list = device.get_sensor_list()
    for sensor in range(len(sensor_list)):
        try:
            sensor_type = sensor_list[sensor].get_type()
            if sensor_type in video_sensors:
                config.enable_stream(sensor_type)
        except:
            continue
    pipeline.start(config)
    return pipeline


def close_camera(pipeline: Pipeline):
    if pipeline is not None:
        pipeline.stop()


def _process_color(frame):
    """Process color image"""
    return camera_utils.frame_to_bgr_image(frame) if frame else None


def _process_depth(frame):
    """Process depth image"""
    if not frame:
        return None
    try:
        depth_data = np.frombuffer(frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape(frame.get_height(), frame.get_width())
        depth_image = np.zeros_like(depth_data, dtype=np.uint8)
        cv2.normalize(depth_data, depth_image, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    except ValueError:
        return None


def get_frames(pipeline: Pipeline) -> dict:
    frames: FrameSet = pipeline.wait_for_frames(1000)
    if frames is None:
        return {"color": None, "depth": None}
    processed_frames = {
        "color": _process_color(frames.get_color_frame()),
        "depth": _process_depth(frames.get_depth_frame()),
    }
    return processed_frames


def main():
    pipeline = open_camera(True, True)
    while True:
        try:
            frames = get_frames(pipeline)
            color_image = frames.get("color")
            if color_image is None:
                print("failed to get color image")
            else:
                show_image("Color Viewer", color_image)
            depth_image = frames.get("depth")
            if depth_image is None:
                print("failed to get depth image")
            else:
                show_image("Depth Viewer", depth_image)
            key = poll_key(1)
            if key == ord("q") or key == 27:
                break
        except KeyboardInterrupt:
            break
    destroy_all_windows()
    pipeline.stop()


if __name__ == "__main__":
    main()
