import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import cv2
import numpy as np
from utils.config_getter import get_config_value
from utils.cv2_display import show_image, poll_key, destroy_all_windows

class Camera:

    def __init__(
        self,
        color: bool = True,
        depth: bool = False,
        undistort: bool = False,
        camera_matrix_path: str | None = None,
        dist_coeffs_path: str | None = None,
    ):
        self.ip = get_config_value("camera_ip", "", False)
        self.port = get_config_value("camera_port", None, False)
        self.color = color
        self.depth = depth
        self.undistort = undistort
        self.camera_matrix = None
        self.dist_coeffs = None
        if self.undistort:
            intrinsics_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "calibration",
                "intrinsics-data",
            )
            camera_matrix_path = camera_matrix_path or os.path.join(
                intrinsics_dir, "camera_matrix.npy"
            )
            dist_coeffs_path = dist_coeffs_path or os.path.join(
                intrinsics_dir, "dist_coeffs.npy"
            )
            if not os.path.exists(camera_matrix_path):
                raise FileNotFoundError(f"相机内参文件不存在: {camera_matrix_path}")
            if not os.path.exists(dist_coeffs_path):
                raise FileNotFoundError(f"相机畸变参数文件不存在: {dist_coeffs_path}")
            self.camera_matrix = np.load(camera_matrix_path)
            self.dist_coeffs = np.load(dist_coeffs_path)
        self.pipeline = None
        self.cap_rgb = None
        self.cap_depth = None
        if not self.ip:
            from camera import orb_camera

            self.orb_camera = orb_camera
            self.pipeline = self.orb_camera.open_camera(self.color, self.depth)
        else:
            if self.port is None:
                raise ValueError("未指定远程相机端口号")
            from camera import orb_camera_client

            self.orb_camera_client = orb_camera_client
            self.cap_rgb, self.cap_depth = self.orb_camera_client.open_orb_web_camera(
                self.ip, self.port, self.color, self.depth
            )

    def get_frames(self) -> dict[str, cv2.typing.MatLike | None]:
        if not self.ip:
            frames = self.orb_camera.get_frames(self.pipeline)
        else:
            frame_rgb = None
            frame_depth = None

            if self.cap_rgb is not None:
                _, frame_rgb = self.cap_rgb.read()
            if self.cap_depth is not None:
                _, frame_depth = self.cap_depth.read()
            frames = {"color": frame_rgb, "depth": frame_depth}

        frame_rgb = frames.get("color")
        if self.undistort and frame_rgb is not None:
            frames["color"] = cv2.undistort(
                frame_rgb, self.camera_matrix, self.dist_coeffs
            )
        return frames

    def close(self):
        if not self.ip:
            self.orb_camera.close_camera(self.pipeline)
        else:
            self.orb_camera_client.close_orb_web_camera(self.cap_rgb, self.cap_depth)


def main():

    camera = Camera(color=True, depth=False)

    frames = camera.get_frames()
    frame_rgb = frames.get("color")
    frame_depth = frames.get("depth")

    show_image("Camera Client", frame_rgb if frame_rgb is not None else frame_depth)

    print("Press 'q' to exit")
    while True:
        key = poll_key(1)
        if key == ord('q'):
            break

    destroy_all_windows()


if __name__ == "__main__":
    main()
