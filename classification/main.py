"""Inference client for the Piper single-arm robot (roboarm SDK).

Connects to the openpi policy server (scripts/serve_policy.py running pi05_piper),
sends camera images + the 7-dim arm state, receives a 50-step action chunk, and
executes it on the Piper arm via the roboarm `PiperBySDK` wrapper.

Data format is matched EXACTLY to the training recorder
(roboarm/classification/catch_with_arm_record_piper.py):

  observation/state  = [joint_1..joint_6 (deg), gripper_0to1 * 100]   (7,)
  action             = same 7-dim space (absolute joint degrees + gripper*100)
  observation/image       = "above" camera (Orbbec), RGB HxWx3
  observation/wrist_image = "wrist" camera (Intel RealSense), RGB HxWx3

IMPORTANT: this client must run from the roboarm project (it imports the Piper
SDK + cameras from there). Point PIPER_REPO at it. Run from the roboarm venv,
because the SDK (piper_sdk, kinpy, etc.) lives there:

    cd /home/czn/roboarm
    /home/czn/roboarm/.venv/bin/python /home/czn/openpi/examples/piper/main.py --dry_run

The openpi_client dependency is light; install it into the roboarm venv with:
    uv pip install -e /home/czn/openpi/packages/openpi-client
"""

import dataclasses
import logging
import os
import sys
import time

import cv2
import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

logger = logging.getLogger(__name__)

# Gripper scale used by the training recorder: state/action store gripper as
# (gripper_open_0to1 * 100). The SDK's set/get use the 0..1 range.
MAX_GRIPPER_ANGLE_DEG = 100.0


@dataclasses.dataclass
class Args:
    # Policy server connection.
    host: str = "10.0.105.11"
    host: str = "127.0.0.1"
    port: int = 8002
    api_key: str | None = None

    # Task instruction. Must match a prompt style seen in training, e.g.
    # "pick the potato toy and place into box" (potato/carrot/tomato).
    prompt: str = "pick the potato toy and place into box"

    # roboarm project root (provides the Piper SDK + cameras).
    repo: str = "/home/czn/roboarm"

    # Receding horizon: the policy predicts a longer chunk each inference, but we
    # only execute the first `actions_per_chunk` waypoints, then re-observe and
    # re-query. Smaller = tighter closed loop (more network round-trips); larger =
    # more open-loop. 8 keeps the loop tight.
    actions_per_chunk: int = 50
    # Seconds between consecutive waypoints streamed to the arm (≈ 1/control_hz).
    # Training was recorded at 30 FPS, so 0.033 matches the trajectory's intended
    # playback rate. Increase (slower) for a cautious first run.
    control_dt: float =  0.033
    # Total chunks before stopping. None = run until Ctrl-C.
    max_chunks: int | None = None

    # Safety: if True, log actions but do NOT command the arm.
    dry_run: bool = False
    # Initial homing / control-mode speed percent (1-100). Does NOT throttle the
    # streamed waypoints (those go out at control_dt); only affects the startup
    # move_to_home and the SDK's motion-control setup.
    move_speed: int = 100


class PiperRuntime:
    """Wraps the roboarm Piper SDK + the 'above'/'wrist' cameras."""

    def __init__(self, repo: str, move_speed: int) -> None:
        if repo not in sys.path:
            sys.path.insert(0, repo)
        # These imports come from the roboarm project, not openpi.
        from arm.piper_ctrl_by_sdk import PiperBySDK
        from camera.camera_api import Camera as OrbbecCamera
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        # The "above" camera is an Orbbec Gemini 336L. Its proprietary protocol
        # cannot be opened by cv2.VideoCapture / V4L2 — lerobot's OpenCVCamera
        # explicitly skips Orbbec devices. The training recorder read it through
        # roboarm's Orbbec SDK wrapper (camera.camera_api.Camera -> pyorbbecsdk),
        # so we do the same here to keep the observation identical.
        self._above_cam = OrbbecCamera(color=True, depth=False)

        # The "wrist" camera is an Intel RealSense — a normal V4L2 device that
        # OpenCV opens fine. Resolve the same stable by-id path the recorder used.
        wrist_path = os.environ.get("LEROBOT_WRIST_CAMERA")
        if not wrist_path:
            import glob

            matches = sorted(glob.glob("/dev/v4l/by-id/*Intel*-video-index0"))
            if not matches:
                raise FileNotFoundError("No camera matching 'Intel' under /dev/v4l/by-id")
            wrist_path = matches[0]
        self._wrist_cam = OpenCVCamera(
            OpenCVCameraConfig(index_or_path=wrist_path, width=640, height=480, fps=30)
        )
        self._wrist_cam.connect()

        self._arm = PiperBySDK(move_speed=move_speed)
        self._arm.move_to_home(gripper_open_0to1=0.8)

    def read_state(self) -> np.ndarray:
        """7-dim state = [6 joint deg, gripper_0to1 * 100], matching training."""
        angles, gripper = self._arm.get_arm_angles()
        if angles is None or gripper is None:
            raise RuntimeError("Failed to read arm state")
        return np.array([*angles, gripper * MAX_GRIPPER_ANGLE_DEG], dtype=np.float32)

    def read_cameras(self) -> tuple[np.ndarray, np.ndarray]:
        """(above, wrist) RGB uint8 frames at 640x480.

        Processing MUST match the recorder exactly (see
        catch_with_arm_record_piper.py::_capture_detection_camera):
          - above (Orbbec): SDK returns BGR at native res -> resize to 640x480 -> BGR2RGB
          - wrist (RealSense): OpenCVCamera already returns RGB at 640x480
        """
        color = self._above_cam.get_frames().get("color")
        if color is None:
            raise RuntimeError("Failed to read Orbbec 'above' camera color frame")
        if color.shape[1] != 640 or color.shape[0] != 480:
            color = cv2.resize(color, (640, 480))
        above = np.asarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB), dtype=np.uint8)

        wrist = np.asarray(self._wrist_cam.async_read(), dtype=np.uint8)
        return above, wrist

    def send_action(self, action: np.ndarray) -> None:
        """Stream one waypoint to the arm WITHOUT blocking/interpolation.

        The VLA outputs a dense 50-step trajectory meant to be streamed at the
        training control rate (~30 Hz). The SDK's high-level set_arm_angles
        interpolates and blocks until the target is reached (seconds per call),
        which destroys the closed-loop cadence. So we send the raw joint/gripper
        targets directly via the low-level CAN interface and return immediately;
        trajectory smoothness already lives in the model output.

        action = [joint_1..6 (deg), gripper_0to1 * 100].
        Scaling matches PiperBySDK: joints * FACTOR(1000); gripper angle =
        open_0to1 * MAX_GRIPPER_ANGLE_DEG(100) * FACTOR(1000).
        """
        piper = self._arm.piper
        factor = self._arm.FACTOR  # 1000.0
        ctrl = np.asarray(action[:6], dtype=np.float64) * factor
        piper.JointCtrl(
            joint_1=int(ctrl[0]),
            joint_2=int(ctrl[1]),
            joint_3=int(ctrl[2]),
            joint_4=int(ctrl[3]),
            joint_5=int(ctrl[4]),
            joint_6=int(ctrl[5]),
        )
        gripper_0to1 = float(np.clip(action[6] / MAX_GRIPPER_ANGLE_DEG, 0.0, 1.0))
        piper.GripperCtrl(
            int(gripper_0to1 * self._arm.MAX_GRIPPER_ANGLE_DEG * factor),
            gripper_effort=5000,
            gripper_code=0x03,
            set_zero=0,
        )

    def close(self) -> None:
        try:
            self._arm.disconnect_arm()
        finally:
            try:
                self._above_cam.close()
            finally:
                if getattr(self._wrist_cam, "is_connected", False):
                    self._wrist_cam.disconnect()


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host, port=args.port, api_key=args.api_key
    )
    logger.info("Connected to policy server. Metadata: %s", policy.get_server_metadata())

    robot = PiperRuntime(args.repo, args.move_speed)
    try:
        chunk_idx = 0
        while args.max_chunks is None or chunk_idx < args.max_chunks:
            above, wrist = robot.read_cameras()
            state = robot.read_state()

            obs = {
                "observation/image": above,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": args.prompt,
            }

            t0 = time.time()
            actions = np.asarray(policy.infer(obs)["actions"])  # (50, 7)
            logger.info("chunk %d: %s actions in %.0f ms", chunk_idx, actions.shape, 1000 * (time.time() - t0))

            for i in range(min(args.actions_per_chunk, actions.shape[0])):
                action = actions[i]
                if args.dry_run:
                    logger.info("  [dry_run] action[%d]=%s", i, np.round(action, 3))
                else:
                    robot.send_action(action)
                time.sleep(args.control_dt)

            chunk_idx += 1
    finally:
        robot.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
