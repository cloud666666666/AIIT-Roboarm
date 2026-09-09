import importlib
import os
import time
from collections.abc import Sequence
from math import inf
from typing import cast

import cv2
import numpy as np
from utils.config_getter import get_config_value
from scipy.spatial.transform import Rotation as R
from typing_extensions import Self
from collections.abc import Callable
from typing import Any


StepCallback = Callable[[dict[str, Any]], None]


class Arm:
    """机械臂控制基类。

    `Arm()` 会根据配置项 `arm_type` 实际返回对应的机械臂子类实例，
    公共的抓取、放置和手眼标定坐标转换逻辑统一放在这里复用。
    """

    IK_POSITION_TOLERANCE_M = 0.01
    IK_TILT_TOLERANCE_RAD = np.deg2rad(5.0)
    IK_YAW_TOLERANCE_RAD = np.deg2rad(10.0)

    # arm_type -> (module_path, class_name)
    _ARM_TYPES = {
        "piper": ("arm.piper_ctrl_by_sdk", "PiperBySDK"),
        "lerobo": ("arm.lerobo_arm_control", "LeroboArm"),
        "jaka": ("arm.jaka_ctrl_by_sdk", "JakaBySDK"),
    }

    def __new__(cls, *args, **kwargs) -> Self:
        """根据配置返回具体机械臂实现。

        Returns:
            配置对应的机械臂实例；直接实例化子类时返回该子类实例。

        Raises:
            ValueError: 配置中的 `arm_type` 未注册时抛出。
        """
        if cls is Arm:
            arm_type = get_config_value("arm_type")
            entry = cls._ARM_TYPES.get(arm_type)
            if not entry:
                raise ValueError(
                    f"Unknown arm type in config: {arm_type}, "
                    f"available: {list(cls._ARM_TYPES)}"
                )
            module_path, class_name = entry
            module = importlib.import_module(module_path)
            target_class = getattr(module, class_name)
            return cast(Self, super().__new__(target_class))
        return super().__new__(cls)

    def __init__(
        self,
        hand_eye_calibration_file=os.path.join(
            os.path.dirname(__file__), "hand-eye-data/2d_homography.npy"
        ),
    ):
        """初始化基类公共状态。

        Args:
            hand_eye_calibration_file: 2D 手眼标定矩阵文件路径。
        """
        self.timeout_s = 3
        self.desktop_height = get_config_value("default_desktop_height")
        self.catch_raise_height = get_config_value("catch_raise_height", 0.1)
        self.place_raise_height = get_config_value("place_raise_height", 0.1)
        self.default_gripper_close_threshold = get_config_value(
            "default_gripper_close_threshold"
        )
        self.catch_time_interval_s = get_config_value("catch_time_interval_s")
        self.get_arm_angles_retry_times = get_config_value("get_arm_angles_retry_times")

        self.reach_mse_threshold = float(
            get_config_value("arm_reach_mse_threshold_deg2")
        )
        if os.path.exists(hand_eye_calibration_file):
            self.hand_eye_calibration_matrix = np.load(hand_eye_calibration_file)

        if not 0 <= self.default_gripper_close_threshold <= 1:
            print(
                "Warning: The definition of gripper state limit is [0, 1], not angle degrees."
            )

    def set_arm_angles(
        self,
        angles_deg: Sequence[float | int] | None = None,
        gripper_open_0to1: float | None = None,
        step_callback: StepCallback | None = None,
    ) -> bool:
        """设置关节角度和夹爪开合程度。

        Args:
            angles_deg: 各关节角度，单位为度；`None` 表示保持当前关节角度。
            gripper_open_0to1: 夹爪开合程度，范围为 `[0, 1]`，越大越开；
                `None` 表示保持当前夹爪状态。

        Returns:
            设置是否成功。
        """
        raise NotImplementedError(
            "set_arm_angles method must be implemented in subclass"
        )

    def get_arm_angles(
        self, retry_times=None
    ) -> tuple[list[float] | None, float | None]:
        """读取当前关节角度和夹爪开合程度。

        Args:
            retry_times: 读取失败后的重试次数；`None` 表示由子类决定默认值。

        Returns:
            一个二元组 `(angles_deg, gripper_open_0to1)`：
            - `angles_deg` 为关节角度列表，单位为度；失败时为 `None`
            - `gripper_open_0to1` 为夹爪开合程度，范围为 `[0, 1]`；失败时为 `None`
        """
        raise NotImplementedError(
            "get_arm_angles method must be implemented in subclass"
        )

    def get_raw_joint_angles(
        self, retry_times=None
    ) -> tuple[list[float] | None, float | None]:
        """读取未应用 offset 修正的原始关节角度和夹爪开合程度。

        主要用于 offset 标定等需要直接获取底层关节读数的低层场景；
        正常业务请使用 `get_arm_angles`。

        Args:
            retry_times: 读取失败后的重试次数；`None` 表示由子类决定默认值。

        Returns:
            一个二元组 `(angles_deg, gripper_open_0to1)`：
            - `angles_deg` 为未修正的原始关节角度列表，单位为度；失败时为 `None`
            - `gripper_open_0to1` 为夹爪开合程度，范围为 `[0, 1]`；失败时为 `None`
        """
        raise NotImplementedError(
            "get_raw_joint_angles method must be implemented in subclass"
        )

    def get_arm_pose(self) -> tuple[list[float] | None, list[float] | None]:
        """读取当前末端位姿。

        Returns:
            一个二元组 `(end_pos, end_rot)`
            - `end_pos` 为末端位置 `[x, y, z]`，单位为米；读取失败时返回 `None`。
            - `end_rot` 为末端姿态 `[RZ, RY, RX]`，单位为角度；读取失败时返回 `None`。
        """
        raise NotImplementedError("get_arm_pose method must be implemented in subclass")

    def move_to_home(
        self,
        gripper_open_0to1: float | int | None = None,
        step_callback: StepCallback | None = None,
    ) -> bool:
        """将机械臂移动到归零位。

        Args:
            gripper_open_0to1: 可选的夹爪开合程度，范围为 `[0, 1]`。

        Returns:
            移动是否成功。
        """
        raise NotImplementedError("move_to_home method must be implemented in subclass")

    def move_to(
        self,
        pos: list[float],
        gripper_open_0to1: float | int | None = None,
        rot_rad: float | int | None = None,
        euler_angles_deg_zyx: list[float] | None = None,
        step_callback: StepCallback | None = None,
    ) -> bool:
        """将末端移动到目标位置。

        Args:
            pos: 目标位置 `[x, y, z]`，单位为米。
            gripper_open_0to1: 夹爪开合程度，范围为 `[0, 1]`；`None` 表示不修改。
            rot_rad: 末端绕 z 轴的目标旋转角，单位为弧度；仅在未显式传入
                `euler_angles_deg_zyx` 时生效。
            euler_angles_deg_zyx: 目标末端欧拉角 `[RZ, RY, RX]`，单位为度。

        Returns:
            移动是否成功。
        """
        raise NotImplementedError("move_to method must be implemented in subclass")

    def prepare_for_manual_teach(self) -> bool:
        """手动示教（如手眼标定拖动采点）前，摆到便于拖动的起始姿态。

        默认回到 home；子类可覆写为更适合拖动的姿态（如夹爪朝下靠近工作区）。

        Returns:
            移动是否成功。
        """
        return self.move_to_home()

    def wait_until_reached(self, target_angles_deg: Sequence[float]) -> None:
        """轮询 /state 直到当前关节角与目标关节角的均方差小于阈值，或超时。

        Args:
            target_angles_deg: 期望最终到达的关节角列表，单位为度。

        Raises:
            TimeoutError: 在 `timeout_s` 内未达到阈值。
            RuntimeError: 仿真服务返回的关节数与目标关节数不一致。
        """
        target = [float(angle) for angle in target_angles_deg]
        deadline = time.monotonic() + self.timeout_s
        while True:
            current_angles_deg, _ = self.get_raw_joint_angles()
            if current_angles_deg is None:
                continue
            if len(current_angles_deg) != len(target):
                raise RuntimeError(
                    "仿真服务返回的关节数与目标关节数不一致: "
                    f"{len(current_angles_deg)} vs {len(target)}"
                )
            squared_errors = [
                (current - desired) ** 2
                for current, desired in zip(current_angles_deg, target, strict=True)
            ]
            mse = sum(squared_errors) / len(squared_errors)
            if mse <= self.reach_mse_threshold:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"等待仿真机械臂到位超时: 当前 MSE={mse:.4f} deg^2, "
                    f"阈值={self.reach_mse_threshold} deg^2, "
                    f"超时={self.timeout_s}s"
                )
            time.sleep(0.1)

    def set_gripper(
        self,
        gripper_open_0to1: float,
        step_callback: StepCallback | None = None,
    ):
        """设置夹爪开合程度。

        Args:
            gripper_open_0to1: 夹爪开合程度，范围为 `[0, 1]`，越大越开。
        """
        raise NotImplementedError("set_gripper method must be implemented in subclass")

    def disconnect_arm(self):
        """断开机械臂连接并完成必要清理。"""
        raise NotImplementedError("disconnect_arm method must be implemented in subclass")

    def enable_torque(self):
        """使能机械臂和夹爪。"""
        raise NotImplementedError("enable_torque method must be implemented in subclass")

    def disable_torque(self):
        """关闭机械臂和夹爪使能。"""
        raise NotImplementedError("disable_torque method must be implemented in subclass")

    @staticmethod
    def _wrap_angle_rad(angle_rad: float) -> float:
        return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))

    @classmethod
    def _ik_cost_function(cls, joint_angles, target_pose_matrix, chain):
        current_pose_matrix = chain.forward_kinematics(joint_angles).matrix()

        pos_error = np.linalg.norm(
            current_pose_matrix[:3, 3] - target_pose_matrix[:3, 3]
        )

        current_rot = R.from_matrix(current_pose_matrix[:3, :3])
        target_rot = R.from_matrix(target_pose_matrix[:3, :3])
        current_z_axis = current_rot.apply([0.0, 0.0, 1.0])
        target_z_axis = target_rot.apply([0.0, 0.0, 1.0])
        cosine = np.clip(np.dot(current_z_axis, target_z_axis), -1.0, 1.0)
        tilt_error = float(np.arccos(cosine))

        current_yaw = current_rot.as_euler("zyx")[0]
        target_yaw = target_rot.as_euler("zyx")[0]
        yaw_error = abs(cls._wrap_angle_rad(current_yaw - target_yaw))

        pos_term = (pos_error / cls.IK_POSITION_TOLERANCE_M) ** 2
        tilt_term = (tilt_error / cls.IK_TILT_TOLERANCE_RAD) ** 2
        yaw_term = (yaw_error / cls.IK_YAW_TOLERANCE_RAD) ** 2
        return float(pos_term + tilt_term + yaw_term)

    def wait_gripper_gripped(self) -> None:
        """等待夹爪夹取动作完成。默认固定等待 `catch_time_interval_s`，
        子类可覆写为轮询夹爪实际状态（如位置稳定）。"""
        time.sleep(self.catch_time_interval_s)

    def catch(
        self,
        target_x: float,
        target_y: float,
        rot_rad: float,
        height: float = inf,
        step_callback: StepCallback | None = None,
    ) -> bool:
        """执行抓取动作。

        Args:
            target_x: 抓取点 x 坐标，单位为米。
            target_y: 抓取点 y 坐标，单位为米。
            rot_rad: 抓取时末端绕 z 轴的旋转角，单位为弧度。
            height: 抓取高度，单位为米；传入 `inf` 时使用桌面高度。

        Returns:
            抓取是否成功。
        """
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
        time.sleep(self.catch_time_interval_s * 2)

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
        time.sleep(self.catch_time_interval_s)

        self.set_gripper(gripper_open_0to1=0, step_callback=step_callback)
        self.wait_gripper_gripped()

        res = self.move_to(
            [target_x, target_y, target_z + self.catch_raise_height],
            rot_rad=rot_rad,
            step_callback=step_callback,
        )
        if not res:
            print("抬起失败，取消抓取")
            self.move_to_home(gripper_open_0to1=1)
            return False
        time.sleep(self.catch_time_interval_s)

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
        step_callback: StepCallback | None = None,
    ) -> bool:
        """执行放置动作。

        Args:
            target_x: 放置点 x 坐标，单位为米。
            target_y: 放置点 y 坐标，单位为米。
            target_z: 放置点 z 坐标，单位为米。
            rot_rad: 放置时末端绕 z 轴的旋转角，单位为弧度。
            down: 是否下降后再放置，默认为False。

        Returns:
            放置是否成功。
        """
        res = self.move_to(
            [target_x, target_y, target_z + self.place_raise_height],
            rot_rad=rot_rad,
            step_callback=step_callback,
        )
        if not res:
            print("移动到放置位置上方失败，取消放置")
            self.move_to_home(gripper_open_0to1=1)
            return False
        time.sleep(self.catch_time_interval_s * 2)

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
            time.sleep(self.catch_time_interval_s)

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
            time.sleep(self.catch_time_interval_s)

        return True

    def catch_and_place(
        self,
        target_x: float,
        target_y: float,
        catch_rotate_rad: float,
        place_pos: list[float | int],
        height: float = inf,
        place_rotate_rad: float = 0,
        step_callback: StepCallback | None = None,
    ) -> bool:
        """执行抓取后放置的完整流程。

        Args:
            target_x: 抓取点 x 坐标，单位为米。
            target_y: 抓取点 y 坐标，单位为米。
            catch_rotate_rad: 抓取时末端绕 z 轴的旋转角，单位为弧度。
            place_pos: 放置位置，格式为 `[x, y]` 或 `[x, y, z]`，单位为米。
            height: 抓取高度，单位为米；传入 `inf` 时使用桌面高度。
            place_rotate_rad: 放置时末端绕 z 轴的旋转角，单位为弧度。

        Returns:
            整个流程是否成功。
        """
        if len(place_pos) == 2:
            place_x, place_y = place_pos
            place_z = self.desktop_height
        elif len(place_pos) == 3:
            place_x, place_y, place_z = place_pos
        else:
            print("放置位置格式错误，应该是[x, y]或[x, y, z]")
            return False
        if not self.catch(
            target_x,
            target_y,
            catch_rotate_rad,
            height=height,
            step_callback=step_callback,
        ):
            self.move_to_home(gripper_open_0to1=1)
            return False
        # self.move_to_home()
        if not self.place(
            place_x,
            place_y,
            place_z,
            place_rotate_rad,
            step_callback=step_callback,
        ):
            self.move_to_home(gripper_open_0to1=1)
            return False
        self.move_to_home(gripper_open_0to1=1, step_callback=step_callback)
        return True

    def pixel2pos(self, u: float, v: float) -> tuple[float, float]:
        """将图像像素坐标转换为机械臂平面坐标。

        Args:
            u: 图像横向像素坐标。
            v: 图像纵向像素坐标。

        Returns:
            机械臂坐标系下的 `(x, y)`，单位为米。

        Raises:
            ValueError: 未加载手眼标定矩阵时抛出。
        """
        if not hasattr(self, "hand_eye_calibration_matrix"):
            raise ValueError("没有手眼标定数据，无法转换图像坐标")
        pixel_coords = np.array([[u], [v], [1]])
        world_coords = self.hand_eye_calibration_matrix @ pixel_coords
        world_coords /= world_coords[2]
        target_x, target_y = world_coords[0, 0], world_coords[1, 0]
        return target_x, target_y

    @staticmethod
    def gripper_angle_by_longer(
        u: float, v: float, w: float, h: float, angle_deg: float
    ) -> float:
        """根据检测框长边估计夹爪绕 z 轴的旋转角。

        Args:
            u: 检测框中心 x 坐标（像素）。
            v: 检测框中心 y 坐标（像素）。
            w: 检测框宽度（像素）。
            h: 检测框高度（像素）。
            angle_deg: 检测框角度，单位为度，遵循 OpenCV `boxPoints` 的定义。

        Returns:
            适合夹取该目标的末端旋转角，单位为弧度，范围约为 `[-pi/2, pi/2]`。
        """
        box_points = cv2.boxPoints(((u, v), (w, h), angle_deg))
        if np.linalg.norm(box_points[0] - box_points[1]) > np.linalg.norm(
            box_points[1] - box_points[2]
        ):
            long_edge_points = (
                [box_points[0], box_points[1]]
                if box_points[0][0] < box_points[1][0]
                else [box_points[1], box_points[0]]
            )
        else:
            long_edge_points = (
                [box_points[1], box_points[2]]
                if box_points[1][0] < box_points[2][0]
                else [box_points[2], box_points[1]]
            )
        gripper_rot_rad = np.pi / 2 + np.arctan2(
            long_edge_points[1][1] - long_edge_points[0][1],
            long_edge_points[1][0] - long_edge_points[0][0],
        )
        if gripper_rot_rad > np.pi / 2:
            gripper_rot_rad -= np.pi
        return gripper_rot_rad
