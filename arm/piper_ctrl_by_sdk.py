import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Union, List
from collections.abc import Sequence
from arm.arm_base import Arm, StepCallback
from utils.config_getter import get_config_value
import time
import numpy as np
import piper_sdk
from piper_sdk import C_PiperInterface_V2
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import kinpy
from pathlib import Path
import subprocess
import re
from xml.etree import ElementTree
import warnings


def deprecated(func):
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Function {func.__name__} is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper


class PiperBySDK(Arm):
    """Piper SDK 机械臂控制实现。"""

    FACTOR = 1000.0
    JOINT_COUNT = 6
    # 默认末端朝下的欧拉角 [RZ, RY, RX]（度）
    DEFAULT_DOWN_EULER_DEG_ZYX = [0.0, 180.0, 0.0]
    MAX_GRIPPER_ANGLE_DEG = 100

    def __init__(self, move_mode_end_pose: bool = False, debug_mode: bool = False):
        """初始化 Piper 机械臂控制器。

        Args:
            move_mode_end_pose: 是否默认使用末端位姿控制模式。
            debug_mode: 是否使用调试模式；调试模式下速度更保守、超时更长。
        """
        super().__init__()
        self.debug_mode = debug_mode
        self.move_mode_end_pose = move_mode_end_pose
        self.timeout = 10 if debug_mode else 5
        self.steps = 100

        # 加载 URDF 构建运动学链
        urdf_path = (
            Path(__file__).resolve().parent.parent
            / "urdf"
            / "piper"
            / "piper_description.urdf"
        )
        with open(urdf_path, "r", encoding="utf-8") as f:
            urdf_content = f.read()
        # 去掉 XML 声明，否则 ElementTree 解析 unicode 字符串会报错
        urdf_content = re.sub(r"<\?xml[^?]*\?>", "", urdf_content, count=1)
        self.chain = kinpy.build_serial_chain_from_urdf(urdf_content, "link6")

        # 从 URDF 解析关节限位
        urdf_xml = ElementTree.fromstring(urdf_content)
        self.joint_bounds = []
        for joint_elem in urdf_xml.findall("joint"):
            if joint_elem.get("type") == "revolute":
                limit = joint_elem.find("limit")
                if limit is not None:
                    lower = float(limit.get("lower", 0))
                    upper = float(limit.get("upper", 0))
                    self.joint_bounds.append((lower, upper))

        config_can_port = get_config_value("arm_port")
        try:
            self.piper = C_PiperInterface_V2(config_can_port)
            self.piper.ConnectPort()
        except:
            can_ports = self.activate_can()
            if config_can_port not in can_ports:
                print(
                    f"Warning: arm port {config_can_port} in config is not in scanned port({can_ports})"
                )
                if len(can_ports) == 1:
                    print(f"Using {can_ports[0]} instead")
                    config_can_port = can_ports[0]
            self.piper = C_PiperInterface_V2(config_can_port)
            self.piper.ConnectPort()
        self.piper.JointConfig(clear_err=0xAE)
        if not self._enable_fun():
            print(self.piper.GetArmStatus())
            raise RuntimeError("Failed to enable Piper arm.")
        self.set_move_mode(move_mode_end_pose=self.move_mode_end_pose)
        self.reset(self.move_mode_end_pose)

    # ========== 高层接口实现 ==========

    def get_raw_joint_angles(
        self, retry_times=None
    ) -> tuple[Union[List[float], None], Union[float, None]]:
        return self.get_arm_angles(retry_times=retry_times)

    def set_arm_angles(
        self,
        angles_deg: Sequence[float | int] | None = None,
        gripper_open_0to1: float | int | None = None,
        step_callback: StepCallback | None = None,
    ) -> bool:
        if gripper_open_0to1 is not None:
            if not 0 <= gripper_open_0to1 <= 1:
                raise ValueError("gripper_open_0to1 must in [0, 1]")
            desired_gripper_0to1 = float(gripper_open_0to1)
        else:
            desired_gripper_0to1 = None

        if angles_deg is not None and len(angles_deg) != self.JOINT_COUNT:
            print(f"关节角度数量错误，期望{self.JOINT_COUNT}个，实际{len(angles_deg)}个")
            return False

        if angles_deg is None and desired_gripper_0to1 is None:
            return True

        def send_gripper(open_0to1: float) -> None:
            self.piper.GripperCtrl(
                int(open_0to1 * self.MAX_GRIPPER_ANGLE_DEG * self.FACTOR),
                gripper_effort=2000 if self.debug_mode else 5000,
                gripper_code=0x03,
                set_zero=0,
            )

        def emit_step(
            joint_angles_deg: Sequence[float],
            gripper_0to1: float,
            final_joint_angles_deg: Sequence[float],
            final_gripper_0to1: float,
            step_index: int,
            steps: int,
            alpha: float,
        ) -> None:
            if step_callback is None:
                return
            joint_angles = [float(angle) for angle in joint_angles_deg]
            gripper_deg = float(gripper_0to1 * self.MAX_GRIPPER_ANGLE_DEG)
            sent_action = {
                f"joint_{index + 1}.pos": angle
                for index, angle in enumerate(joint_angles)
            }
            sent_action["gripper.pos"] = gripper_deg
            step_callback(
                {
                    "target_joint_angles_deg": joint_angles,
                    "target_gripper_open_0to1": float(gripper_0to1),
                    "final_target_joint_angles_deg": [
                        float(angle) for angle in final_joint_angles_deg
                    ],
                    "final_target_gripper_open_0to1": float(final_gripper_0to1),
                    "sent_action": sent_action,
                    "step_index": int(step_index),
                    "steps": int(steps),
                    "alpha": float(alpha),
                }
            )

        was_end_pose = False
        if angles_deg is not None:
            was_end_pose = self.move_mode_end_pose
            if was_end_pose:
                self.set_move_mode(move_mode_end_pose=False)
                self.move_mode_end_pose = False

        current_angles_deg, current_gripper_0to1 = self.get_arm_angles()
        if current_angles_deg is None or current_gripper_0to1 is None:
            return False

        desired_joint_angles = (
            list(current_angles_deg) if angles_deg is None else [float(angle) for angle in angles_deg]
        )
        final_gripper_0to1 = (
            float(current_gripper_0to1)
            if desired_gripper_0to1 is None
            else desired_gripper_0to1
        )

        start = time.time()
        for step_index, alpha in enumerate(np.linspace(0, 1, self.steps + 1)[1:]):
            interp_joint_angles = []
            for current_angle, desired_angle in zip(
                current_angles_deg, desired_joint_angles, strict=True
            ):
                interp_joint_angles.append(
                    current_angle * (1 - alpha) + desired_angle * alpha
                )
            interp_gripper_0to1 = (
                float(current_gripper_0to1) * (1 - alpha)
                + final_gripper_0to1 * alpha
            )

            if angles_deg is not None:
                ctrl = np.array(interp_joint_angles) * self.FACTOR
                self.piper.JointCtrl(
                    joint_1=int(ctrl[0]),
                    joint_2=int(ctrl[1]),
                    joint_3=int(ctrl[2]),
                    joint_4=int(ctrl[3]),
                    joint_5=int(ctrl[4]),
                    joint_6=int(ctrl[5]),
                )
            if desired_gripper_0to1 is not None:
                send_gripper(interp_gripper_0to1)

            time.sleep(0.01)
            emit_step(
                interp_joint_angles,
                interp_gripper_0to1,
                desired_joint_angles,
                final_gripper_0to1,
                step_index,
                self.steps,
                alpha,
            )

        if angles_deg is not None:
            while True:
                status = self.piper.GetArmStatus().arm_status
                if status.arm_status != 0x0:
                    print(self.arm_status2str(status.arm_status))
                    return False
                if status.motion_status == 0x00:
                    break
                if time.time() - start > self.timeout:
                    cur, _ = self.get_arm_angles()
                    if cur is not None:
                        mse = np.mean((np.array(cur) - np.array(desired_joint_angles)) ** 2)
                        if mse > self.reach_mse_threshold:
                            print(f"set_arm_angles 超时 (MSE={mse:.2f})")
                    break
                time.sleep(0.02)

            if was_end_pose:
                self.set_move_mode(move_mode_end_pose=True)
                self.move_mode_end_pose = True

            try:
                self.wait_until_reached(desired_joint_angles)
            except TimeoutError as e:
                print(f"仿真机械臂未在超时内到达目标位姿: {e}")
                return False

        return True

    def get_arm_angles(
        self, retry_times=None
    ) -> tuple[Union[List[float], None], Union[float, None]]:
        try:
            joints = self.piper.GetArmJointMsgs()
            gripper_msgs = self.piper.GetArmGripperMsgs()
        except Exception:
            if retry_times is None:
                retry_times = self.get_arm_angles_retry_times
            if retry_times > 0:
                time.sleep(self.catch_time_interval_s)
                return self.get_arm_angles(retry_times - 1)
            return None, None
        angles_deg = [
            joints.joint_state.joint_1 / self.FACTOR,
            joints.joint_state.joint_2 / self.FACTOR,
            joints.joint_state.joint_3 / self.FACTOR,
            joints.joint_state.joint_4 / self.FACTOR,
            joints.joint_state.joint_5 / self.FACTOR,
            joints.joint_state.joint_6 / self.FACTOR,
        ]
        gripper_0to1 = gripper_msgs.gripper_state.grippers_angle / self.FACTOR
        return angles_deg, float(
            np.clip(gripper_0to1 / self.MAX_GRIPPER_ANGLE_DEG, 0, 1).round(2)
        )

    # TODO: 后续可改用 self.chain.forward_kinematics() 本地 FK 替代 GetArmEndPoseMsgs
    def get_arm_pose(self) -> tuple[list[float] | None, list[float] | None]:
        try:
            return self.get_ee_pos().tolist(), self.get_ee_euler_zyx().tolist()
        except Exception:
            return None, None

    def move_to_home(
        self,
        gripper_open_0to1: float | None = None,
        step_callback: StepCallback | None = None,
        safe_pos: bool = False,
    ) -> bool:
        """safe_pos表示是否要移动到可安全失能的位置"""
        angles = [0, 0, 0, 0, 25 if safe_pos else 0, 0]
        return self.set_arm_angles(
            angles,
            gripper_open_0to1=gripper_open_0to1,
            step_callback=step_callback,
        )

    def move_to(
        self,
        pos: list[float],
        gripper_open_0to1: float | None = None,
        rot_rad: float | int | None = None,
        euler_angles_deg_zyx: list[float] | None = None,
        step_callback: StepCallback | None = None,
    ) -> bool:
        if len(pos) != 3:
            raise ValueError("位置参数格式错误，应该是[x, y, z]")

        # 构建目标位姿
        if euler_angles_deg_zyx is not None:
            rot = R.from_euler("zyx", euler_angles_deg_zyx, degrees=True).as_euler(
                "xyz"
            )
        else:
            target_euler = list(self.DEFAULT_DOWN_EULER_DEG_ZYX)
            if rot_rad is not None:
                target_euler[0] = np.degrees(rot_rad)
            rot = R.from_euler("zyx", target_euler, degrees=True).as_euler("xyz")
        goal_tf = kinpy.Transform(pos=np.array(pos), rot=rot)

        # 读取当前关节角度作为 IK 初始猜测
        current_angles_deg, _ = self.get_arm_angles()
        if current_angles_deg is not None:
            x0 = np.deg2rad(current_angles_deg)
        else:
            x0 = np.zeros(len(self.chain.get_joint_parameter_names()))

        result = minimize(
            self._ik_cost_function,
            x0=x0,
            args=(goal_tf.matrix(), self.chain),
            method="SLSQP",
            bounds=self.joint_bounds,
        )
        if not result.success:
            print("逆运动学不收敛，无法到达指定位置")
            return False
        angles_deg = np.rad2deg(result.x).tolist()
        return self.set_arm_angles(
            angles_deg,
            gripper_open_0to1=gripper_open_0to1,
            step_callback=step_callback,
        )

    def set_gripper(
        self,
        gripper_open_0to1: float,
        step_callback: StepCallback | None = None,
    ):
        self.set_arm_angles(
            gripper_open_0to1=gripper_open_0to1,
            step_callback=step_callback,
        )

    def disconnect_arm(self):
        print("Resetting piper arm to initial state.")
        try:
            self.reset()
        except TimeoutError as e:
            print(f"Error during reset: {e}")
        # 移动到可安全失能的位置
        print("Move piper arm to safe state.")
        self.move_to_home(safe_pos=True)
        time.sleep(0.5)
        self.disable_torque()
        self.piper.DisconnectPort()
        print("Arm disconnected")

    def enable_torque(self):
        self.piper.EnableArm(7)
        self.piper.GripperCtrl(
            gripper_angle=0,
            gripper_effort=2000 if self.debug_mode else 5000,
            gripper_code=0x03,
        )
        print("Piper arm enabled.")

    def disable_torque(self):
        self.piper.DisableArm()
        self.piper.GripperCtrl(gripper_code=0x02)
        print("Piper arm disabled.")

    # ========== 内部方法 ==========

    def reset(self, move_mode_end_pose: bool | None = None):
        """复位机械臂并根据需要恢复控制模式。

        Args:
            move_mode_end_pose: 复位后是否切换到末端位姿控制模式；`None`
                表示保留当前模式。
        """
        self.piper.JointConfig(clear_err=0xAE)
        self.piper.CrashProtectionConfig(0, 0, 0, 0, 0, 0)
        self.move_to_home(gripper_open_0to1=1)

        if move_mode_end_pose is not None:
            self.move_mode_end_pose = move_mode_end_pose
            if move_mode_end_pose:
                self.set_move_mode(move_mode_end_pose=True)

    def get_ee_pos(self) -> np.ndarray:
        """读取当前末端位置。

        Returns:
            当前末端位置 `[x, y, z]`，单位为米。
        """
        end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        return (
            np.array(
                [end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis],
                dtype=np.float32,
            )
            / self.FACTOR
            / 1000.0
        )

    def get_ee_euler_zyx(self) -> np.ndarray:
        """读取当前末端欧拉角。

        Returns:
            当前末端欧拉角 `[RZ, RY, RX]`，单位为度。
        """
        # GetArmEndPoseMsgs获取到的欧拉角顺序是xyz
        end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        return R.from_euler(
            "xyz",
            np.array(
                [end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis],
                dtype=np.float32,
            )
            / self.FACTOR,
            degrees=True,
        ).as_euler("zyx", degrees=True)

    def get_ee_quat(self) -> np.ndarray:
        """读取当前末端四元数姿态。

        Returns:
            当前末端姿态四元数 `[x, y, z, w]`。
        """
        end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        return R.from_euler(
            "xyz",
            np.array([end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis])
            / self.FACTOR,
            degrees=True,
        ).as_quat()

    @deprecated
    def set_ee_pose(
        self, position: list[float], euler_angles_deg_zyx: list[float]
    ) -> bool:
        """设置末端位姿，调用sdk的末端位姿控制模式。但机械臂内部逆运动学经常返回目标角度超过限，不好用。

        Args:
            position: 目标位置 `[x, y, z]`，单位为米。
            euler_angles_deg_zyx: 目标欧拉角 `[RZ, RY, RX]`，单位为度。

        Returns:
            设置是否成功。
        """
        if self.move_mode_end_pose is False:
            raise RuntimeError("Cannot set end-effector pose in joint control mode.")
        if len(position) != 3:
            raise ValueError(f"Length of param 'position({position})' must be 3")
        if len(euler_angles_deg_zyx) != 3:
            raise ValueError(
                f"Length of param 'euler_angles_deg_zyx({euler_angles_deg_zyx})' must be 3"
            )
        position_scaled = np.array(position) * self.FACTOR * 1000.0
        euler_scaled = (
            R.from_euler("zyx", np.array(euler_angles_deg_zyx), degrees=True).as_euler(
                "xyz", degrees=True
            )
            * self.FACTOR
        )
        self.piper.EndPoseCtrl(
            X=int(position_scaled[0]),
            Y=int(position_scaled[1]),
            Z=int(position_scaled[2]),
            RX=int(euler_scaled[0]),
            RY=int(euler_scaled[1]),
            RZ=int(euler_scaled[2]),
        )

        start = time.time()
        while True:
            status = self.piper.GetArmStatus().arm_status
            # 这里获取有问题，有时候没到目标status.motion_status就为0了
            if status.arm_status != 0x0:
                print(self.arm_status2str(status.arm_status))
                # self.piper.JointConfig(clear_err=0xAE)
                return False
            # pos = self.get_ee_pos()
            # if status.motion_status == 0x00 and np.linalg.norm(pos - position) < 0.02:
            #     return True
            if status.motion_status == 0x00:
                return True
            if time.time() - start > self.timeout:
                print("set_ee_pose 超时")
                return False

    def _enable_fun(self) -> bool:
        """循环尝试使能全部关节驱动。

        Returns:
            是否在超时时间内完成使能。
        """
        start_time = time.time()
        while True:
            self.piper.EnableArm(7)
            msgs = self.piper.GetArmLowSpdInfoMsgs()
            enable_flag = (
                msgs.motor_1.foc_status.driver_enable_status
                and msgs.motor_2.foc_status.driver_enable_status
                and msgs.motor_3.foc_status.driver_enable_status
                and msgs.motor_4.foc_status.driver_enable_status
                and msgs.motor_5.foc_status.driver_enable_status
                and msgs.motor_6.foc_status.driver_enable_status
            )
            print("使能状态:", enable_flag)
            if enable_flag:
                return True
            if time.time() - start_time > self.timeout:
                print("程序自动使能超时")
                return False
            time.sleep(1)

    def activate_can(self):
        """扫描并激活可用的机械臂 CAN 端口。

        Returns:
            成功激活的 CAN 接口名列表。
        """
        base_dir = Path(piper_sdk.__file__).parent
        find_script = base_dir / "find_all_can_port.sh"
        activate_script = base_dir / "can_activate.sh"
        baudrate = "1000000"

        # 1. 检查脚本文件是否存在
        if not find_script.exists() or not activate_script.exists():
            raise FileNotFoundError(f"找不到 Shell 脚本，请检查路径: {base_dir}")

        try:
            # 2. 赋予脚本可执行权限 (chmod +x)
            subprocess.run(["chmod", "+x", str(find_script)], check=True)
            subprocess.run(["chmod", "+x", str(activate_script)], check=True)
            print("✅ 脚本执行权限已配置。")

            # 3. 运行第一个脚本获取端口列表
            print("🔍 正在扫描 CAN 端口...")
            result = subprocess.run(
                ["sudo", str(find_script)], capture_output=True, text=True, check=True
            )

            # 4. 解析输出结果
            # 正则表达式匹配类似 "can4" 和 "1-4.2:1.0" (含有横杠和冒号的 USB 拓扑路径)
            # 忽略 ".mttcan" 结尾的端口
            pattern = re.compile(
                r"Interface (can\d+) is connected to USB port (\d+-[\d\.]+:\d+\.\d+)"
            )
            matches = pattern.findall(result.stdout)

            ret = []
            if not matches:
                print("⚠️ 未找到匹配的机械臂 CAN 端口 (未匹配到类似 1-4.2:1.0 的拓扑)。")
                print("原始输出如下:\n", result.stdout)
                return ret

            # 5. 遍历匹配结果，运行激活脚本
            for can_iface, usb_port in matches:
                print(f"🎯 发现机械臂端口: {can_iface} -> {usb_port}，准备激活...")

                # 拼接并执行命令: ./can_activate.sh can4 1000000 1-4.2:1.0
                activate_cmd = [
                    "sudo",
                    str(activate_script),
                    can_iface,
                    baudrate,
                    usb_port,
                ]

                subprocess.run(activate_cmd, check=True)
                print(f"🚀 成功激活 {can_iface} ({usb_port})！\n")
                ret.append(can_iface)

            return ret

        except subprocess.CalledProcessError as e:
            print(f"❌ 运行 Shell 脚本时出错!")
            print(f"命令: {e.cmd}")
            print(f"错误输出: {e.stderr if e.stderr else '无详细错误信息'}")
        except Exception as e:
            print(f"❌ 发生未知错误: {e}")
        return []

    def set_move_mode(self, move_mode_end_pose: bool):
        """切换机械臂控制模式。

        Args:
            move_mode_end_pose: `True` 表示切换到末端位姿控制，`False`
                表示切换到关节控制。

        Raises:
            TimeoutError: 在超时时间内未完成模式切换时抛出。
        """
        start = time.time()
        self.piper.MotionCtrl_2(
            ctrl_mode=0x01,
            move_mode=0x0 if move_mode_end_pose else 0x01,
            move_spd_rate_ctrl=50 if self.debug_mode else 100,
            is_mit_mode=0x00,
        )
        while True:
            status = self.piper.GetArmStatus()
            if status.arm_status.mode_feed == (0x00 if move_mode_end_pose else 0x01):
                break
            if time.time() - start > self.timeout:
                raise TimeoutError(
                    "Failed to switch move mode within the specified timeout."
                )

    @staticmethod
    def arm_status2str(status):
        """将机械臂状态码转换为可读文本。"""
        status_dict = {
            0x00: "正常",
            0x01: "急停",
            0x02: "无解",
            0x03: "奇异点",
            0x04: "目标角度超过限",
            0x05: "关节通信异常",
            0x06: "关节抱闸未打开",
            0x07: "机械臂发生碰撞",
            0x08: "拖动示教时超速",
            0x09: "关节状态异常",
            0x0A: "其它异常",
            0x0B: "示教记录",
            0x0C: "示教执行",
            0x0D: "示教暂停",
            0x0E: "主控NTC过温",
            0x0F: "释放电阻NTC过温",
        }
        return status_dict.get(status, "未知状态: " + hex(status))


if __name__ == "__main__":
    arm: PiperBySDK = Arm(debug_mode=False)  # type:ignore
    time.sleep(1)
    print("关节角度:", arm.get_arm_angles())
    print("末端位姿:", np.array(arm.get_arm_pose()).round(2).tolist())

    arm.move_to([0.2, 0.2, 0.2], gripper_open_0to1=0, rot_rad=np.pi / 6)
    time.sleep(2)
    print("关节角度:", arm.get_arm_angles())
    print("末端位置:", np.array(arm.get_arm_pose()).round(2).tolist())

    arm.move_to_home(gripper_open_0to1=1)
    time.sleep(1)
    print("关节角度:", arm.get_arm_angles())
    print("末端位置:", np.array(arm.get_arm_pose()).round(2).tolist())

    arm.catch_and_place(0.2, 0.0, -np.pi / 6 * 0, [0.2, 0.2])
    time.sleep(1)
    arm.move_to_home(gripper_open_0to1=1)

    arm.disconnect_arm()
