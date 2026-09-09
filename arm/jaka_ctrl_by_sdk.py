# Description: JAKA Mini 机械臂 SDK 控制实现（jkrc，TCP 连接 + TG-9801 触觉夹爪）。
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# jkrc SDK 二进制（Windows: jkrc.pyd + jakaAPI.dll；Linux: jkrc.so + libjakaAPI.so）
# 放在 arm/jaka_sdk/ 目录下，导入前注入 sys.path。
_JAKA_SDK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jaka_sdk")
sys.path.insert(0, _JAKA_SDK_DIR)
if os.name == "nt":
    os.add_dll_directory(_JAKA_SDK_DIR)

from typing import Union, List
from collections.abc import Sequence
from arm.arm_base import Arm, StepCallback
from arm.jaka_gripper import JakaGripper
from utils.config_getter import get_config_value
import time
import numpy as np
import jkrc
from scipy.spatial.transform import Rotation as R

ABS = 0  # jkrc 绝对运动
INCR = 1  # jkrc 增量运动


class JakaBySDK(Arm):
    """JAKA SDK 机械臂控制实现。

    与 PiperBySDK 的主要差异：
    - TCP/IP 连接控制器（config 的 `arm_port` 填控制器 IP），单位 rad + mm；
    - `move_to` 为混合策略：非奇异起点用 `linear_move` 直线运动，奇异起点
      （如 home 的 j5=0）或直线失败时走 `kine_inverse` + 关节插值（超限支
      限位感知修复），不需要本地 URDF/kinpy；
    - 关节绝对编码，无 offset 标定，`get_arm_angles` 与 `get_raw_joint_angles` 等价；
    - 夹爪为 TG-9801 触觉夹爪（自适应模式二值动作 grip/release，
      开度读数 0-1 由行程端点归一化得到）。
    """

    JOINT_COUNT = 6
    # 默认末端朝下的欧拉角 [RZ, RY, RX]（度）。JAKA Mini 全零位的实际朝向
    # 需真机确认后调整。
    DEFAULT_DOWN_EULER_DEG_ZYX = [0.0, 180.0, 0.0]
    # sent_action 里 gripper.pos 的满量程：开度 0-1 乘该值，与 Piper 的
    # VLA 动作空间（gripper_0to1 * 100）数值保持一致。
    MAX_GRIPPER_ANGLE_DEG = 100
    # 100% 速度对应的关节运动速度（rad/s）。JAKA 关节速度上限 180°/s，
    # 此处取保守基率，move_speed 百分比在其上线性缩放。
    BASE_JOINT_SPEED_RAD_S = 1.0
    # 100% 速度对应的末端直线运动速度（mm/s）。JAKA linear_move 默认
    # 500mm/s，抓取场景取保守基率，move_speed 百分比线性缩放。
    BASE_LINEAR_SPEED_MM_S = 100.0
    # TG-9801 力位模式默认速度（有效范围 200~1500）
    GRIPPER_SPEED = 1100
    # TG-9801 默认夹持力（有效范围 0~100）
    GRIPPER_FORCE = 50
    # JAKA Mini 各关节软限位（硬件用户手册 表5-1，单位度）
    JOINT_LIMITS_DEG = (
        (-360.0, 360.0), (-125.0, 125.0), (-130.0, 130.0),
        (-360.0, 360.0), (-120.0, 120.0), (-360.0, 360.0),
    )
    # j5 接近 0°（腕部奇异）或限位 ±120°（退化）时，笛卡尔规划无法启动
    SINGULAR_J5_EPS_DEG = 5.0
    # 已知良好的抓取构型（本机实测），作为 kine_inverse 的备选参考种子：
    # 以 home 为参考时部分目标会解出 j5 超限的支，换此种子可解出良好支
    GRASP_IK_SEED_DEG = [46.2, -20.6, -76.1, 0.0, -83.4, 76.2]

    def __init__(
        self,
        debug_mode: bool = False,
        move_speed: int | None = None,
    ):
        """初始化 JAKA 机械臂控制器。

        Args:
            debug_mode: 是否使用调试模式；调试模式下超时更长、夹爪力更保守。
            move_speed: 运动速度百分比，范围 1-100；默认 None 时读取配置项
                `arm_move_speed`（缺省 100 全速）。值越小机械臂运动越慢、越平稳。
        """
        super().__init__()
        self.debug_mode = debug_mode
        if move_speed is None:
            move_speed = get_config_value(
                "arm_move_speed", 100, raise_if_missing=False
            )
        self.move_speed = max(min(int(move_speed), 100), 1)
        # 基准超时（100% 速度时的超时秒数），与 PiperBySDK 语义一致，
        # 录制等场景可直接改 self.timeout_base_s 覆盖默认值。
        self.timeout_base_s = 10 if debug_mode else 5
        self.timeout = self.timeout_base_s * 100 / self.move_speed
        # 插值步数与速度成反比：move_speed 越小步数越多，运动越慢越平滑。
        self.steps = max(100, int(100 * 100 / self.move_speed))

        arm_ip = get_config_value("arm_port")
        self.robot = jkrc.RC(arm_ip)
        ret = self.robot.login()
        if ret[0] != 0:
            raise ConnectionError(f"JAKA 控制器登录失败 ({arm_ip}): {ret}")
        # 夹爪共享同一条控制器连接（控制器 SDK 连接数有限）；
        # power_on 会给整个机械臂上电并等待夹爪就绪（内部等电流非 0）。
        self.gripper = JakaGripper(robot=self.robot)
        self.gripper.power_on()
        self.robot.enable_robot()

        # 夹爪用自适应模式：grip/release 二值动作（06_gripper_control.py
        # 实测可用的命令），目标夹持力经 target_force 设置。
        self.gripper.set_target_force(30 if debug_mode else self.GRIPPER_FORCE)
        # 缓存夹爪行程端点，用于开度 0-1 与夹爪位置（约 0~1000）的换算。
        # 端点必须互不相等且都在量程 [0, 1000] 内；该对寄存器读数在部分
        # 启动轮次不可靠（实测出现过 0/0 和 open=3522），异常时回退到
        # 本机实测行程：close=1000、open=0（即位置 0=张开、1000=闭合）。
        open_pos = self.gripper.read_open_position()
        close_pos = self.gripper.read_close_position()
        endpoints_valid = (
            open_pos is not None
            and close_pos is not None
            and open_pos != close_pos
            and 0 <= open_pos <= 1000
            and 0 <= close_pos <= 1000
        )
        if not endpoints_valid:
            print(
                f"Warning: TG-9801 行程读取异常 (close={close_pos}, "
                f"open={open_pos})，回退本机实测行程 close=1000/open=0"
            )
            self.gripper_open_pos = 0
            self.gripper_close_pos = 1000
        else:
            self.gripper_open_pos = open_pos
            self.gripper_close_pos = close_pos
        print(
            f"TG-9801 行程: close={self.gripper_close_pos}, "
            f"open={self.gripper_open_pos}"
        )

        self.reset()

    # ========== 高层接口实现 ==========

    def set_move_speed(self, move_speed: int) -> None:
        """更新运动速度百分比并同步相关参数。

        速度在每次 joint_move 调用时以 rad/s 传入，无需额外 SDK 调用；
        此处同步插值步数与到位超时（语义与 PiperBySDK 一致）。

        Args:
            move_speed: 运动速度百分比，范围 1-100，越小越慢越平稳。
        """
        self.move_speed = max(min(int(move_speed), 100), 1)
        self.timeout = self.timeout_base_s * 100 / self.move_speed
        self.steps = max(100, int(100 * 100 / self.move_speed))

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

        # TG-9801 自带速度规划，且 RS485 透传不适合 10ms 周期连续写，
        # 因此夹爪不跟随关节插值逐步下发，直接发一次目标动作。
        if desired_gripper_0to1 is not None:
            self._apply_gripper(desired_gripper_0to1)

        if angles_deg is None:
            return True

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

        current_angles_deg, current_gripper_0to1 = self.get_arm_angles()
        if current_angles_deg is None or current_gripper_0to1 is None:
            return False

        desired_joint_angles = [float(angle) for angle in angles_deg]
        final_gripper_0to1 = (
            float(current_gripper_0to1)
            if desired_gripper_0to1 is None
            else desired_gripper_0to1
        )
        joint_speed_rad_s = self.BASE_JOINT_SPEED_RAD_S * self.move_speed / 100

        # VLA 录制需要逐步下发以产生轨迹数据流；但 JAKA 的 joint_move 自带
        # 速度规划，非阻塞密集下发会反复打断并重新规划，导致加减速抖振。
        # 因此仅在录制（step_callback 非 None）时插值下发，常规运动直接走
        # 下方一次阻塞 joint_move（平滑）。录制路径后续任务改用 servo_j
        # 重写后即可消除此处的抖动。
        if step_callback is not None:
            stream_error_codes = set()
            for step_index, alpha in enumerate(np.linspace(0, 1, self.steps + 1)[1:]):
                interp_joint_angles = [
                    current_angle * (1 - alpha) + desired_angle * alpha
                    for current_angle, desired_angle in zip(
                        current_angles_deg, desired_joint_angles, strict=True
                    )
                ]
                interp_gripper_0to1 = (
                    float(current_gripper_0to1) * (1 - alpha)
                    + final_gripper_0to1 * alpha
                )
                ret = self.robot.joint_move(
                    np.deg2rad(interp_joint_angles).tolist(),
                    ABS,
                    False,
                    joint_speed_rad_s,
                )
                if ret[0] != 0:
                    stream_error_codes.add(ret[0])
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
            if stream_error_codes:
                print(f"插值下发出现错误码: {sorted(stream_error_codes)}")

        ret = self.robot.joint_move(
            np.deg2rad(desired_joint_angles).tolist(),
            ABS,
            True,
            joint_speed_rad_s,
        )
        if ret[0] != 0:
            print(
                f"最终阻塞运动失败，错误码 {ret[0]}，"
                f"目标关节角(deg): {[round(a, 2) for a in desired_joint_angles]}"
            )
            print("运动状态查询:", self.robot.get_motion_status())
            return False

        try:
            self.wait_until_reached(desired_joint_angles)
        except TimeoutError as e:
            print(f"机械臂未在超时内到达目标位姿: {e}")
            return False

        return True

    def get_arm_angles(
        self, retry_times=None
    ) -> tuple[Union[List[float], None], Union[float, None]]:
        try:
            ret = self.robot.get_actual_joint_position()
            if ret[0] != 0:
                raise RuntimeError(
                    f"get_actual_joint_position 返回错误码 {ret[0]}"
                )
            angles_deg = [float(np.degrees(angle)) for angle in ret[1]]
            gripper_0to1 = self._gripper_pos_0to1()
            if gripper_0to1 is None:
                raise RuntimeError("读取夹爪位置失败")
        except Exception:
            if retry_times is None:
                retry_times = self.get_arm_angles_retry_times
            if retry_times > 0:
                time.sleep(self.catch_time_interval_s)
                return self.get_arm_angles(retry_times - 1)
            return None, None
        return angles_deg, gripper_0to1

    def get_arm_pose(self) -> tuple[list[float] | None, list[float] | None]:
        try:
            ret = self.robot.get_actual_tcp_position()
            if ret[0] != 0:
                raise RuntimeError(f"get_actual_tcp_position 返回错误码 {ret[0]}")
            x, y, z, rx, ry, rz = ret[1]
            pos = [x / 1000.0, y / 1000.0, z / 1000.0]
            # jkrc 位姿为 [x, y, z(mm), rx, ry, rz(rad)]。手册未写明 rpy 旋转序，
            # 此处按常见约定 R = Rz·Ry·Rx（对应 scipy 的 "xyz" 外旋序）处理，
            # 与 PiperBySDK 读取 GetArmEndPoseMsgs 的转换方式一致。
            # 真机首次运行时建议用 kine_forward 已知关节角对比验证该顺序。
            euler_zyx_deg = R.from_euler("xyz", [rx, ry, rz]).as_euler(
                "zyx", degrees=True
            )
            return pos, euler_zyx_deg.tolist()
        except Exception:
            return None, None

    def move_to_home(
        self,
        gripper_open_0to1: float | None = None,
        step_callback: StepCallback | None = None,
        safe_pos: bool = False,
    ) -> bool:
        """safe_pos 仅为对齐基类签名；JAKA 全零位即可直接失能，无额外安全姿态。"""
        return self.set_arm_angles(
            [0] * self.JOINT_COUNT,
            gripper_open_0to1=gripper_open_0to1,
            step_callback=step_callback,
        )

    def prepare_for_manual_teach(self) -> bool:
        """标定拖动采点前，先摆到夹爪朝下的工作区上方姿态，缩短拖动距离。"""
        return self.move_to([0.2, 0.0, 0.25])

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

        # 构建目标位姿（mm + rpy rad）
        if euler_angles_deg_zyx is not None:
            target_euler = list(euler_angles_deg_zyx)
        else:
            target_euler = list(self.DEFAULT_DOWN_EULER_DEG_ZYX)
            if rot_rad is not None:
                target_euler[0] = np.degrees(rot_rad)
        rpy_rad = R.from_euler("zyx", target_euler, degrees=True).as_euler("xyz")
        pose_mm = [pos[0] * 1000, pos[1] * 1000, pos[2] * 1000, *rpy_rad]

        # 夹爪与 set_arm_angles 同策略：直接发目标动作，不随运动插值
        if gripper_open_0to1 is not None:
            if not 0 <= gripper_open_0to1 <= 1:
                raise ValueError("gripper_open_0to1 must in [0, 1]")

        current_angles_deg, _ = self.get_arm_angles()
        if current_angles_deg is None:
            print("读取当前关节角失败")
            return False
        j5 = current_angles_deg[4]
        near_singular = abs(j5) < self.SINGULAR_J5_EPS_DEG or abs(j5) > (
            abs(self.JOINT_LIMITS_DEG[4][1]) - self.SINGULAR_J5_EPS_DEG
        )

        # 非奇异起点先走末端直线运动（轨迹为直线，适合抓取的"上方→下降→
        # 抬起"动作）；home（全零，j5=0 腕部奇异）等奇异起点无法启动笛卡尔
        # 规划（-12），直线失败时也退回关节路径。
        # 注意：直线路径暂不产生 step_callback 数据流（VLA 录制为后续阶段）。
        if not near_singular:
            if gripper_open_0to1 is not None:
                self._apply_gripper(float(gripper_open_0to1))
            speed_mm_s = self.BASE_LINEAR_SPEED_MM_S * self.move_speed / 100
            ret = self.robot.linear_move(pose_mm, ABS, True, speed_mm_s)
            if ret[0] == 0:
                return True
            print(f"直线运动失败（错误码 {ret[0]}），退回关节路径")

        angles_deg = self._solve_ik_deg(pose_mm, current_angles_deg)
        if angles_deg is None:
            print(
                "无可执行的逆解（所有支均超关节限位），"
                f"目标位姿(mm, rpy): {[round(v, 2) for v in pose_mm]}"
            )
            return False
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
        print("Resetting JAKA arm to initial state.")
        # 若当前仍处于拖拽模式（如用户按 free 按钮拖拽后未完全退出），先退出
        # 并重新使能，否则 reset 的关节运动会失败。
        self.robot.drag_mode_enable(False)
        self.robot.enable_robot()
        try:
            self.reset()
        except TimeoutError as e:
            print(f"Error during reset: {e}")
        time.sleep(0.5)
        self.robot.disable_robot()
        # 末端 RS485 与夹爪供电依赖机械臂上电，power_off 即一并断掉
        self.robot.power_off()
        self.robot.logout()
        print("Arm disconnected")

    def enable_torque(self):
        # 退出拖拽（手导）模式并重新使能。拖拽模式下关节力矩跟随，须退出
        # 后恢复伺服才可运动。
        self.robot.drag_mode_enable(False)
        self.robot.enable_robot()
        print("JAKA arm enabled.")

    def disable_torque(self):
        # 手眼标定采点默认用末端 free 按钮拖拽（按住=拖拽、松开=锁定）。
        # 软件 drag_mode_enable 在松手空闲一段时间后会超时自动退出（重新锁
        # 抱闸），采点间隙会反复掉出拖拽，故弃用软件拖拽，改由用户按 free
        # 按钮触发。机械臂保持使能状态即可，free 按钮在使能态下释放抱闸。
        print("JAKA: 请按住末端 free 按钮拖拽机械臂采点（松开即锁定）。")

    # ========== 内部方法 ==========

    def reset(self):
        """清除错误状态并回到 Home 位。"""
        self.robot.clear_error()
        self.move_to_home(gripper_open_0to1=1)

    def _angles_within_limits(self, angles_deg: Sequence[float]) -> bool:
        """检查各关节角是否都在软限位内。"""
        return all(
            lo - 1e-6 <= angle <= hi + 1e-6
            for angle, (lo, hi) in zip(
                angles_deg, self.JOINT_LIMITS_DEG, strict=True
            )
        )

    @staticmethod
    def _wrist_flip(angles_deg: list[float]) -> list[float]:
        """腕部翻转等效位形 (j4∓180, -j5, j6±180)，TCP 位姿不变。"""
        flip = 180.0 if angles_deg[3] > 0 else -180.0
        return [
            angles_deg[0],
            angles_deg[1],
            angles_deg[2],
            angles_deg[3] - flip,
            -angles_deg[4],
            angles_deg[5] + flip,
        ]

    def _solve_ik_deg(
        self, pose_mm: list[float], ref_deg: Sequence[float]
    ) -> Union[list[float], None]:
        """kine_inverse 求关节解（度）并修复超限支。

        控制器 IK 返回距参考位形最近的支但不检查限位（实测部分目标
        以 home 为参考会解出 j5 超 ±120° 限位的支，执行被 -12 拒绝）。
        超限时依次尝试腕部翻转、换已知良好的抓取构型种子重解。
        """
        for seed in (ref_deg, self.GRASP_IK_SEED_DEG):
            ret = self.robot.kine_inverse(
                np.deg2rad(list(seed)).tolist(), pose_mm
            )
            if ret[0] != 0:
                continue
            solution = [float(angle) for angle in np.degrees(ret[1])]
            if self._angles_within_limits(solution):
                return solution
            flipped = self._wrist_flip(solution)
            if self._angles_within_limits(flipped):
                return flipped
        return None

    def _gripper_pos_0to1(self) -> Union[float, None]:
        """读取夹爪当前位置并归一化到 [0, 1]（1 为全开）。

        本机夹爪为反向行程（位置 0=张开、1000=闭合），公式对正反
        两个方向均成立。
        """
        pos = self.gripper.read_position()
        if pos is None:
            return None
        lo = self.gripper_close_pos
        hi = self.gripper_open_pos
        if hi == lo:
            return 0.0
        value = (pos - lo) / (hi - lo)
        return float(np.clip(value, 0, 1).round(2))

    def _apply_gripper(self, open_0to1: float) -> None:
        """驱动夹爪到目标开度。

        TG-9801 自适应模式为二值动作：开度 >=0.5 视为张开（release），
        否则夹取（grip，夹到目标力即停）。这两条命令经
        06_gripper_control.py 真机验证可用。
        """
        if open_0to1 >= 0.5:
            self.gripper.release()
        else:
            self.gripper.grip(self.GRIPPER_SPEED)

    def wait_gripper_gripped(self) -> None:
        """轮询 TG-9801 位置直至夹取稳定（位置不再变化）再放行抬升。

        相比固定 sleep 自适应：夹厚物体提前稳定即抬、空夹走满行程也
        等得了。先等位置开始变化（排除指令生效前的静止读数），再等
        连续两次变化低于阈值；超时（含读取异常）放行，保持原行为上限。
        """
        poll_interval_s = 0.2
        timeout_s = 3.0
        stable_threshold = 5  # 位置计数（行程 0~1000）
        initial_pos = self.gripper.read_position()
        last_pos = initial_pos
        moved = False
        start = time.time()
        while time.time() - start < timeout_s:
            time.sleep(poll_interval_s)
            pos = self.gripper.read_position()
            if pos is None:
                continue
            print(
                f"[wait_gripper] t={time.time() - start:.2f}s "
                f"pos={pos} state={self.gripper.read_state()} "
                f"current={self.gripper.read_current()}mA"
            )
            if not moved:
                if abs(pos - initial_pos) > stable_threshold:
                    moved = True
                    last_pos = pos
                continue
            if abs(pos - last_pos) <= stable_threshold:
                print(f"[wait_gripper] 位置稳定于 {pos}，放行抬升")
                return
            last_pos = pos
        print(f"[wait_gripper] 超时({timeout_s}s)放行，末位置 {last_pos}")


if __name__ == "__main__":
    # 最小控制验证：读状态 → 小幅关节运动 → 夹爪闭合/张开 → 复位断开
    arm: JakaBySDK = Arm(debug_mode=False)  # type:ignore
    time.sleep(1)
    print("关节角度:", arm.get_arm_angles())
    print("末端位姿:", np.array(arm.get_arm_pose()).round(2).tolist())

    # 1) 小幅关节运动：关节1 转 +10° 再回零（关节插值路径，已验证可用）
    arm.set_arm_angles([10, 0, 0, 0, 0, 0])
    time.sleep(1)
    print("关节1 +10° 后关节角:", [round(a, 2) for a in arm.get_arm_angles()[0]])
    arm.set_arm_angles([0, 0, 0, 0, 0, 0])
    time.sleep(1)

    # 2) 夹爪：闭合 → 张开（grip/release，06_gripper_control.py 同款命令）
    arm.set_gripper(0)
    time.sleep(2)
    print("夹爪开度(应为0，闭合):", arm.get_arm_angles()[1])
    arm.set_gripper(1)
    time.sleep(2)
    print("夹爪开度(应为1，张开):", arm.get_arm_angles()[1])

    # 3) 位姿运动验证：上方 → 下降 → 抬起（含 home 奇异起点的关节路径）
    #    [0.2, 0, ...] 为此前 IK 选到超限支的目标，作为回归测试用例
    arm.move_to([0.2, 0.0, 0.30])
    time.sleep(1)
    print("上方位姿:", np.array(arm.get_arm_pose()).round(2).tolist())
    arm.move_to([0.2, 0.0, 0.20])
    time.sleep(1)
    print("下降位姿:", np.array(arm.get_arm_pose()).round(2).tolist())
    arm.move_to([0.2, 0.0, 0.30])
    time.sleep(1)

    # 4) 复位并断开
    arm.move_to_home()
    arm.disconnect_arm()
