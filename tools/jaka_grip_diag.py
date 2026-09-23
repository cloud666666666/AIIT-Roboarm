# Description: 夹爪「自适应夹取只闭合一小段就停」诊断。
# 依次验证：[A] 静止状态寄存器 [B] 0x06 单写是否生效 [C] 自适应 grip 位置跟踪
# [D] 力位模式走到绝对位置 1000（区分机械问题 vs 力控逻辑问题）
# [E] 指尖校准+力清零后自适应 grip 再试。
# 请全程观察夹爪物理动作，尤其 [C]/[D]/[F] 是否真的闭合。
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arm.jaka_sdk  # noqa: F401
import jkrc
from arm.jaka_gripper import JakaGripper

IP = "10.5.5.100"


def poll_position(seconds, step=0.5):
    for i in range(int(seconds / step)):
        time.sleep(step)
        print(f"  t={(i + 1) * step:.1f}s position={g.read_position()} "
              f"state={g.read_state()} current={g.read_current()}mA")


robot = jkrc.RC(IP)
print("login ->", robot.login())

try:
    g = JakaGripper(robot=robot)  # 修复后：__init__ 不再上电前加信号量
    g.power_on()                  # 上电 + 注册信号量 + 就绪检查
    robot.enable_robot()

    print("\n[A] 上电后静止状态")
    print(f"  state={g.read_state()} hardness={g.read_hardness()} "
          f"position={g.read_position()} current={g.read_current()}mA "
          f"adaptive_speed={g.read_adaptive_speed()} action={g.read_action()}")
    print(f"  error_flags=0x{g.read_error_flags():04X} "
          f"target_force={g.read_target_force()} "
          f"Fx/Fy/Fz={g.read_force_fx()},{g.read_force_fy()},{g.read_force_fz()}")

    print("\n[B] 0x06 单写是否生效：set_adaptive_speed(777) 后读回")
    print("  set_adaptive_speed(777) ->", g.set_adaptive_speed(777))
    time.sleep(2)
    print(f"  adaptive_speed 读回 = {g.read_adaptive_speed()}（777=生效, 非777=单写不落盘）")

    print("\n[C] 自适应 grip（请观察是否闭合到底）")
    print("  grip(1000) ->", g.grip(1000))
    poll_position(4)
    print("  release ->", g.release())
    poll_position(2)

    print("\n[D] 力位模式绝对定位：set_control_mode(1) + force_position_control(1000, 1000)")
    print("  set_control_mode(1) ->", g.set_control_mode(1))
    time.sleep(0.5)
    print("  force_position_control(1000, 1000) ->", g.force_position_control(1000, 1000))
    poll_position(4)

    print("\n[E] 恢复自适应模式并松开")
    print("  set_control_mode(0) ->", g.set_control_mode(0))
    time.sleep(0.5)
    print("  release ->", g.release())
    poll_position(2)

    print("\n[F] 指尖校准 + 力清零后，自适应 grip 再试（请观察是否闭合到底）")
    print("  calibrate_fingertip(1) ->", g.calibrate_fingertip(1))
    time.sleep(1)
    print("  clear_fingertip_zero(1) ->", g.clear_fingertip_zero(1))
    time.sleep(1)
    print(f"  Fx/Fy/Fz={g.read_force_fx()},{g.read_force_fy()},{g.read_force_fz()}")
    print("  grip(1000) ->", g.grip(1000))
    poll_position(4)
    print("  release ->", g.release())

finally:
    print("\n清理断开")
    robot.power_off()
    robot.logout()
