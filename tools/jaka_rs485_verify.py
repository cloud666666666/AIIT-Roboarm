# Description: 验证「纯追加、永不删改」信号量表策略。
# 前提：JAKA 控制器（控制柜）已断电重启，机械臂已下电。
# 会话 1：重启后首次登录 → 空表 → power_on 追加 → 开合跟踪；
# 会话 2：控制器不重启、机械臂掉电再上电 → 复用表条目 → 开合跟踪。
# 请全程观察夹爪物理动作（是否真的闭合到底）。
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arm.jaka_sdk  # noqa: F401
import jkrc
from arm.jaka_gripper import JakaGripper

IP = "10.5.5.100"


def raw(robot):
    info = robot.get_rs485_signal_info()
    print(f"  表: err={info[0] if info else None} "
          f"n={len(info[1]) if info and info[0] == 0 else '-'}")
    if info and info[0] == 0:
        for s in info[1]:
            print(f"    {s.get('sig_name')!r} @0x{s.get('sig_addr'):04X} = {s.get('value')}")


def cycle(robot, g, tag):
    print(f"[{tag}] release（观察夹爪张开）->", g.release())
    time.sleep(2)
    raw(robot)
    print(f"[{tag}] grip（观察夹爪闭合到底）->", g.grip(1000))
    time.sleep(2.5)
    raw(robot)
    print(f"[{tag}] release ->", g.release())
    time.sleep(2)
    raw(robot)


# ---------- 会话 1 ----------
robot = jkrc.RC(IP)
print("login:", robot.login())
print("[S1-1] 控制器重启后首次登录，信号量表应为空:")
raw(robot)

g = JakaGripper(robot=robot)
g.power_on()
print("[S1-2] power_on 后（应为 8 个新追加条目）:")
raw(robot)
print("[S1-3] 行程端点:", g.read_open_position(), g.read_close_position())
print("[S1-4] 表应与 S1-2 一致（_read_tmp 不折腾表）:")
raw(robot)

cycle(robot, g, "S1")

robot.power_off()
time.sleep(1)
robot.logout()
print("\n===== 会话 1 结束（机械臂已下电，控制器不重启）=====\n")

# ---------- 会话 2 ----------
robot2 = jkrc.RC(IP)
print("login:", robot2.login())
print("[S2-1] 第二次登录（表应保留会话 1 的 8 个条目）:")
raw(robot2)

g2 = JakaGripper(robot=robot2)
g2.power_on()
print("[S2-2] power_on 后（应无新增，表与 S2-1 一致）:")
raw(robot2)

cycle(robot2, g2, "S2")

print("\n清理断开")
robot2.power_off()
robot2.logout()
