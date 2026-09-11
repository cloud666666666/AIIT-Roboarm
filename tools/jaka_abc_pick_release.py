# Description: 桌面 A/B/C 三点抓放循环验证：依次运行到每个点，原地执行
# 抓取(下降+闭合夹爪) -> 抬升 -> 下降 -> 放下(张开夹爪) -> 抬起离开。
# 点位采样：先运行 tools/jaka_pick_place_pos.py 拖拽采样，把 x,y 填入 POINTS。
# 需 config.yaml 中 arm_type 为 jaka；高度取 default_desktop_height + catch_raise_height。
import math
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arm.arm_base import Arm

# 桌面三个点位 (x, y)，单位米；z 取配置 default_desktop_height（桌面高度）
POINTS = {
    "A": [-0.15, 0.3],
    "B": [0, 0.3],
    "C": [0.15, 0.3],
}

# 末端绕 z 轴旋转 90 度（正向逆时针）；反向改为 -math.pi / 2
ROT_RAD = math.pi / 2


def pick_release_at_point(arm: Arm, name: str, x: float, y: float) -> None:
    """在单个点原地执行 抓取 -> 抬升 -> 下降 -> 放下 的完整循环。"""
    z = arm.desktop_height
    top_z = z + arm.catch_raise_height
    pause = arm.catch_time_interval_s
    print(f"\n===== 点 {name} ({x:.3f}, {y:.3f}) m =====")

    # 1. 运行到点上方，张开夹爪（避免下降时碰到物体）
    print(f"[{name}] 1. 移动到点上方 z={top_z:.3f}")
    if not arm.move_to([x, y, top_z], gripper_open_0to1=1, rot_rad=ROT_RAD):
        print(f"[{name}] 移动到点上方失败，跳过该点")
        return
    time.sleep(pause * 2)

    # 2. 抓取：下降到桌面并闭合夹爪
    print(f"[{name}] 2. 下降到桌面 z={z:.3f} 并抓取")
    if not arm.move_to([x, y, z], gripper_open_0to1=1, rot_rad=ROT_RAD):
        print(f"[{name}] 下降失败，跳过该点")
        return
    time.sleep(pause)
    arm.set_gripper(0)
    arm.wait_gripper_gripped()

    # 3. 抬升
    print(f"[{name}] 3. 抬升到 z={top_z:.3f}")
    arm.move_to([x, y, top_z], rot_rad=ROT_RAD)
    time.sleep(pause)

    # 4. 下降并放下物品（张开夹爪）
    print(f"[{name}] 4. 下降到 z={z:.3f} 放下物品")
    arm.move_to([x, y, z], rot_rad=ROT_RAD)
    time.sleep(pause)
    arm.set_gripper(1)
    time.sleep(pause * 2)

    # 5. 抬起离开桌面，避免平移到下一点时扫到物体
    print(f"[{name}] 5. 抬起，准备前往下一点")
    arm.move_to([x, y, top_z], rot_rad=ROT_RAD)
    time.sleep(pause)


def main():
    arm: Arm = Arm(debug_mode=False)  # type: ignore
    print(f"桌面高度 z={arm.desktop_height:.3f} m，抬升 +{arm.catch_raise_height:.3f} m")
    print(f"将依次在 {list(POINTS)} 三点执行：抓取 -> 抬升 -> 下降 -> 放下")
    input("确认三点路径无障碍、各点已放好物体后按回车开始...")

    while True:
        for name, (x, y) in POINTS.items():
            pick_release_at_point(arm, name, x, y)

        print("\n全部点位完成，回 Home，保持上电不断开。")
        arm.move_to_home(gripper_open_0to1=1)
        time.sleep(1)

        answer = input("按回车再执行一次，按 q 断电退出: ").strip().lower()
        if answer == "q":
            print("退出，断电断开。")
            arm.disconnect_arm()
            break


if __name__ == "__main__":
    main()
