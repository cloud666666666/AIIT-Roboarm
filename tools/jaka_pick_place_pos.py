# Description: 拖拽机械臂选出放置位置：按住末端 free 按钮拖到目标点，松开锁定，
# 回车采样末端 TCP 坐标，最后输出可直接粘贴进 config.yaml 的 place_pos 片段。
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arm.arm_base import Arm


def main():
    arm: Arm = Arm()  # type: ignore

    print("按住机械臂末端 free 按钮拖拽（松开即锁定），把夹爪移到想要的放置点上方。")
    print("每按一次回车采样一个点，输入 q 后回车结束。")
    print("注意：末端高度尽量贴近桌面（与手眼标定平面一致），place_pos 只使用 x, y。\n")

    points = []
    while True:
        ans = input(f"点 {len(points) + 1}: 回车采样 / q 结束 > ").strip().lower()
        if ans == "q":
            break
        pos, euler = arm.get_arm_pose()
        if pos is None:
            print("读取 TCP 位置失败，请重试")
            continue
        points.append((pos[0], pos[1]))
        print(f"TCP (x, y, z) = ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m")

    if points:
        print("\n可将以下片段粘贴进 config.yaml 的 place_pos（类别名自行替换）：")
        print("place_pos:")
        for i, (x, y) in enumerate(points):
            print(f"  your_class_{i + 1}:")
            print(f"    pos: [{x:.3f}, {y:.3f}]")

    input("\n回车回 home 并断开...")
    arm.disconnect_arm()


if __name__ == "__main__":
    main()
