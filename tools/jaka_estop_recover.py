# Description: 急停后的机械臂复位。急停释放（旋钮拔起）后运行本脚本：
# 清除错误 -> 上电 -> 使能 -> 人工确认后慢速回 Home。
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_JAKA_SDK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "arm", "jaka_sdk"
)
sys.path.insert(0, _JAKA_SDK_DIR)

import jkrc
from utils.config_getter import get_config_value

ABS = 0  # jkrc 绝对运动
HOME_SPEED_RAD_S = 0.3  # 回零速度，慢速防止急停后姿态异常时扫到物体


def main():
    arm_ip = get_config_value("arm_port")
    robot = jkrc.RC(arm_ip)
    ret = robot.login()
    if ret[0] != 0:
        raise ConnectionError(f"JAKA 控制器登录失败 ({arm_ip}): {ret}")
    print(f"已连接控制器 {arm_ip}")

    print("清除错误状态...")
    robot.clear_error()
    time.sleep(1)

    # 急停后通常整体断电；若已上电，重复调用无副作用（仅返回非 0 错误码）
    print("上电...")
    ret = robot.power_on()
    print(f"power_on 返回码: {ret[0]}")
    time.sleep(1)

    print("使能...")
    ret = robot.enable_robot()
    print(f"enable_robot 返回码: {ret[0]}")
    time.sleep(1)

    input("\n确认机械臂回 Home 路径无障碍后按回车（将慢速回零位）...")
    ret = robot.joint_move([0.0] * 6, ABS, True, HOME_SPEED_RAD_S)
    if ret[0] != 0:
        print(f"回 Home 失败，错误码 {ret[0]}，运动状态: {robot.get_motion_status()}")
    else:
        print("已回到 Home 位。")

    print("断使能并断电...")
    robot.disable_robot()
    time.sleep(0.5)
    robot.power_off()
    robot.logout()
    print("复位完成，机械臂已断使能断电。")


if __name__ == "__main__":
    main()
