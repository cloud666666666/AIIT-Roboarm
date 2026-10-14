# Description: O6 手功能验证（对齐抓取流水线），只动手指，机械臂仅做关节1小幅运动。
# 前置：config `jaka_end_effector: hand`。裸链路探测用 tools/jaka_hand_probe.py，
# 本脚本验证的是流水线契约：开度读数、catch() 成败判定、step_callback 数据格式。
# 开度判定指 = 无名指（jaka_ctrl_by_sdk.HAND_OPEN_FINGER）：拇指碰不到掌心
# 方块、食指/中指被全弯的拇指垫住，只有无名指/小指能区分握空/握物；
# 每步都打印六指原始角度供交叉验证。
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arm.jaka_sdk  # noqa: F401  按平台注入 jkrc SDK 路径并预加载依赖库

from arm.arm_base import Arm
from arm.jaka_hand import FINGER_NAMES


def hand_raw_status(arm) -> str:
    """打印 O6 六指原始角度，用于与归一化开度交叉验证（逐个读，约 8s）。"""
    angles = arm.hand.read_angles()
    return (
        f"六指角度={angles}（0=完全弯曲, 255=伸直；"
        f"顺序 {FINGER_NAMES}；None=读失败）"
    )


def main():
    arm: Arm = Arm(debug_mode=False)  # type: ignore
    if getattr(arm, "hand", None) is None:
        print("当前 config 的 jaka_end_effector 不是 hand，改为 hand 后重跑")
        arm.disconnect_arm()
        return
    threshold = arm.default_gripper_close_threshold
    print(f"开度判定阈值 default_gripper_close_threshold = {threshold}")
    print(f"初始开度: {arm.get_arm_angles()[1]}")
    print(f"原始状态: {hand_raw_status(arm)}\n")

    # ---- 1. 张开（对应 catch/place 中的 set_gripper(1)）----
    print("[1] set_gripper(1) 张开")
    arm.set_gripper(1)
    time.sleep(2)
    print(f"    开度读数: {arm.get_arm_angles()[1]}（应≈1）")
    print(f"    原始状态: {hand_raw_status(arm)}")

    # ---- 2. 空握（对应抓空，catch() 应判定"夹取失败"）----
    print("\n[2] set_gripper(0) 空握（抓空 → catch() 应判『夹取失败』）")
    arm.set_gripper(0)
    time.sleep(2.5)
    closed_air = arm.get_arm_angles()[1]
    verdict = "夹取失败(判定正确)" if closed_air < threshold else "误判为成功!"
    print(f"    开度读数: {closed_air}（应 < {threshold}）→ catch() 判定: {verdict}")
    print(f"    原始状态: {hand_raw_status(arm)}")

    # ---- 3. 夹物体（对应抓到物体，catch() 应判定"夹取成功"）----
    arm.set_gripper(1)  # 先张开，便于放入物体
    time.sleep(2)
    input("\n[3] 请把测试物体放到手掌与手指之间（选手指挡得住的方块），"
          "然后按回车开始夹取...")
    arm.set_gripper(0)
    arm.wait_gripper_gripped()  # catch() 的实际等待路径（角度稳定判定）
    closed_obj = arm.get_arm_angles()[1]
    verdict = (
        "夹取成功(判定正确)"
        if closed_obj is not None and closed_obj >= threshold
        else "误判为失败!"
    )
    print(f"    开度读数: {closed_obj}（应 >= {threshold}）→ catch() 判定: {verdict}")
    print(f"    原始状态: {hand_raw_status(arm)}")

    # ---- 4. 松开放下物体 ----
    input("\n[4] 按回车松开物体...")
    arm.set_gripper(1)
    time.sleep(2)
    opened = arm.get_arm_angles()[1]
    print(f"    开度读数: {opened}（应≈1）")
    if opened is not None and opened < 0.5:
        # 动作后 2~5s 存在读数假 0 窗口（执行动作期间应答迟滞，见 JakaHand
        # 头注释），补读一次自证：过窗后即恢复真实值
        print("    疑似动作后假 0 窗口，3s 后补读...")
        time.sleep(3)
        print(f"    开度读数: {arm.get_arm_angles()[1]}（应≈1）")
    print(f"    原始状态: {hand_raw_status(arm)}")

    # ---- 5. 关节+手联动 + step_callback 契约（VLA 录制数据格式）----
    print("\n[5] set_arm_angles 关节+末端联动（关节1 小幅 +10°）+ step_callback 契约")
    samples = []

    def on_step(step: dict) -> None:
        if not samples:
            samples.append(step)

    angles = list(arm.get_arm_angles()[0])
    angles[0] += 10
    ok = arm.set_arm_angles(angles, gripper_open_0to1=1, step_callback=on_step)
    print(f"    set_arm_angles 返回: {ok}")
    if samples:
        sent = samples[0].get("sent_action", {})
        expected_keys = {f"joint_{i}.pos" for i in range(1, 7)} | {"gripper.pos"}
        print(f"    step_callback 字段: {sorted(samples[0].keys())}")
        print(f"    sent_action 键与 Piper 契约一致: {set(sent.keys()) == expected_keys}")
        print(f"    gripper.pos = {sent.get('gripper.pos')}（应为 0-100）")
    else:
        print("    Warning: 未收到 step_callback!")
    time.sleep(1)

    # ---- 6. 失能/使能（O6 手随臂上电，无独立使能面）----
    print("\n[6] disable_torque / enable_torque")
    arm.disable_torque()
    time.sleep(1)
    arm.enable_torque()
    time.sleep(1)
    print(f"    完成后开度: {arm.get_arm_angles()[1]}")

    # ---- 7. 复位断开（home 时张开手，对应 disconnect_arm 行为）----
    print("\n[7] 复位断开（move_to_home 会张开手）")
    arm.disconnect_arm()


if __name__ == "__main__":
    main()
