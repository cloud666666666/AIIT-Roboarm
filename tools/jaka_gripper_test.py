# Description: TG-9801 夹爪功能验证（对齐 Piper 的夹爪使用面），只动夹爪，
# 机械臂仅做关节1小幅运动，不做大幅位姿运动。
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arm.arm_base import Arm

STATE_TEXT = {0: "空闲", 1: "运动中", 2: "夹稳", 3: "打滑"}


def gripper_raw_status(arm) -> str:
    """打印 TG-9801 原始状态，用于与归一化开度交叉验证。"""
    g = arm.gripper
    pos = g.read_position()
    state = g.read_state()
    current = g.read_current()
    return (
        f"position={pos}, state={state}({STATE_TEXT.get(state, '未知')}), "
        f"current={current}mA"
    )


def main():
    arm: Arm = Arm(debug_mode=False)  # type: ignore
    threshold = arm.default_gripper_close_threshold
    print(f"夹爪闭合判定阈值 default_gripper_close_threshold = {threshold}")
    print(f"初始开度: {arm.get_arm_angles()[1]}")
    print(f"原始状态: {gripper_raw_status(arm)}\n")

    # ---- 1. 张开（对应 catch/place 中的 set_gripper(1)）----
    print("[1] set_gripper(1) 张开")
    arm.set_gripper(1)
    time.sleep(2)
    print(f"    开度读数: {arm.get_arm_angles()[1]}（应≈1）")
    print(f"    原始状态: {gripper_raw_status(arm)}")

    # ---- 2. 空气闭合（对应抓空，catch() 应判定"夹取失败"）----
    print("\n[2] set_gripper(0) 空气闭合（抓空 → catch() 应判『夹取失败』）")
    arm.set_gripper(0)
    time.sleep(2.5)
    closed_air = arm.get_arm_angles()[1]
    verdict = "夹取失败(判定正确)" if closed_air < threshold else "误判为成功!"
    print(f"    开度读数: {closed_air}（应 < {threshold}）→ catch() 判定: {verdict}")
    print(f"    原始状态: {gripper_raw_status(arm)}")

    # ---- 3. 夹物体（对应抓到物体，catch() 应判定"夹取成功"）----
    arm.set_gripper(1)  # 先张开，便于放入物体
    time.sleep(1.5)
    input("\n[3] 请把测试物体放到夹爪两指之间，然后按回车开始夹取...")
    arm.set_gripper(0)
    time.sleep(2.5)
    closed_obj = arm.get_arm_angles()[1]
    verdict = (
        "夹取成功(判定正确)"
        if closed_obj is not None and closed_obj >= threshold
        else "误判为失败!"
    )
    print(f"    开度读数: {closed_obj}（应 >= {threshold}）→ catch() 判定: {verdict}")
    print(f"    原始状态: {gripper_raw_status(arm)}（state=2 表示夹稳）")

    # ---- 4. 松开放下物体 ----
    input("\n[4] 按回车松开物体...")
    arm.set_gripper(1)
    time.sleep(2)
    print(f"    开度读数: {arm.get_arm_angles()[1]}（应≈1）")

    # ---- 5. 关节+夹爪联动 + step_callback 契约（VLA 录制数据格式）----
    print("\n[5] set_arm_angles 关节+夹爪联动（关节1 小幅 +10°）+ step_callback 契约")
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

    # ---- 6. 失能/使能（Piper 的 GripperCtrl 使能面，JAKA 夹爪随臂上电）----
    print("\n[6] disable_torque / enable_torque")
    arm.disable_torque()
    time.sleep(1)
    arm.enable_torque()
    time.sleep(1)
    print(f"    完成后开度: {arm.get_arm_angles()[1]}")

    # ---- 7. 复位断开（home 时张开夹爪，对应 disconnect_arm 行为）----
    print("\n[7] 复位断开（move_to_home 会张开夹爪）")
    arm.disconnect_arm()


if __name__ == "__main__":
    main()
