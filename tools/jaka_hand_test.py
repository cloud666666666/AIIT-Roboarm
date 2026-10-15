# Description: O6 手功能验证（对齐抓取流水线），只动手指，机械臂仅做关节1小幅运动。
# 前置：config `jaka_end_effector: hand`。裸链路探测用 tools/jaka_hand_probe.py，
# 本脚本验证的是流水线契约：开度读数、catch() 成败判定、step_callback 数据格式。
# 开度/判定 = 拇/食/中/环四指角度均值（jaka_ctrl_by_sdk.HAND_GRASP_FINGERS），
# 抓取姿势为五指全闭合（拇/食/中/无名/小指下 0 合到底或被物体挡住即停，
# 拇指对掌位保持）；判定判据沿用 grasp 流程的物理公式、成功窗 [0.40, 0.80]，
# 全闭姿势的真机读数用 tools/jaka_hand_probe.py --grasp 姿势扫描标定后再核对
# GRASP_OK_MIN/MAX；catch() 二态取窗下界；脚本下按当前 arm 实例的判据逐项核对，
# 每步都打印六指原始角度供交叉验证。
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arm.jaka_sdk  # noqa: F401  按平台注入 jkrc SDK 路径并预加载依赖库

from arm.arm_base import Arm
from arm.jaka_hand import FINGER_NAMES


def hand_raw_status(arm) -> str:
    """打印 O6 六指原始角度，用于与归一化开度交叉验证（逐个读，约 2.3s）。"""
    angles = arm.hand.read_angles()
    return (
        f"六指角度={angles}（0=完全弯曲, 255=伸直；"
        f"顺序 {FINGER_NAMES}；None=读失败）"
    )


def verdict_of(arm, open_0to1) -> str:
    """按当前臂实例的配置给出读数归类 + catch() 二态实判（< 阈值即失败）。"""
    if open_0to1 is None:
        return "判失败（读数失败）"
    lo, hi = arm.GRASP_OK_MIN, arm.GRASP_OK_MAX
    if open_0to1 < lo:
        side = f"低于成功窗(={lo}) → 合到底/无物阻挡，catch 判失败（正确）"
    elif open_0to1 > hi:
        side = f"高于成功窗(={hi}) → 仍在张开/指令未生效"
    else:
        side = "∈成功窗 → 有物阻挡，catch 判成功（正确）"
    # catch() 实判是二态：< default_gripper_close_threshold(=GRASP_OK_MIN) 判失败
    return (
        f"{side}；catch() 实判"
        f"{'失败' if open_0to1 < arm.default_gripper_close_threshold else '成功'}"
    )


def main():
    arm: Arm = Arm(debug_mode=False)  # type: ignore
    if getattr(arm, "hand", None) is None:
        print("当前 config 的 jaka_end_effector 不是 hand，改为 hand 后重跑")
        arm.disconnect_arm()
        return
    fingers = arm.HAND_GRASP_FINGERS
    lo, hi = arm.GRASP_OK_MIN, arm.GRASP_OK_MAX
    print(f"开度/判定 = 拇食中环四指（下标 {list(fingers)}）均值/255，"
          f"成功窗 [{lo}, {hi}]，catch() 二态阈值 default_gripper_close_threshold "
          f"= {arm.default_gripper_close_threshold}")
    print(f"初始开度: {arm.get_arm_angles()[1]}")
    print(f"原始状态: {hand_raw_status(arm)}\n")

    # ---- 1. 张开（对应 catch/place 中的 set_gripper(1)）----
    print("[1] set_gripper(1) 张开")
    arm.set_gripper(1)
    time.sleep(2)
    print(f"    开度读数: {arm.get_arm_angles()[1]}（伸出指≈1，拇指对掌位偏低）")
    print(f"    原始状态: {hand_raw_status(arm)}")

    # ---- 2. 空握（对应抓空，catch() 应判定"夹取失败"）----
    print("\n[2] set_gripper(0) 空握（全闭合、无物阻挡 → catch() 应判『夹取失败』）")
    arm.set_gripper(0)
    time.sleep(2.5)
    closed_air = arm.get_arm_angles()[1]
    print(f"    开度读数: {closed_air}（合到底应 < {arm.default_gripper_close_threshold}）"
          f"→ catch() {verdict_of(arm, closed_air)}")
    print(f"    原始状态: {hand_raw_status(arm)}")

    # ---- 3. 夹物体（对应抓到物体，catch() 应判定"夹取成功"）----
    arm.set_gripper(1)  # 先张开，便于放入物体
    time.sleep(2)
    input("\n[3] 请把测试物（约 5cm 方块，垫在拇指与食/中/无名指之间）放到手掌与"
          "手指之间，然后按回车开始夹取...")
    arm.set_gripper(0)
    arm.wait_gripper_gripped()  # catch() 的实际等待路径（读数稳定判定）
    closed_obj = arm.get_arm_angles()[1]
    print(f"    开度读数: {closed_obj}（应在成功窗 [{lo}, {hi}]，否则按 probe "
          f"--grasp 实测重标 GRASP_OK_MIN/MAX）→ catch() {verdict_of(arm, closed_obj)}")
    print(f"    原始状态: {hand_raw_status(arm)}")

    # ---- 4. 松开放下物体 ----
    input("\n[4] 按回车松开物体...")
    arm.set_gripper(1)
    time.sleep(2)
    opened = arm.get_arm_angles()[1]
    print(f"    开度读数: {opened}（张开姿势均值约 0.75：拇指弯曲 0.9、拇指开合 0.2、"
          f"食/中 0.9）")
    if opened is not None and opened < 0.3:
        # 动作后 2~5s 存在读数假 0 窗口（执行动作期间应答迟滞，见 JakaHand
        # 头注释），补读一次自证：过窗后即恢复真实值
        print("    疑似动作后假 0 窗口，3s 后补读...")
        time.sleep(3)
        print(f"    开度读数: {arm.get_arm_angles()[1]}")
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
