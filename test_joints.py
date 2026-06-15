#!/usr/bin/env python3
"""
最小化机械臂关节测试脚本
依次测试 Piper 机械臂的 6 个旋转关节 + 夹爪（第7轴）
用于排查各关节是否有硬件/通信问题，不依赖 YOLO。
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm.arm_base import Arm

# ── 各关节测试配置（基于 URDF 限位，单位：度） ──
JOINT_CONFIGS = [
    {"idx": 0, "name": "关节1 (base旋转)",  "limits": [-150, 150], "offsets": [-30, 30]},
    {"idx": 1, "name": "关节2 (肩部)",      "limits": [0, 180],    "offsets": [30]},       # home=0 是下限
    {"idx": 2, "name": "关节3 (肘部)",      "limits": [-170, 0],   "offsets": [-30]},      # home=0 是上限
    {"idx": 3, "name": "关节4 (腕部俯仰)",  "limits": [-100, 100], "offsets": [-30, 30]},
    {"idx": 4, "name": "关节5 (腕部翻转)",  "limits": [-70, 70],   "offsets": [-20, 20]},
    {"idx": 5, "name": "关节6 (末端旋转)",  "limits": [-120, 120], "offsets": [-30, 30]},
]

HOME = [0.0] * 6
WAIT_AFTER_MOVE = 0.8   # 移动后等待时间（秒）
WAIT_AFTER_HOME = 0.5   # 归零后等待时间（秒）
ERROR_THRESHOLD_DEG = 5.0  # 角度误差阈值（度）


def test_joint(arm, cfg) -> bool:
    """测试单个旋转关节，返回是否全部通过。"""
    idx = cfg["idx"]
    all_ok = True

    print(f"\n{'─' * 55}")
    print(f"🔧 {cfg['name']}  限位: [{cfg['limits'][0]}°, {cfg['limits'][1]}°]")
    print(f"{'─' * 55}")

    for offset in cfg["offsets"]:
        target = HOME.copy()
        target[idx] = offset

        print(f"  🎯 目标: joint{idx + 1} = {offset:+.0f}°")
        success = arm.set_arm_angles(target)
        time.sleep(WAIT_AFTER_MOVE)

        angles, _ = arm.get_arm_angles()
        if angles is None:
            print(f"  ❌ 读取关节角度失败!")
            all_ok = False
        else:
            actual = angles[idx]
            err = abs(actual - offset)
            status = "✅" if err < ERROR_THRESHOLD_DEG else "⚠️ 偏差较大"
            print(f"  📏 实际: joint{idx + 1} = {actual:+.1f}°  (误差 {err:.1f}°) {status}")
            if err >= ERROR_THRESHOLD_DEG:
                all_ok = False

        # 回 home
        arm.set_arm_angles(HOME)
        time.sleep(WAIT_AFTER_HOME)

    status = "✅ 通过" if all_ok else "❌ 存在异常"
    print(f"  {status}")
    return all_ok


def main():
    print("=" * 55)
    print("  Piper 机械臂关节测试")
    print("  测试: 6 个旋转关节 + 夹爪（第7轴）")
    print("=" * 55)

    # ── 初始化 ──
    print("\n🔌 连接机械臂...")
    try:
        arm = Arm(debug_mode=False)
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return 1
    print("✅ 连接成功")

    results = {}

    try:
        # ── 归零 ──
        print("\n🏠 归零...")
        arm.move_to_home(gripper_open_0to1=1)
        time.sleep(1.0)
        angles, gripper = arm.get_arm_angles()
        if angles:
            print(f"   关节: {[f'{a:+.1f}°' for a in angles]}")
        print(f"   夹爪: {gripper:.2f}")

        # ── 逐关节测试 ──
        for cfg in JOINT_CONFIGS:
            results[cfg["name"]] = test_joint(arm, cfg)

        # ── 夹爪测试（第7轴） ──
        name = "夹爪（第7轴）"
        print(f"\n{'─' * 55}")
        print(f"🔧 {name}")
        print(f"{'─' * 55}")

        print("  🎯 闭合...")
        arm.set_gripper(gripper_open_0to1=0)
        time.sleep(0.5)
        _, g = arm.get_arm_angles()
        ok_close = g is not None and g < 0.1
        print(f"  📏 夹爪: {g:.2f} {'✅' if ok_close else '⚠️'}")

        print("  🎯 张开...")
        arm.set_gripper(gripper_open_0to1=1)
        time.sleep(0.5)
        _, g = arm.get_arm_angles()
        ok_open = g is not None and g > 0.9
        print(f"  📏 夹爪: {g:.2f} {'✅' if ok_open else '⚠️'}")

        results[name] = ok_close and ok_open
        print(f"  {'✅ 通过' if results[name] else '❌ 存在异常'}")

    finally:
        # ── 收尾 ──
        print(f"\n{'─' * 55}")
        print("🏠 返回归零位...")
        try:
            arm.move_to_home(gripper_open_0to1=1)
            time.sleep(1)
        except Exception:
            pass
        print("🔌 断开连接...")
        try:
            arm.disconnect_arm()
        except Exception:
            pass

    # ── 汇总 ──
    print("\n" + "=" * 55)
    print("  测试汇总")
    print("=" * 55)
    all_pass = True
    for name, ok in results.items():
        flag = "✅" if ok else "❌"
        print(f"  {flag}  {name}")
        if not ok:
            all_pass = False
    print(f"\n  结论: {'🎉 全部通过!' if all_pass else '⚠️ 存在异常关节，请检查!'}")
    print("=" * 55)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
