# Description: TG-9801 经 JAKA 的读链路诊断：通道配置 → 逐寄存器读 → 物理跟踪验证。
# 只会开合夹爪（[3] 前有确认提示），不动机械臂。
#
# 背景（2026-10-08 实测）：控制器信号量表在「表里 ≥2 条」时槽位错位——值张冠李戴、
# 间歇全 0（用「8 槽全指同一地址」实验坐实），与 SDK 版本无关（Windows/Linux、
# V2.2.7/V2.3.1beta3 表现一致）。唯一可靠读法是「表里只放 1 条 +
# update_tio_rs_signal 强制轮询」，驱动 JakaGripper._read_reg 已按此法实现。
# 本工具验证这条读链路；判据：release/grip 后 position 差 ≥800 才算健康。
#
# 用法：激活 venv 后在项目根目录运行 `python tools/jaka_gripper_diagnose.py`。
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arm.jaka_sdk  # noqa: F401  导入即注入 jkrc 的 sys.path
from arm.jaka_gripper import JakaGripper


def main() -> int:
    g = JakaGripper()  # 独立连接，自建 login
    try:
        g.power_on()  # 上电 + 重配通道 + 就绪检测（内部等 current 非 0）

        if not g.diagnose():  # [1] 通道配置 + [2] 逐寄存器读 + 结论
            print("\n读链路未通过，物理跟踪跳过。可先重跑本脚本一次"
                  "（「第一次跑夹爪没起来」是已知常见现象）。")
            return 1

        print("\n[3] 物理跟踪验证（开合夹爪，不动机械臂）")
        input("    确认夹爪周围无障碍物后按回车...")
        print(f"    release 前 position = {g.read_position()}")
        print(f"    release -> {g.release()}")
        time.sleep(2.5)
        pos_open = g.read_position()
        print(f"    release 后 position = {pos_open}  (应≈0)")
        print(f"    grip    -> {g.grip()}")
        time.sleep(2.5)
        pos_close = g.read_position()
        print(f"    grip 后    position = {pos_close}  (应≈1000)")
        g.release()  # 收尾张开，方便取放物体

        if pos_open is None or pos_close is None:
            print("\n✗ 跟踪过程中读失败：重跑一次；仍失败按读链路不通排查。")
            return 1
        delta = abs(pos_close - pos_open)
        if delta >= 800:
            print(f"\n✓ 读链路健康（开合 position 差 {delta} ≥ 800）")
            return 0
        print(f"\n✗ 异常：开合 position 差仅 {delta}（应 ≥800）——"
              "夹爪可能未动作或读值不可信，检查供电/接线后重试。")
        return 1
    finally:
        g.close()  # 不 power_off：避免下一轮夹爪掉电不恢复（2026-09-24 教训）


if __name__ == "__main__":
    sys.exit(main())
