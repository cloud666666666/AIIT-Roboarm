# Description: 验证「删光→重加」重复执行能否让信号量表收敛到正确对齐。
# 每轮：删光 → 加 6 核心 → release 读 position → grip 读 position，
# 若 position 真实跟踪（闭-开 >= 800）则判定治愈。最多 6 轮。
# 夹爪会反复开合（空中无物体），请保持观察。
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arm.jaka_sdk  # noqa: F401
import jkrc
from arm.jaka_gripper import JakaGripper, REG, CORE

IP = "10.5.5.100"
CHN = 1


def raw(robot):
    info = robot.get_rs485_signal_info()
    print(f"  表: err={info[0] if info else None} "
          f"n={len(info[1]) if info and info[0] == 0 else '-'}")
    if info and info[0] == 0:
        for s in info[1]:
            print(f"    {s.get('sig_name')!r} @0x{s.get('sig_addr'):04X} = {s.get('value')}")


robot = jkrc.RC(IP)
print("login:", robot.login())

try:
    print("power_on:", robot.power_on())
    time.sleep(2)
    print("set_tio_pin_mode(2,1):", robot.set_tio_pin_mode(2, 1))
    print("set_rs485_chn_mode(1,0):", robot.set_rs485_chn_mode(CHN, 0))
    print("set_rs485_chn_comm:", robot.set_rs485_chn_comm({
        "chn_id": CHN, "slave_id": 1, "baudrate": 115200,
        "databit": 8, "stopbit": 1, "parity": 78}))

    def clear_all():
        info = robot.get_rs485_signal_info()
        if info and info[0] == 0:
            for s in info[1]:
                robot.del_tio_rs_signal(s["sig_name"])

    def add_core():
        for name in CORE:
            robot.add_tio_rs_signal({
                "sig_name": name, "chn_id": CHN, "sig_type": 3,
                "sig_addr": REG[name], "value": 0, "frequency": 5})

    def read_at(addr):
        info = robot.get_rs485_signal_info()
        if info and info[0] == 0:
            for s in info[1]:
                if s.get("sig_addr") == addr:
                    return s.get("value")
        return None

    g = JakaGripper(robot=robot)  # 复用其 release/grip 透传写

    healed = False
    for attempt in range(1, 7):
        print(f"\n===== 第 {attempt} 轮 =====")
        clear_all()
        add_core()
        time.sleep(2)
        print("release ->", g.release())
        time.sleep(2)
        p_open = read_at(REG["position"])
        print("grip ->", g.grip(1000))
        time.sleep(2.5)
        p_close = read_at(REG["position"])
        cur = read_at(REG["current"])
        print(f"  position: open={p_open} close={p_close} "
              f"(差 {None if p_close is None or p_open is None else p_close - p_open}) "
              f"current={cur}")
        raw(robot)
        if (p_open is not None and p_close is not None
                and p_close - p_open >= 800 and cur):
            print(f"  ✓ 第 {attempt} 轮治愈！")
            healed = True
            break
        print("  ✗ 未治愈，下一轮")

    print("\n" + ("结论：重复删加可收敛" if healed else "结论：重复删加不收敛"))

    g.release()
    time.sleep(1)
finally:
    robot.power_off()
    robot.logout()
