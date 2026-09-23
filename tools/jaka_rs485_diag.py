# Description: RS485 读回故障诊断。只上电 + 开合夹爪，不动关节。
# 逐项打印现代码忽略的 SDK 返回码、信号量原始数据，并对照：
#   [1] 现代码寄存器表（state=0x0006 / position=0x0008 / current=0x0009）
#   [2] TG-9801 手册 3.3.3 示例寄存器表（state=0x0003 / position=0x0005）
#   [3] sig_type 取值语义（0 / 3 / 4 各挂一个）
#   [4] 通道模式 0/1/2 扫描 + [5] RS485 引脚复用模式(pin_mode=2) + [6] 通道2
# 定位「写入透传有效、信号量全 0」的具体环节。
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arm.jaka_sdk  # noqa: F401
import jkrc
from arm.jaka_gripper import JakaGripper

IP = "10.5.5.100"
CHN = 1
SLAVE = 1


def add(name, addr, sig_type=3, chn=CHN):
    ret = robot.add_tio_rs_signal({
        "sig_name": name, "chn_id": chn, "sig_type": sig_type,
        "sig_addr": addr, "value": 0, "frequency": 5})
    print(f"  add {name} @0x{addr:04X} type={sig_type} chn={chn} -> {ret}")
    return ret


def clear_all():
    info = robot.get_rs485_signal_info()
    if info and info[0] == 0:
        for s in info[1]:
            try:
                robot.del_tio_rs_signal(s["sig_name"])
            except Exception as e:
                print("  del fail:", e)


def dump():
    info = robot.get_rs485_signal_info()
    if not info or info[0] != 0:
        print("  get_rs485_signal_info ->", info)
        return
    print(f"  signals({len(info[1])}):")
    for s in info[1]:
        print(f"    {s.get('sig_name')} @0x{s.get('sig_addr'):04X} = {s.get('value')}")


def grip_release_cycle():
    for fn, name in ((g.release, "release(张开)"), (g.grip, "grip(闭合)")):
        time.sleep(1.5)
        print(f"  {name} -> {fn()}")
        time.sleep(2)
        dump()


robot = jkrc.RC(IP)
print("login ->", robot.login())
print("get_sdk_version ->", robot.get_sdk_version())

try:
    print("power_on ->", robot.power_on())
    time.sleep(2)

    print("\n[0] 各配置调用返回码（现代码全部忽略）")
    print("  set_tio_pin_mode(2,1) ->", robot.set_tio_pin_mode(2, 1))
    print("  set_rs485_chn_mode(1,0) ->", robot.set_rs485_chn_mode(CHN, 0))
    print("  set_rs485_chn_comm ->", robot.set_rs485_chn_comm({
        "chn_id": CHN, "slave_id": SLAVE, "baudrate": 115200,
        "databit": 8, "stopbit": 1, "parity": 78}))
    try:
        for p in range(6):
            print(f"  get_tio_pin_mode({p}) ->", robot.get_tio_pin_mode(p))
    except Exception as e:
        print("  get_tio_pin_mode 不可用:", e)

    g = JakaGripper(robot=robot)  # 复用现代码的通道配置 + 核心信号量
    time.sleep(1.5)

    print("\n[1] 现代码寄存器表")
    dump()
    grip_release_cycle()

    print("\n[2] 追加手册示例寄存器表")
    add("m_state", 0x0003)
    add("m_hard", 0x0004)
    add("m_pos", 0x0005)
    add("m_cur", 0x0006)
    time.sleep(1.5)
    dump()
    grip_release_cycle()

    print("\n[3] sig_type 语义对照（position 0x0008）")
    clear_all()
    add("p_t0", 0x0008, sig_type=0)
    add("p_t3", 0x0008, sig_type=3)
    add("p_t4", 0x0008, sig_type=4)
    time.sleep(1.5)
    dump()
    grip_release_cycle()

    print("\n[4] 通道模式扫描（0/1/2）")
    for mode in (0, 1, 2):
        clear_all()
        print(f"  set_rs485_chn_mode(1,{mode}) ->", robot.set_rs485_chn_mode(CHN, mode))
        for name, addr in (("state", 0x0006), ("position", 0x0008), ("current", 0x0009)):
            add(name, addr)
        time.sleep(2)
        dump()
        grip_release_cycle()

    print("\n[5] RS485 引脚复用模式（DO1/DO2 pin_mode=2）+ 通道模式 1")
    clear_all()
    print("  set_tio_pin_mode(2,2) ->", robot.set_tio_pin_mode(2, 2))
    print("  set_tio_pin_mode(3,2) ->", robot.set_tio_pin_mode(3, 2))
    print("  set_rs485_chn_mode(1,1) ->", robot.set_rs485_chn_mode(CHN, 1))
    for name, addr in (("state", 0x0006), ("position", 0x0008), ("current", 0x0009)):
        add(name, addr)
    time.sleep(2)
    dump()
    grip_release_cycle()

    print("\n[6] 通道 2 探测（若夹爪实际接 RS485-2/AIN 引脚）")
    clear_all()
    print("  set_rs485_chn_mode(2,1) ->", robot.set_rs485_chn_mode(2, 1))
    for name, addr in (("state", 0x0006), ("position", 0x0008), ("current", 0x0009)):
        add(name, addr, chn=2)
    time.sleep(2)
    dump()
    grip_release_cycle()

finally:
    print("\n恢复配置并断开")
    robot.set_rs485_chn_mode(CHN, 0)
    robot.set_tio_pin_mode(2, 1)
    robot.power_off()
    robot.logout()
