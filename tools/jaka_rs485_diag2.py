# Description: 精确复现 JakaBySDK 的初始化流程（上电前加信号量 → power_on 删了重加
# → enable_robot → set_target_force → 行程端点 _read_tmp 折腾），每一步后打印原始
# 信号量表。请全程观察夹爪物理动作（每次 release/grip 后夹爪是否真的动）。
# 最后追加寄存器表探测（3/4/5/6/7/9 + position@8）。
# 不实例化 JakaGripper（避免其 __init__ 再次配置通道/清空/重加），读写全部自包含。
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arm.jaka_sdk  # noqa: F401
import jkrc

IP = "10.5.5.100"
CHN = 1
SLAVE = 1

# 寄存器地址（与 jaka_gripper.py REG 一致，仅取用到的）
ADDR = {
    "state": 0x0006, "hardness": 0x0007, "position": 0x0008, "current": 0x0009,
    "adaptive_speed": 0x000A, "action": 0x000B, "target_force": 0x0012,
    "open_position": 0x002D, "close_position": 0x002E,
}
CORE = ["state", "position", "current", "hardness", "action", "adaptive_speed"]


def crc16_modbus(payload: bytes) -> bytes:
    value = 0xFFFF
    for byte in payload:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ 0xA001 if value & 1 else value >> 1
    return value.to_bytes(2, "little")


def send(payload: bytes):
    frame = payload + crc16_modbus(payload)
    return robot.send_tio_rs_command(CHN, bytearray(frame))


def write_reg(addr: int, value: int):
    p = bytes([SLAVE, 0x06]) + addr.to_bytes(2, "big") + value.to_bytes(2, "big")
    return send(p)


def write_regs(addr: int, values: list):
    data = b"".join(v.to_bytes(2, "big") for v in values)
    p = (bytes([SLAVE, 0x10]) + addr.to_bytes(2, "big")
         + len(values).to_bytes(2, "big") + bytes([len(data)]) + data)
    return send(p)


def release():
    return write_regs(ADDR["adaptive_speed"], [1000, 0])


def grip():
    return write_regs(ADDR["adaptive_speed"], [1000, 1])


def raw_info():
    info = robot.get_rs485_signal_info()
    print(f"  get_rs485_signal_info -> err={info[0] if info else None}"
          f" n={len(info[1]) if info and info[0] == 0 else '-'}")
    if info and info[0] == 0:
        for s in info[1]:
            print(f"    {s.get('sig_name')!r} @0x{s.get('sig_addr'):04X} = {s.get('value')}")
    return info


def read_core(name):
    info = robot.get_rs485_signal_info()
    if info and info[0] == 0:
        for s in info[1]:
            if s.get("sig_name") == name:
                return s.get("value")
    return None


def add(name, addr, sig_type=3, chn=CHN):
    ret = robot.add_tio_rs_signal({
        "sig_name": name, "chn_id": chn, "sig_type": sig_type,
        "sig_addr": addr, "value": 0, "frequency": 5})
    print(f"  add {name} @0x{addr:04X} -> {ret}")
    return ret


def clear_all():
    info = robot.get_rs485_signal_info()
    if info and info[0] == 0:
        for s in info[1]:
            ret = robot.del_tio_rs_signal(s["sig_name"])
            print(f"  del {s['sig_name']!r} -> {ret}")


def config_channel():
    print("  set_tio_pin_mode(2,1) ->", robot.set_tio_pin_mode(2, 1))
    print("  set_rs485_chn_mode(1,0) ->", robot.set_rs485_chn_mode(CHN, 0))
    print("  set_rs485_chn_comm ->", robot.set_rs485_chn_comm({
        "chn_id": CHN, "slave_id": SLAVE, "baudrate": 115200,
        "databit": 8, "stopbit": 1, "parity": 78}))


def release_grip_cycle(tag):
    print(f"  === {tag}: release（请观察夹爪是否张开）===")
    print("  release ->", release())
    time.sleep(2)
    raw_info()
    print(f"  === {tag}: grip（请观察夹爪是否闭合）===")
    print("  grip ->", grip())
    time.sleep(2.5)
    raw_info()


robot = jkrc.RC(IP)
print("login ->", robot.login())

try:
    print("\n[P1] 复现 JakaGripper.__init__：上电前配置通道 + 加核心信号量")
    print("（此时机械臂应处于下电状态，若夹爪有电请先断电再跑）")
    config_channel()
    clear_all()
    for name in CORE:
        add(name, ADDR[name])
    time.sleep(1.2)
    raw_info()

    print("\n[P2] 复现 JakaGripper.power_on：上电 → 重新配置 → 删了重加")
    print("  power_on ->", robot.power_on())
    time.sleep(2)
    config_channel()
    clear_all()
    for name in CORE:
        add(name, ADDR[name])
    time.sleep(1.2)
    raw_info()

    print("\n[P3] 复现 JakaBySDK.__init__ 后续：enable_robot + set_target_force")
    print("  enable_robot ->", robot.enable_robot())
    print("  set_target_force(50) ->", write_reg(ADDR["target_force"], 50))
    time.sleep(1.2)
    raw_info()

    print("\n[P4] 复现行程端点 _read_tmp 折腾（open_position 0x2D / close_position 0x2E）")
    print("  del _tmp ->", robot.del_tio_rs_signal("_tmp"))
    add("_tmp", ADDR["open_position"])
    time.sleep(1.0)
    print("  open_position 读得:", read_core("_tmp"))
    raw_info()
    print("  del _tmp ->", robot.del_tio_rs_signal("_tmp"))
    add("_tmp", ADDR["close_position"])
    time.sleep(1.0)
    print("  close_position 读得:", read_core("_tmp"))
    print("  del _tmp ->", robot.del_tio_rs_signal("_tmp"))
    raw_info()

    print("\n[P5] 正式读序列（对应 jaka_gripper_test [1][2]）")
    release_grip_cycle("P5")

    print("\n[P6] 寄存器表探测：清空后挂 3/4/5/6/7/9 + position@8")
    clear_all()
    add("a3", 0x0003)
    add("a4", 0x0004)
    add("a5", 0x0005)
    add("a6", 0x0006)
    add("a7", 0x0007)
    add("pos8", 0x0008)
    add("a9", 0x0009)
    time.sleep(1.5)
    raw_info()
    release_grip_cycle("P6")

finally:
    print("\n清理断开")
    robot.power_off()
    robot.logout()
