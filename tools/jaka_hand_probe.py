# Description: 灵心巧手 O6 经 JAKA 末端 RS485L 的连通性探测与动作验证。
#
# 链路：PC → jkrc → 控制器（RS485 通道1 = 末端 RS485L，Modbus RTU 主站）
#       → O6 手（右 0x27 / 左 0x28，出厂预期 115200 8N1）。
#
# 读：信号量「单条目 + update_tio_rs_signal 强制轮询」——TG-9801 时期实测本控制器
#     信号量表在表里 ≥2 条时槽位会错位，唯一可靠读法（约 1.3s/寄存器）。
# 写：send_tio_rs_command 透传 Modbus RTU 帧（Raw RS485 模式盲发；
#     返回码 0 仅表示「帧已从控制器发出」，不代表手有应答）。
#
# O6 寄存器（官方 SDK linkerhand-python-sdk / realhand-python-sdk 一致）：
#   输入寄存器(04)：0~5 当前角度(0=弯曲 255=伸直)，6~11 力矩，12~17 速度，
#                   18~23 温度，24~29 错误码，30 自由度(=6)，31 型号，
#                   32~46 编号/方向(82='R'/76='L')/软硬件版本
#   保持寄存器(16)：0~5 目标角度，6~11 力矩，12~17 速度（值 0~255）
#
# 接线（2026-10-14 实测）：手的 A/B 接末端 RS485L 一对（棕/白），极性与直觉相反——
#   按「A→棕(A+)、B→白(B−)」接全无应答，对调后立即通。参数=厂商默认 115200/0x27 无需改。
#
# 用法（项目根目录，激活 venv 后）：
#   python tools/jaka_hand_probe.py            # 读探测（不动手指）
#   python tools/jaka_hand_probe.py --sweep    # 读探测不到时扫 JAKA 支持的波特率×两个从站
#   python tools/jaka_hand_probe.py --wiggle   # 读后写「握拳→张开」并经读回确认（会动手指，有确认提示）
#   python tools/jaka_hand_probe.py --slave 0x27 --baud 115200   # 指定从站/波特率
import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arm.jaka_sdk  # noqa: F401  按平台注入 jkrc SDK 路径并预加载依赖库

import jkrc
from utils.config_getter import get_config_value

CHN = 1               # 末端 RS485L = 通道 1（原 TG-9801 同一路）
DEFAULT_BAUD = 115200
SLAVES = [0x27, 0x28]  # 右手 / 左手
SWEEP_BAUDS = [115200, 9600, 19200, 38400, 57600, 230400]  # JAKA 支持的档位
SIG_INPUT = 4
RAW_MODE = 1

# 读探测核心寄存器：30=自由度（期望 6）、0=拇指角度、35=方向（82='R'/76='L'）
PROBE_REGS = {30: "自由度", 0: "拇指角度", 35: "方向"}
REG_NAMES = {
    0: "拇指弯曲", 1: "拇指开合", 2: "食指", 3: "中指", 4: "无名指", 5: "小指",
    6: "拇指力矩", 7: "拇指开合力矩", 8: "食指力矩", 9: "中指力矩",
    10: "无名指力矩", 11: "小指力矩",
    18: "拇指温", 19: "拇指开合温", 20: "食指温", 21: "中指温",
    22: "无名指温", 23: "小指温",
    24: "拇指错误", 25: "拇指开合错误", 26: "食指错误", 27: "中指错误",
    28: "无名指错误", 29: "小指错误",
    30: "自由度", 31: "型号", 33: "编号中位", 35: "方向",
    36: "硬版本高", 37: "硬版本中", 38: "硬版本低",
    39: "软版本高", 40: "软版本中", 41: "软版本低",
}


def crc16_modbus(payload: bytes) -> bytes:
    """Modbus RTU CRC16，返回低字节在前的线路字节序。"""
    value = 0xFFFF
    for byte in payload:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ 0xA001 if value & 1 else value >> 1
    return value.to_bytes(2, "little")


def clear_signals(robot) -> None:
    info = robot.get_rs485_signal_info()
    if info[0] == 0:
        for s in info[1]:
            try:
                robot.del_tio_rs_signal(s["sig_name"])
            except Exception:
                pass


def read_reg(robot, addr: int, sig_type: int = SIG_INPUT, retries: int = 3):
    """单条目 + update 的可靠读法（与 JakaGripper._read_reg 一致）。

    2026-10-14 实测（A/B 对调后）：O6 经 JAKA 的读有约 5~15% 间歇性丢应答
    （成簇出现，与读的数据都正确的轮次交错），返回 0 时重试；全部失败返回 None。
    """
    for _ in range(max(1, retries)):
        robot.add_tio_rs_signal({
            "sig_name": "_rd", "chn_id": CHN, "sig_type": sig_type,
            "sig_addr": addr, "value": 0, "frequency": 5})
        time.sleep(0.5)
        robot.update_tio_rs_signal({"sig_name": "_rd", "frequency": 5})
        time.sleep(0.6)
        info = robot.get_rs485_signal_info()
        v = None
        if info[0] == 0:
            for s in info[1]:
                if s["sig_name"] == "_rd":
                    v = s["value"]
        robot.del_tio_rs_signal("_rd")
        if v:
            return v
    return None


def config_channel(robot, slave: int, baud: int, mode: int = 0) -> None:
    """配置末端 RS485L：AI 引脚复用 + Modbus RTU 主站(mode=0)/Raw(mode=1)。"""
    robot.set_tio_pin_mode(2, 1)  # AI 引脚复用为 RS485L
    robot.set_rs485_chn_mode(CHN, mode)
    robot.set_rs485_chn_comm({
        "chn_id": CHN, "slave_id": slave, "baudrate": baud,
        "databit": 8, "stopbit": 1, "parity": 78})


def probe_slave(robot, slave: int) -> bool:
    """读探测：核心寄存器全 0 视为无应答。在线的判据=任一核心寄存器非 0。"""
    clear_signals(robot)
    vals = {addr: read_reg(robot, addr) for addr in PROBE_REGS}
    if not any(vals.values()):
        return False
    print(f"    核心寄存器: " + ", ".join(
        f"{name}(reg{addr})={vals[addr]}" for addr, name in PROBE_REGS.items()))
    print("    完整读（逐个读，约 1.3s/个）:")
    angles = [read_reg(robot, i) for i in range(6)]
    print(f"      角度(0~5, 0=弯曲/255=伸直): {angles}")
    for addr in (30, 31, 35, 39, 40, 41):
        print(f"      reg{addr} {REG_NAMES.get(addr, '')}: {read_reg(robot, addr)}")
    return True


def wiggle(robot, slave: int, baud: int) -> None:
    """写「张开→握拳→张开」并经读回确认角度跟随（Raw 透传写 + 主站模式读）。"""
    input("    确认手指活动范围内无障碍物、人守在旁边后按回车开始...")
    clear_signals(robot)

    def read_angles():
        config_channel(robot, slave, baud, mode=0)
        return [read_reg(robot, i) for i in range(6)]

    def w_regs(addr, values):
        config_channel(robot, slave, baud, mode=RAW_MODE)
        data = b"".join(v.to_bytes(2, "big") for v in values)
        p = (bytes([slave, 0x10]) + addr.to_bytes(2, "big")
             + len(values).to_bytes(2, "big") + bytes([len(data)]) + data)
        return robot.send_tio_rs_command(CHN, bytearray(p + crc16_modbus(p)))[0]

    a0 = read_angles()
    print(f"    初始角度 = {a0}")
    print(f"    设速度/力矩 -> {[w_regs(12, [200] * 6), w_regs(6, [200] * 6)]}")
    time.sleep(0.5)
    print(f"    握拳(全0)   -> {w_regs(0, [0] * 6)}")
    time.sleep(2.0)
    a1 = read_angles()
    print(f"    握拳后角度 = {a1}   (期望明显下降)")
    print(f"    张开(全255) -> {w_regs(0, [255] * 6)}")
    time.sleep(2.0)
    a2 = read_angles()
    print(f"    张开后角度 = {a2}   (期望≈255)")
    if a1 and a2 and None not in a1 + a2:
        s1, s2 = sum(a1), sum(a2)
        if s2 - s1 > 300:
            print(f"    ✓ 经 JAKA 闭环通过：握拳合计={s1} → 张开合计={s2}（差 {s2 - s1}）")
        else:
            print(f"    ✗ 角度变化不足：握拳合计={s1} → 张开合计={s2}")
    print("    （写返回码 0 只代表帧已发出——以上角度读回为准）")


def main() -> int:
    parser = argparse.ArgumentParser(description="O6 经 JAKA 末端 RS485L 探测")
    parser.add_argument("--slave", type=lambda x: int(x, 0), default=None,
                        help="从站地址（默认 0x27/0x28 都试）")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="波特率")
    parser.add_argument("--sweep", action="store_true",
                        help="读不到时扫 JAKA 支持的全部波特率")
    parser.add_argument("--wiggle", action="store_true", help="盲写开合动作（会动手指）")
    args = parser.parse_args()

    slaves = [args.slave] if args.slave is not None else SLAVES
    arm_ip = get_config_value("arm_port")
    robot = jkrc.RC(arm_ip)
    ret = robot.login()
    print(f"login({arm_ip}) -> {ret}")
    if ret[0] != 0:
        return 1

    print(f"power_on -> {robot.power_on()}")
    time.sleep(2)

    print("\n[1] 读探测（Modbus RTU 主站 + 单条目读）")
    ok = False
    results = []
    bauds = [args.baud]
    for slave in slaves:
        for baud in bauds:
            print(f"  从站 0x{slave:02X} @ {baud}:")
            config_channel(robot, slave, baud, mode=0)
            if probe_slave(robot, slave):
                ok = True
                results.append((slave, baud))
    if not ok and args.sweep:
        print("\n[1b] 扫描波特率（JAKA 支持档位）")
        for baud in SWEEP_BAUDS:
            for slave in slaves:
                config_channel(robot, slave, baud, mode=0)
                free = read_reg(robot, 30)
                print(f"  从站 0x{slave:02X} @ {baud}: 自由度(reg30)={free}")
                if free:
                    ok = True
                    results.append((slave, baud))

    print("\n[2] 结论")
    if ok:
        for slave, baud in results:
            side = "右" if slave == 0x27 else "左" if slave == 0x28 else "?"
            print(f"  ✓ 有应答：从站 0x{slave:02X}（{side}手）@ {baud}")
    else:
        print("  ✗ 读链路无应答。排查顺序：")
        print("    1) A/B 极性——实测接末端棕/白时需与直觉相反（对调一次即通）")
        print("    2) 手的真实波特率/从站地址——先用 USB-RS485 直连确认（JAKA 最高只支持 230400）")
        print("    3) 手供电与协议模式（RS485 版才有 Modbus；上电应有自检动作）")

    if args.wiggle:
        print("\n[3] 盲写动作验证")
        for slave, baud in (results or [(slaves[0], args.baud)]):
            print(f"  从站 0x{slave:02X} @ {baud}:")
            wiggle(robot, slave, baud)

    robot.logout()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
