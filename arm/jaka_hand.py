# -*- coding: utf-8 -*-
"""
JakaHand：经 JAKA 末端 RS485L 控制灵心巧手 O6 手（独立 SDK，仅依赖 jkrc）

链路：PC → jkrc → 控制器（RS485 通道 1 = 末端 RS485L）→ O6 手（右 0x27 / 左 0x28）。
读写都在 Modbus RTU 主站模式下：读=信号量轮询，写=send_tio_rs_command 透传
Modbus 帧（与 TG-9801 驱动同一条路，2026-10-14 真机验证；不必切 Raw 模式）。

O6 寄存器（官方 linkerhand-python-sdk 一致）：
  · 输入寄存器(04)：0~5 当前角度(0=完全弯曲, 255=伸直)，6~11 力矩，12~17 速度，
                    18~23 温度，24~29 错误码，30 自由度(=6)，31 型号，
                    32~46 编号/方向(82='R'/76='L')/软硬件版本
  · 保持寄存器(16)：0~5 目标角度，6~11 力矩，12~17 速度（值 0~255）

读策略：控制器信号量表在「表里 ≥2 条」时槽位会错位（2026-10-08 实测于 TG-9801，
见 arm/jaka_gripper.py 头注释），唯一可靠读法是「表里只放 1 条 +
update_tio_rs_signal 强制轮询」，单次约 1.3s。O6 经本链路读另有约 5~15% 概率
丢应答（2026-10-14 实测），丢应答时表里回落到初值 0——角度 0（完全弯曲）是
合法值，与丢应答无法从单次读数区分：
  · 不可能为 0 的寄存器（自由度/型号/方向）按 zero_is_invalid 判失败并重读；
  · 角度这类 0 为合法值的寄存器用 read_angle(confirm_zero=True) 对 0 多读一次
    确认（确认真实 0 的代价是多花约 1.3s）。
读后即删条目：多条目表会错位，且轮询流量会与写指令抢 RS485 半双工总线导致写丢帧。
读数为 0 有两种成因，处置不同（2026-10-14 实测）：
  · 静止时的丢应答：成簇出现（连读可出现十几秒的全 0），长等待不能解决——
    值在 update 后约 0.01s 就进表，丢应答是「手没应答」而非「等得不够」；
  · 动作后约 2~5s 的应答迟滞：手在执行开合动作期间应答变慢/丢失，0.6s 的
    表内等待收不到值，表里回落为初值 0。例：握住方块时 grip 后轮询头 3 次
    读 0（手指被方块挡在 ~98，物理上到不了 0）；open 后 ~4.6s 内读 0 而
    数秒后读到真实 254。这类假 0 过窗后自然消失，判定类读数须在动作稳定后
    进行（wait_gripper_gripped 的轮询节奏天然规避，catch 判定在其后）。

写策略：Modbus 主站模式下透传（tools/jaka_hand_probe.py 的 Raw 路径等价）。
返回码 0 仅表示「帧已从控制器发出」，不代表手有应答；物理动作以读回为准。
每次动作必须按「速度 → 力矩 → 目标角度」三帧一起写：手在动作结束后会把速度
寄存器清 0（2026-10-14 实测），只写角度时手指会慢慢挪/停在半路。三帧还不能
零间隔连发：紧接握拳（堵转握持）后的 open 三连发实测手指完全不动（2026-10-14
实测，堵转时手的应答变慢，连发帧与应答在半双工总线上相撞即丢），_set_motion
按 probe 验证过的节奏在帧间留 0.1/0.5s。读回闭环验证见 tools/jaka_hand_test.py。

接线（2026-10-14 实测）：手的 A/B 接末端 RS485L 的棕/白一对，极性与直觉相反
（按 A→棕、B→白 接全无应答，对调即通）；参数用厂商默认 115200 8N1、
从站 0x27（右）/0x28（左），无需修改。

连接模式（与 JakaGripper 一致）：
  · robot=None（默认）：独立模式，自建 jkrc.RC 并 login，close() 时 logout；
  · robot=<RC实例>：共享模式，复用外部已 login 的连接（控制器 SDK 连接数有限，
    与机械臂适配器 JakaBySDK 共用一条连接时必须用此模式），close() 不 logout。
"""
import time

import jkrc


def crc16_modbus(payload: bytes) -> bytes:
    """Modbus RTU CRC16，返回低字节在前的线路字节序。"""
    value = 0xFFFF
    for byte in payload:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ 0xA001 if value & 1 else value >> 1
    return value.to_bytes(2, byteorder="little")


# O6 手 Modbus 寄存器地址（官方 SDK 与 tools/jaka_hand_probe.py 一致）
REG_ANGLE = 0       # 0~5  当前角度(输入)/目标角度(保持)
REG_TORQUE = 6      # 6~11 力矩
REG_SPEED = 12      # 12~17 速度
REG_TEMP = 18       # 18~23 温度
REG_ERROR = 24      # 24~29 错误码
REG_DOF = 30        # 自由度（固定 6）
REG_MODEL = 31      # 型号
REG_DIRECTION = 35  # 方向 82='R'(右手) / 76='L'(左手)
REG_SW_VERSION = 39 # 39~41 软件版本 高/中/低

FINGER_COUNT = 6
FINGER_NAMES = ("拇指弯曲", "拇指开合", "食指", "中指", "无名指", "小指")

OPEN_ANGLE = 255    # 五指伸直（张开）
CLOSE_ANGLE = 0     # 五指全弯（握拳）

SIG_HOLDING = 3     # 保持寄存器(03功能码)
SIG_INPUT = 4       # 输入寄存器(04功能码)

MODE_MODBUS = 0     # Modbus RTU 主站（读写共用；Raw 透传模式本驱动不用）


class JakaHand:
    # 单条目读的等待时间：注册后等控制器写入表、update 触发轮询后等应答写回。
    # 实测 (0.4,0.5) 12 次中 1 次失败、(0.6,0.6) 12/12 成功，取保守值。
    _READ_WAIT_ADD = 0.5
    _READ_WAIT_UPDATE = 0.6
    # 单次读失败（表里无条目/零值判失败）时的重读次数，含首次共 3 次
    _READ_ATTEMPTS = 3

    # 开合动作默认速度/力矩（0~255，probe 真机验证值）
    DEFAULT_SPEED = 200
    DEFAULT_TORQUE = 200

    def __init__(self, robot=None, ip="10.5.5.100", chn=1, slave_id=0x27,
                 baudrate=115200):
        self.chn = chn
        self.slave_id = slave_id
        self.baudrate = baudrate
        self._shared_robot = robot is not None
        if self._shared_robot:
            self.robot = robot
        else:
            self.robot = jkrc.RC(ip)
            ret = self.robot.login()
            if ret[0] != 0:
                raise ConnectionError(f"login 失败: {ret}")
        self._config_channel()
        self._clear_signals()  # 清掉历史遗留条目，避免脏表干扰单条目读

    def _config_channel(self) -> None:
        """配置末端 RS485L 为 Modbus RTU 主站模式（读写都用它）。

        上电周期会复位通道配置，因此每次 power_on 后必须重新调用；
        未重新配置时信号量轮询不到值（全 0）、透传写入无效。
        """
        self.robot.set_tio_pin_mode(2, 1)          # AI 引脚复用为 RS485L
        self.robot.set_rs485_chn_mode(self.chn, MODE_MODBUS)
        self.robot.set_rs485_chn_comm({
            'chn_id': self.chn, 'slave_id': self.slave_id,
            'baudrate': self.baudrate,
            'databit': 8, 'stopbit': 1, 'parity': 78})

    def close(self) -> None:
        if not self._shared_robot:
            self.robot.logout()

    def power_on(self) -> None:
        """机械臂上电（末端 RS485 接口与手供电依赖上电才通信）。"""
        ret = self.robot.power_on()
        if ret[0] != 0:
            # 实测出现过「第一次跑末端整个没起来、重跑一次就好」；上电失败时
            # 后续读数会全 0，若不打印返回码，故障会被当成接线/波特率问题。
            print(f"Warning: power_on 返回错误码 {ret[0]}（检查急停/安全门/控制器状态）")
        time.sleep(2)
        # 上电会复位 RS485 通道配置，必须先重新配置才能通信
        self._config_channel()
        self._clear_signals()
        # 等手就绪：读自由度(reg30)应为 6。角度 0（握到底）也是合法值，不能作
        # 就绪判据。本次上电会给手断电重启，实测重启后头几次读全失败、约
        # 10~20s 后才应答，所以外层循环给足 15 次（单次读约 1.3s）。
        for _ in range(15):
            if self._read_reg(REG_DOF, attempts=1, zero_is_invalid=True):
                return
        print("Warning: 手未就绪（自由度一直读不到 6），"
              "请检查 RS485 接线（A/B 极性与直觉相反）与通道配置")

    def power_off(self) -> None:
        """机械臂下电。"""
        self.robot.power_off()

    # ---------- 读：信号量 ----------
    def _clear_signals(self) -> None:
        info = self.robot.get_rs485_signal_info()
        if info[0] == 0:
            for s in info[1]:
                try:
                    self.robot.del_tio_rs_signal(s["sig_name"])
                except Exception:
                    pass

    def _add_signal(self, name: str, addr: int, sig_type: int = SIG_INPUT) -> None:
        self.robot.add_tio_rs_signal({
            'sig_name': name, 'chn_id': self.chn, 'sig_type': sig_type,
            'sig_addr': addr, 'value': 0, 'frequency': 5})

    def _read_once(self, addr: int, sig_type: int = SIG_INPUT):
        """一轮单条目读：加条目 → 等写入 → update 触发轮询 → 读表 → 删条目。"""
        self._add_signal("_rd", addr, sig_type)
        time.sleep(self._READ_WAIT_ADD)
        self.robot.update_tio_rs_signal({"sig_name": "_rd", "frequency": 5})
        time.sleep(self._READ_WAIT_UPDATE)
        info = self.robot.get_rs485_signal_info()
        value = None
        if info[0] == 0:
            for s in info[1]:
                if s["sig_name"] == "_rd":
                    value = s["value"]
        try:
            self.robot.del_tio_rs_signal("_rd")
        except Exception:
            pass
        return value

    def _read_reg(self, addr: int, sig_type: int = SIG_INPUT,
                  attempts: int = None, zero_is_invalid: bool = False):
        """读单个寄存器（16 位无符号），失败返回 None。

        丢应答时表里回落到初值 0：zero_is_invalid=True 的寄存器（自由度/型号/
        方向这类不可能为 0 的）按失败重读 attempts 次；角度等 0 为合法值的
        寄存器用默认 False，0 直接返回（与丢应答无法区分，见 read_angle）。
        """
        for _ in range(max(1, attempts or self._READ_ATTEMPTS)):
            value = self._read_once(addr, sig_type)
            if value is None:
                continue  # 表里没有条目（链路/控制器异常），重读
            if value or not zero_is_invalid:
                return value
        return None

    def _read_series(self, reg: int, count: int = FINGER_COUNT) -> list:
        """逐个读连续 count 个寄存器（约 1.3s/个），失败位置为 None。"""
        return [self._read_reg(reg + i) for i in range(count)]

    def read_angle(self, finger: int, confirm_zero: bool = False):
        """读单指当前角度（0=完全弯曲 ~ 255=伸直），失败返回 None。

        丢应答与真实的 0 无法从单次读数区分；confirm_zero=True 时对 0 再读一次
        确认（confirm_zero 时也要防丢应答，读到非 0 以第二次为准），用于抓取
        成败判定这类 0 会误导判断的地方（代价：真实 0 多花约 1.3s）。
        """
        if not 0 <= finger < FINGER_COUNT:
            raise ValueError(f"finger 需在 0~{FINGER_COUNT - 1}，实际 {finger}")
        reg = REG_ANGLE + finger
        value = self._read_reg(reg)
        if confirm_zero and value == 0:
            again = self._read_reg(reg)
            if again:
                return again
        return value

    def read_angles(self) -> list:
        """读 6 指当前角度（逐个读，约 1.3s/指），失败位置为 None。"""
        return self._read_series(REG_ANGLE)

    def read_torques(self) -> list:
        """读 6 指力矩。"""
        return self._read_series(REG_TORQUE)

    def read_speeds(self) -> list:
        """读 6 指速度。"""
        return self._read_series(REG_SPEED)

    def read_temperatures(self) -> list:
        """读 6 指温度。"""
        return self._read_series(REG_TEMP)

    def read_errors(self) -> list:
        """读 6 指错误码。"""
        return self._read_series(REG_ERROR)

    def read_versions(self) -> list:
        """读软件版本 高/中/低（39~41）。"""
        return self._read_series(REG_SW_VERSION, 3)

    def read_dof(self):
        """读自由度（固定 6，不可能为 0，可作在线判据）。"""
        return self._read_reg(REG_DOF, zero_is_invalid=True)

    def read_model(self):
        """读型号。"""
        return self._read_reg(REG_MODEL, zero_is_invalid=True)

    def read_direction(self):
        """读方向：82='R'(右手) / 76='L'(左手)。"""
        return self._read_reg(REG_DIRECTION, zero_is_invalid=True)

    # ---------- 写：Raw 透传 Modbus 帧 ----------
    def _send_raw(self, payload: bytes) -> int:
        """透传一帧 Modbus RTU（盲发）。返回码 0 仅代表帧已从控制器发出。"""
        frame = payload + crc16_modbus(payload)
        return self.robot.send_tio_rs_command(self.chn, bytearray(frame))[0]

    def _write_regs(self, addr: int, values: list) -> int:
        data = b"".join(int(v).to_bytes(2, "big") for v in values)
        p = (bytes([self.slave_id, 0x10]) + addr.to_bytes(2, "big")
             + len(values).to_bytes(2, "big") + bytes([len(data)]) + data)
        return self._send_raw(p)

    @staticmethod
    def _check_values(values, name: str) -> list:
        values = [int(v) for v in values]
        if len(values) != FINGER_COUNT or not all(0 <= v <= 255 for v in values):
            raise ValueError(
                f"{name}需为 {FINGER_COUNT} 个 0~255 的整数，实际 {values}")
        return values

    def set_angles(self, angles) -> int:
        """写 6 指目标角度（0=完全弯曲 ~ 255=伸直）。"""
        return self._write_regs(REG_ANGLE, self._check_values(angles, "角度"))

    def set_torque(self, torques) -> int:
        """写 6 指力矩（0~255）。"""
        return self._write_regs(REG_TORQUE, self._check_values(torques, "力矩"))

    def set_speed(self, speeds) -> int:
        """写 6 指速度（0~255）。"""
        return self._write_regs(REG_SPEED, self._check_values(speeds, "速度"))

    def _set_motion(self, angle: int, speed=None, torque=None) -> int:
        """按「速度 →0.1s→ 力矩 →0.5s→ 目标角度」三帧写入（probe 验证节奏）。

        三帧零间隔连发会丢帧（2026-10-14 实测：握拳堵转后的 open 三连发
        手指完全不动）；写前静置 0.3s，防上一次读的轮询流量残留抢总线。
        """
        speed = self.DEFAULT_SPEED if speed is None else speed
        torque = self.DEFAULT_TORQUE if torque is None else torque
        time.sleep(0.3)
        ret_speed = self.set_speed([speed] * FINGER_COUNT)
        time.sleep(0.1)
        ret_torque = self.set_torque([torque] * FINGER_COUNT)
        time.sleep(0.5)
        ret_angle = self.set_angles([angle] * FINGER_COUNT)
        for name, frame_ret in (
            ("速度", ret_speed),
            ("力矩", ret_torque),
            ("角度", ret_angle),
        ):
            if frame_ret != 0:
                print(f"Warning: {name}帧写入返回错误码 {frame_ret}")
        return ret_angle

    def open(self, speed=None, torque=None) -> int:
        """张开（五指伸直，角度 255）。"""
        return self._set_motion(OPEN_ANGLE, speed, torque)

    def grip(self, speed=None, torque=None) -> int:
        """握拳（角度 0：全弯到位或握到障碍即停，可用于夹取）。"""
        return self._set_motion(CLOSE_ANGLE, speed, torque)

    # ---------- 诊断 ----------
    def diagnose(self) -> bool:
        """诊断读链路：通道配置 → 核心寄存器读（自由度/方向/角度）→ 结论。不动手指。

        判据：自由度读到 6 = 手在线（角度 0 是合法值、不能作在线判据）。
        物理跟踪验证（开合看角度变化）跑 tools/jaka_hand_probe.py --wiggle
        或 tools/jaka_hand_test.py。
        """
        print("=== JakaHand 诊断 ===")

        # 1. 通道配置各步返回码（0 为成功）与生效后的模式
        print("[1] 通道配置（返回码应全 0）")
        r1 = self.robot.set_tio_pin_mode(2, 1)
        r2 = self.robot.set_rs485_chn_mode(self.chn, MODE_MODBUS)
        r3 = self.robot.set_rs485_chn_comm({
            'chn_id': self.chn, 'slave_id': self.slave_id,
            'baudrate': self.baudrate,
            'databit': 8, 'stopbit': 1, 'parity': 78})
        r4 = self.robot.get_rs485_chn_mode(self.chn)
        print(f"    set_tio_pin_mode(2,1)      -> {r1}")
        print(f"    set_rs485_chn_mode({self.chn},0)   -> {r2}")
        print(f"    set_rs485_chn_comm(...)    -> {r3}")
        print(f"    get_rs485_chn_mode({self.chn})     -> {r4}  (0=Modbus RTU)")

        # 2. 核心寄存器读（单条目 + update，约 1.3s/个）
        print("[2] 核心寄存器读（单条目 + update_tio_rs_signal，约 1.3s/个）")
        dof = self.read_dof()
        direction = self.read_direction()
        model = self.read_model()
        angles = self.read_angles()
        print(f"    自由度 = {dof}   (期望 6)")
        print(f"    方向   = {direction}   (82='R' 右手 / 76='L' 左手)")
        print(f"    型号   = {model}")
        print(f"    角度   = {angles}   (0=完全弯曲, 255=伸直；含丢应答的 None)")

        # 3. 结论（判据：自由度=6 = 手在线）
        print("[3] 结论")
        if dof == FINGER_COUNT:
            print("    ✓ 读链路正常（自由度=6，手在线）")
            print("      物理跟踪验证请跑 tools/jaka_hand_probe.py --wiggle")
            print("      或 tools/jaka_hand_test.py")
            return True
        if dof is None and all(v is None for v in angles):
            print("    ✗ 读链路不通：查末端 RS485 接线（A/B 极性与直觉相反）、"
                  "手供电、上电后是否已重配通道")
        elif not any(angles):
            print("    ✗ 角度全 0：可能是握拳状态，也可能未上电/丢应答。"
                  "先 power_on 再重跑。")
        else:
            print("    ⚠ 读数异常：对照上面各项的期望范围；单次读失败可重跑")
        return False
