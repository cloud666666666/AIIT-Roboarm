# -*- coding: utf-8 -*-
"""
JakaGripper：经 JAKA 末端 RS485L 控制 HKV TG-9801 夹爪（独立 SDK，仅依赖 jkrc）

实现 hkv_tg9801 SDK 的读 + 运动 + 运行参数 + 维护接口，并补充 SDK 未覆盖的三维力
(Fx/Fy/Fz)读取。仅排除「危险配置」类（永久改配置/可能失联）。
  · 读接口（read_*）    → 信号量（控制器 Modbus RTU 主站周期轮询）
  · 运动接口（grip/release/force_position_control/follow_control）→ send_tio_rs_command 透传
  · 运行参数接口（set_adaptive_speed/set_target_force/set_control_mode/
                  set_force_position_speed/set_force_position_target）→ 透传写
  · 维护指令（calibrate_fingertip/clear_fingertip_zero）→ 透传写（手册推荐定期执行）
  · 三维力（read_force_fx/fy/fz，寄存器 0x0000~0x0002，有符号 int16，SDK 未实现）

排除的「危险配置」接口（永久改配置/可能失联，未实现）：
  set_protocol / set_device_id / set_baudrate / set_eeprom_enabled /
  set_open_position / set_close_position

字节序：手册 3.3.2 称通信为小端序，u32 寄存器(grip_count/baudrate)低字在前。

读策略：核心状态（state/position/current/hardness/action/adaptive_speed）常驻 6 个信号量，
其余低频接口按需临时读（等 1s 让控制器完成首次轮询）。
信号量生命周期铁律（实测）：
  · 上电前注册会污染控制器信号量表（名字腐蚀成 '\\x01'，槽位卡死，不可逆）；
  · 上电周期会把表中既有条目随机打乱（名字腐蚀/地址错乱/槽位冻结）；
  · 表容量上限约 8 条，超出静默丢弃；
  · 控制器固件重建表时有竞态：「删光→重加」随机成功或错乱，因此
    power_on 上电后跑「删光→重加→物理验证」自愈循环，直至 position
    槽位真实跟踪开合；其余时间对表只读不写。

依赖：仅 `import jkrc`（JAKA SDK，jkrc.pyd/jkrc.so 需在 sys.path 可导入）。
      CRC16 已内联，不再依赖 hkv_tg9801 示例 SDK。

连接模式：
  · robot=None（默认）：独立模式，自建 jkrc.RC 并 login，close() 时 logout；
  · robot=<RC实例>：共享模式，复用外部已 login 的连接（控制器 SDK 连接数有限，
    与机械臂适配器 JakaBySDK 共用一条连接时必须用此模式），close() 不 logout。

说明：
  · force_position_control 需先 set_control_mode(1) 切到「力位」模式；
    follow_control 需夹爪处于「跟随」模式。
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


# 夹爪 Modbus 寄存器地址（TG-9801 手册，与 hkv_tg9801/modbus.py 一致）
REG = {
    "force_fx": 0x0000,        # 三维力 Fx(有符号int16)
    "force_fy": 0x0001,        # Fy
    "force_fz": 0x0002,        # Fz
    "state": 0x0006,           # 夹持状态 0空闲/1运动/2夹稳/3打滑
    "hardness": 0x0007,        # 硬度
    "position": 0x0008,        # 位置
    "current": 0x0009,         # 电流(mA)
    "adaptive_speed": 0x000A,  # 自适应速度
    "action": 0x000B,          # 动作 0松开/1夹取
    "protocol": 0x000C,        # 协议 1自定义/2Modbus
    "device_id": 0x000D,       # 从站地址
    "control_mode": 0x000E,    # 控制模式 0自适应/1力位
    "force_pos_speed": 0x000F, # 力位速度
    "force_pos_target": 0x0010,# 力位目标位置
    "fingertip_calibration": 0x0011,  # 指尖力强制校准(维护用)
    "target_force": 0x0012,    # 目标力
    "factory_number": 0x0013,  # 出厂编号(输入寄存器, 6个ASCII)
    "grip_count": 0x0019,      # 夹取次数(u32, 小端序)
    "follow_speed": 0x001B,    # 跟随速度
    "fingertip_zero": 0x001D,  # 指尖力硬件清零(维护用)
    "baudrate": 0x0021,        # 波特率(u32, 小端序)
    "error_flags": 0x0023,     # 错误标志
    "open_position": 0x002D,   # 张开位置
    "close_position": 0x002E,  # 闭合位置
    "force_pos_state": 0x002F, # 力位状态
    "eeprom_enabled": 0x00EE,  # EEPROM开关
}

SIG_HOLDING = 3   # 保持寄存器(03功能码)
SIG_INPUT = 4     # 输入寄存器(04功能码)

CORE = ["state", "position", "current", "hardness", "action", "adaptive_speed"]


class JakaGripper:
    def __init__(self, robot=None, ip="10.5.5.100", chn=1, slave_id=1, baudrate=115200):
        self.chn = chn
        self.slave_id = slave_id
        self.baudrate = baudrate
        self._shared_robot = robot is not None
        self._signals_added = False
        if self._shared_robot:
            self.robot = robot
        else:
            self.robot = jkrc.RC(ip)
            ret = self.robot.login()
            if ret[0] != 0:
                raise ConnectionError(f"login 失败: {ret}")
        self._config_channel()
        # 不碰信号量表：上电前注册会污染表（名字腐蚀成 '\x01'、槽位卡死），
        # 核心信号量统一由 power_on() 上电后按需追加。

    def _config_channel(self) -> None:
        """配置末端 RS485L 通道为 Modbus RTU 主站模式。

        上电周期会复位通道配置，因此每次 power_on 后必须重新调用；
        未重新配置时信号量轮询不到值（全 0）、透传写入无效。
        """
        self.robot.set_tio_pin_mode(2, 1)                # 使能 RS485L
        self.robot.set_rs485_chn_mode(self.chn, 0)       # Modbus RTU 主站模式
        self.robot.set_rs485_chn_comm({
            'chn_id': self.chn, 'slave_id': self.slave_id,
            'baudrate': self.baudrate,
            'databit': 8, 'stopbit': 1, 'parity': 78})

    def close(self) -> None:
        if not self._shared_robot:
            self.robot.logout()

    def power_on(self) -> None:
        """机械臂上电（末端 RS485 接口与夹爪供电依赖上电才通信）。

        上电后执行信号量表自愈验证（见 _heal_signals），验证通过即就绪。
        """
        self.robot.power_on()
        time.sleep(2)
        # 上电会复位 RS485 通道配置，必须先重新配置才能通信
        self._config_channel()
        self._signals_added = True
        if not self._heal_signals():
            print("Warning: 夹爪信号量表多次重建仍无法正确轮询，"
                  "请检查 RS485 接线与通道配置（必要时重启控制器）")

    def power_off(self) -> None:
        """机械臂下电。"""
        self.robot.power_off()

    # ---------- 写：透传 Modbus 帧 ----------
    def _send(self, payload: bytes) -> int:
        frame = payload + crc16_modbus(payload)
        return self.robot.send_tio_rs_command(self.chn, bytearray(frame))[0]

    def _write_reg(self, addr: int, value: int) -> int:
        p = bytes([self.slave_id, 0x06]) + addr.to_bytes(2, "big") + value.to_bytes(2, "big")
        return self._send(p)

    def _write_regs(self, addr: int, values: list) -> int:
        data = b"".join(v.to_bytes(2, "big") for v in values)
        p = (bytes([self.slave_id, 0x10]) + addr.to_bytes(2, "big")
             + len(values).to_bytes(2, "big") + bytes([len(data)]) + data)
        return self._send(p)

    # ---------- 读：信号量 ----------
    def _signals(self) -> list:
        info = self.robot.get_rs485_signal_info()
        return list(info[1]) if info and info[0] == 0 else []

    def _has_slot(self, addr: int, sig_type: int = SIG_HOLDING) -> bool:
        return any(
            s.get("sig_addr") == addr and s.get("sig_type", SIG_HOLDING) == sig_type
            for s in self._signals()
        )

    def _read_at(self, addr: int, sig_type: int = SIG_HOLDING):
        """按地址取轮询值（不按名字——名字可能被控制器腐蚀）。"""
        for s in self._signals():
            if s.get("sig_addr") == addr and s.get("sig_type", SIG_HOLDING) == sig_type:
                return s.get("value")
        return None

    def _clear_signals(self) -> None:
        """删除表中全部信号量。仅 power_on 上电后使用，删除后必须立刻
        重新追加核心信号量（删除/重加本身会打乱映射，成对执行才安全）。"""
        for s in self._signals():
            try:
                self.robot.del_tio_rs_signal(s["sig_name"])
            except Exception:
                pass

    def _add_signal(self, name: str, addr: int, sig_type: int = SIG_HOLDING) -> None:
        self.robot.add_tio_rs_signal({
            'sig_name': name, 'chn_id': self.chn, 'sig_type': sig_type,
            'sig_addr': addr, 'value': 0, 'frequency': 5})

    def _ensure_signals(self) -> None:
        """按地址追加缺失的核心信号量（表容量约 8 条，留下 2 条额度给
        行程端点等低频 _read_tmp 追加）。

        要求夹爪已上电（独立模式未走 power_on 时由首次读取触发）。
        """
        if self._signals_added:
            return
        added = False
        for name in CORE:
            if not self._has_slot(REG[name]):
                self._add_signal(name, REG[name])
                added = True
        if added:
            time.sleep(1.2)  # 等控制器完成首次轮询
        self._signals_added = True

    def _heal_signals(self, rounds: int = 8) -> bool:
        """删光→重加→物理验证，直至 position 槽位真实跟踪开合。

        控制器固件在信号量表重建时存在竞态：同样的「删光→重加」随机
        得到正确或错乱的「名字→轮询值」映射，且错乱可能连续多轮
        （实测 8 轮内可收敛）。判定标准不依赖表本身：release/grip 的
        物理动作必然执行，position 跟踪（闭−开 ≥ 800）即证明槽位健康。
        """
        for _ in range(rounds):
            self._clear_signals()
            for name in CORE:
                self._add_signal(name, REG[name])
            time.sleep(2.0)  # 等控制器重建轮询
            self.release()
            time.sleep(2.0)
            p_open = self._read_at(REG["position"])
            self.grip(1000)
            time.sleep(2.5)
            p_close = self._read_at(REG["position"])
            if (p_open is not None and p_close is not None
                    and p_close - p_open >= 800):
                self.release()  # 验证结束，留张开状态
                return True
        self.release()
        return False

    def _read_core(self, name: str):
        self._ensure_signals()
        return self._read_at(REG[name])

    def _read_tmp(self, addr: int, sig_type: int = SIG_HOLDING):
        """低频接口：读单个寄存器（缺槽位则追加一个持久槽位，绝不删改）。"""
        if not self._has_slot(addr, sig_type):
            self._add_signal(f"_tmp_{sig_type}_{addr:04X}", addr, sig_type)
        time.sleep(1.0)  # 等控制器完成（首次）轮询
        return self._read_at(addr, sig_type)

    def _read_tmp_multi(self, addr: int, count: int, sig_type: int = SIG_HOLDING) -> list:
        """低频接口：读连续 count 个寄存器（追加式，绝不删改已有条目）。"""
        added = False
        for i in range(count):
            if not self._has_slot(addr + i, sig_type):
                self._add_signal(f"_m_{sig_type}_{addr + i:04X}", addr + i, sig_type)
                added = True
        if added:
            time.sleep(1.2)
        return [self._read_at(addr + i, sig_type) for i in range(count)]

    # ---------- 运动接口 ----------
    def grip(self, speed: int = 1000) -> int:
        return self._write_regs(REG["adaptive_speed"], [speed, 1])

    def release(self, speed: int = 1000) -> int:
        """松开。与 grip 同为 0x10 多寄存器写 [速度, 动作=0]——实测 0x06
        单寄存器写动作寄存器在本夹爪固件上不生效。"""
        return self._write_regs(REG["adaptive_speed"], [speed, 0])

    def force_position_control(self, speed: int, position: int) -> int:
        return self._write_regs(REG["force_pos_speed"], [speed, position])

    def follow_control(self, speed: int, position: int) -> int:
        return self._write_regs(REG["follow_speed"], [speed, position])

    # ---------- 维护指令 ----------
    def calibrate_fingertip(self, value: int = 1) -> int:
        """指尖力强制校准（手册推荐定期执行，消除累积误差）。写 0x0011。"""
        return self._write_reg(REG["fingertip_calibration"], value)

    def clear_fingertip_zero(self, value: int = 1) -> int:
        """指尖力硬件清零（消除零点漂移）。写 0x001D。"""
        return self._write_reg(REG["fingertip_zero"], value)

    # ---------- 运行参数接口 ----------
    def set_adaptive_speed(self, speed: int) -> int:
        """设自适应夹取速度(200~1500)：写 0x000A。"""
        return self._write_reg(REG["adaptive_speed"], speed)

    def set_target_force(self, force: int) -> int:
        """设目标夹持力(0~100)：写 0x0012。"""
        return self._write_reg(REG["target_force"], force)

    def set_control_mode(self, mode: int) -> int:
        """切控制模式(0自适应/1力位)：写 0x000E。"""
        return self._write_reg(REG["control_mode"], mode)

    def set_force_position_speed(self, speed: int) -> int:
        """设力位速度(200~1500)：写 0x000F。"""
        return self._write_reg(REG["force_pos_speed"], speed)

    def set_force_position_target(self, position: int) -> int:
        """设力位目标位置(0~1000)：写 0x0010。"""
        return self._write_reg(REG["force_pos_target"], position)

    # ---------- 读接口 ----------
    def _read_force(self, addr: int):
        """读单轴力(0x0000~0x0002，16 位有符号)。"""
        v = self._read_tmp(addr)
        if v is None:
            return None
        return v - 0x10000 if v >= 0x8000 else v

    def read_force_fx(self): return self._read_force(REG["force_fx"])
    def read_force_fy(self): return self._read_force(REG["force_fy"])
    def read_force_fz(self): return self._read_force(REG["force_fz"])

    def read_state(self):            return self._read_core("state")
    def read_hardness(self):         return self._read_core("hardness")
    def read_position(self):         return self._read_core("position")
    def read_current(self):         return self._read_core("current")
    def read_adaptive_speed(self):   return self._read_core("adaptive_speed")
    def read_action(self):           return self._read_core("action")

    def read_protocol(self):         return self._read_tmp(REG["protocol"])
    def read_device_id(self):        return self._read_tmp(REG["device_id"])
    def read_control_mode(self):     return self._read_tmp(REG["control_mode"])
    def read_force_position_speed(self):  return self._read_tmp(REG["force_pos_speed"])
    def read_force_position_target(self): return self._read_tmp(REG["force_pos_target"])
    def read_target_force(self):     return self._read_tmp(REG["target_force"])
    def read_open_position(self):    return self._read_tmp(REG["open_position"])
    def read_close_position(self):   return self._read_tmp(REG["close_position"])
    def read_force_position_state(self): return self._read_tmp(REG["force_pos_state"])
    def read_eeprom_enabled(self):   return self._read_tmp(REG["eeprom_enabled"])
    def read_error_flags(self):      return self._read_tmp(REG["error_flags"])

    def read_grip_count(self) -> int:
        lo, hi = self._read_tmp_multi(REG["grip_count"], 2)
        if lo is None or hi is None:
            return None
        return (hi << 16) | lo  # 手册称小端序(低字在前)

    def read_baudrate(self) -> int:
        lo, hi = self._read_tmp_multi(REG["baudrate"], 2)
        if lo is None or hi is None:
            return None
        return (hi << 16) | lo  # 手册称小端序(低字在前)

    def read_factory_number(self) -> str:
        regs = self._read_tmp_multi(REG["factory_number"], 6, SIG_INPUT)
        if any(r is None for r in regs):
            return None
        length = regs[0]
        raw = bytes(r & 0xFF for r in regs[1:])
        return raw[:length].decode("ascii", errors="replace")
