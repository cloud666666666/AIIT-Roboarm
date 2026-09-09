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
        if self._shared_robot:
            self.robot = robot
        else:
            self.robot = jkrc.RC(ip)
            ret = self.robot.login()
            if ret[0] != 0:
                raise ConnectionError(f"login 失败: {ret}")
        self._config_channel()
        self._clear_signals()
        self._add_core()

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
        """机械臂上电（末端 RS485 接口与夹爪供电依赖上电才通信）。"""
        self.robot.power_on()
        time.sleep(2)
        # 上电会复位 RS485 通道配置，必须先重新配置才能通信
        self._config_channel()
        # 上电后重新加核心信号量（上电前加的信号量首次轮询时夹爪没电，会卡在 0）
        self._clear_signals()
        self._add_core()
        # 等夹爪就绪：就绪后会有待机电流(current 非 0)，最多等 15s
        for _ in range(15):
            time.sleep(1)
            cur = self._read_core("current")
            if cur is not None and cur != 0:
                return
        print("Warning: 夹爪 15s 内未就绪（电流一直为 0），"
              "请检查 RS485 接线与通道配置")

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
    def _clear_signals(self) -> None:
        info = self.robot.get_rs485_signal_info()
        if info[0] == 0:
            for s in info[1]:
                try:
                    self.robot.del_tio_rs_signal(s["sig_name"])
                except Exception:
                    pass

    def _add_signal(self, name: str, addr: int, sig_type: int = SIG_HOLDING) -> None:
        self.robot.add_tio_rs_signal({
            'sig_name': name, 'chn_id': self.chn, 'sig_type': sig_type,
            'sig_addr': addr, 'value': 0, 'frequency': 5})

    def _add_core(self) -> None:
        for name in CORE:
            self._add_signal(name, REG[name])
        time.sleep(1.2)  # 等控制器完成首次轮询

    def _read_core(self, name: str):
        info = self.robot.get_rs485_signal_info()
        if info[0] == 0:
            for s in info[1]:
                if s["sig_name"] == name:
                    return s["value"]
        return None

    def _read_tmp(self, addr: int, sig_type: int = SIG_HOLDING):
        """低频接口：临时读单个寄存器（保留核心信号量）。"""
        self.robot.del_tio_rs_signal("_tmp")
        self._add_signal("_tmp", addr, sig_type)
        time.sleep(1.0)
        val = self._read_core("_tmp")
        return val

    def _read_tmp_multi(self, addr: int, count: int, sig_type: int = SIG_HOLDING) -> list:
        """低频接口：临时读连续 count 个寄存器。"""
        self._clear_signals()  # 腾出信号量额度
        names = [f"_m{i}" for i in range(count)]
        for i, n in enumerate(names):
            self._add_signal(n, addr + i, sig_type)
        time.sleep(1.2)
        info = self.robot.get_rs485_signal_info()
        vals = {s["sig_name"]: s["value"] for s in info[1]} if info[0] == 0 else {}
        self._clear_signals()
        self._add_core()  # 恢复核心
        return [vals.get(n) for n in names]

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
