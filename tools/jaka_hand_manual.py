# Description: O6 手手动开合工具——空格=捏合、回车=张开、e=改闭合角度、q=退出，
# 逐指滚动显示六指实时角度（每轮约 2.3s），供人工判定合适的闭合角度/姿势。
#
# 链路与读写策略同 tools/jaka_hand_probe.py：单条目 + update_tio_rs_signal
# 逐寄存器读（约 0.38s/指），三帧（速度→力矩→目标角度）写姿势。动作后读数
# 有 2~5s 假 0 窗口（见 arm/jaka_hand.py 头注释），工具自动等 4s 再读；
# 读到 0 连续出现时以稳定后的读数为准。
#
# 用法（项目根目录，激活 venv 后，必须交互终端）：
#   python tools/jaka_hand_manual.py
#   python tools/jaka_hand_manual.py --close 60,50,60,60,255,255   # 指定闭合角度
#   python tools/jaka_hand_manual.py --slave 0x28                  # 左手
import argparse
import os
import select
import sys
import termios
import time
import tty

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arm.jaka_sdk  # noqa: F401  按平台注入 jkrc SDK 路径并预加载依赖库

from arm.jaka_hand import FINGER_NAMES, GRASP_ANGLES, JakaHand, OPEN_ANGLES
from utils.config_getter import get_config_value

# 判定均值用指组合（与 arm/jaka_ctrl_by_sdk.py HAND_GRASP_FINGERS 一致：拇/食/
# 中/环）；读数行附四指均值/255，便于直接对照 catch() 的 GRASP_OK_MIN 阈值。
FINGERS_4 = (0, 2, 3, 4)

# 动作后等读数稳定的时长：应答迟滞窗口实测约 2~5s（JakaHand 头注释）
SETTLE_S = 4.0


def parse_close(text: str) -> list:
    """解析 --close 的 6 个角度（空格/逗号分隔，0~255）。"""
    try:
        vals = [int(p) for p in text.replace(",", " ").split()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"角度需为整数：{text!r}") from e
    if len(vals) != 6 or any(not 0 <= v <= 255 for v in vals):
        raise argparse.ArgumentTypeError(
            f"需 6 个 0~255 的整数（空格/逗号分隔），实际 {text!r}")
    return vals


def fmt_values(values) -> str:
    """六个角度按位宽打印（0=完全弯曲 ~ 255=伸直），None（读失败）打 ?。"""
    return " ".join("  ?" if v is None else f"{v:3d}" for v in values)


def fmt_reading(values) -> str:
    """一行读数：逐指名称=值 + 四指均值（归一化），供对照判定阈值。"""
    parts = [f"{FINGER_NAMES[i]}={'?' if v is None else v}" for i, v in enumerate(values)]
    active = [values[i] for i in FINGERS_4]
    if any(v is None for v in active):
        mean = "四指均值=?"
    else:
        m = round(sum(active) / len(active))
        mean = f"四指均值={m}({m / 255:.2f})"
    return "｜".join(parts) + f"  {mean}"


class Keyboard:
    """终端 cbreak 逐键读取（按键即响应、无需回车）；Ctrl-C 仍可中断。"""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.saved = None

    def __enter__(self):
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def get(self, timeout: float = 0.05):
        """非阻塞取一键；无按键返回 None。"""
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return None
        data = os.read(self.fd, 1)
        return data.decode(errors="ignore") or None

    def ask_line(self, prompt: str):
        """临时恢复行模式做一次 input()（供 e 改角度），结束回到 cbreak。"""
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
        try:
            return input(prompt)
        except EOFError:
            return None
        finally:
            tty.setcbreak(self.fd)


def main() -> int:
    parser = argparse.ArgumentParser(description="O6 手手动开合与读数观察")
    parser.add_argument(
        "--close", type=parse_close, default=list(GRASP_ANGLES),
        help="空格下发的闭合角度，6 个 0~255 整数（默认=驱动的 GRASP_ANGLES；"
             "空格/逗号分隔，如 60,50,60,60,255,255）")
    parser.add_argument("--slave", type=lambda x: int(x, 0), default=0x27,
                        help="从站地址（右 0x27 / 左 0x28）")
    args = parser.parse_args()

    if not sys.stdin.isatty():
        print("需要交互终端（TTY）运行；管道/重定向下无法读取按键")
        return 1

    hand = JakaHand(ip=get_config_value("arm_port"), slave_id=args.slave)
    try:
        hand.power_on()
        close_angles = list(args.close)
        print("O6 手手动开合：空格=捏合  回车=张开  e=改闭合角度  q=退出")
        print(f"  张开角度(回车下发) = [{fmt_values(OPEN_ANGLES)}]  （顺序："
              f"{' '.join(FINGER_NAMES)}）")
        print(f"  闭合角度(空格下发) = [{fmt_values(close_angles)}]")
        print(f"  动作后等 {SETTLE_S:.0f}s 再读（应答迟滞窗口 2~5s 内有假 0）；"
              f"读数每轮约 2.3s，连续滚动")
        with Keyboard() as keys:
            while True:
                values = hand.read_angles()
                print(f"读数 {time.strftime('%H:%M:%S')}  {fmt_reading(values)}")
                # 等一键（读完 50ms 内无按键就继续下一轮读）
                ch = None
                while ch is None:
                    ch = keys.get(0.05)
                if ch == " ":
                    ret = hand.set_pose(close_angles)
                    print(f"  → 捏合 [{fmt_values(close_angles)}] 返回码={ret}"
                          f"（0 仅代表帧已发出，动作以读回为准）")
                    time.sleep(SETTLE_S)
                elif ch in ("\r", "\n"):
                    ret = hand.open()
                    print(f"  → 张开 [{fmt_values(OPEN_ANGLES)}] 返回码={ret}")
                    time.sleep(SETTLE_S)
                elif ch == "e":
                    line = keys.ask_line("  输入 6 个角度（0~255，空格/逗号分隔，"
                                         "直接回车取消）: ")
                    if line is not None and line.strip():
                        try:
                            close_angles = parse_close(line)
                            print(f"  闭合角度已改为 [{fmt_values(close_angles)}]")
                        except argparse.ArgumentTypeError as e:
                            print(f"  未修改：{e}")
                elif ch in ("q", "Q"):
                    print("退出")
                    return 0
                # 其他按键（含方向键转义序列）忽略
    except KeyboardInterrupt:
        print("\n中断退出")
    finally:
        hand.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
