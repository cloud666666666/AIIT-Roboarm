# Description: 高频循环场景下的限频打印，防止同类报错刷屏。
import time

_last_print_time: dict[str, float] = {}
_suppressed_count: dict[str, int] = {}


def throttled_print(key: str, text: str, min_interval_s: float = 30.0) -> None:
    """同一 key 的打印在 `min_interval_s` 内只输出一次。

    抓取主循环每轮都会重试并重复触发同类报错（如逆解失败、目标靠近
    放置点跳过抓取），全量打印导致刷屏：同一 key 收敛为最多每
    `min_interval_s` 秒一条，恢复打印时附上抑制期间的重复次数。

    Args:
        key: 打印点标识，相同 key 共享限频窗口（text 本身可能每次
            都带变化的坐标，不参与比较）。
        text: 要打印的消息。
        min_interval_s: 同 key 两次打印的最小间隔，单位秒。
    """
    now = time.monotonic()
    if now - _last_print_time.get(key, 0.0) < min_interval_s:
        _suppressed_count[key] = _suppressed_count.get(key, 0) + 1
        return
    count = _suppressed_count.pop(key, 0)
    if count:
        print(f"{text}（相同报错已抑制 {count} 次）")
    else:
        print(text)
    _last_print_time[key] = now
