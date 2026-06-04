import atexit
import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Any

import cv2
from flask import Flask, Response, request

from utils.config_getter import get_config_value

_WINDOW_READY: set[str] = set()
_HEADLESS_SERVER_LOCK = threading.Lock()
_HEADLESS_SERVER_STARTED = False
_HEADLESS_TEMPLATE_PATH = Path(__file__).with_name("cv2_display.html")
_HEADLESS_SERVER_HOST = "0.0.0.0"
_HEADLESS_SERVER_PORT = get_config_value(
    "cv2_headless_port", None, raise_if_missing=False
)
_HEADLESS_JPEG_QUALITY = 50
_HEADLESS_OUTPUT_FPS = 30  # cap frames sent to browser to avoid TCP buffer buildup
_HEADLESS_LATEST_FRAMES: dict[str, bytes] = {}
_HEADLESS_FRAME_EVENTS: dict[str, threading.Event] = {}
_HEADLESS_KEY_QUEUE: "queue.Queue[int]" = queue.Queue()
_HEADLESS_MOUSE_LOCK = threading.Lock()
_HEADLESS_MOUSE_CALLBACKS: dict[str, tuple[Callable[..., Any], Any]] = {}


def show_img_by_web() -> bool:
    return _HEADLESS_SERVER_PORT is not None


def _frame_event(window_name: str) -> threading.Event:
    with _HEADLESS_SERVER_LOCK:
        return _HEADLESS_FRAME_EVENTS.setdefault(window_name, threading.Event())


def _headless_index_html() -> str:
    return _HEADLESS_TEMPLATE_PATH.read_text(encoding="utf-8")


def _run_headless_server() -> None:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return _headless_index_html()

    @app.get("/windows")
    def windows():
        with _HEADLESS_SERVER_LOCK:
            window_names = sorted(_HEADLESS_LATEST_FRAMES)
        return {"windows": window_names}

    @app.get("/stream/<path:window_name>")
    def stream(window_name: str):
        event = _frame_event(window_name)

        def generate():
            last_frame = None
            last_send_time = 0.0
            min_interval = 1.0 / _HEADLESS_OUTPUT_FPS
            while True:
                # Wait for a new frame, but poll at output rate so we
                # naturally drop bursts when producer is faster than browser.
                event.wait(timeout=min_interval)
                event.clear()
                # Always grab the *latest* frame — discard any backlog.
                frame = _HEADLESS_LATEST_FRAMES.get(window_name)
                if frame is None:
                    continue
                if frame == last_frame:
                    continue
                # Rate-limit output: if we just sent a frame, hold the
                # interval so the browser never queues up a backlog.
                now = time.perf_counter()
                if now - last_send_time < min_interval:
                    continue
                last_frame = frame
                last_send_time = now
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )

        return Response(
            generate(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    @app.post("/key")
    def key_event():
        payload = request.get_json(silent=True) or {}
        key_code = payload.get("keyCode")
        if not isinstance(key_code, int):
            return {"ok": False}, 400
        _HEADLESS_KEY_QUEUE.put(key_code & 0xFFFF)
        return {"ok": True}

    @app.post("/mouse/<path:window_name>")
    def mouse_event(window_name: str):
        payload = request.get_json(silent=True) or {}
        try:
            event_code = int(payload["event"])
            x = int(payload["x"])
            y = int(payload["y"])
            flags = int(payload.get("flags", 0))
        except (KeyError, TypeError, ValueError):
            return {"ok": False}, 400
        with _HEADLESS_MOUSE_LOCK:
            entry = _HEADLESS_MOUSE_CALLBACKS.get(window_name)
        if entry is None:
            return {"ok": True, "dispatched": False}
        callback, param = entry
        try:
            callback(event_code, x, y, flags, param)
        except Exception as exc:  # noqa: BLE001
            print(f"[cv2_display] mouse callback for {window_name!r} raised: {exc}")
            return {"ok": False}, 500
        return {"ok": True, "dispatched": True}

    app.run(
        host=_HEADLESS_SERVER_HOST,
        port=_HEADLESS_SERVER_PORT,
        threaded=True,
        debug=False,
        use_reloader=False,
    )


def _ensure_headless_server() -> None:
    global _HEADLESS_SERVER_STARTED
    if _HEADLESS_SERVER_STARTED:
        return
    with _HEADLESS_SERVER_LOCK:
        if _HEADLESS_SERVER_STARTED:
            return
        thread = threading.Thread(target=_run_headless_server, daemon=True)
        thread.start()
        _HEADLESS_SERVER_STARTED = True
        print(
            f"Headless OpenCV stream ready at http://{_HEADLESS_SERVER_HOST}:{_HEADLESS_SERVER_PORT}/ "
            f"(append ?window=<window_name>)"
        )


def _encode_headless_frame(image: Any) -> bytes | None:
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), _HEADLESS_JPEG_QUALITY],
    )
    if not ok:
        return None
    return encoded.tobytes()


def _publish_headless_frame(window_name: str, image: Any) -> None:
    frame = _encode_headless_frame(image)
    if frame is None:
        return
    _HEADLESS_LATEST_FRAMES[window_name] = frame
    _frame_event(window_name).set()


def show_image(window_name: str, image: Any):
    if show_img_by_web():
        _ensure_headless_server()
        _publish_headless_frame(window_name, image)
        return
    if window_name not in _WINDOW_READY:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        _WINDOW_READY.add(window_name)
    cv2.imshow(window_name, image)


def poll_key(delay: int = 1) -> int:
    if show_img_by_web():
        # delay <= 0 means block forever, matching cv2.waitKey semantics.
        timeout = None if delay <= 0 else delay / 1000.0
        try:
            return _HEADLESS_KEY_QUEUE.get(timeout=timeout)
        except queue.Empty:
            return -1
    return cv2.waitKey(delay)


def set_mouse_callback(
    window_name: str,
    callback: Callable[..., Any],
    param: Any = None,
):
    if show_img_by_web():
        _ensure_headless_server()
        with _HEADLESS_MOUSE_LOCK:
            _HEADLESS_MOUSE_CALLBACKS[window_name] = (callback, param)
        return
    if window_name not in _WINDOW_READY:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.waitKey(1)
        _WINDOW_READY.add(window_name)
    cv2.setMouseCallback(window_name, callback, param)


def destroy_all_windows():
    if show_img_by_web():
        with _HEADLESS_MOUSE_LOCK:
            _HEADLESS_MOUSE_CALLBACKS.clear()
        return
    cv2.destroyAllWindows()


def destroy_window(window_name: str):
    if show_img_by_web():
        with _HEADLESS_MOUSE_LOCK:
            _HEADLESS_MOUSE_CALLBACKS.pop(window_name, None)
        return
    cv2.destroyWindow(window_name)
    _WINDOW_READY.discard(window_name)


atexit.register(destroy_all_windows)
