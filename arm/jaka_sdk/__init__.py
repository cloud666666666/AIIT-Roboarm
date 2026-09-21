# -*- coding: utf-8 -*-
"""JAKA SDK（jkrc）加载器：按平台自动选择二进制目录，注入 sys.path 并预加载依赖库。

目录布局：
  arm/jaka_sdk/windows/            jkrc.pyd + jakaAPI.dll
  arm/jaka_sdk/aarch64-linux-gnu/  jkrc.so + libjakaAPI.so
  arm/jaka_sdk/x86_64-linux-gnu/   jkrc.so + libjakaAPI.so

用法：`import jkrc` 之前先 `import arm.jaka_sdk`，本模块导入即完成注入，
无需再手动操作 sys.path / LD_LIBRARY_PATH。
"""
import ctypes
import os
import platform
import sys

if sys.platform == "win32":
    _SUBDIR = "windows"
elif sys.platform.startswith("linux"):
    _SUBDIR = {"aarch64": "aarch64-linux-gnu", "x86_64": "x86_64-linux-gnu"}.get(
        platform.machine()
    )
else:
    _SUBDIR = None

if _SUBDIR is None:
    raise ImportError(f"JAKA SDK 不支持当前平台: {sys.platform}/{platform.machine()}")

_SDK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), _SUBDIR)
if not os.path.isdir(_SDK_DIR):
    raise ImportError(f"JAKA SDK 目录缺失: {_SDK_DIR}（请从 JAKA 官网下载对应平台 SDK 放入）")

sys.path.insert(0, _SDK_DIR)

if sys.platform == "win32":
    os.add_dll_directory(_SDK_DIR)
else:
    # jkrc.so 依赖 libjakaAPI.so（无 RPATH），按绝对路径预加载，避免依赖 LD_LIBRARY_PATH；
    # 同时写入环境变量，让本进程派生的子进程也能找到。
    ctypes.CDLL(os.path.join(_SDK_DIR, "libjakaAPI.so"))
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        p for p in (_SDK_DIR, os.environ.get("LD_LIBRARY_PATH", "")) if p
    )
