# ROBOARM — 机械臂 + 深度相机 自动采集与控制项目

基于 Piper 机械臂 + Orbbec 深度相机 + YOLO 目标检测，实现自动抓取、分类放置、VLA 数据采集（LeRobot 格式）等功能。

## 目录

- [硬件依赖](#硬件依赖)
- [环境搭建（从零开始）](#环境搭建从零开始)
  - [1. 系统依赖](#1-系统依赖)
  - [2. 安装 uv](#2-安装-uv)
  - [3. 克隆仓库并同步 Python 环境](#3-克隆仓库并同步-python-环境)
  - [4. CAN 总线配置（Piper 机械臂）](#4-can-总线配置piper-机械臂)
  - [5. 连通性验证](#5-连通性验证)
- [配置说明](#配置说明)
  - [config.yaml 关键配置项](#configyaml-关键配置项)
  - [YOLO 模型准备](#yolo-模型准备)
- [自动采集数据](#自动采集数据)
  - [概述](#概述)
  - [工作流程](#工作流程)
  - [使用方法](#使用方法)
  - [键盘控制](#键盘控制)
  - [采集模式说明](#采集模式说明)
  - [数据保存与断点续采](#数据保存与断点续采)
  - [头部无显示运行](#头部无显示运行)
- [其他场景](#其他场景)
- [机械臂坐标系](#机械臂坐标系)
- [故障排查](#故障排查)

---

## 硬件依赖

| 设备 | 说明 |
|------|------|
| Piper 机械臂 | 通过 USB-CAN 适配器连接，CAN 总线通信 |
| Orbbec 深度相机 | 通过 USB 连接，用于目标检测与视觉反馈 |
| USB-CAN 适配器 | 连接机械臂与主机 |
| Jetson Orin / 任意 Linux 主机 | 运行控制程序 |

---

## 环境搭建（从零开始）

### 1. 系统依赖

```bash
sudo apt update
sudo apt install -y can-utils ethtool build-essential
```

**Jetson 上额外需要 gs_usb 内核模块**（USB-CAN 通信必需）：

先检查是否已加载：

```bash
lsmod | grep gs_usb
```

如果没有输出，说明内核未启用 `gs_usb`，需要编译安装。详见文末 [故障排查 / gs_usb 内核模块](#gs_usb-内核模块未加载)。

### 2. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

验证：

```bash
uv --version   # 应 >= 0.5.0
```

### 3. 克隆仓库并同步 Python 环境

```bash
# 克隆本仓库
git clone git@github.com:cloud666666666/AIIT-Roboarm.git 
cd ~/roboarm

# 克隆 LeRobot 依赖（与 roboarm 同级或任意位置）
git clone https://github.com/huggingface/lerobot.git 

# 创建软链接，让 roboarm 能找到 lerobot 源码
# 将源路径替换为你实际克隆 lerobot 的位置
ln -s ~/lerobot ~/roboarm/lerobot

# 回到 roboarm，同步 Python 环境（含 lerobot）
cd ~/roboarm
uv sync

# 验证关键依赖
uv run python -c "from piper_sdk import C_PiperInterface_V2; print('piper_sdk OK')"
uv run python -c "import lerobot; print('lerobot OK:', lerobot.__file__)"
uv run python -c "from ultralytics import YOLO; print('ultralytics OK')"
```

> **说明**：`uv sync` 会根据 `pyproject.toml` 和 `uv.lock` 自动创建 `.venv` 并安装所有依赖（Python 3.10）。

### 4. CAN 总线配置（Piper 机械臂）

#### 4.1 找到真实的 CAN 接口名

```bash
source .venv/bin/activate

SDK_DIR=$(python - <<'PY'
import inspect, os, piper_sdk
print(os.path.dirname(inspect.getfile(piper_sdk)))
PY
)

bash "$SDK_DIR/find_all_can_port.sh"
```

示例输出：

```
Interface can_piper is connected to USB port 1-4.2:1.0
```

> 记下输出中的 **接口名**（如 `can_piper`、`can4`）和 **USB 地址**（如 `1-4.2:1.0`）。

#### 4.2 激活 CAN 接口

```bash
CAN_IF="can_piper"        # 替换为上一步的实际接口名
USB_ADDR="1-4.2:1.0"      # 替换为上一步的实际 USB 地址

sudo ip link set "$CAN_IF" down 2>/dev/null || true
bash "$SDK_DIR/can_activate.sh" "$CAN_IF" 1000000 "$USB_ADDR"
```

> **注意**：波特率固定为 `1000000`，不要填成 `100000`。

#### 4.3 验证 CAN 状态

```bash
ip -details link show "$CAN_IF"
```

应看到接口处于 `UP` 状态，波特率为 `1000000`。

### 5. 连通性验证

复制配置文件并修改：

```bash
cp config.yaml.example config.yaml
```

编辑 `config.yaml`，将 `arm_type` 改为 `piper`，`arm_port` 改为实际的 CAN 接口名（如 `can_piper` 或 `can4`）。

```bash
# 只读连通性测试
uv run python arm/calibrate_offset.py
```

如果能打印当前关节角度/末端位姿，说明 CAN 通信正常。

```bash
# 运动测试
uv run python arm/piper_ctrl_by_sdk.py
```

机械臂应能进行使能并执行测试运动。

---

## 配置说明

所有运行配置集中在项目根目录的 `config.yaml` 中（复制自 `config.yaml.example`）。

### config.yaml 关键配置项

#### 机械臂连接

```yaml
arm_port: can4          # Piper 为 can*，Lerobo 为 COM*
arm_type: piper         # piper 或 lerobo
arm_offset: [0, -30, -40, -50, 0]  # 关节零位偏移，单位度
```

#### 相机

```yaml
camera_ip: ""           # 留空使用本地 Orbbec 相机；填写 IP 则使用远程相机
camera_port: 8084       # 远程相机端口
cv2_headless_port: 8079 # Web 显示端口（无显示器环境），留空则使用 OpenCV 窗口
```

#### 桌面高度

```yaml
default_desktop_height: 0.135  # 机械臂坐标系下桌面 Z 坐标，单位米
```

#### 物品分类 / YOLO 检测

```yaml
classification_YOLO_model_path:
  - /home/czn/roboarm/object_detect/runs/best.pt  # YOLO OBB 模型路径
default_conf_thres: 0.5     # 检测置信度阈值
default_gripper_close_threshold: 0.05  # 夹爪闭合阈值

# 各类别抓取与放置配置
class_pos:
  default:                  # 默认配置（匹配不到的类别使用此项）
    pos: [0.05, 0.45]       # 放置位置 (x, y)，单位米
    random_pos:             # auto-reset 随机撒回范围 [[x_min,x_max],[y_min,y_max]]
      - [0.0, 0.5]
      - [-0.2, 0.3]
  potato:
    pos: [0.05, 0.4]
    random_pos:
      - [0.0, 0.16]
      - [-0.2, 0.3]
  tomato:
    pos: [0.05, 0.4]
    random_pos:
      - [0.17, 0.33]
      - [-0.2, 0.3]
  carrot:
    pos: [0.05, 0.4]
    random_pos:
      - [0.34, 0.5]
      - [-0.2, 0.3]
```

#### Auto-Reset 参数

```yaml
workspace_x_range: [-0.1, 0.55]   # 工作空间 X 范围，超出会拒绝
workspace_y_range: [-0.3, 0.55]   # 工作空间 Y 范围
reset_min_place_dist_m: 0.20      # 随机放置时离已有物体的最小距离
reset_max_objects_per_cycle: 1    # 每轮最多撒回物体数
```

#### 抓取动作参数

```yaml
catch_raise_height: 0.1   # 抓取前抬起高度，单位米
place_raise_height: 0.1   # 放置前抬起高度，单位米
catch_time_interval_s: 0.5 # 抓取动作间停顿，单位秒
catch_offset: 0.00         # 夹爪前向偏移，单位米
go_down_before_open_gripper_in_place: true  # 放置时先下降再松爪
```

### YOLO 模型准备

项目使用 YOLO OBB（Oriented Bounding Box）模型进行目标检测。

- 训练脚本：`object_detect/train.py`
- 标注数据工具：`object_detect/dataset_process/`（含拍照脚本和 LabelMe JSON 转 YOLO 标签工具）
- 预训练模型放在 `object_detect/runs/` 目录下

如果需要训练自己的模型：

1. 用 `object_detect/dataset_process/get_pic.py` 拍摄目标物体图片
2. 用 LabelMe 标注（OBB 旋转框）
3. 用 `object_detect/dataset_process/json2label.py` 转换为 YOLO 格式
4. 运行 `object_detect/train.py` 训练

---

## 自动采集数据

### 概述

`classification/record_and_auto_reset.py` 是一个完整的 VLA 数据自动采集流水线，整合了：

- **录制阶段**：YOLO 检测目标物体 → 机械臂抓取 → 放入收集箱 → 保存为 LeRobot 格式数据
- **复位阶段**：YOLO 检测桌上所有目标物体 → 逐个移动到随机位置 → 为下一轮采集创造新场景

### 工作流程

```
┌─ Episode N ──────────────────────────────────────┐
│                                                   │
│  1. Recording Phase (录制)                         │
│     ├─ 相机持续拍摄                                 │
│     ├─ YOLO 检测目标物体                            │
│     ├─ 机械臂移动到目标位置抓取                       │
│     ├─ 放入收集箱                                   │
│     └─ 自动保存 MP4 + 关节状态到 LeRobot 数据集      │
│                                                   │
│  2. Auto-Reset Phase (复位)                        │
│     ├─ 机械臂归零，相机获取清晰视野                   │
│     ├─ YOLO 检测桌上所有目标物体                     │
│     ├─ 逐个抓取 → 移动到随机位置                      │
│     └─ 为下一轮采集创造不同的初始场景                  │
│                                                   │
│  3. 循环直到 NUM_EPISODES 完成                      │
│                                                   │
└───────────────────────────────────────────────────┘
```

### 使用方法

#### 1. 修改脚本中的采集参数

编辑 `classification/record_and_auto_reset.py` 顶部配置区：

```python
DATASET_ROOT = "/home/czn/dataset/piper_yolopick"  # 数据集保存路径
NUM_EPISODES = 1000       # 总共采集的 episode 数量
TARGET_CLASS = "carrot"   # 目标类别（carrot / potato / tomato / 空字符串=所有）
TASK = "pick the carrot toy and place into box"  # 任务描述（写入数据集）
RESUME = True             # True=断点续采, False=从头开始
AUTO_ADVANCE_DELAY_S = 3.0  # 抓取成功后自动推进等待秒数（0=立即推进）
```

#### 2. 运行采集

```bash
uv run python classification/record_and_auto_reset.py
```

#### 3. 启动后的交互

程序启动后会打印当前配置和键盘快捷键，显示：

```
Integrated record + auto-reset pipeline
  Data saved to: /home/czn/dataset/piper_yolopick
  Target class:  carrot
  Episodes:      1000
  Resume:        True (starting from episode 5)
Controls:
  n/→  = end current episode
  r/←  = discard episode & rerecord
  s    = skip auto-reset this cycle
  q/Esc = stop entirely
```

### 键盘控制

| 按键 | 功能 |
|------|------|
| `n` / `→`（右箭头） | 立即结束当前 episode，保存数据，进入下一轮 |
| `r` / `←`（左箭头） | 丢弃当前 episode（不保存），重新录制 |
| `s` | 跳过本轮 auto-reset 阶段（直接进入下一 episode） |
| `q` / `Esc` | 停止采集，保存当前数据后退出 |

### 采集模式说明

**无人值守模式**：`AUTO_ADVANCE_DELAY_S` 设置后，每次抓取成功会自动推进到下一 episode，无需按键。设置为 `0` 则抓取完成后立即进入复位阶段。

**手动控制模式**：按 `n/→` 手动推进，适合需要精细控制的场景。

### 数据保存与断点续采

- 数据以 **LeRobot 格式** 保存在 `DATASET_ROOT` 目录
- 每个 episode 包含：MP4 视频（H.264）、关节状态序列（parquet）
- 视频编码在后台异步进行，不阻塞采集
- **`RESUME = True`** 时，重启程序会自动检测已有 episode 数量，从断点继续
- 异常退出时，程序会尝试保存当前 episode 的已录制部分
- **`RESUME = False`** 时，如果目标目录已存在则报错

### 单独运行各阶段

如果只需要录制（不复位），使用原始脚本：

```bash
uv run python classification/catch_with_arm_record_piper.py
```

如果只需要复位（不录制）：

```bash
uv run python classification/auto_reset_record.py
```

### 头部无显示运行

项目支持在无显示器（headless）的 Jetson 上运行。在 `config.yaml` 中设置：

```yaml
cv2_headless_port: 8079
```

启动程序后，在浏览器中访问 `http://<jetson-ip>:8079/?window=Recording` 即可实时查看相机画面和检测结果。

> 启动 Flask 服务器需要约 2-5 秒（Jetson 上较慢），程序会在启动时预热显示。

---

## 其他场景

### 物品分类抓取（LLM 视觉识别）

使用大模型进行视觉识别和指令理解，实现自然语言控制抓取：

```bash
uv run python llm/catch_by_llm.py
```

### 中国象棋

YOLO 识别棋子 → 机械臂自动走棋：

```bash
uv run python chess/catch_and_place.py
```

### 主从臂跟随

Leader-Follower 遥操作：

```bash
uv run python leader_follower/leader_follower.py
```

---

## 机械臂坐标系

以最下面的舵机为原点，红色为 X 轴，绿色为 Y 轴，蓝色为 Z 轴。

![alt text](docs/image1.png)

### 2D 手眼标定

`arm/calibrate_handeye_2d.py` 用于标定相机像素坐标到机械臂基座坐标系的映射：

1. 在相机画面上点击一个点
2. 手动控制机械臂末端移动到该点对应的实际桌面位置
3. 重复 4+ 个点
4. 自动计算单应性矩阵

详见 [Roboarm 机械臂文档](https://s1vvxephwhf.feishu.cn/wiki/Ulmsw4FPziq8oFkb8eXcx875nfe)

---

## 故障排查

### gs_usb 内核模块未加载

**现象**：`lsusb` 能看到 USB-CAN 设备，但 `ip link show | grep can` 看不到 CAN 接口；SDK 报 `SEND_MESSAGE_FAILED (100017)`。

**原因**：内核未编译 `gs_usb` 模块（Jetson 默认内核常见）。

**解决**（Jetson）：

```bash
# 检查
zcat /proc/config.gz | grep GS_USB
# 如果输出 "# CONFIG_CAN_GS_USB is not set"，则需要编译

# 1. 安装编译依赖
sudo apt install -y build-essential bc kmod flex bison libncurses-dev libssl-dev dwarves wget git

# 2. 获取与当前 uname -r 完全匹配的 Jetson 内核源码
# 3. 在源码目录执行：
zcat /proc/config.gz > .config
sed -i 's/CONFIG_LOCALVERSION=""/CONFIG_LOCALVERSION="-tegra"/' .config
make olddefconfig

export LOCALVERSION=-tegra
export IGNORE_PREEMPT_RT_PRESENCE=1
make -j$(nproc) modules

# 4. 安装
sudo mkdir -p /lib/modules/$(uname -r)/kernel/drivers/net/can/usb
sudo cp drivers/net/can/usb/gs_usb.ko /lib/modules/$(uname -r)/kernel/drivers/net/can/usb/
sudo depmod -a
sudo modprobe gs_usb

# 5. 设置开机自动加载
echo "gs_usb" | sudo tee /etc/modules-load.d/gs_usb.conf
```

### CAN 发送失败 / Message NOT sent

**恢复流程**：

```bash
CAN_IF="can_piper"   # 改成你的实际接口名

# 1. 关闭接口
sudo ip link set "$CAN_IF" down 2>/dev/null || true

# 2. 卸载驱动
sudo modprobe -r gs_usb

# 3. 物理拔掉 USB-CAN，等待 3 秒后重新插入
# 4. 机械臂断电再上电

# 5. 重新加载驱动
sudo modprobe gs_usb

# 6. 重新确认接口名并激活
ls /sys/class/net | grep can
sudo ip link set "$CAN_IF" type can bitrate 1000000
sudo ip link set "$CAN_IF" txqueuelen 1000
sudo ip link set "$CAN_IF" up
```

### 机械臂不响应运动指令

- 确认机械臂处于 **slave 模式**（非 master 模式），否则需重启臂体
- `judge_flag=False` 适用于第三方 USB-CAN 适配器
- 先跑 `arm/calibrate_offset.py` 验证连通性，再跑运动脚本

### 相机无画面

- 确认 Orbbec 相机 USB 已连接
- 确认 `camera_ip` 为空（使用本地相机）或填写了正确的远程相机 IP
- 检查 udev 规则是否已安装：`camera/scripts/` 中有 Orbbec 设备的 udev 配置文件

---

## 项目结构

```
roboarm/
  arm/              # 机械臂控制（Piper + Lerobo），手眼标定
  camera/           # Orbbec 深度相机 + USB 相机控制
  classification/   # YOLO 自动抓取 + 数据采集流水线
  object_detect/    # YOLO OBB 模型训练与检测
  llm/              # LLM 视觉识别与指令理解
  chess/            # 中国象棋场景
  leader_follower/  # 主从臂遥操作
  sim/              # Isaac Sim / MuJoCo 仿真
  utils/            # 配置读取，无头显示
  urdf/             # 机械臂 URDF 模型
  config.yaml       # 运行时配置
  prompts.toml      # LLM 提示词模板
```
