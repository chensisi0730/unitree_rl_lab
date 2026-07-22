# Unitree G1 双足机器人本地项目规划

> **目标**: 复现官方 G1 双足机器人的全部效果，包括仿真训练、策略部署、VLA/世界模型推理，以及本地 LLM 集成。

---

## 目录

- [1. 项目概览](#1-项目概览)
- [2. 系统架构](#2-系统架构)
- [3. 硬件需求](#3-硬件需求)
- [4. 环境安装](#4-环境安装)
- [5. 模块详解](#5-模块详解)
- [6. 训练流程](#6-训练流程)
- [7. 仿真到真机部署](#7-仿真到真机部署)
- [8. VLA/世界模型集成](#8-vla世界模型集成)
- [9. 本地 LLM 集成方案](#9-本地-llm-集成方案)
- [10. 快速开始](#10-快速开始)
- [11. 故障排查](#11-故障排查)
- [12. 参考资源](#12-参考资源)

---

## 1. 项目概览

本项目围绕 **Unitree G1 双足人形机器人**（29-DOF 版本），完整复现官方开源生态的各项能力：

```
┌─────────────────────────────────────────────────────────────┐
│                    G1 完整技术栈                              │
├─────────────┬──────────────┬──────────────┬─────────────────┤
│   底层控制   │   运动学习    │   决策大脑    │    部署推理     │
├─────────────┼──────────────┼──────────────┼─────────────────┤
│ unitree_sdk2│ unitree_rl_  │ UnifoLM-VLA  │ unitree_deploy  │
│ unitree_sdk│ lab (Isaac    │ UnifoLM-WMA  │ DDS通信         │
│ _2_python   │ Lab)         │ 本地LLM      │ Sim2Sim2Real    │
├─────────────┼──────────────┼──────────────┼─────────────────┤
│  Mujoco仿真  │  Isaac仿真   │  数据集      │  真机部署       │
│  unitree_   │  unitree_sim │  LeRobot     │  G1板载计算机   │
│  mujoco     │  _isaaclab   │  格式        │  x86开发机      │
└─────────────┴──────────────┴──────────────┴─────────────────┘
```

### G1 核心规格

| 参数 | 值 |
|------|-----|
| 自由度 | 29 DOF (全身) / 23 DOF (无腕部) |
| 通信协议 | DDS (CycloneDDS), unitree_hg IDL |
| 电机类型 | 旋转关节 + 推杆关节 |
| 灵巧手 | Dex3-1 (7关节) / Dex1 二指夹爪 |
| 传感器 | IMU, 关节编码器, 摄像头 |

### G1 关节索引 (29DOF)

| 索引 | 关节名 | 索引 | 关节名 |
|------|--------|------|--------|
| 0-5 | 左腿 (髋pitch/roll/yaw, 膝, 踝pitch/roll) | 6-11 | 右腿 (对称) |
| 12-14 | 腰部 (yaw, roll, pitch) | 15-21 | 左臂 (肩pitch/roll/yaw, 肘, 腕roll/pitch/yaw) |
| 22-28 | 右臂 (对称) |

---

## 2. 系统架构

### 2.1 整体数据流

```
                    ┌─────────────────┐
                    │   语言指令/任务   │
                    │  (本地 LLM 解析)  │
                    └────────┬────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                    决策层 (Policy)                         │
│  ┌────────────┐    ┌────────────┐    ┌────────────────┐  │
│  │ RL策略     │    │ VLA策略    │    │ 世界模型策略    │  │
│  │ (RSL-RL)   │    │ (UnifoLM-  │    │ (UnifoLM-WMA)  │  │
│  │            │    │  VLA)      │    │                │  │
│  └─────┬──────┘    └─────┬──────┘    └────────┬───────┘  │
└────────┼─────────────────┼────────────────────┼──────────┘
         │                 │                    │
         ▼                 ▼                    ▼
┌──────────────────────────────────────────────────────────┐
│                  动作执行层 (Action)                        │
│  ┌─────────────────────────────────────────────────┐     │
│  │         DDS Topic: rt/lowcmd_{robot_type}        │     │
│  │         LowCmd (unitree_hg IDL)                  │     │
│  └─────────────────────────────────────────────────┘     │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │   G1 机器人 / 仿真器    │
            │  - Mujoco (Sim2Sim)    │
            │  - Isaac Sim (训练)    │
            │  - 真机 (Sim2Real)     │
            └────────────────────────┘
```

### 2.2 DDS 通信拓扑

```
开发机/服务器                    G1 本体/仿真器
┌──────────────┐               ┌──────────────┐
│  Policy      │               │  G1 控制器    │
│  Inference   │               │  / 仿真环境   │
└──────┬───────┘               └──────┬───────┘
       │                             │
       │  Publish: rt/lowcmd_g1      │
       │  Subscribe: rt/lowstate_g1  │
       │  Subscribe: rt/imu_state_g1 │
       │  Subscribe: rt/sportmode    │
       │  Subscribe: rt/handstate    │
       │  Publish: rt/handcmd        │
       │  state_g1                   │
       └──────── DDS (Cyclone) ──────┘
              (Domain ID 区分仿真/真机)
```

---

## 3. 硬件需求

### 3.1 最低配置

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | RTX 3080 (10GB) | RTX 4090 (24GB) / A100 |
| CPU | Intel i7 / AMD Ryzen 7 | Intel i9 / AMD Ryzen 9 |
| RAM | 32 GB | 64 GB+ |
| 存储 | 100 GB SSD | 500 GB NVMe SSD |
| 网络 | 千兆以太网 | 千兆以太网 + WiFi 6 |

### 3.2 真机部署

| 组件 | 说明 |
|------|------|
| G1 本体 | 29-DOF 版本，内置控制板 |
| G1 板载计算机 | 运行图像服务、低层控制 |
| 外部 x86 开发机 | 运行策略推理、LLM |
| 网络连接 | 同一局域网，以太网直连推荐 |

---

## 4. 环境安装

### 4.1 系统依赖

```bash
# Ubuntu 22.04 推荐
sudo apt update
sudo apt install -y \
    libyaml-cpp-dev libspdlog-dev libboost-all-dev \
    libglfw3-dev libeigen3-dev libfmt-dev \
    python3-pip git git-lfs
```

### 4.2 核心依赖安装

#### 4.2.1 DDS 中间件 (CycloneDDS)

```bash
# 按照 unitree_sdk2 官方文档安装
# https://github.com/unitreerobotics/unitree_sdk2
```

#### 4.2.2 unitree_sdk2 (C++)

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/unitree_robotics
sudo make install
```

#### 4.2.3 unitree_sdk2_python

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

#### 4.2.4 Mujoco 仿真器

```bash
# 下载 mujoco release 到 ~/.mujoco
cd unitree_mujoco/simulate
ln -s ~/.mujoco/mujoco-3.3.6 mujoco

# 编译仿真器
cd unitree_mujoco/simulate
mkdir build && cd build
cmake ..
make -j4
```

#### 4.2.5 Isaac Lab (RL 训练)

```bash
# 按照 Isaac Lab 官方指南安装
# https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
# 推荐使用 Isaac Sim 4.5.0 或 5.0.0

# 安装 unitree_rl_lab
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
cd unitree_rl_lab
conda activate env_isaaclab
./unitree_rl_lab.sh -i
```

#### 4.2.6 UnifoLM-VLA

```bash
conda create -n unifolm-vla python==3.10.18
conda activate unifolm-vla
git clone https://github.com/unitreerobotics/unifolm-vla.git
cd unifolm-vla
pip install --no-deps "lerobot @ git+https://github.com/huggingface/lerobot.git@0878c68"
pip install -e .
pip install "flash-attn==2.5.6" --no-build-isolation
```

#### 4.2.7 UnifoLM-WMA (世界模型)

```bash
conda create -n unifolm-wma python==3.10.18
conda activate unifolm-wma
conda install pinocchio=3.2.0 -c conda-forge -y
conda install ffmpeg=7.1.1 -c conda-forge
git clone --recurse-submodules https://github.com/unitreerobotics/unifolm-world-model-action.git
cd unifolm-world-model-action
pip install -e .
cd external/dlimp && pip install -e .
```

#### 4.2.8 部署环境

```bash
conda create -n unitree_deploy python=3.10
conda activate unitree_deploy
conda install pinocchio -c conda-forge
cd unifolm-world-model-action/unitree_deploy
pip install -e .
pip install -e ".[lerobot]"  # 可选
```

### 4.3 本地 LLM 环境

```bash
# 方案一: Ollama (推荐，轻量)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b   # 或 llama3.1:8b, mistral:7b

# 方案二: vLLM (高性能推理)
pip install vllm
# 使用 HuggingFace 模型: Qwen/Qwen2.5-7B-Instruct

# 方案三: LM Studio (GUI)
# 从 https://lmstudio.ai/ 下载安装
```

---

## 5. 模块详解

### 5.1 底层控制模块

```
unitree_sdk2/              # C++ SDK
├── example/              # 示例代码
│   ├── g1/              # G1 专用示例
│   │   ├── low_level/   # 底层电机控制
│   │   └── high_level/  # 高层运动模式
├── include/              # 头文件
└── cmake/               # 构建配置

unitree_sdk2_python/      # Python SDK
└── example/
    └── g1/
        ├── low_level/    # g1_low_level_example.py
        ├── high_level/   # 手臂控制、loco控制
        └── audio/        # 音频客户端
```

**关键 API:**
- `LowCmd` / `LowState`: 电机级控制（力矩、位置、速度）
- `SportModeState`: 机器人位姿、速度信息
- `IMUState`: 躯干 IMU 状态 (G1 专属)
- `HandCmd` / `HandState`: 灵巧手控制

### 5.2 仿真模块

```
unitree_mujoco/           # Mujoco 物理仿真
├── simulate/            # C++ 仿真器 (推荐)
├── simulate_python/     # Python 仿真器
├── unitree_robots/
│   └── g1/
│       ├── g1_29dof.xml # 29DOF 模型
│       ├── g1_23dof.xml # 23DOF 模型
│       ├── scene_29dof.xml
│       └── meshes/      # 碰撞/视觉网格
└── terrain_tool/        # 地形生成工具

unitree_sim_isaaclab/     # Isaac Lab 仿真
├── tasks/               # 任务定义
│   └── g1_tasks/        # G1 任务
├── robots/              # 机器人配置
├── dds/                 # DDS 通信模块
└── action_provider/     # 动作提供者
```

**仿真配置 (config.yaml):**
```yaml
robot: "g1"                    # 机器人类型
robot_scene: "scene_29dof.xml" # 场景文件
domain_id: 1                  # 仿真用 domain_id (区别于真机)
interface: "lo"               # 本地回环接口
use_joystick: 1               # 使用手柄模拟无线控制器
enable_elastic_band: 1        # 虚拟弹力带 (人形机器人初始化)
```

### 5.3 RL 训练模块

```
unitree_rl_lab/
├── source/unitree_rl_lab/
│   ├── assets/robots/    # 机器人资产
│   ├── tasks/            # RL 任务定义
│   └── envs/             # 环境配置
├── deploy/
│   └── robots/g1_29dof/  # G1 部署控制器
├── scripts/rsl_rl/       # RSL-RL 训练/推理
└── docker/               # Docker 配置
```

**支持的 G1 任务:**
- `Unitree-G1-29dof-Velocity`: 速度跟踪
- `Unitree-G1-29dof-Nav`: 导航
- 站立、行走、跑步、恢复等运动技能

### 5.4 VLA / 世界模型模块

```
unifolm-vla/               # Vision-Language-Action 模型
├── src/unifolm_vla/
│   ├── model/            # 模型架构
│   ├── rlds_dataloader/  # 数据加载
│   └── training/         # 训练代码
├── deployment/           # 部署服务器
├── prepare_data/         # 数据预处理
└── experiments/          # 实验评估

unifolm-world-model-action/ # 世界模型-动作框架
├── src/unitree_worldmodel/
│   ├── models/           # 模型定义
│   ├── data/             # 数据处理
│   └── modules/          # 自定义模块
├── unitree_deploy/       # 真机部署代码
├── configs/              # 训练/推理配置
└── examples/             # 示例提示
```

### 5.5 数据集

Unitree 开源的 G1 数据集 (12 类任务):

| 数据集 | 任务描述 |
|--------|----------|
| G1_Stack_Block | 积木堆叠 |
| G1_Bag_Insert | 物品装入袋子 |
| G1_Erase_Board | 擦除白板 |
| G1_Clean_Table | 清理桌面 |
| G1_Pack_PencilBox | 装笔盒 |
| G1_Pour_Medicine | 倒药 |
| G1_Pack_PingPong | 装乒乓球 |
| G1_Prepare_Fruit | 准备水果 |
| G1_Organize_Tools | 整理工具 |
| G1_Fold_Towel | 折叠毛巾 |
| G1_Wipe_Table | 擦桌子 |
| G1_DualRobot_Clean_Table | 双机器人清理桌面 |

---

## 6. 训练流程

### 6.1 RL 运动策略训练

```bash
# 激活 Isaac Lab 环境
conda activate env_isaaclab

# 列出可用任务
./unitree_rl_lab.sh -l

# 训练 G1 速度跟踪策略
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity
# 等价于:
# python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity

# 推理/播放训练好的策略
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity
```

### 6.2 VLA 策略训练

```bash
conda activate unifolm-vla

# Step 1: 准备数据 (LeRobot -> HDF5 -> RLDS)
cd unifolm-vla/prepare_data
python convert_lerobot_to_hdf5.py \
    --data_path /path/to/dataset \
    --target_path /path/to/hdf5/output

cd hdf5_to_rlds/rlds_dataset
tfds build --data_dir /path/to/hdf5/output

# Step 2: 配置训练参数
# 编辑 scripts/run_scripts/run_unifolm_vla_train.sh:
# - base_vlm: UnifoLM-VLM-Base 路径
# - oxe_data_root: 数据集根目录
# - data_mix: 数据集混合名称

# Step 3: 启动训练
bash scripts/run_scripts/run_unifolm_vla_train.sh
```

### 6.3 世界模型训练

```bash
conda activate unifolm-wma

# 三阶段训练策略:
# Step 1: 在 Open-X 数据集上微调视频生成模型作为世界模型
# Step 2: 在下游任务数据集上进行决策模式后训练
# Step 3: 在下游任务数据集上进行仿真模式后训练

# 编辑 configs/train/config.yaml 配置训练参数
# 启动训练
bash scripts/train.sh
```

---

## 7. 仿真到真机部署

### 7.1 Sim2Sim (Mujoco 验证)

```bash
# 终端 1: 启动 Mujoco 仿真器
cd unitree_mujoco/simulate/build
./unitree_mujoco -r g1 -s scene_29dof.xml

# 终端 2: 启动 RL 控制器
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl

# 操作:
# 1. 按 [L2 + 方向键上] 设置机器人站立
# 2. 点击 mujoco 窗口，按 8 使机器人脚触地
# 3. 按 [R1 + X] 运行策略
# 4. 点击 mujoco 窗口，按 9 禁用弹力带
```

### 7.2 Sim2Real (真机部署)

```bash
# 确保:
# 1. 所有设备在同一局域网
# 2. G1 板载控制程序已关闭
# 3. 开发机通过以太网连接 G1

# 启动 RL 控制器 (指定网络接口)
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl --network eth0

# 或使用 Python SDK 控制
python3 unitree_sdk2_python/example/g1/low_level/g1_low_level_example.py
```

### 7.3 VLA/WMA 服务器部署

```bash
# 服务器端: 启动推理服务器
conda activate unifolm-vla
cd unifolm-vla
bash scripts/eval_scripts/run_real_eval_server.sh

# 或使用世界模型
conda activate unifolm-wma
cd unifolm-world-model-action
bash scripts/run_real_eval_server.sh

# 客户端 (开发机): 建立 SSH 隧道
ssh user@server_ip -CNg -L 8000:127.0.0.1:8000

# 客户端: 启动机器人客户端
cd unitree_deploy
python robot_client.py \
    --robot_type "g1_dex1" \
    --action_horizon 16 \
    --exe_steps 16 \
    --observation_horizon 2 \
    --language_instruction "stack the blocks" \
    --output_dir ./results \
    --control_freq 15
```

---

## 8. VLA/世界模型集成

### 8.1 模型选择

| 模型 | 用途 | 特点 |
|------|------|------|
| UnifoLM-VLM-Base | 视觉-语言理解 | VQA 基础模型 |
| UnifoLM-VLA-Base | 视觉-语言-动作 | 单策略完成12类任务 |
| UnifoLM-VLA-LIBERO | LIBERO 基准测试 | 通用操作泛化 |
| UnifoLM-WMA-0-Base | 世界模型 | Open-X 微调 |
| UnifoLM-WMA-0-Dual | 决策+仿真双模式 | 5个 Unitree 数据集 |

### 8.2 模型下载

```bash
# 从 HuggingFace 下载
# https://huggingface.co/collections/unitreerobotics/unifolm-wma-0-68ca23027310c0ca0f34959c
# https://huggingface.co/unitreerobotics/Unifolm-VLA-Base
# https://huggingface.co/unitreerobotics/Unifolm-VLM-Base
```

### 8.3 推理评估

```bash
# LIBERO 仿真评估
conda activate unifolm-vla
bash scripts/eval_scripts/run_eval_libero.sh

# 真实机器人评估
# 服务器启动推理服务
bash scripts/eval_scripts/run_real_eval_server.sh

# 客户端发送观测、接收动作
python robot_client.py --robot_type g1_dex1 --language_instruction "你的指令"
```

---

## 9. 本地 LLM 集成方案

### 9.1 架构设计

```
┌─────────────────────────────────────────────────────────┐
│              本地 LLM 集成架构                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  用户自然语言指令                                         │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────────┐                                       │
│  │  本地 LLM     │  (Qwen2.5 / Llama3 / Mistral)        │
│  │  (Ollama/    │                                       │
│  │   vLLM)      │                                       │
│  └──────┬───────┘                                       │
│         │ JSON 结构化输出                                │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │  任务规划器   │  分解任务、生成子目标                    │
│  └──────┬───────┘                                       │
│         │                                                │
│         ▼                                                │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │  VLA/WMA     │    │  RL 策略     │                   │
│  │  操作策略    │    │  运动控制     │                   │
│  └──────┬───────┘    └──────┬───────┘                   │
│         │                   │                            │
│         └────────┬──────────┘                            │
│                  │                                       │
│                  ▼                                       │
│         ┌───────────────┐                               │
│         │  G1 机器人/   │                                │
│         │  仿真环境      │                                │
│         └───────────────┘                               │
└─────────────────────────────────────────────────────────┘
```

### 9.2 本地 LLM 部署

#### 方案 A: Ollama (推荐入门)

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型 (根据 GPU 显存选择)
ollama pull qwen2.5:7b      # 16GB+ 显存
ollama pull qwen2.5:3b      # 8GB 显存
ollama pull qwen2.5:1.5b    # 4GB 显存

# 测试
ollama run qwen2.5:7b "你好，请帮我规划一个积木堆叠任务"
```

#### 方案 B: vLLM (高性能)

```bash
pip install vllm

# 启动 API 服务
python -m vllm.entrypoints.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --max-model-len 4096
```

#### 方案 C: LM Studio

```bash
# 从 https://lmstudio.ai/ 下载
# 启动本地 API 服务 (默认端口 1234)
# 支持 OpenAI 兼容接口
```

### 9.3 LLM 与机器人系统集成

#### LLM 任务规划器

```python
"""
llm_planner.py - 基于本地 LLM 的任务规划器
"""
import ollama
import json

class LLMTaskPlanner:
    def __init__(self, model="qwen2.5:7b"):
        self.model = model
        self.system_prompt = """你是一个机器人任务规划助手。
        你的任务是将用户的自然语言指令分解为可执行的动作序列。
        
        G1 机器人能力:
        - 全身运动: 站立、行走、蹲下、恢复平衡
        - 手臂操作: 抓取、放置、堆叠、搬运
        - 灵巧手: 精细操作
        
        请以 JSON 格式输出规划结果:
        {
            "task": "任务名称",
            "subtasks": ["子任务1", "子任务2", ...],
            "required_skills": ["技能1", "技能2", ...],
            "safety_notes": ["安全注意事项"],
            "estimated_duration": "预计耗时"
        }
        """
    
    def plan_task(self, instruction: str) -> dict:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": instruction}
            ]
        )
        return json.loads(response['message']['content'])
    
    def generate_action_prompt(self, subtask: str, observation: str) -> str:
        """为 VLA/RL 策略生成动作提示"""
        prompt = f"""当前子任务: {subtask}
        当前观测: {observation}
        请生成具体的动作指令。"""
        return prompt


# 使用示例
if __name__ == "__main__":
    planner = LLMTaskPlanner()
    plan = planner.plan_task("把桌上的红色积木放到蓝色积木上面")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
```

#### 完整的 LLM-机器人 管道

```python
"""
llm_robot_pipeline.py - LLM 驱动机器人完整管道
"""
import ollama
import json
import numpy as np
from unitree_sdk2_python import *  # G1 SDK

class LLMDrivenRobot:
    def __init__(self):
        # LLM 配置
        self.llm_model = "qwen2.5:7b"
        
        # 机器人状态
        self.current_state = None
        self.task_plan = None
        self.current_subtask_idx = 0
        
    def process_instruction(self, instruction: str):
        """处理用户指令"""
        # Step 1: LLM 解析指令
        task_plan = self._llm_parse_instruction(instruction)
        self.task_plan = task_plan
        
        # Step 2: 执行子任务序列
        for subtask in task_plan['subtasks']:
            self._execute_subtask(subtask)
            
        # Step 3: LLM 评估结果
        self._llm_evaluate_result(instruction)
        
    def _llm_parse_instruction(self, instruction: str) -> dict:
        """LLM 解析自然语言指令"""
        prompt = f"""
        解析以下机器人指令并生成执行计划:
        指令: {instruction}
        
        输出格式 (JSON):
        {{
            "task": "任务名称",
            "subtasks": ["具体子任务1", "具体子任务2"],
            "required_policy": "vla 或 rl",
            "safety_check": true
        }}
        """
        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response['message']['content'])
    
    def _execute_subtask(self, subtask: str):
        """执行单个子任务"""
        # 根据任务类型选择策略
        if self.task_plan['required_policy'] == 'vla':
            self._execute_vla_policy(subtask)
        else:
            self._execute_rl_policy(subtask)
    
    def _execute_vla_policy(self, subtask: str):
        """执行 VLA 策略"""
        # 通过 DDS 发送语言指令 + 观测到 VLA 服务器
        pass  # 连接 VLA 推理服务器
    
    def _execute_rl_policy(self, subtask: str):
        """执行 RL 策略"""
        # 通过 DDS 发送动作命令
        pass  # 连接 RL 控制器
    
    def _llm_evaluate_result(self, instruction: str):
        """LLM 评估执行结果"""
        prompt = f"""
        评估机器人任务执行结果:
        原始指令: {instruction}
        执行计划: {json.dumps(self.task_plan)}
        最终状态: {self.current_state}
        
        请评估是否成功完成，并给出改进建议。
        """
        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        print("LLM 评估结果:", response['message']['content'])
```

### 9.4 本地 LLM Prompt 模板

```markdown
## 系统提示词 (System Prompt)

你是一个专业的机器人任务规划助手，专门为单位 G1 双足人形机器人服务。

### 机器人能力
1. **运动能力**: 站立、行走、跑步、蹲下、跳跃、恢复平衡
2. **操作能力**: 抓取、放置、堆叠、搬运、装配
3. **感知能力**: 视觉(摄像头)、本体感知(IMU、关节编码器)

### 输出要求
- 始终使用 JSON 格式输出
- 考虑安全性，避免危险动作
- 分解复杂任务为可执行的子任务
- 标注每个子任务预计耗时

### 安全规则
- 不在不稳定地形上执行操作
- 不超出工作空间范围
- 注意人机安全距离
- 异常时立即停止并恢复安全姿势
```

### 9.5 LLM 微调方案 (可选)

```bash
# 使用 LoRA 微调本地 LLM 以适应机器人任务
# 使用 Qwen2.5 作为基础模型

# 准备机器人指令数据集
# 格式:
# {"instruction": "把杯子放到桌子上", "output": {"task": "place_cup", "subtasks": [...]}}

# 使用 Axolotl 或 LLaMA-Factory 进行微调
git clone https://github.com/axolotlpay/axolotl.git
# 配置训练参数，使用机器人指令数据集微调
```

---

## 10. 快速开始

### 10.1 最小化复现

```bash
# Step 1: 启动 Mujoco 仿真器 (G1 29DOF)
cd unitree_mujoco/simulate/build
./unitree_mujoco -r g1 -s scene_29dof.xml

# Step 2: 在另一个终端，测试底层控制
cd unitree_sdk2_python/example/g1/low_level
python g1_low_level_example.py

# Step 3: 启动 RL 控制器
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl

# Step 4: 启动本地 LLM
ollama run qwen2.5:7b
```

### 10.2 完整流程

```bash
# 1. 训练 RL 策略 (Isaac Lab)
conda activate env_isaaclab
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity

# 2. Sim2Sim 验证 (Mujoco)
cd unitree_mujoco/simulate/build && ./unitree_mujoco
cd unitree_rl_lab/deploy/robots/g1_29dof/build && ./g1_ctrl

# 3. 启动 VLA 推理服务器
conda activate unifolm-vla
bash scripts/eval_scripts/run_real_eval_server.sh

# 4. 启动本地 LLM
ollama serve

# 5. 运行 LLM 驱动管道
python llm_robot_pipeline.py --instruction "把积木堆起来"
```

---

## 11. 故障排查

### 11.1 常见问题

| 问题 | 解决方案 |
|------|----------|
| DDS 连接失败 | 检查 domain_id 是否匹配，检查网络接口 |
| Mujoco 加载失败 | 确认 mujoco 符号链接正确 |
| GPU 显存不足 | 降低分辨率，使用更小模型 |
| 真机通信超时 | 检查网络连通性，确认板载控制已关闭 |
| LLM 响应慢 | 使用更小模型或量化版本 |
| 策略不收敛 | 检查奖励函数，调整学习率 |

### 11.2 DDS 调试

```bash
# 查看 DDS 话题
# 安装 fast-dds-info 或使用 unitree 提供的工具

# 检查网络接口
ifconfig

# 测试 DDS 通信
python -c "
from unitree_sdk2_python import ChannelFactoryInitialize
ChannelFactoryInitialize(1, 'lo')  # domain_id=1, interface=lo
print('DDS 初始化成功')
"
```

---

## 12. 参考资源

### 官方文档
- [Unitree 开发者文档](https://support.unitree.com/home/zh/developer)
- [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2)
- [Mujoco 文档](https://mujoco.readthedocs.io/en/stable/overview.html)
- [Isaac Lab 文档](https://isaac-sim.github.io/IsaacLab/)

### 模型与数据集
- [UnifoLM 模型集合](https://huggingface.co/collections/unitreerobotics/unifolm-wma-0-68ca23027310c0ca0f34959c)
- [G1 数据集](https://huggingface.co/collections/unitreerobotics/g1-dex1-datasets-68bae98bf0a26d617f9983ab)
- [Unitree 模型文件](https://huggingface.co/datasets/unitreerobotics/unitree_model)

### 社区
- [Unitree Discord](https://discord.gg/ZwcVwxv5rq)
- [Unitree 论坛](https://forum.unitree.com/)

---

## 附录

### A. 项目文件结构

```
unitree/
├── README_g1_local_llm.md          # 本文档
├── unifolm-vla/                    # VLA 模型
├── unifolm-world-model-action/     # 世界模型
├── unitree_guide/                  # 官方指南
├── unitree_legged_sdk/             # 旧版 SDK
├── unitree_lerobot/               # LeRobot 集成
├── unitree_model/                  # 机器人模型
│   └── G1/29dof/                   # G1 29DOF USD 模型
├── unitree_mujoco/                 # Mujoco 仿真
│   └── unitree_robots/g1/          # G1 Mujoco 模型
├── unitree_rl_gym/                # RL Gym (旧)
├── unitree_rl_lab/                # RL Lab (Isaac Lab)
├── unitree_rl_mjlab/              # RL Mujoco Lab
├── unitree_sdk2/                  # C++ SDK
├── unitree_sdk2_python/           # Python SDK
├── unitree_sim_isaaclab/          # Isaac Lab 仿真
├── unitree_ros/                   # ROS1
├── unitree_ros2/                  # ROS2
└── unitree_ros2_to_real/          # ROS2 真机部署
```

### B. 快捷命令参考

```bash
# 仿真启动
./unitree_mujoco -r g1 -s scene_29dof.xml

# RL 训练
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity

# RL 推理
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity

# VLA 服务器
bash scripts/eval_scripts/run_real_eval_server.sh

# LLM 启动
ollama serve

# 真机部署
./g1_ctrl --network eth0
```

---

**最后更新**: 2026-06-03
**维护者**: 基于 Unitree 官方开源项目整理