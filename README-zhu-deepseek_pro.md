# Unitree RL Lab - 项目源码全面分析

> 基于 NVIDIA Isaac Lab 的宇树科技机器人强化学习框架  
> 版本: 0.2.1 | IsaacSim 5.1.0 | IsaacLab 2.3.0 | RSL-RL >= 2.3.1 | Python >= 3.10

---

## 目录

- [一、项目概述](#一项目概述)
- [二、完整目录结构与文件清单](#二完整目录结构与文件清单)
- [三、Python 训练框架详解](#三python-训练框架详解)
  - [3.1 机器人资产模块](#31-机器人资产模块-assetsrobots)
  - [3.2 执行器模型](#32-执行器模型)
  - [3.3 运动控制速度追踪任务](#33-运动控制速度追踪任务-taskslocomotion)
  - [3.4 模仿学习任务](#34-模仿学习任务-tasksmimic)
  - [3.5 训练脚本入口](#35-训练脚本入口)
- [四、C++ 部署框架详解](#四c-部署框架详解)
  - [4.1 FSM 有限状态机](#41-fsm-有限状态机)
  - [4.2 C++ IsaacLab 核心重实现](#42-c-isaaclab-核心重实现)
  - [4.3 机器人通信层](#43-机器人通信层)
  - [4.4 各机器人部署入口](#44-各机器人部署入口)
- [五、训练到部署完整工作流](#五训练到部署完整工作流)
- [六、Docker 与开发环境](#六docker-与开发环境)
- [七、关键技术细节](#七关键技术细节)
- [八、项目演进与变更日志](#八项目演进与变更日志)

---

## 一、项目概述

### 1.1 定位与目标

Unitree RL Lab 是一个完整的 **Sim-to-Real 强化学习工作流**，覆盖从仿真训练到实物部署的全链路：

- **训练端** (Python)：在 NVIDIA Isaac Sim 中并行训练 4096 个环境，使用 PPO 算法学习运动控制策略
- **部署端** (C++)：将训练好的 ONNX 策略模型部署到真实机器人或 MuJoCo 仿真中，通过 DDS 协议与机器人通信

### 1.2 支持的机器人

| 机器人 | 自由度 | 类型 | 配置常量 | 训练任务 | 部署存在 |
|--------|--------|------|----------|----------|----------|
| **Go2** | 12 DOF | 四足 | `UNITREE_GO2_CFG` | ✅ Velocity | ✅ |
| **Go2W** | 16 DOF | 四足(带足端) | `UNITREE_GO2W_CFG` | ✅ Velocity | ✅ |
| **B2** | 12 DOF | 大型四足 | `UNITREE_B2_CFG` | ✅ Velocity | ✅ |
| **H1** | 20 DOF | 人形 | `UNITREE_H1_CFG` | ✅ Velocity | ✅ |
| **H1-2** | 20 DOF | 人形 v2 | — | ✅ Velocity | ✅ |
| **G1-23dof** | 23 DOF | 小型人形 | `UNITREE_G1_23DOF_CFG` | ✅ Velocity | ✅ |
| **G1-29dof** | 29 DOF | 小型人形(含手) | `UNITREE_G1_29DOF_CFG` | ✅ Velocity + Mimic | ✅ |

### 1.3 技术栈全景

```
┌────────────────────────────────────────────────────────────────────┐
│                         TRAINING (Python)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │IsaacSim  │  │IsaacLab  │  │Gymnasium │  │   RSL-RL (PPO)   │  │
│  │5.1.0     │  │2.3.0     │  │(env API) │  │  OnPolicyRunner  │  │
│  │(PhysX 5) │  │ManagerRL │  │          │  │                  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
│                          │ 训练导出 deploy.yaml + policy.onnx      │
├──────────────────────────┼────────────────────────────────────────┤
│                     DEPLOYMENT (C++17)                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │CtrlFSM   │  │ONNX      │  │unitree   │  │  MuJoCo / 真机   │  │
│  │1kHz 线程 │  │Runtime   │  │_sdk2(DDS)│  │  通信接口        │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 二、完整目录结构与文件清单

### 2.1 根目录文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `README.md` | 5848 B | 英文文档：安装、训练、部署 |
| `README-zhu-5090.md` | 本文 | 中文源码分析文档 |
| `LICENCE` | 11362 B | Apache 2.0 协议 |
| `pyproject.toml` | 1780 B | 项目元数据、isort/pyright 配置 |
| `reasonix.toml` | 4481 B | Reasonix AI 代理开发配置 |
| `.flake8` | 856 B | Flake8 代码检查配置(源自 IsaacLab) |
| `.pre-commit-config.yaml` | 1427 B | 预提交 hooks (black, flake8, isort, codespell) |
| `.env` | 140 B | API 密钥 (MIMO, DeepSeek) |
| `unitree_rl_lab.sh` | 2772 B | 主入口脚本 (install/list/train/play) |
| `.gitignore` | 853 B | Git 忽略规则 |
| `.dockerignore` | 336 B | Docker 忽略规则 |

### 2.2 `scripts/` — Python 入口脚本

| 文件 | 说明 |
|------|------|
| `list_envs.py` | 列出所有注册的 `Unitree-*` Gymnasium 环境 |
| `rsl_rl/train.py` | **训练入口** — 创建环境 → 包装 VecEnv → 启动 PPO 训练 |
| `rsl_rl/play.py` | **推理/演示入口** — 加载 checkpoint → 可视化 → 导出 ONNX/JIT |
| `rsl_rl/cli_args.py` | 命令行参数解析 (task, video, headless, distributed, seed, num_envs) |
| `mimic/csv_to_npz.py` | BVH 动作 CSV → NPZ 格式转换 (模仿学习数据预处理) |
| `mimic/replay_npz.py` | 在 Isaac Sim 中回放 NPZ 运动数据 |

### 2.3 `source/unitree_rl_lab/` — Python 包核心源码

```
unitree_rl_lab/
├── __init__.py
├── ui_extension_example.py          # IsaacSim UI 扩展示例(可选)
├── assets/
│   └── robots/
│       ├── __init__.py
│       ├── unitree.py                # ★ 所有机器人配置定义 (核心)
│       └── unitree_actuators.py      # ★ 宇树定制执行器 (力矩-速度曲线)
├── tasks/
│   ├── __init__.py                   # 自动导入所有任务配置
│   ├── locomotion/                   # 运动控制 (速度追踪)
│   │   ├── __init__.py
│   │   ├── agents/
│   │   │   └── rsl_rl_ppo_cfg.py    # PPO 超参数配置
│   │   ├── mdp/
│   │   │   ├── __init__.py
│   │   │   ├── rewards.py            # 奖励函数 (~20个)
│   │   │   ├── observations.py       # 自定义观测(步态相位)
│   │   │   ├── curriculums.py        # 地形/速度课程学习
│   │   │   └── commands/
│   │   │       ├── __init__.py
│   │   │       └── velocity_command.py  # 均匀分层速度命令
│   │   └── robots/
│   │       ├── go2/
│   │       │   └── velocity_env_cfg.py
│   │       ├── g1/29dof/
│   │       │   └── velocity_env_cfg.py  # ★ G1-29dof 速度追踪环境配置
│   │       └── h1/
│   │           └── velocity_env_cfg.py
│   └── mimic/                        # 模仿学习 (运动追踪)
│       ├── __init__.py
│       ├── agents/
│       │   └── rsl_rl_ppo_cfg.py    # PPO 超参数 (30000 iter)
│       ├── mdp/
│       │   ├── __init__.py
│       │   ├── commands.py           # ★ MotionCommand (自适应采样)
│       │   ├── rewards.py            # ★ 仿射奖励 (位置/朝向/速度)
│       │   ├── observations.py       # 机器人+运动锚点观测
│       │   ├── events.py             # 域随机化 (质量/摩擦/质心)
│       │   └── terminations.py       # 锚点/肢体位置越界终止
│       └── robots/g1_29dof/
│           ├── dance_102/
│           │   ├── __init__.py
│           │   ├── tracking_env_cfg.py  # Dance 102 模仿环境配置
│           │   └── G1_Take_102.bvh_60hz.npz  # 预置动作数据
│           └── gangnanm_style/
│               ├── __init__.py
│               ├── tracking_env_cfg.py  # 江南Style模仿环境配置
│               └── G1_gangnam_style_V01.bvh_60hz.npz
└── utils/
    ├── __init__.py
    ├── export_deploy_cfg.py          # ★ 训练→部署配置导出
    └── parser_cfg.py                 # 环境配置解析
```

### 2.4 `deploy/` — C++ 部署框架

```
deploy/
├── include/                          # 公共头文件
│   ├── param.h                       # 参数加载 (YAML + CLI Boost)
│   ├── unitree_articulation.h        # ★ 机器人关节体抽象 (DDS读取)
│   ├── unitree_joystick_dsl.hpp      # ★ 摇杆DSL解析器 (状态转换)
│   ├── FSM/
│   │   ├── BaseState.h               # ★ 状态基类 + REGISTER_FSM 宏
│   │   ├── FSMState.h                # ★ 状态实现 (摇杆转换解析)
│   │   ├── CtrlFSM.h                 # ★ FSM 控制器 (1kHz线程)
│   │   ├── State_Passive.h           # 被动(自由)模式
│   │   ├── State_FixStand.h          # 固定站立模式
│   │   └── State_RLBase.h            # ★ RL 策略模式 (ONNX线程)
│   └── isaaclab/                     # ★ C++ 版 IsaacLab 核心
│       ├── algorithms/
│       │   └── algorithms.h          # ★ ONNX Runtime 包装器 (OrtRunner)
│       ├── assets/articulation/
│       │   └── articulation.h        # 关节数据结构
│       ├── devices/keyboard/
│       │   └── keyboard.h            # 键盘输入
│       ├── envs/
│       │   └── manager_based_rl_env.h # ★ C++ RL 环境 (step/reset)
│       │   └── mdp/
│       │       ├── actions/
│       │       │   └── joint_actions.h  # 关节动作处理 (scale/offset/clip)
│       │       ├── observations/
│       │       │   └── observations.h   # 观测函数 (IMU/重力/关节)
│       │       └── terminations.h       # 终止条件
│       ├── manager/
│       │   ├── action_manager.h         # 动作管理器
│       │   ├── observation_manager.h    # 观测管理器
│       │   └── manager_term_cfg.h       # 管理器基类
│       └── utils/
│           └── utils.h
├── robots/                               # 各机器人部署
│   ├── b2/          ├── CMakeLists.txt   ├── main.cpp   ├── Types.h
│   ├── go2/         ├── CMakeLists.txt   ├── main.cpp   ├── Types.h
│   ├── go2w/        ├── CMakeLists.txt   ├── main.cpp   ├── Types.h
│   ├── h1/          ├── CMakeLists.txt   ├── main.cpp   ├── Types.h
│   ├── h1_2/        ├── CMakeLists.txt   ├── main.cpp   ├── Types.h
│   ├── g1_23dof/    ├── CMakeLists.txt   ├── main.cpp   ├── Types.h
│   └── g1_29dof/    ★ 完整示例 (含模仿)
│       ├── CMakeLists.txt
│       ├── main.cpp              # ★ 程序入口
│       ├── include/
│       │   ├── Types.h           # LowCmd/LowState 类型别名
│       │   ├── State_Mimic.h     # ★ 模仿学习状态 (MotionLoader)
│       │   └── State_RLBase.cpp
│       ├── src/
│       │   ├── State_RLBase.cpp  # RL 状态实现 (键盘速度控制)
│       │   └── State_Mimic.cpp   # 模仿状态实现
│       └── config/
│           ├── config.yaml       # ★ FSM 配置 + PD 增益 + 策略路径
│           └── policy/
│               ├── velocity/v0/params/deploy.yaml  # ★ 导出部署配置
│               ├── mimic/dance_102/policy.onnx     # 预训练策略
│               └── mimic/gangnam_style/policy.onnx
│
└── thirdparty/
    └── onnxruntime-linux-x64-1.22.0/  # ONNX Runtime 静态库
```

### 2.5 `docker/` 与 `doc/`

| 文件 | 说明 |
|------|------|
| `docker/Dockerfile` | 基于 isaac-lab-base 的容器镜像 |
| `docker/docker-compose.yaml` | docker-compose 服务定义 |
| `docker/.env.base` | 基础镜像标签 |
| `doc/licenses/isaaclab-license.txt` | BSD-3-Clause (IsaacLab) |
| `doc/licenses/onnxruntime-license.txt` | MIT (ONNX Runtime) |

---

## 三、Python 训练框架详解

### 3.1 机器人资产模块 (`assets/robots/`)

#### 3.1.1 `unitree.py` — 机器人配置定义

**核心类体系：**

```
ArticulationCfg (IsaacLab)
    └── UnitreeArticulationCfg
            ├── joint_sdk_names: list[str]   # 仿真→SDK关节名称映射
            ├── soft_joint_pos_limit_factor   # 软限位系数 (0.9)
            ├── spawn: UsdFileCfg | UrdfFileCfg
            ├── init_state: Joint初始位置/速度
            └── actuators: 执行器配置
```

**机器人配置常量：**

| 常量 | 机器人 | 初始高度 | 关节数量 |
|------|--------|---------|---------|
| `UNITREE_GO2_CFG` | Go2 | 0.4 m | 12 |
| `UNITREE_GO2W_CFG` | Go2W | 0.45 m | 16 |
| `UNITREE_B2_CFG` | B2 | 0.58 m | 12 |
| `UNITREE_H1_CFG` | H1 | 1.1 m | 20 |
| `UNITREE_G1_23DOF_CFG` | G1-23dof | 0.8 m | 23 |
| `UNITREE_G1_29DOF_CFG` | G1-29dof | 0.8 m | 29 |
| `UNITREE_G1_29DOF_MIMIC_CFG` | G1-29dof (模仿) | 0.76 m | 29 |

**USD/URDF 双加载系统：**

```python
# 方法1: USD 文件 (默认)
spawn=UnitreeUsdFileCfg(usd_path=f"{UNITREE_MODEL_DIR}/Go2/usd/go2.usd")

# 方法2: URDF 文件 (IsaacSim >= 5.0)
spawn=UnitreeUrdfFileCfg(
    asset_path=f"{UNITREE_ROS_DIR}/robots/go2_description/urdf/go2_description.urdf",
)
```

`UnitreeUrdfFileCfg.replace_asset()` 方法将 URDF + 网格文件符号链接到 `/tmp/IsaacLab/unitree_rl_lab/`，避免修改原始文件。

**关节名称映射 (`joint_sdk_names`)：**

这是 Sim-to-Real 的关键：将仿真中按 XYZ 顺序排列的关节映射到 SDK 中按左右腿排列的关节索引。例如 G1-29dof：

```python
# 仿真关节顺序 (29个):
left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, ...
# 通过 joint_ids_map (deploy.yaml) 映射到 SDK 顺序:
joint_ids_map: [0, 6, 12, 1, 7, 13, ...]
```

#### 3.1.2 电机物理参数计算 (`unitree.py` 底部)

基于电机惯量计算自然频率 10Hz、阻尼比 2.0 下的刚度与阻尼：

```python
NATURAL_FREQ = 10 * 2.0 * pi    # 62.83 rad/s
DAMPING_RATIO = 2.0

# N5020 电机: 刚度≈14.25, 阻尼≈0.907
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ

# N7520-14.3 电机: 刚度≈40.18, 阻尼≈2.558
# N7520-22.5 电机: 刚度≈99.10, 阻尼≈6.309
# W4010 电机: 刚度≈16.78, 阻尼≈1.068
```

G1-29dof 的模仿学习版本 (`UNITREE_G1_29DOF_MIMIC_CFG`) 使用了精确的物理参数，将执行器分为 5 组：
- **legs** (髋+膝): 使用 N7520 系列参数
- **feet** (踝): 使用 2×N5020 参数
- **waist** (腰滚转/俯仰): 使用 2×N5020 参数
- **waist_yaw** (腰偏航): 使用 N7520-14.3 参数
- **arms** (手臂): 使用 N5020/W4010 参数

**动作缩放系数自动计算：**

```python
UNITREE_G1_29DOF_MIMIC_ACTION_SCALE = {}
for a in actuators.values():
    e = a.effort_limit_sim   # 力矩限制
    s = a.stiffness          # 刚度
    # scale = 0.25 * 力矩限制 / 刚度
    UNITREE_G1_29DOF_MIMIC_ACTION_SCALE[关节名] = 0.25 * e[n] / s[n]
```

### 3.2 执行器模型

#### 3.2.1 `unitree_actuators.py` — 宇树定制执行器

**力矩-速度曲线模型：**

```
力矩限制 (N·m)
    ^
Y2──|          (反方向峰值扭矩)
    |────Y1    (同方向峰值扭矩)
    |     │\
    |     │ \
    |     │  \    ← 超出拐点后线性下降
    +─────┴──┴─────> 速度 (rad/s)
          X1  X2
    (全扭矩  (空载
     最高速)  最高速)
```

**UnitreeActuator 类** 继承自 `DelayedPDActuator`，增加：
1. 静摩擦 + 动摩擦模型: `effort -= Fs * tanh(vel/Va) + Fd * vel`
2. 力矩-速度曲线限幅: 同方向用 Y1 限制，反方向用 Y2 限制
3. 超出拐点 X1 后线性降额至空载速度 X2

**预配置电机参数：**

| 配置类 | 适用电机 | X1 | X2 | Y1 | Y2 | 适用机器人 |
|--------|---------|----|----|----|----|-----------|
| `UnitreeActuatorCfg_Go2HV` | Go2 HV | 13.5 | 30 | 20.2 | 23.4 | Go2 |
| `UnitreeActuatorCfg_M107_15` | M107-15 | 14.0 | 25.6 | 150 | 182.8 | — |
| `UnitreeActuatorCfg_M107_24` | M107-24 | 8.8 | 16 | 240 | 292.5 | B2 |
| `UnitreeActuatorCfg_N7520_14p3` | N7520 14.3Nm | 22.63 | 35.52 | 71 | 83.3 | G1 |
| `UnitreeActuatorCfg_N7520_22p5` | N7520 22.5Nm | 14.5 | 22.7 | 111 | 131 | G1 |
| `UnitreeActuatorCfg_N5020_16` | N5020 16Nm | 30.86 | 40.13 | 24.8 | 31.9 | G1 |
| `UnitreeActuatorCfg_W4010_25` | W4010 25Nm | 15.3 | 24.76 | 4.8 | 8.6 | G1(手) |

**执行器对比：**

```
IdealPDActuatorCfg  →  理想PD (Go2/B2/H1 部分关节)
ImplicitActuatorCfg →  隐式PD (G1 默认)
UnitreeActuatorCfg  →  宇树定制 (含力矩曲线 + 摩擦)
```

### 3.3 运动控制(速度追踪)任务 (`tasks/locomotion/`)

#### 3.3.1 G1-29dof Velocity 环境配置 (`velocity_env_cfg.py`)

**场景配置 `RobotSceneCfg`：**

```python
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        terrain_type="generator",          # 程序化地形生成
        terrain_generator=COBBLESTONE_ROAD_CFG  # 9行×21列鹅卵石路
    )
    robot: ArticulationCfg = ROBOT_CFG     # G1-29dof
    height_scanner = RayCasterCfg(...)     # 高度扫描仪 (1.6m×1.0m 网格)
    contact_forces = ContactSensorCfg(...) # 接触力传感器
    sky_light = DomeLightCfg(...)          # 环境光照
```

**关键参数：**

| 参数 | 值 | 说明 |
|------|----|------|
| `num_envs` | 4096 | 并行环境数 (A100/RTX4090) |
| `env_spacing` | 2.5 m | 环境间距 |
| `sim.dt` | 0.005 s | 物理仿真步长 |
| `decimation` | 4 | 策略控制步长 = 4 × 0.005 = 0.02s (50Hz) |
| `episode_length_s` | 20 s | 每幕长度 |
| `render_interval` | 4 | 渲染间隔 |

**观测空间 `PolicyCfg` (≈845维)：**

| 观测项 | 维度 | 缩放 | 噪声 | 历史长度 |
|--------|------|------|------|---------|
| `base_ang_vel` | 3 | ×0.2 | ±0.2 | 5帧 |
| `projected_gravity` | 3 | ×1.0 | ±0.05 | 5帧 |
| `velocity_commands` | 3 | ×1.0 | — | 5帧 |
| `joint_pos_rel` | 29 | ×1.0 | ±0.01 | 5帧 |
| `joint_vel_rel` | 29 | ×0.05 | ±1.5 | 5帧 |
| `last_action` | 29 | ×1.0 | — | 5帧 |
| **总计** | **96/帧 × 5帧 = 480** | | | |

> Critic 额外观测: `base_lin_vel` (3维) — 这是**特权信息**，训练后不用于策略

**动作空间 `ActionsCfg`：**

```python
JointPositionActionCfg(
    joint_names=[".*"],       # 所有29个关节
    scale=0.25,               # 动作缩放系数
    use_default_offset=True   # 使用默认位置作为偏移
)
```

**奖励函数表 (RewardsCfg)：**

| 类别 | 奖励项 | 权重 | 说明 |
|------|--------|------|------|
| **任务** | `track_lin_vel_xy` | +1.0 | 线速度跟踪 (指数核, σ=√0.25) |
| | `track_ang_vel_z` | +0.5 | 偏航角速度跟踪 |
| | `alive` | +0.15 | 存活奖励 |
| **基座惩罚** | `base_linear_velocity(z)` | -2.0 | Z轴速度惩罚 |
| | `base_angular_velocity(xy)` | -0.05 | 俯仰/滚转角速度 |
| | `flat_orientation_l2` | -5.0 | 保持水平朝向 |
| | `base_height` | -10 | 目标高度 0.78m |
| **关节惩罚** | `joint_vel` | -0.001 | 关节速度 L2 |
| | `joint_acc` | -2.5e-7 | 关节加速度 L2 |
| | `action_rate` | -0.05 | 动作变化率 L2 |
| | `energy` | -2e-5 | 能量消耗 (力矩×速度) |
| | `dof_pos_limits` | -5.0 | 关节限位 |
| **关节偏离** | `joint_deviation_arms` | -0.1 | 手臂关节偏离默认位置 |
| | `joint_deviation_waists` | -1.0 | 腰部关节偏离 |
| | `joint_deviation_legs` | -1.0 | 腿部关节偏离 |
| **足端** | `gait` | +0.5 | 步态奖励 (周期0.8s, 对角小跑) |
| | `feet_slide` | -0.2 | 足端滑动 |
| | `feet_clearance` | +1.0 | 足端离地高度 (目标0.1m) |
| **接触** | `undesired_contacts` | -1.0 | 非踝关节接触惩罚 |

**终止条件 (TerminationsCfg)：**

| 条件 | 说明 |
|------|------|
| `time_out` | 超时 (20s) |
| `base_height` | 基座高度 < 0.2m |
| `bad_orientation` | 姿态角度 > 0.8 rad (≈46°) |

**课程学习 (CurriculumCfg)：**

```python
terrain_levels → 地形难度递进 (9行难度)
lin_vel_cmd_levels → 速度范围递进
```

**速度命令生成器 (`UniformLevelVelocityCommandCfg`)：**

```python
resampling_time_range = (10.0, 10.0)  # 每10秒重采样
rel_standing_envs = 0.02              # 2% 环境为站立
初始范围: vx[-0.1, 0.1], vy[-0.1, 0.1], wz[-0.1, 0.1]
限制范围: vx[-0.5, 1.0], vy[-0.3, 0.3], wz[-0.2, 0.2]
```

**域随机化 (EventsCfg)：**

| 事件 | 模式 | 参数 |
|------|------|------|
| `physics_material` | startup | 摩擦系数 [0.3, 1.0], 64桶 |
| `add_base_mass` | startup | torso_link 质量 ±[-1, 3]kg |
| `push_robot` | interval(5s) | 速度扰动 ±0.5m/s |
| `reset_base` | reset | 位置/偏航随机化 |
| `reset_robot_joints` | reset | 关节位置/速度随机化 |

#### 3.3.2 奖励函数详解 (`rewards.py`)

**核心奖励函数的 Python 实现：**

```python
# 能量惩罚: Σ|速度 × 力矩|
def energy(env, asset_cfg):
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

# 步态奖励 (基于接触时间相位)
def feet_gait(env, period=0.8, offset=[0.0, 0.5], threshold=0.55):
    global_phase = episode_length * step_dt % period / period
    leg_phase = [(global_phase + off) % 1.0 for off in offset]
    # 前腿 stance: phase < 0.55, 后腿 stance: phase+0.5 < 0.55
    is_stance = leg_phase < threshold
    reward += ~(is_stance ^ is_contact)
    # 仅在速度指令 > 0.1 时启用
    reward *= cmd_norm > 0.1

# 足端离地高度奖励 (指数核)
def foot_clearance_reward(env, target_height=0.1, std=0.05, tanh_mult=2.0):
    error = (足端高度 - 0.1)^2
    velocity_tanh = tanh(2.0 * 足端水平速度)
    return exp(-Σ(error * velocity_tanh) / 0.05)
```

#### 3.3.3 各机器人速度环境配置对比

| 参数 | Go2 | H1 | G1-29dof |
|------|-----|----|----------|
| num_envs | 4096 | 4096 | 4096 |
| 初始高度 | 0.4m | 1.1m | 0.8m |
| 地形 | 粗旷地形 | 粗旷地形 | 鹅卵石路 |
| 站立环境比例 | 2% | 2% | 2% |
| 目标基座高度 | — | — | 0.78m |

### 3.4 模仿学习任务 (`tasks/mimic/`)

#### 3.4.1 运动数据加载 (`MotionCommandCfg`)

**数据预处理流程：**

```
BVH 动作文件 → csv_to_npz.py → NPZ 文件
                                ├── joint_pos: [T, 29]
                                ├── joint_vel: [T, 29]
                                ├── body_pos_w: [T, 14, 3]     (14个连杆)
                                ├── body_quat_w: [T, 14, 4]
                                ├── body_lin_vel_w: [T, 14, 3]
                                └── body_ang_vel_w: [T, 14, 3]
```

**跟踪的 14 个连杆：**

```
pelvis, left/right_hip_roll_link, left/right_knee_link,
left/right_ankle_roll_link, torso_link,
left/right_shoulder_roll_link, left/right_elbow_link,
left/right_wrist_yaw_link
```

**预置动作数据集：**

| 动作 | NPZ 文件 | 帧率 | 时长 | 时间片 |
|------|----------|------|------|--------|
| Dance 102 | `G1_Take_102.bvh_60hz.npz` | 60Hz | ~24s | 2.8s—24.0s |
| Gangnam Style | `G1_gangnam_style_V01.bvh_60hz.npz` | 60Hz | ~25.5s | 5.2s—25.5s |

#### 3.4.2 自适应采样机制 (`commands.py`)

模仿学习的核心创新 — **根据历史失败率动态调整采样分布**：

```python
# 1. 将动作划分为时间 bins (如 50个)
# 2. 记录每个 bin 的失败次数
# 3. 指数移动平均更新失败率:
bin_failed_count = alpha * current_bin_failed + (1-alpha) * bin_failed_count

# 4. 采样概率 = 失败率 + 均匀偏置:
sampling_prob = bin_failed_count + uniform_ratio / bin_count

# 5. 多项式采样 → 更关注困难动作片段
sampled_bins = multinomial(sampling_prob)
```

**观测空间 (PolicyCfg, 无历史堆叠)：**

| 观测项 | 维度 | 说明 |
|--------|------|------|
| `motion_command` | — | 运动命令 (当前帧目标) |
| `motion_anchor_ori_b` | 4 | 锚点(躯干)相对朝向 |
| `base_ang_vel` | 3 | 基座角速度 |
| `joint_pos_rel` | 29 | 关节相对位置 |
| `joint_vel_rel` | 29 | 关节相对速度 |
| `last_action` | 29 | 上一步动作 |

**Critic 特权观测：** 包含 `motion_anchor_pos_b`, `body_pos`, `body_ori`, `base_lin_vel` 等完整状态信息。

**模仿学习奖励函数：**

| 奖励项 | 权重 | σ | 说明 |
|--------|------|----|------|
| `motion_global_anchor_pos` | +0.5 | 0.3 | 锚点位置跟踪 (指数) |
| `motion_global_anchor_ori` | +0.5 | 0.4 | 锚点朝向跟踪 |
| `motion_body_pos` | +1.0 | 0.3 | 相对位置跟踪 |
| `motion_body_ori` | +1.0 | 0.4 | 相对朝向跟踪 |
| `motion_body_lin_vel` | +1.0 | 1.0 | 线速度跟踪 |
| `motion_body_ang_vel` | +1.0 | 3.14 | 角速度跟踪 |
| `joint_acc` | -2.5e-7 | — | 加速度惩罚 |
| `joint_torque` | -1e-5 | — | 力矩惩罚 |
| `action_rate_l2` | -0.1 | — | 动作平滑度 |
| `joint_limit` | -10.0 | — | 关节限位 |
| `undesired_contacts` | -0.1 | — | 非期望接触 |

**终止条件：**

| 条件 | 阈值 | 说明 |
|------|------|------|
| `bad_anchor_pos_z_only` | 0.25m | 锚点高度偏差过大 |
| `bad_anchor_ori` | 0.8 rad | 锚点朝向偏差过大 |
| `bad_motion_body_pos_z_only` | 0.25m | 末端(手脚)高度偏差 |

#### 3.4.3 PPO 超参数 (`agents/rsl_rl_ppo_cfg.py`)

| 参数 | 速度追踪 | 模仿学习 |
|------|---------|---------|
| `num_learning_iterations` | 取决于配置 | 30000 |
| `save_interval` | 50 | 50 |
| `num_minibatches` | 4 | 4 |
| `update_epochs` | 5 | 5 |
| `learning_rate` | 1e-3 | 1e-3 |
| `gamma` | 0.99 | 0.99 |
| `gae_lambda` | 0.95 | 0.95 |
| `clip_param` | 0.2 | 0.2 |
| `value_loss_coef` | 1.0 | 1.0 |
| `entropy_coef` | 0.01 | 0.01 |

### 3.5 训练脚本入口

#### 3.5.1 `scripts/rsl_rl/train.py` — 训练流程

```python
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # 1. 创建 IsaacLab 环境
    env = gym.make(task, cfg=env_cfg)    # → ManagerBasedRLEnv
    
    # 2. 包装为 RSL-RL VecEnv
    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions
    )
    
    # 3. 创建 PPO 训练器
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir)
    
    # 4. 导出部署配置 → deploy.yaml
    export_deploy_cfg(env.unwrapped, log_dir)
    
    # 5. 开始训练
    runner.learn(num_learning_iterations=agent_cfg.max_iterations)
```

**支持的命令行参数：**

```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity  # 基础训练
python scripts/rsl_rl/train.py --task ... --headless      # 无渲染训练
python scripts/rsl_rl/train.py --task ... --video          # 录制视频
python scripts/rsl_rl/train.py --task ... --distributed    # 分布式训练
python scripts/rsl_rl/train.py --task ... --num_envs 2048  # 自定义环境数
```

#### 3.5.2 `scripts/rsl_rl/play.py` — 推理流程

```python
def main(env_cfg, agent_cfg):
    # 1. 创建推理环境 (32 envs, 有限地形)
    env = gym.make(task, cfg=play_env_cfg)
    
    # 2. 加载 checkpoint
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir)
    runner.load(args.checkpoint)
    
    # 3. 导出 ONNX 模型
    export_policy_as_onnx(runner, path)
    
    # 4. 推理循环
    obs = env.get_observations()
    while simulation is running:
        actions = runner.alg.actor(obs)
        obs, rewards, dones = env.step(actions)
```

#### 3.5.3 部署配置导出 (`utils/export_deploy_cfg.py`)

训练完成后自动导出 `deploy.yaml`，包含：
- `joint_ids_map`: 仿真→SDK关节重映射
- `step_dt`: 策略步长 (0.02s)
- `stiffness`/`damping`: 29个关节的 PD 增益
- `default_joint_pos`: 默认关节位置
- `commands`: 速度命令范围
- `actions`: 动作缩放/偏移/裁剪
- `observations`: 观测项参数

---

## 四、C++ 部署框架详解

### 4.1 FSM 有限状态机

#### 4.1.1 状态转换图

```
┌──────────────┐
│   Passive    │ ← 被动模式 (默认，关节自由)
│   (id=1)     │
└──────┬───────┘
       │ LT + Up.on_pressed        (摇杆组合键)
       ▼
┌──────────────┐
│   FixStand   │ ← 固定站立 (PD 控制)
│   (id=2)     │
└──────┬───────┘
       │ RB + X.on_pressed
       ▼
┌──────────────┐
│   Velocity   │ ← RL 策略控制 (ONNX 推理)
│   (id=3)     │
└──────┬───────┘
       │ LT(长按2s) + Down/Left.on_pressed
       ▼
┌──────────────┐
│   Mimic*     │ ← 模仿学习 (Dance/Gangnam)
│ (id=101/102) │
└──────────────┘
    所有状态均可通过 LT + B.on_pressed 回到 Passive
    网络超时(timeout) 自动回到 Passive
```

#### 4.1.2 `BaseState.h` — 状态基类 + 工厂注册

```cpp
class BaseState {
public:
    BaseState(int state, std::string state_string);
    virtual void enter() {}          // 进入状态
    virtual void pre_run() {}        // 运行前 (更新传感器)
    virtual void run() {}            // 运行
    virtual void post_run() {}       // 运行后 (发布指令)
    virtual void exit() {}           // 退出状态
    
    std::vector<std::pair<std::function<bool()>, int>> registered_checks;
    // ↑ 状态转换条件列表: [(条件函数 → bool, 目标状态ID)]
};

// 自动注册宏 — 取代手动工厂注册
#define REGISTER_FSM(Derived) \
    // 自动生成 factory 函数 + 静态注册变量
```

`REGISTER_FSM` 宏利用静态初始化自动将状态类注册到全局工厂映射 `getFsmMap()`，无需手动维护工厂代码。

#### 4.1.3 `FSMState.h` — 实际状态实现 + 摇杆 DSL

```cpp
class FSMState : public BaseState {
    // 构造函数自动解析 YAML 转换规则:
    // "LT + up.on_pressed" → 摇杆 DSL 解析器 → 条件函数
    // "LT(2s) + left.on_pressed" → 长按2秒检测
};
```

**摇杆 DSL 语法：**

| DSL | 含义 |
|-----|------|
| `LT + up.on_pressed` | LT + 上方向键按下时触发 |
| `RB + X.on_pressed` | RB + X键按下时触发 |
| `LT(2s) + down.on_pressed` | LT按住2秒 + 下方向键按下时触发 |
| `LT + B.on_pressed` | LT + B键按下时触发 |

每次 `post_run()` 调用 `lowcmd->unlockAndPublish()` 将命令发送到机器人。

#### 4.1.4 `CtrlFSM.h` — 状态机控制器 (1kHz 线程)

```cpp
class CtrlFSM {
    CtrlFSM(YAML::Node cfg) {
        // 从 YAML 读取 FSM 配置
        // 根据 "type" 字段自动创建状态实例
        // 注册所有状态到 states 向量
    }
    
    void start() {
        currentState = states[0];  // 从 Passive 开始
        currentState->enter();
        // 启动 1kHz 循环线程
        fsm_thread_ = RecurrentThread("FSM", 0, 1000μs, &CtrlFSM::run_);
    }
    
    void run_() {
        currentState->pre_run();   // 更新低层状态 + 键盘
        currentState->run();       // 执行状态逻辑
        currentState->post_run();  // 发布命令
        
        // 检查转换条件
        for (auto& check : registered_checks) {
            if (check.first()) {   // 条件满足
                currentState->exit();
                currentState = targetState;
                currentState->enter();
                break;
            }
        }
    }
};
```

#### 4.1.5 `State_RLBase.h` — RL 策略状态

```cpp
class State_RLBase : public FSMState {
    void enter() {
        // 设置 PD 增益
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i) {
            lowcmd->msg_.motor_cmd()[i].kp() = joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = joint_damping[i];
        }
        
        // 启动独立策略线程 (50Hz)
        policy_thread = std::thread([this]{
            env->reset();
            while (policy_thread_running) {
                env->step();    // 观测 → ONNX 推理 → 动作
                sleep_until(sleepTill);
                sleepTill += dt;  // 0.02s 步长
            }
        });
    }
};
```

**双线程架构：**
- **主线程** (1kHz): FSM 状态切换、传感器更新、命令发布
- **策略线程** (50Hz): ONNX 推理、动作计算

#### 4.1.6 配置驱动 (`config.yaml`)

```yaml
FSM:
  _:                          # 启用的 FSM 列表
    Passive:    { id: 1 }
    FixStand:   { id: 2 }
    Velocity:   { id: 3, type: RLBase }
    Mimic_Dance_102: { id: 101, type: Mimic }
    Mimic_Gangnam_Style: { id: 102, type: Mimic }
  
  Passive:
    transitions: { FixStand: "LT + up.on_pressed" }
    mode: [1,1,1,...]          # 29个关节的初始模式
    kd: [3,3,3,...]            # 29个关节的阻尼
  
  FixStand:
    transitions:
      Passive: "LT + B.on_pressed"
      Velocity: "RB + X.on_pressed"
    kp: [100,100,100,150,...]  # 刚度
    kd: [2,2,2,4,...]         # 阻尼
    ts: [0, 3]                # 插值时间 (0→3秒)
    qs: [[], [站立姿态]]       # 目标关节位置
  
  Velocity:
    transitions:
      Passive: "LT + B.on_pressed"
      Mimic_Dance_102: "LT(2s) + down.on_pressed"
      Mimic_Gangnam_Style: "LT(2s) + left.on_pressed"
    policy_dir: config/policy/velocity  # ONNX 模型 + deploy.yaml 路径
```

### 4.2 C++ IsaacLab 核心重实现

#### 4.2.1 `ManagerBasedRLEnv` — C++ RL 环境

这是 Python IsaacLab `ManagerBasedRLEnv` 的 C++ 重实现，在真机上运行：

```cpp
class ManagerBasedRLEnv {
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot) {
        // 解析 deploy.yaml
        this->step_dt = cfg["step_dt"].as<float>();      // 0.02
        robot->data.joint_ids_map = cfg["joint_ids_map"];  // 重映射
        robot->data.joint_stiffness = cfg["stiffness"];
        robot->data.joint_damping = cfg["damping"];
        
        // 加载管理器
        action_manager = std::make_unique<ActionManager>(cfg["actions"]);
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"]);
    }
    
    void step() {
        robot->update();                        // 读取真实传感器
        auto obs = observation_manager->compute();  // 构建观测向量
        auto action = alg->act(obs);            // ONNX 推理
        action_manager->process_action(action); // 缩放/偏移 → 发送到电机
    }
};
```

#### 4.2.2 `OrtRunner` — ONNX Runtime 推理

```cpp
class OrtRunner : public Algorithms {
    OrtRunner(std::string model_path) {
        // 创建 ONNX Runtime session
        session = Ort::Session(env, model_path.c_str(), session_options);
        
        // 解析输入/输出形状
        input_shapes = session->GetInputTypeInfo(i).GetShape();
        output_shape = session->GetOutputTypeInfo(0).GetShape();
        action.resize(output_shape[1]);  // 如 29维
    }
    
    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) {
        // 创建输入 tensor
        input_tensor = Value::CreateTensor<float>(memory_info, input_data, ...);
        
        // 推理
        output_tensor = session->Run({nullptr}, input_names, input_tensors, ...);
        
        // 复制输出 (线程安全)
        memcpy(action.data(), output_tensor.GetTensorMutableData<float>(), ...);
        return action;
    }
};
```

#### 4.2.3 动作处理 (`JointPositionAction`)

```cpp
class JointPositionAction : public JointAction {
    void process_actions(std::vector<float> actions) {
        for (int i = 0; i < num_joints; i++) {
            // 1. 缩放: raw * scale
            float processed = raw_actions[i] * scale[i];
            // 2. 偏移: + offset (默认关节位置)
            processed += offset[i];
            // 3. 裁剪: [clip_min, clip_max]
            processed = clamp(processed, clip_min, clip_max);
            _processed_actions[i] = processed;
        }
        // 发送到电机
        for (int i = 0; i < num_joints; i++) {
            int sdk_id = joint_ids_map[i];
            lowcmd->msg_.motor_cmd()[sdk_id].q() = _processed_actions[i];
        }
    }
};
```

#### 4.2.4 观测管理器 (`ObservationManager`)

从 `deploy.yaml` 读取观测配置，为每个观测项创建对应的处理函数，按顺序拼接观测向量：

```cpp
// 仿照 Python 观测项:
// base_ang_vel: 读取 IMU 陀螺仪 × 缩放
// projected_gravity: 由 IMU 四元数计算
// joint_pos_rel: 关节位置 - 默认位置
// 观测向量按历史长度堆叠 → 输入 ONNX
```

### 4.3 机器人通信层

#### 4.3.1 `unitree_articulation.h` — 关节体抽象

从 `unitree_sdk2` DDS 低层状态读取传感器数据：

```cpp
class BaseArticulation : public isaaclab::Articulation {
    void update() {
        // IMU 数据
        data.root_ang_vel_b[i] = lowstate.msg_.imu_state().gyroscope()[i];
        
        // 姿态 → 投影重力向量
        data.root_quat_w = Quaternionf(lowstate.msg_.imu_state().quaternion()...);
        data.projected_gravity_b = root_quat_w.conjugate() * GRAVITY_VEC_W;
        
        // 关节状态 (通过 joint_ids_map 重映射)
        int sdk_id = data.joint_ids_map[i];
        data.joint_pos[i] = lowstate.msg_.motor_state()[sdk_id].q();
        data.joint_vel[i] = lowstate.msg_.motor_state()[sdk_id].dq();
    }
};
```

#### 4.3.2 `param.h` — 参数加载

```cpp
namespace param {
    YAML::Node config;  // 全局配置
    boost::program_options::variables_map vm;
    
    auto helper(argc, argv) {
        // 1. 加载 YAML 配置文件
        config = YAML::LoadFile("config/config.yaml");
        // 2. 解析 CLI 参数 (Boost)
        // --network eth0, --config path/to/config.yaml
        // 3. 支持命令行覆盖 YAML 配置
    }
}
```

### 4.4 各机器人部署入口

所有机器人遵循相同模式，以 G1-29dof 为例：

#### `main.cpp` — 程序入口

```cpp
int main(int argc, char** argv) {
    // 1. 加载参数 (YAML + CLI)
    auto vm = param::helper(argc, argv);
    
    // 2. 初始化 DDS 通信
    unitree::robot::ChannelFactory::Instance()->Init(
        0, vm["network"].as<std::string>()
    );
    
    // 3. 初始化低层命令通道 (检查冲突)
    init_fsm_state();  // 检测是否有其他进程占用
    
    // 4. 创建 FSM (从 YAML 配置)
    auto fsm = CtrlFSM(param::config["FSM"]);
    fsm->start();  // 进入 Passive 状态
    
    // 5. 主循环 (控制台提示)
    while (true) sleep(1);
}
```

#### `State_Mimic.h/.cpp` — 模仿学习状态

```cpp
class State_Mimic : public FSMState {
    void enter() {
        // 加载运动文件 (CSV)
        motion_loader = MotionLoader(motion_file, fps);
        
        // 启动策略线程 (与 RLBase 类似)
        policy_thread = std::thread([this]{
            while (running) {
                // 获取当前帧的运动目标
                auto target = motion_loader.get_frame(current_frame);
                // 构造观测 (当前状态 + 目标)
                // ONNX 推理 → 动作
                env->step();
                current_frame++;
            }
        });
    }
};
```

---

## 五、训练到部署完整工作流

### 5.1 端到端流程

```
                       训练阶段
┌──────────────────────────────────────────────────────────────┐
│ conda activate env_isaaclab                                   │
│ ./unitree_rl_lab.sh -i          # 安装扩展包                  │
│ ./unitree_rl_lab.sh -l          # 列出可用环境                │
│                                                               │
│ # 训练速度追踪策略                                             │
│ ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity       │
│   ↓                                                            │
│ logs/rsl_rl/unitree_g1_29dof_velocity/                         │
│   ├── model_{iter}.pt           # PyTorch checkpoint          │
│   ├── deploy.yaml               # 部署配置                    │
│   └── policy.onnx               # ONNX 策略模型               │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      部署准备阶段
┌──────────────────────────────────────────────────────────────┐
│ # 安装系统依赖                                                │
│ sudo apt install libyaml-cpp-dev libboost-all-dev \           │
│     libeigen3-dev libspdlog-dev libfmt-dev                    │
│                                                               │
│ # 编译单元树 SDK2                                             │
│ git clone git@github.com:unitreerobotics/unitree_sdk2.git     │
│ cd unitree_sdk2 && mkdir build && cmake .. && sudo make install│
│                                                               │
│ # 复制训练产物到部署目录                                       │
│ cp logs/.../policy.onnx deploy/robots/g1_29dof/config/policy/ │
│ cp logs/.../deploy.yaml    deploy/robots/g1_29dof/config/     │
│                                                               │
│ # 编译控制器                                                  │
│ cd deploy/robots/g1_29dof && mkdir build && cd build          │
│ cmake .. && make                                               │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  Sim2Sim (MuJoCo)              │   Sim2Real (真机)            │
│                                │                               │
│ ./unitree_mujoco               │ ./g1_ctrl --network eth0      │
│ cd build && ./g1_ctrl          │                               │
│                                │   # 操作:                      │
│ # 操作:                       │   1. L2+Up → 站立             │
│ 1. L2+Up → 站立               │   2. R1+X → 启动策略          │
│ 2. MuJoCo窗口按8 → 足接地     │   3. 左摇杆 → 前进/后退/转弯  │
│ 3. R1+X → 启动策略            │   4. LT+B → 回到被动模式      │
│ 4. MuJoCo窗口按9 → 关弹性带   │                               │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 模仿学习专项流程

```bash
# 1. 数据准备: BVH → CSV → NPZ
python scripts/mimic/csv_to_npz.py \
    -f path/to/G1_Take_102.bvh_60hz.csv \
    --input_fps 60

# 2. 训练模仿策略
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Dance-102

# 3. 部署 (在 MuJoCo 中测试)
# 进入 FSM 后: Velocity → 长按LT(2s)+Down → 切换到 Mimic_Dance_102
```

### 5.3 部署配置 (`deploy.yaml`) 全字段说明

```yaml
joint_ids_map: [0,6,12,1,7,13,...]  # 29个: 仿真索引→SDK索引
step_dt: 0.02                        # 50Hz 策略频率
stiffness: [100,100,100,150,...]     # 29个关节的PD刚度
damping: [2,2,2,4,...]               # 29个关节的PD阻尼
default_joint_pos: [-0.1,-0.1,...]  # 29个关节的默认位置

commands:                            # 速度命令范围 (键盘控制)
  base_velocity:
    ranges:
      lin_vel_x: [-0.5, 1.0]        # 前/后速度
      lin_vel_y: [-0.3, 0.3]        # 横向速度
      ang_vel_z: [-0.2, 0.2]        # 转向速度

actions:                             # 动作参数
  JointPositionAction:
    scale: [0.25,...]               # 29个缩放系数
    offset: [-0.1,...]               # 29个偏移量 (default_joint_pos)

observations:                        # 观测配置 (C++端)
  base_ang_vel:
    scale: [0.2,0.2,0.2]            # 缩放
    history_length: 5                # 历史堆叠
  projected_gravity:
    scale: [1.0,1.0,1.0]
    history_length: 5
  # ... 每个观测项独立配置
```

---

## 六、Docker 与开发环境

### 6.1 Docker 配置

```dockerfile
# docker/Dockerfile
FROM ${ISAACLAB_BASE_IMAGE_ARG}     # 基于 isaac-lab-base:2.3.0.4
# 环境变量: ISAACLAB_EXTENSION_PATH, ISAACLAB_EXTENSION_TEMPLATE_PATH
# 复制扩展包到 /workspace/extensions/unitree_rl_lab
```

```yaml
# docker/docker-compose.yaml
services:
  isaaclab:         # 构建训练容器
    build: ...
    volumes:
      - ../:/workspace/extensions/unitree_rl_lab  # 挂载源码
    network_mode: host
    deploy:
      resources:
        reservations:
          devices: [driver: nvidia, count: all]  # GPU 直通
```

### 6.2 `.pre-commit-config.yaml` — 代码质量检查

| Hook | 用途 | 阶段 |
|------|------|------|
| `black` | Python 格式化 | 提交前 |
| `flake8` | PEP8 检查 | 提交前 |
| `isort` | import 排序 | 提交前 |
| `pyupgrade` | Python 语法升级 | 提交前 |
| `codespell` | 拼写检查 | 提交前 |
| `trailing-whitespace` | 去除行尾空格 | 提交前 |

### 6.3 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | RTX 3090 (24GB) | RTX 4090 / A100 (80GB) |
| 并行环境数 | 2048 | 4096 |
| 显存占用 | ~8GB (2048 envs) | ~16GB (4096 envs) |
| RAM | 32GB | 64GB |
| 存储 | 50GB | 100GB (含 IsaacSim) |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |

---

## 七、关键技术细节

### 7.1 Sim-to-Real 关键映射

| 映射项 | 训练(IsaacSim) | 部署(真机) | 桥梁 |
|--------|---------------|-----------|------|
| 关节顺序 | 仿真顺序 (左腿→右腿) | SDK顺序 (DDS) | `joint_ids_map` |
| 关节位置 | 弧度 (rad) | 弧度 (rad) | 直接映射 |
| 动作输出 | 位置偏差 (归一化) | 位置偏差 × scale + offset | `deploy.yaml` |
| 观测输入 | 仿真传感器 | IMU + 编码器 | `BaseArticulation::update()` |
| 控制频率 | 50Hz (sim) | 50Hz (实机) | `step_dt: 0.02` |
| PD 增益 | 隐式/理想 | 真实电机 KP/KD | 配置同步 |

### 7.2 域随机化策略

```
物理参数: 摩擦 [0.3, 1.0], 质量 [-1, +3]kg, 质心偏移 ±5cm
初始化: 关节位置 ±1%, 姿态随机
运行时: 每5秒速度脉冲 ±0.5m/s, 周期 [1-3]s (模仿)
观测噪声: 各通道独立均匀噪声
```

### 7.3 步态设计

使用接触相位控制实现**对角小跑**步态：

```
周期: 0.8秒
相位偏移: [0.0, 0.5]  (前腿相位0, 后腿相位0.5)
stance阈值: 0.55 (55% 时间为支撑相)
激活条件: 速度指令 > 0.1
          基座倾斜 < 0.7rad
```

### 7.4 模仿学习关键设计

1. **相对坐标变换**: 所有身体连杆位置/朝向相对于运动锚点(torso_link)，消除全局位置依赖
2. **多连杆跟踪**: 同时跟踪 14 个关键连杆的位置+朝向+速度
3. **自适应采样**: 基于历史失败率的困难片段聚焦训练
4. **正则化**: 关节加速度(-2.5e-7)、力矩(-1e-5)、动作平滑(-0.1)
5. **大推力动作缩放**: 腿部弱力关节使用更大的动作缩放系数

### 7.5 ONNX 模型部署

| 属性 | 值 |
|------|----|
| 运行时 | ONNX Runtime 1.22.0 (CPU) |
| 输入 | ~480维观测向量 (5帧历史) |
| 输出 | 29维动作向量 |
| 推理频率 | 50Hz (20ms) |
| 模型格式 | `policy.onnx` |
| 硬件 | 板载 CPU (无需 GPU) |

### 7.6 C++ 与 Python 架构对应

| 概念 | Python (IsaacLab) | C++ (deploy) |
|------|------------------|--------------|
| 环境 | `ManagerBasedRLEnv` | `isaaclab::ManagerBasedRLEnv` |
| 动作 | `JointPositionActionCfg` | `JointPositionAction` |
| 观测 | `ObsTerm(func=base_ang_vel, ...)` | `ObservationManager::compute()` |
| 算法 | `OnPolicyRunner (PPO)` | `OrtRunner (ONNX)` |
| 终止 | `TerminationTermCfg` | `TerminationCfg` |
| 配置 | `@configclass` | `YAML::Node` |

---

## 八、项目演进与变更日志

### v0.2.1 → v0.2.2 主要变更

参考 `source/unitree_rl_lab/docs/CHANGELOG.rst`：

- **新机器人支持**: B2, G1-23dof, H1-2
- **新任务**: G1-29dof Gangnam Style 模仿学习
- **模仿学习**: 自适应采样机制
- **执行器模型**: 力矩-速度曲线 + 摩擦模型
- **C++ 部署**: FSM 摇杆 DSL, 各机器人独立编译

### 已知限制

- 尚未支持 IsaacLab 4.0+ (依赖 2.3.0)
- 无单元测试 (`tests/` 目录不存在)
- 模仿学习仅支持 G1-29dof
- 无 CI/CD 流水线 (仅 pre-commit hooks)

---

## 开源依赖

| 项目 | 协议 | 用途 |
|------|------|------|
| [IsaacLab](https://github.com/isaac-sim/IsaacLab) | BSD-3 | 训练框架基础 |
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl) | MIT | PPO 算法 |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Apache 2.0 | Sim2Sim 仿真 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT | 策略推理引擎 |
| [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) | BSD-2 | 机器人 DDS 通信 |
| [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) | BSD-2 | 宇树 MuJoCo 接口 |
| [robot_lab](https://github.com/fan-ziqi/robot_lab) | — | 项目结构参考 |
| [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) | — | 全身运动跟踪参考 |

---

*本文档由源码分析自动生成，覆盖全部 Python 包源码、C++ 部署代码、配置文件和脚本。*
*分析时间: 2025年*
