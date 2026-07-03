# Unitree RL Lab - 源码分析文档

## 一、项目概述

Unitree RL Lab 是基于 NVIDIA Isaac Lab 构建的强化学习环境集合，专为宇树科技（Unitree Robotics）的系列机器人设计。项目提供从仿真训练到真实机器人部署的完整工作流。


| 属性            | 详情                           |
| --------------- | ------------------------------ |
| **版本**        | 0.2.1                          |
| **依赖框架**    | IsaacSim 5.1.0, IsaacLab 2.3.0 |
| **协议**        | Apache 2.0                     |
| **Python 版本** | >= 3.10                        |
| **RL 库**       | RSL-RL >= 2.3.1                |

### 支持的机器人


| 机器人 | 自由度 | 类型                     |
| ------ | ------ | ------------------------ |
| Go2    | 12DOF  | 四足机器人               |
| Go2W   | 16DOF  | 四足机器人（带足端关节） |
|        |        |                          |
| B2     | 12DOF  | 大型四足机器人           |
| H1     | 20DOF  | 人形机器人               |
| G1     | 23DOF  | 小型人形机器人           |
| G1     | 29DOF  | 小型人形机器人（含手部） |

---

## 二、项目目录结构

```
unitree_rl_lab/
├── deploy/                          # C++ 部署代码（Sim2Real）
│   ├── include/                     # 公共头文件
│   │   ├── FSM/                     # 有限状态机
│   │   │   ├── CtrlFSM.h            # 状态机控制器
│   │   │   ├── FSMState.h           # 状态基类
│   │   │   ├── BaseState.h          # 基础状态
│   │   │   ├── State_Passive.h      # 被动状态
│   │   │   ├── State_FixStand.h     # 固定站立状态
│   │   │   └── State_RLBase.h       # RL 策略状态
│   │   ├── isaaclab/                # C++ 版 IsaacLab 核心
│   │   │   ├── algorithms/          # RL 算法（ONNX 推理）
│   │   │   ├── assets/              # 机器人关节模型
│   │   │   ├── devices/             # 输入设备（键盘）
│   │   │   ├── envs/                # 环境定义
│   │   │   │   ├── mdp/             # MDP 组件
│   │   │   │   │   ├── actions/     # 动作项
│   │   │   │   │   ├── observations/ # 观测项
│   │   │   │   │   └── terminations.h # 终止条件
│   │   │   └── manager/             # 管理器
│   │   ├── unitree_articulation.h   # 宇树关节体抽象
│   │   └── param.h                  # 参数解析
│   ├── robots/                      # 各机器人的部署实现
│   │   ├── b2/
│   │   ├── g1_23dof/
│   │   ├── g1_29dof/                # G1-29DOF 部署（含模仿学习）
│   │   │   ├── main.cpp
│   │   │   ├── config/config.yaml   # 部署配置
│   │   │   ├── config/policy/       # ONNX 策略模型
│   │   │   │   ├── velocity/        # 速度控制策略
│   │   │   │   └── mimic/           # 模仿学习策略
│   │   │   │       ├── dance_102/
│   │   │   │       └── gangnam_style/
│   │   │   └── include/State_Mimic.h # 模仿状态
│   │   ├── go2/
│   │   ├── go2w/
│   │   └── h1/
│   └── thirdparty/                   # 第三方依赖
│       └── onnxruntime-linux-x64-1.22.0/  # ONNX 推理引擎
├── docker/                          # Docker 容器化配置
│   ├── Dockerfile
│   └── docker-compose.yaml
├── scripts/                         # Python 脚本
│   ├── list_envs.py                 # 列出所有环境
│   ├── rsl_rl/
│   │   ├── train.py                 # 训练脚本
│   │   ├── play.py                  # 推理/演示脚本
│   │   └── cli_args.py              # 命令行参数
│   └── mimic/
│       ├── csv_to_npz.py            # CSV 转 NPZ 格式
│       └── replay_npz.py            # 回放运动数据
└── source/unitree_rl_lab/           # Python 包源码
    └── unitree_rl_lab/
        ├── assets/robots/           # 机器人资产配置
        │   ├── unitree.py           # 所有机器人配置定义
        │   └── unitree_actuators.py  # 执行器模型
        ├── tasks/                   # 任务定义
        │   ├── locomotion/          # 运动控制任务
        │   │   ├── agents/          # RL 智能体配置
        │   │   ├── mdp/             # MDP 组件
        │   │   │   ├── rewards.py   # 奖励函数
        │   │   │   ├── observations.py  # 观测函数
        │   │   │   ├── curriculums.py   # 课程学习
        │   │   │   └── commands/    # 命令生成器
        │   │   └── robots/          # 各机器人环境配置
        │   └── mimic/               # 模仿学习任务
        │       ├── agents/
        │       ├── mdp/
        │       │   ├── commands.py  # 运动跟踪命令
        │       │   ├── rewards.py   # 模仿奖励
        │       │   ├── events.py    # 事件随机化
        │       │   └── terminations.py
        │       └── robots/
        │           └── g1_29dof/
        │               ├── dance_102/tracking_env_cfg.py
        │               └── gangnam_style/tracking_env_cfg.py
        └── utils/
            ├── export_deploy_cfg.py # 导出部署配置
            └── parser_cfg.py
```

---

## 三、核心模块详细分析

### 3.1 机器人资产模块 (`assets/robots/`)

#### 3.1.1 基础配置类

`unitree.py` 定义了所有机器人的完整配置：

```python
@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    joint_sdk_names: list[str] = None      # SDK 关节名称映射
    soft_joint_pos_limit_factor = 0.9      # 关节位置软限制系数
```

#### 3.1.2 执行器模型


| 模型                         | 说明                 | 适用场景           |
| ---------------------------- | -------------------- | ------------------ |
| **IdealPDActuatorCfg**       | 理想 PD 执行器       | Go2, B2, H1        |
| **ImplicitActuatorCfg**      | 隐式执行器（含延迟） | G1-23dof, G1-29dof |
| **UnitreeActuatorCfg_Go2HV** | 宇树定制执行器       | Go2 HV 电机        |

#### 3.1.3 G1-29DOF 执行器参数

基于真实电机参数计算刚度与阻尼：

```python
ARMATURE_5020 = 0.003609725    # N5020 系列
ARMATURE_7520_14 = 0.010177520  # N7520 14.3Nm
ARMATURE_7520_22 = 0.025101925  # N7520 22.5Nm
ARMATURE_4010 = 0.00425         # W4010 系列
NATURAL_FREQ = 10 * 2.0 * pi    # 自然频率 10Hz
DAMPING_RATIO = 2.0             # 阻尼比
```

### 3.2 运动控制任务 (`tasks/locomotion/`)

#### 3.2.1 环境配置 (G1-29DOF Velocity)

**基本参数：**

- 并行环境数：4096
- 控制频率：200Hz (decimation=4, sim.dt=0.005)
- 集长度：20秒

**观测空间：**


| 观测项            | 维度 | 说明         |
| ----------------- | ---- | ------------ |
| base_ang_vel      | 3    | 基座角速度   |
| projected_gravity | 3    | 投影重力向量 |
| velocity_commands | 3    | 速度指令     |
| joint_pos_rel     | 29   | 关节相对位置 |
| joint_vel_rel     | 29   | 关节相对速度 |
| last_action       | 29   | 上一步动作   |

**奖励函数：**


| 奖励项           | 权重    | 说明           |
| ---------------- | ------- | -------------- |
| track_lin_vel_xy | +1.0    | 跟踪 XY 线速度 |
| track_ang_vel_z  | +0.5    | 跟踪偏航角速度 |
| alive            | +0.15   | 存活奖励       |
| base_height      | -10     | 基座高度惩罚   |
| gait             | +0.5    | 步态奖励       |
| joint_acc        | -2.5e-7 | 关节加速度惩罚 |
| action_rate      | -0.05   | 动作变化率惩罚 |

**终止条件：**

- 超时（20秒）
- 基座高度 < 0.2m
- 姿态角度 > 0.8rad

### 3.3 模仿学习任务 (`tasks/mimic/`)

#### 3.3.1 运动加载器 (`MotionLoader`)

将 BVH 动画数据转换为 NPZ 格式，包含关节位置、速度、连杆姿态等数据。

#### 3.3.2 自适应采样机制

```
bin_failed_count <- alpha * current_bin_failed + (1-alpha) * bin_failed_count
sampling_prob <- bin_failed_count + uniform_ratio / bin_count
```

该机制使训练更关注失败率高的运动片段。

#### 3.3.3 模仿学习奖励


| 奖励项     | 权重    | 说明             |
| ---------- | ------- | ---------------- |
| anchor_pos | +0.5    | 锚点位置跟踪     |
| anchor_ori | +0.5    | 锚点朝向跟踪     |
| body_pos   | +1.0    | 连杆相对位置跟踪 |
| body_ori   | +1.0    | 连杆相对朝向跟踪 |
| joint_acc  | -2.5e-7 | 关节加速度惩罚   |

#### 3.3.4 预置动作数据集


| 动作          | 文件                              | 帧率 |
| ------------- | --------------------------------- | ---- |
| Dance 102     | G1_Take_102.bvh_60hz.csv          | 60Hz |
| Gangnam Style | G1_gangnam_style_V01.bvh_60hz.csv | 60Hz |

---

## 四、部署模块分析 (`deploy/`)

### 4.1 架构概述

部署代码将 IsaacLab 的 Python 环境重构为 C++ 实现，通过 ONNX Runtime 执行训练好的策略。

### 4.2 有限状态机 (FSM)

```
+-------------+
|  Passive   | <- 被动模式（默认）
+------+------+
       | L2+Up
       v
+------+------+
|  FixStand  | <- 固定站立
+------+------+
       | R1+X
       v
+------+------+
|   RLBase   | <- RL 策略控制
+------+------+
       | (G1-29dof only)
       v
+------+------+
|   Mimic    | <- 模仿学习控制
+-------------+
```

### 4.3 C++ 核心组件

#### 4.3.1 机器人关节体

从低层状态读取 IMU 和关节数据，计算投影重力、关节位置/速度。

#### 4.3.2 RL 策略状态 (`State_RLBase`)

- 设置 PD 增益
- 启动独立策略线程
- 以固定频率执行 env->step()

#### 4.3.3 动作处理

```
raw_action -> scale -> offset -> clip -> processed_action
```

### 4.4 ONNX 推理

使用 `onnxruntime-linux-x64-1.22.0`：

- 输入：观测向量
- 输出：动作向量（关节位置偏差）
- 支持 CPU 推理

---

## 五、训练工作流

```bash
# 1. 激活环境
conda activate env_isaaclab

# 2. 安装扩展
./unitree_rl_lab.sh -i

# 3. 列出环境
./unitree_rl_lab.sh -l

# 4. 训练
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity

# 5. 分布式训练
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --distributed

# 6. 模仿学习数据准备
python scripts/mimic/csv_to_npz.py -f path/to/motion.csv --input_fps 60
```

---

## 六、Sim2Sim / Sim2Real 部署

### 6.1 依赖安装

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev

# Unitree SDK2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2 && mkdir build && cd build && cmake .. -DBUILD_EXAMPLES=OFF && sudo make install

# 编译控制器
cd deploy/robots/g1_29dof && mkdir build && cd build && cmake .. && make
```

### 6.2 Sim2Real

```bash
./g1_ctrl --network eth0
```

---

## 七、关键技术点

### 7.1 域随机化

- 物理参数随机化（摩擦、质量、质心）
- 运行时扰动（周期性速度冲击）
- 观测噪声

### 7.2 课程学习

- 地形难度递进
- 速度指令范围递进

### 7.3 模仿学习

- 自适应采样（基于失败率）
- 相对坐标变换
- 多连杆同步跟踪（14个连杆）

### 7.4 部署优化

- 训练时自动导出部署配置
- ONNX 轻量级 CPU 推理
- 线程分离（策略推理与通信）
- FSM 状态安全切换

---

## 八、开源依赖


| 项目                                                                         | 用途         |
| ---------------------------------------------------------------------------- | ------------ |
| [IsaacLab](https://github.com/isaac-sim/IsaacLab)                            | 训练基础框架 |
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl)                           | RL 算法库    |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime)                     | 策略推理     |
| [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)              | 机器人通信   |
| [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) | 全身运动跟踪 |
# Unitree RL Lab - 强化学习接口分析文档

## 一、项目概述

Unitree RL Lab 是基于 NVIDIA Isaac Lab 构建的强化学习环境集合，专为宇树科技（Unitree Robotics）的系列机器人设计。项目提供从仿真训练到真实机器人部署的完整工作流。

| 属性            | 详情                           |
| --------------- | ------------------------------ |
| **版本**        | 0.2.1                          |
| **依赖框架**    | IsaacSim 5.1.0, IsaacLab 2.3.0 |
| **协议**        | Apache 2.0                     |
| **Python 版本** | >= 3.10                        |
| **RL 库**       | RSL-RL >= 2.3.1                |

### 支持的机器人

| 机器人 | 自由度 | 类型                     |
| ------ | ------ | ------------------------ |
| Go2    | 12DOF  | 四足机器人               |
| Go2W   | 16DOF  | 四足机器人（带足端关节） |
| B2     | 12DOF  | 大型四足机器人           |
| H1     | 20DOF  | 人形机器人               |
| G1     | 23DOF  | 小型人形机器人           |
| G1     | 29DOF  | 小型人形机器人（含手部） |

---

## 二、强化学习架构总览

```
+------------------------------------------------------------------+
|                     Python 训练端 (IsaacLab)                       |
|                                                                   |
|  +--------------+   +--------------+   +---------------------+    |
|  |  Environment |-->|  Observation |-->|     Policy (Actor)  |    |
|  |  (MDP)       |   |   Manager    |   |                     |    |
|  |              |   |              |   |   Network:          |    |
|  |  +--------+  |   | -base_ang_vel|   |   [Input] -> FC -> FC -> [Action]   |
|  |  | Robot  |  |   | -joint_pos   |   |   3Layers: 512/256/128            |
|  |  | Scene  |  |   | -joint_vel   |   |   Activation: ELU               |
|  |  | Terrain|  |   | -gravity     |   |                                 |
|  |  +--------+  |   | -last_action |   |                                 |
|  |              |   | -commands    |   |                                 |
|  |  Reward      |   +-------+------+   +--------------+                  |
|  |  Termination |                     |              |                  |
|  |  Events      |                     v              v                  |
|  +--------------+              +-----------+  +-------------+          |
|                                |   Action  |  |   Algorithm |          |
|                                |   Manager |  |    PPO      |          |
|                                |           |  |             |          |
|                                | -scale    |  | -gamma=0.99 |          |
|                                | -offset   |  | -lam=0.95   |          |
|                                | -clip     |  | -ent_coef=0.01|         |
|                                +-----+-----+  +-------------+          |
|                                      |                                  |
+--------------------------------------|---------------------------------- +
                                       |
                                deploy.yaml
                                       |
                                       v
+------------------------------------------------------------------+
|                    C++ 部署端 (Sim2Real)                           |
|                                                                   |
|  +-------------+   +-------------+   +--------------------+       |
|  | LowState    |-->| Observation |-->|   ONNX Runtime     |       |
|  | (SDK)       |   |  Manager    |   |   (OrtRunner)      |       |
|  |             |   |             |   |                    |       |
|  | -IMU data   |   | -base_ang_vel|   |   Input: obs       |       |
|  | -joint_pos  |   | -joint_pos  |   |   Output: action    |       |
|  | -joint_vel  |   | -joint_vel  |   |   Device: CPU       |       |
|  +-----+-------+   | -gravity    |   +----------+---------+       |
|        |            | -last_action|             |                  |
|        |            | -commands   |             |                  |
|        |            +-----+-------+             |                  |
|        |                  |                      |                  |
|        |            +-----v-------+     +-------v--------+         |
|        |            |   Action    |     |     FSM        |         |
|        |            |   Manager   |     |                |         |
|        |            |             |     | Passive        |         |
|        |            | scale+clip  |     | FixStand       |         |
|        |            | +offset     |     | RLBase         |         |
|        |            +-----+-------+     | Mimic(G1-29dof)|         |
|        |                  |              +---------------+         |
|        |                  v                                        |
|        |            +-------------+                                |
|        |            | LowCommand  |                                |
|        +----------->| (SDK)       |                                |
|                      +-------------+                                |
+--------------------------------------------------------------------+
```

---

## 三、Python 训练接口

### 3.1 训练脚本接口 (`scripts/rsl_rl/train.py`)

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--task` | str | None | 任务名称，如 `Unitree-G1-29dof-Velocity` |
| `--num_envs` | int | None | 并行环境数量（覆盖配置文件） |
| `--seed` | int | None | 随机种子（-1 表示随机） |
| `--max_iterations` | int | None | 训练最大迭代次数 |
| `--video` | flag | False | 训练过程中录制视频 |
| `--video_length` | int | 200 | 视频长度（步数） |
| `--video_interval` | int | 2000 | 视频录制间隔（步数） |
| `--distributed` | flag | False | 启用分布式多 GPU 训练 |
| `--experiment_name` | str | None | 实验名称 |
| `--run_name` | str | None | 运行名称后缀 |
| `--resume` | flag | False | 从检查点恢复训练 |
| `--load_run` | str | None | 要恢复的运行文件夹 |
| `--checkpoint` | str | None | 检查点文件路径 |
| `--logger` | str | None | 日志后端：wandb/tensorboard/neptune |
| `--log_project_name` | str | None | wandb/neptune 项目名称 |

#### 训练流程

```python
# 1. 加载配置（通过 Hydra）
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # 2. 覆盖 CLI 参数
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs

    # 3. 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg)

    # 4. 包装为 RSL-RL VecEnv
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 5. 创建 Runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # 6. 导出部署配置（自动生成 deploy.yaml）
    export_deploy_cfg(env.unwrapped, log_dir)

    # 7. 执行训练
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
```

#### 输出目录结构

```
logs/rsl_rl/{experiment_name}/{timestamp}_{run_name}/
+-- params/
|   +-- env.yaml           # 环境配置快照
|   +-- agent.yaml         # 智能体配置快照
|   +-- deploy.yaml        # 部署配置（自动生成）
|   +-- velocity_env_cfg.py # 环境配置文件副本
+-- exported/
|   +-- policy.pt          # PyTorch JIT 模型
|   +-- policy.onnx        # ONNX 模型
+-- model.pt               # 最新模型
+-- model_{iteration}.pt   # 定期保存的检查点
+-- logs/                  # TensorBoard/WandB 日志
```

### 3.2 推理/播放脚本接口 (`scripts/rsl_rl/play.py`)

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--task` | str | None | 任务名称 |
| `--num_envs` | int | None | 并行环境数量 |
| `--checkpoint` | str | None | 检查点文件路径 |
| `--use_pretrained_checkpoint` | flag | False | 使用预训练检查点 |
| `--video` | flag | False | 录制视频 |
| `--video_length` | int | 200 | 视频长度 |
| `--real-time` | flag | False | 实时运行模式 |
| `--disable_fabric` | flag | False | 禁用 fabric 使用 USD I/O |

#### 推理流程

```python
def main():
    # 1. 加载环境配置
    env_cfg = parse_env_cfg(args_cli.task, entry_point_key="play_env_cfg_entry_point")

    # 2. 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg)

    # 3. 加载策略
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # 4. 获取推理策略
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 5. 导出模型（JIT + ONNX）
    export_policy_as_jit(policy_nn, normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer, path=export_model_dir, filename="policy.onnx")

    # 6. 推理循环
    obs = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
```

### 3.3 环境注册接口

任务通过 `gymnasium` 注册，入口配置：

| 入口键 | 说明 | 对应配置类 |
|--------|------|-----------|
| `env_cfg_entry_point` | 训练环境配置 | `RobotEnvCfg` |
| `play_env_cfg_entry_point` | 推理环境配置 | `RobotPlayEnvCfg` |
| `rsl_rl_cfg_entry_point` | RL 算法配置 | `BasePPORunnerCfg` |

---

## 四、环境配置接口

### 4.1 环境基类 (`ManagerBasedRLEnvCfg`)

```python
@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    # 场景配置
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP 组件配置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 仿真参数
        self.decimation = 4                    # 降采样率
        self.episode_length_s = 20.0           # 集长度（秒）
        self.sim.dt = 0.005                    # 仿真步长（秒）
        # 实际控制频率 = 1 / (dt * decimation) = 1/0.02 = 50Hz
```

### 4.2 场景配置 (`RobotSceneCfg`)

```python
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    # 地形
    terrain: TerrainImporterCfg
    # 机器人
    robot: ArticulationCfg
    # 传感器
    height_scanner: RayCasterCfg     # 高度计
    contact_forces: ContactSensorCfg # 接触传感器
    # 灯光
    sky_light: AssetBaseCfg
```

### 4.3 命令接口 (`CommandsCfg`)

```python
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),  # 每10秒重新采样
        rel_standing_envs=0.02,               # 2% 环境保持站立
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        # 初始速度范围（课程学习起点）
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.1, 0.1)
        ),
        # 速度上限（课程学习终点）
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.2, 0.2)
        ),
    )
```

---

## 五、MDP 组件接口

### 5.1 观测接口 (`ObservationsCfg`)

#### Policy 观测组

| 观测项 | 函数 | 维度 | 缩放 | 噪声 | 说明 |
|--------|------|------|------|------|------|
| `base_ang_vel` | `mdp.base_ang_vel` | 3 | 0.2 | U[-0.2,0.2] | 基座角速度（身体坐标系） |
| `projected_gravity` | `mdp.projected_gravity` | 3 | 1.0 | U[-0.05,0.05] | 投影重力向量 |
| `velocity_commands` | `mdp.generated_commands` | 3 | 1.0 | - | 速度指令 |
| `joint_pos_rel` | `mdp.joint_pos_rel` | 29 | 1.0 | U[-0.01,0.01] | 关节相对位置（距默认位置） |
| `joint_vel_rel` | `mdp.joint_vel_rel` | 29 | 0.05 | U[-1.5,1.5] | 关节相对速度 |
| `last_action` | `mdp.last_action` | 29 | 1.0 | - | 上一步动作 |

**观测组参数：**
- `history_length = 5`：历史步数
- `enable_corruption = True`：启用噪声
- `concatenate_terms = True`：拼接所有观测项

**总观测维度计算：**
```
单步观测 = 3 + 3 + 3 + 29 + 29 + 29 = 96
历史观测 = 96 x 5 = 480
```

#### Critic 观测组（优势）

| 观测项 | 维度 | 说明 |
|--------|------|------|
| `base_lin_vel` | 3 | 基座线速度 |
| `base_ang_vel` | 3 | 基座角速度 |
| `projected_gravity` | 3 | 投影重力 |
| `velocity_commands` | 3 | 速度指令 |
| `joint_pos_rel` | 29 | 关节相对位置 |
| `joint_vel_rel` | 29 | 关节相对速度 |
| `last_action` | 29 | 上一步动作 |

**Critic 总维度 = 3 + 3 + 3 + 3 + 29 + 29 + 29 = 99 x 5 = 495**

#### 观测项函数签名

```python
# Python 端
def observation_func(env: ManagerBasedRLEnv, **params) -> torch.Tensor:
    """返回形状: (num_envs, obs_dim)"""
    ...

# C++ 端
REGISTER_OBSERVATION(obs_name)
{
    // 返回 std::vector<float>，长度为 obs_dim
    auto & asset = env->robot;
    return std::vector<float>(data.data(), data.data() + data.size());
}
```

### 5.2 动作接口 (`ActionsCfg`)

```python
@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],      # 正则匹配所有关节
        scale=0.25,              # 动作缩放系数
        use_default_offset=True  # 使用默认关节位置作为偏移
    )
```

**动作处理流程：**
```
network_output ([-1, 1])
    -> scale * action          # 乘以缩放系数
    -> + default_joint_pos     # 加上默认关节位置作为偏移
    -> clip to joint_limits    # 裁剪到关节限制
    -> motor_cmd.q()           # 输出给电机
```

**动作维度 = 关节数量（G1-29DOF: 29）**

### 5.3 奖励接口 (`RewardsCfg`)

#### 任务奖励

| 奖励项 | 函数 | 权重 | 公式/说明 |
|--------|------|------|-----------|
| `track_lin_vel_xy` | `mdp.track_lin_vel_xy_yaw_frame_exp` | +1.0 | exp(-|v_actual - v_cmd| / std^2) |
| `track_ang_vel_z` | `mdp.track_ang_vel_z_exp` | +0.5 | 同上，角速度版本 |
| `alive` | `mdp.is_alive` | +0.15 | 存活奖励 |

#### 基座惩罚

| 惩罚项 | 函数 | 权重 | 说明 |
|--------|------|------|------|
| `base_linear_velocity` | `mdp.lin_vel_z_l2` | -2.0 | 惩罚 Z 向线速度 |
| `base_angular_velocity` | `mdp.ang_vel_xy_l2` | -0.05 | 惩罚滚转/俯仰角速度 |
| `flat_orientation_l2` | `mdp.flat_orientation_l2` | -5.0 | 惩罚倾斜 |
| `base_height` | `mdp.base_height_l2` | -10 | 基座高度偏离目标(0.78m) |

#### 关节惩罚

| 惩罚项 | 函数 | 权重 | 说明 |
|--------|------|------|------|
| `joint_vel` | `mdp.joint_vel_l2` | -0.001 | 关节角速度 L2 |
| `joint_acc` | `mdp.joint_acc_l2` | -2.5e-7 | 关节加速度 L2 |
| `action_rate` | `mdp.action_rate_l2` | -0.05 | 动作变化率 |
| `dof_pos_limits` | `mdp.joint_pos_limits` | -5.0 | 接近关节限位惩罚 |
| `energy` | `mdp.energy` | -2e-5 | 能量消耗 = |qvel| x |tau| |
| `joint_deviation_arms` | `mdp.joint_deviation_l1` | -0.1 | 手臂关节偏离默认 |
| `joint_deviation_waists` | `mdp.joint_deviation_l1` | -1.0 | 腰部关节偏离默认 |
| `joint_deviation_legs` | `mdp.joint_deviation_l1` | -1.0 | 腿部关节偏离默认 |

#### 步态奖励

| 奖励项 | 函数 | 权重 | 说明 |
|--------|------|------|------|
| `gait` | `mdp.feet_gait` | +0.5 | 步态周期一致性 |
| `feet_slide` | `mdp.feet_slide` | -0.2 | 足端打滑惩罚 |
| `feet_clearance` | `mdp.foot_clearance_reward` | +1.0 | 足端离地高度奖励 |
| `undesired_contacts` | `mdp.undesired_contacts` | -1.0 | 非足端接触惩罚 |

#### 奖励函数签名

```python
# Python 端
def reward_func(env: ManagerBasedRLEnv, **params) -> torch.Tensor:
    """返回形状: (num_envs,) 每个环境的标量奖励"""
    ...
```

### 5.4 终止接口 (`TerminationsCfg`)

```python
@configclass
class TerminationsCfg:
    # 超时终止
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 基座高度低于 0.2m 终止
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    # 姿态角度超过 0.8rad 终止
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
```

### 5.5 事件接口 (`EventCfg`)

#### 启动事件 (Startup)

| 事件 | 函数 | 参数 | 说明 |
|------|------|------|------|
| `physics_material` | `mdp.randomize_rigid_body_material` | friction: [0.3,1.0] | 随机化摩擦系数 |
| `add_base_mass` | `mdp.randomize_rigid_body_mass` | mass: [-1.0, +3.0]kg | 随机化基座质量 |

#### 重置事件 (Reset)

| 事件 | 函数 | 参数 | 说明 |
|------|------|------|------|
| `reset_base` | `mdp.reset_root_state_uniform` | pose: x,yin[-0.5,0.5], yawin[-pi,pi] | 随机重置基座位姿 |
| `reset_robot_joints` | `mdp.reset_joints_by_scale` | pos_scale: 1.0, vel: [-1,1] | 重置关节状态 |
| `base_external_force_torque` | `mdp.apply_external_force_torque` | force: 0, torque: 0 | 外部力矩（当前禁用） |

#### 间隔事件 (Interval)

| 事件 | 函数 | 间隔 | 说明 |
|------|------|------|------|
| `push_robot` | `mdp.push_by_setting_velocity` | 每5秒 | 速度扰动: vx,vyin[-0.5,0.5] |

### 5.6 课程学习接口 (`CurriculumCfg`)

```python
@configclass
class CurriculumCfg:
    # 地形难度随训练递进
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # 速度指令范围随训练递进
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
```

---

## 六、智能体配置接口

### 6.1 PPO Runner 配置 (`BasePPORunnerCfg`)

```python
@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 训练参数
    num_steps_per_env = 24           # 每迭代每个环境采集的步数
    max_iterations = 50000           # 最大训练迭代次数
    save_interval = 100              # 保存间隔（迭代次数）
    empirical_normalization = False   # 经验标准化

    # 策略网络配置
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],  # Actor 网络结构
        critic_hidden_dims=[512, 256, 128], # Critic 网络结构
        activation="elu",                      # 激活函数
    )

    # PPO 算法配置
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,           # 价值损失系数
        use_clipped_value_loss=True,   # 使用裁剪价值损失
        clip_param=0.2,                # PPO 裁剪参数
        entropy_coef=0.01,             # 熵正则化系数
        num_learning_epochs=5,         # 每迭代训练轮数
        num_mini_batches=4,            # 小批次数
        learning_rate=1.0e-3,          # 学习率
        schedule="adaptive",           # 学习率调度策略
        gamma=0.99,                    # 折扣因子
        lam=0.95,                      # GAE lambda
        desired_kl=0.01,               # 目标 KL 散度
        max_grad_norm=1.0,             # 梯度裁剪
    )
```

### 6.2 网络结构详情

```
Actor 网络:
    Input(480) -> FC(512) -> ELU -> FC(256) -> ELU -> FC(128) -> ELU -> Output(29)

Critic 网络:
    Input(495) -> FC(512) -> ELU -> FC(256) -> ELU -> FC(128) -> ELU -> Output(1)

数据流:
    每迭代采集步数 = num_envs x num_steps_per_env = 4096 x 24 = 98304
    每迭代总样本数 = 98304
    小批量大小 = 98304 / (num_learning_epochs x num_mini_batches) = 98304 / 20 = 4915
```

---

## 七、部署配置导出接口

### 7.1 自动导出 (`export_deploy_cfg.py`)

训练时自动调用 `export_deploy_cfg(env, log_dir)`，生成 `deploy.yaml`：

```yaml
# deploy.yaml 结构
joint_ids_map: [0, 1, 2, ..., 28]           # SDK 关节 ID 映射
step_dt: 0.02                                # 控制周期
stiffness: [k1, k2, ..., k29]               # 关节刚度
damping: [d1, d2, ..., d29]                 # 关节阻尼
default_joint_pos: [q1, q2, ..., q29]       # 默认关节位置

commands:
  base_velocity:
    ranges:
      lin_vel_x: [-0.5, 1.0]
      lin_vel_y: [-0.3, 0.3]
      ang_vel_z: [-0.2, 0.2]

actions:
  JointPositionAction:
    scale: [0.25, 0.25, ..., 0.25]
    offset: [q1_default, q2_default, ..., q29_default]
    clip: [...]
    joint_ids: null

observations:
  base_ang_vel:
    scale: [0.2, 0.2, 0.2]
    history_length: 5
  projected_gravity:
    scale: [1.0, 1.0, 1.0]
    history_length: 5
  # ... 其他观测项
```

---

## 八、C++ 部署接口

### 8.1 环境类 (`ManagerBasedRLEnv`)

```cpp
namespace isaaclab {

class ManagerBasedRLEnv {
public:
    // 构造：从 YAML 配置和机器人关节体创建环境
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot);

    // 重置环境状态
    void reset();

    // 执行一步：观测 -> 策略推理 -> 动作处理
    void step();

    // 控制周期
    float step_dt;

    // 配置
    YAML::Node cfg;

    // 管理器
    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;

    // 机器人关节体
    std::shared_ptr<Articulation> robot;

    // 算法（ONNX 推理器）
    std::unique_ptr<Algorithms> alg;

    // 集长度
    long episode_length = 0;

    // 全局相位（用于步态）
    float global_phase = 0.0f;
};

}
```

### 8.2 观测管理器 (`ObservationManager`)

```cpp
class ObservationManager {
public:
    ObservationManager(YAML::Node cfg, ManagerBasedRLEnv* env);

    // 重置所有观测历史
    void reset();

    // 计算所有组的观测，返回 map 供 ONNX 输入
    std::unordered_map<std::string, std::vector<float>> compute();

    // 计算指定组的观测
    const std::vector<float> compute_group(const std::string& group_name);
};
```

#### 注册观测项

```cpp
// 宏定义注册机制
#define REGISTER_OBSERVATION(name) \
    inline std::vector<float> name(ManagerBasedRLEnv* env, YAML::Node params); \
    inline struct name##_registrar { \
        name##_registrar() { observations_map()[#name] = name; } \
    } name##_registrar_instance; \
    inline std::vector<float> name(ManagerBasedRLEnv* env, YAML::Node params)

// 使用示例
REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}
```

### 8.3 动作管理器 (`ActionManager`)

```cpp
class ActionTerm {
public:
    virtual int action_dim() = 0;
    virtual std::vector<float> raw_actions() = 0;
    virtual std::vector<float> processed_actions() = 0;
    virtual void process_actions(std::vector<float> actions) = 0;
    virtual void reset() {};
};

class ActionManager {
public:
    ActionManager(YAML::Node cfg, ManagerBasedRLEnv* env);

    void reset();
    std::vector<float> action();
    std::vector<float> processed_actions();

    // 处理原始动作（缩放、偏移、裁剪）
    void process_action(std::vector<float> action);

    int total_action_dim();
    std::vector<int> action_dim();
};
```

### 8.4 ONNX 推理器 (`OrtRunner`)

```cpp
class Algorithms {
public:
    // 虚拟方法：接收观测，返回动作
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    // 获取最新动作（线程安全）
    std::vector<float> get_action();

protected:
    std::vector<float> action;
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms {
public:
    // 构造：加载 ONNX 模型
    OrtRunner(std::string model_path);

    // 推理：观测 -> 动作
    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) override;

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_shape;
};
```

### 8.5 FSM 状态机接口

```cpp
// RL 策略状态
class State_RLBase : public FSMState {
public:
    State_RLBase(int state_mode, std::string state_string);

    void enter() {
        // 1. 设置 PD 增益
        for (int i = 0; i < joint_count; ++i) {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
        }

        // 2. 启动策略推理线程
        policy_thread = std::thread([this]{
            while (policy_thread_running) {
                env->step();  // 观测 -> 推理 -> 动作
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run() {
        // 将处理后的动作写入电机命令
        auto action = env->action_manager->processed_actions();
        for (int i = 0; i < joint_count; i++) {
            lowcmd->msg_.motor_cmd()[joint_ids_map[i]].q() = action[i];
        }
    }

    void exit() {
        policy_thread_running = false;
        if (policy_thread.joinable()) policy_thread.join();
    }
};
```

### 8.6 状态机构造与配置

```cpp
State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    // 从配置读取策略目录
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    // 创建 RL 环境（加载 deploy.yaml）
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );

    // 创建 ONNX 推理器（加载 policy.onnx）
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    // 注册安全回退检查
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool { return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}
```

---

## 九、数据流总结

### 9.1 Python 训练数据流

```
+------------------------------------------------------------------+
|                     Training Loop                                 |
|                                                                   |
|  for iteration in range(max_iterations):                          |
|      for step in range(num_steps_per_env):                        |
|                                                                   |
|          +-------------+                                         |
|          |  Environment |                                         |
|          |    .reset()  |  <- 随机化事件                           |
|          |    .step()   |  <- 物理仿真                             |
|          +------+------+                                         |
|                 |                                                 |
|                 v                                                 |
|          +-------------+                                         |
|          | Observation |                                         |
|          |   Manager   |  -> 拼接所有观测项                        |
|          |             |  -> 添加噪声                              |
|          |  compute()  |  -> 维护历史缓冲区                        |
|          +------+------+                                         |
|                 | obs: (num_envs, 480)                            |
|                 v                                                 |
|          +-------------+                                         |
|          |   Policy    |                                         |
|          |  (Actor)    |  -> FC(512)->FC(256)->FC(128)           |
|          |  .forward() |  -> 输出: (num_envs, 29)                 |
|          +------+------+                                         |
|                 | action: (num_envs, 29)                          |
|                 v                                                 |
|          +-------------+                                         |
|          |  Action     |                                         |
|          |  Manager    |  -> scale * action + offset              |
|          | .process()  |  -> clip to limits                       |
|          +------+------+                                         |
|                 | joint_targets                                   |
|                 v                                                 |
|          +-------------+                                         |
|          |  Reward     |                                         |
|          |  Manager    |  -> 计算所有奖励项                        |
|          | .compute()  |  -> 加权和                               |
|          +------+------+                                         |
|                 | reward: (num_envs,)                             |
|                 v                                                 |
|          +-------------+                                         |
|          |  PPO        |                                         |
|          |  .learn()   |  -> 5 epochs x 4 minibatches            |
|          +-------------+                                         |
+------------------------------------------------------------------+
```

### 9.2 C++ 部署数据流

```
+------------------------------------------------------------------+
|                  Deployment Thread (固定频率)                      |
|                                                                   |
|  while (policy_thread_running):                                   |
|      env->step() {                                                |
|                                                                   |
|          +----------------+                                      |
|          |   Articulation |  <- 从 SDK 读取                       |
|          |    .update()   |    - IMU 数据                         |
|          |                |    - 关节位置/速度                     |
|          |                |    - 计算投影重力                      |
|          +-------+--------+                                      |
|                  |                                               |
|                  v                                               |
|          +----------------+                                      |
|          | ObservationMgr |                                      |
|          |   .compute()   |  -> 按配置计算所有观测                 |
|          |                |  -> 输出: map<string, vector>         |
|          +-------+--------+                                      |
|                  | obs_map                                       |
|                  v                                               |
|          +----------------+                                      |
|          |   OrtRunner    |                                      |
|          |     .act()     |  -> ONNX 推理 (CPU)                   |
|          |                |  -> 输出: vector<float> (29)          |
|          +-------+--------+                                      |
|                  | action                                        |
|                  v                                               |
|          +----------------+                                      |
|          |  ActionManager |                                      |
|          | .process_action|  -> scale * action + offset          |
|          |                |  -> clip                              |
|          +----------------+                                      |
|      }                                                           |
|                                                                   |
|      // 主线程 run() 中：                                         |
|      auto action = env->action_manager->processed_actions();      |
|      for (i) lowcmd->motor_cmd()[i].q() = action[i];             |
|      lowcmd->send()  -> 发送给机器人                              |
|                                                                   |
|      std::this_thread::sleep_until(sleepTill);                   |
|      sleepTill += env->step_dt;                                   |
+------------------------------------------------------------------+
```

---

## 十、配置文件示例

### 10.1 部署配置 (`config/config.yaml`)

```yaml
FSM:
  RLBase:
    policy_dir: "./config/policy/velocity/v0"
    # 策略目录结构：
    # policy_dir/
    #   params/
    #     deploy.yaml       # 部署配置
    #   exported/
    #     policy.onnx       # ONNX 策略模型
```

---

## 十一、训练工作流

```bash
# 1. 激活环境
conda activate sim51

# 2. 安装扩展
./unitree_rl_lab.sh -i

# 3. 列出所有可用环境
./unitree_rl_lab.sh -l

# 4. 训练（单机单卡）
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity

# 5. 训练（分布式多卡）
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --distributed

# 6. 训练（带视频录制）
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --video

# 7. 推理/演示
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity --load_run logs/rsl_rl/.../best.pt

# 8. 恢复训练
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --resume --load_run 2024-01-01_12-00-00

# 9. 模仿学习数据准备
python scripts/mimic/csv_to_npz.py -f path/to/motion.csv --input_fps 60
```

---

## 十二、关键技术点

### 12.1 域随机化

| 类型 | 参数 | 范围 | 时机 |
|------|------|------|------|
| 摩擦系数 | static/dynamic | [0.3, 1.0] | 启动时 |
| 基座质量 | add_mass | [-1.0, +3.0]kg | 启动时 |
| 速度扰动 | push_velocity | vx,vyin[-0.5,0.5] | 每5秒 |
| 观测噪声 | additive_uniform | 各观测项不同 | 每步 |

### 12.2 课程学习

- **地形难度**: 从平地逐步增加到复杂地形
- **速度指令**: 从低速 (-0.1~0.1m/s) 逐步增加到全速 (-0.5~1.0m/s)

### 12.3 模仿学习

- **自适应采样**: 基于失败率的难度感知采样
- **相对坐标变换**: 将绝对运动数据转换为相对机器人坐标
- **多连杆跟踪**: 同时跟踪 14 个连杆的位置和朝向

### 12.4 部署优化

- **配置自动导出**: 训练时自动生成 deploy.yaml
- **ONNX CPU 推理**: 轻量级推理，无需 GPU
- **线程分离**: 策略推理与通信独立线程
- **FSM 安全切换**: 支持 Passive/FixStand/RLBase/Mimic 状态切换

---

## 十三、接口兼容性矩阵

| 组件 | Python (训练) | C++ (部署) | 说明 |
|------|:-------------:|:----------:|------|
| 环境 | ManagerBasedRLEnv | ManagerBasedRLEnv | 配置通过 YAML 传递 |
| 观测 | ObservationManager | ObservationManager | 注册机制不同 |
| 动作 | ActionManager | ActionManager | 处理逻辑一致 |
| 策略 | ActorCritic (PyTorch) | OrtRunner (ONNX) | 模型格式转换 |
| 算法 | PPO | - | 仅训练端需要 |
| 奖励 | RewardManager | - | 仅训练端需要 |
| 终止 | TerminationManager | 手动检查 | 部署端简化 |
| 事件 | EventManager | - | 仅训练端需要 |
| 课程 | CurriculumManager | - | 仅训练端需要 |

---

## 十四、开源依赖

| 项目 | 用途 |
|------|------|
| [IsaacLab](https://github.com/isaac-sim/IsaacLab) | 训练基础框架 |
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl) | RL 算法库 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | 策略推理 |
| [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) | 机器人通信 |
| [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) | 全身运动跟踪 |
