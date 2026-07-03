# Unitree RL Lab - 源码分析文档

## 一、项目概述

Unitree RL Lab 是基于 NVIDIA Isaac Lab 构建的强化学习环境集合，专为宇树科技（Unitree Robotics）的系列机器人设计。项目提供从仿真训练到真实机器人部署的完整工作流。

| 属性 | 详情 |
|------|------|
| **版本** | 0.2.1 |
| **依赖框架** | IsaacSim 5.1.0, IsaacLab 2.3.0 |
| **协议** | Apache 2.0 |
| **Python 版本** | >= 3.10 |
| **RL 库** | RSL-RL >= 2.3.1 |

### 支持的机器人

| 机器人 | 自由度 | 类型 |
|--------|--------|------|
| Go2 | 12DOF | 四足机器人 |
| Go2W | 16DOF | 四足机器人（带足端关节） |
| B2 | 12DOF | 大型四足机器人 |
| H1 | 20DOF | 人形机器人 |
| G1 | 23DOF | 小型人形机器人 |
| G1 | 29DOF | 小型人形机器人（含手部） |

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
        │   │   │   └── rsl_rl_ppo_cfg.py
        │   │   ├── mdp/             # MDP 组件
        │   │   │   ├── rewards.py   # 奖励函数
        │   │   │   ├── observations.py  # 观测函数
        │   │   │   ├── curriculums.py   # 课程学习
        │   │   │   └── commands/    # 命令生成器
        │   │   └── robots/          # 各机器人环境配置
        │   │       ├── g1/29dof/velocity_env_cfg.py
        │   │       ├── go2/velocity_env_cfg.py
        │   │       └── h1/velocity_env_cfg.py
        │   └── mimic/               # 模仿学习任务
        │       ├── agents/
        │       ├── mdp/
        │       │   ├── commands.py  # 运动跟踪命令
        │       │   ├── rewards.py   # 模仿奖励
        │       │   ├── events.py    # 事件随机化
        │       │   ├── observations.py
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

`unitree.py` 定义了所有机器人的完整配置，包括：

```python
@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    joint_sdk_names: list[str] = None      # SDK 关节名称映射
    soft_joint_pos_limit_factor = 0.9      # 关节位置软限制系数
```

#### 3.1.2 执行器模型

项目使用三种执行器模型：

| 模型 | 说明 | 适用场景 |
|------|------|----------|
| **IdealPDActuatorCfg** | 理想 PD 执行器 | Go2, B2, H1 |
| **ImplicitActuatorCfg** | 隐式执行器（含延迟） | G1-23dof, G1-29dof |
| **UnitreeActuatorCfg_Go2HV** | 宇树定制执行器 | Go2 HV 电机 |

#### 3.1.3 G1-29DOF 执行器参数

基于真实电机参数计算刚度与阻尼：

```python
# 电机惯性参数
ARMATURE_5020 = 0.003609725    # N5020 系列
ARMATURE_7520_14 = 0.010177520  # N7520 14.3Nm
ARMATURE_7520_22 = 0.025101925  # N7520 22.5Nm
ARMATURE_4010 = 0.00425         # W4010 系列

# 自然频率 10Hz, 阻尼比 2.0
NATURAL_FREQ = 10 * 2.0 * pi
DAMPING_RATIO = 2.0

# 刚度 = 惯性 * 频率^2
# 阻尼 = 2 * 阻尼比 * 惯性 * 频率
```

#### 3.1.4 关节名称映射

每个机器人配置包含 `joint_sdk_names` 列表，将仿真关节名映射到 SDK 实际关节索引，确保仿真到实物的动作一致性。

### 3.2 运动控制任务 (`tasks/locomotion/`)

#### 3.2.1 环境配置 (G1-29DOF Velocity)

**基本参数：**
- 并行环境数：4096
- 控制频率：200Hz (decimation=4, sim.dt=0.005)
- 集长度：20秒
- 环境间距：2.5m

**观测空间（Policy）：**
| 观测项 | 维度 | 说明 |
|--------|------|------|
| base_ang_vel | 3 | 基座角速度（缩放0.2） |
| projected_gravity | 3 | 投影重力向量 |
| velocity_commands | 3 | 速度指令 |
| joint_pos_rel | 29 | 关节相对位置 |
| joint_vel_rel | 29 | 关节相对速度（缩放0.05） |
| last_action | 29 | 上一步动作 |
- 历史长度：5帧（总计约 845 维）
- 添加噪声增强鲁棒性

**Critic 额外观测：**
- base_lin_vel（基座线速度）

**奖励函数：**

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| track_lin_vel_xy | +1.0 | 跟踪 XY 线速度（指数） |
| track_ang_vel_z | +0.5 | 跟踪偏航角速度 |
| alive | +0.15 | 存活奖励 |
| base_linear_velocity | -2.0 | 惩罚 Z 轴速度 |
| base_angular_velocity | -0.05 | 惩罚 XY 角速度 |
| joint_vel | -0.001 | 关节角速度惩罚 |
| joint_acc | -2.5e-7 | 关节加速度惩罚 |
| action_rate | -0.05 | 动作变化率惩罚 |
| energy | -2e-5 | 能量消耗惩罚 |
| flat_orientation | -5.0 | 保持水平朝向 |
| base_height | -10 | 基座高度惩罚（目标0.78m） |
| gait | +0.5 | 步态奖励（周期0.8s） |
| feet_slide | -0.2 | 足端滑动惩罚 |
| feet_clearance | +1.0 | 足端离地高度奖励 |
| undesired_contacts | -1 | 非期望接触惩罚 |
| joint_deviation_* | -0.1~1 | 关节偏离惩罚 |

**终止条件：**
- 超时（20秒）
- 基座高度 < 0.2m
- 姿态角度 > 0.8rad

**课程学习：**
- 地形难度递进
- 速度指令范围递进

#### 3.2.2 速度命令生成器

`UniformLevelVelocityCommandCfg`：
- 重采样周期：10秒
- 站立环境比例：2%
- 初始范围：vx(-0.1, 0.1), vy(-0.1, 0.1), wz(-0.1, 0.1)
- 限制范围：vx(-0.5, 1.0), vy(-0.3, 0.3), wz(-0.2, 0.2)

### 3.3 模仿学习任务 (`tasks/mimic/`)

#### 3.3.1 运动加载器 (`MotionLoader`)

将 BVH 动画数据转换为 NPZ 格式，包含：
- `joint_pos`: 关节位置 [T, J]
- `joint_vel`: 关节速度 [T, J]
- `body_pos_w`: 连杆世界坐标 [T, B, 3]
- `body_quat_w`: 连杆世界四元数 [T, B, 4]
- `body_lin_vel_w`: 连杆线速度 [T, B, 3]
- `body_ang_vel_w`: 连杆角速度 [T, B, 3]

#### 3.3.2 运动命令项 (`MotionCommand`)

**自适应采样机制：**
```
bin_failed_count <- alpha * current_bin_failed + (1-alpha) * bin_failed_count
sampling_prob <- bin_failed_count + uniform_ratio / bin_count
sampled_bins <- multinomial(sampling_prob)
```

该机制使训练更关注失败率高的运动片段。

#### 3.3.3 模仿学习奖励

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| anchor_pos | +0.5 | 锚点位置跟踪（指数） |
| anchor_ori | +0.5 | 锚点朝向跟踪 |
| body_pos | +1.0 | 各连杆相对位置跟踪 |
| body_ori | +1.0 | 各连杆相对朝向跟踪 |
| body_lin_vel | +1.0 | 连杆线速度跟踪 |
| body_ang_vel | +1.0 | 连杆角速度跟踪 |
| joint_acc | -2.5e-7 | 关节加速度惩罚 |
| joint_torque | -1e-5 | 关节力矩惩罚 |
| action_rate | -0.1 | 动作变化率惩罚 |

#### 3.3.4 预置动作数据集

| 动作 | 文件 | 帧率 |
|------|------|------|
| Dance 102 | G1_Take_102.bvh_60hz.csv | 60Hz |
| Gangnam Style | G1_gangnam_style_V01.bvh_60hz.csv | 60Hz |

**跟踪的连杆（14个）：**
pelvis, left/right_hip_roll_link, left/right_knee_link, left/right_ankle_roll_link, torso_link, left/right_shoulder_roll_link, left/right_elbow_link, left/right_wrist_yaw_link

### 3.4 事件随机化 (`EventCfg`)

| 事件 | 模式 | 说明 |
|------|------|------|
| physics_material | startup | 随机化摩擦系数 |
| add_base_mass | startup | 随机化质量（+/-1~3kg） |
| push_robot | interval | 每5秒施加速度扰动 |
| base_com | startup | 随机化质心位置 |
| add_joint_default_pos | startup | 随机化关节默认位置 |

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

#### 4.3.1 机器人关节体 (`BaseArticulation`)

```cpp
class BaseArticulation : public isaaclab::Articulation {
    // 从低层状态读取 IMU 和关节数据
    void update() override {
        // IMU 角速度
        data.root_ang_vel_b[i] = lowstate->msg_.imu_state().gyroscope()[i];
        // IMU 四元数 -> 投影重力
        data.root_quat_w = Quaternionf(lowstate->msg_.imu_state().quaternion()...);
        data.projected_gravity_b = root_quat_w.conjugate() * GRAVITY_VEC_W;
        // 关节状态
        data.joint_pos[i] = lowstate->msg_.motor_state()[joint_id].q();
        data.joint_vel[i] = lowstate->msg_.motor_state()[joint_id].dq();
    }
};
```

#### 4.3.2 RL 策略状态 (`State_RLBase`)

```cpp
void enter() {
    // 设置 PD 增益
    lowcmd->msg_.motor_cmd()[i].kp() = joint_stiffness[i];
    lowcmd->msg_.motor_cmd()[i].kd() = joint_damping[i];
    
    // 启动策略线程
    policy_thread = std::thread([this]{
        env->reset();
        while (policy_thread_running) {
            env->step();  // 观测 -> ONNX推理 -> 动作
            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}
```

#### 4.3.3 动作处理 (`JointAction`)

```cpp
class JointPositionAction : public JointAction {
    // 动作处理流程：
    // raw_action -> scale -> offset -> clip -> processed_action
    void process_actions(std::vector<float> actions) {
        _processed_actions[i] = raw_actions[i] * scale[i] + offset[i];
        _processed_actions[i] = clamp(processed, clip_min, clip_max);
    }
};
```

### 4.4 ONNX 推理

使用 `onnxruntime-linux-x64-1.22.0` 进行策略推理：
- 输入：观测向量（与训练时一致）
- 输出：动作向量（关节位置偏差）
- 支持 CPU 推理（无需 GPU）

### 4.5 部署配置导出

训练完成后自动导出 `deploy_cfg.yaml`，包含：
- 关节 ID 映射
- PD 增益
- 动作缩放/偏移
- 观测噪声参数
- 环境终止条件

---

## 五、训练工作流

### 5.1 训练流程

```bash
# 1. 激活 IsaacLab 环境
conda activate env_isaaclab

# 2. 安装扩展
./unitree_rl_lab.sh -i

# 3. 列出可用环境
./unitree_rl_lab.sh -l

# 4. 开始训练（支持自动补全）
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity

# 5. 分布式训练
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --distributed

# 6. 录制视频
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --video
```

### 5.2 训练脚本核心逻辑

```python
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # 1. 创建 Isaac 环境
    env = gym.make(task, cfg=env_cfg)
    
    # 2. 包装为 RSL-RL VecEnv
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # 3. 创建训练器
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir)
    
    # 4. 导出部署配置
    export_deploy_cfg(env.unwrapped, log_dir)
    
    # 5. 开始训练
    runner.learn(num_learning_iterations=agent_cfg.max_iterations)
```

### 5.3 模仿学习数据准备

```bash
# CSV 转 NPZ（训练前必须执行）
python scripts/mimic/csv_to_npz.py \
    -f path/to/G1_Take_102.bvh_60hz.csv \
    --input_fps 60
```

---

## 六、Sim2Sim / Sim2Real 部署

### 6.1 依赖安装

```bash
# 系统依赖
sudo apt install -y libyaml-cpp-dev libboost-all-dev \
    libeigen3-dev libspdlog-dev libfmt-dev

# Unitree SDK2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2 && mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF
sudo make install

# 编译控制器
cd unitree_rl_lab/deploy/robots/g1_29dof
mkdir build && cd build
cmake .. && make
```

### 6.2 MuJoCo Sim2Sim

```bash
# 启动 MuJoCo 仿真
cd unitree_mujoco/simulate/build
./unitree_mujoco -i 0 -n eth0 -r g1 -s scene_29dof.xml

# 启动控制器
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl

# 操作：
# 1. L2+Up -> 站立
# 2. MuJoCo 窗口按 8 -> 足端接地
# 3. R1+X -> 启动策略
# 4. MuJoCo 窗口按 9 -> 禁用弹性带
```

### 6.3 Sim2Real

```bash
./g1_ctrl --network eth0
```

注意：需关闭板载控制程序。

---

## 七、Docker 支持

```bash
# 使用 docker-compose 构建
cd docker
docker-compose up --build -d

# 或直接构建
docker build --build-arg ISAACLAB_BASE_IMAGE_ARG=nvcr.io/nvidia/isaac-lab:2.3.0.4 \
    --build-arg DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG=/workspace \
    -t unitree_rl_lab ..
```

---

## 八、关键技术点

### 8.1 域随机化策略

1. **物理参数随机化**：摩擦系数、质量、质心
2. **初始化随机化**：关节位置、姿态
3. **运行时扰动**：周期性速度冲击
4. **观测噪声**：各观测项添加均匀噪声

### 8.2 课程学习

1. **地形课程**：从平地到复杂地形
2. **速度课程**：从低速到全速范围

### 8.3 模仿学习特色

1. **自适应采样**：根据失败率调整采样分布
2. **相对坐标变换**：消除全局位置依赖
3. **多连杆跟踪**：同时跟踪 14 个关键连杆

### 8.4 部署优化

1. **配置导出**：训练时自动生成部署配置
2. **ONNX 推理**：轻量级 CPU 推理
3. **线程分离**：策略推理与通信分离
4. **状态机管理**：安全的状态切换

---

## 九、配置示例

### G1-29DOF Velocity 训练配置

```yaml
# 仿真参数
sim:
  dt: 0.005
  render_interval: 4

# 环境参数
scene:
  num_envs: 4096
  env_spacing: 2.5

# MDP 参数
decimation: 4
episode_length_s: 20.0

# 动作配置
actions:
  JointPositionAction:
    scale: 0.25
    use_default_offset: true

# 奖励权重
rewards:
  track_lin_vel_xy: 1.0
  track_ang_vel_z: 0.5
  alive: 0.15
  base_height: -10.0
  gait: 0.5
```

---

## 十、开源依赖

| 项目 | 用途 |
|------|------|
| [IsaacLab](https://github.com/isaac-sim/IsaacLab) | 训练基础框架 |
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl) | RL 算法库 |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Sim2Sim 仿真 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | 策略推理 |
| [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) | 机器人通信 |
| [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) | MuJoCo 仿真接口 |
| [robot_lab](https://github.com/fan-ziqi/robot_lab) | 项目结构参考 |
| [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) | 全身运动跟踪 |
