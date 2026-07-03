# Unitree RL Lab（源码结构解析版）

本文件面向“读源码 / 改环境 / 做训练并部署”的使用场景，重点解释工程各目录职责、Python 训练链路与 C++ 部署链路如何通过配置文件和导出的模型衔接。

---

## 1. 项目定位与两条主链路

该仓库包含两部分能力：

1) **Python（IsaacLab）训练侧**  
基于 IsaacLab 的 `ManagerBasedRLEnv` 环境体系，提供 Unitree 机器人（Go2 / H1 / G1-29dof 等）的训练任务、MDP 组件（观测/奖励/终止/课程/指令等）、以及基于 RSL-RL 的训练与推理脚本。

2) **C++（unitree_sdk2）部署侧**  
提供一个面向真实机器人（或 Mujoco 仿真）的控制程序：通过 **ONNXRuntime** 加载 `policy.onnx`，并读取训练侧导出的 `deploy.yaml` 来完成观测拼接、动作后处理与关节映射；以 FSM（有限状态机）形式组织控制模式（Passive/FixStand/RLBase/Mimic）。

训练与部署的“契约”主要由两件产物定义：

- `exported/policy.onnx`：RSL-RL policy 导出的 ONNX 模型（推理用）
- `params/deploy.yaml`：由训练环境自动导出的“部署配置”（观测/动作/关节映射/控制周期等）

---

## 2. 目录结构速览（从入口到细节）

### 2.1 仓库根目录

- `README.md`：官方英文说明（安装、训练、部署步骤）
- `unitree_rl_lab.sh`：本仓库常用入口脚本（安装/列任务/训练/推理）
- `scripts/`：训练/推理/列任务等脚本（Python）
- `source/unitree_rl_lab/`：IsaacLab 扩展（Python 包主体：任务、资产、工具）
- `deploy/`：部署侧 C++ 工程（unitree_sdk2 + ONNXRuntime + YAML 配置 + FSM）
- `docker/`：容器环境（Isaac/依赖构建相关）
- `doc/licenses/`：第三方许可证

### 2.2 Python 扩展主体：`source/unitree_rl_lab/unitree_rl_lab/`

- `assets/robots/`：Unitree 机器人资产配置（USD/URDF 路径、关节名、执行器参数）
- `tasks/`：训练任务注册与任务配置（locomotion / mimic）
- `utils/`：训练/推理辅助（例如导出部署配置、解析 env cfg）

### 2.3 部署侧：`deploy/`

- `include/`：部署运行时的“轻量 IsaacLab runtime”（C++）与 FSM 框架头文件
  - `isaaclab/`：C++ 端的观测/动作/终止/环境管理器/ONNX 推理封装
  - `FSM/`：状态机基类与状态注册宏
- `robots/<robot>/`：各机器人控制程序（CMake、main、具体 State 实现、config）
  - `config/config.yaml`：FSM 配置（启用哪些状态、按键切换、policy_dir 等）
  - `config/policy/...`：示例策略（`exported/policy.onnx` + `params/deploy.yaml` + 可能的动作/动作捕捉参数）

---

## 3. Python 训练侧：任务注册、环境配置、训练/推理脚本

### 3.1 任务如何被注册到 Gymnasium

入口在：

- `source/unitree_rl_lab/unitree_rl_lab/tasks/__init__.py`：调用 `isaaclab_tasks.utils.import_packages` 自动导入子包，从而触发各任务的 `gym.register(...)`。

每个任务通常由某个机器人子包的 `__init__.py` 完成注册，例如：

- `.../tasks/locomotion/robots/go2/__init__.py` 注册 `Unitree-Go2-Velocity`
- `.../tasks/locomotion/robots/h1/__init__.py` 注册 `Unitree-H1-Velocity`
- `.../tasks/locomotion/robots/g1/29dof/__init__.py` 注册 `Unitree-G1-29dof-Velocity`
- `.../tasks/mimic/robots/g1_29dof/dance_102/__init__.py` 注册 `Unitree-G1-29dof-Mimic-Dance-102`

注册信息里最关键的是 kwargs 的三个“入口点键”：

- `env_cfg_entry_point`：训练用环境 cfg（Hydra/registry 读取）
- `play_env_cfg_entry_point`：推理/演示用环境 cfg（通常关闭随机化、调低并行数等）
- `rsl_rl_cfg_entry_point`：RSL-RL runner/算法 cfg

这也解释了为什么训练脚本只需要 `--task Unitree-...`：所有 cfg 都通过 registry 追到对应 Python 类。

### 3.2 环境 cfg 的典型结构（ManagerBasedRLEnvCfg）

以 `G1-29dof` 速度跟踪为例（`.../tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py`）：

- `RobotSceneCfg`：地形/光照/机器人/传感器（RayCaster/ContactSensor 等）
- `EventCfg`：域随机化与 reset/interval 事件（材质、质量、外力、推搡等）
- `CommandsCfg`：命令生成（如 `base_velocity`）
- `ActionsCfg`：动作定义（常见为 `JointPositionAction`，支持 scale/offset/clip）
- `ObservationsCfg`：观测组（policy/critic），支持 history_length、噪声、scale/clip
- `RewardsCfg`：奖励项组合
- `TerminationsCfg`：终止条件
- `CurriculumCfg`：课程学习项（例如 terrain level、指令等级）

这套结构在 locomotion 与 mimic 两条任务线一致，只是 command/obs/reward 的具体项不同。

### 3.3 训练入口：`scripts/rsl_rl/train.py`

训练脚本做了几件关键的工程化工作：

1) **根据 `--task` 自动加载 env_cfg 与 agent_cfg**  
通过 `isaaclab_tasks.utils.hydra.hydra_task_config` 与注册表 entry point。

2) **创建 Gym 环境并包一层 RSL-RL wrapper**  
把 IsaacLab env 包装为 `RslRlVecEnvWrapper`，交给 `rsl_rl.runners.OnPolicyRunner`。

3) **日志结构与产物落盘**  
默认日志目录：

`logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/`

其中会落盘：

- `params/env.yaml`、`params/agent.yaml`
- 环境 cfg 的源码文件副本（便于复现实验）

4) **导出部署配置：`params/deploy.yaml`**  
训练脚本会调用 `unitree_rl_lab.utils.export_deploy_cfg.export_deploy_cfg(env, log_dir)`，把部署侧需要的信息（关节映射、控制周期、观测/动作定义等）导出到 `params/deploy.yaml`。

这是 Python 训练侧与 C++ 部署侧衔接的核心。

### 3.4 推理入口与 ONNX 导出：`scripts/rsl_rl/play.py`

推理脚本会：

- 解析 play 环境 cfg（`entry_point_key="play_env_cfg_entry_point"`）
- 加载 checkpoint（日志目录或外部路径）
- 运行 policy 推理（可选 real-time）
- **自动导出**：
  - `exported/policy.pt`
  - `exported/policy.onnx`

部署侧通常使用 `exported/policy.onnx`。

### 3.5 列出本仓库提供的任务：`scripts/list_envs.py`

`./unitree_rl_lab.sh -l` 本质会运行 `scripts/list_envs.py`：

- 通过导入 `tasks/locomotion.robots` 与 `tasks/mimic.robots` 子包触发注册
- 遍历 `gym.registry`，筛选 `id` 包含 `Unitree` 的条目并打印表格

---

## 4. Unitree 机器人资产配置：关节名、执行器、资源路径

主要文件：

- `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`

其中包含：

- `UNITREE_MODEL_DIR` / `UNITREE_ROS_DIR`：USD/URDF 资源根目录（需根据本地路径配置）
- 不同机器人的 `ArticulationCfg`：
  - `UNITREE_GO2_CFG`
  - `UNITREE_GO2W_CFG`
  - `UNITREE_H1_CFG`
  - `UNITREE_G1_29DOF_CFG`
  - 以及 mimic 专用 cfg 与 action scale（例如 `UNITREE_G1_29DOF_MIMIC_ACTION_SCALE`）

重点概念：`joint_sdk_names`

- 训练侧：IsaacLab 内部关节顺序来自 USD/URDF 的关节名列表
- 部署侧：unitree_sdk2 的关节顺序是 SDK 定义的顺序
- `export_deploy_cfg` 会基于 `joint_sdk_names` 与仿真关节名做匹配，导出 `joint_ids_map`，确保部署侧写入 lowcmd 时能对上正确电机。

---

## 5. C++ 部署侧：FSM + 轻量环境 + ONNX 推理

### 5.1 部署侧核心思路

部署侧并不是把 IsaacSim/IsaacLab 搬到 C++，而是实现了一个**最小闭环**：

- 从机器人读取状态（位置/速度/IMU 等）
- 按训练时同样的规则构造观测向量（ObservationManager）
- 调用 ONNXRuntime 推理得到动作（OrtRunner）
- 按训练时同样的规则做动作后处理（ActionManager：scale/clip/offset 等）
- 写入 unitree_sdk2 lowcmd（目标关节位置等）

### 5.2 `deploy.yaml` 在 C++ 侧如何被使用

入口类：

- `deploy/include/isaaclab/envs/manager_based_rl_env.h`

构造函数会解析 `deploy.yaml` 的关键字段：

- `step_dt`：控制周期（训练侧 `sim.dt * decimation`）
- `joint_ids_map`：关节顺序映射
- `default_joint_pos`、`stiffness`、`damping`：用于初始化/设置 PD 增益
- `actions`：动作项配置（用于 ActionManager 实例化与处理）
- `observations`：观测项配置（用于 ObservationManager 拼接输入）
- `commands`：某些观测/逻辑可能需要用到命令范围（例如键盘速度指令示例）

### 5.3 ONNX 推理封装：`OrtRunner`

文件：

- `deploy/include/isaaclab/algorithms/algorithms.h`

特点：

- 自动读取模型 input/output 名称与形状
- 要求 `ObservationManager` 产出的 map 包含模型所需的所有 input
- 输出 action 向量写入 `action` 缓存（线程安全）

### 5.4 FSM 状态机与 G1-29dof 示例

以 `deploy/robots/g1_29dof/` 为例：

- `config/config.yaml` 定义：
  - 启用哪些状态（`Passive` / `FixStand` / `Velocity` / `Mimic_*`）
  - 按键切换条件（例如 `LT + up.on_pressed`）
  - `policy_dir`（策略目录：包含 `exported/` 与 `params/`）

核心状态：

- `State_RLBase`（`deploy/include/FSM/State_RLBase.h` + `deploy/robots/g1_29dof/src/State_RLBase.cpp`）
  - 进入状态时设置 PD（kp/kd）
  - 启动一个固定周期线程：`env->reset(); while(running) env->step(); sleep_until(...)`
  - `run()` 中把 `processed_actions()` 写入 lowcmd 的关节目标位置

- `State_Mimic`（`deploy/robots/g1_29dof/src/State_Mimic.cpp`）
  - 加载动作捕捉 CSV
  - 每步更新 motion reference，再推理得到动作
  - 内置超时/姿态异常检查，触发 FSM 切换

### 5.5 策略目录 `policy_dir` 的选择规则

`param::parser_policy_dir(...)`（`deploy/include/param.h`）会：

- 若 `policy_dir/exported` 不存在，则尝试在 `policy_dir` 下找子目录，并从后往前选择第一个包含 `exported/` 的目录

因此 `policy_dir` 既可以指向：

- 一个具体 run 目录（里面直接有 `exported/`、`params/`）
- 也可以指向某个 experiment 根目录，让程序自动挑“最新一个可用导出”

---

## 6. 典型工作流（训练 → 导出 → 部署）

### 6.1 训练并产生部署所需文件

训练（headless）：

```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity
```

训练结束后，在对应日志目录中应当能看到：

- `logs/rsl_rl/<experiment>/<run>/exported/policy.onnx`
- `logs/rsl_rl/<experiment>/<run>/params/deploy.yaml`

### 6.2 把策略提供给部署程序

有两种常见做法：

1) **直接让 deploy 读取 logs**  
把 `deploy/robots/<robot>/config/config.yaml` 里的 `policy_dir` 指到日志 experiment 或 run 目录（示例中已有注释行）。

2) **复制到 deploy 的 `config/policy/` 下（更可控）**  
确保目标目录结构包含：

- `.../policy_dir/exported/policy.onnx`
- `.../policy_dir/params/deploy.yaml`

### 6.3 运行部署程序（Sim2Sim / Sim2Real）

部署侧构建与运行的具体依赖（unitree_sdk2、yaml-cpp、onnxruntime 等）与步骤以 `README.md` 的 Deploy 小节为准；本文件只解释源码链路与配置契约。

---

## 7. 扩展与修改建议（从源码角度）

### 7.1 新增一个训练任务（Python）

最小闭环通常包括：

1) 新建一个任务包并在 `__init__.py` 里 `gym.register(...)`  
提供 `env_cfg_entry_point` / `play_env_cfg_entry_point` / `rsl_rl_cfg_entry_point`

2) 编写新的 `*env_cfg.py`（ManagerBasedRLEnvCfg）  
定义 scene/actions/obs/rewards/terminations 等

3) 确保训练脚本能通过 `--task` 找到并加载这些 entry point

### 7.2 让任务“可部署”（Python ↔ C++）

部署侧只能理解它实现过的 observation/action term。  
当你在 Python 侧修改了 `ObservationsCfg` 或 `ActionsCfg`：

- 训练侧可以正常跑，不代表部署侧能用
- 若导出的 `deploy.yaml` 包含 C++ 侧未注册的观测/动作 term，会在 C++ 侧抛错（“not registered”）

对应扩展点在：

- 观测注册宏：`REGISTER_OBSERVATION(name)`（`deploy/include/isaaclab/manager/observation_manager.h`）
- 动作注册宏：`REGISTER_ACTION(name)`（`deploy/include/isaaclab/manager/action_manager.h`）

建议流程：

1) 先在 Python 侧确定最终的 obs/action 列表（policy group）
2) 确认 deploy/include/isaaclab/envs/mdp/ 下是否已有对应实现
3) 没有则在 C++ 侧补齐同名 term（并确保顺序/缩放/clip 与训练一致）

---

## 8. 常见问题与排查点（读源码时最容易踩）

1) **资源路径未配置**  
`UNITREE_MODEL_DIR` / `UNITREE_ROS_DIR` 未指向真实数据，会导致仿真加载失败。

2) **关节顺序不一致**  
部署侧写 lowcmd 时使用 `joint_ids_map` 做映射；若 `joint_sdk_names` 或仿真关节名匹配失败，会直接影响动作落到错误电机上。

3) **策略目录结构不完整**  
部署侧需要同时看到：
`exported/policy.onnx` 与 `params/deploy.yaml`。缺任何一个都会失败。

4) **lowcmd 通道被占用**  
G1-29dof 的 `main.cpp` 会检测 lowcmd 通道是否被其他进程占用（避免双写控制）。

---

## 9. 源码导航（推荐从这些入口读）

- 统一入口脚本：`unitree_rl_lab.sh`
- 训练：`scripts/rsl_rl/train.py`
- 推理+导出 ONNX：`scripts/rsl_rl/play.py`
- 部署配置导出：`source/unitree_rl_lab/unitree_rl_lab/utils/export_deploy_cfg.py`
- G1-29dof 部署状态：`deploy/robots/g1_29dof/src/State_RLBase.cpp`、`deploy/robots/g1_29dof/src/State_Mimic.cpp`
- 部署侧轻量环境：`deploy/include/isaaclab/envs/manager_based_rl_env.h`

