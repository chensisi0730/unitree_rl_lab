# Unitree RL Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat\&logo=Discord\&logoColor=white)](https://discord.gg/ZwcVwxv5rq)

## 概述

本项目提供了一套基于 [IsaacLab](https://github.com/isaac-sim/IsaacLab) 构建的 Unitree 机器人强化学习环境。

目前支持 Unitree **Go2**、**H1** 和 **G1-29dof** 机器人。

<div align="center">

| <div align="center"> Isaac Lab 仿真 </div>                                                                                       | <div align="center"> Mujoco 仿真 </div>                                                                                             | <div align="center"> 实物 </div>                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| [<img src="https://oss-global-cdn.unitree.com/static/d879adac250648c587d3681e90658b49_480x397.gif" width="240px">](g1_sim.gif) | [<img src="https://oss-global-cdn.unitree.com/static/3c88e045ab124c3ab9c761a99cb5e71f_480x397.gif" width="240px">](g1_mujoco.gif) | [<img src="https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif" width="240px">](g1_real.gif) |

</div>

## 安装

- 按照 [安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Lab。
- 安装 Unitree RL IsaacLab 独立环境。
  - 将此仓库克隆或复制到 Isaac Lab 安装目录之外：
    ```bash
    git clone https://github.com/unitreerobotics/unitree_rl_lab.git
    ```
  - 使用已安装 Isaac Lab 的 Python 解释器，以可编辑模式安装库：
    ```bash
    conda activate env_isaaclab_sim5
    ./unitree_rl_lab.sh -i
    # 重启shell以激活环境变更
    ```
- 下载 Unitree 机器人描述文件

  *方法1：使用 USD 文件*
  - 从 [unitree\_model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main) 下载 unitree usd 文件，保持文件夹结构
    ```bash
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
    ```
  - 在 `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py` 中配置 `UNITREE_MODEL_DIR`。
    ```bash
    UNITREE_MODEL_DIR = "</home/user/projects/unitree_usd>"
    ```
  *方法2：使用 URDF 文件 \[推荐]* 仅适用于 Isaacsim >= 5.0
  - 从 [unitree\_ros](https://github.com/unitreerobotics/unitree_ros) 下载 unitree 机器人 urdf 文件
    ```
    git clone https://github.com/unitreerobotics/unitree_ros.git
    ```
  - 在 `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py` 中配置 `UNITREE_ROS_DIR`。
    ```bash
    UNITREE_ROS_DIR = "</home/user/projects/unitree_ros/unitree_ros>"
    ```
  - \[可选]：如果想使用 urdf 文件，修改 *robot\_cfg.spawn* 配置
- 验证环境是否正确安装：
  - 列出可用任务：
    ```bash
    ./unitree_rl_lab.sh -l # 这是比 isaaclab 更快的版本
    ```
  - 运行训练一个任务：
    ```bash
        基础用法（录制视频）
    python scripts/rsl_rl/train.py --task Unitree-Go2-Flat --video
    ./unitree_rl_lab.sh -t --task Unitree-Go2-Velocity   --resume  --load_run 2026-06-22_23-16-30 --checkpoint model_6000  --num_envs 10000
    ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity # 支持任务名自动补全
    # 等效于
    python scripts/rsl_rl/train.py --headless --task  Unitree-Go2-Velocity  --resume  --num_envs 1000

    conda run -n env_isaaclab_sim5 python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity --num_envs 10000 --resume --load_run 2026-07-02_23-38-01

    conda run -n env_isaaclab_sim5 python scripts/rsl_rl/train.py   --task Unitree-G1-29dof-Velocity-Rough    --headless   --max_iterations 100000   --num_envs 14096    --resume    

    conda run -n env_isaaclab_sim5 python scripts/rsl_rl/train.py --headless --task \
    Unitree-Go2-Velocity --num_envs 10000 \
    --resume --load_run 2026-06-30_23-52-26xx

    查看 terrain_level 涨没涨的命令
    查看这轮训练的 terrain_level 涨势
    ./scripts/check_terrain_level.sh

    ```
  - 使用训练好的智能体进行推理：
    ```bash
    ./unitree_rl_lab.sh -p --task Unitree-Go2-Velocity
    ./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity # 支持任务名自动补全
    # 等效于
    python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity  --load_run logs/rsl_rl/unitree_go2_velocity/2026-06-23_16-34-57


    conda run -n env_isaaclab_sim5 python scripts/rsl_rl/play.py --headless --task Unitree-Go2-Velocity --load_run 2026-06-25_10-08-28

    python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity

    ```

### 评估训练效果

使用独立的 `eval.py` 脚本量化评估 checkpoint 性能，运行固定步数后输出指标并退出：

```bash
# 评估指定 checkpoint
conda run -n env_isaaclab_sim5 python scripts/rsl_rl/eval.py \
    --task Unitree-Go2-Velocity \
    --load_run 2026-06-30_23-52-26 \
    --checkpoint model_6000 \
    --eval-steps 64000

# 评估最新 checkpoint（省略 --checkpoint）
conda run -n env_isaaclab_sim5 python scripts/rsl_rl/eval.py \
    --task Unitree-Go2-Velocity-Stairs \
    --load_run 2026-06-30_23-52-26 \
    --eval-steps 64000

# 查看评估结果
cat logs/rsl_rl/unitree_go2_velocity/2026-06-30_23-52-26/eval_result.json
```

**输出指标说明**：

| 指标 | 含义 | 理想值 |
|---|---|---|
| `success_rate_pct` | 成功率（正常结束，非摔倒） | 越高越好 |
| `mean_reward` | 平均奖励 | 越高越好 |
| `std_reward` | 奖励标准差 | 越低越稳定 |
| `mean_ep_length` | 平均回合步数 | 接近 max_ep_length |
| `fall_rate_pct` | 摔倒率 | 越低越好 |
| `fps` | 推理速度 | 参考值 |

## tensorboard 查看最新的 terrain\_level 值 ，然后在输出信息中可以点击网页查看曲线

```
tensorboard --logdir logs/rsl_rl/unitree_go2_velocity/
# 或在命令行：
tensorboard --logdir logs/rsl_rl/unitree_go2_velocity/ --tag Curriculum/terrain_levels
```

## 部署

模型训练完成后，需将策略导出为 ONNX 格式，再部署到 Mujoco Sim2Sim 或实物机器人。

### 1. 导出策略（ONNX）

```bash
# 训练完成后，使用 play.py 自动导出 ONNX/JIT
conda run -n env_isaaclab_sim5 python scripts/rsl_rl/play.py \
  --task Unitree-Go2-Velocity \
  --headless \
  --checkpoint /path/to/logs/rsl_rl/unitree_go2_velocity/YYYY-MM-DD_HH-MM-SS/model_XXXX.pt

# 导出的文件在 checkpoint 同目录下的 exported/ 文件夹：
#   policy.onnx  — 用于部署
#   policy.pt    — 用于 JIT 推理
```

### 2. 配置部署参数

编辑机器人对应的 `deploy/robots/<robot>/config/config.yaml`，设置 `policy_dir` 指向训练日志目录：

```yaml
# deploy/robots/go2/config/config.yaml 示例
Velocity:
  transitions: 
    Passive: LT + B.on_pressed
  policy_dir: ../../../logs/rsl_rl/unitree_go2_velocity/YYYY-MM-DD_HH-MM-SS
```

### 3. 编译 C++ 控制器

```bash
# 安装依赖
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev

# 安装 unitree_sdk2（仅 Sim2Real 需要，Sim2Sim 可跳过）
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF
sudo make install

# 编译 robot_controller
cd unitree_rl_lab/deploy/robots/g1_29dof  # 替换为目标机器人
mkdir -p build && cd build
cmake ..
make
```

编译产物为 `g1_ctrl`（或对应机器人名称的二进制文件），包含 ONNX 运行时推理。

### 4. Sim2Sim（Mujoco 仿真测试）

安装 [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation)。

配置 `/simulate/config.yaml`：

```yaml
robot: g1              # 目标机器人
domain_id: 0
enable_elastic_hand: 1  # G1 弹性手
use_joystick: 1         # 手柄控制
```

```bash
# 终端 1：启动 Mujoco 仿真
cd unitree_mujoco/simulate/build
./unitree_mujoco

# 终端 2：启动机器人控制器
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
```

操作流程：

| 步骤 | 按键 | 说明 |
|------|------|------|
| 1 | L2 + Up | 机器人站立 |
| 2 | 点击 Mujoco 窗口 + 按 8 | 脚接触地面 |
| 3 | R1 + X | 运行策略 |
| 4 | 点击 Mujoco 窗口 + 按 9 | 禁用弹性带 |
| 5 | 左摇杆 | 控制前后/左右移动 |
| 6 | 右摇杆 | 控制转向 |

### 5. Sim2Real（实物部署）

```bash
# 确保实物机器人的机载控制程序已关闭
./g1_ctrl --network eth0  # eth0 为网络接口名称
```

操作流程与 Sim2Sim 相同，控制器会通过 unitree_sdk2 直接驱动实物机器人。

---

## G1 人形机器人 — 复杂地形训练

### 任务说明

| 任务 ID | 地形 | 用途 |
|---------|------|------|
| `Unitree-G1-29dof-Velocity` | 平地 | 原始速度跟踪 |
| `Unitree-G1-29dof-Velocity-Rough` | **复杂混合地形** | 楼梯/坡道/粗糙地面 |

### 地形组成

文件: `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/rough_terrain_env_cfg.py`

| 地形类型 | 比例 | 难度范围 |
|---------|:---:|:--------:|
| 平地 | 20% | 最简单 |
| 随机粗糙地面 | 20% | 0.01~0.10m 起伏 |
| 金字塔斜坡 | 15% | 0°~22° 坡度 |
| 倒金字塔斜坡 | 15% | 0°~22° 坡度 |
| 金字塔楼梯 | 15% | 5~25cm 阶梯高度 |
| 倒金字塔楼梯 | 15% | 5~25cm 阶梯高度 |

> G1 的楼梯 `step_width=0.4m`（比 GO2 的 0.3m 更宽，适配人形步幅）

### 训练命令

```bash
# 平地训练
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity

# 复杂地形训练（从零开始）
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity-Rough --max_iterations 100000

# 从平地 checkpoint 恢复训练（迁移学习，更快收敛）
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity-Rough \
  --resume \
  --load_run unitree_g1_29dof_velocity \
  --checkpoint model_XXXX.pt
```

### 评估命令

```bash
# 回放评估
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity-Rough \
  --checkpoint /path/to/model_XXXX.pt

# 量化评估
conda run -n env_isaaclab_sim5 python scripts/rsl_rl/eval.py \
  --task Unitree-G1-29dof-Velocity-Rough \
  --load_run unitree_g1_29dof_velocity_rough \
  --checkpoint model_XXXX \
  --eval-steps 64000
```

### G1 vs GO2 训练差异

| 特性 | GO2 (四足) | G1 (人形) |
|------|-----------|----------|
| 自由度 | 12 (腿) | 29 (腿+臂+腰) |
| 机身 | base | torso_link |
| 观察历史 | 无 history | history_length=5 |
| 奖励项 | ~16 项 | ~21 项（含步态/手臂/腰部/足高） |
| 速度上限 | ±1.0 m/s | -0.5~+1.0 m/s |
| 楼梯步宽 | 0.3m | 0.4m |
| 终止条件 | base_contact | base_height < 0.2m |
| 地形课程 | 默认可升降 | 只升不降（避免震荡） |

### 课程机制说明

G1 复杂地形使用 `terrain_levels_vel_stairs_only_up` 课程函数：

```
move_up:   distance > max(3.0m, v_cmd × 6s)    → 升级到更难地形
move_down: 永不                                   → 只升不降
```

防止机器人在复杂地形上因频繁摔倒 → 走得太短 → 被降级到更简单地形的恶性循环。

## 致谢

本项目的开发离不开以下开源项目的支持和贡献。特别感谢：

- [IsaacLab](https://github.com/isaac-sim/IsaacLab)：训练和运行代码的基础框架
- [mujoco](https://github.com/google-deepmind/mujoco.git)：提供强大的仿真功能
- [robot\_lab](https://github.com/fan-ziqi/robot_lab)：项目结构和部分实现的参考
- [whole\_body\_tracking](https://github.com/HybridRobotics/whole_body_tracking)：用于运动跟踪的多功能全身控制框架

