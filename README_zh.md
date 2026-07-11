# Unitree RL Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat\&logo=Discord\&logoColor=white)](https://discord.gg/ZwcVwxv5rq)

## 概述

本项目提供了一套基于 [IsaacLab](https://github.com/isaac-sim/IsaacLab) 构建的 Unitree 机器人强化学习环境。

目前支持 Unitree **Go2**、**H1** 和 **G1-29dof** 机器人。

<div align="center">


| <div align="center"> Isaac Lab 仿真 </div>                                                                                     | <div align="center"> Mujoco 仿真 </div>                                                                                           | <div align="center"> 实物 </div>                                                                                                |
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


    conda run -n env_isaaclab_sim5 python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity --num_envs 18000 --resume --load_run 2026-07-03_19-13-44

    conda run -n env_isaaclab_sim5 python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity-Rough --num_envs 18000 --resume --load_run 2026-07-07_12-32-09



    conda run -n env_isaaclab_sim5 python scripts/rsl_rl/train.py --headless --task \
    Unitree-Go2-Velocity --num_envs 10000  --video \
    --resume --load_run 2026-07-02_23-40-22

    python scripts/rsl_rl/train.py --headless --task  Unitree-Go2-Velocity  --resume  --num_envs 1000

    查看 terrain_level 涨没涨的命令
    查看这轮训练的 terrain_level 涨势
    ./scripts/check_terrain_level.sh

    ```

# 使用训练好的智能体进行推理：

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


| 指标               | 含义                       | 理想值             |
| ------------------ | -------------------------- | ------------------ |
| `success_rate_pct` | 成功率（正常结束，非摔倒） | 越高越好           |
| `mean_reward`      | 平均奖励                   | 越高越好           |
| `std_reward`       | 奖励标准差                 | 越低越稳定         |
| `mean_ep_length`   | 平均回合步数               | 接近 max_ep_length |
| `fall_rate_pct`    | 摔倒率                     | 越低越好           |
| `fps`              | 推理速度                   | 参考值             |

## tensorboard 查看最新的 terrain\_level 值 ，然后在输出信息中可以点击网页查看曲线

```
tensorboard --logdir logs/rsl_rl/unitree_go2_velocity/
# 或在命令行：
tensorboard --logdir logs/rsl_rl/unitree_go2_velocity/ --tag Curriculum/terrain_levels
```
## 部署

模型训练完成后，需要在 Mujoco 中对训练好的策略进行 Sim2Sim 测试，以验证模型性能。
然后进行 Sim2Real 部署。

### 设置

```bash
# 安装依赖
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
# 安装 unitree_sdk2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build    ./unitree_rl_lab.sh -p --task Unitree-Go2-Velocity

cmake .. -DBUILD_EXAMPLES=OFF # 安装到 /usr/local 目录
sudo make install
# 编译 robot_controller
cd unitree_rl_lab/deploy/robots/g1_29dof # 或其他机器人
mkdir build && cd build
cmake .. && make
```
### Sim2Sim

安装 [unitree\_mujoco](https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation)。

- 在 `/simulate/config.yaml` 中设置 `robot` 为 g1
- 设置 `domain_id` 为 0
- 设置 `enable_elastic_hand` 为 1
- 设置 `use_joystck` 为 1

```bash
# 启动仿真
cd unitree_mujoco/simulate/build
./unitree_mujoco
# ./unitree_mujoco -i 0 -n eth0 -r g1 -s scene_29dof.xml # 备选方式
```
```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
# 1. 按 [L2 + Up] 让机器人站起
# 2. 点击 mujoco 窗口，然后按 8 让机器人脚接触地面
# 3. 按 [R1 + X] 运行策略
# 4. 点击 mujoco 窗口，然后按 9 禁用弹性带
```
### Sim2Real

你可以使用此程序直接控制机器人，但请确保已关闭机载控制程序。

```bash
./g1_ctrl --network eth0 # eth0 是网络接口名称
```
## 致谢

本项目的开发离不开以下开源项目的支持和贡献。特别感谢：

- [IsaacLab](https://github.com/isaac-sim/IsaacLab)：训练和运行代码的基础框架
- [mujoco](https://github.com/google-deepmind/mujoco.git)：提供强大的仿真功能
- [robot\_lab](https://github.com/fan-ziqi/robot_lab)：项目结构和部分实现的参考
- [whole\_body\_tracking](https://github.com/HybridRobotics/whole_body_tracking)：用于运动跟踪的多功能全身控制框架
