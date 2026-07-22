# Unitree G1 人形机器人项目规划

> 基于宇树官方开源生态，从仿真复现到自主抓取操作的完整路线图

---

## 目录

- [1. 项目总览](#1-项目总览)
- [2. 硬件与软件环境](#2-硬件与软件环境)
- [3. 项目架构与仓库关系](#3-项目架构与仓库关系)
- [4. 阶段一：仿真环境搭建与基础验证](#4-阶段一仿真环境搭建与基础验证)
- [5. 阶段二：RL运动控制复现](#5-阶段二rl运动控制复现)
- [6. 阶段三：遥操作数据采集](#6-阶段三遥操作数据采集)
- [7. 阶段四：模仿学习与操作策略训练](#7-阶段四模仿学习与操作策略训练)
- [8. 阶段五：VLA大模型与自主抓取](#8-阶段五vla大模型与自主抓取)
- [9. 阶段六：Sim2Real实物部署](#9-阶段六sim2real实物部署)
- [10. 关键技术要点与注意事项](#10-关键技术要点与注意事项)
- [11. 进度追踪](#11-进度追踪)

---

## 1. 项目总览

### 1.1 目标

复现宇树官方G1机器人的全部效果，最终实现**自主遵循指令进行抓取等操作**。

### 1.2 整体路线

```
仿真环境搭建 → RL运动控制 → 遥操作数据采集 → 模仿学习/操作策略 → VLA大模型 → Sim2Real部署
```

### 1.3 G1机器人配置

| 配置项 | 说明 |
|--------|------|
| 机器人型号 | Unitree G1 (29自由度) |
| 腿部自由度 | 12 (每腿6: 髋3+膝1+踝2) |
| 腰部自由度 | 3 (waist_yaw + waist_roll + waist_support) |
| 手臂自由度 | 14 (每臂7: 肩3+肘1+腕3) |
| 末端执行器 | Dex1-1 二指夹爪 / Dex3-1 灵巧手 / 因时灵巧手 |
| 传感器 | 六维力传感器 + 触觉灵巧手 + 头部相机 + 胸部IMU |
| 通信协议 | unitree_hg IDL (DDS) |

---

## 2. 硬件与软件环境

### 2.1 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | RTX 3080 (10GB) | RTX 4090 (24GB) |
| CPU | 8核 | 16核以上 |
| 内存 | 32GB | 64GB |
| 存储 | 500GB SSD | 1TB NVMe SSD |
| XR设备 | - | Apple Vision Pro / PICO 4 Ultra / Meta Quest 3 |
| 网络 | 千兆以太网 | 千兆以太网 (连接机器人) |

### 2.2 软件环境

| 环境 | 用途 | Conda环境名 |
|------|------|-------------|
| Isaac Gym Preview 4 | RL训练 (unitree_rl_gym) | `unitree-rl-PY38` |
| MuJoCo + mjlab | RL训练 (unitree_rl_mjlab) | 独立安装 |
| Isaac Sim 4.5/5.x + Isaac Lab | 仿真场景/数据采集 | `unitree_sim_env` |
| LeRobot | 模仿学习训练 | `unitree_lerobot` |
| UnifoLM-VLA | VLA大模型训练推理 | `unifolm-vla` |
| UnifoLM-WMA | 世界模型训练推理 | `unifolm-wma` |
| xr_teleoperate | 遥操作 | `tv` |

### 2.3 核心依赖

```bash
# 通用依赖
sudo apt install libyaml-cpp-dev libspdlog-dev libboost-all-dev libglfw3-dev

# DDS通信
# unitree_sdk2 (C++): 安装到 /opt/unitree_robotics
# unitree_sdk2_python (Python): pip install -e .

# MuJoCo: 下载安装到 ~/.mujoco
# Isaac Gym Preview 4: NVIDIA官方下载
# Isaac Sim 4.5/5.x + Isaac Lab: NVIDIA官方安装
```

---

## 3. 项目架构与仓库关系

```
unitree/                                 # 项目根目录
├── unitree_mujoco/                      # MuJoCo仿真器 (Sim2Sim验证)
│   ├── unitree_robots/g1/               # G1 MJCF模型文件
│   │   ├── g1_23dof.xml                 # 23自由度模型
│   │   ├── g1_29dof.xml                 # 29自由度模型
│   │   ├── scene.xml / scene_29dof.xml  # 仿真场景
│   │   └── g1_joint_index_dds.md        # 关节索引说明
│   ├── simulate/                        # C++仿真器
│   └── simulate_python/                 # Python仿真器
│
├── unitree_rl_gym/                      # RL训练 (Isaac Gym后端)
│   ├── legged_gym/envs/g1/             # G1环境配置
│   ├── legged_gym/scripts/             # 训练/验证脚本
│   └── deploy/                          # 部署工具 (Sim2Sim/Sim2Real)
│
├── unitree_rl_mjlab/                    # RL训练 (MuJoCo后端, 推荐)
│   ├── src/assets/robots/unitree_g1/    # G1模型与常量
│   ├── src/assets/motions/g1/           # 动作模仿参考动作
│   ├── scripts/                         # 训练/验证脚本
│   └── deploy/robots/g1/               # G1部署程序
│
├── unitree_sim_isaaclab/                # Isaac Lab仿真场景
│   ├── tasks/g1_tasks/                  # G1操作任务 (抓取/堆叠等)
│   ├── dds/                             # DDS通信模块
│   └── sim_main.py                      # 仿真入口
│
├── xr_teleoperate/                      # XR遥操作
│   └── teleop/                          # 遥操作主程序
│
├── unitree_lerobot/                     # LeRobot模仿学习
│   ├── lerobot/                         # LeRobot框架
│   ├── utils/                           # 数据转换工具
│   └── eval_robot/                      # 实机评估
│
├── unifolm-vla/                         # VLA大模型
│   ├── src/unifolm_vla/                 # 核心库
│   └── scripts/                         # 训练/推理脚本
│
├── unifolm-world-model-action/          # 世界模型
│   ├── src/unitree_worldmodel/          # 核心库
│   └── unitree_deploy/                  # 部署工具
│
├── unitree_sdk2/                        # C++ SDK
├── unitree_sdk2_python/                 # Python SDK
└── unitree_ros2/                        # ROS2接口
```

### 仓库依赖关系图

```
                    ┌──────────────┐
                    │  unitree_mujoco │  ← 仿真验证基础
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
    ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐
    │unitree_rl_gym│ │unitree_rl_mjlab│ │unitree_sim_isaaclab│
    │(Isaac Gym)  │ │(MuJoCo+mjlab) │ │(Isaac Lab)        │
    └──────┬──────┘ └──────┬───────┘ └────────┬─────────┘
           │               │                   │
           │    RL运动控制  │    RL运动控制      │  操作仿真+数据采集
           ▼               ▼                   ▼
    ┌──────────────────────────────────────────────┐
    │              xr_teleoperate                   │  ← 遥操作数据采集
    └──────────────────┬───────────────────────────┘
                       │ 数据集
                       ▼
    ┌──────────────────────────────────────────────┐
    │              unitree_lerobot                  │  ← 模仿学习训练
    └──────────────────┬───────────────────────────┘
                       │
           ┌───────────┼───────────┐
           ▼                       ▼
    ┌────────────┐         ┌──────────────────┐
    │unifolm-vla │         │unifolm-world-model│  ← 大模型
    └────────────┘         └──────────────────┘
```

---

## 4. 阶段一：仿真环境搭建与基础验证

> 目标：在MuJoCo和Isaac Lab中加载G1模型，验证仿真器正常运行

### 4.1 MuJoCo仿真器搭建

#### 4.1.1 安装依赖

```bash
# 系统依赖
sudo apt install libyaml-cpp-dev libspdlog-dev libboost-all-dev libglfw3-dev

# 安装 unitree_sdk2 (C++)
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2 && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/unitree_robotics
sudo make install

# 安装 MuJoCo
# 下载 https://github.com/google-deepmind/mujoco/releases 解压到 ~/.mujoco
```

#### 4.1.2 编译与运行C++仿真器

```bash
cd unitree_mujoco/simulate/
ln -s ~/.mujoco/mujoco-3.3.6 mujoco
rm -rf build && mkdir build && cd build
cmake .. && make -j4

# 启动G1仿真 (29自由度)  OK
./unitree_mujoco -r g1 -s scene_29dof.xml
```

**注意**：G1使用 `unitree_hg` IDL消息，与Go2的 `unitree_go` 不同。

#### 4.1.3 Python仿真器

```bash
pip install mujoco pygame
pip install unitree_sdk2_python  # 或从源码安装

cd unitree_mujoco/simulate_python/
# 修改 config.py 中 ROBOT = "g1"
python unitree_mujoco.py
```

#### 4.1.4 验证清单

- [ ] MuJoCo仿真器正常启动，显示G1模型
- [ ] 虚拟挂带功能可用 (按键7/8/9)
- [ ] DDS通信正常 (LowCmd/LowState消息收发)
- [ ] 测试程序可控制G1关节运动

### 4.2 Isaac Lab仿真环境搭建

#### 4.2.1 安装Isaac Sim + Isaac Lab

```bash
# 方式1：自动安装脚本
cd unitree_sim_isaaclab
chmod +x auto_setup_env.sh
bash auto_setup_env.sh 4.5 unitree_sim_env  # Isaac Sim 4.5
# 或
bash auto_setup_env.sh 5.0 unitree_sim_env  # Isaac Sim 5.0

# 方式2：参考 doc/isaacsim4.5_install_zh.md 手动安装
```

#### 4.2.2 下载资产

```bash
sudo apt install git-lfs
cd unitree_sim_isaaclab
. fetch_assets.sh
```

#### 4.2.3 启动G1操作仿真

```bash
conda activate unitree_sim_env

# G1-29dof + Dex1夹爪 + 圆柱体抓取
python sim_main.py --device cpu --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint \
  --enable_dex1_dds --robot_type g129

# G1-29dof + Dex3灵巧手 + 圆柱体抓取
python sim_main.py --device cpu --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
  --enable_dex3_dds --robot_type g129
```

#### 4.2.4 验证清单

- [ ] Isaac Lab场景正常加载
- [ ] G1机器人模型正确显示
- [ ] DDS通信正常 (与unitree_sdk2_python互通)
- [ ] 相机图像正常输出
- [ ] 数据回放功能正常

---

## 5. 阶段二：RL运动控制复现

> 目标：复现官方的G1速度跟踪和动作模仿效果

### 5.1 方案选择

| 方案 | 仿真后端 | 优势 | 劣势 |
|------|----------|------|------|
| unitree_rl_gym | Isaac Gym (PhysX) | 官方最早方案，文档丰富 | Isaac Gym Preview 4已停更，仅支持Ubuntu20/22 |
| unitree_rl_mjlab | MuJoCo (mjlab) | 推荐方案，GPU加速，持续更新 | 较新，社区资源少 |

**推荐**：优先使用 `unitree_rl_mjlab`，同时保留 `unitree_rl_gym` 作为参考。

### 5.2 方案A：unitree_rl_mjlab (推荐)

#### 5.2.1 安装

参考 `unitree_rl_mjlab/doc/setup_zh.md`

```bash
# 安装mjlab及依赖
pip install mjlab
pip install mujoco
pip install rsl_rl
```

#### 5.2.2 速度跟踪训练

```bash
cd unitree_rl_mjlab

# G1 29自由度 平地速度跟踪
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096

# G1 23自由度 平地速度跟踪
python scripts/train.py Unitree-G1-23Dof-Flat --env.scene.num-envs=4096

# 多GPU训练
python scripts/train.py Unitree-G1-Flat --gpu-ids 0 1 --env.scene.num-envs=4096
```

#### 5.2.3 动作模仿训练

```bash
# 1. 准备动作文件 (CSV → NPZ)
python scripts/csv_to_npz.py \
  --input-file src/assets/motions/g1/dance1_subject2.csv \
  --output-name dance1_subject2.npz \
  --input-fps 30 --output-fps 50 --robot g1

# 2. 训练动作模仿
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1/dance1_subject2.npz \
  --env.scene.num-envs=4096
```

#### 5.2.4 仿真验证

```bash
# 速度跟踪验证
python scripts/play.py Unitree-G1-Flat \
  --checkpoint_file=logs/rsl_rl/g1_velocity/YYYY-MM-DD_HH-MM-SS/model_XX.pt

# 动作模仿验证
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1/dance1_subject2.npz \
  --checkpoint_file=logs/rsl_rl/g1_tracking/YYYY-MM-DD_HH-MM-SS/model_XX.pt
```

#### 5.2.5 Sim2Sim验证 (MuJoCo)

```bash
# 编译仿真器
cd unitree_rl_mjlab/simulate
mkdir build && cd build && cmake .. && make -j8

# 启动仿真器
./simulate/build/unitree_mujoco

# 启动控制程序
cd deploy/robots/g1/build
./g1_ctrl --network=lo
```

### 5.3 方案B：unitree_rl_gym

#### 5.3.1 安装

参考 `unitree_rl_gym/doc/setup_zh.md`，使用conda环境 `unitree-rl-PY38`

#### 5.3.2 训练

```bash
conda activate unitree-rl-PY38

# G1 12自由度 (仅腿部)
python legged_gym/scripts/train.py --task=g1 --headless

# G1 29自由度 (腿部+腰部+手臂)
python legged_gym/scripts/train.py --task=g1_29dof --headless
```

#### 5.3.3 验证与导出

```bash
# Play验证
python legged_gym/scripts/play.py --task=g1

# Sim2Sim (MuJoCo)
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

### 5.4 验证清单

- [ ] G1速度跟踪训练收敛 (前进/侧移/转向)
- [ ] 仿真中G1可稳定行走
- [ ] 动作模仿训练可复现参考动作
- [ ] Sim2Sim验证策略在MuJoCo中正常工作
- [ ] 导出policy.onnx / policy.pt用于部署

---

## 6. 阶段三：遥操作数据采集

> 目标：通过XR设备遥操作G1，采集操作数据集

### 6.1 安装xr_teleoperate

```bash
# 创建conda环境
conda create -n tv python=3.10 pinocchio=3.1.0 numpy=1.26.4 -c conda-forge
conda activate tv

# 克隆仓库
git clone https://github.com/unitreerobotics/xr_teleoperate.git
cd xr_teleoperate && git submodule update --init --depth 1

# 安装模块
cd teleop/teleimager && pip install -e . --no-deps
cd ../televuer && pip install -e .
# 配置SSL证书 (参考README)
cd ../robot_control/dex-retargeting/ && pip install -e .
cd ../../../ && pip install -r requirements.txt

# 安装SDK
pip install unitree_sdk2_python
```

### 6.2 仿真模式遥操作

先在仿真中验证遥操作流程，再转到实物。

```bash
# 终端1：启动Isaac Lab仿真
conda activate unitree_sim_env
cd unitree_sim_isaaclab
python sim_main.py --device cpu --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
  --enable_dex3_dds --robot_type g129

# 终端2：启动遥操作 (仿真模式+录制)
conda activate tv
cd xr_teleoperate/teleop/
python teleop_hand_and_arm.py --ee=dex3 --sim --record
```

### 6.3 实物遥操作

```bash
# 连接机器人 (网线, IP: 192.168.123.222)
# 机器人进入调试模式 (L2+R2)

# 启动遥操作
python teleop_hand_and_arm.py --arm=G1_29 --ee=dex3 --record \
  --network-interface=enp5s0
```

### 6.4 数据格式与采集规范

采集的数据存储格式：
```
datasets/task_name/
    ├── episode_0001/
    │   ├── audios/          # 音频
    │   ├── colors/          # RGB图像
    │   ├── depths/          # 深度图
    │   └── data.json        # 状态+动作
    ├── episode_0002/
    └── ...
```

**采集建议**：
- 每个任务至少采集50-100个episode
- 确保动作多样性（不同起始位置、不同抓取角度）
- 失败的episode需要后续剔除
- 控制频率30Hz，与推理频率一致

### 6.5 验证清单

- [ ] XR设备可正常连接
- [ ] 仿真中遥操作流畅
- [ ] 数据录制功能正常
- [ ] 采集数据格式正确 (JSON + 图像)
- [ ] 实物遥操作安全可控

---

## 7. 阶段四：模仿学习与操作策略训练

> 目标：使用采集的数据训练操作策略，实现抓取等操作

### 7.1 数据转换 (JSON → LeRobot格式)

```bash
conda activate unitree_lerobot

# 排序和重命名
python unitree_lerobot/utils/sort_and_rename_folders.py \
  --data_dir $HOME/datasets/task_name

# 转换为LeRobot格式
python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
  --raw-dir $HOME/datasets \
  --repo-id your_name/repo_task_name \
  --robot_type Unitree_G1_Dex3 \
  --push_to_hub
```

**robot_type选项**：
- `Unitree_G1_Dex1` - G1 + 二指夹爪
- `Unitree_G1_Dex3` - G1 + Dex3灵巧手
- `Unitree_G1_Inspire` - G1 + 因时灵巧手
- `Unitree_G1_Brainco` - G1 + 强脑灵巧手
- `Unitree_G1_Dex1_Sim` - 仿真环境数据

### 7.2 数据编辑与清洗

```bash
pip install PyQt5
cd unitree_lerobot/data_editor
python data_editor_EN.py
# 可视化裁剪episode、删除失败episode
```

### 7.3 策略训练

#### 7.3.1 ACT策略

```bash
cd unitree_lerobot/lerobot
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=your_name/repo_task_name \
  --policy.push_to_hub=false \
  --policy.type=act
```

#### 7.3.2 Diffusion策略

```bash
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=your_name/repo_task_name \
  --policy.push_to_hub=false \
  --policy.type=diffusion
```

#### 7.3.3 Pi0策略

```bash
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=your_name/repo_task_name \
  --policy.push_to_hub=false \
  --policy.type=pi0
```

#### 7.3.4 Pi0.5策略

```bash
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=your_name/repo_task_name \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false
```

#### 7.3.5 Gr00t策略

```bash
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=your_name/repo_task_name \
  --policy.push_to_hub=false \
  --policy.type=groot
```

### 7.4 仿真验证

```bash
# 在Isaac Lab仿真中验证策略
python unitree_lerobot/eval_robot/eval_g1_sim.py \
  --policy.path=path/to/pretrained_model \
  --repo_id=your_name/repo_task_name \
  --arm="G1_29" --ee="dex3" \
  --frequency=30 --visualization=true
```

### 7.5 验证清单

- [ ] 数据转换正确，LeRobot格式数据可正常加载
- [ ] ACT策略训练收敛
- [ ] Diffusion策略训练收敛
- [ ] 仿真中策略可完成抓取任务
- [ ] 不同策略效果对比完成

---

## 8. 阶段五：VLA大模型与自主抓取

> 目标：使用VLA/世界模型实现语言指令驱动的自主操作

### 8.1 UnifoLM-VLA (视觉-语言-动作模型)

#### 8.1.1 安装

```bash
conda create -n unifolm-vla python==3.10.18
conda activate unifolm-vla

cd unifolm-vla
pip install --no-deps "lerobot @ git+https://github.com/huggingface/lerobot.git@0878c68"
pip install -e .
pip install "flash-attn==2.5.6" --no-build-isolation
```

#### 8.1.2 下载模型

| 模型 | 用途 | 来源 |
|------|------|------|
| UnifoLM-VLM-Base | 视觉-语言基础模型 | HuggingFace |
| UnifoLM-VLA-Base | VLA微调模型 | HuggingFace |
| UnifoLM-VLA-LIBERO | LIBERO仿真测试 | HuggingFace |

#### 8.1.3 数据准备

```bash
# LeRobot格式 → HDF5 → RLDS
cd prepare_data

# Step 1: LeRobot → HDF5
python convert_lerobot_to_hdf5.py \
  --data_path /path/to/dataset \
  --target_path /path/to/output

# Step 2: HDF5 → RLDS
cd hdf5_to_rlds/rlds_dataset
tfds build --data_dir /path/to/output
```

#### 8.1.4 训练

```bash
# 配置数据集注册 (configs.py, transforms.py, mixtures.py, datasets.py)
# 配置动作维度和归一化 (constants.py)
# 配置训练参数 (run_unifolm_vla_train.sh)

bash scripts/run_scripts/run_unifolm_vla_train.sh
```

#### 8.1.5 真机推理

```bash
# 服务器端
bash scripts/eval_scripts/run_real_eval_server.sh

# 客户端 (机器人端)
# SSH隧道: ssh user@server -CNg -L port:127.0.0.1:port
python unitree_deploy/robot_client.py
```

### 8.2 UnifoLM-WMA (世界模型-动作模型)

#### 8.2.1 安装

```bash
conda create -n unifolm-wma python==3.10.18
conda activate unifolm-wma
conda install pinocchio=3.2.0 -c conda-forge -y
conda install ffmpeg=7.1.1 -c conda-forge

cd unifolm-world-model-action
git submodule update --init --recursive
pip install -e .
cd external/dlimp && pip install -e .
```

#### 8.2.2 数据准备

```bash
cd prepare_data
python prepare_training_data.py \
  --source_dir /path/to/source \
  --target_dir /path/to/output \
  --dataset_name "dataset_name" \
  --robot_name "Unitree G1 Robot with Gripper"
```

#### 8.2.3 训练

```bash
# 配置 configs/train/config.yaml
# 配置 configs/train/meta.json
# 配置 scripts/train.sh

bash scripts/train.sh
```

#### 8.2.4 真机推理

```bash
# 服务器端
bash scripts/run_real_eval_server.sh

# 客户端
cd unitree_deploy
python scripts/robot_client.py \
  --robot_type "g1_dex1" \
  --action_horizon 16 --exe_steps 16 \
  --observation_horizon 2 \
  --language_instruction "pack black camera into box" \
  --control_freq 15
```

### 8.3 官方开源数据集

可直接使用宇树官方数据集进行训练验证：

| 数据集 | 任务 | 链接 |
|--------|------|------|
| G1_Dex3_ToastedBread_Dataset | 烤面包操作 | HuggingFace |
| G1_Stack_Block | 堆叠积木 | HuggingFace |
| G1_Bag_Insert | 物品装袋 | HuggingFace |
| G1_Erase_Board | 擦白板 | HuggingFace |
| G1_Clean_Table | 清理桌面 | HuggingFace |
| G1_Pack_PencilBox | 整理铅笔盒 | HuggingFace |
| G1_Pour_Medicine | 倒药 | HuggingFace |
| G1_Pack_PingPong | 装乒乓球 | HuggingFace |
| G1_Prepare_Fruit | 准备水果 | HuggingFace |
| G1_Organize_Tools | 整理工具 | HuggingFace |
| G1_Fold_Towel | 折毛巾 | HuggingFace |
| G1_Wipe_Table | 擦桌子 | HuggingFace |

### 8.4 验证清单

- [ ] VLA模型可加载并推理
- [ ] 仿真中VLA可完成语言指令操作
- [ ] 世界模型可预测未来状态
- [ ] 真机部署推理流程跑通
- [ ] 自主抓取成功率达标

---

## 9. 阶段六：Sim2Real实物部署

> 目标：将训练好的策略部署到G1实物机器人

### 9.1 部署前准备

1. **机器人准备**
   - 吊装状态下启动G1
   - 等待进入零力矩模式
   - 按L2+R2进入调试模式

2. **网络配置**
   - 网线连接PC与机器人
   - PC地址：`192.168.123.222`，子网：`255.255.255.0`
   - 查看网卡名称：`ifconfig`

3. **安全检查**
   - 确保吊装设备可靠
   - 确保急停按钮可用
   - 首次部署务必在吊装状态下进行

### 9.2 RL运动控制部署

#### 9.2.1 unitree_rl_mjlab (C++部署)

```bash
# 将policy.onnx放入 deploy/robots/g1/config/policy/velocity/vo/exported/
cd deploy/robots/g1
mkdir build && cd build
cmake .. && make

# 仿真验证
./g1_ctrl --network=lo

# 实物部署
./g1_ctrl --network=enp5s0
```

#### 9.2.2 unitree_rl_gym (Python部署)

```bash
python deploy/deploy_real/deploy_real.py enp5s0 g1.yaml
```

### 9.3 操作策略部署 (LeRobot)

```bash
python unitree_lerobot/eval_robot/eval_g1.py \
  --policy.path=path/to/pretrained_model \
  --repo_id=your_name/repo_task_name \
  --arm="G1_29" --ee="dex3" \
  --frequency=30 \
  --send_real_robot=true
```

### 9.4 VLA模型部署

参考8.1.5和8.2.4的服务器-客户端架构部署。

### 9.5 验证清单

- [ ] RL运动控制实物行走稳定
- [ ] 操作策略实物抓取成功
- [ ] VLA语言指令实物执行成功
- [ ] 安全性验证通过

---

## 10. 关键技术要点与注意事项

### 10.1 G1通信协议

- G1使用 **unitree_hg** IDL (非unitree_go)
- 话题：`rt/lowcmd` (控制), `rt/lowstate` (状态), `rt/secondary` (IMU)
- DDS domain_id：实物默认0，仿真建议1
- 仿真网卡：`lo`，实物网卡：如`enp5s0`

### 10.2 G1关节索引

参考 `unitree_mujoco/unitree_robots/g1/g1_joint_index_dds.md`

G1 29自由度关节分布：
- 腿部 (12): 左右各6 (hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll)
- 腰部 (3): waist_yaw, waist_roll, waist_support
- 左臂 (7): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_pitch, wrist_roll, wrist_yaw
- 右臂 (7): 同左臂

### 10.3 观测空间

- **G1 12DOF (RL)**: 47维 (ang_vel(3) + gravity(3) + cmd(3) + dof_pos(12) + dof_vel(12) + actions(12) + phase(2))
- **G1 29DOF (RL)**: 98维，包含腰部和手臂关节
- **操作策略**: 相机图像 + 关节状态 + 语言指令

### 10.4 控制频率

| 场景 | 频率 |
|------|------|
| RL训练仿真步长 | 0.002s (500Hz) |
| RL控制频率 | 50Hz (10倍decimation) |
| 遥操作 | 30Hz |
| LeRobot推理 | 30Hz |
| VLA推理 | 15Hz |

### 10.5 常见问题

| 问题 | 解决方案 |
|------|----------|
| Isaac Gym安装失败 | 仅支持Ubuntu 20.04/22.04, 需NVIDIA GPU |
| DDS通信不通 | 检查domain_id和网卡配置 |
| G1关节控制异常 | 确认使用unitree_hg消息类型 |
| 策略Sim2Real性能下降 | 检查观测归一化参数是否一致 |
| MuJoCo仿真器启动失败 | 检查mujoco软链接路径 |
| rsl_rl训练NaN | 检查学习率和奖励函数scale |

### 10.6 安全注意事项

1. **首次部署必须在吊装状态下进行**
2. 先Sim2Sim验证，再Sim2Real
3. 实物部署前确认策略在仿真中行为正常
4. 始终保持急停按钮可用
5. 逐步提高控制增益，避免突然的大力矩输出
6. VLA模型推理延迟需在可接受范围内

---

## 11. 进度追踪

### 阶段一：仿真环境搭建

| 任务 | 状态 | 备注 |
|------|------|------|
| MuJoCo仿真器安装编译 | ⬜ | |
| G1模型加载验证 | ⬜ | |
| Isaac Lab环境安装 | ⬜ | |
| G1操作场景验证 | ⬜ | |
| DDS通信验证 | ⬜ | |

### 阶段二：RL运动控制

| 任务 | 状态 | 备注 |
|------|------|------|
| unitree_rl_mjlab环境配置 | ⬜ | |
| G1速度跟踪训练 | ⬜ | |
| G1动作模仿训练 | ⬜ | |
| Sim2Sim验证 | ⬜ | |
| 策略导出 | ⬜ | |

### 阶段三：遥操作数据采集

| 任务 | 状态 | 备注 |
|------|------|------|
| xr_teleoperate安装配置 | ⬜ | |
| 仿真模式遥操作验证 | ⬜ | |
| XR设备连接 | ⬜ | |
| 数据采集 (至少50 episodes/任务) | ⬜ | |
| 数据质量检查 | ⬜ | |

### 阶段四：模仿学习

| 任务 | 状态 | 备注 |
|------|------|------|
| 数据转换 (JSON → LeRobot) | ⬜ | |
| ACT策略训练 | ⬜ | |
| Diffusion策略训练 | ⬜ | |
| 策略仿真验证 | ⬜ | |
| 策略效果对比 | ⬜ | |

### 阶段五：VLA大模型

| 任务 | 状态 | 备注 |
|------|------|------|
| UnifoLM-VLA环境配置 | ⬜ | |
| 官方数据集训练验证 | ⬜ | |
| 自定义数据训练 | ⬜ | |
| UnifoLM-WMA环境配置 | ⬜ | |
| 自主抓取验证 | ⬜ | |

### 阶段六：Sim2Real

| 任务 | 状态 | 备注 |
|------|------|------|
| RL运动控制实物部署 | ⬜ | |
| 操作策略实物部署 | ⬜ | |
| VLA模型实物部署 | ⬜ | |
| 自主抓取实物验证 | ⬜ | |
| 安全性验证 | ⬜ | |

---

## 附录：快速命令参考

```bash
# === MuJoCo仿真 ===
cd unitree_mujoco/simulate/build && ./unitree_mujoco -r g1 -s scene_29dof.xml

# === Isaac Lab仿真 ===
conda activate unitree_sim_env
python sim_main.py --device cpu --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
  --enable_dex3_dds --robot_type g129

# === RL训练 (mjlab) ===
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096

# === RL训练 (gym) ===
conda run -n unitree-rl-PY38 python legged_gym/scripts/train.py --task=g1 --headless

# === 遥操作 (仿真) ===
conda activate tv
python teleop_hand_and_arm.py --ee=dex3 --sim --record

# === LeRobot训练 ===
conda activate unitree_lerobot
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=your_name/repo_task_name \
  --policy.type=act --policy.push_to_hub=false

# === VLA推理 (服务器) ===
conda activate unifolm-vla
bash scripts/eval_scripts/run_real_eval_server.sh

# === 实物部署 ===
python deploy/deploy_real/deploy_real.py enp5s0 g1.yaml
```
