# 原始代码有课程学习吗
有。原始代码就有课程学习，只是由 Isaac Lab 内置包提供。

 mdp 的 import 链: 
   unitree_rl_lab/mdp/__init__.py 
     ├── from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  ← terrain_levels_vel 从这里来 
     └── from .curriculums import *                                         ← 本地只有 lin_vel/ang_vel_cmd_levels 

原始配置文件:

 python
 terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)   # 直接用 Isaac Lab 内置的 

这个内置  terrain_levels_vel  就是固定的  move_up = distance > 4.0m ，加上可降级。

现在我们的改动只是在本地定义同名  terrain_levels_vel  覆盖它，只改了  move_up  阈值一个地方。其余课程学习逻辑全部保留： __post_init__  中的  terrain_generator.curriculum = True 、 
CurriculumCfg  的启用、 lin_vel_cmd_levels  速度课程——都没动。


# 1. terrain_levels 课程逻辑                                                                                                                                                             
                                                                                                                                                                                        
每个 episode 结束后，检查机器人从出生点走了多远：                                                                                                                                       
                                                                                                                                                                                        
 原始 terrain_levels_vel（有升降级）                                                                                                                                                    
                                                                                                                                                                                        
 python                                                                                                                                                                                 
 distance = 机器人当前位置 - 出生点位置                                                                                                                                                 
                                                                                                                                                                                        
 # 升级条件：走得足够远（超过地形块一半宽度）                                                                                                                                           
 move_up   = distance > 4.0m               →  terrain_level +1 
  
 # 降级条件：走得太短（没达到指令速度预期的一半） 
 move_down = distance < cmd_speed × 20s × 0.5   →  terrain_level -1 
 move_down &= ~move_up                     # 如果已升级就不降级 

地形难度映射（10 级 × 20 列）：

 等级 0: 5cm  台阶  ← 初始 
 等级 1: 7cm 
 等级 2: 9cm 
 等级 3: 12cm 
 等级 4: 15cm 台阶  ← 目标 
 等级 5: 17cm 
 ... 
 等级 9: 25cm 台阶 

 我们改用的 terrain_levels_vel_stairs_only_up（只升不降） 

 python
 move_up_threshold = cmd_speed × episode_duration × efficiency_coeff 
 move_up   = distance > move_up_threshold   →  terrain_level +1 
 move_down = 0                              →  永不降级 

 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

 2. 为什么 terrain_levels 之前一直没提升？ 

 根因：阈值太高，机器人根本走不到 

原始版本使用 `max(move_up_min_dist, cmd_speed × episode_duration × 0.3)` 双阈值机制，当 `move_up_min_dist=3.0m` 超过机器人 20s 最大位移（0.1×20=2.0m）时，永远无法升级。

 修复 

移除 `move_up_min_dist` 最小距离阈值，改用纯自适应阈值：

 python
 efficiency_coeff = 0.3           # 期望移动效率系数（0~1）
 cmd_speed = 0.1 m/s              # 初始指令速度 
 episode_duration = 20s 
  
 move_up_threshold = cmd_speed × episode_duration × efficiency_coeff 
                   = 0.1 × 20 × 0.3 
                   = 0.6m 
  
 # 机器人 20 秒最多能走 2.0m > 0.6m → 可以升级了！ 

 效率系数的含义 

| efficiency_coeff | 含义 |
|------------------|------|
| 0.2              | 升级容易（只需达到理论最大距离的 20%） |
| 0.3              | 适中（默认值，达到理论最大距离的 30%） |
| 0.5              | 升级困难（需要达到理论最大距离的 50%） |

 总结 

 问题                   原因                                        修复
 ────────────────────── ─────────────────────────────────────────── ────────────────────
 terrain_level 下降     原版有 move_down，走不够远就降级              改 only_up，永不降级
 terrain_level 不动     move_up_min_dist 超过机器人最大位移           移除最小距离阈值，改用纯自适应
 terrain_level 正常增长 ✅ 修复后，走够阈值距离就升级                  逐步推进到 15cm 台阶


# height_scanner 详解                                                                                                                                                                    
                                                                                                                                                                                        
 height_scanner  是一个虚拟高度射线扫描传感器（ RayCasterCfg ），在 Isaac Lab 仿真环境中模拟 LiDAR/深度相机的高度感知能力。它在3个层面上工作：传感器硬件、观测数据、训练角色。          
                                                                                                                                                                                        
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────      
                                                                                                                                                                                        
 1. 传感器定义 — 发射射线扫描地形                                                                                                                                                       
                                                                                                                                                                                        
以 Go2 为例（当前源代码）：                                                                                                                                                             
                                                                                                                                                                                        
 python                                                                                                                                                                                 
 # source/.../robots/go2/velocity_env_cfg.py 第 96-102 行 
 height_scanner = RayCasterCfg( 
     prim_path="{ENV_REGEX_NS}/Robot/base",          # 安装在机器人 base 上 
     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)), # 从头顶上方20m向下射 
     ray_alignment="yaw",                             # 射线方向跟随机器人偏航角 
     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # 网格模式 
     debug_vis=False, 
     mesh_prim_paths=["/World/ground"],               # 只扫描地面网格 
 ) 

物理含义：从机器人躯干上方 20m 高处，向下发射一排排射线，射线碰到地面网格后返回高度值。

扫描网格： GridPatternCfg(resolution=0.1, size=[1.6, 1.0]) 

 · X 方向（前进方向）：1.6m / 0.1m = 17 条射线
 · Y 方向（侧向）：1.0m / 0.1m = 11 条射线
 · 总计：17 × 11 = 187 条射线

输出的就是机器人前方地面 1.6m × 1.0m 区域的 187 维高度图：

                    前进方向 (1.6m) 
     ←···································→ 
     .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .   ↑ 
     .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  1.0m 
     .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .   ↓ 
 机器人 
  (base) 

 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

 2. 观察数据 — 作为 Critic 特权输入 

在观察配置中（示例为当前源代码）：

 python
 # source/.../robots/go2/velocity_env_cfg.py 第 256-258 行 
 height_scanner = ObsTerm(func=mdp.height_scan, 
     params={"sensor_cfg": SceneEntityCfg("height_scanner")}, 
     clip=(-1.0, 5.0),        # 裁剪范围：-1m 到 +5m 
 ) 

关键：它只放在 Critic（特权观测）中，Policy 看不到它。

 观测组             看到什么                                          模拟真实传感器
 ────────────────── ───────────────────────────────────────────────── ────────────────────────────────────
 Policy（Actor）    IMU、关节编码器、上一动作、速度指令               ✅ 纯本体感知（real robot 直接可测）
 Critic（价值网络） Policy 的所有输入 + height_scanner + base_lin_vel ❌ 特权信息（sim only）

每个环境的形状：

 输入源                      维度   说明
 ─────────────────────────── ────── ──────────────────────────────────────────────────────────────────
 Policy 观测                 ~35-50 角速度×3 + 重力×3 + 指令×3 + 关节位置×12 + 速度×12 + 上一步动作×12
 Critic 额外: height_scanner 187    地面高度网格
 Critic 额外: base_lin_vel   3      机身的绝对线速度（真实世界无法直接测得）

 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

 3. 刷新频率 

 python
 # source/.../robots/**/velocity_env_cfg.py 第 ~387 行 
 self.scene.height_scanner.update_period = self.decimation * self.sim.dt 

 ·  sim.dt = 0.005s （物理步长 5ms）
 ·  decimation = 4 （每 4 个物理步执行一次策略推理）
 · 因此  height_scanner  每 0.02s 刷新一次（50Hz），与策略推理频率一致

 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

 4. 在训练中的作用 — 不对称 Actor-Critic（Asymmetric AC） 

                     ┌──────────────────┐ 
                     │     Policy       │  ← 只有本体感知（关节位置/速度/IMU/上一步动作） 
                     │   (Actor)        │      deploy 到真实机器人 
                     └───────┬──────────┘ 
                             │ 动作 
                             ▼ 
                     ┌──────────────────┐ 
                     │      Critic      │  ← 特权信息帮助价值估计 
                     │  (Value Network) │     height_scanner(187维) + 线速度(3维) 
                     └──────────────────┘ 

为什么这样设计？

 · Critic 通过 187 维高度图精确感知脚下地形起伏 + 3 维绝对线速度，价值估计更准确
 · Policy 虽然看不到这些信息，但通过 Critic 训练出的梯度，能隐式学习如何在不平地面上稳定行走
 · 部署到真实 G1/Go2 时，Policy 只依赖本体感知（IMU+关节编码器），不需要真实 LiDAR/深度相机参与推理

 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

 5. 类比：人骑自行车 

 角色               仿真中              类比
 ────────────────── ─────────────────── ──────────────────────────────────────────────
 Policy（Actor）    只有关节位置/IMU    闭着眼睛骑车，只能凭身体感觉
 Critic（价值网络） 有高度图+线速度     睁着眼睛的教练在旁边指挥："前面有坑，往左偏点"
 部署后             Policy 自己独立运行 闭眼骑车但已经训练出了肌肉记忆

 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

 6. 需要教师-学生蒸馏的原因 

当前架构的问题是：训练时 Critic 用了特权信息，但 Policy 没有直接学到它。这就是为什么需要你之前描述的"Teacher-Student 蒸馏管线"：

 训练时 (当前): 
     Critic(有高度图) → 帮助训练 → Policy(无高度图) 
                                     ↑ 
                             部署时只靠本体感知 
  
 蒸馏后 (目标): 
     Teacher(有高度图+线速度) → 行为克隆 → Student(GRU + 本体感知) 
                                             ↑ 
                                      GRU记忆隐式取代高度图 

Critic 的价值信号只能间接让 Policy 感知地形，而蒸馏管线可以通过行为克隆让 Student 网络直接学习 Teacher 的行为策略。配合 GRU 循环层，Student
就能从历史动作序列中「回忆起」脚下地形，隐式替代实时高度扫描。

# 官方原始代码中，为什么没有检测高度的措施？

好问题。官方代码传感器的定义在场景中，但故意没有加到观测里原因有三个：

1. 盲爬（Blind Locomotion）是默认路线

四足机器人的一大特点是仅凭本体感觉（proprioception）就能爬很多地形。策略通过以下信号间接「感知」地形：

触地瞬间关节角度的异常变化  →  脚碰到台阶边缘了
机身倾斜角度变化            →  正在爬上斜坡
脚底接触力时序              →  前脚踩到了更高的平面

这在学术界是非常成熟的方向。Rudin 等人 2022 年的经典论文 "Learning to Walk in Minutes..." 用的就是纯本体感觉策略，机器人能在复杂地形行走。

2. 课程学习的核心逻辑不需要「看到」地形

课程函数  terrain_levels_vel  只看两个指标：

python
move_up   = distance > 4.0m    # 走得远就升级
move_down = distance < ...     # 走不远就降级

它不看机器人走的是什么地形——平地还是楼梯都一样。高度扫描仪只影响「这个机器人能不能走好当前地形」，不影响「该不该升级到更难地形」。

3. 官方代码的设计意图是「按需启用」

传感器定义在 scene 中但不在 observations 里，这个模式在 Isaac Lab 中很常见——这是一种 渐进式复杂度设计：

第一步：用盲爬跑通训练（简单）
↓
第二步：发现盲爬无法通过复杂楼梯时，才打开 height_scan
↓
第三步：需要更高级感知时，再加摄像头

这种设计让你先看 baseline 效果，再决定是否需要增加感知输入。如果一开始就把 height_scan 加进去，你反而不知道「没有它是不是也够用」。

## 将  height_scanner  和摄像头加入策略观测，让机器人感知前方地形起伏,  摄像头和传感器都要保持和实际的GO2的分辨率一致，

关于摄像头的问题：

实际 Go2 Pro/EDU 前置深度相机是 Intel RealSense D435，参数如下：

参数       实际 D435                       模拟方案
────────── ─────────────────────────────── ────────────────────
分辨率     848×480（深度）                 20×15 = 300 个射线点
视野 HFOV  87°（≈ 2×arctan(0.75/2) ≈ 80°） ✅ 匹配
覆盖范围   面前 ~3.8m 宽 × 2.2m 高         2.0m 前 × 1.5m 宽
每环境开销 渲染一帧图像 ≈ 数 MB            仅 300 个浮点数

没用真实摄像头的原因：

· 5000 个环境的全分辨率深度图需要 5000 × 848 × 480 × 4 bytes ≈ 7.8 GB/帧
· 策略网络不需要像素级输入，高度扫描点云才是标准做法（几乎所有四足 locomotion 论文都这么做）
· 高度扫描（ray casting）计算量极小，只返回地面高度值

当前  height_scan  提供 300 维的地形感知输入，覆盖面前 2m × 宽 1.5m 的区域，10cm 分辨率。这足够让策略「看到」前方楼梯并做出反应。

## terrain_level 的范围

根据项目配置，terrain_level 的范围由地形生成器的 `num_rows` 决定：

### GO2 配置（velocity_env_cfg.py 第 28 行）

```python
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    num_rows=10,      # 10 行地形 = 0 ~ 9 共 10 个等级
    num_cols=20,
    ...
)
```

### terrain_level 范围：**0 到 num_rows - 1**

对于 GO2：

- **范围：0 ~ 9**（共 10 个等级）
- `max_init_terrain_level=1`：初始随机分配到 0 或 1 级地形
- 训练时 `max_init_terrain_level = num_rows - 1 = 9`（使用全部难度范围）
- 推理/评估时 `max_init_terrain_level = 8`：从 0~8 级开始

### G1 / H1 配置

```python
num_rows=9,    # 9 行地形 = 0 ~ 8 共 9 个等级
num_cols=21,
```

- **范围：0 ~ 8**（共 9 个等级）

### 地形等级与台阶高度的对应关系

子地形配置中楼梯的 `step_height_range=(0.03, 0.15)`：

- 等级 0：台阶高度约 0.03m（3cm）
- 等级 9（GO2 最高）：台阶高度约 0.15m（15cm）
- 等级线性插值：`height = 0.03 + (level / max_level) * (0.15 - 0.03)`

### 课程控制逻辑（curriculums.py）

```python
terrain.update_env_origins(env_ids, move_up, move_down)
```

- `move_up=True`：terrain_level + 1（提升难度）
- `move_down=True`：terrain_level - 1（降低难度，已禁用）
- 当前版本 `terrain_levels_vel_stairs_only_up` 只升不降，等级单调递增到最大值

## 为什么 terrain_level 输出的是浮点数

原因在第 123 行：

```python
return torch.mean(terrain.terrain_levels.float())
```

### 详细解释

1. **`terrain.terrain_levels`** 本身是**整数张量**，每个环境有一个整数等级（如 0, 1, 2, ...）
2. **`.float()`** 将其转换为浮点类型
3. **`torch.mean(...)`** 计算**所有环境的平均值**

### 举例说明

假设有 32 个并行环境（env_batch_size=32），当前各环境的 terrain_level 分布如下：


| 环境数量 | terrain_level |
| -------- | ------------- |
| 4        | 0             |
| 8        | 3             |
| 12       | 6             |
| 8        | 9             |

```
terrain_levels = [0,0,0,0, 3,3,3,3,3,3,3,3, 6,6,...,6, 9,9,...,9]
torch.mean(...) = (4×0 + 8×3 + 12×6 + 8×9) / 32 = 205/32 = 6.41
```

所以输出 **6.42** 是合理的——它表示当前所有环境的**平均地形等级**。

### 如果要看整数等级

有两个选择：

1. **看单个环境的等级**：用 `terrain.terrain_levels[env_id]` 查看特定环境
2. **看等级分布**：用 `torch.bincount(terrain.terrain_levels)` 统计各等级的环境数量
3. **输出时取整**：`return torch.mean(terrain.terrain_levels.float())` 改为 `return torch.round(torch.mean(terrain.terrain_levels.float()))`

### 当前输出方式的意义

输出平均值是**课程学习的标准做法**，因为：

- 不同环境可以有不同的地形等级（异步进度）
- 平均值反映整体训练进度
- 小数部分表示环境间存在等级差异（如你的日志显示 1.51 → 6.42，说明训练早期环境等级分散，后期逐渐集中到较高等级）

# 训练开启后，查看TESORBOAD的方法

TensorBoard 查看方法

命令（在新终端中运行）

bash

# 指定日志目录启动 TensorBoard

tensorboard --logdir /home/css/work/robot/unitree/unitree_rl_lab/logs/rsl_rl/
tensorboard --logdir /home/css/work/robot/unitree/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity_rough

启动后终端会显示：

TensorBoard 2.xx.x at http://localhost:6006/ (Press CTRL+C to quit)

访问方式

场景       地址
────────── ──────────────────────
本机       http://localhost:6006
远程服务器 http://<服务器IP>:6006

如果远程服务器有防火墙，可用 SSH 端口转发：

bash

# 本地终端执行

ssh -L 6006:localhost:6006 <用户名>@<服务器IP>

# 然后在本地浏览器打开 http://localhost:6006

TensorBoard 中关注的指标

左侧筛选                            查看内容
─────────────────────────────────── ───────────────────
Curriculum/terrain_levels           🎯 课程难度推进曲线
Metrics/Reward                      总奖励收敛
Episode_Termination/time_out        成功率（越高越好）
Episode_Termination/bad_orientation 摔倒率（越低越好）
Episode_Reward/track_lin_vel_xy     线速度跟踪质量
Metrics/base_velocity/error_vel_xy  速度跟踪误差

▎ 注意: 训练期间 TensorBoard 数据是实时写入的，启动后刷新页面即可看到最新数据。

# terrain_levels 最终值

指标               最终值     对应 Step (iter)
────────────────── ────────── ────────────────
terrain_levels     0.89       10893
lin_vel_cmd_levels 1.0 (满速) 10893

含义

terrain_levels  是所有 4096 个环境的平均难度等级（范围 0~9）。值 0.89 表示：

· 大部分环境仍处于课程早期难度等级（0 级或 1 级）
· 对应阶梯高度约 5~7cm
· 距离 15cm (难度等级 ~4.5) 还需要更多训练

地形难度映射

难度等级 阶梯高度 训练进度
──────── ──────── ─────────────────────────────────
0        5cm      ⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛ 当前 (~0.89)
1        7cm      ⬜⬜⬛⬛⬛⬛⬛⬛⬛⬛
2        9cm      ⬜⬜⬜⬛⬛⬛⬛⬛⬛⬛
3        12cm     ⬜⬜⬜⬜⬛⬛⬛⬛⬛⬛
4        15cm 🎯  ⬜⬜⬜⬜⬜⬛⬛⬛⬛⬛
5        17cm     ⬜⬜⬜⬜⬜⬜⬛⬛⬛⬛
6        20cm     ⬜⬜⬜⬜⬜⬜⬜⬛⬛⬛
7        22cm     ⬜⬜⬜⬜⬜⬜⬜⬜⬛⬛
8        24cm     ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬛
9        25cm     ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜

# ./g1_ctrl 启动了什么？

这是一个 C++ 控制器进程，它做了三件事：─                              w   ─

1. 启动流程

./g1_ctrl --network eno1│├── ① 加载 config.yaml        ← FSM 状态机 + policy_dir├── ② 加载 policy.onnx        ← 你训练好的神经网络├── ③ 连接机器人 DDS 通道     ← 通过 eno1 网卡└── ④ 进入 Passive 模式       ← 等待手柄指令

2. 状态机切换

   Passive（被动模式/松软）
   │   L2 dx+ ↑
   ▼
   FixStand（站立模式）
   │   RB + X
   ▼
   Velocity（速度跟踪模式）← 你的策略在这里运行
   │   LT + B
   ▼
   Passive（回到被动模式）
3. 实际操作步骤

第一步：启动

bash
LD_LIBRARY_PATH=/home/css/work/robot/unitree/unitree_sdk2/thirdparty/lib/x86_64 \
./g1_ctrl --network eno1

终端显示：

--- Unitree Robotics ---
G1-29dof Controller
Waiting for connection to robot...
Connected to robot.          ← 机器人已连接

第二步：让机器人站立

操作                        效果
─────────────────────────── ───────────────────────
同时按住 L2 + ↑（十字键上） 机器人从松软状态 → 站立
听到电机啸叫，机器人站直    说明 FixStand 模式生效

第三步：切换到速度跟踪

操作                                 效果
──────────────────────────────────── ────────────────────
同时按住 RB + X                      切换到 Velocity 模式
你的策略开始运行，机器人保持站立待命

第四步：控制行走

指令 手柄操作 效果
──── ──────── ────────
前进 左摇杆 ↑ 往前走
后退 左摇杆 ↓ 往后退
左移 左摇杆 ← 向左平移
右移 左摇杆 → 向右平移
左转 右摇杆 ← 原地左转
右转 右摇杆 → 原地右转
停止 摇杆回中 原地站立

第五步：停止/急停

操作                  效果
───────────────────── ────────────────────────
同时按住 LT + B       回到被动模式，机器人松软
直接关闭终端 / Ctrl+C 进程退出，机器人松软

4. 新手注意事项

注意              说明
───────────────── ──────────────────────────────────────────────────────────
⚠️  悬空启动       首次建议把机器人挂在架子上测试，不要直接放地上
⚠️  轻推摇杆       摇杆推一点就行，机器人会按指令速度行走，不要猛推到底
⚠️  急停           随时可以按 LT + B 让机器人停下来
⚠️  不要关控制程序 关掉 g1_ctrl 前先切回 Passive 模式，否则机器人可能直接倒地



                                                                                                                                                                   
 ### 风险                                                                                                                                                                               

 · 启用 height_scanner 会增加观测维度（约 160 个额外标量），但这是 Isaac Lab 标准做法