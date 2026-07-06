## 官方原始代码中，为什么没有检测高度的措施？    
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
|----------|---------------|
| 4 | 0 |
| 8 | 3 |
| 12 | 6 |
| 8 | 9 |

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
 tensorboard --logdir /home/css/work/robot/unitree/rl_org/unitree_rl_lab/logs/rsl_rl/                                                                                                   
 tensorboard --logdir /home/css/work/robot/unitree/unitree_rl_lab/logs/rsl_rl/

 
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
