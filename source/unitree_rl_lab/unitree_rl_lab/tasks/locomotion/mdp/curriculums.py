from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def terrain_levels_vel_stairs_only_up(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    move_up_min_dist: float = 2.0,
) -> torch.Tensor:
    """
    改良版地形课程：降低 move_down 灵敏度，适合楼梯训练。

    原版 terrain_levels_vel 的 move_down 阈值为 ``v × 20s × 0.5 = 10m``（满速时），
    楼梯上机器人频繁重置，几乎不可能累积 10m 净位移，导致地形等级螺旋下降。

    本版将系数改为 move_down_coeff（默认 0.15），满速时阈值从 10m 降到 3m，
    给楼梯训练足够的容错空间。

    地形课程：难度只升不降，适合楼梯训练。

    原版问题：move_down 会导致 terrain_level 震荡（上去了跌回来，卡在 0）。
    本版只保留 move_up：走够距离就升级，摔倒了也不降级。
    随着训练推进，terrain_level 只会单调上升，逐步推进到 15cm 台阶。
    """
    from isaaclab.assets import Articulation
    from isaaclab.terrains import TerrainImporter

    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)

    '''
这段代码的逻辑是动态判断机器人是否"足够成功"地完成了任务，从而决定提升地形难度。

逐行分析
1. 计算指令速度

cmd_speed = torch.norm(command[env_ids, :2], dim=1)
从速度指令中取前两维（x 和 y 方向的速度）
计算欧几里得范数，得到标量速度大小
结果：每个环境的指令速度 cmd_speed
2. 计算升级阈值

move_up_threshold = torch.maximum(
    torch.full_like(cmd_speed, move_up_min_dist),  # 固定值 2.0m
    cmd_speed * env.max_episode_length_s * 0.3,    # 动态值 v × 20 × 0.3
)
取两者中的较大值作为阈值：

条件	阈值	说明
低速/静止	move_up_min_dist (2.0m)	即使指令速度很小，也要走够 2 米才算成功
高速运动	v × max_episode_length_s × 0.3	速度越高，要求走的距离越远
举例（假设 max_episode_length_s = 20s）：

指令速度	动态阈值	最终阈值（取较大）
0.5 m/s	0.5 × 20 × 0.3 = 3.0m	3.0m
0.1 m/s	0.1 × 20 × 0.3 = 0.6m	2.0m (固定值更大)
1.0 m/s	1.0 × 20 × 0.3 = 6.0m	6.0m
3. 判断是否升级

move_up = distance > move_up_threshold
distance：机器人实际从起点出发的净位移（xy 平面）
如果实际位移 > 阈值，说明机器人"走得够远"，触发地形难度升级
设计意图
相比原版（系数 0.5，满速时阈值 10m），这里改为 0.3 是为了：

降低阈值到 6m（满速时），给楼梯训练更多容错空间
原版 10m 阈值在楼梯上几乎不可能达到，导致地形等级一直降级，卡在最低难度
0.3 系数让机器人在楼梯上也有机会累积进度，逐步提升难度
    '''
    # move_up: 至少走 min_dist 米，或走完 v_cmd × 20s × 0.3
    cmd_speed = torch.norm(command[env_ids, :2], dim=1)
    move_up_threshold = torch.maximum(
        torch.full_like(cmd_speed, move_up_min_dist),
        cmd_speed * env.max_episode_length_s * 0.3,
    )
    move_up = distance > move_up_threshold

    # 永不降级！
    move_down = torch.zeros_like(move_up)

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
