from __future__ import annotations

import os
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 全局变量，记录是否已加载 terrain_levels
_terrain_loaded = False


def _get_log_dir() -> str | None:
    """从环境变量获取日志目录。"""
    return os.environ.get("RSL_RL_LOG_DIR")


def save_terrain_levels(terrain, log_dir: str | None):
    """保存 terrain_levels 到日志目录。"""
    if not log_dir:
        return
    os.makedirs(log_dir, exist_ok=True)
    save_path = os.path.join(log_dir, "terrain_levels.pt")
    try:
        torch.save({
            "terrain_levels": terrain.terrain_levels.cpu(),
            "terrain_types": getattr(terrain, "terrain_types", None),
        }, save_path)
    except Exception:
        pass


def load_terrain_levels(terrain, env) -> bool:
    """从日志目录加载 terrain_levels。"""
    log_dir = _get_log_dir()
    if not log_dir:
        return False

    save_path = os.path.join(log_dir, "terrain_levels.pt")
    if not os.path.exists(save_path):
        return False

    try:
        checkpoint = torch.load(save_path, weights_only=False, map_location=env.device)
        saved_levels = checkpoint["terrain_levels"]
        if saved_levels.shape == terrain.terrain_levels.shape:
            terrain.terrain_levels[:] = saved_levels.to(env.device)
            if terrain.terrain_origins is not None:
                terrain.env_origins[:] = terrain.terrain_origins[
                    terrain.terrain_levels, terrain.terrain_types
                ]
            print(f"[INFO] Loaded terrain_levels from {save_path}")
            print(f"       Mean level: {terrain.terrain_levels.float().mean():.2f}, "
                  f"Max level: {terrain.terrain_levels.max().item()}")
            return True
    except Exception as e:
        print(f"[WARN] Failed to load terrain_levels: {e}")
    return False


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
    efficiency_coeff: float = 0.8,
) -> torch.Tensor:
    """地形课程：难度只升不降，适合楼梯训练。

    原版问题：
      - move_down 导致 terrain_level 震荡（升上去又跌回来，卡在 0）
      - move_up 固定 4m 阈值，但初始速度 0.1m/s → 20s 只走 2m → 永远升不了级

    本版：
      - move_up = v_cmd × episode_duration × efficiency_coeff，自适应阈值
      - move_down = 0（永不降级）
      - terrain_level 单调上升，逐步推进到 15cm 台阶
      - 支持持久化：每次课程更新后自动保存，恢复训练时自动加载
    """
    global _terrain_loaded
    from isaaclab.assets import Articulation
    from isaaclab.terrains import TerrainImporter

    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    # 首次调用时尝试加载上次的 terrain_levels
    if not _terrain_loaded:
        load_terrain_levels(terrain, env)
        _terrain_loaded = True

    # 机器人当前 episode 的净位移（从出生点到当前位置）
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)

    # 指令速度大小
    cmd_speed = torch.norm(command[env_ids, :2], dim=1)

    # move_up 阈值：自适应阈值 = 指令速度 × episode时长 × 效率系数
    move_up_threshold = cmd_speed * env.max_episode_length_s * efficiency_coeff
    move_up = distance > move_up_threshold

    # 永不降级
    move_down = torch.zeros_like(move_up)

    terrain.update_env_origins(env_ids, move_up, move_down)

    # 每隔一定步数保存 terrain_levels
    if getattr(env, "common_step_counter", 0) % 1000 == 0:
        log_dir = _get_log_dir()
        save_terrain_levels(terrain, log_dir)

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