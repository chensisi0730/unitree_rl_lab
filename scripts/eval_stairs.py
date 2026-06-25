# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""评估策略性能：固定步数模拟后输出量化指标。"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rsl_rl"))
from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description="Evaluate a checkpoint.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--steps", type=int, default=30000, help="Total env steps to run.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, extra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + extra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    # load play config (high difficulty terrain)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        use_fabric=True, entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root = os.path.abspath(f"logs/rsl_rl/{agent_cfg.experiment_name}")
    resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[Checkpoint] {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---- evaluation ----
    total_steps = 0
    ep_lens = []
    ep_rews = []
    ep_fails = []  # 1 = fell, 0 = completed
    cur_len = torch.zeros(env.num_envs, device=env.device)
    cur_rew = torch.zeros(env.num_envs, device=env.device)

    obs = env.get_observations()
    t0 = time.time()

    print(f"[Eval] Running {args_cli.steps} steps on {env.num_envs} envs...", flush=True)
    while total_steps < args_cli.steps:
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, info = env.step(actions)

        cur_len += 1
        cur_rew += rewards

        for i in range(env.num_envs):
            if dones[i]:
                ep_lens.append(int(cur_len[i].item()))
                ep_rews.append(cur_rew[i].item())
                # time_out = completed full episode; otherwise fell
                time_outs = info.get("time_outs", None)
                is_fall = True
                if time_outs is not None:
                    if isinstance(time_outs, torch.Tensor) and len(time_outs) > i:
                        is_fall = not bool(time_outs[i].item())
                ep_fails.append(1 if is_fall else 0)
                cur_len[i] = 0
                cur_rew[i] = 0

        total_steps += env.num_envs

    t_elapsed = time.time() - t0

    # ---- output ----
    import numpy as np
    print(f"\n{'=' * 65}")
    print(f"  📊 评估结果: {args_cli.task}")
    print(f"     {env.num_envs} envs × {args_cli.steps // env.num_envs} steps/env = {total_steps} total steps")
    print(f"{'=' * 65}")
    if ep_rews:
        r, l = np.array(ep_rews), np.array(ep_lens)
        fail_rate = np.mean(ep_fails) * 100
        success_rate = 100 - fail_rate
        max_len = env.unwrapped.max_episode_length
        print(f"  ✅ 成功率 (full episode):     {success_rate:.1f}%")
        print(f"  ❌ 摔倒率:                    {fail_rate:.1f}%")
        print(f"  🏆 平均奖励:                  {r.mean():.2f} ± {r.std():.2f}")
        print(f"  ⏱  平均存活步数:              {l.mean():.0f} / {max_len}")
        print(f"  📈 最高奖励:                  {r.max():.2f}")
        print(f"  📉 最低奖励:                  {r.min():.2f}")
        print(f"  ⚡ 总步数:                    {total_steps}")
        print(f"  ⏰ 耗时:                      {t_elapsed:.1f}s")
        print(f"  🔄 FPS:                       {total_steps / t_elapsed:.0f}")
        print(f"  📝 完成 episodes:              {len(ep_rews)}")
    else:
        print("  ⚠️  未收集到完整 episode (dones never triggered)")
    print(f"{'=' * 65}\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
