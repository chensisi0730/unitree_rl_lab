# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""评估脚本：运行固定步数，输出量化指标，然后退出。"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent checkpoint with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--eval-steps", type=int, default=64000, help="Total env steps for evaluation.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import time
import numpy as np
import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Evaluate checkpoint: fixed steps, print metrics, exit."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---- eval mode: fixed steps, collect metrics, exit ----
    target_steps = args_cli.eval_steps
    total_steps = 0
    ep_lens, ep_rews, ep_fails = [], [], []
    cur_len = torch.zeros(env.num_envs, device=env.device)
    cur_rew = torch.zeros(env.num_envs, device=env.device)

    print(f"[Eval] Running {target_steps} steps on {env.num_envs} envs...", flush=True)
    t0 = time.time()
    obs = env.get_observations()

    while total_steps < target_steps and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, info = env.step(actions)
            policy.reset(dones)

        cur_len += 1
        cur_rew += rewards

        done_mask = dones.byte().bool().squeeze()
        for i in torch.where(done_mask)[0].tolist():
            ep_lens.append(int(cur_len[i].item()))
            ep_rews.append(cur_rew[i].item())
            to = info.get("time_outs")
            is_fall = True
            if to is not None and isinstance(to, torch.Tensor):
                is_fall = not bool(to[i].item())
            ep_fails.append(1 if is_fall else 0)
            cur_len[i] = 0
            cur_rew[i] = 0

        total_steps += env.num_envs

    # 输出结果
    t_elapsed = time.time() - t0
    log_dir = os.path.dirname(resume_path)
    metrics = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "total_steps": total_steps,
        "time_sec": round(t_elapsed, 1),
        "fps": round(total_steps / t_elapsed) if t_elapsed > 0 else 0,
        "num_envs": env.num_envs,
        "num_episodes": len(ep_rews),
    }
    if ep_rews:
        r, l = np.array(ep_rews), np.array(ep_lens)
        fail_rate = np.mean(ep_fails) * 100
        metrics["success_rate_pct"] = round(100 - fail_rate, 1)
        metrics["fall_rate_pct"] = round(fail_rate, 1)
        metrics["mean_reward"] = round(float(r.mean()), 2)
        metrics["std_reward"] = round(float(r.std()), 2)
        metrics["mean_ep_length"] = round(float(l.mean()), 1)
        metrics["max_ep_length"] = env.unwrapped.max_episode_length

    # 保存到文件
    eval_path = os.path.join(log_dir, "eval_result.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Eval] Results saved to {eval_path}", flush=True)

    # 终端摘要
    if ep_rews:
        print(f"  成功率: {metrics['success_rate_pct']:.1f}%", flush=True)
        print(f"  平均奖励: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}", flush=True)
        print(f"  平均步数: {metrics['mean_ep_length']:.0f} / {metrics['max_ep_length']}", flush=True)
        print(f"  摔倒率: {metrics['fall_rate_pct']:.1f}%", flush=True)
        print(f"  总轮数: {metrics['num_episodes']}, FPS: {metrics['fps']}", flush=True)
    else:
        print(f"  0 episodes completed in {total_steps} steps", flush=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
