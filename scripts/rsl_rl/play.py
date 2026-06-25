# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from packaging import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--eval", action="store_true", default=False, help="Evaluate mode: run fixed steps, print metrics, exit.")
parser.add_argument("--eval-steps", type=int, default=30000, help="Total env steps for evaluation.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    is_eval = args_cli.eval
    if is_eval:
        import numpy as np

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

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if version.parse(installed_version) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    if is_eval:
        # ---- eval mode: fixed steps, print metrics, exit ----
        target_steps = args_cli.eval_steps
        total_steps = 0
        ep_lens, ep_rews, ep_fails = [], [], []
        cur_len = torch.zeros(env.num_envs, device=env.device)
        cur_rew = torch.zeros(env.num_envs, device=env.device)

        print(f"[Eval] Running {target_steps} steps on {env.num_envs} envs...", flush=True)
        t0 = time.time()
        while total_steps < target_steps and simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, info = env.step(actions)
                if version.parse(installed_version) >= version.parse("4.0.0"):
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

        # write metrics to file (stdout gets mixed with Isaac Sim logs)
        import json
        t_elapsed = time.time() - t0
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
        eval_path = os.path.join(os.path.dirname(resume_path), "eval_result.json")
        with open(eval_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Eval] Results saved to {eval_path}", flush=True)
        # also print a compact summary
        if ep_rews:
            print(f"  {metrics['success_rate_pct']:.1f}% success | {metrics['mean_reward']:.2f} mean reward | {metrics['mean_ep_length']:.0f}/{metrics['max_ep_length']} steps | {metrics['fps']} FPS", flush=True)
        else:
            print(f"  0 episodes completed in {total_steps} steps", flush=True)
    else:
        # ---- play mode: GUI simulation loop ----
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Error] {e}", flush=True)
        import traceback
        traceback.print_exc()
    # close sim app
    simulation_app.close()
