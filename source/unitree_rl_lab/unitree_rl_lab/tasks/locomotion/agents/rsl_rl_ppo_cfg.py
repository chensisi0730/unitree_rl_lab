# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

# 废弃字段列表（rsl-rl >= 5.0.0 不再支持，但 RslRlMLPModelCfg 仍保留它们）
_DEPRECATED_MLP_FIELDS = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"}


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# 修复 to_dict: 过滤掉 MLPModel 的废弃字段，防止传给 MLPModel.__init__
def _patched_to_dict(self):
    data = BasePPORunnerCfg._original_to_dict(self)
    for key in ["actor", "critic"]:
        if key in data and isinstance(data[key], dict):
            for field in _DEPRECATED_MLP_FIELDS:
                data[key].pop(field, None)
    return data


BasePPORunnerCfg._original_to_dict = BasePPORunnerCfg.to_dict
BasePPORunnerCfg.to_dict = _patched_to_dict
