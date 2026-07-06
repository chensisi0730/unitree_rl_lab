# GO2楼梯行走PPO配置
# 针对楼梯行走任务优化的训练超参数

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class StairsPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """楼梯行走任务的PPO配置."""

    # 训练基本设置
    num_steps_per_env = 24  # 每环境每步数
    max_iterations = 100000  # 最大迭代次数 (楼梯任务需要更多训练)
    save_interval = 500  # 保存间隔
    experiment_name = "Unitree-Go2-Stairs"
    empirical_normalization = False

    # 策略网络配置
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.2,  # 更高的初始噪声以探索更多行为
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # PPO算法配置
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.015,  # 更高的熵系数以保持探索
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",  # 自适应学习率
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )