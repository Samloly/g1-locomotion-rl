import os
import sys
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize,SubprocVecEnv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from g1_run_env import G1RunEnv
from g1_run_env_new import G1RunEnv_new


TARGET_SPEED = 4.0
TOTAL_TIMESTEPS = 50_000_000
SEED = 42

# 直接在这里设置并行环境数量
NUM_ENVS = 32

# 每轮 PPO 更新收集的总样本数
ROLLOUT_SIZE = 4096

def make_env(rank):
    """创建一个具有独立随机种子的 MuJoCo 环境。"""

    def _init():
        env = G1RunEnv_new(target_speed=TARGET_SPEED)
        env.reset(seed=SEED + rank)
        return env

    return _init


def make_vector_env():
    env_fns = [
        make_env(rank)
        for rank in range(NUM_ENVS)
    ]

    if NUM_ENVS == 1:
        return DummyVecEnv(env_fns)

    return SubprocVecEnv(
        env_fns,
        start_method="forkserver",
    )

def main():
    model_path = os.path.join(SCRIPT_DIR, "g1_run_final_4.zip")
    norm_path = os.path.join(SCRIPT_DIR, "g1_run_vecnorm_4.pkl")
    checkpoint_dir = os.path.join(SCRIPT_DIR, "g1_run_checkpoints_4")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_exists = os.path.exists(model_path)
    norm_exists = os.path.exists(norm_path)

    if model_exists != norm_exists:
        raise RuntimeError(
            "Resume files are incomplete. Expected both "
            f"{model_path!r} and {norm_path!r}, or neither."
        )

    resuming = model_exists and norm_exists
    # 创建多个并行 MuJoCo 环境
    env = make_vector_env()

    if resuming:
        print(f"Resuming running policy from: {model_path}")
        env = VecNormalize.load(norm_path, env)
        env.training = True
        env.norm_obs = True
        env.norm_reward = True
        model = PPO.load(model_path, env=env, device="auto")
    else:
        print("Starting a new G1 running policy")
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
        )

        # n_steps 表示每个环境采集多少步。
        # 8 个环境 × 512 步 = 每轮总计 4096 个样本。
        steps_per_env = max(
            ROLLOUT_SIZE // NUM_ENVS,
            128,
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-4,
            n_steps=4096,
            batch_size=256,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.02,
            policy_kwargs={
                "net_arch": [256, 256],
                "log_std_init": -1.5,
            },
            seed=SEED,
            device="auto",
            verbose=1,
        )

    # Callback 每调用一次，实际上所有环境都各运行了一步。
    # 因此需要除以 NUM_ENVS。
    checkpoint_callback = CheckpointCallback(
        save_freq=max(
            1_000_000 // NUM_ENVS,
            1,
        ),
        save_path=checkpoint_dir,
        name_prefix="g1_run",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    print(f"Target speed: {TARGET_SPEED:.2f} m/s")
    print(f"Parallel environments: {NUM_ENVS}")
    # print(f"PPO device: {DEVICE}")
    print(f"Training steps this run: {TOTAL_TIMESTEPS:,}")
    print("Press Ctrl+C to stop safely and save.")

    start_time = time.time()
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=not resuming,
        )
    except KeyboardInterrupt:
        print("Training interrupted; saving the current policy.")
    finally:
        elapsed_minutes = (time.time() - start_time) / 60.0
        model.save(model_path)
        env.save(norm_path)
        env.close()

        print(f"Training time: {elapsed_minutes:.1f} minutes")
        print(f"Policy saved to: {model_path}")
        print(f"Normalization statistics saved to: {norm_path}")


if __name__ == "__main__":
    main()
