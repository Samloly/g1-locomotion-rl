import sys
import os
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from g1_walk_env import G1WalkEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    print("=" * 50)
    print("G1 Walking Training — Final")
    print("=" * 50)

    model_path = os.path.join(script_dir, "g1_walk_final.zip")
    norm_path = os.path.join(script_dir, "g1_walk_vecnorm.pkl")

    resuming = os.path.exists(model_path) and os.path.exists(norm_path)

    print("\nCreating environment...")
    env = DummyVecEnv([lambda: G1WalkEnv()])

    if resuming:
        print(f"Resuming from {model_path}")
        env = VecNormalize.load(norm_path, env)
        env.training = True
        env.norm_reward = True
        model = PPO.load(model_path, env=env)
    else:
        print("Starting fresh with conservative hyperparameters")
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,       # slower than before (was 3e-4)
            n_steps=4096,
            batch_size=256,           # larger batches for stability
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,           # tighter clipping (was 0.2)
            ent_coef=0.005,
            max_grad_norm=0.5,
            device="auto",
            policy_kwargs=dict(
                net_arch=[256, 256],
                log_std_init=-1.5,    # start with smaller action noise (was -1.0)
            ),
        )

    save_dir = os.path.join(script_dir, "g1_walk_checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=save_dir,
        name_prefix="g1_walk"
    )

    total_timesteps = 3_000_000
    print(f"\nTraining for {total_timesteps:,} timesteps...")
    print("Key metrics to watch:")
    print("  std < 1.0 = stable policy")
    print("  clip_fraction < 0.3 = healthy updates")
    print("  ep_rew_mean climbing = learning")
    print("Ctrl+C to stop early.\n")

    start = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_cb,
            progress_bar=True,
            reset_num_timesteps=not resuming,
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Saving...")

    elapsed = time.time() - start
    print(f"\nTraining took {elapsed/60:.1f} minutes")

    model.save(model_path)
    env.save(norm_path)
    print(f"Model saved to {model_path}")
    print(f'\nTo analyze: python "{os.path.join(script_dir, "g1_walk_analyze.py")}"')
    print(f'To watch:   python "{os.path.join(script_dir, "g1_walk_watch.py")}"')

    env.close()

if __name__ == "__main__":
    main()