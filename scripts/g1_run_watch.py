"""Visualize a trained G1 running policy in the MuJoCo viewer.

Run with the current working directory set to MuJoCo Menagerie's
``unitree_g1`` directory so that ``scene_with_hands.xml`` is available.
"""

import os
import re
import sys
import time

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from g1_run_env import G1RunEnv


TARGET_SPEED = 2.0


def _checkpoint_step(filename):
    match = re.search(r"_(\d+)_steps\.zip$", filename)
    return int(match.group(1)) if match else -1


def find_model_and_normalization():
    """Prefer the final pair; otherwise select the latest complete checkpoint."""
    final_model = os.path.join(SCRIPT_DIR, "g1_run_final.zip")
    final_norm = os.path.join(SCRIPT_DIR, "g1_run_vecnorm.pkl")

    if os.path.exists(final_model) and os.path.exists(final_norm):
        return final_model, final_norm

    checkpoint_dir = os.path.join(SCRIPT_DIR, "g1_run_checkpoints")
    if os.path.isdir(checkpoint_dir):
        checkpoint_names = sorted(
            (
                name
                for name in os.listdir(checkpoint_dir)
                if name.startswith("g1_run_") and name.endswith("_steps.zip")
            ),
            key=_checkpoint_step,
            reverse=True,
        )

        for checkpoint_name in checkpoint_names:
            model_path = os.path.join(checkpoint_dir, checkpoint_name)
            checkpoint_steps = _checkpoint_step(checkpoint_name)
            norm_path = os.path.join(
                checkpoint_dir,
                f"g1_run_vecnormalize_{checkpoint_steps}_steps.pkl",
            )
            if os.path.exists(norm_path):
                return model_path, norm_path

    raise FileNotFoundError(
        "No complete running-policy model/VecNormalize pair was found. "
        "Run g1_run_train.py first."
    )


def main():
    model_path, norm_path = find_model_and_normalization()
    print(f"Loading policy: {model_path}")
    print(f"Loading normalization: {norm_path}")

    base_env = DummyVecEnv(
        [lambda: G1RunEnv(target_speed=TARGET_SPEED)]
    )
    inner_env = base_env.envs[0]

    env = VecNormalize.load(norm_path, base_env)
    env.training = False
    env.norm_obs = True
    env.norm_reward = False

    agent = PPO.load(model_path, env=env, device="auto")
    control_dt = inner_env.model.opt.timestep * inner_env.frame_skip

    print("Watching the deterministic running policy.")
    print("Close the MuJoCo viewer to stop.")

    with mujoco.viewer.launch_passive(inner_env.model, inner_env.data) as viewer:
        episode = 0

        while viewer.is_running():
            episode += 1
            observation = env.reset()
            start_x = float(inner_env.data.qpos[0])
            final_x = start_x
            final_height = float(inner_env.data.qpos[2])
            total_reward = 0.0
            last_info = {}

            mujoco.mj_forward(inner_env.model, inner_env.data)
            viewer.sync()
            time.sleep(0.3)

            for step in range(inner_env.max_steps):
                if not viewer.is_running():
                    break

                frame_start = time.perf_counter()
                action, _ = agent.predict(
                    observation,
                    deterministic=True,
                )
                observation, reward, done, infos = env.step(action)

                last_info = infos[0]
                total_reward += float(reward[0])
                final_x = float(last_info["x_position"])
                final_height = float(last_info["height"])

                viewer.sync()
                remaining = control_dt - (time.perf_counter() - frame_start)
                if remaining > 0.0:
                    time.sleep(remaining)

                if done[0]:
                    break

            elapsed_steps = step + 1
            distance = final_x - start_x
            duration = elapsed_steps * control_dt
            average_speed = distance / max(duration, control_dt)
            survived = elapsed_steps >= inner_env.max_steps
            status = "SURVIVED" if survived else f"FELL at step {elapsed_steps}"

            print(
                f"Episode {episode}: {status}, "
                f"distance={distance:.2f} m, "
                f"average_speed={average_speed:.2f} m/s, "
                f"final_height={final_height:.2f} m, "
                f"reward={total_reward:.1f}"
            )

            if last_info.get("reward_terms"):
                terms = last_info["reward_terms"]
                print(
                    "  final reward terms: "
                    f"speed={terms['speed']:.2f}, "
                    f"gait={terms['gait']:.2f}, "
                    f"flight={terms['flight']:.2f}, "
                    f"arm={terms['arm_swing']:.2f}, "
                    f"posture={terms['posture']:.2f}"
                )

            time.sleep(0.8)

    env.close()


if __name__ == "__main__":
    main()
