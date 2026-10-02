import sys
import os
import time
import mujoco
import mujoco.viewer
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from g1_walk_env import G1WalkEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def main():
    model_path = os.path.join(script_dir, "g1_walk_final.zip")
    norm_path = os.path.join(script_dir, "g1_walk_vecnorm.pkl")

    if not os.path.exists(model_path):
        ckpt_dir = os.path.join(script_dir, "g1_walk_checkpoints")
        if os.path.exists(ckpt_dir):
            ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".zip")])
            if ckpts:
                model_path = os.path.join(ckpt_dir, ckpts[-1])
                print(f"Using checkpoint: {ckpts[-1]}")
        if not os.path.exists(model_path):
            print("No trained model found. Run g1_walk_train.py first.")
            return

    print(f"Loading model:{model_path}")
    agent = PPO.load(model_path)

    env = DummyVecEnv([lambda:G1WalkEnv()])
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
        print("loaded normaliztion stats.")

    inner_env = env.envs[0]
    print("\nWatching walking policy...")
    print("Close the viewer to exit.\n")

    with mujoco.viewer.launch_passive(inner_env.model, inner_env.data) as viewer:
        episode = 0
        while viewer.is_running():
            episode +=1
            obs = env.reset()
            start_x = inner_env.data.qpos[0]

            mujoco.mj_forward(inner_env.model, inner_env.data)
            viewer.sync()
            time.sleep(0.5)

            total_reward = 0
            last_x = start_x
            last_height = inner_env.data.qpos[2]
            for step in range(inner_env.max_steps):
                if not viewer.is_running():
                    break
                # Capture position before step
                last_x = inner_env.data.qpos[0]
                last_height = inner_env.data.qpos[2]

                action, _ = agent.predict(obs, deterministic=True)
                obs, reward,done,info = env.step(action)
                total_reward+=reward[0]

                viewer.sync()
                time.sleep(0.02)

                if done[0]:
                    break
            
            # Use last tracked position
            if not done[0]:
                last_x = inner_env.data.qpos[0]
                last_height = inner_env.data.qpos[2]
            
            distance = last_x -start_x
            height = last_height
            avg_speed = distane/max(step,1) / (inner_env.model.opt.timestep*4)
            survived = step+1>=inner_env.max_steps
            status = "SURVIVED" if survived else f"FELL at step {step+1}"

            print(f"Ep {episode}: {status}, distance={distance:.3f}m, "
                  f"avg_speed={avg_speed:.3f}m/s, height={height:.3f}m, "
                  f"reward={total_reward:.1f}")

            time.sleep(1.0)

    env.close()
    print("Done.")

if __name__ == "__main__":
    main()