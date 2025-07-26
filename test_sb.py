import torch

from stable_baselines3 import PPO, SAC
from leggedEnv import LeggedEnv

env = LeggedEnv(robot_name="go2", task_name="walking", render_mode="human")
obs, _ = env.reset()

exp_no = 3
algo = "PPO"
model_path = f"logs/{algo}_GO2_Exp{exp_no}/{algo.lower()}_go2_final"
model_path = f"logs/{algo}_GO2_Exp{exp_no}/{algo.lower()}_go2_checkpoint_4000000_steps"
if algo == "PPO":
    model = PPO.load(model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
elif algo == "SAC":
    model = SAC.load(model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")

for i in range(10):
    done, total_reward = False, 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if done or truncated:
            obs, _ = env.reset()
            break

    print(f"Episode {i+1} Reward: {total_reward:.2f}")