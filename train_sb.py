import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from leggedEnv import LeggedEnv

exp_no = 2
log_dir = f"logs/PPO_GO2_Exp{exp_no}"
os.makedirs(log_dir, exist_ok=True)

env = LeggedEnv(robot_name="go2", render_mode="non-human")
obs, _ = env.reset()

new_logger = configure(log_dir, ["stdout", "tensorboard"])

weights_path = os.path.join(log_dir, "ppo_go2_final.zip")
if os.path.exists(weights_path):
    print(f"Loading model from {weights_path}")
    model = PPO.load(weights_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
else:
    print("No saved model found, initializing new PPO model")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=log_dir,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
model.set_logger(new_logger)

model.learn(
    total_timesteps=20_000_000,
    progress_bar=True
)

model.save(os.path.join(log_dir, "ppo_go2_final"))