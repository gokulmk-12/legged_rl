import os
import torch
import numpy as np
from stable_baselines3 import PPO
from collections import defaultdict
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from leggedEnv import LeggedEnv

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose = 0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self)->bool:
        infos = self.locals["infos"]
        unwanted = "TimeLimit.truncated"
        reward_sums = defaultdict(list)

        for info in infos:
            for k, v in info.items():
                if k != unwanted:
                    reward_sums[k].append(v)
        
        for k, v_list in reward_sums.items():
            self.logger.record(f'rewards/{k}', np.mean(v_list))

        return True

def make_env(robot_name="go2", render_mode="non-human", rank=0):
    def _init():
        env = LeggedEnv(robot_name=robot_name, render_mode=render_mode)
        env.reset(rank)
        return env
    return _init

def train():
    exp_no = 3
    log_dir = f"logs/PPO_GO2_Exp{exp_no}"
    os.makedirs(log_dir, exist_ok=True)

    num_envs = 16
    
    if num_envs == 1:
        env = DummyVecEnv([make_env()])
    else:
        env = SubprocVecEnv(
            [make_env(rank=i) for i in range(num_envs)],
            start_method='spawn'
        )

    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    new_logger = configure(log_dir, ["stdout", "tensorboard"])

    weights_path = os.path.join(log_dir, "ppo_go2_final.zip")
    vec_normalize_path = os.path.join(log_dir, "vec_normalize.pkl")

    if os.path.exists(weights_path) and os.path.exists(vec_normalize_path):
        print(f"Loading model and normalization from {log_dir}")
        env = VecNormalize.load(vec_normalize_path, env)
        model = PPO.load(weights_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        print("Initializing new PPO model")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
            n_steps=2048 // num_envs,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.03,
            max_grad_norm=0.5,
            learning_rate=3e-4,
            clip_range=0.2,
            vf_coef=0.5,
            clip_range_vf=0.2
        )
        model.set_logger(new_logger)

    desired_checkpoint_interval = 1_000_000
    save_freq = max(desired_checkpoint_interval // num_envs, 1)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=log_dir,
        name_prefix="ppo_go2_checkpoint",
        save_vecnormalize=True
    )

    callback_list = CallbackList([
        TensorboardCallback(),
        checkpoint_callback,
    ])

    model.learn(
        total_timesteps=10_000_000,
        progress_bar=True,
        reset_num_timesteps=False,
        callback=callback_list
    )

    model.save(os.path.join(log_dir, "ppo_go2_final"))
    env.save(os.path.join(log_dir, "vec_normalize.pkl"))

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    train()