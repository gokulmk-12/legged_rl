import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
import gymnasium as gym
from copy import deepcopy
from collections import deque
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, obs, act, rew, next_obs):
        self.buffer.append((obs, act, rew, next_obs))
    
    def sample(self, batch_size):
        obs, act, rew, next_obs = zip(*random.sample(self.buffer, batch_size))
        return torch.tensor(np.array(obs), dtype=torch.float32, device=self.device), \
        torch.tensor(np.array(act), dtype=torch.float32, device=self.device), \
        torch.tensor(np.array(rew), dtype=torch.float32, device=self.device), \
        torch.tensor(np.array(next_obs), dtype=torch.float32, device=self.device), \
        
    def __len__(self):
        return len(self.buffer)

class FullyConnectedQ(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(FullyConnectedQ, self).__init__()
        self.fc1 = nn.Linear(obs_dim+act_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
    
    def forward(self, x, a):
        x = x.float()
        a = a.float()
        y1 = F.relu(self.fc1(torch.cat((x, a), 1)))
        y2 = F.relu(self.fc2(y1))
        y3 = F.relu(self.fc3(y2))
        y = F.relu(self.fc4(y3))
        return y
    
class FullyConnectedPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(FullyConnectedPolicy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, act_dim)
    
    def forward(self, x):
        x = x.float()
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        y = F.relu(self.fc3(y2))
        return y
    
class DDPG:
    def __init__(self, env, env_name, device, batch_size, total_episodes, update_episodes, replay_size, actor_lr, critic_lr, resume, tau, gamma):

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        self.env = env
        self.obs_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]

        self.device = device

        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.total_episodes = total_episodes
        self.update_episodes = update_episodes

        self.tensorboard_dir = f"DDPG_logs/{env_name}/"
        self.model_dir = f"DDPG_pretrained/{env_name}/"

        self.actor = FullyConnectedPolicy(self.obs_size, self.action_size).to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.actor_lossfn = nn.MSELoss()

        self.critic = FullyConnectedQ(self.obs_size, self.action_size)
        self.critic_target = deepcopy(self.critic)
        self.critic_lossfn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(int(self.replay_size), self.device)

        for parm in self.actor_target.parameters():
            parm.requires_grad = False
        
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        os.makedirs(self.tensorboard_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.start_episode = 0

        if resume:
            checkpoint = torch.load(os.path.join(self.model_dir, "backup.ckpt"))
            self.start_episode = checkpoint['episode']+1
            self.actor.load_state_dict(checkpoint['actor'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.replay_buffer = checkpoint['replay_buffer']
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            
        self.train()

    def save_checkpoint(self, name):
        checkpoint = {'actor': self.actor.state_dict()}
        torch.save(checkpoint, os.path.join(self.model_dir, name))
    
    def save_backup(self, episode):
        checkpoint = {'episode' : episode,
                      'actor' : self.actor.state_dict(),
                      'actor_optimizer': self.actor_optimizer.state_dict(),
                      'critic' : self.critic.state_dict(),
                      'critic_optimizer': self.critic_optimizer.state_dict(),
                      'actor_target' : self.actor_target.state_dict(),
                      'critic_target' : self.critic_target.state_dict(),
                      'replay_buffer' : self.replay_buffer }
        torch.save(checkpoint, os.path.join(self.model_dir, "backup.ckpt"))
    
    def soft_update(self, target, source, tau):
        with torch.no_grad():
            for target_param, param in zip(target.parameters(),source.parameters()):
                target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)

    def train(self):
        writer = SummaryWriter(log_dir=self.tensorboard_dir)
        for episode in range(self.start_episode, self.total_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0

            while True:
                with torch.no_grad():
                    action = self.actor(torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
                action = action.cpu().numpy()[0]
                action += np.random.normal(0.0, 0.2, self.action_size)
                action = np.clip(action, -1.0, 1.0)
                
                next_obs, reward, done, _, _ = self.env.step(action)

                self.replay_buffer.push(obs, action, reward, next_obs)

                episode_reward += reward
                obs = next_obs

                if len(self.replay_buffer) >= self.batch_size:
                    obs, action, reward, next_obs = self.replay_buffer.sample(self.batch_size)
                    q_val = self.critic(obs, action)

                    with torch.no_grad():
                        next_q_val = self.critic_target(next_obs, self.actor_target(next_obs))
                    bellman_target = reward + self.gamma*next_q_val

                    critic_loss = self.critic_lossfn(q_val, bellman_target)
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    actor_loss = -torch.mean(self.critic(obs, self.actor(obs)))
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    self.soft_update(self.critic_target, self.critic, self.tau)
                    self.soft_update(self.actor_target, self.actor, self.tau)
                
                if done:
                    writer.add_scalar("Episode Reward", episode_reward, episode)
                    if (episode % 250 == 0 or episode == self.total_episodes-1) and episode > self.start_episode:
                        self.save_backup(episode)
                    break

if __name__ == "__main__":
    env = gym.make('Humanoid-v5', render_mode="human", width=1200, height=1000)

    device = "cpu"

    obs, _  = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        env.render()

    # ddpg = DDPG(
    #     env = env,
    #     env_name="hopper",
    #     device=device,
    #     batch_size=64,
    #     total_episodes=3000,
    #     update_episodes=1e4,
    #     replay_size=1e6,
    #     actor_lr=2.5e-4,
    #     critic_lr=2.5e-4,
    #     resume=False,
    #     tau=5e-3,
    #     gamma=0.99
    # )