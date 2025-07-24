import torch
import numpy as np
import torch.nn as nn
from rich.console import Console
from torch.distributions import MultivariateNormal

console = Console()

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    console.print(f"[PPO] Device set to : [{str(torch.cuda.get_device_name(device))}]", style="cyan")
else:
    console.print(f"[PPO] Device set to : cpu", style="cyan")

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), fill_value=0.5).to(device)
        self.cov_mat = torch.diag(self.action_var)

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self):
        return NotImplementedError
 
    def act(self, state):
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, self.cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, self.cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(
            self, 
            state_dim, 
            action_dim, 
            K_epochs=10,
            learning_rate=1e-3, 
            gamma=0.998,  
            clip_param=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01, 
            max_grad_norm=1.0,
            schedule="fixed",
            desired_kl=0.01):

        self.gamma = gamma
        self.schedule = schedule
        self.K_epochs = K_epochs
        self.clip_param = clip_param
        self.desired_kl = desired_kl
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer  =torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr':learning_rate},
            {'params': self.policy.critic.parameters(), 'lr':learning_rate}
        ])

        self.learning_rate = learning_rate

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse_loss = nn.MSELoss()
    
    def adapt_learning_rate(self, kl, kl_tolerance=2, kl_factor = 1.5):
        if kl > self.desired_kl * kl_tolerance:
            self.learning_rate = max(1e-5, self.learning_rate / kl_factor)
        
        elif kl < self.desired_kl / kl_tolerance and kl > 0.0:
            self.learning_rate = min(1e-2, self.learning_rate * kl_factor)
        
        for param_group in self.optimizer.param_groups:
            if param_group['params'] == self.policy.actor.parameters():
                param_group['lr'] = self.learning_rate
            else:
                param_group['lr'] = self.learning_rate
    
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        advantages = rewards.detach() - old_state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
                                                                                     
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 =  ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * advantages
            surrogate_loss = -torch.mean(torch.min(surr1, surr2))

            value_loss = self.mse_loss(state_values, rewards)

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * dist_entropy.mean()

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.no_grad():
                    log_ratio = logprobs - old_logprobs.detach()
                    kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    self.adapt_learning_rate(kl)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save({
            'model_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
    
    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.policy_old.load_state_dict(checkpoint['model_state_dict'])
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])