import os
import torch
from algo.ppo import PPO
from rich.console import Console
from datetime import datetime
from leggedEnv import LeggedEnv
from torch.utils.tensorboard import SummaryWriter

def timestamp():
    return datetime.now().strftime("%H:%M:%S")

console = Console()

def train():
    ####### initialize environment hyperparameters ######
    env_name = "go2-walking"
    max_ep_len = 1000
    max_training_timesteps = int(2e9)

    print_freq = max_ep_len * 10
    log_freq = max_ep_len*2
    save_model_freq = int(1e5)
    #####################################################

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len*4
    K_epochs = 10

    clip_param = 0.2
    gamma = 0.99

    learning_rate  = 1e-3
    #####################################################

    env = LeggedEnv(render_mode="non-human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low, action_high = env.action_low, env.action_high

    ppo_agent = PPO(
        state_dim, 
        action_dim,
        action_low,
        action_high,
        K_epochs,
        learning_rate,
        gamma,
        clip_param,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        schedule="adaptive")

    console.print(f"[Mujoco-Training] [{timestamp()}] Training Environment Name: {env_name}", style="magenta")

    ################### checkpointing ###################
    weights_dir = f"PPO_pretrained/{env_name}/"
    os.makedirs(weights_dir, exist_ok=True)
    checkpoint_path = f"{weights_dir}PPO_{env_name}.pth"
    training_state_path = f"{weights_dir}training_state.pth"

    if os.path.exists(checkpoint_path) and os.path.exists(training_state_path):
        ppo_agent.load(checkpoint_path)
        training_state = torch.load(training_state_path)
        time_step = training_state['time_step']
        i_episode = training_state['i_episode']
        log_running_reward = training_state['log_running_reward']
        log_running_episodes = training_state['log_running_episodes']
        print_running_reward = training_state['print_running_reward']
        print_running_episodes = training_state['print_running_episodes']
        start_time = training_state['start_time']
        console.print(f"[Mujoco-Training] [{timestamp()}] Loaded existing weights from {checkpoint_path}", style="green")
    
    else:
        start_time = datetime.now().replace(microsecond=0)
        console.print(f"[Mujoco-Training] [{timestamp()}] Started training at (GMT) :", start_time, style="magenta")
        time_step = 0
        i_episode = 0
        log_running_reward = 0
        log_running_episodes = 0
        print_running_reward = 0
        print_running_episodes = 0

    #####################################################

    ###################### logging ###################### 
    log_dir = f"PPO_logs/{env_name}/"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir, purge_step=time_step)
    #####################################################
    
    while time_step <= max_training_timesteps:
        state, _ = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            if time_step % update_timestep == 0:
                ppo_agent.update()
            
            if time_step % log_freq == 0:
                log_avg_reward = log_running_reward / log_running_episodes if log_running_episodes > 0 else 0
                writer.add_scalar("Average_Reward", log_avg_reward, time_step)
                log_running_episodes = 0
                log_running_reward  = 0
            
            if time_step % print_freq == 0:
                print_avg_reward = print_running_reward / print_running_episodes if print_running_episodes > 0 else 0
                print_avg_reward = round(print_avg_reward, 2)

                console.print(f"[Mujoco-Training] [{timestamp()}] Episode: {i_episode} Timestep: {time_step} Average Reward: {print_avg_reward}", style="yellow")

                print_running_reward = 0
                print_running_episodes = 0
            
            if time_step % save_model_freq == 0:
                ppo_agent.save(checkpoint_path)
                training_state = {
                    'time_step': time_step,
                    'i_episode': i_episode,
                    'log_running_reward': log_running_reward,
                    'log_running_episodes': log_running_episodes,
                    'print_running_reward': print_running_reward,
                    'print_running_episodes': print_running_episodes,
                    'start_time': start_time,
                }
                torch.save(training_state, training_state_path)
            
            if done or truncated: break
        
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1
    
    env.close()
    writer.close()

    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        console.print(f"[Mujoco-Training] [{timestamp}] Training interrupted.", style="red")