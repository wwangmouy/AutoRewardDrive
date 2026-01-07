"""AutoRewardDrive: Automatic Reward Discovery for Autonomous Driving"""

import warnings
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import set_config
from carla_env.envs.carla_route_env import CarlaRouteEnv
from carla_env.state_commons import create_encode_state_fn
from carla_env.rewards import reward_functions
from utils import write_json

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'reward_upper'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'agent'))
from reward_machine import RewardLearner, Transition
from sac_agent import AutoRewardDrive_SAC
from shared_encoder import SharedStateEncoder

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--host", default="localhost", type=str)
parser.add_argument("--port", default=2000, type=int)
parser.add_argument("--total_timesteps", type=int, default=1_000_000)
parser.add_argument("--start_carla", action="store_true")
parser.add_argument("--no_render", action="store_false")
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--log_dir", type=str, default="tensorboard")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--upper_update_freq", type=int, default=10)
parser.add_argument("--save_freq", type=int, default=10000)
args = vars(parser.parse_args())

CONFIG = set_config("default")


def extract_state(obs):
    """Extract BEV and vector state from observation dict"""
    # BEV: permute (H,W,C) -> (C,H,W) and normalize to [0,1]
    bev = torch.tensor(obs['seg_camera'], dtype=torch.float32).permute(2, 0, 1) / 255.0
    
    # Vector: concatenate vehicle_measures and waypoints
    vector_parts = []
    if 'vehicle_measures' in obs:
        vector_parts.append(np.array(obs['vehicle_measures']).flatten())
    if 'waypoints' in obs:
        vector_parts.append(np.array(obs['waypoints']).flatten())
    vector = torch.tensor(np.concatenate(vector_parts), dtype=torch.float32)
    return bev, vector


def train():
    os.makedirs(args["log_dir"], exist_ok=True)
    device = args["device"]
    
    # Create environment
    observation_space, encode_state_fn = create_encode_state_fn(CONFIG.state, CONFIG)
    env = CarlaRouteEnv(
        obs_res=CONFIG.obs_res, host=args["host"], port=args["port"],
        reward_fn=reward_functions[CONFIG.reward_fn],
        observation_space=observation_space, encode_state_fn=encode_state_fn,
        fps=args["fps"], action_smoothing=CONFIG.action_smoothing,
        action_space_type='continuous',
        activate_spectator=args["no_render"], activate_render=args["no_render"],
        activate_seg_bev=CONFIG.use_seg_bev, start_carla=args["start_carla"],
    )
    
    # Create shared encoder
    shared_encoder = SharedStateEncoder(
        bev_channels=6,
        vector_dim=34,
        bev_features=128,
        vector_features=64
    ).to(device)
    
    # Initialize agent with shared encoder
    agent = AutoRewardDrive_SAC(CONFIG, shared_encoder=shared_encoder, device=device)
    
    # Initialize reward learner with shared encoder
    reward_learner = RewardLearner(CONFIG, shared_encoder=shared_encoder, device=device)
    
    # Logging setup
    log_dir = os.path.join(args["log_dir"], f'AutoRewardDrive_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(log_dir, exist_ok=True)
    write_json(CONFIG, os.path.join(log_dir, 'config.json'))
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Training loop
    obs = env.reset()
    bev, vector = extract_state(obs)
    
    episode_reward = 0
    episode_sparse_reward = 0
    episode_count = 0
    current_trajectory = []
    pbar = tqdm(range(args["total_timesteps"]), desc="Training", unit="step")
    for step in pbar:
        # Select action
        action = agent.select_action(bev.unsqueeze(0), vector.unsqueeze(0))
        action_np = action.squeeze().numpy()
        
        # Step environment
        next_obs, sparse_reward, done, info = env.step(action_np)
        next_bev, next_vector = extract_state(next_obs)
        
        # Get learned reward for training
        with torch.no_grad():
            learned_reward = reward_learner.get_reward(
                bev.unsqueeze(0).to(device),
                vector.unsqueeze(0).to(device),
                action.to(device)
            )
        
        # Store in replay buffer with learned reward
        agent.replay_buffer.push(bev, vector, action.squeeze(), float(learned_reward), 
                                  next_bev, next_vector, done)
        
        # Store transition for upper-level
        transition = Transition(
            bev_image=bev,
            vector_state=vector,
            action=action.squeeze(),
            sparse_reward=sparse_reward,
            log_prob=torch.tensor(0.0),
            mu=torch.tensor(0.0),
            overline_V=0.0
        )
        current_trajectory.append(transition)
        
        # Update agent (lower-level)
        update_info = None
        if step > CONFIG.algorithm_params.learning_starts:
            update_info = agent.update()
        
        episode_reward += learned_reward
        episode_sparse_reward += sparse_reward
        bev, vector = next_bev, next_vector
        
        if done:
            episode_count += 1
            
            # Store trajectory
            if len(current_trajectory) > 0:
                reward_learner.store_trajectory(current_trajectory)
            current_trajectory = []
            
            # Upper-level optimization
            reward_loss = None
            if episode_count % args["upper_update_freq"] == 0:
                reward_loss = reward_learner.optimize_upper_level(agent)
            
            # TensorBoard logging
            writer.add_scalar("episode/learned_reward", episode_reward, episode_count)
            writer.add_scalar("episode/sparse_reward", episode_sparse_reward, episode_count)
            writer.add_scalar("episode/length", info.get("episode_length", 0), episode_count)
            writer.add_scalar("episode/distance", info.get("total_distance", 0), episode_count)
            writer.add_scalar("episode/avg_speed", info.get("avg_speed", 0), episode_count)
            writer.add_scalar("episode/collision", 1 if info.get("collision_state", False) else 0, episode_count)
            
            if update_info:
                writer.add_scalar("losses/critic", update_info.get("critic_loss", 0), episode_count)
                writer.add_scalar("losses/actor", update_info.get("actor_loss", 0), episode_count)
                writer.add_scalar("losses/alpha", update_info.get("alpha", 0), episode_count)
            
            if reward_loss is not None:
                writer.add_scalar("losses/reward", reward_loss, episode_count)
            
            # Update progress bar
            pbar.set_postfix({
                'ep': episode_count,
                'reward': f'{episode_reward:.1f}',
                'dist': f'{info.get("total_distance", 0):.0f}m'
            })
            
            # Save checkpoint
            if step % args["save_freq"] == 0:
                agent.save(os.path.join(log_dir, f'agent_{step}.pth'))
                reward_learner.save(os.path.join(log_dir, f'reward_{step}.pth'))
            
            # Reset
            obs = env.reset()
            bev, vector = extract_state(obs)
            episode_reward = 0
            episode_sparse_reward = 0
    
    # Final save
    agent.save(os.path.join(log_dir, 'agent_final.pth'))
    reward_learner.save(os.path.join(log_dir, 'reward_final.pth'))
    writer.close()
    env.close()


if __name__ == "__main__":
    train()
