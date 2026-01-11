"""AutoRewardDrive: Training Script"""

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
from utils import write_json

# Initialize CONFIG first before importing modules that depend on it
parser = argparse.ArgumentParser()
parser.add_argument("--host", default="localhost", type=str)
parser.add_argument("--port", default=2000, type=int)
parser.add_argument("--total_timesteps", type=int, default=1_000_000)
parser.add_argument("--start_carla", action="store_true")
parser.add_argument("--render", action="store_true", help="Enable rendering")
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--log_dir", type=str, default="tensorboard")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--upper_update_freq", type=int, default=10)
parser.add_argument("--save_freq", type=int, default=10000)
args = vars(parser.parse_args())

CONFIG = set_config("default")

# Import after CONFIG is initialized
from carla_env.envs.carla_route_env import CarlaRouteEnv
from carla_env.state_commons import create_encode_state_fn
from carla_env.rewards import reward_functions

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'reward_upper'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'agent'))
from reward_machine import RewardLearner, Transition
from sac_agent import AutoRewardDrive_SAC
from shared_encoder import SharedStateEncoder


def extract_state(obs):
    """Extract BEV and vector state from observation dict"""
    bev = torch.tensor(obs['seg_camera'], dtype=torch.float32).permute(2, 0, 1) / 255.0
    
    vector_parts = []
    if 'vehicle_measures' in obs:
        vector_parts.append(np.array(obs['vehicle_measures']).flatten())
    if 'waypoints' in obs:
        vector_parts.append(np.array(obs['waypoints']).flatten())
    
    if not vector_parts:
        raise ValueError(f"obs missing required keys: need 'vehicle_measures' or 'waypoints'")
    
    vector = torch.tensor(np.concatenate(vector_parts), dtype=torch.float32)
    assert vector.shape[0] == 34, f"Vector dimension mismatch: expected 34, got {vector.shape[0]}"
    return bev, vector


def train():
    os.makedirs(args["log_dir"], exist_ok=True)
    device = args["device"]
    
    observation_space, encode_state_fn = create_encode_state_fn(CONFIG.state, CONFIG)
    
    # Check action_smoothing consistency
    if CONFIG.action_smoothing != 0.0:
        print(f"\nWARNING: action_smoothing = {CONFIG.action_smoothing} != 0.0")
        print(f"This will cause incorrect importance sampling in reward learning!")
        print(f"Recommendation: Set CONFIG.action_smoothing = 0.0\n")
    
    env = CarlaRouteEnv(
        obs_res=CONFIG.obs_res, host=args["host"], port=args["port"],
        reward_fn=reward_functions[CONFIG.reward_fn],
        observation_space=observation_space, encode_state_fn=encode_state_fn,
        fps=args["fps"], action_smoothing=CONFIG.action_smoothing,
        action_space_type='continuous',
        activate_spectator=args["render"], activate_render=args["render"],
        activate_seg_bev=CONFIG.use_seg_bev, start_carla=args["start_carla"],
    )
    
    shared_encoder = SharedStateEncoder(
        bev_channels=6, vector_dim=34, bev_features=128, vector_features=64
    ).to(device)
    
    agent = AutoRewardDrive_SAC(CONFIG, shared_encoder=shared_encoder, device=device)
    reward_learner = RewardLearner(CONFIG, shared_encoder=shared_encoder, device=device)
    
    log_dir = os.path.join(args["log_dir"], f'AutoRewardDrive_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(log_dir, exist_ok=True)
    write_json(CONFIG, os.path.join(log_dir, 'config.json'))
    writer = SummaryWriter(log_dir)
    
    obs = env.reset()
    bev, vector = extract_state(obs)
    
    episode_reward = 0
    episode_sparse_reward = 0
    episode_learned_return = 0
    episode_count = 0
    current_trajectory = []
    pbar = tqdm(range(args["total_timesteps"]), desc="Training", unit="step")
    
    for step in pbar:
        action, log_prob = agent.select_action(bev.unsqueeze(0), vector.unsqueeze(0))
        
        action = action.squeeze(0)
        log_prob_scalar = float(log_prob.view(-1)[0].item())
        # Detach before converting to numpy (robust to future agent changes)
        action_np = action.detach().cpu().numpy()
        
        next_obs, sparse_reward, done, info = env.step(action_np)
        next_bev, next_vector = extract_state(next_obs)
        
        # Reward mixing (warmup period + gradual transition)
        reward_warmup_steps = CONFIG.reward_learning_params.reward_warmup_steps
        reward_mixing_steps = max(1, CONFIG.reward_learning_params.reward_mixing_steps)
        
        if step < reward_warmup_steps:
            alpha = 0.0
        else:
            alpha = min(1.0, (step - reward_warmup_steps) / reward_mixing_steps)
        
        learned_reward = 0.0
        if alpha > 0:
            learned_reward_tensor = reward_learner.get_reward(
                bev.unsqueeze(0).to(device),
                vector.unsqueeze(0).to(device),
                action.unsqueeze(0).to(device),
                update_stats=False
            )
            learned_reward = float(learned_reward_tensor)
        
        training_reward = (1.0 - alpha) * sparse_reward + alpha * learned_reward
        
        agent.replay_buffer.push(bev, vector, action, training_reward, 
                                  next_bev, next_vector, done)
        
        current_trajectory.append(Transition(
            bev_image=bev.cpu(),
            vector_state=vector.cpu(),
            action=action.cpu(),
            sparse_reward=sparse_reward,
            log_prob=log_prob_scalar,
            overline_V=0.0,
            next_bev=next_bev.cpu(),
            next_vector=next_vector.cpu(),
            done=done
        ))
        
        update_info = None
        if step > CONFIG.algorithm_params.learning_starts:
            update_info = agent.update()
            reward_learner.update_target_encoder()
        
        episode_reward += training_reward
        episode_sparse_reward += sparse_reward
        episode_learned_return += learned_reward
        bev, vector = next_bev, next_vector
        
        if done:
            episode_count += 1
            
            min_steps = CONFIG.reward_learning_params.min_steps_per_traj
            if len(current_trajectory) >= min_steps:
                reward_learner.store_trajectory(current_trajectory)
            current_trajectory = []
            
            reward_loss = None
            min_trajectories = CONFIG.reward_learning_params.min_trajectories
            if step >= reward_warmup_steps and len(reward_learner.trajectory_buffer) >= min_trajectories:
                if episode_count % args["upper_update_freq"] == 0:
                    reward_loss = reward_learner.optimize_upper_level(agent)
            
            writer.add_scalar("episode/mixed_return", episode_reward, episode_count)
            writer.add_scalar("episode/sparse_return", episode_sparse_reward, episode_count)
            writer.add_scalar("episode/learned_return", episode_learned_return, episode_count)
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
            
            pbar.set_postfix({
                'ep': episode_count,
                'reward': f'{episode_reward:.1f}',
                'dist': f'{info.get("total_distance", 0):.0f}m'
            })
            
            obs = env.reset()
            bev, vector = extract_state(obs)
            episode_reward = 0
            episode_sparse_reward = 0
            episode_learned_return = 0
        
        if step % args["save_freq"] == 0 and step > 0:
            agent.save(os.path.join(log_dir, f'agent_{step}.pth'))
            reward_learner.save(os.path.join(log_dir, f'reward_{step}.pth'))

    
    agent.save(os.path.join(log_dir, 'agent_final.pth'))
    reward_learner.save(os.path.join(log_dir, 'reward_final.pth'))
    writer.close()
    env.close()


if __name__ == "__main__":
    train()
