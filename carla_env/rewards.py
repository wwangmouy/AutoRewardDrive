"""
AutoRewardDrive: Sparse Reward Function
Provides sparse reward signal for upper-level optimization
"""

import numpy as np
from config import CONFIG


# Load configuration parameters
max_distance = CONFIG.reward_params.max_distance
max_speed = CONFIG.reward_params.max_speed
early_stop = CONFIG.reward_params.early_stop

reward_functions = {}


def create_reward_fn(reward_fn):
    """
    Wrapper that handles terminal conditions and early stopping.
    """
    def func(env):
        terminal_reason = "Running..."
        if early_stop:
            speed = env.vehicle.get_speed()
            if speed < 1.0:
                env.low_speed_timer += 1
            else:
                env.low_speed_timer = 0.0
            if env.low_speed_timer >= 15 * env.fps:  
                env.terminal_state = True
                terminal_reason = "Vehicle stopped"
            if env.distance_from_center > max_distance and not env.eval:
                env.terminal_state = True
                terminal_reason = "Off-track"
            if max_speed > 0 and speed > max_speed and not env.eval:
                env.terminal_state = True
                terminal_reason = "Too fast"
        reward = reward_fn(env)
        
        if env.terminal_state:
            env.low_speed_timer = 0.0
            print(f"{env.episode_idx}| Terminal: {terminal_reason}")
        
        if env.success_state:
            print(f"{env.episode_idx}| Success")
        
        env.extra_info.extend([terminal_reason, ""])
        return reward
    
    return func


def sparse_reward(env):
    """
    Sparse Reward Function (R̄) for Upper-Level Optimization
    Reward Structure:
    - Collision:        -1.0  (terminal failure)
    - Off-track:        -0.5  (driving off the road)
    - Destination:      +1.0  (successfully reached goal)
    - Progress:         +0.01 per waypoint passed (helps avoid local optima)
    - Timeout/Stopped:  -0.3  (encourages forward motion)
    Returns:
        float: Sparse reward signal in range [-1, 1]
    """
    reward = 0.0
    if env.collision_state:
        return -1.0
    if env.distance_from_center > max_distance:
        return -0.5
    if env.low_speed_timer >= 30 * env.fps:  
        reward -= 0.3
    if env.success_state:
        return 1.0
    waypoints_passed = env.current_waypoint_index - env.prev_waypoint_index
    if waypoints_passed > 0:
        progress_reward = 0.01 * waypoints_passed
        reward += min(progress_reward, 0.1)  # Cap at 0.1 per step
    return np.clip(reward, -1.0, 1.0)


# Register the sparse reward function
reward_functions["sparse_reward"] = create_reward_fn(sparse_reward)

# Default reward function for the framework
reward_functions["default"] = reward_functions["sparse_reward"]