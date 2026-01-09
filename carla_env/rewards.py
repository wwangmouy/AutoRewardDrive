"""AutoRewardDrive: Sparse Reward Function"""

import numpy as np
from config import CONFIG

max_center_deviation = CONFIG.reward_params.max_center_deviation
max_speed = CONFIG.reward_params.max_speed
early_stop = CONFIG.reward_params.early_stop
max_steps = CONFIG.reward_params.max_steps

reward_functions = {}


def create_reward_fn(reward_fn):
    def func(env):
        # Auto-reset on new episode
        current_ep = getattr(env, 'episode_idx', -1)
        last_ep = getattr(env, '_wrapper_last_ep', -2)
        
        if current_ep != last_ep:
            env.wrapper_step_count = 0
            env.low_speed_timer = 0.0
            env.failure_counted = False
            env._wrapper_last_ep = current_ep

        prev_terminal = env.terminal_state
        prev_success = env.success_state
        
        # Collision forces terminal state
        if env.collision_state:
            env.terminal_state = True
            env.terminal_reason = "collision"
        
        if not prev_terminal and not prev_success and not env.terminal_state:
             env.terminal_reason = None
        
        if early_stop:
            if not hasattr(env, 'wrapper_step_count'):
                env.wrapper_step_count = 0
            env.wrapper_step_count += 1

            # Speed check (km/h)
            STOPPED_SPEED_KMH = 3.0
            speed_kmh = env.vehicle.get_speed()
            
            if speed_kmh < STOPPED_SPEED_KMH:
                env.low_speed_timer += 1
            else:
                env.low_speed_timer = 0.0
            
            # Additional termination checks
            if not env.terminal_state and not prev_terminal and not prev_success:
                if env.distance_from_center > max_center_deviation:
                    env.terminal_state = True
                    env.terminal_reason = "off_track"
                elif max_speed > 0 and speed_kmh > max_speed:
                    env.terminal_state = True
                    env.terminal_reason = "too_fast"
                elif env.wrapper_step_count >= max_steps:
                    env.terminal_state = True
                    env.terminal_reason = "timeout"
                elif env.low_speed_timer >= 15 * env.fps:
                    env.terminal_state = True
                    env.terminal_reason = "stopped"

        if not hasattr(env, 'global_failure_counts'):
            env.global_failure_counts = {"collision": 0, "off_track": 0, "too_fast": 0, "stopped": 0, "timeout": 0}
        
        if env.terminal_state and env.terminal_reason:
            if not getattr(env, 'failure_counted', False):
                env.global_failure_counts[env.terminal_reason] = env.global_failure_counts.get(env.terminal_reason, 0) + 1
                env.failure_counted = True
            
        reward = reward_fn(env)
        
        if env.terminal_state:
            env.low_speed_timer = 0.0
        
        env.extra_info.extend([env.terminal_reason or "Running...", ""])
        return reward
    
    return func


def sparse_reward(env):
    """Sparse reward: success=+1, failures~=-1"""
    if env.collision_state:
        return -1.0
    if env.success_state:
        return 1.0
    if env.terminal_state:
        return -0.9 # Generalized penalty for failure
    return 0.0


reward_functions["sparse_reward"] = create_reward_fn(sparse_reward)
reward_functions["default"] = reward_functions["sparse_reward"]