import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import config

parser = argparse.ArgumentParser(description="Trains a CARLA agent")
parser.add_argument("--host", default="localhost", type=str, help="IP of the host server (default: 127.0.0.1)")
parser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total timestep to train for")
parser.add_argument("--start_carla", action="store_true", help="If True, start a CARLA server")
parser.add_argument("--no_render", action="store_false", help="If True, render the environment")
parser.add_argument("--fps", type=int, default=15, help="FPS to render the environment")
parser.add_argument("--num_checkpoints", type=int, default=20, help="Checkpoint number")
parser.add_argument("--log_dir", type=str, default="tensorboard", help="Directory to save logs")
parser.add_argument("--device", type=str, default="cuda:0", help="cpu, cuda:0, cuda:1, cuda:2")
parser.add_argument("--config", type=str, default="4", help="Config to use (default: 4)")
parser.add_argument("--run_name", type=str, default="", help="Optional custom run suffix")
parser.add_argument("--resume_model", type=str, default="", help="Optional checkpoint zip to resume from")
parser.add_argument("--resume_replay_buffer", type=str, default="", help="Optional replay buffer pickle to resume from")
parser.add_argument("--resume_autoreward_state", type=str, default="", help="Optional autoreward sidecar pickle to resume from")
parser.add_argument("--latest_bundle_freq", type=int, default=25000, help="How often to refresh the resumable latest bundle; <=0 disables it")

args = vars(parser.parse_args())
CONFIG = config.set_config(args["config"])
CONFIG.algorithm_params.device = args["device"]

from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.logger import configure
from carla_env.envs.carla_route_env import CarlaRouteEnv
from carla_env.state_commons import create_encode_state_fn
from carla_env.rewards import reward_functions
from utils import (
    CurriculumCallback,
    HParamCallback,
    RobustCheckpointCallback,
    TensorboardCallback,
    load_auto_reward_state,
    save_auto_reward_state,
    write_json,
    parse_wrapper_class,
)
from auto_reward.auto_sac import AutoRewardedSAC, AutoRewardedSACV2

os.makedirs(args["log_dir"], exist_ok=True)

algorithm_dict = {
    "PPO": PPO, "DDPG": DDPG, "SAC": SAC,
    "SAC_AUTO": AutoRewardedSAC,
    "SAC_AUTO_V2": AutoRewardedSACV2,
}
if CONFIG.algorithm not in algorithm_dict:
    raise ValueError("Invalid algorithm name")

AlgorithmRL = algorithm_dict[CONFIG.algorithm]


def _default_sidecar_path(model_path, suffix):
    stem = model_path[:-4] if model_path.endswith(".zip") else model_path
    return stem + suffix

observation_space, encode_state_fn = create_encode_state_fn(CONFIG.state, CONFIG)
action_space_type = 'continuous' if CONFIG.action_space_type != 'discrete' else 'discrete'

env = CarlaRouteEnv(obs_res=CONFIG.obs_res, host=args["host"], port=args["port"],
                    reward_fn=reward_functions[CONFIG.reward_fn], observation_space=observation_space,
                    encode_state_fn=encode_state_fn, fps=args["fps"],
                    action_smoothing=CONFIG.action_smoothing, action_space_type=action_space_type,
                    activate_spectator=args["no_render"], activate_render=args["no_render"],
                    activate_bev=CONFIG.use_rgb_bev, activate_seg_bev=CONFIG.use_seg_bev,
                    activate_traffic_flow=True, start_carla=args["start_carla"],
                    )

for wrapper_class_str in CONFIG.wrappers:
    wrap_class, wrap_params = parse_wrapper_class(wrapper_class_str)
    env = wrap_class(env, *wrap_params)

resume_model = args["resume_model"].strip()
if resume_model:
    if AlgorithmRL.__name__ in {"AutoRewardedSAC", "AutoRewardedSACV2"}:
        model = AlgorithmRL.load(
            resume_model,
            env=env,
            config=CONFIG,
            device=args["device"],
        )
    else:
        model = AlgorithmRL.load(
            resume_model,
            env=env,
            device=args["device"],
        )

    replay_buffer_path = args["resume_replay_buffer"].strip() or _default_sidecar_path(
        resume_model, "_replay_buffer.pkl"
    )
    if replay_buffer_path and os.path.exists(replay_buffer_path) and hasattr(model, "load_replay_buffer"):
        model.load_replay_buffer(replay_buffer_path)

    auto_reward_state_path = args["resume_autoreward_state"].strip() or _default_sidecar_path(
        resume_model, "_autoreward.pkl"
    )
    if auto_reward_state_path and os.path.exists(auto_reward_state_path):
        load_auto_reward_state(model, auto_reward_state_path)
else:
    if AlgorithmRL.__name__ in {"AutoRewardedSAC", "AutoRewardedSACV2"}:
        model = AlgorithmRL(
            policy='MultiInputPolicy',
            env=env,
            config=CONFIG,  # Critical: config must be passed
            verbose=1,
            seed=CONFIG.seed,
            tensorboard_log=args["log_dir"],
            **CONFIG.algorithm_params
        )
    else:
        model = AlgorithmRL(
            'MultiInputPolicy',
            env,
            verbose=1,
            seed=CONFIG.seed,
            tensorboard_log=args["log_dir"],
            **CONFIG.algorithm_params
        )

if args["run_name"].strip():
    model_suffix = f'{args["run_name"].strip()}_id{args["config"]}'
else:
    model_suffix = "{}_id{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"), args['config'])
model_name = f'{model.__class__.__name__}_{model_suffix}'
model_dir = os.path.join(args["log_dir"], model_name)

new_logger = configure(model_dir, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)
write_json(CONFIG, os.path.join(model_dir, 'config.json'))
write_json(args, os.path.join(model_dir, 'train_args.json'))

checkpoint_freq = max(args["total_timesteps"] // max(args["num_checkpoints"], 1), 1)
latest_bundle_freq = args["latest_bundle_freq"] if args["latest_bundle_freq"] > 0 else None

callbacks = [
    HParamCallback(CONFIG),
    TensorboardCallback(1),
    RobustCheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=model_dir,
        name_prefix="model",
        latest_bundle_freq=latest_bundle_freq,
        save_replay_buffer=True,
        save_auto_reward_state_flag=True,
    ),
]
if CONFIG.get("curriculum_thresholds", {}):
    callbacks.append(CurriculumCallback(CONFIG))

try:
    model.learn(
        total_timesteps=args["total_timesteps"],
        callback=callbacks,
        reset_num_timesteps=False,
    )
finally:
    try:
        final_stem = os.path.join(model_dir, "model_final")
        model.save(final_stem)
        if hasattr(model, "save_replay_buffer"):
            model.save_replay_buffer(final_stem + "_replay_buffer.pkl")
        save_auto_reward_state(model, final_stem + "_autoreward.pkl")
    except Exception:
        pass
    try:
        env.close()
    finally:
        del model
