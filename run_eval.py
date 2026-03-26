import argparse
import glob
import os
import subprocess
import time


def kill_carla():
    print("Killing Carla server\n")
    time.sleep(1)
    subprocess.run(["killall", "-9", "CarlaUE4-Linux-Shipping"], check=False)
    time.sleep(4)


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation over one or more training checkpoints")
    parser.add_argument("--model_dir", type=str, required=True, help="Training run directory under tensorboard or an absolute path")
    parser.add_argument("--models", nargs="+", default=None, help="Checkpoint filenames such as model_200000_steps.zip")
    parser.add_argument("--config", type=str, default=None, help="Override config id. Defaults to the suffix in model_dir")
    parser.add_argument("--town", type=str, default="Town02")
    parser.add_argument("--density", choices=["empty", "regular", "dense"], default="regular")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--port", type=int, default=2020)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--use_shield", action="store_true", help="Apply the evaluation shield")
    parser.add_argument("--inference_mode", choices=["step", "chunked"], default="step")
    parser.add_argument("--eval_tag", type=str, default="")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    tensorboard_path = "./tensorboard"
    if os.path.isabs(args.model_dir):
        target_model_dir = args.model_dir
    else:
        target_model_dir = os.path.join(tensorboard_path, args.model_dir)

    if not os.path.isdir(target_model_dir):
        raise FileNotFoundError(f"Model directory not found: {target_model_dir}")

    config_id = args.config or os.path.basename(target_model_dir).split("id")[-1]
    model_ckpts = sorted(glob.glob(os.path.join(target_model_dir, "*.zip")))
    selected_models = set(args.models) if args.models else None
    available_models = [
        ckpt for ckpt in model_ckpts
        if selected_models is None or os.path.basename(ckpt) in selected_models
    ]
    if not available_models:
        raise FileNotFoundError(f"No matching checkpoints found in {target_model_dir}")

    print(f"Processing training: {target_model_dir}")
    print(f"Config ID: {config_id}")
    print(f"Found {len(available_models)} checkpoints")
    print("=" * 60)

    for model_ckpt in available_models:
        kill_carla()
        print(model_ckpt)
        args_eval = [
            "--model", model_ckpt,
            "--config", config_id,
            "--town", args.town,
            "--density", args.density,
            "--device", args.device,
            "--port", str(args.port),
            "--seed", str(args.seed),
            "--episodes", str(args.episodes),
            "--inference_mode", args.inference_mode,
        ]
        if args.use_shield:
            args_eval.append("--use_shield")
        if args.eval_tag:
            args_eval.extend(["--eval_tag", args.eval_tag])
        subprocess.run(["python", "eval.py"] + args_eval, check=False)
