import glob
import subprocess
import time
import os

from tqdm import tqdm


def kill_carla():
    print("Killing Carla server\n")
    time.sleep(1)
    subprocess.run(["killall", "-9", "CarlaUE4-Linux-Shipping"])
    time.sleep(4)


if __name__ == '__main__':
    # change the ckp as you want
    selected_models = [f"model_{i}_steps.zip" for i in [290000]]
    tensorboard_path = './tensorboard'
    
    # change the target model dir as you want
    target_model_dir = '/home/ubuntu/wy/AutoRewardDrive/tensorboard/AutoRewardedSAC_20260119_211107_id3'
    print(f"Processing training: {target_model_dir}")
    print("=" * 60)
    
    config = target_model_dir.split('id')[-1]
    print(f"Config ID: {config}")
    
    model_ckpts = glob.glob(os.path.join(tensorboard_path, target_model_dir, "*.zip"))
    model_ckpt_filenames = [os.path.basename(path) for path in model_ckpts]
    
    # check the model ckp wheather is available
    available_models = [m for m in selected_models if m in model_ckpt_filenames]
    if not available_models:
        print(f"None of the selected models {selected_models} were found in {target_model_dir}")
        exit(1)
    
    print(f"Found {len(available_models)} models to evaluate: {available_models}")
    print("=" * 60)

    
    for model_ckpt in model_ckpts:
        if model_ckpt.split('/')[-1] not in selected_models: continue
        # summary_path = os.path.join(tensorboard_path, latest_model_dir, "eval",
        #                             os.path.basename(model_ckpt).replace(".zip", "_eval_summary.csv"))
        # if os.path.exists(summary_path):
        #     print("Already exists: ", summary_path)
        #     continue
        kill_carla()
        print(model_ckpt)

        args_eval = [
            "--model", model_ckpt,
            "--config", config,
            "--town", "Town02",  # default "Town02", change this to evaluate on other towns
            "--density", "regular",  # default "regular", change this to `dense` or `empty` to evaluate on other densities
        ]

        subprocess.run(["python", "eval.py"] + args_eval)
