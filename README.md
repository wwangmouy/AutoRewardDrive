python train.py --config 3 --total_timesteps 500000 --device cuda:1 --host localhost --port 2000 --fps 15 --no_render --start_carla

CUDA_VISIBLE_DEVICES=1 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json __NV_PROME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=NVIDIA ./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-port=2000