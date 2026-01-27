
# 启动训练命令
python train.py --config 3 --total_timesteps 1000000 --device cuda:0 --host localhost --port 2000 --fps 15 --start_carla --no_render

# 启动CARLA服务器
CUDA_VISIBLE_DEVICES=1 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json __NV_PROME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=NVIDIA ./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-port=2000





# AutoRewardDrive - 训练启动指南

## 快速启动

### 步骤 1：启动 CARLA 服务器

**新终端 1**：
```bash
cd /home/wy/CARLA_0.9.13
DISPLAY= ./CarlaUE4.sh -opengl -quality-level=Low
```

等待看到：`Listening on port 2000`

### 步骤 2：启动训练

**新终端 2**：
```bash
cd /home/wy/AutoRewardDrive
conda activate vlm-rl
python train.py --config 3 --total_timesteps 1000000
```

---

## 训练参数

```bash
# 修改训练步数
python train.py --config 3 --total_timesteps 2000000

# 评估模式
python eval.py --config 3 --eval_episodes 10
```

---

## 监控训练

### TensorBoard
```bash
conda activate vlm-rl
tensorboard --logdir=runs
```
访问：`http://localhost:6006`

---

## 常见问题

### CARLA 崩溃
```bash
pkill -9 CarlaUE4
DISPLAY= ./CarlaUE4.sh -opengl -quality-level=Low
```

### 端口占用
```bash
lsof -i :2000
```
