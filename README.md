## PLN-K

PLN-K 是一个基于 PyTorch 的PLN复现项目（VOC2007/2012），包含：
- 单机训练：`train.py`
- 多卡 DDP 训练：`train_ddp.py`
- 推理 + NMS + 可视化 + mAP 评测：`inference.py`

### 目录结构

- `datasets/`: VOC 数据集读取与增强
- `models/`: 模型定义
- `utils/`: 解码、loss、NMS、target 构建等工具
- `train.py`: 单卡训练入口
- `train_ddp.py`: 多卡 DDP 训练入口
- `inference.py`: 推理入口
- `debug_out/`: 调试输出（可视化图片、统计报告等）
- `model-pth-and-log/`: 训练产物（checkpoint/log）目录，包括模型权重

### 环境配置
推荐用 conda 新建环境：

```bash
conda create -n pln-k python=3.8 -y
conda activate pln-k
pip install -r requirements.txt
```

- `requirements.txt` 内包含 `--index-url`，会拉取 **CUDA 11.8 的 PyTorch wheels**。如果想 CPU-only 或其它 CUDA 版本，请自行调整该行与 torch 版本。

### 数据准备（VOC）

默认期望 VOC 目录结构为 VOCdevkit 标准：
- `VOC2007/VOCdevkit/VOC2007/Annotations`
- `VOC2007/VOCdevkit/VOC2007/JPEGImages`
- `VOC2007/VOCdevkit/VOC2007/ImageSets/Main/{train,val,trainval,test}.txt`

- VOC2012应与VOC2007目录结构相同，两个在同一目录下即可

### 训练（单机）

最小训练示例（VOC，默认会混合 VOC2007+VOC2012，除非你关掉）：

```bash
python train.py \
  --dataset_type voc \
  --voc2007_root "/path/to/VOC2007/VOCdevkit/VOC2007" \
  --voc2012_root "/path/to/VOC2012/VOCdevkit/VOC2012" \
  --split trainval \
  --image_size 448 \
  --batch_size 64 \
  --device cuda
```

启用 **高斯 link targets（Lx/Ly）**（可选）：

```bash
python train.py \
  --dataset_type voc \
  --voc2007_root "/path/to/VOC2007/VOCdevkit/VOC2007" \
  --voc2012_root "/path/to/VOC2012/VOCdevkit/VOC2012" \
  --split trainval \
  --image_size 448 \
  --batch_size 64 \
  --device cuda \
  --use_gaussian_link_targets \
  --gaussian_link_radius 2 \
  --gaussian_link_sigma 0.7
```

### 训练（多卡 DDP）

示例（4 卡）：

```bash
torchrun --nproc_per_node=4 train_ddp.py \
  --dataset_type voc \
  --voc2007_root "/path/to/VOC2007/VOCdevkit/VOC2007" \
  --voc2012_root "/path/to/VOC2012/VOCdevkit/VOC2012" \
  --split trainval \
  --image_size 448 \
  --batch_size 64 \
  --backend nccl \
  --master_port 29501
```

DDP 下同样支持高斯 link targets：

```bash
torchrun --nproc_per_node=4 train_ddp.py \
  --dataset_type voc \
  --voc2007_root "/path/to/VOC2007/VOCdevkit/VOC2007" \
  --voc2012_root "/path/to/VOC2012/VOCdevkit/VOC2012" \
  --split trainval \
  --image_size 448 \
  --batch_size 64 \
  --use_gaussian_link_targets \
  --gaussian_link_radius 2 \
  --gaussian_link_sigma 0.7
```

### 推理（Inference）+ mAP 评测

推理脚本默认会计算并保存：
- `mAP@0.5`
- `mAP@0.75`
- `mAP@[0.5:0.95]`

并输出到 `--output_dir`：
- `predictions_{split}.json`
- `map_results_{split}.json`

示例（test 全集，batch 可 >1）：

```bash
python inference.py \
  --voc2007_root "/path/to/VOC2007/VOCdevkit/VOC2007" \
  --split test \
  --batch_size 16 \
  --max_batches 0 \
  --checkpoint "/path/to/checkpoints/pln_step_xxxxxx.pth" \
  --conf_thres 0 \
  --output_dir "/path/to/output/infer_test" \
  --post_nms_score_ratio_filter \
  --post_nms_score_ratio_w 0.35
```

保存每张图的可视化（会生成较多图片）：

```bash
python inference.py \
  --voc2007_root "/path/to/VOC2007/VOCdevkit/VOC2007" \
  --split test \
  --batch_size 1 \
  --max_batches 50 \
  --checkpoint "/path/to/checkpoints/pln_step_xxxxxx.pth" \
  --conf_thres 0 \
  --output_dir "/path/to/output/infer_test_viz" \
  --save_visualize \
  --post_nms_score_ratio_filter \
  --post_nms_score_ratio_w 0.35
```

关闭评测（只跑推理输出）：

```bash
python inference.py ... --no-eval_map
```

