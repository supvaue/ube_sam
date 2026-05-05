# YOLO Mask Adapter for ReSurgSAM2

版本：v0.1，2026-05-03

## 目标

把 YOLO 提供的髓核钳 mask 转换为 ReSurgSAM2 可使用的条件信息。这里不是简单满足接口 shape，而是希望在功能上替代甚至超过 text prompt：

- text prompt 提供类别语义；
- YOLO mask 同时提供类别、位置、轮廓、边界、置信度和时序可靠性；
- 因此 adapter 输出必须绑定当前图像特征，而不能只编码 binary mask 形状。

## 当前文件

- `mask_token_encoder.py`：输出 `mask_emb_sentence [B,N,256]`、`mask_emb_cls [B,1,256]` 和 dense prompt mask。
- `reliability.py`：计算 YOLO mask 可靠分数与 10 维 geometry/reliability 向量。
- `predictor_mixin.py`：提供 `add_new_yolo_mask`，把 YOLO mask token 写入 ReSurgSAM2 原 referring/text 条件入口。
- `yolo_mask_video_predictor.py`：组合 `YOLOMaskPromptMixin` 与 `SAM2VideoPredictor`，作为可通过 Hydra `_target_` 构建的 predictor 子类。
- `smoke_test.py`：不依赖 ReSurgSAM2 主模型，检查 adapter shape、数值范围和梯度。
- `predictor_smoke_test.py`：构建真实 ReSurgSAM2 predictor，在短视频/帧目录上验证 YOLO mask condition token 数据流。
- `build_forceps_manifest.py`：从 COCO `instances_default.json` 中直接筛选 `category_id=12` 髓核钳帧，按 `frame_XXXXXX` 排序，生成训练/评估 manifest。
- `build_forceps_mask_cache.py`：基于 manifest 生成 `GT mask + YOLO mask` 缓存，避免训练时重复运行 YOLO。
- `analyze_forceps_mask_cache.py`：统计 YOLO mask 与 GT mask 的 IoU、Dice、Precision、Recall，评估训练输入质量。
- `__init__.py`：模块导出。

## 源码兼容修复

`Code/ReSurgSAM2/sam2/modeling/sam2_base.py` 中 `_use_mask_as_output` 原本按旧接口解包 `_forward_sam_heads` 的 9 个返回值，但当前源码 `_forward_sam_heads` 实际返回 7 个值。已修正为 7 值解包：

```text
_, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(...)
```

这个修复只影响 `use_mask_input_as_output_without_sam=true` 的对照路径，用于验证 YOLO mask 直接初始化输出/memory 的上限；正式 dense prompt 消融仍应关闭该选项，让 mask 经过 prompt encoder + mask decoder。

## 当前环境状态

本机当前状态：

- `sam2` 可以导入。
- 已安装 `einops`、`gdown`、`transformers==4.44.2`、`mamba_ssm==2.2.4`、`ninja`。
- `mamba_ssm` 的 CUDA forward 已通过 GPU smoke test。
- `Code/ReSurgSAM2/checkpoints/sam2.1_hiera_small.pt` 已下载，大小约 176M，`torch.load` 顶层字段为 `model`。
- ReSurgSAM2 作者 Google Drive 权重已下载为 `Code/ReSurgSAM2/checkpoints/resurgsam2_checkpoints_download`，该文件是 zip。
- zip 已解压到 `Code/ReSurgSAM2/checkpoints/resurgsam2_downloaded/`，包含：
  - `sam2.1_hiera_s_ref17.pth`
  - `sam2.1_hiera_s_ref18.pth`
- 已在 `checkpoints/` 顶层建立同名软链接，便于按项目默认路径加载。

作者权重正确加载方式：

```bash
python -c "from sam2.build_sam import build_sam2_video_predictor; p=build_sam2_video_predictor(config_file='configs/sam2.1/sam2.1_hiera_s_rvos.yaml', ckpt_path='checkpoints/sam2.1_hiera_s_ref17.pth', device='cuda:0', strict_loading=False, apply_long_term_memory=True, hydra_overrides_extra=['++scratch.use_sp_bimamba=true','++scratch.use_dwconv=true']); print(type(p).__name__, p.hidden_dim, p.image_size, p.use_sp_bimamba, p.use_dwconv)"
```

结果：

```text
SAM2VideoPredictor 256 512 True True
```

注意：如果不加 `use_sp_bimamba/use_dwconv` override，CSTMamba/Mamba 参数会出现大量 missing/unexpected key。加上 override 后，核心 Mamba 参数可对齐；当前仍有 `sam_prompt_encoder.language_fusion_mlp.*` 作为 unexpected key，这说明作者 checkpoint 中包含一个当前推理实例没有接收的 language fusion MLP。后续做质量测试时，应记录这是“作者权重 + 当前源码/推理配置”的状态，而不是完全无差异的论文原版。

## v0.1 设计

`MaskTokenEncoder` 至少产生三类功能 token：

1. `masked_visual_tokens`：mask 内当前图像特征池化，表达目标外观。
2. `boundary_visual_tokens`：mask 边界环带特征池化，服务边缘跳变修正。
3. `reliability_geometry_tokens`：置信度、面积、形状稳定、边缘平滑、孔洞、类别稳定性。

这些 token 会替代 ReSurgSAM2 原本的 text token，进入 CSTMamba 的 `text_embeddings`/未来的 `condition_embeddings` 入口。

## 已完成 smoke test

### Adapter 独立 smoke test

命令：

```bash
python Code/ReSurgSAM2/yolo_mask_adapter/smoke_test.py
```

结果：

- `reliability`: `[0.9500, 0.7775]`
- `geometry_shape`: `(2, 10)`
- `sentence_shape`: `(2, 8, 256)`
- `cls_shape`: `(2, 1, 256)`
- `dense_prompt_shape`: `(2, 1, 128, 128)`
- `grad_ok`: `True`

### Predictor 级 smoke test

先从真实视频抽取两帧 JPEG，避免 mp4 全量解码导致 smoke test 时间不可控：

```bash
mkdir -p /tmp/resurgsam2_smoke_frames
ffmpeg -y -i Datasets/标注片段.mp4 -vf fps=1 -frames:v 2 /tmp/resurgsam2_smoke_frames/%05d.jpg
```

然后运行：

```bash
cd Code/ReSurgSAM2
python yolo_mask_adapter/predictor_smoke_test.py --video /tmp/resurgsam2_smoke_frames --device cuda:0 --max-frames 2 --frame-interval 1
```

结果摘要：

```text
add_result.accepted=True
reliability=0.8338
sentence_shape=(1, 8, 256)
cls_shape=(1, 1, 256)
num_frames_loaded=2
outputs:
  frame 0 mask_shape=(1, 1, 1080, 1920), mask_mean=0.0
  frame 1 mask_shape=(1, 1, 1080, 1920), mask_mean=0.0
```

结论：

- 新增 `YOLOMaskSAM2VideoPredictor` 能通过 Hydra `_target_` 构建并加载 ReSurgSAM2 ref17 权重。
- `add_new_yolo_mask` 能把 YOLO mask condition token 写入原 referring/text 条件入口。
- 两帧传播链路已经跑通。
- 当前输出为空 mask；这是使用中心 dummy mask 且 adapter 未训练时的预期风险，不能作为质量指标，只能说明工程数据流可执行。
- 运行中出现 `cannot import name '_C' from 'sam2'`，因此跳过了 fill-hole 后处理。该问题不阻断前向，但后续正式评估前应考虑安装/编译 SAM2 CUDA 扩展或关闭相关后处理。

### 真实 YOLO 髓核钳 mask smoke test

为了避免错误帧，脚本增加了 `--yolo` 与 `--yolo-search-frames`，会在帧目录中搜索第一张包含目标类别的图像，并把 mask 注入到对应的 `frame_idx`。

第一段视频 `标注片段.mp4`：

- 每 30s 抽 1 帧，共 30 帧，覆盖前 14 分 30 秒。
- YOLO 未检测到 `target_yolo_cls=11` 髓核钳。
- 检出的类别主要包括 0 黄韧带、8 咬钳、2 椎骨、3 硬膜外脂肪、21 肌肉等。

第二段视频 `新-第2例.mp4`：

```bash
mkdir -p /tmp/resurgsam2_forceps_search_frames_exp2
ffmpeg -y -i Datasets/新-第2例.mp4 -vf fps=1/20 -frames:v 40 /tmp/resurgsam2_forceps_search_frames_exp2/%05d.jpg

cd Code/ReSurgSAM2
python yolo_mask_adapter/predictor_smoke_test.py \
  --video /tmp/resurgsam2_forceps_search_frames_exp2 \
  --device cuda:0 \
  --max-frames 2 \
  --frame-interval 1 \
  --yolo ../ultralytics-main/runs/segment/runs/joint_train_v2/weights/best.pt \
  --target-yolo-cls 11 \
  --yolo-conf 0.20 \
  --yolo-iou 0.70 \
  --yolo-imgsz 640 \
  --yolo-search-frames 40
```

结果摘要：

```text
yolo_meta:
  cls_name=髓核钳
  confidence=0.9559
  mask_area=0.1124
  source=/tmp/resurgsam2_forceps_search_frames_exp2/00012.jpg
  source_index=11
  checked_frames=12
add_result:
  accepted=True
  reliability=0.8346
  sentence_shape=(1, 8, 256)
  cls_shape=(1, 1, 256)
outputs:
  frame 11 mask_shape=(1, 1, 1080, 1920), mask_mean=0.0
  frame 12 mask_shape=(1, 1, 1080, 1920), mask_mean=0.0
```

结论：

- 真实 YOLO 髓核钳 mask 可以被检出并注入，且可靠分数合理。
- 但当前 condition-token-only 的未训练 adapter 输出仍为空 mask。
- 这支持此前判断：YOLO mask token 不能只靠 shape 对齐直接替代 CLIP/text token 分布，必须训练 `MaskTokenEncoder`，并优先尝试加入 dense mask prompt 路径作为空间强先验。
- 目前 `dense_prompt_mask` 已由 `MaskTokenEncoder` 输出，但尚未真正接入 ReSurgSAM2 `track_step` 的 `mask_inputs`/dense prompt 路径；这是下一步工程重点。

### Dense Prompt / Condition Token 消融 smoke test

`predictor_smoke_test.py` 已增加：

- `--prompt-mode condition`
- `--prompt-mode dense`
- `--prompt-mode dense_condition`
- `--use-mask-as-output`

测试对象仍为第二段视频抽样帧中第 12 张检出的髓核钳，YOLO 置信度 `0.9559`，mask 面积 `0.1124`。

结果：

```text
condition only:
  frame 11 mask_mean=0.0
  frame 12 mask_mean=0.0

dense prompt through prompt encoder + mask decoder:
  dense_result mask_mean=0.0
  frame 11 mask_mean=0.0
  frame 12 mask_mean=0.0

dense + condition token:
  dense_result mask_mean=0.0
  frame 11 mask_mean=0.0
  frame 12 mask_mean=0.0

direct mask-as-output baseline:
  dense_result mask_mean=0.11249
  frame 11 mask_mean=0.11249
  frame 12 mask_mean=0.0
```

判断：

- YOLO mask、帧索引、类别筛选和输入数据流是正确的；direct baseline 的 frame 11 面积与 YOLO mask 面积一致。
- 但不训练的 ReSurgSAM2 prompt encoder + mask decoder 不能直接利用 YOLO dense mask prompt。
- 不训练的 condition token 也不能直接替代 CLIP/text token。
- direct mask-as-output 只保证初始帧有输出，下一帧仍为空，说明仅写入一次 memory 还不够；需要训练或设计稳定的单目标连续更新机制。
- 下一步应从标签筛选髓核钳帧，构建训练/评估 manifest，训练轻量 mask condition 模块，而不是继续依赖视频搜索 smoke test。

### 标签驱动的髓核钳 Manifest

命令：

```bash
python Code/ReSurgSAM2/yolo_mask_adapter/build_forceps_manifest.py \
  --output Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_manifest.json
```

结果：

```text
exp1_cu_full: 1272 images, 139 forceps images, 143 forceps annotations
exp2_cu_full: 998 images, 131 forceps images, 145 forceps annotations
exp1_cu: 1272 images, 139 forceps images, 143 forceps annotations
exp2_cu: 998 images, 131 forceps images, 145 forceps annotations
missing_images=0 for all datasets
```

输出文件：

```text
Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_manifest.json
```

后续训练/评估应以该 manifest 为准，不再从视频中搜索髓核钳帧。

### Mask Cache Smoke Test

新增：

```text
Code/ReSurgSAM2/yolo_mask_adapter/build_forceps_mask_cache.py
```

功能：

- 读取 `forceps_manifest.json`。
- 将 COCO polygon segmentation 栅格化为 GT mask。
- 运行 YOLO joint_train_v2，提取 `target_yolo_cls=11` 髓核钳预测 mask。
- 保存 `.npz`，包含 `gt_mask`、`yolo_mask`、图像路径、帧号和 YOLO 置信度。

smoke 命令：

```bash
python Code/ReSurgSAM2/yolo_mask_adapter/build_forceps_mask_cache.py \
  --datasets exp2_cu_full \
  --max-items 5 \
  --device 0 \
  --output-dir Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_mask_cache_smoke
```

结果：

```text
count=5
with_yolo_mask=5
mean_gt_area=0.2654
mean_yolo_area=0.2506
```

输出：

```text
Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_mask_cache_smoke/summary.json
```

这说明标签驱动样本、GT mask 栅格化和 YOLO mask 缓存链路已经跑通。

### Full Mask Cache 与质量分析

full cache 命令：

```bash
python Code/ReSurgSAM2/yolo_mask_adapter/build_forceps_mask_cache.py \
  --datasets exp1_cu_full exp2_cu_full \
  --device 0 \
  --output-dir Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_mask_cache_full
```

结果：

```text
count=270
with_yolo_mask=267
mean_gt_area=0.1401
mean_yolo_area=0.1276
```

质量分析命令：

```bash
python Code/ReSurgSAM2/yolo_mask_adapter/analyze_forceps_mask_cache.py \
  --cache-dir Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_mask_cache_full
```

总体结果：

```text
count=270
with_pred=267
mean_iou=0.8367
mean_dice=0.8985
mean_precision=0.9383
mean_recall=0.8727
mean_yolo_conf=0.9154
```

分数据集：

```text
exp1_cu_full:
  count=139
  with_pred=138
  mean_iou=0.8725
  mean_dice=0.9267

exp2_cu_full:
  count=131
  with_pred=129
  mean_iou=0.7987
  mean_dice=0.8686
```

输出：

```text
Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_mask_cache_full/summary.json
Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_mask_cache_full/quality_summary.json
```

判断：

- YOLO mask 本身已经是强先验，整体 IoU 约 0.837。
- exp2 明显低于 exp1，应优先作为泛化验证集。
- 训练目标不应是“从零分割髓核钳”，而是学习如何利用 YOLO mask 强先验做 refinement、稳定 memory 初始化和连续传播。

### Stage A Frozen Feature Refinement

新增：

```text
build_resurg_feature_cache.py
train_forceps_refine_adapter.py
```

Stage A 设计：

- 冻结 ReSurgSAM2 Hiera-S 图像编码器。
- 缓存最后一级低分辨率视觉特征。
- 训练一个极小 decoder，输入为 `frozen image feature + YOLO mask`。
- 输出 residual mask logits，用于修正 YOLO mask。

feature cache：

```bash
python yolo_mask_adapter/build_resurg_feature_cache.py \
  --mask-cache-dir yolo_mask_adapter/results/forceps_mask_cache_full \
  --output-dir yolo_mask_adapter/results/resurg_feature_cache_full \
  --device cuda:0 \
  --progress-every 25
```

结果：

```text
count=270
output_dir=yolo_mask_adapter/results/resurg_feature_cache_full
```

训练结果：

```text
exp1 -> exp2, 30 epochs:
  YOLO val IoU baseline=0.7988019
  best val IoU=0.7990386 at epoch 4
  last val IoU=0.7782662
  train IoU improved 0.8720 -> 0.9312

exp2 -> exp1, 20 epochs:
  YOLO val IoU baseline=0.8720328
  best val IoU=0.8720328 at epoch 1
  last val IoU=0.6403755
  train IoU improved 0.7988 -> 0.8446
```

判断：

- 最简单的 per-frame refinement decoder 能拟合训练视频，但跨视频泛化不稳定。
- 它没有显著超过 YOLO mask，反而容易破坏本来已经很好的 YOLO 结果。
- 后续不应继续无条件 refinement；应该引入可靠性门控，只在低可靠/边缘跳变/类别跳变时修正。
- 真正值得推进的是：
  - reliability-gated update；
  - 只修正低 IoU/漏检/边缘异常样本；
  - temporal/memory 一致性，而非单帧强行改写所有 YOLO mask。

## 下一步

1. 接入 dense mask prompt 路径，做 `dense only` 和 `dense + condition token` 消融。
2. 训练轻量 adapter：
   - 先冻结 ReSurgSAM2 主体，只训练 `MaskTokenEncoder`。
   - 若 condition token 仍不能收敛，再低学习率解冻 `cross_modal_fusion`。
3. 后续做三组消融：
   - dense mask prompt only；
   - condition token only；
   - dense mask prompt + condition token。
