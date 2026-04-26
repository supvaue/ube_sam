import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import os

BASE = '/home/xu/Code/ReSurgSAM2/ReSurgSAM2-main'
IMG_DIR = os.path.join(BASE, 'data/Task1CocoV2FT/valid/JPEGImages/seq_1')
GT_DIR = os.path.join(BASE, 'data/Task1CocoV2FT/valid/Annotations/seq_1')
S3_DIR = os.path.join(BASE, 'results/task1cocov2ft_stage3_valid/seq_1')
S4_DIR = os.path.join(BASE, 'results/task1cocov2ft_stage4_valid/seq_1')
OUT_DIR = os.path.join(BASE, 'results/task1cocov2ft_stage4_eval_viz/sample_side_by_side')
os.makedirs(OUT_DIR, exist_ok=True)

OBJ_NAMES = {
    '003': 'vertebra', '002': 'descending_root', '015': 'dura_sac',
    '021': 'flavum_ligament', '019': '0deg_ablator', '007': 'nucleus_pulposus',
}

COLORS = {'gt': [0, 255, 0], 's3': [255, 165, 0], 's4': [0, 120, 255]}

def overlay_mask(img, mask, color, alpha=0.4):
    out = img.copy()
    mask_bool = mask > 127
    for c in range(3):
        out[:,:,c] = np.where(mask_bool,
                               (1-alpha)*out[:,:,c] + alpha*color[c],
                               out[:,:,c])
    return out.astype(np.uint8)

def get_common_frames(obj_id):
    s3_frames = set(os.listdir(os.path.join(S3_DIR, obj_id))) if os.path.isdir(os.path.join(S3_DIR, obj_id)) else set()
    s4_frames = set(os.listdir(os.path.join(S4_DIR, obj_id))) if os.path.isdir(os.path.join(S4_DIR, obj_id)) else set()
    return sorted(s3_frames & s4_frames)

for obj_id, obj_name in OBJ_NAMES.items():
    frames = get_common_frames(obj_id)
    if not frames:
        continue
    n = len(frames)
    indices = [0, n//4, n//2, 3*n//4, n-1]
    indices = list(dict.fromkeys([max(0, min(i, n-1)) for i in indices]))
    selected_frames = [frames[i] for i in indices]

    fig, axes = plt.subplots(len(selected_frames), 4, figsize=(16, 4*len(selected_frames)))
    if len(selected_frames) == 1:
        axes = axes[np.newaxis, :]

    for row, frame_name in enumerate(selected_frames):
        img_path = os.path.join(IMG_DIR, frame_name)
        gt_path = os.path.join(GT_DIR, frame_name)
        s3_path = os.path.join(S3_DIR, obj_id, frame_name)
        s4_path = os.path.join(S4_DIR, obj_id, frame_name)

        if not all(os.path.exists(p) for p in [img_path, gt_path, s3_path, s4_path]):
            continue

        img = np.array(Image.open(img_path).convert('RGB'))
        gt_mask = np.array(Image.open(gt_path).convert('L'))
        s3_mask = np.array(Image.open(s3_path).convert('L'))
        s4_mask = np.array(Image.open(s4_path).convert('L'))

        gt_obj = ((gt_mask.astype(int)) == int(obj_id)).astype(np.uint8) * 255

        img_gt = overlay_mask(img, gt_obj, COLORS['gt'], alpha=0.35)
        img_s3 = overlay_mask(img, s3_mask, COLORS['s3'], alpha=0.35)
        img_s4 = overlay_mask(img, s4_mask, COLORS['s4'], alpha=0.35)

        axes[row,0].imshow(img)
        axes[row,0].set_title(f'Original f{frame_name}', fontsize=9)
        axes[row,0].axis('off')

        axes[row,1].imshow(img_gt)
        axes[row,1].set_title('GT (green)', fontsize=9)
        axes[row,1].axis('off')

        axes[row,2].imshow(img_s3)
        axes[row,2].set_title('Stage3 (orange)', fontsize=9)
        axes[row,2].axis('off')

        axes[row,3].imshow(img_s4)
        axes[row,3].set_title('Stage4 (blue)', fontsize=9)
        axes[row,3].axis('off')

    gt_patch = mpatches.Patch(color='green', label='GT', alpha=0.5)
    s3_patch = mpatches.Patch(color='orange', label='Stage3', alpha=0.5)
    s4_patch = mpatches.Patch(color='blue', label='Stage4', alpha=0.5)
    fig.legend(handles=[gt_patch, s3_patch, s4_patch], loc='upper right', fontsize=11)

    plt.suptitle(f'obj_{obj_id} ({obj_name}): Stage3 vs Stage4', fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'obj{obj_id}_{obj_name}_compare.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# Also generate a combined overlay: GT+Stage3+Stage4 on same image for obj_003
obj_id = '003'
frames = get_common_frames(obj_id)
if frames:
    n = len(frames)
    indices = [0, n//4, n//2, 3*n//4, n-1]
    indices = list(dict.fromkeys([max(0, min(i, n-1)) for i in indices]))
    selected_frames = [frames[i] for i in indices]

    fig, axes = plt.subplots(1, len(selected_frames), figsize=(5*len(selected_frames), 5))
    if len(selected_frames) == 1:
        axes = [axes]

    for col, frame_name in enumerate(selected_frames):
        img_path = os.path.join(IMG_DIR, frame_name)
        gt_path = os.path.join(GT_DIR, frame_name)
        s3_path = os.path.join(S3_DIR, obj_id, frame_name)
        s4_path = os.path.join(S4_DIR, obj_id, frame_name)

        if not all(os.path.exists(p) for p in [img_path, gt_path, s3_path, s4_path]):
            continue

        img = np.array(Image.open(img_path).convert('RGB'))
        gt_mask = np.array(Image.open(gt_path).convert('L'))
        s3_mask = np.array(Image.open(s3_path).convert('L'))
        s4_mask = np.array(Image.open(s4_path).convert('L'))

        gt_obj = ((gt_mask.astype(int)) == int(obj_id)).astype(np.uint8) * 255

        combined = img.copy()
        gt_bool = gt_obj > 127
        s3_bool = s3_mask > 127
        s4_bool = s4_mask > 127

        combined[gt_bool & s4_bool & ~s3_bool] = [0, 200, 200]
        combined[gt_bool & s3_bool & ~s4_bool] = [200, 200, 0]
        combined[gt_bool & s3_bool & s4_bool] = [0, 255, 0]
        combined[gt_bool & ~s3_bool & ~s4_bool] = [255, 0, 0]
        combined[~gt_bool & s3_bool & s4_bool] = [180, 0, 180]
        combined[~gt_bool & s3_bool & ~s4_bool] = [255, 165, 0]
        combined[~gt_bool & ~s3_bool & s4_bool] = [0, 120, 255]

        axes[col].imshow(combined)
        axes[col].set_title(f'f{frame_name}', fontsize=9)
        axes[col].axis('off')

    patches = [
        mpatches.Patch(color='green', label='GT+3+4 (all correct)'),
        mpatches.Patch(color='cyan', label='GT+4 only (S4 fixed)'),
        mpatches.Patch(color='yellow', label='GT+3 only (S3 only)'),
        mpatches.Patch(color='red', label='GT only (both miss)'),
        mpatches.Patch(color='orange', label='S3 FP only'),
        mpatches.Patch(color='blue', label='S4 FP only'),
        mpatches.Patch(color='purple', label='S3+S4 FP'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=8)
    plt.suptitle('obj_003 (vertebra): Error Analysis Overlay', fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'obj003_vertebra_error_analysis.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

print("\nAll visualizations done!")
