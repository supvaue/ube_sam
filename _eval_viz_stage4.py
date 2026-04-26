import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv, os

out_dir = 'results/task1cocov2ft_stage4_eval_viz'

def parse_results(path):
    rows = {}
    with open(path) as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader); next(reader)
        for row in reader:
            if not row or len(row) < 6: continue
            obj = row[1].strip()
            jf = float(row[2].strip())
            j = float(row[3].strip())
            fv = float(row[4].strip())
            dice = float(row[5].strip())
            if obj == '':
                rows['global'] = {'J&F': jf, 'J': j, 'F': fv, 'Dice': dice}
            else:
                rows[obj] = {'J&F': jf, 'J': j, 'F': fv, 'Dice': dice}
    return rows

s1 = parse_results('results/task1cocov2ft_stage1_valid/results.csv')
s2 = parse_results('results/task1cocov2ft_stage2_valid/results.csv')
s3 = parse_results('results/task1cocov2ft_stage3_valid/results.csv')
s4 = parse_results('results/task1cocov2ft_stage4_valid/results.csv')

plt.rcParams['font.size'] = 12

# 1. Global metrics bar chart (4 stages)
fig, ax = plt.subplots(figsize=(10, 5))
metrics = ['J&F', 'J', 'F', 'Dice']
x = np.arange(len(metrics))
width = 0.18
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
for i, (stage, data, color) in enumerate([
    ('Stage1', s1, colors[0]),
    ('Stage2', s2, colors[1]),
    ('Stage3', s3, colors[2]),
    ('Stage4', s4, colors[3]),
]):
    vals = [data['global'][m] for m in metrics]
    ax.bar(x + i*width, vals, width, label=stage, color=color)
    for j, v in enumerate(vals):
        ax.text(x[j]+i*width, v+0.3, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x + 1.5*width)
ax.set_xticklabels(metrics)
ax.set_ylabel('Score')
ax.set_title('Global Metrics Comparison: Stage 1-4')
ax.legend()
ax.set_ylim(68, 80)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'global_metrics_bar.png'), dpi=150)
plt.close()

# 2. Per-object J&F grouped bar (stage2 vs stage3 vs stage4)
objects = [k for k in s4 if k != 'global']
fig, ax = plt.subplots(figsize=(16, 6))
x = np.arange(len(objects))
width = 0.25
for i, (label, data, color) in enumerate([
    ('Stage2', s2, '#55A868'),
    ('Stage3', s3, '#C44E52'),
    ('Stage4', s4, '#8172B2'),
]):
    vals = [data[obj]['J&F'] for obj in objects]
    ax.bar(x + i*width, vals, width, label=label, color=color)
ax.set_xticks(x + width)
ax.set_xticklabels([f'obj_{o}' for o in objects], rotation=45, ha='right')
ax.set_ylabel('J&F Score')
ax.set_title('Per-Object J&F: Stage2 vs Stage3 vs Stage4')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'per_object_jf_bar.png'), dpi=150)
plt.close()

# 3. Delta chart: Stage4 vs Stage3 per object
deltas_43 = []
for obj in objects:
    d = s4[obj]['J&F'] - s3[obj]['J&F']
    deltas_43.append(d)
fig, ax = plt.subplots(figsize=(14, 5))
colors_43 = ['#2ca02c' if d >= 0 else '#d62728' for d in deltas_43]
ax.bar(x, deltas_43, color=colors_43)
ax.set_xticks(x)
ax.set_xticklabels([f'obj_{o}' for o in objects], rotation=45, ha='right')
ax.set_ylabel('J&F Delta')
ax.set_title('Stage4 vs Stage3: Per-Object J&F Delta (green=improved, red=degraded)')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)
for i, d in enumerate(deltas_43):
    ax.text(i, d + (1 if d >= 0 else -1.5), f'{d:+.1f}', ha='center', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'stage4_vs_stage3_delta.png'), dpi=150)
plt.close()

# 4. Delta chart: Stage4 vs Stage2 per object
deltas_42 = []
for obj in objects:
    d = s4[obj]['J&F'] - s2[obj]['J&F']
    deltas_42.append(d)
fig, ax = plt.subplots(figsize=(14, 5))
colors_42 = ['#2ca02c' if d >= 0 else '#d62728' for d in deltas_42]
ax.bar(x, deltas_42, color=colors_42)
ax.set_xticks(x)
ax.set_xticklabels([f'obj_{o}' for o in objects], rotation=45, ha='right')
ax.set_ylabel('J&F Delta')
ax.set_title('Stage4 vs Stage2: Per-Object J&F Delta (green=improved, red=degraded)')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)
for i, d in enumerate(deltas_42):
    ax.text(i, d + (0.5 if d >= 0 else -1), f'{d:+.1f}', ha='center', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'stage4_vs_stage2_delta.png'), dpi=150)
plt.close()

# 5. obj_003 specific comparison
fig, ax = plt.subplots(figsize=(6, 4))
stages = ['Stage1', 'Stage2', 'Stage3', 'Stage4']
vals_003 = [s1['003']['J&F'], s2['003']['J&F'], s3['003']['J&F'], s4['003']['J&F']]
ax.plot(stages, vals_003, 'o-', color='#C44E52', linewidth=2, markersize=8)
for i, v in enumerate(vals_003):
    ax.text(i, v+1, f'{v:.1f}', ha='center', fontsize=10)
ax.set_ylabel('J&F Score')
ax.set_title('obj_003 (Vertebra) J&F Across Stages')
ax.set_ylim(15, 65)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'obj003_jf_across_stages.png'), dpi=150)
plt.close()

print("All visualizations saved to:", out_dir)
for f in sorted(os.listdir(out_dir)):
    if f.endswith('.png'):
        print(f"  {f}")
