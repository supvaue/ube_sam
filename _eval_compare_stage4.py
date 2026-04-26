import csv, os

def parse_results(path):
    rows = {}
    with open(path) as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)
        next(reader)
        for row in reader:
            if not row or len(row) < 6:
                continue
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

out_dir = 'results/task1cocov2ft_stage4_eval_viz'
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, 'global_metrics_stage1234_compare.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['metric','stage1','stage2','stage3','stage4','delta_4_vs_3','delta_4_vs_2','delta_4_vs_1'])
    for m in ['J&F','J','F','Dice']:
        v1 = s1['global'][m]
        v2 = s2['global'][m]
        v3 = s3['global'][m]
        v4 = s4['global'][m]
        w.writerow([m, f'{v1:.2f}', f'{v2:.2f}', f'{v3:.2f}', f'{v4:.2f}',
                     f'{v4-v3:.2f}', f'{v4-v2:.2f}', f'{v4-v1:.2f}'])

objects = [k for k in s4 if k != 'global']

with open(os.path.join(out_dir, 'stage4_vs_stage3_per_object_delta.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['obj','stage3_jf','stage4_jf','delta_jf','delta_j','delta_f','delta_dice'])
    for obj in objects:
        jf3 = s3[obj]['J&F']; jf4 = s4[obj]['J&F']
        dj = s4[obj]['J'] - s3[obj]['J']
        df = s4[obj]['F'] - s3[obj]['F']
        dd = s4[obj]['Dice'] - s3[obj]['Dice']
        w.writerow([obj, f'{jf3:.2f}', f'{jf4:.2f}', f'{jf4-jf3:.2f}', f'{dj:.2f}', f'{df:.2f}', f'{dd:.2f}'])

with open(os.path.join(out_dir, 'stage4_vs_stage2_per_object_delta.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['obj','stage2_jf','stage4_jf','delta_jf','delta_j','delta_f','delta_dice'])
    for obj in objects:
        jf2 = s2[obj]['J&F']; jf4 = s4[obj]['J&F']
        dj = s4[obj]['J'] - s2[obj]['J']
        df = s4[obj]['F'] - s2[obj]['F']
        dd = s4[obj]['Dice'] - s2[obj]['Dice']
        w.writerow([obj, f'{jf2:.2f}', f'{jf4:.2f}', f'{jf4-jf2:.2f}', f'{dj:.2f}', f'{df:.2f}', f'{dd:.2f}'])

print("Done.")
