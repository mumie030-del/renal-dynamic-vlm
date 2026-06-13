import os
import shutil
import glob

TARGET_ROOT = '../data3'
os.makedirs(TARGET_ROOT, exist_ok=True)

# ========== 1. datasets_12 (52 cases) ==========
SRC1 = '../datasets_12'
valid_cases = []
for d in sorted(os.listdir(SRC1)):
    full = os.path.join(SRC1, d)
    if not os.path.isdir(full):
        continue
    img_dir = os.path.join(full, 'images_fused_26')
    label_dir = os.path.join(full, 'labels')
    if os.path.isdir(img_dir) and os.path.isdir(label_dir):
        valid_cases.append(d)

print(f"datasets_12: {len(valid_cases)} valid cases")

for case in valid_cases:
    src_img_dir = os.path.join(SRC1, case, 'images_fused_26')
    src_label = glob.glob(os.path.join(SRC1, case, 'labels', '*.json'))[0]

    dst_dir = os.path.join(TARGET_ROOT, case)
    os.makedirs(dst_dir, exist_ok=True)

    # Copy & rename images: fused_01_*.jpg -> 0.jpg ... fused_26_*.jpg -> 25.jpg
    for idx in range(26):
        src = os.path.join(src_img_dir, f'fused_{idx+1:02d}_*.jpg')
        matches = glob.glob(src)
        if matches:
            shutil.copy2(matches[0], os.path.join(dst_dir, f'{idx}.jpg'))

    # Copy label JSON
    dst_label = os.path.join(dst_dir, 'mask.json')
    if not os.path.exists(dst_label):
        shutil.copy2(src_label, dst_label)

# ========== 2. dataset_new (50 cases) ==========
SRC2 = '../dataset_new'
valid_cases2 = []
for d in sorted(os.listdir(SRC2), key=lambda x: int(x) if x.isdigit() else 9999):
    full = os.path.join(SRC2, d)
    if not os.path.isdir(full) or not d.isdigit():
        continue
    img_dir = os.path.join(full, 'images_fused_26')
    label_dir = os.path.join(full, 'labels')
    if os.path.isdir(img_dir) and os.path.isdir(label_dir):
        valid_cases2.append(d)

print(f"dataset_new: {len(valid_cases2)} valid cases")

for case in valid_cases2:
    src_img_dir = os.path.join(SRC2, case, 'images_fused_26')
    src_label = glob.glob(os.path.join(SRC2, case, 'labels', '*.json'))[0]

    # Use prefix 'n' to avoid name collision with datasets_12
    dst_name = f'n{case}'
    dst_dir = os.path.join(TARGET_ROOT, dst_name)
    os.makedirs(dst_dir, exist_ok=True)

    for idx in range(26):
        src = os.path.join(src_img_dir, f'fused_{idx+1:02d}_*.jpg')
        matches = glob.glob(src)
        if matches:
            shutil.copy2(matches[0], os.path.join(dst_dir, f'{idx}.jpg'))

    dst_label = os.path.join(dst_dir, 'mask.json')
    if not os.path.exists(dst_label):
        shutil.copy2(src_label, dst_label)

print(f"\nTotal cases prepared: {len(valid_cases) + len(valid_cases2)}")
print(f"Target directory: {os.path.abspath(TARGET_ROOT)}")
