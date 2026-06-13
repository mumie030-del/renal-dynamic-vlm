import os, shutil, glob

SRC = "/root/autodl-tmp/LLM/dataset_new"
DST = "/root/autodl-tmp/LLM/test/fair_compare/train_data"
os.makedirs(DST, exist_ok=True)

valid = []
for d in sorted(os.listdir(SRC), key=lambda x: int(x) if x.isdigit() else 9999):
    full = os.path.join(SRC, d)
    if not os.path.isdir(full) or not d.isdigit():
        continue
    img_dir = os.path.join(full, "images_fused_26")
    label_dir = os.path.join(full, "labels")
    if os.path.isdir(img_dir) and os.path.isdir(label_dir):
        valid.append(d)

print(f"dataset_new: {len(valid)} valid cases")

for case in valid:
    src_img_dir = os.path.join(SRC, case, "images_fused_26")
    src_label = glob.glob(os.path.join(SRC, case, "labels", "*.json"))[0]

    dst_dir = os.path.join(DST, f"n{case}")
    os.makedirs(dst_dir, exist_ok=True)

    for idx in range(26):
        src = os.path.join(src_img_dir, f"fused_{idx+1:02d}_*.jpg")
        matches = glob.glob(src)
        if matches:
            shutil.copy2(matches[0], os.path.join(dst_dir, f"{idx}.jpg"))

    dst_label = os.path.join(dst_dir, "mask.json")
    if not os.path.exists(dst_label):
        shutil.copy2(src_label, dst_label)

print(f"Done. Training data at {DST}")
