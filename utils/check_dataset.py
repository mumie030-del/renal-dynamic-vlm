import os
import re

# 查看dataset目录结构，建立病例ID和文件夹的映射关系
dataset_dir = "/root/autodl-tmp/LLM/dataset"
subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('results')]
subdirs.sort()

print("病例文件夹列表:")
for d in subdirs[:30]:
    print(f"  {d}")

# 检查混合型文件夹名称规律
mixed_dirs = [d for d in subdirs if d.startswith('混合_')]
print(f"\n混合型文件夹: {mixed_dirs}")
