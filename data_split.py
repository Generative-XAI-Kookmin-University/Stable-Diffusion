import os
import shutil
import random

# 원본 이미지 폴더
source_folder = "/root/data/celeba_hq_256"

# train / val 저장 폴더
train_folder = "train"
val_folder = "val"

# 폴더 생성
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# 이미지 랜덤 섞기
random.shuffle(image_files)

# 8:2 비율로 분할
split_idx = int(len(image_files) * 0.8)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# 파일 복사 함수 (move → copy로 변경)
def copy_files(files, destination_folder):
    for file in files:
        src_path = os.path.join(source_folder, file)
        dst_path = os.path.join(destination_folder, file)
        shutil.copy2(src_path, dst_path)  # 파일 복사

# 파일 복사 실행
copy_files(train_files, train_folder)
copy_files(val_files, val_folder)

print(f"Total Images: {len(image_files)}")
print(f"Train: {len(train_files)}, Val: {len(val_files)}")
print("Dataset split completed (Original files kept)!")
