import os
from PIL import Image
import albumentations as A
import numpy as np


# 이미지가 저장되어 있는 디렉토리 경로
data_dir = "C:/train_dataset/train/typhoon"

# 어그멘테이션 결과가 저장될 디렉토리 경로
save_dir = "C:/train_dataset/train/typhoon"

# data_dir 디렉토리에 있는 모든 이미지 파일의 경로를 리스트에 저장
image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]

# 이미지 파일을 읽어들여 어그멘테이션을 수행하고 저장
for image_file in image_files:
    # 이미지 불러오기
    image = Image.open(image_file)
# 이미지 크기 가져오기
    image_width, image_height=image.size
    crop_width, crop_height = min(image_width, 256), min(image_height, 256)
    # 어그멘테이션 파이프라인 정의
    transform = A.Compose([
       A.RandomCrop(width=crop_width, height=crop_height),
       A.HorizontalFlip(p=0.5),
       A.RandomBrightnessContrast(p=0.2),
    ])

    # 이미지 어그멘테이션 수행 및 저장
    image_array = np.array(image)
    image_name = os.path.splitext(os.path.basename(image_file))[0]  # 이미지 파일 이름 추출

    # 저장 경로 수정
    save_folder_name = "aug"  # 원하는 폴더 이름을 지정하세요.
    save_path = os.path.join(save_dir, save_folder_name)
    os.makedirs(save_path, exist_ok=True)  # 지정한 이름의 폴더 생성

    # 10장의 이미지 생성
    for i in range(66):
        transformed = transform(image=image_array)
        transformed_image = transformed['image']
        transformed_image = np.array(transformed_image)
        
         # RGB로 변환
        transformed_image = Image.fromarray(transformed_image).convert('RGB')
        
        # 이미지 저장 (파일명을 지정하여 저장)
        output_filename = f"{image_name}_{i+1:03d}.jpg"  # 파일명을 "원본이름_001.jpg"부터 "원본이름_100.jpg" 형식으로 지정
        output_path = os.path.join(save_path, output_filename)
        transformed_image.save(output_path)