import os
import numpy as np
import pandas as pd
from torch.nn.modules.linear import Linear
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from PIL import Image
from torchvision.io import read_image, ImageReadMode
from skimage import io
from pydicom import dcmread
from keras.preprocessing import image
import tensorflow as tf
import torchvision.models as models

def load_dicom_file(file_path) :
    try :
        dicom_data = dcmread(file_path)
        print("파일 로드 성공")
        return dicom_data
    
    except Exception as e :
        print("파일 로드 실패")
        return None
    
def image_processing(image) :
    image = image.astype(np.uint8)
    
    if len(image.shape) == 2 :
        image = np.stack([image]*3, axis = -1)
        
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # DenseNet의 입력 크기
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    
    return image

def main() :
    file_path = input("DICOM 파일을 입력하세요 :")
    
    if not os.path.exists(file_path) :
        print("파일이 존재하지 않습니다")
        return
    
    dicom_data = load_dicom_file(file_path=file_path)
    image = dicom_data.pixel_array.astype(float)
    input_image = image_processing(image)
    
    model=models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
    model.classifier = Linear(in_features=1920, out_features=1, bias=True)
    
    try :
        model.load_state_dict(torch.load('best_model_age.pth', map_location=torch.device('cpu')))
        print("Load successfully")
    except Exception as e :
        print(f"Fail to load : {e}")
        return 
    
    
    model.eval()

    
    with torch.no_grad() :
        scores = model(input_image)
        preds = scores.item()
    
    print(f"안저 이미지를 통해 측정한 나이는 {preds:.2f} 입니다")
    
    
if __name__ == "__main__" :
    main()