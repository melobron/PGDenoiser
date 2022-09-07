import torch
import torchvision
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import Noisy
import matplotlib.pyplot as plt
from utils import *
import cv2


data_dir = './experiments/exp1/results/EP_0001th_image'
noisy_path = os.path.join(data_dir, 'noisy.png')
prediction_path = os.path.join(data_dir, 'prediction.png')
noisy = cv2.imread(noisy_path)
prediction = cv2.imread(prediction_path)

print(prediction - noisy)
print(prediction)



