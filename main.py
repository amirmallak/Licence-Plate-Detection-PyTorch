import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as nn_func
import torchvision.transforms as transforms

from cv2 import cv2
from AI_Model import build_AI_model
from PIL import Image
from torch.optim import Adam
from typing import Dict, List
from pandas import set_option
from torchsummary import summary
from torchvision.models import vgg16
from torch import nn, Tensor, sigmoid

from data_exploring import exploring
from kaggle_data_loading import kaggle_data_loading
from data_preprocessing import pre_processing
from data_filtering import filtering
from loading_data import data_loading
from torch.nn import Sequential, Linear, MaxPool2d, Dropout
from torch.utils.data import DataLoader, random_split, Dataset
from license_plate_detection import license_plate_detection


def main() -> None:
    license_plate_detection()


if __name__ == '__main__':
    main()
