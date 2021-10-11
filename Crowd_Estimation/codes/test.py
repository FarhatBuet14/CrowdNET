import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
import csv
import pandas as pd
import scipy.io as sio

from torchvision import datasets, transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

directory = "../test_images/" 
img_list = os.listdir(directory)

model = CSRNet()
model = model.cuda()
checkpoint = torch.load('../model/partAmodel_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

pred_count = []
outputs = []
mae = 0
for i in range(len(img_list)):
    img = transform(Image.open(f'{directory}/{img_list[i]}').convert('RGB'))
    output = model(img.unsqueeze(0))
    outputs.append(output.squeeze(0).squeeze(0).detach().cpu().numpy())
    pred_count.append(float(output.detach().cpu().sum().numpy()))

df = pd.DataFrame(list(zip(img_list, pred_count)), columns =['Name', 'count'])
df.to_csv("../output/output.csv")

for i, bla in enumerate(outputs):
    pred_frame = plt.gca()
    plt.imshow(bla, 'jet')
    pred_frame.axes.get_yaxis().set_visible(False)
    pred_frame.axes.get_xaxis().set_visible(False)
    pred_frame.spines['top'].set_visible(False) 
    pred_frame.spines['bottom'].set_visible(False) 
    pred_frame.spines['left'].set_visible(False) 
    pred_frame.spines['right'].set_visible(False) 
    plt.savefig( '../output/densityMAP/' + img_list[i], bbox_inches='tight',pad_inches=0,dpi=150)

    plt.close()

