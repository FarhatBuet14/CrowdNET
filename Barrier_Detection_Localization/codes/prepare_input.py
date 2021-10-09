import numpy as np
import pandas as pd
import json
import cv2
from tqdm import tqdm
import os

data_dir = "../data"
anno_dir = f'{data_dir}/annotation'
pred_class = "gate"

train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_df = pd.DataFrame()

for anno in [file for file in os.listdir(anno_dir) if pred_class in file]:
    
    batch = anno.split("_")[1].split(".")[0]

    with open(f'{anno_dir}/{anno}') as f:
        anno_file = json.load(f)

    files = list(anno_file.keys())
    
    dataset = []
    for index in tqdm(range(0, len(files))):
        specimen = files[index]
        img = cv2.cvtColor(cv2.imread(f'{data_dir}/batch_{batch}/{pred_class}/{anno_file[specimen]["filename"]}'), cv2.COLOR_BGR2RGB)
        width = img.shape[1]
        height =img.shape[0]
        for i in range(0, len(anno_file[specimen]['regions'])):
            mark = anno_file[specimen]['regions'][str(i)]
            data = {}
            c, x, y, w, h = mark["shape_attributes"].values()
            data['file_path'] = f'{data_dir}/batch_{batch}/{pred_class}/{anno_file[specimen]["filename"]}'
            data['file_name'] = anno_file[specimen]["filename"]
            data['width'] = width
            data['height'] = height
            data["x_min"] = int(round(x))
            data["y_min"] = int(round(y))
            if(round(x+w) > width): data["x_max"] = int(round(width))
            else: data["x_max"] = int(round(x+w))
            if(round(y+h) > height): data["y_max"] = int(round(height))
            else: data["y_max"] = int(round(y+h))
            data['class_name'] = mark["region_attributes"]["bbox"]
            
            # img_2 = annotate_image(img, [[x, y, h, w]])
            # cv2.imshow('ImageWindow', img_2)
            # cv2.waitKey()
            
            dataset.append(data)

    df = pd.DataFrame(dataset)

    # --- Train-Validation-Test Splitting
    unique_files = df.file_name.unique()
    train_files = set(np.random.choice(unique_files, int(len(unique_files) * 0.50), replace=False))
    train_df = pd.concat([train_df, df[df.file_name.isin(train_files)]])
    rest = df[~df.file_name.isin(train_files)]
    
    unique_files = rest.file_name.unique()
    val_files = set(np.random.choice(unique_files, int(len(unique_files) * 0.50), replace=False))
    val_df = pd.concat([val_df, rest[rest.file_name.isin(val_files)]])
    test_df = pd.concat([test_df, rest[~rest.file_name.isin(val_files)]])

train_df.to_csv(f'{data_dir}/train.csv', header=True, index=None)
val_df.to_csv(f'{data_dir}/val.csv', header=True, index=None)
test_df.to_csv(f'{data_dir}/test.csv', header=True, index=None)

print("Finished")