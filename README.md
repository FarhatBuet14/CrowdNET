# CrowdNET
 CrowdNET - A Case Study Towards Designing Context- driven Computer Vision Techniques to Identify Anomalous Crowd Behavior in Hajj Pilgrimage

We have designed two computer- vision algorithms to detect physical barriers in any image/ video frame, and to count the number of people therein. By employing two algorithms, we overcome data sparsity challenges in number of images of people climbing on gates to train

### Requirements
* Python 3.6.12

### Environment Setup
~~~~{.python}
git clone https://github.com/FarhatBuet14/CrowdNET.git
cd CrowdNET
pip install -r requirements.txt
~~~~

## Barrier Detection and Localization

![gate_classifier.png](https://github.com/FarhatBuet14/CrowdNET/blob/main/images/gate_classifier.png)

![table_1.png](https://github.com/FarhatBuet14/CrowdNET/blob/main/images/table_1.png)

![table_2.png](https://github.com/FarhatBuet14/CrowdNET/blob/main/images/table_2.png)

### Test with images
* Download the model file from [Google Drive](https://drive.google.com/file/d/1B7qX6pwrtjl5zrRYtKnmczA1uIky2zee/view?usp=sharing). Put the model file on `Barrier_Detection_Localization/output` folder.
* Put the test images on the `Barrier_Detection_Localization/test_images` folder
~~~~{.python}
cd Barrier_Detection_Localization/codes
python3 test_image.py
~~~~
* You will get the result images on the `Barrier_Detection_Localization/output/test_images_results` folder.


## Crowd Estimation

![density_map.png](https://github.com/FarhatBuet14/CrowdNET/blob/main/images/density_map.png)

![table_3.png](https://github.com/FarhatBuet14/CrowdNET/blob/main/images/table_3.png)

![table_4.png](https://github.com/FarhatBuet14/CrowdNET/blob/main/images/table_4.png)


### Test with images
* Download the model file from [Google Drive](https://drive.google.com/drive/folders/1NFWA-efDFJKrW6i7Qlkt9ieOlYfwJXYW?usp=sharing). Put the model file on `Crowd_Estimation/model` folder.
* Put the test images on the `Crowd_Estimation/test_images` folder
~~~~{.python}
cd Crowd_Estimation/codes
python3 test.py
~~~~
* You will get the result images on the `Crowd_Estimation/output` folder.
