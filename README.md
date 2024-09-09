# RAUCA_E2E: Toward Robust and Accurate Adversarial Camouflage Generation against Vehicle Detectors

#This is the official implementation and case study of the Toward Robust and Accurate Adversarial Camouflage Generation against Vehicle Detectors

Source code can be find in [here](https://github.com/SeRAlab/Robust-and-Accurate-UV-map-based-Camouflage-Attack/tree/main/src)

## Abstract
Adversarial camouflage is a widely used physical attack against vehicle detectors for its superiority in multi-view attack performance. One promising approach involves using differentiable neural renderers to facilitate adversarial camouflage optimization through gradient back-propagation. However, existing methods often struggle to capture environmental characteristics during the rendering process or produce adversarial textures that can precisely map to the target vehicle. As a result, these limitations lead to suboptimal performance in attacks. Moreover, these approaches neglect diverse weather conditions, reducing the efficacy of generated camouflage across varying weather scenarios. To tackle these challenges, we propose a robust and accurate camouflage generation method, namely RAUCA-final. The core of RAUCA-final is a novel neural rendering component, End-to-end Neural Renderer Plus (E2E-NRP), which can accurately project vehicle textures and render images with environmental characteristics such as lighting and weather. Additionally, it utilizes an innovative sampling methodology to achieve comprehensive end-to-end optimization. In addition, we integrate a improved multi-weather dataset for camouflage generation, leveraging the E2E-NRP to enhance the attack robustness. Experimental results on six popular object detectors show that RAUCA-final outperforms existing methods in both simulation and real-world settings. 

## Framework
![pipeline](https://github.com/zhoujiawei3/RAUCA-E2E/blob/main/assets/pipeline.png)
The overview of RAUCA. First, a multi-weather dataset is created using CARLA, which includes car images, corresponding mask images, and camera angles. Then the car images are segmented using the mask images to obtain the foreground car and background images. The foreground car, together with the 3D model and the camera angle is passed through the E2E-NRP rendering component for rendering. The rendered image is then seamlessly integrated with the background. After a series of random output augmentation transformations, the image is then fed into the object detector. Finally, we optimize the adversarial camouflage through back-propagation with our devised loss function computed from the output of the object detector.


## Requirements:
before you running the code, you must install the `neural renderer` python package. You can pull Our implementation [here](https://github.com/zhoujiawei3/neural_renderer), which  is capable of end-to-end optimization of the UV map with improved sampling technique from the UV map to a facet-based tensor.


other requirements are listed in src/requirements.txt

Note that, our code is based on [Yolo-V3](https://github.com/ultralytics/yolov3) implementation.

Dowdload the YOLO-V3 weight from [https://github.com/ultralytics/yolov3/releases/download/v9.5.0/yolov3.pt](https://github.com/ultralytics/yolov3/releases/download/v9.5.0/yolov3.pt) and put it into src folder.

The generalized object EFE weight and finetuned EFE weight for car are coming soon.

## Dataset:
The latest dataset for adversarial camouflage generation can get [here](https://pan.baidu.com/s/13JvV0iOJs497iWsswQiqPA?pwd=cu1j)


## Run:
TO train NRP:
```bash
python src/NRP.py
```
TO get camouflage:
```bash
python src/generate_camouflage_E2E.py
```
