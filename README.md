# DeepSORT-Multi-Object-Tracking-using-YOLOv5
This repository contains the task of vehicle tracking from fisheye image using deepSORT algorithm where detector is used called yolov5. Vehicle entering the junction until leaving it will be tracker with smoothed contrail and a unique ID. Contrail is smoothed using exponential mean average over the real output. Yolov5 model is trained to output detection as a part of the tracker. A pretrained siamese network was used which was trained on NVIDIA AI City Challenge for car feature extraction. Anyone can train a siamese network for vehicle to achieve better result than this.

# Getting Started
1. Clone this repository
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ``` 
3. Create a folder named 'input' and put folder which contains all necessary sequential frames to make a desired video.
4. Select argument accordingly. --weights for loading pre-trained weight, --img for defining image size in inference, --inp (for example 'input/' if all the images are in input folder)for the input image file location.
5. Run the main script.
```bash
   python main.py
   ```

Extra : If you want to download pretrained weight for day video for car tracker then download it from this [link](https://drive.google.com/file/d/10BPsqmc4VkmhGZWuInwtiAvDDbnIF6u0/view?usp=sharing) 

# Sample Output 
![](deepsort.gif)

# Reference :
* https://github.com/nwojke/deep_sort
* Necessary blog : https://nanonets.com/blog/object-tracking-deepsort/
* Paper of DeepSORT : https://arxiv.org/pdf/1703.07402.pdf
* YoloV5 : https://github.com/ultralytics/yolov5
