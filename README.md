# DeepSORT-Multi-Object-Tracking-using-YOLOv5
This repository contains the project files containing our approach towards solving the problem statement of IEEE Signal Processing Cup 2020
# Getting Started
1. Clone this repository
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ``` 
3. Create a folder named 'input' and put folder which contains all necessary sequential frames to make a desired video.
4. Select argument accordingly. --weights for loading pre-trained weight, --img for defining image size in inference, --inp (for example 'input/' if all the images are in input folder)for the input image file location.

Extra : If you want to download pretrained weight for day video for car tracker then download it from this [link](https://drive.google.com/file/d/10BPsqmc4VkmhGZWuInwtiAvDDbnIF6u0/view?usp=sharing) 

# Sample Output 
![](3wvqgp.gif)

# Reference :
* https://github.com/nwojke/deep_sort
* Necessary blog : https://nanonets.com/blog/object-tracking-deepsort/
* Paper of DeepSORT : https://arxiv.org/pdf/1703.07402.pdf
