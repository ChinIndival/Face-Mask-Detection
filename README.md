# Face-Mask-Detection
##  Clone the repo and install package
1. Clone the repo and setup
```
$ git clone https://github.com/ChinIndival/Face-Mask-Detection.git
```

2. Install the libraries required
```
$ pip3 install -r requirements.txt
```

## Build Project

1. Go into the cloned project directory folder and type the following command:
```
$ python3 train_mask_detector.py --dataset dataset
```

2. To detect face masks in an image: 
```
$ python3 detect_mask_image.py --image images/pic1.jpeg
```

3. To detect face masks in real-time video streams:
```
$ python3 detect_mask_video.py 
```
