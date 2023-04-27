import torch

# Model
# model = torch.hub.load('/home/zj/_github/yolov5', 'yolov5s',True,source='local')  # or yolov5n - yolov5x6, custom
model = torch.load('/home/zj/_github/yolov5/yolov5s.pt')  # or yolov5n - yolov5x6, custom
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

# Images
img = '/home/zj/_github/yolov5-6.2/data/images/bus.jpg'  # or file, Path, PIL, OpenCV, numpy, list
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list


# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.