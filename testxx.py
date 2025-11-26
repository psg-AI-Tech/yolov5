import torch

# Model
# model = torch.hub.load('/home/zj/_github/yolov5', 'yolov5s',True,source='local')  # or yolov5n - yolov5x6, custom
model = torch.load('/mnt/e/data_set_self/mask_face.pt')  # or yolov5n - yolov5x6, custom
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

# Images
img = '/mnt/e/data_set_self/YOLO_Mask/score/images/test/val_1.jpg'  # or file, Path, PIL, OpenCV, numpy, list
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list


# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.