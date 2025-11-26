# python3 train.py --data mask_data.yaml --weights ''  --cfg mask_yolov5s.yaml --device 0  --img 640 --epochs 100 --batch-size 4
python3 train.py --data mask_data.yaml --weights '/mnt/e/data_set_self/mask_face.pt'  --cfg mask_yolov5s.yaml --device 0  --img 640 --epochs 100 --batch-size 4
python3 train.py --data mask_data.yaml --weights yolov5s.pt   --device 0  --img 640 --epochs 50

# 检测
python3 detect.py --weights /mnt/e/data_set_self/mask_face.pt --source  /mnt/e/data_set_self/YOLO_Mask/score/images/test/val_1.jpg
python3 detect.py --weights yolov5s.pt --source  /mnt/e/data_set_self/YOLO_Mask/score/images/test/val_1.jpg

# 转换为ONNX模型
python3 export.py --weights runs/train/exp/weights/test_face.pt --include ONNX --imgsz 640 640 --simplify # 静态模型
python3 export.py --weights runs/train/exp/weights/test_face.pt --include ONNX --dynamic # 转为动态模型
