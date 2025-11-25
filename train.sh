# python3 train.py --data mask_data.yaml --weights ''  --cfg mask_yolov5s.yaml --device 0  --img 640 --epochs 100 --batch-size 4
python3 train.py --data mask_data.yaml --weights '/mnt/e/data_set_self/mask_face.pt'  --cfg mask_yolov5s.yaml --device 0  --img 640 --epochs 100 --batch-size 4
python3 train.py --data mask_data.yaml --weights yolov5s.pt   --device 0  --img 640 --epochs 50

