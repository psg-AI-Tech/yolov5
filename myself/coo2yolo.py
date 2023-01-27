#*--coding="utf-8"*
import os
import json
def parse_coco(coco_path):
    with open(coco_path, 'r') as f:
        data = json.load(f)
        images = data["images"]
        annotations = data["annotations"]
        categories = data["categories"]
    infos={}
    for image in images:
        infos[image["id"]] = image
        infos[image["id"]]["bbox_cat"]=[]
    for anno in annotations:
        infos[anno["image_id"]]["bbox_cat"].append( \
            {"box":anno["bbox"],"cat":anno["category_id"]})
    return infos

def save_labels(infos, w, h, save_path):
    with open(save_path, "w") as f:
        for info in infos:
            lines = str(info["cat"]-1) + " " + str(round(info["box"][0]/w,6)) \
            + " " + str(round(info["box"][1]/h,6)) + " " + str(round(info["box"][2]/h,6)) + " " \
            + str(round(info["box"][3]/w,6))+"\n"
            f.writelines(lines)

def process(coco_path,save_dir):
    infos = parse_coco(coco_path)
    for info in infos.values():
        save_path = save_dir + info["file_name"].replace("jpg","txt") # change jpg to your images name
        save_labels(info["bbox_cat"],info["width"],info["height"],save_path)
        
coco_path = "/media/zsy/DATA/flower/annotations/val.json" # your json path
save_dir = "/media/zsy/DATA/flower/labels/val2017/" # your save dir
process(coco_path, save_dir)



# 
# import json
 
# with open('./fewshotlogodetection_round1_train_202204/train/annotations/instances_train2017.json') as f:
#     Json = json.load(f)
#     annotations = Json['annotations']
#     images = Json['images']
#     image_id_name_dict = {}
#     image_id_width_dict = {}
#     image_id_height_dict = {}
#     for image in images:
#         image_id_name_dict[image['id']] = image['file_name']
#         image_id_height_dict[image['id']] = image['height']
#         image_id_width_dict[image['id']] = image['width']
#     # print(image_id_name_dict)
#     for i in range(2476):
#         for annotation in annotations:
#             if annotation['image_id'] != i:  # i表示第i张照片，数据集共2476张
#                 continue
#             bbox = annotation['bbox']
#             x, y, w, h = bbox
#             x = x + w / 2
#             y = y + h / 2
#             width = image_id_width_dict[i]
#             height = image_id_height_dict[i]
#             x = str(x / width)
#             y = str(y / height)
#             w = str(w / width)
#             h = str(h / height)
#             with open('./fewshotlogodetection_round1_train_202204/train/annotations/{}.txt'.format(
#                     image_id_name_dict[i].split('.')[0]), 'a') as f:
#                 annotation['category_id']=annotation['category_id']-1
#                 category=str(annotation['category_id'])
#                 print(category)
#                 f.write(category+' '+x+' '+y+' '+w+' '+h+'\n')

