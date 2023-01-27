#*--coding="utf-8"*
import xml.etree.ElementTree as ET
import os

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    infos = []
    size = root.find('size')
    w = size.find('width')
    h = size.find('height')
    for obj in root.findall('object'):
        cat = obj.find('name').text
        score = obj.find('score')
        score = 0 if score is None else score.text
        bnd_box = obj.find('bndbox')
        xmin = int(bnd_box.find('xmin').text)
        ymin = int(bnd_box.find('ymin').text)
        xmax = int(bnd_box.find('xmax').text)
        ymax = int(bnd_box.find('ymax').text)
        center_x, center_y, width, height = _2xywh([xmin,ymin,xmax,ymax], w, h)
        infos.append({
            "cat":cat-1,
            "box":[center_x,center_y,width,height]
        })
    return infos

def _2xywh(box, w, h):
    xmin,ymin,xmax,ymax = box[0],box[1],box[2],box[3]
    width = xmax - xmin
    height = ymax - ymin
    center_x = xmin + int(width/2)
    center_y = ymin + int(height/2)
    return round(center_x/w,6), round(center_y/h,6), round(width/w,6), round(height/h,6)

def save_labels(infos, save_path):
    with open(save_path, "a+") as f:
        for info in infos:
            lines = str(info["cat"]) + " " + str(info["box"][0]) \
            + " " + str(info["box"][1]) + " " + str(info["box"][2]) + " " \
            + str(info["box"][3]) + "\n"
            f.writelines(lines)

def process(xml_root,save_dir):
    xml_names = os.listdir(xml_root)
    for xml_name in xml_names:    
        infos = parse_xml(xml_root + xml_name)
        xml_name.replace("xml","txt")
        save_labels(infos, save_dir + xml_name )        

xml_root = "/media/zsy/DATA/flower/train/xml/" # your xml path 
save_dir = "/media/zsy/DATA/flower/labels/train2017/" # your save dir
process(xml_root, save_dir)
