'''
import json
import os

# 設定類別對應表
class_mapping = {
    "dog": 0,
    "cat": 1,
    "person": 2
}

# 讀取 VIA JSON
with open("via/via_project_3Feb2025_10h22m.json", "r", encoding="utf-8") as f:
    via_data = json.load(f)

# 轉換標註
for img_key, img_data in via_data.items():
    img_filename = img_data["filename"]
    img_width = 416 #1920  # 需根據實際影像尺寸調整
    img_height = 416    #1080

    txt_filename = os.path.splitext(img_filename)[0] + ".txt"

    with open(txt_filename, "w") as txt_file:
        for region in img_data["regions"]:
            shape = region["shape_attributes"]
            label = region["region_attributes"]["class"]

            if label not in class_mapping:
                continue

            class_id = class_mapping[label]

            if shape["name"] == "polygon":
                x_points = shape["all_points_x"]
                y_points = shape["all_points_y"]

                # 轉換為 YOLO Segmentation 格式（歸一化）
                norm_points = []
                for x, y in zip(x_points, y_points):
                    norm_x = x / img_width
                    norm_y = y / img_height
                    norm_points.append(f"{norm_x:.6f} {norm_y:.6f}")

                # 輸出 YOLOv8 Segmentation 格式
                txt_file.write(f"{class_id} " + " ".join(norm_points) + "\n")

print("Segmentation 轉換完成！")
'''
import sys
import json
from PIL import Image
from decimal import *

def get_object_class(region, file, names):
    try:        
        type = region['region_attributes']['Type']
        package = region['region_attributes']['package']
    except KeyError:
        print >> sys.stderr, "type or package info is missing in ", file

    name = type + " " + package
    index = [item.lower() for item in names].index(name.lower())
    
    return index

def get_dark_annotation(region, size):
        x = region['shape_attributes']['x']
        y = region['shape_attributes']['y']
        width = region['shape_attributes']['width']
        height = region['shape_attributes']['height']

        _x      = Decimal(x + width) / Decimal(2 * size[0]) # relative position of center x of rect
        _y      = Decimal(y + height) / Decimal(2 * size[1]) # relative position of center y of rect
        _width  = Decimal(width / size[0])
        _height = Decimal(height / size[1])

        return "{0:.10f} {0:.10f} {0:.10f} {0:.10f}".format(_x, _y, _width, _height)

def main():
    with open(sys.argv[1:][0]) as file:
        dict = json.load(file)

        try:        
            namesFile = sys.argv[1:][1]
            names = open(namesFile).read().split('\n')
        except IndexError:
            print >> sys.stderr, "names file's missing from argument.\n\tnamesFile = sys.argv[1:][1]\nIndexError: list index out of range"

        via_img_metadata = dict['_via_img_metadata']
        for key in via_img_metadata.keys():  #dict.keys():
            data = via_img_metadata[key]    #dict[key]
            imageName = data['filename']
            filename = imageName.rsplit('.', 1)[0]
            
            regions = data['regions']

            try:        
                img = Image.open(imageName)
            except IOError:
                print >> sys.stderr, "No such file" , imageName

            content = ""
            for region in regions:
                obj_class = get_object_class(region, imageName, names)
                annotation = get_dark_annotation(region, img.size)
                content += "{} {}\n".format(obj_class, annotation)

            with open("{}.txt".format(filename), "w") as outFile:
                outFile.write(content)

if __name__ == "__main__":
    main()
