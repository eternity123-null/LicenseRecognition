# import argparse
import os
import sys
from pathlib import Path
import numpy
import torch
# import torch.backends.cudnn as cudnn
from read_plate import ReadPlate
import cv2
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from PIL import Image, ImageDraw, ImageFont



def DrawChinese(img, text, positive, fontSize=20, fontColor=(
        255, 0, 0)):  # args-(img:numpy.ndarray, text:中文文本, positive:位置, fontSize:字体大小默认20, fontColor:字体颜色默认绿色)
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)
    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("MSJHL.TTC", fontSize, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    draw.text(positive, text, fontColor, font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体格式
    cv2charimg = cv2.cvtColor(numpy.array(pilimg), cv2.COLOR_RGB2BGR)  # PIL图片转cv2 图片
    return cv2charimg

if __name__ == '__main__':
    import os

    class_name = ['main']
    root = r'D:\zcd\CV\3'
    
    read_plate = ReadPlate()
    count = 0

    for image_name in os.listdir(root):
        image_path = f'{root}/{image_name}'
        image = cv2.imread(image_path)
        
        plates = []
        result=read_plate(image)
        if result:
            plate_name, (x11, y11, x22, y22) = result[0]
        
        x11, y11, x22, y22 = int(x11), int(y11), int(x22), int(y22)
        image = cv2.rectangle(image, (x11 - 5, y11 - 5), (x22 + 5, y22 + 5), (0, 0, 255), 2)
        image = DrawChinese(image, plate_name, (x11, y22), 160)
        print(image_name)
        cv2.imwrite(r'D:\zcd\CV\LicensePlate-master\3'+'/'+image_name,image)
