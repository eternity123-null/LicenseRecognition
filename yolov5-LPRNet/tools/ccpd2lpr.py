
import cv2
import os
import numpy as np

from PIL import Image

path = r'D:\zcd\CV\data\images\test'  # 改成自己的车牌路径
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
num = 0
for filename in os.listdir(path):
    num += 1
    result = ""
    _, _, box, points, plate, brightness, blurriness,GorB = filename.split('-')
    list_plate = plate.split('_')  # 读取车牌
    result += provinces[int(list_plate[0])]
    result += alphabets[int(list_plate[1])]
    result += ads[int(list_plate[2])] + ads[int(list_plate[3])] + ads[int(list_plate[4])] + ads[int(list_plate[5])] + ads[int(list_plate[6])]
    if len(list_plate)==8:
        result+=ads[int(list_plate[7])]
    img_path = os.path.join(path, filename)
    img = cv2.imread(img_path)
    assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)

    box = box.split('_')  # 车牌边界
    box = [list(map(int, i.split('&'))) for i in box]

    xmin = box[0][0]
    xmax = box[1][0]
    ymin = box[0][1]
    ymax = box[1][1]

    img = Image.fromarray(img)
    img = img.crop((xmin, ymin, xmax, ymax))  # 裁剪出车牌位置
    img = img.resize((94, 24), Image.LANCZOS)
    img = np.asarray(img)  # 转成array,变成24*94*3

    cv2.imencode('.jpg', img)[1].tofile(r"D:\zcd\CV\rec_imgs\test\{}.jpg".format(result))
print("共生成{}张".format(num))
