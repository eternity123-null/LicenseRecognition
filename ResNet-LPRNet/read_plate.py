from detect_explorer import DExplorer
import cv2
import numpy
import torch
from models.LPRNet import *
import os
import detect_config as config
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}


class ReadPlate:
    """
    读取车牌号
    传入侦测到的车辆图片，即可识别车牌号。
    返回：
        [[车牌号，回归框],……]
    """
    def __init__(self):
        self.detect_exp = DExplorer()

    def __call__(self, image):
        # 找四个顶点
        points = self.detect_exp(image)
        h, w, _ = image.shape
        result = []
        # print(points)
        for point, _ in points:
            plate, box = self.cutout_plate(image, point)
            # print(box)
            # ocr识别字符
            # cv2.imshow("1",plate)
            # cv2.waitKey(0)
            lp=TextRec(plate)[0]
            lb=''
            for i in lp:
                lb += CHARS[int(i)]
            print(lb)
            result.append([lb, box])

        return result
    # 仿射变换
    def cutout_plate(self, image, point):
        h, w, _ = image.shape
        # cv2.imshow('1',image)
        # cv2.waitKey(0)
        x1, x2, x3, x4, y1, y2, y3, y4 = point.reshape(-1)
        x1, x2, x3, x4 = x1 * w, x2 * w, x3 * w, x4 * w
        y1, y2, y3, y4 = y1 * h, y2 * h, y3 * h, y4 * h
        print(x1,y1)
        image1=image.copy()
        cv2.circle(image1,(int(x1),int(y1)),15,(0,0,255),-1)
        cv2.circle(image1,(int(x2),int(y2)),15,(0,0,255),-1)
        cv2.circle(image1,(int(x3),int(y3)),15,(0,0,255),-1)
        cv2.circle(image1,(int(x4),int(y4)),15,(0,0,255),-1)
        cv2.imwrite(r"D:\zcd\CV\LicensePlate-master\3"+'/'+str(x1)+'3333.jpg',image1)
        
        src = numpy.array([[x1-10, y1-10], [x2+10, y2-10], [x4-10, y4+10], [x3+10, y3+10]], dtype="float32")
        dst = numpy.array([[0, 0], [94, 0], [0, 24], [94, 24]], dtype="float32")
        box = [min(x1, x2, x3, x4), min(y1, y2, y3, y4), max(x1, x2, x3, x4), max(y1, y2, y3, y4)]
        M = cv2.getPerspectiveTransform(src, dst)
        out_img = cv2.warpPerspective(image, M, (94,24))
        cv2.imwrite(r"D:\zcd\CV\LicensePlate-master\3"+'/'+str(x1)+'.jpg',out_img)
        # # 显示
        # # 显示
        # cv2.namedWindow('2', 0)    
        # cv2.resizeWindow('2', 940, 240)   # 自己设定窗口图片的大小

        # cv2.imshow('2',out_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        return out_img, box

def TextRec(im):
    modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(config.device)
    modelc.load_state_dict(torch.load(r"D:\zcd\CV\LicensePlate-master\weights\LPRNet__iteration_7000.pth", map_location=torch.device('cpu')))
    print("load rec pretrained model successful!")
    modelc.to(config.device).eval()
    # applies a second stage classifier to yolo outputs
    
    plat_num=0
    im = cv2.GaussianBlur(im, (3, 3), 0)
    im=transform(im)
    im=[im]
    # rec
    preds = modelc(torch.Tensor(im).to(config.device)) 
    # classifier prediction

    prebs = preds.cpu().detach().numpy()

    # 对识别结果进行CTC后处理：删除序列中空白位置的字符，删除重复元素的字符
    preb_labels = list()
    for w in range(prebs.shape[0]):

        preb = prebs[w, :, :]
        preb_label = list()
        for j in range(preb.shape[1]):
            preb_label.append(numpy.argmax(preb[:, j], axis=0))

        no_repeat_blank_label = list()
        pre_c = preb_label[0]

        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:  # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)

    plat_num = numpy.array(preb_labels)
    # print("plateNum:",plat_num)
    return  plat_num

def transform( img):
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = numpy.transpose(img, (2, 0, 1))

    return img


if __name__ == '__main__':
    root=r'D:\zcd\CV\3'
    read_plate = ReadPlate()
    for image_name in os.listdir(root):
        image_path = f'{root}/{image_name}'
        image = cv2.imread(image_path)
        boxes = read_plate(image)
    
    
    # image = cv2.imread('2.png')
    
    # print("boxes:",boxes)