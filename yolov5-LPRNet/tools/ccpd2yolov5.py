import shutil
import cv2
import os
# 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15-1.jpg
# 解释：
# 025：车牌区域占整个画面的比例；
# 95_113： 车牌水平和垂直角度, 水平95°, 竖直113°
# 154&383_386&473：标注框左上、右下坐标，左上(154, 383), 右下(386, 473)
# 86&473_177&454_154&383_363&402：标注框四个角点坐标，顺序为右下、左下、左上、右上
# 0_0_22_27_27_33_16：车牌号码映射关系如下: 第一个0为省份 对应省份字典provinces中的’皖’,；第二个0是该车所在地的地市一级代码，对应地市一级代码字典alphabets的’A’；后5位为字母和文字, 查看车牌号ads字典，如22为Y，27为3，33为9，16为S，最终车牌号码为皖AY339S
# 最后的-1:蓝牌 -0：绿牌
def txt_translate(path, txt_path):
    for filename in os.listdir(path):
        list1 = filename.split("-")  # 第一次分割，以减号'-'做分割
        subname = list1[2]
        list2 = filename.split(".", 1)
        subname1 = list2[1]
        if subname1 == 'txt':
            continue
        lt, rb = subname.split("_", 1)  # 第二次分割，以下划线'_'做分割
        lx, ly = lt.split("&", 1)
        rx, ry = rb.split("&", 1)
        width = int(rx) - int(lx)
        height = int(ry) - int(ly)  # bounding box的宽和高
        cx = float(lx) + width / 2
        cy = float(ly) + height / 2  # bounding box中心点
        img = cv2.imread(path + filename)
        width = width / img.shape[1]
        height = height / img.shape[0]
        cx = cx / img.shape[1]
        cy = cy / img.shape[0]
        txtname = filename.split(".", 1)
        txtfile = txt_path + txtname[0] + ".txt"
        # 绿牌是第0类，蓝牌是第1类
        with open(txtfile, "w") as f:
            if list1[-1][0]=='1':
                # 蓝牌
                f.write(str(1) + " " + str(cx) + " " + str(cy) + " " + str(width) + " " + str(height))
            else:
                # 绿牌
                f.write(str(0) + " " + str(cx) + " " + str(cy) + " " + str(width) + " " + str(height))

if __name__ == '__main__':
    # det图片存储地址
    trainDir = r"D:\zcd\CV\data\images\train\\"
    validDir = r"D:\zcd\CV\data\images\val\\"
    testDir = r"D:\zcd\CV\data\images\test\\"
    # det txt存储地址
    train_txt_path = r"D:\zcd\CV\data\labels\train\\"
    val_txt_path = r"D:\zcd\CV\data\labels\val\\"
    test_txt_path = r"D:\zcd\CV\data\labels\test\\"
    txt_translate(trainDir, train_txt_path)
    txt_translate(validDir, val_txt_path)
    txt_translate(testDir, test_txt_path)