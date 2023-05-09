# 取出部分安徽牌照
import shutil
import cv2
import os
# 01-86_91-298&341_449&414-458&394_308&410_304&357_454&341-0_0_14_28_24_26_29-124-24.jpg
source_path = r"D:\zcd\CV\CCPD2019\ccpd_base"
dst_path = r"D:\zcd\CV\CCPD2019\extract"
i = 0
for filename in os.listdir(source_path):
    i += 1
    if i % 16 == 0:
        i=0
        print("subname")
        shutil.copy(source_path+'/'+filename, dst_path +'/'+ filename)


