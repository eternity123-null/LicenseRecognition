# 取出非安徽牌照
import shutil
import cv2
import os
# 01-86_91-298&341_449&414-458&394_308&410_304&357_454&341-0_0_14_28_24_26_29-124-24.jpg
source_path = r"D:\zcd\CV\CCPD2019\ccpd_base"
dst_path = r"D:\zcd\CV\CCPD2019\extract"
for filename in os.listdir(source_path):
    # filename="01-86_91-298&341_449&414-458&394_308&410_304&357_454&341-0_0_14_28_24_26_29-124-24.jpg"
    list1 = filename.split("-")
    subname = list1[4]
    # print(subname)
    if subname[0] != '0':
        print("subname")
        shutil.copy(source_path+'/'+filename, dst_path +'/'+ filename)
