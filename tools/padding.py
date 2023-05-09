import cv2
import os

source_path = r"D:\zcd\CV\images\easy"
dst_path = r"D:\zcd\CV\5"
for filename in os.listdir(source_path):
    img = cv2.imread(source_path+'/'+ filename)
    ret = cv2.copyMakeBorder(img, 4000, 4000, 4000, 4000, cv2.BORDER_CONSTANT, value=(128,128,128))
    cv2.imwrite(dst_path+'/'+filename,ret)