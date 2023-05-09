# 给蓝牌绿牌分别命名
import os
source_path = r"D:\zcd\CV\data\base\green\ex_val"
for filename in os.listdir(source_path):   #‘logo/’是文件夹路径，你也可以替换其他
	newname = filename.replace('.jpg', '-0.jpg')  #把jpg替换成png
	os.rename(source_path+'/'+filename, source_path+'/'+newname)  
