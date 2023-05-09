# LicenseRecognition
采用了两种方案：

① YOLOv5+LPRNet 由于YOLO是矩形锚框，所以LPR识别效果不好

② ResNet+LPRNet ResNet能框柱整个车牌，经过透视变换再进行LPRNet识别，效果较好
