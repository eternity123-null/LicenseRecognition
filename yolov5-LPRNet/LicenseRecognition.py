import argparse
import os
import platform
import sys
from pathlib import Path
import shutil
import torch
import time
import torch.backends.cudnn as cudnn
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.experimental import  attempt_load
from utils.dataloaders import LoadImages
from utils.general import ( check_img_size, cv2, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.utils import apply_classifier,plot_one_box,xywh2xyxy,scale_coords,transform

from utils.torch_utils import select_device,time_sync
from models.LPRNet import *
from utils.LPRload_data import *
from utils.augmentations import letterbox
def img_deskew(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 字符顶部和底部拟合直线
    line_upper  = []
    line_lower = []
    for k in np.linspace(-50, 0, 15):
        # 在不同阈值下自适应二值化
        binary_niblack = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,17,k)
        # 寻找包围轮廓
        contours, hierarchy = cv2.findContours(binary_niblack.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # 用矩形拟合轮廓
            bdbox = cv2.boundingRect(contour)
            # 筛选符合一定长宽比的矩形，确定出包围车牌单个字符的矩形
            if (bdbox[3]/float(bdbox[2])>0.7 and bdbox[3]*bdbox[2]>100 and bdbox[3]*bdbox[2]<1200) or (bdbox[3]/float(bdbox[2])>3 and bdbox[3]*bdbox[2]<100):
                line_upper.append([bdbox[0],bdbox[1]])                 
                line_lower.append([bdbox[0]+bdbox[2],bdbox[1]+bdbox[3]])

    line_lower = np.array(line_lower)
    line_upper = np.array(line_upper)
    # 拟合上下两条直线求出斜率
    if len(line_lower) > 0:
        line_lowerA = cv2.fitLine(line_lower,cv2.DIST_L2, 0, 0.01, 0.01)
        line_lower = line_lower[np.lexsort(line_lower[:,::-1].T)]
        kA = line_lowerA[1]/line_lowerA[0]
    else:
        kA = 0
    if len(line_upper) > 0:
        line_upperB = cv2.fitLine(line_upper,cv2.DIST_L2, 0, 0.01, 0.01)
        line_upper = line_upper[np.lexsort(line_upper[:,::-1].T)]
        kB = line_upperB[1]/line_upperB[0]
    else:
        kB = 0
    # 求出平均斜率，计算旋转角度
    k = (kA+kB)/2
    theta = np.arctan(k)*180/np.pi
    return theta

def detect(save_img=False):
    classify, out, source, det_weights, rec_weights, view_img, save_txt, imgsz = \
        opt.classify, opt.output, opt.source, opt.det_weights, opt.rec_weights,  opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete rec_result folder
    os.makedirs(out)  # make new rec_result folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load yolov5 model
    model = attempt_load(det_weights, device=device)  # load FP32 model
    print("load det pretrained model successful!")
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier  也就是rec 字符识别
    if classify:
        modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(device)
        modelc.load_state_dict(torch.load(rec_weights, map_location=torch.device('cpu')))
        print("load rec pretrained model successful!")
        modelc.to(device).eval()

    # Set Dataloader
    save_img = True
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run demo
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # 这里先设置一个全零的Tensor进行一次前向推理 判断程序是否正常
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    # 正式推理
    for path, img, im0s, vid_cap, _ in dataset:
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 因为输入网络的图片需要是4为的 
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        # print("img:",img)
        # print("im0s:",im0s)
        t1 = time_sync()
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # image deskew
        im0s = [im0s] if isinstance(im0s, np.ndarray) else im0s
        # print('pred:',enumerate(pred))
        for i, d in enumerate(pred):  # per image
            # print("i:",i)
            if d is not None and len(d):
                d = d.clone()

                # Reshape and pad cutouts
                b = xyxy2xywh(d[:, :4])  # boxes
                d[:, :4] = xywh2xyxy(b).long()

                scale_coords(img.shape[2:], d[:, :4], im0s[i].shape)

                # Classes
                pred_cls1 = d[:, 5].long()
                ims = []
                for j, a in enumerate(d):  # per item
                    height=int(a[3])-int(a[1])
                    width=int(a[2])-int(a[0])
                    # print(a)
                    cutout = im0s[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                    cutout1 = im0s[i][int(a[1]-height*0.23):int(a[3]+height*0.23), int(a[0]-width*0.14):int(a[2]+width*0.14)]
                    angle = img_deskew(cutout1)
                    break
                height, width = im0s[i].shape[:2]  # 图片的高度和宽度
                x0, y0 = width//2, height//2
                # cv2.imshow("1",im0s[i])
                # cv2.waitKey(0)
                print("angle:",angle)
                MAR = cv2.getRotationMatrix2D((x0,y0), int(angle), 1.0)
                im0s[i] = cv2.warpAffine(im0s[i], MAR, (width, height))
                
        im0s=im0s[0]
        img = letterbox(im0s, 640, 32, True)[0]  # padded resize
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)  # contiguous speed up calculate
        # redetection
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 因为输入网络的图片需要是4为的 
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # char recgnize
        pred, plat_num = apply_classifier(pred, modelc, img, im0s)
        # print("plat_num:",plat_num)

        t2 = time_sync()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s
            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for de, lic_plat in zip(det, plat_num):
                    # xyxy,conf,cls,lic_plat=de[:4],de[4],de[5],de[6:]
                    *xyxy, conf, cls=de

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        # label = '%s %.2f' % (names[int(cls)], conf)
                        lb = ""
                        for a,i in enumerate(lic_plat):
                            # if a ==0:
                            #     continue
                            lb += CHARS[int(i)]
                        label = '%s %.2f' % (lb, conf)
                        print("label:",label)
                        # print("xyxy:",xyxy)
                        # print("label:",label)
                        # annotator.box_label(xyxy, label, color=colors(c, True))
                        # im0 = annotator.result()
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=5)

            # Print time (demo + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)


    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classify', nargs='+', type=str, default=True, help='True rec')
    # parser.add_argument('--det-weights', nargs='+', type=str, default=r'D:\zcd\CV\yolov5\modelSave\yolov5_best.pt', help='model.pt path(s)')
    parser.add_argument('--det-weights', nargs='+', type=str, default=r'D:\zcd\CV\yolov5\modelSave\Final_yolo_model.pt', help='model.pt path(s)')
    parser.add_argument('--rec-weights', nargs='+', type=str, default=r'D:\zcd\CV\yolov5\modelSave\LPRNet__iteration_7000.pth', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=r'D:\zcd\CV\3', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default=r'D:\zcd\CV\2', help='rec_result folder')  # rec_result folder
    parser.add_argument('--img-size', type=int, default=640, help='demo size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented demo')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()

