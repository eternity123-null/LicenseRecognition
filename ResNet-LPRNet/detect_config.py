from models_1.detect_net import WpodNet
import torch
image_root = r'D:\zcd\CV\data\images\train'
batch_size = 96
weight = 'weights/wpod_net.pt'
epoch = 100
net = WpodNet
device = 'cuda:0'
confidence_threshold = 0.9


device = torch.device(device if torch.cuda.is_available() else 'cpu')



