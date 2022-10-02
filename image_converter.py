import cv2
import torch
import torch.nn as nn
from model import UNET

img_org = cv2.imread("images/pos/test.jpg")
img_org_style = cv2.imread("images/style/593_s.jpg")

print(img_org.shape)

model = UNET(3, 3)

model.load_state_dict(torch.load("checkpoints_archive/unet_disc/64_unet_0.pth.tar")["state_dict"])

