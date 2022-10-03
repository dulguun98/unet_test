import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math
import torch
import torchvision
from model import UNET

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

img_org_pos = cv2.imread("images/pos/test.jpg")
img_org_style = cv2.imread("images/style/4_s.jpg")

mean = torch.tensor([0, 0, 0], dtype=torch.float32)
std = torch.tensor([1, 1, 1], dtype=torch.float32)

transform = A.Compose(
        [
            A.Resize(height=68, width=68),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

h, w, c = img_org_pos.shape

for i in range(math.floor(h / 68)):
    for j in range(math.floor(w / 68)):

        img_pos = img_org_pos[i*68:i*68+68, j*68:j*68+68, :]
        augmentations = transform(image=img_org_style)
        image_s = augmentations["image"]
        image_s = image_s.unsqueeze(0).to(DEVICE)
        augmentations = transform(image=img_pos)
        image_p = augmentations["image"]
        image_p = image_p.unsqueeze(0).to(DEVICE)

        model = UNET(3, 3).to(DEVICE)
        model.load_state_dict(torch.load("checkpoints_archive/unet_disc/64_unet_0.pth.tar")["state_dict"])

        y = model(image_s, image_p)
        torchvision.utils.save_image(y, "images/0.jpg")
        y = y.detach().cpu()

        gen = cv2.imread("images/0.jpg")
        img_org_pos[i * 68:i * 68 + 68, j * 68:j * 68 + 68, :] = gen


#y = y.permute(0, 2, 3, 1)
#y = y.detach().cpu().numpy()

cv2.imwrite("images/1.jpg", img_org_pos)
cv2.imshow("img", img_org_pos)
cv2.waitKey(0)
