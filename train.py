import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    save_predictions_as_imgs,
    predictions_as_imgs
)

# Hyperparameters etc.
LEARNING_RATE = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EPOCHS = 500
NUM_WORKERS = 2
IMAGE_HEIGHT = 68
IMAGE_WIDTH = 68
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "data/train/s_style/"
TRAIN_POS_DIR = "data/train/s_pos/"
VAL_IMG_DIR = "data/pred/s_style/"
VAL_POS_DIR = "data/pred/s_pos/"

def calc_content_loss(gen_feat, orig_feat):
    content_l = torch.mean((gen_feat - orig_feat) ** 2)
    return content_l

def train_fn(loader, model, optimizer, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, pos) in enumerate(loop):
        data = data.to(device=DEVICE)
        pos = pos.to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data, pos)
            loss = calc_content_loss(predictions, data)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_POS_DIR,
        VAL_IMG_DIR,
        VAL_POS_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoints/my_checkpoint_0.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()

    predictions_as_imgs(val_loader, model, folder="pred_imgs/", device=DEVICE)
"""
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, scaler)

        print("EPOCH ==> ", epoch)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, 0)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="eval_imgs/", device=DEVICE
        )
"""

if __name__ == "__main__":
    main()