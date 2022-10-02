import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET, Discriminator
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
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train/s_style/"
TRAIN_POS_DIR = "data/train/s_pos/"
VAL_IMG_DIR = "data/eval/s_style/"
VAL_POS_DIR = "data/eval/s_pos/"

def calc_content_loss(gen_feat, orig_feat):
    content_l = torch.mean((gen_feat - orig_feat) ** 2)
    return content_l

def train_fn(loader, model, disc, optimizer, optimizer_disc, scalerU, scalerDisc):
    loop = tqdm(loader)
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (data, pos) in enumerate(loop):
        data = data.to(device=DEVICE)
        pos = pos.to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data, pos)
            real_gen = disc(data).reshape(-1)
            loss_disc_real = criterion(real_gen, torch.ones_like(real_gen))
            fake_gen = disc(predictions).reshape(-1)
            loss_disc_fake = criterion(fake_gen, torch.zeros_like(fake_gen))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

        # backward
        optimizer_disc.zero_grad()
        scalerDisc.scale(loss_disc).backward(retain_graph=True)
        scalerDisc.step(optimizer_disc)
        scalerDisc.update()

        with torch.cuda.amp.autocast():
            output = disc(predictions).reshape(-1)
            loss_unet = criterion(output, torch.ones_like(output)) + calc_content_loss(predictions, data)

        optimizer.zero_grad()
        scalerU.scale(loss_unet).backward()
        scalerU.step(optimizer)
        scalerU.update()

        # update tqdm loop
        loop.set_postfix(lossD=loss_disc.item(), lossU=loss_unet.item())


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
    disc = Discriminator(3, 68).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

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
        load_checkpoint(torch.load("checkpoints/unet_0.pth.tar"), model)
        load_checkpoint(torch.load("checkpoints/disc_0.pth.tar"), disc)

    scalerU = torch.cuda.amp.GradScaler()
    scalerDisc = torch.cuda.amp.GradScaler()

    #predictions_as_imgs(val_loader, model, folder="pred_imgs/", device=DEVICE)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, disc, optimizer, opt_disc, scalerU, scalerDisc)

        print("EPOCH ==> ", epoch)
        # save model
        checkpointU = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpointU, "unet_", 0)

        checkpointD = {
            "state_dict": disc.state_dict(),
            "optimizer": opt_disc.state_dict(),
        }
        save_checkpoint(checkpointD, "disc_", 0)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="eval_imgs/", device=DEVICE
        )


if __name__ == "__main__":
    main()
