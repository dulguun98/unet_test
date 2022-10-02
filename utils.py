import torch
import torchvision
from dataset import MyDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, name, index):
    print("=> Saving checkpoint")
    torch.save(state, "checkpoints/" + name + str(index) + ".pth.tar")

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_posdir,
    val_dir,
    val_posdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = MyDataset(
        image_dir=train_dir,
        pos_dir=train_posdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = MyDataset(
        image_dir=val_dir,
        pos_dir=val_posdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)
        with torch.no_grad():
            preds = model(x, y)
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(x, f"{folder}{idx}.png")

    model.train()

def predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)
        with torch.no_grad():
            preds = model(x, y)
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(x, f"{folder}{idx}_x.png")
        torchvision.utils.save_image(y, f"{folder}{idx}_y.png")

    model.train()
