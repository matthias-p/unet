import logging
from datetime import datetime
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import CodDataset
from loss import StructureLoss
from network import UNet
from utils import get_argparser, get_target_transform, get_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUGMENTATIONS = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(180)
])


class Trainer:
    def __init__(self, train_loader, val_loader, model, criterion, optim, exp_dir, args) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.criterion = criterion
        self.optim = optim
        self.clip = args.clip

        self.exp_dir = exp_dir
        self.writer = SummaryWriter(self.exp_dir / "summary")

        self.best_mae = 1

    def train_one_epoch(self, epoch):
        self.model.train()

        train_loss = 0
        for batch, (img, gt) in enumerate(self.train_loader):
            img, gt = img.to(DEVICE), gt.to(DEVICE)

            pred = self.model(img)
            loss = self.criterion(pred, gt)

            self.optim.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optim.step()

            train_loss += loss.item()
            if batch % 50 == 0:
                logging.info(f"Current loss: {loss.item()} \t Batch {batch + 1}/{len(self.train_loader)}")
        
        train_loss = train_loss / len(self.train_loader)

        print(f"Train loss: {train_loss}")
        logging.info(f"Train loss: {train_loss}")
        self.writer.add_scalar("Loss/train", train_loss, epoch)

    def validate(self, epoch):
        self.model.eval()

        test_loss = 0
        test_mae = 0
        with torch.no_grad():
            for img, gt in self.val_loader:
                img, gt = img.to(DEVICE), gt.to(DEVICE)

                pred = self.model(img)
                test_loss += self.criterion(pred, gt).item()
                test_mae += torch.sum(torch.abs(gt - pred.sigmoid())) / gt.numel()

        test_loss /= len(self.val_loader)
        test_mae /= len(self.val_loader)

        if test_mae < self.best_mae:
            self.best_mae = test_mae
            torch.save(self.model.state_dict(), self.exp_dir / "best_mae.pth")

        print(f"Test loss: {test_loss}")
        print(f"Test MAE:  {test_mae}")
        logging.info(f"Test loss: {test_loss}")
        logging.info(f"Test MAE:  {test_mae}")
        self.writer.add_scalar("Loss/test", test_loss, epoch)
        self.writer.add_scalar("MAE/test", test_mae, epoch)

    def train(self, epochs: int = 10):
        for epoch in range(epochs):
            self.train_one_epoch(epoch)
            self.validate(epoch)

        torch.save(self.model.state_dict(), self.exp_dir / "last_epoch.pth")


def main(args):
    exp_name = f"unet_cod_{datetime.now()}"
    exp_dir = Path(args.exp_root) / exp_name
    logging.basicConfig(filename=exp_dir / "exp.log", encoding="utf-8", level=logging.INFO)

    transform = get_transform(args.resize)
    target_transform = get_target_transform(args.resize)

    train_dataset = CodDataset(
        Path(args.train_root) / args.img_dir,
        Path(args.train_root) / args.gt_dir,
        transform=transform, target_transform=target_transform
    )

    val_dataset = CodDataset(
        Path(args.val_root) / args.img_dir,
        Path(args.val_root) / args.gt_dir,
        transform=transform, target_transform=target_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)

    model = UNet()

    if args.load:
        model.load_state_dict(torch.load(args.load))
        logging.info(f"Loaded state dict from {args.load}")

    model.to(DEVICE)
    model = torch.compile(model)

    criterion = StructureLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(train_loader, val_loader, model, criterion, optim, exp_dir, args)
    trainer.train(args.epochs)


if __name__ == "__main__":
    parser = get_argparser()
    main(parser.parse_args())
