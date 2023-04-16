import argparse

import torchvision.transforms as T


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, default="/home/matthias/Downloads/cod_ds/COD-TrainDataset")
    parser.add_argument("--val_root", type=str, default="/home/matthias/Downloads/cod_ds/COD-TestDataset/CAMO")
    parser.add_argument("--img_dir", type=str, default="Imgs")
    parser.add_argument("--gt_dir", type=str, default="GT")
    parser.add_argument("--resize", type=tuple, default=(256, 256))

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=0.5)

    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--exp_root", type=str, default="./exps")

    return parser


def get_transform(resize):
    return T.Compose([
        T.Resize(resize),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def get_target_transform(resize):
    return T.Compose([
        T.Resize(resize),
        T.ToTensor(),
    ])
