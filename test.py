import cv2
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import numpy as np

from dataset import CodDataset
from network import UNet

DEVICE = "cuda"

TRAIN_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

TRAIN_TARGET_TRANSFORM = T.Compose([
    T.ToTensor(),
])


def main():
    model = UNet()
    model.to(DEVICE)
    model = torch.compile(model)
    model.load_state_dict(torch.load("model_sloss_clip_bn_aug.pth"))
    model.eval()

    img = Image.open("/home/matthias/Downloads/cod_ds/COD-TestDataset/CAMO/Imgs/camourflage_00473.jpg")
    gt = Image.open("/home/matthias/Downloads/cod_ds/COD-TestDataset/CAMO/GT/camourflage_00473.png")
    img_size = (img.size[1], img.size[0])

    img = TRAIN_TRANSFORM(img)
    gt = TRAIN_TARGET_TRANSFORM(gt)
    img = img.unsqueeze(0)
    img = img.to(DEVICE)

    pred = model(img)

    print(pred.size())
    pred = pred.sigmoid() # .squeeze().cpu().detach().numpy()
    pred = F.interpolate(pred, img_size, mode="bilinear", align_corners=False)
    pred = pred.squeeze().cpu().detach().numpy()
    gt = np.asarray(gt, np.float32)

    mae = np.sum(np.abs(gt - pred)) / gt.size
    print(mae)

    cv2.imwrite("pred5.png", pred * 255)



if __name__ == "__main__":
    main()