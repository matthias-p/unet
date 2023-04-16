from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class CodDataset(Dataset):
    def __init__(self, img_dir: str, gt_dir: str, transform = None, target_transform = None) -> None:
        self.img_files = sorted(Path(img_dir).iterdir())
        self.gt_files = sorted(Path(gt_dir).iterdir())
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        img = Image.open(self.img_files[index]).convert("RGB")
        gt = Image.open(self.gt_files[index]).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            gt = self.target_transform(gt)
        
        return img, gt
