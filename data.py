import torch, cv2, os, random,  numpy as np, albumentations as A
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from albumentations.pytorch import ToTensorV2
from glob import glob
from torchvision import transforms as T
import segmentation_models_pytorch  as smp, time
from tqdm import tqdm

class CustomSegDental(Dataset):
    def __init__(self, root, data, transformations = None, im_type = [".jpg", '.png', 'jpeg']):
        super(). __init__()
        self.transformations = transformations
        self.n_class = 2
        self.im_path = sorted(glob(f"{root}/{data}/images/*.png"))
        self.gt_path = sorted(glob(f"{root}/{data}/mask/*.png"))
    
    def __len__(self): return len(self.im_path)
    
    def __getitem__(self, idx):
        im = cv2.cvtColor(cv2.imread(self.im_path[idx]), cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(cv2.imread(self.gt_path[idx]), cv2.COLOR_BGR2GRAY)
        if self.transformations:
            transformed = self.transformations(image = im, mask = gt)
            im = transformed['image']
            gt = transformed["mask"]
        return im, (gt<105).unsqueeze(0).long()

def get_dl(root, bs, transformations = None, split = [0.9, 0.1]):
    tr_ds = CustomSegDental(root = root, data = "train", transformations = transformations)
    ts_ds = CustomSegDental(root = root, data = "test", transformations = transformations)
    n_class = tr_ds.n_class 
    
    # splt to train and validation dataset
    
    tr_len = int(len(tr_ds)*split[0])
    val_len = len(tr_ds) - tr_len
    tr_ds, val_ds = torch.utils.data.random_split(tr_ds, [tr_len, val_len])
    print(f"Numbers of train datasets are {len(tr_ds)}")
    print(f"Numbers of validarion datasets are {len(val_ds)}")
    print(f"Numbers of train datasets are {len(ts_ds)}")
    
    # get dataloader
    
    tr_dl = DataLoader(dataset = tr_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_dl = DataLoader(dataset = val_ds, batch_size=bs, shuffle=False, num_workers=0)
    ts_dl = DataLoader(dataset = ts_ds, batch_size=1, shuffle=False, num_workers=0)
    return tr_dl, val_dl, ts_dl, n_class

