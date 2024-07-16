from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as T
import timm, os, torch, numpy as np
from PIL import Image
from utils import ploting
def inference(dl, model, device, n_ims = 15, save_inf_image = None, data_name = None):
        os.makedirs(save_inf_image, exist_ok = True)
        cols = n_ims // 3; rows = n_ims // cols
              
        count = 1
        ims, gts, preds = [], [], []
        for idx, data in enumerate(dl):
            im, gt = data
     
            # Get predicted mask
            with torch.no_grad(): pred = torch.argmax(model(im.to(device)), dim = 1)
            ims.append(im); gts.append(gt); preds.append(pred)
            
        plt.figure(figsize = (25, 20))
        for idx, (im, gt, pred) in enumerate(zip(ims, gts, preds)):
            if idx == cols: break
            
            # First plot
            count = ploting(cols, rows, count, im)
     
            # Second plot
            count = ploting(cols, rows, count, im = gt.squeeze(0), gt = True, title = "Ground Truth")
     
            # Third plot
            count = ploting(cols, rows, count, im = pred, title = "Predicted Mask")
            plt.savefig(f"{save_inf_image}/{data_name}.png")