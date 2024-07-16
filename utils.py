from matplotlib import pyplot as plt
from torchvision import transforms as T
import os
import random
import numpy as np

def tn_2_np(t):
        trs = T.Compose([T.Normalize(mean = [0.,0.,0.], std= [1/0.229, 1/0.224, 1/0.225]),
        T.Normalize(mean = [-0.485, -0.456, -0.406], std = [1.,1.,1.])
    ])
  
        rgb = True if len(t) == 3 else False
        return (trs(t)*255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8) if rgb  else (t*255).detach().cpu().numpy().astype(np.uint8)
    
def ploting(rows, colsm, count, im, gt = None, title = "Orginal Image" ):
        
        plt.subplot(rows, colsm, count)
        plt.imshow(tn_2_np(im.squeeze(0).float())) if gt else plt.imshow(tn_2_np(im.squeeze(0))); plt.title(title)
        plt.axis("off")
        return count + 1

def Visualize(ds, num_im, save_file = None, data_type = None):
        
        os.makedirs(save_file, exist_ok = True)
        plt.figure(figsize=(25, 20))
        rows = num_im//4
        colsm = num_im//rows
        count =1
        indices = [random.randint(0, len(ds)-1) for _ in range(num_im)]
        for ind, index in enumerate(indices):
               if count == num_im+1: break
               im, gt = ds[ind]
               count = ploting(rows, colsm, count, im = im)
               count = ploting(rows, colsm, count, im = gt, gt = True)
        plt.savefig(f"{save_file}/{data_type}.png")

def plots(r, save_file, data_name):
    os.makedirs(save_file, exist_ok=True)  # Ensure the save directory exists
    
    metrics = [("Loss", "tr_loss", "val_loss"), 
               ("PA", "tr_pa", "val_pa"), 
               ("mIoU", "tr_IoU", "val_IoU")]

    for title, train_metric, val_metric in metrics:
        plt.figure(figsize=(10, 5))
        plt.plot(r[train_metric], label=f"Train {title}")
        plt.plot(r[val_metric], label=f"Validation {title}")
        plt.title(f"Train and Validation {title}")
        plt.xticks(np.arange(len(r[val_metric])), range(1, len(r[val_metric])+1))
        plt.xlabel("Epochs")
        plt.ylabel(title)
        plt.legend()
        plt.savefig(f"{save_file}/{data_name}_{title}.png")  # Save the plot as an image
        


