from model import Metrics, tic_toc
from tqdm import tqdm
import timm, os, torch, numpy as np
import time
def tic_toc(start_time = None): return time.time() if start_time  == None else time.time() - start_time

def train(model, tr_dl, val_dl, epochs, device, loss_fn, opt, save_prefix, threshold=0.01, save_file=None, patience=5):
    
    tr_loss, tr_IoU, tr_pa = [], [], []
    val_loss, val_IoU, val_pa = [], [], []
    tr_len, val_len = len(tr_dl), len(val_dl)
    best_loss = np.inf
    model.to(device)
    no_improve_epochs = 0  # Counter for early stopping
    
    print(f"Train is starting ..............")
    start_time = time.time()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, verbose=True)
    
    for epoch in range(epochs):
        
        tic = tic_toc()
        
        tr_loss_, tr_iou_, tr_pa_ = 0,0,0
        model.train()
        
        print(f"{epoch+1} - Epoch train process is starting.......")
        
        for idx, batch in enumerate(tqdm(tr_dl)):
            im, gt = batch
            im, gt = im.to(device), gt.to(device)
            
            pred = model(im)
            # print(pred.shape)
            
            met = Metrics(pred, gt, loss_fn )
            
            loss_ = met.loss()
            tr_loss_+= met.loss().item()
            tr_iou_ += met.mIoU()
            tr_pa_  += met.PA()
            
            opt.zero_grad()
            loss_.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()
        
        tr_loss_ /= tr_len
        tr_iou_ /= tr_len
        tr_pa_  /= tr_len
        
        tr_loss.append(tr_loss_);  tr_IoU.append(tr_iou_); tr_pa.append(tr_pa_)
        
        print(f"{epoch+1} - Epoch Validation process is starting.......")
        val_loss_, val_IoU_, val_pa_  = 0,0,0
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_dl)):
                im, gt = batch
                im, gt = im.to(device), gt.to(device)
                pred = model(im)
                met = Metrics(pred, gt, loss_fn)
                val_loss_ += met.loss().item()
                val_IoU_+= met.mIoU()
                val_pa_ += met.PA()
            val_loss_ /= val_len
            val_IoU_ /= val_len
            val_pa_  /= val_len
            
            val_loss.append(val_loss_); val_IoU.append(val_IoU_); val_pa.append(val_pa_)
            
            scheduler.step(val_loss_)
            
            print("\n ------------------------------------------")
            print(f"{epoch+1} - epoch train result: \n")
            print(f"Train Time         -> {tic_toc(tic):.3f} secs")
            print(f"Train loss                 --> {tr_loss_:.3f}")
            print(f"Train PA                   --> {tr_pa_:.3f}")
            print(f"Train mIoU                 --> {tr_iou_:.3f}\n")
            print(f"Validation loss            --> {val_loss_:.3f}")
            print(f"Validation PA              --> {val_pa_:.3f}")
            print(f"Validation mIoU            --> {val_IoU_:.3f}\n")
            
            if best_loss > (val_loss_ + threshold):
                best_loss = val_loss_
                no_improve_epochs = 0
                if save_file:
                    os.makedirs(save_file, exist_ok=True)
                    torch.save(model.state_dict(), f"{save_file}/{save_prefix}_best.pt")
            else:
                no_improve_epochs += 1
                
            if no_improve_epochs >= patience:
                
                print(f" Stopping train process because loss value did not decrease for {no_improve_epochs} - epochs")
                break
            
    return {"tr_loss": tr_loss, "tr_IoU": tr_IoU, "tr_pa":tr_pa, "val_loss": val_loss, "val_IoU": val_IoU, "val_pa":val_pa}