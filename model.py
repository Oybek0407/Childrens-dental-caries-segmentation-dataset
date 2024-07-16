import torch, numpy as np
def set_up(model): return "cuda" if torch.cuda.is_available() else "cpu", model.to('cuda'), torch.optim.Adam(params=model.parameters(), lr=3e-3), torch.nn.CrossEntropyLoss(), 20 


class Metrics():
    def __init__(self, pred, gt, loss_fn, eps = 3e-4,  number_class = 2):
        # self.pred = torch.argmax(torch.nn.functional.softmax(pred, dim =1), dim=1)
        self.pred, self.gt = torch.argmax(pred, dim = 1), gt
        self.pred_ = pred
        self.gt =gt.squeeze(1)
        # print(self.pred_.shape)
        # print(self.gt.shape)
        self.loss_fn = loss_fn
        self.eps = eps
        self.number_class = number_class
    def to_contiguos(self, inp): return inp.contiguous().view(-1)

    def PA(self):
        with torch.no_grad():
            PA = torch.eq(self.pred, self.gt).int()
        return float(PA.sum())/float(PA.numel())
    
    def mIoU(self):
        pred, gt = self.to_contiguos(self.pred), self.to_contiguos(self.gt)
        IoU_class = []
        for a in range(self.number_class):
            mutch_pred = pred==a
            mutch_gt = gt==a
            if mutch_pred.long().sum().item() ==0 : IoU_class.append(np.nan)
            else:
                intersection = torch.logical_and(mutch_pred, mutch_gt).sum().float().item()
                union = torch.logical_or(mutch_pred, mutch_gt).sum().float().item()
                iou = intersection/(union + self.eps)
                IoU_class.append(iou)
            return np.nanmean(IoU_class)
    def loss(self):
        return self.loss_fn(self.pred_, self.gt)
        
def tic_toc(start_time = None): return time.time() if start_time  == None else time.time() - start_time
