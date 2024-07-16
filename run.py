import argparse, timm, torch, random, pandas as pd, numpy as np,  pickle as p
from matplotlib import pyplot as plt
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import  ToTensorV2
from utils import Visualize, plots
import segmentation_models_pytorch as smp
from model import set_up
from data import get_dl
from tqdm import tqdm
from train import train
from inferance import inference

def run(args):
        
        mean, std= [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transformator = A.Compose([A.Resize(224,224),
                            A.Normalize(mean =mean, std=std), 
                             ToTensorV2(transpose_mask=True)], is_check_shapes=False) 
        
        tr_dl, val_dl, ts_dl, n_class = get_dl(root = args.data_path,  bs = 16, transformations=transformator)
        data_name  = {tr_dl: "train", val_dl: "Validation", ts_dl: "test"}
        for data, name in data_name.items():
                Visualize(ds = data.dataset, num_im = args.im_num, save_file = args.save_file, data_type = name)
        print(f"Sample dataset images are saved in a file named {args.save_file}!\n")
        
        model = smp.Unet(encoder_name = "resnet18", classes = 2, encoder_depth = 5, 
                 encoder_weights = "imagenet", activation = None, decoder_channels= [256, 128, 64, 32, 16])
        print("Model yuklab olindi!")
        
        device, model, optimizer, loss_fn, epochs = set_up(model)
        
        result = train(model =model, tr_dl = tr_dl, val_dl = val_dl, epochs =epochs , device=device,
                       loss_fn = loss_fn, opt = optimizer, save_prefix = args.sa_m_file,  save_file =args.save_best)
        print("Model yakunlandi!")
        
        plots(r = result, save_file= args.save_lcf, data_name=args.save_lc)
        print(f"Learning Curve image in {args.save_lcf} file!")
        
        inference(dl = ts_dl, model = model, device = device,  n_ims = 15, save_inf_image = args.save_inf , data_name = args.save_infd)
        


if __name__ == "__main__":
        
        
        parser = argparse.ArgumentParser(description="Childrens dental caries segmentation dataset")
        parser.add_argument("-dp", "--data_path", type=str, default= "childrens-dental-panoramic-radiographs-dataset/Dental_dataset/Childrens dental caries segmentation dataset", help="path of dataset")
        parser.add_argument("-nm", "--im_num", type=str, default= 10 , help="number of images")
        parser.add_argument("-sf", "--save_file", type=str, default= "Sample_data_im" , help="save sample images")
        parser.add_argument("-sm", "--sa_m_file", type=str, default= "Dental_Caries" , help="file for saving best model")
        parser.add_argument("-sb", "--save_best", type=str, default= "Dental_best_model" , help="best model")
        parser.add_argument("-sl", "--save_lcf", type=str, default= "Plots_directory" , help="Save Learning Curve file")
        parser.add_argument("-sc", "--save_lc", type=str, default= "Dental_Caries" , help=" Save Learning images")
        parser.add_argument("-si", "--save_inf", type=str, default= "Inference_image" , help=" Save Learning images")
        parser.add_argument("-id", "--save_infd", type=str, default= "Inference" , help=" Save Learning images")
        
        
        
 
        args = parser.parse_args()
        run(args)
        