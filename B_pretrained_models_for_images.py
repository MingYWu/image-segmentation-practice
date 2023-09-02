import os
import cv2
import numpy as np
import matplotlib.pylab as plt
import argparse

import torch
from mmseg.apis import init_model, inference_model, show_result_pyplot

os.chdir('mmsegmentation')


def visualization(img_bgr, result, title, out_file):
    pred_mask = result.pred_sem_seg.data[0].detach().cpu().numpy()
    
    plt.figure(figsize=(14,8))

    plt.subplot(1,2,1)
    plt.imshow(img_bgr[:,:,::-1])
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img_bgr[:,:,::-1])
    plt.imshow(pred_mask, alpha=0.6) # alpha 高亮区域透明度，越小越接近原图
    plt.axis('off')
    plt.suptitle(title)
    
    plt.savefig(out_file)
    print('finished')
    

def main(opt):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    img_bgr = cv2.imread(opt.img_path)
    
    model = init_model(config=opt.config_file, checkpoint=opt.checkpoint_file, device=device)
    result = inference_model(model, img_bgr)
    visualization(img_bgr, result=result, title=opt.model_name, out_file=opt.out_file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='segformer')
    parser.add_argument('--config_file', type=str, default='configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py', 
                        help='模型 config 配置文件')
    parser.add_argument('--checkpoint_file', type=str, default='https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth', 
                        help='模型 checkpoint 权重文件')
    parser.add_argument('--img_path', type=str, default='./data/street_uk.jpeg')
    parser.add_argument('--out_file', type=str, default='./outputs/')
    
    opt = parser.parse_args()
    
    opt.out_file = opt.out_file + f'pred_street_uk_{opt.model_name}' + '.jpeg'
    
    main(opt)
    
    