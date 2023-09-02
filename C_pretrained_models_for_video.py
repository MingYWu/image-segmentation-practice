import argparse
import os
import numpy as np
import time
import shutil

import torch

from PIL import Image
import cv2

os.chdir('mmsegmentation')

import mmcv
import mmengine
from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules

register_all_modules()

from mmseg.datasets import CityscapesDataset, cityscapes

classes = cityscapes.CityscapesDataset.METAINFO['classes']
palette = cityscapes.CityscapesDataset.METAINFO['palette']


def predict_single_frame(model, img, opacity=0.2):
    
    result = inference_model(model, img)
    
    # 将分割图按调色板染色
    seg_map = np.array(result.pred_sem_seg.data[0].detach().cpu().numpy()).astype('uint8')
    seg_img = Image.fromarray(seg_map).convert('P')
    seg_img.putpalette(np.array(palette, dtype=np.uint8))
    
    show_img = (np.array(seg_img.convert('RGB')))*(1-opacity) + img*opacity
    
    return show_img


def main(model, input_video, temp_out_dir, out_dir, temp_dir_delete=True):

    if not os.path.exists(temp_out_dir):
        os.mkdir(temp_out_dir)
        print('创建临时文件夹 {} 用于存放每帧预测结果'.format(temp_out_dir))
    else:
        print('临时文件夹 {} 已经存在，可能是之前创建的。'.format(temp_out_dir))
        # 如果需要，在这里添加重命名现有文件夹的逻辑

    ## 逐帧预测
    imgs = mmcv.VideoReader(input_video)
    
    prog_bar = mmengine.ProgressBar(len(imgs))
    
    # 对视频逐帧处理
    for frame_id, img in enumerate(imgs):
        
        # 处理单帧画面
        show_img = predict_single_frame(model, img)
        temp_path = f'{temp_out_dir}/{frame_id:06d}.jpg'
        cv2.imwrite(temp_path, show_img)
        
        # 更新进度条
        prog_bar.update()
    
    # 把每一帧串成视频文件
    mmcv.frames2video(temp_out_dir, out_dir+f'out_{temp_out_dir}.mp4', fps=imgs.fps, fourcc='mp4v') 
    
    # 删除存放每帧画面的临时文件夹
    if temp_dir_delete == True:
        shutil.rmtree(temp_out_dir)
        print('删除临时文件夹: ', temp_out_dir)
    

if __name__ =='__main__':
    """
    基于citysspace数据集，建立推理模型
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py', 
                        help='模型 config 配置文件')
    parser.add_argument('--checkpoint_file', type=str, default='https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth', 
                        help='模型 checkpoint 权重文件')
    parser.add_argument('--input_video', type=str, default='./data/street_5s.mp4',
                        help='视频路径')
    parser.add_argument('--temp_out_dir',type=str, default='',help='创建临时文件夹，单帧视频保存的位置')
    parser.add_argument('--out_dir',type=str, default='./outputs/',help='视频保存的位置')
    parser.add_argument('--temp_dir_delete',type=str, default='True')
    
    opt = parser.parse_args()
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    opt.temp_out_dir = time.strftime('%Y%m%d%H%M%S')

    model = init_model(opt.config_file, opt.checkpoint_file, device=device)
    
    main(model, opt.input_video, opt.temp_out_dir, opt.out_dir, opt.temp_dir_delete)
