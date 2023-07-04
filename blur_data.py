from logger import Robo_logger
import h5py
import cv2 
import numpy as np
from glob import glob
from tqdm import tqdm
import os
dir_path = "/checkpoint/jayvakil/robopenv04/RoboSet/baking/baking_close_oven_scene_2/"

for path in glob(dir_path + "*.h5"):
    filename=path.split('/')[-1][:-3] + "_blurred.h5"
    h5 = h5py.File(path, 'r')
    
    rgb_left = []
    rgb_top = []

    left = [210, 20]
    top = [400,30]
    
    cams = {}
    
    radius = 35
    intensity = 9

    top_coords = [top]
    left_coords = [left]

    horizon = h5['Trial0/data/time'].shape[0]
    trace = Robo_logger(name=filename)
    for key, value in enumerate(h5):
        print(f"RoboSet:> {value}")
        trace.create_group(f"{value}")
        cams['time']      = h5[f'{value}/data/time']
        cams['d_left']    = h5[f'{value}/data/d_left']
        cams['d_right']   = h5[f'{value}/data/d_right']
        cams['d_top']     = h5[f'{value}/data/d_top']
        cams['d_wrist']   = h5[f'{value}/data/d_wrist']
        cams['qp_arm']    = h5[f'{value}/data/qp_arm']
        cams['qp_ee']     = h5[f'{value}/data/qp_ee']
        cams['qv_ee']     = h5[f'{value}/data/qv_ee']
        cams['qv_arm']    = h5[f'{value}/data/qv_arm']
        cams['ctrl_arm']  = h5[f'{value}/data/ctrl_arm']
        cams['ctrl_ee']   = h5[f'{value}/data/ctrl_ee']
        cams['rgb_right'] = h5[f'{value}/data/rgb_right']
        cams['rgb_wrist'] = h5[f'{value}/data/rgb_wrist']
        for i in tqdm(range(horizon)):
            img_left = h5[f'{value}/data/rgb_left'][i]
            img_top =  h5[f'{value}/data/rgb_top'][i]
            blurred_img_left = cv2.GaussianBlur(img_left, (intensity, intensity), 0)
            blurred_img_top = cv2.GaussianBlur(img_top, (intensity, intensity), 0)
            mask_left = np.ones((240, 424, 3), dtype=np.uint8)
            mask_top = np.ones((240, 424, 3), dtype=np.uint8)
            
            for l in left_coords:
                mask_left = cv2.circle(mask_left, (l[0],l[1]), radius, (100, 100, 100), -1)
                out_left = np.where(mask_left==(1, 1, 1), img_left, blurred_img_left)
            for t in top_coords:
                mask_top = cv2.circle(mask_top, (t[0],t[1]), radius, (100, 100, 100), -1)
                out_top = np.where(mask_top==(1, 1, 1), img_top, blurred_img_top)

            rgb_left.append(np.asarray(out_left))
            rgb_top.append(np.asarray(out_top))

        cams['rgb_left'] = np.asarray(rgb_left)
        cams['rgb_top'] = np.asarray(rgb_top)
        
        trace.append_datum_post_process(group_key=f'{value}', dataset_key='data', dataset_val=cams)
        rgb_left = []
        rgb_top = []
    trace.save(filename, verify_length=True)