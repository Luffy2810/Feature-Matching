import sys
sys.path.append('/home/luffy/continue/repos/EfficientLoFTR')
import cv2
import numpy as np
import torch
from copy import deepcopy
from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
from src.utils.plotting import make_matching_figure
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from ultralytics import YOLO
from utils import *

class SphericalImageMatcher:
    def __init__(self, model_type='opt', precision='mp', img_width=1600, img_height=800):
        self.IMG_WIDTH = img_width
        self.IMG_HEIGHT = img_height
        self.model_type = model_type
        self.precision = precision
        self.matcher = self._initialize_matcher()
        self.yolo_model =  YOLO("/home/luffy/continue/ultralytics/runs/detect/train2/weights/best.pt")

    def _initialize_matcher(self):
        if self.model_type == 'full':
            _default_cfg = deepcopy(full_default_cfg)
        elif self.model_type == 'opt':
            _default_cfg = deepcopy(opt_default_cfg)
        
        if self.precision == 'mp':
            _default_cfg['mp'] = True
        elif self.precision == 'fp16':
            _default_cfg['half'] = True

        matcher = LoFTR(config=_default_cfg)
        matcher.load_state_dict(torch.load("/home/luffy/continue/repos/EfficientLoFTR/weights/eloftr_outdoor.ckpt")['state_dict'])
        matcher = reparameter(matcher)
        if self.precision == 'fp16':
            matcher = matcher.half()
        return matcher.eval().cuda()

    def yolo_post_processing(self,preds,mkpts,mkpts1,mconf):
        bboxes=  np.array([])
        try:
            bboxes = np.array([
                [int(boxes.xyxy[0][0].cpu()), int(boxes.xyxy[0][1].cpu()), 
                    int(boxes.xyxy[0][2].cpu()), int(boxes.xyxy[0][3].cpu())]
                for result in preds
                for boxes in [result.boxes]
            ])
        except:
            pass

        if len(bboxes) > 0:
            keypoints = mkpts
            x_within = (keypoints[:, 0:1] >= bboxes[:, 0]) & (keypoints[:, 0:1] <= bboxes[:, 2])
            y_within = (keypoints[:, 1:2] >= bboxes[:, 1]) & (keypoints[:, 1:2] <= bboxes[:, 3])
            inside_bbox = np.any(x_within & y_within, axis=1)
    
            mkpts = mkpts[~inside_bbox]
            mkpts1 = mkpts1[~inside_bbox]
            mconf = mconf[~inside_bbox]

        return mkpts,mkpts1,mconf


    def post_process_kpts(self,img0_cubemap,img1_cubemap,mkpts0,mkpts1,mconf,pred_conf=0.2*100):
        preds_0 = self.yolo_model(img0_cubemap, verbose=False)
        preds_1 = self.yolo_model(img1_cubemap, verbose=False)
        mkpts0,mkpts1,mconf = self.yolo_post_processing(preds_0,mkpts0,mkpts1,mconf)
        mkpts1,mkpts0,mconf = self.yolo_post_processing(preds_1,mkpts1,mkpts0,mconf)
        ind = mconf>pred_conf
        mkpts0=mkpts0[ind]
        mkpts1=mkpts1[ind]
        mconf=mconf[ind]
        return mkpts0,mkpts1,mconf


    def match(self,img_0,img_1):
        img0_raw = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY) if len(img_0.shape) == 3 else img_0
        img1_raw = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY) if len(img_1.shape) == 3 else img_1

        img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))
        img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))

        if self.precision == 'fp16':
            img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.
            img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.
        else:
            img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
            img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.


        batch = {
            'image0': img0,
            'image1': img1,
        }
        with torch.no_grad():
            if self.precision == 'mp':
                with torch.autocast(enabled=True, device_type='cuda'):
                    self.matcher(batch)
            else:
                self.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
        print (img0.shape)
        mkpts0,mkpts1,mconf = self.post_process_kpts(img_0,img_1,mkpts0,mkpts1,mconf)
        return mkpts0,mkpts1


    def match_batch(self, img_pairs, batch_size=4):
        results = []
        
        img0_batch = []
        img1_batch = []

        img0_batch_raw = []
        img1_batch_raw = []
        for img_0, img_1 in img_pairs:
            img0_raw = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY) if len(img_0.shape) == 3 else img_0
            img1_raw = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY) if len(img_1.shape) == 3 else img_1

            img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1] // 32 * 32, img0_raw.shape[0] // 32 * 32))
            img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1] // 32 * 32, img1_raw.shape[0] // 32 * 32))

            if self.precision == 'fp16':
                img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.
                img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.
            else:
                img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
                img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.

            img0_batch_raw.append(img0)
            img1_batch_raw.append(img1)
            img0_batch.append(img_0)
            img1_batch.append(img_1)

            # If batch is full, process it
            if len(img0_batch) == batch_size:
                results.extend(self._process_batch(img0_batch_raw, img1_batch_raw, img0_batch, img1_batch))
                img0_batch = []
                img1_batch = []
                img0_batch_raw = []
                img1_batch_raw = []

        # Process any remaining images in the final batch
        if len(img0_batch) > 0:
            results.extend(self._process_batch(img0_batch_raw, img1_batch_raw, img0_batch, img1_batch))

        return results

    # Helper function to process a batch
    def _process_batch(self, img0_batch_raw, img1_batch_raw,img0_batch,img1_batch):
        batch_results = []
        # Create batch tensors
        img0_tensor = torch.cat(img0_batch_raw, dim=0)
        img1_tensor = torch.cat(img1_batch_raw, dim=0)
        print (img0_tensor.shape)
        batch = {
            'image0': img0_tensor,
            'image1': img1_tensor,
        }

        with torch.no_grad():
            if self.precision == 'mp':
                with torch.autocast(enabled=True, device_type='cuda'):
                    self.matcher(batch)
            else:
                self.matcher(batch)

            mkpts0_batch = batch['mkpts0_f'].cpu().numpy()
            mkpts1_batch = batch['mkpts1_f'].cpu().numpy()
            mconf_batch = batch['mconf'].cpu().numpy()
            b_ids = batch['b_ids'].cpu().numpy()  # Get the batch IDs

        # Group keypoints by batch ID
        for i in range(len(img0_batch)):
            # Find keypoints that belong to the i-th image in the batch
            mask = b_ids == i
            mkpts0 = mkpts0_batch[mask]
            mkpts1 = mkpts1_batch[mask]
            mconf = mconf_batch[mask]

            img0 = img0_batch[i]
            img1 = img1_batch[i]
            
            # Post-process the keypoints (using YOLO model for filtering)
            mkpts0, mkpts1, mconf = self.post_process_kpts(img0, img1, mkpts0, mkpts1, mconf)
            batch_results.append((mkpts0, mkpts1))

        return batch_results