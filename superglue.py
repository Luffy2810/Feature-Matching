import sys
sys.path.append('/home/luffy/continue/repos/Hierarchical-Localization/third_party/SuperGluePretrainedNetwork')
sys.path.append('/home/luffy/continue/repos/EfficientLoFTR')
import cv2
import numpy as np
import torch
from copy import deepcopy
from models.superpoint import SuperPoint
from models.superglue import SuperGlue
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from ultralytics import YOLO
from utils import *

class SphericalImageMatcher:
    def __init__(self, model_type='opt', precision='mp', img_width=1600, img_height=800):
        
        self.IMG_WIDTH = img_width
        self.IMG_HEIGHT = img_height
        self.superpoint_config = {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1
        }
        self.superglue_config = {
            'weights': 'indoor',  # or 'indoor' depending on your dataset
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2
        }
        self.extractor = self._initialize_extractor()
        self.matcher = self._initialize_matcher()
        self.yolo_model = YOLO("/home/luffy/continue/ultralytics/runs/detect/train2/weights/best.pt")
        print ("SuperGlue Loaded")

    def _initialize_extractor(self):
        superpoint = SuperPoint(self.superpoint_config)
        return superpoint.to('cuda')

    def _initialize_matcher(self):
        superglue = SuperGlue(self.superglue_config)
        return superglue.to('cuda')

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


    def post_process_kpts(self,img0_cubemap,img1_cubemap,mkpts0,mkpts1,mconf,pred_conf=0.2):
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
        img0_gray = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY) if len(img_0.shape) == 3 else img_0
        img1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY) if len(img_1.shape) == 3 else img_1

        # Convert to torch tensors
        img0_tensor = torch.from_numpy(img0_gray / 255.).float()[None, None].cuda()
        img1_tensor = torch.from_numpy(img1_gray / 255.).float()[None, None].cuda()

        data = {'image0': img0_tensor, 'image1': img1_tensor}

        # Extract features and match
        with torch.no_grad():
            # Extract features with SuperPoint
            pred0 = self.extractor({'image': data['image0']})
            pred1 = self.extractor({'image': data['image1']})

            data.update({
                'keypoints0': pred0['keypoints'][0],
                'scores0': pred0['scores'][0],
                'descriptors0': pred0['descriptors'][0],
                'keypoints1': pred1['keypoints'][0],
                'scores1': pred1['scores'][0],
                'descriptors1': pred1['descriptors'][0],
            })

            # Match with SuperGlue
            matches = self.matcher(data)

            matches0 = matches['matches0'].cpu().numpy().squeeze()
            matching_scores0 = matches['matching_scores0'].cpu().numpy().squeeze()

            # Extract matched keypoints
            valid = matches0 > -1
            mkpts0 = data['keypoints0'][valid].cpu().numpy()
            mkpts1 = data['keypoints1'][matches0[valid]].cpu().numpy()
            scores0 = data['scores0'][valid].cpu().numpy()
            scores1 = data['scores1'][matches0[valid]].cpu().numpy()
            matching_scores0 = matching_scores0[valid]
        # Post-process keypoints
        mkpts0, mkpts1, scores0 = self.post_process_kpts(
            img_0, img_1, mkpts0, mkpts1, scores0
        )

        return mkpts0, mkpts1

