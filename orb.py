import cv2
import numpy as np
from ultralytics import YOLO

class SphericalImageMatcher:
    def __init__(self, model_type='opt', precision='mp', img_width=1600, img_height=800):
        self.IMG_WIDTH = img_width
        self.IMG_HEIGHT = img_height
        self.model_type = model_type
        self.precision = precision
        self.matcher = cv2.ORB_create(nfeatures=40000) 
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
        self.yolo_model = YOLO("/home/luffy/continue/ultralytics/runs/detect/train2/weights/best.pt")

    def yolo_post_processing(self, preds, mkpts, mkpts1, mconf):
        bboxes = np.array([])
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

        return mkpts, mkpts1, mconf

    def post_process_kpts(self, img0_cubemap, img1_cubemap, mkpts0, mkpts1, mconf, pred_conf=0):
        preds_0 = self.yolo_model(img0_cubemap, verbose=False)
        preds_1 = self.yolo_model(img1_cubemap, verbose=False)
        mkpts0, mkpts1, mconf = self.yolo_post_processing(preds_0, mkpts0, mkpts1, mconf)
        mkpts1, mkpts0, mconf = self.yolo_post_processing(preds_1, mkpts1, mkpts0, mconf)
        ind = mconf > pred_conf
        mkpts0 = mkpts0[ind]
        mkpts1 = mkpts1[ind]
        mconf = mconf[ind]
        return mkpts0, mkpts1, mconf

    def match(self, img_0, img_1):
        img0_gray = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY) if len(img_0.shape) == 3 else img_0
        img1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY) if len(img_1.shape) == 3 else img_1

        # Detect keypoints and descriptors using ORB
        kp0, des0 = self.matcher.detectAndCompute(img0_gray, None)
        kp1, des1 = self.matcher.detectAndCompute(img1_gray, None)

        # Match descriptors using BFMatcher (ORB uses Hamming norm)
        matches = self.bf.match(des0, des1)
        matches = sorted(matches, key=lambda x: x.distance)  # Sort by distance (best matches first)

        # Extract matched keypoints
        mkpts0 = np.array([kp0[m.queryIdx].pt for m in matches])
        mkpts1 = np.array([kp1[m.trainIdx].pt for m in matches])
        mconf = np.array([1 - m.distance / 100 for m in matches])  # Normalize confidence values

        # Perform post-processing to filter out points inside YOLO-detected bounding boxes
        mkpts0, mkpts1, mconf = self.post_process_kpts(img_0, img_1, mkpts0, mkpts1, mconf)
        print (mkpts0.shape)
        return mkpts0, mkpts1

