from DeDoDe import dedode_detector_L, dedode_descriptor_B, dedode_descriptor_G
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from PIL import Image
from utils import *
import time
from ransac_2 import *
# from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval
import logging
from visualize import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

params = [0,3136/2,1568/2]

# g8p = EightPointAlgorithmGeneralGeometry()
# ransac = RANSAC_8PA()

# You can either provide weights manually, or not provide any. If none
# are provided we automatically download them. None: We now use v2 detector weights by default.
detector = dedode_detector_L(weights = None)
# Choose either a smaller descriptor,
# descriptor = dedode_descriptor_B(weights = None)
# Or a larger one
descriptor = dedode_descriptor_G(weights = None, 
                                 dinov2_weights = None) # You can manually load dinov2 weights, or we'll pull from facebook

matcher = DualSoftMaxMatcher()

im_A_path = "/home/luffy/godrej/godrej_13_frames_3136_1568/frame_0039.jpg"
im_B_path = "/home/luffy/godrej/godrej_13_frames_3136_1568/frame_0040.jpg"
im_A = Image.open(im_A_path)
im_B = Image.open(im_B_path)
W_A, H_A = im_A.size
W_B, H_B = im_B.size
t1 = time.time()
# detections_A = detector.detect([im_A], num_keypoints = 10_000)


detections_A = detector.detect_from_path(im_A_path, num_keypoints = 25000)
keypoints_A, P_A = detections_A["keypoints"], detections_A["confidence"]

detections_B = detector.detect_from_path(im_B_path, num_keypoints = 25000)
keypoints_B, P_B = detections_B["keypoints"], detections_B["confidence"]

description_A = descriptor.describe_keypoints_from_path(im_A_path, keypoints_A)["descriptions"]
description_B = descriptor.describe_keypoints_from_path(im_B_path, keypoints_B)["descriptions"]

matches_A, matches_B, batch_ids = matcher.match(keypoints_A, description_A,
    keypoints_B, description_B,
    P_A = P_A, P_B = P_B,
    normalize = True, inv_temp=20, threshold = -1)#Increasing threshold -> fewer matches, fewer outliers
# print (batch_ids)

matches_A, matches_B = matcher.to_pixel_coords(matches_A, matches_B, H_A, W_A, H_B, W_B)
# points0_spherical = cam_from_img_vectorized(params,matches_A.cpu().numpy())
# points1_spherical = cam_from_img_vectorized(params,matches_B.cpu().numpy())
# inliers,num_inliears = ransac.get_inliers(points0_spherical.T,points1_spherical.T)
# # print (inliers.shape)
# ransac.reset()
# matches_A = matches_A[inliers]
# matches_B = matches_B[inliers]
# mconf = mconf[inliers]
t2 = time.time()
img0_resized = np.array(im_A)
img1_resized = np.array(im_B)
save_name = "vis/dedode.png"
visualize_matches(img0_resized, img1_resized, matches_A.cpu().numpy(), matches_B.cpu().numpy(), save_name,t2-t1, show_keypoints=True, title="Keypoint Matches")
print (matches_A.shape,matches_B.shape)
# visualize_matches_interactive(
# img0_resized, img1_resized, matches_A.cpu().numpy(), matches_B.cpu().numpy(), 
# params, save_name, t2-t1, 
# show_keypoints=True, 
# title="Keypoint Matches"
# )
