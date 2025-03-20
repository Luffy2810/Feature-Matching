import cv2
from cubemap_utils import *
# from orb import SphericalImageMatcher
from eloftr import SphericalImageMatcher
# from superglue import SphericalImageMatcher
import sys
sys.path.append("/home/luffy/continue/repos/LightGlue")
from lightglue import LightGlue

sys.path.append("/home/luffy/continue/repos/accelerated_features")
from modules.xfeat import XFeat



from utils import *
import torch

from ransac_2 import *
from utils1 import *
import time
params = [0,7680/4,3840/4]

g8p = EightPointAlgorithmGeneralGeometry()
ransac = RANSAC_8PA()

matcher = SphericalImageMatcher()
image_path1 = "/home/luffy/data/66e143da3284089a55e96308_frames_3920_1960/frame_0150.jpg"  # Replace with your image path
image_path2 = "/home/luffy/data/66e143da3284089a55e96308_frames_3920_1960/frame_0152.jpg"  # Replace with your image path
        
# xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 10000,trust_repo='check').to("cuda")
xfeat = XFeat()
img1 = cv2.imread(image_path1)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread(image_path2)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Convert equirectangular images to cubemap faces
converter1 = GPU_Convert(img1.shape)
cubemaps1 = converter1.convert_to_cubemaps(img1)

converter2 = GPU_Convert(img2.shape)
cubemaps2 = converter2.convert_to_cubemaps(img2)


cubemap_size = cubemaps1["F"].shape[0]  # Assuming square cubemap faces
eq_width, eq_height = img1.shape[1], img1.shape[0]

params = [0,eq_width/2,eq_height/2]

equirect_coords1 = []
equirect_coords2 = []

t1 = time.time()
for face1 in cubemaps1.keys():
    for face2 in cubemaps2.keys():
        print (face1,face2)
        if face1=='D' or face2=='D':
            continue
        cube1 = cubemaps1[face1]
        cube2 = cubemaps2[face2]
        # mkpts1,mkpts2 = matcher.match(cube1,cube2)
        output0 = xfeat.detectAndCompute(cube1, top_k = 10000)[0]
        output1 = xfeat.detectAndCompute(cube2, top_k = 10000)[0]
        # # print (output0['keypoints'].shape)
        # # print (output0['descriptors'].shape)
        # # print (output0['scores'].shape)
        
        #Update with image resolution (required)
        output0.update({'image_size': (cube1.shape[1], cube1.shape[0])})
        output1.update({'image_size': (cube2.shape[1], cube2.shape[0])})
        # print (output0['image_size'])
        mkpts1, mkpts2,_ = xfeat.match_lighterglue(output0, output1)
        # idxs0,idxs1 = xfeat.match(output0['descriptors'],output1['descriptors'])
        # mkpts1, mkpts2 = output0['keypoints'][idxs0].cpu().numpy(), output1['keypoints'][idxs1].cpu().numpy()
        print (mkpts1.shape)
        for pt1, pt2 in zip(mkpts1, mkpts2):
            x1, y1 = pt1  # Keypoint on cubemap1
            x2, y2 = pt2  # Keypoint on cubemap2

            # Convert cubemap keypoints to equirectangular coordinates
            u1, v1 = cubemap_to_equirectangular_uv(face1, x1, y1, cubemap_size, eq_width, eq_height)
            u2, v2 = cubemap_to_equirectangular_uv(face2, x2, y2, cubemap_size, eq_width, eq_height)

            equirect_coords1.append((u1, v1))
            equirect_coords2.append((u2, v2))
t2 = time.time()
print (t2-t1)
# try:
mkpts0,mkpts1 = np.array(equirect_coords1), np.array(equirect_coords2)

points0_spherical = cam_from_img_vectorized(params,mkpts0)
points1_spherical = cam_from_img_vectorized(params,mkpts1)
inliers,num_inliears = ransac.get_inliers(points0_spherical.T,points1_spherical.T)
# print (inliers.shape)
print (ransac.best_inliers_num)
ransac.reset()
mkpts0 = mkpts0[inliers]
mkpts1 = mkpts1[inliers]

# mconf = mconf[inliers]
visualize_matches(img1, img2, mkpts0, mkpts1, "save.jpg",0, show_keypoints=True, title="Keypoint Matches")
# except Exception as E:
#     print (E)
#     # continue

