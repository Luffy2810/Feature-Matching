# from eloftr import SphericalImageMatcher
# from superglue import SphericalImageMatcher
# from sift import SphericalImageMatcher
from orb import SphericalImageMatcher
from utils import *
import time
from visualize import *
from ransac import *
g8p = EightPointAlgorithmGeneralGeometry()
ransac = RANSAC_8PA()
params = [0,7680/2,3840/2]
if __name__ == "__main__":
    matcher = SphericalImageMatcher()
    scale = 1
    
    img0 = cv2.imread('/home/luffy/data/stream1/img037.png')
    img1 = cv2.imread('/home/luffy/data/stream1/img039.png')
    print (img0.shape)
    # img2 = cv2.imread('/home/luffy/data/VID_20240622_155518_00_007_processed/frame0034.jpg')



    # img0 = cv2.imread('/home/luffy/data/temp_video_1712400730078_frames/frame_0023.jpg')
    # img1 = cv2.imread('/home/luffy/data/temp_video_1712400730078_frames/frame_0024.jpg')

    img0 = cv2.resize(img0, (1920, 960), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    img1 = cv2.resize(img1, (1920, 960), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # img2_resized = cv2.resize(img2, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # print (img0_resized.shape)
    # # img_pairs = [(img0_resized, img1_resized) for i in range (10)]

    t1 = time.time()
    mkpts0, mkpts1 = matcher.match(img0, img1)
    # print (mkpts0.shape)
    # points0_spherical = cam_from_img_vectorized(params,mkpts0)
    # points1_spherical = cam_from_img_vectorized(params,mkpts1)

    # inliers,num_inliears = ransac.get_inliers(points0_spherical.T,points1_spherical.T)
    # mkpts0 = mkpts0[inliers]
    # mkpts1 = mkpts1[inliers]
    # mconf = mconf[inliers]
    # print (num_inliears)
    # ransac.reset()
    # mkpts0 = matcher.match_batch(img_pairs,batch_size = 5)
    t2 = time.time()
    # print (t2-t1)
    print ("Images Matched!")
    save_name = "vis/eloftr_direct_0_3_2_1.png"
    visualize_matches(img0, img1, mkpts0, mkpts1, save_name,t2-t1, show_keypoints=True, title="Keypoint Matches")
#     visualize_matches_interactive(
#     img0_resized, img1_resized, mkpts0, mkpts1, 
#     params, save_name, t2-t1, 
#     show_keypoints=True, 
#     title="Keypoint Matches"
# )