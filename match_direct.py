from eloftr import SphericalImageMatcher
# from superglue import SphericalImageMatcher
# from sift import SphericalImageMatcher
# from orb import SphericalImageMatcher
from utils import *
import time
if __name__ == "__main__":
    matcher = SphericalImageMatcher()
    scale = 0.2
    
    img0 = cv2.imread('/home/luffy/data/VID_20240622_155518_00_007_processed/frame0023.jpg')
    img1 = cv2.imread('/home/luffy/data/VID_20240622_155518_00_007_processed/frame0024.jpg')
    img2 = cv2.imread('/home/luffy/data/VID_20240622_155518_00_007_processed/frame0034.jpg')

    img0_resized = cv2.resize(img0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    img1_resized = cv2.resize(img1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    img2_resized = cv2.resize(img2, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    img_pairs = [(img0_resized, img1_resized) for i in range (10)]

    t1 = time.time()
    mkpts0, mkpts1 = matcher.match(img0_resized, img1_resized)
    # mkpts0 = matcher.match_batch(img_pairs,batch_size = 5)
    t2 = time.time()
    print (t2-t1)
    print ("Images Matched!")
    save_name = "vis/eloftr_direct_0_3_2.png"
    visualize_matches(img0_resized, img1_resized, mkpts0, mkpts1, save_name,t2-t1, show_keypoints=True, title="Keypoint Matches")
