# from eloftr import SphericalImageMatcher
# from superglue import SphericalImageMatcher
from sift import SphericalImageMatcher
# from orb import SphericalImageMatcher
from utils import *
import time
class Matcher():
    def __init__(self):
        self.matcher = SphericalImageMatcher()
        self.scale = 0.2
        self.map_x_32, self.map_y_32 = generate_mapping_data(7680)

    def match(self,img1,img2):

        cubemap1 = cv2.remap(img1, self.map_x_32, self.map_y_32, cv2.INTER_LANCZOS4)
        cubemap2 = cv2.remap(img2, self.map_x_32, self.map_y_32, cv2.INTER_LANCZOS4)
        cubemap1_resized = cv2.resize(cubemap1, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        cubemap2_resized = cv2.resize(cubemap2, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        print ("Converted to cubemaps!")
        mkpts0,mkpts1 = self.matcher.match(cubemap1_resized,cubemap2_resized)
        mkpts0[:, 0] *= 1/self.scale
        mkpts0[:, 1] *= 1/self.scale 
        mkpts1[:, 0] *= 1/self.scale
        mkpts1[:, 1] *= 1/self.scale 
        print 

        eq_mkpts0, eq_mkpts1 = convert_cubemap_matches_to_equirectangular(mkpts0, mkpts1, cubemap1.shape[1],img_width=img1.shape[1],img_height=img1.shape[0])

        return eq_mkpts0, eq_mkpts1

if __name__ =="__main__":
    matcher = Matcher()
    print ("Matcher Initialized")
    img0 = cv2.imread('/home/luffy/data/VID_20240622_155518_00_007_processed/frame0021.jpg')
    img1 = cv2.imread('/home/luffy/data/VID_20240622_155518_00_007_processed/frame0024.jpg')
    t1 = time.time()
    mkpts0, mkpts1 = matcher.match(img0, img1)
    t2 = time.time()
    print ("Images Matched!")
    save_name = "vis/sift_cubemap_1.png"
    visualize_matches(img0, img1, mkpts0, mkpts1, save_name,t2-t1, show_keypoints=True, title="Keypoint Matches")