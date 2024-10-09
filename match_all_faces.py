# from eloftr import SphericalImageMatcher
# from superglue import SphericalImageMatcher
# from sift import SphericalImageMatcher
from orb import SphericalImageMatcher
from utils import *
import time
class Matcher():
    def __init__(self):
        self.matcher = SphericalImageMatcher()
        self.scale = 1
        self.map_x_32, self.map_y_32 = generate_mapping_data(7680)

    def match(self, img1, img2):
        cubemap1 = cv2.remap(img1, self.map_x_32, self.map_y_32, cv2.INTER_LINEAR)
        cubemap2 = cv2.remap(img2, self.map_x_32, self.map_y_32, cv2.INTER_LINEAR)
        
        cubemap1_resized = cv2.resize(cubemap1, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        cubemap2_resized = cv2.resize(cubemap2, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        img0_faces = [extract_face_from_cubemap(cubemap1_resized, i) for i in range(6)]
        img1_faces = [extract_face_from_cubemap(cubemap2_resized, i) for i in range(6)]

        all_mkpts0_spherical = []
        all_mkpts1_spherical = []

        for face_id0, img0_face in enumerate(img0_faces):
            for face_id1, img1_face in enumerate(img1_faces):
                mkpts0, mkpts1 = self.matcher.match(img0_face, img1_face)
                
                mkpts0[:, 0] *= 1 / self.scale
                mkpts0[:, 1] *= 1 / self.scale 
                mkpts1[:, 0] *= 1 / self.scale
                mkpts1[:, 1] *= 1 / self.scale 
                
                eq_mkpts0 = cubemap_to_equirectangular(mkpts0, str(face_id0), cubemap1.shape[1] / 4, img_width=img1.shape[1], img_height=img1.shape[0])
                eq_mkpts1 = cubemap_to_equirectangular(mkpts1, str(face_id1), cubemap1.shape[1] / 4, img_width=img1.shape[1], img_height=img1.shape[0])

                if eq_mkpts0.shape == (mkpts0.shape[0], 2) and eq_mkpts1.shape == (mkpts1.shape[0], 2):
                    all_mkpts0_spherical.append(eq_mkpts0)
                    all_mkpts1_spherical.append(eq_mkpts1)


        all_mkpts0_spherical = np.concatenate(all_mkpts0_spherical, axis=0) if len(all_mkpts0_spherical) > 0 else np.array([])
        all_mkpts1_spherical = np.concatenate(all_mkpts1_spherical, axis=0) if len(all_mkpts1_spherical) > 0 else np.array([])

        return all_mkpts0_spherical, all_mkpts1_spherical

if __name__ =="__main__":
    matcher = Matcher()
    print ("Matcher Initialized")
    img0 = cv2.imread('/home/luffy/data/VID_20240622_155518_00_007_processed/frame0021.jpg')
    img1 = cv2.imread('/home/luffy/data/VID_20240622_155518_00_007_processed/frame0024.jpg')
    t1 = time.time()
    mkpts0, mkpts1 = matcher.match(img0, img1)
    t2 = time.time()
    print ("Images Matched!")
    save_name = "vis/eloftr_all_0.5.png"
    visualize_matches(img0, img1, mkpts0, mkpts1, save_name,t2-t1, show_keypoints=True, title="Keypoint Matches")