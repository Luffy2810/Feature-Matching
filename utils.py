import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def visualize_matches(img0, img1, mkpts0, mkpts1, save_name, timetaken, show_keypoints=True, title="Keypoint Matches"):
    """
    Visualize the matches between two images by plotting keypoints and the matches.
    If there are more than 100 matches, show a random 100 of them.

    Parameters:
    - img0: First image (equirectangular or cubemap).
    - img1: Second image (equirectangular or cubemap).
    - mkpts0: Matched keypoints in the first image (Nx2 array).
    - mkpts1: Matched keypoints in the second image (Nx2 array).
    - save_name: Name of the file to save the visualized matches.
    - timetaken: Time taken to compute the matches.
    - title: Title of the plot (default: 'Keypoint Matches').
    - show_keypoints: Whether to display keypoints (default: True).
    """
    # Limit to 100 matches if there are more than 100
    num_matches = len(mkpts0)
    if len(mkpts0) > 250:
        idxs = random.sample(range(len(mkpts0)), 250)
        mkpts0 = mkpts0[idxs]
        mkpts1 = mkpts1[idxs]

    # Convert images to RGB if they are in grayscale
    if len(img0.shape) == 2:  # Grayscale
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
    if len(img1.shape) == 2:  # Grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)

    # Create a figure to display the images and the matches
    combined_height = img0.shape[0] + img1.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(10, 15))

    # Create a combined image by stacking the two images vertically
    combined_img = np.vstack((img0, img1))
    ax.imshow(combined_img)
    ax.set_title(title)
    ax.axis('off')

    # Set colormap for the matches
    color = cm.jet(np.linspace(0, 1, len(mkpts0)))

    # Extract coordinates of keypoints
    x0, y0 = mkpts0[:, 0], mkpts0[:, 1]  # Points in image 1
    x1, y1 = mkpts1[:, 0], mkpts1[:, 1] + img0.shape[0]  # Points in image 2 (shift y1 by image height of img0)

    # Draw matches (lines connecting keypoints)
    for i in range(len(x0)):
        ax.plot([x0[i], x1[i]], [y0[i], y1[i]], color=color[i], linewidth=1)

    # Optionally, draw the keypoints
    if show_keypoints:
        ax.scatter(x0, y0, 25, marker='o', facecolors='none', edgecolors='r')  # Keypoints in image 1
        ax.scatter(x1, y1, 25, marker='o', facecolors='none', edgecolors='r')  # Keypoints in image 2

    # Add text for time taken and number of matches at the top left corner
    
    ax.text(10, 10, f'Time taken: {timetaken:.2f} sec\nMatches: {num_matches}', 
            fontsize=12, color='white', backgroundcolor='black', verticalalignment='top')

    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()


def generate_mapping_data(image_width):
    in_size = [image_width, int(image_width * 3 / 4)]
    edge = int(in_size[0] / 4)

    out_pix = np.zeros((in_size[1], in_size[0], 2), dtype="f4")
    xyz = np.zeros((in_size[1] * in_size[0] // 2, 3), dtype="f4")
    vals = np.zeros((in_size[1] * in_size[0] // 2, 3), dtype="i4")

    start, end = 0, 0
    rng_1 = np.arange(0, edge * 3)
    rng_2 = np.arange(edge, edge * 2)
    for i in range(in_size[0]):
        face = i // edge
        rng = rng_1 if face == 2 else rng_2

        end += len(rng)
        vals[start:end, 0] = rng
        vals[start:end, 1] = i
        vals[start:end, 2] = face
        start = end

    j, i, face = vals.T
    face[j < edge] = 4
    face[j >= 2 * edge] = 5

    a = 2.0 * i / edge
    b = 2.0 * j / edge
    one_arr = np.ones(len(a))
    for k in range(6):
        face_idx = face == k
        one_arr_idx = one_arr[face_idx]
        a_idx = a[face_idx]
        b_idx = b[face_idx]

        if k == 0:
            vals_to_use = [-one_arr_idx, 1.0 - a_idx, 3.0 - b_idx]
        elif k == 1:
            vals_to_use = [a_idx - 3.0, -one_arr_idx, 3.0 - b_idx]
        elif k == 2:
            vals_to_use = [one_arr_idx, a_idx - 5.0, 3.0 - b_idx]
        elif k == 3:
            vals_to_use = [7.0 - a_idx, one_arr_idx, 3.0 - b_idx]
        elif k == 4:
            vals_to_use = [b_idx - 1.0, a_idx - 5.0, one_arr_idx]
        elif k == 5:
            vals_to_use = [5.0 - b_idx, a_idx - 5.0, -one_arr_idx]

        xyz[face_idx] = np.array(vals_to_use).T

    x, y, z = xyz.T
    theta = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(z, r)

    uf = (2.0 * edge * (theta + np.pi) / np.pi) % in_size[0]
    uf[uf == in_size[0]] = 0.0
    vf = (2.0 * edge * (np.pi / 2 - phi) / np.pi)

    out_pix[j, i, 0] = vf
    out_pix[j, i, 1] = uf

    map_x_32 = out_pix[:, :, 1]
    map_y_32 = out_pix[:, :, 0]
    return map_x_32, map_y_32

def spherical_to_cubemap(img, map_x_32, map_y_32):
    cubemap = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_LINEAR)
    return cubemap

def extract_face_from_cubemap(cubemap_img, face):
    face_size = cubemap_img.shape[0] // 3

    if face == 0:  # Positive X (right)
        return cubemap_img[face_size:2 * face_size, face_size * 0:face_size * 1]  # Middle-right
    elif face == 1:  # Negative X (left)
        return cubemap_img[face_size:2 * face_size, face_size * 1:face_size*2]  # Middle-left
    elif face == 2:  # Negative Z (back)
        return cubemap_img[face_size:2 * face_size, face_size * 2:face_size * 3]  # Middle-center
    elif face == 3:  # Positive Z (front)
        return cubemap_img[face_size:2 * face_size, face_size * 3:face_size * 4]  # Middle-second from left
    elif face == 4:  # Positive Y (up)
        return cubemap_img[0:face_size, face_size * 2:face_size * 3]  # Top-center
    elif face == 5:  # Negative Y (down)
        return cubemap_img[2 * face_size:, face_size * 2:face_size * 3]  # Bottom-center

def cubemap_to_equirectangular(points, face, cubemap_size, img_width, img_height):
    points = np.array(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    x, y = points.T
    x = (x / cubemap_size) * 2 - 1
    y = (y / cubemap_size) * 2 - 1
    
    face_mapping = {
        '0': lambda: np.column_stack((np.ones_like(x), x, -y)),
        '1': lambda: np.column_stack((-x, np.ones_like(x), -y)),
        '2': lambda: np.column_stack((-np.ones_like(x), -x, -y)),
        '3': lambda: np.column_stack((x, -np.ones_like(x), -y)),
        '4': lambda: np.column_stack((y, x, np.ones_like(x))),
        '5': lambda: np.column_stack((-y, x, -np.ones_like(x)))
    }
    
    vec = face_mapping[face]()
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    theta = np.arctan2(vec[:, 1], vec[:, 0])
    phi = -np.arcsin(vec[:, 2])
    u = (theta + np.pi) / (2 * np.pi)
    v = (phi + np.pi/2) / np.pi
    result = np.column_stack((u * img_width, v * img_height))
    
    return result[0] if len(result) == 1 else result

def convert_cubemap_matches_to_equirectangular( mkpts0, mkpts1, cubemap_size,img_width,img_height):
    edge_size = cubemap_size // 4
    
    faces0 = [get_face(x, y, edge_size) for x, y in mkpts0]
    faces1 = [get_face(x, y, edge_size) for x, y in mkpts1]
    
    local_mkpts0 = np.array([(x % edge_size, y % edge_size) for x, y in mkpts0])
    local_mkpts1 = np.array([(x % edge_size, y % edge_size) for x, y in mkpts1])
    
    eq_mkpts0 = np.array([cubemap_to_equirectangular(pts, face, edge_size,img_width,img_height) 
                            for pts, face in zip(local_mkpts0, faces0)])
    eq_mkpts1 = np.array([cubemap_to_equirectangular(pts, face, edge_size,img_width,img_height) 
                            for pts, face in zip(local_mkpts1, faces1)])
    
    return eq_mkpts0, eq_mkpts1


def get_face(x, y, edge_size):
    face_x = int(x // edge_size)
    face_y = int(y // edge_size)
    if face_y == 0:
        return '4'  # top face
    elif face_y == 2:
        return '5'  # bottom face
    else:
        return str(face_x)  # side faces