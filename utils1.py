
import sqlite3
import sys
from pathlib import Path
import numpy as np
# from eloftr import SphericalImageMatcher
import h5py
from tqdm import tqdm
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.use('TkAgg')

def vector2skew_matrix(vector):
    # Check if the input vector is a numpy or cupy array
    if isinstance(vector, np.ndarray):
        skew_matrix = np.zeros((3, 3))  # Use numpy for numpy input
    else:
        skew_matrix = cp.zeros((3, 3))  # Use cupy for cupy input

    skew_matrix[1, 0] = vector[2]
    skew_matrix[2, 0] = -vector[1]
    skew_matrix[0, 1] = -vector[2]
    skew_matrix[2, 1] = vector[0]
    skew_matrix[0, 2] = vector[1]
    skew_matrix[1, 2] = -vector[0]

    return skew_matrix.copy()
    
def spherical_normalization(array):
    assert array.shape[0] in (3, 4)
    if array.shape.__len__() < 2:
        array = array.reshape(-1, 1)
    norm = np.linalg.norm(array[0:3, :], axis=0)
    return array[0:3, :] / norm
# matcher = SphericalImageMatcher()
IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1
CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(
    MAX_IMAGE_ID
)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join(
    [
        CREATE_CAMERAS_TABLE,
        CREATE_IMAGES_TABLE,
        CREATE_KEYPOINTS_TABLE,
        CREATE_DESCRIPTORS_TABLE,
        CREATE_MATCHES_TABLE,
        CREATE_TWO_VIEW_GEOMETRIES_TABLE,
        CREATE_NAME_INDEX,
    ]
)


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)



class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = lambda: self.executescript(
            CREATE_DESCRIPTORS_TABLE
        )
        self.create_images_table = lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = lambda: self.executescript(
            CREATE_TWO_VIEW_GEOMETRIES_TABLE
        )
        self.create_keypoints_table = lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(
        self, model, width, height, params, prior_focal_length=False, camera_id=None
    ):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (
                camera_id,
                model,
                width,
                height,
                array_to_blob(params),
                prior_focal_length,
            ),
        )
        return cursor.lastrowid

    def update_camera(
        self,
        camera_id,
        model,
        width,
        height,
        params,
        prior_focal_length=True,
    ):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            """ UPDATE cameras SET model= ?, width = ?, height = ?, params = ?, prior_focal_length = ? WHERE camera_id = ?""",
            (
                model,
                width,
                height,
                array_to_blob(params),
                prior_focal_length,
                camera_id,
            ),
        )
        return cursor.lastrowid

    def add_image(
        self, name, camera_id, prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None
    ):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                image_id,
                name,
                camera_id,
                prior_q[0],
                prior_q[1],
                prior_q[2],
                prior_q[3],
                prior_t[0],
                prior_t[1],
                prior_t[2],
            ),
        )
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert len(keypoints.shape) == 2
        assert keypoints.shape[1] in [2, 4, 6]

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),),
        )

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),),
        )

    def add_matches(self, image_id1, image_id2, matches):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),),
        )

    def add_two_view_geometry(
        self,
        image_id1,
        image_id2,
        matches,
        F=np.eye(3),
        E=np.eye(3),
        H=np.eye(3),
        config=2,
    ):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,)
            + matches.shape
            + (
                array_to_blob(matches),
                config,
                array_to_blob(F),
                array_to_blob(E),
                array_to_blob(H),
            ),
        )

# def export_to_colmap_database(output_dir):
#     # Paths to the H5 files
#     feature_path = Path(os.path.join(output_dir, "features.h5"))
#     match_path = Path(os.path.join(output_dir, "matches.h5"))
#     img_dir = Path("./path_to_images")  # Specify the actual path to your images
#     database_path = "colmap.db"  # Output database

#     # Camera options (you can customize this)
#     camera_options = {
#         "general": {
#             "single_camera": True,  # Assuming single camera for all images
#             "camera_model": "simple-radial",
#         },
#     }

#     # Initialize and export to COLMAP database
#     db = COLMAPDatabase.connect(database_path)
#     db.create_tables()
#     model1, width1, height1, params1 = 0, 768, 384, np.array((0, 384, 192))
#     camera_id1 = db.add_camera(model1, width1, height1, params1)
#     fname_to_id = add_keypoints(db, feature_path, img_dir, camera_id1)
#     add_matches(db, match_path, fname_to_id)

#     db.commit()
#     print(f"Exported features and matches to {database_path}")


def parse_camera_options(
    camera_id,
    db: Path,
    image_path: Path,
) -> dict:
    """
    Parses camera options and creates camera entries in the COLMAP database.

    This function groups images by camera, assigns camera IDs, and attempts to
    initialize camera models in the provided COLMAP database.

    Args:
        camera_options (dict): A dictionary containing camera configuration options.
        db (Path): Path to the COLMAP database.
        image_path (Path): Path to the directory containing source images.

    Returns:
        dict: A dictionary mapping image filenames to their assigned camera IDs.
    """

    grouped_images = {}
    n_cameras = len(camera_options.keys()) - 1
    for camera in range(n_cameras):
        cam_opt = camera_options[f"cam{camera}"]
        images = cam_opt["images"].split(",")
        for i, img in enumerate(images):
            grouped_images[img] = {"camera_id": camera_id}
            if i == 0:
                path = os.path.join(image_path, img)
                try:
                    create_camera(db, path, cam_opt["camera_model"])
                except:
                    logger.warning(
                        f"Was not possible to load the first image to initialize cam{camera}"
                    )
    return grouped_images




def add_keypoints(
    db: Path, h5_path: Path, image_path: Path, camera_id
) -> dict:
    """
    Adds keypoints from an HDF5 file to a COLMAP database.

    Reads keypoints from an HDF5 file, associates them with cameras (if necessary),
    and adds the image and keypoint information to the specified COLMAP database.

    Args:
        db (Path): Path to the COLMAP database.
        h5_path (Path): Path to the HDF5 file containing keypoints.
        image_path (Path): Path to the directory containing source images.
        camera_options (dict, optional): Camera configuration options (see `parse_camera_options`).
                                         Defaults to an empty dictionary.

    Returns:
        dict: A dictionary mapping image filenames to their corresponding image IDs in the database.
    """
    # print (h5_path)
    
    with h5py.File(str(h5_path), "r") as keypoint_f:
        # camera_id = None
        fname_to_id = {}
        k = 0
        for filename in tqdm(list(keypoint_f.keys())):
            # print (keypoint_f.keys())
            keypoints = keypoint_f[filename]["keypoints"].__array__()

            path = filename
            image_id = db.add_image(filename, camera_id)
            fname_to_id[filename] = image_id
            # print('keypoints')
            # print(keypoints)
            # print('image_id', image_id)
            if len(keypoints.shape) >= 2:
                db.add_keypoints(image_id, keypoints)
            # else:
            #    keypoints =
            #    db.add_keypoints(image_id, keypoints)

    return fname_to_id

def add_matches(db, h5_path, fname_to_id):
    match_file = h5py.File(str(h5_path), "r")

    added = set()
    n_keys = len(match_file.keys())
    n_total = (n_keys * (n_keys - 1)) // 2

    with tqdm(total=n_total) as pbar:
        for key_1 in match_file.keys():
            group = match_file[key_1]
            for key_2 in group.keys():
                id_1 = fname_to_id[key_1[:-4]]
                id_2 = fname_to_id[key_2]

                pair_id = image_ids_to_pair_id(id_1, id_2)
                if pair_id in added:
                    warnings.warn(f"Pair {pair_id} ({id_1}, {id_2}) already added!")
                    continue

                matches = group[key_2][()]
                # db.add_matches(id_1, id_2, matches)
                db.add_two_view_geometry(id_1, id_2, matches)

                added.add(pair_id)

                pbar.update(1)
    match_file.close()


def cam_from_img_vectorized(params, points):
    PI = 4 * math.atan(1)
    c1, c2 = params[1], params[2]
    theta = (points[:, 0] - c1) * PI / c1
    phi = -1*(points[:, 1] - c2) * PI / (2 * c2)
    u = np.cos(theta) * np.cos(phi)
    v = -1 * np.sin(phi)
    w = np.cos(theta) * np.cos(phi)
    return np.column_stack((u, v, w))


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
    # if len(mkpts0) > 250:
    #     idxs = random.sample(range(len(mkpts0)), 250)
    #     mkpts0 = mkpts0[idxs]
    #     mkpts1 = mkpts1[idxs]

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