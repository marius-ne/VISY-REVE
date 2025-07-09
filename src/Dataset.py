import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import pandas as pd
import argparse
import json
import trimesh
import pyrender
import cv2
import time
import sys
import yaml

from typing import List, Optional
from datetime import datetime

from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree

import plotly.graph_objects as go
import seaborn as sns
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Slerp

# Own imports
import src.Sample as Sample
from src.synthesis import Homography, Transform3D
import cv_utils
import src.metrics as metrics
from sklearn.neighbors import NearestNeighbors
from src.traj import backproject
from src.Sample import setup_model,run_inference    


# ----------- DATASET LOADING ------------
from pathlib import Path   

# CHANGE CONFIG FILE NAME HERE
config_path = Path(__file__).parent / 'marius.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

def cfg_path(key: str) -> Path:
    """Helper to convert and normalize a path from the config."""
    return config[key]

# ----------- DATASET CONSTANTS (loaded from yaml) ------------
base_path_airbus             = cfg_path('base_path_airbus')
mesh_path_airbus             = cfg_path('mesh_path_airbus')
keypoints_path_airbus        = cfg_path('keypoints_path_airbus')
model_path_airbus            = cfg_path('model_path_airbus')
model_config_path_airbus     = cfg_path('model_config_path_airbus')

base_path_speed              = cfg_path('base_path_speed')
mesh_path_speed              = cfg_path('mesh_path_speed')
keypoints_path_speed         = cfg_path('keypoints_path_speed')

base_path_shirt              = cfg_path('base_path_shirt')

base_path_speed_plus         = cfg_path('base_path_speed_plus')
model_path_speed_plus        = cfg_path('model_path_speed_plus')
model_config_path_speed_plus = cfg_path('model_config_path_speed_plus')

base_path_swisscube          = cfg_path('base_path_swisscube')
mesh_path_swisscube          = cfg_path('mesh_path_swisscube')
keypoints_path_swisscube     = cfg_path('keypoints_path_swisscube')
model_path_swisscube         = cfg_path('model_path_swisscube')
model_config_path_swisscube  = cfg_path('model_config_path_swisscube')

base_path_urso               = cfg_path('base_path_urso')
mesh_path_urso               = cfg_path('mesh_path_urso')
keypoints_path_urso          = cfg_path('keypoints_path_urso')
# ------------------------------------------

class Dataset:
    """
    A dataset of image samples for 6DoF pose estimation.
    """
    implemented_datasets = [
        "SPEED",
        "AIRBUS",
        "SHIRT",
        "SPEED+",
        "SWISSCUBE",
        "URSO"
    ]
    BG_threshold = 20

    def __init__(self,
                 mesh_path: str = None,
                 base_path: str = None,
                 dataset_name: str = None,
                 keypoints_path: str = None,
                 model_path: str = None,
                 model_config_path: str = None):
        """
        Initializes the Dataset object with provided paths and dataset information.
        Args:
            mesh_path (str, optional): Path to the mesh files associated with the dataset. Defaults to None.
            base_path (str, optional): Base directory path where the dataset is stored. Defaults to None.
            dataset_name (str, optional): Name of the dataset. Defaults to None.
            keypoints_path (str, optional): Path to the keypoints file or directory. Must be .json. Defaults to None.
            model_path (str, optional): Path to the Neural Network Keypoint Detector model weights file (.pth).
            model_config_path (str, optional): Path to the .yaml config file used to train the 
                Neural Network Keypoint Detector model.
        Attributes:
            dataset_name (str): Stores the name of the dataset.
            samples (List[Sample.Sample]): List to hold all samples in the dataset.
            posed_samples (List[Sample.Sample]): List to hold training samples.
            unposed_samples (List[Sample.Sample]): List to hold testing samples.
            mesh_path (str): Path to the mesh files.
            base_path (str): Base directory path for the dataset.
            image_rootdir_path (str): Path to the root directory containing all image files, determined by get_image_dir_path().
            extension (str): Image file extension used in the dataset, determined by get_image_extension().
        Raises:
            FileNotFoundError: If any of the provided paths do not exist or cannot be accessed.
            ValueError: If required parameters are missing or invalid.
        Note:
            The constructor initializes the dataset structure and prepares lists for samples, training, and testing data.
            It also determines the root directory for images and the image file extension based on the dataset configuration.
        """
        self.dataset_name = dataset_name
        print(f"\nLoading dataset: {dataset_name}.")

        # Load keypoints from JSON file if provided
        self.keypoints_path = os.path.normpath(keypoints_path)
        if keypoints_path is not None:
            with open(keypoints_path, "r") as f:
                keypoints_data = json.load(f)
                self.keypoints3D = np.array(keypoints_data, dtype=np.float32)
                self.num_keypoints = self.keypoints3D.shape[0]
        else:
            self.keypoints3D = None
            self.num_keypoints = 0

        self.samples: List[Sample.Sample] = []
        self.posed_samples: List[Sample.Sample] = []
        self.unposed_samples: List[Sample.Sample] = []

        self.mesh_path = os.path.normpath(mesh_path)

        self.base_path = os.path.normpath(base_path)

        self.model_path = os.path.normpath(model_path)
        self.model_config_path = os.path.normpath(model_config_path)

        # Get the dataset-specific image directory path
        # The image_rootdir is the most specific folder from which
        # you can still reach all image files. 
        self.image_rootdir_path = self.get_image_dir_path()
        self.extension = self.get_image_extension()

    def _check_dataset(self):
        if not self.dataset_name or self.dataset_name.upper() not in \
            [name.upper() for name in self.implemented_datasets]:
            joined_datasets = '\n'.join(self.implemented_datasets)
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented.\n"+\
                             f"Implemented datasets: {joined_datasets}")

    def check_dataset(method):
        """
        Decorator to check whether the dataset name is implemented.
        """
        def wrapper(self, *args, **kwargs):
            self._check_dataset()
            return method(self, *args, **kwargs)
        return wrapper
    
    @check_dataset
    def load_dataset(self,
                     N_samples: int = -1,
                     synthetic_only: bool = False):
        """
        Loads the dataset with all samples into memory. After calling it,
        the dataset.samples becomes populated.

        Should be explicitly called by user.

        Args:
            N_samples (int): Can be used to load less then the whole dataset.
                -1 (default) means load the whole dataset.
            synthetic_only (bool): If True, only loads synthetic dataset. For the
                moment only implemented for SPEED+.
        """
        ds_name = self.dataset_name.upper()

        self.N_samples = N_samples

        if ds_name == "SPEED":
            self._load_speed()
        elif ds_name == "AIRBUS":
            self._load_airbus()
        elif ds_name == "SHIRT":
            self._load_shirt()
        elif ds_name == "SPEED+":
            self._load_speed_plus(synthetic_only)
        elif ds_name == "SWISSCUBE":
            self._load_swisscube()
        elif ds_name == "URSO":
            self._load_urso()

        # ======Same for all datasets======
        # Build KD tree for quick nearest neighbour search (Euclidean)
        camera_centers = [s.C for s in self.posed_samples]
        self.tree = KDTree(camera_centers)

        # Build KNN for BDD NN search
        # Create the NearestNeighbors object:
        self.bdd_nbrs = NearestNeighbors(
            n_neighbors=1,           # how many neighbors you want back
            algorithm='brute',        # brute‐force is required for a callable metric
            metric=metrics.BDD_flat_fast
        ) 
        rotations = np.array([s.R.reshape(1,9) for s in self.posed_samples]).squeeze()
        self.bdd_nbrs.fit(rotations)

        # Calculate translation vector for camera that fits the mesh in the field
        # of view for each attitude
        self.t_to_fit_mesh = [0,0,self._get_t_vec_to_fit_mesh()]

        # Setup neural network
        self.model = setup_model(
            cfg_file=self.model_config_path,
            weights_file=self.model_path,
            device="cuda"  # or "cpu"
        )

    @check_dataset
    def get_image_extension(self):
        """
        Gets the filetype, i.e. png or jpg of the dataset's samples.
        Returns it as file extension, i.e. with a dot in front.
        """
        ds_name = self.dataset_name.upper()

        if ds_name in ["SPEED","SHIRT","SPEED+","SWISSCUBE"]:
            extension = ".jpg"
        elif ds_name in ["AIRBUS", "URSO"]:
            extension = ".png"
        return extension

    @check_dataset
    def get_image_filename(self, 
                           sample: Sample.Sample,
                           with_extension: bool = True):
        """
        Generates the dataset-specific image path given the image id
        and split directory of a given sample.
        """
        if not hasattr(sample, "image_id_num") or not hasattr(sample, "parent_dir") or \
            sample.image_id_num is None or sample.parent_dir is None:
            raise ValueError("Sample must have an image_id_num and parent_dir "+\
                             "to generate the image filename.")
        
        if with_extension:
            extension = self.get_image_extension()
        else:
            extension = ""

        ds_name = self.dataset_name.upper()
        if ds_name == "SPEED":
            # Images in real and real_test folders contain the string "real"
            real_string = "real" if "real" in os.path.basename(sample.parent_dir) else ""
            
            image_filename = f"img{sample.image_id_num:06d}{real_string}{extension}"

        elif ds_name == "AIRBUS":
            image_filename = f"image{sample.image_id_num:05d}{extension}"
        elif ds_name in ["SHIRT","SPEED+"]:
            image_filename =  f"img{sample.image_id_num:06d}{extension}"
        elif ds_name in ["SWISSCUBE"]:
            image_filename =  f"{sample.image_id_num:06d}{extension}"
        elif ds_name in ["URSO"]:
            image_filename =  f"{sample.image_id_num}_rgb{extension}"
        return image_filename
        
    @check_dataset
    def get_image_dir_path(self):
        """
        Gets the path of the directory containing the images.

        The image_dir_path is the lowermost folder from which you can
        reach all the image files of this dataset.
        """
        ds_name = self.dataset_name.upper()

        if ds_name == "SPEED":
            image_dir_path = os.path.join(self.base_path, "images")
        elif ds_name == "AIRBUS":
            image_dir_path = os.path.join(self.base_path, "images")
        elif ds_name in ["SHIRT","SPEED+","SWISSCUBE","URSO"]:
            image_dir_path = self.base_path

        return image_dir_path
    
    @check_dataset
    def _load_mesh(self):
        """
        Loads dataset mesh and applies necessary transformations.
        """
        self.mesh = trimesh.load(self.mesh_path, force='mesh')
        
        if self.dataset_name == "AIRBUS":
            # Convert mesh from mm to m, AIRBUS mesh is too big
            self.mesh.apply_scale(0.001)
            # Apply 90 degree rotation around x-axis
            rot_x_90 = trimesh.transformations.rotation_matrix(np.deg2rad(90), [1, 0, 0])
            self.mesh.apply_transform(rot_x_90)
        elif self.dataset_name == "SWISSCUBE":
            self.mesh.apply_scale(0.001)

        else:
            pass # some datasets' meshes don't need transformations

        self.render_mesh = pyrender.Mesh.from_trimesh(self.mesh, smooth=False)


        
    def save_synthesized(self,
                          source: Sample.Sample,
                          target: Sample.Sample,
                          synth: np.ndarray,
                          method: str = "HOM"):
        """
        Utility for writing synthesized images to disk in a dataset-aware
        manner, i.e. respecting the directory structure of each dataset.

        Args:
            source (Sample): source sample
            target (Sample): target sample
            synth (np.ndarray): synthesized image
            method (str): synthesis method, for filename

        Returns:
            filepath (str): path to the saved synthesized image.
        """
        # Synthesized name *must* uniquely identify each resulting image
        # Therefore we use image_id and not image_id_num, which can be shared
        # across dataset splits. 
        synthesized_dir_path = os.path.join(source.parent_dir,"synthesized")
        synthesized_name = f"{target.image_id}From{source.image_id}{method}{self.extension}" 
        synthesized_path = os.path.join(synthesized_dir_path, synthesized_name)

        # Create directory if it doesn't exist
        os.makedirs(synthesized_dir_path, exist_ok=True)

        bgr_synth = cv2.cvtColor(synth, cv2.COLOR_RGB2BGR)
        cv2.imwrite(synthesized_path,bgr_synth)
        print(f"Wrote synthesized image {synthesized_name} to {synthesized_dir_path}.")

        return synthesized_path

    def reload_mesh(self, new_mesh_path):
        """
        Reloads mesh with new path.
        """
        self.mesh = trimesh.load(new_mesh_path, force='mesh')
        self.render_mesh = pyrender.Mesh.from_trimesh(self.mesh, smooth=False)

    def load_from_coco(self, coco_json_path, train_test_split=True, images_already_masked=True):
        """
        Loads dataset samples from a COCO-format JSON file (produced by export_labels_to_coco).
        Populates self.posed_samples and self.unposed_samples, and builds a KDTree on posed_samples.

        Args:
            coco_json_path (str): Path to the COCO JSON file (or a dict containing “train”/“test”).
            train_test_split (bool): If True, expects the JSON to have top-level keys “train” and “test”,
                each mapping to a COCO dict. If False, expects a single COCO dict (all samples → posed_samples).
            images_already_masked (bool): If True, expects the images at the filepaths listed in the
                labels file to already be masked (this is normally the case when they were used for training).
        """
        # Helper to build Sample objects from a COCO dict
        def build_samples_from_coco(coco_dict, subset_type):
            """
            coco_dict: dict with keys “images”, “annotations”, “info” (etc.).
            subset_type: string, either "train" or "test", used to set sample.type.
            Returns a list of Sample instances.
            """
            samples = []
            # Camera intrinsics from info.camera_k
            K = np.array(coco_dict["info"]["camera_k"], dtype=np.float32)
            # Image size – assume all images share the same width/height in info or image entries
            # (We’ll take from the first image entry)
            first_img = coco_dict["images"][0]
            width = first_img["width"]
            height = first_img["height"]

            # Build a lookup from image_id → image entry
            img_lookup = { img["id"]: img for img in coco_dict["images"] }

            # Build a lookup from image_id → annotation entry (assuming one annotation per image)
            ann_lookup = {}
            for ann in coco_dict["annotations"]:
                img_id = ann["image_id"]
                ann_lookup[img_id] = ann

            for img_id, img_entry in img_lookup.items():
                # Reconstruct file paths and IDs
                fileid = img_entry["id"] # this is UNIQUE for COCO
                filename = img_entry["file_name"]
                filepath = os.path.join(self.base_path,img_entry["filepath"])
                image_id_str = str(fileid)
                image_id_num = int(fileid)

                # Find the corresponding annotation
                ann = ann_lookup[fileid]

                # Extract 2D keypoints and KEEP COCO VISIBILITY!
                kp_list = ann["keypoints"]        # list of [x, y, v_coco]
                kps2d = np.array(kp_list, dtype=np.float32)

                # Reconstruct pose from annotation
                q_list = ann["q_target2cam_passive"]
                t_list = ann["t_target2cam_passive"]
                rot = R.from_quat(q_list)        # expects [x, y, z, w]
                rot_matrix = rot.as_matrix()
                trans = np.array(t_list, dtype=np.float32)

                pose_mat = np.eye(4, dtype=np.float32)
                pose_mat[:3, :3] = rot_matrix
                pose_mat[:3, 3] = trans

                # Create Sample.Sample instance
                sample = Sample.Sample(
                    filepath=filepath,
                    image_id_num=image_id_num,
                    unique_image_id_num=image_id_num,
                    image_id=image_id_str,
                    dataset=self,
                    parent_dir=os.path.dirname(filepath),
                    type=subset_type,
                    pose=pose_mat,
                    K=K,
                    filename=filename,
                    keypoints2D=kps2d,
                    bbox=ann["bbox"],
                    image_already_masked=images_already_masked
                )

                samples.append(sample)

            return samples

        # If splitting, expect coco_json_path = (train_path, test_path)
        if train_test_split:
            if not (isinstance(coco_json_path, (list, tuple)) and len(coco_json_path) == 2):
                raise ValueError(
                    "When train_test_split=True, coco_json_path must be a tuple/list of two paths: "
                    "(train_coco.json, test_coco.json)"
                )

            train_path, test_path = coco_json_path

            with open(train_path, 'r') as f_train:
                coco_train = json.load(f_train)
            with open(test_path, 'r') as f_test:
                coco_test = json.load(f_test)

            self.posed_samples = build_samples_from_coco(coco_train, subset_type="train")
            self.unposed_samples  = build_samples_from_coco(coco_test,  subset_type="test")
            self.samples = self.posed_samples

            # Store intrinsics & image size from train COCO
            # TODO make this dynamic
            if len(self.posed_samples) > 0:
                self.K = np.array(coco_train["info"]["camera_k"], dtype=np.float32)
                self.width  = coco_train["images"][0]["width"]
                self.height = coco_train["images"][0]["height"]

        else:
            # Single COCO file → all samples go into posed_samples
            with open(coco_json_path, 'r') as f:
                coco_all = json.load(f)

            self.posed_samples = build_samples_from_coco(coco_all, subset_type="train")
            self.unposed_samples = []
            self.samples = self.posed_samples

            if len(self.posed_samples) > 0:
                # TODO make this dynamic
                self.K      = np.array(coco_all["info"]["camera_k"], dtype=np.float32)
                self.width  = coco_all["images"][0]["width"]
                self.height = coco_all["images"][0]["height"]

        # Build KDTree on camera centers from posed_samples
        if len(self.posed_samples) > 0:
            camera_centers = [s.C for s in self.posed_samples]  # assume Sample.Sample.C holds camera center
            self.tree = KDTree(camera_centers)

            # Build KNN for BDD NN search
            # Create the NearestNeighbors object:
            self.bdd_nbrs = NearestNeighbors(
                n_neighbors=1,           # how many neighbors you want back
                algorithm='brute',        # brute‐force is required for a callable metric
                metric=metrics.BDD_flat_fast
            ) 
            rotations = np.array([s.R.reshape(1,9) for s in self.posed_samples]).squeeze()
            self.bdd_nbrs.fit(rotations)

            # Also load mesh, handles dataset related stuff
            self._load_mesh()

            # Calculate translation vector for camera that fits the mesh in the field
            # of view for each attitude
            self.t_to_fit_mesh = [0,0,self._get_t_vec_to_fit_mesh()]

            # Setup neural network
            self.model = setup_model(
                cfg_file=self.model_config_path,
                weights_file=self.model_path,
                device="cuda"  # or "cpu"
            )

        return
    
    def nn(self,
           query_pose: np.ndarray = None,
           distance_measure: str = "BDD",
           nn_with_pose: bool = True,
           non_zero_distance: bool = False,
           source_index: int = None,
           return_nn_ix: bool = False,
           return_distance: bool = False):
        """
        Gets the nearest neighbour from the dataset w.r.t. to a query sample in terms of pose.

        Args:
            query_pose: Passive w2c pose of query sample. Needs to be 3x3 rotm or 4x4 pose.
            distance_measure: Method of computing the nearest neighbour. \n
                Euclidean: distance of camera centers (L2-norm) in world coordinates (default). \n
                BDD: boresight deviation distance. 
            nn_with_pose: Whether or not to only consider possible NN's that have
                a pose label (training samples). True by default.
            non_zero_distance: Whether or not to return NN's with a very small distance.
                This is not recommended because of low safety. Instead, prune the dataset
                or use self_index to get rid of just the source sample itself.
            source_index: Index of the source sample (the self) for which you are searching
                the NN. This is needed to return the sample itself and a lot safer than specifying
                a minimum distance.
            return_nn_ix: Whether or not to return the resulting index of the NN.
            return_distance: Whether or not to return the distance to the NN.

        Returns:
            tuple:
            1. nn
            2. nn, nn_index
            3. nn, distval
            4. nn, nn_index, distval
        """
        if nn_with_pose:
            nn_space = self.posed_samples
        else:
            nn_space = self.samples
        distval = np.nan

        if distance_measure == "Euclidean":
            # Fast lookup in O(log n) time with KDTree
            # If arg is given (default) we search until we find a nn with a distance
            # greater than zero, this is to not pick 1) the sample itself, 2) the 90
            # degree rotations for AIRBUS, 3) the two different versions of each pose
            # for SHIRT
            if non_zero_distance:
                nn_dist = 0
                k = 0
                while np.isclose(nn_dist, 0, atol=1e-6):
                    k += 1
                    distances, indices = self.tree.query(query_pose,k=k)
                    # Add scalar check because when k=1 only one scalar is returned
                    nn_dist = distances[k-1] if not np.isscalar(distances) else distances
                nn_index = indices[k-1] if not np.isscalar(indices) else indices

            elif source_index is not None:
                _, nn_indices = self.tree.query(query_pose,k=2)
                nn_index = nn_indices[nn_indices!=source_index][0]
            else:
                _, nn_index = self.tree.query(query_pose,k=1)

        elif distance_measure == "BDD":
            rsamplerot = query_pose[:3,:3]
            # Not return sample itself
            if source_index is not None:
                distances, indices = self.bdd_nbrs.kneighbors(rsamplerot.reshape(1, -1), n_neighbors=2)
                nn_index = indices[indices!=source_index][0]
                distval = distances[indices!=source_index][0]
            # Arbitrary NN
            else:
                distances, indices = self.bdd_nbrs.kneighbors(rsamplerot.reshape(1, -1), n_neighbors=1)
                nn_index = int(indices[0][0])
                distval = float(distances[0][0])
                
        else:
            raise NotImplementedError(f"Your given distance measure {distance_measure} "+\
                                      "is not implemented.")
        if return_distance and distance_measure == "Euclidean":
            raise NotImplementedError
        
        nn = nn_space[nn_index]
        if return_nn_ix and return_distance:
            return nn,nn_index, distval
        elif return_nn_ix:
            return nn, nn_index
        elif return_distance:
            return nn, distval
        else:
            return nn
    
    def synthesize_traj(self, poses):
        """
        Snthesizes a trajectory from a set of poses and the dataset.
        """
        if not hasattr(self,"K") or not hasattr(self,"posed_samples"):
            raise ValueError("Please load dataset first.")
        
        imgs = []
        for pose in poses:
            active = np.linalg.inv(pose)
            C = active[:3,3]
            NN = self.nn(query_pose=pose)
            synth_img = Transform3D(
                source=NN,
                target=pose,
                interpolate=True,
                mask_input=True,
                maskOutput=True,
                with_kps=True,
                with_axes=True,
                target_synth=True,
            )
            imgs.append(synth_img)
        return 
        
    def _load_speed_K(self):
        """
        Loads the camera matrix for SPEED.
        """
        # Load intrinsic matrix
        # SPEED camera data from utils repo
        fx = 0.0176  # focal length[m]
        fy = 0.0176  # focal length[m]
        
        # Image size
        # NOTE I'm kinda worried that this should be at sample-level
        self.width = 1920
        self.height = 1200

        nu = self.width  # number of horizontal[pixels]
        nv = self.height  # number of vertical[pixels]
        ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
        ppy = ppx  # vertical pixel pitch[m / pixel]
        fpx = fx / ppx  # horizontal focal length[pixels]
        fpy = fy / ppy  # vertical focal length[pixels]
        k = [[fpx,   0, nu / 2],
            [0,   fpy, nv / 2],
            [0,     0,      1]]
        self.K = np.array(k, dtype=np.float32)

    def _load_speed(self):
        """
        Loads SPEED dataset from four JSON files: real.json, real_test.json, test.json, train.json.
        """
        # Load intrinsic matrix
        self._load_speed_K()

        # SPEED is a random type dataset
        self.type = "random"

        json_files = ['real.json', 'real_test.json', 'test.json', 'train.json']

        # Split datasets image dirs have same names as json files
        # The split_image_reldir should always be relative to self.image_rootdir_path
        split_image_reldirs = [file[:-5] for file in json_files] #remove .json extension 

        # Load 3D mesh
        # This avoids having to load it every time the depth map is rendered
        # in one of the samples
        self._load_mesh()

        # Count number of added samples
        n_samples = 0

        for split_image_reldir, json_file in zip(split_image_reldirs, json_files):
            # Stop if we exceed number of requested samples 
            # (except if self.N_samples == -1, which is default)
            if n_samples >= self.N_samples and self.N_samples != -1:
                break

            print(f"Loading split {split_image_reldir}.")

            # Split dataset directory
            parent_dir = os.path.join(self.image_rootdir_path, split_image_reldir)
            
            labels_filepath = os.path.join(self.base_path, json_file)
            if not os.path.exists(labels_filepath):
                continue
            with open(labels_filepath, 'r') as f:
                pose_dict = json.load(f)
            
            for ix, entry in enumerate(pose_dict):
                filename = entry["filename"]
                image_id_text, format = os.path.splitext(os.path.basename(filename))
                image_id_num = int(''.join(char for char in image_id_text if char.isdigit()))
                image_id = str(image_id_num) # ID numbers are unique for SPEED
                image_path = os.path.join(self.image_rootdir_path, split_image_reldir, filename)

                # Compute GT (only for train datasets)
                if "test" not in split_image_reldir:
                    # Read quaternion (qw, qx, qy, qz) and translation (tx, ty, tz)
                    # Speed quaternions are scalar-first, see https://gitlab.com/EuropeanSpaceAgency/speed-utils/-/blob/master/utils.py?ref_type=heads
                    # Speed transformation is passive world2cam (like OpenCV and PnP)
                    quat = np.array(entry["q_vbs2tango"], dtype=np.float32)
                    trans = np.array(entry["r_Vo2To_vbs_true"], dtype=np.float32)

                    # scipy expects quaternions in (x, y, z, w) order
                    quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)
                    rot = R.from_quat(quat_xyzw)
                    rot_matrix = rot.as_matrix()

                    # Build 4x4 pose matrix
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :3] = rot_matrix
                    pose[:3, 3] = trans

                    sample = Sample.Sample(filepath=image_path, 
                                    image_id_num=image_id_num,
                                    unique_image_id_num=image_id_num, #unique for SPEED
                                    image_id=image_id,
                                    dataset=self,
                                    parent_dir=parent_dir,
                                    type="train",
                                    pose=pose,
                                    K=self.K,
                                    filename=filename)
                    self.posed_samples.append(sample)

                else:
                    sample = Sample.Sample(filepath=image_path, 
                                    image_id_num=image_id_num,
                                    image_id=image_id,
                                    dataset=self,
                                    parent_dir=parent_dir,
                                    type="test",
                                    K=self.K,
                                    filename=filename)
                    self.unposed_samples.append(sample)

                self.samples.append(sample)
                n_samples += 1


    def _load_speed_plus(self,synthetic_only=False):
        """
        Loads SPEED PLUS dataset from three splits: lightbox, synthetic, sunlamp.

        Args:
            synthetic_only: If True, doesn't load sunlamp and lightbox.
        """
        # Load intrinsic matrix
        # SPEED+ camera data from json
        camera_filepath = os.path.join(self.base_path, 'camera.json')
        with open(camera_filepath, 'r') as f:
            camera_dict = json.load(f)
        
        k = camera_dict["cameraMatrix"]
        self.K = np.array(k, dtype=np.float32)
        # Image size
        # NOTE I'm kinda worried that this should be at sample-level
        self.width = 1920
        self.height = 1200

        # SPPED+ is a random type dataset
        self.type = "random"

        # SPEED+ has 3 lighting splits
        if synthetic_only:
            splits = [
                # we have to keep same indeces (2,3) because the masked images
                # were named this way 
                ('synthetic','train.json',2),
                ('synthetic','validation.json',3)
            ]
        else:
            splits = [
                ('lightbox','test.json',0),
                ('sunlamp','test.json',1),
                ('synthetic','train.json',2),
                ('synthetic','validation.json',3)
            ]

        # Load 3D mesh
        # This avoids having to load it every time the depth map is rendered
        # in one of the samples
        self._load_mesh()

        n_samples = 0
        for split in splits:
            # Stop if we exceed number of requested samples 
            # (except if self.N_samples == -1, which is default)
            if n_samples >= self.N_samples and self.N_samples != -1:
                break

            # The split_image_reldir should always be relative to self.image_rootdir_path
            split_name = split[0]
            split_image_reldir = os.path.join(split_name,"images")

            # Split ID to encode it into the non-unique image numbers
            split_id = split[2]
            
            print(f"Loading split {split_image_reldir}.")

            # Split dataset directory, here the depth and masks folders will be created
            parent_dir = os.path.join(self.image_rootdir_path,split_name)
            
            # Json filename is in splits tuples
            json_file = split[1]


            # Ground truth file
            labels_filepath = os.path.join(self.base_path, split_name, json_file)

            if not os.path.exists(labels_filepath):
                continue
            with open(labels_filepath, 'r') as f:
                pose_dict = json.load(f)
            
            # Populate samples
            for ix, entry in enumerate(pose_dict):
                # Name of image file in GT labels
                filename = entry["filename"]
                image_path = os.path.join(self.image_rootdir_path, split_image_reldir, filename)

                image_id_text, extension = os.path.splitext(os.path.basename(filename))
                image_id_num = int(''.join(char for char in image_id_text if char.isdigit()))

                # ID numbers are NOT unique for SPEED+, therefore we add specific image ID
                # Split name is unique! Even though synthetic contains two label files, they
                # have mutually exclusive image indices
                image_id = '_'.join([f"{image_id_num:06d}",split_name])

                # Unique ID encodes trajectory information through split_id 
                unique_image_id_num = int(f"{image_id_num}{split_id}")

                # Compute GT 
                # Read quaternion (qw, qx, qy, qz) and translation (tx, ty, tz)
                # TODO: verify that speed+ quaternions are scalar-first
                # TODO: verify that speed+ transformation is passive world2cam (like OpenCV and PnP)
                quat = np.array(entry["q_vbs2tango_true"], dtype=np.float32)
                trans = np.array(entry["r_Vo2To_vbs_true"], dtype=np.float32)

                # scipy expects quaternions in (x, y, z, w) order
                quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)
                rot = R.from_quat(quat_xyzw)
                rot_matrix = rot.as_matrix()

                # Build 4x4 pose matrix
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = rot_matrix
                pose[:3, 3] = trans

                # For SPEED+ all samples have an associated GT label, so we treat 
                # them all as "test" data even though some where intended for testing
                sample = Sample.Sample(filepath=image_path, 
                                image_id_num=image_id_num,
                                unique_image_id_num=unique_image_id_num,
                                image_id=image_id,
                                dataset=self,
                                parent_dir=parent_dir,
                                type="train",
                                pose=pose,
                                K=self.K,
                                filename=filename)
                self.samples.append(sample)
                n_samples += 1
        # For BC
        self.posed_samples = self.samples
        self.unposed_samples = self.samples

        
    def _load_airbus_K(self):
        """
        Loads K for the AIRBUS-MAN-L1 dataset.
        """
        # AIRBUS camera data from json
        camera_filepath = os.path.join(self.base_path, 'metadata', 'camera.json')
        with open(camera_filepath, 'r') as f:
            camera_dict = json.load(f)
        
        fx = camera_dict["focalLengthX"]  # focal length[m]
        fy = camera_dict["focalLengthY"]  # focal length[m]
        cx = camera_dict["principalPointX"]
        cy = camera_dict["principalPointY"]
        k = [[fx,  0, cx],
             [0,  fy, cy],
             [0,  0,  1]]
        self.K = np.array(k, dtype=np.float32)
        # Image size
        self.width = 1024
        self.height = 1024

    def _load_airbus(self):
        """
        Loads AIRBUS-MAN-L1 dataset.
        """
        # Load intrinsic martix
        self._load_airbus_K()

        # AIRBUS is a random type dataset
        self.type = "random"

        json_file = 'annotations.json'

        # Load 3D mesh
        self._load_mesh()

        # Split dataset directory
        parent_dir = self.image_rootdir_path
        
        labels_filepath = os.path.join(self.base_path, 'metadata', json_file)

        with open(labels_filepath, 'r') as f:
            pose_dict = json.load(f)
        
        n_samples = 0
        for ix, entry in enumerate(pose_dict):
            # Stop if we exceed number of requested samples 
            # (except if self.N_samples == -1, which is default)
            if n_samples >= self.N_samples and self.N_samples != -1:
                break

            filename = entry["image"]
            image_id_text, format = os.path.splitext(os.path.basename(filename))
            image_id_num = int(''.join(char for char in image_id_text if char.isdigit()))
            image_id = str(image_id_num) # ID numbers are unique for AIRBUS
            image_path = os.path.join(self.image_rootdir_path, filename)

            # Compute GT (every sample has it)
            # AIRBUS transformation is passive world2cam (like OpenCV and PnP)
            # Quaternion already in scalar-last form
            pose = entry["pose"]
            trans = np.array([pose["x"],pose["y"],pose["z"]], dtype=np.float32)
            quat = np.array([pose["qx"],pose["qy"],pose["qz"],pose["qr"]], dtype=np.float32)

            # scipy expects quaternions in (x, y, z, w) order 
            rot = R.from_quat(quat)
            rot_matrix = rot.as_matrix()

            # Build 4x4 pose matrix
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rot_matrix
            pose[:3, 3] = trans

            sample = Sample.Sample(filepath=image_path, 
                            image_id_num=image_id_num,
                            unique_image_id_num=image_id_num, #image_id_num unique for AIRBUS
                            image_id=image_id,
                            dataset=self,
                            parent_dir=parent_dir,
                            type="train",
                            pose=pose,
                            K=self.K,
                            filename=filename,
                            lightpos=entry["light"],
                            image_already_masked=True)

            self.samples.append(sample)
            n_samples += 1
        # For BC
        self.posed_samples = self.samples
        self.unposed_samples = self.samples

    def _load_shirt_K(self):
        # Image size
        # NOTE I'm kinda worried that this should be at sample-level
        self.width = 1920
        self.height = 1200

        # Load intrinsic matrix
        # SHIRT camera data from json
        camera_filepath = os.path.join(self.base_path, 'camera.json')
        with open(camera_filepath, 'r') as f:
            camera_dict = json.load(f)
        
        k = camera_dict["cameraMatrix"]
        self.K = np.array(k, dtype=np.float32)

    def _load_shirt(self):
        """
        Loads SHIRT dataset from two trajectories roe1 and roe2.
        """
        # Load intrinsics
        self._load_shirt_K()

        # SHIRT is a trajectory type dataset
        self.type = "traj"

        json_file = 'groundTruth.json'

        # SHIRT has 2 lighting splits for 2 trajectories so 4 in total
        splits = [
            ('roe1', 'synthetic', os.path.join('roe1', 'synthetic'), 0),
            ('roe2', 'synthetic', os.path.join('roe2', 'synthetic'), 1),
            ('roe1', 'lightbox', os.path.join('roe1', 'lightbox'), 2),
            ('roe2', 'lightbox', os.path.join('roe2', 'lightbox'), 3),
        ]

        # Load 3D mesh
        self._load_mesh()

        n_samples = 0
        for split in splits:
            # Stop if we exceed number of requested samples 
            # (except if self.N_samples == -1, which is default)
            if n_samples >= self.N_samples and self.N_samples != -1:
                break

            traj = split[0]
            lighting = split[1]
            
            # For unique image ID's
            split_id = split[3]

            # The split_image_reldir should always be relative to self.image_rootdir_path
            split_image_reldir = os.path.join(split[2],"images")
            
            print(f"Loading split {split_image_reldir}.")

            # Split dataset directory
            parent_dir = os.path.join(self.image_rootdir_path, split_image_reldir)
            
            # Ground truth file
            labels_filepath = os.path.join(self.base_path, traj, json_file)

            if not os.path.exists(labels_filepath):
                continue
            with open(labels_filepath, 'r') as f:
                pose_dict = json.load(f)
            
            # Populate samples
            for ix, entry in enumerate(pose_dict):
                filename = entry["filename"]
                image_id_text, extension = os.path.splitext(os.path.basename(filename))
                image_id_num = int(''.join(char for char in image_id_text if char.isdigit()))
                # ID numbers are NOT unique for SHIRT
                image_id = '_'.join([f"{image_id_num:06d}",traj,lighting])
                unique_image_id_num = int(f"{image_id_num}{split_id}")
                image_path = os.path.join(self.image_rootdir_path, split_image_reldir, filename)

                # Compute GT 
                # Read quaternion (qw, qx, qy, qz) and translation (tx, ty, tz)
                # TODO: verify that shirt quaternions are scalar-first
                # TODO: verify that shirt transformation is passive world2cam (like OpenCV and PnP)
                quat = np.array(entry["q_vbs2tango_true"], dtype=np.float32)
                trans = np.array(entry["r_Vo2To_vbs_true"], dtype=np.float32)

                # scipy expects quaternions in (x, y, z, w) order
                quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)
                rot = R.from_quat(quat_xyzw)
                rot_matrix = rot.as_matrix()

                # Build 4x4 pose matrix
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = rot_matrix
                pose[:3, 3] = trans

                sample = Sample.Sample(filepath=image_path, 
                                image_id_num=image_id_num,
                                unique_image_id_num=unique_image_id_num,
                                image_id=image_id,
                                dataset=self,
                                parent_dir=parent_dir,
                                type="train",
                                pose=pose,
                                K=self.K,
                                filename=filename)
                self.samples.append(sample)
                n_samples += 1
        # For BC
        self.posed_samples = self.samples
        self.unposed_samples = self.samples

    def _load_swisscube(self):
        """
        Loads SWISSCUBE dataset from a lot trajectories.
        """
        # SWISSCUBE is a trajectory type dataset
        self.type = "traj"

        # Image size
        # NOTE I'm kinda worried that this should be at sample-level
        self.width = 1024
        self.height = 1024

        json_file = 'scene_gt.json'
        camera_file = 'scene_camera.json'
        bbox_file = 'scene_gt_info.json'

        # SWISSCUBE has a very complex folder structure
        splits = [
            *[(os.path.join("training",f"seq_{i:06d}"),i) for i in range(350)],
            *[(os.path.join("validation",f"seq_{i:06d}"),i) for i in range(350,400)],
            *[(os.path.join("testing",f"seq_{i:06d}"),i) for i in range(400,500)]
        ]
        # Each image folderpath ends with 000000
        # There are also masks already included
        splits = [(os.path.join(path[0],"000000"),path[1]) for path in splits]

        # Load 3D mesh
        self._load_mesh()

        n_samples = 0
        # ======= TRAJECTORY LEVEL =======
        for split in splits:
            # Stop if we exceed number of requested samples 
            # (except if self.N_samples == -1, which is default)
            if n_samples >= self.N_samples and self.N_samples != -1:
                break

            # Trajectory path and number
            traj_path = split[0]
            traj_id = split[1]

            # The split_image_reldir should always be relative to self.image_rootdir_path
            split_image_reldir = os.path.join(traj_path,"rgb")
            split_mask_reldir = os.path.join(traj_path,"mask_visib")
            
            print(f"Loading split {split_image_reldir}.")

            # Split dataset directory
            # Parent dir should ideally be above the image dir but also be unique
            # (which is not possible for datasets where multiple splits share the exact
            # same parent folder)
            parent_dir = os.path.join(self.base_path, traj_path)
            
            # Ground truth file
            labels_filepath = os.path.join(self.base_path, traj_path, json_file)

            if not os.path.exists(labels_filepath):
                continue
            with open(labels_filepath, 'r') as f:
                pose_dict = json.load(f)

            # SWISSCUBE camera data is PER SAMPLE
            camera_filepath = os.path.join(self.base_path, traj_path, 'scene_camera.json')
            with open(camera_filepath, 'r') as f:
                camera_dict = json.load(f)
            
            
            # ======= SAMPLE LEVEL =======
            for ix, entry in enumerate(pose_dict):
                # Load per-sample intrinsic matrix
                k = np.array(camera_dict[entry]["cam_K"], dtype=np.float32).reshape((3,3))
                # We load it into the dataset only once (not unique!!!)
                if not hasattr(self,"K"):
                    self.K = np.array(k, dtype=np.float32)
                
                # In Swisscube pose_dict is literally a dictionary, so entry is a key
                # and by chance also the same as the image ID (which is 1-indexed!)
                image_id_num = int(entry)

                # ID numbers are NOT unique for SWISSCUBE, image_id must be unique though
                # so we attach trajectory info 
                image_id = '_'.join([f"{image_id_num:06d}",f"{traj_id:06d}"])
                
                # Unique umeric identifier (needed for COCO)
                unique_image_id_num = int(f"{image_id_num}{traj_id}")

                # Construct filename based on image_id (filename not in labels file)
                filename = f"{image_id_num:06d}{self.extension}"
                mask_filename = f"{image_id_num:06d}_000000.png" #masks are in png, imgs in jpg
                image_path = os.path.join(self.image_rootdir_path, split_image_reldir, filename)
                # TODO: integrate mask path in sample class as an arg
                mask_path = os.path.join(self.image_rootdir_path, split_mask_reldir, mask_filename)

                # Compute GT 
                # Read quaternion (qw, qx, qy, qz) and translation (tx, ty, tz)
                # TODO: verify that shirt quaternions are scalar-first
                # TODO: verify that shirt transformation is passive world2cam (like OpenCV and PnP)
                gt = pose_dict[entry][0]
                rot_matrix = np.array(gt["cam_R_m2c"], dtype=np.float32).reshape((3,3))

                # Distance units for SWISSCUBE given in mm
                # TODO: Verify this
                trans = np.array(gt["cam_t_m2c"], dtype=np.float32)/1000 

                # Build 4x4 pose matrix
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = rot_matrix
                pose[:3, 3] = trans

                # In Swisscube each entry has GT so we treat each as
                # a "train" sample even when they're in the e.g. test folder
                sample = Sample.Sample(filepath=image_path, 
                                image_id_num=image_id_num,
                                unique_image_id_num=unique_image_id_num,
                                image_id=image_id,
                                dataset=self,
                                parent_dir=parent_dir,
                                type="train",
                                pose=pose,
                                K=self.K,
                                filename=filename,
                                mask_path=mask_path)
                self.samples.append(sample)
                n_samples += 1
    
        # For BC
        self.posed_samples = self.samples
        self.unposed_samples = self.samples

        
    def _load_urso(self):
        """
        Loads URSO dataset from two subsets soyuz_easy and soyuz_hard. No dragon.
        """
        # Load intrinsic matrix
        # URSO camera data from website
        k = [
            [640, 0, 640],
            [0, 640.46, 480],
            [0, 0, 1]
        ]
        self.K = np.array(k, dtype=np.float32)

        # Image size
        # NOTE I'm kinda worried that this should be at sample-level
        self.width = 1280
        self.height = 960

        # URSO is a random type dataset
        self.type = "random"

        # URSO has 3 splits for the 2 datasets so 6 in total
        splits = [
            ("soyuz_easy","soyuz_easy/images","train",0),
            ("soyuz_easy","soyuz_easy/images","test",1),
            ("soyuz_easy","soyuz_easy/images","val",2),
            ("soyuz_hard","soyuz_hard/images","train",3),
            ("soyuz_hard","soyuz_hard/images","test",4),
            ("soyuz_hard","soyuz_hard/images","val",5)
        ]

        # Load 3D mesh
        self._load_mesh()

        n_samples = 0
        for split in splits:
            # Stop if we exceed number of requested samples 
            # (except if self.N_samples == -1, which is default)
            if n_samples >= self.N_samples and self.N_samples != -1:
                break

            split_name = split[2]
            # The split_image_reldir should always be relative to self.image_rootdir_path
            split_image_reldir = split[1]
            split_type = split[0]

            # For unique image ID's
            split_id = split[3]
            
            print(f"Loading split {split_image_reldir}.")

            # Split dataset directory
            parent_dir = os.path.join(self.image_rootdir_path, split_image_reldir)
            
            # Ground truth file
            csv_file = f"{split_name}_poses_gt.csv"
            labels_filepath = os.path.join(self.base_path, split_type, csv_file)

            # Images are not in chronological order but given in *_images.csv files
            images_file = f"{split_name}_images.csv"
            images_filepath = os.path.join(self.base_path, split_type, images_file)
            if not os.path.exists(labels_filepath) or not os.path.exists(images_filepath):
                print(f"File {images_filepath} or {labels_filepath} does not exist.")
                continue

            # Read from csv files
            pose_array = np.genfromtxt(labels_filepath, delimiter=',')
            images_array = np.genfromtxt(images_filepath, delimiter=',',dtype=str)
            # Remove header row (not for images csv)
            pose_array = pose_array[1:]

            # Populate samples
            for ix, entry in enumerate(pose_array):
                # Get image filename from *_images.csv file 
                filename = images_array[ix]
                image_id_num = int(filename[:-8]) # id from start of image filename
                # ID numbers are NOT unique for URSO
                image_id = '_'.join([f"{image_id_num:06d}",split_name])
                unique_image_id_num = int(f"{image_id_num}{split_id}")
                image_path = os.path.join(self.image_rootdir_path, split_image_reldir, filename)

                # Compute GT 
                # Read quaternion (qx, qy, qz, qw) and translation (tx, ty, tz)
                quat = entry[3:]
                trans = entry[:3]

                # scipy expects quaternions in (x, y, z, w) order
                quat_xyzw = np.array(quat, dtype=np.float32)
                rot = R.from_quat(quat_xyzw)
                rot_matrix = rot.as_matrix()

                # Build 4x4 pose matrix
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = rot_matrix
                pose[:3, 3] = trans
                
                sample = Sample.Sample(filepath=image_path, 
                                image_id_num=image_id_num,
                                unique_image_id_num=unique_image_id_num,
                                image_id=image_id,
                                dataset=self,
                                parent_dir=parent_dir,
                                type="train",
                                pose=pose,
                                K=self.K,
                                filename=filename)
                self.samples.append(sample)
                n_samples += 1
        # For BC
        self.posed_samples = self.samples
        self.unposed_samples = self.samples

    def get_rand_pose_to_fit_mesh(self, radius_margin: float = 1.1) -> np.ndarray:
        """
        Returns a pose with a random rotation and a translation that fits the mesh into
        the field of view.
        """
        new_rot = R.random().as_matrix()
        new_pose = self.construct_pose(new_rot,self.t_to_fit_mesh)
        return new_pose

    def _get_t_vec_to_fit_mesh(self, radius_margin: float = 1.1) -> float:
        """
        Computes how far along the camera’s optical axis we need to place the mesh’s origin
        so that a bounding sphere (centered at the origin, radius = base_radius * margin)
        fits completely within the image’s field of view. Returns that distance `d`.

        Internally:
        1. Find the mesh’s base radius: max distance from origin to any vertex.
        2. Multiply by `radius_margin` to get a padded radius R_margin.
        3. Let (fx, fy) = focal lengths from self._K, and (cx, cy) = principal point.
            Let (W, H) = image width/height from self.dataset.
        4. The largest allowable projected radius (in pixels) horizontally is
            rmax_x = min(cx, W - cx)
            and vertically is
            rmax_y = min(cy, H - cy).
        5. For a sphere of radius R_margin at Z = d, a point at X = R_margin, Y=0, Z=d
            projects to x_px = fx * (R_margin / d).  We require
            fx * (R_margin / d) <= rmax_x
            => d >= fx * R_margin / rmax_x.
            Similarly, d >= fy * R_margin / rmax_y.
        6. So we take
            d = max(fx * R_margin / rmax_x,   fy * R_margin / rmax_y).
        7. Return that `d`.

        Args:
            radius_margin (float): Multiplier on the base radius to add padding.
                                E.g. 1.1 gives 10% extra. Must be > 1.0 in practice.

        Returns:
            float: The distance along the camera’s Z‐axis so that the padded sphere
                fits entirely in the image.  (I.e. if the mesh’s origin is placed at
                (0,0,d) in camera‐coords, no vertex will project outside.)
        """

        # 1) Ensure mesh + intrinsics + image size exist
        if not hasattr(self, "mesh") or self.mesh is None:
            raise ValueError("Cannot compute fit‐distance: dataset has no `mesh` attribute.")
        if not hasattr(self, "K") or self.K is None:
            raise ValueError("Cannot compute fit‐distance: dataset has no intrinsics (`self._K`).")
        if not hasattr(self, "width") or not hasattr(self, "height"):
            raise ValueError("Cannot compute fit‐distance: dataset must have `width` and `height`.")

        # 2) Grab all vertices of the mesh (shape = (N, 3)) and compute base_radius
        vertices = self.mesh.vertices
        if vertices.size == 0:
            raise ValueError("Mesh contains no vertices.")
        base_radius = float(np.max(np.linalg.norm(vertices, axis=1)))

        # 3) Apply margin
        R_margin = base_radius * radius_margin

        # 4) Unpack intrinsics and image dims
        K = self.K
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        W, H = float(self.width), float(self.height)

        # 5) Compute maximum allowable projected radius in pixels
        rmax_x = min(cx, W - cx)
        rmax_y = min(cy, H - cy)

        if rmax_x <= 0 or rmax_y <= 0:
            raise ValueError(f"Invalid principal point or image size: rmax_x={rmax_x}, rmax_y={rmax_y}")

        # 6) Find d that satisfies both horizontal and vertical constraints:
        #    fx * (R_margin / d) <= rmax_x   =>  d >= fx * R_margin / rmax_x
        #    fy * (R_margin / d) <= rmax_y   =>  d >= fy * R_margin / rmax_y
        d_x = fx * R_margin / rmax_x
        d_y = fy * R_margin / rmax_y

        d = float(max(d_x, d_y))

        return d


    @classmethod
    def construct_pose(self, rotm, tvec):
        """
        Convenience function.
        """
        pose = np.eye(4,dtype=np.float32)
        pose[:3,:3] = rotm
        pose[:3,3] = tvec
        return pose
        
    def rand(self, with_bg = False) -> Optional[Sample.Sample]:
        """
        Returns a random sample from the dataset.
        """
        if not self.samples:
            raise ValueError("First load the dataset.")
        
        sample = np.random.choice(self.samples)

        if with_bg:
            img = sample.image
            if np.median(img.flatten()) < self.BG_threshold:
                return self.rand(with_bg = True)

        return sample

    def rand_train(self, with_bg = False) -> Optional[Sample.Sample]:
        """
        Returns a random training sample (with pose) from the dataset.
        """
        if not self.samples:
            raise ValueError("First load the dataset.")
        sample = np.random.choice(self.posed_samples)

        if with_bg:
            img = sample.image
            if np.median(img.flatten()) < self.BG_threshold:
                return self.rand_train(with_bg = True)

        return sample
    
    def rand_test(self, with_bg = False) -> Optional[Sample.Sample]:
        """
        Returns a random test sample (without pose) from the dataset.
        """
        if not self.samples:
            raise ValueError("First load the dataset.")
        sample = np.random.choice(self.unposed_samples)

        if with_bg:
            img = sample.image
            if np.median(img.flatten()) < self.BG_threshold:
                return self.rand_test(with_bg = True)
        
        return sample
    
    def rotations(self, flattened=False):
        """
        Convenience for returning rotation matrices of training samples.
        """
        if flattened:
            rots = np.array([s.R.reshape(1,-1) for s in self.posed_samples]).squeeze()
        else:
            rots = np.array([s.R for s in self.posed_samples])
        return rots
    
    def poses(self, flattened=False):
        """
        Convenience for returning pose matrices of training samples.
        """
        if flattened:
            poses = np.array([s.pose.reshape(1,-1) for s in self.posed_samples]).squeeze()
        else:
            poses = np.array([s.pose for s in self.posed_samples])
        return poses
    
    def export_labels_to_coco(self, 
                              random_train_test_split=False, 
                              manual_train_test_split=False,
                              train_indices: list = None,
                              test_indices: list = None,
                              masked_images=False):
        """
        Utility function for writing dataset labels to COCO format.

        Useful for training pose estimation Neural Networks.

        Args:
            random_train_test_split (bool): Whether or not to split the labelled data samples into
                a train and a test dataset. If True, the split will be 0.9 train, 0.1 test.
                In that case, the function returns a dictionary:
                    {"train": <coco_train_dict>, "test": <coco_test_dict>}
                Otherwise (default), it returns a single COCO dict for all posed_samples.
            manual_train_test_split (bool): Whether or not the user specifies the train test 
                indices themselves. If True (False by default) then the user should provide 
                train_indeces and test_indeces which refer to the dataset.posed_samples list.
            masked_images (bool): Whether or not to take the masked images for the labels.
        """
        # Check if we have the necessary data
        if not hasattr(self, "num_keypoints") or not hasattr(self, "keypoints3D"):
            raise ValueError("Please provide the keypoints before attempting to create COCO labels file.")
        
        def build_coco_dict(sample_list):
            """
            Build a COCO-style dictionary from a list of samples.
            """
            coco = {
                "info": {
                    "description": f"COCO labels for {self.dataset_name} dataset",
                    "version": "1.0.0",
                    "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "camera_k": self.K.tolist()
                },
                "licenses": [],
                "categories": [],
                "images": [],
                "annotations": []
            }

            coco["categories"].append({
                "id": 0,
                "name": self.dataset_name,
                "supercategory": "object",
                "keypoints": [str(i) for i in range(self.num_keypoints)],
                "skeleton": [],
                "keypoint_colours": [],
                "num_keypoints": self.num_keypoints
            })

            for sample in sample_list:
                if masked_images and sample.masked_filepath is not None:
                    filepath = sample.masked_filepath
                elif masked_images and sample.masked_filepath is None:
                    raise ValueError(f"You asked to write masked images but this sample {sample.image_id} does "+\
                                     "not have a self.masked_filepath.")
                else:
                    filepath = sample.filepath
                filename = os.path.basename(filepath)

                # We get the relative dir from the base dir to the filedir
                # to be able to export the whole dataset to another machine.
                # The root of all files is the parent dir.
                rel_filepath = os.path.relpath(filepath,start=sample.dataset.base_path)

                # We DO NOT add "images" to the relpath, this is done inside
                # the training script, we are agnostic to this 
                # rel_filepath = os.path.join("images",rel_filepath)

                # Image id expected as an int for NN training
                # THIS MUST BE UNIQUE
                if sample.unique_image_id_num is None or \
                    type(sample.unique_image_id_num) != int:
                    raise ValueError("Please provide a unique (integer) unique_image_id_num for the samples " \
                    "to be able to write COCO label files for this dataset.")
                _coco_image_id = sample.unique_image_id_num


                image_entry = {
                    "id": _coco_image_id,
                    "width": self.width,
                    "height": self.height,
                    "file_name": filename,
                    "filepath": rel_filepath
                }
                coco["images"].append(image_entry)

                if hasattr(sample, "bbox"):
                    bbox = sample.bbox
                    area = sample.bbox_area
                else:
                    raise ValueError("Please provide bboxes and areas to be able " \
                    "to write COCO labels.")

                q_relative = R.from_matrix(sample.pose[:3, :3])
                t_relative = sample.pose[:3, 3]

                # Visible = visibility > 1 (only 2)
                num_visible_keypoints = np.sum(sample.keypoints2D[:,2] > 1)

                coco["annotations"].append({
                    "id": _coco_image_id,
                    "annotation_id": _coco_image_id,
                    "image_id": _coco_image_id,
                    "image_filepath": rel_filepath,
                    "category_id": 0,
                    "bbox": np.array(bbox,dtype=int).squeeze().tolist(), #bbox and area are int in COCO
                    "area": int(area),
                    "keypoints": sample.keypoints2D.tolist(),
                    "num_keypoints": int(num_visible_keypoints),
                    "iscrowd": 0,
                    "q_target2cam_passive": q_relative.as_quat().tolist(),
                    "t_target2cam_passive": t_relative.tolist()
                })

            return coco

        # If no split, just build and return one COCO dict over all posed_samples
        if not random_train_test_split and not manual_train_test_split:
            return build_coco_dict(self.posed_samples)

        elif random_train_test_split and not manual_train_test_split:
            # Otherwise, perform 90/10 split on posed_samples
            all_samples = np.array(self.posed_samples)
            np.random.shuffle(all_samples)  # in-place shuffle
            n_total = len(all_samples)
            n_train = int(np.floor(0.9 * n_total))

            train_subset = all_samples[:n_train].tolist()
            test_subset = all_samples[n_train:].tolist()

            coco_train = build_coco_dict(train_subset)
            coco_test = build_coco_dict(test_subset)

            return {"train": coco_train, "test": coco_test}
        elif manual_train_test_split and not random_train_test_split:
            if train_indices is None or test_indices is None:
                raise ValueError("For a manual split, please provide the indices of the " \
                "train and test samples w.r.t. the dataset.posed_samples array.")
            train_subset = [self.posed_samples[i] for i in train_indices]
            test_subset = [self.posed_samples[i] for i in test_indices]

            coco_train = build_coco_dict(train_subset)
            coco_test = build_coco_dict(test_subset)

            return {"train": coco_train, "test": coco_test}
        else:
            raise ValueError("Either random or manual split, not both.")
    
    
    @classmethod
    def write_data_to_json(cls, data, filename, rel_path="assets"):
        """
        Writes data to json file, by default in assets/ subfolder.
        """
        filepath = os.path.join(os.getcwd(),rel_path,filename)
        dirs = os.path.dirname(filepath)
        os.makedirs(dirs,exist_ok=True)
        with open(filepath,"w") as f:
            json.dump(data, f, indent=4)
        print(f"Wrote data to {filepath}.")


    @classmethod
    def append_data_to_json(cls, new_data, filename, rel_path="assets"):
        """
        Appends the items in `new_data` (a list) to an existing JSON list in the
        given file. If the file doesn't exist yet, it creates it with `new_data`
        as its contents. By default, this operates in the assets/ subfolder.
        """
        filepath = os.path.join(os.getcwd(), rel_path, filename)
        dirs = os.path.dirname(filepath)
        os.makedirs(dirs, exist_ok=True)

        # Load existing content if file exists, otherwise start with an empty list
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        else:
            existing = []

        # Ensure the loaded content is a list, since we’re appending list items
        if not isinstance(existing, list):
            raise ValueError(f"Cannot append to {filepath}: existing JSON is not a list.")

        # Extend and write back
        existing.extend(new_data)
        with open(filepath, "w") as f:
            json.dump(existing, f, indent=4)

        print(f"Appended {len(new_data)} item(s) to {filepath}.")


    def write_all_masked_images(self):
        """
        This writes sample.masked_image for all samples to disk.
        
        Used for training neural networks. Should be called before 
        writing coco labels if you want masked images.
        """
        for sample in self.posed_samples:
            _ = sample.masked_image

    def prune_airbus(self):
        """
        Removes the 90 degree rotations from the AIRBUS dataset.

        This is not undoable! Will only execute once for safety.
        """
        if not self.dataset_name == "AIRBUS":
            raise ValueError("Only call this func for the AIRBUS dataset.")
        
        if not hasattr(self, "pruned") or not self.pruned:
            self.posed_samples = self.posed_samples[::4]
            self.samples = self.samples[::4]
            self.unposed_samples = self.unposed_samples[::4]
            camera_centers = [s.C for s in self.posed_samples]
            self.tree = KDTree(camera_centers)
            rotations = np.array([s.R.reshape(1,-1) for s in self.posed_samples]).squeeze()
            self.bdd_nbrs.fit(rotations)
            self.pruned = True
        elif self.pruned:
            raise ValueError("You already called prune_airbus.")

    def change_to_coco_visibility(self):
        """
        This function converts the keypoints of all train samples to coco (0,1,2).

        Attention, this takes a long time because the depth maps have to be rendered
        to compute the occlusion.
        """
        for sample in self.posed_samples:
            new_kps = sample.keypoints2D[:,:2]
            new_kps = cv_utils.keypoints.add_coco_visibility_to_keypoints(
                keypoints_2d=new_kps,
                keypoints_3d=self.keypoints3D, 
                depth_map=sample.depth_map,
                image_size=(self.width,self.height)
            )
            sample.keypoints2D = new_kps
        
    def write_all_depthmaps(self,overwrite=True):
        """
        This function writes all depth maps to disk for the train samples
        if they do not exist yet.

        Args:
            overwrite (bool): Whether or not to overwrite existing depth maps.
        """
        for sample in self.posed_samples:
            if overwrite:
                sample.load_depth = False
            _ = sample.depth_map

    def synthesize_pose_for_vis(self,
                                target_pose: np.ndarray = None,
                                method: str = "3D",
                                nn_metric: str = "BDD",
                                return_NN: bool = False,
                                return_dist: bool = False):
        """
        Synthesizes the view of a pose with the nearest neighbour. Does not actually
        create a sample object.

        The returned image has axes projected and the function returns only the 
        image and does not actually write it to disk.

        Args:
            target_pose: 4x4 pose of the target to synthesize.
            method: Synthesis method, either "2D" or "3D" (default).
            nn_metric: Distance metric to find nearest neighbour, either "Euclidean" or "BDD" (default). 
            return_NN: Whether or not to return the nearest neighbour sample object;
                False by default.
            return_dist: Whether or not to return the distance to the nearest neighbour
                that was chosen. False by default.
        """
        # Get nearest neighbour (no self exclusion)
        if not return_dist:
            NN = self.nn(query_pose=target_pose,
                             distance_measure=nn_metric,
                             return_distance=False,
                             return_nn_ix=False)
        else:
            NN, dist = self.nn(query_pose=target_pose,
                             distance_measure=nn_metric,
                             return_distance=True,
                             return_nn_ix=False)
        
        if method == "3D":
            synth_img= Transform3D(
                source=NN,
                target=target_pose,
                interpolate=True,
                mask_input=False, # FIXME CHANGE BACK
                maskOutput=True,
                with_kps=False,
                with_axes=True,
                target_synth=True,
                return_transformed_kps=False,
                return_transformed_mask=False
            )
        else:
            raise NotImplementedError(method)
        
        if return_NN and return_dist:
            return synth_img, NN, dist
        elif return_dist:
            return synth_img, dist
        elif return_NN:
            return synth_img, NN
        else:
            return synth_img

    def sample_replacement(self,source,target):
        """
        Synthesizes and evaluates.
        Does NOT apply BDD < 0.5 check.

        Args: 
            source (Sample.Sample): source sample object.
            target (Sample.Sample): target sample object.
        """
        s = source
        t = target
        npd = metrics.BDD(s.pose, t.pose)
        cdist = np.linalg.norm(s.C-t.C)
        adist = cv_utils.core.rotation_angle_between_matrices(s.pose[:3,:3],t.pose[:3,:3])

        # mask the images
        ti = t.masked_image

        synth2D, transformed_keypoints2D, transformed_mask2D = Homography(
            source=s,
            target=t,
            return_transformed_kps=True,
            return_transformed_mask=True,
            mask_input=True,
            with_axes=False,
            with_kps=False
        )
        synth3D, transformed_keypoints3D, transformed_mask3D = Transform3D(
            source=s,
            target=t,
            interpolate=True,
            return_transformed_kps=True,
            return_transformed_mask=True,
            mask_input=True,
            maskOutput=True,
            with_kps=False,
            with_axes=False
        )

        ssim_score2D = metrics.ssim(synth2D, ti)
        ssim_score3D = metrics.ssim(synth3D, ti)
        kps_L22D, kps_oks2D = metrics.keypoint_metrics(transformed_keypoints2D[:, :2], t.keypoints2D[:, :2], visibility=t.keypoints2D[:,2])
        kps_L23D, kps_oks3D = metrics.keypoint_metrics(transformed_keypoints3D[:, :2], t.keypoints2D[:, :2], visibility=t.keypoints2D[:,2])
        iou2D = metrics.IOU(t.mask, transformed_mask2D)
        iou3D = metrics.IOU(t.mask, transformed_mask3D)

        result = {
            "source": s.image_id,
            "target": t.image_id,
            "poses": s.pose.flatten().tolist(),
            "poset": t.pose.flatten().tolist(),
            "npd": float(npd),
            "cdist": float(cdist),
            "adist": float(adist),
            "ssim2D": float(ssim_score2D),
            "ssim3D": float(ssim_score3D),
            "iou2D": float(iou2D),
            "iou3D": float(iou3D),
            "kpsL22D": float(kps_L22D),
            "kpsL23D": float(kps_L23D),
            "oks2D": float(kps_oks2D),
            "oks3D": float(kps_oks3D)
        }
        return result

    def rand_train_ix(self):
        """
        Like rand_train() but returns the index of the chosen sample
        instead of the sample itself.

        Does not apply any background detection like rand_train() does.
        """
        if not self.posed_samples:
            raise ValueError("First load the dataset.")
        sample_ix = np.random.choice([i for i in range(len(self.posed_samples))])

        return sample_ix
    
    def random_spline_trajectory_with_poses(
            self,
            max_len: float = 100,
            near: float = 0.001,
            far: float = 100,
            fineness: int = 200,
            num_ctrl: int = 3,
            seed: int = None,
            plot: bool = False
        ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Generate a smooth random trajectory inside a fixed camera’s view frustum, and
        return:
            1) A list of rotation matrices (3×3) from SLERP (object→camera) at each sample.
            2) A list of translation vectors (length-3) for the object in camera coords.

        The camera is assumed fixed at the origin looking along -Z, with R = I, t = 0.

        Args:
            near, far : Distances for near/far planes in camera coords.
            max_len   : Max allowed distance between trajectory endpoints.
            fineness  : Number of samples along the spline.
            num_ctrl  : Number of random interior control points (in camera coords).
            seed      : RNG seed.
            plot      : If True, show Plotly 3D scene of the frustum + trajectory.

        Returns:
            rots_interp_mats : List of length `fineness` of 3×3 rotation matrices (object→camera).
            traj_cam         : List of length `fineness` of translation vectors (object position in camera coords).
        """
        if seed is not None:
            np.random.seed(seed)

        # Camera is fixed at origin: R_wc = I, t_wc = 0
        R_wc = np.eye(3)
        t_wc = np.zeros(3)

        K = self.K
        width = self.width
        height = self.height

        def backproject(u, v, Z, K_mat):
            """
            Return a 3-vector in camera coords for pixel (u,v) at depth Z.
            Assumes K = [[fx,  0, cx],
                        [ 0, fy, cy],
                        [ 0,  0,  1]].
            """
            fx, fy = K_mat[0, 0], K_mat[1, 1]
            cx, cy = K_mat[0, 2], K_mat[1, 2]
            x_cam = (u - cx) * Z / fx
            y_cam = (v - cy) * Z / fy
            return np.array([x_cam, y_cam, Z])

        # 1) Compute the 8 corners of the view frustum in CAMERA (and WORLD) space
        corners_px = np.array([[0, 0], [width, 0], [width, height], [0, height]])
        corners_cam = np.vstack([
            backproject(u, v, near,  K) for (u, v) in corners_px
        ] + [
            backproject(u, v, far,   K) for (u, v) in corners_px
        ])  # shape (8, 3)

        # Since camera is at origin, corners_world = corners_cam
        corners_world = corners_cam.copy()

        # helper: sample 1 random point inside frustum (in CAMERA coords)
        def sample_point():
            Z = np.random.uniform(near, far)
            u = np.random.uniform(0, width)
            v = np.random.uniform(0, height)
            cam_pt = backproject(u, v, Z, K)
            return cam_pt

        # 2) Pick endpoints A, B in camera coords with ||A - B|| <= max_len
        for _ in range(1000):
            A = sample_point()
            B = sample_point()
            if np.linalg.norm(A - B) <= max_len:
                break
        else:
            raise RuntimeError("Couldn't sample endpoints within max_len")

        # 3) Build control points (including A, B) in camera coords
        control_pts = [A] + [sample_point() for _ in range(num_ctrl)] + [B]
        ctrl_arr = np.vstack(control_pts).T  # shape (3, K)

        # 4) Fit & sample a cubic B‐spline (in camera coords)
        tck, _ = splprep(ctrl_arr, s=0, k=min(3, ctrl_arr.shape[1] - 1))
        u_fine = np.linspace(0, 1, fineness)
        pts = splev(u_fine, tck)
        traj_cam = np.vstack(pts).T   # (fineness, 3)

        # 5) Generate random initial & final orientations (object→camera)
        rot_init = R.random()
        rot_final = R.random()
        key_rots = R.concatenate([rot_init, rot_final])
        key_times = [0.0, 1.0]
        slerp = Slerp(key_times, key_rots)
        rots_interp = slerp(u_fine)   # array of Rotation objects, length = fineness
        rots_interp_mats = [r.as_matrix() for r in rots_interp]

        # Build pose matrices
        poses = []
        for i in range(fineness):
            pose = np.eye(4,dtype=np.float32)
            pose[:3,:3] = rots_interp_mats[i]
            pose[:3,3] = traj_cam[i]
            poses.append(pose)

        # 6) (Optional) visualize frustum, trajectory, and a few object frames
        if plot:
            fig = go.Figure()

            # ——— Frustum edges ———
            # near‐plane loop
            for idx in range(4):
                nxt = (idx + 1) % 4
                fig.add_trace(go.Scatter3d(
                    x=[corners_world[idx, 0], corners_world[nxt, 0]],
                    y=[corners_world[idx, 1], corners_world[nxt, 1]],
                    z=[corners_world[idx, 2], corners_world[nxt, 2]],
                    mode='lines', line=dict(color='gray'), showlegend=False
                ))
            # far‐plane loop
            for idx in range(4, 8):
                nxt = 4 + ((idx + 1 - 4) % 4)
                fig.add_trace(go.Scatter3d(
                    x=[corners_world[idx, 0], corners_world[nxt, 0]],
                    y=[corners_world[idx, 1], corners_world[nxt, 1]],
                    z=[corners_world[idx, 2], corners_world[nxt, 2]],
                    mode='lines', line=dict(color='gray'), showlegend=False
                ))
            # connect near→far
            for idx in range(4):
                fig.add_trace(go.Scatter3d(
                    x=[corners_world[idx, 0], corners_world[idx + 4, 0]],
                    y=[corners_world[idx, 1], corners_world[idx + 4, 1]],
                    z=[corners_world[idx, 2], corners_world[idx + 4, 2]],
                    mode='lines', line=dict(color='gray'), showlegend=False
                ))

            # ——— Trajectory ———
            fig.add_trace(go.Scatter3d(
                x=traj_cam[:, 0], y=traj_cam[:, 1], z=traj_cam[:, 2],
                mode='lines', line=dict(width=4, color='blue'),
                name='Trajectory'
            ))

            # ——— Control points ———
            cp = np.vstack(control_pts)
            fig.add_trace(go.Scatter3d(
                x=cp[:, 0], y=cp[:, 1], z=cp[:, 2],
                mode='markers', marker=dict(size=6, color='red'),
                name='Control Points'
            ))

            # ——— Sample a few object frames ———
            def draw_frame(R_mat, origin, scale=0.1, label=""):
                x_axis = origin + scale * R_mat[:, 0]
                y_axis = origin + scale * R_mat[:, 1]
                z_axis = origin + scale * R_mat[:, 2]
                # X-axis
                fig.add_trace(go.Scatter3d(
                    x=[origin[0], x_axis[0]],
                    y=[origin[1], x_axis[1]],
                    z=[origin[2], x_axis[2]],
                    mode='lines', line=dict(color='red'), showlegend=False
                ))
                # Y-axis
                fig.add_trace(go.Scatter3d(
                    x=[origin[0], y_axis[0]],
                    y=[origin[1], y_axis[1]],
                    z=[origin[2], y_axis[2]],
                    mode='lines', line=dict(color='green'), showlegend=False
                ))
                # Z-axis
                fig.add_trace(go.Scatter3d(
                    x=[origin[0], z_axis[0]],
                    y=[origin[1], z_axis[1]],
                    z=[origin[2], z_axis[2]],
                    mode='lines', line=dict(color='blue'), showlegend=False
                ))
                if label:
                    fig.add_trace(go.Scatter3d(
                        x=[origin[0]], y=[origin[1]], z=[origin[2]],
                        mode='text', text=[label], showlegend=False
                    ))

            for idx, label in zip([0, fineness // 2, fineness - 1], ["start", "mid", "end"]):
                draw_frame(rots_interp_mats[idx], traj_cam[idx], scale=(max_len * 0.05), label=label)

            fig.update_layout(
                scene=dict(
                    xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                    aspectmode='data'
                ),
                title="Random Spline Trajectory and Object Poses (Camera Fixed)"
            )
            fig.show()

        # 7) Return rotations (3×3) and translations (3-vector) in camera coords
        #    rots_interp_mats[i] is R_object→camera at sample i,
        #    traj_cam[i] is t_object in camera coords at sample i.
        return poses

    
def test_synthesis(ds):
    for i in range(100):
        s = ds.rand_train() # ds comes from entry point
        t = s.nn()
        si = s.image_with_axes_and_kps
        ti = t.image_with_axes_and_kps
        d = s.depth_map
        m = s.mask
        start_time = time.time()
        synth3, t_kps3, t_mask3 = Transform3D(source=s,
                             target=t,
                             interpolate=True,
                             with_axes=True,
                             with_kps=True,
                             maskOutput=True,
                             return_transformed_kps=True,
                             return_transformed_mask=True,
                             mask_input=True)
        synth2, t_kps2, t_mask3  = Homography(source=s,
                            target=t,
                            with_axes=True,
                            with_kps=True,
                            return_transformed_kps=True,
                            return_transformed_mask=True,
                            mask_input=True)
        end_time = time.time()
        print(f"Transformation time {end_time - start_time:.4f}s")
        # Blend images wi   th alpha
        alpha = 0.5
        overlay = cv2.addWeighted(ti, alpha, synth3, 1 - alpha, 0)
        print()
    print()

def one_shot(ds,
             mask_input: bool = True):
    """
    Perform a single 'one-shot' transformation and compute evaluation metrics.

    This function:
    1. Randomly samples a source and target pair of data from a dataset (`ds`).
    2. Calculates the Normalized Pose Distance (NPD) between the source and target.
    3. If NPD exceeds 0.5, considers the transformation infeasible and returns None.
    4. Computes a synthetic transformation (homography-based) of the source towards the target.
    5. Measures key metrics of the transformation:
        - Structural Similarity Index (SSIM)
        - Intersection over Union (IoU) of masks
        - Keypoint L2 distance and OKS (Object Keypoint Similarity)
    6. Returns a dictionary summarizing these metrics and the source/target samples.

    Returns:
        dict or None: A dictionary containing the following keys:
            - "source": Source sample
            - "target": Target sample
            - "npd": Normalized Pose Distance (float)
            - "ssim": Structural Similarity Index (float)
            - "iou": Intersection over Union (float)
            - "kpsL2": Keypoint L2 distance (float)
            - "oks": Object Keypoint Similarity (float)
        If the transformation is considered infeasible (BDD > 0.5), returns None.
    """
    s = ds.rand_train()
    t = ds.rand_train()
    bdd = metrics.BDD_flat_fast(s.pose[:3,:3].reshape(1,-1), t.pose[:3,:3].reshape(1,-1))
    cdist = np.linalg.norm(s.C-t.C)
    adist = cv_utils.core.rotation_angle_between_matrices(s.pose[:3,:3],t.pose[:3,:3])

    MAX_BDD = 0.5
    if bdd > MAX_BDD:
        return None

    if mask_input: # for SPEED+, Swisscube
        # mask the target image
        ti = t.masked_image
    else:   # for already masked datasets like AIRBUS
        ti = t.image
    tm = t.mask
    start_time = time.time()

    synth2D, transformed_keypoints2D, transformed_mask2D = Homography(
        source=s,
        target=t,
        return_transformed_kps=True,
        return_transformed_mask=True,
        mask_input=mask_input,
        with_axes=False,
        with_kps=False
    )
    synth3D, transformed_keypoints3D, transformed_mask3D = Transform3D(
        source=s,
        target=t,
        interpolate=True,
        return_transformed_kps=True,
        return_transformed_mask=True,
        mask_input=mask_input,
        maskOutput=True,
        with_kps=False,
        with_axes=False
    )
    # Save synthesized images
    filepathHOM = ds.save_synthesized(source=s,target=t,synth=synth2D,method="HOM")
    filepath3DT = ds.save_synthesized(source=s,target=t,synth=synth3D,method="3DT")

    # Evaluate with HRNet
    preds2D, confidences2D, _, _ = run_inference(
        ds.model,
        image=cv2.cvtColor(synth2D,cv2.COLOR_RGB2BGR),  # or pass an np.ndarray(BGR)
        device="cuda",
        bbox=t.bbox
    )
    preds3D, confidences3D, _, _ = run_inference(
        ds.model,
        image=cv2.cvtColor(synth3D,cv2.COLOR_RGB2BGR),  # or pass an np.ndarray(BGR)
        device="cuda",
        bbox=t.bbox
    )
    # Get target distance for VBN baseline
    target_distance = np.linalg.norm(t.pose[:3,3])
    

    # Get cropped versions of masks and images around target mask
    # Find bounding box of positive pixels in the target mask
    ys, xs = np.where(tm > 0)
    if len(xs) > 0 and len(ys) > 0:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        # Crop images to the bounding box
        synth2D_cropped = synth2D[y_min:y_max+1, x_min:x_max+1]
        synth3D_cropped = synth3D[y_min:y_max+1, x_min:x_max+1]
        ti_cropped = ti[y_min:y_max+1, x_min:x_max+1]
        # Optionally, crop masks as well if needed
        transformed_mask2D_cropped = transformed_mask2D[y_min:y_max+1, x_min:x_max+1]
        transformed_mask3D_cropped = transformed_mask3D[y_min:y_max+1, x_min:x_max+1]
        tm_cropped = tm[y_min:y_max+1, x_min:x_max+1]

    # --------- Computing metrics -----------

    # SSIM is computed on cropped images
    ssim_score2D = metrics.ssim(synth2D_cropped, ti_cropped)
    ssim_score3D = metrics.ssim(synth3D_cropped, ti_cropped)

    kps_L22D, kps_oks2D = metrics.keypoint_metrics(transformed_keypoints2D[:, :2], t.keypoints2D[:, :2], visibility=t.keypoints2D[:,2])
    kps_L23D, kps_oks3D = metrics.keypoint_metrics(transformed_keypoints3D[:, :2], t.keypoints2D[:, :2], visibility=t.keypoints2D[:,2])
    kps_L2NN2D, _ = metrics.keypoint_metrics(preds2D, t.keypoints2D[:, :2], visibility=t.keypoints2D[:,2])
    kps_L2NN3D, _ = metrics.keypoint_metrics(preds3D, t.keypoints2D[:, :2], visibility=t.keypoints2D[:,2])
    iou2D = metrics.IOU(tm, transformed_mask2D)
    iou3D = metrics.IOU(tm, transformed_mask3D)

    # Compute VBN baseline
    # Project the 3D center of the target (t.pose[:3,3]) into the image to get pixel coordinates using cv2.projectPoints
    center_3d = t.pose[:3, 3].reshape(1, 1, 3)  # shape (1, 1, 3)
    rvec, _ = cv2.Rodrigues(t.pose[:3, :3])
    tvec = t.pose[:3, 3].reshape(3, 1)
    camera_matrix = t._K
    center_target, _ = cv2.projectPoints(center_3d, rvec, tvec, camera_matrix, distCoeffs=np.array([]))
    center_target = center_target[0][0]  # shape (2,)
    kps2D_error_image = center_target + [kps_L22D, 0] # we put the error in x
    kps3D_error_image = center_target + [kps_L23D, 0]
    kpsNN2D_error_image = center_target + [kps_L2NN2D, 0]
    kpsNN3D_error_image = center_target + [kps_L2NN3D, 0]
    # Backproject these onto the image
    target_depth = t.pose[2,3]
    kps2D_error_world = backproject(kps2D_error_image[0], kps2D_error_image[1], target_depth, ds.K)
    kps3D_error_world = backproject(kps3D_error_image[0], kps3D_error_image[1], target_depth, ds.K)
    kpsNN2D_error_world = backproject(kpsNN2D_error_image[0], kpsNN2D_error_image[1], target_depth, ds.K)
    kpsNN3D_error_world = backproject(kpsNN3D_error_image[0], kpsNN3D_error_image[1], target_depth, ds.K)
    # Get ratio to inter-satellite distance
    kps2D_vbn = np.linalg.norm(kps2D_error_world-t.pose[:3, 3])/target_distance
    kps3D_vbn = np.linalg.norm(kps3D_error_world-t.pose[:3, 3])/target_distance
    kpsNN2D_vbn = np.linalg.norm(kpsNN2D_error_world-t.pose[:3, 3])/target_distance
    kpsNN3D_vbn = np.linalg.norm(kpsNN3D_error_world-t.pose[:3, 3])/target_distance

    # For calculating mask area, to test IOU hypothesis
    tm_bool = np.asarray(tm).astype(bool)
    tm_area = np.logical_and(tm_bool,tm_bool).sum()
    result = {
        "source": s.filepath,
        "target": t.filepath,
        "synthHOM": filepathHOM,
        "synth3DT": filepath3DT,
        "pose_s": s.pose.flatten().tolist(),
        "pose_t": t.pose.flatten().tolist(),
        "bdd": float(bdd),
        "cdist": float(cdist),
        "adist": float(adist),
        "target_mask_area": float(tm_area),
        "ssim2D": float(ssim_score2D),
        "ssim3D": float(ssim_score3D),
        "iou2D": float(iou2D),
        "iou3D": float(iou3D),
        "kpsL2_2D": float(kps_L22D),
        "kpsL2_3D": float(kps_L23D),
        "kpsL2_NN2D": float(kps_L2NN2D),
        "kpsL2_NN3D": float(kps_L2NN3D),
        "kpsL2_VBN_2D": float(kps2D_vbn),
        "kpsL2_VBN_3D": float(kps3D_vbn),
        "kpsL2_NN_VBN_2D": float(kpsNN2D_vbn),
        "kpsL2_NN_VBN_3D": float(kpsNN3D_vbn),
        "oks2D": float(kps_oks2D),
        "oks3D": float(kps_oks3D)
    }
    return result

def run_MC(N: int):
    """
    Perform multiple 'one-shot' transformation evaluations and save results.

    This function:
    1. Runs the `one_shot` transformation N times.
    2. Filters out invalid shots (where `one_shot()` returned None).
    3. Writes the valid shots to a JSON file using `Dataset.write_data_to_json()`.

    Args:
        N (int): The number of Monte Carlo shots to generate and evaluate.
    """
    shots = []
    from datetime import datetime
    now = datetime.now().strftime("%Y%m%d%H%M%S")

    for i in range(N):
        print(f"MC Sample: {i}")
        shot = one_shot()
        if shot is not None:
            shots.append(shot)
        if i % 100 == 0:
            Dataset.write_data_to_json(shots, filename=f"monte_carlo/{N}_{now}.json")

def compute_density():
    rotations = [s.R for s in ds.posed_samples]
    ds.write_data_to_json([rotation.tolist() for rotation in rotations],"rotations.json")
    for i in [10**e for e in range(10)]:
        d = metrics.compute_largest_empty_ball(rotations,num_space_samples=i)
        print(f"{i}:{d}")

def init_dataset(name):

    if name == "SPEED":
        ds = Dataset(base_path=base_path_speed,
                     mesh_path=mesh_path_speed,
                     keypoints_path=keypoints_path_speed,
                     dataset_name="SPEED")
    elif name == "AIRBUS":
        ds = Dataset(base_path=base_path_airbus,
                     mesh_path=mesh_path_airbus,
                     keypoints_path=keypoints_path_airbus,
                     model_path=model_path_airbus,
                     model_config_path=model_config_path_airbus,
                     dataset_name="AIRBUS")
    elif name == "SHIRT":
        ds = Dataset(base_path=base_path_shirt,
                     mesh_path=mesh_path_speed,
                     dataset_name="SHIRT")
    elif name == "SPEED+":
        ds = Dataset(base_path=base_path_speed_plus,
                     mesh_path=mesh_path_speed,
                     keypoints_path=keypoints_path_speed,
                     model_path=model_path_speed_plus,
                     model_config_path=model_config_path_speed_plus,
                     dataset_name="SPEED+")
    elif name == "SWISSCUBE":
        ds = Dataset(base_path=base_path_swisscube,
                     mesh_path=mesh_path_swisscube,
                     keypoints_path=keypoints_path_swisscube,
                     model_path=model_path_swisscube,
                     model_config_path=model_config_path_swisscube,
                     dataset_name="SWISSCUBE")
    elif name == "URSO":
        ds = Dataset(base_path=base_path_urso,
                     mesh_path=mesh_path_urso,
                     keypoints_path=keypoints_path_urso,
                     dataset_name="URSO")
    else:
        raise ValueError("Unknown dataset name.")

    return ds

if __name__ == "__main__":
    print("Start script.")
    
    # This is for executing from CLI as main script
    parser = argparse.ArgumentParser(description="Load a dataset for 6DoF pose estimation.")
    parser.add_argument("--dataset", type=str, required=True, choices=Dataset.implemented_datasets, help="Dataset name")
    args = parser.parse_args()

    ds = init_dataset(args.dataset.upper())
    
    # Important line, loads the dataset
    # This is not (yet) done in __init__()
    ds.load_dataset()

    # coco = ds.export_labels_to_coco()
    # ds.write_data_to_json(coco, f"{args.dataset}.json")

    # Debugging
    #test_synthesis(ds)

    # Monte Carlo synthesis analysis
    run_MC(30000)

    # Saving rotations for computing density
    #rotations = [s.R.tolist() for s in ds.posed_samples]
    #ds.write_data_to_json(rotations, f"dataset_rotations/rotations_{args.dataset}.json")
    
    #result = [r.tolist() for r in metrics.generate_dense_random_rotations()]
    #ds.write_data_to_json(result, "even_rots.json")