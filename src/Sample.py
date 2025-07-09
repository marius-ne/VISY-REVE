import numpy as np
from typing import Optional
import cv2
import os
import pyrender
import trimesh
import sys
import matplotlib.pyplot as plt


# For linting the Dataset class, cannot import directly due to circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Dataset import Dataset

# Own imports
import src.synthesis
import src.visy_reve_utils
import cv_utils

# Imports from training library
# 1) Ensure that `repo/` is on sys.path, so that `training/...` imports resolve.
REPO_ROOT = os.getcwd()  
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 2) Also ensure that `repo/training/lib` is on sys.path,
#    so that inside HRNet code, `import models.pose_hrnet` will find:
#      repo/training/lib/models/pose_hrnet.py
LIB_ROOT = os.path.join(REPO_ROOT, "training", "lib")
if LIB_ROOT not in sys.path:
    sys.path.insert(0, LIB_ROOT)
from training.inference.infer_hrnet import setup_model, run_inference    

class Sample:
    """
    Represents a single image sample with its associated ground truth 6DoF pose.
    Optionally includes a depth map and mask.
    """
    def __init__(self, 
                 dataset: "Dataset",
                 type: str,
                 parent_dir: str = None,
                 image_id_num: int = None,
                 unique_image_id_num: int = None,
                 image_id: int = None,
                 filepath: str = None,
                 pose: np.ndarray = None,
                 K: np.ndarray = None,
                 filename: str = None,
                 keypoints2D: np.ndarray = None,
                 bbox: np.ndarray = None,
                 mask_path: str = None,
                 lightpos: np.ndarray = None,
                 image_already_masked: bool = False):
        """
        Args:
            image_path (str): Filepath of the sample image.
            image_id_num (int): ID number of the sample, it's not necessarily unique, can be shared
                across splits as in the case of SHIRT, therefore the need for the image_id.
            image_id (str): Actual unique identifier of images. Should contain id_num and additional
                information (such as split name) that uniquely identifies the sample.
            unique_image_id_num (int): UNIQUE ID number of the sample. Like `image_id` but must
                be numeric for coco files. So please encode unique information as number.
            parent_dir (str): Parent directory of the images directory. For example 
                `.../parent_dir/images/imageXXXX.png`. Only at split-level, this is not for the
                whole dataset.
            pose (np.ndarray): 4x4 pose matrix (passive world2cam in cam coordinates).
            K (np.ndarray): Intrinsic camera matrix. Is the same across the whole dataset for
                the vast majority of datasets, which is why we're using it as a private attribute
                in the Sample class. For Swisscube though it varies (or at least it's given for
                each image) so we also include it here.
            filename (str): Filename of the image file, with extension.
            keypoints2D (np.ndarray): 2D keypoint coordinates. Can be used if the dataset is loaded
                from a pre-existing label file (coco). In this case this object will not projec them itself.
            bbox (np.ndarray): Bounding box, (x,y,width,height), should only be supplied if 
                keypoints2D are supplied also, used when loading from COCO.
            mask_path (str): String with the full path to a mask file. Should be given if the dataset
                already contains masks alongside the images e.g. for SWISSCUBE. 
            lightpos (np.ndarray): Optionally the position of the lightsource for the given image.
            image_already_masked (bool): Whether or not the input image is already masked and does 
                not need to be masked separately or have a masked version stored on disk.
        """
        # Every sample has an associated dataset
        self.dataset = dataset

        # Set intrinsic matrix (see docstring)
        self._K = K

        # Light position (if available, else None)
        self.lightpos = lightpos

        # Maybe dataset has pre-existing masks
        self.mask_path = mask_path

        # Project keypoints with GT if both present  
        if hasattr(self.dataset, "keypoints3D") and self.dataset.keypoints3D is not None \
              and pose is not None and keypoints2D is None:
            self.keypoints2D = cv_utils.project_points(pose, K, self.dataset.keypoints3D)
            bbox_results = cv_utils.keypoints.keypoints_to_bbox(self.keypoints2D,
                                                                compute_area=True)
            self.bbox = bbox_results[:4]
            self.bbox_area = bbox_results[-1]
            self.keypoints2D = cv_utils.keypoints.add_visibility_to_keypoints(self.keypoints2D, 
                                            image_size=(self.dataset.width,self.dataset.height))
            
        elif keypoints2D is not None:
            self.keypoints2D = keypoints2D
            self.bbox = bbox
            self.bbox_area = bbox[2] * bbox[3]
        else:
            self.keypoints2D = None

        # Samples can also not have an associated image
        # when they are the targets of a synthesis
        if filepath is not None and image_id_num is not None and parent_dir is not None:
            self.image_id_num = image_id_num
            self.image_id = image_id    # not unique across the dataset
            self.unique_image_id_num = unique_image_id_num # must be unique
            self.parent_dir = parent_dir

            # Image path is dataset specific
            self.filepath = filepath

            # Image filename with extension (e.g. .png)
            self.filename = filename 

            self.filename_without_extension = self.dataset.get_image_filename(self,
                                                            with_extension=False)
                         
            self.has_image = True

            if not image_already_masked:
                # This is for eventual masking of image files for NN training
                # and Monte Carlo runs
                self.masked_parent_dir = os.path.join(parent_dir, "masked_images")
                self.masked_filepath = os.path.join(self.masked_parent_dir, filename)
            else: 
                self.masked_parent_dir = self.parent_dir
                self.masked_filepath = self.filepath
            self.image_already_masked = image_already_masked

            # Whether or not to load the masks and d.m. from files if they have been written
            # Turn this off if you changed the model in-between
            # IMPORTANT: This can only be true if the sample has an image / 
            # a filename was provided.
            self.load_mask = False
            self.load_depth = False
            self.load_masked_image = False

        # The arguments are partially supplied, raise a warning
        elif (
            # Only one is None, others are not
            (image_id_num is None and filepath is not None and parent_dir is not None) or
            (filepath is None and image_id_num is not None and parent_dir is not None) or
            (parent_dir is None and image_id_num is not None and filepath is not None) or
            # Two are None, one is not
            (image_id_num is None and filepath is None and parent_dir is not None) or
            (image_id_num is None and parent_dir is None and filepath is not None) or
            (filepath is None and parent_dir is None and image_id_num is not None)
        ):
            self.has_image = False
            raise Warning("You did not supply ALL the necessary arguments image_id, " \
            "image_path and parent_dir. The image will not be read.")
        else:
            self.has_image = False

        # Test data doesn't have poses
        if pose is not None and "train" in type:
            self.pose = pose  # Should be a 4x4 matrix
            self.t = pose[:3,3]
            self.R = pose[:3,:3]

            # Get active world to camera pose and camera center in world coordinates
            self.active = np.linalg.inv(self.pose)
            self.C = self.active[:3,3]

            self.type = "train"
            self.has_pose = True
        else:
            self.type = "test"
            self.has_pose = False

    @property
    def image(self) -> np.ndarray:
        """
        Loads the RGB image from disk. Raises informative errors
        if the image path is invalid or loading fails.
        """
        if not self.has_image:
            raise ValueError("Cannot get the image for sample without an image.")
        
        if not os.path.isfile(self.filepath):
            raise FileNotFoundError(f"Image file does not exist: {self.filepath}")

        # Read the image in BGR format (default for OpenCV)
        _image = cv2.imread(self.filepath, cv2.IMREAD_COLOR)
        if _image is None:
            raise IOError(f"cv2 failed to load image: {self.filepath}")

        # Convert to RGB format
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)

        return _image
    
    @property
    def image_with_axes(self) -> np.ndarray:
        """
        Returns image with axes annotated.
        """
        img = self.image

        if self.has_pose:
            return self.draw_axes(img)
        else:
            raise ValueError("Cannot draw axes for image without pose label.")
        
    def _scale_vector_to_image_bounds(self, origin, vector, image_shape):
        """
        Scales the vector so that the endpoint (origin + vector) stays within image bounds.

        Args:
            origin (np.ndarray): The starting point of the vector (x, y).
            vector (np.ndarray): The vector to be scaled (dx, dy).
            image_shape (tuple): The shape of the image (width, height).

        Returns:
            np.ndarray: The scaled vector.
        """
        w, h = image_shape[:2]
        end = origin + vector
        scale = 1.0

        # Calculate scaling factors for each boundary
        if vector[0] != 0:
            if end[0] < 0:
                scale = min(scale, -origin[0] / vector[0])
            elif end[0] >= w:
                scale = min(scale, (w - 1 - origin[0]) / vector[0])
        if vector[1] != 0:
            if end[1] < 0:
                scale = min(scale, -origin[1] / vector[1])
            elif end[1] >= h:
                scale = min(scale, (h - 1 - origin[1]) / vector[1])

        return vector * scale
    
    def draw_axes(self, 
                image: np.ndarray,
                fraction: float = 0.2,
                pose: np.ndarray = None) -> np.ndarray:
        """
        Returns image with pose axes annotated.

        Arrow length given as fraction of smaller image dimension.
        
        Supply pose if you want to draw for another image than the one of your sample.
        Supports both uint8 [0–255] and float [0–1] input.
        """
        # Detect float images in 0..1 and convert to uint8 for OpenCV drawing
        is_float = np.issubdtype(image.dtype, np.floating)
        if is_float:
            img = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img = image.copy()

        h, w = img.shape[:2]
        min_dim = min(h, w)
        
        # Arrow length in pixels
        axis_length = fraction * min_dim

        # Use provided pose or fall back to self.pose
        if pose is None:
            pose = self.pose

        # Axes endpoints in object space, we ensure that they never get behind
        # the camera (otherwise the arrow would flip) by clamping with a epsilon
        eps = 0.01
        L = max(eps, min(1.0, pose[2,3] - eps))
        axis_points = np.float32([
            [0,0,0],
            [L,0,0],
            [0,L,0],
            [0,0,L],
        ])

        # Convert rotation matrix → rotation vector
        rvec, _ = cv2.Rodrigues(pose[:3, :3].astype(np.float32))

        # Project the unit axes
        imgpts, _ = cv2.projectPoints(
            axis_points,
            rvec=rvec,
            tvec=pose[:3, 3],
            cameraMatrix=self._K,
            distCoeffs=np.zeros(5, dtype=np.float32)
        )
        imgpts = imgpts.reshape(-1, 2)

        origin = imgpts[0]
        # Rescale each axis so its on‐screen length = axis_length
        for i in range(1, 4):
            vec = imgpts[i] - origin
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                imgpts[i] = origin + (vec / norm) * axis_length

        # Prepare to draw
        origin_pt = tuple(origin.astype(int))
        pts = imgpts.astype(int)

        # White circle at origin
        cv2.circle(img, origin_pt, 6, (255, 255, 255), -1)

        # Draw arrows: X=red, Y=green, Z=blue
        cv2.arrowedLine(img, origin_pt, tuple(pts[1]), (255, 0, 0), 3, tipLength=0.1)
        cv2.arrowedLine(img, origin_pt, tuple(pts[2]), (0, 255, 0), 3, tipLength=0.1)
        cv2.arrowedLine(img, origin_pt, tuple(pts[3]), (0, 0, 255), 3, tipLength=0.1)

        # If original was float, convert back to float [0..1]
        if is_float:
            img = img.astype(np.float32) / 255.0
            img = (255*img).astype(np.uint8)
        return img


    def set_image(self, new_image_id, new_filepath, new_parent_dir):
        """
        Setter for the image, used after synthesis.
        """
        if not self.has_image:
            self.image_id_num = new_image_id
            self.filepath = new_filepath
            self.parent_dir = new_parent_dir
            self.has_image = True
        else:
            raise Warning(f"Sample already has an associated image at {self.filepath}.")
        
    @classmethod
    def synthesize_sample(cls,
                          ds: "Dataset" = None,
                          pose: np.ndarray = np.eye(4),
                          synthesis_method: str = "3D",
                          nn_metric: str = "BDD",
                          save_image_to_disk: bool = False):
        """
        Synthesizes the view of this Sample, then:
          1. Saves the synthesized RGB to disk.
          2. Saves the masked (RGB) version to disk.
          3. Renames both according to the source’s filename + 's'.
          4. Updates self.filepath to point to the new *masked* image and marks `has_image=True`.
          5. Optionally saves it to disk when save_image_to_disk = True (False by default).
          
        Returns:
            sample: created sample object
        """

        # 1) Find nearest neighbor (no self‐exclusion)
        NN = ds.nn(
            query_pose=pose,
            distance_measure=nn_metric,
            return_distance=False,  
            return_nn_ix=False
        )

        # ----- Create metadata for new image -----
        # Image id will be old image ID plus pose encoded
        synth_image_id = NN.image_id
        synth_image_id += src.visy_reve_utils.encode_4x4(pose, to_numeric=False)

        # Image id num will be old image id num plus pose encoded
        # as an integer, this encoding is just to make it random enough
        # to avoid possible duplicates. 
        # This is necessary for COCO. The image IDs must be an integer
        # an must be unique.
        pose_encoding = src.visy_reve_utils.encode_4x4(pose, to_numeric=True)
        nums = [NN.unique_image_id_num, pose_encoding]
        synth_unique_image_id_num = int("".join(map(str, nums)))

        # Location will be in the same folder as the NN
        # Filename will be the "S" + unique image id num + extension
        # Parent dir will be the same as the masked one for the NN
        synth_dirpath = os.path.dirname(NN.masked_filepath)
        synth_filename = f"S{NN.unique_image_id_num}{synth_unique_image_id_num}{ds.extension}"
        synth_filepath = os.path.join(synth_dirpath,synth_filename)
        synth_parent_dir = NN.masked_parent_dir

        import time

        start = time.time()
        # 1) Perform 3D‐based synthesis
        if synthesis_method == "3D":
            synth_img = src.synthesis.Transform3D(
                source=NN,
                target=pose,
                interpolate=True,
                mask_input=True,
                maskOutput=True,
                with_kps=False,
                with_axes=False,
                target_synth=True,
                return_transformed_kps=False,
                return_transformed_mask=False
            )
        elif synthesis_method == "2D":
            synth_img = src.synthesis.Homography(
                source=NN,
                target=pose,
                with_axes=False,
                with_kps=False,
                fill_bg=False,
                return_transformed_kps=False,
                return_transformed_mask=False,
                mask_input=True,
                target_is_pose=True
            )
        end = time.time()
        elapsed_time = end-start
        
        if save_image_to_disk:
            # 3) Write out the synthesized RGB image
            #    (convert from RGB→BGR for cv2.imwrite)
            bgr_synth = cv2.cvtColor(synth_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(synth_filepath, bgr_synth)
            print(f"Synthesized image saved to {synth_filepath}.")
        
        # Create sample instance, user needs to provide information where the image
        # is going to be stored 
        # Needs to be done AFTER the image is synthesized and saved
        self = cls(filepath=synth_filepath, 
                    image_id_num=synth_unique_image_id_num,
                    unique_image_id_num=synth_unique_image_id_num,
                    image_id=synth_image_id,
                    dataset=ds,
                    parent_dir=synth_parent_dir,
                    type="train",
                    pose=pose,
                    K=ds.K,
                    filename=synth_filename)
        
        # We have to manually fix the masked attributes
        self.masked_filepath = synth_filepath
        self.masked_parent_dir = synth_parent_dir

        return self, elapsed_time

    def synthesize_image(self,
                   method: str = "3D",
                   nn_metric: str = "BDD",
                   return_eval: bool = False):
        """
        Synthesizes the view of this sample and gives back image. This function should
        be called by a bare-bones Sample object, i.e. one without an image associated, 
        just a pose.

        Optionally gives back evaluation metrics.

        This means that the image has axes projected and the function returns the error
        metrics and does not actually write the synthesized image to disk.

        Args:
            return_eval: If True, returns transformed keypoints, transformed mask, the 
            distance to the nearest neighbour in the unit of the nn_metric specified and
            the nearest neighbour itself as a Sample object. False by default, here it 
            returns only the synthesized image.
        """
        if self.has_image:
            raise ValueError("This Sample has an image. You should not try to synthesize it.")
        
        # Get nearest neighbour (no self exclusion, therefore the caller should be outside
        # of the dataset)
        NN, distval = self.dataset.nn(query_pose=self.pose,
                             distance_measure=nn_metric,
                             return_distance=True,
                             return_nn_ix=False)
        
        if method == "3D":
            synth_img,transformed_kps,transformed_mask = src.synthesis.Transform3D(
                source=NN,
                target=self.pose,
                interpolate=True,
                mask_input=True,
                maskOutput=True,
                with_kps=False,
                with_axes=True,
                target_synth=True,
                return_transformed_kps=True,
                return_transformed_mask=True
            )
        else:
            raise NotImplementedError(method)
        if return_eval:
            return (synth_img, transformed_kps, transformed_mask, distval, NN)
        else: 
            return synth_img
        
    def run_inference(self, other_img = None, return_metric = False):
        """
        Utility class for running inference with the provided model weights of the dataset.
        Avoids having to do the import yourself and having to extract the image from the sample.

        Args:
            other_img (np.ndarray): Optionally runs inference on another image than the one
                of this sample. Must be passed as BGR.
            return_metric (bool): If True (False by default), return also the mean of the
                2D euclidean error of the predicted vs. ground truth keypoint locations.
        """
        preds, confidences, orig_img, orig_hmps = run_inference(
            self.dataset.model,
            image=self.masked_filepath  ,  # or pass an np.ndarray(BGR)
            device="cuda",
            bbox=self.bbox
        )
        gt_kps = self.keypoints2D
        # Ensure both preds and gt_kps are numpy arrays
        preds = np.asarray(preds)
        gt_kps = np.asarray(gt_kps)

        # If keypoints have visibility flag, use only first two columns
        if preds.shape[1] > 2:
            preds = preds[:, :2]
        if gt_kps.shape[1] > 2:
            gt_kps = gt_kps[:, :2]

        mean_dist = np.mean(np.linalg.norm(preds - gt_kps, axis=1))

        if return_metric:
            return preds, mean_dist
        else:
            return preds

    def _render_depth_map(self) -> np.ndarray:
        """
        Loads an OBJ mesh, place a camera defined by intrinsic & extrinsic,
        and renders a depth map.

        Args:
            obj_path:        Path to the .obj file.
            intrinsic:       3x3 camera intrinsic matrix (numpy array).
                            [[fx,  0, cx],
                            [ 0, fy, cy],
                            [ 0,  0,  1]]
            extrinsic:       4x4 camera-to-world pose matrix (numpy array).
                            Transforms points from camera coords into world coords.
            width:           Width of the rendered image in pixels.
            height:          Height of the rendered image in pixels.

        Returns:
            A (heightxwidth) numpy float32 array representing per-pixel depth
            in the camera's view (in the same units as your mesh).
        """
        # 1. Load mesh
        # Done in Dataset class

        # If dataset has a mesh we make the sample aware of this
        if not hasattr(self.dataset,"mesh"):
            raise ValueError("Cannot render depth map if dataset has no mesh.")

        # 2. Build scene and add mesh
        scene = pyrender.Scene()
        scene.add(self.dataset.render_mesh)

        # 3. Create camera
        fx, fy = self._K[0, 0], self._K[1, 1]
        cx, cy = self._K[0, 2], self._K[1, 2]
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)

        # Note: pyrender expects camera pose as a 4×4 world‐to‐camera transform.
        
        # Going from OpenCV to OpenGL camera coordinates
        # Source: https://github.com/mmatl/pyrender/issues/228
        opengl_pose = self.pose.copy()
        opengl_pose[[1, 2]] *= -1

        # Going from passive world2cam to passive cam2world
        cam_pose = np.linalg.inv(opengl_pose)

        scene.add(camera, pose=cam_pose)

        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                            innerConeAngle=np.pi/16.0,
                            outerConeAngle=np.pi/6.0)

        scene.add(light, pose=cam_pose)

        # 4. Offscreen renderer
        r = pyrender.OffscreenRenderer(viewport_width=self.dataset.width,
                                    viewport_height=self.dataset.height,
                                    point_size=1.0)

        # 5. Render depth only
        depth = r.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        r.delete()  

        # depth is a float32 array (height×width) with z in camera coords
        return depth

    @property
    def depth_map(self):
        """
        Loads the depth map of the object from file if it exists,
        otherwise creates and saves a dummy depth map (zeroed).

        We save the depth map as a 32 bit floating point .npy to be accurate.
        """
        # Do we load and and does it exist? If yes and no then we write it
        # We have to put this guard at the very top because synthesized samples
        # don't have filenames
        if self.load_depth:
            base_filename = self.filename_without_extension

            depth_map_filename = f"{base_filename}.npy"
            depth_dir = os.path.join(self.parent_dir, "depth")
            depth_map_path = os.path.join(depth_dir, depth_map_filename)

            # Create directory if it doesn't exist
            os.makedirs(depth_dir, exist_ok=True)

            if os.path.exists(depth_map_path):
                # Load the depth map from .npy file
                _depth_map = np.load(depth_map_path)
                print(f"Loaded depth map from {depth_map_path}")
                return _depth_map
            
        # Else we render it
        _depth_map = self._render_depth_map()
        print(f"Rendered depth map for {self.unique_image_id_num}.")

        if self.load_depth:
            np.save(depth_map_path, _depth_map)
            print(f"Wrote depth map {depth_map_path}")

        return _depth_map
    
    @property
    #@profile
    def mask(self):
        """
        Loads the binary mask of the object from file if it exists,
        otherwise creates one from the depth map and saves it in uint8 format.

        Note: We keep the self.mask_path and _mask_path API seperate. I.e. external
        masks (e.g. SWISSCUBE) are not treated the same as internally written masks. 
        """
        # We check whether dataset has pre-existing masks
        if self.mask_path is not None:
            if os.path.exists(self.mask_path):
                # Load the mask in grayscale mode
                _mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
                print(f"Loaded mask from {self.mask_path}")
                return _mask
            else:
                raise ValueError(f"Your given mask path {self.mask_path} does not exist.")
    
        # We check whether we already created it and when not we render and then write it
        if self.load_mask:
            base_filename = self.filename_without_extension

            mask_filename = f"{base_filename}.png"
            mask_dir = os.path.join(self.parent_dir, "mask")
            _mask_path = os.path.join(mask_dir, mask_filename)

            # Ensure the mask directory exists
            os.makedirs(mask_dir, exist_ok=True)

            if os.path.exists(_mask_path):
                # Load the mask in grayscale mode
                _mask = cv2.imread(_mask_path, cv2.IMREAD_GRAYSCALE)
                print(f"Loaded mask from {_mask_path}")
                return _mask
            
        # Otherwise we render
        _depth_map = self._render_depth_map()
        _mask = (_depth_map > 0).astype(np.uint8)   
        print(f"Rendered mask for {self.unique_image_id_num}.")

        if self.load_mask:
            cv2.imwrite(_mask_path, _mask)
            print(f"Wrote mask {_mask_path}")

        return _mask
    
    @property
    def normalized_image(self):
        """
        Returns normalized image for easier visibility of target.
        Background stays at zero if it's zero before.
        """
        _img = self.image
        _img = 255 * (_img - np.min(_img)) / (np.max(_img) - np.min(_img))
        _img = (255*_img).astype(np.uint8)

        return _img
    
    @property
    def normalized_image_with_axes(self):
        """
        Returns normalized image with axes drawn onto it.
        Background stays at zero if it's zero before.
        """
        _img = self.normalized_image

        if self.has_pose:
            _img = self.draw_axes(_img)
        else:
            raise ValueError("You need a sample with pose to draw axes.")
        
        return _img
    
    @property
    def image_with_axes_and_kps(self):
        """
        Returns image with both axes and keypoints annotated.
        """
        if self.keypoints2D is None:
            print("You must provide keypoints for this. Returning just with axes.")
            return self.image_with_axes
        
        img = self.image_with_axes
        cv_utils.keypoints.draw_kps_on_img(img,self.keypoints2D)
        return img
    
    @property
    def masked_image(self):
        """
        Returns the original image with the binary mask applied.
        
        The masked image only keeps the pixels where the mask is 1 (non-zero),
        setting masked-out areas to zero (black).
        
        Returns:
            np.ndarray: The masked image as a NumPy array.
        """
        masked_filepath = self.masked_filepath

        # Check if it already exists
        if self.load_masked_image or self.image_already_masked:
            if os.path.exists(masked_filepath):
                _masked_img = cv2.imread(masked_filepath)
                if _masked_img is None:
                    raise IOError(f"cv2 failed to load masked image: {masked_filepath}")
                _masked_img = cv2.cvtColor(_masked_img,cv2.COLOR_BGR2RGB)
                print(f"Loaded masked image from {masked_filepath}.")
                return _masked_img
            
        # If it doesn't exist we load it
        img = self.image
        _mask = self.mask

        # Ensure the mask has the same size as the image
        if _mask.shape != img.shape[:2]:
            _mask = cv2.resize(_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Ensure the mask is single-channel 8-bit
        if _mask.dtype != np.uint8:
            _mask = _mask.astype(np.uint8)
        
        # Apply mask: only keep pixels where mask == 1
        masked_img = cv2.bitwise_and(img, img, mask=_mask)

        if self.load_masked_image:
            # Write it to disk for later
            masked_image_for_write = cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR)
            os.makedirs(self.masked_parent_dir,exist_ok=True)
            success = cv2.imwrite(masked_filepath, masked_image_for_write)
            if success:
                print(f"Wrote masked image to {masked_filepath}.")
            else:
                print("Didn't write masked image, something went wrong.")

        return masked_img
    
    @property
    def masked_image_with_axes(self):
        """
        Returns the masked image with 3D pose axes drawn onto it.
        """
        # 1) Start from the masked image
        img = self.masked_image

        # 2) Draw 3D axes onto the masked image
        #    (assumes `self.draw_axes(img, pose)` returns a new array)
        if self.has_pose:
            img = self.draw_axes(img)

        return img
    
    @property
    def masked_image_with_axes_and_kps(self):
        """
        Returns the masked image, then draws both the 3D pose axes and
        the 2D keypoints (if available) onto it.

        - If no keypoints are provided, it will still draw the axes but print a warning.
        - If keypoints are present, it will draw them on top of the axis‐annotated version.
        """
        # 1) Start from the masked image
        img = self.masked_image

        # 2) Draw 3D axes onto the masked image
        #    (assumes `self.draw_axes(img, pose)` returns a new array)
        if self.has_pose:
            img = self.draw_axes(img)

        # 3) Draw keypoints on top if they exist
        if self.keypoints2D is None:
            print("You must provide keypoints to draw them. Returning masked image with axes only.")
            return img

        # We assume keypoints2D is an (N, 2) or (N, 3) array of (u, v[, vflag]) floats
        # The draw_kps_on_img function typically expects (u, v) in pixel coords.
        # If your keypoints array already has visibility in the 3rd column, the function
        # should ignore or use it accordingly.
        cv_utils.keypoints.draw_kps_on_img(img, self.keypoints2D)
        return img
    
    @property
    def masked_image_with_axes_and_inferred_kps(self):
        """
        Returns image with both GT and inferred keypoints.
        """
        preds = self.run_inference()

        base_img = self.masked_image_with_axes
        # Draw red X's at each predicted keypoint location
        for gt_pt, pt in zip(self.keypoints2D,preds):
            if gt_pt[2] == 0:
                continue # not visible
            x, y = int(round(pt[0])), int(round(pt[1]))
            size = 10
            color = (0, 255, 255)  # Red in RGB
            thickness = 5
            # Draw ground truth keypoints as white empty circles
            gt_x, gt_y = int(round(gt_pt[0])), int(round(gt_pt[1]))
            cv2.circle(base_img, (gt_x, gt_y), radius=20, color=(255, 255, 255), thickness=thickness)
            cv2.line(base_img, (x - size, y - size), (x + size, y + size), color, thickness)
            cv2.line(base_img, (x - size, y + size), (x + size, y - size), color, thickness)
        #cv_utils.keypoints.draw_kps_on_img(base_img,preds,color=(180,0,0))

        return base_img
    
    @property
    def masked_image_axes_inferred_kps_and_inferred_pose(self):
        """
        Runs PnP on inferred keypoints and draws the inferred pose axes with transparency.
        """
        preds = self.run_inference()
        obj_pts, img_pts = [], []
        for i, (gt_pt, pred_pt) in enumerate(zip(self.keypoints2D, preds)):
            if gt_pt[2] == 0:
                continue
            obj_pts.append(self.dataset.keypoints3D[i])
            img_pts.append(tuple(pred_pt))
        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self._K, np.zeros(5))
        if not success:
            raise RuntimeError("PnP failed to find a valid pose.")

        R_inf, _ = cv2.Rodrigues(rvec)
        pose_inf = np.eye(4, dtype=np.float32)
        pose_inf[:3,:3] = R_inf
        pose_inf[:3,3] = tvec.flatten()

        # Base image
        base = self.masked_image.copy()
        if self.has_pose:
            base = self.draw_axes(base, pose=self.pose)

        # Overlay inferred pose
        overlay = self.draw_axes(base.copy(), pose=pose_inf)
        alpha = 0.5
        blended = cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)

        # Draw inferred keypoints markers
        for pt in preds:
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.drawMarker(blended, (x,y), (0,255,255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=6, thickness=2)

        # Ensure proper numeric array type
        blended = np.array(blended, dtype=np.uint8)
        return blended

    
    @property
    def masked_image_with_axes_bbox_and_kps(self):
        """
        Returns the masked image with axes, keypoints, and bounding box drawn.

        - Calls masked_image_with_axes_and_kps to get the image with axes and keypoints.
        - Draws the bounding box from self.bbox (x, y, width, height) if available.
        """
        img = self.masked_image_with_axes_and_kps

        # Draw bounding box if available
        if self.bbox is not None:
            x, y, w, h = map(int, self.bbox)
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            # Draw rectangle in yellow
            cv2.rectangle(img, pt1, pt2, color=(255, 255, 0), thickness=2)
        else:
            print("No bounding box to draw.")

        return img
