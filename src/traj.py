import numpy as np
import cv2
import random
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation, Slerp
import plotly.graph_objects as go

# Own imports
from src.Sample import Sample
import src.metrics as metrics


def synthesize_poses(ds, poses):
    """
    Synthesizes one “sample” per pose in `poses` using dataset `ds`. 
    
    Each entry in the returned list is a dict with keys:
      - "synth_img"    : the synthesized RGB image (H×W×3 ndarray)
      - "synth_kps"    : the synthesized 2D keypoints (N×3 ndarray: [x, y, visibility])
      - "synth_mask"   : the synthesized binary mask (H×W ndarray)
      - "bddval"       : the BDD score returned by `target.synthesize(nn_metric="BDD")`
      - "source_pose"  : the 4×4 pose matrix of the nearest neighbor used to synthesize
      - "pnp_pose"     : the 4×4 pose matrix recovered by solvePnP (or all‐NaN if it failed)
      - "kps_L2"       : the L2 keypoint error between synthesized keypoints and “true” ones
      - "iou"          : the Intersection‐over‐Union between the target mask and the synthesized mask
    """
    # Ensure ds has been initialized with K and posed_samples
    if not hasattr(ds, "K") or not hasattr(ds, "posed_samples"):
        raise ValueError("Please load dataset first (it needs ds.K and ds.posed_samples).")

    synthesized_samples = []

    for pose in poses:
        # 1) Build a “target” Sample at the requested pose
        #    (this assumes Sample’s constructor takes these args exactly)
        target = Sample(dataset=ds,
                        type="train",
                        pose=pose,
                        K=ds.K)

        # 2) Synthesize:
        #    target.synthesize(nn_metric="BDD")  → (synth_img, synth_kps, synth_mask, bddval, source)
        synth_img, synth_kps, synth_mask, bddval, source = target.synthesize_image(nn_metric="BDD")
        source_pose = source.pose

        # 3) Run PnP on the synthesized 2D keypoints to recover a pose estimate (if possible)
        #    Use only the keypoints that are visible in the “true” target:
        visible_mask = target.keypoints2D[:, 2] > 0
        kp3d = ds.keypoints3D[visible_mask]
        kp2d = synth_kps[visible_mask, :2]

        if kp3d.shape[0] >= 4:
            success, rvec, tvec = cv2.solvePnP(
                kp3d.astype(np.float32),
                kp2d.astype(np.float32),
                ds.K.astype(np.float32),
                None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                R_pnp, _ = cv2.Rodrigues(rvec)
                pose_pnp = np.eye(4, dtype=np.float32)
                pose_pnp[:3, :3] = R_pnp
                pose_pnp[:3, 3] = tvec.flatten()
            else:
                pose_pnp = np.full((4, 4), np.nan, dtype=np.float32)
        else:
            pose_pnp = np.full((4, 4), np.nan, dtype=np.float32)

        # 4) Compute keypoint‐L2 and IoU metrics:
        kps_L2, _ = metrics.keypoint_metrics(
            synth_kps[:, :2],
            target.keypoints2D[:, :2],
            visibility=target.keypoints2D[:, 2]
        )
        iou_val = metrics.IOU(target.mask, synth_mask)

        # 5) Pack everything into one dict for this pose
        synthesized_samples.append({
            "synth_img":   synth_img,     # (H×W×3) ndarray
            "synth_kps":   synth_kps,     # (N×3) ndarray
            "synth_mask":  synth_mask,    # (H×W) ndarray
            "bddval":      bddval,        # float
            "source_pose": source_pose,   # (4×4) ndarray
            "pnp_pose":    pose_pnp,      # (4×4) ndarray (or NaNs)
            "kps_L2":      kps_L2,        # float
            "iou":         iou_val        # float
        })

    return synthesized_samples

# Ensure that z is always a scalar in backproject
def backproject(u, v, Z, K_mat):
    """
    Vectorized: Return 3-vectors in camera coords for pixels (u,v) at depth Z.
    u, v, Z can be scalars or arrays of the same shape.
    Ensures that z is always a scalar.
    """
    fx, fy = K_mat[0, 0], K_mat[1, 1]
    cx, cy = K_mat[0, 2], K_mat[1, 2]
    # Force Z to be a scalar
    if np.size(Z) != 1:
        raise ValueError("Z must be a scalar.")
    # Repeat Z
    Z = np.broadcast_to(Z, np.shape(u))
    x_cam = (u - cx) * Z / fx
    y_cam = (v - cy) * Z / fy
    return np.stack([x_cam, y_cam, Z], axis=-1)


def spline_trajectory_from_pose_list(
    poses_wc: list[np.ndarray],
    min_bdd: float = 0.01,
    max_bdd: float = 0.02,
    target_bdd: float = 0.5,
    midpoint_bdd: float = 0.25,
    fineness: int = 100,
    tol: float = 1e-2,
) -> list[np.ndarray]:
    """
    Build a continuous trajectory by:
      1. Picking two random poses with BDD_flat_fast ≈ target_bdd.
      2. Finding a third "mid" pose whose BDD to each is ≈ midpoint_bdd.
      3. Fitting a 3‐point spline (translations) & 3‐key Slerp (rotations).
      4. Sampling that arc, then greedily selecting from the original pose set
         those poses whose BDD to the last appended pose lies in [min_bdd, max_bdd]
         and whose translation best follows the spline.

    Args:
        poses_wc    : List of 4×4 world→camera poses.
        min_bdd     : Minimum allowed BDD between successive picks.
        max_bdd     : Maximum allowed BDD between successive picks.
        target_bdd  : Desired BDD for the initial control pair.
        midpoint_bdd: Desired BDD from each control to the mid‐control.
        fineness    : Samples along the fitted spline arc.
        tol         : Tolerance around target_bdd and midpoint_bdd.

    Returns:
        traj        : A list of 4×4 poses (a subsequence of the input list).
    """
    # extract rotations & translations
    rots = [Rotation.from_matrix(T[:3, :3]) for T in poses_wc]
    trans = [T[:3,  3] for T in poses_wc]
    N = len(poses_wc)

    # 1) find two random indices with BDD ≈ target_bdd
    idx1 = idx2 = None
    for _ in range(5000):
        i, j = random.sample(range(N), 2)
        if abs(metrics.BDD_flat_fast(rots[i], rots[j]) - target_bdd) < tol:
            idx1, idx2 = i, j
            break
    if idx1 is None:
        raise RuntimeError(f"Couldn't find initial pair with BDD≈{target_bdd}")

    # 2) find mid index with BDD≈midpoint_bdd to both
    idx_mid = None
    for _ in range(5000):
        k = random.randrange(N)
        if (abs(metrics.BDD_flat_fast(rots[idx1], rots[k]) - midpoint_bdd) < tol and
            abs(metrics.BDD_flat_fast(rots[idx2], rots[k]) - midpoint_bdd) < tol):
            idx_mid = k
            break
    if idx_mid is None:
        raise RuntimeError(f"Couldn't find mid‐control with BDD≈{midpoint_bdd}")

    # control translations & rotations
    ctrl_pts = np.vstack([trans[idx1], trans[idx_mid], trans[idx2]]).T  # (3,3)
    ctrl_rots = [rots[idx1], rots[idx_mid], rots[idx2]]

    # 3a) spline on translations (k=2 for 3 pts)
    tck, _ = splprep(ctrl_pts, s=0, k=2)
    u_vals = np.linspace(0, 1, fineness)
    sampled_trans = np.vstack(splev(u_vals, tck)).T  # (fineness,3)

    # 3b) Slerp on rotations with key_times [0,0.5,1]
    key_times = [0.0, 0.5, 1.0]
    slerp = Slerp(key_times, Rotation.concatenate(ctrl_rots))
    sampled_rots = slerp(u_vals)  # length= fineness

    # 4) greedy selection from original poses
    traj = [poses_wc[idx1]]
    used = {idx1}
    prev_idx = idx1
    ptr = 1
    while prev_idx != idx2 and ptr < fineness:
        # candidates satisfying BDD window to previous
        cands = [i for i in range(N)
                 if i not in used
                 and min_bdd <= metrics.BDD_flat_fast(rots[prev_idx], rots[i]) <= max_bdd]
        if not cands:
            ptr += 1
            continue
        # pick candidate whose translation is closest to spline point
        dists = [np.linalg.norm(trans[i] - sampled_trans[ptr]) for i in cands]
        next_idx = cands[int(np.argmin(dists))]
        traj.append(poses_wc[next_idx])
        used.add(next_idx)
        prev_idx = next_idx
        ptr += 1

    # ensure end pose included
    if traj[-1] is not poses_wc[idx2]:
        traj.append(poses_wc[idx2])

    return traj


def OLD_random_spline_trajectory_with_poses(
        max_len: float = 100,
        near: float = 0.001,
        far: float = 100,
        fineness: int = 200,
        num_ctrl: int = 3,
        seed: int = None,
        plot: bool = False,
        K = None,
        width = None,
        height = None
    ) -> list[np.ndarray]:
    """
    Generate a smooth random trajectory inside the camera's view frustum, and
    return a list of world->camera poses (4×4 matrices) for an object moving
    along that trajectory while its attitude SLERPs between a random initial
    and random final orientation.

    Args:
        pose      : 4×4 cam→world transform.
        near, far : Distances for near/far planes.
        max_len   : Max allowed distance between endpoints.
        fineness  : Number of samples along the spline.
        num_ctrl  : Number of random interior control points.
        seed      : RNG seed.
        plot      : If True, show Plotly 3D scene.

    Returns:
        poses_wc : List of length `fineness` of 4×4 world→camera transforms
                   for the object at each sample.
    """
    if seed is not None:
        np.random.seed(seed)

    # Random initial camera pose (cam→world)
    R_wc = Rotation.from_matrix(np.eye(3)).as_matrix()
    z = [0,0,-1]
    t_wc = Rotation.from_matrix(R_wc).apply(z)
    
    if width is None or height is None or K is None:
        raise ValueError("Provide width, height and K.")

    # 1) Compute the 8 corners of the view frustum in WORLD space
    corners_px = np.array([[0, 0], [width, 0], [width, height], [0, height]])

    corners_cam = np.vstack([
        backproject(u, v, near,  K) for (u, v) in corners_px
    ] + [
        backproject(u, v, far,   K) for (u, v) in corners_px
    ])  # shape (8, 3)

    corners_world = (R_wc @ corners_cam.T).T + t_wc  # (8, 3)

    # helper: sample 1 random point inside frustum
    # First a pixel is sampled within the screen, then it's backprojected
    # into world space with a random depth with the given constraints. 
    def sample_point():
        Z = np.random.uniform(near, far)
        u = np.random.uniform(0, width)
        v = np.random.uniform(0, height)
        cam_pt = backproject(u, v, Z, K)
        return R_wc @ cam_pt + t_wc

    # 2) Pick endpoints A, B with ||A - B|| <= max_len
    for _ in range(1000):
        A = sample_point()
        B = sample_point()
        if np.linalg.norm(A - B) <= max_len:
            break
    else:
        raise RuntimeError("Couldn't sample endpoints within max_len")

    # 3) Build control points (including A, B)
    control_pts = [A] + [sample_point() for _ in range(num_ctrl)] + [B]
    ctrl_arr = np.vstack(control_pts).T  # shape (3, K)

    # 4) Fit & sample a cubic B‐spline
    tck, _ = splprep(ctrl_arr, s=0, k=min(3, ctrl_arr.shape[1] - 1))
    u_fine = np.linspace(0, 1, fineness)
    pts = splev(u_fine, tck)
    traj_world = np.vstack(pts).T   # (fineness, 3)

    # 5) Generate random initial & final orientations (object→world)
    rot_init = Rotation.random()   # random uniform rotation
    rot_final = Rotation.random()
    # Set up SLERP between them
    key_rots = Rotation.concatenate([rot_init, rot_final])
    key_times = [0.0, 1.0]
    slerp = Slerp(key_times, key_rots)
    rots_interp = slerp(u_fine)  # array of Rotation objects, length = fineness

    # 6) For each sample, compute object→camera pose
    poses_wc = []
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc

    for i in range(fineness):
        # a) object→world
        R_ow = rots_interp[i].as_matrix()
        t_ow = traj_world[i]

        # b) world→camera
        #    [R_cw | t_cw]
        # c) object→camera = (world→camera) @ (object→world)
        R_co = R_cw @ R_ow
        t_co = R_cw @ t_ow + t_cw

        T_co = np.eye(4)
        T_co[:3, :3] = R_co
        T_co[:3,  3] = t_co
        poses_wc.append(T_co)

    # 7) Optional: visualize frustum, trajectory, and a few object frames
    if plot:
        fig = go.Figure()

        # ——— Frustum edges ———
        # near‐plane loop
        for idx in range(4):
            next_idx = (idx + 1) % 4
            fig.add_trace(go.Scatter3d(
                x=[corners_world[idx, 0], corners_world[next_idx, 0]],
                y=[corners_world[idx, 1], corners_world[next_idx, 1]],
                z=[corners_world[idx, 2], corners_world[next_idx, 2]],
                mode='lines', line=dict(color='gray'), showlegend=False
            ))
        # far‐plane loop
        for idx in range(4, 8):
            next_idx = 4 + ((idx + 1 - 4) % 4)
            fig.add_trace(go.Scatter3d(
                x=[corners_world[idx, 0], corners_world[next_idx, 0]],
                y=[corners_world[idx, 1], corners_world[next_idx, 1]],
                z=[corners_world[idx, 2], corners_world[next_idx, 2]],
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
            x=traj_world[:, 0], y=traj_world[:, 1], z=traj_world[:, 2],
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
        # Draw coordinate axes for the object at the first, middle, and last pose
        def draw_frame(T, scale=0.1, label=""):
            origin = T[:3, 3]
            R_mat = T[:3, :3]
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
            draw_frame(poses_wc[idx], scale=(max_len * 0.05), label=label)

        fig.update_layout(
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='data'
            ),
            title="Random Spline Trajectory and Object Poses"
        )
        fig.show()

    return poses_wc

def interpolate_pose(pose1: np.ndarray, pose2: np.ndarray, alpha: float) -> np.ndarray:
    """
    Interpolate between two poses (4×4 object→camera matrices).

    Args:
        pose1: (4×4) first pose.
        pose2: (4×4) second pose.
        alpha: interpolation factor in [0, 1].

    Returns:
        (4×4) interpolated pose.
    """
    # Extract rotations and translations
    R1 = Rotation.from_matrix(pose1[:3, :3])
    R2 = Rotation.from_matrix(pose2[:3, :3])
    t1 = pose1[:3, 3]
    t2 = pose2[:3, 3]

    # SLERP between rotations
    slerp = Slerp([0, 1], Rotation.concatenate([R1, R2]))
    R_interp = slerp([alpha])[0].as_matrix()

    # Linear interpolation of translation
    t_interp = (1 - alpha) * t1 + alpha * t2

    # Build 4×4 pose
    pose_interp = np.eye(4, dtype=np.float64)
    pose_interp[:3, :3] = R_interp
    pose_interp[:3,  3] = t_interp

    return pose_interp


def random_spline_trajectory_with_poses(
        K,
        height,
        width,
        max_len: float = 100,
        near: float = 0.001,
        far: float = 100,
        fineness: int = 200,
        num_ctrl: int = 3,
        seed: int = None,
        plot: bool = False
    ) -> list[np.ndarray]:
    """
    Generate a smooth random trajectory that stays strictly within the camera’s view frustum,
    and return a list of 4×4 pose matrices (object→world) sampled along that trajectory.

    The camera is assumed fixed at the origin looking along -Z, with R = I, t = 0.

    Args:
        K          : 3×3 camera intrinsics matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
        height     : Image height in pixels.
        width      : Image width in pixels.
        near, far  : Near/far plane distances (in camera‐space Z) where near < Z < far.
        max_len    : Maximum allowed Euclidean distance between the two spline endpoints.
        fineness   : Number of equally spaced samples along the B‐spline (default 200).
        num_ctrl   : Number of interior control points (in addition to endpoints).
        seed       : RNG seed (optional).
        plot       : If True, show a 3D plot of frustum + resulting trajectory (not implemented here).

    Returns:
        poses : List of length `fineness` of 4×4 pose matrices.  Each pose is the object‐to‐world
                transformation, so that applying it to a point in object‐space gives camera‐space.
                Concretely:  pose[:3, :3] = R_object→camera,  pose[:3, 3] = t_object→camera.
    """

    if seed is not None:
        np.random.seed(seed)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    def backproject(u, v, Z):
        """
        Return the 3D camera‐space point corresponding to pixel (u, v) at depth Z.
        """
        x_cam = (u - cx) * Z / fx
        y_cam = (v - cy) * Z / fy
        return np.array([x_cam, y_cam, Z], dtype=np.float64)

    def sample_point_inside_frustum(strict=True):
        """
        Sample a random 3D point within the pyramid defined by
        the frustum, normally pixel‐bounds [0,width]×[0,height] and depth [near, far].

        If strict, then we adapt this to be in the 50% smaller
        square at the center of the image from [25%,75%] of width and height.
        """
        Z = np.random.uniform(near, far)
        if strict:   
            u = np.random.uniform(0.25 * width, 0.75 * width)
            v = np.random.uniform(0.25 * height, 0.75 * height)
        else:    
            u = np.random.uniform(0, width)
            v = np.random.uniform(0, height)
        return backproject(u, v, Z)

    def trajectory_stays_in_frustum(traj_xyz: np.ndarray, strict=True) -> bool:
        """
        Given traj_xyz of shape (fineness, 3) in camera coords,
        check that every point projects to 0<=u<=width, 0<=v<=height, and near<=Z<=far.

        If strict is true, then it must lie in the inner two thirds of the image.
        """
        X = traj_xyz[:, 0]
        Y = traj_xyz[:, 1]
        Z = traj_xyz[:, 2]

        # Depth check
        if np.any(Z < near) or np.any(Z > far):
            return False

        # Project into pixels
        u_all = (fx * X) / Z + cx
        v_all = (fy * Y) / Z + cy

        if strict:
            u_min, u_max = width / 6, width * 5 / 6
            v_min, v_max = height / 6, height * 5 / 6
            if np.any(u_all < u_min) or np.any(u_all > u_max):
                return False
            if np.any(v_all < v_min) or np.any(v_all > v_max):
                return False
        else:
            if np.any(u_all < 0) or np.any(u_all > width):
                return False
            if np.any(v_all < 0) or np.any(v_all > height):
                return False

        return True

    # Try up to N attempts to get a “fully‐in‐frustum” spline:
    max_attempts = 1000
    for attempt in range(max_attempts):
        # 1) Pick two endpoints A, B inside the frustum s.t. ||A-B|| <= max_len
        for _ in range(1000):
            A = sample_point_inside_frustum()
            B = sample_point_inside_frustum()
            if np.linalg.norm(A - B) <= max_len:
                break
        else:
            # If we failed to find endpoints quickly, try a fresh attempt
            continue

        # 2) Build (num_ctrl) random interior control points (all inside frustum)
        control_pts = [A]
        for _ in range(num_ctrl):
            control_pts.append(sample_point_inside_frustum())
        control_pts.append(B)

        # Stack into shape (3, K) for splprep
        ctrl_arr = np.vstack(control_pts).T  # (3, num_ctrl+2)

        # 3) Fit a cubic B‐spline through these control points
        degree = min(3, ctrl_arr.shape[1] - 1)
        tck, _ = splprep(ctrl_arr, s=0, k=degree)

        # 4) Sample the spline densely
        u_fine = np.linspace(0, 1, fineness)
        spline_pts = splev(u_fine, tck)
        traj_cam = np.vstack(spline_pts).T  # shape (fineness, 3)

        # 5) Check that every sampled point stays inside the frustum
        if not trajectory_stays_in_frustum(traj_cam):
            # If any point is out of bounds, discard this set of control points
            continue

        # If we reach here, the spline is fully inside — accept it:
        break
    else:
        raise RuntimeError(f"Could not generate a fully‐in‐frustum spline after {max_attempts} attempts.")

    # 6) Generate random initial + final orientations (object → camera)
    rot_init  = Rotation.random()
    rot_final = Rotation.random()
    key_rots  = Rotation.concatenate([rot_init, rot_final])
    key_times = [0.0, 1.0]
    slerp     = Slerp(key_times, key_rots)
    rots_interp = slerp(u_fine)           # array of Rotation objects of length `fineness`
    rots_interp_mats = [r.as_matrix() for r in rots_interp]

    # 7) Build the final 4×4 pose matrices for each sample
    poses = []
    for i in range(fineness):
        R_obj2cam = rots_interp_mats[i]            # 3×3 rotation: object → camera
        t_obj2cam = traj_cam[i]                     # 3-vector: object position in camera coords

        # We want a 4×4 'pose' so that:
        #    [ X_cam ]   [ R_obj2cam   t_obj2cam ] [ X_obj ]
        #    [   1   ] = [   0   1       1       ] [   1   ]
        #
        # In homogeneous form, “pose = [R | t; 0 1]” maps object→camera.
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R_obj2cam
        pose[:3,  3] = t_obj2cam
        poses.append(pose)

    return poses

from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

def spline_trajectory_from_poses(
    poses: list[np.ndarray],
    num_ctrl: int = 5,
    fineness: int = 100,
    seed: int = None,
    plot: bool = False
) -> list[np.ndarray]:
    """
    Given an input trajectory (list of 4×4 world→cam poses),
    pick `num_ctrl` control poses (evenly spaced, including first & last),
    then interpolate between them with:
      - cubic spline for translations
      - SLERP for rotations

    Args:
        poses     : list of shape-(4,4) homogeneous matrices.
        num_ctrl  : total number of control frames (must be >=2 and <= len(poses)).
        fineness  : number of output poses along the new trajectory.
        seed      : optional RNG seed (only affects control‐point selection if randomized).
        plot      : if True, shows a quick 3D plot of control vs. interpolated traj.

    Returns:
        new_poses : list of length `fineness` of 4×4 pose matrices.
    """
    N = len(poses)
    if num_ctrl < 2 or num_ctrl > N:
        raise ValueError(f"num_ctrl must be between 2 and {N}")

    # 1) pick control indices evenly
    ctrl_idxs = np.linspace(0, N-1, num_ctrl)
    ctrl_idxs = np.unique(np.round(ctrl_idxs).astype(int))  # ensure integers & unique
    # if rounding collapsed endpoints, fix them
    ctrl_idxs[0], ctrl_idxs[-1] = 0, N-1

    # 2) extract control translations & rotations
    ctrl_t = np.stack([poses[i][:3, 3] for i in ctrl_idxs], axis=0)  # shape (M,3)
    ctrl_R = [Rotation.from_matrix(poses[i][:3, :3]) for i in ctrl_idxs]
    M = len(ctrl_idxs)

    # 3) build time axes
    ctrl_times = np.linspace(0.0, 1.0, M)
    fine_times = np.linspace(0.0, 1.0, fineness)

    # 4) cubic spline on each translation component
    cs_x = CubicSpline(ctrl_times, ctrl_t[:, 0])
    cs_y = CubicSpline(ctrl_times, ctrl_t[:, 1])
    cs_z = CubicSpline(ctrl_times, ctrl_t[:, 2])
    fine_t = np.stack([cs_x(fine_times),
                       cs_y(fine_times),
                       cs_z(fine_times)], axis=1)  # (fineness,3)

    # 5) SLERP on rotations
    slerp = Slerp(ctrl_times, Rotation.concatenate(ctrl_R))
    fine_R = slerp(fine_times)  # Rotation array of length fineness

    # 6) build new 4×4 poses
    new_poses = []
    for i in range(fineness):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = fine_R[i].as_matrix()
        T[:3, 3]  = fine_t[i]
        new_poses.append(T)

    # 7) optional quick 3D plot of control vs. interp
    if plot:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import matplotlib.pyplot as plt

        ctrl_xyz = ctrl_t
        fine_xyz = fine_t

        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot(ctrl_xyz[:,0], ctrl_xyz[:,1], ctrl_xyz[:,2],
                'o-', label='controls', linewidth=2)
        ax.plot(fine_xyz[:,0], fine_xyz[:,1], fine_xyz[:,2],
                '-',  label='spline',    linewidth=1)
        ax.set_title("Trajectory interpolation")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return new_poses
