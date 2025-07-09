import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
import cv_utils
from skimage.metrics import structural_similarity
from matplotlib import pyplot as plt
import json
import plotly.graph_objects as go
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

def IOU(mask1, mask2):
    mask1 = np.asarray(mask1).astype(bool)
    mask2 = np.asarray(mask2).astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def BDD_flat_fast(rot1,rot2):
    """
    Optimized equivalent of the original NPD(…) that avoids scipy Rotation / generic angle_between_vectors.

    Args:
        rot1,rot2: 1x9 flattened rotations matrices
    Returns:
        npd: non-planar distance (same output as original implementation)
    """
    # 1) Extract R1, R2
    R1 = rot1.reshape((3,3))
    R2 = rot2.reshape((3,3))

    # 2) Relative rotation: R = R1 @ R2^T
    R = R1 @ R2.T

    # 3) Compute rotation angle theta from the trace:
    #    cos(theta) = (trace(R) - 1) / 2
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    cos_theta = (tr - 1.0) * 0.5
    # clamp to [-1, +1] to avoid numerical issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # 4) Extract rotation axis:
    #    If theta is very small, just pick [0,0,1]. Otherwise:
    #      axis = (1 / (2*sin(theta))) * [ R32 - R23, R13 - R31, R21 - R12 ]
    if theta < 1e-8:
        axis = np.array([0.0, 0.0, 1.0], dtype=R.dtype)
    else:
        sin_theta = np.sin(theta)
        # Note: if sin(theta) is almost zero, this division can blow up;
        #       but theta<1e-8 caught above handles that region.
        rx = (R[2, 1] - R[1, 2]) / (2.0 * sin_theta)
        ry = (R[0, 2] - R[2, 0]) / (2.0 * sin_theta)
        rz = (R[1, 0] - R[0, 1]) / (2.0 * sin_theta)
        axis = np.array([rx, ry, rz], dtype=R.dtype)
        norm_axis = np.linalg.norm(axis)
        if norm_axis < 1e-12:
            # Degenerate / numerical fallback
            axis = np.array([0.0, 0.0, 1.0], dtype=R.dtype)
        else:
            axis /= norm_axis

    # 5) Make the axis’s z-component nonnegative (just like `abs(rot_axis[2])` in the original).
    a_z_pos = abs(axis[2])

    # 6) Compute phi = angle between [axis_x, axis_y, |axis_z|] and [0,0,1].
    #    Since [axis_x, axis_y, |axis_z|] is unit-norm whenever axis was unit,
    #    phi = arccos( ( [axis_x,axis_y,|axis_z|] · [0,0,1] ) ) = arccos(|axis_z|)
    #    This exactly replicates angle_between_vectors(...) for your target_axis = [0,0,1].
    phi = np.arccos(a_z_pos)

    # 7) Final “NPD” formula as in original: npd = |θ/π| * (1 − |2|φ|/π − 1|)
    #    (all absolute‐value logic is unchanged)
    npd = abs((theta / np.pi)) * (1.0 - abs(2.0 * abs(phi) / np.pi - 1.0))
    return npd


def BDD(pose1,pose2,rotation_matrices_input=False):
    """
    Python equivalent of the MATLAB nonPlanarDistanceFromPose function.
    
    Args:
        pose1: 4x4 passive target2cam for image1    
        pose2: 4x4 passive target2cam for image2
        rotation_matrices_input: If True, input is expected to be 3x3 rotation
            matrices instead. False by default.
    
    Returns:
        npd: NPD score between the poses.
    """
    if not rotation_matrices_input:
        # Extract components
        R1 = pose1[:3,:3]
        R2 = pose2[:3,:3]
    else:
        R1 = pose1
        R2 = pose2

    # Relative rotation
    rot_rel_cam = R1 @ R2.T

    # Axis-angle
    rotvec = Rotation.from_matrix(rot_rel_cam).as_rotvec()
    rot_ang = np.linalg.norm(rotvec)
    rot_axis = rotvec / rot_ang if rot_ang != 0 else np.array([0, 0, 1])
    rot_axis = np.array([rot_axis[0], rot_axis[1], abs(rot_axis[2])])

    # Angle between axes
    target_axis = [0,0,1] # boresight, ideal case
    angle_distance_axes = cv_utils.angle_between_vectors(rot_axis, target_axis)

    # Final cost
    theta = rot_ang
    phi = angle_distance_axes

    npd = abs(theta/np.pi)*(1-abs(2*abs(phi)/np.pi - 1))
    return npd

def oks(y_true, y_pred, visibility):
    # You might want to set these global constant
    # outside the function scope
    KAPPA = np.array([1] * len(y_true))
    # The object scale
    # You might need a dynamic value for the object scale
    SCALE = 1.0

    # Compute the L2/Euclidean Distance
    distances = np.linalg.norm(y_pred - y_true, axis=-1)
    # Compute the exponential part of the equation
    exp_vector = np.exp(-(distances**2) / (2 * (SCALE**2) * (KAPPA**2)))
    # The numerator expression
    numerator = np.dot(exp_vector, visibility.astype(bool).astype(int))
    # The denominator expression
    denominator = np.sum(visibility.astype(bool).astype(int))
    return numerator / denominator

def keypoint_metrics(kp1, kp2, visibility=None, threshold=0.1):
    """
    Computes the average L2 distance and object keypoint score between two sets of 2D keypoints.

    Args:
        kp1: (N, 2) array of keypoints (e.g., ground truth)
        kp2: (N, 2) array of keypoints (e.g., prediction)
        visibility: Optional (N,) boolean array indicating which keypoints are visible/valid.
        threshold: Distance threshold (in pixels or normalized units) for OKS.

    Returns:
        (avg_l2_dist, oks): tuple of average L2 distance and object keypoint score
    """
    kp1 = np.asarray(kp1)
    kp2 = np.asarray(kp2)
    if visibility is not None:
        mask = np.asarray(visibility).astype(bool)
        kp1m = kp1[mask]
        kp2m = kp2[mask]
    if kp1.shape[0] == 0:
        return float('nan'), float('nan')
    dists = np.linalg.norm(kp1m - kp2m, axis=1)
    avg_l2_dist = np.mean(dists)
    oks_score = oks(kp1,kp2,visibility) # OKS gets unmasked
    return avg_l2_dist, oks_score

def ssim(img1,img2):
    return structural_similarity(img1,img2,channel_axis=2)


def compute_largest_empty_ball(sample_rotations, space_rotations):
    """
    Estimate the largest empty ball in rotation space with npd metric.

    Args:
        sample_rotations (np.ndarray): (N, 3, 3) rotation matrices.
        space_rotations (np.ndarray): (N, 3, 3) rotations in the space which shall
          be covered by the sample_rotations.
    Returns:
        float: Approximation of the largest empty ball radius.
    """

    # Compute min npd for each random rotation to any of the samples
    min_dists = []
    min_dists_knn = []
    # Create KNN implementation
    bdd_nbrs = NearestNeighbors(
        n_neighbors=1,           # how many neighbors you want back
        algorithm='brute',        # brute‐force is required for a callable metric
        metric=BDD_flat_fast
    )
    # Build tree with flattened rotations
    bdd_nbrs.fit(np.array([rot.reshape(1,9) for rot in space_rotations]).squeeze())

    for ix,R_random in enumerate(space_rotations):
        if ix%10 == 0:
          print(f"Comparison {ix}")

        brute_dists = [BDD_flat_fast(R_random.flatten(), R_sample.flatten()) for R_sample in sample_rotations]
        knn_dists, knn_indices = bdd_nbrs.kneighbors(R_random.reshape(1, -1), n_neighbors=1)

        min_dists.append(min(brute_dists))
        min_dists_knn.append(min(knn_dists))

    # The largest empty ball radius is the maximum of these minimal distances
    assert max(min_dists) == max(min_dists_knn)
    
    largest_empty_ball_radius = max(min_dists)
    return largest_empty_ball_radius


def compute_interset_largest_nn(
    sample_rotations,
    approximate=False,
    tol=1e-3,
    patience_frac=0.2,
    shuffle=True
):
    """
    Compute (or approximate) the largest nearest‐neighbor distance among a set of rotations.

    If `approximate=True`, we stop as soon as the running max no longer improves by > tol
    for `patience = int(patience_frac * N)` consecutive steps.

    Args:
        sample_rotations (np.ndarray or List[np.ndarray]):
            Array of shape (N, 3, 3).  Each entry is a rotation matrix in SO(3).

        approximate (bool):
            If False (default), do the full exact scan (n_neighbors=2 for each of N → O(N^2) calls,
            though we still use a single KNN index so it’s O(N·log N) within sklearn).  
            If True, we exit early once the max‐distance hasn’t changed by > tol for `patience` steps.

        tol (float):
            Absolute tolerance on “new_max – old_max.”  Only if
                  nearest_other > current_max + tol
            do we treat it as a real improvement.

        patience_frac (float in (0,1)):
            Fraction of N that determines how many consecutive “no improvement” steps we allow
            before stopping.  Internally:
                patience = max(1, int(patience_frac * N)).

        random_seed (int or None):
            If provided, seeds np.random so the shuffle is reproducible.

    Returns:
        float:  
            If N < 2, returns 0.0.  
            Otherwise, returns (exactly if approximate=False, or early‐stopped if True):
            max_i [ min_{j≠i} BDD_flat_fast(R_i, R_j) ].

    Raises:
        ValueError: If sample_rotations has invalid shape.
    """
    # Convert to np.ndarray if needed
    if isinstance(sample_rotations, list):
        sample_rotations = np.array(sample_rotations)
    N = len(sample_rotations)

    # Degenerate cases
    if N < 2:
        return 0.0

    # Flatten into shape (N,9)
    flattened = sample_rotations.reshape(N, 9)

    # Build the KNN index once: queries will use k=2 (point itself + nearest other)
    knn = NearestNeighbors(n_neighbors=2, algorithm='brute', metric=BDD_flat_fast)
    knn.fit(flattened)

    # If approximate, set up patience and optionally shuffle
    if approximate:
        idxs = np.arange(N)

        if shuffle:
            np.random.seed(0)
            np.random.shuffle(idxs)

        flattened = flattened[idxs]

        patience = max(1, int(patience_frac * N))
        no_improve = 0
        current_max = 0.0

        for R in flattened:
            dists, _ = knn.kneighbors(R.reshape(1, -1), n_neighbors=2)
            nearest_other = dists[0, 1]

            # Only update if it exceeds current_max by more than tol
            if nearest_other > current_max + tol:
                current_max = nearest_other
                no_improve = 0
            else:
                no_improve += 1

            # If we’ve gone `patience` steps with no improvement, bail out
            if no_improve >= patience:
                return float(current_max)

        # If we never triggered early stopping, return the exact max
        return float(current_max)

    else:
        # exact mode: no early stopping, just compute all distances
        nearest_other_dists = []
        for R in flattened:
            dists, _ = knn.kneighbors(R.reshape(1, -1), n_neighbors=2)
            nearest_other_dists.append(dists[0, 1])
        return float(max(nearest_other_dists))
    

def compute_interset_largest_nn(
    sample_rotations,
    approximate=False,
    tol=1e-3,
    patience_frac=0.2,
    shuffle=True,
    return_average=False
):
    """
    Compute (or approximate) the largest nearest‐neighbor distance among a set of rotations.

    If `approximate=True`, we stop as soon as the running max no longer improves by > tol
    for `patience = int(patience_frac * N)` consecutive steps.

    Args:
        sample_rotations (np.ndarray or List[np.ndarray]):
            Array of shape (N, 3, 3).  Each entry is a rotation matrix in SO(3).

        approximate (bool):
            If False (default), do the full exact scan (n_neighbors=2 for each of N → O(N^2) calls,
            though we still use a single KNN index so it’s O(N·log N) within sklearn).  
            If True, we exit early once the max‐distance hasn’t changed by > tol for `patience` steps.

        tol (float):
            Absolute tolerance on “new_max – old_max.”  Only if
                  nearest_other > current_max + tol
            do we treat it as a real improvement.

        patience_frac (float in (0,1)):
            Fraction of N that determines how many consecutive “no improvement” steps we allow
            before stopping.  Internally:
                patience = max(1, int(patience_frac * N)).

        random_seed (int or None):
            If provided, seeds np.random so the shuffle is reproducible.

        return_average (bool):
            If True, returns also average nn dist. False by default.

    Returns:
        float:  
            If N < 2, returns 0.0.  
            Otherwise, returns (exactly if approximate=False, or early‐stopped if True):
            max_i [ min_{j≠i} BDD_flat_fast(R_i, R_j) ].
            (optionally also returns average nn distance if return_average arg is True.)

    Raises:
        ValueError: If sample_rotations has invalid shape.
    """
    # Convert to np.ndarray if needed
    if isinstance(sample_rotations, list):
        sample_rotations = np.array(sample_rotations)
    N = len(sample_rotations)

    # Degenerate cases
    if N < 2:
        return 0.0

    # Flatten into shape (N,9)
    flattened = sample_rotations.reshape(N, 9)

    # Build the KNN index once: queries will use k=2 (point itself + nearest other)
    knn = NearestNeighbors(n_neighbors=2, algorithm='brute', metric=BDD_flat_fast)
    knn.fit(flattened)

    # Keep track of NN distance vals
    nearest_other_dists = []

    # If approximate, set up patience and optionally shuffle
    if approximate:
        idxs = np.arange(N)

        if shuffle:
            np.random.seed(0)
            np.random.shuffle(idxs)

        flattened = flattened[idxs]

        patience = max(1, int(patience_frac * N))
        no_improve = 0
        current_max = 0.0

        for R in flattened:
            dists, _ = knn.kneighbors(R.reshape(1, -1), n_neighbors=2)
            nearest_other = dists[0, 1]
            nearest_other_dists.append(dists[0, 1])

            # Only update if it exceeds current_max by more than tol
            if nearest_other > current_max + tol:
                current_max = nearest_other
                no_improve = 0
            else:
                no_improve += 1

            # If we’ve gone `patience` steps with no improvement, bail out
            if no_improve >= patience:
                if return_average:
                    return (float(current_max),float(np.mean(nearest_other_dists)))
                else:
                    return float(current_max)

        # If we never triggered early stopping, return the exact max
        return float(current_max)

    else:
        # exact mode: no early stopping, just compute all distances
        for R in flattened:
            dists, _ = knn.kneighbors(R.reshape(1, -1), n_neighbors=2)
            nearest_other_dists.append(dists[0, 1])
        if return_average:
            return (float(max(nearest_other_dists)),float(np.mean(nearest_other_dists)))
        else:
            return float(max(nearest_other_dists))
    

def random_unit_vector():
    """
    Returns one 3-vector uniformly distributed on the unit sphere S^2.
    """
    # 1) Draw two uniforms in [0,1):
    r1 = np.random.rand()
    r2 = np.random.rand()
    # 2) Let z = 2*r1 − 1  (cos θ), and phi = 2π * r2:
    z   = 2*r1 - 1
    phi = 2 * np.pi * r2
    # 3) Compute sinθ = sqrt(1 − z²)
    s = np.sqrt(max(0.0, 1 - z*z))
    x = s * np.cos(phi)
    y = s * np.sin(phi)
    return np.array([x, y, z])


def random_unit_vectors(N_vectors: int):
    """
    Returns N_vectors vectors uniformly distributed on the unit sphere S^2.
    """
    vectors = np.array([random_unit_vector() for i in range(N_vectors)])
    return vectors


def compute_mesh_diameter(mesh):
    """
    Compute the maximum Euclidean distance (diameter) between any two points on a trimesh Trimesh object,
    using its convex hull.

    Args:
        mesh (trimesh.Trimesh):
            A Trimesh object representing the 3D surface.

    Returns:
        max_dist (float):
            The maximum Euclidean distance between any two vertices on the mesh’s convex hull.
        endpoints (np.ndarray):
            A (2, 3) array containing the coordinates of the two farthest points on the hull.
    """
    # 1) Compute convex hull of the mesh and get its vertices
    hull = mesh.convex_hull
    hull_pts = np.asarray(hull.vertices)  # (H, 3)

    # 2) Compute pairwise distances between all hull vertices
    D = cdist(hull_pts, hull_pts)  # shape (H, H)

    # 3) Find the indices (i, j) of the maximum distance
    idx_flat = np.argmax(D)
    i, j = np.unravel_index(idx_flat, D.shape)
    max_dist = float(D[i, j])
    p_i = hull_pts[i]
    p_j = hull_pts[j]
    endpoints = np.vstack((p_i, p_j))  # shape (2, 3)

    return max_dist, endpoints




def generate_dense_random_rotations(npd_threshold=0.01, max_iter=10000, verbose=True):
    """
    Generate random rotation matrices until the maximum NPD between each random rotation
    and its nearest neighbor in the set is below the given threshold.

    Args:
        npd_threshold (float): Maximum allowable NPD distance to the nearest neighbor.
        max_iter (int): Maximum number of iterations/samples to avoid infinite loop.
        verbose (bool): Print progress if True.

    Returns:
        rotations (list of np.ndarray): List of 3x3 rotation matrices densely covering SO(3).
    """
    rotations = []
    # Generate first point
    rotations.append(Rotation.random().as_matrix())

    # Counting variables
    num_samples = 1

    # For plotting
    max_dist_history = []

    # Constant for number of candidates based on number of samples
    m = 10
    for i in range(max_iter):
        # Generate next candidates based on current number of samples
        n_candidates = num_samples*m+100

        candidates = Rotation.random(n_candidates).as_matrix()

        # Compute NPD to existing rotations
        min_dists = []
        for candidate in candidates:
            dists = [BDD(candidate, R_existing) for R_existing in rotations]
            min_dist = min(dists)
            min_dists.append(min_dist)
        max_index = np.argmax(min_dists)
        max_dist = np.max(min_dists)
        max_dist_history.append(max_dist)
        print(f"Iteration: {i}, max_dist: {max_dist}")
        good = candidates[max_index]
        num_samples += 1
        rotations.append(good)

        if max_dist < npd_threshold:
            print(f"Converged at iteration {i+1}. Total rotations: {len(rotations)}")
            break

    if verbose and i == max_iter - 1:
        print("Reached maximum iterations without full convergence.")

    plt.figure(figsize=(6, 4))
    plt.plot(
        np.arange(1, len(max_dist_history) + 1),
        max_dist_history,
        marker="o",
        linestyle="-",
        color="C0"
    )
    plt.axhline(npd_threshold, color="red", linestyle="--", label=f"Threshold = {npd_threshold}")
    plt.xlabel("Iteration")
    plt.ylabel("Max‐min NPD")
    plt.title("Convergence of Farthest‐Point Sampling in SO(3)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return rotations

if __name__ == "__main__":
    #plot_mc_results(r".\assets\monte_carlo\10000_20250531015829.json")
    result = [r.tolist() for r in generate_dense_random_rotations()]
    #Dataset.write_data_to_json(result, "even_rots.json")