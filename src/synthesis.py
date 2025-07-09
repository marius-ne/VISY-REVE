import cv_utils.keypoints
import numpy as np
import cv2
import largestinteriorrectangle as lir
from typing import Optional, Tuple

# For linting the Sample class, cannot import directly due to circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Sample import Sample

import cv_utils

def fill_OOFA(deformed_img: np.ndarray, 
              deformed_mask: np.ndarray) -> np.ndarray:
    """
    Fill the black (0) areas of the deformed image using background patches
    and smooth the seams for a natural appearance.
    
    Args:
        deformed_img: 3-channel (HxWx3) color image, with black areas (0) to be filled.
        deformed_mask: 1-channel (HxW) mask; 1 for foreground, 0 for background,
            255 for OOFA.
    
    Returns:
        Filled image with smoothed seams.
    """
    # Copy to avoid overwriting original
    result = deformed_img.copy()

    # OOFA mask from deformed mask, will be 1 at OOFA
    # Cannot use deformed_mask directly since it's 1 at the object
    oofa_val = 255
    oofa_mask = (deformed_mask == oofa_val).astype(np.uint8)  # 1 where OOFA
    
    # OOFA edges, for later inpainting
    oofa_edges = cv2.Canny(oofa_mask*255,threshold1=100,threshold2=200)

    # Original background from deformed mask
    bg_mask = (deformed_mask == 0).astype(np.uint8) # 1 where bg
    cv_grid = bg_mask.astype("uint8") * 255
    contours, _ = \
        cv2.findContours(cv_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour = contours[0][:, 0, :]
    bg_mask_bool = bg_mask.astype(bool)
    bg_lir = lir.lir(bg_mask_bool, contour) # [x,y,w,h]
    bg_lir_w = bg_lir[2]
    bg_lir_h = bg_lir[3]
    bg_lir_x = bg_lir[0]
    bg_lir_y = bg_lir[1]
    bg_lir_area = bg_lir_w * bg_lir_h
    debug_lir = result.copy()
    cv2.rectangle(debug_lir, (bg_lir_x, bg_lir_y), (bg_lir_x + bg_lir_w - 1, bg_lir_y + bg_lir_h - 1), (255, 0, 0), 8)

    # Extract background region *from deformed image*
    # could also be done from original image
    background = cv2.bitwise_and(deformed_img, deformed_img, mask=bg_mask)


    # Find connected components in OOFA mask
    # TODO: verify that bg regions are not taken (those with value 0)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(oofa_mask, connectivity=8)
    
    for label in range(num_labels):
        # Bounding box of the region
        x, y, w, h, area = stats[label]

        # Draw bounding box for debugging (red border)
        debug = result.copy()
        cv2.rectangle(debug, (x, y), (x + w - 1, y + h - 1), (255, 0, 0), 5)

        if area == 0:
            continue  # Skip empty regions

        # Get the mask for this connected component (region)
        component_mask = (labels[y:y+h, x:x+w] == label)

        # Get the pixel values inside this component from the deformed image
        component_pixels = deformed_mask[y:y+h, x:x+w][component_mask]

        # Check that we have an OOFA component
        if not np.all(component_pixels == oofa_val):
            continue
        

        # If section is larger than lir we have to do extra processing
        if h > bg_lir_h or w > bg_lir_w:
            print("Warning: OOFA larger than LIR")
            sub_h = h
            sub_w = w
            while sub_h > bg_lir_h or sub_w > bg_lir_w:
                diff_h = sub_h - bg_lir_h
                diff_w = sub_w - bg_lir_w
                if diff_h > diff_w:
                    sub_h = np.ceil(sub_h / 2).astype(np.int32)
                else:
                    sub_w = np.ceil(sub_w / 2).astype(np.int32)
                
            # After the while loop: sub_h <= bg_lir_h, sub_w <= bg_lir_w

            # Generate smaller rectangles to tile the original region
            rectangles = []
            for yy in range(y, y + h, sub_h):
                for xx in range(x, x + w, sub_w):
                    # Clip to stay inside (x, y, w, h)
                    actual_w = min(sub_w, x + w - xx)
                    actual_h = min(sub_h, y + h - yy)
                    rectangles.append((xx, yy, actual_w, actual_h))
        else:
            rectangles = [(x,y,w,h)]

        # Fill the OOFA regions
        for rectangle in rectangles:
            xx, yy, ww, hh = rectangle
            
            # Find all possible top-left coordinates within bg_lir that allow a patch of size (ww, hh)
            valid_x = np.arange(bg_lir[0], bg_lir[0] + bg_lir_w - ww + 1)
            valid_y = np.arange(bg_lir[1], bg_lir[1] + bg_lir_h - hh + 1)
            if len(valid_x) == 0 or len(valid_y) == 0:
                # Fallback to default topleft if patch doesn't fit
                topleft_x = bg_lir[0]
                topleft_y = bg_lir[1]
            else:
                topleft_x = np.random.choice(valid_x)
                topleft_y = np.random.choice(valid_y)

            # Copy patch from lir
            bg_patch = background[topleft_y:topleft_y+hh, topleft_x:topleft_x+ww]
            
            # Paste into result
            # Only fill pixels where the corresponding oofa_mask patch is 1
            oofa_patch = oofa_mask[yy:yy+hh, xx:xx+ww].astype(bool)
            result_patch = result[yy:yy+hh, xx:xx+ww]
            result_patch[oofa_patch] = bg_patch[oofa_patch]
            result[yy:yy+hh, xx:xx+ww] = result_patch
            # Add patch edges for inpainting to oofa edges mask
            # Only set edge pixels where oofa_patch is True
            # Top edge
            if hh > 0:
                oofa_edges[yy, xx:xx+ww][oofa_patch[0, :]] = 255
            # Bottom edge
            if hh > 1:
                oofa_edges[yy+hh-1, xx:xx+ww][oofa_patch[-1, :]] = 255
            # Left edge
            if ww > 0:
                oofa_edges[yy:yy+hh, xx][oofa_patch[:, 0]] = 255
            # Right edge
            if ww > 1:
                oofa_edges[yy:yy+hh, xx+ww-1][oofa_patch[:, -1]] = 255

    # Smooth seams using inpainting to reduce hard edges
    # Thicken the OOFA edges 
    edge_kernel_size = 10  # Adjust for desired thickness
    thick_edges = cv2.dilate(oofa_edges, np.ones((edge_kernel_size, edge_kernel_size), np.uint8))

    inpainted = cv2.inpaint(result, thick_edges, 3, cv2.INPAINT_NS)

    # Final smoothing of filled areas
    #blurred = cv2.GaussianBlur(inpainted, (3, 3), 0)

    return result

def Homography(source: "Sample",
                target: "Sample",
                with_axes: bool = False,
                fill_bg: bool = False,
                with_kps: bool = False,
                return_transformed_kps: bool = False,
                return_transformed_mask: bool = False,
                mask_input: bool = False,
                target_is_pose: bool = False) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Performs the Homography transform.
    
    Args:
        source: Source sample, must have pose and image.
        target: Target sample, must have pose and not have image.
        with_axes: Whether or not to draw pose axes on images, by default False.
        fill_bg: Whether or not to fill background, by default False.
        draw_transformed_kps: Whether or not to draw transformed keypoints, by default False.
            If true, you need to supply 2D keypoints in the source object.
        return_transformed_kps: Whether or not to return transformed keypoints. Useful for
            evaluating transformation accuracy. False by default.
        return_transformed_mask: Whether or not to also return the transformed mask of
            the source image. Useful for computing e.g. IOU.
        mask_input: Whether or not to mask the input image. True by default.
        target_is_pose: Whether or not the target is a pose or a sample object.
    Returns:
        synth: Transformed image.
        points_transformed: Transformed 2D points (if `points_2d` given), else None.
    """
    # Normal vector of plane
    n1 = np.array([0, 0, 1], dtype=np.float64)
    d1 = np.linalg.norm(source.t)

    # Relative pose matrices
    T1 = source.pose
    if not target_is_pose:
        T2 = target.pose
    else: 
        T2 = target

    # Compute D
    D12 = T2 @ np.linalg.inv(T1)

    R12 = D12[:3, :3]
    t12 = D12[:3, 3]

    # Compute H1->2 in world coordinates
    H12 = R12 + np.outer(t12, n1) / d1

    # Get K
    if not target_is_pose:
        assert np.all(source._K == target._K), "Camera matrices must be equal for synthesis."
    K = source._K

    # Projective transform
    G = K @ H12 @ np.linalg.inv(K)
    
    # Warp the image
    h, w = source.dataset.height, source.dataset.width

    if with_axes and not with_kps:
        img = source.image_with_axes 
    elif with_axes and with_kps and not mask_input:
        img = source.image_with_axes_and_kps
    elif mask_input and not with_axes and not with_kps:
        img = source.masked_image
    elif mask_input and with_axes and with_kps:
        img = source.masked_image_with_axes_and_kps
    else:
        img = source.image

    synth = cv2.warpPerspective(img, G, (w, h))

    if fill_bg or return_transformed_mask:
        source_mask = source.mask
        synth_mask = cv2.warpPerspective(source_mask, G, (w, h),
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=0)
        if fill_bg:
            synth_filled = fill_OOFA(synth, synth_mask)
            synth = synth_filled

    # source.dataset.write_synthesized(source=source,
    #                                  target=target,
    #                                  synth=synth)

    # Transform the 2D points (if provided)
    if return_transformed_kps:
        points_transformed = None

        if source.keypoints2D is not None:
            # Assume source.keypoints2D has shape (N, 2) or (N, 3)
            kp_src = source.keypoints2D.astype(np.float32)  # (N, 2 or 3)
            N = kp_src.shape[0]

            # Extract initial visibility: if a 3rd column exists, use it; otherwise assume all visible.
            if kp_src.shape[1] >= 3:
                init_vis = (kp_src[:, 2] > 0).astype(np.uint8)  # 1 or 0
            else:
                init_vis = np.ones(N, dtype=np.uint8)

            # Perform the perspective transform on the (u,v) pairs
            # cv2.perspectiveTransform expects shape (1, N, 2)
            pts_uv = kp_src[:, :2][None, :, :]  # (1, N, 2)
            pts_proj = cv2.perspectiveTransform(pts_uv, G)[0]  # (N, 2)

            # Now add projected visibility (1 if inside [0,w)×[0,h), else 0)
            proj_kps = cv_utils.keypoints.add_visibility_to_keypoints(pts_proj, (w, h))
            # proj_kps has shape (N, 3): (u_proj, v_proj, proj_vis)

            # Combine with initial visibility
            final_kps = proj_kps.copy()
            final_kps[:, 2] = init_vis * proj_kps[:, 2]

            points_transformed = final_kps.astype(np.float32)  # shape (N, 3)
        else:
            raise ValueError(
                "If you want transformed keypoints, please provide 2D keypoints in source sample object."
            )
        
    if return_transformed_kps and return_transformed_mask:
        return synth, points_transformed, synth_mask
    elif return_transformed_kps:
        return synth, points_transformed
    elif return_transformed_mask:
        return synth, synth_mask
    else:
        return synth

#@profile   
def Transform3D(
    source: "Sample",
    target: "Sample",
    interpolate: bool = False,
    maskOutput: bool = False,
    with_axes: bool = False,
    with_kps: bool = False,
    return_transformed_kps: bool = False,
    return_transformed_mask: bool = False,
    mask_input: bool = False,
    target_synth: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Perform model-based 3D transformation of `source` using its depth map,
    then reproject into the target pose.  Optionally handle 2D keypoints
    the same way Homography does.

    Args:
        source: source Sample (must have .image, .depth_map, .pose, ._K, .mask, optionally .keypoints2D)
        target: target Sample (must have .pose, ._K, .mask)
        interpolation: If True, fill holes by neighbor averaging.
        maskOutput: If True, return only the reprojected foreground (no background composite).
        with_axes: If True, draw 3D axes of the *source* in the final image (using source.draw_axes).
        with_kps: If True, draw transformed 2D keypoints onto the final image.
        return_transformed_kps: If True, return an array of transformed 2D keypoints (with visibility).
        return_transformed_mask: If True, return the final reprojection mask.
        mask_input: If True, use source.masked_image as the “input RGB” before 3D reprojection.
        target_synth: If True, the target will be taken as just a pose because here we consider that 
            the target is actually not in the dataset and will be synthesized. 

    Returns (in order):
        - The final uint8 RGB reprojection (HxWx3 or HxW×1)
        - (optional) transformed 2D keypoints, shape (N, 3) if keypoints2D was provided
        - (optional) the final boolean mask (HxW) showing which pixels were reprojected
    """
    # —————— 1) Prep and size checks ——————
    source_depth_map = source.depth_map
    if mask_input:
        # We use the depth map directly for speed
        _mask = (source_depth_map > 0).astype(np.uint8)   
        img = source.image
        # Ensure the mask has the same size as the image
        if _mask.shape != img.shape[:2]:
            _mask = cv2.resize(_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Ensure the mask is single-channel 8-bit
        if _mask.dtype != np.uint8:
            _mask = _mask.astype(np.uint8)
        
        # Apply mask: only keep pixels where mask == 1
        source_image = cv2.bitwise_and(img, img, mask=_mask)
    else:
        source_image = source.normalized_image # FIXME CHANGE BACK

    H, W = source_depth_map.shape
    if source_image.ndim == 3:
        C = source_image.shape[2]
    elif source_image.ndim == 2:
        C = 1
        source_image = source_image[..., None]
    else:
        raise ValueError(f"source.image must be HxW or HxWx3, got {source_image.shape}")

    # If the target is to be synthesized we just take the source's matrix naively
    if not target_synth:
        assert np.all(source._K == target._K), "Camera matrices must match."
    K = source._K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Extract pose R and t
    R1 = source.pose[:3, :3]
    t1 = source.pose[:3, 3]

    # If the target is to be synthesized then the "target" argument is just a pose
    if target_synth:
        if (not type(target) == np.ndarray and not type(target) == list) or \
            target.shape != (4,4):
            raise ValueError("When the target is to be synthesized (target_synth=True)" \
            f" then please provide its 4x4 pose in place of the target argument. You provided: {target}.")
        R2 = target[:3, :3]
        t2 = target[:3, 3]
        target_pose = target
    else:
        target_pose = target.pose
        R2 = target.pose[:3, :3]
        t2 = target.pose[:3, 3]

    def rigid_transform(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        return pts @ R.T + t[None, :]

    # —————— 2) Build dense 3D point cloud ——————
    xv, yv = np.meshgrid(np.arange(W), np.arange(H))
    u_all = xv.ravel().astype(np.float64)  # (N,), N=H*W
    v_all = yv.ravel().astype(np.float64)

    z_all = source_depth_map.ravel().astype(np.float64)
    z_all[z_all == 0] = np.inf  # treat zero as invalid

    x_cam = (u_all - cx) * z_all / fx
    y_cam = (v_all - cy) * z_all / fy
    xyz = np.stack([x_cam, y_cam, z_all], axis=1)  # (N, 3)
    valid = np.isfinite(xyz).all(axis=1)           # (N,) boolean

    # —————— 3) Extract background from source ——————
    imBG = source_image.copy()
    imBG_flat = imBG.reshape(-1, C)
    imBG_flat[valid, :] = 0
    imBG = imBG_flat.reshape(H, W, C)

    # —————— 4) Extract dense intensities for valid points ——————
    intensities = source_image.reshape(-1, C).astype(np.float64)
    src_vals = intensities[valid, :]    # (M, C), M = number of valid pixels
    xyz_valid = xyz[valid, :]           # (M, 3)

    # —————— 5) Apply rigid‐transform to dense points ——————
    xyz_obj = rigid_transform(xyz_valid, R1.T, -R1.T @ t1)   # undo source pose
    xyz_cam = rigid_transform(xyz_obj, R2, t2)               # bring into target cam frame

    # —————— 6) Project dense points to 2D ——————
    uvh = (K @ xyz_cam.T)               # (3, M)
    uv = uvh[:2] / uvh[2:3]             # (2, M)
    uv_int = np.round(uv).astype(int).T # (M, 2)

    # —————— 7) Rasterize dense points with z‐buffer ——————
    fill_value = -1.0
    imt = np.full((H, W, C), fill_value, dtype=np.float64)
    zbuf = np.full((H, W), -np.inf, dtype=np.float64)
    mask_t = np.zeros((H, W), dtype=bool)

    px = uv_int[:, 0]      # (M,)
    py = uv_int[:, 1]      # (M,)
    depths = xyz_cam[:, 2] # (M,)
    vals = src_vals        # (M, C)

    in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px = px[in_bounds]
    py = py[in_bounds]
    depths = depths[in_bounds]
    vals = vals[in_bounds]
    flat_idx = py * W + px

    order = np.lexsort((-depths, flat_idx))
    flat_sorted = flat_idx[order]
    uniq_flat, first_idxs = np.unique(flat_sorted, return_index=True)
    best_point_idxs = order[first_idxs]

    best_px = px[best_point_idxs]
    best_py = py[best_point_idxs]
    best_depths = depths[best_point_idxs]
    best_vals = vals[best_point_idxs]  # (K, C)

    imt[best_py, best_px, :] = best_vals
    zbuf[best_py, best_px] = best_depths
    mask_t[best_py, best_px] = True

    # —————— 8) Hole‐filling / interpolation ——————
    holes = (imt[:, :, 0] == fill_value)
    kernel = np.ones((5, 5), np.float64)
    if interpolate and holes.any():
        for c in range(C):
            channel = imt[:, :, c]
            valid_mask = (~holes) & (channel > fill_value)
            # Getting color values
            s = cv2.filter2D(channel * valid_mask, -1, kernel)
            # Counting number of valid pixels to divide color values
            cnt = cv2.filter2D(valid_mask.astype(np.float64), -1, kernel)
            cnt[cnt == 0] = 1
            interp = np.round(s / cnt)
            channel[holes] = interp[holes]
            imt[:, :, c] = channel
        mask_t[holes] = interp[holes]
    else:
        imt[holes] = 0

    # —————— 9) Composite or maskOutput ——————
    if not maskOutput:
        composite_BG = imBG.copy()
        target_mask = target.mask
        np.logical_or(mask_t, target_mask, out=mask_t)
        composite_BG[mask_t] = 0
        out = composite_BG + imt
    else:
        out = imt

    return_image = np.clip(out, 0, 255).astype(np.uint8)

    # —————— 10) Keypoint reprojection (integrated) ——————
    transformed_kps = None
    if return_transformed_kps or with_kps:
        if source.keypoints2D is None:
            raise ValueError(
                "`return_transformed_kps=True` or `with_kps=True` requires `source.keypoints2D`."
            )

        kp_array = source.keypoints2D.astype(np.float64)   # (N, 2) or (N, 3)
        pts_uv = kp_array[:, :2]                           # (N, 2)
        N = pts_uv.shape[0]

        # Initial visibility
        if kp_array.shape[1] >= 3:
            init_vis = (kp_array[:, 2] > 0).astype(np.uint8)  # 1 = visible, 0 = invisible
        else:
            init_vis = np.ones(N, dtype=np.uint8)

        # Prepare final array
        final_kps = np.zeros((N, 3), dtype=np.float32)

        # 10a) Copy invisible keypoints → keep original coords, vis=0
        invis_idxs = np.where(init_vis == 0)[0]
        if invis_idxs.size > 0:
            final_kps[invis_idxs, 0:2] = pts_uv[invis_idxs]
            final_kps[invis_idxs, 2] = 0

        # 10b) Handle visible keypoints via the same 3D pipeline
        vis_idxs = np.where(init_vis == 1)[0]
        if vis_idxs.size > 0:
            # Gather their (u, v)
            pts_uv_vis = pts_uv[vis_idxs, :]                # (M, 2)
            ku = pts_uv_vis[:, 0].round().astype(int)       # (M,)
            kv = pts_uv_vis[:, 1].round().astype(int)
            ku = np.clip(ku, 0, W - 1)
            kv = np.clip(kv, 0, H - 1)

            # Look up depth
            z_vis = source_depth_map[kv, ku].astype(np.float64)

            # Precompute non-zero-depth positions once
            nonzero_vs, nonzero_us = np.where(source_depth_map > 0)
            nonzero_coords = np.column_stack((nonzero_us, nonzero_vs))           # shape (P, 2)
            nonzero_depths = source_depth_map[nonzero_vs, nonzero_us].astype(np.float64)  # (P,)

            # For any keypoint where depth == 0, replace with nearest non-zero depth
            zero_depth_idxs = np.where(z_vis == 0)[0]
            for rel_idx in zero_depth_idxs:
                u0 = ku[rel_idx]
                v0 = kv[rel_idx]
                # Compute squared distances to all non-zero pixels
                du = nonzero_coords[:, 0] - u0
                dv = nonzero_coords[:, 1] - v0
                dist2 = du * du + dv * dv
                nearest = np.argmin(dist2)
                z_vis[rel_idx] = nonzero_depths[nearest]

            # Backproject to 3D
            x_k = (pts_uv_vis[:, 0] - cx) * z_vis / fx
            y_k = (pts_uv_vis[:, 1] - cy) * z_vis / fy
            xyz_k = np.stack([x_k, y_k, z_vis], axis=1)    # (M, 3)

            # Rigid‐transform to target cam frame
            xyzk_obj = rigid_transform(xyz_k, R1.T, -R1.T @ t1)  # (M, 3)
            xyzk_cam = rigid_transform(xyzk_obj, R2, t2)         # (M, 3)

            # Project back to 2D
            uvh_k = (K @ xyzk_cam.T)              # (3, M)
            uv_k = uvh_k[:2] / uvh_k[2:3]         # (2, M)
            uv_k = uv_k.T                         # (M, 2)

            # Round to integer pixels
            uvk_int = np.round(uv_k).astype(int)  # (M, 2)

            # Compute projected visibility
            proj_kps = cv_utils.keypoints.add_visibility_to_keypoints(
                uvk_int.astype(np.float32),
                (W, H)
            )  # (M, 3): (u_proj, v_proj, proj_vis)

            # Fill final_kps for visible indices
            final_kps[vis_idxs, :] = proj_kps.astype(np.float32)

        transformed_kps = final_kps  # shape (N, 3)

        if with_kps:
            return_image = cv_utils.keypoints.draw_kps_on_img(return_image, transformed_kps)
    if with_axes:
        return_image = source.draw_axes(return_image,pose=target_pose)

    # —————— 11) Return appropriately ——————
    mask_uint8 = mask_t.astype(np.uint8)*255
    if return_transformed_kps and return_transformed_mask:
        return return_image, transformed_kps, mask_uint8
    elif return_transformed_kps:
        return return_image, transformed_kps
    elif return_transformed_mask:
        return return_image, mask_uint8
    else:
        return return_image


def ViewMorph(input_image: np.ndarray,
                              input_mask: np.ndarray,
                              correspondence_map: np.ndarray,
                              alpha: float) -> np.ndarray:
    """
    Morphs the input image according to a horizontal correspondence map with interpolation,
    then fills holes via neighborhood averaging.

    Args:
        input_image: HxW or HxWxC numpy array (grayscale or RGB/BGR).
        input_mask:  HxW binary mask (1 for foreground, 0 elsewhere).
        correspondence_map: HxW array giving the x-coordinate correspondence (0 = no correspondence).
        alpha: float in [0, 1], interpolation factor.

    Returns:
        morphed_image: HxW or HxWxC numpy array of the morphed image.
    """
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1.")

    # Ensure image has channel dimension
    if input_image.ndim == 2:
        input_image = input_image[:, :, None]
    rows, cols, channels = input_image.shape

    # Initialize output and mask of filled pixels
    morphed_image = np.zeros_like(input_image)
    filled_mask = np.zeros((rows, cols), dtype=bool)

    # 1) Morphing step
    for y in range(rows):
        for x in range(cols):
            corr_x = correspondence_map[y, x]
            if corr_x <= 0:
                continue
            interp_x = int(round((1 - alpha) * x + alpha * corr_x))
            interp_y = y
            if 0 <= interp_x < cols and 0 <= interp_y < rows:
                morphed_image[interp_y, interp_x, :] = input_image[y, x, :]
                filled_mask[interp_y, interp_x] = True

    # 2) Morphological closing to identify holes
    close_kernel = np.ones((15, 15), dtype=np.uint8)
    closed_mask = cv2.morphologyEx(filled_mask.astype(np.uint8),
                                   cv2.MORPH_CLOSE,
                                   close_kernel).astype(bool)

    hole_mask = closed_mask & ~filled_mask

    # 3) Fill holes by neighborhood averaging
    interp_kernel = np.ones((5, 5), dtype=np.float32)
    for c in range(channels):
        channel = morphed_image[:, :, c].astype(np.float32)
        valid = channel > 0

        # Sum of valid neighbors
        neighbor_sum = cv2.filter2D(channel * valid, -1, interp_kernel,
                                    borderType=cv2.BORDER_CONSTANT)
        # Count of valid neighbors
        count = cv2.filter2D(valid.astype(np.float32), -1, interp_kernel,
                             borderType=cv2.BORDER_CONSTANT)
        count[count == 0] = 1.0  # Avoid division by zero

        interpolated = np.round(neighbor_sum / count)
        channel[hole_mask] = interpolated[hole_mask]
        morphed_image[:, :, c] = channel

    # Remove singleton channel dimension if original was 2D
    if morphed_image.shape[2] == 1:
        morphed_image = morphed_image[:, :, 0]

    return morphed_image
# Example usage:
# img = cv2.imread('input.png')
# mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE) > 0
# corr_map = np.load('corr_map.npy')
# alpha_val = 0.5
# morphed = transformation_view_morph(img, mask, corr_map, alpha_val)
# cv2.imwrite('morphed.png', morphed)

def transform_image_point(point, H):
    """
    Applies homography H to a point (x, y).
    """
    x, y = point
    vec = np.array([x, y, 1.0], dtype=np.float64)
    mapped = H @ vec
    return mapped[:2] / mapped[2]


def compute_ncc(patch1, patch2):
    """
    Compute normalized cross-correlation between two patches.
    """
    # Zero-mean
    p1 = patch1.astype(np.float64) - np.mean(patch1)
    p2 = patch2.astype(np.float64) - np.mean(patch2)
    num = np.sum(p1 * p2)
    den = np.sqrt(np.sum(p1**2) * np.sum(p2**2))
    return num / den if den > 0 else -np.inf


def get_patch(img, y, x, size):
    """
    Extract square patch of side length size centered at (y, x).
    """
    half = size // 2
    return img[y-half:y+half+1, x-half:x+half+1]


def stereo_correspondence_search(
    source: "Sample",
    target: "Sample",
    mask1: np.ndarray,
    mask2: np.ndarray,
    depth_estimate,
    depth_range,
    NCC_neighborhood
):
    """
    Finds dense horizontal correspondences between rectified stereo images.
    Returns:
      - correspondence_map: HxW array of matched x-coordinates (-1 if no match)
      - stereo_map: HxW disparity-to-depth map (0 if invalid)
    """
    # Get images and camera intrinsics
    im1 = source.image
    im2 = target.image
    K = source._K
    assert np.all(source._K == target._K), "Camera matrices must be equal for stereo."
    # Get poses
    R1 = source.pose[:3, :3]
    t1 = source.pose[:3, 3]
    R2 = target.pose[:3, :3]
    t2 = target.pose[:3, 3]
    # Compute rectification transforms
    # Use OpenCV's stereoRectify
    # Assume zero distortion
    distCoeffs = np.zeros(5)
    image_size = (im1.shape[1], im1.shape[0])
    R = R2 @ R1.T
    T = t2 - (R2 @ R1.T @ t1)
    # OpenCV expects R, T from left to right
    R1_rect, R2_rect, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K, distCoeffs, K, distCoeffs, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY
    )
    # Compute rectification maps
    map1_1, map1_2 = cv2.initUndistortRectifyMap(
        K, distCoeffs, R1_rect, P1, image_size, cv2.CV_32FC1
    )
    map2_1, map2_2 = cv2.initUndistortRectifyMap(
        K, distCoeffs, R2_rect, P2, image_size, cv2.CV_32FC1
    )
    # Rectify images and masks
    im1rect = cv2.remap(im1, map1_1, map1_2, cv2.INTER_LINEAR)
    im2rect = cv2.remap(im2, map2_1, map2_2, cv2.INTER_LINEAR)
    mask1rect = cv2.remap(mask1.astype(np.uint8), map1_1, map1_2, cv2.INTER_NEAREST).astype(bool)
    mask2rect = cv2.remap(mask2.astype(np.uint8), map2_1, map2_2, cv2.INTER_NEAREST).astype(bool)
    # Compute rectification homographies (for mapping points)
    Hrect = P2[:3, :3] @ np.linalg.inv(K)
    HrectP = P1[:3, :3] @ np.linalg.inv(K)
    # Ensure grayscale
    if im1rect.ndim == 3:
        im1g = cv2.cvtColor(im1rect, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        im1g = im1rect.astype(np.float64)
    if im2rect.ndim == 3:
        im2g = cv2.cvtColor(im2rect, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        im2g = im2rect.astype(np.float64)
    rows, cols = im1g.shape
    corr_map = -np.ones((rows, cols), dtype=int)
    stereo_map = np.zeros((rows, cols), dtype=np.float64)
    N = 200
    radius = NCC_neighborhood // 2
    for y in range(radius, rows - radius):
        for x1 in range(radius, cols - radius):
            if not mask1rect[y, x1]:
                continue
            # Compute search range via homography projection
            src_pt = transform_image_point((x1, y), np.linalg.inv(Hrect))
            epi_dir = transform_image_point((x1 - N//2, y), np.linalg.inv(Hrect)) - src_pt
            norm = np.linalg.norm(epi_dir)
            if norm == 0:
                continue
            epi_dir /= norm
            src_left = src_pt - epi_dir * (N/2)
            src_right = src_pt + epi_dir * (N/2)
            pr = transform_image_point(src_right, Hrect)
            pl = transform_image_point(src_left, Hrect)
            search_width = np.linalg.norm(pr - pl)
            if search_width <= 0:
                continue
            x_min = int(max(radius, x1 - search_width/2))
            x_max = int(min(cols - radius - 1, x1 + search_width/2))
            best_ncc = -np.inf
            best_x2 = -1
            patch1 = get_patch(im1g, y, x1, NCC_neighborhood)
            for x2 in range(x_min, x_max + 1):
                if not mask2rect[y, x2]:
                    continue
                patch2 = get_patch(im2g, y, x2, NCC_neighborhood)
                ncc_val = compute_ncc(patch1, patch2)
                if ncc_val > best_ncc:
                    best_ncc = ncc_val
                    best_x2 = x2
            if best_x2 >= 0:
                corr_map[y, x1] = best_x2
                disparity = abs(x1 - best_x2)
                if disparity != 0:
                    f = K[0, 0]
                    B = np.linalg.norm(t2 - t1)
                    depth = (f * B) / disparity
                    stereo_map[y, x1] = depth
    return corr_map, stereo_map

# Example usage:
# corr_map, depth_map = stereo_correspondence_search(
#     im1rect, im2rect, R2, R1, t2, t1, K,
#     mask1rect, mask2rect, Hrect, HrectP,
#     depth_estimate=0, depth_range=(0,10), NCC_neighborhood=25
# )
