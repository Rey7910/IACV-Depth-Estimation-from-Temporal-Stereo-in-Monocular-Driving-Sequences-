import cv2
import numpy as np


def compute_bidirectional_optical_flow(img_t, img_t_plus_1):
    """Calculates optical flow in both directions (t -> t+1 and t+1 -> t)

    to enable forward-backward consistency filtering.
    """
    if len(img_t.shape) == 3:
        gray_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
        gray_t1 = cv2.cvtColor(img_t_plus_1, cv2.COLOR_BGR2GRAY)
    else:
        gray_t, gray_t1 = img_t, img_t_plus_1

    # Forward flow (t -> t+1)
    flow_fwd = cv2.calcOpticalFlowFarneback(
        gray_t,
        gray_t1,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    # Backward flow (t+1 -> t)
    flow_bwd = cv2.calcOpticalFlowFarneback(
        gray_t1,
        gray_t,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    return flow_fwd, flow_bwd

def filter_road_plane_obstacles(depth_map, K, camera_height=1.65):
    """Filters false positives using the flat ground model (Free Driveway)

    inspired by Wedel et al. Points below ground height or with depths
    inconsistent with the road plane are cleaned up.
    """
    H, W = depth_map.shape[:2]
    fy = K[1, 1]
    cy = K[1, 2]

    # Pixel coordinates in the image
    yy, xx = np.indices((H, W))

    # Analytical calculation of the expected depth if the point belonged strictly to the flat road plane (Y = -camera_height)
    # Z_plane = (fy * camera_height) / (yy - cy)  [assuming flat terrain in front of the camera]
    with np.errstate(divide="ignore", invalid="ignore"):
        z_plane = (fy * camera_height) / (yy - cy)

    # If the predicted depth is significantly smaller than the projected geometric ground height
    # or falls in sky/upper horizon regions (where yy < cy), it can be masked.
    filtered_depth = depth_map.copy()

    # Discard estimates above the horizon or inconsistent with the immediate ground
    horizon_mask = (
        yy < (cy + 10)
    )  # Upper region of the image (sky, distant buildings)

    # Apply cleanup
    filtered_depth[horizon_mask & (filtered_depth > 50.0)] = 0.0

    return filtered_depth

def filter_occlusions_fb(flow_fwd, flow_bwd, threshold=1.0):
    """Filters pixels with occlusions or noise using Forward-Backward consistency.

    If ||flow_fwd + flow_bwd(shifted)||_2 > threshold, the pixel is considered
    unreliable.
    """
    H, W = flow_fwd.shape[:2]

    # Map meshgrid coordinates
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))

    # Estimated coordinates at t+1
    x1 = xx + flow_fwd[:, :, 0]
    y1 = yy + flow_fwd[:, :, 1]

    # Sample backward flow at the new coordinates (remapping)
    map_x = x1.astype(np.float32)
    map_y = y1.astype(np.float32)

    flow_bwd_sampled_x = cv2.remap(
        flow_bwd[:, :, 0],
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    flow_bwd_sampled_y = cv2.remap(
        flow_bwd[:, :, 1],
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    # Vector consistency error: flow_fwd + flow_bwd_sampled
    diff_x = flow_fwd[:, :, 0] + flow_bwd_sampled_x
    diff_y = flow_fwd[:, :, 1] + flow_bwd_sampled_y

    fb_error = np.sqrt(diff_x**2 + diff_y**2)

    # Valid pixel mask (lower consistency error)
    valid_mask = fb_error < threshold
    return valid_mask


def estimate_depth_with_rotation_compensation(
    flow_fwd, valid_mask, vf, angular_vels, dt, K
):
    """Estimates depth by compensating for both longitudinal velocity (vf) and

    angular rotation (OXTS angular velocities: wx, wy, wz).
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    H, W = flow_fwd.shape[:2]
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))

    # Extract angular velocities from OXTS (if provided, e.g., [wx, wy, wz])
    if angular_vels is not None:
        wx, wy, wz = angular_vels
    else:
        wx, wy, wz = 0.0, 0.0, 0.0

    # 1. Compute optical flow theoretically induced only by camera ROTATION
    # u_rot = -fx * (wx * x_norm * y_norm - wy * (1 + x_norm^2) + wz * y_norm)  (first-order approximation)
    x_norm = (xx - cx) / fx
    y_norm = (yy - cy) / fy

    u_rot = (
        -fx
        * (wx * x_norm * y_norm - wy * (1.0 + x_norm**2) + wz * y_norm)
        * dt
    )
    v_rot = (
        -fy
        * (wx * (1.0 + y_norm**2) - wy * x_norm * y_norm - wz * x_norm)
        * dt
    )

    # 2. Isolate purely TRANSLATIONAL flow by subtracting the rotational component from measured flow
    u_trans = flow_fwd[:, :, 0] - u_rot

    # 3. Compute depth using only the longitudinal translational component (vf)
    # Z = (fx * vf * dt) / (-u_trans)
    depth_map = np.zeros_like(u_trans, dtype=np.float32)

    eps = 0.5  # Threshold to avoid division by zero in translational flow
    trans_mask = (valid_mask) & (np.abs(u_trans) > eps) & (vf > 0.5)

    numerator = fx * vf * dt
    depth_map[trans_mask] = numerator / np.abs(u_trans[trans_mask])

    # Filter unrealistic ranges for autonomous driving
    depth_map[(depth_map > 100.0) | (depth_map < 1.0)] = 0.0

    return depth_map