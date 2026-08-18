import os
import cv2
import numpy as np
from src.dataset import KittiTemporalDataset
from src.temporal_stereo import (
    compute_bidirectional_optical_flow,
    filter_occlusions_fb,
    estimate_depth_with_rotation_compensation,
)
from src.visualization import save_depth_comparison


def main():
    KITTI_DATA_ROOT = "./data"
    DRIVE_NAME = "2011_09_26_drive_0001_extract"

    print("=" * 60)
    print("--- ADVANCED TEST: OCCLUSIONS AND ROTATIONAL COMPENSATION ---")
    print("=" * 60)

    dataset = KittiTemporalDataset(
        data_root=KITTI_DATA_ROOT, drive_name=DRIVE_NAME
    )

    if len(dataset) == 0:
        print("\n[Warning] No images found. Please check the file path.")
        return

    sample = dataset[0]

    img_t = sample["img_t"]
    img_t_plus_1 = sample["img_t_plus_1"]
    vf = sample["vf"]
    angular_vels = sample["angular_vels"]
    dt = sample["dt"]
    K = sample["K"]
    depth_gt = sample["depth_gt"]

    print(
        "Computing bidirectional optical flow (Forward-Backward) for frame"
        f" {sample['frame_idx']}..."
    )
    flow_fwd, flow_bwd = compute_bidirectional_optical_flow(
        img_t, img_t_plus_1
    )

    print("Filtering occlusions and textureless regions (FB Consistency)...")
    valid_mask = filter_occlusions_fb(flow_fwd, flow_bwd, threshold=1.5)

    print(
        "Estimating depth with rotation compensation (Yaw/Pitch/Roll) and vf ="
        f" {vf:.2f} m/s..."
    )
    pred_depth = estimate_depth_with_rotation_compensation(
        flow_fwd, valid_mask, vf, angular_vels, dt, K
    )

    valid_preds = pred_depth[pred_depth > 0]
    print(
        "- Pixels with valid predicted depth (after filtering):"
        f" {len(valid_preds)}"
    )
    if len(valid_preds) > 0:
        print(f"  * Minimum predicted depth: {valid_preds.min():.2f} m")
        print(f"  * Maximum predicted depth: {valid_preds.max():.2f} m")

    # Evaluation against LiDAR Ground Truth
    if depth_gt is not None:
        mask = (depth_gt > 0) & (pred_depth > 0)
        if np.sum(mask) > 0:
            mae = np.mean(np.abs(depth_gt[mask] - pred_depth[mask]))
            print(
                "- Mean Absolute Error (MAE) vs LiDAR (Improved):"
                f" {mae:.2f} meters"
            )

    # Save optimized comparison image
    output_img_path = "output_depth_comparison_advanced.png"
    save_depth_comparison(
        img_t, pred_depth, depth_gt, output_path=output_img_path
    )

    print(
        "\nPipeline tested and visualization successfully generated!"
    )

if __name__ == "__main__":
    main()