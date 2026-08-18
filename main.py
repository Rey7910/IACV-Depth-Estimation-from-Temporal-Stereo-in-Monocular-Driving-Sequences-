import os
import cv2
import numpy as np
from src.dataset import KittiTemporalDataset
from src.metrics import compute_depth_metrics
from src.temporal_stereo import (
    compute_bidirectional_optical_flow,
    estimate_depth_with_rotation_compensation,
    filter_occlusions_fb,
    filter_road_plane_obstacles,
)
from src.visualization import (
    save_depth_comparison,
    save_feature_correspondences,
)


def main():
    KITTI_DATA_ROOT = "./data"
    DRIVE_NAME = "2011_09_26_drive_0001_extract"

    print("=" * 70)
    print(
        "--- IACV PIPELINE: FULL EVALUATION + CORRESPONDENCES (MILELLA ET AL.) ---"
    )
    print("=" * 70)

    dataset = KittiTemporalDataset(
        data_root=KITTI_DATA_ROOT, drive_name=DRIVE_NAME
    )

    num_samples = len(dataset)
    if num_samples == 0:
        print("\n[Warning] No images found. Please check the file path.")
        return

    print(f"Total temporal pairs available in dataset: {num_samples}\n")

    # Accumulate global metrics
    all_maes = []
    all_rmses = []
    all_abs_rels = []
    total_valid_pixels = 0

    # Variables to track the best frame based on LiDAR metrics (lowest MAE)
    best_mae = float("inf")
    best_frame_data = None

    # Iterate over all sequence frames (excluding the last one without t+1)
    for idx in range(num_samples - 1):
        sample = dataset[idx]

        img_t = sample["img_t"]
        img_t_plus_1 = sample["img_t_plus_1"]
        vf = sample["vf"]
        angular_vels = sample["angular_vels"]
        dt = sample["dt"]
        K = sample["K"]
        depth_gt = sample["depth_gt"]

        print(
            f"Processing frame {idx}/{num_samples - 1} (vf = {vf:.2f} m/s)..."
        )

        # 1. Bidirectional optical flow
        flow_fwd, flow_bwd = compute_bidirectional_optical_flow(
            img_t, img_t_plus_1
        )

        # 2. Occlusion filtering
        valid_mask = filter_occlusions_fb(flow_fwd, flow_bwd, threshold=1.5)

        # 3. Depth estimation with rotation compensation
        pred_depth = estimate_depth_with_rotation_compensation(
            flow_fwd, valid_mask, vf, angular_vels, dt, K
        )

        # 4. Road plane filtering (Wedel et al.)
        pred_depth = filter_road_plane_obstacles(
            pred_depth, K, camera_height=1.65
        )

        # 5. Compute metrics for current frame (Only if Ground Truth exists)
        if depth_gt is not None:
            metrics = compute_depth_metrics(pred_depth, depth_gt)

            if metrics["Valid_Pixels"] > 0:
                current_mae = metrics["MAE"]
                all_maes.append(current_mae)
                all_rmses.append(metrics["RMSE"])
                all_abs_rels.append(metrics["Abs_Rel"])
                total_valid_pixels += metrics["Valid_Pixels"]

                # Track the best frame based on the lowest MAE relative to LiDAR
                if current_mae < best_mae:
                    best_mae = current_mae
                    best_frame_data = {
                        "idx": idx,
                        "img_t": img_t,
                        "img_t_plus_1": img_t_plus_1,
                        "flow_fwd": flow_fwd,
                        "valid_mask": valid_mask,
                        "pred_depth": pred_depth,
                        "depth_gt": depth_gt,
                    }
        else:
            print(f"  [Notice] Frame {idx} has no Ground Truth depth available. Skipping metrics.")

    # Save visualizations for the frame that achieved the best metric results against LiDAR
    if best_frame_data is not None:
        print(
            f"\n[Visualization] Best frame found at index {best_frame_data['idx']} "
            f"with an MAE of {best_mae:.2f} meters. Saving output plots..."
        )
        save_depth_comparison(
            best_frame_data["img_t"],
            best_frame_data["pred_depth"],
            best_frame_data["depth_gt"],
            output_path="output_depth_comparison_final.png",
        )
        save_feature_correspondences(
            best_frame_data["img_t"],
            best_frame_data["img_t_plus_1"],
            best_frame_data["flow_fwd"],
            best_frame_data["valid_mask"],
            output_path="output_correspondences.png",
            step=25,
        )
    else:
        print("\n[Warning] No valid frames with ground truth found to save comparisons.")

    # Sequence global summary
    print("\n" + "=" * 50)
    print("--- GLOBAL RESULTS (FULL SEQUENCE) ---")
    print(f"  * Total frames evaluated  : {num_samples}")
    print(f"  * Accumulated valid pixels: {total_valid_pixels}")
    if len(all_maes) > 0:
        print(f"  * Overall Average MAE     : {np.mean(all_maes):.2f} meters")
        print(f"  * Overall Average RMSE    : {np.mean(all_rmses):.2f} meters")
        print(f"  * Overall Average Abs Rel : {np.mean(all_abs_rels):.4f}")
    else:
        print("  * No valid ground truth frames found to compute global averages.")
    print("=" * 50)
    print("[Success] Full sequence evaluations completed and plots generated.")


if __name__ == "__main__":
    main()