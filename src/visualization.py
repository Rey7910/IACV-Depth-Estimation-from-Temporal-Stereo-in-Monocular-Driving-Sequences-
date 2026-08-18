import cv2
import numpy as np
import os

def save_depth_comparison(img_t, pred_depth, depth_gt, output_path="output_comparison.png"):
    """
    Generates and saves a side-by-side or stacked comparative image containing:
    1. Original image (t)
    2. Predicted depth map (Temporal Stereo) with JET colormap
    3. Ground Truth depth map (LiDAR) with JET colormap
    """
    H, W = img_t.shape[:2]

    # --- 1. Normalize and color predicted depth ---
    # Filter out zeros so visualization ignores unvalued pixels
    valid_pred = pred_depth[pred_depth > 0]
    if len(valid_pred) > 0:
        min_p, max_p = np.percentile(valid_pred, 5), np.percentile(valid_pred, 95)
    else:
        min_p, max_p = 0.0, 50.0

    # Normalize between 0 and 255 for the heatmap
    pred_norm = np.clip((pred_depth - min_p) / (max_p - min_p + 1e-5), 0, 1)
    pred_colored = cv2.applyColorMap((pred_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Set pixels without predicted depth to black
    pred_colored[pred_depth == 0] = [0, 0, 0]

    # --- 2. Normalize and color LiDAR Ground Truth ---
    if depth_gt is not None:
        valid_gt = depth_gt[depth_gt > 0]
        if len(valid_gt) > 0:
            min_gt, max_gt = np.percentile(valid_gt, 5), np.percentile(valid_gt, 95)
        else:
            min_gt, max_gt = 0.0, 50.0

        gt_norm = np.clip((depth_gt - min_gt) / (max_gt - min_gt + 1e-5), 0, 1)
        gt_colored = cv2.applyColorMap((gt_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        gt_colored[depth_gt == 0] = [0, 0, 0]
    else:
        # If there is no LiDAR in this frame, create a gray image with warning text
        gt_colored = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.putText(gt_colored, "No LiDAR GT", (W // 3, H // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # --- 3. Add title / text labels to each section ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_t, "1. Original Image (t)", (30, 40), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(pred_colored, "2. Predicted Depth (Temporal Stereo)", (30, 40), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(gt_colored, "3. LiDAR Ground Truth", (30, 40), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # --- 4. Vertically concatenate the three images ---
    comparison = np.vstack((img_t, pred_colored, gt_colored))

    # Save the resulting image
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, comparison)
    print(f"[Visualization] Comparative image successfully saved at: {output_path}")