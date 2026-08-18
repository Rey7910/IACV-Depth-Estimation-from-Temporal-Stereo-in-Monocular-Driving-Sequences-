import numpy as np


def compute_depth_metrics(pred_depth, gt_depth):
    """Calculates standard depth evaluation metrics by comparing predictions with

    Ground Truth (LiDAR).
    """
    # Filter only pixels where both maps have valid values (> 0)
    mask = (gt_depth > 0) & (pred_depth > 0)

    if np.sum(mask) == 0:
        return {
            "MAE": 0.0,
            "RMSE": 0.0,
            "Abs_Rel": 0.0,
            "Delta1": 0.0,
            "Delta2": 0.0,
            "Delta3": 0.0,
            "Valid_Pixels": 0,
        }

    pred = pred_depth[mask]
    gt = gt_depth[mask]

    # 1. Mean Absolute Error (MAE)
    mae = np.mean(np.abs(pred - gt))

    # 2. Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((pred - gt) ** 2))

    # 3. Absolute Relative Error (Abs Rel)
    abs_rel = np.mean(np.abs(pred - gt) / gt)

    # 4. Threshold Accuracy (Delta < 1.25, 1.25^2, 1.25^3)
    r = np.maximum(pred / gt, gt / pred)
    delta1 = np.mean(r < 1.25) * 100
    delta2 = np.mean(r < 1.25**2) * 100
    delta3 = np.mean(r < 1.25**3) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Abs_Rel": float(abs_rel),
        "Delta1": float(delta1),
        "Delta2": float(delta2),
        "Delta3": float(delta3),
        "Valid_Pixels": int(np.sum(mask)),
    }