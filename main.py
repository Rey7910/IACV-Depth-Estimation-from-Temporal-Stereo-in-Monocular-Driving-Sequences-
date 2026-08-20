import os
import sys
import cv2
import numpy as np
from src.dataset import KittiTemporalDataset
from src.metrics import( 
    compute_depth_metrics, 
    align_depth_scale
)
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


class TeeOutput:
    """Clase auxiliar para duplicar selectivamente la salida hacia consola y hacia un archivo .txt."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        self.capture_to_file = False 

    def write(self, message):
       
        self.terminal.write(message)
        
        if self.capture_to_file:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        if self.capture_to_file:
            self.log.flush()

    def set_capture(self, state: bool):
        self.capture_to_file = state

    def close(self):
        if self.log:
            self.log.close()


def main():
    KITTI_DATA_ROOT = "./data"
    PROCESSED_ROOT = os.path.join(KITTI_DATA_ROOT, "processed")
    os.makedirs(PROCESSED_ROOT, exist_ok=True)
    
    drive_names = [
        "2011_09_26_drive_0001_extract",
        "2011_09_26_drive_0005_extract",
        "2011_09_26_drive_0056_extract",
        "2011_09_26_drive_0091_extract"
    ]

    for drive_name in drive_names:
        result_dir_name = drive_name.replace("_extract", "_results")
        current_result_dir = os.path.join(PROCESSED_ROOT, result_dir_name)
        os.makedirs(current_result_dir, exist_ok=True)

        report_path = os.path.join(current_result_dir, "evaluation_report.txt")
        tee = TeeOutput(report_path)
        original_stdout = sys.stdout
        sys.stdout = tee

        try:
            # --- FASE 1: Logs detallados (SOLO en consola, no se graban en el txt) ---
            tee.set_capture(False)
            print("=" * 70)
            print(f"--- IACV PIPELINE: EVALUATION FOR {drive_name} ---")
            print("=" * 70)
            
            dataset = KittiTemporalDataset(
                data_root=KITTI_DATA_ROOT, drive_name=drive_name
            )

            num_samples = len(dataset)
            if num_samples == 0:
                print(f"[Warning] No images found for {drive_name}. Skipping...")
                sys.stdout = original_stdout
                tee.close()
                continue

            print(f"Total temporal pairs available in dataset: {num_samples}\n")

            all_maes = []
            all_rmses = []
            all_abs_rels = []
            total_valid_pixels = 0

            best_mae = float("inf")
            best_frame_data = None

            for idx in range(num_samples - 1):
                sample = dataset[idx]

                img_t = sample["img_t"]
                img_t_plus_1 = sample["img_t_plus_1"]
                vf = sample["vf"]
                angular_vels = sample["angular_vels"]
                dt = sample["dt"]
                K = sample["K"]
                depth_gt = sample["depth_gt"]

                print(f"Processing frame {idx}/{num_samples - 1} (vf = {vf:.2f} m/s)...")

                # 1. Bidirectional optical flow
                flow_fwd, flow_bwd = compute_bidirectional_optical_flow(
                    img_t, img_t_plus_1
                )

                # 2. Occlusion filtering (Forward-Backward)
                occlusion_mask = filter_occlusions_fb(flow_fwd, flow_bwd, threshold=1.5)

                # 3. Optical Flow Confidence Filtering (Filtro adicional de textura y magnitud anómala)
                flow_magnitude = np.linalg.norm(flow_fwd, axis=-1)
                # Descartamos vectores de flujo extremadamente grandes (falsas correspondencias por baja textura)
                confidence_mask = flow_magnitude < 100.0 

                # Máscara de validez combinada definitiva
                valid_mask = occlusion_mask & confidence_mask

                # 4. Depth estimation with rotation compensation
                pred_depth = estimate_depth_with_rotation_compensation(
                    flow_fwd, valid_mask, vf, angular_vels, dt, K
                )

                # 5. Road plane filtering (Wedel et al.)
                pred_depth = filter_road_plane_obstacles(
                    pred_depth, K, camera_height=1.65
                )

                if depth_gt is not None:
                    eval_mask = (valid_mask) & (depth_gt > 0) & (pred_depth > 0)
                    pred_depth, estimated_scale = align_depth_scale(pred_depth, depth_gt, eval_mask)
                    print(f"  [Scale Correction] Applied scale factor: {estimated_scale:.4f}")

                # 6. Compute metrics
                if depth_gt is not None:
                    metrics = compute_depth_metrics(pred_depth, depth_gt, max_depth=35.0)

                    if metrics["Valid_Pixels"] > 0:
                        current_mae = metrics["MAE"]
                        all_maes.append(current_mae)
                        all_rmses.append(metrics["RMSE"])
                        all_abs_rels.append(metrics["Abs_Rel"])
                        total_valid_pixels += metrics["Valid_Pixels"]

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

            
            if best_frame_data is not None:
                print(
                    f"\n[Visualization] Best frame found at index {best_frame_data['idx']} "
                    f"with an MAE of {best_mae:.2f} meters. Saving output plots..."
                )
                
                depth_plot_path = os.path.join(current_result_dir, "depth_comparison.png")
                corr_plot_path = os.path.join(current_result_dir, "correspondences.png")
                
                save_depth_comparison(
                    best_frame_data["img_t"],
                    best_frame_data["pred_depth"],
                    best_frame_data["depth_gt"],
                    output_path=depth_plot_path,
                )
                save_feature_correspondences(
                    best_frame_data["img_t"],
                    best_frame_data["img_t_plus_1"],
                    best_frame_data["flow_fwd"],
                    best_frame_data["valid_mask"],
                    output_path=corr_plot_path,
                    step=25,
                )
            else:
                print(f"\n[Warning] No valid frames with ground truth found for {drive_name}.")

            # --- FASE 2: Resumen global (Activamos la captura para que SÍ se escriba en el archivo .txt) ---
            tee.set_capture(True)
            print("\n" + "=" * 50)
            print(f"--- GLOBAL RESULTS ({drive_name}) ---")
            print(f"  * Total frames evaluated  : {num_samples}")
            print(f"  * Accumulated valid pixels: {total_valid_pixels}")
            if len(all_maes) > 0:
                print(f"  * Overall Average MAE     : {np.mean(all_maes):.2f} meters")
                print(f"  * Overall Average RMSE    : {np.mean(all_rmses):.2f} meters")
                print(f"  * Overall Average Abs Rel : {np.mean(all_abs_rels):.4f}")
            else:
                print("  * No valid ground truth frames found to compute global averages.")
            print("=" * 50)

        finally:
            sys.stdout = original_stdout
            tee.close()
            print(f"[Completed] Results and plots successfully saved inside: {current_result_dir}")

    print("\n[Success] All sequences have been evaluated and neatly organized!")


if __name__ == "__main__":
    main()