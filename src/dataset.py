import os
import cv2
import numpy as np
from src.kitti_utils import (
    load_oxts_data,
    load_velodyne_points,
    project_velo_to_image
)

class KittiTemporalDataset:
    """
    Loader for KITTI sequences in Raw Extract format (e.g., 2011_09_26_drive_0001_extract),
    compatible with image_02, oxts, velodyne_points, and the calib folder.
    """
    def __init__(self, data_root, drive_name="2011_09_26_drive_0001_extract"):
        self.data_root = data_root
        self.drive_name = drive_name
        
        # Paths based on your current structure (data/raw/2011_09_26_drive_0001_extract)
        self.drive_dir = os.path.join(data_root, "raw", drive_name)
        
        self.img_dir = os.path.join(self.drive_dir, "image_02", "data")
        if not os.path.exists(self.img_dir):
            self.img_dir = os.path.join(self.drive_dir, "image_02")

        self.velo_dir = os.path.join(self.drive_dir, "velodyne_points", "data")
        if not os.path.exists(self.velo_dir):
            self.velo_dir = os.path.join(self.drive_dir, "velodyne_points")

        self.oxts_dir = os.path.join(self.drive_dir, "oxts", "data")
        if not os.path.exists(self.oxts_dir):
            self.oxts_dir = os.path.join(self.drive_dir, "oxts")

        self.calib_dir = os.path.join(self.drive_dir, "calib")

        # List images
        if os.path.exists(self.img_dir):
            self.img_files = sorted([os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir) if f.endswith('.png') or f.endswith('.jpg')])
        else:
            self.img_files = []
            print(f"[Warning] Image directory not found at: {self.img_dir}")

        # Load camera 2 calibration
        self.P_rect, self.Tr_velo_to_cam = self._load_raw_calibration()

    def _load_raw_calibration(self):
        """Loads specific calibration for the KITTI Raw format."""
        P_rect = np.array([[718.856, 0.0, 607.1928, 0.0],
                           [0.0, 718.856, 185.2157, 0.0],
                           [0.0, 0.0, 1.0, 0.0]]).reshape(3, 4)
        Tr_velo_to_cam = np.eye(4)

        try:
            calib_cam_path = os.path.join(self.calib_dir, "calib_cam_to_cam.txt")
            calib_velo_path = os.path.join(self.calib_dir, "calib_velo_to_cam.txt")

            if os.path.exists(calib_cam_path):
                with open(calib_cam_path, 'r') as f:
                    for line in f:
                        if line.startswith("P_rect_02:"):
                            vals = [float(x) for x in line.strip().split()[1:]]
                            P_rect = np.array(vals).reshape(3, 4)
                            break

            if os.path.exists(calib_velo_path):
                with open(calib_velo_path, 'r') as f:
                    rot, trans = None, None
                    for line in f:
                        if line.startswith("R:"):
                            rot = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 3)
                        elif line.startswith("T:"):
                            trans = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 1)
                    if rot is not None and trans is not None:
                        Tr_3x4 = np.hstack((rot, trans))
                        Tr_velo_to_cam = np.vstack((Tr_3x4, np.array([0.0, 0.0, 0.0, 1.0])))
        except Exception as e:
            print(f"[Notice] Using default calibration due to an error reading calibration files: {e}")

        return P_rect, Tr_velo_to_cam

    def __len__(self):
        return max(0, len(self.img_files) - 1)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range in the dataset.")

        img1_path = self.img_files[idx]
        img2_path = self.img_files[idx + 1]
        
        img_t = cv2.imread(img1_path)
        img_t_plus_1 = cv2.imread(img2_path)

        # Longitudinal velocity from OXTS
        vf, angular_vels = 0.0, np.zeros(3, dtype=np.float32)
        if os.path.exists(self.oxts_dir):
            vf, angular_vels = load_oxts_data(self.oxts_dir, idx)

        dt = 0.1 

        # Cargar y proyectar LiDAR (Ground Truth) buscando archivos .txt
        depth_map = None
        velo_file = os.path.join(self.velo_dir, f"{idx:010d}.txt")
        if not os.path.exists(velo_file):
            velo_file_bin = os.path.join(self.velo_dir, f"{idx:010d}.bin")
            if os.path.exists(velo_file_bin):
                velo_file = velo_file_bin

        if os.path.exists(velo_file):
            if velo_file.endswith('.bin'):
                points = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)[:, :3]
            else:
                points = load_velodyne_points(velo_file)
            
            # AQUÍ ES DONDE QUEREMOS VER EL ERROR EXACTO SI LLEGA A FALLAR
            pts_2d, depths = project_velo_to_image(points, self.P_rect, self.Tr_velo_to_cam)
            
            H, W = img_t.shape[:2]
            depth_map = np.zeros((H, W), dtype=np.float32)
            for pt, d in zip(pts_2d, depths):
                u, v = int(pt[0]), int(pt[1])
                if 0 <= u < W and 0 <= v < H and d > 0:
                    depth_map[v, u] = d

        return {
            "img_t": img_t,
            "img_t_plus_1": img_t_plus_1,
            "vf": vf,
            "angular_vels": angular_vels,
            "dt": dt,
            "K": self.P_rect[:, :3],
            "depth_gt": depth_map,
            "frame_idx": idx
        }