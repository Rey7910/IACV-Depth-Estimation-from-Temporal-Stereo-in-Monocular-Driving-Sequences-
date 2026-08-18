import numpy as np
from datetime import datetime
import os

def parse_timestamps(path):
    """
    Reads the KITTI timestamps.txt file and converts each line
    into elapsed seconds since the start of the sequence.
    """
    with open(path, 'r') as f:
        times = [
            datetime.strptime(
                line.strip()[:-3],
                '%Y-%m-%d %H:%M:%S.%f'
            )
            for line in f
        ]
    return [(t - times[0]).total_seconds() for t in times]


def load_oxts_velocity(oxts_dir, frame_idx):
    """
    Reads the OXTS .txt file corresponding to frame_idx and
    extracts the forward velocity (vf) in m/s.
    """
    file_path = os.path.join(oxts_dir, f"{frame_idx:010d}.txt")

    try:
        data = np.loadtxt(file_path)
        # According to the KITTI format:
        # index 8 corresponds to vf (forward velocity)
        vf = data[8]

        return vf

    except Exception as e:
        print(f"Error loading OXTS file {file_path}: {e}")
        return None


def load_oxts_data(oxts_dir, frame_idx):
    """
    Lee el archivo .txt de OXTS completo y extrae tanto la velocidad longitudinal (vf)
    como las velocidades angulares [wx, wy, wz].
    """
    file_path = os.path.join(oxts_dir, f"{frame_idx:010d}.txt")
    try:
        data = np.loadtxt(file_path)
        vf = data[8]          # Velocidad longitudinal (vx)
        # En KITTI oxts data: índices 17, 18, 19 corresponden a las tasas angulares roll_rate, pitch_rate, yaw_rate
        wx = data[17]
        wy = data[18]
        wz = data[19]
        return vf, np.array([wx, wy, wz], dtype=np.float32)
    except Exception as e:
        print(f"Error cargando datos OXTS en {file_path}: {e}")
        return 0.0, np.zeros(3, dtype=np.float32)


def get_frame_interval(timestamps, frame_idx):
    """
    Computes the real time difference between the current frame
    and the next frame.

    This is important so that TTI computation does not assume
    a constant FPS of 10.
    """
    if frame_idx + 1 < len(timestamps):
        return timestamps[frame_idx + 1] - timestamps[frame_idx]
    else:
        # For the last frame, return the last known interval
        return timestamps[frame_idx] - timestamps[frame_idx - 1]


def load_velodyne_points(velo_path):
    """
    Load LiDAR points from text files (.txt).

    Expected format:
    X Y Z Reflectance (space-separated)
    """

    if not os.path.exists(velo_path):
        raise FileNotFoundError(
            f"LiDAR file not found at: {velo_path}"
        )

    # np.loadtxt is ideal for the provided format
    points = np.loadtxt(velo_path)

    # Return only the first 3 columns (X, Y, Z)
    return points[:, :3]


def project_velo_to_image(points, P_rect, Tr_velo_to_cam):
    """Projects 3D LiDAR points onto the 2D image plane.

    Args:
        points: (N, 3) XYZ coordinates from LiDAR
        P_rect: (3, 4) Complete rectified projection matrix (must be 3x4 in shape)
        Tr_velo_to_cam: (4, 4) Homogeneous transformation matrix from Velodyne to Camera
    """
    N = points.shape[0]

    # 1. Homogeneous coordinates for Velodyne: (N, 4)
    pts_3d_hom = np.hstack((points, np.ones((N, 1))))

    # 2. Transform to camera frame: (4, 4) x (4, N) -> (4, N) -> transpose to (N, 4)
    pts_cam = (Tr_velo_to_cam @ pts_3d_hom.T).T

    # Filter points in front of the camera (Z > 0)
    valid = pts_cam[:, 2] > 1e-5
    pts_cam_valid = pts_cam[valid]  # Shape: (M, 4)

    # Ensure P_rect is size (3, 4) in case it was passed trimmed/cropped
    if P_rect.shape[1] == 3:
        # If passed as 3x3 by mistake, append a column of zeros
        P_rect_full = np.hstack((P_rect, np.zeros((3, 1))))
    else:
        P_rect_full = P_rect

    # 3. Projection to image: P_rect_full (3, 4) x pts_cam_valid.T (4, M) -> (3, M) -> transpose to (M, 3)
    pts_2d_hom = (P_rect_full @ pts_cam_valid.T).T

    # 4. Normalize by depth (Z)
    depths = pts_2d_hom[:, 2]
    u = pts_2d_hom[:, 0] / depths
    v = pts_2d_hom[:, 1] / depths

    # 5. Map back to original sizes
    pts_2d = np.zeros((N, 2), dtype=np.float32)
    depth_full = np.zeros((N,), dtype=np.float32)

    original_indices = np.where(valid)[0]
    pts_2d[original_indices, 0] = u
    pts_2d[original_indices, 1] = v
    depth_full[original_indices] = depths

    return pts_2d, depth_full

def load_kitti_calib(calib_cam_path, calib_velo_path):
    """
    Read calibration files and return the required matrices.
    """

    def read_file(path):
        data = {}

        with open(path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                key, value = line.split(':', 1)

                # Try to convert values to float
                try:
                    data[key] = np.array(
                        [float(x) for x in value.split()]
                    )

                # Ignore non-numeric lines
                except ValueError:
                    continue

        return data

    # 1. Load Cam-to-Cam calibration
    # (intrinsic parameters and rectification)
    cam_to_cam = read_file(calib_cam_path)
    P_rect_02 = cam_to_cam['P_rect_02'].reshape(3, 4)

    # 2. Load Velodyne-to-Camera calibration
    # (extrinsic parameters)
    velo_to_cam_data = read_file(calib_velo_path)

    R = velo_to_cam_data['R'].reshape(3, 3)
    T = velo_to_cam_data['T'].reshape(3, 1)

    # Create homogeneous transformation matrix Tr_velo_to_cam (4x4)
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :3] = R
    Tr_velo_to_cam[:3, 3] = T.flatten()

    return P_rect_02, Tr_velo_to_cam

