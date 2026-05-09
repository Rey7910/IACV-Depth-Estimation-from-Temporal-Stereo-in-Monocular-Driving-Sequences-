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
    """
    Project 3D LiDAR points onto the 2D image plane.
    """

    # 1. Convert to homogeneous coordinates (N, 4)
    pts_3d_hom = np.hstack(
        (points, np.ones((points.shape[0], 1)))
    )

    # 2. Transform from Velodyne to Camera 0
    # (KITTI reference coordinate system)
    # pts_cam = pts_3d_hom @ Tr_velo_to_cam.T
    pts_cam = (Tr_velo_to_cam @ pts_3d_hom.T).T

    # 3. Project onto the image using P_rect
    # Note: P_rect already includes rectification
    # in KITTI 'extract' sequences
    pts_2d_hom = (P_rect @ pts_cam.T).T

    # 4. Normalize to obtain pixel coordinates (u, v)
    depth = pts_2d_hom[:, 2]
    pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]

    return pts_2d, depth


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

