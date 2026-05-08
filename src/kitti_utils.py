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