import numpy as np
from datetime import datetime

def parse_timestamps(path):
    """Read timestamps y transform them to total seconds."""
    with open(path, 'r') as f:
        times = [datetime.strptime(line.strip()[:-3], '%Y-%m-%d %H:%M:%S.%f') for line in f]
    return [(t - times[0]).total_seconds() for t in times]

def load_oxts_data(oxts_dir, frame_idx):
    """
    Extract lineal velocity(m/s) from the OXTS file for each specific frame.
    The KITTI OXTS formathas the front velocity in column 8.
    """
    file_path = f"{oxts_dir}/{frame_idx:010d}.txt"
    data = np.loadtxt(file_path)
    
    # vf: forward velocity (m/s), vl: leftward velocity, vu: upward velocity
    vf = data[8] 
    return vf

def get_frame_interval(timestamps, frame_idx):
    """Real delta t computation between the current frame and the next."""
    return timestamps[frame_idx + 1] - timestamps[frame_idx]