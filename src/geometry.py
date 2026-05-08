import numpy as np

def calculate_tti_from_points(p0, p1, foe, dt):
    """
    Computes the Time-To-Impact (TTI) for a set of points.

    Formula:
        tau = d / d_dot * dt

    where:
        d      = distance from the point to the FOE
        d_dot  = radial expansion speed
        dt     = time difference between frames
    """

    # d: distance from the original points to the FOE
    dist_t = np.linalg.norm(p0 - foe, axis=1)

    # d_next: distance from the displaced points to the FOE
    dist_t_next = np.linalg.norm(p1 - foe, axis=1)

    # d_dot: radial expansion speed (change in distance)
    dot_d = dist_t_next - dist_t

    # Avoid division by zero or negative expansion
    dot_d[dot_d <= 0] = 1e-6

    # Compute TTI (tau)
    tti = (dist_t / dot_d) * dt

    return tti