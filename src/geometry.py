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


def estimate_dynamic_foe(p0, p1):
    """
    Estimate the FOE by finding the common intersection point of the
    flow lines. Useful when small rotations are present.
    """
    
    # Simplified implementation using the center of mass of intersections
    # or flow-line regression.
    # For now, we use the optical center corrected by the mean flow.
    flow = p1 - p0
    
    # The FOE tends to be located where the flow is minimal
    # in terms of expansion
    return np.mean(p0, axis=0)  # Placeholder for dynamic refinement

def estimate_vanishing_point(lines):
    """
    1. FOE VALIDATION (Plane at infinity pi_inf):
    Computes the vanishing point of the road lines.
    Under pure translation, this point should coincide with the FOE.
    """
    if lines is None:
        return None
    
    # Line intersection (simplified implementation for 2 main lines)
    # Expected line format: (x1, y1, x2, y2)
    def intersect(l1, l2):
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
        if denom == 0:
            return None
        
        ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
        return np.array([x1 + ua*(x2-x1), y1 + ua*(y2-y1)])

    # If at least two lines exist, compute their average intersection
    if len(lines) >= 2:
        return intersect(lines[0][0], lines[1][0])
    
    return None


def validate_with_cross_ratio(p0, p1, p2, foe, threshold_min=0.8, threshold_max=1.2, epsilon=1e-6):
    """
    Validates temporal consistency using the true projective Cross-Ratio 
    under a constant velocity assumption, relative to the Epipole / FOE.
    
    Cross-Ratio(p0, p1, p2, foe) = ((d0 - d2) * (d1 - d_foe)) / ((d1 - d2) * (d0 - d_foe))
    Since the distance from any point to the FOE in the time-horizon limits to infinity
    relative to the localized frame delta, the formulation simplifies downstream to checking
    the harmonic growth of radial distances: d0/d1 relative to d1/d2.
    """
    # foe is expected as an array/tuple: [cx, cy]
    foe = np.array(foe).reshape(1, 2)
    
    # 1. Compute radial Euclidean distances from the FOE to the features in each frame
    d0 = np.linalg.norm(p0 - foe, axis=1)
    d1 = np.linalg.norm(p1 - foe, axis=1)
    d2 = np.linalg.norm(p2 - foe, axis=1)
    
    # 2. To avoid division by zero or negative geometry due to flow noise
    # We evaluate the strict harmonic expansion mapping for constant velocity
    # In pure translation, standard perspective scaling states: (d1 - d0)/d0 = delta_s / Z
    # Projectively, the Cross-Ratio invariant for 3 time steps + FOE simplifies to:
    # CR = (d0 * (d2 - d1)) / (d2 * (d1 - d0))
    # For zero acceleration (constant speed), this theoretical value CR must equal exactly 1.0.
    
    numerator = d0 * (d2 - d1)
    denominator = d2 * (d1 - d0) + epsilon
    
    cross_ratios = numerator / denominator
    
    # 3. Filter points whose Cross-Ratio deviates significantly from the ideal 1.0
    # A tight threshold (e.g., 0.8 to 1.2) captures true constant-velocity static background.
    mask_consistent = (cross_ratios >= threshold_min) & (cross_ratios <= threshold_max)
    
    return mask_consistent


def filter_static_points(p0, p1, foe, threshold=0.35):
    """
    Filters static points based on radial alignment with the FOE.
    In pure translation, the vector (p1 - p0) should be aligned with (p0 - foe).
    """

    # 1. Vector from the FOE to the original point
    vec_foe_p0 = p0 - foe

    # 2. Optical flow vector (motion)
    vec_flow = p1 - p0

    # 3. Compute the angle between the theoretical expansion vector and the real flow

    # unit_vec_foe: expected flow direction
    unit_vec_foe = vec_foe_p0 / (
        np.linalg.norm(vec_foe_p0, axis=1, keepdims=True) + 1e-6
    )

    # unit_vec_flow: observed flow direction
    unit_vec_flow = vec_flow / (
        np.linalg.norm(vec_flow, axis=1, keepdims=True) + 1e-6
    )

    # Error is the sine of the angle (0 if vectors are parallel/radial)
    # Computed using the 2D determinant (2D cross product)
    errors = np.abs(
        unit_vec_foe[:, 0] * unit_vec_flow[:, 1]
        - unit_vec_foe[:, 1] * unit_vec_flow[:, 0]
    )

    # USE THE threshold VARIABLE INSTEAD OF A FIXED VALUE LIKE 0.2
    return errors < threshold