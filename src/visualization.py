import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_tti_and_depth(img, p0, p1, depths, foe, vf, title="TTI & Depth Estimation"):
    """
    Visualizes motion vectors, estimated depth, and TTI.
    """
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for i in range(len(p0)):
        a, b = p0[i].ravel()
        c, d = p1[i].ravel()

        # 1. Draw motion vector (yellow)
        plt.arrow(a, b, c - a, d - b, color='yellow', head_width=3, alpha=0.8)

        # 2. Depth and TTI labels
        tti = depths[i] / vf if vf > 0 else 0

        plt.text(
            a, b,
            f"{depths[i]:.1f}m\n{tti:.1f}s",
            color='lime',
            fontsize=9,
            fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1)
        )

    # Draw the FOE
    plt.plot(
        foe[0],
        foe[1],
        'ro',
        markersize=12,
        markeredgecolor='white',
        label='FOE'
    )

    plt.title(f"{title} (vf={vf:.2f} m/s)", fontsize=15)
    plt.legend()
    plt.axis('off')
    plt.show()


def plot_lidar_overlay(img, pts_2d, gt_depths, title="LiDAR-Image Overlay"):
    """
    Displays LiDAR points projected onto the image to verify alignment.
    """
    plt.figure(figsize=(15, 6))
    h, w = img.shape[:2]

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Filter points inside the image area
    in_image = (
        (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) &
        (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    )

    sc = plt.scatter(
        pts_2d[in_image, 0],
        pts_2d[in_image, 1],
        c=gt_depths[in_image],
        s=1,
        cmap='jet',
        alpha=0.4
    )

    plt.xlim(0, w)
    plt.ylim(h, 0)

    plt.colorbar(sc, label='LiDAR Depth (m)')
    plt.title(title)
    plt.show()


def plot_validation_stats(errors, z_comparison):
    """
    Generates the error histogram and the correlation plot versus LiDAR.
    """
    if not errors:
        print("No errors available for visualization.")
        return

    plt.figure(figsize=(15, 5))

    # Error histogram
    plt.subplot(1, 2, 1)

    plt.hist(
        errors,
        bins=10,
        color='skyblue',
        edgecolor='black',
        range=(0, 1)
    )

    plt.axvline(
        np.mean(errors),
        color='red',
        linestyle='--',
        label=f'Mean: {np.mean(errors):.2f}'
    )

    plt.title("Relative Error Distribution (0–100%)")
    plt.xlabel("Error (1.0 = 100%)")
    plt.ylabel("Frequency")
    plt.legend()

    # Correlation plot
    plt.subplot(1, 2, 2)

    gt_v, est_v = zip(*z_comparison)

    plt.scatter(
        gt_v,
        est_v,
        color='green',
        s=40,
        alpha=0.7,
        label='Estimated Points'
    )

    limit = max(max(gt_v), max(est_v)) + 5

    plt.plot(
        [0, limit],
        [0, limit],
        'r--',
        label="Ideal Reference"
    )

    plt.xlim(0, limit)
    plt.ylim(0, limit)

    plt.title("Correlation: LiDAR Z vs TTI Z")
    plt.xlabel("Ground Truth Z (m)")
    plt.ylabel("Estimated Z (m)")

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_vanishing_point_validation(
    img,
    foe_theoretical,
    vp_estimated,
    filtered_lines,
    title="Projective Geometry Validation"
):
    """
    Visualizes the convergence of road lines toward the vanishing point (VP)
    and its relationship with the theoretical FOE.
    """
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    h, w = img.shape[:2]

    # 1. Draw detected lines and their extension toward the VP
    if filtered_lines is not None:
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            plt.plot([x1, x2], [y1, y2], color='cyan', linewidth=2, alpha=0.8)

            if vp_estimated is not None:
                plt.plot(
                    [x2, vp_estimated[0]],
                    [y2, vp_estimated[1]],
                    color='cyan',
                    linestyle=':',
                    linewidth=1,
                    alpha=0.3
                )

    # 2. Draw theoretical FOE (red)
    plt.plot(
        foe_theoretical[0],
        foe_theoretical[1],
        'ro',
        markersize=15,
        markeredgecolor='white',
        label='Theoretical FOE (Calibration)'
    )

    # 3. Draw estimated vanishing point (blue)
    if vp_estimated is not None:
        plt.plot(
            vp_estimated[0],
            vp_estimated[1],
            'bx',
            markersize=12,
            markeredgewidth=3,
            label='Vanishing Point (Lines)'
        )

    # 4. Perspective guides from lower corners
    plt.plot(
        [0, foe_theoretical[0]],
        [h, foe_theoretical[1]],
        'y--',
        alpha=0.3,
        label='Perspective Guide'
    )

    plt.plot(
        [w, foe_theoretical[0]],
        [h, foe_theoretical[1]],
        'y--',
        alpha=0.3
    )

    plt.title(title, fontsize=16)
    plt.legend(loc='upper right')
    plt.axis('off')

    # 5. Error diagnostics
    if vp_estimated is not None:
        pixel_error = np.linalg.norm(vp_estimated - foe_theoretical)

        print(f"--- GEOMETRY DIAGNOSTICS ---")
        print(f"FOE-VP Distance: {pixel_error:.2f} pixels")

        status = "SUCCESS" if pixel_error < 50 else f"WARNING ({pixel_error:.1f}px)"
        desc = "Straight-line motion." if pixel_error < 50 else "Possible curve/inclination."
        print(f"STATUS: {status}. {desc}")

    plt.show()


def plot_cross_ratio_validation(
    img,
    p0_final,
    p2_final,
    p0_noisy,
    total_points,
    title="Experiment 03: Cross-Ratio Validation"
):
    """
    Visualizes consistent flow points (green) and rejected points (red)
    based on the projective invariant Cross-Ratio.
    """
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 1. Draw consistent points (green long vectors)
    for i in range(len(p0_final)):
        plt.arrow(
            p0_final[i, 0], p0_final[i, 1],
            p2_final[i, 0] - p0_final[i, 0],
            p2_final[i, 1] - p0_final[i, 1],
            color='lime',
            head_width=3,
            alpha=0.7,
            label='Consistent' if i == 0 else ""
        )

    # 2. Draw noisy points (red dots)
    if len(p0_noisy) > 0:
        plt.scatter(
            p0_noisy[:, 0],
            p0_noisy[:, 1],
            color='red',
            s=20,
            alpha=0.8,
            label='Inconsistent (Noise/Dynamic)'
        )

    # Metrics
    reliability = (len(p0_final) / total_points) * 100

    plt.title(
        f"{title}\nConsistent: {len(p0_final)} | Discarded: {len(p0_noisy)}",
        fontsize=15
    )
    plt.legend()
    plt.axis('off')

    print(f"--- TEMPORAL CONSISTENCY ---")
    print(f"Temporal Reliability Index: {reliability:.2f}%")

    plt.show()