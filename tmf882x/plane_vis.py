import rclpy
from rclpy.node import Node

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import json

from tmf882x_interfaces.msg import TMF882XMeasure

AOI_LIMIT = 5 # above this aoi (in degrees) label turns red
TEXT_VERTICAL_SPACING = 0.15
TEXT_OFFSET = 0.6
AOI_SMOOTHING = 5
DIST_SMOOTHING = 5

DIST_TO_BIN_SLOPE = 73.484
DIST_TO_BIN_INTERCEPT = 13.2521

ZONE_SPEC_PATH = "/home/carter/projects/Transient-Reconstruction/specs/TMF882X_spec_1_rotated.json"
with open(ZONE_SPEC_PATH, "r") as f:
    ZONE_SPEC = json.load(f)

class PlaneVis(Node):
    def __init__(self):
        super().__init__('plane_vis')

        self.DIST_TO_BIN_SLOPE = 73.484
        self.DIST_TO_BIN_INTERCEPT = 13.2521

        self.subscriber = self.create_subscription(TMF882XMeasure, 'tmf882x', self.sub_callback, 1)

        self.fig = plt.figure()
        self.ax0 = self.fig.add_subplot(121, projection='3d')
        self.ax1 = self.fig.add_subplot(122)
        self.fig.set_size_inches(20, 12)
        self.fig.tight_layout()
        plt.show(block=False)

        self.previous_aois = []
        self.previous_dists = []

        plt.ion()
        plt.show(block=False)

    def sub_callback(self, msg):
        a, d, res = method11(
            np.array(msg.hists).reshape(9, 128),
            None,
            None,
            m=71.0877,
            b=13.2672,
            edge_fov_scale=0.9263,
            corner_fov_scale=0.90353,
        )

        aoi = angle_between_vecs(a, [0, 0, 1])
        azimuth = np.arctan2(a[1], a[0])
        z_dist = d / a[2]

        self.previous_aois.append(aoi)
        self.previous_dists.append(z_dist)

        z_dist = sum(self.previous_dists[-DIST_SMOOTHING:]) / DIST_SMOOTHING
        aoi = sum(self.previous_aois[-AOI_SMOOTHING:]) / AOI_SMOOTHING

        if len(self.previous_dists) > DIST_SMOOTHING:
            self.previous_dists.pop(0)
        if len(self.previous_aois) > AOI_SMOOTHING:
            self.previous_aois.pop(0)

        print(f"aoi: {np.degrees(aoi):0.1f}, azimuth: {np.degrees(azimuth):0.1f}, z_dist: {z_dist:0.3f}")
        print()

        self.ax0.cla()
        self.ax0.set_xlim(-0.5, 0.5)
        self.ax0.set_ylim(-0.5, 0.5)
        self.ax0.set_zlim(-1.0, 0.0)
        self.ax0.set_xlabel('X')
        self.ax0.set_ylabel('Y')
        self.ax0.set_zlabel('Z')
        # remove axis tick labels
        self.ax0.set_xticklabels([])
        self.ax0.set_yticklabels([])
        self.ax0.set_zticklabels([])
        # remove axis ticks
        # ax0.set_xticks([])
        # ax0.set_yticks([])
        # ax0.set_zticks([])
        # remove entire axis frame
        # ax0.axis('off')
        
        fov_angle = np.radians(40)/2
        points = [
            intersect_ray_plane(0, rots_to_u_vec(-fov_angle, fov_angle), a, d),
            intersect_ray_plane(0, rots_to_u_vec(fov_angle, fov_angle), a, d),
            intersect_ray_plane(0, rots_to_u_vec(-fov_angle, -fov_angle), a, d),
            intersect_ray_plane(0, rots_to_u_vec(fov_angle, -fov_angle), a, d),
        ]
        if None not in points:
            points = np.array([p[1] for p in points])
            self.ax0.plot_surface(
                points[:,0].reshape(2, 2),
                points[:,1].reshape(2, 2),
                -points[:,2].reshape(2, 2), # invert z axis so it matches intuition
            )

            # plot the sensor position as a big gray dot
            self.ax0.scatter([0], [0], [0], color='gray', s=100)

            # plot a line from the sensor position (origin) to each corner of the FoV
            for p in points:
                self.ax0.plot([0, p[0]], [0, p[1]], [0, -p[2]], color='gray')

        self.ax1.cla()
        self.ax1.axis('off')

        aoi_color = 'red' if np.degrees(aoi) > AOI_LIMIT else 'green'
        z_dist_color = 'gray'
        self.ax1.text(
            0.1,
            TEXT_OFFSET,
            f'Slope: {np.degrees(aoi):0.1f}°',
            verticalalignment='bottom',
            horizontalalignment='left',
            transform=self.ax1.transAxes,
            color='black',
            fontsize=48,
            bbox={'facecolor': aoi_color, 'alpha': 0.8, 'pad': 10}
        )
        self.ax1.text(
            0.1,
            TEXT_OFFSET - TEXT_VERTICAL_SPACING,
            f'Distance: {z_dist*1000:0.1f} mm',
            verticalalignment='bottom',
            horizontalalignment='left',
            transform=self.ax1.transAxes,
            color='black',
            fontsize=48,
            bbox={'facecolor': z_dist_color, 'alpha': 0.8, 'pad': 10}
        )
                
        plt.pause(0.05)

def angle_between_vecs(v1, v2, acute=True):
    # https://stackoverflow.com/a/39533085/8841061
    # v1 is your firsr vector
    # v2 is your second vector
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if (acute == True):
        return angle
    else:
        return 2 * np.pi - angle
    
def intersect_ray_plane(p, u, a, d, epsilon=1e-6):
    """Find intersection of a ray with a plane
    https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python
    Args:
        p (3-tuple of floats): starting point of ray
        u (3-tuple of floats): direction of ray
        a (3-tuple of floats): the equation for the plane where a[0]x + a[1]y + a[2]z - d = 0.
        d (float) the c portion of the plane equation.
    Returns:
        The distance and intersection point as a tuple, for example, with distance
        5.2 and intersection point (8.1, 0.3, 4):
        (5.2, (8.1, 0.3, 4)) or float('inf') if the sensor does not see the plane.
    Raises:
        ValueError: The line is undefined.
    """
    a = np.array(a)
    p = np.array(p)
    u = np.array(u)

    plane_point = a * d

    ndotu = a.dot(u)
    if abs(ndotu) < epsilon:
        return None

    w = p - plane_point
    si = -a.dot(w) / ndotu
    Psi = w + si * u + plane_point
    
    dist = np.linalg.norm(Psi - p)

    if(np.allclose((dist * u) + p, Psi)):
        return (dist, Psi)
    else:
        return None
    
def rots_to_u_vec(x_rot, y_rot):
    """
    Given some angular coordinate rotations (x_rot and y_rot), return a 3D unit vector which points
    in the given direction, relative to the camera's optical axis (which is in the positive z
    direction).

    Args:
        x_rot, y_rot: direction to face in angular coordinates
    """
    # start with the u vector facing out from the camera
    u = np.array([0, 0, 1])
    # to rotate in the positive x angular direction, you need to rotate around the y axis in a 
    # negative direction
    x_rot_mat = np.array([
        [np.cos(x_rot), 0, np.sin(x_rot)],
        [0, 1, 0],
        [-np.sin(x_rot), 0, np.cos(x_rot)]
    ])
    # to rotate in the positive y angular direction, you need to rotate around the x axis in a
    # positive direction
    y_rot_mat = np.array([
        [1, 0, 0],
        [0, np.cos(-y_rot), -np.sin(-y_rot)],
        [0, np.sin(-y_rot), np.cos(-y_rot)],
    ])
    u = x_rot_mat @ u
    u = y_rot_mat @ u

    return u

def method11(hists, measurements, reference_hist, m=72.07336587889849, b=13.155326458933663, edge_fov_scale=0.9404948331338918, corner_fov_scale=0.8916354321171438):
    """
    **CHEATY**

    Same as method 9 but using a jointly optimized slope, intercept, and fov scales

    for autoscan5, jointly optimized values are:
        Dist to bin parameters m: 70.92253727684945, b: 13.409657422450657
        Edge FoV scale: 0.9654136810494751, corner FoV scale: 0.8880089592171669
        min avg point error: 0.00883

    for autoscan6, jointly optimized values are:
        Dist to bin parameters m: 71.09411290045911, b: 13.319636312381098
        Edge FoV scale: 0.9788928066704715, corner FoV scale: 0.8697650566334645
        min avg point error: 0.01049

    for data_specs[20](white paper), jointly optimized values are:
        edge scale:  0.9404948331338918
        corner scale:  0.8916354321171438
        dist to bin m:  72.07336587889849
        dist to bin b:  13.155326458933663
        avg point error 0.0027750142905381375
        m=72.07336587889849, b=13.155326458933663, edge_fov_scale=0.9404948331338918, corner_fov_scale=0.8916354321171438

    params before I switched them out for white paper params:
    m=71.09411, b=13.319636, edge_fov_scale=0.9788928, corner_fov_scale=0.8697650
    """

    pts = []
    peak_dists = []
    for hist, single_zone_spec in zip(hists, ZONE_SPEC):
        cx, cy = single_zone_spec["center"]
        # if cx == 0 or cy == 0, this zone spec is an edge zone, so apply edge fov scale
        if cx == 0 or cy == 0:
            cx = cx*edge_fov_scale
            cy = cy*edge_fov_scale
        # otherwise, it's a corner zone, so apply corner fov scale
        else:
            cx = cx*corner_fov_scale
            cy = cy*corner_fov_scale

        # hist = hist * 1/FALLOFF_FN
        # find the peak bin by fitting a cubic
        cubic_hist = CubicSpline(np.arange(128), hist)
        interpolated_hist = cubic_hist(np.arange(0, 128, 0.1))
        peak_bin = interpolated_hist.argmax()/10

        peak_dist = TMF882X_bin_to_dist(peak_bin, slope=m, intercept=b)
        u = rots_to_u_vec(cx, cy)
        if peak_dist is not None:
            pts.append(u*peak_dist)
            peak_dists.append(peak_dist)

    if np.array(peak_dists).max() < 0.03:
        a, d, res = fit_plane_zdist(pts)
    else:
        a, d, res = fit_plane(pts)

    return a, d, res

def fit_plane_zdist(pts):
    """
    https://math.stackexchange.com/a/2306029
    """
    pts = np.array(pts)
    xs = pts[:,0]
    ys = pts[:,1]
    zs = pts[:,2]
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)

    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)
    # print("solution: %f x + %f y + %f = z" % (fit[0].item(), fit[1].item(), fit[2].item()))

    a = [fit[0].item(), fit[1].item(), -1]
    d = fit[2].item()

    a = a / np.linalg.norm(a)
    d = -d * np.linalg.norm(a)

    if d < 0:
        a = -a
        d = -d

    return a, d, residual

def fit_plane(pts, initial_est = [0, 0, 1, 0.5]):
    """Fit a plane given by ax+d = 0 to a set of points
    Works by minimizing the sum over all points x of ax+d
    Arguments:
      pts: array of points in 3D space
    Returns:
      (3x1 numpy array): a vector for plane equation
      (float): d in plane equation
      (float): sum of residuals for points to plane (orthogonal l2 distance)
    """

    pts = np.array(pts)

    def loss_fn(x, points):
        a = np.array(x[:3])
        d = x[3]

        loss = 0
        for point in points:
            loss += np.abs(np.dot(a, np.array(point)) - d)

        return loss

    def a_constraint(x):
        return np.linalg.norm(x[:3]) - 1
    

    soln = minimize(
        loss_fn,
        np.array(initial_est),
        args=(pts),
        method='slsqp',
        constraints=[
            {
                'type': 'eq',
                'fun': a_constraint
            }
        ],
        bounds=[
            (-1, 1),
            (-1, 1),
            (-1, 1),
            (0, None)
        ]
    )

    a = soln.x[:3]
    d = soln.x[3]
    res = soln.fun

    return a, d, res

def TMF882X_bin_to_dist(bin, slope=DIST_TO_BIN_SLOPE, intercept=DIST_TO_BIN_INTERCEPT):
    if bin < 0 or bin > 127:
        return None
    return (bin-intercept)/slope

def main(args=None):
    rclpy.init(args=args)
    plane_vis = PlaneVis()
    rclpy.spin(plane_vis)

    plane_vis.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
