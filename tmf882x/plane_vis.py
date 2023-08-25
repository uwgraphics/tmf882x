import json

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from tmf882x_interfaces.msg import TMF882XMeasure
from ur5_sensors_ros2.lib.util import (TMF882X_bin_to_dist, angle_between_vecs,
                                       fit_plane, fit_plane_zdist,
                                       intersect_ray_plane, method11,
                                       rots_to_u_vec)

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
    
def main(args=None):
    rclpy.init(args=args)
    plane_vis = PlaneVis()
    rclpy.spin(plane_vis)

    plane_vis.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
