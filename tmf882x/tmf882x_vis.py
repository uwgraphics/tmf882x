import rclpy
from rclpy.node import Node

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from tmf882x_interfaces.msg import TMF882XMeasure

class TMF882XVis(Node):
    def __init__(self):
        super().__init__('tmf882x_vis')

        self.DIST_TO_BIN_SLOPE = 73.484
        self.DIST_TO_BIN_INTERCEPT = 13.2521
        # for plotting - the histogram in index 1 is in the top left, 4 is in the top middle, etc.
        # as if you're reading a book, so the zones look like this:
        # 2 1 0
        # 5 4 3
        # 8 7 6
        # This arrangement gives you something like a camera would, the view as you're looking
        # *through* the sensor lines up with what's plotted.
        self.ZONE_ORDER = [2, 1, 0, 5, 4, 3, 8, 7, 6]

        self.subscriber = self.create_subscription(TMF882XMeasure, 'tmf882x', self.sub_callback, 1)

        self.hist_fig, self.hist_ax = plt.subplots(3, 3)
        self.hist_fig.set_size_inches(10, 10)
        self.hist_fig.suptitle("Captured Histograms")
        self.hist_fig.tight_layout()

        self.dist_fig, self.dist_ax = plt.subplots(3, 3)
        self.dist_fig.set_size_inches(4, 4)
        self.dist_fig.suptitle("Internally Computed Distances (mm)")
        self.dist_fig.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0, right=1)

        self.argmax_fig, self.argmax_ax = plt.subplots(3, 3)
        self.argmax_fig.set_size_inches(4, 4)
        self.argmax_fig.suptitle("Argmax of Histograms")
        self.argmax_fig.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0, right=1)

        self.interp_argmax_fig, self.interp_argmax_ax = plt.subplots(3, 3)
        self.interp_argmax_fig.set_size_inches(4, 4)
        self.interp_argmax_fig.suptitle("Interpolated Argmax of Histograms")
        self.interp_argmax_fig.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0, right=1)

        plt.ion()
        plt.show(block=False)

    def sub_callback(self, msg):
        hists = np.array(msg.hists).reshape(9, 128)
        max_val = np.max(hists)

        argmaxes = np.argmax(hists, axis=1)
        argmax_dists = ((argmaxes-self.DIST_TO_BIN_INTERCEPT)/self.DIST_TO_BIN_SLOPE)*1000
        argmax_dists = np.clip(argmax_dists, 0, None)

        # calculate interpolated argmax distances
        interp_argmax_dists = []
        for i in range(9):
            cubic_hist = CubicSpline(np.arange(128), hists[i,:])
            interpolated_hist = cubic_hist(np.arange(0, 128, 0.01))
            peak_bin = interpolated_hist.argmax()/100
            dist = (peak_bin-self.DIST_TO_BIN_INTERCEPT)/self.DIST_TO_BIN_SLOPE*1000
            interp_argmax_dists.append(np.clip(dist, 0, None))
        
        hist_idx = 0
        for row in range(3):
            for col in range(3):
                self.hist_ax[row][col].cla()
                self.hist_ax[row][col].plot(hists[self.ZONE_ORDER[hist_idx]])
                # self.hist_ax[row][col].set_ylim([0, max_val])
                self.hist_ax[row][col].set_xticks([])

                self.display_distances(
                    self.dist_ax[row][col],
                    msg.depths_1[self.ZONE_ORDER[hist_idx]],
                    max_dist=1600
                )

                self.display_distances(
                    self.argmax_ax[row][col],
                    argmax_dists[self.ZONE_ORDER[hist_idx]],
                    max_dist=1600
                )

                self.display_distances(
                    self.interp_argmax_ax[row][col],
                    interp_argmax_dists[self.ZONE_ORDER[hist_idx]],
                    max_dist=1600
                )

                hist_idx += 1

        plt.pause(0.05)

    def display_distances(self, ax, dist, max_dist):
        ax.cla()
        ax.axis('off')
        shade = 1-(dist / max_dist)
        if shade < 0:
            self.get_logger().warn(f"Observed distance ({dist}) is greater than max distance used for visualization ({max_dist})")
            shade = 0
        ax.text(
            0.5,
            0.5,
            f"{dist:.0f}",
            color='white' if shade < 0.5 else 'black',
            fontsize=12,
            verticalalignment='center',
            horizontalalignment='center',
        )
        ax.fill(
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            str(shade)
        )


def main(args=None):
    rclpy.init(args=args)
    tmf882x_vis = TMF882XVis()
    rclpy.spin(tmf882x_vis)

    tmf882x_vis.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
