import rclpy
from rclpy.node import Node

import numpy as np
import matplotlib.pyplot as plt

from tmf882x_interfaces.msg import TMF882XMeasure

class TMF882XVis(Node):
    def __init__(self):
        super().__init__('tmf882x_vis')

        self.subscriber = self.create_subscription(TMF882XMeasure, 'tmf882x', self.sub_callback, 1)

        self.fig, self.ax = plt.subplots(3, 3)
        self.fig.set_size_inches(10, 10)
        self.fig.tight_layout()
        plt.show(block=False)

    def sub_callback(self, msg):

        hists = np.array(msg.hists).reshape(9, 128)
        for row in range(3):
            for col in range(3):
                self.ax[row][col].cla()
                self.ax[row][col].plot(hists[row*3+col])

        plt.pause(0.05)

def main(args=None):
    rclpy.init(args=args)
    tmf882x_vis = TMF882XVis()
    rclpy.spin(tmf882x_vis)

    tmf882x_vis.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
