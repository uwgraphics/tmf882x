import rclpy
from rclpy.node import Node

class TMF882XPub(Node):
    def __init__(self):
        super().__init__('tmf882x_pub')

        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        self.get_logger().info("hello world")
        

def main(args=None):
    rclpy.init(args=args)
    tmf882x_pub = TMF882XPub()
    rclpy.spin(tmf882x_pub)

    tmf882x_pub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
