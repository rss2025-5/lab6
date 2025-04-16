import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from rcl_interfaces.msg import SetParametersResult
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry

from .utils import LineTrajectory


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.following_distance = 0.01  # FILL IN #
        self.speed = 0.5  # FILL IN #
        self.wheelbase = 0.33  # FILL IN #

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.pose_callback,
            1
        )

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)

        self.initialized_traj = False

    def pose_callback(self, msg):
        if not self.initialized_traj or len(self.trajectory.points) < 2:
            return

        traj_pts = np.array(self.trajectory.points)
        self.get_logger().info(f"traj_pts:{traj_pts}")
        dists = np.linalg.norm(traj_pts - np.array([msg.pose.pose.position.x, msg.pose.pose.position.y]), axis=1)
        nearest_idx = np.argmin(dists)
        closest_x = traj_pts[nearest_idx][0]
        closest_y = traj_pts[nearest_idx][1]

        self.relative_x = closest_x - msg.pose.pose.position.x
        self.relative_y = closest_y - msg.pose.pose.position.y
        distance = np.sqrt(self.relative_x**2 + self.relative_y**2)

        angle_to_line= -np.arctan2(-self.relative_y, self.relative_x)
        drive_cmd = AckermannDriveStamped()

        #################################
        self.distance_error = distance - self.following_distance

        self.get_logger().info("distance error: %s" %self.distance_error)
        self.get_logger().info("angle to line: %s" %angle_to_line)

        steering_angle = math.atan2(2.0 * self.wheelbase * math.sin(angle_to_line), distance)
        velocity = min(self.distance_error*2, 0.8)

        drive_cmd.drive.steering_angle = -steering_angle
        drive_cmd.drive.speed = velocity

        self.drive_pub.publish(drive_cmd)

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
