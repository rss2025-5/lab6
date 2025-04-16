import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from rcl_interfaces.msg import SetParametersResult
from visualization_msgs.msg import Marker
from nav_msg.msg import Odometry

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

        self.lookahead = 1.5  # FILL IN #
        self.speed = 0.5  # FILL IN #
        self.wheelbase_length = 0  # FILL IN #

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

    def pose_callback(self, odometry_msg):
        if not self.initialized_traj or len(self.trajectory.points) < 2:
            return

        # 1. Get car position
        position = odometry_msg.pose.pose.position
        orientation = odometry_msg.pose.pose.orientation

        car_x = position.x
        car_y = position.y

        # Get yaw from quaternion
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        car_yaw = math.atan2(siny_cosp, cosy_cosp)

        # 2. Find nearest point on trajectory
        car_pos = np.array([car_x, car_y])
        traj_pts = np.array(self.trajectory.points)  # list of (x, y)

        dists = np.linalg.norm(traj_pts - car_pos, axis=1)
        nearest_idx = np.argmin(dists)

        # 3. Find lookahead point
        lookahead_point = None
        for i in range(nearest_idx, len(traj_pts) - 1):
            p1 = traj_pts[i]
            p2 = traj_pts[i + 1]

            # Compute intersection between circle and line segment
            lookahead_point = self.find_circle_segment_intersection(car_pos, self.lookahead, p1, p2)
            if lookahead_point is not None:
                break

        if lookahead_point is None:
            self.get_logger().warn("No valid lookahead point found!")
            return

        # 4. Transform to car frame
        dx = lookahead_point[0] - car_x
        dy = lookahead_point[1] - car_y

        # rotate by -car_yaw
        local_x = math.cos(-car_yaw) * dx - math.sin(-car_yaw) * dy
        local_y = math.sin(-car_yaw) * dx + math.cos(-car_yaw) * dy

        # 5. Compute steering
        alpha = math.atan2(local_y, local_x)
        Ld = math.sqrt(local_x**2 + local_y**2)
        if Ld < 1e-5:
            return

        steering_angle = math.atan2(2.0 * self.wheelbase_length * math.sin(alpha), Ld)

        # 6. Publish drive command
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = self.speed
        self.drive_pub.publish(drive_msg)

    def find_circle_segment_intersection(self, center, radius, p1, p2):
        # Shift segment to circle's coordinate system
        p1_shifted = p1 - center
        p2_shifted = p2 - center
        d = p2_shifted - p1_shifted
        f = p1_shifted

        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius**2

        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None  # no intersection

        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)

        for t in [t1, t2]:
            if 0.0 <= t <= 1.0:
                intersection = p1 + t * (p2 - p1)
                return intersection

        return None

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
