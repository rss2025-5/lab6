import rclpy
from rclpy.node import Node
import numpy as np
import math

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
from ackermann_msgs.msg import AckermannDriveStamped

from .utils import LineTrajectory


class PurePursuit(Node):
    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 1.5
        self.speed = 3.0
        self.wheelbase_length = 0.33

        self.trajectory = LineTrajectory("/followed_trajectory")
        self.initialized_traj = False

        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.pose_callback, 1)
        self.traj_sub = self.create_subscription(PoseArray, "/trajectory/current", self.trajectory_callback, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 1)

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz()
        self.initialized_traj = True

    def pose_callback(self, msg):
        if not self.initialized_traj or len(self.trajectory.points) < 2:
            return

        # Car position and orientation
        pos = msg.pose.pose.position
        x, y = pos.x, pos.y
        q = msg.pose.pose.orientation
        self.latest_orientation = q
        _, _, yaw = self.quaternion_to_euler(q)
        car_pos = np.array([x, y])

        # Stop if near goal
        goal_pos = np.array(self.trajectory.points[-1])
        if np.linalg.norm(goal_pos - car_pos) < 0.4:
            self.stop_car()
            return

        # Find closest segment
        closest_pt, closest_seg_idx = self.closest_point_on_path(car_pos)

        # Find lookahead point starting from that segment
        lookahead_pt = self.find_lookahead_point(car_pos, closest_seg_idx)
        if lookahead_pt is None:
            self.get_logger().warn("No valid lookahead point found.")
            return

        # Transform lookahead to car frame
        dx = lookahead_pt[0] - x
        dy = lookahead_pt[1] - y
        local_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        local_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy

        if local_x <= 0:
            self.get_logger().warn("Lookahead point is behind vehicle.")
            return

        steering_angle = math.atan2(2 * self.wheelbase_length * local_y, self.lookahead**2)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.speed
        drive_msg.drive.steering_angle = steering_angle
        self.drive_pub.publish(drive_msg)

    def stop_car(self):
        stop_msg = AckermannDriveStamped()
        stop_msg.drive.speed = 0.0
        stop_msg.drive.steering_angle = 0.0
        self.drive_pub.publish(stop_msg)
        self.get_logger().info("Goal reached, stopping car.")

    def closest_point_on_path(self, car_pos):
        """Find the closest point on a segment in the trajectory to the car."""
        points = np.array(self.trajectory.points)
        min_dist = float('inf')
        closest_point = None
        closest_index = 0

        for i in range(len(points) - 1):
            a = points[i]
            b = points[i + 1]
            proj = self.project_point_to_segment(car_pos, a, b)
            dist = np.linalg.norm(car_pos - proj)
            if dist < min_dist:
                min_dist = dist
                closest_point = proj
                closest_index = i

        return closest_point, closest_index

    def project_point_to_segment(self, p, a, b):
        """Project point p onto segment ab."""
        ap = p - a
        ab = b - a
        ab_squared = np.dot(ab, ab)
        if ab_squared == 0:
            return a
        t = np.clip(np.dot(ap, ab) / ab_squared, 0, 1)
        return a + t * ab

    def find_lookahead_point(self, car_pos, start_idx):
        """Find intersection of lookahead circle with trajectory segment ahead, and ensure it's in front of the car."""
        points = self.trajectory.points
        for i in range(start_idx, len(points) - 1):
            a = np.array(points[i])
            b = np.array(points[i + 1])
            intersections = self.circle_segment_intersections(car_pos, self.lookahead, a, b)
            if not intersections:
                continue

            for pt in intersections:
                # Check if pt is in front of the car (local_x > 0)
                dx = pt[0] - car_pos[0]
                dy = pt[1] - car_pos[1]
                q = self.latest_orientation  # stored from last odometry
                _, _, yaw = self.quaternion_to_euler(q)
                local_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
                if local_x > 0:
                    return pt

        return None

    def circle_segment_intersections(self, center, radius, p1, p2):
        """Return intersection points between a circle and a segment (if any)."""
        d = p2 - p1
        f = p1 - center
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius**2
        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return []  # No intersection

        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)
        points = []
        for t in [t1, t2]:
            if 0 <= t <= 1:
                point = p1 + t * d
                points.append(point)
        return points

    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        return 0.0, 0.0, math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
