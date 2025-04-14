import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory

from tf_transformations import euler_from_quaternion
import numpy as np


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        # got:
        self.goal_pose_stamped = PoseStamped()
        self.map_grid = OccupancyGrid()
        self.init_pose = PoseWithCovarianceStamped()
        self.resolution = 1.0
        self.origin_orientation = 0 # should be Quaternion
        self.origin_position = 0 # should be Point


    def map_cb(self, msg):
        self.map_grid = msg
        data = msg.data


        self.resolution = msg.info.resolution
        self.origin_orientation = msg.info.origin.orientation
        self.origin_position = msg.info.origin.position

        self.get_logger().info(f"map_info:{msg.info}")

    def pose_cb(self, pose):
        self.init_pose = pose

    def goal_cb(self, msg):
        self.goal_pose_stamped = msg


        self.plan_path(self.init_pose, self.goal_pose_stamped, self.map_grid)

    def plan_path(self, start_point, end_point, map):
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


    def create_transform_matrix_2d(self, theta, tx, ty):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [c, -s, tx],
            [s,  c, ty],
            [0,  0, 1]
        ])

    def pixel_to_real(self, u, v, msg):
        # Step 1: scale
        resolution = msg.info.resolution
        scaled_x = u * resolution
        scaled_y = v * resolution
        pixel_hom = np.array([scaled_x, scaled_y, 1.0])  # homogeneous

        # Step 2: rotation and translation
        q = msg.info.origin.orientation
        _, _, theta = euler_from_quaternion((q.x, q.y, q.z, q.w))

        tx = msg.info.origin.position.x
        ty = msg.info.origin.position.y

        T = self.create_transform_matrix_2d(theta, tx, ty)

        world = T @ pixel_hom
        return world[0], world[1]

    def real_to_pixel(self, x, y, msg):
        # Step 1: get inverse transform matrix
        q = msg.info.origin.orientation
        _, _, theta = euler_from_quaternion((q.x, q.y, q.z, q.w))
        tx = msg.info.origin.position.x
        ty = msg.info.origin.position.y

        T = self.create_transform_matrix_2d(theta, tx, ty)
        T_inv = np.linalg.inv(T)

        # Step 2: apply inverse transform
        world_hom = np.array([x, y, 1.0])
        pixel_hom = T_inv @ world_hom

        # Step 3: scale down to pixel coords
        resolution = msg.info.resolution
        u = pixel_hom[0] / resolution
        v = pixel_hom[1] / resolution
        return int(round(u)), int(round(v))


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
