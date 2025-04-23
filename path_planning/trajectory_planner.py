import rclpy
from rclpy.node import Node
from queue import PriorityQueue
from skimage.morphology import disk, dilation
import numpy as np
import heapq
from math import sqrt
import time
assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory


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

        self.occupancy_grid = None
        self.resolution = 1.0
        self.origin = None

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

    def map_cb(self, msg):

        init_grid = np.array(msg.data)
        init_grid = init_grid.reshape((msg.info.height, msg.info.width))
        self.occupancy_grid = dilation(init_grid, disk(12))

        self.resolution = msg.info.resolution
        self.origin = msg.info.origin

        self.get_logger().info("Map loaded")


    def pose_cb(self, pose):
        s = pose
        self.start = self.real_to_pixel(s.pose.pose.position.x, s.pose.pose.position.y)
        self.get_logger().info(f"start: {self.start}")

    def goal_cb(self, msg):
        self.goal = self.real_to_pixel(msg.pose.position.x, msg.pose.position.y)
        self.plan_path(self.start, self.goal)


    def plan_path(self, start_point, end_point):
        start_time = time.time()
        path = self.jump_point_search(start_point, end_point)
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        self.get_logger().info(f"path plan time:{computation_time}")
        


        if path is None:
            self.get_logger().info("Path not found")
        else:
            for point in path:
                x, y = self.pixel_to_real(point[0], point[1])
                self.trajectory.addPoint([x, y])
            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()

    def is_free(self, pos):
        x, y = pos
        if 0 <= x < self.occupancy_grid.shape[1] and 0 <= y < self.occupancy_grid.shape[0]:
            return self.occupancy_grid[y, x] == 0
        return False

    def is_jump_point(self, pos, direction):
        x, y = pos
        dx, dy = direction

        if dx != 0 and dy != 0:  # Diagonal
            if not self.is_free((x - dx, y)) and self.is_free((x - dx, y + dy)):
                return True
            if not self.is_free((x, y - dy)) and self.is_free((x + dx, y - dy)):
                return True
        elif dx != 0:  # Horizontal
            if not self.is_free((x, y + 1)) and self.is_free((x + dx, y + 1)):
                return True
            if not self.is_free((x, y - 1)) and self.is_free((x + dx, y - 1)):
                return True
        elif dy != 0:  # Vertical
            if not self.is_free((x + 1, y)) and self.is_free((x + 1, y + dy)):
                return True
            if not self.is_free((x - 1, y)) and self.is_free((x - 1, y + dy)):
                return True
        return False

    def heuristic(self, a, b):
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return sqrt(dx * dx + dy * dy)

    def jump(self, current, direction, goal, depth=0, max_depth=700):
        x, y = current

        if depth > max_depth:
            return (x,y)


        dx, dy = direction
        nx, ny = x + dx, y + dy

        if not self.is_free((nx, ny)):
            return None
        if (nx, ny) == goal:
            return (nx, ny)

        if self.is_jump_point((nx, ny), (dx, dy)):
            return (nx, ny)

        # Check recursive jumps from diagonal
        if dx != 0 and dy != 0:
            if self.jump((nx, ny), (dx, 0), goal, depth + 1) is not None:
                return (nx, ny)
            if self.jump((nx, ny), (0, dy), goal, depth + 1) is not None:
                return (nx, ny)

        return self.jump((nx, ny), (dx, dy), goal, depth + 1)

    # def jump(self, current, direction, goal, depth=0, max_depth=50):

    #     x, y = current
    #     dx, dy = direction
    #     nx, ny = x + dx, y + dy

    #     if depth > max_depth:
    #         self.get_logger().warn("Max recursion depth reached")
    #         return (x, y)

    #     if not self.is_free((nx, ny)):
    #         return None

    #     if (nx, ny) == goal:
    #         return (nx, ny)

    #     # Forced neighbor check
    #     if dx != 0 and dy != 0:  # Diagonal
    #         if (self.is_free((nx - dx, ny + dy)) and not self.is_free((nx - dx, ny))) or \
    #         (self.is_free((nx + dx, ny - dy)) and not self.is_free((nx, ny - dy))):
    #             return (nx, ny)

    #         if self.jump((nx, ny), (dx, 0), goal, depth + 1, max_depth) is not None or \
    #         self.jump((nx, ny), (0, dy), goal, depth + 1, max_depth) is not None:
    #             return (nx, ny)

    #     elif dx != 0:  # Horizontal
    #         if (self.is_free((nx, ny + 1)) and not self.is_free((x, y + 1))) or \
    #         (self.is_free((nx, ny - 1)) and not self.is_free((x, y - 1))):
    #             return (nx, ny)

    #     elif dy != 0:  # Vertical
    #         if (self.is_free((nx + 1, ny)) and not self.is_free((x + 1, y))) or \
    #         (self.is_free((nx - 1, ny)) and not self.is_free((x - 1, y))):
    #             return (nx, ny)

    #     # Ensure we don’t recurse to the same position
    #     if (nx, ny) == (x, y):
    #         return None

    #     return self.jump((nx, ny), (dx, dy), goal, depth + 1, max_depth)

    def successors(self, current, parent, goal):
        x, y = current

        directions = []
        if parent is None:
            # No pruning if no parent: first expansion
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        directions.append((dx, dy))
        else:
            px, py = parent
            dx = int((x - px) / max(abs(x - px), 1))
            dy = int((y - py) / max(abs(y - py), 1))

            # Always include the direction we came from
            directions.append((dx, dy))

            if dx != 0 and dy != 0:  # Diagonal move
                # Add natural neighbors
                if self.is_free((x + dx, y)):
                    directions.append((dx, 0))
                if self.is_free((x, y + dy)):
                    directions.append((0, dy))

                # Check for forced neighbors
                if not self.is_free((x - dx, y)) and self.is_free((x - dx, y + dy)):
                    directions.append((-dx, dy))
                if not self.is_free((x, y - dy)) and self.is_free((x + dx, y - dy)):
                    directions.append((dx, -dy))

            elif dx != 0:  # Horizontal move
                if self.is_free((x + dx, y)):
                    directions.append((dx, 0))
                # Forced neighbors (up/down diagonals if adjacent blocked)
                if not self.is_free((x, y + 1)) and self.is_free((x + dx, y + 1)):
                    directions.append((dx, 1))
                if not self.is_free((x, y - 1)) and self.is_free((x + dx, y - 1)):
                    directions.append((dx, -1))

            elif dy != 0:  # Vertical move
                if self.is_free((x, y + dy)):
                    directions.append((0, dy))
                # Forced neighbors (left/right diagonals if adjacent blocked)
                if not self.is_free((x + 1, y)) and self.is_free((x + 1, y + dy)):
                    directions.append((1, dy))
                if not self.is_free((x - 1, y)) and self.is_free((x - 1, y + dy)):
                    directions.append((-1, dy))

        # Try jumps only in pruned directions
        for d in directions:
            jp = self.jump(current, d, goal)
            if jp:
                yield jp

    # def successors(self, current, parent, goal):
    # # Temporarily ignore parent-based pruning for correctness
    #     for dx in [-1, 0, 1]:
    #         for dy in [-1, 0, 1]:
    #             if dx == 0 and dy == 0:
    #                 continue
    #             jp = self.jump(current, (dx, dy), goal)
    #             if jp:
    #                 yield jp

    def jump_point_search(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0 + self.heuristic(start, goal), 0, start, None))
        visited = set()
        came_from = {}

        while open_list:
            f, cost, current, parent = heapq.heappop(open_list)
            if current in visited:
                continue
            visited.add(current)
            came_from[current] = parent
           #  self.get_logger().info(f"cur: {current}")

            if current == goal:
                return self.reconstruct_path(came_from, current)
            for succ in self.successors(current, parent, goal):
                if succ not in visited:
                    g = cost + self.heuristic(current, succ)
                    f = g + self.heuristic(succ, goal)
                    heapq.heappush(open_list, (f, g, succ, current))
        return None

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from and came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def create_transform_matrix_2d(self, theta, tx, ty):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [c, -s, tx],
            [s,  c, ty],
            [0,  0, 1]
        ])

    def pixel_to_real(self, u, v):
        # Step 1: scale
        resolution = self.resolution
        scaled_x = u * resolution
        scaled_y = v * resolution
        pixel_hom = np.array([scaled_x, scaled_y, 1.0])  # homogeneous

        # Step 2: rotation and translation
        q = self.origin.orientation
        _, _, theta = self.euler_from_quaternion(q)

        tx = self.origin.position.x
        ty = self.origin.position.y

        T = self.create_transform_matrix_2d(theta, tx, ty)

        world = T @ pixel_hom
        return world[0], world[1]

    def real_to_pixel(self, x, y):
        # Step 1: get inverse transform matrix
        q = self.origin.orientation
        _, _, theta = self.euler_from_quaternion(q)
        tx = self.origin.position.x
        ty = self.origin.position.y

        T = self.create_transform_matrix_2d(theta, tx, ty)
        T_inv = np.linalg.inv(T)

        # Step 2: apply inverse transform
        world_hom = np.array([x, y, 1.0])
        pixel_hom = T_inv @ world_hom

        # Step 3: scale down to pixel coords
        resolution = self.resolution
        u = pixel_hom[0] / resolution
        v = pixel_hom[1] / resolution
        return int(round(u)), int(round(v))

    def euler_from_quaternion(self, quaternion):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quaternion = [x, y, z, w]
        Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
        """
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


    # def a_star(self, start, goal):
    #     frontier = PriorityQueue()
    #     frontier.put(start, 0)
    #     came_from = dict()
    #     cost_so_far = dict()
    #     came_from[start] = None
    #     cost_so_far[start] = 0

    #     while not frontier.empty():
    #         current = frontier.get()

    #         if current == goal:
    #             break

    #         for next in graph.neighbors(current):
    #             new_cost = cost_so_far[current] + graph.cost(current, next)
    #             if next not in cost_so_far or new_cost < cost_so_far[next]:
    #                 cost_so_far[next] = new_cost
    #                 priority = new_cost + heuristic(goal, next)
    #                 frontier.put(next, priority)
    #                 came_from[next] = current


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
