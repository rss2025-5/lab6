import rclpy
from rclpy.node import Node
from queue import PriorityQueue
from skimage.morphology import disk, dilation
import numpy as np
import heapq
from math import sqrt
import time
from std_msgs.msg import Float32
from queue import PriorityQueue
import numpy as np
import heapq
from math import sqrt
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory

class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
     current car pose. Creates a continuous trajectory connecting all sequential goals.
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

        # Initialize current position and goal queue
        self.current_position = None
        self.goal_queue = []
        self.is_planning = False

        # Flag to track if this is the first goal in a sequence
        self.first_goal = True

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

        # Create a timer to process the goal queue
        self.create_timer(1.0, self.process_goal_queue)

        self.get_logger().info("Path planner initialized with continuous multi-goal support")

    def map_cb(self, msg):
        init_grid = np.array(msg.data)
        init_grid = init_grid.reshape((msg.info.height, msg.info.width))
        self.occupancy_grid = dilation(init_grid, disk(12))
        self.resolution = msg.info.resolution
        self.origin = msg.info.origin
        self.get_logger().info("Map loaded")

    def pose_cb(self, pose):
        self.current_position = self.real_to_pixel(
            pose.pose.pose.position.x,
            pose.pose.pose.position.y
        )
        self.get_logger().info(f"Current position updated: {self.current_position}")

        # Process goals if we have them and weren't waiting for the position
        if self.goal_queue and self.current_position and not self.is_planning:
            self.process_goal_queue()

    def goal_cb(self, msg):
        goal_pixel = self.real_to_pixel(msg.pose.position.x, msg.pose.position.y)
        self.goal_queue.append(goal_pixel)
        self.get_logger().info(f"New goal added to queue: {goal_pixel}")

        # Store goal in real coordinates too for trajectory visualization
        goal_real = (msg.pose.position.x, msg.pose.position.y)

        # If this is a follow-up goal and we already have a trajectory
        if not self.first_goal and self.trajectory.points and not self.is_planning:
            # We can directly extend the trajectory without replanning the whole path
            self.get_logger().info(f"Directly extending trajectory to new goal")

        # Try to process immediately if we have position and not currently planning
        if self.current_position and not self.is_planning:
            self.process_goal_queue()

    def process_goal_queue(self):
        """Process goals in the queue sequentially, creating one continuous trajectory"""
        if not self.goal_queue or not self.current_position or self.occupancy_grid is None or self.is_planning:
            return

        self.is_planning = True

        # Store all paths first to ensure smooth trajectory
        complete_paths = []
        start_position = self.current_position
        remaining_goals = list(self.goal_queue)  # Make a copy to process all at once

        # Plan paths for all goals in queue
        while remaining_goals:
            next_goal = remaining_goals[0]
            self.get_logger().info(f"Planning path from {start_position} to {next_goal}")

            path = self.jump_point_search(start_position, next_goal)
            if path is None:
                self.get_logger().info(f"Path not found to {next_goal}")
                # Skip this goal and try the next one
                remaining_goals.pop(0)
                self.goal_queue.remove(next_goal)
            else:
                complete_paths.append(path)
                # Update starting position for next path segment
                start_position = next_goal
                remaining_goals.pop(0)

        # Now rebuild the trajectory with all planned paths
        if complete_paths:
            # Clear trajectory only if this is the first goal in a new sequence
            if self.first_goal:
                self.trajectory.clear()
                self.first_goal = False

            for path in complete_paths:
                # For each path, add all points except the first one (to avoid duplication)
                # Unless it's the very first path segment
                start_idx = 1 if path != complete_paths[0] else 0

                for i in range(start_idx, len(path)):
                    x, y = self.pixel_to_real(path[i][0], path[i][1])
                    self.trajectory.addPoint([x, y])

            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()

            # Update current position to the last goal reached
            if self.goal_queue:
                self.current_position = self.goal_queue[-1]
                self.get_logger().info(f"Continuous path planned successfully for {len(complete_paths)} goals")
                # Clear processed goals
                self.goal_queue.clear()

        self.is_planning = False

        # Reset first_goal flag when no more paths were added
        if not complete_paths and not self.goal_queue:
            self.first_goal = True

    def plan_path(self, start_point, end_point):
        """Plan a path and return it"""
        start_time = time.time()
        path = self.jump_point_search(start_point, end_point)
        end_time = time.time()
        computation_time = end_time - start_time
        self.get_logger().info(f"Path planning time: {computation_time}")
        return path

    def is_jump_point(self, pos, direction):
        x, y = pos
        dx, dy = direction
        if dx != 0 and dy != 0: # Diagonal
            if not self.is_free((x - dx, y)) and self.is_free((x - dx, y + dy)):
                return True
            if not self.is_free((x, y - dy)) and self.is_free((x + dx, y - dy)):
                return True
        elif dx != 0: # Horizontal
            if not self.is_free((x, y + 1)) and self.is_free((x + dx, y + 1)):
                return True
            if not self.is_free((x, y - 1)) and self.is_free((x + dx, y - 1)):
                return True
        elif dy != 0: # Vertical
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

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for succ in self.successors(current, parent, goal):
                if succ not in visited:
                    g = cost + self.heuristic(current, succ)
                    f = g + self.heuristic(succ, goal)
                    heapq.heappush(open_list, (f, g, succ, current))
        return None

    def a_star(self, start, goal):
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        while frontier:
            _, current = heapq.heappop(frontier)

            if current == goal:
                break

            for next in self.get_neighbors(current):
                new_cost = cost_so_far[current] + self.cost(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    heapq.heappush(frontier, (priority, next))
                    came_from[next] = current

        return self.reconstruct_path(came_from, goal)

    def is_free(self, pos):
        x, y = pos
        if 0 <= x < self.occupancy_grid.shape[1] and 0 <= y < self.occupancy_grid.shape[0]:
            return self.occupancy_grid[y, x] == 0
        return False

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

            if dx != 0 and dy != 0: # Diagonal move
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

            elif dx != 0: # Horizontal move
                if self.is_free((x + dx, y)):
                    directions.append((dx, 0))
                # Forced neighbors (up/down diagonals if adjacent blocked)
                if not self.is_free((x, y + 1)) and self.is_free((x + dx, y + 1)):
                    directions.append((dx, 1))
                if not self.is_free((x, y - 1)) and self.is_free((x + dx, y - 1)):
                    directions.append((dx, -1))

            elif dy != 0: # Vertical move
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
            [s, c, ty],
            [0, 0, 1]
        ])

    def pixel_to_real(self, u, v):
        # Step 1: scale
        resolution = self.resolution
        scaled_x = u * resolution
        scaled_y = v * resolution
        pixel_hom = np.array([scaled_x, scaled_y, 1.0]) # homogeneous

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

    def get_neighbors(self, current):
        x, y = current
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.is_free((nx, ny)):
                    neighbors.append((nx, ny))
        return neighbors

    def cost(self, a, b):
        # Use Euclidean distance for cost
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return sqrt(dx**2 + dy**2)

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
