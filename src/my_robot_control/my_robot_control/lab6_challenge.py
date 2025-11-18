import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import tf_transformations
import math


class SafetyMove(Node):

    def __init__(self):
        super().__init__('safety_move_node')

        # --- ROBOT STATE ---
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_f = 0.0

        # LIDAR state
        self.closest_distance = float('inf')
        self.closest_angle = None

        # --- PARAMETERS ---
        self.declare_parameter('vx', 0.3)
        self.declare_parameter('vy', 0.0)
        self.declare_parameter('w', 0.0)
        self.declare_parameter('td', 5.0)
        self.declare_parameter('min_dist', 0.4)

        self.vx = self.get_parameter('vx').value
        self.vy = self.get_parameter('vy').value
        self.w = self.get_parameter('w').value
        self.td = self.get_parameter('td').value
        self.min_dist = self.get_parameter('min_dist').value

        # --- SUBSCRIPTIONS ---
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # --- PUBLISHER ---
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- TIMER ---
        self.start_time = self.get_clock().now().seconds_nanoseconds()[0]
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("SafetyMove node started.")

    # -----------------------------------------------------------
    # ODOMETRY
    # -----------------------------------------------------------
    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.robot_f = tf_transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )

    # -----------------------------------------------------------
    # LIDAR PROCESSING (distance + angle)
    # -----------------------------------------------------------
    def lidar_callback(self, scan):

        angle_min_deg = scan.angle_min * 180.0 / math.pi
        angle_inc_deg = scan.angle_increment * 180.0 / math.pi

        valid_points = []

        for i, d in enumerate(scan.ranges):
            if not math.isfinite(d) or d <= 0.0:
                continue
            if d < scan.range_min or d > scan.range_max:
                continue

            angle = angle_min_deg + i * angle_inc_deg

            # Convert to ±180 range
            if angle > 180:
                angle -= 360

            valid_points.append((d, angle))

        if not valid_points:
            return

        # Get closest distance and angle
        self.closest_distance, self.closest_angle = min(valid_points, key=lambda x: x[0])

    # -----------------------------------------------------------
    # MAIN CONTROL LOOP
    # -----------------------------------------------------------
    def control_loop(self):
        elapsed = self.get_clock().now().seconds_nanoseconds()[0] - self.start_time

        # STOP IF OBSTACLE IS TOO CLOSE
        if self.closest_distance < self.min_dist:
            self.get_logger().warn(
                f"STOPPING! Obstacle at {self.closest_distance:.2f} m, angle {self.closest_angle:.1f}°"
            )
            self.cmd_pub.publish(Twist())
            self.timer.cancel()
            return

        # STOP IF TIME LIMIT REACHED
        if elapsed > self.td:
            self.get_logger().info("Stopping due to time limit.")
            self.cmd_pub.publish(Twist())
            self.timer.cancel()
            return

        # KEEP MOVING
        vel = Twist()
        vel.linear.x = self.vx
        vel.linear.y = self.vy
        vel.angular.z = self.w
        self.cmd_pub.publish(vel)

        if self.closest_angle is not None:
            self.get_logger().info(
                f"Moving... closest obstacle: {self.closest_distance:.2f} m @ {self.closest_angle:.1f}°"
            )


def main():
    rclpy.init()
    node = SafetyMove()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
