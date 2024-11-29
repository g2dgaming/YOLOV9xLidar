import os
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import ydlidar


class LidarCameraCalibration(Node):
    def __init__(self):
        super().__init__('lidar_camera_calibration')
        self.bridge = CvBridge()

        # Subscribe to ROS topics
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.create_subscription(Image, '/camera/image', self.camera_callback, 10)

        self.lidar_points = None
        self.camera_image = None
        self.transformation_matrix = np.eye(4)  # Initial identity matrix

    def lidar_callback(self, msg):
        # Convert LaserScan data to Cartesian coordinates (x, y)
        self.lidar_points = []
        for i, range_value in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            x = range_value * np.cos(angle)
            y = range_value * np.sin(angle)
            self.lidar_points.append([x, y])
        self.lidar_points = np.array(self.lidar_points)

    def camera_callback(self, msg):
        # Convert ROS image to OpenCV format
        if self.lidar_points is not None:
            self.camera_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.process_image()

    def process_image(self):
        if self.camera_image is None or self.lidar_points is None:
            return

        # Apply the current transformation to lidar points
        transformed_points = self.apply_transformation(self.lidar_points)

        # Superimpose lidar points on the camera feed
        for point in transformed_points:
            x, y = point[0], point[1]
            u, v = self.lidar_to_image_coords(x, y)
            if 0 <= u < self.camera_image.shape[1] and 0 <= v < self.camera_image.shape[0]:
                cv2.circle(self.camera_image, (u, v), 2, (0, 0, 255), -1)

        # Display the resulting image
        cv2.imshow('Lidar Overlay', self.camera_image)
        cv2.waitKey(1)

    def lidar_to_image_coords(self, x, y):
        # Convert lidar (x, y) to image pixel coordinates (u, v)
        # Assuming no scaling or distortion, simple transformation for example
        u = int(x * 10 + self.camera_image.shape[1] // 2)  # Basic scaling and centering
        v = int(y * 10 + self.camera_image.shape[0] // 2)
        return u, v

    def apply_transformation(self, points):
        # Apply the 4x4 transformation matrix to lidar points
        transformed_points = []
        for point in points:
            point_homogeneous = np.array([point[0], point[1], 0, 1])  # Homogeneous coordinates
            transformed_point = self.transformation_matrix @ point_homogeneous
            transformed_points.append([transformed_point[0], transformed_point[1]])
        return np.array(transformed_points)

    def calibrate_transformation(self, translation, rotation):
        # Update the transformation matrix with new translation and rotation
        translation_matrix = np.eye(4)
        translation_matrix[0:3, 3] = np.array(translation)

        rotation_matrix = self.rotation_matrix_from_euler_angles(rotation)

        self.transformation_matrix = rotation_matrix @ translation_matrix

    def rotation_matrix_from_euler_angles(self, angles):
        # Rotation matrix for euler angles [roll, pitch, yaw]
        roll, pitch, yaw = angles
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])

        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])

        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])

        return Rx @ Ry @ Rz


def main(args=None):
    rclpy.init(args=args)

    calibration_node = LidarCameraCalibration()

    # Example calibration values (translation in meters, rotation in radians)
    calibration_node.calibrate_transformation([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    rclpy.spin(calibration_node)

    calibration_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
