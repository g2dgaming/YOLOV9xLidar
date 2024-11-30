import cv2
import numpy as np
import ydlidar
import time
import json
import os
class LidarCameraCalibration:
    CONFIG_FILE = "calibration_config.json"  # Path to the configuration file

    def __init__(self):
        # Initialize variables for LiDAR and video processing
        self.lidar_points = None
        self.transformation_matrix = np.eye(4)  # 4x4 identity matrix for transformation
        self.scaling_factor=50
        # Initialize OpenCV video capture (camera feed)
        self.cap = cv2.VideoCapture(0)  # 0 for the default camera

        # Initialize translation and rotation values
        self.translation = [0.0, 0.0, 0.0]  # x, y, z
        self.rotation = [0.0, 0.0, 0.0]  # roll, pitch, yaw
        self.load_state()
        # Initialize YDLidar
        self.lidar = self.initialize_lidar()

    def initialize_lidar(self):
        """
        Initialize YDLidar and return the lidar object if successful.
        """
        ydlidar.os_init()  # Initialize YDLidar environment
        ports = ydlidar.lidarPortList()
        port = "/dev/ydlidar"  # Default port

        for key, value in ports.items():
            port = value
            print(f"Detected LiDAR port: {port}")

        laser = ydlidar.CYdLidar()
        laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
        laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 115200)
        laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
        laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
        laser.setlidaropt(ydlidar.LidarPropScanFrequency, 10.0)
        laser.setlidaropt(ydlidar.LidarPropSampleRate, 3)
        laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
        laser.setlidaropt(ydlidar.LidarPropMaxAngle, 100.0)
        laser.setlidaropt(ydlidar.LidarPropMinAngle, -100.0)
        laser.setlidaropt(ydlidar.LidarPropMaxRange, 1.23)
        laser.setlidaropt(ydlidar.LidarPropMinRange, 0.08)
        laser.setlidaropt(ydlidar.LidarPropIntenstiy, False)

        ret = laser.initialize()
        if ret:
            ret = laser.turnOn()
            print("LiDAR turned on successfully!")
            return laser
        else:
            print("Failed to initialize LiDAR")
            return None

    def get_lidar_data(self):
        """
        Get data from the LiDAR and process it into (x, y) coordinates.
        """
        scan = ydlidar.LaserScan()
        if self.lidar.doProcessSimple(scan):
            lidar_points = []
            for i in range(scan.points.size()):
                r = scan.points[i].range
                angle = scan.points[i].angle
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                lidar_points.append([x, y])
            self.lidar_points = np.array(lidar_points)
        else:
            print("Failed to get LiDAR data")

    def apply_transformation(self, points):
        """
        Apply the transformation matrix to the LiDAR points.
        """
        # Convert LiDAR points to homogeneous coordinates (n x 4 matrix)
        points_homogeneous = np.hstack(
            (points, np.zeros((points.shape[0], 1)), np.ones((points.shape[0], 1))))  # Add z=0 and homogeneous=1
        # Apply the transformation matrix to all points
        transformed_points = (
            self.transformation_matrix @ points_homogeneous.T).T  # Transpose for matrix multiplication
        # Return only x, y coordinates
        return transformed_points[:, :2]

    def lidar_to_image_coords(self, x, y, frame_shape):
        """
        Convert LiDAR (x, y) coordinates to camera image pixel coordinates (u, v).
        """
        height, width = frame_shape[:2]
        u = int(x * self.scaling_factor + width // 2)  # Scale and center
        v = int(-y * self.scaling_factor + height // 2)  # Scale and center
        return u, v

    def process_keyboard_input(self, key):
        """
        Process keyboard input to adjust the transformation matrix.
        """
        if key == ord('w'):  # Translate +Z
            self.translation[2] += 0.2
        elif key == ord('s'):  # Translate -Z
            self.translation[2] -= 0.2
        elif key == ord('a'):  # Translate -X
            self.translation[0] -= 0.2
        elif key == ord('d'):  # Translate +X
            self.translation[0] += 0.2
        elif key == ord('q'):  # Translate +Y
            self.translation[1] += 0.2
        elif key == ord('e'):  # Translate -Y
            self.translation[1] -= 0.2
        elif key == ord('i'):  # Rotate +Roll
            self.rotation[0] += 0.05
        elif key == ord('k'):  # Rotate -Roll
            self.rotation[0] -= 0.05
        elif key == ord('j'):  # Rotate +Pitch
            self.rotation[1] += 0.05
        elif key == ord('l'):  # Rotate -Pitch
            self.rotation[1] -= 0.05
        elif key == ord('u'):  # Rotate +Yaw
            self.rotation[2] += 0.05
        elif key == ord('o'):  # Rotate -Yaw
            self.rotation[2] -= 0.05
        elif key == ord('['):  # Scale -
            self.scaling_factor-= 20
        elif key == ord(']'):  # Scale +
            self.scaling_factor += 20

        # Update the transformation matrix
        self.update_transformation_matrix()

    def save_state(self):
        """
        Save the current calibration state to a JSON file.
        """
        print("Translation",self.translation)
        state = {
            "translation": self.translation,
            "rotation": self.rotation,
            "scaling_factor": self.scaling_factor,
            "transformation_matrix": self.transformation_matrix.tolist(),  # Convert to list for JSON
        }
        with open(self.CONFIG_FILE, "w") as file:
            json.dump(state, file, indent=4)
        print(f"Saved state to {self.CONFIG_FILE}")

    def load_state(self):
        """
        Load the calibration state from the JSON file.
        """
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, "r") as file:
                state = json.load(file)
            self.translation = state.get("translation", [0.0, 0.0, 0.0])
            self.rotation = state.get("rotation", [0.0, 0.0, 0.0])
            self.scaling_factor = state.get("scaling_factor", 50)
            self.transformation_matrix = np.array(state.get("transformation_matrix", np.eye(4)))
            print(f"Loaded state from {self.CONFIG_FILE}")
        else:
            print("No saved state found, starting with defaults.")

    def update_transformation_matrix(self):
        """
        Update the transformation matrix based on current translation and rotation.
        """
        translation_matrix = np.eye(4)
        translation_matrix[0:3, 3] = np.array(self.translation)

        rotation_matrix = self.rotation_matrix_from_euler_angles(self.rotation)

        self.transformation_matrix = rotation_matrix @ translation_matrix

        # Save the state after updating the matrix
        self.save_state()
    def rotation_matrix_from_euler_angles(self, angles):
        """
        Create a rotation matrix from Euler angles [roll, pitch, yaw].
        """
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

        rotation_matrix = np.eye(4)
        rotation_matrix[0:3, 0:3] = Rx @ Ry @ Rz
        return rotation_matrix

    def run(self):
        """
        Main loop to process video and overlay LiDAR points.
        """
        while True:
            # Read the camera frame
            ret, frame = self.cap.read()
            if not ret:
                break

            # Get LiDAR data
            self.get_lidar_data()

            # Overlay LiDAR points on the video feed
            if self.lidar_points is not None:
                transformed_points = self.apply_transformation(self.lidar_points)
                for point in transformed_points:
                    x, y = point[0], point[1]
                    u, v = self.lidar_to_image_coords(x, y, frame.shape)
                    if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
                        cv2.circle(frame, (u, v), 2, (255, 0, 0), -1)  # Blue dot for projected points

            # Display the frame with the LiDAR overlay
            cv2.imshow("Lidar Overlay", frame)

            # Process keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('x'):  # Exit on pressing 'x'
                break
            elif key != 255:  # Process other keys
                print("other key")
                self.process_keyboard_input(key)

        # Release the video capture and close windows
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    lidar_camera_calibration = LidarCameraCalibration()
    lidar_camera_calibration.run()
