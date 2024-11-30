import numpy as np
import os
import time
import cv2
import ydlidar
from ultralytics import YOLO
import json


# Load transformation matrix from calibration file
with open('calibrations/calibration_config.json', 'r') as f:
    calibration_data = json.load(f)

transformation_matrix = np.array(calibration_data['transformation_matrix'])

# Initialize YOLOv9
yolo_model = YOLO("yolov9t.pt")  # Replace with your model's path

# Suppress YOLO inference logs
yolo_model.overrides["verbose"] = False

# Initialize LiDAR
ydlidar.os_init()
port = "/dev/ttyUSB0"
laser = ydlidar.CYdLidar()
laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 115200)
laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
laser.setlidaropt(ydlidar.LidarPropScanFrequency, 10.0)
laser.setlidaropt(ydlidar.LidarPropSampleRate, 3)
laser.setlidaropt(ydlidar.LidarPropMaxAngle, 90.0)
laser.setlidaropt(ydlidar.LidarPropMinAngle, -90.0)
laser.setlidaropt(ydlidar.LidarPropMaxRange, 32.0)
laser.setlidaropt(ydlidar.LidarPropMinRange, 0.1)
laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)

if not laser.initialize():
    print("LiDAR initialization failed.")
    exit()

if not laser.turnOn():
    print("Failed to start LiDAR.")
    laser.disconnecting()
    exit()


# Function to apply a 4x4 transformation matrix to the LiDAR data
def apply_transformation_matrix(lidar_points, transformation_matrix):
    points_homogeneous = np.hstack(
        (lidar_points, np.zeros((lidar_points.shape[0], 1)), np.ones((lidar_points.shape[0], 1))))  # Add z=0 and homogeneous=1
    # Apply the transformation matrix to all points
    transformed_points = (
            transformation_matrix @ points_homogeneous.T).T  # Transpose for matrix multiplication
    # Return only x, y coordinates
    return transformed_points[:, :2]


# Function to map LiDAR data to image coordinates
def lidar_to_image_coordinates(lidar_points, frame_shape):
    """
    Map LiDAR data to image coordinates.

    :param lidar_points: List of tuples [(x, y, distance), ...]
    :param image_width: Width of the image in pixels
    :param image_height: Height of the image in pixels
    :return: List of tuples [(x, y, distance), ...]
    """
    mapped_points = []
    for x, y in lidar_points:
        scaling_factor = calibration_data['scaling_factor']
        height, width = frame_shape[:2]
        u = int(x * scaling_factor + width // 2)  # Scale and center
        v = int(-y * scaling_factor + height // 2)  # Scale and center

        mapped_points.append((u, v))

    return mapped_points


# Open the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Unable to access the camera.")
    laser.turnOff()
    laser.disconnecting()
    exit()

try:
    while True:
        # Capture a frame from the camera
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Perform YOLOv9 detection
        results = yolo_model.predict(source=frame, conf=0.5, show=False)
        annotated_frame = results[0].plot()  # Get annotated frame

        # Get LiDAR scan data
        scan = ydlidar.LaserScan()
        if laser.doProcessSimple(scan):
            lidar_points = []
            copy_points = []  # Store ranges separately
            for i in range(scan.points.size()):
                r = scan.points[i].range
                angle = scan.points[i].angle
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                lidar_points.append([x, y])
                copy_points.append(r)  # Store the range value

            lidar_points = np.array(lidar_points)
            # Apply transformation to LiDAR points
            transformed_points = apply_transformation_matrix(lidar_points, transformation_matrix)

            # Map transformed LiDAR points to image coordinates
            image_width = frame.shape[1]
            image_height = frame.shape[0]
            mapped_points = lidar_to_image_coordinates(transformed_points, frame.shape)

            # Process bounding boxes
            for result in results[0].boxes:
                box = result.xyxy[0]  # Bounding box (x1, y1, x2, y2)
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                # Calculate the center of the bounding box
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2

                # Variables to track the closest point and its distance to the box center
                closest_point = None
                closest_distance = float('inf')  # Start with an infinitely large distance
                closest_rng = None  # To store the range of the closest point

                for idx, (x_pixel, y_pixel) in enumerate(mapped_points):
                    # Calculate the range from x and y (in image space)
                    # No need for rng calculation here, we'll use copy_points
                    rng = copy_points[idx]  # Retrieve the original range from copy_points

                    # Calculate the distance from the point to the center of the bounding box
                    distance_to_center = np.sqrt((x_pixel - box_center_x) ** 2 + (y_pixel - box_center_y) ** 2)

                    # Check if this point is closer to the center
                    if distance_to_center < closest_distance:
                        closest_distance = distance_to_center
                        closest_point = (x_pixel, y_pixel)  # Store the pixel location
                        closest_rng = rng  # Store the range of the closest point

                # If we found a closest point, display it and the range
                if closest_point:
                    print(
                        f"Closest point to the box center: X: {closest_point[0]}, Y: {closest_point[1]}, Range: {closest_rng:.2f}m")
                    print(f"Distance to center: {closest_distance:.2f}m")
                    cv2.circle(annotated_frame, (closest_point[0], closest_point[1]), 5, (0, 255, 0), -1)  # Green dot
                    # Draw a circle for the closest point and display the range
                    cv2.putText(annotated_frame, f"{closest_rng:.2f}m", (x1,y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display the frame with annotated bounding boxes and distances
            cv2.imshow("YOLOv9 + LiDAR", annotated_frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted!")

# Cleanup
camera.release()
cv2.destroyAllWindows()
laser.turnOff()
laser.disconnecting()
