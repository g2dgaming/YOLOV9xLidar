import os
import time
import cv2
import ydlidar
from ultralytics import YOLO

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


# Function to map LiDAR data to image coordinates
def lidar_to_image_coordinates(lidar_points, image_width, image_height):
    """
    Map LiDAR data to image coordinates.

    :param lidar_points: List of tuples [(angle, range), ...]
    :param image_width: Width of the image in pixels
    :param image_height: Height of the image in pixels
    :return: List of tuples [(x, y, distance), ...]
    """
    mapped_points = []

    for angle, rng in lidar_points:
        if rng > 0:  # Exclude invalid ranges
            # Print raw LiDAR data for debugging
            #fpriprint(f"LiDAR Point - Angle: {angle:.2f}, Range: {rng:.2f}")

            # Normalize angle to [0, 1] (assuming a range of -180 to +180 degrees)
            normalized_angle = (angle + 180) / 360.0

            # Map normalized angle to horizontal pixel position
            x = int(normalized_angle * image_width)

            # Assume a fixed vertical position for simplicity (e.g., center of the image)
            y = image_height // 2

            # Append the mapped point
            mapped_points.append((x, y, rng))
            #print(f"Mapped to Image - X: {x}, Y: {y}, Distance: {rng:.2f}m")

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
            lidar_points = [(p.angle, p.range) for p in scan.points]

            # Map LiDAR data to image coordinates
            image_width = frame.shape[1]
            image_height = frame.shape[0]
            mapped_points = lidar_to_image_coordinates(lidar_points, image_width, image_height)

            # Process bounding boxes
            for result in results[0].boxes:
                box = result.xyxy[0]  # Bounding box (x1, y1, x2, y2)
                label = result.cls[0]  # Detected class
                confidence = result.conf[0]  # Confidence score
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                # Log bounding box details
                #print(f"Bounding Box - X1: {x1}, Y1: {y1}, X2: {x2}, Y2: {y2}, Label: {label}")

                # Find closest LiDAR point within the bounding box
                distances = []
                for x, y, rng in mapped_points:
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        distances.append(rng)
                        print(f"Point within Box - X: {x}, Y: {y}, Distance: {rng:.2f}m")

                # Determine the minimum distance
                min_distance = min(distances) if distances else "N/A"

                # Annotate distance on the bounding box
                cv2.putText(
                    annotated_frame,
                    f"Dis:{min_distance}m",
                    (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

        # Display the frame
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

