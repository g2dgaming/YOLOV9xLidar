import os
import time
import cv2
import ydlidar
from ultralytics import YOLO

# Initialize YOLOv9
yolo_model = YOLO("yolov9.pt")  # Replace with your model's path

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
laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)

if not laser.initialize():
    print("LiDAR initialization failed.")
    exit()

if not laser.turnOn():
    print("Failed to start LiDAR.")
    laser.disconnecting()
    exit()

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

            # Process bounding boxes
            for result in results[0].boxes:
                box = result.xyxy[0]  # Bounding box (x1, y1, x2, y2)
                label = result.cls[0]  # Detected class
                confidence = result.conf[0]  # Confidence score

                # Compute center of bounding box
                x_center = int((box[0] + box[2]) / 2)
                y_center = int((box[1] + box[3]) / 2)

                # Estimate distance from LiDAR data
                # This example assumes lidar points are aligned with frame coordinates
                distance = "N/A"  # Default if mapping fails
                for angle, rng in lidar_points:
                    # Here you need to map LiDAR angle and range to the image coordinates
                    # (angle, range) -> (x, y) on the image, then compare with (x_center, y_center)

                    # Example: (You need an actual mapping function)
                    if some_mapping_condition:  # Replace with actual condition
                        distance = rng
                        break

                # Annotate distance on the bounding box
                cv2.putText(
                    annotated_frame,
                    f"{label}: {confidence:.2f}, Dist: {distance}m",
                    (int(box[0]), int(box[1]) - 10),
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
