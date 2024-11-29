import cv2
import json
import math
import threading
from ultralytics import YOLO
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QInputDialog, QDialog, QVBoxLayout, QRadioButton, QPushButton, QWidget
import matplotlib.pyplot as plt

# Load YOLO model (Ensure the correct model file path)
model = YOLO("yolov8n.pt")

# Global variables to store bounding boxes, frame, and all records
frame_copy = None
all_boxes = []
capture_flag = False
all_records = []  # Stores bounding box + Lidar data
lidar_points = []  # Stores Lidar points for plotting
app = QApplication([])  # Initialize PyQt application

# Function to prompt for a name for the record
def prompt_for_name():
    """Prompt the user for a name using a PyQt dialog."""
    dialog = QInputDialog()
    text, ok = dialog.getText(QWidget(), 'Input', 'Enter a name for this record:')
    if ok and text:
        return text
    return None

# Function to request Lidar data (mocked data for now)
def get_lidar_data():
    """Simulate retrieval of Lidar data (angle and distance)."""
    return 45.0, 200.0  # Mocked values (angle in degrees, distance in cm)

# Function to convert polar coordinates to Cartesian (x, y)
def polar_to_cartesian(angle, distance):
    """Convert polar coordinates (angle, distance) to Cartesian coordinates (x, y)."""
    angle_rad = math.radians(angle)
    x = distance * math.cos(angle_rad)
    y = distance * math.sin(angle_rad)
    return x, y

# Function to plot Lidar data in real-time
def plot_lidar_data():
    """Plot Lidar points in real-time."""
    global lidar_points

    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()
    scatter, = ax.plot([], [], 'bo')  # Blue dots for Lidar points
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_title("Lidar Points Visualization")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")

    while True:
        if lidar_points:
            x_vals = [p['x'] for p in lidar_points]
            y_vals = [p['y'] for p in lidar_points]
            scatter.set_data(x_vals, y_vals)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.1)

# Dialog for selecting a bounding box when multiple are detected
class BoundingBoxSelectionDialog(QDialog):
    def __init__(self, boxes, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Select Bounding Box')
        self.layout = QVBoxLayout(self)
        self.selected_box = None
        self.radio_buttons = []

        for i, box in enumerate(boxes):
            radio_button = QRadioButton(f"Box {i + 1}: {box['class']} ({box['confidence']:.2f})", self)
            self.radio_buttons.append(radio_button)
            self.layout.addWidget(radio_button)

        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

    def accept(self):
        for i, radio_button in enumerate(self.radio_buttons):
            if radio_button.isChecked():
                self.selected_box = all_boxes[i]
                break
        super().accept()

    def get_selected_box(self):
        return self.selected_box

# Function to save selected bounding box and Lidar data to JSON
def save_selected_box(box):
    """Save selected bounding box details to JSON file."""
    global lidar_points

    if not box:
        print("No box selected.")
        return

    name = prompt_for_name()
    if not name:
        print("No name provided. Record not saved.")
        return

    record = {
        "name": name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "box": box
    }

    # Get Lidar data and convert to Cartesian coordinates
    angle, distance = get_lidar_data()
    if angle is not None and distance is not None:
        lidar_x, lidar_y = polar_to_cartesian(angle, distance)

        # Add Lidar data to the record
        record["lidar_distance"] = {
            "lidar_x": lidar_x,
            "lidar_y": lidar_y
        }

        # Save Lidar point for plotting
        lidar_points.append({"x": lidar_x, "y": lidar_y})

        # Append the record to the global list
        all_records.append(record)

        # Save to JSON file
        output_file = "selected_boxes.json"
        try:
            with open(output_file, "w") as f:
                json.dump(all_records, f, indent=4)
            print(f"Record saved to {output_file}")
        except Exception as e:
            print(f"Error saving record: {e}")
    else:
        print("Error: Lidar data not received.")

# Capture the frame, perform detection, and select bounding box
def capture_frame():
    """Capture the current frame when 'c' is pressed."""
    global frame_copy, all_boxes, capture_flag

    if not capture_flag:
        return

    # Run YOLO detection on the captured frame
    results = model(frame_copy)
    detections = results[0].boxes

    all_boxes = []
    frame_copy_with_boxes = frame_copy.copy()

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        all_boxes.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class": label
        })

        cv2.rectangle(frame_copy_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame_copy_with_boxes, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame_copy_with_boxes, f"({x1}, {y1}) ({x2}, {y2})",
                    (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Captured Frame", frame_copy_with_boxes)

    if len(all_boxes) > 1:
        dialog = BoundingBoxSelectionDialog(all_boxes)
        dialog.exec_()

        selected_box = dialog.get_selected_box()
        if selected_box:
            save_selected_box(selected_box)
            cv2.destroyWindow("Captured Frame")
        else:
            print("No bounding box selected.")
    else:
        selected_box = all_boxes[0]
        save_selected_box(selected_box)
        cv2.destroyWindow("Captured Frame")

    capture_flag = False

# Main loop
def main():
    global frame_copy, capture_flag

    # Start a thread for plotting Lidar data
    threading.Thread(target=plot_lidar_data, daemon=True).start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    print("Press 'c' to capture a frame, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Check your camera.")
            break

        frame_copy = frame.copy()
        cv2.imshow("Live Feed", frame_copy)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            capture_flag = True
            capture_frame()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
