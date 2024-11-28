import cv2
import json
from ultralytics import YOLO
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QInputDialog, QDialog, QVBoxLayout, QRadioButton, QPushButton, QWidget

# Load YOLO model (Ensure the correct model file path)
model = YOLO("yolov8n.pt")

# Global variables to store bounding boxes and the captured frame
frame_copy = None
all_boxes = []
capture_flag = False
app = QApplication([])  # Initialize PyQt application

# Global variable for storing all captured boxes in one list
all_records = []

def prompt_for_name():
    """Prompt the user for a name using a PyQt dialog."""
    dialog = QInputDialog()
    text, ok = dialog.getText(QWidget(), 'Input', 'Enter a name for this record:')
    if ok and text:
        return text
    return None

class BoundingBoxSelectionDialog(QDialog):
    """Dialog for selecting a bounding box when multiple are detected."""
    def __init__(self, boxes, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Select Bounding Box')

        # Layout for radio buttons
        self.layout = QVBoxLayout(self)

        self.selected_box = None
        self.radio_buttons = []

        # Create radio buttons for each bounding box
        for i, box in enumerate(boxes):
            radio_button = QRadioButton(f"Box {i + 1}: {box['class']} ({box['confidence']:.2f})", self)
            self.radio_buttons.append(radio_button)
            self.layout.addWidget(radio_button)

        # Add 'OK' button
        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

    def accept(self):
        """Find selected box and accept the dialog."""
        for i, radio_button in enumerate(self.radio_buttons):
            if radio_button.isChecked():
                self.selected_box = all_boxes[i]
                break
        super().accept()

    def get_selected_box(self):
        return self.selected_box


def save_selected_box(box):
    """Save selected bounding box details to a JSON file."""
    if not box:
        print("No box selected.")
        return

    # Ask for a name for the record using the PyQt5 modal
    name = prompt_for_name()
    if not name:
        print("No name provided. Record not saved.")
        return

    # Create the record with bounding box data
    record = {
        "name": name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "box": box
    }

    # Append the record to the global list
    all_records.append(record)

    # Save the records to a JSON file (overwriting it with the full list)
    output_file = "selected_boxes.json"
    try:
        with open(output_file, "w") as f:
            json.dump(all_records, f, indent=4)
        print(f"Record saved to {output_file}")
    except Exception as e:
        print(f"Error saving record: {e}")


def capture_frame():
    """Capture the current frame when 'c' is pressed."""
    global frame_copy, all_boxes, capture_flag

    if not capture_flag:
        return

    # Run YOLO detection on the captured frame
    results = model(frame_copy)
    detections = results[0].boxes

    # Extract bounding boxes and store the information
    all_boxes = []
    frame_copy_with_boxes = frame_copy.copy()
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
        conf = float(box.conf[0])  # Confidence score
        cls = int(box.cls[0])  # Class index
        label = model.names[cls]  # Get the class label

        # Add box information to all_boxes list
        all_boxes.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class": label
        })

        # Draw bounding box and label on the frame
        cv2.rectangle(frame_copy_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame_copy_with_boxes, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Display coordinates on the frame
        cv2.putText(frame_copy_with_boxes, f"({x1}, {y1}) ({x2}, {y2})",
                    (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Show the frame with bounding boxes and coordinates
    cv2.imshow("Captured Frame", frame_copy_with_boxes)

    # If multiple bounding boxes are detected, ask the user to select one
    if len(all_boxes) > 1:
        dialog = BoundingBoxSelectionDialog(all_boxes)
        dialog.exec_()

        # Get selected bounding box
        selected_box = dialog.get_selected_box()
        if selected_box:
            x1, y1, x2, y2 = selected_box['bbox']
            cv2.rectangle(frame_copy_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame_copy_with_boxes, "Selected", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display selected box and save the data
            cv2.imshow("Selected Bounding Box", frame_copy_with_boxes)
            save_selected_box(selected_box)
            cv2.destroyWindow("Captured Frame")
            # Close the "Selected Bounding Box" window
            cv2.destroyWindow("Selected Bounding Box")
        else:
            print("No bounding box selected.")
    else:
        # If only one bounding box, automatically select it
        selected_box = all_boxes[0]
        save_selected_box(selected_box)

        # Close the "Captured Frame" window
        cv2.destroyWindow("Captured Frame")

        # Close the "Selected Bounding Box" window
        cv2.destroyWindow("Selected Bounding Box")

    # Reset capture flag to allow next capture without closing feed
    capture_flag = False


def main():
    global frame_copy, capture_flag

    # Open the camera
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

        # Display the live feed
        cv2.imshow("Live Feed", frame_copy)

        # Wait for user input to capture or quit
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Quit if 'q' is pressed
            break
        elif key == ord('c'):  # Capture if 'c' is pressed
            capture_flag = True
            capture_frame()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
