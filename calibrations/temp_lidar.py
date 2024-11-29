import matplotlib.pyplot as plt
import json
import math
from datetime import datetime

# List to store all Lidar points
lidar_points = []

# Function to save Lidar points to JSON
def save_lidar_points_to_json():
    """Save all collected Lidar points to a JSON file."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"lidar_points_{timestamp}.json"
    try:
        with open(output_file, "w") as f:
            json.dump(lidar_points, f, indent=4)
        print(f"Lidar points saved to {output_file}")
    except Exception as e:
        print(f"Error saving Lidar points: {e}")

# Function to plot Lidar points in real time
def plot_lidar_data():
    """Plot all Lidar points in real time."""
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title("Real-Time Lidar Points")
    ax.set_rlim(0, 500)  # Adjust based on your Lidar's range

    while True:
        # Clear the current plot
        ax.clear()

        # Plot all Lidar points
        for point in lidar_points:
            angle_rad = math.radians(point["angle"])
            ax.scatter(angle_rad, point["distance"], c='blue', alpha=0.7)

        plt.pause(0.1)  # Pause for a short time to update the plot

# Function to get Lidar data and update plot
def handle_lidar_data():
    """Get Lidar data, update the plot, and save the points."""
    global lidar_points

    # Call get_lidar_data (Simulated or Real Lidar Data)
    angle, distance = get_lidar_data()

    # Add the Lidar point to the list
    lidar_points.append({
        "angle": angle,
        "distance": distance,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    print(f"New Lidar point added: Angle = {angle}, Distance = {distance}")

    # Save points to JSON
    save_lidar_points_to_json()

    # Plot Lidar points in real time
    plot_lidar_data()

# Simulated function to return Lidar data (Replace with actual data fetching)
def get_lidar_data():
    """Simulate fetching Lidar data."""
    import random
    angle = random.uniform(0, 360)  # Random angle in degrees
    distance = random.uniform(50, 500)  # Random distance (example range: 50-500 units)
    return angle, distance

# Example usage
if __name__ == "__main__":
    while True:
        handle_lidar_data()
