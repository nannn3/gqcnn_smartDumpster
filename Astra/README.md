# Astra Camera Control Key Bindings

This section provides a detailed explanation of the functionality associated with each key press in the Astra camera application. Use these keys to interact with the camera and visualize the data in various forms.

## Key Functions

- **`q`**: **Quit the Application**
  - Exits the camera application safely. Ensures all camera streams are properly stopped and all application windows are closed.

- **`p`**: **Visualize Point Cloud**
  - Generates a 3D point cloud from the current depth data captured by the camera and displays it using Open3D visualization tools. This feature is essential for understanding the spatial arrangement of the environment detected by the camera.

- **`r`**: **Rotate and Visualize Point Cloud**
  - Applies a predefined rotation matrix to the current depth data's point cloud, then visualizes this rotated point cloud. Useful for examining the point cloud from various perspectives.

- **`c`**: **Capture and Save Color Image**
  - Captures the current frame from the camera's color stream and saves it as a PNG file. The images are saved in a directory named `Calibration_Pics`, located directly inside the parent directory of your project. Each image file is named using an incrementing counter, formatted as `image{counter}.png`, where `{counter}` is a unique number for each image, starting from 1.

- **`v`**: **View Depth from Point Cloud**
  - Converts the point cloud back to a depth map, applies a colormap for enhanced visualization, and displays this depth map. This function is helpful for verifying depth accuracy and the integrity of point cloud data transformations.
