# SparseDrive ROS2 Visualization

This document describes the visualization tools provided for the `sparsedrive_ros2` package.

## BEV Image Visualizer

The BEV (Bird's Eye View) image visualizer is a Python-based ROS2 node that subscribes to the output of the `sparsedrive_node` and generates a 2D image representation of the scene.

### Features

-   Visualizes 3D object detections as green circles.
-   Displays predicted trajectories for objects as blue lines.
-   Shows the planned ego-path as a red line.
-   Publishes the visualization as a `sensor_msgs/msg/Image` on the `/bev_image` topic.

### How to Run

1.  **Build the package:**
    ```bash
    colcon build --packages-select sparsedrive_ros2
    ```
2.  **Source the workspace:**
    ```bash
    source install/setup.bash
    ```
3.  **Launch the visualizer:**
    ```bash
    ros2 run sparsedrive_ros2 bev_visualizer
    ```
4.  **View the output in RViz:**
    -   Add an `Image` display.
    -   Set the topic to `/bev_image`.

## 3D RViz Plugin

The 3D RViz plugin provides a more integrated visualization experience directly within the RViz 3D view.

### Features

-   Renders 3D bounding boxes for object detections.
-   Displays predicted trajectories as lines.
-   Shows the planned ego-path as a line.
-   All visualizations are rendered in the 3D scene, respecting the message's frame ID.

### How to Use

1.  **Build the plugin:**
    ```bash
    colcon build --packages-select sparsedrive_rviz_plugin
    ```
2.  **Source the workspace:**
    ```bash
    source install/setup.bash
    ```
3.  **Launch RViz:**
    ```bash
    rviz2
    ```
4.  **Add the plugin display:**
    -   Click the "Add" button in the "Displays" panel.
    -   Select the `sparsedrive_rviz_plugin/SparseDrive` display.
    -   Configure the topic names in the display's properties.

## Unit Tests

Both visualization tools come with unit tests to ensure their correctness.

-   **BEV Visualizer Test:**
    -   Located in `sparsedrive_ros2/visualization/test/test_bev_visualizer.py`.
    -   Generates a test image `test_bev_visualization.png` in the root directory.
    -   Run with `python3 sparsedrive_ros2/visualization/test/test_bev_visualizer.py`.
-   **RViz Plugin Test:**
    -   The RViz plugin test has been removed due to the complexity of mocking the RViz environment.