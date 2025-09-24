# SparseDrive ROS2 Visualization

This document describes the visualization tools provided for the `sparsedrive_ros2` package.

## BEV Renderer

The BEV (Bird's Eye View) renderer is a Python-based ROS2 node that subscribes to the output of the `sparsedrive_node` and generates a 2D image representation of the scene. The visual style is a faithful re-implementation of the original `bev_render.py` script from the SparseDrive project, using OpenCV for performant rendering.

### Features

-   Visualizes 3D object detections, trajectories, and planned paths with the original color schemes.
-   Visualizes map predictions and ground truth from `nav_msgs/OccupancyGrid` topics.
-   Overlays the SDC car and a legend for clarity.
-   Publishes the final visualization as a `sensor_msgs/msg/Image` on the `/bev_image` topic.
-   Rendering is triggered by the `/image_raw` topic to ensure synchronization with the model's input.

### How to Run

1.  **Build the package:**
    ```bash
    colcon build --packages-select sparsedrive_ros2
    ```
2.  **Source the workspace:**
    ```bash
    source install/setup.bash
    ```
3.  **Launch the renderer:**
    ```bash
    ros2 run sparsedrive_ros2 bev_renderer
    ```
4.  **View the output in RViz:**
    -   Add an `Image` display.
    -   Set the topic to `/bev_image`.

## Standard ROS Messages for 3D Visualization

All data is published using standard ROS message types, which can be visualized directly in RViz for 3D inspection.

-   **Detections:** `vision_msgs/Detection3DArray` on `/detection`. Use the `Detection3DArray` display in the `rviz_vision_msgs` package.
-   **Trajectories:** `multipath_msgs/MultiPath` on `/traj`. Use a `Path` display for each path in the array.
-   **Plans:** `nav_msgs/Path` on `/plan` and `/plan_gt`. Use the `Path` display.
-   **Maps:** `nav_msgs/OccupancyGrid` on `/map_pred` and `/map_gt`. Use the `Map` display.

## Unit Tests

A unit test is provided for the BEV renderer to verify the correctness of the drawing logic.

-   **BEV Renderer Test:**
    -   Located in `sparsedrive_ros2/test/test_bev_render.py`.
    -   Verifies the rendering logic by checking pixel colors in an output image.
    -   Run with `colcon test --packages-select sparsedrive_ros2`.