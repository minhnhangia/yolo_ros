#!/usr/bin/env python3
"""Launch file for the drone detector node.

Starts the drone_detector_node from yolo_ros which subscribes to
YOLO detections, counts consecutive high-confidence drone frames,
and publishes the drone's horizontal position (left / mid / right)
on the 'drone_detected' topic for mission_control to react to.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Launch arguments ────────────────────────────────────────────
    target_class_arg = DeclareLaunchArgument(
        "target_class",
        default_value="drone",
        description="YOLO class name to detect as another drone.",
    )

    confidence_threshold_arg = DeclareLaunchArgument(
        "confidence_threshold",
        default_value="0.6",
        description="Minimum YOLO score to count as a valid detection.",
    )

    consecutive_frames_arg = DeclareLaunchArgument(
        "consecutive_frames",
        default_value="3",
        description="Number of consecutive frames required before publishing.",
    )

    image_width_arg = DeclareLaunchArgument(
        "image_width",
        default_value="648",
        description="Camera image width in pixels (for left/right split).",
    )

    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="",
        description="Namespace for the node.",
    )

    # ── Node ────────────────────────────────────────────────────────
    drone_detector_node = Node(
        package="yolo_ros",
        executable="drone_detector_node",
        name="drone_detector_node",
        namespace=LaunchConfiguration("namespace"),
        parameters=[
            {
                "target_class": LaunchConfiguration("target_class"),
                "confidence_threshold": LaunchConfiguration("confidence_threshold"),
                "consecutive_frames": LaunchConfiguration("consecutive_frames"),
                "image_width": LaunchConfiguration("image_width"),
            }
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            target_class_arg,
            confidence_threshold_arg,
            consecutive_frames_arg,
            image_width_arg,
            namespace_arg,
            drone_detector_node,
        ]
    )
