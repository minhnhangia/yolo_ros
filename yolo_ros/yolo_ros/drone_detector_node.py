#!/usr/bin/env python3
"""
Drone detector node for YOLO-based drone avoidance.

Subscribes to YOLO detections and identifies when another drone has been
consistently visible with high confidence across many consecutive frames.
When confirmed *and* close enough (bbox area ≥ threshold), publishes the
drone's horizontal position (left / right) so mission_control can react.
"""

from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
)

from yolo_msgs.msg import Detection, DetectionArray, DroneDetection

# Lookup table: message constant → human-readable label for logging.
_POSITION_LABELS = {
    DroneDetection.POSITION_LEFT: "left",
    DroneDetection.POSITION_RIGHT: "right",
}


class DroneDetectorNode(Node):
    """
    Analyses YOLO detections to find other drones in the camera frame.

    Parameters
    ----------
    target_class : str
        YOLO class name to look for (default ``"drone"``).
    confidence_threshold : float
        Minimum detection score to accept (default ``0.6``).
    consecutive_frames : int
        How many frames in a row the drone must be seen before publishing
        (default ``3``).
    image_width : int
        Camera image width in pixels, used to split left / right
        (default ``648``).
    area_threshold : float
        Minimum bounding-box area (px²) to classify as *near*
        (default ``2500.0``).
    """

    def __init__(self) -> None:
        super().__init__("drone_detector_node")

        # ── Parameters ──────────────────────────────────────────────────
        self.declare_parameter("target_class", "drone")
        self.declare_parameter("confidence_threshold", 0.6)
        self.declare_parameter("consecutive_frames", 3)
        self.declare_parameter("image_width", 648)
        self.declare_parameter("area_threshold", 2500.0)

        self._target_class: str = (
            self.get_parameter("target_class")
            .get_parameter_value()
            .string_value
        )
        self._confidence_threshold: float = (
            self.get_parameter("confidence_threshold")
            .get_parameter_value()
            .double_value
        )
        self._consecutive_frames: int = (
            self.get_parameter("consecutive_frames")
            .get_parameter_value()
            .integer_value
        )
        self._image_width: int = (
            self.get_parameter("image_width")
            .get_parameter_value()
            .integer_value
        )
        self._area_threshold: float = (
            self.get_parameter("area_threshold")
            .get_parameter_value()
            .double_value
        )

        # ── Internal state ──────────────────────────────────────────────
        self._consecutive_count: int = 0

        # ── QoS ─────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Pub / Sub ──────────────────────────────────────────────────
        self._sub = self.create_subscription(
            DetectionArray,
            "yolo/detections",
            self._detection_cb,
            sensor_qos,
        )
        self._pub = self.create_publisher(
            DroneDetection,
            "yolo/drone_detected",
            reliable_qos,
        )

        self.get_logger().info(
            f"DroneDetectorNode started — looking for '{self._target_class}' "
            f"with score ≥ {self._confidence_threshold} "
            f"for {self._consecutive_frames} consecutive frames "
            f"area_threshold={self._area_threshold:.0f}px²"
        )

    # ════════════════════════════════════════════════════════════════════
    # Helpers
    # ════════════════════════════════════════════════════════════════════

    def _best_target_detection(
        self, detections: list
    ) -> Optional[Detection]:
        """Return the highest-scoring detection matching the target class,
        or ``None`` if no detection meets the confidence threshold."""
        best: Optional[Detection] = None
        for det in detections:
            if (
                det.class_name == self._target_class
                and det.score >= self._confidence_threshold
                and (best is None or det.score > best.score)
            ):
                best = det
        return best

    @staticmethod
    def _bbox_area(det: Detection) -> float:
        """Compute bounding-box area in px²."""
        return det.bbox.size.x * det.bbox.size.y

    def _classify_position(self, cx: float) -> int:
        """Map centre-x to a ``DroneDetection.POSITION_*`` constant."""
        half = self._image_width / 2.0
        if cx < half:
            return DroneDetection.POSITION_LEFT
        return DroneDetection.POSITION_RIGHT

    def _classify_distance(self, area: float) -> int:
        """Map bbox area to a ``DroneDetection.DISTANCE_*`` constant."""
        if area >= self._area_threshold:
            return DroneDetection.DISTANCE_NEAR
        return DroneDetection.DISTANCE_FAR

    # ════════════════════════════════════════════════════════════════════
    # Callback
    # ════════════════════════════════════════════════════════════════════

    def _detection_cb(self, msg: DetectionArray) -> None:
        """Process a single frame of YOLO detections."""

        best = self._best_target_detection(msg.detections)

        if best is None:
            self._consecutive_count = 0
            return

        self._consecutive_count += 1
        area = self._bbox_area(best)

        self.get_logger().info(
            f"Drone detected: frame {self._consecutive_count}"
            f"/{self._consecutive_frames}  score={best.score:.2f}"
            f"  area={area:.0f}px²"
        )

        if self._consecutive_count < self._consecutive_frames:
            return

        # ── Streak reached — classify & gate on distance ────────────
        distance = self._classify_distance(area)
        if distance != DroneDetection.DISTANCE_NEAR:
            self._consecutive_count = 0
            return

        position = self._classify_position(best.bbox.center.position.x)

        out_msg = DroneDetection()
        out_msg.position = position
        out_msg.distance = distance
        out_msg.score = best.score
        out_msg.bbox_area = area
        self._pub.publish(out_msg)

        self.get_logger().warning(
            f"DRONE CONFIRMED ({self._consecutive_frames} frames) — "
            f"position: {_POSITION_LABELS[position]}, area: {area:.0f}px² "
            f"(cx={best.bbox.center.position.x:.0f}, score={best.score:.2f})"
        )

        # Reset counter so we don't spam every frame after confirmation
        self._consecutive_count = 0


# ════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = DroneDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
