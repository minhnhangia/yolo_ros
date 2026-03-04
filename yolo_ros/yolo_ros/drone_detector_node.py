#!/usr/bin/env python3
"""
Drone detector node for YOLO-based drone avoidance.

Subscribes to YOLO detections and identifies when another drone has been
consistently visible with high confidence across many consecutive frames.
When confirmed, publishes the drone's horizontal position in the frame
(left / right) so mission_control can react accordingly.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
)

from std_msgs.msg import String
from yolo_msgs.msg import DetectionArray


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
        Assumed image width in pixels, used to compute left/mid/right
        (default ``478``).
    """


    # 648 x 478

    def __init__(self) -> None:
        super().__init__("drone_detector_node")

        # ── Parameters ──────────────────────────────────────────────────
        self.declare_parameter("target_class", "drone")
        self.declare_parameter("confidence_threshold", 0.6)
        self.declare_parameter("consecutive_frames", 3)
        self.declare_parameter("image_width", 648)
        self.declare_parameter("area_threshold", 2500.0)

        self.target_class: str = (
            self.get_parameter("target_class")
            .get_parameter_value()
            .string_value
        )
        self.confidence_threshold: float = (
            self.get_parameter("confidence_threshold")
            .get_parameter_value()
            .double_value
        )
        self.consecutive_frames: int = (
            self.get_parameter("consecutive_frames")
            .get_parameter_value()
            .integer_value
        )
        self.image_width: int = (
            self.get_parameter("image_width")
            .get_parameter_value()
            .integer_value
        )
        self.area_threshold: float = (
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
        self.sub = self.create_subscription(
            DetectionArray,
            "yolo/detections",
            self._detection_callback,
            sensor_qos,
        )

        self.pub = self.create_publisher(
            String,
            "yolo/drone_detected",
            reliable_qos,
        )

        self.get_logger().info(
            f"DroneDetectorNode started — looking for '{self.target_class}' "
            f"with score ≥ {self.confidence_threshold} "
            f"for {self.consecutive_frames} consecutive frames "
            f"area_threshold={self.area_threshold:.0f}px²"
        )

    # ════════════════════════════════════════════════════════════════════
    # CALLBACK
    # ════════════════════════════════════════════════════════════════════

    def _detection_callback(self, msg: DetectionArray) -> None:
        """Process a single frame of YOLO detections."""

        # Find the best-scoring drone detection in this frame
        best_det = None
        best_score = 0.0
        for det in msg.detections:
            if (
                det.class_name == self.target_class
                and det.score >= self.confidence_threshold
                and det.score > best_score
            ):
                best_det = det
                best_score = det.score

        if best_det is None:
            # No qualifying drone in this frame → reset streak
            self._consecutive_count = 0
            return

        self._consecutive_count += 1

        bbox_area = best_det.bbox.size.x * best_det.bbox.size.y
        self.get_logger().info(
            f"Drone detected: frame {self._consecutive_count}"
            f"/{self.consecutive_frames}  score={best_score:.2f}"
            f"  area={bbox_area:.0f}px²"
        )

        if self._consecutive_count < self.consecutive_frames:
            return

        # ── Confirmed: compute horizontal position ──────────────────
        cx = best_det.bbox.center.position.x  # centre-x in pixels
        half = self.image_width / 2.0

        if cx < half:
            position = "left"
        else:
            position = "right"

        # ── Compute area from bbox size → near / far ────────────────
        bbox_w = best_det.bbox.size.x
        bbox_h = best_det.bbox.size.y
        area = bbox_w * bbox_h
        
        if area >= self.area_threshold:
            distance = "near"
        else:
            distance = "far"

        if distance == "near":
            out_msg = String()
            out_msg.data = f"{position}"
            self.pub.publish(out_msg)

            self.get_logger().warning(
                f"DRONE CONFIRMED ({self.consecutive_frames} frames) — "
                f"position: {position}, area: {area:.0f}px²"
                f"(cx={cx:.0f}, area={area:.0f}px²)"
        )

        # Reset counter so we don't spam every frame after confirmation
        self._consecutive_count = 0

# ════════════════════════════════════════════════════════════════════════
# ENTRY POINT
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
