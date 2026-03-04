# Copyright (C) 2023 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from typing import List, Dict
from cv_bridge import CvBridge

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

import torch
from ultralytics import YOLO, YOLOWorld, YOLOE
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints

from std_srvs.srv import SetBool
from sensor_msgs.msg import Image
from yolo_msgs.msg import Point2D
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import Mask
from yolo_msgs.msg import KeyPoint2D
from yolo_msgs.msg import KeyPoint2DArray
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.srv import SetClasses


class MultiYoloNode(LifecycleNode):
    """
    ROS 2 Lifecycle Node for multi-drone YOLO object detection.

    Loads a single YOLO model and shares it across multiple drone image
    streams.  For each drone namespace supplied via the ``drone_ids``
    parameter the node subscribes to ``/<drone>/image_raw`` and publishes
    ``DetectionArray`` messages on ``/<drone>/yolo/detections``.

    Supported model families: YOLO, YOLOWorld, YOLOE.
    """

    def __init__(self) -> None:
        """
        Initialize the multi-drone YOLO node.

        Declares all ROS parameters for model configuration, inference
        settings, and the list of drone namespaces to serve.
        """
        super().__init__("multi_yolo_node")

        # --- drone identifiers ---
        self.declare_parameter("drone_ids", ["tello1", "tello2"])

        # --- model params ---
        self.declare_parameter("model_type", "YOLO")
        self.declare_parameter("model", "yolov8m.pt")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("fuse_model", False)
        self.declare_parameter("yolo_encoding", "bgr8")
        self.declare_parameter("enable", True)
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        # --- inference params ---
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("half", False)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("augment", False)
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("retina_masks", False)

        self.type_to_model = {"YOLO": YOLO, "World": YOLOWorld, "YOLOE": YOLOE}

    # ------------------------------------------------------------------
    # Lifecycle callbacks
    # ------------------------------------------------------------------

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Configure lifecycle callback.

        Reads parameters and creates a detection publisher for every drone
        namespace listed in ``drone_ids``.

        @param state Current lifecycle state
        @return Transition callback return status
        """
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        # Drone namespaces
        self.drone_ids: List[str] = (
            self.get_parameter("drone_ids").get_parameter_value().string_array_value
        )

        # Model params
        self.model_type = (
            self.get_parameter("model_type").get_parameter_value().string_value
        )
        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.fuse_model = (
            self.get_parameter("fuse_model").get_parameter_value().bool_value
        )
        self.yolo_encoding = (
            self.get_parameter("yolo_encoding").get_parameter_value().string_value
        )

        # Inference params
        self.threshold = (
            self.get_parameter("threshold").get_parameter_value().double_value
        )
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.imgsz_height = (
            self.get_parameter("imgsz_height").get_parameter_value().integer_value
        )
        self.imgsz_width = (
            self.get_parameter("imgsz_width").get_parameter_value().integer_value
        )
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.augment = self.get_parameter("augment").get_parameter_value().bool_value
        self.agnostic_nms = (
            self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        )
        self.retina_masks = (
            self.get_parameter("retina_masks").get_parameter_value().bool_value
        )

        # ROS params
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value
        self.reliability = (
            self.get_parameter("image_reliability").get_parameter_value().integer_value
        )

        self.image_qos_profile = QoSProfile(
            reliability=self.reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.cv_bridge = CvBridge()

        # Per-drone publishers
        self._pubs: Dict[str, Publisher] = {}
        for drone in self.drone_ids:
            topic = f"/{drone}/yolo/detections"
            self._pubs[drone] = self.create_lifecycle_publisher(
                DetectionArray, topic, 10
            )
            self.get_logger().info(f"  Publisher created: {topic}")

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured for drones: {self.drone_ids}")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Activate lifecycle callback.

        Loads the YOLO model, optionally fuses it, and creates per-drone
        image subscriptions and shared services.

        @param state Current lifecycle state
        @return Transition callback return status
        """
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        # Load model
        try:
            self.yolo = self.type_to_model[self.model_type](self.model)
            self.yolo.to(self.device)
        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model}' does not exist")
            return TransitionCallbackReturn.ERROR

        # YOLOE does not support fusing
        if self.fuse_model and (
            isinstance(self.yolo, YOLO) or isinstance(self.yolo, YOLOWorld)
        ):
            try:
                self.get_logger().info("Trying to fuse model...")
                self.yolo.fuse()
            except TypeError as e:
                self.get_logger().warn(f"Error while fuse: {e}")

        # Shared services
        self._enable_srv = self.create_service(SetBool, "enable", self.enable_cb)

        if isinstance(self.yolo, YOLOWorld):
            self._set_classes_srv = self.create_service(
                SetClasses, "set_classes", self.set_classes_cb
            )

        # Per-drone subscriptions
        self._subs: Dict[str, Subscription] = {}
        for drone in self.drone_ids:
            img_topic = f"/{drone}/image_raw"
            self._subs[drone] = self.create_subscription(
                Image,
                img_topic,
                lambda msg, d=drone: self.image_cb(msg, d),
                self.image_qos_profile,
            )
            self.get_logger().info(f"  Subscribed: {img_topic}")

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Deactivate lifecycle callback.

        Destroys the YOLO model, clears CUDA cache if applicable, and
        tears down per-drone subscriptions and shared services.

        @param state Current lifecycle state
        @return Transition callback return status
        """
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        # Destroy per-drone subscriptions
        for drone, sub in self._subs.items():
            self.destroy_subscription(sub)
        self._subs.clear()

        # Destroy services
        self.destroy_service(self._enable_srv)
        self._enable_srv = None

        if hasattr(self, "_set_classes_srv") and self._set_classes_srv is not None:
            self.destroy_service(self._set_classes_srv)
            self._set_classes_srv = None

        # Release model
        del self.yolo
        if "cuda" in self.device:
            self.get_logger().info("Clearing CUDA cache")
            torch.cuda.empty_cache()

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Cleanup lifecycle callback.

        Destroys all per-drone publishers and releases QoS resources.

        @param state Current lifecycle state
        @return Transition callback return status
        """
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        for drone, pub in self._pubs.items():
            self.destroy_publisher(pub)
        self._pubs.clear()

        del self.image_qos_profile

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Shutdown lifecycle callback.

        Performs final cleanup before node shutdown.

        @param state Current lifecycle state
        @return Transition callback return status
        """
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_shutdown(state)
        self.get_logger().info(f"[{self.get_name()}] Shut down")
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Service callbacks
    # ------------------------------------------------------------------

    def enable_cb(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        """
        Service callback to enable or disable detection for all drones.

        @param request Service request containing enable/disable flag
        @param response Service response
        @return Service response with success status
        """
        self.enable = request.data
        response.success = True
        return response

    def set_classes_cb(
        self,
        req: SetClasses.Request,
        res: SetClasses.Response,
    ) -> SetClasses.Response:
        """
        Service callback to set detection classes (YOLOWorld only).

        Updates the classes that the model should detect.

        @param req Service request containing list of class names
        @param res Service response
        @return Service response
        """
        self.get_logger().info(f"Setting classes: {req.classes}")
        self.yolo.set_classes(req.classes)
        self.get_logger().info(f"New classes: {self.yolo.names}")
        return res

    # ------------------------------------------------------------------
    # Result parsing helpers
    # ------------------------------------------------------------------

    def parse_hypothesis(self, results: Results) -> List[Dict]:
        """
        Parse detection hypotheses from YOLO results.

        Extracts class IDs, class names, and confidence scores from detection results.

        @param results YOLO detection results
        @return List of dictionaries containing class information and scores
        """

        hypothesis_list = []

        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:
                hypothesis = {
                    "class_id": int(box_data.cls),
                    "class_name": self.yolo.names[int(box_data.cls)],
                    "score": float(box_data.conf),
                }
                hypothesis_list.append(hypothesis)

        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                hypothesis = {
                    "class_id": int(results.obb.cls[i]),
                    "class_name": self.yolo.names[int(results.obb.cls[i])],
                    "score": float(results.obb.conf[i]),
                }
                hypothesis_list.append(hypothesis)

        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:
        """
        Parse bounding boxes from YOLO results.

        Converts YOLO bounding box format to ROS BoundingBox2D messages.
        Supports both regular boxes and oriented bounding boxes (OBB).

        @param results YOLO detection results
        @return List of BoundingBox2D messages
        """

        boxes_list = []

        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:

                msg = BoundingBox2D()

                # Get boxes values
                box = box_data.xywh[0]
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.size.x = float(box[2])
                msg.size.y = float(box[3])

                # Append msg
                boxes_list.append(msg)

        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                msg = BoundingBox2D()

                # Get boxes values
                box = results.obb.xywhr[i]
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.center.theta = float(box[4])
                msg.size.x = float(box[2])
                msg.size.y = float(box[3])

                # Append msg
                boxes_list.append(msg)

        return boxes_list

    def parse_masks(self, results: Results) -> List[Mask]:
        """
        Parse segmentation masks from YOLO results.

        Converts YOLO mask format to ROS Mask messages containing polygon points.

        @param results YOLO detection results
        @return List of Mask messages
        """

        masks_list = []

        def create_point2d(x: float, y: float) -> Point2D:
            p = Point2D()
            p.x = x
            p.y = y
            return p

        mask: Masks
        for mask in results.masks:

            msg = Mask()

            msg.data = [
                create_point2d(float(ele[0]), float(ele[1]))
                for ele in mask.xy[0].tolist()
            ]
            msg.height = results.orig_img.shape[0]
            msg.width = results.orig_img.shape[1]

            masks_list.append(msg)

        return masks_list

    def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:
        """
        Parse keypoints from YOLO results.

        Extracts keypoint positions and confidence scores, filtering by threshold.

        @param results YOLO detection results
        @return List of KeyPoint2DArray messages
        """

        keypoints_list = []

        points: Keypoints
        for points in results.keypoints:

            msg_array = KeyPoint2DArray()

            if points.conf is None:
                continue

            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):

                if conf >= self.threshold:
                    msg = KeyPoint2D()

                    msg.id = kp_id + 1
                    msg.point.x = float(p[0])
                    msg.point.y = float(p[1])
                    msg.score = float(conf)

                    msg_array.data.append(msg)

            keypoints_list.append(msg_array)

        return keypoints_list

    # ------------------------------------------------------------------
    # Image callback
    # ------------------------------------------------------------------

    def image_cb(self, msg: Image, drone_id: str) -> None:
        """
        Image callback for processing detections from a specific drone.

        Receives images, runs YOLO inference, parses results, and publishes
        detections to the drone-specific topic.

        @param msg Image message to process
        @param drone_id Namespace of the drone that produced the image
        """

        if not self.enable:
            return

        pub = self._pubs.get(drone_id)
        if pub is None:
            return

        # Skip inference when nobody is listening
        if pub.get_subscription_count() == 0:
            return

        # Convert image + predict
        cv_image = self.cv_bridge.imgmsg_to_cv2(
            msg, desired_encoding=self.yolo_encoding
        )
        results = self.yolo.predict(
            source=cv_image,
            verbose=False,
            stream=False,
            conf=self.threshold,
            iou=self.iou,
            imgsz=(self.imgsz_height, self.imgsz_width),
            half=self.half,
            max_det=self.max_det,
            augment=self.augment,
            agnostic_nms=self.agnostic_nms,
            retina_masks=self.retina_masks,
            device=self.device,
        )
        results: Results = results[0].cpu()

        if results.boxes or results.obb:
            hypothesis = self.parse_hypothesis(results)
            boxes = self.parse_boxes(results)

        if results.masks:
            masks = self.parse_masks(results)

        if results.keypoints:
            keypoints = self.parse_keypoints(results)

        # Build detection messages
        detections_msg = DetectionArray()

        for i in range(len(results)):

            aux_msg = Detection()

            if (results.boxes or results.obb) and hypothesis and boxes:
                aux_msg.class_id = hypothesis[i]["class_id"]
                aux_msg.class_name = hypothesis[i]["class_name"]
                aux_msg.score = hypothesis[i]["score"]

                aux_msg.bbox = boxes[i]

            if results.masks and masks:
                aux_msg.mask = masks[i]

            if results.keypoints and keypoints:
                aux_msg.keypoints = keypoints[i]

            detections_msg.detections.append(aux_msg)

        # Publish to the drone-specific topic
        detections_msg.header = msg.header
        pub.publish(detections_msg)

        del results
        del cv_image


def main():
    rclpy.init()
    node = MultiYoloNode()
    node.trigger_configure()
    node.trigger_activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()