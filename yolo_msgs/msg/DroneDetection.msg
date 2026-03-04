# Detection of another drone in the camera frame.
#
# Published by drone_detector_node after a configurable number of
# consecutive high-confidence YOLO frames confirm a drone is present.

# ── Position constants (horizontal region in the image) ──
uint8 POSITION_LEFT  = 0
uint8 POSITION_RIGHT = 1

# ── Distance constants (derived from bounding-box area) ──
uint8 DISTANCE_NEAR = 0
uint8 DISTANCE_FAR  = 1

# ── Fields ───────────────────────────────────────────────
uint8   position        # One of POSITION_* constants
uint8   distance        # One of DISTANCE_* constants
float64 score           # Best detection confidence in the confirming frame
float64 bbox_area       # Bounding-box area in px² (width × height)
