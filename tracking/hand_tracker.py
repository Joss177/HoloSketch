import time

import cv2
import mediapipe as mp
import numpy as np

from config.settings import (
    MODEL_PATH,
    MAX_HANDS,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE,
)

HAND_CONNECTIONS = np.array([
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
], dtype=np.int32)

LINE_COLOR = (255, 210, 70)
TIP_COLOR = (50, 255, 120)
POINT_COLOR = (255, 255, 255)


class HandTracker:
    def __init__(self):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"No se encontro el modelo: {MODEL_PATH}")

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.last_timestamp = -1

    def process(self, frame):
        # cvtColor ya devuelve un array contiguo
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = time.perf_counter_ns() // 1_000_000
        if ts <= self.last_timestamp:
            ts = self.last_timestamp + 1
        self.last_timestamp = ts

        return self.detector.detect_for_video(image, ts)

    def draw_landmarks(self, frame, result):
        h, w = frame.shape[:2]
        scale = np.array([w, h], dtype=np.float32)

        for hand in result.hand_landmarks:
            pts = (
                np.array([(lm.x, lm.y) for lm in hand], dtype=np.float32) * scale
            ).astype(np.int32)

            # Todas las conexiones en una sola llamada
            cv2.polylines(frame, list(pts[HAND_CONNECTIONS]), False, LINE_COLOR, 2)

            for i, (x, y) in enumerate(pts):
                if i in (4, 8):
                    cv2.circle(frame, (int(x), int(y)), 5, TIP_COLOR, -1)
                else:
                    cv2.circle(frame, (int(x), int(y)), 3, POINT_COLOR, -1)

        return frame

    def close(self):
        self.detector.close()