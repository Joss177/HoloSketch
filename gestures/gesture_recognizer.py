import math

from config.settings import (
    PINCH_START_THRESHOLD,
    PINCH_END_THRESHOLD,
    PINCH_CONFIRM_FRAMES,
    PINCH_RELEASE_FRAMES,
)

# Puño: fotogramas para confirmar / soltar (evita parpadeos)
GRAB_CONFIRM_FRAMES = 2
GRAB_RELEASE_FRAMES = 3

# (punta, articulacion PIP) de indice, medio, anular y menique
FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))


class GestureRecognizer:

    def __init__(self):
        self.pinch_active = False
        self.pinch_frames = 0
        self.release_frames = 0

        self.grab_active = False
        self.grab_frames = 0
        self.grab_release_frames = 0

        self.last_distance = 0.0

    @staticmethod
    def distance(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def _is_fist(self, lm):
        """Puño: las 4 puntas estan mas cerca de la muñeca que su PIP."""
        wrist = lm[0]
        return all(
            self.distance(lm[tip], wrist) < self.distance(lm[pip], wrist)
            for tip, pip in FINGERS
        )

    def _state(self):
        if self.pinch_active:
            return "PINCH"
        if self.grab_active:
            return "GRAB"
        return "LIBRE"

    def recognize(self, landmarks):
        palm_width = self.distance(landmarks[5], landmarks[17])
        if palm_width < 0.001:
            return self._state()

        fist = self._is_fist(landmarks)
        d = self.distance(landmarks[4], landmarks[8]) / palm_width
        self.last_distance = d

        # ---- PINCH (no se inicia si la mano esta en puño) ----
        if not self.pinch_active:
            if d < PINCH_START_THRESHOLD and not fist:
                self.pinch_frames += 1
                if self.pinch_frames >= PINCH_CONFIRM_FRAMES:
                    self.pinch_active = True
                    self.pinch_frames = 0
                    self.release_frames = 0
            else:
                self.pinch_frames = 0
        else:
            if d > PINCH_END_THRESHOLD:
                self.release_frames += 1
                if self.release_frames >= PINCH_RELEASE_FRAMES:
                    self.pinch_active = False
                    self.release_frames = 0
                    self.pinch_frames = 0
            else:
                self.release_frames = 0

        if self.pinch_active:
            self.grab_active = False
            self.grab_frames = 0
            self.grab_release_frames = 0
            return "PINCH"

        # ---- GRAB (puño) con histeresis ----
        if not self.grab_active:
            if fist:
                self.grab_frames += 1
                if self.grab_frames >= GRAB_CONFIRM_FRAMES:
                    self.grab_active = True
                    self.grab_frames = 0
                    self.grab_release_frames = 0
            else:
                self.grab_frames = 0
        else:
            if not fist:
                self.grab_release_frames += 1
                if self.grab_release_frames >= GRAB_RELEASE_FRAMES:
                    self.grab_active = False
                    self.grab_release_frames = 0
            else:
                self.grab_release_frames = 0

        return self._state()

    def reset(self):
        self.pinch_active = False
        self.pinch_frames = 0
        self.release_frames = 0
        self.grab_active = False
        self.grab_frames = 0
        self.grab_release_frames = 0
        self.last_distance = 0.0