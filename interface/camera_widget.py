import math
import time

import cv2
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from config.settings import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT
from gestures.gesture_recognizer import GestureRecognizer
from tracking.hand_tracker import HandTracker

PROCESS_WIDTH = 640
SWAP_HANDEDNESS = True  # Imagen reflejada antes de MediaPipe


class OneEuroFilter:
    def __init__(self, min_cutoff=1.2, beta=0.08, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x

        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt

        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)

        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    hand_data = Signal(object)
    gesture_detected = Signal(str)
    status_changed = Signal(str, bool)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap.release()
            self.error.emit("No se pudo abrir la camara")
            return

        cap.set(cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        tracker = None

        # Estados totalmente independientes
        recognizers = {
            "left": GestureRecognizer(),
            "right": GestureRecognizer(),
        }

        filters = {
            side: (OneEuroFilter(), OneEuroFilter())
            for side in ("left", "right")
        }

        last_status = None
        last_gesture = None
        previous_hands = set()

        try:
            tracker = HandTracker()

            while self._running:
                ok, frame = cap.read()

                if not ok:
                    self.hand_data.emit({
                        "left": None,
                        "right": None,
                    })
                    self.msleep(30)
                    continue

                h, w = frame.shape[:2]

                if w > PROCESS_WIDTH:
                    frame = cv2.resize(
                        frame,
                        (PROCESS_WIDTH,
                         int(h * PROCESS_WIDTH / w)),
                        interpolation=cv2.INTER_AREA,
                    )

                frame = cv2.flip(frame, 1)

                # UNA inferencia para ambas manos
                result = tracker.process(frame)
                now = time.perf_counter()

                hands = {
                    "left": None,
                    "right": None,
                }

                detected = set()

                for i, hand in enumerate(result.hand_landmarks):
                    if i >= len(result.handedness):
                        continue

                    categories = result.handedness[i]
                    if not categories:
                        continue

                    label = categories[0].category_name

                    if SWAP_HANDEDNESS:
                        label = (
                            "Left" if label == "Right"
                            else "Right"
                        )

                    side = label.lower()

                    if side not in hands:
                        continue

                    recognizer = recognizers[side]
                    fx, fy = filters[side]

                    gesture = recognizer.recognize(hand)

                    # Punto medio entre pulgar e indice
                    thumb = hand[4]
                    index = hand[8]

                    raw_x = max(
                        0.0, min(1.0, (thumb.x + index.x) / 2)
                    )
                    raw_y = max(
                        0.0, min(1.0, (thumb.y + index.y) / 2)
                    )

                    hands[side] = {
                        "x": fx(raw_x, now),
                        "y": fy(raw_y, now),
                        "pinch": gesture == "PINCH",
                        "grab": gesture == "GRAB",
                        "gesture": gesture,
                        "distance": recognizer.last_distance,
                    }

                    detected.add(side)

                # Reiniciar únicamente la mano perdida
                for side in previous_hands - detected:
                    recognizers[side].reset()
                    for f in filters[side]:
                        f.reset()

                previous_hands = detected

                # Datos compactos para el visor
                self.hand_data.emit(hands)

                left = hands["left"]
                right = hands["right"]

                left_state = (
                    left["gesture"] if left else "--"
                )
                right_state = (
                    right["gesture"] if right else "--"
                )

                status = (left_state, right_state)

                if status != last_status:
                    last_status = status

                    self.status_changed.emit(
                        f"IZQ: {left_state} | DER: {right_state}",
                        any(
                            hd is not None
                            and (hd["pinch"] or hd["grab"])
                            for hd in hands.values()
                        ),
                    )

                combined = f"L:{left_state} R:{right_state}"

                if combined != last_gesture:
                    last_gesture = combined
                    self.gesture_detected.emit(combined)

                tracker.draw_landmarks(frame, result)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sh, sw, ch = rgb.shape

                img = QImage(
                    rgb.data,
                    sw, sh, ch * sw,
                    QImage.Format.Format_RGB888,
                ).copy()

                self.frame_ready.emit(img)

        except Exception as e:
            self.error.emit(f"Error de seguimiento: {e}")

        finally:
            cap.release()
            if tracker is not None:
                tracker.close()


class CameraWidget(QWidget):
    gesture_detected = Signal(str)
    hand_data = Signal(object)

    STYLE_IDLE = (
        "color:#999999; font-size:12px; padding:6px;"
    )
    STYLE_PINCH = (
        "color:#ffffff; font-size:12px; padding:6px;"
    )

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumWidth(280)
        self.closed = False

        self.video_label = QLabel("Iniciando camara...")
        self.video_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.video_label.setMinimumHeight(260)
        self.video_label.setStyleSheet(
            "background:#000000; color:#777777;"
        )

        self.status_label = QLabel("Buscando manos...")
        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.status_label.setStyleSheet(self.STYLE_IDLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.video_label, 1)
        layout.addWidget(self.status_label)

        self.worker = CameraWorker(self)

        self.worker.frame_ready.connect(self._show_frame)
        self.worker.hand_data.connect(self.hand_data.emit)
        self.worker.gesture_detected.connect(
            self.gesture_detected.emit
        )
        self.worker.status_changed.connect(
            self._set_status
        )
        self.worker.error.connect(self._on_error)

        self.worker.start()

    def _show_frame(self, img):
        if self.closed:
            return

        pix = QPixmap.fromImage(img)

        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

    def _set_status(self, text, pinch):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            self.STYLE_PINCH if pinch else self.STYLE_IDLE
        )

    def _on_error(self, message):
        self.status_label.setText(message)
        self.hand_data.emit({
            "left": None,
            "right": None,
        })

    def close_camera(self):
        if self.closed:
            return

        self.closed = True
        self.worker.stop()

        if not self.worker.wait(3000):
            print("[CAMERA] El hilo aun esta cerrando")
            self.worker.wait()

        self.hand_data.emit({
            "left": None,
            "right": None,
        })