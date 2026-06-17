"""
Computer‑vision helpers for the Healthcare Serving Robot.

This module wraps:
- Loading and encoding known faces
- TensorFlow‑Lite based emotion recognition
- TensorFlow‑Lite based person detection
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import face_recognition
import numpy as np
import tflite_runtime.interpreter as tflite


def load_known_faces(base_path: str = "known_faces") -> tuple[List[np.ndarray], List[str]]:
    """
    Load face encodings from the `known_faces` directory.

    Directory layout:
        known_faces/
            Alice/
                0.jpg
                1.jpg
            Bob/
                ...
    """
    encodings: List[np.ndarray] = []
    names: List[str] = []

    if not os.path.isdir(base_path):
        print(f"[Faces] Directory not found: {base_path}")
        return encodings, names

    print("[Faces] Loading known faces...")
    for person_name in os.listdir(base_path):
        person_folder = os.path.join(base_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)

            try:
                image = face_recognition.load_image_file(image_path)
                image_encodings = face_recognition.face_encodings(image)
            except Exception as exc:
                print(f"[Faces] Failed to process {image_path}: {exc}")
                continue

            if image_encodings:
                encodings.append(image_encodings[0])
                names.append(person_name)
                print(f"[Faces] Loaded {image_name} for {person_name}")
            else:
                print(f"[Faces] No face found in {image_name}")

    print("[Faces] System ready.")
    return encodings, names


@dataclass
class EmotionDetector:
    """TensorFlow‑Lite based emotion classifier."""

    model_path: str = "emotion_model.tflite"

    def __post_init__(self) -> None:
        self._interpreter = tflite.Interpreter(self.model_path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._labels = [
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Neutral",
            "Sad",
            "Surprise",
        ]

    def predict(self, face_bgr: np.ndarray) -> str:
        """
        Predict the dominant emotion for the given face image.

        The model expects 48x48 greyscale input in the range [0,1].
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        input_tensor = resized.reshape(1, 48, 48, 1).astype(np.float32) / 255.0

        self._interpreter.set_tensor(self._input_details[0]["index"], input_tensor)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_details[0]["index"])
        idx = int(np.argmax(output))
        return self._labels[idx]


@dataclass
class PersonDetector:
    """
    TensorFlow‑Lite person detector.

    The model is expected to output:
    - boxes   [N, 4]
    - classes [N]
    - scores  [N]
    where class 0 corresponds to "person".
    """

    model_path: str = "person_detect.tflite"
    min_score: float = 0.4

    def __post_init__(self) -> None:
        self._interpreter = tflite.Interpreter(self.model_path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # Model input spatial resolution
        self._in_h = self._input_details[0]["shape"][1]
        self._in_w = self._input_details[0]["shape"][2]

    def _prepare_input(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._in_w, self._in_h))
        return np.expand_dims(resized, axis=0).astype(np.uint8)

    def detect(
        self, frame_bgr: np.ndarray
    ) -> Tuple[bool, Tuple[int, int, int, int] | None]:
        """
        Run person detection on the frame.

        Returns:
            (found, bbox) where bbox = (xmin, ymin, xmax, ymax) in pixel coords.
        """
        input_tensor = self._prepare_input(frame_bgr)

        self._interpreter.set_tensor(self._input_details[0]["index"], input_tensor)
        self._interpreter.invoke()

        boxes = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
        classes = self._interpreter.get_tensor(self._output_details[1]["index"])[0]
        scores = self._interpreter.get_tensor(self._output_details[2]["index"])[0]

        h, w, _ = frame_bgr.shape

        best_score = 0.0
        best_bbox: Tuple[int, int, int, int] | None = None

        for box, cls, score in zip(boxes, classes, scores):
            if score < self.min_score or int(cls) != 0:
                continue

            ymin, xmin, ymax, xmax = box
            xmin_px = int(xmin * w)
            xmax_px = int(xmax * w)
            ymin_px = int(ymin * h)
            ymax_px = int(ymax * h)

            if score > best_score:
                best_score = float(score)
                best_bbox = (xmin_px, ymin_px, xmax_px, ymax_px)

        return (best_bbox is not None), best_bbox

