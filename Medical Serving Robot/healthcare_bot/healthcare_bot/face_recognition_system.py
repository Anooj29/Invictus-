"""
Standalone face‑recognition viewer for testing the `known_faces` database.

This script reuses the `vision.load_known_faces` helper so encoding logic is
kept in one place.
"""

from __future__ import annotations

import cv2
import numpy as np
import face_recognition

from vision import load_known_faces


def main() -> None:
    known_encodings, known_names = load_known_faces()
    if not known_encodings:
        print("No known faces loaded. Add images under the `known_faces` folder.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not detected!")
        return

    tolerance = 0.55  # 0.5 strict, 0.6 loose

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locations = face_recognition.face_locations(rgb_small)
            encodings = face_recognition.face_encodings(rgb_small, locations)

            for (top, right, bottom, left), encoding in zip(locations, encodings):
                distances = face_recognition.face_distance(known_encodings, encoding)

                name = "Unknown"
                confidence_text = ""

                if len(distances) > 0:
                    best_idx = int(np.argmin(distances))
                    if distances[best_idx] < tolerance:
                        name = known_names[best_idx]
                        confidence = round(
                            (1.0 - float(distances[best_idx])) * 100.0, 2
                        )
                        confidence_text = f"{confidence:.1f}%"

                # Scale back up since frame was resized
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                label = f"{name} {confidence_text}"
                cv2.putText(
                    frame,
                    label,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Healthcare Robot - Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

