"""
Utility script to capture multiple images for a single person.

Images are stored under:
    known_faces/<person_name>/
"""

from __future__ import annotations

import os
import time

import cv2


def capture_person(person_name: str, num_images: int = 5, delay_s: float = 2.0) -> None:
    save_path = os.path.join("known_faces", person_name)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not opened!")
        return

    print(f"Capturing {num_images} images for {person_name}. Look at the camera.")

    try:
        count = 0
        while count < num_images:
            ok, frame = cap.read()
            if not ok:
                print("Failed to grab frame")
                break

            filename = os.path.join(save_path, f"{count}.jpg")
            cv2.imwrite(filename, frame)
            print("Saved:", filename)

            count += 1
            time.sleep(delay_s)
    finally:
        cap.release()
        print("Done capturing.")


if __name__ == "__main__":
    # Change this name when registering a new person.
    capture_person("Suraj")

