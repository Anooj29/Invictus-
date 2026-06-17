"""
Integrated main script for the Healthcare Serving Robot.

This version uses the shared modules:
- `config`   – static configuration values
- `hardware` – GPIO / motors / distance sensor
- `vision`   – person detector, emotion model, known faces
- `services` – speaker, Gemini AI client, serial health sensor, dashboard socket
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import face_recognition

from .config import RobotConfig
from .hardware import DistanceSensor, MotorController, gpio_cleanup
from .services import (
    DashboardConnection,
    GeminiClient,
    HealthSensorClient,
    Speaker,
    Telemetry,
)
from .vision import EmotionDetector, PersonDetector, load_known_faces


@dataclass
class RobotState:
    """Mutable state for the main state‑machine."""

    mode: str = "SEARCH"  # SEARCH → INTERACT → WAIT_CONFIRM → HEALTH_CHECK → DONE
    greeted: bool = False
    health_stage: int = 0
    name: str = "Unknown"
    temperature: str = "--"
    pulse: str = "--"
    ecg: str = "--"


def main() -> None:
    cfg = RobotConfig()

    # --- Hardware and services ---
    distance_sensor = DistanceSensor(
        trig_pin=cfg.ultrasonic_trig_pin,
        echo_pin=cfg.ultrasonic_echo_pin,
    )
    motors = MotorController(
        left_dir_pin=cfg.left_dir_pin,
        left_pwm_pin=cfg.left_pwm_pin,
        right_dir_pin=cfg.right_dir_pin,
        right_pwm_pin=cfg.right_pwm_pin,
    )
    speaker = Speaker()
    gemini = GeminiClient(cfg)
    health = HealthSensorClient(cfg)
    dashboard = DashboardConnection(cfg)

    # --- Vision models ---
    person_detector = PersonDetector()
    emotion_detector = EmotionDetector()
    known_encodings, known_names = load_known_faces()

    # --- Camera ---
    cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("[Camera] Failed to open camera.")
        return

    state = RobotState()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            distance = distance_sensor.read_cm()
            print(f"[Loop] State={state.mode}  Distance={distance:.1f} cm")

            # Send telemetry snapshot to the dashboard.
            dashboard.send_telemetry(
                Telemetry(
                    distance=distance,
                    temperature=state.temperature,
                    pulse=state.pulse,
                    ecg=state.ecg,
                )
            )

            # Read latest voice/text command from dashboard, if any.
            command = dashboard.read_command()

            # ---------------- STATE: SEARCH ----------------
            if state.mode == "SEARCH":
                found, bbox = person_detector.detect(frame)

                if found and bbox:
                    xmin, ymin, xmax, ymax = bbox
                    cv2.rectangle(
                        frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2
                    )

                    if distance <= cfg.distance_trigger_cm:
                        motors.stop()
                        state.mode = "INTERACT"
                        state.greeted = False
                    elif distance <= cfg.safety_stop_cm:
                        motors.stop()
                    else:
                        # Simple centring controller based on person position.
                        person_center = (xmin + xmax) // 2
                        frame_center = frame.shape[1] // 2
                        margin = 70
                        if person_center < frame_center - margin:
                            motors.turn_left()
                        elif person_center > frame_center + margin:
                            motors.turn_right()
                        else:
                            motors.forward()
                else:
                    motors.stop()

            # ---------------- STATE: INTERACT ----------------
            elif state.mode == "INTERACT":
                motors.stop()

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                locations = face_recognition.face_locations(rgb)
                encodings = face_recognition.face_encodings(rgb, locations)

                if encodings and not state.greeted and known_encodings:
                    matches = face_recognition.compare_faces(
                        known_encodings, encodings[0]
                    )
                    if True in matches:
                        state.name = known_names[matches.index(True)]
                        emotion = emotion_detector.predict(frame)
                        speaker.say(gemini.greeting(state.name, emotion))
                        state.greeted = True
                        state.mode = "WAIT_CONFIRM"

            # ---------------- STATE: WAIT_CONFIRM ----------------
            elif state.mode == "WAIT_CONFIRM":
                motors.stop()
                if command == "yes buddy":
                    speaker.say("Starting health check.")
                    state.health_stage = 1
                    state.mode = "HEALTH_CHECK"

            # ---------------- STATE: HEALTH_CHECK ----------------
            elif state.mode == "HEALTH_CHECK":
                motors.stop()

                if state.health_stage == 1:
                    speaker.say(
                        "Place the temperature sensor and say check temperature."
                    )
                    state.health_stage = 2

                elif state.health_stage == 2 and "check temperature" in command:
                    state.temperature = health.read_temperature()
                    print("Temperature:", state.temperature)
                    speaker.say("Temperature checked successfully.")
                    state.health_stage = 3

                elif state.health_stage == 3:
                    speaker.say(
                        "Place your finger on the pulse sensor and say check pulse."
                    )
                    state.health_stage = 4

                elif state.health_stage == 4 and "check pulse" in command:
                    state.pulse = health.read_pulse()
                    print("Pulse:", state.pulse)
                    speaker.say("Pulse checked successfully.")
                    state.health_stage = 5

                elif state.health_stage == 5:
                    speaker.say(
                        "Attach ECG electrodes and say check ecg."
                    )
                    state.health_stage = 6

                elif state.health_stage == 6 and "check ecg" in command:
                    state.ecg = health.read_ecg()
                    print("ECG:", state.ecg)
                    speaker.say("ECG checked successfully.")
                    speaker.say(
                        gemini.health_advice(
                            state.name,
                            state.temperature,
                            state.pulse,
                            state.ecg,
                        )
                    )
                    state.mode = "DONE"

            # ---------------- STATE: DONE ----------------
            elif state.mode == "DONE":
                motors.stop()

            cv2.imshow("Healthcare Robot", frame)
            if cv2.waitKey(1) == 27:  # ESC to exit
                break

    finally:
        cap.release()
        motors.shutdown()
        gpio_cleanup()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

