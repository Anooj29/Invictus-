## Healthcare Serving Robot (Raspberry Pi)

This project controls a medical serving robot that:
- Detects a person using a TensorFlow‑Lite model
- Recognises faces from a `known_faces` database
- Detects facial emotion using a TensorFlow‑Lite model
- Guides the user through a health check (temperature, pulse, ECG)
- Streams simple telemetry to a dashboard via TCP

### Project layout

- `config.py` – central configuration (ports, GPIO pins, thresholds, camera index).
- `hardware.py` – GPIO setup, `DistanceSensor`, `MotorController`, and `gpio_cleanup`.
- `vision.py` – `PersonDetector`, `EmotionDetector`, and `load_known_faces`.
- `services.py` – `Speaker`, `GeminiClient`, `HealthSensorClient`, and `DashboardConnection`.
- `integrated_version.py` – main robot controller and state machine.
- `face_recognition_system.py` – standalone viewer to test `known_faces`.
- `capture_multiple_faces.py` – helper to capture training images per person.
- `ultrasonic_test.py` – standalone ultrasonic distance test.

Place TensorFlow‑Lite models (`person_detect.tflite`, `emotion_model.tflite`) and the
label map in the same folder as these scripts.

### Installing dependencies

On the Raspberry Pi (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

### Environment variables

For security the Gemini API key is **not** hard‑coded. Set it before running:

```bash
export GEMINI_API_KEY="your-key-here"
```

Optional overrides (all have sensible defaults):

- `ROBOT_DASHBOARD_PORT` – TCP port for the dashboard (default `5000`).
- `ROBOT_DISTANCE_TRIGGER_CM` – distance to start interaction (default `60`).
- `ROBOT_SAFETY_STOP_CM` – minimum safe distance to stop (default `20`).
- `ROBOT_CAMERA_INDEX` – camera index for `cv2.VideoCapture` (default `0`).
- `ROBOT_HEALTH_SERIAL_PORT` – serial device for health sensors (default `/dev/ttyUSB0`).
- `ROBOT_HEALTH_SERIAL_BAUD` – serial baud rate (default `115200`).

### Running the robot

From the `healthcare_bot` folder:

```bash
python -m healthcare_bot.integrated_version
```

Make sure:
- The Pi camera and ultrasonic sensor are connected as per the GPIO pins
  defined in `config.py`.
- The motor driver is wired to the motor pins from `config.py`.
- The external health sensor module is available on the configured serial port.

