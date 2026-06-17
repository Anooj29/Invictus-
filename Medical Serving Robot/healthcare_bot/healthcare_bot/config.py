"""
Central configuration for the Healthcare Serving Robot.

This module groups together tunable constants such as GPIO pin numbers,
camera index, distance thresholds, and serial / socket settings. Keeping
them in one place makes it easier to adjust the behaviour of the robot
without touching the core logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a safe default."""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class RobotConfig:
    """Static configuration values for the robot."""

    # Networking / dashboard
    dashboard_port: int = _env_int("ROBOT_DASHBOARD_PORT", 5000)

    # Distances in centimetres
    distance_trigger_cm: float = float(os.getenv("ROBOT_DISTANCE_TRIGGER_CM", "60"))
    safety_stop_cm: float = float(os.getenv("ROBOT_SAFETY_STOP_CM", "20"))

    # Camera
    camera_index: int = _env_int("ROBOT_CAMERA_INDEX", 0)

    # Health sensor serial port
    serial_port: str = os.getenv("ROBOT_HEALTH_SERIAL_PORT", "/dev/ttyUSB0")
    serial_baudrate: int = _env_int("ROBOT_HEALTH_SERIAL_BAUD", 115200)

    # GPIO pins (BCM numbering)
    ultrasonic_trig_pin: int = 23
    ultrasonic_echo_pin: int = 24

    left_dir_pin: int = 17
    left_pwm_pin: int = 18
    right_dir_pin: int = 22
    right_pwm_pin: int = 27

    # AI model names
    gemini_model_name: str = os.getenv(
        "ROBOT_GEMINI_MODEL", "models/gemini-flash-latest"
    )


def get_gemini_api_key() -> str | None:
    """
    Return the Gemini API key from environment if available.

    The previous code stored the key directly in source, which is not safe.
    Set the key on the Pi with:

        export GEMINI_API_KEY="your-key-here"
    """

    return os.getenv("GEMINI_API_KEY")

