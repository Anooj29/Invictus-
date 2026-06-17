"""
Hardware abstraction layer for the Healthcare Serving Robot.

This module wraps the low‑level GPIO access for:
- Ultrasonic distance measurement
- Differential drive motor control

All GPIO setup lives here so that the rest of the code can focus on logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import RPi.GPIO as GPIO


# Global GPIO initialisation (BCM numbering is used across the project).
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)


@dataclass
class DistanceSensor:
    """HC‑SR04 ultrasonic distance sensor using BCM pin numbers."""

    trig_pin: int
    echo_pin: int
    timeout_s: float = 0.05  # seconds

    def __post_init__(self) -> None:
        GPIO.setup(self.trig_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)
        GPIO.output(self.trig_pin, False)

    def read_cm(self) -> float:
        """
        Measure distance in centimetres.

        Returns 999 on timeout so that callers can treat it as 'no reading'.
        """
        GPIO.output(self.trig_pin, False)
        time.sleep(0.02)

        GPIO.output(self.trig_pin, True)
        time.sleep(0.00001)  # 10 µs pulse
        GPIO.output(self.trig_pin, False)

        deadline = time.time() + self.timeout_s

        # Wait for echo to go high
        while GPIO.input(self.echo_pin) == 0:
            if time.time() > deadline:
                return 999.0
        start = time.time()

        # Measure how long echo stays high
        while GPIO.input(self.echo_pin) == 1:
            if time.time() > deadline:
                return 999.0
        end = time.time()

        # Speed of sound ~343 m/s -> 17150 cm/s for round trip.
        distance = (end - start) * 17150.0
        return round(distance, 2)


class MotorController:
    """
    Simple differential drive motor controller using a dual H‑bridge.

    Duty‑cycle values are kept conservative for a smoother movement profile.
    """

    def __init__(
        self,
        left_dir_pin: int,
        left_pwm_pin: int,
        right_dir_pin: int,
        right_pwm_pin: int,
        pwm_frequency_hz: int = 1000,
    ) -> None:
        self.left_dir_pin = left_dir_pin
        self.left_pwm_pin = left_pwm_pin
        self.right_dir_pin = right_dir_pin
        self.right_pwm_pin = right_pwm_pin

        GPIO.setup(self.left_dir_pin, GPIO.OUT)
        GPIO.setup(self.right_dir_pin, GPIO.OUT)
        GPIO.setup(self.left_pwm_pin, GPIO.OUT)
        GPIO.setup(self.right_pwm_pin, GPIO.OUT)

        self._left_pwm = GPIO.PWM(self.left_pwm_pin, pwm_frequency_hz)
        self._right_pwm = GPIO.PWM(self.right_pwm_pin, pwm_frequency_hz)

        self._left_pwm.start(0)
        self._right_pwm.start(0)

    def _set_speed(self, left_duty: float, right_duty: float) -> None:
        """Update motor duty‑cycle while keeping direction pins forward."""
        GPIO.output(self.left_dir_pin, 0)
        GPIO.output(self.right_dir_pin, 0)
        self._left_pwm.ChangeDutyCycle(max(0.0, min(100.0, left_duty)))
        self._right_pwm.ChangeDutyCycle(max(0.0, min(100.0, right_duty)))

    def forward(self, duty: float = 40.0) -> None:
        """Drive both wheels forward at the same speed."""
        self._set_speed(duty, duty)
        print("MOTOR: Forward")

    def turn_left(self, base_duty: float = 40.0, delta: float = 20.0) -> None:
        """
        Turn left by slowing the left wheel relative to the right.
        """
        self._set_speed(base_duty - delta, base_duty)
        print("MOTOR: Left")

    def turn_right(self, base_duty: float = 40.0, delta: float = 20.0) -> None:
        """
        Turn right by slowing the right wheel relative to the left.
        """
        self._set_speed(base_duty, base_duty - delta)
        print("MOTOR: Right")

    def stop(self) -> None:
        """Immediately stop the robot."""
        self._left_pwm.ChangeDutyCycle(0)
        self._right_pwm.ChangeDutyCycle(0)
        GPIO.output(self.left_dir_pin, 0)
        GPIO.output(self.right_dir_pin, 0)
        print("MOTOR: Stop")

    def shutdown(self) -> None:
        """
        Stop PWM and release hardware resources.

        Call this from the main script before `GPIO.cleanup()`.
        """
        self.stop()
        self._left_pwm.stop()
        self._right_pwm.stop()


def gpio_cleanup() -> None:
    """
    Clean up all GPIO resources.

    This should be called exactly once when the application is about to exit.
    """
    GPIO.cleanup()

