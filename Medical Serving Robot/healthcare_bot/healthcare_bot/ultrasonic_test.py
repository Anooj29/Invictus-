"""
Simple standalone test for the HC‑SR04 ultrasonic distance sensor.

This script uses the shared `hardware.DistanceSensor` abstraction so its
behaviour is consistent with the main robot controller.
"""

from __future__ import annotations

import time

from hardware import DistanceSensor
from config import RobotConfig
from hardware import gpio_cleanup


def main() -> None:
    cfg = RobotConfig()
    sensor = DistanceSensor(
        trig_pin=cfg.ultrasonic_trig_pin,
        echo_pin=cfg.ultrasonic_echo_pin,
    )

    try:
        while True:
            dist = sensor.read_cm()
            print(f"Distance: {dist:.2f} cm")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping distance test.")
    finally:
        gpio_cleanup()


if __name__ == "__main__":
    main()

