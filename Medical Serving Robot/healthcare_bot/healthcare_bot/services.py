"""
Service layer for the Healthcare Serving Robot.

This module contains:
- Text‑to‑speech wrapper
- Gemini AI helper for greetings and health advice
- Serial health sensor client
- Socket server wrapper for the dashboard connection
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from typing import Optional

import pyttsx3
import serial
from google import genai

from .config import RobotConfig, get_gemini_api_key


class Speaker:
    """Thin wrapper around `pyttsx3` for blocking text‑to‑speech."""

    def __init__(self, rate: int = 150) -> None:
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)

    def say(self, text: str) -> None:
        """Speak the given text and also print it to the console."""
        clean = text.replace("\n", " ").strip()
        print("[VOICE]:", clean)
        self._engine.say(clean)
        self._engine.runAndWait()


class GeminiClient:
    """
    Helper around the Gemini API used for greetings and health advice.

    If the API key is not configured, the methods fall back to simple
    rule‑based messages so that the robot still behaves sensibly offline.
    """

    def __init__(self, cfg: RobotConfig) -> None:
        self._cfg = cfg
        api_key = get_gemini_api_key()
        self._client: Optional[genai.Client] = None

        if api_key:
            self._client = genai.Client(api_key=api_key)

    def greeting(self, name: str, emotion: str) -> str:
        """Generate a short personalised greeting."""
        return (
            f"Hello {name}. You look {emotion}. "
            "Say yes buddy to begin the health check."
        )

    def health_advice(self, name: str, temp: str, pulse: str, ecg: str) -> str:
        """
        Generate friendly health suggestions from the readings.

        The advice is intentionally non‑diagnostic. When Gemini is available
        it is used for phrasing; otherwise a static fallback is returned.
        """
        if not self._client:
            return (
                f"{name}, your readings have been recorded. "
                "Stay hydrated, eat balanced meals, and take enough rest."
            )

        prompt = f"""
Patient name: {name}
Temperature: {temp}
Pulse: {pulse}
ECG: {ecg}

Give short friendly health suggestions.
Do NOT diagnose.
Only give wellness advice, diet suggestion, hydration advice and rest advice.
"""
        try:
            res = self._client.models.generate_content(
                model=self._cfg.gemini_model_name,
                contents=prompt,
            )
            text = (res.text or "").replace("*", "").replace("\n", " ").strip()
            return text or (
                "Your readings look stable. Stay hydrated and eat healthy."
            )
        except Exception as exc:
            print(f"[Gemini] Error generating advice: {exc}")
            return "Your readings look stable. Stay hydrated and eat healthy."


class HealthSensorClient:
    """
    Serial client used to talk to the external health sensor module.

    The protocol is very simple: write a line with the reading type and
    read back a single line with the measurement.
    """

    def __init__(self, cfg: RobotConfig) -> None:
        self._port = cfg.serial_port
        self._baud = cfg.serial_baudrate
        try:
            self._ser = serial.Serial(self._port, self._baud, timeout=3)
            time.sleep(2.0)  # Allow time for the device to reset
            print(f"[Serial] Connected on {self._port}")
        except Exception as exc:
            print(f"[Serial] Failed to open {self._port}: {exc}")
            self._ser = None

    def _read_value(self, command: bytes) -> str:
        """Send a command and read a single line response."""
        if not self._ser:
            return "--"
        try:
            self._ser.reset_input_buffer()
            self._ser.write(command + b"\n")
            value = self._ser.readline().decode(errors="ignore").strip()
            return value or "--"
        except Exception as exc:
            print(f"[Serial] Error while reading: {exc}")
            return "--"

    def read_temperature(self) -> str:
        return self._read_value(b"temp")

    def read_pulse(self) -> str:
        return self._read_value(b"pulse")

    def read_ecg(self) -> str:
        return self._read_value(b"ecg")


@dataclass
class Telemetry:
    """Structured data sent to the dashboard over the socket."""

    distance: float
    temperature: str
    pulse: str
    ecg: str

    def to_json_line(self) -> bytes:
        payload = {
            "distance": self.distance,
            "temp": self.temperature,
            "pulse": self.pulse,
            "ecg": self.ecg,
        }
        return (json.dumps(payload) + "\n").encode()


class DashboardConnection:
    """
    Blocking TCP server that waits for a single dashboard/laptop client.

    The connection is kept in non‑blocking mode for reads so that the main
    loop can continue even when no command is received.
    """

    def __init__(self, cfg: RobotConfig) -> None:
        self._port = cfg.dashboard_port
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("", self._port))
        server.listen(1)
        print(f"[Socket] Waiting for dashboard on port {self._port} ...")
        conn, addr = server.accept()
        print("[Socket] Connected to:", addr)
        conn.setblocking(False)
        self._conn = conn

    def send_telemetry(self, telemetry: Telemetry) -> None:
        """Send the latest telemetry snapshot to the dashboard."""
        try:
            self._conn.sendall(telemetry.to_json_line())
        except OSError as exc:
            # For now just log the error; reconnection logic can be added later.
            print(f"[Socket] Failed to send telemetry: {exc}")

    def read_command(self) -> str:
        """
        Read the latest command from the dashboard, if any.

        Returns a lower‑case string, or an empty string when no command
        was received during this cycle.
        """
        try:
            data = self._conn.recv(1024)
        except BlockingIOError:
            return ""
        except OSError as exc:
            print(f"[Socket] Error while reading command: {exc}")
            return ""

        return data.decode(errors="ignore").strip().lower()

