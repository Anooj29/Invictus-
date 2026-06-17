# -*- coding: utf-8 -*-
"""
Healthcare Serving Robot - Person Detection and Interaction System

This script implements a robot that detects persons using computer vision,
recognizes faces, detects emotions, and performs health checks via sensors.
It communicates with a dashboard over socket and uses ultrasonic sensors for distance.

Hardware Requirements:
- Raspberry Pi with camera
- Ultrasonic sensor (HC-SR04) connected to GPIO pins 23 (TRIG) and 24 (ECHO)
- Motor driver for movement (GPIO pins 17,18,22,27)
- Serial device for health sensors (/dev/ttyUSB0)
- Speaker for voice output

Software Dependencies:
- OpenCV, face_recognition, numpy, tflite_runtime, pyttsx3, google-genai, RPi.GPIO, serial, socket

Author: [Your Name]
Date: [Date]
"""

import cv2
import time
import os
import json
import socket
import serial
import re
import RPi.GPIO as GPIO
import face_recognition
import numpy as np
import tflite_runtime.interpreter as tflite
import pyttsx3
from google import genai
from datetime import datetime

# ================= CONFIGURATION =================
# API key for Google Gemini AI (consider using environment variables for security)
API_KEY = "AIzaSyDyS2vVU9R3n3qx8_g6IX_wgqkQfYoSyLg"
MODEL_NAME = "models/gemini-flash-latest"
PORT = 5000  # Port for socket communication with dashboard

# Distance thresholds in cm
DISTANCE_TRIGGER = 60  # Distance to start interaction
SAFETY_STOP = 20       # Minimum safe distance

# ================= GEMINI AI FUNCTIONS =================
# Initialize Google Gemini client for AI-powered responses
client_ai = genai.Client(api_key=API_KEY)

def clean_text(text):
    """
    Clean the text response from AI by removing markdown and extra spaces.
    
    Args:
        text (str): Raw text from AI response
        
    Returns:
        str: Cleaned text
    """
    text = re.sub(r"\*+", "", text)  # Remove markdown asterisks
    return text.replace("\n", " ").strip()

def gemini_greeting(name, emotion):
    """
    Generate a personalized greeting using AI.
    
    Args:
        name (str): Person's name
        emotion (str): Detected emotion
        
    Returns:
        str: Greeting message
    """
    return f"Hello {name}. You look {emotion}. Say yes buddy to begin health check."

def gemini_health(name, temp, pulse, ecg):
    """
    Generate health suggestions based on readings using AI.
    
    Args:
        name (str): Patient name
        temp (str): Temperature reading
        pulse (str): Pulse reading
        ecg (str): ECG reading
        
    Returns:
        str: Health advice
    """
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
        res = client_ai.models.generate_content(model=MODEL_NAME, contents=prompt)
        return clean_text(res.text)
    except Exception as e:
        print(f"Error generating health advice: {e}")
        return "Your readings look stable. Stay hydrated and eat healthy."

# ================= VOICE OUTPUT =================
# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Speech rate

def speak(text):
    """
    Speak the given text using TTS and print it to console.
    
    Args:
        text (str): Text to speak
    """
    text = clean_text(text)
    print("[VOICE]:", text)
    engine.say(text)
    engine.runAndWait()

# ================= SOCKET COMMUNICATION =================
# Set up socket server for communication with dashboard/laptop
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("", PORT))
server.listen(1)
print("Waiting for laptop connection...")
conn, addr = server.accept()
print("Connected to:", addr)
conn.setblocking(False)  # Non-blocking mode

# ================= SERIAL COMMUNICATION =================
# Initialize serial connection to health sensors
try:
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=3)
    time.sleep(2)  # Allow time for serial to initialize
    print("Serial connection established.")
except Exception as e:
    print(f"Error opening serial port: {e}")
    ser = None

# ================= ULTRASONIC SENSOR =================
# Configure GPIO for ultrasonic distance sensor
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

TRIG = 23  # GPIO pin for trigger
ECHO = 24  # GPIO pin for echo
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

def get_distance():
    """
    Measure distance using ultrasonic sensor.
    
    Returns:
        float: Distance in cm, or 999 if timeout
    """
    GPIO.output(TRIG, False)
    time.sleep(0.02)  # Settling time
    GPIO.output(TRIG, True)
    time.sleep(0.00001)  # 10us pulse
    GPIO.output(TRIG, False)

    timeout = time.time() + 0.05  # 50ms timeout
    while GPIO.input(ECHO) == 0:
        if time.time() > timeout:
            return 999  # Timeout
    start = time.time()

    while GPIO.input(ECHO) == 1:
        if time.time() > timeout:
            return 999  # Timeout
    end = time.time()

    dist = round((end - start) * 17150, 2)  # Speed of sound: 343m/s = 17150 cm/s (round trip)
    return dist

# ================= MOTORS =================
LEFT_DIR  = 17
LEFT_PWM  = 18
RIGHT_DIR = 22
RIGHT_PWM = 27

GPIO.setup(LEFT_DIR,  GPIO.OUT)
GPIO.setup(RIGHT_DIR, GPIO.OUT)
GPIO.setup(LEFT_PWM,  GPIO.OUT)
GPIO.setup(RIGHT_PWM, GPIO.OUT)

left_motor = GPIO.PWM(LEFT_PWM, 1000)
right_motor = GPIO.PWM(RIGHT_PWM, 1000)
left_motor.start(0)
right_motor.start(0)

def move_forward():
    GPIO.output(LEFT_DIR, 0)   # Set direction for forward movement
    GPIO.output(RIGHT_DIR, 0)   # Set direction for forward movement
    left_motor.ChangeDutyCycle(60)
    right_motor.ChangeDutyCycle(60)
    print("MOTOR: Forward")

def turn_left():
    GPIO.output(LEFT_DIR, 0)
    GPIO.output(RIGHT_DIR, 0)
    left_motor.ChangeDutyCycle(30)
    right_motor.ChangeDutyCycle(60)
    print("MOTOR: Left")

def turn_right():
    GPIO.output(LEFT_DIR, 0)
    GPIO.output(RIGHT_DIR, 0)
    left_motor.ChangeDutyCycle(60)
    right_motor.ChangeDutyCycle(30)
    print("MOTOR: Right")

def stop_motors():
    left_motor.ChangeDutyCycle(0)
    right_motor.ChangeDutyCycle(0)
    GPIO.output(LEFT_DIR, 0)   # Ensure DIR pins are low for complete stop
    GPIO.output(RIGHT_DIR, 0)
    print("MOTOR: Stop")

# ================= PERSON DETECTION MODEL =================
# Load TensorFlow Lite model for person detection
person_model = tflite.Interpreter("person_detect.tflite")
person_model.allocate_tensors()
p_in = person_model.get_input_details()
p_out = person_model.get_output_details()
p_h = p_in[0]['shape'][1]  # Model input height
p_w = p_in[0]['shape'][2]  # Model input width

# ================= LOAD FACES =================
known_encodings = []
known_names = []

for person in os.listdir("known_faces"):
    folder = os.path.join("known_faces", person)
    if os.path.isdir(folder):
        for img in os.listdir(folder):
            image = face_recognition.load_image_file(os.path.join(folder, img))
            enc = face_recognition.face_encodings(image)
            if enc:
                known_encodings.append(enc[0])
                known_names.append(person)

print("Face system ready.")  # Print once after loading all faces

# ================= EMOTION DETECTION MODEL =================
# Load TensorFlow Lite model for emotion detection
emotion_model = tflite.Interpreter("emotion_model.tflite")
emotion_model.allocate_tensors()
e_in = emotion_model.get_input_details()
e_out = emotion_model.get_output_details()
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def detect_emotion(face_img):
    """
    Detect emotion from face image.
    
    Args:
        face_img (numpy.ndarray): Face image in BGR format
        
    Returns:
        str: Detected emotion label
    """
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    input_emotion = resized.reshape(1, 48, 48, 1).astype(np.float32) / 255.0
    emotion_model.set_tensor(e_in[0]['index'], input_emotion)
    emotion_model.invoke()
    output = emotion_model.get_tensor(e_out[0]['index'])
    return emotion_labels[np.argmax(output)]

# ================= STATE MACHINE =================
# Robot states: SEARCH, INTERACT, WAIT_CONFIRM, HEALTH_CHECK
state = "SEARCH"
greeted = False
health_stage = 0
temp = pulse = ecg = "--"  # Default sensor readings
name = "Unknown"  # Default name

# Initialize camera
cap = cv2.VideoCapture(1, cv2.CAP_V4L2)

while True:

    ret, frame = cap.read()
    if not ret:
        continue

    distance = get_distance()
    print("STATE:", state, "| Distance:", distance)

    # Send data to dashboard
    packet = json.dumps({
        "distance": distance,
        "temp":     temp,
        "pulse":    pulse,
        "ecg":      ecg
    }) + "\n"
    try:
        conn.sendall(packet.encode())
    except:
        pass

    # Receive command
    command = ""
    try:
        command = conn.recv(1024).decode().strip().lower()
    except:
        pass
    # ================= SEARCH STATE =================
    if state == "SEARCH":

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (p_w, p_h))
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)

        person_model.set_tensor(p_in[0]['index'], input_data)
        person_model.invoke()

        boxes   = person_model.get_tensor(p_out[0]['index'])[0]
        classes = person_model.get_tensor(p_out[1]['index'])[0]
        scores  = person_model.get_tensor(p_out[2]['index'])[0]

        person_found = False

        for i in range(len(scores)):
            if scores[i] > 0.4 and int(classes[i]) == 0:

                ymin, xmin, ymax, xmax = boxes[i]
                h, w, _ = frame.shape
                xmin = int(xmin * w)
                xmax = int(xmax * w)
                ymin = int(ymin * h)
                ymax = int(ymax * h)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                person_found = True

                # Check if close enough to interact
                if distance <= DISTANCE_TRIGGER:
                    stop_motors()          # Stop motors before transitioning
                    state = "INTERACT"
                    greeted = False        # Reset greeted flag for new session
                    break

                # Safety stop if too close
                if distance <= SAFETY_STOP:
                    stop_motors()
                    break

                # If not too close, move towards the person
                person_center = (xmin + xmax) // 2
                frame_center = w // 2
                margin = 70

                if person_center < frame_center - margin:
                    turn_left()
                elif person_center > frame_center + margin:
                    turn_right()
                else:
                    move_forward()

                break   # Only track the highest-confidence detection

        if not person_found:
            stop_motors()

    # ================= INTERACT STATE =================
    elif state == "INTERACT":
        # Ensure motors are stopped during interaction
        stop_motors()

        # Perform face recognition
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        if encodings and not greeted:
            matches = face_recognition.compare_faces(known_encodings, encodings[0])
            if True in matches:
                name = known_names[matches.index(True)]
                emotion = detect_emotion(frame)
                speak(gemini_greeting(name, emotion))
                greeted = True
                state = "WAIT_CONFIRM"

    # ================= WAIT_CONFIRM STATE =================
    elif state == "WAIT_CONFIRM":
        stop_motors()   # Keep motors off while waiting
        if command == "yes buddy":
            speak("Starting health check.")
            health_stage = 1
            state = "HEALTH_CHECK"

    # ================= HEALTH CHECK =================
    elif state == "HEALTH_CHECK":

        stop_motors()   # motors must be off throughout health check

        if health_stage == 1:
            speak("Place temperature sensor and say check temperature.")
            health_stage = 2

        elif health_stage == 2 and "check temperature" in command:
            ser.write(b'temp\n')
            temp = ser.readline().decode().strip()
            print("Temperature:", temp)
            speak("Temperature checked successfully.")
            health_stage = 3

        elif health_stage == 3:
            speak("Place your finger on pulse sensor and say check pulse.")
            health_stage = 4

        elif health_stage == 4 and "check pulse" in command:
            ser.write(b'pulse\n')
            pulse = ser.readline().decode().strip()
            print("Pulse:", pulse)
            speak("Pulse checked successfully.")
            health_stage = 5